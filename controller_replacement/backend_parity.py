#!/usr/bin/env python3
"""Re-run recorded SONIC inputs to isolate inference-backend differences.

The deterministic rollout stores the exact encoder and decoder inputs in
``policy_telemetry.npz``.  This tool selects a small, fixed set of inferences,
replays those immutable tensors through ONNX Runtime, and compares the outputs
with the token/action that the rollout recorded.  MuJoCo, reference generation,
history construction, PD control, and message timing are therefore outside the
comparison.

TensorRT is intentionally represented only by an explicit unavailable
interface in this revision.  No TensorRT result is inferred from ORT output.
"""

from __future__ import annotations

import argparse
import errno
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
from typing import Any, Callable, Mapping, Protocol, Sequence
import uuid

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controller_replacement.output import (  # noqa: E402
    decode_ragged,
    sha256_file,
    write_json_atomic,
)


FORMAT_VERSION = 1
TOKEN_DIM = 64
DECODER_INPUT_DIM = 994
ACTION_DIM = 29
DEFAULT_CASES_NAME = "backend_parity_cases.npz"
DEFAULT_REPORT_NAME = "backend_parity.json"
RUN_MANIFEST_NAME = "run_manifest.json"
RUN_COMPLETE_NAME = "run_complete.json"
PARITY_COMPLETE_NAME = "backend_parity_complete.json"
PARITY_OWNED_MARKER = ".backend_parity_output"


class BackendParityError(RuntimeError):
    """A telemetry/model/backend contract required for parity is invalid."""


class BackendUnavailableError(BackendParityError):
    """A requested inference backend has no executable implementation."""


class _ValueInfo(Protocol):
    name: str
    shape: Sequence[Any]
    type: str


class _Session(Protocol):
    def get_inputs(self) -> Sequence[_ValueInfo]: ...

    def get_outputs(self) -> Sequence[_ValueInfo]: ...

    def run(
        self,
        output_names: Sequence[str] | None,
        inputs: Mapping[str, Any],
    ) -> list[Any]: ...


SessionFactory = Callable[[Path, str], _Session]


@dataclass(frozen=True)
class SonicParityCase:
    """One immutable encoder/decoder input pair from policy telemetry."""

    telemetry_index: int
    policy_seq: int
    encoder_input: np.ndarray
    decoder_input: np.ndarray
    recorded_token: np.ndarray
    recorded_raw_action: np.ndarray

    def __post_init__(self) -> None:
        if self.telemetry_index < 0 or self.policy_seq < 0:
            raise BackendParityError("telemetry_index and policy_seq must be non-negative")
        arrays = {
            "encoder_input": (self.encoder_input, None),
            "decoder_input": (self.decoder_input, DECODER_INPUT_DIM),
            "recorded_token": (self.recorded_token, TOKEN_DIM),
            "recorded_raw_action": (self.recorded_raw_action, ACTION_DIM),
        }
        for name, (value, expected_size) in arrays.items():
            array = np.asarray(value, dtype=np.float32).reshape(-1)
            if expected_size is not None and array.size != expected_size:
                raise BackendParityError(
                    f"{name} has {array.size} values; expected {expected_size}"
                )
            if name == "encoder_input" and array.size <= TOKEN_DIM:
                raise BackendParityError("encoder_input is unexpectedly short")
            if not np.all(np.isfinite(array)):
                raise BackendParityError(f"{name} contains NaN or infinity")
            array = np.ascontiguousarray(array)
            array.setflags(write=False)
            object.__setattr__(self, name, array)


@dataclass(frozen=True)
class ModelPaths:
    encoder: Path
    decoder: Path
    manifest_path: Path | None
    manifest: Mapping[str, Any] | None


@dataclass(frozen=True)
class ParityOutputs:
    cases_npz: Path
    report_json: Path
    report: Mapping[str, Any]


def select_case_indices(inference_count: int) -> tuple[int, ...]:
    """Select ``0, 10, middle, last`` in stable order, removing duplicates."""

    count = int(inference_count)
    if count <= 0:
        raise BackendParityError("policy telemetry contains no inferences")
    candidates = (0, 10, count // 2, count - 1)
    selected: list[int] = []
    for index in candidates:
        if index >= count or index in selected:
            continue
        selected.append(index)
    return tuple(selected)


def _scalar_text(value: np.ndarray, *, name: str) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "SU":
        raise BackendParityError(f"telemetry field {name!r} must be a string scalar")
    result = array.item()
    return result.decode("utf-8") if isinstance(result, bytes) else str(result)


def _decode_required(
    archive: Mapping[str, np.ndarray], field_name: str, index: int
) -> np.ndarray:
    try:
        value = decode_ragged(archive, field_name, index)
    except (KeyError, IndexError, ValueError) as exc:
        raise BackendParityError(
            f"cannot decode telemetry field {field_name!r} at index {index}"
        ) from exc
    if value is None:
        raise BackendParityError(
            f"SONIC telemetry field {field_name!r} is absent at index {index}"
        )
    return np.asarray(value)


def load_sonic_parity_cases(
    telemetry_path: str | Path,
) -> tuple[SonicParityCase, ...]:
    """Decode the fixed parity cases from a SONIC telemetry archive."""

    path = Path(telemetry_path).expanduser().resolve()
    if not path.is_file():
        raise BackendParityError(f"policy telemetry does not exist: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            family = _scalar_text(archive["controller_family"], name="controller_family")
            if not family.strip().lower().startswith("sonic"):
                raise BackendParityError(
                    f"backend parity requires SONIC telemetry; got {family!r}"
                )
            policy_seq = np.asarray(archive["policy_seq"], dtype=np.int64)
            if policy_seq.ndim != 1:
                raise BackendParityError("policy_seq must be one-dimensional")
            indices = select_case_indices(policy_seq.size)
            cases = tuple(
                SonicParityCase(
                    telemetry_index=index,
                    policy_seq=int(policy_seq[index]),
                    encoder_input=_decode_required(
                        archive, "extra_encoderInput", index
                    ),
                    decoder_input=_decode_required(archive, "observation", index),
                    recorded_token=_decode_required(archive, "token", index),
                    recorded_raw_action=_decode_required(
                        archive, "raw_action", index
                    ),
                )
                for index in indices
            )
    except BackendParityError:
        raise
    except (OSError, KeyError, ValueError) as exc:
        raise BackendParityError(f"failed to read SONIC telemetry {path}: {exc}") from exc
    encoder_sizes = {case.encoder_input.size for case in cases}
    if len(encoder_sizes) != 1:
        raise BackendParityError(
            f"selected encoder inputs have inconsistent sizes {sorted(encoder_sizes)}"
        )
    return cases


def _load_json_object(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackendParityError(f"cannot read JSON metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BackendParityError(f"JSON metadata {path} must contain an object")
    return value


def _validate_rollout_completion(
    telemetry_path: Path,
    *,
    telemetry_sha256: str,
) -> tuple[Path | None, str | None]:
    """Validate the completion certificate for rollout-owned telemetry.

    A directory containing ``run_manifest.json`` is treated as a formal
    rollout.  Its final ``run_complete.json`` certificate is therefore
    mandatory and must bind the exact policy telemetry bytes used here.  A
    standalone telemetry/model fixture without either file remains supported.
    If a completion certificate exists without a manifest, it is still
    honored rather than silently ignored.
    """

    rollout_directory = telemetry_path.parent
    manifest_path = rollout_directory / RUN_MANIFEST_NAME
    complete_path = rollout_directory / RUN_COMPLETE_NAME
    if manifest_path.is_symlink() or complete_path.is_symlink():
        raise BackendParityError(
            "formal rollout manifest/completion certificate must not be a symlink"
        )
    has_manifest = manifest_path.is_file()
    has_complete = complete_path.is_file()
    if not has_manifest and not has_complete:
        return None, None
    if has_manifest and not has_complete:
        raise BackendParityError(
            f"formal rollout has {RUN_MANIFEST_NAME} but no {RUN_COMPLETE_NAME}: "
            f"{rollout_directory}"
        )
    if not has_complete:
        return None, None

    complete = _load_json_object(complete_path)
    try:
        protocol_revision = int(complete.get("protocol_revision", -1))
    except (TypeError, ValueError):
        protocol_revision = -1
    if not (
        protocol_revision == 2
        and complete.get("complete") is True
        and complete.get("fixed_step_schedule_complete") is True
        and complete.get("finite_state_complete") is True
    ):
        raise BackendParityError(
            f"{RUN_COMPLETE_NAME} does not certify a complete finite "
            "fixed-step protocol-2 rollout"
        )
    artifact_hashes = complete.get("artifact_sha256")
    if not isinstance(artifact_hashes, Mapping):
        raise BackendParityError(
            f"{RUN_COMPLETE_NAME} has no artifact_sha256 object"
        )
    expected_hash = artifact_hashes.get(telemetry_path.name)
    if not isinstance(expected_hash, str) or not expected_hash.strip():
        raise BackendParityError(
            f"{RUN_COMPLETE_NAME} does not certify {telemetry_path.name}"
        )
    expected_hash = expected_hash.strip().lower()
    if expected_hash != telemetry_sha256:
        raise BackendParityError(
            f"{telemetry_path.name} SHA-256 differs from {RUN_COMPLETE_NAME}: "
            f"expected {expected_hash}, got {telemetry_sha256}"
        )
    if has_manifest:
        expected_manifest_hash = artifact_hashes.get(RUN_MANIFEST_NAME)
        if not isinstance(expected_manifest_hash, str) or not (
            expected_manifest_hash.strip()
        ):
            raise BackendParityError(
                f"{RUN_COMPLETE_NAME} does not certify {RUN_MANIFEST_NAME}"
            )
        expected_manifest_hash = expected_manifest_hash.strip().lower()
        actual_manifest_hash = sha256_file(manifest_path)
        if actual_manifest_hash != expected_manifest_hash:
            raise BackendParityError(
                f"{RUN_MANIFEST_NAME} SHA-256 differs from {RUN_COMPLETE_NAME}: "
                f"expected {expected_manifest_hash}, got {actual_manifest_hash}"
            )
    return complete_path, sha256_file(complete_path)


def _default_output_directory(
    telemetry_path: Path,
    *,
    device: str,
    backend: str = "ort",
) -> Path:
    return (
        telemetry_path.parent.parent
        / "analysis"
        / f"{telemetry_path.parent.name}_backend_parity_{backend}_{device}"
    ).resolve()


def _validate_output_directory(
    telemetry_path: Path,
    output_directory: str | Path | None,
    *,
    device: str,
    backend: str = "ort",
) -> Path:
    rollout_directory = telemetry_path.parent.resolve()
    if output_directory is not None:
        raw_destination = Path(output_directory).expanduser()
        if raw_destination.is_symlink():
            raise BackendParityError(
                f"refusing symbolic-link output: {raw_destination}"
            )
    destination = (
        _default_output_directory(
            telemetry_path,
            device=device,
            backend=backend,
        )
        if output_directory is None
        else Path(output_directory).expanduser().resolve()
    )
    overlaps = False
    for candidate, parent in (
        (destination, rollout_directory),
        (rollout_directory, destination),
    ):
        try:
            candidate.relative_to(parent)
        except ValueError:
            continue
        overlaps = True
        break
    if overlaps:
        raise BackendParityError(
            "backend parity output cannot equal, contain, or be contained by "
            f"the certified rollout directory: {destination}"
        )
    return destination


def _reject_output_containing_inputs(
    destination: Path, inputs: Mapping[str, Path | None]
) -> None:
    """Prevent replacement of an output directory from deleting its inputs."""

    for name, value in inputs.items():
        if value is None:
            continue
        path = value.expanduser().resolve()
        try:
            path.relative_to(destination)
        except ValueError:
            continue
        raise BackendParityError(
            f"backend parity output contains its {name} input: {destination}"
        )


def _manifest_model(
    manifest: Mapping[str, Any], name: str
) -> Mapping[str, Any] | None:
    models = manifest.get("models")
    if not isinstance(models, Mapping):
        return None
    value = models.get(name)
    return value if isinstance(value, Mapping) else None


def _path_from_manifest(manifest_path: Path, entry: Mapping[str, Any], name: str) -> Path:
    raw_path = entry.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise BackendParityError(f"manifest model {name!r} has no path")
    value = Path(raw_path).expanduser()
    return value.resolve() if value.is_absolute() else (manifest_path.parent / value).resolve()


def resolve_model_paths(
    telemetry_path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    encoder_path: str | Path | None = None,
    decoder_path: str | Path | None = None,
) -> ModelPaths:
    """Resolve exact encoder/decoder artifacts from a manifest or CLI paths."""

    telemetry = Path(telemetry_path).expanduser().resolve()
    default_manifest = telemetry.parent / "run_manifest.json"
    selected_manifest = (
        default_manifest
        if manifest_path is None and default_manifest.is_file()
        else (
            None
            if manifest_path is None
            else Path(manifest_path).expanduser().resolve()
        )
    )
    manifest = None if selected_manifest is None else _load_json_object(selected_manifest)
    if manifest is not None:
        family = str(manifest.get("controller_family", ""))
        if not family.strip().lower().startswith("sonic"):
            raise BackendParityError(
                f"manifest controller_family is not SONIC: {family!r}"
            )

    resolved: dict[str, Path] = {}
    for name, explicit in (("encoder", encoder_path), ("decoder", decoder_path)):
        entry = None if manifest is None else _manifest_model(manifest, name)
        if explicit is not None:
            path = Path(explicit).expanduser().resolve()
        elif entry is not None:
            assert selected_manifest is not None
            path = _path_from_manifest(selected_manifest, entry, name)
        else:
            raise BackendParityError(
                f"no {name} model path: provide --{name} or a run_manifest.json"
            )
        if not path.is_file():
            raise BackendParityError(f"{name} model does not exist: {path}")
        if entry is not None and isinstance(entry.get("sha256"), str):
            expected_hash = str(entry["sha256"]).lower()
            actual_hash = sha256_file(path)
            if actual_hash != expected_hash:
                raise BackendParityError(
                    f"{name} model SHA-256 differs from run_manifest.json: "
                    f"expected {expected_hash}, got {actual_hash}"
                )
        resolved[name] = path
    return ModelPaths(
        encoder=resolved["encoder"],
        decoder=resolved["decoder"],
        manifest_path=selected_manifest,
        manifest=manifest,
    )


def _make_ort_session(path: Path, device: str) -> _Session:
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise BackendUnavailableError(
            "ONNX Runtime is not installed in this Python environment"
        ) from exc
    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    normalized = str(device).strip().lower()
    available = set(ort.get_available_providers())
    if normalized == "cpu":
        providers = ["CPUExecutionProvider"]
    elif normalized in {"cuda", "gpu"}:
        if "CUDAExecutionProvider" not in available:
            raise BackendUnavailableError(
                "ORT CUDAExecutionProvider was requested but is unavailable"
            )
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        raise BackendParityError("ORT device must be 'cpu' or 'cuda'")
    return ort.InferenceSession(str(path), sess_options=options, providers=providers)


def _shape_tuple(value: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(
        item if isinstance(item, (int, str)) or item is None else str(item)
        for item in value
    )


def _validate_session(
    session: _Session,
    *,
    name: str,
    input_size: int,
    output_size: int,
) -> tuple[str, str]:
    inputs = list(session.get_inputs())
    outputs = list(session.get_outputs())
    if len(inputs) != 1 or len(outputs) != 1:
        raise BackendParityError(
            f"{name} model must have one input/output; got {len(inputs)}/{len(outputs)}"
        )
    if _shape_tuple(inputs[0].shape) != (1, input_size):
        raise BackendParityError(
            f"{name} input shape is {_shape_tuple(inputs[0].shape)}; "
            f"expected {(1, input_size)}"
        )
    if _shape_tuple(outputs[0].shape) != (1, output_size):
        raise BackendParityError(
            f"{name} output shape is {_shape_tuple(outputs[0].shape)}; "
            f"expected {(1, output_size)}"
        )
    if inputs[0].type != "tensor(float)" or outputs[0].type != "tensor(float)":
        raise BackendParityError(
            f"{name} model must use float tensors; got "
            f"{inputs[0].type}/{outputs[0].type}"
        )
    return str(inputs[0].name), str(outputs[0].name)


def _run_vector(
    session: _Session,
    *,
    input_name: str,
    output_name: str,
    value: np.ndarray,
    output_size: int,
) -> np.ndarray:
    output = session.run(
        [output_name], {input_name: np.asarray(value, dtype=np.float32)[None, :]}
    )
    if len(output) != 1:
        raise BackendParityError(f"backend returned {len(output)} outputs; expected one")
    array = np.asarray(output[0], dtype=np.float32)
    if array.shape != (1, output_size) or not np.all(np.isfinite(array)):
        raise BackendParityError(
            f"backend output has shape {array.shape}; expected {(1, output_size)}"
        )
    return array[0].copy()


def _error_metrics(expected: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    expected_array = np.asarray(expected, dtype=np.float64)
    actual_array = np.asarray(actual, dtype=np.float64)
    if expected_array.shape != actual_array.shape:
        raise BackendParityError(
            f"cannot compare shapes {expected_array.shape} and {actual_array.shape}"
        )
    finite = bool(np.all(np.isfinite(expected_array)) and np.all(np.isfinite(actual_array)))
    if not finite:
        return {
            "element_count": int(expected_array.size),
            "all_finite": False,
            "max_abs": None,
            "mean_abs": None,
            "rmse": None,
            "cosine_similarity": None,
            "bitwise_equal_after_float32_cast": False,
        }
    difference = actual_array - expected_array
    left = expected_array.reshape(-1)
    right = actual_array.reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    cosine = (
        1.0
        if denominator == 0.0 and np.array_equal(left, right)
        else (None if denominator == 0.0 else float(np.dot(left, right) / denominator))
    )
    return {
        "element_count": int(expected_array.size),
        "all_finite": True,
        "max_abs": float(np.max(np.abs(difference))) if difference.size else 0.0,
        "mean_abs": float(np.mean(np.abs(difference))) if difference.size else 0.0,
        "rmse": float(np.sqrt(np.mean(difference * difference))) if difference.size else 0.0,
        "cosine_similarity": cosine,
        "bitwise_equal_after_float32_cast": bool(
            np.array_equal(
                expected_array.astype(np.float32), actual_array.astype(np.float32)
            )
        ),
    }


def _comparison_report(expected: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    return {
        "aggregate": _error_metrics(expected, actual),
        "per_case": [
            {"case_position": index, **_error_metrics(left, right)}
            for index, (left, right) in enumerate(zip(expected, actual, strict=True))
        ],
    }


def _save_npz_atomic(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            np.savez_compressed(output, **payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _validate_parity_output(
    path: Path, *, expected_identity: Mapping[str, Any]
) -> None:
    """Allow replacement only of a complete output for the same inputs."""

    if not path.exists():
        return
    if path.is_symlink() or not path.is_dir():
        raise BackendParityError(f"refusing to replace non-directory output: {path}")
    marker = path / PARITY_OWNED_MARKER
    complete_path = path / PARITY_COMPLETE_NAME
    if (
        marker.is_symlink()
        or not marker.is_file()
        or marker.read_text(encoding="utf-8").strip() != "1"
        or complete_path.is_symlink()
        or not complete_path.is_file()
    ):
        raise BackendParityError(
            "refusing to replace an unowned or incomplete backend-parity "
            f"directory: {path}"
        )
    complete = _load_json_object(complete_path)
    if not (
        complete.get("complete") is True
        and complete.get("format_version") == FORMAT_VERSION
        and complete.get("identity") == dict(expected_identity)
    ):
        raise BackendParityError(
            "existing backend-parity output belongs to different inputs or is "
            f"not complete: {path}"
        )
    artifacts = complete.get("artifact_sha256")
    if not isinstance(artifacts, Mapping):
        raise BackendParityError("backend-parity completion has no artifact hashes")
    for name in (DEFAULT_CASES_NAME, DEFAULT_REPORT_NAME):
        artifact = path / name
        if artifact.is_symlink() or not artifact.is_file():
            raise BackendParityError(f"backend-parity output is missing {name}")
        if artifacts.get(name) != sha256_file(artifact):
            raise BackendParityError(
                f"backend-parity {name} differs from its completion certificate"
            )


class _ParityOutputTransaction:
    """Publish the NPZ, JSON and completion certificate as one directory."""

    def __init__(self, output: Path, *, identity: Mapping[str, Any]) -> None:
        raw = output.expanduser()
        if raw.is_symlink():
            raise BackendParityError(f"refusing symbolic-link output: {raw}")
        self.output = raw.resolve()
        self.identity = dict(identity)
        self.work: Path | None = None
        self.backup: Path | None = None
        self.lock_fd: int | None = None
        self.published = False

    @property
    def lock_path(self) -> Path:
        return self.output.parent / f".{self.output.name}.lock"

    def __enter__(self) -> Path:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        self.lock_fd = os.open(self.lock_path, flags, 0o600)
        if not stat.S_ISREG(os.fstat(self.lock_fd).st_mode):
            self._release()
            raise BackendParityError(f"parity lock is not regular: {self.lock_path}")
        try:
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._release()
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise BackendParityError(
                    f"another process is publishing this parity output: {self.output}"
                ) from exc
            raise
        try:
            _validate_parity_output(
                self.output, expected_identity=self.identity
            )
            self.work = Path(
                tempfile.mkdtemp(
                    prefix=f".{self.output.name}.work-", dir=self.output.parent
                )
            )
            (self.work / PARITY_OWNED_MARKER).write_text("1\n", encoding="utf-8")
        except BaseException:
            if self.work is not None:
                shutil.rmtree(self.work, ignore_errors=True)
            self._release()
            raise
        return self.work

    def _release(self) -> None:
        if self.lock_fd is not None:
            try:
                fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(self.lock_fd)
                self.lock_fd = None

    @staticmethod
    def _remove_owned(path: Path) -> None:
        marker = path / PARITY_OWNED_MARKER
        if (
            path.is_symlink()
            or not path.is_dir()
            or marker.is_symlink()
            or not marker.is_file()
            or marker.read_text(encoding="utf-8").strip() != "1"
        ):
            raise BackendParityError(f"refusing to remove unowned output: {path}")
        shutil.rmtree(path)

    def publish(self) -> None:
        if self.work is None:
            raise BackendParityError("parity transaction has no work directory")
        _validate_parity_output(self.work, expected_identity=self.identity)
        _validate_parity_output(self.output, expected_identity=self.identity)
        try:
            if self.output.exists():
                self.backup = self.output.parent / (
                    f".{self.output.name}.backup-{uuid.uuid4().hex}"
                )
                self.output.replace(self.backup)
            self.work.replace(self.output)
        except BaseException as publish_error:
            if self.backup is not None and self.backup.exists():
                if self.output.exists():
                    self._remove_owned(self.output)
                self.backup.replace(self.output)
                self.backup = None
            elif (
                self.backup is None
                and self.work is not None
                and not self.work.exists()
                and self.output.exists()
            ):
                # The install rename completed just before an asynchronous
                # exception was delivered.  Do not advertise a failed command
                # as a published parity result.
                self._remove_owned(self.output)
            raise publish_error
        self.work = None
        self.published = True
        if self.backup is not None:
            self._remove_owned(self.backup)
            self.backup = None

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            if not self.published and self.backup is not None and self.backup.exists():
                if self.output.exists():
                    self._remove_owned(self.output)
                self.backup.replace(self.output)
                self.backup = None
            elif (
                not self.published
                and self.backup is None
                and self.work is not None
                and not self.work.exists()
                and self.output.exists()
            ):
                self._remove_owned(self.output)
            if self.work is not None and self.work.exists():
                self._remove_owned(self.work)
                self.work = None
        finally:
            self._release()


def tensorrt_backend_status() -> dict[str, Any]:
    """Return the honest status of the reserved TensorRT parity interface."""

    return {
        "backend": "tensorrt",
        "status": "not_available",
        "numerical_results_present": False,
        "reason": (
            "the deterministic Python TensorRT backend and a usable NVIDIA "
            "driver have not yet been validated in this revision"
        ),
    }


def run_tensorrt_parity(*args: Any, **kwargs: Any) -> None:
    """Reserved interface; never substitutes ORT numbers for TensorRT."""

    del args, kwargs
    raise BackendUnavailableError(tensorrt_backend_status()["reason"])


def run_ort_parity(
    telemetry_path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    encoder_path: str | Path | None = None,
    decoder_path: str | Path | None = None,
    output_directory: str | Path | None = None,
    device: str = "cpu",
    session_factory: SessionFactory | None = None,
) -> ParityOutputs:
    """Run fixed-input ORT self-checks and atomically write both artifacts."""

    telemetry = Path(telemetry_path).expanduser().resolve()
    telemetry_sha256 = sha256_file(telemetry)
    run_complete_path, run_complete_sha256 = _validate_rollout_completion(
        telemetry,
        telemetry_sha256=telemetry_sha256,
    )
    destination = _validate_output_directory(
        telemetry,
        output_directory,
        device=device,
    )
    cases = load_sonic_parity_cases(telemetry)
    if sha256_file(telemetry) != telemetry_sha256:
        raise BackendParityError(
            "policy telemetry changed while fixed parity cases were loaded"
        )
    models = resolve_model_paths(
        telemetry,
        manifest_path=manifest_path,
        encoder_path=encoder_path,
        decoder_path=decoder_path,
    )
    _reject_output_containing_inputs(
        destination,
        {
            "policy telemetry": telemetry,
            "run completion": run_complete_path,
            "run manifest": models.manifest_path,
            "encoder model": models.encoder,
            "decoder model": models.decoder,
        },
    )
    model_sha256 = {
        "encoder": sha256_file(models.encoder),
        "decoder": sha256_file(models.decoder),
    }
    factory = _make_ort_session if session_factory is None else session_factory
    encoder_session = factory(models.encoder, device)
    decoder_session = factory(models.decoder, device)
    if {
        "encoder": sha256_file(models.encoder),
        "decoder": sha256_file(models.decoder),
    } != model_sha256:
        raise BackendParityError(
            "SONIC models changed while parity inference sessions were created"
        )
    encoder_input_name, encoder_output_name = _validate_session(
        encoder_session,
        name="encoder",
        input_size=cases[0].encoder_input.size,
        output_size=TOKEN_DIM,
    )
    decoder_input_name, decoder_output_name = _validate_session(
        decoder_session,
        name="decoder",
        input_size=DECODER_INPUT_DIM,
        output_size=ACTION_DIM,
    )

    encoder_inputs = np.stack([case.encoder_input for case in cases])
    decoder_inputs = np.stack([case.decoder_input for case in cases])
    recorded_tokens = np.stack([case.recorded_token for case in cases])
    recorded_actions = np.stack([case.recorded_raw_action for case in cases])
    ort_tokens: list[np.ndarray] = []
    ort_tokens_repeat: list[np.ndarray] = []
    ort_actions: list[np.ndarray] = []
    ort_actions_repeat: list[np.ndarray] = []
    ort_actions_end_to_end: list[np.ndarray] = []
    for case in cases:
        token = _run_vector(
            encoder_session,
            input_name=encoder_input_name,
            output_name=encoder_output_name,
            value=case.encoder_input,
            output_size=TOKEN_DIM,
        )
        token_repeat = _run_vector(
            encoder_session,
            input_name=encoder_input_name,
            output_name=encoder_output_name,
            value=case.encoder_input,
            output_size=TOKEN_DIM,
        )
        action = _run_vector(
            decoder_session,
            input_name=decoder_input_name,
            output_name=decoder_output_name,
            value=case.decoder_input,
            output_size=ACTION_DIM,
        )
        action_repeat = _run_vector(
            decoder_session,
            input_name=decoder_input_name,
            output_name=decoder_output_name,
            value=case.decoder_input,
            output_size=ACTION_DIM,
        )
        end_to_end_input = case.decoder_input.copy()
        end_to_end_input[:TOKEN_DIM] = token
        end_to_end_action = _run_vector(
            decoder_session,
            input_name=decoder_input_name,
            output_name=decoder_output_name,
            value=end_to_end_input,
            output_size=ACTION_DIM,
        )
        ort_tokens.append(token)
        ort_tokens_repeat.append(token_repeat)
        ort_actions.append(action)
        ort_actions_repeat.append(action_repeat)
        ort_actions_end_to_end.append(end_to_end_action)

    ort_token_array = np.stack(ort_tokens)
    ort_token_repeat_array = np.stack(ort_tokens_repeat)
    ort_action_array = np.stack(ort_actions)
    ort_action_repeat_array = np.stack(ort_actions_repeat)
    ort_action_end_to_end_array = np.stack(ort_actions_end_to_end)
    telemetry_indices = np.asarray(
        [case.telemetry_index for case in cases], dtype=np.int64
    )
    policy_seq = np.asarray([case.policy_seq for case in cases], dtype=np.int64)
    selected_cases = [
        {
            "case_position": position,
            "telemetry_index": case.telemetry_index,
            "policy_seq": case.policy_seq,
        }
        for position, case in enumerate(cases)
    ]
    try:
        import onnxruntime as ort

        ort_version: str | None = str(ort.__version__)
    except ImportError:  # injected sessions in unit tests
        ort_version = None
    runtime = {
        "implementation": (
            "onnxruntime" if session_factory is None else "injected_session_factory"
        ),
        "onnxruntime_version": ort_version,
        "requested_device": str(device),
        "encoder_session_providers": (
            list(encoder_session.get_providers())
            if hasattr(encoder_session, "get_providers")
            else None
        ),
        "decoder_session_providers": (
            list(decoder_session.get_providers())
            if hasattr(decoder_session, "get_providers")
            else None
        ),
    }
    if sha256_file(telemetry) != telemetry_sha256:
        raise BackendParityError("policy telemetry changed during parity inference")
    if {
        "encoder": sha256_file(models.encoder),
        "decoder": sha256_file(models.decoder),
    } != model_sha256:
        raise BackendParityError("SONIC models changed during parity inference")
    if run_complete_path is not None:
        assert run_complete_sha256 is not None
        if sha256_file(run_complete_path) != run_complete_sha256:
            raise BackendParityError(
                f"{RUN_COMPLETE_NAME} changed during parity inference"
            )
        _validate_rollout_completion(
            telemetry,
            telemetry_sha256=telemetry_sha256,
        )
    report: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "backend": "onnxruntime",
        "status": "completed",
        "acceptance_threshold_applied": False,
        "interpretation": (
            "fixed-input numerical comparison only; this is not a closed-loop "
            "MuJoCo or TensorRT comparison"
        ),
        "source": {
            "policy_telemetry": str(telemetry),
            "policy_telemetry_sha256": telemetry_sha256,
            "run_manifest": (
                None if models.manifest_path is None else str(models.manifest_path)
            ),
            "run_complete": (
                None if run_complete_path is None else str(run_complete_path)
            ),
            "run_complete_sha256": run_complete_sha256,
        },
        "models": {
            "encoder": {
                "path": str(models.encoder),
                "sha256": model_sha256["encoder"],
            },
            "decoder": {
                "path": str(models.decoder),
                "sha256": model_sha256["decoder"],
            },
        },
        "runtime": runtime,
        "selected_cases": selected_cases,
        "comparisons": {
            "recorded_token_vs_ort_encoder": _comparison_report(
                recorded_tokens, ort_token_array
            ),
            "recorded_action_vs_ort_fixed_decoder_input": _comparison_report(
                recorded_actions, ort_action_array
            ),
            "recorded_action_vs_ort_end_to_end": _comparison_report(
                recorded_actions, ort_action_end_to_end_array
            ),
            "recorded_token_vs_decoder_input_token_slice": _comparison_report(
                recorded_tokens, decoder_inputs[:, :TOKEN_DIM]
            ),
            "ort_encoder_repeatability": _comparison_report(
                ort_token_array, ort_token_repeat_array
            ),
            "ort_decoder_repeatability": _comparison_report(
                ort_action_array, ort_action_repeat_array
            ),
        },
        "tensorrt": tensorrt_backend_status(),
    }
    metadata_json = json.dumps(report, ensure_ascii=False, sort_keys=True)
    payload = {
        "format_version": np.asarray(FORMAT_VERSION, dtype=np.int64),
        "telemetry_index": telemetry_indices,
        "policy_seq": policy_seq,
        "encoder_input": encoder_inputs.astype(np.float32),
        "decoder_input": decoder_inputs.astype(np.float32),
        "recorded_token": recorded_tokens.astype(np.float32),
        "recorded_raw_action": recorded_actions.astype(np.float32),
        "ort_token": ort_token_array.astype(np.float32),
        "ort_token_repeat": ort_token_repeat_array.astype(np.float32),
        "ort_raw_action_fixed_decoder_input": ort_action_array.astype(np.float32),
        "ort_raw_action_fixed_decoder_input_repeat": ort_action_repeat_array.astype(
            np.float32
        ),
        "ort_raw_action_end_to_end": ort_action_end_to_end_array.astype(np.float32),
        "metadata_json": np.asarray(metadata_json, dtype=np.str_),
    }
    if any(value.dtype.hasobject for value in payload.values()):
        raise AssertionError("backend parity payload must remain pickle-free")
    identity = {
        "format_version": FORMAT_VERSION,
        "backend": "onnxruntime",
        "requested_device": str(device),
        "runtime": runtime,
        "policy_telemetry_sha256": telemetry_sha256,
        "model_sha256": model_sha256,
    }
    transaction = _ParityOutputTransaction(destination, identity=identity)
    with transaction as work:
        work_cases = work / DEFAULT_CASES_NAME
        work_report = work / DEFAULT_REPORT_NAME
        _save_npz_atomic(work_cases, payload)
        write_json_atomic(work_report, report)
        write_json_atomic(
            work / PARITY_COMPLETE_NAME,
            {
                "format_version": FORMAT_VERSION,
                "complete": True,
                "identity": identity,
                "artifact_sha256": {
                    DEFAULT_CASES_NAME: sha256_file(work_cases),
                    DEFAULT_REPORT_NAME: sha256_file(work_report),
                },
            },
        )
        transaction.publish()
    cases_path = destination / DEFAULT_CASES_NAME
    report_path = destination / DEFAULT_REPORT_NAME
    return ParityOutputs(
        cases_npz=cases_path,
        report_json=report_path,
        report=report,
    )


def _resolve_telemetry(value: Path) -> Path:
    path = value.expanduser().resolve()
    return path / "policy_telemetry.npz" if path.is_dir() else path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "rollout",
        type=Path,
        help="rollout directory or its policy_telemetry.npz",
    )
    parser.add_argument("--backend", choices=("ort", "tensorrt"), default="ort")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--encoder", type=Path)
    parser.add_argument("--decoder", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    telemetry = _resolve_telemetry(args.rollout)
    default_destination = _default_output_directory(
        telemetry,
        device=args.device,
        backend=args.backend,
    )
    if args.backend == "tensorrt":
        status = tensorrt_backend_status()
        try:
            destination = _validate_output_directory(
                telemetry,
                args.output_directory,
                device=args.device,
                backend=args.backend,
            )
        except BackendParityError as exc:
            print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        destination.mkdir(parents=True, exist_ok=True)
        write_json_atomic(destination / "backend_parity_tensorrt.json", status)
        print(json.dumps(status, indent=2, ensure_ascii=False))
        return 3
    try:
        outputs = run_ort_parity(
            telemetry,
            manifest_path=args.manifest,
            encoder_path=args.encoder,
            decoder_path=args.decoder,
            output_directory=(
                default_destination
                if args.output_directory is None
                else args.output_directory
            ),
            device=args.device,
        )
    except BackendParityError as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "completed",
                "cases_npz": str(outputs.cases_npz),
                "report_json": str(outputs.report_json),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
