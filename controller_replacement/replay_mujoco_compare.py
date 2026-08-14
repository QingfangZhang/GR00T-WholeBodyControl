#!/usr/bin/env python3
"""Replay a controller-replacement result with its exact source-qpos ghost.

This is a small compatibility wrapper around the established visual-only ghost
viewer.  Controller-replacement rollouts use strict eight-row 50 Hz holds and
can begin at a phase-matched sub-row time, so raw ``initial_row + sample``
reconstruction is not reliable.  The wrapper uses the per-sample mapping saved
in ``source_timeline.npz`` instead.
"""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from change_ckpt_track import replay_mujoco_compare as _viewer  # noqa: E402
from controller_replacement.output import sha256_file  # noqa: E402
from controller_replacement.provenance import (  # noqa: E402
    compiled_mujoco_model_fingerprint,
    snapshot_xml_hashes,
)
from controller_replacement.references import ReferenceSequence  # noqa: E402


_ORIGINAL_BUILD_GHOST_TRACK = _viewer.build_ghost_track
_ORIGINAL_PRINT_SUMMARY = _viewer._print_summary
_ORIGINAL_STAGE_RECORDING_SNAPSHOT = _viewer.stage_recording_snapshot
_CERTIFIED_RUN_MANIFEST: dict[str, Any] | None = None
_CERTIFIED_REPLAY_STATE: dict[str, Any] | None = None


def _regular_file(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise _viewer.ReplayCompareError(
            f"protocol-2 replay requires a regular {label}: {path}"
        )
    return path


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    _regular_file(path, label=label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _viewer.ReplayCompareError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise _viewer.ReplayCompareError(f"{label} must contain a JSON object: {path}")
    return value


def _validate_completion_certificate(
    rollout_dir: Path, *, ghost_mode: str
) -> dict[str, Any]:
    """Authenticate every rollout artifact consumed by this replay wrapper."""

    complete_path = rollout_dir / "run_complete.json"
    manifest_path = rollout_dir / "run_manifest.json"
    complete = _json_object(complete_path, label="run_complete.json")
    manifest = _json_object(manifest_path, label="run_manifest.json")
    try:
        revision = int(complete.get("protocol_revision", -1))
        manifest_revision = int(manifest.get("protocol_revision", -1))
    except (TypeError, ValueError):
        revision = manifest_revision = -1
    if not (
        revision == 2
        and manifest_revision == 2
        and complete.get("complete") is True
        and complete.get("fixed_step_schedule_complete") is True
        and complete.get("finite_state_complete") is True
    ):
        raise _viewer.ReplayCompareError(
            "rollout is not certified as a complete finite fixed-step protocol-2 run"
        )
    certified = complete.get("artifact_sha256")
    if not isinstance(certified, Mapping):
        raise _viewer.ReplayCompareError(
            "run_complete.json has no artifact_sha256 object"
        )
    required = [
        "data.csv",
        "source_timeline.npz",
        "run_manifest.json",
        "run_metadata.json",
        "launch_manifest.json",
    ]
    if ghost_mode == "reference":
        required.append("prepared_reference.npz")
    for name in required:
        path = _regular_file(rollout_dir / name, label=name)
        expected = certified.get(name)
        actual = sha256_file(path)
        if not isinstance(expected, str) or expected.lower() != actual:
            raise _viewer.ReplayCompareError(
                f"{name} SHA-256 is not certified by run_complete.json: "
                f"expected {expected!r}, got {actual}"
            )
    return manifest


def _recorded_staged_scene() -> Mapping[str, Any]:
    manifest = _CERTIFIED_RUN_MANIFEST or {}
    provenance = manifest.get("provenance", {})
    staged = provenance.get("staged_scene", {}) if isinstance(provenance, Mapping) else {}
    if not isinstance(staged, Mapping):
        raise _viewer.ReplayCompareError(
            "certified run_manifest.json has no staged_scene provenance"
        )
    return staged


def _remember_source_csv(source_path: Path) -> str:
    """Bind the external source CSV consumed by the viewer to this replay."""

    actual_path = source_path.expanduser().resolve(strict=True)
    actual_sha256 = sha256_file(actual_path)
    manifest_source = (_CERTIFIED_RUN_MANIFEST or {}).get("provenance", {})
    manifest_source = (
        manifest_source.get("source_recording", {})
        if isinstance(manifest_source, Mapping)
        else {}
    )
    if (
        not isinstance(manifest_source, Mapping)
        or manifest_source.get("sha256_before") != actual_sha256
    ):
        raise _viewer.ReplayCompareError(
            "source data.csv SHA-256 disagrees with certified run provenance"
        )
    if _CERTIFIED_REPLAY_STATE is not None:
        previous_path = _CERTIFIED_REPLAY_STATE.get("source_csv_path")
        previous_sha256 = _CERTIFIED_REPLAY_STATE.get("source_csv_sha256")
        if previous_path is not None and previous_path != str(actual_path):
            raise _viewer.ReplayCompareError(
                "replay resolved two different source data.csv paths"
            )
        if previous_sha256 is not None and previous_sha256 != actual_sha256:
            raise _viewer.ReplayCompareError(
                "source data.csv changed during replay initialization"
            )
        _CERTIFIED_REPLAY_STATE["source_csv_path"] = str(actual_path)
        _CERTIFIED_REPLAY_STATE["source_csv_sha256"] = actual_sha256
    return actual_sha256


def _revalidate_rollout_certificate(rollout_dir: Path) -> None:
    """Recheck the initial certificate without trusting a second manifest."""

    state = _CERTIFIED_REPLAY_STATE
    if state is None:
        raise _viewer.ReplayCompareError("replay certificate state is unavailable")
    try:
        current_rollout = rollout_dir.expanduser().resolve(strict=True)
    except OSError as exc:
        raise _viewer.ReplayCompareError(
            f"rollout path changed during replay initialization: {exc}"
        ) from exc
    if str(current_rollout) != state["rollout_dir"]:
        raise _viewer.ReplayCompareError(
            "rollout path resolved to a different directory during initialization"
        )
    complete_path = _regular_file(
        current_rollout / "run_complete.json", label="run_complete.json"
    )
    if sha256_file(complete_path) != state["run_complete_sha256"]:
        raise _viewer.ReplayCompareError(
            "run_complete.json changed during replay initialization"
        )
    current_manifest = _validate_completion_certificate(
        current_rollout, ghost_mode=state["ghost_mode"]
    )
    if current_manifest != state["run_manifest"]:
        raise _viewer.ReplayCompareError(
            "run_manifest.json changed during replay initialization"
        )


def _revalidate_consumed_inputs(rollout_dir: Path) -> None:
    """Recheck rollout and source bytes after every replay input is loaded."""

    _revalidate_rollout_certificate(rollout_dir)
    state = _CERTIFIED_REPLAY_STATE
    assert state is not None
    source_path_value = state.get("source_csv_path")
    source_sha256 = state.get("source_csv_sha256")
    if not isinstance(source_path_value, str) or not isinstance(source_sha256, str):
        raise _viewer.ReplayCompareError(
            "source data.csv was not bound during replay initialization"
        )
    source_path = _regular_file(Path(source_path_value), label="source data.csv")
    if source_path.resolve(strict=True) != Path(source_path_value):
        raise _viewer.ReplayCompareError(
            "source data.csv path changed during replay initialization"
        )
    if sha256_file(source_path) != source_sha256:
        raise _viewer.ReplayCompareError(
            "source data.csv changed during replay initialization"
        )


def _stage_recording_snapshot(
    recording_dir: Path,
    destination_parent: Path | None,
    asset_model_root: str | Path | None = None,
) -> Any:
    """Stage the certified XML with the recorded asset root by default."""

    staged_scene = _recorded_staged_scene()
    selected_assets: str | Path | None = asset_model_root
    recorded_assets = staged_scene.get("asset_model_root")
    if selected_assets is None:
        if not isinstance(recorded_assets, str) or not recorded_assets:
            raise _viewer.ReplayCompareError(
                "run manifest does not record the rollout asset_model_root; "
                "pass --asset-model-root explicitly"
            )
        selected_assets = recorded_assets
    elif recorded_assets is not None and Path(str(selected_assets)).expanduser().resolve() != (
        Path(str(recorded_assets)).expanduser().resolve()
    ):
        print(
            "[ghost-replay] warning: explicit --asset-model-root differs from "
            "the recorded rollout asset root; compiled-model validation below applies"
        )
    staged = _ORIGINAL_STAGE_RECORDING_SNAPSHOT(
        recording_dir,
        destination_parent,
        asset_model_root=selected_assets,
    )
    expected_xml = staged_scene.get("snapshot_xml_sha256")
    actual_xml = snapshot_xml_hashes(staged.snapshot_root)
    if not isinstance(expected_xml, Mapping) or {
        str(name): str(digest) for name, digest in expected_xml.items()
    } != actual_xml:
        staged.close()
        raise _viewer.ReplayCompareError(
            "replay snapshot XML differs from the certified rollout snapshot"
        )
    return staged


def _build_ghost_track(**kwargs: Any) -> Any:
    mode = kwargs["mode"]
    metadata = kwargs["run_metadata"]
    source = kwargs["source"]
    actual_source_sha256 = _remember_source_csv(source.path)
    if mode == "reference":
        target = kwargs["target"]
        prepared = kwargs.get("prepared")
        path = target.path.parent / "prepared_reference.npz"
        try:
            strict = ReferenceSequence.load_prepared_npz(
                path, source_csv_path=source.path
            )
        except (OSError, ValueError, KeyError) as exc:
            raise _viewer.ReplayCompareError(
                f"prepared reference failed protocol-2 validation: {exc}"
            ) from exc
        if prepared is None or not np.array_equal(
            strict.policy_seq, prepared.policy_seq
        ):
            raise _viewer.ReplayCompareError(
                "prepared reference policy sequence differs after strict validation"
            )
        return _ORIGINAL_BUILD_GHOST_TRACK(**kwargs)
    if mode != "source":
        return _ORIGINAL_BUILD_GHOST_TRACK(**kwargs)
    if metadata.get("source_timeline_path") != "source_timeline.npz":
        raise _viewer.ReplayCompareError(
            "protocol-2 source_timeline_path must be exactly "
            "'source_timeline.npz'"
        )

    target = kwargs["target"]
    _viewer._validate_source_layout(target, source)
    _viewer._validate_target_sample_indices(target)
    sidecar = target.path.parent / "source_timeline.npz"
    if not sidecar.is_file():
        raise _viewer.ReplayCompareError(
            f"source timeline sidecar is missing: {sidecar}"
        )
    try:
        with np.load(sidecar, allow_pickle=False) as archive:
            source_rows = np.asarray(archive["source_row_index"], dtype=np.int64)
            if "source_qpos" not in archive:
                raise _viewer.ReplayCompareError(
                    "protocol-2 source timeline is missing exact source_qpos"
                )
            source_qpos = np.asarray(archive["source_qpos"], dtype=np.float64)
            raw_metadata = str(archive["metadata_json"].item())
            decoded_metadata = json.loads(raw_metadata)
    except (OSError, KeyError, ValueError) as exc:
        raise _viewer.ReplayCompareError(
            f"cannot read source timeline {sidecar}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise _viewer.ReplayCompareError(
            f"source timeline metadata is not valid JSON: {sidecar}"
        ) from exc
    if not isinstance(decoded_metadata, dict):
        raise _viewer.ReplayCompareError("source timeline metadata must be an object")
    try:
        format_version = int(decoded_metadata.get("format_version", -1))
        protocol_revision = int(decoded_metadata.get("protocol_revision", -1))
    except (TypeError, ValueError):
        format_version = protocol_revision = -1
    if (
        (format_version not in {-1, 2})
        or (protocol_revision not in {-1, 2})
    ):
        raise _viewer.ReplayCompareError("source timeline has an incompatible format")
    if format_version == -1 or protocol_revision == -1:
        # Early protocol-2 outputs predate the redundant version scalars, but
        # their exact source_qpos sidecar bytes are still authenticated by the
        # completion certificate.  Use the certified manifest source hash.
        print(
            "[ghost-replay] note: legacy protocol-2 source timeline has no "
            "embedded version fields; run_complete + run_manifest certify it"
        )
    recorded_source_sha256 = decoded_metadata.get("source_csv_sha256")
    if (
        recorded_source_sha256 is not None
        and recorded_source_sha256 != actual_source_sha256
    ):
        raise _viewer.ReplayCompareError(
            "original source data.csv differs from source_timeline metadata: "
            f"expected {recorded_source_sha256!r}, got {actual_source_sha256}"
        )
    if source_rows.shape != (len(target.qpos),):
        raise _viewer.ReplayCompareError(
            f"source timeline shape {source_rows.shape} does not match "
            f"rollout rows {(len(target.qpos),)}"
        )
    if np.any(source_rows < 0) or np.any(source_rows >= len(source.qpos)):
        raise _viewer.ReplayCompareError("source timeline contains out-of-range rows")
    if np.any(np.diff(source_rows) < 0):
        raise _viewer.ReplayCompareError("source timeline rows are not monotonic")
    if source.policy_seq is None:
        raise _viewer.ReplayCompareError("source data.csv has no policy_seq")
    ghost_qpos = source.qpos[source_rows].copy()
    if source_qpos.shape != ghost_qpos.shape:
        raise _viewer.ReplayCompareError(
            f"source_qpos shape {source_qpos.shape} does not match "
            f"rollout/source qpos shape {ghost_qpos.shape}"
        )
    if not np.all(np.isfinite(source_qpos)):
        raise _viewer.ReplayCompareError("source_qpos contains NaN or infinity")
    ghost_qpos = source_qpos
    return _viewer.GhostTrack(
        qpos=ghost_qpos,
        source_row_index=source_rows,
        reference_frame_index=np.full(len(source_rows), -1, dtype=np.int64),
        policy_seq=source.policy_seq[source_rows],
        mode="source",
        description=(
            "original recording qpos interpolated on the exact per-sample "
            "phase-matched 400 Hz source clock saved by controller_replacement"
        ),
    )


def _print_summary(*args: Any, **kwargs: Any) -> None:
    """Reuse the viewer summary while replacing its protocol-v1-only note."""

    _revalidate_consumed_inputs(Path(kwargs["rollout_dir"]))

    captured = io.StringIO()
    with redirect_stdout(captured):
        _ORIGINAL_PRINT_SUMMARY(*args, **kwargs)
    text = captured.getvalue().replace(
        "recorded root xyz is drawn for world-space comparison, but protocol "
        "v1 did not send root xyz to the encoder.",
        "recorded root xyz is drawn for phase-matched world-space comparison; "
        "it is not automatically a controller input or tracking target.",
    )
    print(text, end="")
    staged_scene = _recorded_staged_scene()
    recorded_model = staged_scene.get("compiled_mujoco_model")
    if not isinstance(recorded_model, Mapping):
        raise _viewer.ReplayCompareError(
            "run manifest has no compiled MuJoCo model fingerprint"
        )
    current_model = compiled_mujoco_model_fingerprint(kwargs["model"])
    if recorded_model.get("mujoco_version") == current_model.get("mujoco_version"):
        if dict(recorded_model) != current_model:
            raise _viewer.ReplayCompareError(
                "replay compiled model differs from the certified rollout model"
            )
        print("[ghost-replay] compiled model fingerprint matches the rollout")
    else:
        print(
            "[ghost-replay] warning: rollout/replay MuJoCo versions differ "
            f"({recorded_model.get('mujoco_version')} vs "
            f"{current_model.get('mujoco_version')}); XML and recorded asset-root "
            "conditions were verified, but MJB bytes are version-specific"
        )


def main(argv: Sequence[str] | None = None) -> int:
    global _CERTIFIED_REPLAY_STATE, _CERTIFIED_RUN_MANIFEST

    args_list = list(sys.argv[1:] if argv is None else argv)
    parsed = _viewer.build_parser().parse_args(args_list)
    rollout_dir, _ = _viewer.resolve_recording(parsed.rollout)
    _CERTIFIED_RUN_MANIFEST = _validate_completion_certificate(
        rollout_dir, ghost_mode=parsed.ghost_mode
    )
    complete_path = _regular_file(
        rollout_dir / "run_complete.json", label="run_complete.json"
    )
    _CERTIFIED_REPLAY_STATE = {
        "rollout_dir": str(rollout_dir.resolve(strict=True)),
        "ghost_mode": parsed.ghost_mode,
        "run_complete_sha256": sha256_file(complete_path),
        "run_manifest": _CERTIFIED_RUN_MANIFEST,
    }
    # Ensure the certificate and the identity captured immediately afterward
    # did not straddle an atomic directory replacement.
    _revalidate_rollout_certificate(rollout_dir)
    _viewer.build_ghost_track = _build_ghost_track
    _viewer._print_summary = _print_summary
    _viewer.stage_recording_snapshot = _stage_recording_snapshot
    return _viewer.main(args_list)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (_viewer.ReplayCompareError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
