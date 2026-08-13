"""Deterministic, in-process adapters for the released SONIC controllers.

This module is a NumPy/ONNX Runtime translation of the G1 path in
``g1_deploy_onnx_ref.cpp``.  It deliberately does not use DDS, ZMQ, wall-clock
sampling, or the C++ state logger.  The reference and robot state are supplied
explicitly for one 50 Hz policy step.

There are two joint orders in this path and they must not be conflated:

* :class:`~controller_replacement.simulator.SceneState` and the returned PD
  command use the 29-joint MuJoCo/URDF order;
* SONIC reference arrays, decoder histories, raw actions, and both ONNX models
  use the interleaved IsaacLab order.

The regular encoder ONNX in ``change_ckpt/models/regular`` already contains
the historical training-time ``[q+dq] -> [10,58]`` reshape.  Consequently this
adapter supplies the canonical C++ input slice ``q[290], dq[290], ori[60]``;
it must not apply that reshape a second time.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from change_ckpt.reference_data import (
    quat_conjugate_wxyz,
    quat_multiply_wxyz,
    quat_to_matrix_wxyz,
)
from change_ckpt_track.qpos_reference_data import (
    G1_ISAACLAB_JOINT_NAMES,
    G1_MUJOCO_JOINT_NAMES,
)
from controller_replacement.references.base import ReferenceFrame


FloatArray = np.ndarray
REPO_ROOT = Path(__file__).resolve().parents[2]

POLICY_HZ = 50.0
HISTORY_LENGTH = 10
SOURCE_PREFILL_LENGTH = HISTORY_LENGTH - 1
NUM_JOINTS = 29
TOKEN_DIM = 64
DECODER_INPUT_DIM = 994

# policy_parameters.hpp.  Despite the names used in that C++ header, these
# are best read as an index lookup for the order named on the left here.
MUJOCO_INDEX_FOR_ISAACLAB = np.asarray(
    [
        0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
        16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
    ],
    dtype=np.int64,
)
ISAACLAB_INDEX_FOR_MUJOCO = np.asarray(
    [
        0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
        11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28,
    ],
    dtype=np.int64,
)

# G1 deploy constants, all in MuJoCo/URDF order.
DEFAULT_DOF_POS_MUJOCO = np.asarray(
    [
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        0.0, 0.0, 0.0,
        0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
        0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    ],
    dtype=np.float64,
)
DEFAULT_DOF_POS_ISAACLAB = DEFAULT_DOF_POS_MUJOCO[MUJOCO_INDEX_FOR_ISAACLAB]

_ARMATURE_5020 = 0.003609725
_ARMATURE_7520_14 = 0.010177520
_ARMATURE_7520_22 = 0.025101925
_ARMATURE_4010 = 0.00425
# Preserve the literal used by policy_parameters.hpp rather than silently
# changing the exported controller constants by a few final bits.
_NATURAL_FREQ = 10.0 * 2.0 * 3.1415926535
_DAMPING_RATIO = 2.0


def _stiffness(armature: float) -> float:
    return armature * _NATURAL_FREQ * _NATURAL_FREQ


def _damping(armature: float) -> float:
    return 2.0 * _DAMPING_RATIO * armature * _NATURAL_FREQ


_KP_5020 = _stiffness(_ARMATURE_5020)
_KP_7520_14 = _stiffness(_ARMATURE_7520_14)
_KP_7520_22 = _stiffness(_ARMATURE_7520_22)
_KP_4010 = _stiffness(_ARMATURE_4010)
_KD_5020 = _damping(_ARMATURE_5020)
_KD_7520_14 = _damping(_ARMATURE_7520_14)
_KD_7520_22 = _damping(_ARMATURE_7520_22)
_KD_4010 = _damping(_ARMATURE_4010)

KPS_MUJOCO = np.asarray(
    [
        _KP_7520_22, _KP_7520_22, _KP_7520_14, _KP_7520_22,
        2.0 * _KP_5020, 2.0 * _KP_5020,
        _KP_7520_22, _KP_7520_22, _KP_7520_14, _KP_7520_22,
        2.0 * _KP_5020, 2.0 * _KP_5020,
        _KP_7520_14, 2.0 * _KP_5020, 2.0 * _KP_5020,
        _KP_5020, _KP_5020, _KP_5020, _KP_5020, _KP_5020,
        _KP_4010, _KP_4010,
        _KP_5020, _KP_5020, _KP_5020, _KP_5020, _KP_5020,
        _KP_4010, _KP_4010,
    ],
    dtype=np.float64,
)
KDS_MUJOCO = np.asarray(
    [
        _KD_7520_22, _KD_7520_22, _KD_7520_14, _KD_7520_22,
        2.0 * _KD_5020, 2.0 * _KD_5020,
        _KD_7520_22, _KD_7520_22, _KD_7520_14, _KD_7520_22,
        2.0 * _KD_5020, 2.0 * _KD_5020,
        _KD_7520_14, 2.0 * _KD_5020, 2.0 * _KD_5020,
        _KD_5020, _KD_5020, _KD_5020, _KD_5020, _KD_5020,
        _KD_4010, _KD_4010,
        _KD_5020, _KD_5020, _KD_5020, _KD_5020, _KD_5020,
        _KD_4010, _KD_4010,
    ],
    dtype=np.float64,
)
# C++ stores the gain tables as ``float`` before publishing them.  The runner
# uses float64 MuJoCo arrays, so round through float32 here to preserve the
# exact native command values while avoiding mixed-dtype arithmetic later.
KPS_MUJOCO = KPS_MUJOCO.astype(np.float32).astype(np.float64)
KDS_MUJOCO = KDS_MUJOCO.astype(np.float32).astype(np.float64)
TORQUE_LIMITS_MUJOCO = np.asarray(
    [
        139.0, 139.0, 88.0, 139.0, 25.0, 25.0,
        139.0, 139.0, 88.0, 139.0, 25.0, 25.0,
        88.0, 25.0, 25.0,
        25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
        25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
    ],
    dtype=np.float64,
)
ACTION_SCALE_MUJOCO = 0.25 * TORQUE_LIMITS_MUJOCO / np.asarray(
    [
        _KP_7520_22, _KP_7520_22, _KP_7520_14, _KP_7520_22,
        _KP_5020, _KP_5020,
        _KP_7520_22, _KP_7520_22, _KP_7520_14, _KP_7520_22,
        _KP_5020, _KP_5020,
        _KP_7520_14, _KP_5020, _KP_5020,
        _KP_5020, _KP_5020, _KP_5020, _KP_5020, _KP_5020,
        _KP_4010, _KP_4010,
        _KP_5020, _KP_5020, _KP_5020, _KP_5020, _KP_5020,
        _KP_4010, _KP_4010,
    ],
    dtype=np.float64,
)


class SonicControllerError(RuntimeError):
    """A model, state, reference, or history violates the SONIC contract."""


class SonicVariant(str, Enum):
    REGULAR = "regular"
    LOW_LATENCY = "low_latency"
    SONIC_V1_1 = "sonic_v1_1"

    @classmethod
    def parse(cls, value: "SonicVariant | str") -> "SonicVariant":
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower().replace("-", "_")
        aliases = {
            "regular": cls.REGULAR,
            "low_latency": cls.LOW_LATENCY,
            "lowlatency": cls.LOW_LATENCY,
            "sonic_v1_1": cls.SONIC_V1_1,
            "sonic_v11": cls.SONIC_V1_1,
            "v1_1": cls.SONIC_V1_1,
            "v1.1": cls.SONIC_V1_1,
        }
        try:
            return aliases[normalized]
        except KeyError as exc:
            raise SonicControllerError(f"unsupported SONIC variant {value!r}") from exc


@dataclass(frozen=True)
class SonicModelSpec:
    variant: SonicVariant
    encoder_path: Path
    decoder_path: Path
    encoder_input_dim: int
    orientation_mode: str
    reference_view: str


def default_model_spec(variant: SonicVariant | str) -> SonicModelSpec:
    parsed = SonicVariant.parse(variant)
    if parsed is SonicVariant.REGULAR:
        directory = REPO_ROOT / "change_ckpt/models/regular"
        dimension = 1751
        orientation_mode = "full_base"
        view = "regular"
    elif parsed is SonicVariant.LOW_LATENCY:
        directory = REPO_ROOT / "change_ckpt/models/low_latency"
        dimension = 1247
        orientation_mode = "full_base"
        view = "consecutive"
    else:
        directory = REPO_ROOT / "change_ckpt/models/v1.1"
        dimension = 1751
        orientation_mode = "robot_heading"
        view = "regular"
    return SonicModelSpec(
        variant=parsed,
        encoder_path=directory / "model_encoder.onnx",
        decoder_path=directory / "model_decoder.onnx",
        encoder_input_dim=dimension,
        orientation_mode=orientation_mode,
        reference_view=view,
    )


class _ValueInfo(Protocol):
    name: str
    shape: Sequence[Any]
    type: str


class _Session(Protocol):
    def get_inputs(self) -> Sequence[_ValueInfo]: ...
    def get_outputs(self) -> Sequence[_ValueInfo]: ...
    def run(self, output_names: Sequence[str] | None, inputs: Mapping[str, Any]) -> list[Any]: ...


def _shape_tuple(value: Sequence[Any]) -> tuple[int | str | None, ...]:
    return tuple(item if isinstance(item, (int, str)) or item is None else str(item) for item in value)


def _validate_session(
    session: _Session,
    *,
    path: Path,
    expected_input: int,
    expected_output: int,
    expected_output_name: str,
) -> tuple[str, str]:
    inputs = list(session.get_inputs())
    outputs = list(session.get_outputs())
    if len(inputs) != 1 or len(outputs) != 1:
        raise SonicControllerError(
            f"{path} must have one input and one output; got {len(inputs)}/{len(outputs)}"
        )
    input_shape = _shape_tuple(inputs[0].shape)
    output_shape = _shape_tuple(outputs[0].shape)
    if input_shape != (1, expected_input):
        raise SonicControllerError(
            f"{path} input shape is {input_shape}; expected {(1, expected_input)}"
        )
    if output_shape != (1, expected_output):
        raise SonicControllerError(
            f"{path} output shape is {output_shape}; expected {(1, expected_output)}"
        )
    if inputs[0].type != "tensor(float)" or outputs[0].type != "tensor(float)":
        raise SonicControllerError(
            f"{path} must use float tensors; got {inputs[0].type}/{outputs[0].type}"
        )
    if outputs[0].name != expected_output_name:
        raise SonicControllerError(
            f"{path} output is {outputs[0].name!r}; expected {expected_output_name!r}"
        )
    return inputs[0].name, outputs[0].name


def _make_ort_session(path: Path, device: str) -> _Session:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SonicControllerError(
            "onnxruntime is required for SONIC inference; install it in the "
            "controller_replacement runtime environment"
        ) from exc
    if not path.is_file():
        raise FileNotFoundError(path)
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
            raise SonicControllerError(
                "CUDAExecutionProvider was requested but is not available"
            )
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        raise SonicControllerError("device must be 'cpu' or 'cuda'")
    return ort.InferenceSession(str(path), sess_options=options, providers=providers)


def _finite_vector(value: Any, size: int, name: str) -> FloatArray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (size,):
        raise SonicControllerError(f"{name} has shape {result.shape}; expected {(size,)}")
    if not np.all(np.isfinite(result)):
        raise SonicControllerError(f"{name} contains NaN or infinity")
    return result


def _unit_quaternion(value: Any, name: str) -> FloatArray:
    result = _finite_vector(value, 4, name)
    norm = float(np.linalg.norm(result))
    if norm < 1e-8:
        raise SonicControllerError(f"{name} has zero norm")
    if abs(norm - 1.0) > 2e-4:
        raise SonicControllerError(f"{name} is not unit-normalized (norm={norm:g})")
    return result / norm


def _heading_quaternion(quaternion_wxyz: Any) -> FloatArray:
    quaternion = _unit_quaternion(quaternion_wxyz, "robot root quaternion")
    # This matches calc_heading_d(): rotate world +X and take atan2(y, x).
    w, x, y, z = quaternion
    direction_x = 1.0 - 2.0 * (y * y + z * z)
    direction_y = 2.0 * (x * y + w * z)
    yaw = math.atan2(direction_y, direction_x)
    return np.asarray(
        [math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)],
        dtype=np.float64,
    )


def _gravity_in_body(quaternion_wxyz: Any) -> FloatArray:
    quaternion = _unit_quaternion(quaternion_wxyz, "history root quaternion")
    # quat_rotate(conjugate(q), [0, 0, -1]), written explicitly to avoid a
    # dependency on a simulator-specific quaternion helper.
    w, x, y, z = quaternion
    inverse = np.asarray([w, -x, -y, -z], dtype=np.float64)
    q_w = inverse[0]
    q_vec = inverse[1:]
    vector = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
    return (
        vector * (2.0 * q_w * q_w - 1.0)
        + np.cross(q_vec, vector) * (2.0 * q_w)
        + q_vec * (2.0 * float(np.dot(q_vec, vector)))
    )


def _orientation_6d(robot_quat_wxyz: Any, reference_quat_wxyz: Any, mode: str) -> FloatArray:
    robot = _unit_quaternion(robot_quat_wxyz, "robot root quaternion")
    reference = np.asarray(reference_quat_wxyz, dtype=np.float64)
    if reference.shape != (10, 4) or not np.all(np.isfinite(reference)):
        raise SonicControllerError(
            f"reference anchor quaternions have shape {reference.shape}; expected (10, 4)"
        )
    norms = np.linalg.norm(reference, axis=1, keepdims=True)
    if np.any(norms < 1e-8) or float(np.max(np.abs(norms - 1.0))) > 2e-4:
        raise SonicControllerError("reference anchor quaternions are not unit-normalized")
    reference = reference / norms
    left = _heading_quaternion(robot) if mode == "robot_heading" else robot
    relative = quat_multiply_wxyz(quat_conjugate_wxyz(left), reference)
    matrices = quat_to_matrix_wxyz(relative)
    return np.asarray(matrices[..., :2].reshape(10, 6), dtype=np.float32)


@dataclass(frozen=True)
class SonicHistoryFrame:
    """One decoder history row, with joint/action values in IsaacLab order."""

    body_q: FloatArray
    body_dq: FloatArray
    root_quat_wxyz: FloatArray
    root_ang_vel_b: FloatArray
    last_action: FloatArray

    def __post_init__(self) -> None:
        values = {
            "body_q": _finite_vector(self.body_q, 29, "history body_q"),
            "body_dq": _finite_vector(self.body_dq, 29, "history body_dq"),
            "root_quat_wxyz": _unit_quaternion(
                self.root_quat_wxyz, "history root quaternion"
            ),
            "root_ang_vel_b": _finite_vector(
                self.root_ang_vel_b, 3, "history root angular velocity"
            ),
            "last_action": _finite_vector(
                self.last_action, 29, "history last action"
            ),
        }
        for name, value in values.items():
            copied = np.asarray(value, dtype=np.float32).copy()
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)

    @property
    def gravity_dir(self) -> FloatArray:
        return _gravity_in_body(self.root_quat_wxyz).astype(np.float32)


@dataclass(frozen=True)
class SonicInference:
    """All native values produced/consumed by one SONIC policy inference."""

    variant: SonicVariant
    encoder_input: FloatArray
    decoder_input: FloatArray
    history: FloatArray
    token: FloatArray
    last_action: FloatArray
    raw_action: FloatArray
    q_target: FloatArray
    q_target_isaaclab: FloatArray
    kp: FloatArray
    kd: FloatArray
    torque_limit: FloatArray
    received_dof_pos: FloatArray

    @property
    def observation(self) -> FloatArray:
        """Controller-native decoder observation (994 values)."""

        return self.decoder_input


def action_to_q_target(raw_action_isaaclab: Any) -> FloatArray:
    """Decode one SONIC action into absolute MuJoCo-order joint targets."""

    action = _finite_vector(raw_action_isaaclab, NUM_JOINTS, "raw action")
    return (
        DEFAULT_DOF_POS_MUJOCO
        + action[ISAACLAB_INDEX_FOR_MUJOCO] * ACTION_SCALE_MUJOCO
    ).astype(np.float64)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SonicController:
    """In-process regular, low-latency, or v1.1 SONIC controller."""

    controller_family = "sonic"
    joint_names = G1_MUJOCO_JOINT_NAMES
    policy_hz = POLICY_HZ
    history_length = HISTORY_LENGTH

    def __init__(
        self,
        variant: SonicVariant | str,
        *,
        encoder_path: str | Path | None = None,
        decoder_path: str | Path | None = None,
        device: str = "cpu",
        encoder_session: _Session | None = None,
        decoder_session: _Session | None = None,
        require_source_history: bool = True,
        validate_prefill_current_state: bool = True,
    ) -> None:
        default = default_model_spec(variant)
        self.spec = SonicModelSpec(
            variant=default.variant,
            encoder_path=(
                default.encoder_path
                if encoder_path is None
                else Path(encoder_path).expanduser().resolve()
            ),
            decoder_path=(
                default.decoder_path
                if decoder_path is None
                else Path(decoder_path).expanduser().resolve()
            ),
            encoder_input_dim=default.encoder_input_dim,
            orientation_mode=default.orientation_mode,
            reference_view=default.reference_view,
        )
        self.require_source_history = bool(require_source_history)
        self.validate_prefill_current_state = bool(validate_prefill_current_state)
        self.encoder_session = (
            _make_ort_session(self.spec.encoder_path, device)
            if encoder_session is None
            else encoder_session
        )
        self.decoder_session = (
            _make_ort_session(self.spec.decoder_path, device)
            if decoder_session is None
            else decoder_session
        )
        self.encoder_input_name, self.encoder_output_name = _validate_session(
            self.encoder_session,
            path=self.spec.encoder_path,
            expected_input=self.spec.encoder_input_dim,
            expected_output=TOKEN_DIM,
            expected_output_name="encoded_tokens",
        )
        self.decoder_input_name, self.decoder_output_name = _validate_session(
            self.decoder_session,
            path=self.spec.decoder_path,
            expected_input=DECODER_INPUT_DIM,
            expected_output=NUM_JOINTS,
            expected_output_name="action",
        )
        self._history: deque[SonicHistoryFrame] = deque(maxlen=HISTORY_LENGTH)
        self._pending_current_last_action: FloatArray | None = None
        self._expected_current: Mapping[str, Any] | None = None
        self._inference_count = 0

    @property
    def name(self) -> str:
        return self.spec.variant.value

    @property
    def kp(self) -> FloatArray:
        return KPS_MUJOCO.copy()

    @property
    def kd(self) -> FloatArray:
        return KDS_MUJOCO.copy()

    @property
    def torque_limit(self) -> FloatArray:
        return TORQUE_LIMITS_MUJOCO.copy()

    @property
    def inference_count(self) -> int:
        return self._inference_count

    def metadata(self) -> dict[str, Any]:
        return {
            "controller_family": self.controller_family,
            "controller_name": self.name,
            "variant": self.spec.variant.value,
            "policy_hz": POLICY_HZ,
            "history_length": HISTORY_LENGTH,
            "joint_order_external": "g1_mujoco_urdf_29dof",
            "joint_names_external": list(G1_MUJOCO_JOINT_NAMES),
            "joint_order_internal": "g1_isaaclab_29dof",
            "joint_names_internal": list(G1_ISAACLAB_JOINT_NAMES),
            "reference_view": self.spec.reference_view,
            "orientation_mode": self.spec.orientation_mode,
            "heading_reinitialisation": False,
            "encoder": {
                "path": str(self.spec.encoder_path),
                "sha256": _sha256(self.spec.encoder_path),
                "input_dimension": self.spec.encoder_input_dim,
                "output_dimension": TOKEN_DIM,
            },
            "decoder": {
                "path": str(self.spec.decoder_path),
                "sha256": _sha256(self.spec.decoder_path),
                "input_dimension": DECODER_INPUT_DIM,
                "output_dimension": NUM_JOINTS,
            },
            "source_history_required": self.require_source_history,
        }

    def reset(
        self,
        source_history_prefill: Mapping[str, Any] | str | Path | None = None,
    ) -> None:
        """Clear controller state and optionally load the nine source frames.

        Formal experiments pass the version-1 payload produced by
        ``build_source_history_prefill``.  No implicit zero/repeated-state
        padding is performed: when ``require_source_history`` is true, the
        first call to :meth:`infer` refuses an incomplete history.
        """

        self._history.clear()
        self._pending_current_last_action = None
        self._expected_current = None
        self._inference_count = 0
        if source_history_prefill is not None:
            self.prefill_source_history(source_history_prefill)

    def prefill_history(
        self,
        frames: Sequence[SonicHistoryFrame],
        *,
        current_last_action: Any,
        expected_current: Mapping[str, Any] | None = None,
    ) -> None:
        """Install nine explicit oldest-to-newest history frames."""

        if len(frames) != SOURCE_PREFILL_LENGTH:
            raise SonicControllerError(
                f"source history must contain exactly {SOURCE_PREFILL_LENGTH} frames"
            )
        self._history.clear()
        for frame in frames:
            if not isinstance(frame, SonicHistoryFrame):
                raise SonicControllerError("source history contains a non-SonicHistoryFrame")
            self._history.append(frame)
        self._pending_current_last_action = _finite_vector(
            current_last_action, 29, "source current last action"
        ).astype(np.float32, copy=True)
        self._expected_current = expected_current

    def prefill_source_history(
        self, payload: Mapping[str, Any] | str | Path
    ) -> None:
        """Load the existing validated source-history JSON contract."""

        if isinstance(payload, (str, Path)):
            path = Path(payload).expanduser().resolve()
            with path.open("r", encoding="utf-8") as stream:
                value = json.load(stream)
        else:
            value = dict(payload)
        if value.get("format") != "g1_decoder_source_history_prefill" or value.get("version") != 1:
            raise SonicControllerError("unsupported source-history prefill format/version")
        if value.get("history_order") != "oldest_to_newest":
            raise SonicControllerError("source-history entries must be oldest_to_newest")
        entries = value.get("entries")
        if not isinstance(entries, list) or len(entries) != SOURCE_PREFILL_LENGTH:
            raise SonicControllerError(
                f"source-history prefill must contain exactly {SOURCE_PREFILL_LENGTH} entries"
            )
        if int(value.get("history_entry_count", -1)) != SOURCE_PREFILL_LENGTH:
            raise SonicControllerError("source-history entry count metadata is inconsistent")
        frames = [
            SonicHistoryFrame(
                body_q=entry["body_q"],
                body_dq=entry["body_dq"],
                root_quat_wxyz=entry["base_quat"],
                root_ang_vel_b=entry["base_ang_vel"],
                last_action=entry["last_action"],
            )
            for entry in entries
        ]
        sequences = [int(entry["policy_seq"]) for entry in entries]
        if any(right != left + 1 for left, right in zip(sequences, sequences[1:])):
            raise SonicControllerError("source-history policy_seq values are not contiguous")
        current = value.get("current")
        if not isinstance(current, Mapping):
            raise SonicControllerError("source-history payload has no current snapshot")
        if int(current.get("policy_seq", -1)) != sequences[-1] + 1:
            raise SonicControllerError("source-history current policy_seq is not contiguous")
        self.prefill_history(
            frames,
            current_last_action=current["last_action"],
            expected_current=current,
        )

    @staticmethod
    def _state_arrays(state: Any) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        q_mujoco = _finite_vector(state.joint_pos, 29, "robot joint_pos")
        dq_mujoco = _finite_vector(state.joint_vel, 29, "robot joint_vel")
        root_quat = _unit_quaternion(state.root_quat_wxyz, "robot root quaternion")
        root_ang_vel = _finite_vector(
            state.root_ang_vel_b, 3, "robot root angular velocity"
        )
        return q_mujoco, dq_mujoco, root_quat, root_ang_vel

    def _validate_expected_current(self, state: Any) -> None:
        if self._expected_current is None or not self.validate_prefill_current_state:
            return
        q_mujoco, dq_mujoco, root_quat, root_ang_vel = self._state_arrays(state)
        expected = self._expected_current
        q_relative = q_mujoco[MUJOCO_INDEX_FOR_ISAACLAB] - DEFAULT_DOF_POS_ISAACLAB
        dq_isaaclab = dq_mujoco[MUJOCO_INDEX_FOR_ISAACLAB]
        expected_quat = _unit_quaternion(expected["base_quat"], "expected root quaternion")
        quat_error = min(
            float(np.max(np.abs(root_quat - expected_quat))),
            float(np.max(np.abs(root_quat + expected_quat))),
        )
        errors = {
            "quat": quat_error,
            "angular_velocity": float(
                np.max(np.abs(root_ang_vel - _finite_vector(expected["base_ang_vel"], 3, "expected angular velocity")))
            ),
            "q": float(
                np.max(np.abs(q_relative - _finite_vector(expected["body_q"], 29, "expected body_q")))
            ),
            "dq": float(
                np.max(np.abs(dq_isaaclab - _finite_vector(expected["body_dq"], 29, "expected body_dq")))
            ),
        }
        if errors["quat"] > 1e-4 or errors["q"] > 1e-4 or errors["angular_velocity"] > 2e-3 or errors["dq"] > 2e-3:
            raise SonicControllerError(
                "live first state does not match source-history current snapshot: "
                + ", ".join(f"{name}={value:.6g}" for name, value in errors.items())
            )

    def build_encoder_input(self, state: Any, reference: ReferenceFrame) -> FloatArray:
        """Construct the full multiplexed encoder tensor for G1 mode 0."""

        _, _, root_quat, _ = self._state_arrays(state)
        if self.spec.reference_view == "regular":
            positions = np.asarray(reference.sonic_regular_joint_pos, dtype=np.float32)
            velocities = np.asarray(reference.sonic_regular_joint_vel, dtype=np.float32)
            quaternions = np.asarray(
                reference.sonic_regular_anchor_quat_wxyz, dtype=np.float32
            )
        else:
            positions = np.asarray(reference.sonic_consecutive_joint_pos, dtype=np.float32)
            velocities = np.asarray(reference.sonic_consecutive_joint_vel, dtype=np.float32)
            quaternions = np.asarray(
                reference.sonic_consecutive_anchor_quat_wxyz, dtype=np.float32
            )
        if positions.shape != (10, 29) or velocities.shape != (10, 29):
            raise SonicControllerError(
                f"SONIC reference q/dq shapes are {positions.shape}/{velocities.shape}; expected (10, 29)"
            )
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
            raise SonicControllerError("SONIC reference q/dq contains NaN or infinity")
        orientation = _orientation_6d(
            root_quat, quaternions, self.spec.orientation_mode
        )
        result = np.zeros(self.spec.encoder_input_dim, dtype=np.float32)
        # encoder_mode_4 for G1 mode 0 is [0, 0, 0, 0].  The remaining encoder
        # branches stay zero because the ONNX is a shared multiplexed model.
        result[4:294] = positions.reshape(-1)
        result[294:584] = velocities.reshape(-1)
        result[584:644] = orientation.reshape(-1)
        return result

    @staticmethod
    def build_decoder_input(token: Any, history: Sequence[SonicHistoryFrame]) -> FloatArray:
        """Build the exact C++ observation order: 64 + 30 + 290*3 + 30."""

        token_array = _finite_vector(token, TOKEN_DIM, "SONIC token")
        if len(history) != HISTORY_LENGTH:
            raise SonicControllerError(
                f"decoder needs {HISTORY_LENGTH} history frames; got {len(history)}"
            )
        angular_velocity = np.stack([frame.root_ang_vel_b for frame in history])
        body_q = np.stack([frame.body_q for frame in history])
        body_dq = np.stack([frame.body_dq for frame in history])
        actions = np.stack([frame.last_action for frame in history])
        gravity = np.stack([frame.gravity_dir for frame in history])
        result = np.concatenate(
            (
                token_array,
                angular_velocity.reshape(-1),
                body_q.reshape(-1),
                body_dq.reshape(-1),
                actions.reshape(-1),
                gravity.reshape(-1),
            )
        ).astype(np.float32)
        if result.shape != (DECODER_INPUT_DIM,):
            raise AssertionError(f"internal decoder input shape error: {result.shape}")
        return result

    @staticmethod
    def history_matrix(history: Sequence[SonicHistoryFrame]) -> FloatArray:
        """Return an analysis-friendly oldest-to-newest ``[10,93]`` view."""

        return np.stack(
            [
                np.concatenate(
                    (
                        frame.root_ang_vel_b,
                        frame.body_q,
                        frame.body_dq,
                        frame.last_action,
                        frame.gravity_dir,
                    )
                )
                for frame in history
            ]
        ).astype(np.float32)

    def infer(self, state: Any, reference: ReferenceFrame) -> SonicInference:
        """Run one 50 Hz encoder/decoder step and advance controller history."""

        q_mujoco, dq_mujoco, root_quat, root_ang_vel = self._state_arrays(state)
        try:
            left_hand_pos = _finite_vector(
                state.left_hand_pos, 7, "robot left hand position"
            )
            right_hand_pos = _finite_vector(
                state.right_hand_pos, 7, "robot right hand position"
            )
        except AttributeError as exc:
            raise SonicControllerError(
                "SONIC telemetry requires both seven-joint hand snapshots"
            ) from exc
        # The legacy CSV is *not* raw XML qpos[7:50] order.  The policy
        # receiver stores the 29 body joints first, followed by left and right
        # hands.  Construct that explicitly instead of forwarding the scene's
        # convenient XML-order snapshot.
        received_dof_pos = np.concatenate(
            (q_mujoco, left_hand_pos, right_hand_pos)
        )
        if self._inference_count == 0:
            self._validate_expected_current(state)
        if self._pending_current_last_action is not None:
            last_action = self._pending_current_last_action.copy()
            self._pending_current_last_action = None
        elif self._history:
            # After the first inference ``self._last_raw_action`` is always set.
            last_action = self._last_raw_action.copy()
        else:
            last_action = np.zeros(29, dtype=np.float32)

        current = SonicHistoryFrame(
            body_q=(
                q_mujoco[MUJOCO_INDEX_FOR_ISAACLAB] - DEFAULT_DOF_POS_ISAACLAB
            ),
            body_dq=dq_mujoco[MUJOCO_INDEX_FOR_ISAACLAB],
            root_quat_wxyz=root_quat,
            root_ang_vel_b=root_ang_vel,
            last_action=last_action,
        )
        self._history.append(current)
        if len(self._history) != HISTORY_LENGTH:
            if self.require_source_history:
                # Undo the mutation so a caller can still load a valid prefill.
                self._history.pop()
                raise SonicControllerError(
                    "formal SONIC inference requires source-history prefill; "
                    f"only {len(self._history)} prior frames are available"
                )
            zero = SonicHistoryFrame(
                body_q=np.zeros(29),
                body_dq=np.zeros(29),
                root_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
                root_ang_vel_b=np.zeros(3),
                last_action=np.zeros(29),
            )
            while len(self._history) < HISTORY_LENGTH:
                self._history.appendleft(zero)

        encoder_input = self.build_encoder_input(state, reference)
        token_output = self.encoder_session.run(
            [self.encoder_output_name],
            {self.encoder_input_name: encoder_input[None]},
        )[0]
        token = np.asarray(token_output, dtype=np.float32)
        if token.shape != (1, TOKEN_DIM) or not np.all(np.isfinite(token)):
            raise SonicControllerError(
                f"encoder returned shape {token.shape}; expected {(1, TOKEN_DIM)}"
            )
        token = token[0].copy()
        history = tuple(self._history)
        decoder_input = self.build_decoder_input(token, history)
        action_output = self.decoder_session.run(
            [self.decoder_output_name],
            {self.decoder_input_name: decoder_input[None]},
        )[0]
        action = np.asarray(action_output, dtype=np.float32)
        if action.shape != (1, NUM_JOINTS) or not np.all(np.isfinite(action)):
            raise SonicControllerError(
                f"decoder returned shape {action.shape}; expected {(1, NUM_JOINTS)}"
            )
        raw_action = action[0].copy()
        q_target = action_to_q_target(raw_action)
        self._last_raw_action = raw_action.copy()
        self._inference_count += 1
        self._expected_current = None
        return SonicInference(
            variant=self.spec.variant,
            encoder_input=encoder_input,
            decoder_input=decoder_input,
            history=self.history_matrix(history),
            token=token,
            last_action=last_action.astype(np.float32, copy=True),
            raw_action=raw_action,
            q_target=q_target,
            q_target_isaaclab=q_target[MUJOCO_INDEX_FOR_ISAACLAB].copy(),
            kp=KPS_MUJOCO.copy(),
            kd=KDS_MUJOCO.copy(),
            torque_limit=TORQUE_LIMITS_MUJOCO.copy(),
            received_dof_pos=received_dof_pos.astype(np.float32, copy=True),
        )


__all__ = [
    "ACTION_SCALE_MUJOCO",
    "DECODER_INPUT_DIM",
    "DEFAULT_DOF_POS_ISAACLAB",
    "DEFAULT_DOF_POS_MUJOCO",
    "HISTORY_LENGTH",
    "ISAACLAB_INDEX_FOR_MUJOCO",
    "KDS_MUJOCO",
    "KPS_MUJOCO",
    "MUJOCO_INDEX_FOR_ISAACLAB",
    "SonicController",
    "SonicControllerError",
    "SonicHistoryFrame",
    "SonicInference",
    "SonicModelSpec",
    "SonicVariant",
    "TOKEN_DIM",
    "TORQUE_LIMITS_MUJOCO",
    "action_to_q_target",
    "default_model_spec",
]
