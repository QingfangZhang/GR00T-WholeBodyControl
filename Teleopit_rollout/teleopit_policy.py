"""Minimal Teleopit v0.5.0 observation and ONNX inference adapter.

This module intentionally implements only the inference surface required by
the recorded task rollout. The formulas follow Teleopit v0.5.0
``observation.py``, ``reference_processing.py`` and ``rl_policy.py``. See
``THIRD_PARTY_NOTICE.md`` for provenance.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np

try:  # Package imports for tests; script imports for direct CLI execution.
    from .constants import (
        ACTION_DIM,
        ACTION_SCALE,
        DEFAULT_DOF_POS,
        G1_JOINT_NAMES,
        HISTORY_LENGTH,
        OBSERVATION_DIM,
        POLICY_HZ,
    )
except ImportError:  # pragma: no cover - exercised by direct script execution
    from constants import (
        ACTION_DIM,
        ACTION_SCALE,
        DEFAULT_DOF_POS,
        G1_JOINT_NAMES,
        HISTORY_LENGTH,
        OBSERVATION_DIM,
        POLICY_HZ,
    )


FloatArray = np.ndarray
GRAVITY_UNIT_W = np.asarray([0.0, 0.0, -1.0], dtype=np.float32)


def quat_inverse(q: FloatArray) -> FloatArray:
    value = np.asarray(q, dtype=np.float32).copy()
    value[..., 1:] *= -1.0
    return value


def quat_multiply(q1: FloatArray, q2: FloatArray) -> FloatArray:
    a = np.asarray(q1, dtype=np.float32)
    b = np.asarray(q2, dtype=np.float32)
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    ).astype(np.float32)


def quat_rotate(q: FloatArray, vector: FloatArray) -> FloatArray:
    quat = normalize_quaternion(q)
    vec = np.asarray(vector, dtype=np.float32)
    vec_quat = np.zeros((*vec.shape[:-1], 4), dtype=np.float32)
    vec_quat[..., 1:4] = vec
    return quat_multiply(
        quat_multiply(quat, vec_quat), quat_inverse(quat)
    )[..., 1:4]


def normalize_quaternion(q: FloatArray) -> FloatArray:
    value = np.asarray(q, dtype=np.float32)
    if value.shape[-1] != 4 or not np.all(np.isfinite(value)):
        raise ValueError(f"invalid wxyz quaternion: shape={value.shape}")
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("zero-length quaternion")
    return np.asarray(value / norm, dtype=np.float32)


def quat_to_rot6d(q: FloatArray) -> FloatArray:
    """Use Teleopit's exact six-value matrix layout."""

    w, x, y, z = normalize_quaternion(q).reshape(4)
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - w * z)
    r10 = 2.0 * (x * y + w * z)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r20 = 2.0 * (x * z - w * y)
    r21 = 2.0 * (y * z + w * x)
    return np.asarray([r00, r01, r10, r11, r20, r21], dtype=np.float32)


@dataclass(frozen=True)
class RobotState:
    joint_pos: FloatArray
    joint_vel: FloatArray
    root_pos: FloatArray
    root_quat_wxyz: FloatArray
    root_ang_vel_b: FloatArray
    timestamp_s: float


@dataclass(frozen=True)
class ReferenceFeatures:
    qpos36: FloatArray
    joint_vel: FloatArray
    anchor_lin_vel_w: FloatArray
    anchor_ang_vel_w: FloatArray


class TeleopitObservationBuilder:
    """Build the exact 167D ``velcmd_history`` observation."""

    def __init__(self, robot_xml: str | Path, anchor_body: str = "torso_link") -> None:
        path = Path(robot_xml).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Teleopit FK robot XML not found: {path}")
        self.robot_xml = path
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self._neutral_qpos = self.data.qpos.copy()
        self._joint_qpos_addresses = self._addresses(G1_JOINT_NAMES, qpos=True)
        try:
            self._anchor_body_id = self.model.body(anchor_body).id
        except KeyError as exc:
            raise ValueError(f"anchor body {anchor_body!r} missing from {path}") from exc

    def _addresses(self, names: Sequence[str], *, qpos: bool) -> np.ndarray:
        addresses: list[int] = []
        for name in names:
            try:
                joint_id = self.model.joint(name).id
            except KeyError as exc:
                raise ValueError(f"joint {name!r} missing from {self.robot_xml}") from exc
            addresses.append(
                int(
                    self.model.jnt_qposadr[joint_id]
                    if qpos
                    else self.model.jnt_dofadr[joint_id]
                )
            )
        return np.asarray(addresses, dtype=np.int32)

    def _anchor_pose(self, qpos36: FloatArray) -> tuple[FloatArray, FloatArray]:
        motion = np.asarray(qpos36, dtype=np.float64).reshape(-1)
        if motion.shape != (36,) or not np.all(np.isfinite(motion)):
            raise ValueError(f"reference qpos must be finite 36D, got {motion.shape}")
        self.data.qpos[:] = self._neutral_qpos
        self.data.qpos[0:3] = motion[0:3]
        self.data.qpos[3:7] = normalize_quaternion(motion[3:7]).astype(np.float64)
        self.data.qpos[self._joint_qpos_addresses] = motion[7:36]
        mujoco.mj_kinematics(self.model, self.data)
        return (
            np.asarray(self.data.xpos[self._anchor_body_id], dtype=np.float32).copy(),
            np.asarray(self.data.xquat[self._anchor_body_id], dtype=np.float32).copy(),
        )

    def reference_features(
        self,
        current_qpos36: FloatArray,
        previous_qpos36: FloatArray | None,
    ) -> ReferenceFeatures:
        current = np.asarray(current_qpos36, dtype=np.float32).reshape(36)
        current_anchor_pos, current_anchor_quat = self._anchor_pose(current)
        if previous_qpos36 is None:
            zeros3 = np.zeros(3, dtype=np.float32)
            return ReferenceFeatures(
                current.copy(),
                np.zeros(ACTION_DIM, dtype=np.float32),
                zeros3,
                zeros3.copy(),
            )

        previous = np.asarray(previous_qpos36, dtype=np.float32).reshape(36)
        previous_anchor_pos, previous_anchor_quat = self._anchor_pose(previous)
        joint_vel = np.asarray(
            (current[7:36] - previous[7:36]) * np.float32(POLICY_HZ),
            dtype=np.float32,
        )
        anchor_lin_vel = np.asarray(
            (current_anchor_pos - previous_anchor_pos) * np.float32(POLICY_HZ),
            dtype=np.float32,
        )
        delta = quat_multiply(current_anchor_quat, quat_inverse(previous_anchor_quat))
        if float(delta[0]) < 0.0:
            delta = -delta
        half_angle = np.float32(np.arccos(float(np.clip(delta[0], -1.0, 1.0))))
        sin_half = np.float32(np.sin(half_angle))
        if sin_half > 1e-6:
            axis = delta[1:4] / sin_half
            anchor_ang_vel = np.asarray(
                axis * 2.0 * half_angle * np.float32(POLICY_HZ),
                dtype=np.float32,
            )
        else:
            anchor_ang_vel = np.zeros(3, dtype=np.float32)
        return ReferenceFeatures(
            current.copy(), joint_vel, anchor_lin_vel, anchor_ang_vel
        )

    def build(
        self,
        robot: RobotState,
        reference: ReferenceFeatures,
        previous_raw_action: FloatArray,
    ) -> FloatArray:
        robot_joint_pos = np.asarray(robot.joint_pos, dtype=np.float32).reshape(ACTION_DIM)
        robot_joint_vel = np.asarray(robot.joint_vel, dtype=np.float32).reshape(ACTION_DIM)
        robot_quat = normalize_quaternion(robot.root_quat_wxyz).reshape(4)
        robot_ang_vel = np.asarray(robot.root_ang_vel_b, dtype=np.float32).reshape(3)
        previous_action = np.asarray(previous_raw_action, dtype=np.float32).reshape(ACTION_DIM)

        ref = np.asarray(reference.qpos36, dtype=np.float32).reshape(36)
        ref_anchor_pos, ref_anchor_quat = self._anchor_pose(ref)
        robot_fk = np.concatenate(
            (
                np.asarray(robot.root_pos, dtype=np.float32).reshape(3),
                robot_quat,
                robot_joint_pos,
            )
        )
        _, robot_anchor_quat = self._anchor_pose(robot_fk)

        relative_anchor_quat = quat_multiply(
            quat_inverse(robot_anchor_quat), ref_anchor_quat
        )
        robot_projected_gravity = quat_rotate(quat_inverse(robot_quat), GRAVITY_UNIT_W)
        robot_anchor_inverse = quat_inverse(robot_anchor_quat)
        ref_anchor_lin_vel_b = quat_rotate(
            robot_anchor_inverse, reference.anchor_lin_vel_w
        )
        ref_anchor_ang_vel_b = quat_rotate(
            robot_anchor_inverse, reference.anchor_ang_vel_w
        )
        ref_projected_gravity = quat_rotate(
            quat_inverse(ref_anchor_quat), GRAVITY_UNIT_W
        )

        observation = np.concatenate(
            [
                ref[7:36],
                np.asarray(reference.joint_vel, dtype=np.float32),
                quat_to_rot6d(relative_anchor_quat),
                robot_ang_vel,
                robot_joint_pos - DEFAULT_DOF_POS,
                robot_joint_vel,
                previous_action,
                robot_projected_gravity,
                ref_anchor_lin_vel_b,
                ref_anchor_ang_vel_b,
                ref_projected_gravity,
                ref_anchor_pos[2:3],
            ],
            dtype=np.float32,
        )
        if observation.shape != (OBSERVATION_DIM,):
            raise AssertionError(
                f"Teleopit observation shape {observation.shape}, expected {(OBSERVATION_DIM,)}"
            )
        if not np.all(np.isfinite(observation)):
            indices = np.flatnonzero(~np.isfinite(observation)).tolist()
            raise ValueError(f"non-finite Teleopit observation at indices {indices}")
        return observation


class TeleopitOnnxPolicy:
    """Dual-input ONNX wrapper with Teleopit's exact history behavior."""

    def __init__(self, checkpoint: str | Path, device: str = "cpu") -> None:
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Teleopit checkpoint not found: {path}")
        try:
            import onnxruntime as ort
        except ModuleNotFoundError as exc:
            raise ImportError(
                "onnxruntime is required; run Teleopit_rollout/setup_env.sh"
            ) from exc

        available = ort.get_available_providers()
        if device == "cpu":
            providers = ["CPUExecutionProvider"]
        elif device == "auto":
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if "CUDAExecutionProvider" in available
                else ["CPUExecutionProvider"]
            )
        else:
            raise ValueError("device must be 'cpu' or 'auto'")
        self.checkpoint = path
        self.session = ort.InferenceSession(str(path), providers=providers)
        self.providers = tuple(self.session.get_providers())
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        input_signature = [(item.name, tuple(item.shape)) for item in inputs]
        output_signature = [(item.name, tuple(item.shape)) for item in outputs]
        expected_inputs = [
            ("obs", (1, OBSERVATION_DIM)),
            ("obs_history", (1, HISTORY_LENGTH, OBSERVATION_DIM)),
        ]
        expected_outputs = [("actions", (1, ACTION_DIM))]
        if input_signature != expected_inputs:
            raise ValueError(
                f"unexpected Teleopit ONNX inputs {input_signature}; expected {expected_inputs}"
            )
        if output_signature != expected_outputs:
            raise ValueError(
                f"unexpected Teleopit ONNX outputs {output_signature}; expected {expected_outputs}"
            )
        self._history: deque[FloatArray] = deque(maxlen=HISTORY_LENGTH)
        self.last_history: FloatArray | None = None

    def reset(self) -> None:
        self._history.clear()
        self.last_history = None

    def infer(self, observation: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        obs = np.asarray(observation, dtype=np.float32).reshape(OBSERVATION_DIM)
        if not self._history:
            for _ in range(HISTORY_LENGTH):
                self._history.append(obs.copy())
        else:
            self._history.append(obs.copy())
        history = np.stack(tuple(self._history), axis=0).astype(np.float32, copy=False)
        raw_action = np.asarray(
            self.session.run(
                ["actions"],
                {"obs": obs[None, :], "obs_history": history[None, :, :]},
            )[0],
            dtype=np.float32,
        ).reshape(ACTION_DIM)
        if not np.all(np.isfinite(raw_action)):
            raise ValueError("Teleopit ONNX returned NaN/inf")
        target = DEFAULT_DOF_POS + ACTION_SCALE * np.clip(raw_action, -10.0, 10.0)
        self.last_history = history.copy()
        return raw_action, np.asarray(target, dtype=np.float32), history.copy()
