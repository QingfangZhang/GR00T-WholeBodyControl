"""Deterministic MuJoCo task-scene control for the Teleopit tracker.

The recorded task XML contains 29 body motors, two seven-DoF Dex3 hands and
possibly task actuators.  The body and hand joints are interleaved in XML
order, so every mapping in this module is resolved by joint name.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np

try:
    from .constants import (
        ACTION_DIM,
        DEX3_MAX_LEFT,
        DEX3_MAX_RIGHT,
        DEX3_MIN_LEFT,
        DEX3_MIN_RIGHT,
        G1_JOINT_NAMES,
        HAND_KD,
        HAND_KP,
        HAND_MAX_TARGET_DELTA,
        KDS,
        KPS,
        LEFT_HAND_JOINT_NAMES,
        RIGHT_HAND_JOINT_NAMES,
        ROOT_ASSIST_WIDTH,
        TORQUE_LIMITS,
    )
    from .teleopit_policy import RobotState
except ImportError:  # pragma: no cover - direct script execution
    from constants import (
        ACTION_DIM,
        DEX3_MAX_LEFT,
        DEX3_MAX_RIGHT,
        DEX3_MIN_LEFT,
        DEX3_MIN_RIGHT,
        G1_JOINT_NAMES,
        HAND_KD,
        HAND_KP,
        HAND_MAX_TARGET_DELTA,
        KDS,
        KPS,
        LEFT_HAND_JOINT_NAMES,
        RIGHT_HAND_JOINT_NAMES,
        ROOT_ASSIST_WIDTH,
        TORQUE_LIMITS,
    )
    from teleopit_policy import RobotState


FloatArray = np.ndarray


def _named_id(model: mujoco.MjModel, kind: str, name: str) -> int:
    object_type = {
        "joint": mujoco.mjtObj.mjOBJ_JOINT,
        "body": mujoco.mjtObj.mjOBJ_BODY,
    }[kind]
    identifier = int(mujoco.mj_name2id(model, object_type, name))
    if identifier < 0:
        raise ValueError(f"MuJoCo {kind} {name!r} is missing")
    return identifier


def _qpos_labels(model: mujoco.MjModel, first: int) -> list[str]:
    labels = [f"qpos[{index}]" for index in range(first, model.nq)]
    ordered = sorted(range(model.njnt), key=lambda jid: int(model.jnt_qposadr[jid]))
    for position, joint_id in enumerate(ordered):
        start = int(model.jnt_qposadr[joint_id])
        stop = (
            int(model.jnt_qposadr[ordered[position + 1]])
            if position + 1 < len(ordered)
            else model.nq
        )
        if stop <= first:
            continue
        name = model.joint(joint_id).name or f"joint_{joint_id}"
        for qpos_index in range(max(start, first), stop):
            suffix = "" if stop - start == 1 else f"[{qpos_index - start}]"
            labels[qpos_index - first] = f"{name}{suffix}"
    return labels


@dataclass
class RootAssistStatistics:
    mode: str
    ticks: int = 0
    squared_error_sum: float = 0.0
    maximum_error: float = 0.0

    @property
    def width(self) -> int:
        return ROOT_ASSIST_WIDTH[self.mode]

    def update(self, error: FloatArray) -> None:
        magnitude = float(np.linalg.norm(error))
        self.ticks += 1
        self.squared_error_sum += magnitude * magnitude
        self.maximum_error = max(self.maximum_error, magnitude)

    def as_dict(self) -> dict[str, object]:
        return {
            "enabled": self.width > 0,
            "mode": self.mode,
            "position_components": ["x", "y", "z"][: self.width],
            "ticks": self.ticks,
            "pre_alignment_error_rms_m": (
                float(np.sqrt(self.squared_error_sum / self.ticks))
                if self.ticks
                else 0.0
            ),
            "pre_alignment_error_max_m": self.maximum_error,
            "position_update": (
                "hard overwrite after each 200 Hz PD interval; root orientation "
                "is never overwritten"
            ),
            "velocity_update": (
                "copy matching source linear velocity while source advances; "
                "use zero at source hold"
            ),
            "evaluation_note": (
                "Oracle source-root assistance changes the dynamics. Assisted "
                "results measure controller replacement under a common external "
                "root stabilizer, not standalone root tracking."
            ),
        }


class TaskSceneController:
    """Own one recorded MuJoCo scene and apply Teleopit/Dex3 PD commands."""

    def __init__(self, scene_xml: str | Path, *, root_assist: str = "none") -> None:
        if root_assist not in ROOT_ASSIST_WIDTH:
            raise ValueError(f"unsupported root-assist mode {root_assist!r}")
        self.scene_xml = Path(scene_xml).expanduser().resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_xml))
        self.data = mujoco.MjData(self.model)
        self.root_assist = RootAssistStatistics(root_assist)

        self.body_joint_ids = self._joint_ids(G1_JOINT_NAMES)
        self.left_hand_joint_ids = self._joint_ids(LEFT_HAND_JOINT_NAMES)
        self.right_hand_joint_ids = self._joint_ids(RIGHT_HAND_JOINT_NAMES)
        self.body_qpos_addresses = self.model.jnt_qposadr[self.body_joint_ids].astype(np.int32)
        self.body_dof_addresses = self.model.jnt_dofadr[self.body_joint_ids].astype(np.int32)
        self.left_hand_qpos_addresses = self.model.jnt_qposadr[
            self.left_hand_joint_ids
        ].astype(np.int32)
        self.left_hand_dof_addresses = self.model.jnt_dofadr[
            self.left_hand_joint_ids
        ].astype(np.int32)
        self.right_hand_qpos_addresses = self.model.jnt_qposadr[
            self.right_hand_joint_ids
        ].astype(np.int32)
        self.right_hand_dof_addresses = self.model.jnt_dofadr[
            self.right_hand_joint_ids
        ].astype(np.int32)

        robot_joint_id_set = set(
            np.concatenate(
                (
                    self.body_joint_ids,
                    self.left_hand_joint_ids,
                    self.right_hand_joint_ids,
                )
            ).tolist()
        )
        joint_to_actuator: dict[int, int] = {}
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            if joint_id not in robot_joint_id_set:
                continue
            transmission = int(self.model.actuator_trntype[actuator_id])
            if transmission != int(mujoco.mjtTrn.mjTRN_JOINT):
                raise ValueError(
                    f"robot actuator {actuator_id} does not use joint transmission"
                )
            if joint_id in joint_to_actuator:
                raise ValueError(
                    f"robot joint {joint_id} maps to multiple actuators: "
                    f"{joint_to_actuator[joint_id]} and {actuator_id}"
                )
            joint_to_actuator[joint_id] = actuator_id
        self.body_actuator_ids = self._actuator_ids(self.body_joint_ids, joint_to_actuator)
        self.left_hand_actuator_ids = self._actuator_ids(
            self.left_hand_joint_ids, joint_to_actuator
        )
        self.right_hand_actuator_ids = self._actuator_ids(
            self.right_hand_joint_ids, joint_to_actuator
        )
        robot_actuators = np.concatenate(
            (self.body_actuator_ids, self.left_hand_actuator_ids, self.right_hand_actuator_ids)
        )
        if len(set(robot_actuators.tolist())) != 43:
            raise ValueError("the 43 robot joints do not map one-to-one to actuators")
        self.robot_actuator_ids = robot_actuators
        self.nonrobot_actuator_ids = np.asarray(
            sorted(set(range(self.model.nu)).difference(robot_actuators.tolist())),
            dtype=np.int32,
        )

        all_robot_ids = np.concatenate(
            (self.body_joint_ids, self.left_hand_joint_ids, self.right_hand_joint_ids)
        )
        self.robot_joint_ids_xml_order = all_robot_ids[
            np.argsort(self.model.jnt_qposadr[all_robot_ids])
        ]
        xml_addresses = self.model.jnt_qposadr[self.robot_joint_ids_xml_order]
        if not np.array_equal(xml_addresses, np.arange(7, 50)):
            raise ValueError(
                "expected floating root qpos[0:7] followed by 43 robot joints "
                f"at qpos[7:50], got {xml_addresses.tolist()}"
            )
        self.task_qpos_start = 50
        if self.model.nq < self.task_qpos_start:
            raise ValueError(f"task model nq={self.model.nq} is smaller than 50")
        self.task_qpos_labels = _qpos_labels(self.model, self.task_qpos_start)
        self.pelvis_body_id = _named_id(self.model, "body", "pelvis")

        self.body_target = np.zeros(ACTION_DIM, dtype=np.float64)
        self.left_hand_desired = np.zeros(7, dtype=np.float64)
        self.right_hand_desired = np.zeros(7, dtype=np.float64)
        self.left_hand_applied = np.zeros(7, dtype=np.float64)
        self.right_hand_applied = np.zeros(7, dtype=np.float64)
        self._last_command: dict[str, FloatArray] | None = None

    def _joint_ids(self, names: Sequence[str]) -> FloatArray:
        return np.asarray(
            [_named_id(self.model, "joint", name) for name in names], dtype=np.int32
        )

    @staticmethod
    def _actuator_ids(joint_ids: FloatArray, mapping: dict[int, int]) -> FloatArray:
        missing = [int(joint_id) for joint_id in joint_ids if int(joint_id) not in mapping]
        if missing:
            raise ValueError(f"robot joints without actuators: {missing}")
        return np.asarray([mapping[int(joint_id)] for joint_id in joint_ids], dtype=np.int32)

    def initialize(self, qpos: FloatArray, qvel: FloatArray) -> None:
        positions = np.asarray(qpos, dtype=np.float64)
        velocities = np.asarray(qvel, dtype=np.float64)
        if positions.shape != (self.model.nq,) or velocities.shape != (self.model.nv,):
            raise ValueError(
                f"CSV/model state mismatch: qpos {positions.shape}/{(self.model.nq,)}, "
                f"qvel {velocities.shape}/{(self.model.nv,)}"
            )
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
            raise ValueError("initial qpos/qvel contains NaN or inf")
        self.data.qpos[:] = positions
        self.data.qvel[:] = velocities
        self.data.time = 0.0
        self.data.ctrl[:] = 0.0
        self.data.qacc_warmstart[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.body_target[:] = self.body_joint_pos
        self.left_hand_desired[:] = self.left_hand_joint_pos
        self.right_hand_desired[:] = self.right_hand_joint_pos
        self.left_hand_applied[:] = self.left_hand_joint_pos
        self.right_hand_applied[:] = self.right_hand_joint_pos
        self._last_command = None

    @property
    def body_joint_pos(self) -> FloatArray:
        return self.data.qpos[self.body_qpos_addresses].copy()

    @property
    def body_joint_vel(self) -> FloatArray:
        return self.data.qvel[self.body_dof_addresses].copy()

    @property
    def left_hand_joint_pos(self) -> FloatArray:
        return self.data.qpos[self.left_hand_qpos_addresses].copy()

    @property
    def right_hand_joint_pos(self) -> FloatArray:
        return self.data.qpos[self.right_hand_qpos_addresses].copy()

    def robot_state(self) -> RobotState:
        velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.pelvis_body_id,
            velocity,
            1,
        )
        return RobotState(
            joint_pos=self.body_joint_pos.astype(np.float32),
            joint_vel=self.body_joint_vel.astype(np.float32),
            root_pos=self.data.qpos[0:3].astype(np.float32, copy=True),
            root_quat_wxyz=self.data.qpos[3:7].astype(np.float32, copy=True),
            root_ang_vel_b=velocity[0:3].astype(np.float32, copy=True),
            timestamp_s=float(self.data.time),
        )

    def set_policy_command(
        self,
        body_target: FloatArray,
        left_hand_target: FloatArray,
        right_hand_target: FloatArray,
    ) -> None:
        target = np.asarray(body_target, dtype=np.float64).reshape(ACTION_DIM)
        left = np.asarray(left_hand_target, dtype=np.float64).reshape(7)
        right = np.asarray(right_hand_target, dtype=np.float64).reshape(7)
        if not all(np.all(np.isfinite(item)) for item in (target, left, right)):
            raise ValueError("policy/hand target contains NaN or inf")
        self.body_target[:] = target
        self.left_hand_desired[:] = np.clip(left, DEX3_MIN_LEFT, DEX3_MAX_LEFT)
        self.right_hand_desired[:] = np.clip(right, DEX3_MIN_RIGHT, DEX3_MAX_RIGHT)

    def _actuator_clip(self, torque: FloatArray, actuator_ids: FloatArray) -> FloatArray:
        result = np.asarray(torque, dtype=np.float64).copy()
        limited = self.model.actuator_ctrllimited[actuator_ids].astype(bool)
        ranges = self.model.actuator_ctrlrange[actuator_ids]
        result[limited] = np.clip(result[limited], ranges[limited, 0], ranges[limited, 1])
        return result

    def update_pd_command(self) -> dict[str, FloatArray]:
        body_q = self.body_joint_pos
        body_dq = self.body_joint_vel
        body_tau = KPS * (self.body_target - body_q) - KDS * body_dq
        body_tau = np.clip(body_tau, -TORQUE_LIMITS, TORQUE_LIMITS)
        body_tau = self._actuator_clip(body_tau, self.body_actuator_ids)

        left_q = self.left_hand_joint_pos
        right_q = self.right_hand_joint_pos
        self.left_hand_applied[:] = left_q + np.clip(
            self.left_hand_desired - left_q,
            -HAND_MAX_TARGET_DELTA,
            HAND_MAX_TARGET_DELTA,
        )
        self.right_hand_applied[:] = right_q + np.clip(
            self.right_hand_desired - right_q,
            -HAND_MAX_TARGET_DELTA,
            HAND_MAX_TARGET_DELTA,
        )
        left_dq = self.data.qvel[self.left_hand_dof_addresses]
        right_dq = self.data.qvel[self.right_hand_dof_addresses]
        left_tau = HAND_KP * (self.left_hand_applied - left_q) - HAND_KD * left_dq
        right_tau = HAND_KP * (self.right_hand_applied - right_q) - HAND_KD * right_dq
        left_tau = self._actuator_clip(left_tau, self.left_hand_actuator_ids)
        right_tau = self._actuator_clip(right_tau, self.right_hand_actuator_ids)

        self.data.ctrl[:] = 0.0
        self.data.ctrl[self.body_actuator_ids] = body_tau
        self.data.ctrl[self.left_hand_actuator_ids] = left_tau
        self.data.ctrl[self.right_hand_actuator_ids] = right_tau
        if self.nonrobot_actuator_ids.size:
            self.data.ctrl[self.nonrobot_actuator_ids] = 0.0

        target_by_joint = {
            int(joint): float(value)
            for joint, value in zip(self.body_joint_ids, self.body_target, strict=True)
        }
        target_by_joint.update(
            {
                int(joint): float(value)
                for joint, value in zip(
                    self.left_hand_joint_ids, self.left_hand_applied, strict=True
                )
            }
        )
        target_by_joint.update(
            {
                int(joint): float(value)
                for joint, value in zip(
                    self.right_hand_joint_ids, self.right_hand_applied, strict=True
                )
            }
        )
        zero7 = np.zeros(7, dtype=np.float64)
        command = {
            "received_dof_pos": np.asarray(
                [target_by_joint[int(joint)] for joint in self.robot_joint_ids_xml_order],
                dtype=np.float64,
            ),
            "left_hand_q": self.left_hand_applied.copy(),
            "left_hand_dq": zero7.copy(),
            "left_hand_kp": np.full(7, HAND_KP, dtype=np.float64),
            "left_hand_kd": np.full(7, HAND_KD, dtype=np.float64),
            "left_hand_tau": zero7.copy(),
            "right_hand_q": self.right_hand_applied.copy(),
            "right_hand_dq": zero7.copy(),
            "right_hand_kp": np.full(7, HAND_KP, dtype=np.float64),
            "right_hand_kd": np.full(7, HAND_KD, dtype=np.float64),
            "right_hand_tau": zero7.copy(),
        }
        self._last_command = command
        return {name: values.copy() for name, values in command.items()}

    def command_snapshot(self) -> dict[str, FloatArray]:
        if self._last_command is None:
            raise RuntimeError("PD command has not been initialized")
        return {name: values.copy() for name, values in self._last_command.items()}

    def physics_step(self, substeps: int) -> None:
        if substeps <= 0:
            raise ValueError("physics substeps must be positive")
        mujoco.mj_step(self.model, self.data, nstep=substeps)

    def apply_root_assist(
        self,
        source_qpos: FloatArray,
        source_qvel: FloatArray,
        *,
        source_velocity_active: bool,
    ) -> None:
        width = self.root_assist.width
        if width == 0:
            return
        qpos = np.asarray(source_qpos, dtype=np.float64)
        qvel = np.asarray(source_qvel, dtype=np.float64)
        if qpos.shape != (self.model.nq,) or qvel.shape != (self.model.nv,):
            raise ValueError("source qpos/qvel shape changed during root assist")
        error = qpos[:width] - self.data.qpos[:width]
        self.data.qpos[:width] = qpos[:width]
        self.data.qvel[:width] = qvel[:width] if source_velocity_active else 0.0
        self.data.qacc_warmstart[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.root_assist.update(error)

    def validate_state(self) -> None:
        if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
            raise FloatingPointError("MuJoCo qpos/qvel contains NaN or inf")
