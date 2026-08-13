"""Deterministic task-scene dynamics shared by all controller adapters.

The neural policy and low-level impedance are intentionally separated.  A
controller supplies its native joint targets, Kp/Kd and torque limits at
50 Hz; this scene recomputes the corresponding torque at exactly 200 Hz and
advances the recorded MuJoCo task model at 2 kHz.  State logging is handled by
the runner at 400 Hz and never drives simulation time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import mujoco
import numpy as np

from Teleopit_rollout.constants import (
    DEX3_MAX_LEFT,
    DEX3_MAX_RIGHT,
    DEX3_MIN_LEFT,
    DEX3_MIN_RIGHT,
    G1_JOINT_NAMES,
    HAND_KD,
    HAND_KP,
    HAND_MAX_TARGET_DELTA,
    LEFT_HAND_JOINT_NAMES,
    RIGHT_HAND_JOINT_NAMES,
)


FloatArray = np.ndarray
PHYSICS_HZ = 2000.0
PD_HZ = 200.0
LOG_HZ = 400.0
POLICY_HZ = 50.0
PHYSICS_DT_S = 1.0 / PHYSICS_HZ


def _finite_vector(value: Any, size: int, name: str) -> FloatArray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (size,):
        raise ValueError(f"{name} has shape {result.shape}; expected {(size,)}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains NaN or infinity")
    return result.copy()


def _named_id(model: mujoco.MjModel, kind: str, name: str) -> int:
    object_type = {
        "joint": mujoco.mjtObj.mjOBJ_JOINT,
        "body": mujoco.mjtObj.mjOBJ_BODY,
    }[kind]
    identifier = int(mujoco.mj_name2id(model, object_type, name))
    if identifier < 0:
        raise ValueError(f"MuJoCo {kind} {name!r} is missing")
    return identifier


def _qpos_labels(model: mujoco.MjModel, first: int) -> tuple[str, ...]:
    """Return stable human-readable labels for the task qpos suffix."""

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
    return tuple(labels)


@dataclass(frozen=True)
class SceneState:
    """Controller-neutral state snapshot in G1 canonical joint-name order."""

    joint_pos: FloatArray
    joint_vel: FloatArray
    root_pos: FloatArray
    root_quat_wxyz: FloatArray
    root_ang_vel_b: FloatArray
    timestamp_s: float
    left_hand_pos: FloatArray
    left_hand_vel: FloatArray
    right_hand_pos: FloatArray
    right_hand_vel: FloatArray
    received_dof_pos: FloatArray
    xml_robot_joint_pos: FloatArray


@dataclass(frozen=True)
class AppliedPdCommand:
    """The command and actual torques installed for the next PD interval."""

    body_q_target: FloatArray
    body_kp: FloatArray
    body_kd: FloatArray
    body_torque_limit: FloatArray
    body_torque: FloatArray
    body_torque_saturation: FloatArray
    left_hand_target: FloatArray
    right_hand_target: FloatArray
    left_hand_torque: FloatArray
    right_hand_torque: FloatArray

    def csv_command_fields(self) -> dict[str, FloatArray]:
        """Return fields whose legacy CSV semantics are controller-neutral."""

        zero7 = np.zeros(7, dtype=np.float64)
        return {
            "left_hand_q": self.left_hand_target.copy(),
            "left_hand_dq": zero7.copy(),
            "left_hand_kp": np.full(7, HAND_KP, dtype=np.float64),
            "left_hand_kd": np.full(7, HAND_KD, dtype=np.float64),
            # The legacy hand tau field is feed-forward torque, not the total
            # PD torque.  The actual applied torque lives in telemetry.
            "left_hand_tau": zero7.copy(),
            "right_hand_q": self.right_hand_target.copy(),
            "right_hand_dq": zero7.copy(),
            "right_hand_kp": np.full(7, HAND_KP, dtype=np.float64),
            "right_hand_kd": np.full(7, HAND_KD, dtype=np.float64),
            "right_hand_tau": zero7.copy(),
        }


@dataclass(frozen=True)
class ContactSummary:
    """Compact contact diagnostics at one logged simulation state."""

    total_contacts: int
    robot_self_contacts: int
    robot_world_contacts: int
    robot_environment_contacts: int
    robot_normal_force_sum_n: float
    robot_normal_force_max_n: float
    robot_environment_normal_force_sum_n: float
    robot_environment_normal_force_max_n: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


@dataclass
class RootAssistStatistics:
    mode: str
    ticks: int = 0
    squared_error_sum: float = 0.0
    maximum_error: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in ("none", "xy"):
            raise ValueError("root assist must be 'none' or 'xy'")

    @property
    def enabled(self) -> bool:
        return self.mode == "xy"

    def update(self, error: FloatArray) -> None:
        magnitude = float(np.linalg.norm(error))
        self.ticks += 1
        self.squared_error_sum += magnitude * magnitude
        self.maximum_error = max(self.maximum_error, magnitude)

    def metadata(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "enabled": self.enabled,
            "position_components": ["x", "y"] if self.enabled else [],
            "ticks": self.ticks,
            "pre_alignment_error_rms_m": (
                float(np.sqrt(self.squared_error_sum / self.ticks))
                if self.ticks
                else 0.0
            ),
            "pre_alignment_error_max_m": self.maximum_error,
            "position_update": (
                "hard overwrite after each 200 Hz PD interval"
                if self.enabled
                else "none"
            ),
            "orientation_update": "never overwritten",
        }


class DeterministicTaskScene:
    """Recorded G1 task scene with controller-supplied native body impedance."""

    def __init__(
        self,
        scene_xml: str | Path,
        *,
        root_assist: str = "none",
        enforce_physics_dt: float = PHYSICS_DT_S,
    ) -> None:
        self.scene_xml = Path(scene_xml).expanduser().resolve()
        if not self.scene_xml.is_file():
            raise FileNotFoundError(self.scene_xml)
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_xml))
        self.data = mujoco.MjData(self.model)
        if not np.isclose(
            float(self.model.opt.timestep),
            float(enforce_physics_dt),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "recorded task physics must use the fixed 2 kHz timestep "
                f"{enforce_physics_dt:g}s; got {self.model.opt.timestep:g}s"
            )
        self.root_assist = RootAssistStatistics(root_assist)

        self.body_joint_ids = self._joint_ids(G1_JOINT_NAMES)
        self.left_hand_joint_ids = self._joint_ids(LEFT_HAND_JOINT_NAMES)
        self.right_hand_joint_ids = self._joint_ids(RIGHT_HAND_JOINT_NAMES)
        self.body_qpos_addresses = self.model.jnt_qposadr[self.body_joint_ids].astype(
            np.int32
        )
        self.body_dof_addresses = self.model.jnt_dofadr[self.body_joint_ids].astype(
            np.int32
        )
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

        all_robot_joint_ids = np.concatenate(
            (self.body_joint_ids, self.left_hand_joint_ids, self.right_hand_joint_ids)
        )
        joint_to_actuator: dict[int, int] = {}
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            if joint_id not in set(all_robot_joint_ids.tolist()):
                continue
            if int(self.model.actuator_trntype[actuator_id]) != int(
                mujoco.mjtTrn.mjTRN_JOINT
            ):
                raise ValueError(
                    f"robot actuator {actuator_id} does not use joint transmission"
                )
            if joint_id in joint_to_actuator:
                raise ValueError(f"robot joint {joint_id} maps to multiple actuators")
            joint_to_actuator[joint_id] = actuator_id
        self.body_actuator_ids = self._actuator_ids(
            self.body_joint_ids, joint_to_actuator
        )
        self.left_hand_actuator_ids = self._actuator_ids(
            self.left_hand_joint_ids, joint_to_actuator
        )
        self.right_hand_actuator_ids = self._actuator_ids(
            self.right_hand_joint_ids, joint_to_actuator
        )
        robot_actuators = np.concatenate(
            (
                self.body_actuator_ids,
                self.left_hand_actuator_ids,
                self.right_hand_actuator_ids,
            )
        )
        if len(set(robot_actuators.tolist())) != 43:
            raise ValueError("the 43 robot joints do not map one-to-one to actuators")
        self.nonrobot_actuator_ids = np.asarray(
            sorted(set(range(self.model.nu)).difference(robot_actuators.tolist())),
            dtype=np.int32,
        )

        self.robot_joint_ids_xml_order = all_robot_joint_ids[
            np.argsort(self.model.jnt_qposadr[all_robot_joint_ids])
        ]
        robot_qpos_addresses = self.model.jnt_qposadr[
            self.robot_joint_ids_xml_order
        ]
        if not np.array_equal(robot_qpos_addresses, np.arange(7, 50)):
            raise ValueError(
                "expected floating root qpos[0:7] followed by 43 robot joints; "
                f"got {robot_qpos_addresses.tolist()}"
            )
        self.task_qpos_start = 50
        if self.model.nq < self.task_qpos_start:
            raise ValueError(f"task model nq={self.model.nq} is smaller than 50")
        self.pelvis_body_id = _named_id(self.model, "body", "pelvis")
        self.task_qpos_labels = _qpos_labels(self.model, self.task_qpos_start)
        self.robot_body_ids = frozenset(
            {
                self.pelvis_body_id,
                *(
                    int(self.model.jnt_bodyid[joint_id])
                    for joint_id in all_robot_joint_ids
                ),
            }
        )

        self.body_q_target = np.zeros(29, dtype=np.float64)
        self.body_kp = np.zeros(29, dtype=np.float64)
        self.body_kd = np.zeros(29, dtype=np.float64)
        self.body_torque_limit = np.zeros(29, dtype=np.float64)
        self.left_hand_desired = np.zeros(7, dtype=np.float64)
        self.right_hand_desired = np.zeros(7, dtype=np.float64)
        self.left_hand_applied = np.zeros(7, dtype=np.float64)
        self.right_hand_applied = np.zeros(7, dtype=np.float64)
        self.last_applied_command: AppliedPdCommand | None = None

    def _joint_ids(self, names: Sequence[str]) -> FloatArray:
        return np.asarray(
            [_named_id(self.model, "joint", name) for name in names],
            dtype=np.int32,
        )

    @staticmethod
    def _actuator_ids(
        joint_ids: FloatArray, mapping: Mapping[int, int]
    ) -> FloatArray:
        missing = [int(joint) for joint in joint_ids if int(joint) not in mapping]
        if missing:
            raise ValueError(f"robot joints without actuators: {missing}")
        return np.asarray([mapping[int(joint)] for joint in joint_ids], dtype=np.int32)

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

    def initialize(self, qpos: Any, qvel: Any) -> None:
        positions = _finite_vector(qpos, self.model.nq, "initial qpos")
        velocities = _finite_vector(qvel, self.model.nv, "initial qvel")
        self.data.qpos[:] = positions
        self.data.qvel[:] = velocities
        self.data.time = 0.0
        self.data.ctrl[:] = 0.0
        self.data.qacc_warmstart[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.body_q_target[:] = self.body_joint_pos
        self.left_hand_desired[:] = self.left_hand_joint_pos
        self.right_hand_desired[:] = self.right_hand_joint_pos
        self.left_hand_applied[:] = self.left_hand_joint_pos
        self.right_hand_applied[:] = self.right_hand_joint_pos
        self.last_applied_command = None

    def state(self) -> SceneState:
        velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.pelvis_body_id,
            velocity,
            1,
        )
        xml_robot_joint_pos = self.data.qpos[
            self.model.jnt_qposadr[self.robot_joint_ids_xml_order]
        ].astype(np.float32, copy=True)
        # Legacy ``policy_received_dof_pos`` order is body29 + left7 + right7,
        # not raw XML qpos[7:50] (the XML interleaves the left hand between the
        # two arms).  Expose both explicitly to prevent a silent schema error.
        received = np.concatenate(
            (
                self.body_joint_pos,
                self.left_hand_joint_pos,
                self.right_hand_joint_pos,
            )
        ).astype(np.float32)
        return SceneState(
            joint_pos=self.body_joint_pos.astype(np.float32),
            joint_vel=self.body_joint_vel.astype(np.float32),
            root_pos=self.data.qpos[0:3].astype(np.float32, copy=True),
            root_quat_wxyz=self.data.qpos[3:7].astype(np.float32, copy=True),
            root_ang_vel_b=velocity[0:3].astype(np.float32, copy=True),
            timestamp_s=float(self.data.time),
            left_hand_pos=self.left_hand_joint_pos.astype(np.float32),
            left_hand_vel=self.data.qvel[self.left_hand_dof_addresses].astype(
                np.float32, copy=True
            ),
            right_hand_pos=self.right_hand_joint_pos.astype(np.float32),
            right_hand_vel=self.data.qvel[self.right_hand_dof_addresses].astype(
                np.float32, copy=True
            ),
            received_dof_pos=received,
            xml_robot_joint_pos=xml_robot_joint_pos,
        )

    def set_controller_command(
        self,
        *,
        q_target: Any,
        kp: Any,
        kd: Any,
        torque_limit: Any,
        left_hand_target: Any,
        right_hand_target: Any,
    ) -> None:
        self.body_q_target[:] = _finite_vector(q_target, 29, "body q_target")
        self.body_kp[:] = _finite_vector(kp, 29, "body kp")
        self.body_kd[:] = _finite_vector(kd, 29, "body kd")
        self.body_torque_limit[:] = _finite_vector(
            torque_limit, 29, "body torque_limit"
        )
        if np.any(self.body_kp < 0.0) or np.any(self.body_kd < 0.0):
            raise ValueError("body Kp/Kd must be non-negative")
        if np.any(self.body_torque_limit <= 0.0):
            raise ValueError("body torque limits must be positive")
        self.left_hand_desired[:] = np.clip(
            _finite_vector(left_hand_target, 7, "left hand target"),
            DEX3_MIN_LEFT,
            DEX3_MAX_LEFT,
        )
        self.right_hand_desired[:] = np.clip(
            _finite_vector(right_hand_target, 7, "right hand target"),
            DEX3_MIN_RIGHT,
            DEX3_MAX_RIGHT,
        )

    def _clip_actuator(self, torque: FloatArray, actuator_ids: FloatArray) -> tuple[FloatArray, FloatArray]:
        result = np.asarray(torque, dtype=np.float64).copy()
        saturated = np.zeros(result.shape, dtype=np.bool_)
        limited = self.model.actuator_ctrllimited[actuator_ids].astype(bool)
        ranges = self.model.actuator_ctrlrange[actuator_ids]
        if np.any(limited):
            clipped = np.clip(
                result[limited], ranges[limited, 0], ranges[limited, 1]
            )
            saturated[limited] = clipped != result[limited]
            result[limited] = clipped
        return result, saturated

    def update_pd(self) -> AppliedPdCommand:
        body_q = self.body_joint_pos
        body_dq = self.body_joint_vel
        raw_body_tau = self.body_kp * (self.body_q_target - body_q) - self.body_kd * body_dq
        body_tau = np.clip(
            raw_body_tau, -self.body_torque_limit, self.body_torque_limit
        )
        body_saturation = body_tau != raw_body_tau
        body_tau, actuator_saturation = self._clip_actuator(
            body_tau, self.body_actuator_ids
        )
        body_saturation |= actuator_saturation

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
        left_tau, _ = self._clip_actuator(left_tau, self.left_hand_actuator_ids)
        right_tau, _ = self._clip_actuator(right_tau, self.right_hand_actuator_ids)

        self.data.ctrl[:] = 0.0
        self.data.ctrl[self.body_actuator_ids] = body_tau
        self.data.ctrl[self.left_hand_actuator_ids] = left_tau
        self.data.ctrl[self.right_hand_actuator_ids] = right_tau
        if self.nonrobot_actuator_ids.size:
            self.data.ctrl[self.nonrobot_actuator_ids] = 0.0

        command = AppliedPdCommand(
            body_q_target=self.body_q_target.copy(),
            body_kp=self.body_kp.copy(),
            body_kd=self.body_kd.copy(),
            body_torque_limit=self.body_torque_limit.copy(),
            body_torque=body_tau.copy(),
            body_torque_saturation=body_saturation.copy(),
            left_hand_target=self.left_hand_applied.copy(),
            right_hand_target=self.right_hand_applied.copy(),
            left_hand_torque=left_tau.copy(),
            right_hand_torque=right_tau.copy(),
        )
        self.last_applied_command = command
        return command

    def physics_step(self, substeps: int) -> None:
        if int(substeps) <= 0:
            raise ValueError("physics substeps must be positive")
        mujoco.mj_step(self.model, self.data, nstep=int(substeps))

    def apply_root_assist(
        self,
        source_qpos: Any,
        source_qvel: Any,
        *,
        source_velocity_active: bool,
    ) -> None:
        if not self.root_assist.enabled:
            return
        qpos = _finite_vector(source_qpos, self.model.nq, "source qpos")
        qvel = _finite_vector(source_qvel, self.model.nv, "source qvel")
        error = qpos[:2] - self.data.qpos[:2]
        self.data.qpos[:2] = qpos[:2]
        self.data.qvel[:2] = qvel[:2] if source_velocity_active else 0.0
        self.data.qacc_warmstart[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.root_assist.update(error)

    def validate_state(self) -> None:
        if not np.all(np.isfinite(self.data.qpos)) or not np.all(
            np.isfinite(self.data.qvel)
        ):
            raise FloatingPointError("MuJoCo qpos/qvel contains NaN or infinity")

    def task_qpos(self) -> FloatArray:
        return self.data.qpos[self.task_qpos_start :].copy()

    def contact_summary(self) -> ContactSummary:
        robot_self = 0
        robot_world = 0
        robot_environment = 0
        robot_force_sum = 0.0
        robot_force_max = 0.0
        environment_force_sum = 0.0
        environment_force_max = 0.0
        force = np.zeros(6, dtype=np.float64)
        for contact_index in range(int(self.data.ncon)):
            contact = self.data.contact[contact_index]
            body1 = int(self.model.geom_bodyid[int(contact.geom1)])
            body2 = int(self.model.geom_bodyid[int(contact.geom2)])
            first_robot = body1 in self.robot_body_ids
            second_robot = body2 in self.robot_body_ids
            if first_robot and second_robot:
                robot_self += 1
                continue
            if first_robot == second_robot:
                continue
            force[:] = 0.0
            mujoco.mj_contactForce(self.model, self.data, contact_index, force)
            normal_force = max(0.0, float(force[0]))
            robot_force_sum += normal_force
            robot_force_max = max(robot_force_max, normal_force)
            other_body = body2 if first_robot else body1
            if other_body == 0:
                robot_world += 1
            else:
                robot_environment += 1
                environment_force_sum += normal_force
                environment_force_max = max(
                    environment_force_max, normal_force
                )
        return ContactSummary(
            total_contacts=int(self.data.ncon),
            robot_self_contacts=robot_self,
            robot_world_contacts=robot_world,
            robot_environment_contacts=robot_environment,
            robot_normal_force_sum_n=robot_force_sum,
            robot_normal_force_max_n=robot_force_max,
            robot_environment_normal_force_sum_n=environment_force_sum,
            robot_environment_normal_force_max_n=environment_force_max,
        )


__all__ = [
    "AppliedPdCommand",
    "ContactSummary",
    "DeterministicTaskScene",
    "LOG_HZ",
    "PD_HZ",
    "PHYSICS_DT_S",
    "PHYSICS_HZ",
    "POLICY_HZ",
    "RootAssistStatistics",
    "SceneState",
]
