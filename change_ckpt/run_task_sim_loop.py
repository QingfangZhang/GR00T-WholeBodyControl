#!/usr/bin/env python3
"""Run the recorded task scene against the existing SONIC DDS deployment.

The original ``gear_sonic/scripts/run_sim_loop.py`` is deliberately not
modified.  This variant loads an XML snapshot from a recording, initializes all
MuJoCo qpos/qvel values at a policy boundary, displays the live simulation, and
writes a replay-compatible CSV under ``change_ckpt/data`` by default.

It remains a simulator process: start the regular or low-latency C++ deploy in
another terminal, just as for the repository's normal sim2sim workflow.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Sequence

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.utils.mujoco_sim.base_sim import DefaultEnv  # noqa: E402
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig  # noqa: E402
from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import UnitreeSdk2Bridge  # noqa: E402
from unitree_sdk2py.core.channel import ChannelFactoryInitialize  # noqa: E402

from task_sim_io import (  # noqa: E402
    CsvTimeline,
    ReplayCsvWriter,
    make_output_directory,
    resolve_recording,
    stage_recording_snapshot,
    write_metadata,
)


DEFAULT_RECORDING = REPO_ROOT / "sample_data" / "ztj" / "20260720_144342_g1_sim"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "change_ckpt" / "data"

# Keep this synchronized with
# gear_sonic_deploy/.../include/policy_parameters.hpp::default_angles.  These
# differ from the legacy DEFAULT_DOF_ANGLES in the simulator YAML and are used
# to distinguish deploy's three-second INIT ramp from real policy commands.
DEPLOY_DEFAULT_ANGLES = np.asarray(
    [
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        0.0,
        0.0,
        0.0,
        0.2,
        0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
        0.2,
        -0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float64,
)

ROOT_ASSIST_WIDTH = {
    "none": 0,
    "xy": 2,
    "xyz": 3,
}


def _apply_source_root_assist(
    qpos: np.ndarray,
    qvel: np.ndarray,
    source_qpos: np.ndarray,
    source_qvel: np.ndarray,
    mode: str,
    *,
    source_velocity_active: bool,
) -> np.ndarray:
    """Hard-align selected floating-base coordinates to the source recording.

    Returns the pre-correction position error ``source - simulation``.  The
    source velocity is copied for active source playback and set to zero while
    the final source row is held.
    """

    try:
        width = ROOT_ASSIST_WIDTH[mode]
    except KeyError as exc:
        raise ValueError(f"unsupported root assist mode: {mode}") from exc
    if width == 0:
        return np.empty(0, dtype=np.float64)
    for name, value, minimum in (
        ("qpos", qpos, width),
        ("source_qpos", source_qpos, width),
        ("qvel", qvel, width),
        ("source_qvel", source_qvel, width),
    ):
        if value.ndim != 1 or len(value) < minimum:
            raise ValueError(
                f"{name} must be a 1-D vector with at least {minimum} values, "
                f"got shape {value.shape}"
            )
        if not np.isfinite(value[:minimum]).all():
            raise ValueError(
                f"{name} contains non-finite root-assist values: "
                f"{value[:minimum].tolist()}"
            )

    error = source_qpos[:width] - qpos[:width]
    qpos[:width] = source_qpos[:width]
    if source_velocity_active:
        qvel[:width] = source_qvel[:width]
    else:
        qvel[:width] = 0.0
    return error.copy()


def _joint_values(model: mujoco.MjModel, data: mujoco.MjData, joint_ids: np.ndarray, kind: str):
    addresses = model.jnt_qposadr[joint_ids] if kind == "qpos" else model.jnt_dofadr[joint_ids]
    values = data.qpos if kind == "qpos" else data.qvel
    # NumPy advanced indexing already returns a new array.
    return values[addresses]


class HandStatePublisher:
    """Publish immutable control-rate hand snapshots while MuJoCo releases the GIL."""

    _STOP = object()

    def __init__(self, env: "TaskSnapshotEnv") -> None:
        self.env = env
        self.queue: queue.Queue[object] = queue.Queue(maxsize=1)
        self.submitted = 0
        self.published = 0
        self.overruns = 0
        self.total_latency_s = 0.0
        self.max_latency_s = 0.0
        self.error: BaseException | None = None
        self.thread = threading.Thread(
            target=self._run,
            name="change-ckpt-hand-state-publisher",
            daemon=True,
        )
        self.thread.start()

    def _run(self) -> None:
        while True:
            payload = self.queue.get()
            try:
                if payload is self._STOP:
                    return
                queued_at, left_q, left_dq, right_q, right_dq = payload
                self.env._publish_hand_state(left_q, left_dq, right_q, right_dq)
                latency = time.perf_counter() - queued_at
                self.total_latency_s += latency
                self.max_latency_s = max(self.max_latency_s, latency)
                self.published += 1
            except BaseException as exc:  # propagate worker failures on the control thread
                self.error = exc
                return
            finally:
                self.queue.task_done()

    def submit(
        self,
        left_q: np.ndarray,
        left_dq: np.ndarray,
        right_q: np.ndarray,
        right_dq: np.ndarray,
    ) -> None:
        if self.error is not None:
            raise RuntimeError("hand-state publisher failed") from self.error
        payload = (time.perf_counter(), left_q, left_dq, right_q, right_dq)
        try:
            self.queue.put_nowait(payload)
        except queue.Full:
            # Preserve every control-rate state rather than silently dropping one,
            # but record the overrun so the experiment is marked invalid.
            self.overruns += 1
            self.queue.put(payload, timeout=0.05)
        self.submitted += 1

    def close(self) -> None:
        if not self.thread.is_alive():
            if self.error is not None:
                raise RuntimeError("hand-state publisher failed") from self.error
            return
        self.queue.join()
        self.queue.put(self._STOP)
        self.thread.join(timeout=2.0)
        if self.thread.is_alive():
            raise RuntimeError("hand-state publisher did not stop")
        if self.error is not None:
            raise RuntimeError("hand-state publisher failed") from self.error


class TaskSnapshotEnv(DefaultEnv):
    """Default DDS environment with task-safe actuator and reset behavior."""

    def __init__(self, *args, fall_height: float, **kwargs):
        self.fall_height = fall_height
        self.fallen = False
        self.invalid_state = False
        self.fall_message_printed = False
        self.control_timing = {
            "publish": 0.0,
            "command_pd": 0.0,
            "physics": 0.0,
            "max_publish": 0.0,
            "max_command_pd": 0.0,
            "max_physics": 0.0,
            "ticks": 0,
        }
        self.hand_state_publisher: HandStatePublisher | None = None
        super().__init__(*args, **kwargs)

    def init_scene(self):
        # ROBOT_SCENE is an absolute staged XML path. pathlib keeps an absolute
        # RHS unchanged when DefaultEnv joins it to GEAR_SONIC_ROOT.
        super().init_scene()
        self.elastic_band = None
        self._build_named_actuator_map()

    def _build_named_actuator_map(self) -> None:
        self.body_joint_names = [self.mj_model.joint(int(jid)).name for jid in self.body_joint_index]
        self.left_hand_joint_names = [
            self.mj_model.joint(int(jid)).name for jid in self.left_hand_index
        ]
        self.right_hand_joint_names = [
            self.mj_model.joint(int(jid)).name for jid in self.right_hand_index
        ]
        joint_to_actuator: dict[int, int] = {}
        for actuator_id in range(self.mj_model.nu):
            joint_id = int(self.mj_model.actuator_trnid[actuator_id, 0])
            if joint_id >= 0 and joint_id not in joint_to_actuator:
                joint_to_actuator[joint_id] = actuator_id

        def lookup(joints: Sequence[int], label: str) -> np.ndarray:
            missing = [self.mj_model.joint(int(jid)).name for jid in joints if int(jid) not in joint_to_actuator]
            if missing:
                raise ValueError(f"{label} joints have no motor actuator: {missing}")
            return np.asarray([joint_to_actuator[int(jid)] for jid in joints], dtype=np.int32)

        self.body_actuator_ids = lookup(self.body_joint_index, "body")
        self.left_hand_actuator_ids = lookup(self.left_hand_index, "left hand")
        self.right_hand_actuator_ids = lookup(self.right_hand_index, "right hand")
        self.robot_actuator_ids = np.concatenate(
            (self.body_actuator_ids, self.left_hand_actuator_ids, self.right_hand_actuator_ids)
        )
        self.drawer_actuator_ids = np.asarray(
            [
                actuator_id
                for actuator_id in range(self.mj_model.nu)
                if "drawer" in (self.mj_model.actuator(actuator_id).name or "").lower()
            ],
            dtype=np.int32,
        )
        overlap = set(self.robot_actuator_ids.tolist()) & set(self.drawer_actuator_ids.tolist())
        if overlap:
            raise ValueError(f"drawer and robot actuator sets overlap: {sorted(overlap)}")

        def actuator_limits(actuator_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            effort = np.asarray(self.torque_limit, dtype=np.float64)[actuator_ids].copy()
            limited = self.mj_model.actuator_ctrllimited[actuator_ids].astype(bool)
            ranges = self.mj_model.actuator_ctrlrange[actuator_ids].copy()
            return effort, limited, ranges

        self.body_control_limits = actuator_limits(self.body_actuator_ids)
        self.left_hand_control_limits = actuator_limits(self.left_hand_actuator_ids)
        self.right_hand_control_limits = actuator_limits(self.right_hand_actuator_ids)

        # CSV policy_received_dof_pos follows the robot joints' MuJoCo/XML order.
        robot_joints = np.concatenate(
            (self.body_joint_index, self.left_hand_index, self.right_hand_index)
        )
        self.robot_joints_qpos_order = robot_joints[
            np.argsort(self.mj_model.jnt_qposadr[robot_joints])
        ]
        if len(self.robot_joints_qpos_order) != 43:
            raise ValueError(
                f"expected 43 actuated robot joints, got {len(self.robot_joints_qpos_order)}"
            )
        robot_qpos_addresses = np.sort(
            self.mj_model.jnt_qposadr[self.robot_joints_qpos_order]
        )
        expected_robot_addresses = np.arange(7, 50, dtype=robot_qpos_addresses.dtype)
        if not np.array_equal(robot_qpos_addresses, expected_robot_addresses):
            raise ValueError(
                "expected floating base qpos[0:7] followed by 43 robot joints at "
                f"qpos[7:50], got {robot_qpos_addresses.tolist()}"
            )
        self.task_qpos_start = 50
        self.task_qpos_labels = self._qpos_labels_from(self.task_qpos_start)
        self._command_index_by_joint: dict[str, tuple[str, int]] = {}
        for group, names in (
            ("body", self.body_joint_names),
            ("left_hand", self.left_hand_joint_names),
            ("right_hand", self.right_hand_joint_names),
        ):
            self._command_index_by_joint.update(
                {name: (group, index) for index, name in enumerate(names)}
            )
        group_offset = {"body": 0, "left_hand": 29, "right_hand": 36}
        self.received_command_indices = np.asarray(
            [
                group_offset[self._command_index_by_joint[
                    self.mj_model.joint(int(joint_id)).name
                ][0]]
                + self._command_index_by_joint[
                    self.mj_model.joint(int(joint_id)).name
                ][1]
                for joint_id in self.robot_joints_qpos_order
            ],
            dtype=np.int32,
        )

    def _qpos_labels_from(self, first_qpos: int) -> list[str]:
        labels = [f"qpos[{index}]" for index in range(first_qpos, self.mj_model.nq)]
        joints = sorted(
            range(self.mj_model.njnt),
            key=lambda joint_id: int(self.mj_model.jnt_qposadr[joint_id]),
        )
        for position, joint_id in enumerate(joints):
            start = int(self.mj_model.jnt_qposadr[joint_id])
            end = (
                int(self.mj_model.jnt_qposadr[joints[position + 1]])
                if position + 1 < len(joints)
                else self.mj_model.nq
            )
            if end <= first_qpos:
                continue
            name = self.mj_model.joint(joint_id).name or f"joint_{joint_id}"
            for qpos_index in range(max(start, first_qpos), end):
                suffix = "" if end - start == 1 else f"[{qpos_index - start}]"
                labels[qpos_index - first_qpos] = f"{name}{suffix}"
        return labels

    def initialize_state(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        if qpos.shape != (self.mj_model.nq,):
            raise ValueError(f"qpos shape {qpos.shape}; model requires ({self.mj_model.nq},)")
        if qvel.shape != (self.mj_model.nv,):
            raise ValueError(f"qvel shape {qvel.shape}; model requires ({self.mj_model.nv},)")
        self.mj_data.qpos[:] = qpos
        self.mj_data.qvel[:] = qvel
        self.mj_data.time = 0.0
        self.mj_data.ctrl[:] = 0.0
        self.mj_data.qacc_warmstart[:] = 0.0
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self._initial_qpos = qpos.copy()
        self._initial_qvel = qvel.copy()

    def reset(self):
        # Manual reset, if called by a future UI hook, returns to the selected
        # recording state rather than MuJoCo's unrelated XML default state.
        if hasattr(self, "_initial_qpos"):
            self.initialize_state(self._initial_qpos, self._initial_qvel)
            self.fallen = False
            self.invalid_state = False
            self.fall_message_printed = False
        else:
            mujoco.mj_resetData(self.mj_model, self.mj_data)

    def prepare_obs(self):
        obs: dict[str, np.ndarray | float] = {
            # PublishLowState consumes these synchronously before mj_step, so
            # views are sufficient and avoid four allocations per control tick.
            "floating_base_pose": self.mj_data.qpos[:7],
            "floating_base_vel": self.mj_data.qvel[:6],
            "floating_base_acc": self.mj_data.qacc[:6],
            "secondary_imu_quat": self.mj_data.xquat[self.torso_index],
            "body_q": _joint_values(
                self.mj_model, self.mj_data, self.body_joint_index, "qpos"
            ),
            "body_dq": _joint_values(
                self.mj_model, self.mj_data, self.body_joint_index, "qvel"
            ),
            "body_ddq": self.mj_data.qacc[
                self.mj_model.jnt_dofadr[self.body_joint_index]
            ],
            "body_tau_est": self.mj_data.actuator_force[self.body_actuator_ids],
            "left_hand_q": _joint_values(
                self.mj_model, self.mj_data, self.left_hand_index, "qpos"
            ),
            "left_hand_dq": _joint_values(
                self.mj_model, self.mj_data, self.left_hand_index, "qvel"
            ),
            "right_hand_q": _joint_values(
                self.mj_model, self.mj_data, self.right_hand_index, "qpos"
            ),
            "right_hand_dq": _joint_values(
                self.mj_model, self.mj_data, self.right_hand_index, "qvel"
            ),
            "time": float(self.mj_data.time),
        }
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.mj_model,
            self.mj_data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.mj_model.body("torso_link").id,
            velocity,
            1,
        )
        # MuJoCo returns angular then linear; the bridge expects linear then angular.
        obs["secondary_imu_vel"] = np.concatenate((velocity[3:6], velocity[0:3]))
        return obs

    def publish_state(self) -> None:
        """Synchronous full publication used only during the startup gate."""
        self.obs = self.prepare_obs()
        self.unitree_bridge.PublishLowState(self.obs)

    def start_async_hand_state(self) -> None:
        if self.hand_state_publisher is None:
            self.hand_state_publisher = HandStatePublisher(self)

    def stop_async_hand_state(self) -> None:
        if self.hand_state_publisher is not None:
            self.hand_state_publisher.close()

    def _publish_body_state(self, obs: dict[str, np.ndarray | float]) -> None:
        """Publish policy-critical body/IMU topics synchronously.

        The C++ deployment has no odostate subscriber, so the unused odometry
        serialization is omitted during the measured rollout.  Startup still
        uses the repository bridge's complete publication method.
        """
        bridge = self.unitree_bridge
        for index in range(bridge.num_body_motor):
            motor = bridge.low_state.motor_state[index]
            motor.q = obs["body_q"][index]
            motor.dq = obs["body_dq"][index]
            motor.ddq = obs["body_ddq"][index]
            motor.tau_est = obs["body_tau_est"][index]
        bridge.low_state.imu_state.quaternion[:] = obs["floating_base_pose"][3:7]
        bridge.low_state.imu_state.gyroscope[:] = obs["floating_base_vel"][3:6]
        bridge.low_state.imu_state.accelerometer[:] = obs["floating_base_acc"][:3]
        bridge.torso_imu_state.quaternion[:] = obs["secondary_imu_quat"]
        bridge.torso_imu_state.gyroscope[:] = obs["secondary_imu_vel"][3:6]
        bridge.low_state.tick = int(float(obs["time"]) * 1e3)
        bridge.low_state_puber.Write(bridge.low_state)
        bridge.torso_imu_puber.Write(bridge.torso_imu_state)

    def _publish_hand_state(
        self,
        left_q: np.ndarray,
        left_dq: np.ndarray,
        right_q: np.ndarray,
        right_dq: np.ndarray,
    ) -> None:
        bridge = self.unitree_bridge
        for index in range(bridge.num_hand_motor):
            bridge.left_hand_state.motor_state[index].q = left_q[index]
            bridge.left_hand_state.motor_state[index].dq = left_dq[index]
        bridge.left_hand_state_puber.Write(bridge.left_hand_state)
        for index in range(bridge.num_hand_motor):
            bridge.right_hand_state.motor_state[index].q = right_q[index]
            bridge.right_hand_state.motor_state[index].dq = right_dq[index]
        bridge.right_hand_state_puber.Write(bridge.right_hand_state)

    def _pd_torques(
        self,
        joint_ids: np.ndarray,
        command_values: np.ndarray,
        control_limits: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> np.ndarray:
        q = _joint_values(self.mj_model, self.mj_data, joint_ids, "qpos")
        dq = _joint_values(self.mj_model, self.mj_data, joint_ids, "qvel")
        torque = (
            command_values[:, 4]
            + command_values[:, 2] * (command_values[:, 0] - q)
            + command_values[:, 3] * (command_values[:, 1] - dq)
        )
        effort_limits, limited, ranges = control_limits
        torque = np.clip(torque, -effort_limits, effort_limits)
        torque[limited] = np.clip(torque[limited], ranges[limited, 0], ranges[limited, 1])
        return torque

    @staticmethod
    def _command_values(commands: Sequence) -> np.ndarray:
        """Snapshot q/dq/kp/kd/tau once while the corresponding DDS lock is held."""
        return np.fromiter(
            (
                value
                for command in commands
                for value in (command.q, command.dq, command.kp, command.kd, command.tau)
            ),
            dtype=np.float64,
            count=len(commands) * 5,
        ).reshape(len(commands), 5)

    def _snapshot_commands(
        self,
        body_values: np.ndarray,
        left_values: np.ndarray,
        right_values: np.ndarray,
    ) -> dict[str, np.ndarray]:
        q_by_group = np.concatenate(
            (body_values[:, 0], left_values[:, 0], right_values[:, 0])
        )
        result = {"received_dof_pos": q_by_group[self.received_command_indices]}
        fields = ("q", "dq", "kp", "kd", "tau")
        for side, values in (("left", left_values), ("right", right_values)):
            for column, name in enumerate(fields):
                result[f"{side}_hand_{name}"] = values[:, column].copy()
        return result

    def command_snapshot(self) -> dict[str, np.ndarray]:
        bridge = self.unitree_bridge
        # Each DDS callback replaces the complete message object instead of
        # mutating it.  Holding a local Python reference therefore gives a
        # coherent snapshot without waiting on a callback lock.  This matters
        # at high control rates: a callback can otherwise be descheduled while
        # owning the lock and stall MuJoCo for tens of milliseconds.
        body_message = bridge.low_cmd
        left_message = bridge.left_hand_cmd
        right_message = bridge.right_hand_cmd
        body_commands = body_message.motor_cmd[: bridge.num_body_motor]
        left_commands = left_message.motor_cmd[: bridge.num_hand_motor]
        right_commands = right_message.motor_cmd[: bridge.num_hand_motor]
        return self._snapshot_commands(
            self._command_values(body_commands),
            self._command_values(left_commands),
            self._command_values(right_commands),
        )

    def control_tick(
        self,
        physics_substeps: int,
        *,
        capture_command: bool,
        defer_state_check: bool = False,
    ) -> dict[str, np.ndarray] | None:
        tick_started = time.perf_counter()
        self.obs = self.prepare_obs()
        self._publish_body_state(self.obs)
        publish_done = time.perf_counter()
        bridge = self.unitree_bridge
        body_message = bridge.low_cmd
        left_message = bridge.left_hand_cmd
        right_message = bridge.right_hand_cmd
        body_commands = body_message.motor_cmd[: bridge.num_body_motor]
        left_commands = left_message.motor_cmd[: bridge.num_hand_motor]
        right_commands = right_message.motor_cmd[: bridge.num_hand_motor]
        body_values = self._command_values(body_commands)
        left_values = self._command_values(left_commands)
        right_values = self._command_values(right_commands)
        applied_command = (
            self._snapshot_commands(body_values, left_values, right_values)
            if capture_command
            else None
        )

        self.mj_data.ctrl[:] = 0.0
        self.mj_data.ctrl[self.body_actuator_ids] = self._pd_torques(
            self.body_joint_index, body_values, self.body_control_limits
        )
        self.mj_data.ctrl[self.left_hand_actuator_ids] = self._pd_torques(
            self.left_hand_index, left_values, self.left_hand_control_limits
        )
        self.mj_data.ctrl[self.right_hand_actuator_ids] = self._pd_torques(
            self.right_hand_index, right_values, self.right_hand_control_limits
        )
        # All non-robot actuators, including all three drawer motors, stay zero.
        if len(self.drawer_actuator_ids):
            self.mj_data.ctrl[self.drawer_actuator_ids] = 0.0

        command_done = time.perf_counter()
        if self.hand_state_publisher is None:
            raise RuntimeError("asynchronous hand-state publisher was not started")
        self.hand_state_publisher.submit(
            self.obs["left_hand_q"],
            self.obs["left_hand_dq"],
            self.obs["right_hand_q"],
            self.obs["right_hand_dq"],
        )
        mujoco.mj_step(self.mj_model, self.mj_data, nstep=physics_substeps)
        if not defer_state_check:
            self.check_fall()
        physics_done = time.perf_counter()
        self.control_timing["publish"] += publish_done - tick_started
        self.control_timing["command_pd"] += command_done - publish_done
        self.control_timing["physics"] += physics_done - command_done
        self.control_timing["max_publish"] = max(
            self.control_timing["max_publish"], publish_done - tick_started
        )
        self.control_timing["max_command_pd"] = max(
            self.control_timing["max_command_pd"], command_done - publish_done
        )
        self.control_timing["max_physics"] = max(
            self.control_timing["max_physics"], physics_done - command_done
        )
        self.control_timing["ticks"] += 1
        return applied_command

    def check_fall(self):
        # Deliberately do not call reset: a fall is an experiment result.
        self.invalid_state = not (
            np.isfinite(self.mj_data.qpos).all() and np.isfinite(self.mj_data.qvel).all()
        )
        if self.invalid_state:
            print("[task-sim] non-finite qpos/qvel detected; state is preserved")
            return
        self.fallen = bool(self.mj_data.qpos[2] < self.fall_height)
        if self.fallen and not self.fall_message_printed:
            print(
                f"[task-sim] robot fell: base height={self.mj_data.qpos[2]:.3f} m "
                f"(< {self.fall_height:.3f} m); state is preserved"
            )
            self.fall_message_printed = True


@dataclass
class RunResult:
    reason: str
    samples: int
    fallen: bool
    invalid_state: bool
    simulated_seconds: float
    wall_seconds: float
    timing_valid: bool
    max_schedule_lag_s: float
    deadline_rebases: int


class TaskSimulator:
    def __init__(self, config: dict, args: argparse.Namespace, timeline: CsvTimeline):
        self.args = args
        self.timeline = timeline
        self.physics_dt = float(args.physics_dt)
        self.control_dt = float(args.control_dt)
        ratio = self.control_dt / self.physics_dt
        self.physics_substeps = int(round(ratio))
        if self.physics_substeps < 1 or not np.isclose(
            self.physics_substeps * self.physics_dt, self.control_dt, atol=1e-12
        ):
            raise ValueError(
                f"control_dt ({self.control_dt}) must be an integer multiple of "
                f"physics_dt ({self.physics_dt})"
            )
        source_ratio = self.control_dt / float(args.source_dt)
        self.source_rows_per_control = int(round(source_ratio))
        if self.source_rows_per_control < 1 or not np.isclose(
            self.source_rows_per_control * args.source_dt,
            self.control_dt,
            atol=1e-12,
        ):
            raise ValueError(
                f"control_dt ({self.control_dt}) must be an integer multiple of "
                f"source_dt ({args.source_dt})"
            )

        try:
            if config.get("INTERFACE"):
                ChannelFactoryInitialize(config["DOMAIN_ID"], config["INTERFACE"])
            else:
                ChannelFactoryInitialize(config["DOMAIN_ID"])
        except Exception as exc:
            print(f"[task-sim] DDS channel initialization note: {exc}")

        self.env = TaskSnapshotEnv(
            config,
            env_name="default",
            onscreen=args.viewer,
            offscreen=False,
            enable_image_publish=False,
            fall_height=args.fall_height,
        )
        self.bridge = UnitreeSdk2Bridge(config)
        self.env.set_unitree_bridge(self.bridge)
        qpos, qvel = timeline.state()
        self.env.initialize_state(qpos, qvel)
        self.writer: ReplayCsvWriter | None = None
        self.max_schedule_lag_s = 0.0
        self.deadline_rebases = 0
        self.schedule_lag_events: list[dict[str, float | int]] = []
        self.root_assist_width = ROOT_ASSIST_WIDTH[args.root_assist]
        self.root_assist_ticks = 0
        self.root_assist_pre_alignment_sq_sum = 0.0
        self.root_assist_pre_alignment_max = 0.0
        self.root_assist_post_alignment_sq_sum = 0.0
        self.root_assist_post_alignment_max = 0.0
        self.root_assist_time_s = 0.0
        self.root_assist_time_max_s = 0.0

    def _write_gate_status(self, status: str) -> None:
        if self.args.gate_status_file is None:
            return
        path = self.args.gate_status_file.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(status + "\n", encoding="utf-8")

    def _viewer_running(self) -> bool:
        return self.env.viewer is None or self.env.viewer.is_running()

    def _apply_root_assist(
        self,
        source_qpos: np.ndarray,
        source_qvel: np.ndarray,
        *,
        source_velocity_active: bool,
    ) -> None:
        if self.root_assist_width == 0:
            return
        started = time.perf_counter()
        error = _apply_source_root_assist(
            self.env.mj_data.qpos,
            self.env.mj_data.qvel,
            source_qpos,
            source_qvel,
            self.args.root_assist,
            source_velocity_active=source_velocity_active,
        )
        # The floating base was moved discontinuously.  Clear the stale solver
        # warm start and recompute kinematics, contacts and sensors before the
        # next policy observation/physics tick.
        self.env.mj_data.qacc_warmstart[:] = 0.0
        mujoco.mj_forward(self.env.mj_model, self.env.mj_data)
        elapsed = time.perf_counter() - started
        correction = float(np.linalg.norm(error))
        self.root_assist_ticks += 1
        self.root_assist_pre_alignment_sq_sum += correction * correction
        self.root_assist_pre_alignment_max = max(
            self.root_assist_pre_alignment_max, correction
        )
        self.root_assist_time_s += elapsed
        self.root_assist_time_max_s = max(
            self.root_assist_time_max_s, elapsed
        )

    def _record_root_assist_post_alignment(self, source_qpos: np.ndarray) -> None:
        if self.root_assist_width == 0:
            return
        error = (
            source_qpos[: self.root_assist_width]
            - self.env.mj_data.qpos[: self.root_assist_width]
        )
        magnitude = float(np.linalg.norm(error))
        self.root_assist_post_alignment_sq_sum += magnitude * magnitude
        self.root_assist_post_alignment_max = max(
            self.root_assist_post_alignment_max, magnitude
        )

    def root_assist_summary(self) -> dict[str, object]:
        ticks = self.root_assist_ticks
        components = ["x", "y", "z"][: self.root_assist_width]
        source_rate_hz = 1.0 / float(self.args.source_dt)
        control_rate_hz = 1.0 / self.control_dt
        return {
            "enabled": self.root_assist_width > 0,
            "mode": self.args.root_assist,
            "position_components": components,
            "source_clock": (
                f"current {source_rate_hz:g} Hz recording row sampled by the "
                f"{control_rate_hz:g} Hz simulator"
            ),
            "position_update": (
                f"hard overwrite after each {control_rate_hz:g} Hz control/physics "
                "tick and before CSV capture/the next policy observation"
            ),
            "velocity_update": (
                "copy matching source linear velocity; zero it while the final "
                "source row is held"
            ),
            "ticks": ticks,
            "pre_alignment_error_rms_m": (
                float(np.sqrt(self.root_assist_pre_alignment_sq_sum / ticks))
                if ticks
                else 0.0
            ),
            "pre_alignment_error_max_m": self.root_assist_pre_alignment_max,
            "post_alignment_error_rms_m": (
                float(np.sqrt(self.root_assist_post_alignment_sq_sum / ticks))
                if ticks
                else 0.0
            ),
            "post_alignment_error_max_m": self.root_assist_post_alignment_max,
            "mean_compute_ms": (
                self.root_assist_time_s * 1000.0 / ticks if ticks else 0.0
            ),
            "max_compute_ms": self.root_assist_time_max_s * 1000.0,
            "evaluation_note": (
                "Oracle source-root assistance changes the dynamics and is not "
                "an unassisted checkpoint result."
            ),
        }

    def wait_for_command(self) -> bool:
        if not self.args.wait_for_lowcmd:
            return True
        print("[task-sim] waiting for rt/lowcmd; physics is paused at the CSV state")
        self._write_gate_status("waiting_for_lowcmd")
        next_viewer = time.monotonic()
        first_command_at: float | None = None
        deploy_init_complete = False
        try:
            while self._viewer_running():
                self.env.publish_state()
                now = time.monotonic()
                if self.bridge.low_cmd_received:
                    if first_command_at is None:
                        first_command_at = now
                        print("[task-sim] first rt/lowcmd received; waiting for deploy INIT")
                        self._write_gate_status("waiting_for_deploy_init")
                    with self.bridge.low_cmd_lock:
                        target = np.asarray(
                            [
                                self.bridge.low_cmd.motor_cmd[index].q
                                for index in range(self.bridge.num_body_motor)
                            ],
                            dtype=np.float64,
                        )
                    distance = float(np.max(np.abs(target - DEPLOY_DEFAULT_ANGLES)))
                    if not self.args.wait_for_policy_command:
                        print(
                            "[task-sim] policy-command gate disabled; starting on first lowcmd"
                        )
                        return True
                    if not deploy_init_complete and distance <= self.args.command_gate_tolerance:
                        deploy_init_complete = True
                        print(
                            "[task-sim] deploy INIT reached its default pose; "
                            "waiting for the first policy command"
                        )
                        self._write_gate_status("deploy_init_ready")
                    elif deploy_init_complete and distance > self.args.command_gate_tolerance:
                        print(
                            f"[task-sim] policy command detected (max target delta "
                            f"{distance:.6f} rad); starting physics"
                        )
                        self._write_gate_status("running")
                        return True
                    if (
                        first_command_at is not None
                        and now - first_command_at > self.args.command_gate_timeout
                    ):
                        raise TimeoutError(
                            "timed out waiting for deploy's first policy command. "
                            "Ensure the CSV publisher sent start=true, or use "
                            "--no-wait-for-policy-command for manual diagnostics."
                        )
                if self.env.viewer is not None and now >= next_viewer:
                    self.env.update_viewer()
                    next_viewer = now + self.args.viewer_dt
                time.sleep(min(self.control_dt, 0.01))
        except KeyboardInterrupt:
            return False
        return False

    def _write_current(
        self,
        sample_index: int,
        command: dict[str, np.ndarray] | None = None,
    ) -> None:
        if self.writer is None:
            return
        self.writer.write(
            source_row_index=self.timeline.current_row_index,
            sample_index=sample_index,
            control_time=sample_index * self.control_dt,
            mujoco_time=float(self.env.mj_data.time),
            qpos=self.env.mj_data.qpos.copy(),
            qvel=self.env.mj_data.qvel.copy(),
            command=self.env.command_snapshot() if command is None else command,
        )

    def run(self) -> RunResult:
        if not self.wait_for_command():
            return RunResult(
                "stopped while waiting for lowcmd",
                0,
                False,
                False,
                0.0,
                0.0,
                False,
                0.0,
                0,
            )
        print(
            f"[task-sim] running: control={1 / self.control_dt:.1f} Hz, "
            f"physics={1 / self.physics_dt:.1f} Hz, substeps={self.physics_substeps}, "
            f"source_rows_per_control={self.source_rows_per_control}, "
            f"root_assist={self.args.root_assist}"
        )
        if self.root_assist_width:
            print(
                "[task-sim] WARNING: oracle source-root assistance is enabled; "
                "this is an assisted diagnostic, not unassisted checkpoint performance"
            )
        self.env.start_async_hand_state()
        sample_index = 0
        self._write_current(sample_index)
        source_end_at: int | None = None
        post_steps = int(round(self.args.post_rollout_seconds / self.control_dt))
        reason = "viewer closed"
        next_deadline = time.monotonic() + self.control_dt
        wall_started = time.monotonic()
        next_viewer = time.monotonic()

        try:
            while self._viewer_running():
                if self.args.max_steps is not None and sample_index >= self.args.max_steps:
                    reason = "max steps reached"
                    break
                if self.env.fallen and self.args.stop_on_fall:
                    reason = "robot fell"
                    break
                if self.env.invalid_state:
                    reason = "non-finite simulation state"
                    break

                advanced = True
                for _ in range(self.source_rows_per_control):
                    if not self.timeline.advance():
                        advanced = False
                        break
                if not advanced and source_end_at is None:
                    source_end_at = sample_index
                    print(
                        f"[task-sim] source CSV ended; holding the final command for "
                        f"{self.args.post_rollout_seconds:.3f} s"
                    )
                if source_end_at is not None:
                    if self.args.stop_at_source_end and sample_index - source_end_at >= post_steps:
                        reason = "source CSV ended"
                        break

                source_qpos: np.ndarray | None = None
                if self.root_assist_width:
                    source_qpos, source_qvel = self.timeline.state()
                applied_command = self.env.control_tick(
                    self.physics_substeps,
                    capture_command=self.writer is not None,
                    # XYZ assistance may restore base height after the physics
                    # step.  Judge the resulting state only after that
                    # explicitly requested correction has been applied.
                    defer_state_check=self.root_assist_width > 0,
                )
                if source_qpos is not None:
                    self._apply_root_assist(
                        source_qpos,
                        source_qvel,
                        source_velocity_active=advanced,
                    )
                    self._record_root_assist_post_alignment(source_qpos)
                    self.env.check_fall()
                sample_index += 1
                self._write_current(sample_index, applied_command)

                now = time.monotonic()
                if self.env.viewer is not None and now >= next_viewer:
                    self.env.update_viewer()
                    next_viewer = now + self.args.viewer_dt
                before_sleep = time.monotonic()
                self.max_schedule_lag_s = max(
                    self.max_schedule_lag_s, before_sleep - next_deadline
                )
                current_lag = before_sleep - next_deadline
                if current_lag > 0.01 and len(self.schedule_lag_events) < 50:
                    self.schedule_lag_events.append(
                        {
                            "sample_index": sample_index,
                            "simulated_time_s": float(self.env.mj_data.time),
                            "schedule_lag_s": current_lag,
                        }
                    )
                sleep_time = next_deadline - before_sleep
                if sleep_time > 0:
                    time.sleep(sleep_time)
                next_deadline += self.control_dt
                if next_deadline < time.monotonic() - 0.25:
                    # Ordinary scheduler/storage jitter is recoverable when the
                    # mean tick cost stays inside the selected control period;
                    # do not permanently shift physics behind the wall-clock
                    # reference. Only abandon catch-up after a genuinely long
                    # UI/OS stall, which also invalidates the experiment.
                    self.deadline_rebases += 1
                    next_deadline = time.monotonic() + self.control_dt
        except KeyboardInterrupt:
            reason = "keyboard interrupt"
        self.env.stop_async_hand_state()
        simulated_seconds = float(self.env.mj_data.time)
        wall_seconds = time.monotonic() - wall_started
        real_time_factor = simulated_seconds / max(wall_seconds, 1e-9)
        # The reference publisher advances from wall time.  A slow physics loop
        # therefore invalidates the checkpoint comparison even if MuJoCo itself
        # remains finite.  Keep tolerance below three 50 Hz reference frames.
        timing_valid = (
            real_time_factor >= 0.98
            and wall_seconds - simulated_seconds <= 0.05
            and self.max_schedule_lag_s <= 0.05
            and self.env.hand_state_publisher is not None
            and self.env.hand_state_publisher.overruns == 0
            and self.env.hand_state_publisher.published
            == self.env.hand_state_publisher.submitted
        )
        return RunResult(
            reason=reason,
            samples=sample_index + 1,
            fallen=self.env.fallen,
            invalid_state=self.env.invalid_state,
            simulated_seconds=simulated_seconds,
            wall_seconds=wall_seconds,
            timing_valid=timing_valid,
            max_schedule_lag_s=self.max_schedule_lag_s,
            deadline_rebases=self.deadline_rebases,
        )

    def close(self) -> None:
        self.env.stop_async_hand_state()
        if self.writer is not None:
            self.writer.close()
        if self.env.viewer is not None:
            self.env.viewer.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a recorded MuJoCo task snapshot with the existing SONIC DDS deploy."
    )
    parser.add_argument(
        "recording",
        nargs="?",
        default=str(DEFAULT_RECORDING),
        help="recording directory or its data.csv",
    )
    initial = parser.add_mutually_exclusive_group()
    initial.add_argument("--policy-seq", type=int, help="initialize at this policy_seq")
    initial.add_argument(
        "--policy-offset",
        type=int,
        help="initialize at this zero-based unique policy_seq offset (offset 0 may be partial)",
    )
    initial.add_argument("--row-index", type=int, help="initialize at this zero-based CSV data row")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT, help="root for timestamped runs"
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="use this exact existing/new run directory (for the unified launcher)",
    )
    parser.add_argument(
        "--asset-model-root",
        type=Path,
        help="mujoco/model directory containing g1/meshes and task_assets",
    )
    parser.add_argument("--save-csv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--viewer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--wait-for-lowcmd", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--wait-for-policy-command",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="after lowcmd arrives, keep physics frozen through deploy's INIT ramp",
    )
    parser.add_argument("--command-gate-timeout", type=float, default=120.0)
    parser.add_argument("--command-gate-tolerance", type=float, default=1e-4)
    parser.add_argument(
        "--gate-status-file",
        type=Path,
        help="optional status file for launcher synchronization",
    )
    parser.add_argument(
        "--stop-at-source-end", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--stop-on-fall", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--physics-dt", type=float, default=0.001)
    # The copied simulator accepts any exact integer timing ratio.  The unified
    # launcher selects either its default 200 Hz profile (two 400 Hz source
    # rows/tick) or the diagnostic 400 Hz profile (one source row/tick).
    parser.add_argument("--control-dt", type=float, default=0.005)
    parser.add_argument("--source-dt", type=float, default=0.0025)
    parser.add_argument(
        "--root-assist",
        choices=tuple(ROOT_ASSIST_WIDTH),
        default="none",
        help=(
            "hard-align robot root position/linear velocity to the current source "
            "recording row after each physics tick and before CSV capture/the next "
            "policy observation: none (default), xy (recommended diagnostic), or xyz"
        ),
    )
    parser.add_argument("--viewer-dt", type=float, default=0.02)
    parser.add_argument("--post-rollout-seconds", type=float, default=1.0)
    parser.add_argument("--fall-height", type=float, default=0.2)
    parser.add_argument("--interface", default="sim", help="Unitree DDS interface (default: sim/lo)")
    parser.add_argument("--max-steps", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="stage/load/validate the model and CSV without starting DDS or physics",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    recording_dir, csv_path = resolve_recording(args.recording)
    timeline = CsvTimeline(
        csv_path,
        policy_seq=args.policy_seq,
        row_index=args.row_index,
        policy_offset=args.policy_offset,
    )
    run_dir: Path | None = None
    if not args.dry_run and args.run_dir is not None:
        # Unified runs always retain metadata/log-sidecars in their assigned
        # directory, even when the large replay CSV is disabled.
        run_dir = args.run_dir.expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
    elif args.save_csv and not args.dry_run:
        if args.run_dir is None:
            run_dir = make_output_directory(
                args.output_dir, recording_dir, timeline.start_policy_seq
            )
    staged = stage_recording_snapshot(
        recording_dir, run_dir, asset_model_root=args.asset_model_root
    )

    print(f"[task-sim] source: {csv_path}")
    print(
        f"[task-sim] initialization: row={timeline.start_row_index}, "
        f"policy_seq={timeline.start_policy_seq}"
    )
    print(f"[task-sim] scene: {staged.scene_path}")
    print(f"[task-sim] assets: {staged.asset_model_root}")
    if run_dir is not None:
        print(f"[task-sim] output: {run_dir}")
    else:
        print("[task-sim] CSV saving disabled")

    # Validate dimensions before DDS creates any background callbacks.
    model = mujoco.MjModel.from_xml_path(str(staged.scene_path))
    qpos, qvel = timeline.state()
    if len(qpos) != model.nq or len(qvel) != model.nv:
        raise ValueError(
            f"CSV/model state mismatch: qpos {len(qpos)}/{model.nq}, "
            f"qvel {len(qvel)}/{model.nv}"
        )
    print(
        f"[task-sim] validated model: nq={model.nq}, nv={model.nv}, "
        f"nu={model.nu}, XML timestep={model.opt.timestep:g}"
    )
    del model
    if args.dry_run:
        timeline.close()
        staged.close()
        print("[task-sim] dry-run passed")
        return 0

    config_obj = SimLoopConfig(
        interface=args.interface,
        sim_frequency=int(round(1 / args.physics_dt)),
        enable_onscreen=args.viewer,
        enable_offscreen=False,
        with_hands=True,
    )
    config = config_obj.load_wbc_yaml()
    config.update(
        {
            "ENV_NAME": "default",
            "ROBOT_SCENE": str(staged.scene_path),
            "SIMULATE_DT": args.physics_dt,
            "ENABLE_ONSCREEN": args.viewer,
            "ENABLE_OFFSCREEN": False,
            "ENABLE_ELASTIC_BAND": False,
        }
    )

    simulator: TaskSimulator | None = None
    metadata_path = run_dir / "run_metadata.json" if run_dir is not None else None
    try:
        simulator = TaskSimulator(config, args, timeline)
        if run_dir is not None and args.save_csv:
            simulator.writer = ReplayCsvWriter(
                run_dir / "data.csv",
                csv_path,
                timeline.header,
                staged.scene_path,
                timeline.qpos_columns,
                timeline.qvel_columns,
            )
        result = simulator.run()
        print(
            f"[task-sim] stopped: {result.reason}; samples={result.samples}, "
            f"simulated={result.simulated_seconds:.3f}s, fallen={result.fallen}, "
            f"invalid={result.invalid_state}, wall={result.wall_seconds:.3f}s, "
            f"real_time_factor="
            f"{result.simulated_seconds / max(result.wall_seconds, 1e-9):.3f}"
        )
        print(
            f"[task-sim] wall-clock validity: valid={result.timing_valid}, "
            f"max_schedule_lag={result.max_schedule_lag_s * 1000:.3f}ms, "
            f"deadline_rebases={result.deadline_rebases}"
        )
        timing = simulator.env.control_timing
        if timing["ticks"]:
            scale = 1000.0 / timing["ticks"]
            print(
                "[task-sim] mean control cost: "
                f"publish={timing['publish'] * scale:.3f}ms, "
                f"command_pd={timing['command_pd'] * scale:.3f}ms, "
                f"physics={timing['physics'] * scale:.3f}ms; max: "
                f"publish={timing['max_publish'] * 1000:.3f}ms, "
                f"command_pd={timing['max_command_pd'] * 1000:.3f}ms, "
                f"physics={timing['max_physics'] * 1000:.3f}ms"
            )
        hand_timing = simulator.env.hand_state_publisher
        if hand_timing is not None:
            hand_mean_ms = (
                hand_timing.total_latency_s * 1000.0 / hand_timing.published
                if hand_timing.published
                else 0.0
            )
            print(
                "[task-sim] asynchronous hand state: "
                f"published={hand_timing.published}/{hand_timing.submitted}, "
                f"overruns={hand_timing.overruns}, mean_latency={hand_mean_ms:.3f}ms, "
                f"max_latency={hand_timing.max_latency_s * 1000:.3f}ms"
            )
        root_assist = simulator.root_assist_summary()
        if root_assist["enabled"]:
            print(
                "[task-sim] root assist: "
                f"mode={root_assist['mode']}, ticks={root_assist['ticks']}, "
                f"pre-alignment RMS/max="
                f"{root_assist['pre_alignment_error_rms_m']:.6f}/"
                f"{root_assist['pre_alignment_error_max_m']:.6f} m, "
                f"post-alignment RMS/max="
                f"{root_assist['post_alignment_error_rms_m']:.6f}/"
                f"{root_assist['post_alignment_error_max_m']:.6f} m"
            )
        if metadata_path is not None:
            source_qpos, _ = timeline.state()
            initial_qpos = simulator.env._initial_qpos.copy()
            final_qpos = simulator.env.mj_data.qpos.copy()
            task_start = simulator.env.task_qpos_start
            task_initial = initial_qpos[task_start:]
            task_final = final_qpos[task_start:]
            task_source = source_qpos[task_start:]
            write_metadata(
                metadata_path,
                {
                    "source_recording": str(recording_dir),
                    "source_csv": str(csv_path),
                    "scene_path": str(staged.scene_path),
                    "asset_model_root": str(staged.asset_model_root),
                    "initial_row_index": timeline.start_row_index,
                    "initial_policy_seq": timeline.start_policy_seq,
                    "physics_dt": args.physics_dt,
                    "physics_rate_hz": 1.0 / args.physics_dt,
                    "physics_substeps_per_control": simulator.physics_substeps,
                    "control_dt": args.control_dt,
                    "control_rate_hz": 1.0 / args.control_dt,
                    "source_dt": args.source_dt,
                    "source_rate_hz": 1.0 / args.source_dt,
                    "source_rows_per_control": simulator.source_rows_per_control,
                    "viewer_dt": args.viewer_dt,
                    "stop_reason": result.reason,
                    "samples": result.samples,
                    "simulated_seconds": result.simulated_seconds,
                    "wall_seconds": result.wall_seconds,
                    "real_time_factor": (
                        result.simulated_seconds / max(result.wall_seconds, 1e-9)
                    ),
                    "wall_clock_timing_valid": result.timing_valid,
                    "wall_clock_validity_rule": (
                        "RTF >= 0.98, wall-sim drift <= 0.05 s, and maximum "
                        "single schedule lag <= 0.05 s"
                    ),
                    "max_schedule_lag_s": result.max_schedule_lag_s,
                    "deadline_rebases": result.deadline_rebases,
                    "schedule_lag_events": simulator.schedule_lag_events,
                    "root_assist": root_assist,
                    "control_timing_mean_ms": {
                        name: (
                            timing[name] * 1000.0 / timing["ticks"]
                            if timing["ticks"]
                            else 0.0
                        )
                        for name in ("publish", "command_pd", "physics")
                    },
                    "control_timing_max_ms": {
                        name.removeprefix("max_"): timing[name] * 1000.0
                        for name in ("max_publish", "max_command_pd", "max_physics")
                    },
                    "hand_state_publish": {
                        "rate_hz": 1.0 / args.control_dt,
                        "mode": (
                            "two hand DDS topics overlap MuJoCo physics; "
                            "body and IMU remain synchronous"
                        ),
                        "submitted": hand_timing.submitted if hand_timing is not None else 0,
                        "published": hand_timing.published if hand_timing is not None else 0,
                        "overruns": hand_timing.overruns if hand_timing is not None else 0,
                        "mean_end_to_end_ms": (
                            hand_timing.total_latency_s * 1000.0 / hand_timing.published
                            if hand_timing is not None and hand_timing.published
                            else 0.0
                        ),
                        "max_end_to_end_ms": (
                            hand_timing.max_latency_s * 1000.0
                            if hand_timing is not None
                            else 0.0
                        ),
                    },
                    "fallen": result.fallen,
                    "invalid_state": result.invalid_state,
                    "final_base_height_m": float(final_qpos[2]),
                    "initial_qpos": initial_qpos.tolist(),
                    "final_qpos": final_qpos.tolist(),
                    "current_source_qpos": source_qpos.tolist(),
                    "task_qpos_start": task_start,
                    "task_qpos_labels": simulator.env.task_qpos_labels,
                    "initial_task_qpos": task_initial.tolist(),
                    "final_task_qpos": task_final.tolist(),
                    "current_source_task_qpos": task_source.tolist(),
                    "task_qpos_max_abs_error_to_current_source": float(
                        np.max(np.abs(task_final - task_source))
                    ),
                    "task_qpos_motion_l2_from_initial": float(
                        np.linalg.norm(task_final - task_initial)
                    ),
                    "source_task_qpos_motion_l2_from_initial": float(
                        np.linalg.norm(task_source - task_initial)
                    ),
                    "policy_telemetry_valid": False,
                    "policy_telemetry_note": (
                        "DDS exposes q targets but not encoder tokens/raw actions; "
                        "policy_valid is 0 and those fields are zero in this CSV."
                    ),
                    "source_reference_columns_note": (
                        "policy_seq/reference_motion are copied from the old source only "
                        "for provenance and replay timing. They are not a dump of the "
                        "new checkpoint's local encoder input, especially for low_latency. "
                        + (
                            "Source root qpos/qvel are also hard-applied by the explicitly "
                            f"enabled root-assist mode {args.root_assist!r}."
                            if args.root_assist != "none"
                            else "Source qpos/qvel are used only for initialization."
                        )
                    ),
                },
            )
        if result.invalid_state:
            return 5
        if result.fallen:
            return 4
        if not result.timing_valid:
            return 6
        return 0
    finally:
        if simulator is not None:
            simulator.close()
        timeline.close()
        staged.close()


if __name__ == "__main__":
    raise SystemExit(main())
