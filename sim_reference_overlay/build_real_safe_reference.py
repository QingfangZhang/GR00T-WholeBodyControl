#!/usr/bin/env python3
"""Build a staged-entry G1 reference for guarded real-robot experiments.

This tool does not certify the source motion as safe.  It wraps an explicitly
selected source interval with a stationary policy-default pose, a slow
minimum-jerk entry, a smooth stop, and a slow return to the policy-default
joint pose.  Synthetic body poses are recomputed with the exact MuJoCo model
used by the SONIC simulator.

The default frame interval (12..1158) is a conservative stage-1 subset chosen
specifically for ``dun.npz``.  It retains 74.5 percent of the usable source
motion and deliberately excludes the high-dynamic return segment.  Generate a
near-full variant only after the stage-1 motion has passed suspended tests.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Sequence

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

import convert_npz_motions as csvio
from ghost_overlay import G1_MUJOCO_JOINT_NAMES, inspect_robot_visual


FPS = csvio.EXPECTED_FPS
DT = 1.0 / FPS
DEFAULT_SOURCE_START_FRAME = 12
DEFAULT_SOURCE_END_FRAME = 1158
LOWER_BODY_ISAACLAB_INDEXES = np.asarray(
    [0, 1, 3, 4, 6, 7, 9, 10, 13, 14, 17, 18], dtype=np.int64
)
ARM_ISAACLAB_INDEXES = np.asarray(
    [
        index
        for index, name in enumerate(csvio.G1_ISAACLAB_JOINT_NAMES)
        if any(part in name for part in ("shoulder", "elbow", "wrist"))
    ],
    dtype=np.int64,
)
NON_ARM_ISAACLAB_INDEXES = np.setdiff1d(
    np.arange(csvio.NUM_JOINTS, dtype=np.int64), ARM_ISAACLAB_INDEXES
)

# Assignment semantics match policy_parameters.hpp:
#   q_mujoco[i] = q_isaaclab[ISAACLAB_TO_MUJOCO[i]]
ISAACLAB_TO_MUJOCO = np.asarray(
    [
        0,
        3,
        6,
        9,
        13,
        17,
        1,
        4,
        7,
        10,
        14,
        18,
        2,
        5,
        8,
        11,
        15,
        19,
        21,
        23,
        25,
        27,
        12,
        16,
        20,
        22,
        24,
        26,
        28,
    ],
    dtype=np.int64,
)

# Exact standing pose used by g1_deploy_onnx_ref during INIT, in MuJoCo / G1
# hardware order.  It is converted below instead of hand-maintaining a second
# IsaacLab-ordered constant.
DEFAULT_ANGLES_MUJOCO = np.asarray(
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
DEFAULT_ANGLES_ISAACLAB = np.empty(csvio.NUM_JOINTS, dtype=np.float64)
DEFAULT_ANGLES_ISAACLAB[ISAACLAB_TO_MUJOCO] = DEFAULT_ANGLES_MUJOCO

RELEASE_BODY_NAMES = tuple(
    csvio.G1_ISAACLAB_BODY_NAMES[int(index)]
    for index in csvio.RELEASE_BODY_INDEXES
)
FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")

# Bottom corners of the foot collision boxes in g1_29dof_with_hand.xml.
FOOT_BOTTOM_CORNERS_LOCAL = np.asarray(
    [
        [-0.05, -0.03, -0.035],
        [-0.05, 0.03, -0.035],
        [0.12, -0.03, -0.035],
        [0.12, 0.03, -0.035],
    ],
    dtype=np.float64,
)


class SafeMotionError(ValueError):
    """Raised when a requested safe wrapper violates its explicit contract."""


@dataclass(frozen=True)
class BuildConfig:
    source_start_frame: int = DEFAULT_SOURCE_START_FRAME
    source_end_frame: int = DEFAULT_SOURCE_END_FRAME
    initial_hold_s: float = 5.0
    entry_s: float = 5.0
    source_settle_s: float = 1.0
    brake_s: float = 0.5
    stop_hold_s: float = 1.0
    return_s: float = 6.0
    final_hold_s: float = 5.0
    brake_joint_speed_limit: float = 0.7
    brake_non_arm_speed_limit: float = 0.7


@dataclass(frozen=True)
class BuiltTrajectory:
    joint_pos: np.ndarray
    root_pos: np.ndarray
    root_quat_wxyz: np.ndarray
    segments: dict[str, slice]
    entry_foot_anchors: np.ndarray
    return_foot_anchors: np.ndarray


@dataclass(frozen=True)
class ValidationReport:
    total_frames: int
    duration_s: float
    retained_source_fraction: float
    retained_max_joint_speed: float
    retained_max_lower_body_speed: float
    retained_min_root_z: float
    entry_max_marker_displacement: float
    return_max_marker_displacement: float
    entry_max_joint_speed: float
    brake_max_joint_speed: float
    brake_max_non_arm_speed: float
    return_max_joint_speed: float
    max_quaternion_norm_deviation: float
    min_adjacent_quaternion_dot: float
    raw_reset_root_translation: float
    raw_reset_heading_deg: float
    max_joint_limit_violation: float


def _seconds_to_frames(seconds: float, label: str) -> int:
    if not np.isfinite(seconds) or seconds <= 0.0:
        raise SafeMotionError(f"{label} must be positive and finite")
    frames_float = seconds * FPS
    frames = int(round(frames_float))
    if abs(frames - frames_float) > 1.0e-9:
        raise SafeMotionError(f"{label} must be an exact multiple of {DT:g} seconds")
    return frames


def minimum_jerk(unit_time: np.ndarray) -> np.ndarray:
    """C2 interpolation scalar with zero velocity and acceleration at both ends."""

    u = np.asarray(unit_time, dtype=np.float64)
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def brake_displacement(unit_time: np.ndarray) -> np.ndarray:
    """Integrate a velocity smoothly from its initial value to zero.

    The polynomial has h(0)=0, h'(0)=1, h''(0)=0 and
    h(1)=1/2, h'(1)=h''(1)=0.
    """

    u = np.asarray(unit_time, dtype=np.float64)
    return u - u**3 + 0.5 * u**4


def rotation_from_wxyz(quaternion: np.ndarray) -> Rotation:
    value = np.asarray(quaternion, dtype=np.float64)
    return Rotation.from_quat(np.concatenate((value[..., 1:4], value[..., 0:1]), axis=-1))


def rotation_to_wxyz(rotation: Rotation) -> np.ndarray:
    value = rotation.as_quat()
    return np.concatenate((value[..., 3:4], value[..., 0:3]), axis=-1)


def heading_rotation(rotation: Rotation) -> Rotation:
    matrix = rotation.as_matrix()
    yaw = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
    return Rotation.from_euler("z", yaw)


def slerp_pair(start: Rotation, end: Rotation, fractions: np.ndarray) -> Rotation:
    keys = Rotation.from_quat(np.stack((start.as_quat(), end.as_quat()), axis=0))
    return Slerp([0.0, 1.0], keys)(np.asarray(fractions, dtype=np.float64))


def repeat_rotation(rotation: Rotation, count: int) -> Rotation:
    return Rotation.from_quat(np.repeat(rotation.as_quat()[None, :], count, axis=0))


class G1Kinematics:
    """Name-resolved FK using the same 43-DoF scene as the SONIC simulator."""

    def __init__(self, scene_path: Path) -> None:
        scene = scene_path.expanduser().resolve()
        if not scene.is_file():
            raise SafeMotionError(f"MuJoCo scene does not exist: {scene}")
        self.scene_path = scene
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.visual = inspect_robot_visual(self.model)
        if self.visual.body_joint_names != G1_MUJOCO_JOINT_NAMES:
            raise SafeMotionError("resolved G1 joint order does not match SONIC")

        self.release_body_ids = np.asarray(
            [self._body_id(name) for name in RELEASE_BODY_NAMES], dtype=np.int64
        )
        self.foot_body_ids = np.asarray(
            [self._body_id(name) for name in FOOT_BODY_NAMES], dtype=np.int64
        )
        self.joint_lower_isaaclab = np.full(csvio.NUM_JOINTS, -np.inf)
        self.joint_upper_isaaclab = np.full(csvio.NUM_JOINTS, np.inf)
        for mujoco_index, joint_name in enumerate(G1_MUJOCO_JOINT_NAMES):
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            if joint_id < 0:
                raise SafeMotionError(f"MuJoCo scene is missing joint {joint_name!r}")
            if bool(self.model.jnt_limited[joint_id]):
                isaaclab_index = int(ISAACLAB_TO_MUJOCO[mujoco_index])
                self.joint_lower_isaaclab[isaaclab_index] = self.model.jnt_range[
                    joint_id, 0
                ]
                self.joint_upper_isaaclab[isaaclab_index] = self.model.jnt_range[
                    joint_id, 1
                ]

    def _body_id(self, name: str) -> int:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise SafeMotionError(f"MuJoCo scene is missing body {name!r}")
        return int(body_id)

    def _set_pose(
        self,
        joint_pos_isaaclab: np.ndarray,
        root_pos: np.ndarray,
        root_rotation: Rotation,
    ) -> None:
        q = np.asarray(joint_pos_isaaclab, dtype=np.float64)
        p = np.asarray(root_pos, dtype=np.float64)
        if q.shape != (csvio.NUM_JOINTS,) or p.shape != (3,):
            raise SafeMotionError("internal FK pose has the wrong shape")
        self.data.qpos[:] = self.model.qpos0
        self.data.qvel[:] = 0.0
        root = self.visual.root_qpos_indices
        self.data.qpos[root[:3]] = p
        self.data.qpos[root[3:7]] = rotation_to_wxyz(root_rotation)
        self.data.qpos[self.visual.body_joint_qpos_indices] = q[
            ISAACLAB_TO_MUJOCO
        ]
        mujoco.mj_forward(self.model, self.data)

    def foot_markers(
        self,
        joint_pos_isaaclab: np.ndarray,
        root_pos: np.ndarray,
        root_rotation: Rotation,
    ) -> np.ndarray:
        self._set_pose(joint_pos_isaaclab, root_pos, root_rotation)
        result = np.empty((len(self.foot_body_ids), 4, 3), dtype=np.float64)
        for foot_index, body_id in enumerate(self.foot_body_ids):
            body_rotation = self.data.xmat[body_id].reshape(3, 3)
            result[foot_index] = (
                FOOT_BOTTOM_CORNERS_LOCAL @ body_rotation.T
                + self.data.xpos[body_id]
            )
        return result

    def fit_root_translation(
        self,
        joint_pos_isaaclab: np.ndarray,
        root_rotation: Rotation,
        fixed_foot_markers: np.ndarray,
    ) -> np.ndarray:
        relative = self.foot_markers(
            joint_pos_isaaclab, np.zeros(3, dtype=np.float64), root_rotation
        )
        anchors = np.asarray(fixed_foot_markers, dtype=np.float64)
        if anchors.shape != relative.shape:
            raise SafeMotionError("foot-marker anchors have the wrong shape")
        return np.mean(anchors - relative, axis=(0, 1))

    def release_body_pose(
        self,
        joint_pos_isaaclab: np.ndarray,
        root_pos: np.ndarray,
        root_rotation: Rotation,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._set_pose(joint_pos_isaaclab, root_pos, root_rotation)
        return (
            np.array(self.data.xpos[self.release_body_ids], copy=True),
            np.array(self.data.xquat[self.release_body_ids], copy=True),
        )


def _append_segment(
    name: str,
    joint_pos: np.ndarray,
    root_pos: np.ndarray,
    root_rotation: Rotation,
    *,
    joint_parts: list[np.ndarray],
    root_parts: list[np.ndarray],
    rotation_parts: list[Rotation],
    segments: dict[str, slice],
) -> None:
    count = int(joint_pos.shape[0])
    if root_pos.shape != (count, 3) or len(root_rotation) != count:
        raise SafeMotionError(f"internal segment {name!r} has inconsistent lengths")
    start = sum(part.shape[0] for part in joint_parts)
    segments[name] = slice(start, start + count)
    joint_parts.append(np.asarray(joint_pos, dtype=np.float64))
    root_parts.append(np.asarray(root_pos, dtype=np.float64))
    rotation_parts.append(root_rotation)


def build_root_and_joint_trajectory(
    source: csvio.PreparedMotion,
    kinematics: G1Kinematics,
    config: BuildConfig,
) -> BuiltTrajectory:
    start = config.source_start_frame
    end = config.source_end_frame
    if not 0 <= start < end < source.timesteps:
        raise SafeMotionError(
            f"source frame interval must satisfy 0 <= start < end < {source.timesteps}, "
            f"got {start}..{end}"
        )

    initial_count = _seconds_to_frames(config.initial_hold_s, "initial hold")
    entry_count = _seconds_to_frames(config.entry_s, "entry")
    settle_count = _seconds_to_frames(config.source_settle_s, "source settle")
    brake_count = _seconds_to_frames(config.brake_s, "brake")
    stop_count = _seconds_to_frames(config.stop_hold_s, "stop hold")
    return_count = _seconds_to_frames(config.return_s, "return")
    final_count = _seconds_to_frames(config.final_hold_s, "final hold")

    source_q = source.arrays["joint_pos"].astype(np.float64)
    source_dq = source.arrays["joint_vel"].astype(np.float64)
    source_root_pos = source.arrays["body_pos_w"][:, 0].astype(np.float64)
    source_root_quat = source.arrays["body_quat_w"][:, 0].astype(np.float64)
    source_root_lin_vel = source.arrays["body_lin_vel_w"][:, 0].astype(np.float64)
    source_root_ang_vel = source.arrays["body_ang_vel_w"][:, 0].astype(np.float64)
    source_root_rotations = rotation_from_wxyz(source_root_quat)

    entry_target_q = source_q[start]
    entry_target_p = source_root_pos[start]
    entry_target_rotation = source_root_rotations[start]
    entry_heading = heading_rotation(entry_target_rotation)
    entry_anchors = kinematics.foot_markers(
        entry_target_q, entry_target_p, entry_target_rotation
    )
    safe_start_p = kinematics.fit_root_translation(
        DEFAULT_ANGLES_ISAACLAB, entry_heading, entry_anchors
    )

    joint_parts: list[np.ndarray] = []
    root_parts: list[np.ndarray] = []
    rotation_parts: list[Rotation] = []
    segments: dict[str, slice] = {}

    _append_segment(
        "initial_hold",
        np.repeat(DEFAULT_ANGLES_ISAACLAB[None, :], initial_count, axis=0),
        np.repeat(safe_start_p[None, :], initial_count, axis=0),
        repeat_rotation(entry_heading, initial_count),
        joint_parts=joint_parts,
        root_parts=root_parts,
        rotation_parts=rotation_parts,
        segments=segments,
    )

    entry_u = np.arange(1, entry_count + 1, dtype=np.float64) / entry_count
    entry_s = minimum_jerk(entry_u)
    entry_q = (
        (1.0 - entry_s[:, None]) * DEFAULT_ANGLES_ISAACLAB
        + entry_s[:, None] * entry_target_q
    )
    entry_rotation = slerp_pair(entry_heading, entry_target_rotation, entry_s)
    entry_p = np.stack(
        [
            kinematics.fit_root_translation(entry_q[index], entry_rotation[index], entry_anchors)
            for index in range(entry_count)
        ],
        axis=0,
    )
    _append_segment(
        "entry",
        entry_q,
        entry_p,
        entry_rotation,
        joint_parts=joint_parts,
        root_parts=root_parts,
        rotation_parts=rotation_parts,
        segments=segments,
    )

    _append_segment(
        "source_settle",
        np.repeat(entry_target_q[None, :], settle_count, axis=0),
        np.repeat(entry_target_p[None, :], settle_count, axis=0),
        repeat_rotation(entry_target_rotation, settle_count),
        joint_parts=joint_parts,
        root_parts=root_parts,
        rotation_parts=rotation_parts,
        segments=segments,
    )

    _append_segment(
        "source_motion",
        source_q[start + 1 : end + 1],
        source_root_pos[start + 1 : end + 1],
        source_root_rotations[start + 1 : end + 1],
        joint_parts=joint_parts,
        root_parts=root_parts,
        rotation_parts=rotation_parts,
        segments=segments,
    )

    brake_u = np.arange(1, brake_count + 1, dtype=np.float64) / brake_count
    brake_h = brake_displacement(brake_u)
    brake_q = source_q[end] + (
        config.brake_s * brake_h[:, None] * source_dq[end]
    )
    brake_p = source_root_pos[end] + (
        config.brake_s * brake_h[:, None] * source_root_lin_vel[end]
    )
    brake_delta = Rotation.from_rotvec(
        config.brake_s * brake_h[:, None] * source_root_ang_vel[end]
    )
    brake_rotation = brake_delta * source_root_rotations[end]
    _append_segment(
        "brake",
        brake_q,
        brake_p,
        brake_rotation,
        joint_parts=joint_parts,
        root_parts=root_parts,
        rotation_parts=rotation_parts,
        segments=segments,
    )

    stop_q = brake_q[-1]
    stop_p = brake_p[-1]
    stop_rotation = brake_rotation[-1]
    return_anchors = kinematics.foot_markers(stop_q, stop_p, stop_rotation)
    _append_segment(
        "stop_hold",
        np.repeat(stop_q[None, :], stop_count, axis=0),
        np.repeat(stop_p[None, :], stop_count, axis=0),
        repeat_rotation(stop_rotation, stop_count),
        joint_parts=joint_parts,
        root_parts=root_parts,
        rotation_parts=rotation_parts,
        segments=segments,
    )

    return_u = np.arange(1, return_count + 1, dtype=np.float64) / return_count
    return_s = minimum_jerk(return_u)
    return_q = (
        (1.0 - return_s[:, None]) * stop_q
        + return_s[:, None] * DEFAULT_ANGLES_ISAACLAB
    )
    return_heading = heading_rotation(stop_rotation)
    return_rotation = slerp_pair(stop_rotation, return_heading, return_s)
    return_p = np.stack(
        [
            kinematics.fit_root_translation(
                return_q[index], return_rotation[index], return_anchors
            )
            for index in range(return_count)
        ],
        axis=0,
    )
    _append_segment(
        "return",
        return_q,
        return_p,
        return_rotation,
        joint_parts=joint_parts,
        root_parts=root_parts,
        rotation_parts=rotation_parts,
        segments=segments,
    )

    safe_end_p = return_p[-1]
    _append_segment(
        "final_hold",
        np.repeat(DEFAULT_ANGLES_ISAACLAB[None, :], final_count, axis=0),
        np.repeat(safe_end_p[None, :], final_count, axis=0),
        repeat_rotation(return_heading, final_count),
        joint_parts=joint_parts,
        root_parts=root_parts,
        rotation_parts=rotation_parts,
        segments=segments,
    )

    joint_pos = np.concatenate(joint_parts, axis=0)
    root_pos = np.concatenate(root_parts, axis=0)
    root_rotation = Rotation.concatenate(rotation_parts)
    root_quat = rotation_to_wxyz(root_rotation)
    return BuiltTrajectory(
        joint_pos=joint_pos,
        root_pos=root_pos,
        root_quat_wxyz=root_quat,
        segments=segments,
        entry_foot_anchors=entry_anchors,
        return_foot_anchors=return_anchors,
    )


def _make_quaternions_continuous(quaternion: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternion, dtype=np.float64).copy()
    norms = np.linalg.norm(result, axis=-1, keepdims=True)
    if np.any(norms == 0.0):
        raise SafeMotionError("FK produced a zero quaternion")
    result /= norms
    for frame in range(1, result.shape[0]):
        flip = np.sum(result[frame - 1] * result[frame], axis=-1) < 0.0
        result[frame, flip] *= -1.0
    return result


def _gradient(array: np.ndarray) -> np.ndarray:
    return np.gradient(np.asarray(array, dtype=np.float64), DT, axis=0).astype(
        np.float32
    )


def _angular_velocity(quaternion_wxyz: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    timesteps, bodies, _ = quaternion.shape
    if timesteps < 3:
        raise SafeMotionError("at least three frames are required for angular velocity")
    result = np.zeros((timesteps, bodies, 3), dtype=np.float64)
    for body in range(bodies):
        rotation = rotation_from_wxyz(quaternion[:, body])
        centered_delta = rotation[2:] * rotation[:-2].inv()
        result[1:-1, body] = centered_delta.as_rotvec() / (2.0 * DT)
        result[0, body] = result[1, body]
        result[-1, body] = result[-2, body]
    return result.astype(np.float32)


def compute_release_arrays(
    trajectory: BuiltTrajectory,
    kinematics: G1Kinematics,
) -> dict[str, np.ndarray]:
    timesteps = trajectory.joint_pos.shape[0]
    body_count = len(RELEASE_BODY_NAMES)
    body_pos = np.empty((timesteps, body_count, 3), dtype=np.float64)
    body_quat = np.empty((timesteps, body_count, 4), dtype=np.float64)
    root_rotations = rotation_from_wxyz(trajectory.root_quat_wxyz)
    for frame in range(timesteps):
        body_pos[frame], body_quat[frame] = kinematics.release_body_pose(
            trajectory.joint_pos[frame],
            trajectory.root_pos[frame],
            root_rotations[frame],
        )
    body_quat = _make_quaternions_continuous(body_quat)

    joint_pos = trajectory.joint_pos.astype(np.float32)
    body_pos_f32 = body_pos.astype(np.float32)
    body_quat_f32 = body_quat.astype(np.float32)
    return {
        "joint_pos": joint_pos,
        "joint_vel": _gradient(joint_pos),
        "body_pos_w": body_pos_f32,
        "body_quat_w": body_quat_f32,
        "body_lin_vel_w": _gradient(body_pos_f32),
        "body_ang_vel_w": _angular_velocity(body_quat_f32),
    }


def _maximum_foot_marker_displacement(
    trajectory: BuiltTrajectory,
    kinematics: G1Kinematics,
    segment: slice,
    anchors: np.ndarray,
) -> float:
    rotations = rotation_from_wxyz(trajectory.root_quat_wxyz[segment])
    maximum = 0.0
    for local_index, frame in enumerate(range(segment.start, segment.stop)):
        markers = kinematics.foot_markers(
            trajectory.joint_pos[frame],
            trajectory.root_pos[frame],
            rotations[local_index],
        )
        maximum = max(maximum, float(np.max(np.linalg.norm(markers - anchors, axis=-1))))
    return maximum


def validate_built_motion(
    source: csvio.PreparedMotion,
    trajectory: BuiltTrajectory,
    arrays: dict[str, np.ndarray],
    kinematics: G1Kinematics,
    config: BuildConfig,
) -> ValidationReport:
    for label, value in (
        ("brake joint-speed limit", config.brake_joint_speed_limit),
        ("brake non-arm speed limit", config.brake_non_arm_speed_limit),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise SafeMotionError(f"{label} must be positive and finite")

    timesteps = int(arrays["joint_pos"].shape[0])
    if not np.array_equal(arrays["joint_pos"][0], DEFAULT_ANGLES_ISAACLAB.astype(np.float32)):
        raise SafeMotionError("frame 0 is not the exact SONIC policy-default joint pose")
    if not np.array_equal(arrays["joint_pos"][-1], DEFAULT_ANGLES_ISAACLAB.astype(np.float32)):
        raise SafeMotionError("final frame is not the exact SONIC policy-default joint pose")

    for segment_name in ("initial_hold", "final_hold"):
        segment = trajectory.segments[segment_name]
        q = arrays["joint_pos"][segment]
        if not np.array_equal(q, np.repeat(q[:1], q.shape[0], axis=0)):
            raise SafeMotionError(f"{segment_name} is not exactly stationary")
        interior = slice(segment.start + 2, segment.stop - 2)
        if float(np.max(np.abs(arrays["joint_vel"][interior]))) > 1.0e-6:
            raise SafeMotionError(f"{segment_name} has nonzero interior joint velocity")
        if float(np.max(np.abs(arrays["body_lin_vel_w"][interior]))) > 1.0e-6:
            raise SafeMotionError(f"{segment_name} has nonzero interior body velocity")
        if float(np.max(np.abs(arrays["body_ang_vel_w"][interior]))) > 1.0e-6:
            raise SafeMotionError(f"{segment_name} has nonzero interior angular velocity")

    lower_violation = kinematics.joint_lower_isaaclab[None, :] - arrays["joint_pos"]
    upper_violation = arrays["joint_pos"] - kinematics.joint_upper_isaaclab[None, :]
    max_limit_violation = float(max(0.0, np.max(lower_violation), np.max(upper_violation)))
    if max_limit_violation > 2.0e-5:
        raise SafeMotionError(
            f"generated joint pose exceeds MuJoCo limits by {max_limit_violation:.9g} rad"
        )

    quaternion = arrays["body_quat_w"].astype(np.float64)
    quaternion_norm_deviation = float(
        np.max(np.abs(np.linalg.norm(quaternion, axis=-1) - 1.0))
    )
    adjacent_dot = np.sum(quaternion[:-1] * quaternion[1:], axis=-1)
    minimum_dot = float(np.min(adjacent_dot))
    if quaternion_norm_deviation > csvio.QUATERNION_NORM_TOLERANCE:
        raise SafeMotionError("generated body quaternion norm check failed")
    if minimum_dot < 0.0:
        raise SafeMotionError("generated body quaternions are not hemisphere-continuous")

    source_slice = slice(config.source_start_frame, config.source_end_frame + 1)
    source_velocity = np.abs(source.arrays["joint_vel"][source_slice])
    retained_max_joint_speed = float(np.max(source_velocity))
    retained_max_lower_body_speed = float(
        np.max(source_velocity[:, LOWER_BODY_ISAACLAB_INDEXES])
    )
    retained_min_root_z = float(
        np.min(source.arrays["body_pos_w"][source_slice, 0, 2])
    )
    retained_fraction = (config.source_end_frame - config.source_start_frame + 1) / (
        source.timesteps - config.source_start_frame
    )

    entry_marker_displacement = _maximum_foot_marker_displacement(
        trajectory,
        kinematics,
        trajectory.segments["entry"],
        trajectory.entry_foot_anchors,
    )
    return_marker_displacement = _maximum_foot_marker_displacement(
        trajectory,
        kinematics,
        trajectory.segments["return"],
        trajectory.return_foot_anchors,
    )

    joint_vel = arrays["joint_vel"]
    entry_max = float(np.max(np.abs(joint_vel[trajectory.segments["entry"]])))
    brake_max = float(np.max(np.abs(joint_vel[trajectory.segments["brake"]])))
    brake_non_arm_max = float(
        np.max(
            np.abs(
                joint_vel[trajectory.segments["brake"]][
                    :, NON_ARM_ISAACLAB_INDEXES
                ]
            )
        )
    )
    return_max = float(np.max(np.abs(joint_vel[trajectory.segments["return"]])))
    if (
        entry_max > 0.4
        or brake_max > config.brake_joint_speed_limit
        or brake_non_arm_max > config.brake_non_arm_speed_limit
        or return_max > 0.4
    ):
        raise SafeMotionError(
            "synthetic transition exceeded its guarded joint-speed bound: "
            f"entry={entry_max:.6g} (limit 0.4), "
            f"brake={brake_max:.6g} (limit {config.brake_joint_speed_limit:.6g}), "
            f"brake non-arm={brake_non_arm_max:.6g} "
            f"(limit {config.brake_non_arm_speed_limit:.6g}), "
            f"return={return_max:.6g} (limit 0.4)"
        )

    root_start = trajectory.root_pos[0]
    root_end = trajectory.root_pos[-1]
    start_heading = heading_rotation(rotation_from_wxyz(trajectory.root_quat_wxyz[0]))
    end_heading = heading_rotation(rotation_from_wxyz(trajectory.root_quat_wxyz[-1]))
    heading_delta = end_heading * start_heading.inv()
    return ValidationReport(
        total_frames=timesteps,
        duration_s=(timesteps - 1) * DT,
        retained_source_fraction=retained_fraction,
        retained_max_joint_speed=retained_max_joint_speed,
        retained_max_lower_body_speed=retained_max_lower_body_speed,
        retained_min_root_z=retained_min_root_z,
        entry_max_marker_displacement=entry_marker_displacement,
        return_max_marker_displacement=return_marker_displacement,
        entry_max_joint_speed=entry_max,
        brake_max_joint_speed=brake_max,
        brake_max_non_arm_speed=brake_non_arm_max,
        return_max_joint_speed=return_max,
        max_quaternion_norm_deviation=quaternion_norm_deviation,
        min_adjacent_quaternion_dot=minimum_dot,
        raw_reset_root_translation=float(np.linalg.norm(root_end - root_start)),
        raw_reset_heading_deg=float(np.degrees(heading_delta.magnitude())),
        max_joint_limit_violation=max_limit_violation,
    )


def build_safe_motion(
    source_path: str | Path,
    scene_path: str | Path,
    motion_name: str,
    config: BuildConfig,
    *,
    assume_isaaclab_order: bool,
) -> tuple[csvio.PreparedMotion, ValidationReport]:
    source = csvio.load_and_prepare(
        source_path,
        assume_isaaclab_order=assume_isaaclab_order,
    )
    kinematics = G1Kinematics(Path(scene_path))
    trajectory = build_root_and_joint_trajectory(source, kinematics, config)
    arrays = compute_release_arrays(trajectory, kinematics)
    report = validate_built_motion(source, trajectory, arrays, kinematics, config)
    quaternion_max_deviation = float(
        np.max(
            np.abs(
                np.linalg.norm(arrays["body_quat_w"].astype(np.float64), axis=-1)
                - 1.0
            )
        )
    )
    notes = (
        "SAFETY SCOPE: staged entry/exit wrapper only; source dynamics are not certified for a real robot",
        f"Source frames retained: {config.source_start_frame}..{config.source_end_frame} inclusive ({100.0 * report.retained_source_fraction:.3f}% of frames from the selected start to source end)",
        f"Segments: hold {config.initial_hold_s:g}s, entry {config.entry_s:g}s, settle {config.source_settle_s:g}s, brake {config.brake_s:g}s, stop hold {config.stop_hold_s:g}s, return {config.return_s:g}s, final hold {config.final_hold_s:g}s",
        "Entry/return joint interpolation: quintic minimum jerk; brake: C2 quartic velocity-to-zero integration",
        f"Synthetic transition speed guards: entry 0.4 rad/s, brake all joints {config.brake_joint_speed_limit:g} rad/s, brake non-arm joints {config.brake_non_arm_speed_limit:g} rad/s, return 0.4 rad/s",
        "Brake guard partition: arm means every shoulder, elbow, and wrist joint; non-arm means every waist and leg joint",
        f"Measured brake speeds: all joints {report.brake_max_joint_speed:.6g} rad/s, non-arm joints {report.brake_max_non_arm_speed:.6g} rad/s",
        f"Synthetic body FK scene: {kinematics.scene_path}",
        "End state: policy-default joints, zero velocity and upright root; end world x/y/yaw intentionally retained",
        "Reset contract: current g1_deploy_onnx_ref resets frame to 0 and reinitializes heading after natural completion",
        f"Retained source risk: max joint speed {report.retained_max_joint_speed:.6g} rad/s, max lower-body speed {report.retained_max_lower_body_speed:.6g} rad/s, min pelvis z {report.retained_min_root_z:.6g} m",
        f"Foot-marker fit: entry max displacement {report.entry_max_marker_displacement:.6g} m, return max displacement {report.return_max_marker_displacement:.6g} m",
        "Required rollout order: full MuJoCo closed-loop validation, suspended frame-0 CONTROL test, suspended staged playback; never start with unsupported playback",
    )
    motion = csvio.PreparedMotion(
        name=motion_name,
        source_path=source.source_path,
        source_sha256=source.source_sha256,
        fps=FPS,
        timesteps=report.total_frames,
        arrays=arrays,
        joint_order_validation=source.joint_order_validation,
        body_order_validation=source.body_order_validation,
        quaternion_max_norm_deviation=quaternion_max_deviation,
        derivation_notes=notes,
    )
    csvio.validate_prepared_motion(motion)
    return motion, report


def default_scene_path() -> Path:
    return (
        Path(__file__).resolve().parent.parent
        / "gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Wrap an explicitly selected G1 motion interval with guarded stationary "
            "entry and exit segments. This is not a real-robot safety certification."
        )
    )
    parser.add_argument("npz_file", help="Source 50 Hz G1 NPZ motion")
    parser.add_argument("--output-root", required=True, help="Dataset root to create")
    parser.add_argument(
        "--name",
        default="dun_real_safe_stage1",
        help="Inner motion directory name (default: dun_real_safe_stage1)",
    )
    parser.add_argument(
        "--scene", default=str(default_scene_path()), help="MuJoCo scene XML used for FK"
    )
    parser.add_argument("--source-start-frame", type=int, default=DEFAULT_SOURCE_START_FRAME)
    parser.add_argument("--source-end-frame", type=int, default=DEFAULT_SOURCE_END_FRAME)
    parser.add_argument("--initial-hold-s", type=float, default=5.0)
    parser.add_argument("--entry-s", type=float, default=5.0)
    parser.add_argument("--source-settle-s", type=float, default=1.0)
    parser.add_argument("--brake-s", type=float, default=0.5)
    parser.add_argument("--stop-hold-s", type=float, default=1.0)
    parser.add_argument("--return-s", type=float, default=6.0)
    parser.add_argument("--final-hold-s", type=float, default=5.0)
    parser.add_argument(
        "--brake-joint-speed-limit",
        type=float,
        default=0.7,
        help=(
            "maximum generated speed of any joint during the brake segment "
            "(default: 0.7 rad/s)"
        ),
    )
    parser.add_argument(
        "--brake-non-arm-speed-limit",
        type=float,
        default=0.7,
        help=(
            "maximum generated speed of every leg/waist joint during the brake "
            "segment (default: 0.7 rad/s)"
        ),
    )
    parser.add_argument(
        "--assume-isaaclab-order",
        action="store_true",
        help="accept absent names only when the source is known to use canonical G1 IsaacLab order",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = BuildConfig(
        source_start_frame=args.source_start_frame,
        source_end_frame=args.source_end_frame,
        initial_hold_s=args.initial_hold_s,
        entry_s=args.entry_s,
        source_settle_s=args.source_settle_s,
        brake_s=args.brake_s,
        stop_hold_s=args.stop_hold_s,
        return_s=args.return_s,
        final_hold_s=args.final_hold_s,
        brake_joint_speed_limit=args.brake_joint_speed_limit,
        brake_non_arm_speed_limit=args.brake_non_arm_speed_limit,
    )
    try:
        motion, report = build_safe_motion(
            args.npz_file,
            args.scene,
            args.name,
            config,
            assume_isaaclab_order=args.assume_isaaclab_order,
        )
        output_dir, serialization_errors = csvio.publish_prepared_motion(
            motion, args.output_root
        )
    except (csvio.ConversionError, SafeMotionError, OSError, ValueError) as exc:
        print(f"Safe reference generation failed: {exc}", file=sys.stderr)
        return 1

    print(f"Generated staged real-robot reference: {output_dir}")
    print(
        f"  frames={report.total_frames}, duration={report.duration_s:.2f}s, "
        f"retained source={100.0 * report.retained_source_fraction:.3f}%"
    )
    print(
        "  synthetic max |dq| [entry/brake/return]="
        f"{report.entry_max_joint_speed:.6g}/"
        f"{report.brake_max_joint_speed:.6g}/"
        f"{report.return_max_joint_speed:.6g} rad/s"
    )
    print(
        "  brake max non-arm |dq|="
        f"{report.brake_max_non_arm_speed:.6g} rad/s "
        f"(guards: all={config.brake_joint_speed_limit:.6g}, "
        f"non-arm={config.brake_non_arm_speed_limit:.6g})"
    )
    print(
        "  foot-marker max displacement [entry/return]="
        f"{report.entry_max_marker_displacement:.6g}/"
        f"{report.return_max_marker_displacement:.6g} m"
    )
    print(
        "  retained-source risk: max |dq|="
        f"{report.retained_max_joint_speed:.6g} rad/s, lower-body="
        f"{report.retained_max_lower_body_speed:.6g} rad/s, "
        f"min pelvis z={report.retained_min_root_z:.6g} m"
    )
    print(
        "  raw final-to-frame0 root difference="
        f"{report.raw_reset_root_translation:.6g} m, "
        f"{report.raw_reset_heading_deg:.6g} deg "
        "(expected; controller heading is reinitialized)"
    )
    print(
        "  quaternion max norm deviation="
        f"{report.max_quaternion_norm_deviation:.6g}, "
        f"minimum adjacent dot={report.min_adjacent_quaternion_dot:.6g}"
    )
    print("  CSV maximum serialization errors:")
    for filename, error in serialization_errors.items():
        print(f"    {filename}: {error:.9g}")
    print(
        "WARNING: this staged wrapper is not a safety certificate. Do not press T "
        "during the frame-0 CONTROL test."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
