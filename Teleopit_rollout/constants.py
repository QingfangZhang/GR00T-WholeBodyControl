"""Teleopit v0.5.0 G1 runtime constants used by the rollout adapter.

The numerical controller parameters are copied from
``teleopit/configs/robot/g1.yaml`` at Teleopit commit
``f9263865c581802ad531854b8e547e2403a945f3`` (tag ``v0.5.0``).
Teleopit is Copyright 2026 BotRunner64 and licensed under Apache-2.0; see
``THIRD_PARTY_NOTICE.md`` in this directory.
"""

from __future__ import annotations

import numpy as np


TELEOPIT_VERSION = "0.5.0"
TELEOPIT_COMMIT = "f9263865c581802ad531854b8e547e2403a945f3"

POLICY_HZ = 50.0
PD_HZ = 200.0
LOG_HZ = 400.0
OBSERVATION_DIM = 167
HISTORY_LENGTH = 10
ACTION_DIM = 29

G1_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

LEFT_HAND_JOINT_NAMES: tuple[str, ...] = (
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
)

RIGHT_HAND_JOINT_NAMES: tuple[str, ...] = tuple(
    name.replace("left_", "right_", 1) for name in LEFT_HAND_JOINT_NAMES
)

DEFAULT_DOF_POS = np.asarray(
    [
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        0.0, 0.0, 0.0,
        0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
        0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    ],
    dtype=np.float32,
)

ACTION_SCALE = np.asarray(
    [
        0.5475, 0.3507, 0.5475, 0.3507, 0.4386, 0.4386,
        0.5475, 0.3507, 0.5475, 0.3507, 0.4386, 0.4386,
        0.5475, 0.4386, 0.4386,
        0.4386, 0.4386, 0.4386, 0.4386, 0.4386, 0.0745, 0.0745,
        0.4386, 0.4386, 0.4386, 0.4386, 0.4386, 0.0745, 0.0745,
    ],
    dtype=np.float32,
)

KPS = np.asarray(
    [
        40.2, 99.1, 40.2, 99.1, 28.5, 28.5,
        40.2, 99.1, 40.2, 99.1, 28.5, 28.5,
        40.2, 28.5, 28.5,
        14.3, 14.3, 14.3, 14.3, 14.3, 16.8, 16.8,
        14.3, 14.3, 14.3, 14.3, 14.3, 16.8, 16.8,
    ],
    dtype=np.float64,
)

KDS = np.asarray(
    [
        2.6, 6.3, 2.6, 6.3, 1.8, 1.8,
        2.6, 6.3, 2.6, 6.3, 1.8, 1.8,
        2.6, 1.8, 1.8,
        0.9, 0.9, 0.9, 0.9, 0.9, 1.1, 1.1,
        0.9, 0.9, 0.9, 0.9, 0.9, 1.1, 1.1,
    ],
    dtype=np.float64,
)

TORQUE_LIMITS = np.asarray(
    [
        88, 139, 88, 139, 50, 50,
        88, 139, 88, 139, 50, 50,
        88, 50, 50,
        25, 25, 25, 25, 25, 5, 5,
        25, 25, 25, 25, 25, 5, 5,
    ],
    dtype=np.float64,
)

# The existing SONIC Dex3 command path uses these gains while hand targets are
# held between 50 Hz source updates.
HAND_KP = 1.5
HAND_KD = 0.1
HAND_MAX_TARGET_DELTA = 0.25

DEX3_MIN_LEFT = np.asarray(
    [-1.05, -0.724, 0.0, -1.57, -1.75, -1.57, -1.75], dtype=np.float64
)
DEX3_MAX_LEFT = np.asarray(
    [1.05, 1.05, 1.75, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
)
DEX3_MIN_RIGHT = np.asarray(
    [-1.05, -1.05, -1.75, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
)
DEX3_MAX_RIGHT = np.asarray(
    [1.05, 0.742, 0.0, 1.57, 1.75, 1.57, 1.75], dtype=np.float64
)

ROOT_ASSIST_WIDTH = {"none": 0, "xy": 2, "xyz": 3}
