"""Adapt recorded SONIC reference motion to Teleopit's current-pose interface.

The 640 active SONIC values do not contain reference root translation and
encode reference pelvis orientation relative to the recorded robot.  This
adapter therefore constructs an explicitly hybrid 36-D pose at 50 Hz:

``recorded actual root xyz + (recorded actual pelvis quaternion * relative
reference pelvis quaternion) + reference_motion slot-0 body joints``.

Teleopit's runtime still derives joint/torso velocities from adjacent hybrid
poses and builds its ten-observation *past* history online.  SONIC's future
slots are never reinterpreted as Teleopit history.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from .constants import G1_JOINT_NAMES
    from .reference_data import (
        PreparedReference,
        PreparedReferenceError,
        _joint_reorder_indices,
    )
except ImportError:  # pragma: no cover - direct script import
    from constants import G1_JOINT_NAMES
    from reference_data import (
        PreparedReference,
        PreparedReferenceError,
        _joint_reorder_indices,
    )

from change_ckpt.reference_data import (
    ReferenceDataError,
    load_reference_sequence,
)
from change_ckpt_track.qpos_reference_data import (
    QposReferenceError,
    load_qpos_reference,
)


REFERENCE_SOURCE = "sonic_reference_hybrid"
DEFAULT_BASE_SAMPLE_MODE = "previous-index5"
ALIGNMENT_TIME_TOLERANCE_S = 1e-9
ALIGNMENT_VALUE_TOLERANCE = 1e-6


def _sonic_indices_for_policy_seq(
    qpos_policy_seq: np.ndarray,
    sonic_policy_seq: np.ndarray,
) -> np.ndarray:
    sonic_values = np.asarray(sonic_policy_seq, dtype=np.int64)
    if sonic_values.size != np.unique(sonic_values).size:
        raise PreparedReferenceError("SONIC reference contains duplicate policy_seq")
    index_by_policy = {
        int(policy_seq): index for index, policy_seq in enumerate(sonic_values)
    }
    missing = [
        int(policy_seq)
        for policy_seq in np.asarray(qpos_policy_seq, dtype=np.int64)
        if int(policy_seq) not in index_by_policy
    ]
    if missing:
        raise PreparedReferenceError(
            "qpos/SONIC reference policy_seq sets do not align; missing from "
            f"reference_motion: {missing[:10]}"
        )
    return np.asarray(
        [index_by_policy[int(policy_seq)] for policy_seq in qpos_policy_seq],
        dtype=np.int64,
    )


def load_reference_motion_prepared_reference(
    recording: str | Path,
    *,
    policy_offset: int = 0,
    policy_count: int | None = None,
    drop_truncated_edges: bool = True,
    base_sample_mode: str = DEFAULT_BASE_SAMPLE_MODE,
) -> PreparedReference:
    """Return a policy-aligned hybrid SONIC-reference pose for Teleopit."""

    if policy_offset < 0:
        raise PreparedReferenceError("policy_offset must be non-negative")
    if policy_count is not None and policy_count <= 0:
        raise PreparedReferenceError("policy_count must be positive")
    try:
        qpos_full = load_qpos_reference(
            recording,
            drop_truncated_edges=drop_truncated_edges,
        )
        sonic = load_reference_sequence(
            recording,
            base_sample_mode=base_sample_mode,
        )
    except (QposReferenceError, ReferenceDataError) as exc:
        raise PreparedReferenceError(str(exc)) from exc

    if qpos_full.csv_path.resolve() != sonic.csv_path.resolve():
        raise PreparedReferenceError(
            "qpos and SONIC reference parsers resolved different source CSV files"
        )
    if policy_offset >= qpos_full.num_frames:
        raise PreparedReferenceError(
            f"policy_offset {policy_offset} is outside the validated reference "
            f"with {qpos_full.num_frames} frames"
        )

    selected = qpos_full.slice(policy_offset, policy_count)
    sonic_indices = _sonic_indices_for_policy_seq(
        selected.policy_seq,
        sonic.policy_seq,
    )
    sonic_time = np.asarray(sonic.control_time_s[sonic_indices], dtype=np.float64)
    time_error = float(
        np.max(np.abs(sonic_time - np.asarray(selected.control_time_s)))
    )
    if time_error > ALIGNMENT_TIME_TOLERANCE_S:
        raise PreparedReferenceError(
            "qpos/SONIC policy groups disagree in control_time_s: max error "
            f"{time_error:.9g}s"
        )

    sonic_left = np.asarray(sonic.left_hand_target[sonic_indices], dtype=np.float32)
    sonic_right = np.asarray(sonic.right_hand_target[sonic_indices], dtype=np.float32)
    hand_error = float(
        max(
            np.max(np.abs(sonic_left - selected.left_hand_target)),
            np.max(np.abs(sonic_right - selected.right_hand_target)),
        )
    )
    if hand_error > ALIGNMENT_VALUE_TOLERANCE:
        raise PreparedReferenceError(
            "qpos/SONIC policy groups disagree in hand targets: max error "
            f"{hand_error:.9g}"
        )

    reorder = _joint_reorder_indices(selected.joint_names)
    reference_joint_pos = np.asarray(
        sonic.joint_pos[sonic_indices][:, reorder], dtype=np.float32
    )
    source_reference_joint_vel = np.asarray(
        sonic.joint_vel[sonic_indices][:, reorder], dtype=np.float32
    )
    reference_root_quat = np.asarray(
        sonic.reference_anchor_quat_wxyz[sonic_indices], dtype=np.float32
    )
    qpos36 = np.concatenate(
        (
            np.asarray(selected.root_pos, dtype=np.float32),
            reference_root_quat,
            reference_joint_pos,
        ),
        axis=1,
    ).astype(np.float32, copy=False)

    return PreparedReference(
        source_csv_path=selected.csv_path,
        joint_names=G1_JOINT_NAMES,
        qpos36=qpos36,
        recorded_joint_vel=np.asarray(
            selected.joint_vel[:, reorder], dtype=np.float32
        ),
        left_hand_target=np.asarray(selected.left_hand_target, dtype=np.float32),
        right_hand_target=np.asarray(selected.right_hand_target, dtype=np.float32),
        policy_seq=np.asarray(selected.policy_seq, dtype=np.int64),
        control_time_s=np.asarray(selected.control_time_s, dtype=np.float64),
        source_row_index=np.asarray(selected.source_row_indices, dtype=np.int64),
        source_csv_row_number=np.asarray(
            selected.source_csv_row_numbers, dtype=np.int64
        ),
        group_row_counts=np.asarray(selected.group_row_counts, dtype=np.int32),
        policy_offset=int(selected.frame_offset),
        source_frame_count=qpos_full.num_frames,
        reference_source=REFERENCE_SOURCE,
        source_reference_joint_vel=source_reference_joint_vel,
        orientation_base_sample_mode=base_sample_mode,
        policy_alignment_max_time_error_s=time_error,
        policy_alignment_max_hand_error=hand_error,
    )


__all__ = [
    "DEFAULT_BASE_SAMPLE_MODE",
    "REFERENCE_SOURCE",
    "load_reference_motion_prepared_reference",
]
