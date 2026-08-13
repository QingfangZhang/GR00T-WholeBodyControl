"""Primary provider: recorded SONIC ``reference_motion`` intent.

This is the controller-neutral form of the earlier ``change_ckpt`` idea.  The
ten regular SONIC slots are retained verbatim.  A separate consecutive view is
built only from slot zero across policy groups for low-latency controllers.

Because SONIC's 640 active values do not contain root translation, the
Teleopit pose is explicitly hybrid: recorded executed root xyz, recovered
reference pelvis world orientation, and reference slot-zero body joints.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from change_ckpt.reference_data import (
    ReferenceDataError,
    _relative_quaternions,
    load_reference_sequence,
    quat_multiply_wxyz,
)
from change_ckpt_track.qpos_reference_data import (
    QposReferenceError,
    load_qpos_reference,
)

from .base import (
    CONSECUTIVE_SLOT_OFFSETS,
    ReferenceError,
    ReferenceMode,
    ReferenceProvenance,
    ReferenceSequence,
    clamped_future_indices,
    reorder_isaaclab_to_teleopit,
    validate_policy_slice,
)


DEFAULT_BASE_SAMPLE_MODE = "previous-index5"
ALIGNMENT_TIME_TOLERANCE_S = 1e-9
ALIGNMENT_HAND_TOLERANCE = 1e-6


def _indices_for_policy_seq(
    requested: np.ndarray, available: np.ndarray
) -> np.ndarray:
    requested = np.asarray(requested, dtype=np.int64)
    available = np.asarray(available, dtype=np.int64)
    if requested.ndim != 1 or available.ndim != 1:
        raise ReferenceError("policy_seq arrays must be one-dimensional")
    if available.size != np.unique(available).size:
        raise ReferenceError("SONIC reference contains duplicate policy_seq values")
    indices = np.searchsorted(available, requested)
    valid = indices < available.size
    if np.any(valid):
        valid[valid] &= available[indices[valid]] == requested[valid]
    if not np.all(valid):
        missing = requested[~valid][:10].tolist()
        raise ReferenceError(
            "executed-qpos and reference_motion policy groups do not align; "
            f"missing from reference_motion: {missing}"
        )
    return indices.astype(np.int64, copy=False)


def _world_anchor_quaternions(
    base_quat_wxyz: np.ndarray, relative_anchor_6d: np.ndarray
) -> np.ndarray:
    """Recover every recorded reference anchor in world coordinates."""

    relative_quat, _ = _relative_quaternions(relative_anchor_6d)
    result = quat_multiply_wxyz(
        np.asarray(base_quat_wxyz, dtype=np.float64)[:, None, :],
        np.asarray(relative_quat, dtype=np.float64),
    )
    # Quaternion sign is arbitrary.  Make each slot continuous over time so
    # finite-difference consumers do not see artificial sign flips.
    result = np.asarray(result, dtype=np.float64)
    for slot in range(result.shape[1]):
        for frame in range(1, result.shape[0]):
            if float(np.dot(result[frame - 1, slot], result[frame, slot])) < 0.0:
                result[frame, slot] *= -1.0
    return result.astype(np.float32)


@dataclass(frozen=True)
class ReferenceMotionProvider:
    """Load the original exogenous SONIC reference as the main experiment."""

    mode = ReferenceMode.REFERENCE_MOTION

    base_sample_mode: str = DEFAULT_BASE_SAMPLE_MODE
    drop_truncated_edges: bool = True

    def load(
        self,
        recording: str | Path,
        *,
        policy_offset: int = 0,
        policy_count: int | None = None,
    ) -> ReferenceSequence:
        validate_policy_slice(policy_offset, policy_count)
        if not self.base_sample_mode:
            raise ReferenceError("base_sample_mode must not be empty")
        try:
            qpos_full = load_qpos_reference(
                recording,
                drop_truncated_edges=self.drop_truncated_edges,
            )
            sonic = load_reference_sequence(
                recording,
                base_sample_mode=self.base_sample_mode,
            )
        except (QposReferenceError, ReferenceDataError) as exc:
            raise ReferenceError(str(exc)) from exc

        if qpos_full.csv_path.resolve() != sonic.csv_path.resolve():
            raise ReferenceError(
                "qpos and SONIC parsers resolved different source CSV files"
            )
        if policy_offset >= qpos_full.num_frames:
            raise ReferenceError(
                f"policy_offset {policy_offset} is outside the validated "
                f"reference with {qpos_full.num_frames} frames"
            )

        selected = qpos_full.slice(policy_offset, policy_count)
        all_sonic_indices = _indices_for_policy_seq(
            qpos_full.policy_seq, sonic.policy_seq
        )
        selected_full_indices = np.arange(
            selected.frame_offset,
            selected.frame_offset + selected.num_frames,
            dtype=np.int64,
        )
        selected_sonic_indices = all_sonic_indices[selected_full_indices]

        aligned_time = np.asarray(
            sonic.control_time_s[all_sonic_indices], dtype=np.float64
        )
        time_error = float(
            np.max(
                np.abs(aligned_time - np.asarray(qpos_full.control_time_s, dtype=np.float64))
            )
        )
        if time_error > ALIGNMENT_TIME_TOLERANCE_S:
            raise ReferenceError(
                "qpos/reference_motion policy groups disagree in control_time_s: "
                f"max error {time_error:.9g}s"
            )
        aligned_left = np.asarray(
            sonic.left_hand_target[all_sonic_indices], dtype=np.float32
        )
        aligned_right = np.asarray(
            sonic.right_hand_target[all_sonic_indices], dtype=np.float32
        )
        hand_error = float(
            max(
                np.max(np.abs(aligned_left - qpos_full.left_hand_target)),
                np.max(np.abs(aligned_right - qpos_full.right_hand_target)),
            )
        )
        if hand_error > ALIGNMENT_HAND_TOLERANCE:
            raise ReferenceError(
                "qpos/reference_motion policy groups disagree in hand targets: "
                f"max error {hand_error:.9g}"
            )

        source_reference = np.asarray(
            sonic.reference_motion[selected_sonic_indices], dtype=np.float32
        )
        active = source_reference[:, :640]
        regular_joint_pos = active[:, :290].reshape(-1, 10, 29)
        regular_joint_vel = active[:, 290:580].reshape(-1, 10, 29)
        regular_relative_6d = active[:, 580:640].reshape(-1, 10, 6)
        all_regular_world_quat = _world_anchor_quaternions(
            sonic.policy_base_quat_wxyz[all_sonic_indices],
            sonic.relative_anchor_6d[all_sonic_indices],
        )
        regular_world_quat = all_regular_world_quat[selected_full_indices]

        consecutive_indices = clamped_future_indices(
            start_indices=selected_full_indices,
            offsets=CONSECUTIVE_SLOT_OFFSETS,
            source_frame_count=qpos_full.num_frames,
        )
        all_current_joint_pos = np.asarray(
            sonic.joint_pos[all_sonic_indices], dtype=np.float32
        )
        all_current_joint_vel = np.asarray(
            sonic.joint_vel[all_sonic_indices], dtype=np.float32
        )
        all_current_world_quat = all_regular_world_quat[:, 0]
        consecutive_joint_pos = all_current_joint_pos[consecutive_indices]
        consecutive_joint_vel = all_current_joint_vel[consecutive_indices]
        consecutive_world_quat = all_current_world_quat[consecutive_indices]

        teleopit_joint_pos = reorder_isaaclab_to_teleopit(
            consecutive_joint_pos[:, 0]
        ).astype(np.float32)
        teleopit_joint_vel = reorder_isaaclab_to_teleopit(
            consecutive_joint_vel[:, 0]
        ).astype(np.float32)
        teleopit_qpos36 = np.concatenate(
            (
                np.asarray(selected.root_pos, dtype=np.float32),
                np.asarray(consecutive_world_quat[:, 0], dtype=np.float32),
                teleopit_joint_pos,
            ),
            axis=1,
        )

        provenance = ReferenceProvenance(
            mode=self.mode,
            reference_kind=(
                "recorded_sonic_reference_motion_with_recorded_root_translation"
            ),
            regular_slot_source="recorded reference_motion ten slots, verbatim",
            regular_slot_offsets=None,
            orientation_base_sample_mode=self.base_sample_mode,
            component_sources=(
                (
                    "sonic_regular_reference",
                    "data.csv reference_motion[0:640] recorded ten-slot q/dq/anchor",
                ),
                (
                    "sonic_consecutive_reference",
                    "slot zero from consecutive policy_seq groups; final frame clamped",
                ),
                (
                    "reference_anchor_world_orientation",
                    "recorded pelvis base quaternion multiplied by reference_motion relative anchor",
                ),
                (
                    "teleopit_root_translation",
                    "recorded actual qpos root xyz because SONIC reference_motion has no root xyz",
                ),
                (
                    "teleopit_root_orientation",
                    "recovered slot-zero reference pelvis world orientation",
                ),
                (
                    "teleopit_body_joint_position",
                    "reference_motion slot-zero joints reordered by name",
                ),
                ("hand_targets", "recorded left_hand_q/right_hand_q policy command"),
                ("source_root", "recorded actual qpos root xyz and quat_wxyz"),
            ),
            source_frame_count=qpos_full.num_frames,
            selected_policy_offset=int(selected.frame_offset),
            drop_truncated_edges=self.drop_truncated_edges,
            policy_alignment_max_time_error_s=time_error,
            policy_alignment_max_hand_error=hand_error,
        )
        return ReferenceSequence(
            source_csv_path=selected.csv_path,
            provenance=provenance,
            policy_seq=selected.policy_seq,
            control_time_s=selected.control_time_s,
            source_row_index=selected.source_row_indices,
            source_csv_row_number=selected.source_csv_row_numbers,
            group_row_counts=selected.group_row_counts,
            source_root_pos=selected.root_pos,
            source_root_quat_wxyz=selected.root_quat_wxyz,
            left_hand_target=selected.left_hand_target,
            right_hand_target=selected.right_hand_target,
            sonic_regular_joint_pos=regular_joint_pos,
            sonic_regular_joint_vel=regular_joint_vel,
            sonic_regular_anchor_quat_wxyz=regular_world_quat,
            sonic_consecutive_joint_pos=consecutive_joint_pos,
            sonic_consecutive_joint_vel=consecutive_joint_vel,
            sonic_consecutive_anchor_quat_wxyz=consecutive_world_quat,
            teleopit_qpos36=teleopit_qpos36,
            teleopit_reference_joint_vel=teleopit_joint_vel,
            source_reference_motion=source_reference,
            source_recorded_relative_anchor_6d=regular_relative_6d,
        )


def load_reference_motion(
    recording: str | Path,
    *,
    policy_offset: int = 0,
    policy_count: int | None = None,
    drop_truncated_edges: bool = True,
    base_sample_mode: str = DEFAULT_BASE_SAMPLE_MODE,
) -> ReferenceSequence:
    """Functional convenience wrapper for :class:`ReferenceMotionProvider`."""

    return ReferenceMotionProvider(
        base_sample_mode=base_sample_mode,
        drop_truncated_edges=drop_truncated_edges,
    ).load(
        recording,
        policy_offset=policy_offset,
        policy_count=policy_count,
    )


__all__ = [
    "ALIGNMENT_HAND_TOLERANCE",
    "ALIGNMENT_TIME_TOLERANCE_S",
    "DEFAULT_BASE_SAMPLE_MODE",
    "ReferenceMotionProvider",
    "load_reference_motion",
]
