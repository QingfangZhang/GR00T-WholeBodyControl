"""Auxiliary provider: the original controller's executed qpos trajectory.

This is the controller-neutral form of the earlier ``change_ckpt_track``
diagnostic.  It answers whether a replacement controller can track a robot
trajectory that the source controller physically executed; it must not be
reported as the same reference condition as :mod:`reference_motion`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from change_ckpt_track.qpos_reference_data import (
    QposReferenceError,
    load_qpos_reference,
)

from .base import (
    CONSECUTIVE_SLOT_OFFSETS,
    REGULAR_QPOS_SLOT_OFFSETS,
    ReferenceError,
    ReferenceMode,
    ReferenceProvenance,
    ReferenceSequence,
    clamped_future_indices,
    reorder_isaaclab_to_teleopit,
    validate_policy_slice,
)


@dataclass(frozen=True)
class ExecutedQposProvider:
    """Load actual recorded robot state as a secondary tracking reference."""

    mode = ReferenceMode.EXECUTED_QPOS

    drop_truncated_edges: bool = True

    def load(
        self,
        recording: str | Path,
        *,
        policy_offset: int = 0,
        policy_count: int | None = None,
    ) -> ReferenceSequence:
        validate_policy_slice(policy_offset, policy_count)
        try:
            full = load_qpos_reference(
                recording,
                drop_truncated_edges=self.drop_truncated_edges,
            )
        except QposReferenceError as exc:
            raise ReferenceError(str(exc)) from exc
        if policy_offset >= full.num_frames:
            raise ReferenceError(
                f"policy_offset {policy_offset} is outside the validated "
                f"reference with {full.num_frames} frames"
            )
        selected = full.slice(policy_offset, policy_count)
        selected_full_indices = np.arange(
            selected.frame_offset,
            selected.frame_offset + selected.num_frames,
            dtype=np.int64,
        )
        regular_indices = clamped_future_indices(
            start_indices=selected_full_indices,
            offsets=REGULAR_QPOS_SLOT_OFFSETS,
            source_frame_count=full.num_frames,
        )
        consecutive_indices = clamped_future_indices(
            start_indices=selected_full_indices,
            offsets=CONSECUTIVE_SLOT_OFFSETS,
            source_frame_count=full.num_frames,
        )

        regular_joint_pos = np.asarray(full.joint_pos[regular_indices], dtype=np.float32)
        regular_joint_vel = np.asarray(full.joint_vel[regular_indices], dtype=np.float32)
        regular_world_quat = np.asarray(
            full.root_quat_wxyz[regular_indices], dtype=np.float32
        )
        consecutive_joint_pos = np.asarray(
            full.joint_pos[consecutive_indices], dtype=np.float32
        )
        consecutive_joint_vel = np.asarray(
            full.joint_vel[consecutive_indices], dtype=np.float32
        )
        consecutive_world_quat = np.asarray(
            full.root_quat_wxyz[consecutive_indices], dtype=np.float32
        )

        teleopit_joint_pos = reorder_isaaclab_to_teleopit(
            consecutive_joint_pos[:, 0]
        ).astype(np.float32)
        teleopit_joint_vel = reorder_isaaclab_to_teleopit(
            consecutive_joint_vel[:, 0]
        ).astype(np.float32)
        teleopit_qpos36 = np.concatenate(
            (
                np.asarray(selected.root_pos, dtype=np.float32),
                np.asarray(selected.root_quat_wxyz, dtype=np.float32),
                teleopit_joint_pos,
            ),
            axis=1,
        )

        provenance = ReferenceProvenance(
            mode=self.mode,
            reference_kind="recorded_executed_robot_qpos_track",
            regular_slot_source="executed qpos sampled at declared future offsets",
            regular_slot_offsets=REGULAR_QPOS_SLOT_OFFSETS,
            orientation_base_sample_mode=None,
            component_sources=(
                (
                    "sonic_regular_reference",
                    "recorded actual body qpos/qvel/root orientation sampled at 0,5,...,45 policy offsets",
                ),
                (
                    "sonic_consecutive_reference",
                    "recorded actual body qpos/qvel/root orientation at ten consecutive policy frames",
                ),
                (
                    "teleopit_qpos36",
                    "recorded actual root pose and body qpos, joints reordered by name",
                ),
                ("hand_targets", "recorded left_hand_q/right_hand_q policy command"),
                ("source_root", "recorded actual qpos root xyz and quat_wxyz"),
            ),
            source_frame_count=full.num_frames,
            selected_policy_offset=int(selected.frame_offset),
            drop_truncated_edges=self.drop_truncated_edges,
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
        )


def load_executed_qpos(
    recording: str | Path,
    *,
    policy_offset: int = 0,
    policy_count: int | None = None,
    drop_truncated_edges: bool = True,
) -> ReferenceSequence:
    """Functional convenience wrapper for :class:`ExecutedQposProvider`."""

    return ExecutedQposProvider(
        drop_truncated_edges=drop_truncated_edges
    ).load(
        recording,
        policy_offset=policy_offset,
        policy_count=policy_count,
    )


__all__ = ["ExecutedQposProvider", "load_executed_qpos"]
