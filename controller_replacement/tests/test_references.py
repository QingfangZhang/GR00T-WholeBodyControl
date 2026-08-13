"""Regression tests for the unified controller-reference providers."""

from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from change_ckpt.reference_data import load_reference_sequence
from change_ckpt_track.qpos_reference_data import (
    G1_ISAACLAB_JOINT_NAMES,
    G1_MUJOCO_JOINT_NAMES,
    load_qpos_reference,
)
from Teleopit_rollout.reference_data import load_prepared_reference
from Teleopit_rollout.reference_motion_data import (
    load_reference_motion_prepared_reference,
)

from controller_replacement.references import (
    REGULAR_QPOS_SLOT_OFFSETS,
    ReferenceError,
    ReferenceMode,
    ReferenceProvider,
    load_reference,
    make_reference_provider,
)


class SyntheticRecording:
    """Small CSV accepted by both existing validated source parsers."""

    def __init__(self, root: Path, frame_count: int = 5) -> None:
        self.root = root
        self.csv_path = root / "data.csv"
        root.mkdir(parents=True, exist_ok=True)
        self._write(frame_count)

    def _write(self, frame_count: int) -> None:
        root_qpos = [
            "qpos:pelvis.floating_base_joint.x[qpos0]",
            "qpos:pelvis.floating_base_joint.y[qpos1]",
            "qpos:pelvis.floating_base_joint.z[qpos2]",
            "qpos:pelvis.floating_base_joint.qw[qpos3]",
            "qpos:pelvis.floating_base_joint.qx[qpos4]",
            "qpos:pelvis.floating_base_joint.qy[qpos5]",
            "qpos:pelvis.floating_base_joint.qz[qpos6]",
        ]
        body_qpos = [
            f"qpos:{name.removesuffix('_joint')}_link.{name}.angle[qpos{7 + index}]"
            for index, name in enumerate(G1_MUJOCO_JOINT_NAMES)
        ]
        root_qvel = [
            f"qvel:pelvis.floating_base_joint.{name}[qvel{index}]"
            for index, name in enumerate(("vx", "vy", "vz", "wx", "wy", "wz"))
        ]
        body_qvel = [
            f"qvel:{name.removesuffix('_joint')}_link.{name}.omega[qvel{6 + index}]"
            for index, name in enumerate(G1_MUJOCO_JOINT_NAMES)
        ]
        reference_columns = [f"reference_motion[{index}]" for index in range(1024)]
        hand_columns = [
            *(f"left_hand_q[{index}]" for index in range(7)),
            *(f"right_hand_q[{index}]" for index in range(7)),
        ]
        header = [
            "mujoco_time_s",
            "control_time_s",
            *root_qpos,
            *body_qpos,
            *root_qvel,
            *body_qvel,
            "policy_valid",
            "policy_seq",
            "policy_reference_motion_size",
            *reference_columns,
            *hand_columns,
        ]
        isaac_index = {
            name: index for index, name in enumerate(G1_ISAACLAB_JOINT_NAMES)
        }
        rows: list[list[object]] = []
        tick = 0
        for frame in range(frame_count):
            yaw = 0.04 * frame
            quaternion = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
            reference = np.zeros(1024, dtype=np.float64)
            for slot in range(10):
                for joint in range(29):
                    reference[slot * 29 + joint] = (
                        frame * 1000.0 + slot * 100.0 + joint
                    )
                    reference[290 + slot * 29 + joint] = (
                        frame * 100.0 + slot * 10.0 + joint
                    )
                reference[580 + slot * 6 : 580 + (slot + 1) * 6] = (
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                )
            for subrow in range(8):
                values: dict[str, object] = {
                    "mujoco_time_s": tick * 0.0025,
                    "control_time_s": tick * 0.0025,
                    "policy_valid": 1,
                    "policy_seq": 700 + frame,
                    "policy_reference_motion_size": 640,
                }
                values.update(
                    zip(
                        root_qpos,
                        [frame, frame + 0.25, 0.8 + frame * 0.01, *quaternion],
                        strict=True,
                    )
                )
                values.update({name: 0.0 for name in root_qvel})
                for source_index, joint_name in enumerate(G1_MUJOCO_JOINT_NAMES):
                    output_index = isaac_index[joint_name]
                    values[body_qpos[source_index]] = (
                        frame * 10000.0 + output_index + subrow * 0.001
                    )
                    values[body_qvel[source_index]] = frame * 10.0 + output_index
                for index, name in enumerate(reference_columns):
                    values[name] = reference[index]
                for index in range(7):
                    values[f"left_hand_q[{index}]"] = frame + index * 0.01
                    values[f"right_hand_q[{index}]"] = -frame - index * 0.01
                rows.append([values[name] for name in header])
                tick += 1
        with self.csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)


class ReferenceProviderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.recording = SyntheticRecording(Path(self.temp.name)).root

    def test_reference_motion_preserves_recorded_slots_and_builds_hybrid(self) -> None:
        reference = load_reference(
            self.recording,
            mode=ReferenceMode.REFERENCE_MOTION,
            policy_offset=1,
            policy_count=2,
        )
        old_sonic = load_reference_sequence(self.recording)
        old_hybrid = load_reference_motion_prepared_reference(
            self.recording,
            policy_offset=1,
            policy_count=2,
        )
        indices = np.searchsorted(old_sonic.policy_seq, reference.policy_seq)

        self.assertEqual(reference.mode, ReferenceMode.REFERENCE_MOTION)
        self.assertEqual(reference.num_frames, 2)
        np.testing.assert_array_equal(reference.policy_seq, [701, 702])
        np.testing.assert_array_equal(reference.source_row_index, [8, 16])
        np.testing.assert_array_equal(reference.source_csv_row_number, [10, 18])
        np.testing.assert_array_equal(
            reference.sonic_regular_joint_pos,
            old_sonic.reference_motion[indices, :290].reshape(2, 10, 29),
        )
        np.testing.assert_array_equal(
            reference.sonic_regular_joint_vel,
            old_sonic.reference_motion[indices, 290:580].reshape(2, 10, 29),
        )
        # Consecutive slots come from slot zero of successive policy groups,
        # not from the remaining nine recorded regular slots.
        np.testing.assert_array_equal(
            reference.sonic_consecutive_joint_pos[0, :4, 0],
            [1000.0, 2000.0, 3000.0, 4000.0],
        )
        np.testing.assert_array_equal(
            reference.sonic_consecutive_joint_pos[0, 4:, 0],
            np.full(6, 4000.0),
        )
        np.testing.assert_array_equal(reference.teleopit_qpos36, old_hybrid.qpos36)
        np.testing.assert_array_equal(
            reference.teleopit_reference_joint_vel,
            old_hybrid.source_reference_joint_vel,
        )
        np.testing.assert_array_equal(reference.left_hand_target, old_hybrid.left_hand_target)
        self.assertIsNotNone(reference.source_reference_motion)
        self.assertIsNotNone(reference.source_recorded_relative_anchor_6d)
        np.testing.assert_array_equal(
            reference.sonic_current_joint_pos,
            reference.sonic_consecutive_joint_pos[:, 0],
        )
        np.testing.assert_array_equal(
            reference.teleopit_reference_joint_pos,
            reference.teleopit_qpos36[:, 7:],
        )
        self.assertEqual(reference[0].policy_seq, 701)
        self.assertFalse(reference.policy_seq.flags.writeable)
        self.assertFalse(reference.frame(0).sonic_regular_joint_pos.flags.writeable)

        metadata = reference.metadata()
        json.dumps(metadata)
        self.assertEqual(metadata["reference_mode"], "reference_motion")
        self.assertIsNone(metadata["regular_slot_offsets"])
        self.assertEqual(metadata["orientation_base_sample_mode"], "previous-index5")
        self.assertEqual(metadata["policy_alignment_max_time_error_s"], 0.0)
        self.assertEqual(metadata["policy_alignment_max_hand_error"], 0.0)
        self.assertTrue(metadata["source_reference_motion_present"])

    def test_executed_qpos_matches_old_qpos_adapter_and_clamps_windows(self) -> None:
        reference = load_reference(
            self.recording,
            mode="executed_qpos",
            policy_offset=1,
            policy_count=2,
        )
        qpos = load_qpos_reference(self.recording)
        old_teleopit = load_prepared_reference(
            self.recording,
            policy_offset=1,
            policy_count=2,
        )

        self.assertEqual(reference.mode, ReferenceMode.EXECUTED_QPOS)
        np.testing.assert_array_equal(reference.teleopit_qpos36, old_teleopit.qpos36)
        np.testing.assert_array_equal(
            reference.teleopit_reference_joint_vel,
            old_teleopit.recorded_joint_vel,
        )
        # At selected source frame 1, regular offset 0 is frame 1 and every
        # offset >=5 is clamped to the final source frame (frame 4).
        np.testing.assert_array_equal(
            reference.sonic_regular_joint_pos[0, 0], qpos.joint_pos[1]
        )
        np.testing.assert_array_equal(
            reference.sonic_regular_joint_pos[0, 1:],
            np.repeat(qpos.joint_pos[4][None, :], 9, axis=0),
        )
        self.assertIsNone(reference.source_reference_motion)
        self.assertIsNone(reference.source_recorded_relative_anchor_6d)
        metadata = reference.metadata()
        self.assertEqual(metadata["reference_mode"], "executed_qpos")
        self.assertEqual(
            metadata["regular_slot_offsets"], list(REGULAR_QPOS_SLOT_OFFSETS)
        )
        self.assertFalse(metadata["source_reference_motion_present"])

    def test_public_interface_and_strict_validation(self) -> None:
        provider = make_reference_provider("reference_motion")
        self.assertIsInstance(provider, ReferenceProvider)
        with self.assertRaisesRegex(ReferenceError, "unsupported reference mode"):
            make_reference_provider("not-a-mode")
        with self.assertRaisesRegex(ReferenceError, "policy_offset"):
            provider.load(self.recording, policy_offset=-1)
        with self.assertRaisesRegex(ReferenceError, "policy_count"):
            provider.load(self.recording, policy_count=0)
        with self.assertRaisesRegex(ReferenceError, "does not exist"):
            provider.load(self.recording / "missing")

        reference = provider.load(self.recording, policy_count=1)
        with self.assertRaisesRegex(ReferenceError, "frame index"):
            reference.frame(-1)
        with self.assertRaisesRegex(ReferenceError, "frame index"):
            reference.frame(reference.num_frames)
        bad_root = np.array(reference.source_root_pos, copy=True)
        bad_root[0, 0] = np.nan
        with self.assertRaisesRegex(ReferenceError, "NaN or infinity"):
            replace(reference, source_root_pos=bad_root)
        with self.assertRaises(ValueError):
            reference.policy_seq[0] = 1

    def test_slice_preserves_absolute_provenance_without_reparse(self) -> None:
        sequence = load_reference(self.recording, mode="reference_motion")
        selected = sequence.slice(2, 2)
        self.assertEqual(selected.num_frames, 2)
        np.testing.assert_array_equal(selected.policy_seq, sequence.policy_seq[2:4])
        self.assertEqual(
            selected.provenance.selected_policy_offset,
            sequence.provenance.selected_policy_offset + 2,
        )


if __name__ == "__main__":
    unittest.main()
