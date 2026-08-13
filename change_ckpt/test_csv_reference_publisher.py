"""Focused tests for the copied CSV reference publisher."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from change_ckpt.csv_reference_publisher import (
    HEADER_SIZE,
    PROTOCOL_VERSION,
    PUBLISHES_EXTERNAL_TOKEN,
    build_pose_payload,
    build_parser,
)
from change_ckpt.reference_data import (
    ReferenceDataError,
    quat_multiply_wxyz,
    quat_to_matrix_wxyz,
    load_reference_sequence,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message


def _six_d(quat: np.ndarray) -> np.ndarray:
    return quat_to_matrix_wxyz(quat)[:, :2].reshape(6)


class CsvReferencePublisherTest(unittest.TestCase):
    def test_standalone_parser_has_a_supported_default_layout(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual(args.checkpoint_layout, "low_latency")

    def _write_csv(self, path: Path) -> None:
        qpos = [f"qpos:test[qpos{index}]" for index in range(7)]
        reference = [f"reference_motion[{index}]" for index in range(1024)]
        hands = [f"left_hand_q[{index}]" for index in range(7)] + [
            f"right_hand_q[{index}]" for index in range(7)
        ]
        header = [
            "control_time_s",
            *qpos,
            "policy_valid",
            "policy_seq",
            "policy_reference_motion_size",
            *reference,
            *hands,
        ]
        base_quats = [
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([np.cos(0.05), 0.0, 0.0, np.sin(0.05)]),
            np.array([np.cos(0.10), 0.0, 0.0, np.sin(0.10)]),
        ]
        relative = np.array([np.cos(0.025), 0.0, 0.0, np.sin(0.025)])
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            for policy_index in range(3):
                reference_value = np.zeros(1024)
                for future in range(10):
                    reference_value[future * 29 : (future + 1) * 29] = policy_index + future
                    reference_value[290 + future * 29 : 290 + (future + 1) * 29] = 0.1
                    reference_value[580 + future * 6 : 580 + (future + 1) * 6] = _six_d(
                        relative
                    )
                for subrow in range(2):
                    base = base_quats[policy_index]
                    row = [
                        policy_index * 0.02 + subrow * 0.0025,
                        0.0,
                        0.0,
                        0.8,
                        *base,
                        1,
                        100 + policy_index,
                        640,
                        *reference_value,
                        *([float(policy_index)] * 7),
                        *([float(-policy_index)] * 7),
                    ]
                    writer.writerow(row)

    def test_dedup_recovery_and_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "data.csv"
            self._write_csv(csv_path)
            sequence = load_reference_sequence(csv_path, base_sample_mode="first")
        self.assertEqual(sequence.num_frames, 3)
        np.testing.assert_array_equal(sequence.policy_seq, [100, 101, 102])
        np.testing.assert_allclose(sequence.joint_pos[:, 0], [0.0, 1.0, 2.0])
        expected = quat_multiply_wxyz(
            sequence.policy_base_quat_wxyz,
            np.repeat(np.array([[np.cos(0.025), 0.0, 0.0, np.sin(0.025)]]), 3, axis=0),
        )
        np.testing.assert_allclose(sequence.reference_anchor_quat_wxyz, expected, atol=1e-6)

        payload = build_pose_payload(sequence, 2, 10, include_hands=True)
        self.assertNotIn("token_state", payload)
        self.assertEqual(payload["joint_pos"].shape, (10, 29))
        self.assertEqual(payload["body_quat_w"].shape, (10, 4))
        np.testing.assert_array_equal(payload["frame_index"], np.arange(102, 112))
        np.testing.assert_allclose(payload["left_hand_joints"], 2.0)

        regular = build_pose_payload(
            sequence,
            0,
            46,
            include_hands=False,
            checkpoint_layout="regular",
        )
        offsets = np.arange(10) * 5
        np.testing.assert_allclose(regular["joint_pos"][offsets, 0], np.arange(10))
        np.testing.assert_allclose(regular["joint_vel"][offsets, 0], 0.1)
        self.assertEqual(regular["joint_pos"][1, 0], 1.0)
        self.assertEqual(regular["joint_pos"][6, 0], 2.0)

        sonic_v1_1 = build_pose_payload(
            sequence,
            0,
            46,
            include_hands=False,
            checkpoint_layout="sonic_v1_1",
        )
        self.assertEqual(sonic_v1_1.keys(), regular.keys())
        for field in regular:
            np.testing.assert_array_equal(sonic_v1_1[field], regular[field])
        self.assertNotIn("token_state", sonic_v1_1)
        self.assertNotIn("reference_motion", sonic_v1_1)

    def test_sonic_v1_1_rejects_short_step_five_packet(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "data.csv"
            self._write_csv(csv_path)
            sequence = load_reference_sequence(csv_path, base_sample_mode="first")

        with self.assertRaisesRegex(ReferenceDataError, "packet_frames >= 46"):
            build_pose_payload(
                sequence,
                0,
                45,
                include_hands=False,
                checkpoint_layout="sonic_v1_1",
            )

    def test_wire_header_is_protocol_v1_without_token(self) -> None:
        self.assertEqual(PROTOCOL_VERSION, 1)
        self.assertFalse(PUBLISHES_EXTERNAL_TOKEN)
        payload = {
            "joint_pos": np.zeros((10, 29), dtype=np.float32),
            "joint_vel": np.zeros((10, 29), dtype=np.float32),
            "body_quat_w": np.tile(np.array([[1, 0, 0, 0]], dtype=np.float32), (10, 1)),
            "frame_index": np.arange(10, dtype=np.int64),
            "catch_up": np.array([False], dtype=bool),
        }
        message = pack_pose_message(payload, topic="pose", version=PROTOCOL_VERSION)
        header = json.loads(message[4 : 4 + HEADER_SIZE].rstrip(b"\x00"))
        self.assertEqual(header["v"], 1)
        fields = {field["name"]: field for field in header["fields"]}
        self.assertNotIn("token_state", fields)
        self.assertEqual(fields["catch_up"]["dtype"], "bool")


if __name__ == "__main__":
    unittest.main()
