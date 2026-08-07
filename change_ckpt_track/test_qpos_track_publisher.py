from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from change_ckpt_track import csv_reference_publisher as publisher
from change_ckpt_track.csv_reference_publisher import (
    TrackPublisherError,
    build_pose_payload,
)


def _sequence(frames: int = 60) -> SimpleNamespace:
    base = np.arange(frames, dtype=np.float32)
    joint_pos = np.repeat(base[:, None], 29, axis=1)
    joint_vel = joint_pos + 1000.0
    root_quat = np.zeros((frames, 4), dtype=np.float32)
    root_quat[:, 0] = 1.0
    left = np.repeat((base + 2000.0)[:, None], 7, axis=1)
    right = np.repeat((base + 3000.0)[:, None], 7, axis=1)
    return SimpleNamespace(
        csv_path="/tmp/test-recording/data.csv",
        policy_seq=np.arange(100, 100 + frames, dtype=np.int64),
        source_row_index=np.arange(frames, dtype=np.int64),
        source_csv_row_number=np.arange(2, 2 + frames, dtype=np.int64),
        group_row_counts=np.full(frames, 8, dtype=np.int32),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        root_pos=np.zeros((frames, 3), dtype=np.float32),
        root_quat_wxyz=root_quat,
        left_hand_target=left,
        right_hand_target=right,
        joint_names=tuple(f"joint_{index}" for index in range(29)),
    )


class QposTrackPayloadTest(unittest.TestCase):
    def test_init_gate_precedes_one_shot_mode_and_pose_messages(self) -> None:
        events: list[str] = []

        class FakeSocket:
            def setsockopt(self, *_args: object) -> None:
                pass

            def bind(self, _endpoint: str) -> None:
                pass

            def send(self, _message: bytes) -> None:
                events.append("direct_send")

            def close(self, _linger: int) -> None:
                pass

        class FakeContext:
            socket_instance = FakeSocket()

            def socket(self, _kind: int) -> FakeSocket:
                return self.socket_instance

            def term(self) -> None:
                pass

        def build_command_message(
            *, start: bool, stop: bool, planner: bool
        ) -> bytes:
            del stop, planner
            return b"start" if start else b"switch"

        def pack_pose_message(
            _payload: object, *, topic: str, version: int
        ) -> bytes:
            del topic, version
            return b"pose"

        def send_repeated(
            _socket: object, message: bytes, repeats: int, interval_s: float
        ) -> None:
            del repeats, interval_s
            events.append(message.decode())

        def wait_for_gate(
            _args: object,
            _should_stop: object,
            ready_values: set[str] | None = None,
        ) -> None:
            events.append("wait_running" if ready_values else "wait_init")

        args = publisher.build_parser().parse_args(
            [
                "--checkpoint-layout",
                "low_latency",
                "--gate-status-file",
                "/tmp/qpos-track-test-gate",
            ]
        )
        args.checkpoint_layout = "low_latency"
        sequence = _sequence(1)
        with (
            mock.patch("zmq.Context", return_value=FakeContext()),
            mock.patch.object(
                publisher,
                "_import_wire_helpers",
                return_value=(build_command_message, pack_pose_message),
            ),
            mock.patch.object(
                publisher, "_send_repeated", side_effect=send_repeated
            ),
            mock.patch.object(
                publisher, "_wait_for_gate", side_effect=wait_for_gate
            ),
            mock.patch.object(publisher.time, "sleep", return_value=None),
        ):
            self.assertEqual(publisher.run_publisher(args, sequence), 0)

        self.assertEqual(
            events,
            ["wait_init", "switch", "pose", "start", "wait_running"],
        )

    def test_regular_packet_is_consecutive_not_recorded_slots(self) -> None:
        sequence = _sequence()
        payload = build_pose_payload(
            sequence,
            current=3,
            packet_frames=50,
            include_hands=True,
            checkpoint_layout="regular",
        )
        np.testing.assert_array_equal(payload["joint_pos"][:, 0], np.arange(3, 53))
        # These are exactly the ten frames the regular encoder obtains with
        # its step-5 gatherer from this 50-frame consecutive packet.
        np.testing.assert_array_equal(
            payload["joint_pos"][np.arange(10) * 5, 0],
            np.arange(3, 49, 5),
        )
        np.testing.assert_array_equal(
            payload["frame_index"], np.arange(103, 153, dtype=np.int64)
        )
        np.testing.assert_array_equal(
            payload["left_hand_joints"], np.full(7, 2003.0)
        )
        np.testing.assert_array_equal(
            payload["right_hand_joints"], np.full(7, 3003.0)
        )
        self.assertNotIn("token_state", payload)
        self.assertNotIn("reference_motion", payload)

    def test_regular_recorded_lags_are_woven_for_all_five_residues(self) -> None:
        sequence = _sequence(80)
        sequence.root_quat_wxyz[:, 0] = (
            np.arange(80, dtype=np.float32) + 4000.0
        )
        lags = (0, 5, 9, 9, 9, 9, 9, 9, 9, 9)
        payload = build_pose_payload(
            sequence,
            current=3,
            packet_frames=50,
            include_hands=True,
            checkpoint_layout="regular",
            regular_future_lags=lags,
        )
        for residue in range(5):
            packet_rows = residue + np.arange(10) * 5
            expected_source = 3 + residue + np.asarray(lags)
            np.testing.assert_array_equal(
                payload["joint_pos"][packet_rows, 0], expected_source
            )
            np.testing.assert_array_equal(
                payload["joint_vel"][packet_rows, 0],
                expected_source + 1000.0,
            )
            np.testing.assert_array_equal(
                payload["body_quat_w"][packet_rows, 0],
                expected_source + 4000.0,
            )
        # The merger clock stays strictly consecutive even though values are
        # rearranged/repeated for the encoder.
        np.testing.assert_array_equal(
            payload["frame_index"], np.arange(103, 153, dtype=np.int64)
        )
        np.testing.assert_array_equal(
            payload["left_hand_joints"], np.full(7, 2003.0)
        )

    def test_regular_recorded_lags_hold_source_tail(self) -> None:
        payload = build_pose_payload(
            _sequence(12),
            current=10,
            packet_frames=50,
            include_hands=False,
            checkpoint_layout="regular",
            regular_future_lags=(0, 5, 9, 9, 9, 9, 9, 9, 9, 9),
        )
        np.testing.assert_array_equal(
            payload["joint_pos"][np.arange(10) * 5, 0],
            np.asarray([10.0] + [11.0] * 9, dtype=np.float32),
        )
        np.testing.assert_array_equal(
            payload["frame_index"], np.arange(110, 160, dtype=np.int64)
        )

    def test_low_latency_uses_current_through_current_plus_nine(self) -> None:
        payload = build_pose_payload(
            _sequence(),
            current=7,
            packet_frames=10,
            include_hands=False,
            checkpoint_layout="low_latency",
        )
        np.testing.assert_array_equal(payload["joint_pos"][:, 0], np.arange(7, 17))
        self.assertNotIn("left_hand_joints", payload)
        self.assertNotIn("right_hand_joints", payload)

    def test_sonic_v1_1_uses_canonical_step_five_absolute_quaternions(self) -> None:
        sequence = _sequence()
        sequence.root_quat_wxyz[:, 0] = np.arange(60, dtype=np.float32) + 4000.0
        payload = build_pose_payload(
            sequence,
            current=3,
            packet_frames=50,
            include_hands=False,
            checkpoint_layout="sonic_v1_1",
        )
        np.testing.assert_array_equal(payload["joint_pos"][:, 0], np.arange(3, 53))
        np.testing.assert_array_equal(
            payload["joint_pos"][np.arange(10) * 5, 0],
            np.arange(3, 49, 5),
        )
        # The wire carries the recorded world-frame quaternion unchanged.  The
        # v1.1 heading normalization is performed later by the local encoder.
        np.testing.assert_array_equal(
            payload["body_quat_w"][:, 0], np.arange(4003, 4053)
        )
        self.assertNotIn("token_state", payload)
        self.assertNotIn("reference_motion", payload)

    def test_sonic_v1_1_diagnostics_report_canonical_policy_lags(self) -> None:
        args = publisher.build_parser().parse_args(
            ["--checkpoint-layout", "sonic_v1_1"]
        )
        publisher._validate_args(args)
        diagnostics = publisher._sequence_diagnostics(
            _sequence(), args, full_frames=60
        )
        self.assertEqual(
            diagnostics["encoder_source_policy_lags"],
            [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
        )
        self.assertFalse(diagnostics["reference_motion_columns_used"])

    def test_tail_data_is_held_but_frame_index_remains_consecutive(self) -> None:
        payload = build_pose_payload(
            _sequence(12),
            current=10,
            packet_frames=10,
            include_hands=True,
            checkpoint_layout="low_latency",
        )
        np.testing.assert_array_equal(
            payload["joint_pos"][:, 0],
            np.asarray([10.0] + [11.0] * 9, dtype=np.float32),
        )
        np.testing.assert_array_equal(
            payload["frame_index"], np.arange(110, 120, dtype=np.int64)
        )

    def test_regular_rejects_short_packet(self) -> None:
        with self.assertRaisesRegex(TrackPublisherError, "at least 46"):
            build_pose_payload(
                _sequence(),
                current=0,
                packet_frames=45,
                include_hands=True,
                checkpoint_layout="regular",
            )

    def test_sonic_v1_1_rejects_short_packet(self) -> None:
        with self.assertRaisesRegex(TrackPublisherError, "at least 46"):
            build_pose_payload(
                _sequence(),
                current=0,
                packet_frames=45,
                include_hands=True,
                checkpoint_layout="sonic_v1_1",
            )

    def test_sonic_v1_1_direct_api_rejects_recorded_lags(self) -> None:
        with self.assertRaisesRegex(TrackPublisherError, "canonical encoder lags"):
            build_pose_payload(
                _sequence(),
                current=0,
                packet_frames=46,
                include_hands=True,
                checkpoint_layout="sonic_v1_1",
                regular_future_lags=(0, 5, 9, 9, 9, 9, 9, 9, 9, 9),
            )

    def test_recorded_window_rejects_low_latency_layout(self) -> None:
        args = publisher.build_parser().parse_args(
            [
                "--checkpoint-layout",
                "low_latency",
                "--regular-future-window",
                "recorded",
            ]
        )
        with self.assertRaisesRegex(TrackPublisherError, "regular"):
            publisher._validate_args(args)

    def test_recorded_window_rejects_sonic_v1_1_layout(self) -> None:
        args = publisher.build_parser().parse_args(
            [
                "--checkpoint-layout",
                "sonic_v1_1",
                "--regular-future-window",
                "recorded",
            ]
        )
        with self.assertRaisesRegex(TrackPublisherError, "regular"):
            publisher._validate_args(args)


if __name__ == "__main__":
    unittest.main()
