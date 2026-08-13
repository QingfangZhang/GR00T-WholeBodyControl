#!/usr/bin/env python3
"""Tests for source CSV -> decoder startup-history reconstruction."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from change_ckpt_track.source_history_prefill_core import (
    BODY_QPOS_IDS,
    BODY_QVEL_IDS,
    DEFAULT_ANGLES,
    MUJOCO_TO_ISAACLAB,
    ROBOT_QPOS_IDS_IN_RECEIVED_ORDER,
    build_source_history_prefill,
)


def _write_recording(root: Path) -> Path:
    recording = root / "synthetic_g1_sim"
    recording.mkdir()
    header = ["policy_valid", "policy_seq", "control_time_s"]
    header.extend(f"qpos:test[qpos{index}]" for index in range(50))
    header.extend(f"qvel:test[qvel{index}]" for index in range(49))
    header.extend(f"policy_last_action_in[{index}]" for index in range(29))
    header.extend(f"policy_raw_action_out[{index}]" for index in range(29))
    header.extend(f"policy_received_dof_pos[{index}]" for index in range(43))

    row_index = 0
    with (recording / "data.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for group in range(11):
            # Mimic a recording that starts part-way through offset 0.
            count = 7 if group == 0 else 8
            for within in range(count):
                qpos = [group + index * 0.01 + within * 1e-4 for index in range(50)]
                qvel = [group * 0.1 + index * 0.001 for index in range(49)]
                last_action = [group - 1 + index * 0.001 for index in range(29)]
                raw_action = [group + index * 0.001 for index in range(29)]
                received = [
                    qpos[index] for index in ROBOT_QPOS_IDS_IN_RECEIVED_ORDER
                ]
                writer.writerow(
                    [1, 1000 + group, row_index * 0.0025]
                    + qpos
                    + qvel
                    + last_action
                    + raw_action
                    + received
                )
                row_index += 1
    return recording


def _write_phase_lag_recording(root: Path) -> Path:
    """Create continuous joint motion with policy metadata 10.5 ms late."""

    recording = root / "phase_lag_g1_sim"
    recording.mkdir()
    header = ["policy_valid", "policy_seq", "control_time_s"]
    header.extend(f"qpos:test[qpos{index}]" for index in range(50))
    header.extend(f"qvel:test[qvel{index}]" for index in range(49))
    header.extend(f"policy_last_action_in[{index}]" for index in range(29))
    header.extend(f"policy_raw_action_out[{index}]" for index in range(29))
    header.extend(f"policy_received_dof_pos[{index}]" for index in range(43))

    dt = 0.0025
    row_index = 0
    with (recording / "data.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for group in range(11):
            count = 7 if group == 0 else 8
            group_start = row_index
            received_time = (group_start - 4) * dt - 0.0005
            for _within in range(count):
                row_time = row_index * dt
                qvel = [0.1 + index * 0.001 for index in range(49)]
                qpos = [0.0] * 50
                qpos[3] = 1.0
                for qpos_index in range(7, 50):
                    qpos[qpos_index] = (
                        qpos_index * 0.01
                        + qvel[qpos_index - 1] * row_time
                    )
                received = [
                    qpos_index * 0.01
                    + qvel[qpos_index - 1] * received_time
                    for qpos_index in ROBOT_QPOS_IDS_IN_RECEIVED_ORDER
                ]
                last_action = [group - 1 + index * 0.001 for index in range(29)]
                raw_action = [group + index * 0.001 for index in range(29)]
                writer.writerow(
                    [1, 2000 + group, row_time]
                    + qpos
                    + qvel
                    + last_action
                    + raw_action
                    + received
                )
                row_index += 1
    return recording


class SourceHistoryPrefillTest(unittest.TestCase):
    def test_offset_10_uses_nine_previous_groups(self) -> None:
        with tempfile.TemporaryDirectory(prefix="source_history_test_") as temporary:
            recording = _write_recording(Path(temporary))
            payload = build_source_history_prefill(
                recording, start_policy_offset=10
            )
            expected_source_hash = hashlib.sha256(
                (recording / "data.csv").read_bytes()
            ).hexdigest()

        self.assertEqual(payload["history_entry_count"], 9)
        self.assertEqual(
            [entry["policy_seq"] for entry in payload["entries"]],
            list(range(1001, 1010)),
        )
        self.assertEqual(payload["current"]["policy_seq"], 1010)
        self.assertEqual(
            payload["source_csv_sha256"],
            expected_source_hash,
        )
        self.assertEqual(
            [entry["source_group_row_count"] for entry in payload["entries"]],
            [8] * 9,
        )
        self.assertEqual(payload["current"]["source_group_row_count"], 8)
        self.assertEqual(
            [entry["matched_source_row_index"] for entry in payload["entries"]],
            [entry["source_row_index"] for entry in payload["entries"]],
        )
        self.assertEqual(
            payload["validation"][
                "current_last_action_vs_previous_raw_action_max_abs"
            ],
            0.0,
        )

    def test_body_mapping_matches_cpp_gather(self) -> None:
        with tempfile.TemporaryDirectory(prefix="source_history_test_") as temporary:
            recording = _write_recording(Path(temporary))
            payload = build_source_history_prefill(
                recording, start_policy_offset=10
            )
        first = payload["entries"][0]  # group 1, first row
        body_qpos_ids = np.asarray(BODY_QPOS_IDS, dtype=np.int64)
        q_mujoco = 1.0 + body_qpos_ids * 0.01
        expected_q = (
            q_mujoco[MUJOCO_TO_ISAACLAB]
            - DEFAULT_ANGLES[MUJOCO_TO_ISAACLAB]
        )
        body_qvel_ids = np.asarray(BODY_QVEL_IDS, dtype=np.int64)
        dq_mujoco = 0.1 + body_qvel_ids * 0.001
        expected_dq = dq_mujoco[MUJOCO_TO_ISAACLAB]
        np.testing.assert_allclose(first["body_q"], expected_q, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(first["body_dq"], expected_dq, atol=0.0, rtol=0.0)

    def test_accepts_async_seven_row_group_inside_history(self) -> None:
        with tempfile.TemporaryDirectory(prefix="source_history_test_") as temporary:
            recording = _write_recording(Path(temporary))
            payload = build_source_history_prefill(
                recording, start_policy_offset=9
            )
        self.assertEqual(payload["entries"][0]["source_group_row_count"], 7)

    def test_phase_matches_received_state_before_policy_boundary(self) -> None:
        with tempfile.TemporaryDirectory(prefix="source_history_test_") as temporary:
            recording = _write_phase_lag_recording(Path(temporary))
            payload = build_source_history_prefill(
                recording, start_policy_offset=10
            )

        selected = payload["entries"] + [payload["current"]]
        self.assertEqual(
            [item["source_row_index"] for item in selected],
            [7, 15, 23, 31, 39, 47, 55, 63, 71, 79],
        )
        self.assertEqual(
            [item["matched_source_row_index"] for item in selected],
            [3, 11, 19, 27, 35, 43, 51, 59, 67, 75],
        )
        np.testing.assert_allclose(
            [item["interpolation_offset_s"] for item in selected],
            -0.0005,
            atol=1e-12,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            payload["validation"]["policy_boundary_delay_s"],
            0.0105,
            atol=1e-12,
            rtol=0.0,
        )
        self.assertLess(
            payload["validation"]["received_state_phase_fit_max_abs"],
            1e-12,
        )


if __name__ == "__main__":
    unittest.main()
