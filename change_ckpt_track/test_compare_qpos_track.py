#!/usr/bin/env python3
"""Unit tests for the qpos-track comparison and its CSV fallbacks."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from compare_qpos_track import (  # noqa: E402
    MJ_JOINT_NAMES,
    MUJOCO_TO_ISAACLAB,
    analyze_run,
    compare_analyses,
    read_target_motion,
)


def _write_split_log(
    path: Path, prefix: str, values: np.ndarray, indices: np.ndarray
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "index",
                "time_ms",
                "time_realtime_ms",
                "time_monotonic_ms",
                "ros_timestamp",
                *[f"{prefix}_{column}" for column in range(values.shape[1])],
            ]
        )
        for row, index in enumerate(indices):
            time_ms = row * 20.0
            writer.writerow(
                [index, time_ms, 1000.0 + time_ms, 2000.0 + time_ms, 0.0]
                + values[row].tolist()
            )


def _make_run(
    root: Path,
    name: str,
    tracking_offset: float,
    playing: list[float],
    target_frame_sequence: list[int] | None = None,
) -> Path:
    run_dir = root / name
    deploy_dir = run_dir / "deploy_csv"
    indices = np.arange(len(playing), dtype=np.int64) + 10
    if target_frame_sequence is None:
        target_frame_sequence = list(range(len(playing)))
    if len(target_frame_sequence) != len(playing):
        raise ValueError("target-frame sequence must match playing rows")
    reference_frame_count = max(target_frame_sequence) + 1
    reference_target = np.asarray(
        [
            np.arange(29, dtype=np.float64) * 0.01 + row * 0.001
            for row in range(reference_frame_count)
        ]
    )
    target = reference_target[
        np.asarray(target_frame_sequence, dtype=np.int64)
    ]
    measured = target + tracking_offset
    _write_split_log(deploy_dir / "q.csv", "q", measured, indices)
    _write_split_log(deploy_dir / "dq.csv", "dq", np.zeros_like(target), indices)
    raw_action = np.asarray(
        [np.full(29, row * 0.25) for row in range(len(playing))],
        dtype=np.float64,
    )
    _write_split_log(deploy_dir / "action.csv", "act", raw_action, indices)
    token = np.asarray(
        [np.full(64, row / 16.0) for row in range(len(playing))],
        dtype=np.float64,
    )
    _write_split_log(deploy_dir / "token_state.csv", "token", token, indices)
    _write_split_log(
        deploy_dir / "motion_playing.csv",
        "playing",
        np.asarray(playing, dtype=np.float64)[:, None],
        indices,
    )
    (deploy_dir / "metadata.json").write_text(
        json.dumps({"logging": {"dt": 0.02}, "robot_config": {}}),
        encoding="utf-8",
    )
    with (run_dir / "target_motion.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        for joints in target:
            writer.writerow([0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0, *joints, ""])
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "fallen": False,
                "invalid_state": False,
                "task_qpos_start": 50,
                "task_qpos_labels": ["drawer_joint", "prop_joint"],
                "initial_task_qpos": [0.0, 1.0],
                "final_task_qpos": [0.5, 1.25],
                "current_source_task_qpos": [1.0, 1.5],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "launch_manifest.json").write_text(
        json.dumps(
            {
                "checkpoint": name,
                "initialization": {"warmup_exclusion_s": 0.0},
                "models": {},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "reference_diagnostics.json").write_text(
        json.dumps(
            {
                "joint_order": "isaaclab",
                "joint_names": list(MJ_JOINT_NAMES),
                "source_row_indices": list(range(reference_frame_count)),
                "policy_seq": list(
                    range(100, 100 + reference_frame_count)
                ),
            }
        ),
        encoding="utf-8",
    )
    reference_isaac = np.empty_like(reference_target)
    reference_isaac[:, MUJOCO_TO_ISAACLAB] = reference_target
    np.savez_compressed(
        run_dir / "prepared_reference.npz",
        joint_pos=reference_isaac,
        policy_seq=np.arange(
            100, 100 + reference_frame_count, dtype=np.int64
        ),
    )
    (run_dir / "publisher.log").write_text(
        "[qpos-track] source complete; late_ticks=0. Simulator may flush.\n",
        encoding="utf-8",
    )
    (run_dir / "deploy.log").write_text(
        "\n".join(
            [
                "[ZMQEndpointInterface] Merged streamed data: 50 current-rate "
                "frames, window [100..149], did_catchup=1",
                "[ZMQEndpointInterface] Catch-up: Reset to frame 0 at global frame 100",
                "[ZMQEndpointInterface] Merged streamed data: 51 current-rate "
                "frames, window [100..150], did_catchup=0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir


class CompareQposTrackTest(unittest.TestCase):
    def test_motion_gate_alignment_and_tracking_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            regular_dir = _make_run(
                root, "demo_regular", tracking_offset=0.1, playing=[0, 1, 1, 1, 0]
            )
            low_dir = _make_run(
                root,
                "demo_low_latency",
                tracking_offset=0.2,
                playing=[0, 1, 1, 1, 0],
            )
            regular = analyze_run(regular_dir)
            low = analyze_run(low_dir)

            self.assertEqual(regular.indices.tolist(), [11, 12, 13])
            self.assertEqual(
                regular.summary["playback_alignment"]["evaluation_rows"], 3
            )
            self.assertAlmostEqual(
                regular.summary["tracking"]["metrics"]["overall"]["rmse_rad"],
                0.1,
                places=12,
            )
            self.assertAlmostEqual(
                low.summary["tracking"]["metrics"]["overall"]["rmse_rad"],
                0.2,
                places=12,
            )
            self.assertTrue(regular.summary["signals"]["raw_action"]["all_finite"])
            self.assertAlmostEqual(
                regular.summary["signals"]["raw_action"]["smoothness"][
                    "step_delta_rms"
                ],
                0.25,
                places=12,
            )
            task = regular.summary["task_object_terminal_state"]
            self.assertTrue(task["available"])
            self.assertAlmostEqual(task["motion_l2_from_initial"], np.sqrt(0.3125))
            self.assertAlmostEqual(task["drawer"]["source_normalized_progress"], 0.5)

            comparison = compare_analyses(regular, low)
            self.assertAlmostEqual(
                comparison["tracking"]["overall"][
                    "rmse_rad_low_minus_regular"
                ],
                0.1,
                places=12,
            )
            self.assertTrue(comparison["alignment"]["same_target_within_1e-6"])
            self.assertTrue(
                regular.summary["protocol_timing_audit"]["valid"]
            )
            self.assertTrue(comparison["runtime_validity"]["both_valid"])

    def test_prepared_reference_alignment_removes_one_cycle_duplicate_bias(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            regular_dir = _make_run(
                root,
                "duplicate_regular",
                tracking_offset=0.1,
                playing=[1] * 9,
                target_frame_sequence=[0, 1, 2, 3, 3, 4, 5, 6, 7],
            )
            low_dir = _make_run(
                root,
                "duplicate_low_latency",
                tracking_offset=0.2,
                playing=[1] * 9,
                target_frame_sequence=[0, 1, 2, 3, 4, 5, 6, 7, 7],
            )

            regular = analyze_run(regular_dir, warmup_s=0.0)
            low = analyze_run(low_dir, warmup_s=0.0)
            comparison = compare_analyses(regular, low)

            self.assertEqual(
                regular.reference_frame_indices.tolist(),
                [0, 1, 2, 3, 3, 4, 5, 6, 7],
            )
            regular_match = regular.summary["playback_alignment"][
                "reference_frame_alignment"
            ]
            low_match = low.summary["playback_alignment"][
                "reference_frame_alignment"
            ]
            self.assertEqual(regular_match["unexpected_duplicate_count"], 1)
            self.assertEqual(
                regular_match["unexpected_duplicate_target_rows"], [4]
            )
            self.assertEqual(low_match["unexpected_duplicate_count"], 0)
            self.assertEqual(low_match["terminal_hold_count"], 1)
            alignment = comparison["alignment"]
            self.assertEqual(alignment["common_reference_frames"], 8)
            self.assertFalse(alignment["same_reference_schedule_by_ordinal"])
            self.assertGreater(
                alignment["ordinal_target_max_abs_difference_rad"], 0.0
            )
            self.assertEqual(alignment["target_max_abs_difference_rad"], 0.0)
            self.assertTrue(alignment["same_target_within_1e-6"])
            overall = comparison["tracking"]["overall"]
            self.assertAlmostEqual(overall["regular_rmse_rad"], 0.1)
            self.assertAlmostEqual(overall["low_latency_rmse_rad"], 0.2)
            self.assertAlmostEqual(
                overall["rmse_rad_low_minus_regular"], 0.1
            )

    def test_only_first_playing_segment_is_evaluated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = _make_run(
                Path(temporary),
                "two_segments_regular",
                tracking_offset=0.1,
                playing=[0, 1, 1, 0, 1],
            )
            result = analyze_run(run_dir, warmup_s=0.0)
            self.assertEqual(result.indices.tolist(), [11, 12])
            self.assertEqual(
                len(result.summary["playback_alignment"]["segments"]), 2
            )
            self.assertTrue(
                any(
                    "only the first" in warning
                    for warning in result.summary["warnings"]
                )
            )

    def test_named_target_columns_are_reordered_to_mujoco_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "target_motion.csv"
            reversed_names = list(reversed(MJ_JOINT_NAMES))
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    [
                        "frame_index",
                        "root_x",
                        "root_y",
                        "root_z",
                        "root_qw",
                        "root_qx",
                        "root_qy",
                        "root_qz",
                        *[f"{name}_joint" for name in reversed_names],
                    ]
                )
                writer.writerow(
                    [7, 0, 0, 0.8, 1, 0, 0, 0]
                    + [MJ_JOINT_NAMES.index(name) for name in reversed_names]
                )
            parsed = read_target_motion(path)
            self.assertTrue(parsed.has_header)
            self.assertEqual(parsed.explicit_indices.tolist(), [7])
            np.testing.assert_array_equal(
                parsed.joint_pos_mj[0], np.arange(29, dtype=np.float64)
            )

    def test_late_ticks_and_noninitial_catchup_invalidate_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = _make_run(
                Path(temporary),
                "timing_failure_regular",
                tracking_offset=0.1,
                playing=[0, 1, 1, 1, 0],
            )
            (run_dir / "publisher.log").write_text(
                "[qpos-track] source complete; late_ticks=2.\n",
                encoding="utf-8",
            )
            (run_dir / "deploy.log").write_text(
                "\n".join(
                    [
                        "[ZMQEndpointInterface] Merged streamed data: 50 frames, "
                        "did_catchup=1",
                        "[ZMQEndpointInterface] Catch-up: Reset to frame 0 at "
                        "global frame 100",
                        "[StreamedMotionMerger] WARNING: incoming_frame_end "
                        "(120) <= stream_window_end (130) - forcing catch-up",
                        "[StreamedMotionMerger] CATCH-UP: gap too large or old "
                        "data expired",
                        "[ZMQEndpointInterface] Merged streamed data: 50 frames, "
                        "did_catchup=1",
                        "[ZMQEndpointInterface] Catch-up: Reset to frame 0 at "
                        "global frame 120",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            result = analyze_run(run_dir, warmup_s=0.0)
            audit = result.summary["protocol_timing_audit"]
            self.assertFalse(audit["valid"])
            self.assertEqual(audit["publisher"]["late_ticks"], 2)
            stream = audit["deploy_stream"]
            self.assertEqual(stream["non_initial_did_catchup_count"], 1)
            self.assertEqual(stream["forcing_catchup_count"], 1)
            self.assertEqual(stream["gap_catchup_count"], 1)
            self.assertEqual(stream["normal_initial_reset_count"], 1)
            self.assertEqual(stream["non_initial_catchup_reset_count"], 1)
            self.assertGreaterEqual(stream["abnormal_related_line_count"], 4)
            self.assertFalse(result.summary["runtime_validity"]["valid"])
            self.assertTrue(result.summary["runtime_validity"]["warnings"])


if __name__ == "__main__":
    unittest.main()
