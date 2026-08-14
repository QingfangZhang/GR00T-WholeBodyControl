from __future__ import annotations

import math
import unittest

import numpy as np

from controller_replacement.metrics import (
    RolloutMetrics,
    rot6d_orientation_error_rad,
)


def _policy_args(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "robot_joint_pos": np.full(29, 0.5),
        "controller_reference_joint_pos": np.zeros(29),
        "q_target": np.zeros(29),
        "robot_root_pos": np.zeros(3),
        "phase_matched_source_root_pos": np.zeros(3),
        "robot_root_quat_wxyz": np.asarray([1, 0, 0, 0]),
        "phase_matched_source_root_quat_wxyz": np.asarray([1, 0, 0, 0]),
        "controller_reference_orientation_error_rad": 0.0,
        "controller_reference_height_error_m": None,
        "warmup": False,
    }
    values.update(overrides)
    return values


class RolloutMetricsTest(unittest.TestCase):
    def _metrics(self, *, warmup: int = 1) -> RolloutMetrics:
        return RolloutMetrics(
            body_joint_names=[f"j{i}" for i in range(29)],
            task_qpos_labels=["drawer"],
            warmup_inferences=warmup,
        )

    def test_warmup_is_saved_but_excluded_from_tracking(self) -> None:
        metrics = self._metrics()
        metrics.observe_fall(
            qpos=[0, 0, 0.7],
            time_s=0.0,
            log_row_exclusive=0,
            source_row_index=7,
        )
        metrics.record_policy(**_policy_args(robot_joint_pos=np.ones(29), warmup=True))
        metrics.record_policy(**_policy_args())
        metrics.record_log(
            qpos=np.asarray([0, 0, 0.7]),
            task_qpos=[0.1],
            source_row_index=7,
        )
        metrics.record_log(
            qpos=np.asarray([0, 0, 0.6]),
            task_qpos=[0.4],
            source_row_index=8,
        )
        report = metrics.report()
        self.assertEqual(report["format_version"], 3)
        self.assertEqual(report["counts"]["policy_inferences"], 2)
        self.assertEqual(report["counts"]["evaluation_policy_inferences"], 1)
        self.assertAlmostEqual(
            report["tracking"]["robot_vs_reference_rad"]["aggregate_rmse"],
            0.5,
        )
        self.assertEqual(report["task_object_qpos"]["end"], [0.4])
        self.assertIsNone(report["semantic_task_success"]["value"])
        self.assertFalse(report["stability"]["post_fall_physics_continued"])
        self.assertFalse(
            report["task_object_qpos"]["includes_post_fall_physics"]
        )
        self.assertFalse(report["contacts"]["includes_post_fall_physics"])

    def test_body_torque_samples_every_evaluation_pd_update(self) -> None:
        metrics = self._metrics(warmup=0)
        saturation = np.zeros(29, dtype=bool)
        saturation[3] = True
        metrics.record_pd_control(
            body_torque=np.full(29, 100.0),
            body_torque_saturation=saturation,
            evaluation=False,
        )
        for torque, saturated in ((1.0, False), (2.0, True), (-4.0, True), (3.0, False)):
            update_saturation = np.zeros(29, dtype=bool)
            update_saturation[3] = saturated
            metrics.record_pd_control(
                body_torque=np.full(29, torque),
                body_torque_saturation=update_saturation,
                evaluation=True,
            )

        report = metrics.report()
        self.assertEqual(report["counts"]["body_pd_updates"], 5)
        self.assertEqual(report["counts"]["evaluation_body_pd_updates"], 4)
        self.assertEqual(
            report["control"]["per_joint_torque_saturation_fraction"][3],
            0.5,
        )
        self.assertEqual(
            report["control"][
                "pd_updates_with_any_torque_saturation_fraction"
            ],
            0.5,
        )
        self.assertEqual(
            report["control"]["per_joint_max_abs_applied_torque"][3],
            4.0,
        )

    def test_source_and_controller_reference_root_targets_are_separate(self) -> None:
        metrics = self._metrics(warmup=0)
        yaw_90 = np.asarray(
            [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]
        )
        metrics.record_policy(
            **_policy_args(
                robot_root_pos=np.asarray([1.0, 0.0, 0.0]),
                phase_matched_source_root_pos=np.zeros(3),
                phase_matched_source_root_quat_wxyz=yaw_90,
                controller_reference_orientation_error_rad=0.25,
                controller_reference_height_error_m=0.1,
            )
        )
        root = metrics.report()["tracking"]["root"]
        source = root["robot_vs_phase_matched_source"]
        controller = root["robot_vs_controller_reference"]
        self.assertAlmostEqual(
            source["translation_xyz_m"]["aggregate_rmse"],
            1.0 / math.sqrt(3.0),
        )
        self.assertAlmostEqual(
            source["orientation_error_rad"]["rmse_rad"], math.pi / 2
        )
        self.assertAlmostEqual(
            controller["anchor_orientation_error_rad"]["rmse_rad"], 0.25
        )
        self.assertTrue(controller["anchor_height_available"])
        self.assertAlmostEqual(
            controller["anchor_height_error_m"]["rmse_m"], 0.1
        )
        self.assertFalse(controller["absolute_translation_available"])
        self.assertIsNone(controller["absolute_translation_xyz_m"])

    def test_rot6d_error_uses_the_exact_relative_rotation(self) -> None:
        identity = np.asarray([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        self.assertAlmostEqual(rot6d_orientation_error_rad(identity), 0.0)
        yaw_90 = np.asarray([0.0, -1.0, 1.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(
            rot6d_orientation_error_rad(yaw_90), math.pi / 2
        )

    def test_fall_latches_and_only_stops_tracking_metrics(self) -> None:
        metrics = self._metrics(warmup=0)
        self.assertFalse(
            metrics.observe_fall(
                qpos=[0, 0, 0.7],
                time_s=0.0,
                log_row_exclusive=0,
                source_row_index=10,
            )
        )
        metrics.record_policy(**_policy_args())
        metrics.record_log(
            qpos=[0, 0, 0.7], task_qpos=[0.0], source_row_index=10
        )
        self.assertTrue(
            metrics.observe_fall(
                qpos=[0, 0, 0.1],
                time_s=0.0025,
                log_row_exclusive=1,
                source_row_index=11,
            )
        )
        # Recovering height does not clear the latched event.
        self.assertTrue(
            metrics.observe_fall(
                qpos=[0, 0, 0.8],
                time_s=0.005,
                log_row_exclusive=2,
                source_row_index=12,
            )
        )
        metrics.record_policy(**_policy_args(robot_joint_pos=np.ones(29)))
        metrics.record_log(
            qpos=[0, 0, 0.8], task_qpos=[0.6], source_row_index=12
        )
        report = metrics.report()
        self.assertTrue(report["stability"]["fallen"])
        self.assertAlmostEqual(report["stability"]["root_z_min_m"], 0.1)
        self.assertAlmostEqual(report["stability"]["root_z_max_m"], 0.8)
        self.assertFalse(report["stability"]["initially_fallen"])
        self.assertEqual(report["stability"]["first_fall_log_row_exclusive"], 1)
        self.assertEqual(report["counts"]["evaluation_policy_inferences"], 1)
        self.assertEqual(report["counts"]["post_fall_policy_inferences"], 1)
        self.assertEqual(report["task_object_qpos"]["end"], [0.6])
        self.assertFalse(report["evaluation_validity"]["task_success_eligible"])
        self.assertFalse(report["semantic_task_success"]["value"])
        self.assertTrue(report["stability"]["post_fall_physics_continued"])
        self.assertTrue(
            report["task_object_qpos"]["includes_post_fall_physics"]
        )
        self.assertTrue(report["contacts"]["includes_post_fall_physics"])

    def test_initial_and_terminal_boundary_fall_semantics(self) -> None:
        initial = self._metrics(warmup=0)
        initial.observe_fall(
            qpos=[0, 0, 0.19],
            time_s=0.0,
            log_row_exclusive=0,
            source_row_index=1,
        )
        self.assertTrue(initial.report()["stability"]["initially_fallen"])

        terminal = self._metrics(warmup=0)
        terminal.record_log(
            qpos=[0, 0, 0.7], task_qpos=[0.0], source_row_index=3
        )
        terminal.observe_fall(
            qpos=[0, 0, 0.1],
            time_s=0.0025,
            log_row_exclusive=1,
            source_row_index=3,
        )
        report = terminal.report()
        self.assertEqual(report["evaluation_validity"]["valid_log_rows_exclusive"], 1)
        self.assertTrue(report["stability"]["fallen"])
        self.assertFalse(report["stability"]["post_fall_physics_continued"])
        self.assertFalse(
            report["task_object_qpos"]["includes_post_fall_physics"]
        )
        self.assertFalse(report["contacts"]["includes_post_fall_physics"])

        hidden_terminal = self._metrics(warmup=0)
        hidden_terminal.record_log(
            qpos=[0, 0, 0.7], task_qpos=[0.0], source_row_index=3
        )
        hidden_terminal.observe_fall(
            qpos=[0, 0, 0.1],
            time_s=0.0025,
            log_row_exclusive=1,
            source_row_index=3,
        )
        hidden_terminal.record_terminal(
            qpos=[0, 0, 0.1], task_qpos=[0.0], time_s=0.0025
        )
        self.assertFalse(
            hidden_terminal.report()["stability"][
                "post_fall_physics_continued"
            ]
        )

    def test_hidden_terminal_endpoint_updates_task_and_root_without_csv_row(self) -> None:
        metrics = self._metrics(warmup=0)
        metrics.record_log(
            qpos=[0, 0, 0.7], task_qpos=[0.1], source_row_index=3
        )
        metrics.record_terminal(
            qpos=[0, 0, 0.65], task_qpos=[0.8], time_s=0.0025
        )
        report = metrics.report()
        self.assertEqual(report["counts"]["logged_400hz_rows"], 1)
        self.assertEqual(report["task_object_qpos"]["end"], [0.8])
        self.assertTrue(
            report["task_object_qpos"]["includes_hidden_terminal_endpoint"]
        )
        self.assertAlmostEqual(report["stability"]["root_z_min_m"], 0.65)
        with self.assertRaisesRegex(ValueError, "already recorded"):
            metrics.record_terminal(
                qpos=[0, 0, 0.65], task_qpos=[0.8], time_s=0.0025
            )


if __name__ == "__main__":
    unittest.main()
