from __future__ import annotations

import unittest

import numpy as np

from controller_replacement.metrics import RolloutMetrics


class RolloutMetricsTest(unittest.TestCase):
    def test_warmup_is_saved_but_excluded_from_tracking(self) -> None:
        metrics = RolloutMetrics(
            body_joint_names=[f"j{i}" for i in range(29)],
            task_qpos_labels=["drawer"],
            warmup_inferences=1,
        )
        common = dict(
            reference_joint_pos=np.zeros(29),
            q_target=np.zeros(29),
            robot_root_pos=np.zeros(3),
            reference_root_pos=np.zeros(3),
            robot_root_quat_wxyz=np.asarray([1, 0, 0, 0]),
            reference_root_quat_wxyz=np.asarray([1, 0, 0, 0]),
            body_torque=np.zeros(29),
            body_torque_saturation=np.zeros(29, dtype=bool),
        )
        metrics.record_policy(robot_joint_pos=np.ones(29), warmup=True, **common)
        metrics.record_policy(robot_joint_pos=np.full(29, 0.5), warmup=False, **common)
        metrics.record_log(qpos=np.asarray([0, 0, 0.7]), task_qpos=[0.1], source_row_index=7)
        metrics.record_log(qpos=np.asarray([0, 0, 0.6]), task_qpos=[0.4], source_row_index=8)
        report = metrics.report()
        self.assertEqual(report["counts"]["policy_inferences"], 2)
        self.assertEqual(report["counts"]["evaluation_policy_inferences"], 1)
        self.assertAlmostEqual(
            report["tracking"]["robot_vs_reference_rad"]["aggregate_rmse"],
            0.5,
        )
        self.assertEqual(report["task_object_qpos"]["end"], [0.4])
        self.assertIsNone(report["semantic_task_success"]["value"])


if __name__ == "__main__":
    unittest.main()
