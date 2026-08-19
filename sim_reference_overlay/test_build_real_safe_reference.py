#!/usr/bin/env python3
"""Tests for the staged real-robot reference builder."""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

import build_real_safe_reference as safe
import convert_npz_motions as csvio


def stationary_source(timesteps: int = 10) -> csvio.PreparedMotion:
    joint_pos = np.repeat(
        safe.DEFAULT_ANGLES_ISAACLAB.astype(np.float32)[None, :],
        timesteps,
        axis=0,
    )
    body_pos = np.zeros((timesteps, len(safe.RELEASE_BODY_NAMES), 3), dtype=np.float32)
    body_pos[:, 0, 2] = 0.8
    body_quat = np.zeros((timesteps, len(safe.RELEASE_BODY_NAMES), 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    return csvio.PreparedMotion(
        name="stationary",
        source_path=Path("stationary_fixture.npz"),
        source_sha256="0" * 64,
        fps=safe.FPS,
        timesteps=timesteps,
        arrays={
            "joint_pos": joint_pos,
            "joint_vel": np.zeros_like(joint_pos),
            "body_pos_w": body_pos,
            "body_quat_w": body_quat,
            "body_lin_vel_w": np.zeros_like(body_pos),
            "body_ang_vel_w": np.zeros_like(body_pos),
        },
        joint_order_validation="unit-test canonical order",
        body_order_validation="unit-test canonical order",
        quaternion_max_norm_deviation=0.0,
    )


class BuildRealSafeReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.kinematics = safe.G1Kinematics(safe.default_scene_path())

    def test_default_pose_mapping_matches_policy_order(self) -> None:
        expected = np.asarray(
            [
                -0.312,
                -0.312,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.669,
                0.669,
                0.2,
                0.2,
                -0.363,
                -0.363,
                0.2,
                -0.2,
                0.0,
                0.0,
                0.0,
                0.0,
                0.6,
                0.6,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        np.testing.assert_array_equal(safe.DEFAULT_ANGLES_ISAACLAB, expected)

    def test_interpolation_boundary_conditions(self) -> None:
        u = np.asarray([0.0, 1.0])
        np.testing.assert_array_equal(safe.minimum_jerk(u), [0.0, 1.0])
        np.testing.assert_array_equal(safe.brake_displacement(u), [0.0, 0.5])

        epsilon = 1.0e-5
        self.assertAlmostEqual(
            float((safe.minimum_jerk(np.asarray([epsilon]))[0]) / epsilon),
            0.0,
            places=7,
        )
        self.assertAlmostEqual(
            float(
                (0.5 - safe.brake_displacement(np.asarray([1.0 - epsilon]))[0])
                / epsilon
            ),
            0.0,
            places=7,
        )

    def test_stationary_fixture_has_static_safe_endpoints(self) -> None:
        source = stationary_source()
        config = safe.BuildConfig(
            source_start_frame=1,
            source_end_frame=8,
            initial_hold_s=0.1,
            entry_s=0.1,
            source_settle_s=0.1,
            brake_s=0.1,
            stop_hold_s=0.1,
            return_s=0.1,
            final_hold_s=0.1,
        )
        trajectory = safe.build_root_and_joint_trajectory(
            source, self.kinematics, config
        )
        arrays = safe.compute_release_arrays(trajectory, self.kinematics)
        report = safe.validate_built_motion(
            source, trajectory, arrays, self.kinematics, config
        )

        self.assertEqual(report.total_frames, arrays["joint_pos"].shape[0])
        np.testing.assert_array_equal(arrays["joint_pos"][0], arrays["joint_pos"][-1])
        self.assertEqual(float(np.max(np.abs(arrays["joint_vel"]))), 0.0)
        self.assertLessEqual(report.max_quaternion_norm_deviation, 1.0e-6)
        self.assertGreaterEqual(report.min_adjacent_quaternion_dot, 0.0)
        for key in csvio.REQUIRED_ARRAYS:
            self.assertTrue(np.isfinite(arrays[key]).all(), key)

    def test_relaxed_brake_guard_applies_to_arms_only(self) -> None:
        config = safe.BuildConfig(
            source_start_frame=1,
            source_end_frame=8,
            initial_hold_s=0.1,
            entry_s=0.1,
            source_settle_s=0.1,
            brake_s=0.5,
            stop_hold_s=0.1,
            return_s=1.5,
            final_hold_s=0.1,
            brake_joint_speed_limit=1.1,
            brake_non_arm_speed_limit=0.7,
        )

        arm_source = stationary_source()
        arm_source.arrays["joint_vel"][8, 21] = 1.0  # left elbow
        arm_trajectory = safe.build_root_and_joint_trajectory(
            arm_source, self.kinematics, config
        )
        arm_arrays = safe.compute_release_arrays(arm_trajectory, self.kinematics)
        arm_report = safe.validate_built_motion(
            arm_source, arm_trajectory, arm_arrays, self.kinematics, config
        )
        self.assertGreater(arm_report.brake_max_joint_speed, 0.7)
        self.assertEqual(arm_report.brake_max_non_arm_speed, 0.0)

        non_arm_source = stationary_source()
        non_arm_source.arrays["joint_vel"][8, 8] = 1.0  # waist pitch
        non_arm_trajectory = safe.build_root_and_joint_trajectory(
            non_arm_source, self.kinematics, config
        )
        non_arm_arrays = safe.compute_release_arrays(
            non_arm_trajectory, self.kinematics
        )
        with self.assertRaisesRegex(safe.SafeMotionError, "brake non-arm"):
            safe.validate_built_motion(
                non_arm_source,
                non_arm_trajectory,
                non_arm_arrays,
                self.kinematics,
                config,
            )


if __name__ == "__main__":
    unittest.main()
