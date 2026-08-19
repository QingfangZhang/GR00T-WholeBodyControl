#!/usr/bin/env python3
"""Tests for deployment CSV to canonical G1 NPZ reconstruction."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

import build_real_safe_reference as safe
import convert_csv_motions_to_npz as reverse
import convert_npz_motions as csvio


class ConvertCsvMotionsToNpzTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.kinematics = safe.G1Kinematics(safe.default_scene_path())

    def make_stationary_motion(self, timesteps: int = 5) -> csvio.PreparedMotion:
        joint_pos = np.repeat(
            safe.DEFAULT_ANGLES_ISAACLAB.astype(np.float32)[None, :],
            timesteps,
            axis=0,
        )
        root_pos = np.asarray([0.1, -0.2, 0.78], dtype=np.float64)
        root_rotation = Rotation.identity()
        body_pos, body_quat = self.kinematics.release_body_pose(
            joint_pos[0], root_pos, root_rotation
        )
        body_pos = np.repeat(body_pos.astype(np.float32)[None, :], timesteps, axis=0)
        body_quat = np.repeat(
            body_quat.astype(np.float32)[None, :], timesteps, axis=0
        )
        return csvio.PreparedMotion(
            name="stationary_csv_fixture",
            source_path=Path("stationary_csv_fixture"),
            source_sha256="0" * 64,
            fps=csvio.EXPECTED_FPS,
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
            body_order_validation="unit-test release order",
            quaternion_max_norm_deviation=float(
                np.max(
                    np.abs(
                        np.linalg.norm(body_quat.astype(np.float64), axis=-1) - 1.0
                    )
                )
            ),
        )

    def test_reconstructs_canonical_archive_and_round_trips_release_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.make_stationary_motion()
            motion_dir, _ = csvio.publish_prepared_motion(source, root / "csv_fixture")
            output, report = reverse.convert_csv_to_npz(
                motion_dir.parent,
                root / "stationary_csv_fixture.npz",
            )

            self.assertEqual(report.timesteps, source.timesteps)
            self.assertLessEqual(report.fk_release_position_error, 2.0e-5)
            self.assertLessEqual(report.fk_release_quaternion_error, 2.0e-5)
            with np.load(output, allow_pickle=False) as archive:
                self.assertEqual(tuple(archive.files), reverse.NPZ_KEYS)
                self.assertEqual(
                    archive["body_pos_w"].shape,
                    (source.timesteps, csvio.NUM_SOURCE_BODIES, 3),
                )
                self.assertEqual(
                    archive["body_quat_w"].shape,
                    (source.timesteps, csvio.NUM_SOURCE_BODIES, 4),
                )
                self.assertEqual(archive["joint_pos"].dtype, np.dtype(np.float32))
                self.assertEqual(archive["fps"].dtype, np.dtype(np.int64))
                np.testing.assert_array_equal(
                    archive["joint_names"], csvio.G1_ISAACLAB_JOINT_NAMES
                )
                np.testing.assert_array_equal(
                    archive["body_names"], csvio.G1_ISAACLAB_BODY_NAMES
                )

            prepared = csvio.load_and_prepare(output)
            for key in csvio.REQUIRED_ARRAYS:
                np.testing.assert_array_equal(prepared.arrays[key], source.arrays[key])

            with self.assertRaisesRegex(csvio.ConversionError, "refusing to overwrite"):
                reverse.convert_csv_to_npz(motion_dir, output)


if __name__ == "__main__":
    unittest.main()
