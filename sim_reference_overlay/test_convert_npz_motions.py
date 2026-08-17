#!/usr/bin/env python3
"""Tests for the NPZ-to-deployment-CSV converter."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

import convert_npz_motions as converter


def fixture_arrays(timesteps: int = 3) -> dict[str, np.ndarray]:
    joint_pos = np.arange(timesteps * converter.NUM_JOINTS, dtype=np.float32).reshape(
        timesteps, converter.NUM_JOINTS
    )
    joint_vel = -joint_pos.copy()
    body_pos = np.arange(
        timesteps * converter.NUM_SOURCE_BODIES * 3, dtype=np.float32
    ).reshape(timesteps, converter.NUM_SOURCE_BODIES, 3)
    body_quat = np.zeros(
        (timesteps, converter.NUM_SOURCE_BODIES, 4), dtype=np.float32
    )
    body_quat[..., 0] = 1.0
    return {
        "fps": np.asarray([converter.EXPECTED_FPS], dtype=np.int64),
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "body_lin_vel_w": body_pos * np.float32(0.01),
        "body_ang_vel_w": body_pos * np.float32(-0.02),
        "joint_names": np.asarray(converter.G1_ISAACLAB_JOINT_NAMES),
        "body_names": np.asarray(converter.G1_ISAACLAB_BODY_NAMES),
    }


class ConvertNpzMotionsTest(unittest.TestCase):
    def write_fixture(self, directory: Path, **updates: np.ndarray) -> Path:
        payload = fixture_arrays()
        payload.update(updates)
        path = directory / "fixture.npz"
        np.savez(path, **payload)
        return path

    def test_valid_names_and_release_body_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = self.write_fixture(Path(temporary))
            motion = converter.load_and_prepare(source)
            self.assertEqual(motion.timesteps, 3)
            self.assertEqual(motion.arrays["body_pos_w"].shape, (3, 14, 3))
            expected = fixture_arrays()["body_pos_w"][:, converter.RELEASE_BODY_INDEXES, :]
            np.testing.assert_array_equal(motion.arrays["body_pos_w"], expected)
            self.assertIn("validated 29", motion.joint_order_validation)

    def test_missing_names_requires_explicit_assumption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            payload = fixture_arrays()
            payload.pop("joint_names")
            payload.pop("body_names")
            source = Path(temporary) / "unnamed.npz"
            np.savez(source, **payload)
            with self.assertRaisesRegex(converter.ConversionError, "assume-isaaclab-order"):
                converter.load_and_prepare(source)
            motion = converter.load_and_prepare(source, assume_isaaclab_order=True)
            self.assertIn("explicitly assumed", motion.joint_order_validation)
            self.assertIn("explicitly assumed", motion.body_order_validation)

    def test_conflicting_names_are_rejected_even_with_assumption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            names = list(converter.G1_ISAACLAB_JOINT_NAMES)
            names[0], names[1] = names[1], names[0]
            source = self.write_fixture(
                Path(temporary), joint_names=np.asarray(names)
            )
            with self.assertRaisesRegex(converter.ConversionError, "index 0"):
                converter.load_and_prepare(source, assume_isaaclab_order=True)

    def test_nonfinite_data_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            joint_vel = fixture_arrays()["joint_vel"]
            joint_vel[1, 2] = np.nan
            source = self.write_fixture(Path(temporary), joint_vel=joint_vel)
            with self.assertRaisesRegex(converter.ConversionError, "NaN or infinite"):
                converter.load_and_prepare(source)

    def test_hidden_motion_name_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = self.write_fixture(Path(temporary))
            with self.assertRaisesRegex(converter.ConversionError, "must be visible"):
                converter.load_and_prepare(source, motion_name=".hidden")

    def test_conversion_round_trips_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.write_fixture(root)
            output_root = root / "reference_fixture"
            motion_dir, errors = converter.convert_npz(
                source, output_root, motion_name="fixture"
            )
            self.assertEqual(motion_dir, output_root / "fixture")
            self.assertEqual(set(errors), {spec.filename for spec in converter.CSV_SPECS})
            self.assertTrue(all(error >= 0.0 for error in errors.values()))
            self.assertEqual(
                {path.name for path in motion_dir.iterdir()},
                {spec.filename for spec in converter.CSV_SPECS}
                | {"metadata.txt", "info.txt"},
            )
            with self.assertRaisesRegex(converter.ConversionError, "refusing to overwrite"):
                converter.convert_npz(source, output_root, motion_name="fixture")


if __name__ == "__main__":
    unittest.main()
