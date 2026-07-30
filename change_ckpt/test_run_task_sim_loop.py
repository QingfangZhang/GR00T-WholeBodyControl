#!/usr/bin/env python3
"""Unit tests for the optional source-root assistance."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


# ``run_task_sim_loop.py`` is primarily an executable and imports its sibling
# ``task_sim_io`` by its script name.  Reproduce the path setup Python provides
# when that executable is launched directly.
CHANGE_CKPT_DIR = Path(__file__).resolve().parent
if str(CHANGE_CKPT_DIR) not in sys.path:
    sys.path.insert(0, str(CHANGE_CKPT_DIR))

from change_ckpt.run_task_sim_loop import _apply_source_root_assist  # noqa: E402


class SourceRootAssistTest(unittest.TestCase):
    def test_none_leaves_state_unchanged(self) -> None:
        qpos = np.asarray([1.0, 2.0, 3.0, 4.0])
        qvel = np.asarray([5.0, 6.0, 7.0, 8.0])
        original_qpos = qpos.copy()
        original_qvel = qvel.copy()

        error = _apply_source_root_assist(
            qpos,
            qvel,
            np.asarray([10.0, 20.0, 30.0, 40.0]),
            np.asarray([50.0, 60.0, 70.0, 80.0]),
            "none",
            source_velocity_active=True,
        )

        np.testing.assert_array_equal(qpos, original_qpos)
        np.testing.assert_array_equal(qvel, original_qvel)
        self.assertEqual(error.shape, (0,))

    def test_xy_updates_only_xy_position_and_velocity(self) -> None:
        qpos = np.asarray([10.0, 20.0, 30.0, 40.0])
        qvel = np.asarray([1.0, 2.0, 3.0, 4.0])

        error = _apply_source_root_assist(
            qpos,
            qvel,
            np.asarray([11.0, 18.0, 999.0, 888.0]),
            np.asarray([5.0, 6.0, 777.0, 666.0]),
            "xy",
            source_velocity_active=True,
        )

        np.testing.assert_array_equal(error, [1.0, -2.0])
        np.testing.assert_array_equal(qpos, [11.0, 18.0, 30.0, 40.0])
        np.testing.assert_array_equal(qvel, [5.0, 6.0, 3.0, 4.0])

    def test_xyz_updates_only_xyz_position_and_velocity(self) -> None:
        qpos = np.asarray([10.0, 20.0, 30.0, 40.0, 50.0])
        qvel = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])

        error = _apply_source_root_assist(
            qpos,
            qvel,
            np.asarray([11.0, 18.0, 35.0, 999.0, 888.0]),
            np.asarray([5.0, 6.0, 7.0, 777.0, 666.0]),
            "xyz",
            source_velocity_active=True,
        )

        np.testing.assert_array_equal(error, [1.0, -2.0, 5.0])
        np.testing.assert_array_equal(qpos, [11.0, 18.0, 35.0, 40.0, 50.0])
        np.testing.assert_array_equal(qvel, [5.0, 6.0, 7.0, 4.0, 5.0])

    def test_eof_hold_zeros_assisted_velocity_only(self) -> None:
        qpos = np.asarray([10.0, 20.0, 30.0, 40.0])
        qvel = np.asarray([1.0, 2.0, 3.0, 4.0])

        _apply_source_root_assist(
            qpos,
            qvel,
            np.asarray([11.0, 18.0, 999.0, 888.0]),
            np.asarray([5.0, 6.0, 777.0, 666.0]),
            "xy",
            source_velocity_active=False,
        )

        np.testing.assert_array_equal(qpos, [11.0, 18.0, 30.0, 40.0])
        np.testing.assert_array_equal(qvel, [0.0, 0.0, 3.0, 4.0])

    def test_nonfinite_input_fails_before_mutating_state(self) -> None:
        input_names = ("qpos", "qvel", "source_qpos", "source_qvel")
        for input_name in input_names:
            with self.subTest(input_name=input_name):
                values = {
                    "qpos": np.asarray([10.0, 20.0, 30.0, 40.0]),
                    "qvel": np.asarray([1.0, 2.0, 3.0, 4.0]),
                    "source_qpos": np.asarray([11.0, 18.0, 35.0, 45.0]),
                    "source_qvel": np.asarray([5.0, 6.0, 7.0, 8.0]),
                }
                values[input_name][1] = np.nan
                original_qpos = values["qpos"].copy()
                original_qvel = values["qvel"].copy()

                with self.assertRaisesRegex(ValueError, input_name):
                    _apply_source_root_assist(
                        values["qpos"],
                        values["qvel"],
                        values["source_qpos"],
                        values["source_qvel"],
                        "xyz",
                        source_velocity_active=True,
                    )

                np.testing.assert_array_equal(
                    values["qpos"], original_qpos, strict=True
                )
                np.testing.assert_array_equal(
                    values["qvel"], original_qvel, strict=True
                )


if __name__ == "__main__":
    unittest.main()
