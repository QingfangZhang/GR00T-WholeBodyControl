from __future__ import annotations

import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from controller_replacement.timeline import CsvTimeline, TimelineError


def _yaw_quaternion(degrees: float) -> np.ndarray:
    radians = np.deg2rad(degrees)
    return np.asarray(
        [np.cos(radians / 2.0), 0.0, 0.0, np.sin(radians / 2.0)],
        dtype=np.float64,
    )


class TimelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def _write_csv(
        self,
        times: list[float],
        *,
        quaternions: list[np.ndarray] | None = None,
    ) -> Path:
        path = self.directory / "data.csv"
        header = ["control_time_s", "policy_seq"]
        header.extend(f"root_component_{index}[qpos{index}]" for index in range(8))
        header.extend(f"velocity_{index}[qvel{index}]" for index in range(3))
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            for row_index, time_s in enumerate(times):
                quaternion = (
                    quaternions[row_index]
                    if quaternions is not None
                    else _yaw_quaternion(time_s * 10_000.0)
                )
                qpos = [time_s * 1_000.0, 2.0, 3.0]
                qpos.extend(quaternion.tolist())
                qpos.append(10.0 + time_s * 1_000.0)
                qvel = [time_s * 100.0, 2.0 * time_s * 100.0, -1.0]
                writer.writerow([time_s, row_index // 2, *qpos, *qvel])
        return path

    def test_phase_override_applies_only_to_first_sample(self) -> None:
        path = self._write_csv([0.0, 0.0024, 0.0050, 0.0075, 0.0101])
        timeline = CsvTimeline(
            path,
            start_time_s=0.00125,
            # The phase matcher may retain the later row in a half-period tie.
            start_row_index=1,
        )
        initial_qpos = np.arange(8, dtype=np.float64) + 90.0
        initial_qpos[3:7] = _yaw_quaternion(7.0)
        initial_qvel = np.asarray([8.0, 9.0, 10.0])
        timeline.override_initial_state(initial_qpos, initial_qvel)

        np.testing.assert_array_equal(timeline.state()[0], initial_qpos)
        self.assertAlmostEqual(timeline.current_time_s, 0.00125, places=12)
        self.assertEqual(timeline.current_row_index, 1)
        self.assertEqual(timeline.available_log_rows, 4)

        self.assertTrue(timeline.advance())
        qpos, qvel = timeline.state()
        # The next state is sampled at absolute t=0.00375, not at source row 2
        # and not at an override-relative time.
        self.assertAlmostEqual(timeline.current_time_s, 0.00375, places=12)
        self.assertAlmostEqual(qpos[0], 3.75, places=12)
        self.assertAlmostEqual(qpos[7], 13.75, places=12)
        self.assertAlmostEqual(qvel[0], 0.375, places=12)
        np.testing.assert_allclose(qpos[3:7], _yaw_quaternion(37.5), atol=1.0e-12)
        self.assertEqual(timeline.current_row_index, 2)

        self.assertTrue(timeline.advance())
        self.assertTrue(timeline.advance())
        self.assertTrue(timeline.exhausted)
        self.assertFalse(timeline.advance())
        self.assertEqual(timeline.current_row_index, 3)
        self.assertEqual(timeline.last_row_index, 4)

    def test_from_source_history_uses_exact_time_and_initial_state(self) -> None:
        path = self._write_csv([0.0, 0.0025, 0.0050])
        qpos = np.asarray([4.0, 5.0, 6.0, 1.0, 0.0, 0.0, 0.0, 7.0])
        qvel = np.asarray([1.0, 2.0, 3.0])
        context = SimpleNamespace(
            timeline_start_control_time_s=0.001,
            timeline_start_row_index=0,
            initial_qpos=qpos,
            initial_qvel=qvel,
        )
        timeline = CsvTimeline.from_source_history(path, context)
        actual_qpos, actual_qvel = timeline.state()
        np.testing.assert_array_equal(actual_qpos, qpos)
        np.testing.assert_array_equal(actual_qvel, qvel)
        self.assertAlmostEqual(timeline.current_time_s, 0.001)
        self.assertTrue(timeline.advance())
        # A fresh interpolation at 0.0035 proves that the initial override did
        # not mutate the source row arrays.
        self.assertAlmostEqual(timeline.state()[0][0], 3.5)

    def test_quaternion_slerp_uses_shortest_arc_across_sign_flip(self) -> None:
        q0 = _yaw_quaternion(20.0)
        q1 = -_yaw_quaternion(40.0)
        path = self._write_csv(
            [0.0, 0.0025, 0.0050],
            quaternions=[q0, q1, -_yaw_quaternion(60.0)],
        )
        timeline = CsvTimeline(path, start_time_s=0.00125)
        qpos, _ = timeline.state()
        expected = _yaw_quaternion(30.0)
        # q and -q encode the same orientation.
        self.assertAlmostEqual(abs(float(np.dot(qpos[3:7], expected))), 1.0, places=12)

    def test_state_returns_copies_and_override_is_shape_checked(self) -> None:
        path = self._write_csv([0.0, 0.0025])
        timeline = CsvTimeline(path, start_time_s=0.0)
        original, _ = timeline.state()
        modified, _ = timeline.state()
        modified[0] = 999.0
        np.testing.assert_array_equal(timeline.state()[0], original)
        with self.assertRaisesRegex(TimelineError, "qpos override shape"):
            timeline.override_initial_state(np.zeros(7), np.zeros(3))
        self.assertTrue(timeline.advance())
        with self.assertRaisesRegex(TimelineError, "after advancing"):
            timeline.override_initial_state(np.zeros(8), np.zeros(3))

    def test_small_clock_jitter_is_accepted(self) -> None:
        path = self._write_csv([0.0, 0.0024, 0.0050, 0.00745, 0.0100])
        timeline = CsvTimeline(path, start_time_s=0.001)
        self.assertGreater(timeline.available_log_rows, 1)

    def test_non_monotonic_clock_is_rejected(self) -> None:
        path = self._write_csv([0.0, 0.0025, 0.0025, 0.0050])
        with self.assertRaisesRegex(TimelineError, "strictly increasing"):
            CsvTimeline(path, start_time_s=0.0)

    def test_non_400hz_clock_is_rejected(self) -> None:
        path = self._write_csv([0.0, 0.005, 0.010, 0.015])
        with self.assertRaisesRegex(TimelineError, "expected approximately"):
            CsvTimeline(path, start_time_s=0.0)

    def test_far_row_hint_is_rejected(self) -> None:
        path = self._write_csv([0.0, 0.0025, 0.0050])
        with self.assertRaisesRegex(TimelineError, "not a nearest-row candidate"):
            CsvTimeline(path, start_time_s=0.0, start_row_index=2)


if __name__ == "__main__":
    unittest.main()
