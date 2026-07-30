from __future__ import annotations

import unittest

import numpy as np

from change_ckpt_track.recorded_slot_lags import (
    _infer_from_active_reference,
)


class RecordedSlotLagInferenceTest(unittest.TestCase):
    def test_recovers_source_clamped_ten_slot_schedule(self) -> None:
        rng = np.random.default_rng(7)
        frames = 140
        positions = rng.normal(size=(frames, 29))
        velocities = rng.normal(size=(frames, 29))
        lags = np.asarray([0, 5, 9, 9, 9, 9, 9, 9, 9, 9])
        slot_positions = np.empty((frames, 10, 29))
        slot_velocities = np.empty((frames, 10, 29))
        for frame in range(frames):
            source = np.minimum(frame + lags, frames - 1)
            slot_positions[frame] = positions[source]
            slot_velocities[frame] = velocities[source]
        active = np.zeros((frames, 640), dtype=np.float64)
        active[:, :290] = slot_positions.reshape(frames, 290)
        active[:, 290:580] = slot_velocities.reshape(frames, 290)

        report = _infer_from_active_reference(active)

        self.assertEqual(report["future_slot_policy_lags"], lags.tolist())
        self.assertTrue(
            all(item["score"] == 0.0 for item in report["slots"])
        )


if __name__ == "__main__":
    unittest.main()
