#!/usr/bin/env python3
"""Focused tests for qpos-track source-history selection and phase matching."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from change_ckpt_track.test_source_history_prefill_core import (
    _write_phase_lag_recording,
    _write_recording,
)
from change_ckpt_track.source_history_prefill import (
    SourceHistoryError,
    build_source_history_prefill,
    resolve_raw_policy_offset,
)


class TrackSourceHistoryPrefillTest(unittest.TestCase):
    def test_actual_sequence_resolves_past_truncated_first_group(self) -> None:
        with tempfile.TemporaryDirectory(prefix="track_source_history_") as temporary:
            recording = _write_recording(Path(temporary))
            self.assertEqual(
                resolve_raw_policy_offset(recording, start_policy_seq=1010), 10
            )
            payload = build_source_history_prefill(
                recording, start_policy_seq=1010
            )

        self.assertEqual(payload["current"]["policy_seq"], 1010)
        self.assertEqual(payload["raw_start_policy_offset"], 10)
        self.assertEqual(
            payload["selection"]["resolved_raw_policy_group_offset"], 10
        )
        self.assertEqual(
            [entry["policy_seq"] for entry in payload["entries"]],
            list(range(1001, 1010)),
        )

    def test_raw_offset_and_actual_sequence_have_identical_payload_state(self) -> None:
        with tempfile.TemporaryDirectory(prefix="track_source_history_") as temporary:
            recording = _write_phase_lag_recording(Path(temporary))
            by_sequence = build_source_history_prefill(
                recording, start_policy_seq=2010
            )
            by_offset = build_source_history_prefill(
                recording, start_policy_offset=10
            )

        self.assertEqual(by_sequence["entries"], by_offset["entries"])
        self.assertEqual(by_sequence["current"], by_offset["current"])
        self.assertLess(
            by_sequence["validation"]["received_state_phase_fit_max_abs"],
            1e-12,
        )
        self.assertAlmostEqual(
            max(by_sequence["validation"]["policy_boundary_delay_s"]),
            0.0105,
            places=12,
        )

    def test_requires_exactly_one_selector(self) -> None:
        with tempfile.TemporaryDirectory(prefix="track_source_history_") as temporary:
            recording = _write_recording(Path(temporary))
            with self.assertRaisesRegex(SourceHistoryError, "exactly one"):
                build_source_history_prefill(recording)
            with self.assertRaisesRegex(SourceHistoryError, "exactly one"):
                build_source_history_prefill(
                    recording, start_policy_seq=1010, start_policy_offset=10
                )


if __name__ == "__main__":
    unittest.main()
