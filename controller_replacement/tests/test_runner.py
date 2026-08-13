from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from controller_replacement.runner import (
    LOGS_PER_PD,
    LOGS_PER_POLICY,
    PHYSICS_STEPS_PER_LOG,
    RolloutError,
    _OWNED_OUTPUT_MARKER,
    _validate_owned_output,
    _remove_owned_output,
)


class RunnerContractTest(unittest.TestCase):
    def test_fixed_rate_integer_ratios(self) -> None:
        self.assertEqual(PHYSICS_STEPS_PER_LOG, 5)
        self.assertEqual(LOGS_PER_PD, 2)
        self.assertEqual(LOGS_PER_POLICY, 8)

    def test_output_overwrite_refuses_unowned_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result"
            output.mkdir()
            (output / "user.txt").write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(RolloutError, "unmarked"):
                _remove_owned_output(output)
            self.assertTrue((output / "user.txt").is_file())

    def test_output_overwrite_accepts_own_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result"
            output.mkdir()
            (output / _OWNED_OUTPUT_MARKER).write_text("1\n", encoding="utf-8")
            _remove_owned_output(output)
            self.assertFalse(output.exists())

    def test_output_validation_keeps_previous_complete_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result"
            output.mkdir()
            (output / _OWNED_OUTPUT_MARKER).write_text("1\n", encoding="utf-8")
            previous = output / "run_complete.json"
            previous.write_text("{}\n", encoding="utf-8")
            _validate_owned_output(output)
            self.assertTrue(previous.is_file())


if __name__ == "__main__":
    unittest.main()
