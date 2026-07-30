#!/usr/bin/env python3
"""Tests for launcher argument forwarding and deterministic output naming."""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import unittest

from change_ckpt import launch_checkpoint_rollout as launcher


class RunDirectoryNamingTest(unittest.TestCase):
    def test_recording_name_and_overwrite(self) -> None:
        with tempfile.TemporaryDirectory(prefix="change_ckpt_naming_") as temporary:
            temporary_path = Path(temporary)
            recording = temporary_path / "original_recording"
            recording.mkdir()
            recording_csv = recording / "data.csv"
            recording_csv.write_text("time\n0\n", encoding="utf-8")
            output_root = temporary_path / "outputs"

            args = argparse.Namespace(
                output_root=str(output_root),
                recording=str(recording),
            )
            first = launcher._new_run_dir(args, "regular")
            self.assertEqual(first.name, "original_recording_regular")
            marker = first / "old_result.txt"
            marker.write_text("must be replaced", encoding="utf-8")

            # Passing data.csv directly must derive the same recording name.
            args.recording = str(recording_csv)
            second = launcher._new_run_dir(args, "regular")
            self.assertEqual(second, first)
            self.assertTrue(second.is_dir())
            self.assertFalse(marker.exists())

    def test_low_latency_suffix(self) -> None:
        with tempfile.TemporaryDirectory(prefix="change_ckpt_naming_") as temporary:
            temporary_path = Path(temporary)
            recording = temporary_path / "take_001"
            recording.mkdir()
            (recording / "data.csv").write_text("time\n0\n", encoding="utf-8")
            args = argparse.Namespace(
                output_root=str(temporary_path / "outputs"),
                recording=str(recording),
            )

            result = launcher._new_run_dir(args, "low_latency")
            self.assertEqual(result.name, "take_001_low_latency")

    def test_root_assisted_output_is_named_separately_from_baseline(self) -> None:
        with tempfile.TemporaryDirectory(prefix="change_ckpt_naming_") as temporary:
            temporary_path = Path(temporary)
            recording = temporary_path / "take_002"
            recording.mkdir()
            (recording / "data.csv").write_text("time\n0\n", encoding="utf-8")
            output_root = temporary_path / "outputs"

            baseline_args = argparse.Namespace(
                output_root=str(output_root),
                recording=str(recording),
                root_assist="none",
            )
            baseline = launcher._new_run_dir(baseline_args, "regular")
            baseline_marker = baseline / "baseline.txt"
            baseline_marker.write_text("keep baseline", encoding="utf-8")

            assisted_args = argparse.Namespace(
                output_root=str(output_root),
                recording=str(recording),
                root_assist="xy",
            )
            assisted = launcher._new_run_dir(assisted_args, "regular")

            self.assertEqual(baseline.name, "take_002_regular")
            self.assertEqual(
                assisted.name,
                "take_002_regular_root_assist_xy",
            )
            self.assertNotEqual(assisted, baseline)
            self.assertTrue(baseline_marker.is_file())

    def test_400hz_output_is_named_separately_from_200hz(self) -> None:
        with tempfile.TemporaryDirectory(prefix="change_ckpt_naming_") as temporary:
            temporary_path = Path(temporary)
            recording = temporary_path / "take_400"
            recording.mkdir()
            (recording / "data.csv").write_text("time\n0\n", encoding="utf-8")
            output_root = temporary_path / "outputs"

            baseline_args = argparse.Namespace(
                output_root=str(output_root),
                recording=str(recording),
                root_assist="xyz",
                sim_frequency=200,
            )
            baseline = launcher._new_run_dir(baseline_args, "regular")
            marker = baseline / "keep_200hz.txt"
            marker.write_text("keep", encoding="utf-8")

            high_rate_args = argparse.Namespace(
                output_root=str(output_root),
                recording=str(recording),
                root_assist="xyz",
                sim_frequency=400,
            )
            high_rate = launcher._new_run_dir(high_rate_args, "regular")

            self.assertEqual(
                baseline.name,
                "take_400_regular_root_assist_xyz",
            )
            self.assertEqual(
                high_rate.name,
                "take_400_regular_400hz_root_assist_xyz",
            )
            self.assertNotEqual(high_rate, baseline)
            self.assertTrue(marker.is_file())


class RootAssistLauncherTest(unittest.TestCase):
    def test_parser_defaults_to_none_and_accepts_supported_modes(self) -> None:
        parser = launcher.build_parser()
        self.assertEqual(parser.parse_args(["run"]).root_assist, "none")

        for mode in ("xy", "xyz"):
            with self.subTest(mode=mode):
                args = parser.parse_args(["run", "--root-assist", mode])
                self.assertEqual(args.root_assist, mode)

    def test_simulator_command_forwards_root_assist(self) -> None:
        args = launcher.build_parser().parse_args(
            ["run", "--root-assist", "xy"]
        )
        command = launcher.simulator_command(args, Path("/tmp/root_assist_run"))

        option_index = command.index("--root-assist")
        self.assertEqual(command[option_index + 1], "xy")
        self.assertEqual(command.count("--root-assist"), 1)

    def test_sim_arg_cannot_override_root_assist(self) -> None:
        for override in ("--root-assist", "--root-assist=xyz"):
            with self.subTest(override=override):
                args = launcher.build_parser().parse_args(["run"])
                args.sim_arg = [override]
                with self.assertRaisesRegex(
                    launcher.PreflightError,
                    "may not override",
                ):
                    launcher.simulator_command(
                        args,
                        Path("/tmp/root_assist_run"),
                    )


class TimingProfileLauncherTest(unittest.TestCase):
    def test_default_profile_resolves_to_200hz(self) -> None:
        args = launcher.build_parser().parse_args(["run"])
        launcher._resolve_sim_timing(args)

        self.assertEqual(args.sim_frequency, 200)
        self.assertAlmostEqual(args.control_dt, 0.005)
        self.assertAlmostEqual(args.physics_dt, 0.001)
        self.assertAlmostEqual(args.source_dt, 0.0025)

    def test_400hz_profile_resolves_and_forwards_exact_timing(self) -> None:
        args = launcher.build_parser().parse_args(
            ["run", "--sim-frequency", "400"]
        )
        command = launcher.simulator_command(args, Path("/tmp/high_rate_run"))

        self.assertAlmostEqual(args.control_dt, 0.0025)
        self.assertAlmostEqual(args.physics_dt, 0.0005)
        self.assertAlmostEqual(args.source_dt, 0.0025)
        expected = {
            "--control-dt": "0.0025",
            "--physics-dt": "0.0005",
            "--source-dt": "0.0025",
        }
        for option, value in expected.items():
            with self.subTest(option=option):
                self.assertEqual(command.count(option), 1)
                self.assertEqual(command[command.index(option) + 1], value)

    def test_control_dt_must_match_selected_frequency(self) -> None:
        args = launcher.build_parser().parse_args(
            [
                "run",
                "--sim-frequency",
                "400",
                "--control-dt",
                "0.005",
            ]
        )
        with self.assertRaisesRegex(
            launcher.PreflightError,
            "requires --control-dt 0.0025",
        ):
            launcher._resolve_sim_timing(args)


if __name__ == "__main__":
    unittest.main()
