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

    def test_sonic_v1_1_suffix(self) -> None:
        with tempfile.TemporaryDirectory(prefix="change_ckpt_naming_") as temporary:
            temporary_path = Path(temporary)
            recording = temporary_path / "take_v11"
            recording.mkdir()
            (recording / "data.csv").write_text("time\n0\n", encoding="utf-8")
            args = argparse.Namespace(
                output_root=str(temporary_path / "outputs"),
                recording=str(recording),
                root_assist="xy",
            )

            result = launcher._new_run_dir(args, "sonic_v1_1")
            self.assertEqual(
                result.name,
                "take_v11_sonic_v1_1_root_assist_xy",
            )

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
            ["run", "--root-assist", "xy", "--no-source-history-prefill"]
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


class SourceHistoryLauncherTest(unittest.TestCase):
    def test_prefill_is_default_and_can_be_disabled(self) -> None:
        parser = launcher.build_parser()
        baseline = parser.parse_args(["run"])
        self.assertEqual(baseline.start_policy_offset, 10)
        self.assertTrue(baseline.source_history_prefill)
        self.assertFalse(baseline.source_state_init)
        self.assertEqual(
            launcher._selected_deploy_binary(baseline),
            launcher.SOURCE_HISTORY_DEPLOY_BINARY,
        )

        zero_history = parser.parse_args(
            ["run", "--no-source-history-prefill"]
        )
        self.assertFalse(zero_history.source_history_prefill)
        self.assertEqual(
            launcher._selected_deploy_binary(zero_history),
            launcher.DEPLOY_BINARY,
        )

        state_only = parser.parse_args(
            ["run", "--no-source-history-prefill", "--source-state-init"]
        )
        self.assertTrue(state_only.source_state_init)
        self.assertFalse(state_only.source_history_prefill)
        self.assertEqual(
            launcher._selected_deploy_binary(state_only),
            launcher.SOURCE_HISTORY_DEPLOY_BINARY,
        )

    def test_prefill_deploy_command_forwards_prepared_file(self) -> None:
        args = launcher.build_parser().parse_args(
            [
                "run",
                "--checkpoint",
                "regular",
                "--start-policy-offset",
                "10",
                "--source-history-prefill",
            ]
        )
        args.source_history_prefill_file = Path("/tmp/source_history.json")
        model = launcher.resolve_model_files(args)
        command = launcher.deploy_command(
            args, model, Path("/tmp/source_history_deploy_logs")
        )
        self.assertEqual(command[0], str(launcher.SOURCE_HISTORY_DEPLOY_BINARY))
        self.assertEqual(command.count("--source-history-prefill-file"), 1)
        option = command.index("--source-history-prefill-file")
        self.assertEqual(command[option + 1], "/tmp/source_history.json")

    def test_offset_and_history_mode_are_omitted_from_output_name(self) -> None:
        with tempfile.TemporaryDirectory(prefix="change_ckpt_naming_") as temporary:
            root = Path(temporary)
            recording = root / "take_history"
            recording.mkdir()
            (recording / "data.csv").write_text("time\n0\n", encoding="utf-8")
            common = {
                "output_root": str(root / "outputs"),
                "recording": str(recording),
                "root_assist": "none",
                "start_policy_offset": 10,
            }
            baseline = launcher._new_run_dir(
                argparse.Namespace(**common, source_history_prefill=False),
                "regular",
            )
            marker = baseline / "replaced.txt"
            marker.write_text("old", encoding="utf-8")
            prefilled = launcher._new_run_dir(
                argparse.Namespace(**common, source_history_prefill=True),
                "regular",
            )
            self.assertEqual(baseline.name, "take_history_regular")
            self.assertEqual(prefilled.name, "take_history_regular")
            self.assertEqual(prefilled, baseline)
            self.assertFalse(marker.exists())

            state_only = launcher._new_run_dir(
                argparse.Namespace(
                    **common,
                    source_history_prefill=False,
                    source_state_init=True,
                ),
                "regular",
            )
            self.assertEqual(state_only.name, "take_history_regular")

    def test_public_offset_zero_maps_to_second_raw_group(self) -> None:
        parser = launcher.build_parser()
        for public, raw in ((0, 1), (10, 11)):
            with self.subTest(public=public):
                args = parser.parse_args(
                    [
                        "run",
                        "--start-policy-offset",
                        str(public),
                        "--no-source-history-prefill",
                    ]
                )
                self.assertEqual(launcher._raw_start_policy_offset(args), raw)
                simulator = launcher.simulator_command(
                    args, Path("/tmp/change_ckpt_offset_test")
                )
                self.assertEqual(
                    simulator[simulator.index("--policy-offset") + 1], str(raw)
                )
                publisher = launcher.publisher_command(
                    args, Path("/tmp/change_ckpt_offset_test")
                )
                self.assertEqual(
                    publisher[publisher.index("--start-policy-offset") + 1],
                    str(raw),
                )

    def test_source_state_baseline_uses_matched_row_and_official_deploy(self) -> None:
        args = launcher.build_parser().parse_args(
            [
                "run",
                "--checkpoint",
                "regular",
                "--start-policy-offset",
                "10",
                "--no-source-history-prefill",
                "--source-state-init",
            ]
        )
        args.source_history_initial_state_file = Path("/tmp/source_state.json")
        args.source_history_sim_row_index = 75
        command = launcher.simulator_command(args, Path("/tmp/state_only_run"))
        self.assertEqual(command[command.index("--row-index") + 1], "75")
        self.assertEqual(
            command[command.index("--initial-state-json") + 1],
            "/tmp/source_state.json",
        )
        self.assertNotIn("--policy-offset", command)
        model = launcher.resolve_model_files(args)
        deploy = launcher.deploy_command(
            args, model, Path("/tmp/state_only_deploy_logs")
        )
        self.assertEqual(deploy[0], str(launcher.SOURCE_HISTORY_DEPLOY_BINARY))
        self.assertNotIn("--source-history-prefill-file", deploy)


class SonicV11LauncherTest(unittest.TestCase):
    def test_model_defaults_and_aliases(self) -> None:
        parser = launcher.build_parser()
        for alias in (
            "sonic_v1_1",
            "sonic-v1-1",
            "sonic-v1.1",
            "v1.1",
            "v1_1",
        ):
            with self.subTest(alias=alias):
                args = parser.parse_args(
                    ["preflight", "--checkpoint", alias, "--models-only"]
                )
                model = launcher.resolve_model_files(args)
                self.assertEqual(model.name, "sonic_v1_1")
                self.assertEqual(model.expected_encoder_input, 1751)
                self.assertEqual(model.expected_decoder_input, 994)
                self.assertEqual(model.expected_token_dim, 64)
                self.assertEqual(model.expected_action_dim, 29)
                self.assertEqual(
                    model.encoder,
                    (
                        launcher.REPO_ROOT
                        / "change_ckpt/models/v1.1/model_encoder.onnx"
                    ).resolve(),
                )
                self.assertEqual(
                    model.decoder,
                    (
                        launcher.REPO_ROOT
                        / "change_ckpt/models/v1.1/model_decoder.onnx"
                    ).resolve(),
                )
                self.assertEqual(
                    model.obs_config,
                    (
                        launcher.REPO_ROOT
                        / "change_ckpt/models/v1.1/observation_config.yaml"
                    ).resolve(),
                )

    def test_publisher_command_preserves_explicit_v11_layout(self) -> None:
        args = launcher.build_parser().parse_args(
            ["run", "--checkpoint", "sonic_v1_1", "--dry-run"]
        )
        command = launcher.publisher_command(args, Path("/tmp/change-ckpt-v11-test"))
        index = command.index("--checkpoint-layout")
        self.assertEqual(command[index + 1], "sonic_v1_1")

    def test_heading_observation_dimensions(self) -> None:
        expected = {
            "motion_anchor_orientation_heading_10frame_step5": 60,
            "motion_anchor_orientation_heading_10frame_step1": 60,
            "motion_anchor_orientation_heading": 6,
            "smpl_anchor_orientation_heading_10frame_step1": 60,
        }
        for name, dimension in expected.items():
            with self.subTest(name=name):
                self.assertEqual(
                    launcher.ENCODER_OBSERVATION_DIMS[name],
                    dimension,
                )


class DeployCsvLoggingTest(unittest.TestCase):
    def test_enabled_by_default_and_can_be_disabled(self) -> None:
        parser = launcher.build_parser()
        self.assertTrue(parser.parse_args(["run"]).deploy_csv_logs)
        self.assertFalse(
            parser.parse_args(["run", "--no-deploy-csv-logs"]).deploy_csv_logs
        )

    def test_disabled_logging_removes_only_deploy_csv_arguments(self) -> None:
        args = launcher.build_parser().parse_args(
            [
                "run",
                "--checkpoint",
                "sonic_v1_1",
                "--no-deploy-csv-logs",
                "--no-source-history-prefill",
            ]
        )
        model = launcher.resolve_model_files(args)
        command = launcher.deploy_command(
            args,
            model,
            Path("/tmp/change-ckpt-deploy-csv-test"),
        )
        self.assertNotIn("--enable-csv-logs", command)
        self.assertNotIn("--logs-dir", command)
        self.assertNotIn("--target-motion-logfile", command)
        self.assertIn("--encoder-file", command)
        self.assertIn("--obs-config", command)

    def test_disabled_logging_has_distinct_run_directory(self) -> None:
        with tempfile.TemporaryDirectory(prefix="change_ckpt_naming_") as temporary:
            root = Path(temporary)
            recording = root / "take_timing"
            recording.mkdir()
            (recording / "data.csv").write_text("time\n0\n", encoding="utf-8")
            args = argparse.Namespace(
                output_root=str(root / "outputs"),
                recording=str(recording),
                root_assist="xy",
                deploy_csv_logs=False,
            )
            result = launcher._new_run_dir(args, "sonic_v1_1")
            self.assertEqual(
                result.name,
                "take_timing_sonic_v1_1_no_deploy_csv_root_assist_xy",
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
            ["run", "--sim-frequency", "400", "--no-source-history-prefill"]
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
