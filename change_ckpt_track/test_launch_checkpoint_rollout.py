from __future__ import annotations

import contextlib
import io
from pathlib import Path
import tempfile
import unittest

from change_ckpt_track import launch_checkpoint_rollout as launcher


class QposTrackLauncherTest(unittest.TestCase):
    def test_deploy_ready_requires_complete_object_not_encoder_setup(self) -> None:
        self.assertFalse(
            launcher._matches_deploy_ready_line(
                "Initializing encoder observations..."
            )
        )
        self.assertFalse(
            launcher._matches_deploy_ready_line(
                "Initialized 12 encoder observations (total dim: 1247)"
            )
        )
        self.assertTrue(
            launcher._matches_deploy_ready_line(
                "[DEBUG] G1Deploy object created successfully!"
            )
        )

    def test_qpos_track_defaults(self) -> None:
        args = launcher.build_parser().parse_args(["run", "--dry-run"])
        self.assertEqual(args.start_policy_offset, 0)
        self.assertTrue(args.drop_truncated_edges)
        self.assertFalse(args.heading_correction)
        self.assertEqual(args.chunk_size, 100)
        self.assertEqual(args.lookahead, 100)
        self.assertEqual(args.regular_future_window, "canonical")
        self.assertEqual(args.root_assist, "none")
        self.assertEqual(
            Path(args.recording).name, "20260612_144117_g1_sim"
        )
        self.assertEqual(
            Path(args.output_root).resolve(),
            (launcher.REPO_ROOT / "change_ckpt_track/data").resolve(),
        )

    def test_model_defaults_are_change_ckpt_files(self) -> None:
        regular_args = launcher.build_parser().parse_args(
            ["preflight", "--checkpoint", "regular", "--models-only"]
        )
        regular = launcher.resolve_model_files(regular_args)
        self.assertEqual(regular.expected_encoder_input, 1751)
        self.assertEqual(
            regular.encoder,
            (
                launcher.REPO_ROOT
                / "change_ckpt/models/regular/model_encoder.onnx"
            ).resolve(),
        )
        self.assertEqual(
            regular.obs_config,
            (
                launcher.REPO_ROOT
                / "change_ckpt/observation_config_sonic_release.yaml"
            ).resolve(),
        )
        low_args = launcher.build_parser().parse_args(
            ["preflight", "--checkpoint", "low_latency", "--models-only"]
        )
        low = launcher.resolve_model_files(low_args)
        self.assertEqual(low.expected_encoder_input, 1247)
        self.assertEqual(
            low.encoder,
            (
                launcher.REPO_ROOT
                / "change_ckpt/models/low_latency/model_encoder.onnx"
            ).resolve(),
        )

        v11_args = launcher.build_parser().parse_args(
            ["preflight", "--checkpoint", "sonic_v1_1", "--models-only"]
        )
        v11 = launcher.resolve_model_files(v11_args)
        self.assertEqual(v11.name, "sonic_v1_1")
        self.assertEqual(v11.expected_encoder_input, 1751)
        self.assertEqual(v11.expected_decoder_input, 994)
        self.assertEqual(v11.expected_token_dim, 64)
        self.assertEqual(v11.expected_action_dim, 29)
        self.assertEqual(
            v11.encoder,
            (
                launcher.REPO_ROOT
                / "change_ckpt/models/v1.1/model_encoder.onnx"
            ).resolve(),
        )
        self.assertEqual(
            v11.decoder,
            (
                launcher.REPO_ROOT
                / "change_ckpt/models/v1.1/model_decoder.onnx"
            ).resolve(),
        )
        self.assertEqual(
            v11.obs_config,
            (
                launcher.REPO_ROOT
                / "change_ckpt/models/v1.1/observation_config.yaml"
            ).resolve(),
        )

    def test_sonic_v11_aliases_use_canonical_regular_publisher_layout(self) -> None:
        parser = launcher.build_parser()
        for alias in ("sonic_v1_1", "sonic-v1-1", "v1.1", "v1_1"):
            with self.subTest(alias=alias):
                args = parser.parse_args(
                    ["run", "--checkpoint", alias, "--dry-run"]
                )
                self.assertEqual(
                    launcher.resolve_model_files(args).name,
                    "sonic_v1_1",
                )
                command = launcher.publisher_command(
                    args, Path("/tmp/qpos-track-v11-test")
                )
                layout_index = command.index("--checkpoint-layout")
                self.assertEqual(command[layout_index + 1], "sonic_v1_1")
                window_index = command.index("--regular-future-window")
                self.assertEqual(command[window_index + 1], "canonical")
                self.assertEqual(
                    launcher._required_future_frames(alias), 46
                )

    def test_sonic_v11_heading_observation_dimensions(self) -> None:
        expected = {
            "motion_anchor_orientation_heading_10frame_step5": 60,
            "motion_anchor_orientation_heading_10frame_step1": 60,
            "motion_anchor_orientation_heading": 6,
            "smpl_anchor_orientation_heading_10frame_step1": 60,
        }
        for name, dimension in expected.items():
            with self.subTest(name=name):
                self.assertEqual(
                    launcher.ENCODER_OBSERVATION_DIMS[name], dimension
                )

    def test_sonic_v11_rejects_recorded_future_window(self) -> None:
        args = launcher.build_parser().parse_args(
            [
                "run",
                "--checkpoint",
                "sonic_v1_1",
                "--regular-future-window",
                "recorded",
                "--dry-run",
            ]
        )
        model = launcher.resolve_model_files(args)
        with self.assertRaisesRegex(
            launcher.PreflightError,
            "valid only with --checkpoint regular",
        ):
            launcher._validate_launch_settings(args, model, {})

    def test_simulator_receives_actual_policy_seq_after_edge_trim(self) -> None:
        recording = (
            launcher.REPO_ROOT
            / "sample_data/ztj/20260612/20260612_144127_g1_sim"
        )
        args = launcher.build_parser().parse_args(
            ["run", "--recording", str(recording), "--dry-run"]
        )
        command = launcher.simulator_command(args, Path("/tmp/qpos-track-test"))
        index = command.index("--policy-seq")
        self.assertEqual(command[index + 1], "20163")
        self.assertNotIn("--policy-offset", command)

        publisher = launcher.publisher_command(
            args, Path("/tmp/qpos-track-test")
        )
        offset_index = publisher.index("--start-policy-offset")
        self.assertEqual(publisher[offset_index + 1], "0")
        self.assertIn("--drop-truncated-edges", publisher)
        timing_index = publisher.index("--regular-future-window")
        self.assertEqual(publisher[timing_index + 1], "canonical")

    def test_recorded_window_has_separate_stable_output_name(self) -> None:
        recording = (
            launcher.REPO_ROOT
            / "sample_data/ztj/20260612/20260612_144127_g1_sim"
        )
        with tempfile.TemporaryDirectory() as temporary:
            args = launcher.build_parser().parse_args(
                [
                    "run",
                    "--checkpoint",
                    "regular",
                    "--recording",
                    str(recording),
                    "--output-root",
                    temporary,
                    "--regular-future-window",
                    "recorded",
                    "--dry-run",
                ]
            )
            run_dir = launcher._new_run_dir(args, "regular")
            self.assertEqual(
                run_dir.name,
                "20260612_144127_g1_sim_regular_recorded_lags",
            )
            self.assertTrue(run_dir.is_dir())

            publisher = launcher.publisher_command(args, run_dir)
            timing_index = publisher.index("--regular-future-window")
            self.assertEqual(publisher[timing_index + 1], "recorded")

    def test_root_assist_parser_accepts_only_supported_modes(self) -> None:
        parser = launcher.build_parser()
        for mode in ("none", "xy", "xyz"):
            with self.subTest(mode=mode):
                args = parser.parse_args(
                    ["run", "--root-assist", mode, "--dry-run"]
                )
                self.assertEqual(args.root_assist, mode)

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(
                    ["run", "--root-assist", "yaw", "--dry-run"]
                )

    def test_simulator_forwards_root_assist_once(self) -> None:
        recording = (
            launcher.REPO_ROOT
            / "sample_data/ztj/20260612/20260720_144342_g1_sim"
        )
        args = launcher.build_parser().parse_args(
            [
                "run",
                "--recording",
                str(recording),
                "--root-assist",
                "xy",
                "--dry-run",
            ]
        )

        command = launcher.simulator_command(
            args, Path("/tmp/qpos-track-root-assist-test")
        )

        option_index = command.index("--root-assist")
        self.assertEqual(command[option_index + 1], "xy")
        self.assertEqual(command.count("--root-assist"), 1)

    def test_sim_arg_cannot_override_root_assist(self) -> None:
        for override in ("--root-assist", "--root-assist=xyz"):
            with self.subTest(override=override):
                args = launcher.build_parser().parse_args(
                    ["run", "--dry-run"]
                )
                args.sim_arg = [override]
                with self.assertRaisesRegex(
                    launcher.PreflightError,
                    "may not override",
                ):
                    launcher.simulator_command(
                        args,
                        Path("/tmp/qpos-track-root-assist-test"),
                    )

    def test_assisted_output_names_do_not_replace_baseline(self) -> None:
        recording = (
            launcher.REPO_ROOT
            / "sample_data/ztj/20260612/20260720_144342_g1_sim"
        )
        with tempfile.TemporaryDirectory() as temporary:
            parser = launcher.build_parser()
            baseline_args = parser.parse_args(
                [
                    "run",
                    "--checkpoint",
                    "regular",
                    "--recording",
                    str(recording),
                    "--output-root",
                    temporary,
                    "--dry-run",
                ]
            )
            baseline = launcher._new_run_dir(baseline_args, "regular")
            marker = baseline / "baseline.txt"
            marker.write_text("keep baseline", encoding="utf-8")

            expected_names = {
                "canonical": (
                    "20260720_144342_g1_sim_regular_root_assist_xy"
                ),
                "recorded": (
                    "20260720_144342_g1_sim_regular_recorded_lags_"
                    "root_assist_xy"
                ),
            }
            assisted_dirs = []
            for window, expected_name in expected_names.items():
                with self.subTest(window=window):
                    assisted_args = parser.parse_args(
                        [
                            "run",
                            "--checkpoint",
                            "regular",
                            "--recording",
                            str(recording),
                            "--output-root",
                            temporary,
                            "--regular-future-window",
                            window,
                            "--root-assist",
                            "xy",
                            "--dry-run",
                        ]
                    )
                    assisted = launcher._new_run_dir(
                        assisted_args, "regular"
                    )
                    assisted_dirs.append(assisted)
                    self.assertEqual(assisted.name, expected_name)
                    self.assertNotEqual(assisted, baseline)
                    self.assertTrue(marker.is_file())

            self.assertNotEqual(assisted_dirs[0], assisted_dirs[1])


if __name__ == "__main__":
    unittest.main()
