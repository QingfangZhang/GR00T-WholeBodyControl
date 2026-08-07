"""Regression checks for the pinned Teleopit task-rollout adapter."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from change_ckpt_track.task_sim_io import (  # noqa: E402
    CsvTimeline,
    stage_recording_snapshot,
)
from Teleopit_rollout.constants import ACTION_SCALE, DEFAULT_DOF_POS  # noqa: E402
from Teleopit_rollout.launch_teleopit_rollout import (  # noqa: E402
    DeterministicRollout,
)
from Teleopit_rollout.reference_data import load_prepared_reference  # noqa: E402
from Teleopit_rollout.task_simulator import TaskSceneController  # noqa: E402
from Teleopit_rollout.teleopit_policy import (  # noqa: E402
    TeleopitObservationBuilder,
    TeleopitOnnxPolicy,
)


TRASH_RECORDING = (
    REPO_ROOT / "sample_data/ztj/20260612/20260612_144127_g1_sim"
)
DRAWER_RECORDING = (
    REPO_ROOT / "sample_data/ztj/20260612/20260720_144342_g1_sim"
)
ROBOT_XML = SCRIPT_DIR / "assets/robot_assets/unitree_g1/g1_29dof.xml"
CHECKPOINT = SCRIPT_DIR / "assets/checkpoints/track_g1.onnx"


@unittest.skipUnless(
    TRASH_RECORDING.is_dir() and ROBOT_XML.is_file() and CHECKPOINT.is_file(),
    "real recordings and pinned Teleopit assets are required",
)
class AdapterRegressionTest(unittest.TestCase):
    def _initial_fixture(self):
        reference = load_prepared_reference(TRASH_RECORDING, policy_count=2)
        timeline = CsvTimeline(
            TRASH_RECORDING / "data.csv",
            policy_seq=None,
            row_index=reference.first_source_row_index,
            policy_offset=None,
        )
        staged = stage_recording_snapshot(TRASH_RECORDING, None)
        self.addCleanup(timeline.close)
        self.addCleanup(staged.close)
        scene = TaskSceneController(staged.scene_path)
        scene.initialize(*timeline.state())
        builder = TeleopitObservationBuilder(ROBOT_XML)
        return reference, scene, builder

    def test_reference_boundaries_and_layout(self) -> None:
        trash = load_prepared_reference(TRASH_RECORDING, policy_count=3)
        drawer = load_prepared_reference(DRAWER_RECORDING, policy_count=3)
        self.assertEqual(trash.qpos36.shape, (3, 36))
        self.assertEqual(drawer.qpos36.shape, (3, 36))
        np.testing.assert_array_equal(trash.policy_seq, [20163, 20164, 20165])
        np.testing.assert_array_equal(trash.source_row_index, [3, 11, 19])
        np.testing.assert_array_equal(drawer.policy_seq, [31116, 31117, 31118])
        # Recorded policy groups are not forced to exactly eight CSV rows;
        # the third boundary is row 17, which is why rollout timing uses the
        # fixed 50 Hz reference clock rather than treating row count as policy time.
        np.testing.assert_array_equal(drawer.source_row_index, [2, 10, 17])
        np.testing.assert_allclose(
            np.linalg.norm(trash.root_quat_wxyz, axis=1), 1.0, atol=2e-6
        )

    def test_first_observation_and_action_golden(self) -> None:
        reference, scene, builder = self._initial_fixture()
        features = builder.reference_features(reference.qpos36[0], None)
        observation = builder.build(
            scene.robot_state(), features, np.zeros(29, dtype=np.float32)
        )
        np.testing.assert_allclose(
            observation[:8],
            [
                -0.28053543,
                0.08347935,
                0.25177258,
                0.17281131,
                -0.02598528,
                -0.01561820,
                -0.10737339,
                -0.13261074,
            ],
            rtol=0.0,
            atol=1e-7,
        )
        policy = TeleopitOnnxPolicy(CHECKPOINT)
        raw_action, target, history = policy.infer(observation)
        np.testing.assert_allclose(
            raw_action[:8],
            [
                0.31990710,
                0.41339207,
                0.20451586,
                -0.63909709,
                0.75186408,
                0.12185377,
                0.14079101,
                -0.27867970,
            ],
            rtol=0.0,
            atol=2e-6,
        )
        np.testing.assert_allclose(
            target,
            DEFAULT_DOF_POS + ACTION_SCALE * np.clip(raw_action, -10.0, 10.0),
            rtol=0.0,
            atol=1e-7,
        )
        np.testing.assert_allclose(history, np.repeat(observation[None, :], 10, axis=0))

    def test_history_and_task_joint_mapping(self) -> None:
        reference, scene, builder = self._initial_fixture()
        first_features = builder.reference_features(reference.qpos36[0], None)
        first = builder.build(
            scene.robot_state(), first_features, np.zeros(29, dtype=np.float32)
        )
        second_features = builder.reference_features(
            reference.qpos36[1], reference.qpos36[0]
        )
        second = builder.build(
            scene.robot_state(), second_features, np.zeros(29, dtype=np.float32)
        )
        policy = TeleopitOnnxPolicy(CHECKPOINT)
        policy.infer(first)
        _, body_target, history = policy.infer(second)
        np.testing.assert_allclose(history[:9], np.repeat(first[None, :], 9, axis=0))
        np.testing.assert_allclose(history[9], second)

        scene.set_policy_command(
            body_target,
            reference.left_hand_target[1],
            reference.right_hand_target[1],
        )
        command = scene.update_pd_command()
        self.assertEqual(command["received_dof_pos"].shape, (43,))
        self.assertEqual(command["left_hand_q"].shape, (7,))
        self.assertEqual(len(set(scene.robot_actuator_ids.tolist())), 43)
        np.testing.assert_array_equal(
            scene.model.jnt_qposadr[scene.robot_joint_ids_xml_order],
            np.arange(7, 50),
        )
        self.assertTrue(np.all(np.isfinite(scene.data.ctrl)))

    def test_sliced_post_rollout_repeats_policy_and_holds_source(self) -> None:
        reference = load_prepared_reference(TRASH_RECORDING, policy_count=2)
        timeline = CsvTimeline(
            TRASH_RECORDING / "data.csv",
            policy_seq=None,
            row_index=reference.first_source_row_index,
            policy_offset=None,
        )
        staged = stage_recording_snapshot(TRASH_RECORDING, None)
        self.addCleanup(timeline.close)
        self.addCleanup(staged.close)
        scene = TaskSceneController(staged.scene_path, root_assist="xy")
        scene.initialize(*timeline.state())

        class CaptureWriter:
            def __init__(self) -> None:
                self.source_rows: list[int] = []

            def write(self, **values) -> None:
                self.source_rows.append(int(values["source_row_index"]))

        writer = CaptureWriter()
        runner = DeterministicRollout(
            args=SimpleNamespace(
                post_rollout_seconds=0.04,
                viewer=False,
                fall_height=0.2,
                stop_on_fall=True,
                pace_real_time=False,
            ),
            reference=reference,
            timeline=timeline,
            scene=scene,
            observation_builder=TeleopitObservationBuilder(ROBOT_XML),
            policy=TeleopitOnnxPolicy(CHECKPOINT),
            writer=writer,
        )
        result = runner.run()
        self.assertEqual(result.samples, 33)
        self.assertEqual(result.policy_ticks, 4)
        self.assertEqual(runner.telemetry.reference_frame_index, [0, 1, 1, 1])
        self.assertEqual(
            runner.telemetry.reference_is_hold, [False, False, True, True]
        )
        self.assertEqual(len(set(writer.source_rows[16:])), 1)


if __name__ == "__main__":
    unittest.main()
