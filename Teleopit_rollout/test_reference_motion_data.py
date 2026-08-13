"""Regression tests for the SONIC-reference Teleopit adapter."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from change_ckpt.reference_data import load_reference_sequence
from change_ckpt_track.qpos_reference_data import load_qpos_reference
from Teleopit_rollout.launch_teleopit_rollout import (
    _default_run_name,
    build_parser,
)
from Teleopit_rollout.reference_data import _joint_reorder_indices
from Teleopit_rollout.reference_motion_data import (
    load_reference_motion_prepared_reference,
)
from Teleopit_rollout.task_simulator import TaskSceneController
from Teleopit_rollout.teleopit_policy import (
    TeleopitObservationBuilder,
    TeleopitOnnxPolicy,
)
from change_ckpt_track.task_sim_io import CsvTimeline, stage_recording_snapshot


REPO_ROOT = Path(__file__).resolve().parent.parent
TRASH_RECORDING = (
    REPO_ROOT / "sample_data/ztj/20260612/20260612_144127_g1_sim"
)
THIRD_RECORDING = (
    REPO_ROOT / "sample_data/ztj/20260612/20260722_154958_g1_sim"
)
ROBOT_XML = (
    REPO_ROOT
    / "Teleopit_rollout/assets/robot_assets/unitree_g1/g1_29dof.xml"
)
CHECKPOINT = REPO_ROOT / "Teleopit_rollout/assets/checkpoints/track_g1.onnx"


class ReferenceSourceCliTest(unittest.TestCase):
    def test_default_remains_qpos_and_names_are_disjoint(self) -> None:
        parser = build_parser()
        self.assertEqual(parser.parse_args(["recording"]).reference_source, "qpos")
        hybrid = parser.parse_args(
            ["recording", "--reference-source", "sonic_reference_hybrid"]
        )
        self.assertEqual(hybrid.reference_source, "sonic_reference_hybrid")
        recording = Path("take_g1_sim")
        self.assertEqual(
            _default_run_name(recording, "none"),
            "take_g1_sim_teleopit_no_root_assist",
        )
        self.assertEqual(
            _default_run_name(recording, "xy", "sonic_reference_hybrid"),
            "take_g1_sim_teleopit_sonic_reference_hybrid_root_assist_xy",
        )


@unittest.skipUnless(
    TRASH_RECORDING.is_dir() and ROBOT_XML.is_file() and CHECKPOINT.is_file(),
    "real recordings and pinned Teleopit assets are required",
)
class ReferenceMotionAdapterTest(unittest.TestCase):
    def test_hybrid_components_align_by_policy_seq(self) -> None:
        hybrid = load_reference_motion_prepared_reference(
            TRASH_RECORDING, policy_count=3
        )
        qpos = load_qpos_reference(TRASH_RECORDING).slice(0, 3)
        sonic = load_reference_sequence(TRASH_RECORDING)
        indices = np.searchsorted(sonic.policy_seq, hybrid.policy_seq)
        reorder = _joint_reorder_indices(qpos.joint_names)

        np.testing.assert_array_equal(hybrid.policy_seq, [20163, 20164, 20165])
        np.testing.assert_array_equal(hybrid.source_row_index, [3, 11, 19])
        np.testing.assert_allclose(hybrid.root_pos, qpos.root_pos, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            hybrid.root_quat_wxyz,
            sonic.reference_anchor_quat_wxyz[indices],
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            hybrid.joint_pos,
            sonic.joint_pos[indices][:, reorder],
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            hybrid.source_reference_joint_vel,
            sonic.joint_vel[indices][:, reorder],
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            hybrid.recorded_joint_vel,
            qpos.joint_vel[:, reorder],
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(hybrid.left_hand_target, qpos.left_hand_target)
        np.testing.assert_allclose(hybrid.right_hand_target, qpos.right_hand_target)
        self.assertGreater(
            float(np.sqrt(np.mean((hybrid.joint_pos - qpos.joint_pos[:, reorder]) ** 2))),
            0.05,
        )
        metadata = hybrid.metadata()
        self.assertEqual(metadata["reference_source"], "sonic_reference_hybrid")
        self.assertEqual(metadata["orientation_base_sample_mode"], "previous-index5")
        self.assertEqual(metadata["policy_alignment_max_time_error_s"], 0.0)
        self.assertEqual(metadata["policy_alignment_max_hand_error"], 0.0)

    def test_offset_is_relative_to_trimmed_qpos_sequence(self) -> None:
        full = load_reference_motion_prepared_reference(THIRD_RECORDING)
        selected = load_reference_motion_prepared_reference(
            THIRD_RECORDING, policy_offset=1, policy_count=2
        )
        np.testing.assert_array_equal(selected.policy_seq, full.policy_seq[1:3])
        np.testing.assert_array_equal(
            selected.source_row_index, full.source_row_index[1:3]
        )
        self.assertEqual(selected.policy_offset, 1)

    def test_npz_and_first_policy_inference_are_finite(self) -> None:
        reference = load_reference_motion_prepared_reference(
            TRASH_RECORDING, policy_count=2
        )
        with tempfile.TemporaryDirectory() as directory:
            path = reference.save_npz(Path(directory) / "prepared_reference.npz")
            saved = np.load(path, allow_pickle=False)
            self.assertIn("source_reference_joint_vel", saved.files)
            metadata = json.loads(str(saved["metadata_json"]))
            self.assertEqual(metadata["reference_source"], "sonic_reference_hybrid")

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
        features = builder.reference_features(reference.qpos36[0], None)
        observation = builder.build(
            scene.robot_state(), features, np.zeros(29, dtype=np.float32)
        )
        policy = TeleopitOnnxPolicy(CHECKPOINT)
        raw_action, target, history = policy.infer(observation)
        self.assertEqual(observation.shape, (167,))
        self.assertEqual(history.shape, (10, 167))
        self.assertEqual(raw_action.shape, (29,))
        self.assertEqual(target.shape, (29,))
        self.assertTrue(np.all(np.isfinite(observation)))
        self.assertTrue(np.all(np.isfinite(raw_action)))
        self.assertTrue(np.all(np.isfinite(target)))


if __name__ == "__main__":
    unittest.main()
