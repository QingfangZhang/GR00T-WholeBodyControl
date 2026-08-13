"""Regression checks for the shared deterministic task-scene dynamics."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from change_ckpt_track.task_sim_io import CsvTimeline, stage_recording_snapshot
from controller_replacement.simulator import DeterministicTaskScene
from Teleopit_rollout.constants import KDS, KPS, TORQUE_LIMITS


REPO_ROOT = Path(__file__).resolve().parents[2]
RECORDING = REPO_ROOT / "sample_data/ztj/20260612/20260612_144127_g1_sim"


@unittest.skipUnless(RECORDING.is_dir(), "real task recording is required")
class DeterministicTaskSceneTest(unittest.TestCase):
    def _make_scene(self, root_assist: str):
        temporary = tempfile.TemporaryDirectory(prefix="controller_scene_test_")
        self.addCleanup(temporary.cleanup)
        staged = stage_recording_snapshot(RECORDING, Path(temporary.name))
        timeline = CsvTimeline(
            RECORDING / "data.csv", policy_seq=None, row_index=3, policy_offset=None
        )
        self.addCleanup(timeline.close)
        scene = DeterministicTaskScene(staged.scene_path, root_assist=root_assist)
        qpos, qvel = timeline.state()
        scene.initialize(qpos, qvel)
        return scene, timeline

    def test_native_pd_and_400hz_physics_interval(self) -> None:
        scene, _ = self._make_scene("none")
        state = scene.state()
        scene.set_controller_command(
            q_target=state.joint_pos + 0.01,
            kp=KPS,
            kd=KDS,
            torque_limit=TORQUE_LIMITS,
            left_hand_target=state.left_hand_pos,
            right_hand_target=state.right_hand_pos,
        )
        command = scene.update_pd()
        self.assertEqual(command.body_torque.shape, (29,))
        self.assertEqual(command.body_torque_saturation.shape, (29,))
        scene.physics_step(5)
        self.assertAlmostEqual(float(scene.data.time), 0.0025, places=12)
        scene.validate_state()
        contacts = scene.contact_summary()
        self.assertGreaterEqual(contacts.total_contacts, 0)
        self.assertGreaterEqual(contacts.robot_normal_force_max_n, 0.0)

    def test_xy_is_the_only_assisted_mode(self) -> None:
        scene, timeline = self._make_scene("xy")
        source_qpos, source_qvel = timeline.state()
        scene.data.qpos[0:2] += np.asarray([0.1, -0.2])
        scene.apply_root_assist(
            source_qpos, source_qvel, source_velocity_active=True
        )
        np.testing.assert_allclose(scene.data.qpos[:2], source_qpos[:2])
        self.assertEqual(scene.root_assist.ticks, 1)
        with self.assertRaisesRegex(ValueError, "none.*xy"):
            DeterministicTaskScene(scene.scene_xml, root_assist="xyz")

    def test_received_state_is_actual_43_joint_snapshot(self) -> None:
        scene, _ = self._make_scene("none")
        state = scene.state()
        self.assertEqual(state.received_dof_pos.shape, (43,))
        np.testing.assert_allclose(
            state.received_dof_pos,
            np.concatenate(
                (
                    scene.body_joint_pos,
                    scene.left_hand_joint_pos,
                    scene.right_hand_joint_pos,
                )
            ),
            rtol=0.0,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            state.xml_robot_joint_pos,
            scene.data.qpos[7:50],
            rtol=0.0,
            atol=1e-7,
        )


if __name__ == "__main__":
    unittest.main()
