"""Regression checks for the shared deterministic task-scene dynamics."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from change_ckpt_track.task_sim_io import CsvTimeline, stage_recording_snapshot
from controller_replacement.hand_control import (
    HandTorqueProfile,
    SONIC_RELEASE_HAND_TORQUE_LIMIT_NM,
)
from controller_replacement.simulator import DeterministicTaskScene
from Teleopit_rollout.constants import KDS, KPS, TORQUE_LIMITS


REPO_ROOT = Path(__file__).resolve().parents[2]
RECORDING = REPO_ROOT / "sample_data/ztj/20260612/20260612_144127_g1_sim"


@unittest.skipUnless(RECORDING.is_dir(), "real task recording is required")
class DeterministicTaskSceneTest(unittest.TestCase):
    def _make_scene(
        self,
        root_assist: str,
        hand_torque_profile: HandTorqueProfile | str = (
            HandTorqueProfile.SONIC_RELEASE
        ),
    ):
        temporary = tempfile.TemporaryDirectory(prefix="controller_scene_test_")
        self.addCleanup(temporary.cleanup)
        staged = stage_recording_snapshot(RECORDING, Path(temporary.name))
        timeline = CsvTimeline(
            RECORDING / "data.csv", policy_seq=None, row_index=3, policy_offset=None
        )
        self.addCleanup(timeline.close)
        scene = DeterministicTaskScene(
            staged.scene_path,
            root_assist=root_assist,
            hand_torque_profile=hand_torque_profile,
        )
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

    def test_state_exposes_distinct_controller_native_angular_velocities(self) -> None:
        scene, _ = self._make_scene("none")
        scene.data.qvel[3:6] = np.asarray([0.31, -0.27, 0.19])
        mujoco.mj_forward(scene.model, scene.data)
        state = scene.state()
        expected_pelvis_local = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            scene.model,
            scene.data,
            mujoco.mjtObj.mjOBJ_BODY,
            scene.pelvis_body_id,
            expected_pelvis_local,
            1,
        )
        np.testing.assert_array_equal(
            state.sonic_base_ang_vel_qvel,
            scene.data.qvel[3:6].astype(np.float32),
        )
        np.testing.assert_array_equal(
            state.teleopit_pelvis_ang_vel_b,
            expected_pelvis_local[:3].astype(np.float32),
        )

    def test_scene_actuator_limit_applies_without_controller_cap(self) -> None:
        scene, _ = self._make_scene("none")
        state = scene.state()
        target = state.joint_pos.copy()
        # left_ankle_pitch is index 4 in the canonical 29-DoF order.  The
        # recording's staged XML gives this actuator a +/-50 Nm ctrlrange.
        target[4] += 100.0
        scene.set_controller_command(
            q_target=target,
            kp=np.ones(29),
            kd=np.zeros(29),
            torque_limit=None,
            left_hand_target=state.left_hand_pos,
            right_hand_target=state.right_hand_pos,
        )
        command = scene.update_pd()
        self.assertIsNone(command.body_torque_limit)
        self.assertAlmostEqual(float(command.body_torque[4]), 50.0, places=12)
        self.assertTrue(bool(command.body_torque_saturation[4]))

    def test_controller_cap_is_still_intersected_with_scene_limit(self) -> None:
        scene, _ = self._make_scene("none")
        state = scene.state()
        target = state.joint_pos.copy()
        target[4] += 100.0
        scene.set_controller_command(
            q_target=target,
            kp=np.ones(29),
            kd=np.zeros(29),
            torque_limit=np.full(29, 25.0),
            left_hand_target=state.left_hand_pos,
            right_hand_target=state.right_hand_pos,
        )
        command = scene.update_pd()
        self.assertAlmostEqual(float(command.body_torque[4]), 25.0, places=12)
        self.assertTrue(bool(command.body_torque_saturation[4]))

    def test_sonic_release_hand_profile_precedes_xml_limit(self) -> None:
        scene, _ = self._make_scene("none")
        state = scene.state()
        scene.set_controller_command(
            q_target=state.joint_pos,
            kp=np.zeros(29),
            kd=np.zeros(29),
            torque_limit=None,
            left_hand_target=state.left_hand_pos,
            right_hand_target=state.right_hand_pos,
        )
        # Joint 1 has a +/-1.4 Nm staged-XML range.  Damping generates +/-1
        # Nm, so only the formal SONIC release profile should clip it to 0.7.
        scene.data.qvel[scene.left_hand_dof_addresses[1]] = -10.0
        scene.data.qvel[scene.right_hand_dof_addresses[1]] = 10.0
        command = scene.update_pd()
        self.assertAlmostEqual(float(command.left_hand_torque[1]), 0.7)
        self.assertAlmostEqual(float(command.right_hand_torque[1]), -0.7)
        self.assertTrue(bool(command.left_hand_torque_saturation[1]))
        self.assertTrue(bool(command.right_hand_torque_saturation[1]))
        metadata = scene.hand_control_metadata()
        self.assertEqual(metadata["profile"], "sonic_release")
        self.assertEqual(
            metadata["software_torque_limit_nm"],
            SONIC_RELEASE_HAND_TORQUE_LIMIT_NM.tolist(),
        )
        self.assertEqual(
            metadata["torque_limit_application_order"],
            ["hand_torque_profile", "staged_mujoco_actuator_ctrlrange"],
        )
        self.assertEqual(metadata["sides"]["left"]["xml_ctrlrange_upper_nm"][1], 1.4)
        self.assertEqual(metadata["sides"]["left"]["effective_upper_nm"][1], 0.7)

    def test_staged_xml_profile_retains_recording_hand_range(self) -> None:
        scene, _ = self._make_scene("none", HandTorqueProfile.STAGED_XML)
        state = scene.state()
        scene.set_controller_command(
            q_target=state.joint_pos,
            kp=np.zeros(29),
            kd=np.zeros(29),
            torque_limit=None,
            left_hand_target=state.left_hand_pos,
            right_hand_target=state.right_hand_pos,
        )
        scene.data.qvel[scene.left_hand_dof_addresses[1]] = -10.0
        command = scene.update_pd()
        self.assertAlmostEqual(float(command.left_hand_torque[1]), 1.0)
        self.assertFalse(bool(command.left_hand_torque_saturation[1]))
        metadata = scene.hand_control_metadata()
        self.assertEqual(metadata["profile"], "staged_xml")
        self.assertIsNone(metadata["software_torque_limit_nm"])
        self.assertEqual(metadata["sides"]["left"]["effective_upper_nm"][1], 1.4)

    def test_xy_is_the_only_assisted_mode(self) -> None:
        scene, timeline = self._make_scene("xy")
        source_qpos, source_qvel = timeline.state()
        source_qvel = source_qvel.copy()
        source_qvel[:2] = np.asarray([0.3, -0.4])
        scene.data.qpos[0:2] += np.asarray([0.1, -0.2])
        scene.apply_root_assist(
            source_qpos, source_qvel, source_velocity_active=True
        )
        np.testing.assert_allclose(scene.data.qpos[:2], source_qpos[:2])
        np.testing.assert_allclose(scene.data.qvel[:2], source_qvel[:2])
        self.assertEqual(scene.root_assist.ticks, 1)
        metadata = scene.root_assist.metadata()
        self.assertEqual(metadata["position_components"], ["x", "y"])
        self.assertEqual(metadata["velocity_components"], ["x", "y"])
        self.assertEqual(metadata["source_velocity_active_ticks"], 1)
        self.assertEqual(metadata["zero_velocity_fallback_ticks"], 0)

        scene.apply_root_assist(
            source_qpos, source_qvel, source_velocity_active=False
        )
        np.testing.assert_allclose(scene.data.qvel[:2], 0.0)
        metadata = scene.root_assist.metadata()
        self.assertEqual(metadata["source_velocity_active_ticks"], 1)
        self.assertEqual(metadata["zero_velocity_fallback_ticks"], 1)
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
