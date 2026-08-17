#!/usr/bin/env python3
"""Headless tests for the custom in-window reference ghost."""

from __future__ import annotations

from contextlib import contextmanager, redirect_stderr
from io import StringIO
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig  # noqa: E402
from sim_reference_overlay.ghost_overlay import (  # noqa: E402
    GhostOverlayConfig,
    G1_MUJOCO_JOINT_NAMES,
    ReferenceGhostError,
    ReferenceGhostOverlay,
    ReferenceRootAlignment,
    TargetPose,
    apply_target_pose,
    fill_reference_ghost_scene,
    inspect_robot_visual,
)
from sim_reference_overlay.run_sim_loop import (  # noqa: E402
    ReferenceGhostSimConfig,
    ReferenceGhostSimulator,
)


def target_message(
    *,
    index: int = 0,
    root_position: tuple[float, float, float] = (1.0, 2.0, 0.9),
    joint_offset: float = 0.0,
) -> dict[str, object]:
    return {
        "index": index,
        "base_trans_target": np.asarray(root_position),
        "base_quat_target": np.asarray(
            [0.9800665778, 0.0, 0.1986693308, 0.0]
        ),
        "body_q_target": np.linspace(-0.2, 0.2, 29) + joint_offset,
    }


class TargetPoseTest(unittest.TestCase):
    def test_reference_root_mode_is_the_public_default(self) -> None:
        self.assertEqual(GhostOverlayConfig().root_mode, "reference")
        self.assertEqual(ReferenceGhostSimConfig().ghost_root_mode, "reference")

    def test_valid_message_is_copied_and_quaternion_is_normalized(self) -> None:
        message = target_message()
        message["base_quat_target"] = np.asarray([1.001, 0.0, 0.0, 0.0])
        target = TargetPose.from_debug_message(message)
        np.testing.assert_allclose(target.root_position, [1.0, 2.0, 0.9])
        np.testing.assert_allclose(target.root_quaternion_wxyz, [1.0, 0.0, 0.0, 0.0])
        self.assertFalse(target.body_joint_positions.flags.writeable)
        self.assertEqual(target.source_index, 0)

    def test_bad_shape_and_nonfinite_values_are_rejected(self) -> None:
        message = target_message()
        message["body_q_target"] = np.zeros(28)
        with self.assertRaisesRegex(ReferenceGhostError, r"shape \(29,\)"):
            TargetPose.from_debug_message(message)

        message = target_message()
        message["base_trans_target"][0] = np.nan
        with self.assertRaisesRegex(ReferenceGhostError, "NaN or infinity"):
            TargetPose.from_debug_message(message)

        message = target_message()
        message["base_trans_target"][0] = 1.0e100
        with self.assertRaisesRegex(ReferenceGhostError, "scene floats|safety limit"):
            TargetPose.from_debug_message(message)

        message = target_message()
        message["body_q_target"][4] = 101.0
        with self.assertRaisesRegex(ReferenceGhostError, "safety limit"):
            TargetPose.from_debug_message(message)

    def test_config_rejects_ambiguous_or_unsafe_values(self) -> None:
        with self.assertRaisesRegex(ReferenceGhostError, "root_mode"):
            GhostOverlayConfig(root_mode="both")  # type: ignore[arg-type]
        with self.assertRaisesRegex(ReferenceGhostError, "alpha"):
            GhostOverlayConfig(alpha=0.0)
        with self.assertRaisesRegex(ReferenceGhostError, "port"):
            GhostOverlayConfig(port=0)


class ReferenceRootAlignmentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.alignment = ReferenceRootAlignment()
        self.actual_start = np.asarray([0.3, -0.4, 0.82])
        self.start = TargetPose.from_debug_message(target_message(index=10))

    def test_first_frame_and_paused_updates_align_all_xyz(self) -> None:
        self.assertEqual(
            self.alignment.observe(self.start, self.actual_start), "anchored"
        )
        np.testing.assert_allclose(
            self.alignment.aligned_position(self.start), self.actual_start
        )

        moved_actual = np.asarray([0.35, -0.45, 0.86])
        repeated_start = TargetPose.from_debug_message(target_message(index=11))
        self.assertIsNone(self.alignment.observe(repeated_start, moved_actual))
        np.testing.assert_allclose(
            self.alignment.aligned_position(repeated_start), moved_actual
        )

    def test_departure_freezes_anchor_and_preserves_reference_delta(self) -> None:
        self.alignment.observe(self.start, self.actual_start)
        departure = TargetPose.from_debug_message(
            target_message(
                index=11,
                root_position=(1.25, 1.5, 1.0),
                joint_offset=0.02,
            )
        )
        self.alignment.observe(departure, np.asarray([5.0, 6.0, 7.0]))
        np.testing.assert_allclose(
            self.alignment.aligned_position(departure),
            self.actual_start + np.asarray([0.25, -0.5, 0.1]),
        )

        later = TargetPose.from_debug_message(
            target_message(
                index=12,
                root_position=(1.5, 2.75, 0.7),
                joint_offset=0.04,
            )
        )
        self.alignment.observe(later, np.asarray([-8.0, 9.0, 3.0]))
        np.testing.assert_allclose(
            self.alignment.aligned_position(later),
            self.actual_start + np.asarray([0.5, 0.75, -0.2]),
        )

    def test_three_start_messages_reanchor_for_replay(self) -> None:
        self.alignment.observe(self.start, self.actual_start)
        departure = TargetPose.from_debug_message(
            target_message(index=11, root_position=(1.4, 2.2, 0.95), joint_offset=0.1)
        )
        self.alignment.observe(departure, self.actual_start)

        new_actual = np.asarray([2.0, -3.0, 0.75])
        for index in (12, 13):
            returning = TargetPose.from_debug_message(target_message(index=index))
            self.assertIsNone(self.alignment.observe(returning, new_actual))
        returned = TargetPose.from_debug_message(target_message(index=14))
        self.assertEqual(
            self.alignment.observe(returned, new_actual), "reanchored"
        )
        np.testing.assert_allclose(
            self.alignment.aligned_position(returned), new_actual
        )

        second_departure = TargetPose.from_debug_message(
            target_message(
                index=15,
                root_position=(1.2, 2.3, 0.95),
                joint_offset=0.1,
            )
        )
        self.alignment.observe(second_departure, np.asarray([9.0, 9.0, 9.0]))
        np.testing.assert_allclose(
            self.alignment.aligned_position(second_departure),
            new_actual + np.asarray([0.2, 0.3, 0.05]),
        )

    def test_two_start_pose_crossings_do_not_reanchor(self) -> None:
        self.alignment.observe(self.start, self.actual_start)
        departure = TargetPose.from_debug_message(
            target_message(index=11, root_position=(1.2, 2.0, 0.9), joint_offset=0.1)
        )
        self.alignment.observe(departure, self.actual_start)
        crossing = TargetPose.from_debug_message(target_message(index=12))
        self.alignment.observe(crossing, np.asarray([9.0, 9.0, 9.0]))
        second_crossing = TargetPose.from_debug_message(target_message(index=13))
        self.alignment.observe(second_crossing, np.asarray([9.0, 9.0, 9.0]))
        continued = TargetPose.from_debug_message(
            target_message(index=14, root_position=(1.3, 2.0, 0.9), joint_offset=0.2)
        )
        self.alignment.observe(continued, np.asarray([9.0, 9.0, 9.0]))
        np.testing.assert_allclose(
            self.alignment.aligned_position(continued),
            self.actual_start + np.asarray([0.3, 0.0, 0.0]),
        )

    def test_controller_index_rollback_starts_a_new_anchor(self) -> None:
        first = TargetPose.from_debug_message(target_message(index=100))
        self.alignment.observe(first, self.actual_start)
        departure = TargetPose.from_debug_message(
            target_message(index=101, root_position=(1.2, 2.0, 0.9), joint_offset=0.1)
        )
        self.alignment.observe(departure, self.actual_start)

        restarted_actual = np.asarray([-1.0, 4.0, 0.91])
        restarted = TargetPose.from_debug_message(
            target_message(index=0, root_position=(8.0, 7.0, 1.1), joint_offset=0.5)
        )
        self.assertEqual(
            self.alignment.observe(restarted, restarted_actual),
            "controller-restarted",
        )
        np.testing.assert_allclose(
            self.alignment.aligned_position(restarted), restarted_actual
        )


class SimulatorIsolationTest(unittest.TestCase):
    def test_overlay_error_disables_only_ghost_and_still_updates_viewer(self) -> None:
        class FailingOverlay:
            def __init__(self) -> None:
                self.closed = False

            def update(self) -> int:
                raise RuntimeError("synthetic render failure")

            def close(self) -> None:
                self.closed = True
                raise RuntimeError("synthetic cleanup failure")

        overlay = FailingOverlay()
        viewer_updates: list[bool] = []
        harness = SimpleNamespace(
            reference_ghost=overlay,
            _viewer_update_without_ghost=lambda: viewer_updates.append(True),
        )
        with redirect_stderr(StringIO()) as stderr:
            ReferenceGhostSimulator._update_viewer_with_reference_ghost(harness)
        self.assertIsNone(harness.reference_ghost)
        self.assertTrue(overlay.closed)
        self.assertEqual(viewer_updates, [True])
        self.assertIn("physics will continue", stderr.getvalue())
        self.assertIn("cleanup also failed", stderr.getvalue())


class ActualG1ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = SimLoopConfig().load_wbc_yaml()
        scene_path = Path(config["ROBOT_SCENE"])
        if not scene_path.is_absolute():
            scene_path = REPO_ROOT / scene_path
        cls.model = mujoco.MjModel.from_xml_path(str(scene_path))
        cls.actual_data = mujoco.MjData(cls.model)
        cls.ghost_data = mujoco.MjData(cls.model)
        mujoco.mj_forward(cls.model, cls.actual_data)
        cls.visual = inspect_robot_visual(cls.model)

    def setUp(self) -> None:
        mujoco.mj_resetData(self.model, self.actual_data)
        mujoco.mj_resetData(self.model, self.ghost_data)
        self.actual_data.qpos[self.visual.root_qpos_indices[:3]] = [0.3, -0.4, 0.82]
        self.actual_data.qpos[self.visual.root_qpos_indices[3:7]] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(self.model, self.actual_data)
        self.target = TargetPose.from_debug_message(target_message())

    def test_model_contract_is_29_dof_floating_g1(self) -> None:
        self.assertEqual(self.visual.root_qpos_indices.shape, (7,))
        self.assertEqual(self.visual.body_joint_qpos_indices.shape, (29,))
        self.assertEqual(self.visual.body_joint_names, G1_MUJOCO_JOINT_NAMES)
        self.assertGreater(len(self.visual.geom_ids), 0)

    def test_actual_mode_aligns_only_translation(self) -> None:
        actual_qpos_before = self.actual_data.qpos.copy()
        apply_target_pose(
            model=self.model,
            actual_data=self.actual_data,
            ghost_data=self.ghost_data,
            visual=self.visual,
            target=self.target,
            root_mode="actual",
        )
        np.testing.assert_array_equal(self.actual_data.qpos, actual_qpos_before)
        root = self.visual.root_qpos_indices
        np.testing.assert_allclose(
            self.ghost_data.qpos[root[:3]], self.actual_data.qpos[root[:3]]
        )
        np.testing.assert_allclose(
            self.ghost_data.qpos[root[3:7]], self.target.root_quaternion_wxyz
        )
        np.testing.assert_allclose(
            self.ghost_data.qpos[self.visual.body_joint_qpos_indices],
            self.target.body_joint_positions,
        )

    def test_reference_mode_starts_aligned_and_preserves_relative_trajectory(self) -> None:
        alignment = ReferenceRootAlignment()
        alignment.observe(
            self.target,
            self.actual_data.qpos[self.visual.root_qpos_indices[:3]],
        )
        apply_target_pose(
            model=self.model,
            actual_data=self.actual_data,
            ghost_data=self.ghost_data,
            visual=self.visual,
            target=self.target,
            root_mode="reference",
            aligned_reference_root_position=alignment.aligned_position(self.target),
        )
        root = self.visual.root_qpos_indices
        np.testing.assert_allclose(
            self.ghost_data.qpos[root[:3]], self.actual_data.qpos[root[:3]]
        )
        np.testing.assert_allclose(
            self.ghost_data.qpos[root[3:7]], self.target.root_quaternion_wxyz
        )

        next_target = TargetPose.from_debug_message(
            target_message(
                index=1,
                root_position=(1.25, 1.5, 1.0),
                joint_offset=0.05,
            )
        )
        alignment.observe(next_target, np.asarray([8.0, 9.0, 10.0]))
        apply_target_pose(
            model=self.model,
            actual_data=self.actual_data,
            ghost_data=self.ghost_data,
            visual=self.visual,
            target=next_target,
            root_mode="reference",
            aligned_reference_root_position=alignment.aligned_position(next_target),
        )
        np.testing.assert_allclose(
            self.ghost_data.qpos[root[:3]],
            np.asarray([0.3, -0.4, 0.82]) + np.asarray([0.25, -0.5, 0.1]),
        )

    def test_visual_mesh_can_be_built_headlessly(self) -> None:
        apply_target_pose(
            model=self.model,
            actual_data=self.actual_data,
            ghost_data=self.ghost_data,
            visual=self.visual,
            target=self.target,
            root_mode="actual",
        )
        scene = mujoco.MjvScene(self.model, maxgeom=1000)
        count = fill_reference_ghost_scene(
            scene=scene,
            model=self.model,
            ghost_data=self.ghost_data,
            visual=self.visual,
            alpha=0.3,
        )
        self.assertGreater(count, 0)
        self.assertEqual(scene.ngeom, count)
        for index in range(count):
            self.assertAlmostEqual(float(scene.geoms[index].rgba[3]), 0.3, places=6)

    def test_overlay_lifecycle_with_injected_subscriber(self) -> None:
        class FakeSubscriber:
            def __init__(self) -> None:
                self.message = target_message()
                self.closed = False

            def get_msg(self, clear: bool = True):
                message, self.message = self.message, None
                return message

            def close(self) -> None:
                self.closed = True

        class FakeViewer:
            def __init__(self, model: mujoco.MjModel) -> None:
                self.user_scn = mujoco.MjvScene(model, maxgeom=1000)

            @contextmanager
            def lock(self):
                yield

        subscriber = FakeSubscriber()
        viewer = FakeViewer(self.model)
        overlay = ReferenceGhostOverlay(
            model=self.model,
            actual_data=self.actual_data,
            viewer=viewer,
            config=GhostOverlayConfig(root_mode="actual", alpha=0.25),
            subscriber=subscriber,
        )
        count = overlay.update()
        self.assertGreater(count, 0)
        self.assertEqual(viewer.user_scn.ngeom, count)
        overlay.close()
        self.assertTrue(subscriber.closed)
        self.assertEqual(viewer.user_scn.ngeom, 0)
        self.assertEqual(overlay.update(), 0)

    def test_reference_overlay_wires_initial_alignment_and_relative_motion(self) -> None:
        class QueueSubscriber:
            def __init__(self) -> None:
                self.messages = [target_message(index=0)]

            def get_msg(self, clear: bool = True):
                return self.messages.pop(0) if self.messages else None

            def close(self) -> None:
                pass

        class FakeViewer:
            def __init__(self, model: mujoco.MjModel) -> None:
                self.user_scn = mujoco.MjvScene(model, maxgeom=1000)

            @contextmanager
            def lock(self):
                yield

        subscriber = QueueSubscriber()
        overlay = ReferenceGhostOverlay(
            model=self.model,
            actual_data=self.actual_data,
            viewer=FakeViewer(self.model),
            config=GhostOverlayConfig(root_mode="reference", alpha=0.25),
            subscriber=subscriber,
        )
        root = self.visual.root_qpos_indices
        overlay.update()
        np.testing.assert_allclose(
            overlay.ghost_data.qpos[root[:3]], self.actual_data.qpos[root[:3]]
        )

        paused_actual = np.asarray([0.5, -0.6, 0.84])
        self.actual_data.qpos[root[:3]] = paused_actual
        subscriber.messages.append(target_message(index=1))
        overlay.update()
        np.testing.assert_allclose(overlay.ghost_data.qpos[root[:3]], paused_actual)

        self.actual_data.qpos[root[:3]] = [9.0, 9.0, 9.0]
        subscriber.messages.append(
            target_message(
                index=2,
                root_position=(1.25, 1.5, 1.0),
                joint_offset=0.05,
            )
        )
        overlay.update()
        np.testing.assert_allclose(
            overlay.ghost_data.qpos[root[:3]],
            paused_actual + np.asarray([0.25, -0.5, 0.1]),
        )
        overlay.close()

    def test_reference_overlay_reanchors_after_stale_stream(self) -> None:
        class QueueSubscriber:
            def __init__(self) -> None:
                self.messages = [
                    target_message(index=10),
                    target_message(
                        index=11,
                        root_position=(1.25, 1.5, 1.0),
                        joint_offset=0.05,
                    ),
                    None,
                    target_message(
                        index=500,
                        root_position=(8.0, 7.0, 1.1),
                        joint_offset=0.5,
                    ),
                ]

            def get_msg(self, clear: bool = True):
                return self.messages.pop(0)

            def close(self) -> None:
                pass

        class FakeViewer:
            def __init__(self, model: mujoco.MjModel) -> None:
                self.user_scn = mujoco.MjvScene(model, maxgeom=1000)

            @contextmanager
            def lock(self):
                yield

        subscriber = QueueSubscriber()
        viewer = FakeViewer(self.model)
        overlay = ReferenceGhostOverlay(
            model=self.model,
            actual_data=self.actual_data,
            viewer=viewer,
            config=GhostOverlayConfig(
                root_mode="reference", alpha=0.25, stale_timeout_s=1.0
            ),
            subscriber=subscriber,
        )
        with patch(
            "sim_reference_overlay.ghost_overlay.time.monotonic",
            side_effect=[0.0, 0.1, 2.0, 2.1],
        ):
            overlay.update()
            overlay.update()
            self.assertEqual(overlay.update(), 0)
            self.assertFalse(overlay.reference_root_alignment.anchored)
            self.assertEqual(viewer.user_scn.ngeom, 0)

            resumed_actual = np.asarray([-2.0, 3.0, 0.88])
            self.actual_data.qpos[self.visual.root_qpos_indices[:3]] = resumed_actual
            overlay.update()
            np.testing.assert_allclose(
                overlay.ghost_data.qpos[self.visual.root_qpos_indices[:3]],
                resumed_actual,
            )
        overlay.close()


if __name__ == "__main__":
    unittest.main()
