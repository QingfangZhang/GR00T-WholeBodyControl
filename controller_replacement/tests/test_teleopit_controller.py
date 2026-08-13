"""Golden and lifecycle tests for the pinned Teleopit controller adapter."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controller_replacement.controllers.teleopit import (  # noqa: E402
    SONIC_DEFAULT_DOF_POS_MUJOCO,
    SONIC_MUJOCO_INDEX_FOR_ISAACLAB,
    TeleopitAdapterError,
    TeleopitController,
    build_sonic_source_history_prefill,
    convert_sonic_previous_action,
)
from Teleopit_rollout.constants import (  # noqa: E402
    ACTION_SCALE,
    DEFAULT_DOF_POS,
    KDS,
    KPS,
    TORQUE_LIMITS,
)
from Teleopit_rollout.teleopit_policy import (  # noqa: E402
    RobotState,
    TeleopitObservationBuilder,
)


ROBOT_XML = (
    REPO_ROOT
    / "Teleopit_rollout/assets/robot_assets/unitree_g1/g1_29dof.xml"
)
CHECKPOINT = REPO_ROOT / "Teleopit_rollout/assets/checkpoints/track_g1.onnx"
OFFICIAL_COMMIT = "f9263865c581802ad531854b8e547e2403a945f3"
OFFICIAL_OBSERVATION_SHA256 = (
    "de15c2a46587ebf4faad3eeaaf1d5b97a8d3f145c160320d08ece344ff718722"
)


OBSERVATION_BLOCKS = {
    "reference_joint_position": slice(0, 29),
    "reference_joint_velocity": slice(29, 58),
    "relative_torso_orientation_6d": slice(58, 64),
    "robot_base_angular_velocity_body": slice(64, 67),
    "robot_joint_position_relative_default": slice(67, 96),
    "robot_joint_velocity": slice(96, 125),
    "previous_raw_action": slice(125, 154),
    "robot_projected_gravity": slice(154, 157),
    "reference_torso_linear_velocity_body": slice(157, 160),
    "reference_torso_angular_velocity_body": slice(160, 163),
    "reference_projected_gravity": slice(163, 166),
    "reference_torso_height": slice(166, 167),
}


def _quat_inverse(q: np.ndarray) -> np.ndarray:
    result = np.asarray(q, dtype=np.float32).copy()
    result[1:] *= -1.0
    return result


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.asarray(left, dtype=np.float32)
    w2, x2, y2, z2 = np.asarray(right, dtype=np.float32)
    return np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def _quat_rotate(q: np.ndarray, vector: np.ndarray) -> np.ndarray:
    vector_q = np.zeros(4, dtype=np.float32)
    vector_q[1:] = np.asarray(vector, dtype=np.float32)
    return _quat_multiply(
        _quat_multiply(q, vector_q), _quat_inverse(q)
    )[1:]


def _rot6d(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float32)
    return np.asarray(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
        ],
        dtype=np.float32,
    )


class _IndependentFk:
    """Small FK oracle independent of the production observation builder."""

    def __init__(self, xml: Path) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        self.torso_id = self.model.body("torso_link").id

    def torso(self, qpos36: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pose = np.asarray(qpos36, dtype=np.float64)
        self.data.qpos[:] = 0.0
        self.data.qpos[:36] = pose
        mujoco.mj_kinematics(self.model, self.data)
        return (
            np.asarray(self.data.xpos[self.torso_id], dtype=np.float32).copy(),
            np.asarray(self.data.xquat[self.torso_id], dtype=np.float32).copy(),
        )


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    value = np.asarray(axis, dtype=np.float32)
    value /= np.linalg.norm(value)
    return np.concatenate(
        (
            np.asarray([np.cos(angle / 2.0)], dtype=np.float32),
            value * np.float32(np.sin(angle / 2.0)),
        )
    )


@dataclass(frozen=True)
class _Fixture:
    robot: RobotState
    previous_reference: np.ndarray
    current_reference: np.ndarray
    previous_action: np.ndarray


def _fixture() -> _Fixture:
    rng = np.random.default_rng(617)
    robot_quat = _axis_angle(np.asarray([0.25, -0.1, 0.4]), 0.37)
    previous_quat = _axis_angle(np.asarray([-0.2, 0.3, 0.1]), -0.21)
    delta_quat = _axis_angle(np.asarray([0.1, -0.4, 0.2]), 0.035)
    current_quat = _quat_multiply(delta_quat, previous_quat)
    robot_joint_pos = DEFAULT_DOF_POS + rng.normal(0.0, 0.08, 29).astype(np.float32)
    previous_joint_pos = DEFAULT_DOF_POS + rng.normal(0.0, 0.11, 29).astype(np.float32)
    current_joint_pos = previous_joint_pos + rng.normal(0.0, 0.006, 29).astype(np.float32)
    previous_reference = np.concatenate(
        (
            np.asarray([0.41, -0.26, 0.79], dtype=np.float32),
            previous_quat,
            previous_joint_pos,
        )
    )
    current_reference = np.concatenate(
        (
            np.asarray([0.419, -0.255, 0.792], dtype=np.float32),
            current_quat,
            current_joint_pos,
        )
    )
    return _Fixture(
        robot=RobotState(
            joint_pos=robot_joint_pos,
            joint_vel=rng.normal(0.0, 0.25, 29).astype(np.float32),
            root_pos=np.asarray([0.3, -0.2, 0.77], dtype=np.float32),
            root_quat_wxyz=robot_quat,
            root_ang_vel_b=np.asarray([0.17, -0.08, 0.23], dtype=np.float32),
            timestamp_s=1.24,
        ),
        previous_reference=previous_reference,
        current_reference=current_reference,
        previous_action=np.linspace(-0.42, 0.39, 29, dtype=np.float32),
    )


def _independent_oracle(fixture: _Fixture) -> np.ndarray:
    fk = _IndependentFk(ROBOT_XML)
    previous_anchor_pos, previous_anchor_quat = fk.torso(fixture.previous_reference)
    current_anchor_pos, current_anchor_quat = fk.torso(fixture.current_reference)
    robot_qpos36 = np.concatenate(
        (
            fixture.robot.root_pos,
            fixture.robot.root_quat_wxyz,
            fixture.robot.joint_pos,
        )
    )
    _, robot_anchor_quat = fk.torso(robot_qpos36)

    reference_joint_velocity = (
        fixture.current_reference[7:] - fixture.previous_reference[7:]
    ) * np.float32(50.0)
    reference_linear_velocity_w = (
        current_anchor_pos - previous_anchor_pos
    ) * np.float32(50.0)
    delta = _quat_multiply(current_anchor_quat, _quat_inverse(previous_anchor_quat))
    if delta[0] < 0.0:
        delta = -delta
    half_angle = np.float32(np.arccos(float(np.clip(delta[0], -1.0, 1.0))))
    sin_half = np.float32(np.sin(half_angle))
    reference_angular_velocity_w = (
        delta[1:] / sin_half * np.float32(2.0 * half_angle * 50.0)
        if sin_half > 1e-6
        else np.zeros(3, dtype=np.float32)
    )

    inverse_robot_anchor = _quat_inverse(robot_anchor_quat)
    gravity = np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
    return np.concatenate(
        (
            fixture.current_reference[7:],
            reference_joint_velocity,
            _rot6d(_quat_multiply(inverse_robot_anchor, current_anchor_quat)),
            fixture.robot.root_ang_vel_b,
            fixture.robot.joint_pos - DEFAULT_DOF_POS,
            fixture.robot.joint_vel,
            fixture.previous_action,
            _quat_rotate(_quat_inverse(fixture.robot.root_quat_wxyz), gravity),
            _quat_rotate(inverse_robot_anchor, reference_linear_velocity_w),
            _quat_rotate(inverse_robot_anchor, reference_angular_velocity_w),
            _quat_rotate(_quat_inverse(current_anchor_quat), gravity),
            current_anchor_pos[2:3],
        )
    ).astype(np.float32)


@unittest.skipUnless(ROBOT_XML.is_file(), "pinned Teleopit robot XML is required")
class ObservationGoldenTest(unittest.TestCase):
    def test_all_167d_blocks_match_independent_oracle(self) -> None:
        fixture = _fixture()
        builder = TeleopitObservationBuilder(ROBOT_XML)
        features = builder.reference_features(
            fixture.current_reference, fixture.previous_reference
        )
        actual = builder.build(
            fixture.robot, features, fixture.previous_action
        )
        oracle = _independent_oracle(fixture)
        self.assertEqual(actual.shape, (167,))
        for name, block in OBSERVATION_BLOCKS.items():
            with self.subTest(block=name):
                np.testing.assert_allclose(
                    actual[block], oracle[block], rtol=0.0, atol=2e-6
                )

    def test_matches_official_v050_source_when_checkout_is_available(self) -> None:
        source_root = Path(
            os.environ.get("TELEOPIT_V050_SOURCE", "/tmp/Teleopit-v050")
        )
        observation_source = source_root / "teleopit/controllers/observation.py"
        if not observation_source.is_file():
            self.skipTest(
                "official Teleopit v0.5 checkout unavailable; independent oracle still ran"
            )
        digest = hashlib.sha256(observation_source.read_bytes()).hexdigest()
        self.assertEqual(digest, OFFICIAL_OBSERVATION_SHA256)

        # Import the pinned checkout only for this golden comparison.  The
        # production adapter does not depend on /tmp or on an online checkout.
        sys.path.insert(0, str(source_root))
        self.addCleanup(lambda: sys.path.remove(str(source_root)))
        official_observation = importlib.import_module(
            "teleopit.controllers.observation"
        )
        official_interfaces = importlib.import_module("teleopit.interfaces")
        fixture = _fixture()
        our_builder = TeleopitObservationBuilder(ROBOT_XML)
        features = our_builder.reference_features(
            fixture.current_reference, fixture.previous_reference
        )
        actual = our_builder.build(
            fixture.robot, features, fixture.previous_action
        )

        official_builder = official_observation.VelCmdObservationBuilder(
            {
                "num_actions": 29,
                "default_dof_pos": DEFAULT_DOF_POS,
                "xml_path": str(ROBOT_XML),
                "anchor_body_name": "torso_link",
            }
        )
        official_state = official_interfaces.RobotState(
            qpos=fixture.robot.joint_pos,
            qvel=fixture.robot.joint_vel,
            quat=fixture.robot.root_quat_wxyz,
            ang_vel=fixture.robot.root_ang_vel_b,
            timestamp=fixture.robot.timestamp_s,
            base_pos=fixture.robot.root_pos,
        )
        official = official_builder.build(
            official_state,
            fixture.current_reference,
            features.joint_vel,
            fixture.previous_action,
            features.anchor_lin_vel_w,
            features.anchor_ang_vel_w,
        )
        for name, block in OBSERVATION_BLOCKS.items():
            with self.subTest(block=name):
                np.testing.assert_allclose(
                    actual[block], official[block], rtol=0.0, atol=2e-6
                )


class PreviousActionConversionTest(unittest.TestCase):
    def test_round_trip_is_exact_in_physical_target_space(self) -> None:
        source_raw = np.linspace(-0.8, 0.7, 29, dtype=np.float32)
        result = convert_sonic_previous_action(source_raw)
        self.assertFalse(result.clipped)
        self.assertLess(result.max_abs_q_target_residual, 2e-7)
        np.testing.assert_allclose(
            DEFAULT_DOF_POS + ACTION_SCALE * result.teleopit_raw_action,
            result.source_q_target_mujoco,
            rtol=0.0,
            atol=2e-7,
        )
        self.assertEqual(
            result.metadata()["conversion"],
            "sonic raw action -> SONIC physical q_target in MuJoCo order -> "
            "inverse Teleopit default pose/action scale",
        )

    def test_unrepresentable_source_target_is_not_silently_clipped(self) -> None:
        source_raw = np.full(29, 100.0, dtype=np.float32)
        with self.assertRaisesRegex(TeleopitAdapterError, "outside Teleopit"):
            convert_sonic_previous_action(source_raw)
        diagnostic = convert_sonic_previous_action(source_raw, strict=False)
        self.assertTrue(diagnostic.clipped)
        self.assertGreater(diagnostic.max_abs_q_target_residual, 0.0)


def _source_snapshot(
    policy_seq: int,
    q_mujoco: np.ndarray,
    dq_mujoco: np.ndarray,
    source_raw_isaaclab: np.ndarray,
) -> dict[str, object]:
    return {
        "policy_seq": policy_seq,
        "base_quat": _axis_angle(
            np.asarray([0.1, -0.2, 0.4]), 0.001 * policy_seq
        ).tolist(),
        "base_ang_vel": [0.02, -0.03, 0.04],
        "body_q": (
            q_mujoco[SONIC_MUJOCO_INDEX_FOR_ISAACLAB]
            - SONIC_DEFAULT_DOF_POS_MUJOCO[SONIC_MUJOCO_INDEX_FOR_ISAACLAB]
        ).tolist(),
        "body_dq": dq_mujoco[SONIC_MUJOCO_INDEX_FOR_ISAACLAB].tolist(),
        "last_action": source_raw_isaaclab.tolist(),
    }


@unittest.skipUnless(
    ROBOT_XML.is_file() and CHECKPOINT.is_file(),
    "pinned Teleopit XML and checkpoint are required",
)
class SourceHistoryLifecycleTest(unittest.TestCase):
    def _payload_and_reference(self):
        rng = np.random.default_rng(901)
        policy_seq = np.arange(99, 110, dtype=np.int64)
        qpos36: list[np.ndarray] = []
        snapshots: list[dict[str, object]] = []
        measured_states: list[RobotState] = []
        for index, sequence in enumerate(policy_seq):
            joint_pos = DEFAULT_DOF_POS + rng.normal(0.0, 0.03, 29).astype(np.float32)
            joint_vel = rng.normal(0.0, 0.08, 29).astype(np.float32)
            root_quat = _axis_angle(
                np.asarray([0.2, -0.1, 0.3]), 0.02 * index
            )
            qpos36.append(
                np.concatenate(
                    (
                        np.asarray(
                            [0.4 + 0.002 * index, -0.2, 0.78],
                            dtype=np.float32,
                        ),
                        root_quat,
                        joint_pos + np.float32(0.004 * index),
                    )
                )
            )
            source_raw = rng.normal(0.0, 0.2, 29).astype(np.float32)
            snapshot = _source_snapshot(
                int(sequence), joint_pos, joint_vel, source_raw
            )
            if index > 0:
                snapshots.append(snapshot)
            measured_states.append(
                RobotState(
                    joint_pos=joint_pos,
                    joint_vel=joint_vel,
                    root_pos=np.asarray([1.2, -0.7, 0.77], dtype=np.float32),
                    root_quat_wxyz=np.asarray(snapshot["base_quat"], dtype=np.float32),
                    root_ang_vel_b=np.asarray(
                        snapshot["base_ang_vel"], dtype=np.float32
                    ),
                    timestamp_s=float(sequence) / 50.0,
                )
            )
        payload = {
            "format": "g1_decoder_source_history_prefill",
            "version": 1,
            "history_order": "oldest_to_newest",
            "history_entry_count": 9,
            "source_csv": "/recording/data.csv",
            "source_csv_sha256": "a" * 64,
            "entries": snapshots[:9],
            "current": snapshots[9],
        }
        reference = SimpleNamespace(
            policy_seq=policy_seq,
            teleopit_qpos36=np.asarray(qpos36, dtype=np.float32),
        )
        return payload, reference, measured_states[-1]

    def test_prefill_is_nine_prior_plus_first_live_and_runs_onnx(self) -> None:
        payload, reference_sequence, current_state = self._payload_and_reference()
        controller = TeleopitController(
            checkpoint=CHECKPOINT,
            robot_xml=ROBOT_XML,
        )
        prefill = build_sonic_source_history_prefill(
            payload=payload,
            reference_sequence=reference_sequence,
            observation_builder=controller.observation_builder,
        )
        self.assertEqual(prefill.prior_observations.shape, (9, 167))
        self.assertEqual(prefill.expected_current_observation.shape, (167,))
        self.assertEqual(prefill.takeover_policy_seq, 109)
        self.assertFalse(prefill.metadata()["previous_action_any_clipped"])

        controller.reset(prefill)
        current_reference = SimpleNamespace(
            policy_seq=109,
            teleopit_qpos36=reference_sequence.teleopit_qpos36[-1],
        )
        step = controller.infer(current_state, current_reference)
        np.testing.assert_allclose(
            step.observation_history[:9],
            prefill.prior_observations,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            step.observation_history[9], step.observation, rtol=0.0, atol=0.0
        )
        np.testing.assert_allclose(
            step.previous_raw_action,
            prefill.takeover_previous_raw_action,
            rtol=0.0,
            atol=0.0,
        )
        self.assertIsNotNone(step.prefill_validation_max_abs)
        self.assertLess(float(step.prefill_validation_max_abs), 2e-6)
        np.testing.assert_allclose(
            step.q_target,
            DEFAULT_DOF_POS + ACTION_SCALE * np.clip(step.raw_action, -10.0, 10.0),
            rtol=0.0,
            atol=2e-7,
        )
        self.assertEqual(
            controller.metadata()["history_initialization"],
            "source_history_prefill",
        )
        self.assertEqual(controller.metadata()["teleopit_commit"], OFFICIAL_COMMIT)

    def test_native_pd_is_used_and_clipped(self) -> None:
        controller = TeleopitController(
            checkpoint=CHECKPOINT,
            robot_xml=ROBOT_XML,
        )
        q = np.zeros(29, dtype=np.float32)
        dq = np.linspace(-0.5, 0.5, 29, dtype=np.float32)
        target = np.full(29, 50.0, dtype=np.float32)
        torque = controller.compute_pd_torque(q, dq, target)
        expected = np.clip(KPS * (target - q) - KDS * dq, -TORQUE_LIMITS, TORQUE_LIMITS)
        np.testing.assert_allclose(torque, expected, rtol=0.0, atol=1e-12)
        self.assertTrue(np.all(np.abs(torque) <= TORQUE_LIMITS))

    def test_formal_adapter_rejects_missing_source_history(self) -> None:
        _, reference_sequence, current_state = self._payload_and_reference()
        controller = TeleopitController(
            checkpoint=CHECKPOINT,
            robot_xml=ROBOT_XML,
        )
        controller.reset(None)
        current_reference = SimpleNamespace(
            policy_seq=109,
            teleopit_qpos36=reference_sequence.teleopit_qpos36[-1],
        )
        with self.assertRaisesRegex(TeleopitAdapterError, "requires source-history"):
            controller.infer(current_state, current_reference)

    def test_prefill_requires_reference_predecessor(self) -> None:
        payload, reference_sequence, _ = self._payload_and_reference()
        truncated = SimpleNamespace(
            policy_seq=reference_sequence.policy_seq[1:],
            teleopit_qpos36=reference_sequence.teleopit_qpos36[1:],
        )
        with self.assertRaisesRegex(TeleopitAdapterError, "immediately before"):
            build_sonic_source_history_prefill(
                payload=payload,
                reference_sequence=truncated,
                observation_builder=TeleopitObservationBuilder(ROBOT_XML),
            )


if __name__ == "__main__":
    unittest.main()
