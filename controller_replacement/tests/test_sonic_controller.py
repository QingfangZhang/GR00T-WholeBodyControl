"""Regression tests for the deterministic in-process SONIC adapter."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from controller_replacement.controllers.sonic import (
    ACTION_SCALE_MUJOCO,
    DECODER_INPUT_DIM,
    DEFAULT_DOF_POS_ISAACLAB,
    DEFAULT_DOF_POS_MUJOCO,
    HISTORY_LENGTH,
    ISAACLAB_INDEX_FOR_MUJOCO,
    MUJOCO_INDEX_FOR_ISAACLAB,
    SonicController,
    SonicControllerError,
    SonicHistoryFrame,
    SonicVariant,
    TOKEN_DIM,
    action_to_q_target,
    default_model_spec,
)
from controller_replacement.references.reference_motion import (
    ReferenceMotionProvider,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RECORDING = REPO_ROOT / "sample_data/ztj/20260612/20260720_144342_g1_sim"

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - exercised in the light simulator env
    ort = None


@dataclass(frozen=True)
class _ValueInfo:
    name: str
    shape: tuple[int, int]
    type: str = "tensor(float)"


class _FakeSession:
    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        output_name: str,
        *,
        output: np.ndarray | None = None,
    ) -> None:
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.output_name = output_name
        self.output = output
        self.last_input: np.ndarray | None = None

    def get_inputs(self):
        return [_ValueInfo("obs_dict", (1, self.input_dimension))]

    def get_outputs(self):
        return [_ValueInfo(self.output_name, (1, self.output_dimension))]

    def run(self, output_names, inputs):
        self.last_input = np.asarray(inputs["obs_dict"], dtype=np.float32).copy()
        if self.last_input.shape != (1, self.input_dimension):
            raise AssertionError(self.last_input.shape)
        if self.output is None:
            value = np.zeros((1, self.output_dimension), dtype=np.float32)
        else:
            value = np.asarray(self.output, dtype=np.float32).reshape(
                1, self.output_dimension
            )
        return [value]


def _fake_controller(variant: str = "regular", *, require_source_history: bool = False):
    spec = default_model_spec(variant)
    encoder = _FakeSession(spec.encoder_input_dim, TOKEN_DIM, "encoded_tokens")
    decoder = _FakeSession(
        DECODER_INPUT_DIM,
        29,
        "action",
        output=np.linspace(-0.4, 0.4, 29, dtype=np.float32),
    )
    controller = SonicController(
        variant,
        encoder_session=encoder,
        decoder_session=decoder,
        require_source_history=require_source_history,
    )
    return controller, encoder, decoder


def _identity_reference() -> SimpleNamespace:
    regular_q = np.arange(10 * 29, dtype=np.float32).reshape(10, 29) / 100.0
    regular_dq = regular_q + 10.0
    consecutive_q = regular_q + 20.0
    consecutive_dq = regular_q + 30.0
    quaternion = np.tile(
        np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (10, 1)
    )
    return SimpleNamespace(
        sonic_regular_joint_pos=regular_q,
        sonic_regular_joint_vel=regular_dq,
        sonic_regular_anchor_quat_wxyz=quaternion,
        sonic_consecutive_joint_pos=consecutive_q,
        sonic_consecutive_joint_vel=consecutive_dq,
        sonic_consecutive_anchor_quat_wxyz=quaternion,
    )


def _state(
    *,
    joint_pos: np.ndarray | None = None,
    joint_vel: np.ndarray | None = None,
    quaternion: np.ndarray | None = None,
    angular_velocity: np.ndarray | None = None,
) -> SimpleNamespace:
    body_joint_pos = (
            DEFAULT_DOF_POS_MUJOCO.copy()
            if joint_pos is None
            else np.asarray(joint_pos, dtype=np.float64)
        )
    return SimpleNamespace(
        joint_pos=body_joint_pos,
        joint_vel=(
            np.zeros(29, dtype=np.float64)
            if joint_vel is None
            else np.asarray(joint_vel, dtype=np.float64)
        ),
        root_quat_wxyz=(
            np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            if quaternion is None
            else np.asarray(quaternion, dtype=np.float64)
        ),
        root_ang_vel_b=(
            np.zeros(3, dtype=np.float64)
            if angular_velocity is None
            else np.asarray(angular_velocity, dtype=np.float64)
        ),
        left_hand_pos=np.arange(7, dtype=np.float64) / 100.0,
        right_hand_pos=-np.arange(7, dtype=np.float64) / 100.0,
    )


class SonicControllerUnitTest(unittest.TestCase):
    def test_joint_order_mappings_are_exact_inverses(self) -> None:
        values = np.arange(29)
        np.testing.assert_array_equal(
            values[MUJOCO_INDEX_FOR_ISAACLAB][ISAACLAB_INDEX_FOR_MUJOCO],
            values,
        )

    def test_regular_and_low_latency_use_their_native_reference_views(self) -> None:
        reference = _identity_reference()
        state = _state()
        regular, _, _ = _fake_controller("regular")
        low_latency, _, _ = _fake_controller("low_latency")
        regular_input = regular.build_encoder_input(state, reference)
        low_input = low_latency.build_encoder_input(state, reference)
        self.assertEqual(regular_input.shape, (1751,))
        self.assertEqual(low_input.shape, (1247,))
        np.testing.assert_array_equal(
            regular_input[4:294], reference.sonic_regular_joint_pos.reshape(-1)
        )
        np.testing.assert_array_equal(
            regular_input[294:584], reference.sonic_regular_joint_vel.reshape(-1)
        )
        np.testing.assert_array_equal(
            low_input[4:294], reference.sonic_consecutive_joint_pos.reshape(-1)
        )
        np.testing.assert_array_equal(
            low_input[294:584], reference.sonic_consecutive_joint_vel.reshape(-1)
        )
        expected_identity_6d = np.tile(
            np.asarray([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]), 10
        )
        np.testing.assert_allclose(regular_input[584:644], expected_identity_6d)
        self.assertTrue(np.all(regular_input[644:] == 0.0))
        self.assertTrue(np.all(low_input[644:] == 0.0))

    def test_v11_uses_robot_heading_instead_of_full_orientation(self) -> None:
        reference = _identity_reference()
        # Non-zero roll and yaw.  Full-base and heading-only normalization must
        # differ because v1.1 deliberately retains the reference's tilt in a
        # yaw-canonical frame.
        roll = 0.25
        yaw = -0.4
        q_roll = np.asarray([np.cos(roll / 2), np.sin(roll / 2), 0.0, 0.0])
        q_yaw = np.asarray([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        w1, x1, y1, z1 = q_yaw
        w2, x2, y2, z2 = q_roll
        quaternion = np.asarray(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ]
        )
        state = _state(quaternion=quaternion)
        regular, _, _ = _fake_controller("regular")
        v11, _, _ = _fake_controller("sonic_v1_1")
        regular_orientation = regular.build_encoder_input(state, reference)[584:644]
        v11_orientation = v11.build_encoder_input(state, reference)[584:644]
        self.assertGreater(float(np.max(np.abs(regular_orientation - v11_orientation))), 0.1)

    def test_decoder_input_has_exact_block_order_and_shape(self) -> None:
        history = [
            SonicHistoryFrame(
                body_q=np.full(29, frame + 1.0),
                body_dq=np.full(29, frame + 101.0),
                root_quat_wxyz=[1.0, 0.0, 0.0, 0.0],
                root_ang_vel_b=np.full(3, frame + 201.0),
                last_action=np.full(29, frame + 301.0),
            )
            for frame in range(HISTORY_LENGTH)
        ]
        token = np.arange(TOKEN_DIM, dtype=np.float32)
        value = SonicController.build_decoder_input(token, history)
        self.assertEqual(value.shape, (DECODER_INPUT_DIM,))
        np.testing.assert_array_equal(value[:64], token)
        np.testing.assert_array_equal(
            value[64:94], np.stack([item.root_ang_vel_b for item in history]).reshape(-1)
        )
        np.testing.assert_array_equal(
            value[94:384], np.stack([item.body_q for item in history]).reshape(-1)
        )
        np.testing.assert_array_equal(
            value[384:674], np.stack([item.body_dq for item in history]).reshape(-1)
        )
        np.testing.assert_array_equal(
            value[674:964], np.stack([item.last_action for item in history]).reshape(-1)
        )
        np.testing.assert_array_equal(
            value[964:994], np.tile([0.0, 0.0, -1.0], HISTORY_LENGTH)
        )

    def test_action_to_q_target_matches_cpp_order_and_formula(self) -> None:
        raw_action = np.linspace(-1.0, 1.0, 29)
        actual = action_to_q_target(raw_action)
        expected = DEFAULT_DOF_POS_MUJOCO + (
            raw_action[ISAACLAB_INDEX_FOR_MUJOCO] * ACTION_SCALE_MUJOCO
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)

    def test_source_prefill_drives_first_decoder_history_without_zero_padding(self) -> None:
        entries = []
        for index in range(9):
            entries.append(
                {
                    "policy_seq": 100 + index,
                    "body_q": np.full(29, index / 10.0).tolist(),
                    "body_dq": np.full(29, index / 20.0).tolist(),
                    "base_quat": [1.0, 0.0, 0.0, 0.0],
                    "base_ang_vel": np.full(3, index / 30.0).tolist(),
                    "last_action": np.full(29, index / 40.0).tolist(),
                }
            )
        current_body_q = np.linspace(-0.1, 0.1, 29)
        current_body_dq = np.linspace(-0.2, 0.2, 29)
        current_last_action = np.linspace(-0.3, 0.3, 29)
        payload = {
            "format": "g1_decoder_source_history_prefill",
            "version": 1,
            "history_order": "oldest_to_newest",
            "history_entry_count": 9,
            "entries": entries,
            "current": {
                "policy_seq": 109,
                "body_q": current_body_q.tolist(),
                "body_dq": current_body_dq.tolist(),
                "base_quat": [1.0, 0.0, 0.0, 0.0],
                "base_ang_vel": [0.1, 0.2, 0.3],
                "last_action": current_last_action.tolist(),
            },
        }
        current_q_isaaclab = current_body_q + DEFAULT_DOF_POS_ISAACLAB
        current_q_mujoco = current_q_isaaclab[ISAACLAB_INDEX_FOR_MUJOCO]
        current_dq_mujoco = current_body_dq[ISAACLAB_INDEX_FOR_MUJOCO]
        controller, _, decoder = _fake_controller(
            "regular", require_source_history=True
        )
        controller.reset(payload)
        result = controller.infer(
            _state(
                joint_pos=current_q_mujoco,
                joint_vel=current_dq_mujoco,
                angular_velocity=[0.1, 0.2, 0.3],
            ),
            _identity_reference(),
        )
        self.assertEqual(result.history.shape, (10, 93))
        self.assertEqual(result.received_dof_pos.shape, (43,))
        np.testing.assert_allclose(
            result.received_dof_pos,
            np.concatenate(
                (
                    current_q_mujoco,
                    np.arange(7, dtype=np.float64) / 100.0,
                    -np.arange(7, dtype=np.float64) / 100.0,
                )
            ),
            atol=1e-8,
        )
        np.testing.assert_allclose(result.last_action, current_last_action, atol=2e-8)
        np.testing.assert_allclose(result.history[-1, 3:32], current_body_q, atol=2e-8)
        self.assertIsNotNone(decoder.last_input)
        self.assertEqual(decoder.last_input.shape, (1, DECODER_INPUT_DIM))

    def test_formal_mode_refuses_implicit_zero_history(self) -> None:
        controller, _, _ = _fake_controller("regular", require_source_history=True)
        with self.assertRaisesRegex(SonicControllerError, "source-history prefill"):
            controller.infer(_state(), _identity_reference())


@unittest.skipIf(ort is None, "onnxruntime is required for real-model checks")
class SonicOnnxRegressionTest(unittest.TestCase):
    def test_all_released_onnx_signatures(self) -> None:
        expected = {
            SonicVariant.REGULAR: 1751,
            SonicVariant.LOW_LATENCY: 1247,
            SonicVariant.SONIC_V1_1: 1751,
        }
        for variant, encoder_dimension in expected.items():
            with self.subTest(variant=variant.value):
                spec = default_model_spec(variant)
                encoder = ort.InferenceSession(
                    str(spec.encoder_path), providers=["CPUExecutionProvider"]
                )
                decoder = ort.InferenceSession(
                    str(spec.decoder_path), providers=["CPUExecutionProvider"]
                )
                self.assertEqual(
                    [(item.name, item.shape, item.type) for item in encoder.get_inputs()],
                    [("obs_dict", [1, encoder_dimension], "tensor(float)")],
                )
                self.assertEqual(
                    [(item.name, item.shape, item.type) for item in encoder.get_outputs()],
                    [("encoded_tokens", [1, 64], "tensor(float)")],
                )
                self.assertEqual(
                    [(item.name, item.shape, item.type) for item in decoder.get_inputs()],
                    [("obs_dict", [1, 994], "tensor(float)")],
                )
                self.assertEqual(
                    [(item.name, item.shape, item.type) for item in decoder.get_outputs()],
                    [("action", [1, 29], "tensor(float)")],
                )

    @unittest.skipUnless(RECORDING.is_dir(), "verified source recording is required")
    def test_regular_encoder_reproduces_recorded_token_exactly(self) -> None:
        sequence = ReferenceMotionProvider().load(
            RECORDING, policy_offset=10, policy_count=1
        )
        reference = sequence.frame(0)
        spec = default_model_spec("regular")
        encoder = ort.InferenceSession(
            str(spec.encoder_path), providers=["CPUExecutionProvider"]
        )
        decoder = _FakeSession(994, 29, "action")
        controller = SonicController(
            "regular",
            encoder_session=encoder,
            decoder_session=decoder,
            require_source_history=False,
        )
        # Encoder G1 mode does not consume live q/dq, but it does consume the
        # live pelvis quaternion when constructing relative anchor rotation.
        state = _state(quaternion=reference.source_root_quat_wxyz)
        encoder_input = controller.build_encoder_input(state, reference)
        actual = encoder.run(
            ["encoded_tokens"], {"obs_dict": encoder_input[None]}
        )[0][0]

        with sequence.source_csv_path.open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.reader(stream)
            header = next(reader)
            column = {name: index for index, name in enumerate(header)}
            row = None
            for source_index, candidate in enumerate(reader):
                if source_index == reference.source_row_index:
                    row = candidate
                    break
        self.assertIsNotNone(row)
        assert row is not None
        expected = np.asarray(
            [float(row[column[f"token_state[{index}]"]]) for index in range(64)],
            dtype=np.float32,
        )
        difference = np.abs(actual - expected)
        self.assertEqual(float(np.max(difference)), 0.0)


if __name__ == "__main__":
    unittest.main()
