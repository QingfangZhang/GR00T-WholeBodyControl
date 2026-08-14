"""Focused tests for the formal controller-replacement startup context."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from change_ckpt.source_history_prefill import (
    ROBOT_QPOS_IDS_IN_RECEIVED_ORDER,
)
from change_ckpt_track.qpos_reference_data import G1_MUJOCO_JOINT_NAMES
from controller_replacement.history import (
    DEFAULT_RAW_POLICY_GROUP_OFFSET,
    SourceHistoryContextError,
    build_source_history_context,
    load_reference_from_raw_policy_group,
    resolve_raw_policy_group_offset,
    write_source_history_artifacts,
)
from controller_replacement.launch_rollout import build_parser
from controller_replacement.references import load_reference


LEFT_HAND_NAMES = (
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
)
RIGHT_HAND_NAMES = tuple(name.replace("left_", "right_") for name in LEFT_HAND_NAMES)


def _joint_column(kind: str, joint: str, index: int) -> str:
    body = joint.removesuffix("_joint") + "_link"
    suffix = "angle" if kind == "qpos" else "omega"
    return f"{kind}:{body}.{joint}.{suffix}[{kind}{index}]"


def _write_recording(root: Path, *, first_group_rows: int = 1) -> Path:
    """Write the real 43-joint body/hand ordering used by source history."""

    recording = root / "history_g1_sim"
    recording.mkdir()
    root_qpos = [
        "qpos:pelvis.floating_base_joint.x[qpos0]",
        "qpos:pelvis.floating_base_joint.y[qpos1]",
        "qpos:pelvis.floating_base_joint.z[qpos2]",
        "qpos:pelvis.floating_base_joint.qw[qpos3]",
        "qpos:pelvis.floating_base_joint.qx[qpos4]",
        "qpos:pelvis.floating_base_joint.qy[qpos5]",
        "qpos:pelvis.floating_base_joint.qz[qpos6]",
    ]
    root_qvel = [
        f"qvel:pelvis.floating_base_joint.{name}[qvel{index}]"
        for index, name in enumerate(("vx", "vy", "vz", "wx", "wy", "wz"))
    ]
    # The XML ordering is left body through wrist, left hand, right arm, right
    # hand.  This is the non-contiguous 29-body mapping audited by the proven
    # source-history builder.
    left_body = G1_MUJOCO_JOINT_NAMES[:22]
    right_body = G1_MUJOCO_JOINT_NAMES[22:]
    xml_joints = (*left_body, *LEFT_HAND_NAMES, *right_body, *RIGHT_HAND_NAMES)
    qpos_names = [
        _joint_column("qpos", joint, index + 7)
        for index, joint in enumerate(xml_joints)
    ]
    qvel_names = [
        _joint_column("qvel", joint, index + 6)
        for index, joint in enumerate(xml_joints)
    ]
    header = [
        "mujoco_time_s",
        "control_time_s",
        *root_qpos,
        *qpos_names,
        *root_qvel,
        *qvel_names,
        "policy_valid",
        "policy_seq",
        *(f"policy_last_action_in[{index}]" for index in range(29)),
        *(f"policy_raw_action_out[{index}]" for index in range(29)),
        *(f"policy_received_dof_pos[{index}]" for index in range(43)),
        *(f"left_hand_q[{index}]" for index in range(7)),
        *(f"right_hand_q[{index}]" for index in range(7)),
    ]

    elapsed = 0
    with (recording / "data.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for group in range(14):
            row_count = first_group_rows if group == 0 else 8
            for within in range(row_count):
                qpos = np.zeros(50, dtype=np.float64)
                qvel = np.zeros(49, dtype=np.float64)
                qpos[:3] = [group * 0.01, -group * 0.005, 0.78]
                qpos[3] = 1.0
                qpos[7:] = (
                    group * 0.03
                    + np.arange(43, dtype=np.float64) * 0.001
                    + within * 1e-5
                )
                received = qpos[
                    np.asarray(ROBOT_QPOS_IDS_IN_RECEIVED_ORDER, dtype=np.int64)
                ]
                row = [
                    elapsed * 0.0025,
                    elapsed * 0.0025,
                    *qpos[:7],
                    *qpos[7:],
                    *qvel[:6],
                    *qvel[6:],
                    1,
                    5000 + group,
                    *np.zeros(29),
                    *np.zeros(29),
                    *received,
                    *np.linspace(0.0, 0.3, 7),
                    *np.linspace(0.3, 0.0, 7),
                ]
                writer.writerow(row)
                elapsed += 1
    return recording


class _FakeTeleopitObservationBuilder:
    """Duck-typed deterministic builder that exposes the adapter contract."""

    @staticmethod
    def source_pelvis_ang_vel_b(
        base_quat: np.ndarray,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        base_ang_vel_qvel: np.ndarray,
    ) -> np.ndarray:
        del base_quat, joint_pos, joint_vel
        return np.asarray(base_ang_vel_qvel, dtype=np.float32)

    @staticmethod
    def reference_features(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
        return np.concatenate((np.asarray(current), np.asarray(previous)))

    @staticmethod
    def build(state: object, features: np.ndarray, previous_action: np.ndarray) -> np.ndarray:
        values = np.concatenate(
            (
                np.asarray(state.joint_pos),
                np.asarray(state.joint_vel),
                np.asarray(state.root_quat_wxyz),
                np.asarray(state.root_ang_vel_b),
                np.asarray(features),
                np.asarray(previous_action),
            )
        ).astype(np.float32)
        # The exact contents are irrelevant here; the native adapter tests
        # separately audit the official 167-D builder against independent FK.
        return np.pad(values[:167], (0, max(0, 167 - values.size)))


class SourceHistoryContextTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="replacement_history_")
        self.addCleanup(self.temporary.cleanup)
        self.recording = _write_recording(Path(self.temporary.name))

    def test_launcher_defaults_to_twelfth_raw_group(self) -> None:
        args = build_parser().parse_args(
            [str(self.recording), "--controller", "regular"]
        )
        self.assertEqual(args.raw_policy_group_offset, 11)

    def test_exact_policy_selection_survives_truncated_edge_trim(self) -> None:
        # The one-row raw group 5000 is trimmed.  Raw offset 11 must still
        # select policy 5011 even though its processed reference index is 10.
        reference = load_reference_from_raw_policy_group(
            self.recording,
            mode="executed_qpos",
            raw_policy_group_offset=DEFAULT_RAW_POLICY_GROUP_OFFSET,
            policy_count=2,
        )
        self.assertEqual(int(reference.policy_seq[0]), 5011)
        self.assertEqual(
            resolve_raw_policy_group_offset(
                self.recording, selected_policy_seq=5011
            ),
            11,
        )

        context = build_source_history_context(
            self.recording, selected_reference=reference
        )
        self.assertEqual(context.selected_policy_seq, 5011)
        self.assertEqual(context.raw_policy_group_offset, 11)
        self.assertEqual(
            [
                entry["policy_seq"]
                for entry in context.sonic_prefill_payload["entries"]
            ],
            list(range(5002, 5011)),
        )
        self.assertEqual(context.timeline_start_row_index, 81)
        self.assertEqual(context.initial_qpos.shape, (50,))
        self.assertEqual(context.initial_qvel.shape, (49,))
        self.assertFalse(context.initial_qpos.flags.writeable)
        self.assertIsNone(context.teleopit_prefill)
        metadata = context.metadata()
        self.assertEqual(metadata["reference_mode"], "executed_qpos")
        self.assertEqual(metadata["raw_policy_group_offset"], 11)
        self.assertEqual(metadata["resolved_raw_policy_group_offset"], 11)
        self.assertEqual(metadata["excluded_initialization_modes"][0], "zero_padding")
        json.dumps(metadata)

    def test_raw_offset_is_independent_of_complete_first_group(self) -> None:
        complete_root = Path(self.temporary.name) / "complete"
        complete_root.mkdir()
        complete = _write_recording(complete_root, first_group_rows=8)
        reference = load_reference_from_raw_policy_group(
            complete,
            mode="executed_qpos",
            raw_policy_group_offset=DEFAULT_RAW_POLICY_GROUP_OFFSET,
            policy_count=1,
        )
        self.assertEqual(int(reference.policy_seq[0]), 5011)
        self.assertEqual(reference.provenance.selected_policy_offset, 11)
        context = build_source_history_context(
            complete, selected_reference=reference
        )
        self.assertEqual(context.raw_policy_group_offset, 11)

    def test_teleopit_prefill_loads_predecessor_of_oldest_history(self) -> None:
        reference = load_reference(
            self.recording,
            mode="executed_qpos",
            policy_offset=10,
            policy_count=2,
        )
        context = build_source_history_context(
            self.recording,
            selected_reference=reference,
            teleopit_observation_builder=_FakeTeleopitObservationBuilder(),
        )
        self.assertIsNotNone(context.teleopit_prefill)
        assert context.teleopit_prefill is not None
        self.assertEqual(context.teleopit_prefill.prior_observations.shape, (9, 167))
        self.assertEqual(context.teleopit_prefill.takeover_policy_seq, 5011)
        # Entries are 5002..5010; policy 5001 is loaded only to calculate the
        # oldest reference velocity and is retained in provenance.
        self.assertEqual(context.reference_history_first_policy_seq, 5001)
        self.assertEqual(context.reference_history_last_policy_seq, 5011)
        self.assertEqual(
            context.metadata()["teleopit_reference_history_policy_seq"],
            [5001, 5011],
        )

    def test_writes_atomic_native_payload_and_metadata(self) -> None:
        reference = load_reference(
            self.recording,
            mode="executed_qpos",
            policy_offset=10,
            policy_count=1,
        )
        context = build_source_history_context(
            self.recording, selected_reference=reference
        )
        paths = write_source_history_artifacts(
            context, Path(self.temporary.name) / "run"
        )
        payload = json.loads(paths["source_history_prefill"].read_text())
        metadata = json.loads(paths["source_history_context"].read_text())
        self.assertEqual(payload["current"]["policy_seq"], 5011)
        self.assertEqual(metadata["selected_policy_seq"], 5011)
        self.assertEqual(len(paths), 2)

    def test_rejects_takeover_without_nine_predecessors(self) -> None:
        reference = load_reference(
            self.recording,
            mode="executed_qpos",
            policy_offset=0,
            policy_count=1,
        )
        with self.assertRaisesRegex(
            SourceHistoryContextError, "preceding groups"
        ):
            build_source_history_context(
                self.recording, selected_reference=reference
            )


if __name__ == "__main__":
    unittest.main()
