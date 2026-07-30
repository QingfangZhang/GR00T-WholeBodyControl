"""Unit tests for qpos-derived SONIC deploy reference preparation."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    from change_ckpt_track.qpos_reference_data import (
        G1_ISAACLAB_JOINT_NAMES,
        G1_MUJOCO_JOINT_NAMES,
        QposReferenceError,
        build_diagnostics,
        load_qpos_reference,
        write_deploy_reference,
    )
except ModuleNotFoundError:
    from qpos_reference_data import (  # type: ignore[no-redef]
        G1_ISAACLAB_JOINT_NAMES,
        G1_MUJOCO_JOINT_NAMES,
        QposReferenceError,
        build_diagnostics,
        load_qpos_reference,
        write_deploy_reference,
    )


class QposReferenceDataTest(unittest.TestCase):
    REAL_RECORDING = (
        Path(__file__).resolve().parents[1]
        / "sample_data"
        / "ztj"
        / "20260612"
        / "20260612_144117_g1_sim"
    )

    def _table(
        self,
        *,
        group_counts: tuple[int, ...] = (2, 8, 8, 8, 2),
        policy_sequences: tuple[int, ...] | None = None,
        frame_dt: float = 0.02,
    ) -> tuple[list[str], list[list[object]]]:
        if policy_sequences is None:
            policy_sequences = tuple(
                100 + index for index in range(len(group_counts))
            )
        self.assertEqual(len(group_counts), len(policy_sequences))

        root_qpos = [
            "qpos:pelvis.floating_base_joint.x[qpos0]",
            "qpos:pelvis.floating_base_joint.y[qpos1]",
            "qpos:pelvis.floating_base_joint.z[qpos2]",
            "qpos:pelvis.floating_base_joint.qw[qpos3]",
            "qpos:pelvis.floating_base_joint.qx[qpos4]",
            "qpos:pelvis.floating_base_joint.qy[qpos5]",
            "qpos:pelvis.floating_base_joint.qz[qpos6]",
        ]
        body_qpos = [
            f"qpos:{name.removesuffix('_joint')}_link.{name}.angle[qpos{7 + index}]"
            for index, name in enumerate(G1_MUJOCO_JOINT_NAMES)
        ]
        root_qvel = [
            f"qvel:pelvis.floating_base_joint.{name}[qvel{index}]"
            for index, name in enumerate(("vx", "vy", "vz", "wx", "wy", "wz"))
        ]
        body_qvel = [
            f"qvel:{name.removesuffix('_joint')}_link.{name}.omega[qvel{6 + index}]"
            for index, name in enumerate(G1_MUJOCO_JOINT_NAMES)
        ]
        hands = [
            *(f"left_hand_q[{index}]" for index in range(7)),
            *(f"right_hand_q[{index}]" for index in range(7)),
        ]
        header = [
            "mujoco_time_s",
            "control_time_s",
            *root_qpos,
            *body_qpos,
            *root_qvel,
            *body_qvel,
            "policy_valid",
            "policy_seq",
            *hands,
            "unrelated",
        ]
        isaac_index = {
            name: index for index, name in enumerate(G1_ISAACLAB_JOINT_NAMES)
        }
        rows: list[list[object]] = []
        source_tick = 0
        for group_index, (sequence, count) in enumerate(
            zip(policy_sequences, group_counts, strict=True)
        ):
            angle = 0.02 * group_index
            quaternion = np.array(
                [np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)]
            )
            # Prove sign continuity is repaired without changing orientation.
            if group_index == 2:
                quaternion *= -1.0
            for subrow in range(count):
                values: dict[str, object] = {
                    "mujoco_time_s": source_tick * 0.0025,
                    "control_time_s": group_index * frame_dt
                    + subrow * 0.0025,
                    "policy_valid": 1,
                    "policy_seq": sequence,
                    "unrelated": 123,
                }
                root = [
                    group_index,
                    group_index + 0.1,
                    0.8 + group_index * 0.01,
                    *quaternion,
                ]
                values.update(zip(root_qpos, root, strict=True))
                values.update(
                    {name: 0.0 for name in root_qvel}
                )
                for source_index, joint_name in enumerate(
                    G1_MUJOCO_JOINT_NAMES
                ):
                    output_index = isaac_index[joint_name]
                    values[body_qpos[source_index]] = (
                        group_index * 100.0
                        + output_index
                        + subrow * 0.001
                    )
                    values[body_qvel[source_index]] = (
                        group_index * 10.0 + output_index
                    )
                for hand_index in range(7):
                    values[f"left_hand_q[{hand_index}]"] = (
                        group_index + hand_index * 0.01
                    )
                    values[f"right_hand_q[{hand_index}]"] = (
                        -group_index - hand_index * 0.01
                    )
                rows.append([values[name] for name in header])
                source_tick += 1
        return header, rows

    @staticmethod
    def _write(
        path: Path, header: list[str], rows: list[list[object]]
    ) -> None:
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)

    def test_load_reorders_by_name_trims_edges_and_tracks_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recording = Path(directory)
            header, rows = self._table()
            self._write(recording / "data.csv", header, rows)
            sequence = load_qpos_reference(recording)

        self.assertEqual(sequence.num_frames, 3)
        np.testing.assert_array_equal(sequence.policy_seq, [101, 102, 103])
        np.testing.assert_array_equal(sequence.group_row_counts, [8, 8, 8])
        np.testing.assert_array_equal(sequence.source_row_indices, [2, 10, 18])
        np.testing.assert_array_equal(
            sequence.source_csv_row_numbers, [4, 12, 20]
        )
        np.testing.assert_allclose(
            sequence.joint_pos[0], 100.0 + np.arange(29)
        )
        np.testing.assert_allclose(
            sequence.joint_vel[1], 20.0 + np.arange(29)
        )
        np.testing.assert_allclose(sequence.left_hand_target[:, 0], [1, 2, 3])
        np.testing.assert_allclose(
            sequence.right_hand_target[:, 0], [-1, -2, -3]
        )
        np.testing.assert_allclose(
            np.linalg.norm(sequence.root_quat_wxyz, axis=1), 1.0
        )
        self.assertTrue(
            np.all(
                np.sum(
                    sequence.root_quat_wxyz[:-1]
                    * sequence.root_quat_wxyz[1:],
                    axis=1,
                )
                >= 0.0
            )
        )
        self.assertEqual(sequence.quaternion_sign_flips_fixed, 1)
        self.assertEqual(
            [(item.side, item.policy_seq, item.row_count)
             for item in sequence.dropped_edge_groups],
            [("first", 100, 2), ("last", 104, 2)],
        )
        self.assertEqual(sequence.start_policy_seq, 101)
        self.assertEqual(sequence.start_source_row_index, 2)

        selected = sequence.slice(1, 1)
        self.assertEqual(selected.num_frames, 1)
        self.assertEqual(selected.start_policy_seq, 102)
        self.assertEqual(selected.start_source_row_index, 10)
        self.assertEqual(selected.frame_offset, 1)

    def test_keep_edges_and_strict_less_than_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "data.csv"
            header, rows = self._table(group_counts=(4, 8, 8, 8, 4))
            self._write(csv_path, header, rows)
            retained = load_qpos_reference(csv_path)
            self.assertEqual(retained.num_frames, 5)
            self.assertFalse(retained.dropped_edge_groups)

            kept = load_qpos_reference(
                csv_path, drop_truncated_edges=False
            )
            self.assertEqual(kept.num_frames, 5)

    def test_diagnostics_and_deploy_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "data.csv"
            header, rows = self._table()
            self._write(csv_path, header, rows)
            sequence = load_qpos_reference(csv_path)
            motion_dir = write_deploy_reference(
                sequence, root / "reference", motion_name="clip"
            )

            expected = {
                "joint_pos.csv",
                "joint_vel.csv",
                "body_pos.csv",
                "body_quat.csv",
                "metadata.txt",
                "frame_map.csv",
                "conversion_report.json",
            }
            self.assertEqual(
                {path.name for path in motion_dir.iterdir()}, expected
            )
            joint_pos = np.loadtxt(
                motion_dir / "joint_pos.csv",
                delimiter=",",
                skiprows=1,
            )
            self.assertEqual(joint_pos.shape, (3, 29))
            np.testing.assert_allclose(joint_pos, sequence.joint_pos)
            root_pos = np.loadtxt(
                motion_dir / "body_pos.csv",
                delimiter=",",
                skiprows=1,
            )
            self.assertEqual(root_pos.shape, (3, 3))
            self.assertIn(
                "Body part indexes:\n[0]",
                (motion_dir / "metadata.txt").read_text(encoding="utf-8"),
            )
            report = json.loads(
                (motion_dir / "conversion_report.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(report["sampling"]["retained_frame_count"], 3)
            self.assertEqual(
                report["column_mapping"]["joint_order"],
                "G1 IsaacLab 29-DOF",
            )
            self.assertTrue(
                report["validation"][
                    "joint_name_mapping_is_complete_and_one_to_one"
                ]
            )
            with (motion_dir / "frame_map.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                mapping = list(csv.DictReader(stream))
            self.assertEqual(mapping[0]["policy_seq"], "101")
            self.assertEqual(mapping[0]["source_row_index"], "2")

            diagnostics = build_diagnostics(sequence.slice(0, 1))
            self.assertIsNone(
                diagnostics["joint_velocity_consistency"][
                    "aggregate_rmse_rad_s"
                ]
            )

    def test_rejects_bad_schema_values_sequence_and_timing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            header, rows = self._table()
            missing_joint = header.index(
                next(
                    name
                    for name in header
                    if ".left_hip_pitch_joint.angle[" in name
                )
            )
            header[missing_joint] = header[missing_joint].replace(
                "left_hip_pitch_joint", "unknown_joint"
            )
            path = root / "missing_joint.csv"
            self._write(path, header, rows)
            with self.assertRaisesRegex(
                QposReferenceError, "missing qpos columns"
            ):
                load_qpos_reference(path)

            header, rows = self._table()
            rows[3].pop()
            path = root / "short_row.csv"
            self._write(path, header, rows)
            with self.assertRaisesRegex(QposReferenceError, "fields; expected"):
                load_qpos_reference(path)

            header, rows = self._table()
            selected = header.index(
                next(
                    name
                    for name in header
                    if ".right_knee_joint.angle[" in name
                )
            )
            rows[3][selected] = "nan"
            path = root / "nonfinite.csv"
            self._write(path, header, rows)
            with self.assertRaisesRegex(QposReferenceError, "non-finite"):
                load_qpos_reference(path)

            header, rows = self._table()
            qw = header.index(
                "qpos:pelvis.floating_base_joint.qw[qpos3]"
            )
            rows[3][qw] = 2.0
            path = root / "bad_quaternion.csv"
            self._write(path, header, rows)
            with self.assertRaisesRegex(
                QposReferenceError, "quaternion norm error"
            ):
                load_qpos_reference(path)

            header, rows = self._table(
                policy_sequences=(100, 101, 103, 104, 105)
            )
            path = root / "sequence_gap.csv"
            self._write(path, header, rows)
            with self.assertRaisesRegex(
                QposReferenceError, "policy_seq is not consecutive"
            ):
                load_qpos_reference(path)

            header, rows = self._table(frame_dt=0.04)
            path = root / "wrong_rate.csv"
            self._write(path, header, rows)
            with self.assertRaisesRegex(
                QposReferenceError, "not consistent with 50 Hz"
            ):
                load_qpos_reference(path)

    def test_rejects_noncontiguous_state_width_and_hand_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header, rows = self._table()
            qpos_column = next(
                index
                for index, name in enumerate(header)
                if name.endswith("[qpos20]")
            )
            header[qpos_column] = header[qpos_column].replace(
                "[qpos20]", "[qpos99]"
            )
            path = root / "qpos_gap.csv"
            self._write(path, header, rows)
            with self.assertRaisesRegex(
                QposReferenceError, "not contiguous"
            ):
                load_qpos_reference(path)

            header, rows = self._table()
            hand_column = header.index("left_hand_q[6]")
            header[hand_column] = "left_hand_q[7]"
            path = root / "hand_gap.csv"
            self._write(path, header, rows)
            with self.assertRaisesRegex(
                QposReferenceError, "must be exactly"
            ):
                load_qpos_reference(path)

    @unittest.skipUnless(
        REAL_RECORDING.joinpath("data.csv").is_file(),
        "real sample recording is not available",
    )
    def test_real_recording_schema_mapping_and_edge_trim(self) -> None:
        sequence = load_qpos_reference(self.REAL_RECORDING)
        self.assertEqual(sequence.num_frames, 363)
        self.assertEqual(sequence.source_header_width, 1564)
        self.assertEqual(sequence.source_qpos_width, 52)
        self.assertEqual(sequence.source_qvel_width, 51)
        np.testing.assert_array_equal(
            sequence.policy_seq[[0, -1]], [19639, 20001]
        )
        np.testing.assert_array_equal(
            sequence.source_row_indices[[0, -1]], [2, 2898]
        )
        self.assertEqual(sequence.joint_pos.shape, (363, 29))
        self.assertEqual(sequence.joint_vel.shape, (363, 29))
        self.assertEqual(sequence.root_quat_wxyz.shape, (363, 4))
        self.assertEqual(sequence.left_hand_target.shape, (363, 7))
        self.assertEqual(
            [
                (item.side, item.policy_seq, item.row_count)
                for item in sequence.dropped_edge_groups
            ],
            [("first", 19638, 2), ("last", 20002, 2)],
        )
        self.assertTrue(
            sequence.joint_qpos_columns[0].endswith(
                ".left_hip_pitch_joint.angle[qpos7]"
            )
        )
        self.assertTrue(
            sequence.joint_qpos_columns[1].endswith(
                ".right_hip_pitch_joint.angle[qpos13]"
            )
        )


if __name__ == "__main__":
    unittest.main()
