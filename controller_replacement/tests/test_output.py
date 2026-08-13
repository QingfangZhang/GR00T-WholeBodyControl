from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from controller_replacement.output import (
    ControllerTelemetryWriter,
    PolicySnapshot,
    ReplayCsvWriter,
    TelemetryRecord,
    decode_ragged,
    sha256_file,
    write_data_schema,
    write_run_manifest,
)


def _header() -> list[str]:
    return [
        "scene_path",
        "sample_index",
        "control_time_s",
        "mujoco_time_s",
        "qpos:root.x[qpos0]",
        "qpos:joint.angle[qpos1]",
        "qvel:root.vx[qvel0]",
        "qvel:joint.omega[qvel1]",
        "policy_valid",
        "policy_seq",
        "policy_token_size",
        "policy_reference_motion_size",
        "token_state[0]",
        "token_state[1]",
        "token_state[2]",
        "token_state[3]",
        "reference_motion[0]",
        "policy_last_action_in[0]",
        "policy_last_action_in[1]",
        "policy_raw_action_out[0]",
        "policy_raw_action_out[1]",
        "policy_received_dof_pos[0]",
        "policy_received_dof_pos[1]",
        "policy_received_dof_pos[2]",
        "joint_target[0]",
        "joint_target[1]",
        "gripper_close",
    ]


def _write_source(path: Path) -> None:
    header = _header()
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(header)
        for index in range(3):
            row = [str(1000 + column + 100 * index) for column in range(len(header))]
            row[header.index("scene_path")] = "old_scene.xml"
            row[header.index("reference_motion[0]")] = str(90 + index)
            writer.writerow(row)


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", newline="", encoding="utf-8") as source:
        reader = csv.reader(source)
        return next(reader), list(reader)


class ReplayCsvWriterTest(unittest.TestCase):
    def test_sonic_snapshot_is_real_and_held_at_high_rate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.csv"
            output = root / "result.csv"
            _write_source(source)
            snapshot = PolicySnapshot(
                policy_seq=7,
                token=np.asarray([1.25, 2.5]),
                last_action=np.asarray([3.0, 4.0]),
                raw_action=np.asarray([5.0, 6.0]),
                received_dof_pos=np.asarray([7.0, 8.0, 9.0]),
            )
            with ReplayCsvWriter(
                output,
                source,
                controller_family="sonic_v1.1",
                scene_path=root / "new_scene.xml",
            ) as writer:
                self.assertEqual(writer.token_capacity, 4)
                writer.write_frame(
                    source_row_index=0,
                    sample_index=20,
                    control_time_s=0.0,
                    mujoco_time_s=1.0,
                    qpos=[0.1, 0.2],
                    qvel=[0.3, 0.4],
                    command_fields={"joint_target": [0.5, 0.6], "gripper_close": 1},
                    policy_snapshot=snapshot,
                    reference_motion=[12.5],
                )
                # No policy argument: the same 50 Hz inference must be held on
                # the next 400 Hz state row.
                writer.write_frame(
                    source_row_index=1,
                    sample_index=21,
                    control_time_s=0.0025,
                    mujoco_time_s=1.0025,
                    qpos=[0.11, 0.21],
                    qvel=[0.31, 0.41],
                    command_fields={"joint_target": [0.5, 0.6], "gripper_close": 1},
                    reference_motion=[12.5],
                )

            header, rows = _read_csv(output)
            self.assertEqual(header, _header())
            self.assertEqual(len(rows), 2)
            lookup = {name: index for index, name in enumerate(header)}
            for row in rows:
                self.assertEqual(row[lookup["policy_valid"]], "1")
                self.assertEqual(row[lookup["policy_seq"]], "7")
                self.assertEqual(row[lookup["policy_token_size"]], "2")
                self.assertEqual(
                    [float(row[lookup[f"token_state[{index}]"]]) for index in range(4)],
                    [1.25, 2.5, 0.0, 0.0],
                )
                self.assertEqual(
                    [float(row[lookup[f"policy_last_action_in[{index}]"]]) for index in range(2)],
                    [3.0, 4.0],
                )
                self.assertEqual(
                    [float(row[lookup[f"policy_raw_action_out[{index}]"]]) for index in range(2)],
                    [5.0, 6.0],
                )
                self.assertEqual(
                    [float(row[lookup[f"policy_received_dof_pos[{index}]"]]) for index in range(3)],
                    [7.0, 8.0, 9.0],
                )
                self.assertEqual(
                    [float(row[lookup[f"joint_target[{index}]"]]) for index in range(2)],
                    [0.5, 0.6],
                )
            self.assertEqual(float(rows[0][lookup["qpos:root.x[qpos0]"]]), 0.1)
            self.assertEqual(float(rows[1][lookup["qpos:root.x[qpos0]"]]), 0.11)
            self.assertEqual(float(rows[0][lookup["reference_motion[0]"]]), 12.5)
            self.assertEqual(float(rows[1][lookup["reference_motion[0]"]]), 12.5)
            self.assertEqual(rows[0][lookup["policy_reference_motion_size"]], "1")
            self.assertTrue(rows[0][lookup["scene_path"]].endswith("new_scene.xml"))

    def test_sonic_rejects_token_larger_than_source_capacity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.csv"
            _write_source(source)
            writer = ReplayCsvWriter(root / "result.csv", source, controller_family="sonic")
            with self.assertRaisesRegex(ValueError, "capacity is 4"):
                writer.set_policy_snapshot(
                    PolicySnapshot(
                        policy_seq=0,
                        token=np.arange(5),
                        last_action=np.zeros(2),
                        raw_action=np.zeros(2),
                        received_dof_pos=np.zeros(3),
                    )
                )
            writer.abort()

    def test_non_sonic_clears_every_legacy_policy_vector(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.csv"
            output = root / "result.csv"
            _write_source(source)
            with ReplayCsvWriter(output, source, controller_family="teleopit") as writer:
                writer.write_frame(
                    source_row_index=0,
                    sample_index=0,
                    control_time_s=0,
                    mujoco_time_s=0,
                    qpos=[1, 2],
                    qvel=[3, 4],
                    # Even an accidental generic override cannot leak into a
                    # non-SONIC legacy policy field.
                    command_fields={"policy_received_dof_pos": [8, 8, 8]},
                    policy_snapshot=PolicySnapshot(policy_seq=12),
                    clear_reference_motion=True,
                )
            header, rows = _read_csv(output)
            lookup = {name: index for index, name in enumerate(header)}
            row = rows[0]
            self.assertEqual(row[lookup["policy_valid"]], "0")
            self.assertEqual(row[lookup["policy_token_size"]], "0")
            self.assertEqual(row[lookup["policy_seq"]], "12")
            self.assertEqual(row[lookup["policy_reference_motion_size"]], "0")
            self.assertEqual(float(row[lookup["reference_motion[0]"]]), 0.0)
            for prefix, size in (
                ("token_state", 4),
                ("policy_last_action_in", 2),
                ("policy_raw_action_out", 2),
                ("policy_received_dof_pos", 3),
            ):
                self.assertEqual(
                    [float(row[lookup[f"{prefix}[{index}]"]]) for index in range(size)],
                    [0.0] * size,
                )


class TelemetryWriterTest(unittest.TestCase):
    def test_ragged_native_telemetry_and_masks_round_trip_without_pickle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "policy_telemetry.npz"
            with ControllerTelemetryWriter(
                output,
                controller_name="tracker",
                controller_family="teleopit",
                metadata={"checkpoint": "v0.2.0"},
            ) as writer:
                writer.append(
                    TelemetryRecord(
                        policy_seq=10,
                        policy_time_s=0.0,
                        source_row_index=80,
                        reference_index=4,
                        warmup=True,
                        evaluation=False,
                        observation=np.arange(6).reshape(2, 3),
                        history=np.arange(5),
                        token=None,
                        q_target=[0.1, 0.2],
                        torque=[1.0, 2.0],
                        torque_saturation=[False, True],
                        reference=np.arange(4),
                        extra_arrays={"torso_velocity": [1, 2, 3]},
                    )
                )
                writer.append(
                    TelemetryRecord(
                        policy_seq=11,
                        policy_time_s=0.02,
                        source_row_index=88,
                        reference_index=5,
                        warmup=False,
                        evaluation=True,
                        observation=np.arange(3),
                        history=np.arange(8).reshape(2, 4),
                        token=np.arange(2),
                        q_target=[0.3, 0.4],
                        torque=[3.0, 4.0],
                        torque_saturation=[False, False],
                        reference=np.arange(6),
                    )
                )

            with np.load(output, allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["warmup_mask"], [True, False])
                np.testing.assert_array_equal(archive["evaluation_mask"], [False, True])
                np.testing.assert_array_equal(archive["source_row_index"], [80, 88])
                np.testing.assert_array_equal(
                    decode_ragged(archive, "observation", 0), np.arange(6).reshape(2, 3)
                )
                np.testing.assert_array_equal(
                    decode_ragged(archive, "history", 1), np.arange(8).reshape(2, 4)
                )
                self.assertIsNone(decode_ragged(archive, "token", 0))
                np.testing.assert_array_equal(decode_ragged(archive, "token", 1), [0, 1])
                np.testing.assert_array_equal(
                    decode_ragged(archive, "torque_saturation", 0), [False, True]
                )
                np.testing.assert_array_equal(
                    decode_ragged(archive, "extra_torso_velocity", 0), [1, 2, 3]
                )
                self.assertIsNone(decode_ragged(archive, "extra_torso_velocity", 1))


class JsonHelperTest(unittest.TestCase):
    def test_schema_and_manifest_include_contract_and_model_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.csv"
            _write_source(source)
            model = root / "model.onnx"
            model.write_bytes(b"test-model")
            schema = write_data_schema(
                root / "data_schema.json",
                header=_header(),
                controller_family="sonic_regular",
                source_csv=source,
            )
            manifest = write_run_manifest(
                root / "run_manifest.json",
                controller_name="regular",
                controller_family="sonic",
                source_recording=root,
                reference_mode="reference_motion",
                root_assist="xy",
                rates_hz={"physics": 2000, "pd": 200, "policy": 50, "logging": 400},
                model_paths={"encoder": model},
            )
            self.assertEqual(schema["header"], _header())
            self.assertEqual(schema["policy_columns"]["mode"], "sonic_native")
            self.assertEqual(schema["groups"]["token_state"]["size"], 4)
            self.assertEqual(manifest["models"]["encoder"]["sha256"], sha256_file(model))
            self.assertEqual(manifest["root_assist"], "xy")
            with (root / "data_schema.json").open(encoding="utf-8") as source_json:
                self.assertEqual(json.load(source_json)["column_count"], len(_header()))


if __name__ == "__main__":
    unittest.main()
