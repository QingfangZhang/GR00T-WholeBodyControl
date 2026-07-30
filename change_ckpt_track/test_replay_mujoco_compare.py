from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import mujoco
import numpy as np

try:
    from change_ckpt_track.replay_mujoco_compare import (
        CsvQpos,
        PreparedReference,
        ReplayCompareError,
        _read_prepared_reference,
        _reference_path_from_sidecars,
        _scene_data_id,
        _scene_uses_texture_coordinates,
        build_parser,
        build_ghost_track,
    )
except ModuleNotFoundError:
    from replay_mujoco_compare import (  # type: ignore[no-redef]
        CsvQpos,
        PreparedReference,
        ReplayCompareError,
        _read_prepared_reference,
        _reference_path_from_sidecars,
        _scene_data_id,
        _scene_uses_texture_coordinates,
        build_parser,
        build_ghost_track,
    )


def _csv_qpos(
    *,
    path: str,
    qpos: np.ndarray,
    sample_index: np.ndarray | None = None,
    control_dt: float = 0.005,
    policy_seq: np.ndarray | None = None,
) -> CsvQpos:
    qpos = np.asarray(qpos, dtype=np.float64)
    count = len(qpos)
    return CsvQpos(
        path=Path(path),
        header=("pose[qpos0]",),
        qpos_names=("pose[qpos0]",),
        qpos=qpos,
        sample_index=(
            np.arange(count, dtype=np.int64)
            if sample_index is None
            else np.asarray(sample_index, dtype=np.int64)
        ),
        control_time_s=np.arange(count, dtype=np.float64) * control_dt,
        mujoco_time_s=np.arange(count, dtype=np.float64) * control_dt,
        policy_seq=(
            None
            if policy_seq is None
            else np.asarray(policy_seq, dtype=np.int64)
        ),
    )


class GhostTrackTest(unittest.TestCase):
    def setUp(self) -> None:
        source_values = np.arange(24, dtype=np.float64).reshape(-1, 1)
        self.source = _csv_qpos(
            path="source.csv",
            qpos=source_values,
            control_dt=0.0025,
            policy_seq=np.arange(100, 124, dtype=np.int64),
        )

    def test_reference_mode_holds_each_50hz_frame_for_four_samples(self) -> None:
        target = _csv_qpos(
            path="rollout.csv",
            qpos=np.zeros((10, 1)),
            policy_seq=np.arange(10, dtype=np.int64),
        )
        prepared = PreparedReference(
            path=Path("prepared_reference.npz"),
            source_row_index=np.asarray([2, 10, 18], dtype=np.int64),
            policy_seq=np.asarray([102, 110, 118], dtype=np.int64),
            rate_hz=50.0,
            metadata={},
        )

        ghost = build_ghost_track(
            target=target,
            source=self.source,
            playback_time_s=target.control_time_s,
            run_metadata={},
            manifest={},
            prepared=prepared,
            mode="reference",
            reference_time_offset_s=0.0,
        )

        np.testing.assert_array_equal(
            ghost.reference_frame_index,
            np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 2, 2]),
        )
        np.testing.assert_array_equal(
            ghost.source_row_index,
            np.asarray([2, 2, 2, 2, 10, 10, 10, 10, 18, 18]),
        )
        np.testing.assert_array_equal(
            ghost.qpos[:, 0],
            np.asarray([2, 2, 2, 2, 10, 10, 10, 10, 18, 18]),
        )

    def test_source_mode_uses_recorded_row_formula_and_eof_clamp(self) -> None:
        samples = np.arange(14, dtype=np.int64)
        mapped_rows = np.minimum(2 + samples * 2, 23)
        target = _csv_qpos(
            path="rollout.csv",
            qpos=np.zeros((len(samples), 1)),
            sample_index=samples,
            policy_seq=self.source.policy_seq[mapped_rows],
        )

        ghost = build_ghost_track(
            target=target,
            source=self.source,
            playback_time_s=target.control_time_s,
            run_metadata={
                "initial_row_index": 2,
                "source_rows_per_control": 2,
            },
            manifest={},
            prepared=None,
            mode="source",
            reference_time_offset_s=0.0,
        )

        np.testing.assert_array_equal(ghost.source_row_index, mapped_rows)
        np.testing.assert_array_equal(ghost.qpos[:, 0], mapped_rows)

    def test_source_mode_rejects_wrong_recording(self) -> None:
        target = _csv_qpos(
            path="rollout.csv",
            qpos=np.zeros((3, 1)),
            policy_seq=np.asarray([999, 999, 999]),
        )
        with self.assertRaisesRegex(
            ReplayCompareError, "disagrees with rollout policy_seq"
        ):
            build_ghost_track(
                target=target,
                source=self.source,
                playback_time_s=target.control_time_s,
                run_metadata={
                    "initial_row_index": 2,
                    "source_rows_per_control": 2,
                },
                manifest={},
                prepared=None,
                mode="source",
                reference_time_offset_s=0.0,
            )


class CliCompatibilityTest(unittest.TestCase):
    def test_source_is_default_and_reference_remains_selectable(self) -> None:
        parser = build_parser()
        self.assertEqual(parser.parse_args(["rollout"]).ghost_mode, "source")
        self.assertEqual(
            parser.parse_args(
                ["rollout", "--ghost-mode", "reference"]
            ).ghost_mode,
            "reference",
        )

    def test_legacy_change_ckpt_npz_is_accepted_without_source_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prepared_reference.npz"
            np.savez_compressed(
                path,
                policy_seq=np.asarray([100, 101], dtype=np.int64),
                control_time_s=np.asarray([1.0, 1.02], dtype=np.float64),
                reference_motion=np.zeros((2, 1024), dtype=np.float32),
                joint_pos=np.zeros((2, 29), dtype=np.float32),
                joint_vel=np.zeros((2, 29), dtype=np.float32),
                body_quat_w=np.asarray(
                    [[1.0, 0.0, 0.0, 0.0]] * 2, dtype=np.float32
                ),
            )
            prepared = _read_prepared_reference(path)
            self.assertIsNone(prepared.source_row_index)
            np.testing.assert_array_equal(prepared.policy_seq, [100, 101])
            np.testing.assert_allclose(prepared.control_time_s, [1.0, 1.02])
            self.assertAlmostEqual(prepared.rate_hz, 50.0)

    def test_legacy_reference_mode_recovers_policy_boundary_rows(self) -> None:
        source = _csv_qpos(
            path="source.csv",
            qpos=np.arange(24, dtype=np.float64).reshape(-1, 1),
            control_dt=0.0025,
            policy_seq=np.repeat(
                np.asarray([100, 101, 102], dtype=np.int64), 8
            ),
        )
        target = _csv_qpos(
            path="rollout.csv",
            qpos=np.zeros((10, 1)),
            policy_seq=np.repeat(
                np.asarray([100, 101, 102], dtype=np.int64), [4, 4, 2]
            ),
        )
        prepared = PreparedReference(
            path=Path("legacy_prepared_reference.npz"),
            source_row_index=None,
            policy_seq=np.asarray([100, 101, 102], dtype=np.int64),
            rate_hz=50.0,
            metadata={},
            control_time_s=np.asarray([0.0, 0.02, 0.04]),
        )

        ghost = build_ghost_track(
            target=target,
            source=source,
            playback_time_s=target.control_time_s,
            run_metadata={},
            manifest={},
            prepared=prepared,
            mode="reference",
            reference_time_offset_s=0.0,
        )

        np.testing.assert_array_equal(
            ghost.source_row_index,
            np.asarray([0, 0, 0, 0, 8, 8, 8, 8, 16, 16]),
        )
        self.assertIn("legacy policy_seq recovered", ghost.description)

    def test_stale_sidecar_path_is_uniquely_relocated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relocated = root / "20260612" / "recording_a"
            relocated.mkdir(parents=True)
            (relocated / "data.csv").touch()

            result = _reference_path_from_sidecars(
                root / "rollout",
                {
                    "source_recording": (
                        "/old/location/sample_data/ztj/recording_a"
                    )
                },
                {},
                None,
                relocation_roots=[root],
            )
            self.assertEqual(result, relocated.resolve())


class GhostMeshEncodingTest(unittest.TestCase):
    def test_mesh_asset_id_is_encoded_like_mjv_add_geoms(self) -> None:
        model = SimpleNamespace(
            geom_dataid=np.asarray([28, 7, -1], dtype=np.int32),
            geom_type=np.asarray(
                [
                    int(mujoco.mjtGeom.mjGEOM_MESH),
                    int(mujoco.mjtGeom.mjGEOM_HFIELD),
                    int(mujoco.mjtGeom.mjGEOM_BOX),
                ],
                dtype=np.int32,
            ),
            mesh_texcoordadr=np.full(29, -1, dtype=np.int32),
        )
        model.mesh_texcoordadr[28] = 100

        self.assertEqual(_scene_data_id(model, 0), 56)
        self.assertEqual(_scene_data_id(model, 1), 7)
        self.assertEqual(_scene_data_id(model, 2), -1)
        self.assertEqual(_scene_uses_texture_coordinates(model, 0), 1)
        self.assertEqual(_scene_uses_texture_coordinates(model, 2), 0)


if __name__ == "__main__":
    unittest.main()
