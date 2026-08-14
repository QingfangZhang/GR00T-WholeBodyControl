from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from controller_replacement.output import sha256_file
from controller_replacement import replay_mujoco_compare as replay


class ReplayContractTest(unittest.TestCase):
    def setUp(self) -> None:
        replay._CERTIFIED_RUN_MANIFEST = None
        replay._CERTIFIED_REPLAY_STATE = None

    def tearDown(self) -> None:
        replay._CERTIFIED_RUN_MANIFEST = None
        replay._CERTIFIED_REPLAY_STATE = None

    def _certified_rollout(self, root: Path) -> Path:
        rollout = root / "rollout"
        rollout.mkdir()
        manifest = {
            "protocol_revision": 2,
            "provenance": {
                "source_recording": {"sha256_before": "a" * 64},
                "staged_scene": {},
            },
        }
        artifacts = {
            "data.csv": "data\n",
            "source_timeline.npz": "sidecar\n",
            "run_metadata.json": "{}\n",
            "launch_manifest.json": "{}\n",
            "prepared_reference.npz": "reference\n",
            "run_manifest.json": json.dumps(manifest, sort_keys=True) + "\n",
        }
        for name, content in artifacts.items():
            (rollout / name).write_text(content, encoding="utf-8")
        hashes = {name: sha256_file(rollout / name) for name in artifacts}
        (rollout / "run_complete.json").write_text(
            json.dumps(
                {
                    "protocol_revision": 2,
                    "complete": True,
                    "fixed_step_schedule_complete": True,
                    "finite_state_complete": True,
                    "artifact_sha256": hashes,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return rollout

    def test_completion_certificate_authenticates_every_consumed_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            rollout = self._certified_rollout(Path(temporary))
            manifest = replay._validate_completion_certificate(
                rollout, ghost_mode="reference"
            )
            self.assertEqual(manifest["protocol_revision"], 2)
            (rollout / "data.csv").write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(
                replay._viewer.ReplayCompareError, "not certified"
            ):
                replay._validate_completion_certificate(
                    rollout, ghost_mode="source"
                )

    def test_source_ghost_requires_exact_certified_source_qpos(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.csv"
            source_path.write_text("source bytes\n", encoding="utf-8")
            target_path = root / "data.csv"
            target_path.write_text("target bytes\n", encoding="utf-8")
            source_hash = sha256_file(source_path)
            replay._CERTIFIED_RUN_MANIFEST = {
                "provenance": {
                    "source_recording": {"sha256_before": source_hash}
                }
            }
            source = SimpleNamespace(
                path=source_path,
                qpos_names=tuple(f"qpos[{index}]" for index in range(8)),
                qpos=np.arange(24, dtype=np.float64).reshape(3, 8),
                policy_seq=np.asarray([10, 10, 11], dtype=np.int64),
            )
            target = SimpleNamespace(
                path=target_path,
                qpos_names=source.qpos_names,
                qpos=np.zeros((2, 8), dtype=np.float64),
                sample_index=np.asarray([0, 1], dtype=np.int64),
            )
            metadata = {
                "format_version": 2,
                "protocol_revision": 2,
                "source_csv_sha256": source_hash,
            }
            sidecar = root / "source_timeline.npz"
            np.savez_compressed(
                sidecar,
                source_row_index=np.asarray([0, 2], dtype=np.int64),
                source_qpos=np.asarray(
                    [[100.0] * 8, [200.0] * 8], dtype=np.float64
                ),
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
            ghost = replay._build_ghost_track(
                mode="source",
                run_metadata={"source_timeline_path": sidecar.name},
                target=target,
                source=source,
            )
            np.testing.assert_array_equal(
                ghost.qpos,
                np.asarray([[100.0] * 8, [200.0] * 8], dtype=np.float64),
            )

            with self.assertRaisesRegex(
                replay._viewer.ReplayCompareError,
                "must be exactly 'source_timeline.npz'",
            ):
                replay._build_ghost_track(
                    mode="source",
                    run_metadata={"source_timeline_path": str(sidecar)},
                    target=target,
                    source=source,
                )

            np.savez_compressed(
                sidecar,
                source_row_index=np.asarray([0, 2], dtype=np.int64),
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
            with self.assertRaisesRegex(
                replay._viewer.ReplayCompareError, "missing exact source_qpos"
            ):
                replay._build_ghost_track(
                    mode="source",
                    run_metadata={"source_timeline_path": sidecar.name},
                    target=target,
                    source=source,
                )

    def test_initial_certificate_identity_is_rechecked_before_viewing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            rollout = self._certified_rollout(Path(temporary))
            manifest = replay._validate_completion_certificate(
                rollout, ghost_mode="source"
            )
            replay._CERTIFIED_RUN_MANIFEST = manifest
            replay._CERTIFIED_REPLAY_STATE = {
                "rollout_dir": str(rollout.resolve(strict=True)),
                "ghost_mode": "source",
                "run_complete_sha256": sha256_file(
                    rollout / "run_complete.json"
                ),
                "run_manifest": manifest,
            }
            replay._revalidate_rollout_certificate(rollout)
            complete_path = rollout / "run_complete.json"
            complete_path.write_text(
                complete_path.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                replay._viewer.ReplayCompareError,
                "run_complete.json changed",
            ):
                replay._revalidate_rollout_certificate(rollout)

    def test_reference_mode_also_checks_certified_source_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "data.csv"
            source.write_text("source\n", encoding="utf-8")
            replay._CERTIFIED_RUN_MANIFEST = {
                "provenance": {
                    "source_recording": {"sha256_before": "0" * 64}
                }
            }
            with self.assertRaisesRegex(
                replay._viewer.ReplayCompareError,
                "certified run provenance",
            ):
                replay._remember_source_csv(source)


if __name__ == "__main__":
    unittest.main()
