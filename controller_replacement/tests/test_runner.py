from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from controller_replacement.runner import (
    LOGS_PER_PD,
    LOGS_PER_POLICY,
    PHYSICS_STEPS_PER_LOG,
    REPO_ROOT,
    RolloutConfig,
    RolloutError,
    _OutputTransaction,
    _OWNED_OUTPUT_MARKER,
    _fit_reference_to_timeline,
    _validate_automatic_scene_reuse,
    _validate_output_path_safety,
    _validate_owned_output,
    _remove_owned_output,
)
from controller_replacement.launch_rollout import (
    _validate_automatic_output_reuse,
    default_output_directory,
)


class RunnerContractTest(unittest.TestCase):
    @staticmethod
    def _certify_complete_output(path: Path) -> None:
        manifest_path = path / "run_manifest.json"
        if not manifest_path.is_file():
            manifest_path.write_text("{}\n", encoding="utf-8")
        manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        (path / "run_complete.json").write_text(
            json.dumps(
                {
                    "protocol_revision": 2,
                    "complete": True,
                    "fixed_step_schedule_complete": True,
                    "finite_state_complete": True,
                    "artifact_sha256": {
                        "run_manifest.json": manifest_sha256
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def _make_complete_output(cls, path: Path, payload: str) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / _OWNED_OUTPUT_MARKER).write_text("1\n", encoding="utf-8")
        cls._certify_complete_output(path)
        (path / "payload.txt").write_text(payload, encoding="utf-8")

    def test_fixed_rate_integer_ratios(self) -> None:
        self.assertEqual(PHYSICS_STEPS_PER_LOG, 5)
        self.assertEqual(LOGS_PER_PD, 2)
        self.assertEqual(LOGS_PER_POLICY, 8)

    def test_reference_horizon_drops_only_incomplete_source_interval(self) -> None:
        class StubReference:
            def __init__(self, frames: int) -> None:
                self.num_frames = frames

            def slice(
                self, policy_offset: int = 0, policy_count: int | None = None
            ) -> "StubReference":
                self.assertions = (policy_offset, policy_count)
                assert policy_count is not None
                return StubReference(policy_count)

        original = StubReference(5)
        complete, complete_metadata = _fit_reference_to_timeline(
            original, available_log_rows=40
        )
        self.assertIs(complete, original)
        self.assertFalse(
            complete_metadata["truncated_to_complete_source_intervals"]
        )

        truncated, metadata = _fit_reference_to_timeline(
            original, available_log_rows=39
        )
        self.assertEqual(truncated.num_frames, 4)
        self.assertEqual(metadata["requested_reference_frames"], 5)
        self.assertEqual(metadata["executed_reference_frames"], 4)
        self.assertEqual(metadata["source_400hz_rows_consumed"], 32)
        self.assertEqual(metadata["source_400hz_rows_unconsumed"], 7)
        self.assertTrue(metadata["truncated_to_complete_source_intervals"])

        with self.assertRaisesRegex(RolloutError, "fewer than eight"):
            _fit_reference_to_timeline(original, available_log_rows=7)

    def test_output_overwrite_refuses_unowned_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result"
            output.mkdir()
            (output / "user.txt").write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(RolloutError, "unmarked"):
                _remove_owned_output(output)
            self.assertTrue((output / "user.txt").is_file())

    def test_output_overwrite_accepts_own_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result"
            output.mkdir()
            (output / _OWNED_OUTPUT_MARKER).write_text("1\n", encoding="utf-8")
            _remove_owned_output(output)
            self.assertFalse(output.exists())

    def test_output_marker_symlink_never_authorizes_recursive_delete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "result"
            output.mkdir()
            authority = root / "authority.txt"
            authority.write_text("1\n", encoding="utf-8")
            (output / _OWNED_OUTPUT_MARKER).symlink_to(authority)
            with self.assertRaisesRegex(RolloutError, "unmarked"):
                _remove_owned_output(output)
            self.assertTrue(output.is_dir())

    def test_output_validation_keeps_previous_complete_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result"
            output.mkdir()
            (output / _OWNED_OUTPUT_MARKER).write_text("1\n", encoding="utf-8")
            previous = output / "run_complete.json"
            previous.write_text("{}\n", encoding="utf-8")
            _validate_owned_output(output)
            self.assertTrue(previous.is_file())

    def test_transaction_refuses_marker_only_final_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result"
            output.mkdir()
            (output / _OWNED_OUTPUT_MARKER).write_text("1\n", encoding="utf-8")
            with self.assertRaisesRegex(RolloutError, "incomplete marked output"):
                with _OutputTransaction(output):
                    pass
            self.assertTrue(output.is_dir())

    def test_transaction_refuses_false_completion_certificate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result"
            output.mkdir()
            (output / _OWNED_OUTPUT_MARKER).write_text("1\n", encoding="utf-8")
            (output / "run_manifest.json").write_text("{}\n", encoding="utf-8")
            (output / "run_complete.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(RolloutError, "not certified"):
                with _OutputTransaction(output):
                    pass
            self.assertTrue(output.is_dir())

    def test_transaction_refuses_terminal_output_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            self._make_complete_output(target, "keep")
            output = root / "result"
            output.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(RolloutError, "symbolic link"):
                _OutputTransaction(output)
            self.assertEqual(
                (target / "payload.txt").read_text(encoding="utf-8"),
                "keep",
            )

    def test_same_output_transaction_lock_is_nonblocking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result"
            with _OutputTransaction(output) as first:
                self.assertIsNotNone(first.work_output)
                with self.assertRaisesRegex(RolloutError, "already writing"):
                    with _OutputTransaction(output):
                        pass

    def test_existing_output_identity_validator_runs_under_transaction_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result"
            self._make_complete_output(output, "old")
            calls: list[Path] = []

            def validator(path: Path) -> None:
                calls.append(path)
                self.assertTrue((path / "payload.txt").is_file())

            with _OutputTransaction(
                output, existing_output_validator=validator
            ) as transaction:
                self.assertEqual(calls, [output.resolve()])
                self.assertIsNotNone(transaction.work_output)

    def test_transaction_uses_unique_work_and_cleans_on_exception(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result"
            work: Path | None = None
            with self.assertRaisesRegex(RuntimeError, "injected"):
                with _OutputTransaction(output) as transaction:
                    work = transaction.work_output
                    self.assertIsNotNone(work)
                    assert work is not None
                    self.assertEqual(work.parent, output.parent)
                    self.assertNotEqual(work, output.with_name(output.name + ".incomplete"))
                    raise RuntimeError("injected")
            assert work is not None
            self.assertFalse(work.exists())
            # Cleanup also released the advisory lock.
            with _OutputTransaction(output) as second:
                self.assertIsNotNone(second.work_output)
                self.assertNotEqual(second.work_output, work)

    def test_publish_failure_restores_previous_complete_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result"
            self._make_complete_output(output, "old")
            with _OutputTransaction(output) as transaction:
                work = transaction.work_output
                assert work is not None
                self._certify_complete_output(work)
                (work / "payload.txt").write_text("new", encoding="utf-8")
                original_replace = Path.replace

                def injected_replace(path: Path, target: Path) -> Path:
                    if path == work and Path(target) == output:
                        raise OSError("injected publish rename failure")
                    return original_replace(path, target)

                with mock.patch.object(Path, "replace", new=injected_replace):
                    with self.assertRaisesRegex(OSError, "injected publish"):
                        transaction.publish()
                self.assertEqual(
                    (output / "payload.txt").read_text(encoding="utf-8"),
                    "old",
                )
            self.assertFalse(any(output.parent.glob(".result.backup-*")))

    def test_interruption_after_old_output_rename_restores_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result"
            self._make_complete_output(output, "old")
            with _OutputTransaction(output) as transaction:
                work = transaction.work_output
                assert work is not None
                self._certify_complete_output(work)
                (work / "payload.txt").write_text("new", encoding="utf-8")
                original_replace = Path.replace

                def interrupting_replace(path: Path, target: Path) -> Path:
                    result = original_replace(path, target)
                    if path == output:
                        raise KeyboardInterrupt("injected after old rename")
                    return result

                with mock.patch.object(Path, "replace", new=interrupting_replace):
                    with self.assertRaisesRegex(
                        KeyboardInterrupt, "injected after old rename"
                    ):
                        transaction.publish()
                self.assertEqual(
                    (output / "payload.txt").read_text(encoding="utf-8"),
                    "old",
                )
            self.assertFalse(any(output.parent.glob(".result.backup-*")))

    def test_interruption_after_first_publish_rename_removes_partial_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result"
            with _OutputTransaction(output) as transaction:
                work = transaction.work_output
                assert work is not None
                self._certify_complete_output(work)
                original_replace = Path.replace

                def interrupting_replace(path: Path, target: Path) -> Path:
                    result = original_replace(path, target)
                    if path == work:
                        raise KeyboardInterrupt("injected after install rename")
                    return result

                with mock.patch.object(Path, "replace", new=interrupting_replace):
                    with self.assertRaisesRegex(
                        KeyboardInterrupt, "injected after install rename"
                    ):
                        transaction.publish()
                self.assertFalse(output.exists())

    def test_successful_transaction_replaces_complete_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result"
            self._make_complete_output(output, "old")
            with _OutputTransaction(output) as transaction:
                work = transaction.work_output
                assert work is not None
                self._certify_complete_output(work)
                (work / "payload.txt").write_text("new", encoding="utf-8")
                transaction.publish()
            self.assertEqual(
                (output / "payload.txt").read_text(encoding="utf-8"),
                "new",
            )
            self.assertFalse(any(output.parent.glob(".result.backup-*")))

    def test_output_path_must_not_overlap_inputs_or_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            recording = root / "recording"
            recording.mkdir()
            source_csv = recording / "data.csv"
            source_csv.write_text("qpos[0]\n", encoding="utf-8")
            models = root / "models"
            models.mkdir()
            model = models / "controller.onnx"
            model.write_bytes(b"model")
            safe_output = root / "results" / "run"
            _validate_output_path_safety(
                safe_output,
                recording_dir=recording,
                source_csv=source_csv,
                model_paths={"controller": model},
            )
            unsafe = {
                "recording itself": recording,
                "inside recording": recording / "result",
                "contains recording": root,
                "contains model": models,
            }
            for label, output in unsafe.items():
                with self.subTest(label=label):
                    with self.assertRaisesRegex(RolloutError, "unsafe --output"):
                        _validate_output_path_safety(
                            output,
                            recording_dir=recording,
                            source_csv=source_csv,
                            model_paths={"controller": model},
                        )
            with self.assertRaisesRegex(RolloutError, "repository root"):
                _validate_output_path_safety(
                    REPO_ROOT,
                    recording_dir=recording,
                    source_csv=source_csv,
                    model_paths={"controller": model},
                )
            assets = root / "assets"
            assets.mkdir()
            with self.assertRaisesRegex(RolloutError, "asset-model root overlap"):
                _validate_output_path_safety(
                    assets / "result",
                    recording_dir=recording,
                    source_csv=source_csv,
                    model_paths={"controller": model},
                    asset_model_root=assets,
                )

    def test_formal_defaults_and_sensitivity_output_names_are_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            recording = Path(temporary) / "task"
            recording.mkdir()
            root = Path(temporary) / "outputs"
            formal = default_output_directory(
                recording,
                controller="regular",
                reference_mode="reference_motion",
                root_assist="none",
                output_root=root,
            )
            sensitivity = default_output_directory(
                recording,
                controller="regular",
                reference_mode="reference_motion",
                root_assist="none",
                output_root=root,
                hand_torque_profile="staged_xml",
                fall_height_m=0.25,
            )
            self.assertEqual(
                formal.name,
                "task_regular_reference_motion_protocol2",
            )
            self.assertIn("hand_staged_xml", sensitivity.name)
            self.assertIn("fall_z_0p25m", sensitivity.name)
            diagnostic = default_output_directory(
                recording,
                controller="regular",
                reference_mode="reference_motion",
                root_assist="none",
                output_root=root,
                raw_policy_group_offset=14,
                policy_count=12,
                device="cuda",
            )
            self.assertIn("raw_offset_14", diagnostic.name)
            self.assertIn("policy_count_12", diagnostic.name)
            self.assertTrue(diagnostic.name.endswith("_cuda"))
            viewer = default_output_directory(
                Path("/tmp/recording"),
                controller="regular",
                reference_mode="reference_motion",
                root_assist="none",
                output_root=Path("/tmp/output"),
                viewer=True,
            )
            self.assertTrue(viewer.name.endswith("_viewer"))

    def test_rollout_config_rejects_unnamed_hand_or_invalid_fall_profile(self) -> None:
        common = {
            "recording": Path("recording"),
            "output_directory": Path("output"),
            "reference": object(),
        }
        with self.assertRaisesRegex(ValueError, "hand_torque_profile"):
            RolloutConfig(**common, hand_torque_profile="mystery")
        with self.assertRaisesRegex(ValueError, "fall_height_m"):
            RolloutConfig(**common, fall_height_m=0.0)

    def test_automatic_output_reuse_binds_source_and_model_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recording = root / "recording"
            output = root / "output"
            recording.mkdir()
            output.mkdir()
            expected_condition = {
                "controller_name": "regular",
                "controller_family": "sonic",
                "reference_mode": "reference_motion",
                "root_assist": "none",
                "hand_torque_profile": "sonic_release",
                "fall_height_m": 0.2,
                "raw_policy_group_offset": 11,
                "selected_policy_count": 120,
                "requested_device": "cpu",
                "viewer": False,
            }
            manifest = {
                "protocol_revision": 2,
                "controller_name": "regular",
                "controller_family": "sonic",
                "source_recording": str(recording.resolve()),
                "reference_mode": "reference_motion",
                "root_assist": "none",
                "provenance": {
                    "source_recording": {"sha256_before": "a" * 64}
                },
                "models": {"encoder": {"sha256": "b" * 64}},
                "extra": {
                    "hand_control": {"profile": "sonic_release"},
                    "fall_detection": {"threshold_m": 0.2},
                    "source_history": {"raw_policy_group_offset": 11},
                    "reference": {"selected_frames": 120},
                    "controller": {
                        "inference_runtime": {"requested_device": "cpu"}
                    },
                    "viewer": {"enabled": False},
                },
            }

            def write_certified(value: dict[str, object]) -> None:
                manifest_bytes = json.dumps(value, sort_keys=True).encode("utf-8")
                (output / "run_manifest.json").write_bytes(manifest_bytes)
                (output / "run_complete.json").write_text(
                    json.dumps(
                        {
                            "protocol_revision": 2,
                            "complete": True,
                            "artifact_sha256": {
                                "run_manifest.json": hashlib.sha256(
                                    manifest_bytes
                                ).hexdigest()
                            },
                        }
                    ),
                    encoding="utf-8",
                )

            write_certified(manifest)
            _validate_automatic_output_reuse(
                output,
                recording,
                source_csv_sha256="a" * 64,
                model_sha256={"encoder": "b" * 64},
                expected_condition=expected_condition,
            )
            with self.assertRaisesRegex(ValueError, "different source CSV bytes"):
                _validate_automatic_output_reuse(
                    output,
                    recording,
                    source_csv_sha256="c" * 64,
                    model_sha256={"encoder": "b" * 64},
                    expected_condition=expected_condition,
                )

            condition_mutations = (
                (("controller_name",), "low_latency"),
                (("controller_family",), "teleopit"),
                (("reference_mode",), "executed_qpos"),
                (("root_assist",), "xy"),
                (("extra", "hand_control", "profile"), "staged_xml"),
                (("extra", "fall_detection", "threshold_m"), 0.20000001),
                (("extra", "source_history", "raw_policy_group_offset"), 12),
                (("extra", "reference", "selected_frames"), 119),
                (("extra", "viewer", "enabled"), True),
                (
                    (
                        "extra",
                        "controller",
                        "inference_runtime",
                        "requested_device",
                    ),
                    "cuda",
                ),
            )
            for path, replacement in condition_mutations:
                with self.subTest(condition_path=path):
                    changed = json.loads(json.dumps(manifest))
                    parent = changed
                    for key in path[:-1]:
                        parent = parent[key]
                    parent[path[-1]] = replacement
                    write_certified(changed)
                    with self.assertRaisesRegex(
                        ValueError, "different experiment conditions"
                    ):
                        _validate_automatic_output_reuse(
                            output,
                            recording,
                            source_csv_sha256="a" * 64,
                            model_sha256={"encoder": "b" * 64},
                            expected_condition=expected_condition,
                        )

            write_certified(manifest)
            (output / "run_manifest.json").write_text(
                json.dumps({**manifest, "root_assist": "xy"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "completion certificate"):
                _validate_automatic_output_reuse(
                    output,
                    recording,
                    source_csv_sha256="a" * 64,
                    model_sha256={"encoder": "b" * 64},
                    expected_condition=expected_condition,
                )

    def test_automatic_scene_reuse_binds_compiled_model_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result"
            output.mkdir()
            fingerprint = {
                "serialization": "MuJoCo MJB memory buffer",
                "mujoco_version": "3.10.0",
                "size_bytes": 123,
                "sha256": "a" * 64,
            }
            (output / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "provenance": {
                            "staged_scene": {
                                "compiled_mujoco_model": fingerprint
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            _validate_automatic_scene_reuse(output, fingerprint)
            changed = {**fingerprint, "sha256": "b" * 64}
            with self.assertRaisesRegex(RolloutError, "different compiled scene"):
                _validate_automatic_scene_reuse(output, changed)


if __name__ == "__main__":
    unittest.main()
