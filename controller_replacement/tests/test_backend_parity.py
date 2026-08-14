"""Tests for fixed-input ONNX Runtime parity artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from controller_replacement.backend_parity import (
    ACTION_DIM,
    BackendParityError,
    BackendUnavailableError,
    DECODER_INPUT_DIM,
    PARITY_COMPLETE_NAME,
    PARITY_OWNED_MARKER,
    TOKEN_DIM,
    _ParityOutputTransaction,
    load_sonic_parity_cases,
    run_ort_parity,
    run_tensorrt_parity,
    select_case_indices,
    tensorrt_backend_status,
)
from controller_replacement.output import (
    ControllerTelemetryWriter,
    TelemetryRecord,
    sha256_file,
    write_json_atomic,
    write_run_manifest,
)


@dataclass(frozen=True)
class _ValueInfo:
    name: str
    shape: tuple[int, int]
    type: str = "tensor(float)"


class _FakeSession:
    def __init__(
        self,
        kind: str,
        input_size: int,
        output_size: int,
        provider: str = "FakeExecutionProvider",
    ) -> None:
        self.kind = kind
        self.input_size = input_size
        self.output_size = output_size
        self.provider = provider

    def get_inputs(self):
        return [_ValueInfo("obs_dict", (1, self.input_size))]

    def get_outputs(self):
        return [_ValueInfo("output", (1, self.output_size))]

    def get_providers(self):
        return [self.provider]

    def run(self, output_names, inputs):
        del output_names
        value = np.asarray(inputs["obs_dict"], dtype=np.float32)
        self.assert_shape(value)
        if self.kind == "encoder":
            output = value[:, :TOKEN_DIM] * np.float32(0.25)
        else:
            output = value[:, :ACTION_DIM] - np.float32(0.125)
        return [np.asarray(output, dtype=np.float32)]

    def assert_shape(self, value: np.ndarray) -> None:
        if value.shape != (1, self.input_size):
            raise AssertionError(value.shape)


class BackendParityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def _make_rollout(
        self,
        *,
        count: int = 22,
        controller_family: str = "sonic",
        perturb_recorded_token_at: int | None = None,
    ) -> tuple[Path, Path, Path]:
        rollout = self.root / "rollout"
        rollout.mkdir()
        encoder = rollout / "model_encoder.onnx"
        decoder = rollout / "model_decoder.onnx"
        encoder.write_bytes(b"synthetic encoder model identity")
        decoder.write_bytes(b"synthetic decoder model identity")
        telemetry = rollout / "policy_telemetry.npz"
        writer = ControllerTelemetryWriter(
            telemetry,
            controller_name="regular",
            controller_family=controller_family,
        )
        for index in range(count):
            encoder_input = (
                np.linspace(-1.0, 1.0, 1247, dtype=np.float32)
                + np.float32(index / 100.0)
            )
            token = encoder_input[:TOKEN_DIM] * np.float32(0.25)
            if perturb_recorded_token_at == index:
                token = token.copy()
                token[0] += np.float32(0.5)
            decoder_input = np.linspace(
                -0.3, 0.3, DECODER_INPUT_DIM, dtype=np.float32
            )
            decoder_input[:TOKEN_DIM] = token
            raw_action = decoder_input[:ACTION_DIM] - np.float32(0.125)
            writer.append(
                TelemetryRecord(
                    policy_seq=900 + index,
                    policy_time_s=index * 0.02,
                    source_row_index=index * 8,
                    warmup=index < 10,
                    evaluation=index >= 10,
                    observation=decoder_input,
                    token=token,
                    raw_action=raw_action,
                    extra_arrays={"encoderInput": encoder_input},
                )
            )
        writer.close()
        write_run_manifest(
            rollout / "run_manifest.json",
            controller_name="regular",
            controller_family=controller_family,
            source_recording=self.root,
            reference_mode="reference_motion",
            root_assist="none",
            rates_hz={"policy": 50.0},
            model_paths={"encoder": encoder, "decoder": decoder},
        )
        write_json_atomic(
            rollout / "run_complete.json",
            {
                "protocol_revision": 2,
                "complete": True,
                "fixed_step_schedule_complete": True,
                "finite_state_complete": True,
                "artifact_sha256": {
                    telemetry.name: sha256_file(telemetry),
                    "run_manifest.json": sha256_file(
                        rollout / "run_manifest.json"
                    ),
                },
            },
        )
        return rollout, encoder, decoder

    @staticmethod
    def _session_factory(path: Path, device: str):
        if device != "cpu":
            raise AssertionError(device)
        if "encoder" in path.name:
            return _FakeSession("encoder", 1247, TOKEN_DIM)
        return _FakeSession("decoder", DECODER_INPUT_DIM, ACTION_DIM)

    @staticmethod
    def _certify_parity_bundle(path: Path, identity: dict[str, object]) -> None:
        (path / PARITY_OWNED_MARKER).write_text("1\n", encoding="utf-8")
        cases = path / "backend_parity_cases.npz"
        report = path / "backend_parity.json"
        cases.write_bytes(b"cases")
        report.write_text("{}\n", encoding="utf-8")
        write_json_atomic(
            path / PARITY_COMPLETE_NAME,
            {
                "format_version": 1,
                "complete": True,
                "identity": identity,
                "artifact_sha256": {
                    cases.name: sha256_file(cases),
                    report.name: sha256_file(report),
                },
            },
        )

    def test_fixed_index_selection_is_stable_and_deduplicated(self) -> None:
        self.assertEqual(select_case_indices(1), (0,))
        self.assertEqual(select_case_indices(10), (0, 5, 9))
        self.assertEqual(select_case_indices(21), (0, 10, 20))
        self.assertEqual(select_case_indices(22), (0, 10, 11, 21))
        with self.assertRaisesRegex(BackendParityError, "no inferences"):
            select_case_indices(0)

    def test_ort_self_check_writes_pickle_free_cases_and_zero_errors(self) -> None:
        rollout, _, _ = self._make_rollout()
        result = run_ort_parity(
            rollout / "policy_telemetry.npz",
            session_factory=self._session_factory,
        )
        analysis = self.root / "analysis" / "rollout_backend_parity_ort_cpu"
        self.assertEqual(result.cases_npz, analysis / "backend_parity_cases.npz")
        self.assertEqual(result.report_json, analysis / "backend_parity.json")
        report = json.loads(result.report_json.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "completed")
        self.assertFalse(report["acceptance_threshold_applied"])
        self.assertEqual(
            [item["telemetry_index"] for item in report["selected_cases"]],
            [0, 10, 11, 21],
        )
        for comparison in report["comparisons"].values():
            self.assertEqual(comparison["aggregate"]["max_abs"], 0.0)
            self.assertTrue(
                comparison["aggregate"]["bitwise_equal_after_float32_cast"]
            )
        self.assertEqual(report["tensorrt"]["status"], "not_available")
        self.assertFalse(report["tensorrt"]["numerical_results_present"])
        self.assertEqual(
            report["source"]["run_complete"],
            str(rollout / "run_complete.json"),
        )
        self.assertEqual(
            report["source"]["run_complete_sha256"],
            sha256_file(rollout / "run_complete.json"),
        )
        completion = json.loads(
            (analysis / PARITY_COMPLETE_NAME).read_text(encoding="utf-8")
        )
        self.assertTrue(completion["complete"])
        self.assertEqual(
            completion["artifact_sha256"]["backend_parity_cases.npz"],
            sha256_file(result.cases_npz),
        )
        self.assertEqual(
            completion["artifact_sha256"]["backend_parity.json"],
            sha256_file(result.report_json),
        )

        with np.load(result.cases_npz, allow_pickle=False) as archive:
            self.assertTrue(all(not archive[name].dtype.hasobject for name in archive.files))
            np.testing.assert_array_equal(archive["telemetry_index"], [0, 10, 11, 21])
            self.assertEqual(archive["encoder_input"].shape, (4, 1247))
            self.assertEqual(archive["decoder_input"].shape, (4, DECODER_INPUT_DIM))
            self.assertEqual(archive["recorded_token"].shape, (4, TOKEN_DIM))
            self.assertEqual(archive["recorded_raw_action"].shape, (4, ACTION_DIM))
            embedded = json.loads(str(archive["metadata_json"].item()))
            self.assertEqual(embedded, report)

    def test_encoder_difference_is_reported_without_inventing_threshold(self) -> None:
        rollout, _, _ = self._make_rollout(perturb_recorded_token_at=10)
        result = run_ort_parity(
            rollout / "policy_telemetry.npz",
            session_factory=self._session_factory,
        )
        comparison = result.report["comparisons"]["recorded_token_vs_ort_encoder"]
        self.assertEqual(comparison["aggregate"]["max_abs"], 0.5)
        self.assertFalse(
            comparison["aggregate"]["bitwise_equal_after_float32_cast"]
        )
        # The recorded decoder input deliberately contains the recorded token,
        # so its causal consistency check remains exact despite the encoder
        # replay mismatch.
        self.assertEqual(
            result.report["comparisons"][
                "recorded_token_vs_decoder_input_token_slice"
            ]["aggregate"]["max_abs"],
            0.0,
        )

    def test_explicit_model_paths_work_without_manifest(self) -> None:
        rollout, encoder, decoder = self._make_rollout(count=2)
        (rollout / "run_manifest.json").unlink()
        result = run_ort_parity(
            rollout / "policy_telemetry.npz",
            encoder_path=encoder,
            decoder_path=decoder,
            session_factory=self._session_factory,
        )
        self.assertIsNone(result.report["source"]["run_manifest"])

    def test_output_cannot_modify_the_rollout_directory(self) -> None:
        rollout, _, _ = self._make_rollout(count=2)
        for destination in (
            rollout,
            rollout / "derived" / "parity",
            self.root,
        ):
            with self.subTest(destination=destination):
                with self.assertRaisesRegex(
                    BackendParityError, "cannot equal, contain, or be contained"
                ):
                    run_ort_parity(
                        rollout / "policy_telemetry.npz",
                        output_directory=destination,
                        session_factory=self._session_factory,
                    )

    def test_output_symlink_is_rejected_before_resolution(self) -> None:
        rollout, _, _ = self._make_rollout(count=2)
        real_output = self.root / "real-output"
        real_output.mkdir()
        linked_output = self.root / "linked-output"
        linked_output.symlink_to(real_output, target_is_directory=True)
        with self.assertRaisesRegex(BackendParityError, "symbolic-link output"):
            run_ort_parity(
                rollout / "policy_telemetry.npz",
                output_directory=linked_output,
                session_factory=self._session_factory,
            )

    def test_output_cannot_contain_external_model_inputs(self) -> None:
        rollout, _, _ = self._make_rollout(count=2)
        (rollout / "run_manifest.json").unlink()
        external = self.root / "external-models"
        external.mkdir()
        encoder = external / "encoder.onnx"
        decoder = external / "decoder.onnx"
        encoder.write_bytes(b"external encoder")
        decoder.write_bytes(b"external decoder")
        with self.assertRaisesRegex(BackendParityError, "contains its encoder model"):
            run_ort_parity(
                rollout / "policy_telemetry.npz",
                encoder_path=encoder,
                decoder_path=decoder,
                output_directory=external,
                session_factory=self._session_factory,
            )

    def test_same_output_is_bound_to_actual_runtime_providers(self) -> None:
        rollout, _, _ = self._make_rollout(count=2)
        destination = self.root / "parity"
        run_ort_parity(
            rollout / "policy_telemetry.npz",
            output_directory=destination,
            session_factory=self._session_factory,
        )

        def alternate_factory(path: Path, device: str):
            if device != "cpu":
                raise AssertionError(device)
            if "encoder" in path.name:
                return _FakeSession(
                    "encoder", 1247, TOKEN_DIM, "AlternateExecutionProvider"
                )
            return _FakeSession(
                "decoder",
                DECODER_INPUT_DIM,
                ACTION_DIM,
                "AlternateExecutionProvider",
            )

        with self.assertRaisesRegex(BackendParityError, "different inputs"):
            run_ort_parity(
                rollout / "policy_telemetry.npz",
                output_directory=destination,
                session_factory=alternate_factory,
            )

    def test_first_publish_interruption_removes_installed_bundle(self) -> None:
        output = self.root / "parity-interrupted"
        identity: dict[str, object] = {"condition": "test"}
        transaction = _ParityOutputTransaction(output, identity=identity)
        with transaction as work:
            self._certify_parity_bundle(work, identity)
            original_replace = Path.replace

            def interrupting_replace(path: Path, target: Path) -> Path:
                result = original_replace(path, target)
                if path == work:
                    raise KeyboardInterrupt("injected after parity install")
                return result

            with mock.patch.object(Path, "replace", new=interrupting_replace):
                with self.assertRaisesRegex(
                    KeyboardInterrupt, "injected after parity install"
                ):
                    transaction.publish()
            self.assertFalse(output.exists())

    def test_formal_rollout_requires_completion_certificate(self) -> None:
        rollout, _, _ = self._make_rollout(count=2)
        (rollout / "run_complete.json").unlink()
        with self.assertRaisesRegex(BackendParityError, "no run_complete.json"):
            run_ort_parity(
                rollout / "policy_telemetry.npz",
                session_factory=self._session_factory,
            )

    def test_completion_certificate_must_bind_exact_telemetry(self) -> None:
        rollout, _, _ = self._make_rollout(count=2)
        write_json_atomic(
            rollout / "run_complete.json",
            {
                "protocol_revision": 2,
                "complete": True,
                "fixed_step_schedule_complete": True,
                "finite_state_complete": True,
                "artifact_sha256": {
                    "policy_telemetry.npz": "0" * 64,
                },
            },
        )
        with self.assertRaisesRegex(
            BackendParityError, "differs from run_complete.json"
        ):
            run_ort_parity(
                rollout / "policy_telemetry.npz",
                session_factory=self._session_factory,
            )

    def test_completion_certificate_must_bind_model_manifest(self) -> None:
        rollout, _, _ = self._make_rollout(count=2)
        manifest_path = rollout / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["root_assist"] = "xy"
        write_json_atomic(manifest_path, manifest)
        with self.assertRaisesRegex(
            BackendParityError, "run_manifest.json SHA-256 differs"
        ):
            run_ort_parity(
                rollout / "policy_telemetry.npz",
                session_factory=self._session_factory,
            )

    def test_incomplete_certificate_is_rejected(self) -> None:
        rollout, _, _ = self._make_rollout(count=2)
        write_json_atomic(
            rollout / "run_complete.json",
            {
                "complete": False,
                "artifact_sha256": {
                    "policy_telemetry.npz": sha256_file(
                        rollout / "policy_telemetry.npz"
                    ),
                },
            },
        )
        with self.assertRaisesRegex(BackendParityError, "does not certify"):
            run_ort_parity(
                rollout / "policy_telemetry.npz",
                session_factory=self._session_factory,
            )

    def test_non_fixed_or_wrong_revision_certificate_is_rejected(self) -> None:
        rollout, _, _ = self._make_rollout(count=2)
        manifest_hash = sha256_file(rollout / "run_manifest.json")
        telemetry_hash = sha256_file(rollout / "policy_telemetry.npz")
        for revision, fixed, finite in (
            (1, True, True),
            (2, False, True),
            (2, True, False),
        ):
            with self.subTest(revision=revision, fixed=fixed, finite=finite):
                write_json_atomic(
                    rollout / "run_complete.json",
                    {
                        "protocol_revision": revision,
                        "complete": True,
                        "fixed_step_schedule_complete": fixed,
                        "finite_state_complete": finite,
                        "artifact_sha256": {
                            "policy_telemetry.npz": telemetry_hash,
                            "run_manifest.json": manifest_hash,
                        },
                    },
                )
                with self.assertRaisesRegex(
                    BackendParityError, "finite fixed-step protocol-2"
                ):
                    run_ort_parity(
                        rollout / "policy_telemetry.npz",
                        session_factory=self._session_factory,
                    )

    def test_non_sonic_telemetry_is_rejected(self) -> None:
        rollout, _, _ = self._make_rollout(controller_family="teleopit")
        with self.assertRaisesRegex(BackendParityError, "requires SONIC"):
            load_sonic_parity_cases(rollout / "policy_telemetry.npz")

    def test_tensorrt_placeholder_never_claims_numerical_results(self) -> None:
        status = tensorrt_backend_status()
        self.assertEqual(status["status"], "not_available")
        self.assertFalse(status["numerical_results_present"])
        with self.assertRaises(BackendUnavailableError):
            run_tensorrt_parity()


if __name__ == "__main__":
    unittest.main()
