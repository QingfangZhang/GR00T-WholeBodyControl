"""Single-process deterministic controller-replacement rollout engine."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import errno
import fcntl
import json
import os
import shutil
import stat
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping
import uuid

import mujoco
import numpy as np

from change_ckpt_track.task_sim_io import (
    resolve_recording,
    stage_recording_snapshot,
)
from controller_replacement.controllers import SonicController, TeleopitController
from controller_replacement.metrics import (
    DEFAULT_FALL_HEIGHT_M,
    RolloutMetrics,
    rot6d_orientation_error_rad,
)
from controller_replacement.history import write_source_history_artifacts
from controller_replacement.output import (
    ControllerTelemetryWriter,
    PolicySnapshot,
    ReplayCsvWriter,
    TelemetryRecord,
    sha256_file,
    write_data_schema,
    write_json_atomic,
    write_run_manifest,
)
from controller_replacement.provenance import (
    artifact_hashes,
    compiled_mujoco_model_fingerprint,
    git_provenance,
    runtime_environment,
    snapshot_symlink_targets,
    snapshot_xml_hashes,
    source_tree_hashes,
)
from controller_replacement.references import ReferenceSequence
from controller_replacement.simulator import (
    LOG_HZ,
    PD_HZ,
    PHYSICS_HZ,
    POLICY_HZ,
    AppliedPdCommand,
    DeterministicTaskScene,
)
from controller_replacement.timeline import CsvTimeline


PHYSICS_STEPS_PER_LOG = int(PHYSICS_HZ / LOG_HZ)
LOGS_PER_PD = int(LOG_HZ / PD_HZ)
LOGS_PER_POLICY = int(LOG_HZ / POLICY_HZ)
SELF_WARMUP_INFERENCES = 10
REPO_ROOT = Path(__file__).resolve().parents[1]
_BEHAVIOR_SOURCE_DIRECTORIES = (
    "controller_replacement",
    "change_ckpt",
    "change_ckpt_track",
    "Teleopit_rollout",
)


class RolloutError(RuntimeError):
    """A rollout contract or deterministic timing invariant was violated."""


_OWNED_OUTPUT_MARKER = ".controller_replacement_output"


def _validate_owned_output(path: Path) -> None:
    """Reject replacement of anything not previously created by this runner."""

    if not path.exists():
        return
    if path.is_symlink() or not path.is_dir():
        raise RolloutError(f"refusing to replace non-directory output: {path}")
    marker = path / _OWNED_OUTPUT_MARKER
    if (
        marker.is_symlink()
        or not marker.is_file()
        or marker.read_text(encoding="utf-8").strip() != "1"
    ):
        raise RolloutError(
            "refusing to delete an unmarked directory; choose another --output "
            f"or remove it yourself: {path}"
        )


def _remove_owned_output(path: Path) -> None:
    """Remove only a directory previously created by this runner."""

    _validate_owned_output(path)
    if not path.exists():
        return
    shutil.rmtree(path)


def _validate_replaceable_output(path: Path) -> None:
    """Require an existing final output to be both owned and complete.

    ``_remove_owned_output`` deliberately retains its older marker-only
    contract because it is also useful for cleaning a work directory created
    by this process.  Replacing a *published* result is stricter: a marker by
    itself must never authorize overwriting an interrupted or hand-crafted
    directory.
    """

    _validate_owned_output(path)
    if not path.exists():
        return
    missing = [
        name
        for name in ("run_complete.json", "run_manifest.json")
        if (path / name).is_symlink() or not (path / name).is_file()
    ]
    if missing:
        raise RolloutError(
            "refusing to replace an incomplete marked output; missing "
            f"{', '.join(missing)}: {path}"
        )
    complete_path = path / "run_complete.json"
    manifest_path = path / "run_manifest.json"
    try:
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RolloutError(
            f"refusing to replace output with unreadable completion certificate: {path}"
        ) from exc
    if not isinstance(complete, dict):
        raise RolloutError(
            f"refusing to replace output with malformed completion certificate: {path}"
        )
    try:
        revision = int(complete.get("protocol_revision", -1))
    except (TypeError, ValueError):
        revision = -1
    if not (
        revision == 2
        and complete.get("complete") is True
        and complete.get("fixed_step_schedule_complete") is True
        and complete.get("finite_state_complete") is True
    ):
        raise RolloutError(
            f"refusing to replace output not certified as a complete protocol-2 run: {path}"
        )
    certified_artifacts = complete.get("artifact_sha256")
    certified_manifest_sha256 = (
        certified_artifacts.get("run_manifest.json")
        if isinstance(certified_artifacts, Mapping)
        else None
    )
    if certified_manifest_sha256 != sha256_file(manifest_path):
        raise RolloutError(
            "refusing to replace output whose run_manifest.json is not bound "
            f"by run_complete.json: {path}"
        )


def _path_contains(container: Path, candidate: Path) -> bool:
    """Return whether ``candidate`` is equal to or below ``container``."""

    container = container.expanduser().resolve()
    candidate = candidate.expanduser().resolve()
    return candidate == container or container in candidate.parents


def _validate_output_path_safety(
    output: Path,
    *,
    recording_dir: Path,
    source_csv: Path,
    model_paths: Mapping[str, Path],
    asset_model_root: Path | None = None,
) -> None:
    """Reject output/input overlap before creating, deleting, or locking data."""

    output = output.expanduser().resolve()
    recording_dir = recording_dir.expanduser().resolve()
    source_csv = source_csv.expanduser().resolve()
    repo_root = REPO_ROOT.resolve()

    if _path_contains(output, repo_root):
        raise RolloutError(
            "unsafe --output: it equals or contains the repository root: "
            f"{output}"
        )
    if _path_contains(output, recording_dir) or _path_contains(
        recording_dir, output
    ):
        raise RolloutError(
            "unsafe --output: output and source recording directories overlap: "
            f"output={output}, recording={recording_dir}"
        )
    if _path_contains(output, source_csv):
        raise RolloutError(
            "unsafe --output: it equals or contains the source CSV: "
            f"output={output}, source_csv={source_csv}"
        )
    for name, model_path in model_paths.items():
        resolved_model = Path(model_path).expanduser().resolve()
        if _path_contains(output, resolved_model):
            raise RolloutError(
                "unsafe --output: it equals or contains a controller model "
                f"artifact ({name}): output={output}, model={resolved_model}"
            )
    if asset_model_root is not None:
        assets = Path(asset_model_root).expanduser().resolve()
        if _path_contains(output, assets) or _path_contains(assets, output):
            raise RolloutError(
                "unsafe --output: output and custom asset-model root overlap: "
                f"output={output}, assets={assets}"
            )


def _validate_automatic_scene_reuse(
    output: Path, current_compiled_model: Mapping[str, Any]
) -> None:
    """Bind automatic-name reuse to the exact compiled scene condition.

    This runs while the output lock is held and after the current recording
    snapshot/assets have been compiled.  It therefore catches XML or external
    asset changes that path/source/controller checks alone cannot see.
    """

    if not output.exists():
        return
    manifest_path = output / "run_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RolloutError(
            "cannot validate the existing automatic output's compiled scene; "
            f"choose an explicit --output: {output}"
        ) from exc
    provenance = manifest.get("provenance", {}) if isinstance(manifest, dict) else {}
    staged_scene = (
        provenance.get("staged_scene", {})
        if isinstance(provenance, Mapping)
        else {}
    )
    recorded = (
        staged_scene.get("compiled_mujoco_model")
        if isinstance(staged_scene, Mapping)
        else None
    )
    if not isinstance(recorded, Mapping) or dict(recorded) != dict(
        current_compiled_model
    ):
        raise RolloutError(
            "automatic output was produced with different compiled scene/XML/"
            f"asset bytes; choose an explicit --output: {output}"
        )


class _OutputTransaction:
    """Serialize and failure-atomically publish one rollout output directory."""

    def __init__(
        self,
        output: Path,
        *,
        existing_output_validator: Callable[[Path], None] | None = None,
    ) -> None:
        raw_output = output.expanduser()
        if raw_output.is_symlink():
            raise RolloutError(
                f"refusing to use a symbolic link as the final output: {raw_output}"
            )
        self.output = raw_output.resolve()
        self.work_output: Path | None = None
        self.backup_output: Path | None = None
        self._lock_fd: int | None = None
        self._published = False
        self._existing_output_validator = existing_output_validator

    @property
    def lock_path(self) -> Path:
        # The lock file is intentionally persistent.  Unlinking a lock file
        # while waiters have it open can create two independently locked
        # inodes for the same logical output.
        return self.output.parent / f".{self.output.name}.lock"

    def _release_lock(self) -> None:
        if self._lock_fd is None:
            return
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(self._lock_fd)
            self._lock_fd = None

    def __enter__(self) -> "_OutputTransaction":
        self.output.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        self._lock_fd = os.open(self.lock_path, flags, 0o600)
        if not stat.S_ISREG(os.fstat(self._lock_fd).st_mode):
            self._release_lock()
            raise RolloutError(
                f"output lock path is not a regular file: {self.lock_path}"
            )
        try:
            fcntl.flock(
                self._lock_fd,
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except OSError as exc:
            self._release_lock()
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise RolloutError(
                    "another rollout is already writing this output: "
                    f"{self.output}"
                ) from exc
            raise

        try:
            _validate_replaceable_output(self.output)
            if self._existing_output_validator is not None:
                self._existing_output_validator(self.output)
            work = Path(
                tempfile.mkdtemp(
                    prefix=f".{self.output.name}.work-",
                    dir=self.output.parent,
                )
            )
            self.work_output = work
            (work / _OWNED_OUTPUT_MARKER).write_text("1\n", encoding="utf-8")
        except BaseException:
            if self.work_output is not None and self.work_output.exists():
                # This path was uniquely created by mkdtemp in this context;
                # it is safe to clean even if marker creation itself failed.
                shutil.rmtree(self.work_output, ignore_errors=True)
            self.work_output = None
            self._release_lock()
            raise
        return self

    def _unique_backup_path(self) -> Path:
        while True:
            candidate = self.output.parent / (
                f".{self.output.name}.backup-{uuid.uuid4().hex}"
            )
            if not candidate.exists() and not candidate.is_symlink():
                return candidate

    def publish(self) -> None:
        """Atomically install work, restoring the old result on rename failure."""

        if self.work_output is None:
            raise RolloutError("output transaction has no work directory")
        if self._published:
            raise RolloutError("output transaction was already published")
        _validate_replaceable_output(self.work_output)
        _validate_replaceable_output(self.output)

        try:
            if self.output.exists():
                self.backup_output = self._unique_backup_path()
                self.output.replace(self.backup_output)
            self.work_output.replace(self.output)
        except BaseException as publish_error:
            if self.backup_output is not None and self.backup_output.exists():
                try:
                    # An interruption can arrive just after the work directory
                    # was installed but before the transaction was marked as
                    # published.  Remove only that owned result, then restore
                    # the previous complete output.
                    if self.output.exists() or self.output.is_symlink():
                        _remove_owned_output(self.output)
                    self.backup_output.replace(self.output)
                    self.backup_output = None
                except BaseException as restore_error:
                    raise RolloutError(
                        "publishing the new output failed and restoring the old "
                        "output also failed; the old result is preserved at "
                        f"{self.backup_output}"
                    ) from restore_error
            elif (
                self.backup_output is None
                and self.work_output is not None
                and not self.work_output.exists()
                and self.output.exists()
            ):
                # There was no previous result, but the install rename
                # completed before an asynchronous exception was delivered.
                # Remove only the newly installed owned directory so an
                # interrupted command never advertises success by presence.
                _remove_owned_output(self.output)
            raise publish_error

        self.work_output = None
        self._published = True
        if self.backup_output is not None:
            _remove_owned_output(self.backup_output)
            self.backup_output = None

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            # Defense against a BaseException arriving between the two rename
            # operations in ``publish``.  A failed transaction must leave the
            # previous published result at its original path, never stranded
            # under a private backup name.
            if (
                not self._published
                and self.backup_output is not None
                and self.backup_output.exists()
            ):
                if self.output.exists() or self.output.is_symlink():
                    _remove_owned_output(self.output)
                self.backup_output.replace(self.output)
                self.backup_output = None
            elif (
                not self._published
                and self.backup_output is None
                and self.work_output is not None
                and not self.work_output.exists()
                and self.output.exists()
            ):
                _remove_owned_output(self.output)
            if self.work_output is not None and self.work_output.exists():
                # Only this transaction knows the random work path.  Retain
                # the marker check as defense in depth before recursive delete.
                _remove_owned_output(self.work_output)
                self.work_output = None
        finally:
            self._release_lock()


@dataclass(frozen=True)
class RolloutConfig:
    recording: Path
    output_directory: Path
    reference: ReferenceSequence
    root_assist: str = "none"
    viewer: bool = False
    asset_model_root: Path | None = None
    self_warmup_inferences: int = SELF_WARMUP_INFERENCES
    hand_torque_profile: str = "sonic_release"
    fall_height_m: float = DEFAULT_FALL_HEIGHT_M
    expected_source_csv_sha256: str | None = None
    expected_behavior_source_hashes: Mapping[str, Mapping[str, str]] | None = None
    automatic_output: bool = False
    existing_output_validator: Callable[[Path], None] | None = None

    def __post_init__(self) -> None:
        if self.root_assist not in {"none", "xy"}:
            raise ValueError("root_assist must be 'none' or 'xy'")
        if self.self_warmup_inferences != SELF_WARMUP_INFERENCES:
            raise ValueError(
                "the formal protocol fixes self-warmup at ten controller inferences"
            )
        if self.hand_torque_profile not in {"sonic_release", "staged_xml"}:
            raise ValueError(
                "hand_torque_profile must be 'sonic_release' or 'staged_xml'"
            )
        if not np.isfinite(self.fall_height_m) or self.fall_height_m <= 0.0:
            raise ValueError("fall_height_m must be finite and positive")
        if self.expected_source_csv_sha256 is not None:
            value = str(self.expected_source_csv_sha256)
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError("expected_source_csv_sha256 must be lowercase SHA-256")
        if self.expected_behavior_source_hashes is not None and not isinstance(
            self.expected_behavior_source_hashes, Mapping
        ):
            raise ValueError("expected_behavior_source_hashes must be a mapping")
        if self.automatic_output and self.existing_output_validator is None:
            raise ValueError(
                "automatic_output requires an under-lock existing-output validator"
            )
        if self.existing_output_validator is not None and not callable(
            self.existing_output_validator
        ):
            raise ValueError("existing_output_validator must be callable")


@dataclass(frozen=True)
class RolloutResult:
    output_directory: Path
    data_csv: Path
    policy_telemetry: Path
    metrics_json: Path
    policy_inferences: int
    logged_rows: int
    wall_seconds: float
    simulated_seconds: float


def _reference_joint_pos(reference: Any) -> np.ndarray:
    # Teleopit and scene use canonical MuJoCo/URDF order.  SONIC arrays are in
    # IsaacLab order, but the neutral Teleopit view is always available.
    return np.asarray(reference.teleopit_qpos36[7:36], dtype=np.float64)


def _controller_model_paths(controller: Any) -> dict[str, Path]:
    if isinstance(controller, SonicController):
        return {
            "encoder": controller.spec.encoder_path,
            "decoder": controller.spec.decoder_path,
        }
    if isinstance(controller, TeleopitController):
        return {
            "tracker": controller.checkpoint,
            "fk_robot_xml": controller.robot_xml,
        }
    raise TypeError(f"unsupported controller type {type(controller).__name__}")


def _behavior_source_hashes() -> dict[str, dict[str, str]]:
    return {
        relative: source_tree_hashes(REPO_ROOT / relative)
        for relative in _BEHAVIOR_SOURCE_DIRECTORIES
        if (REPO_ROOT / relative).is_dir()
    }


def _path_hashes(paths: Mapping[str, Path]) -> dict[str, str]:
    return {
        str(name): sha256_file(Path(path).expanduser().resolve())
        for name, path in paths.items()
    }


def _fit_reference_to_timeline(
    reference: ReferenceSequence,
    *,
    available_log_rows: int,
) -> tuple[ReferenceSequence, dict[str, Any]]:
    """Select the exact reference prefix the 400 Hz source can support.

    A recording can end partway through its final 50 Hz ``policy_seq`` group.
    The formal scheduler never invents the missing 400 Hz source samples: it
    executes only complete eight-row intervals and makes that tail truncation
    explicit in every output contract.
    """

    rows = int(available_log_rows)
    if rows < 0:
        raise RolloutError("source timeline available_log_rows must be non-negative")
    requested_frames = int(reference.num_frames)
    available_intervals = rows // LOGS_PER_POLICY
    executed_frames = min(requested_frames, available_intervals)
    if executed_frames <= 0:
        raise RolloutError(
            "source timeline has fewer than eight 400 Hz rows after takeover"
        )
    executed = (
        reference
        if executed_frames == requested_frames
        else reference.slice(policy_offset=0, policy_count=executed_frames)
    )
    consumed_rows = executed_frames * LOGS_PER_POLICY
    return executed, {
        "requested_reference_frames": requested_frames,
        "executed_reference_frames": executed_frames,
        "source_available_400hz_rows": rows,
        "source_full_50hz_intervals_available": available_intervals,
        "source_400hz_rows_consumed": consumed_rows,
        "source_400hz_rows_unconsumed": rows - consumed_rows,
        "truncated_to_complete_source_intervals": (
            executed_frames < requested_frames
        ),
        "tail_policy": (
            "drop any reference frame that lacks all eight phase-matched "
            "400 Hz source samples"
        ),
    }


def _loaded_controller_model_hashes(
    controller: SonicController | TeleopitController,
) -> dict[str, str]:
    hashes = getattr(controller, "loaded_model_sha256", None)
    if not isinstance(hashes, Mapping):
        raise RolloutError("controller does not expose loaded model hashes")
    return {str(name): str(value) for name, value in hashes.items()}


def _command_fields(command: AppliedPdCommand) -> dict[str, np.ndarray]:
    return command.csv_command_fields()


def _model_quaternion_qpos_slices(model: mujoco.MjModel) -> tuple[tuple[int, int], ...]:
    """Return every free/ball-joint quaternion range in MuJoCo qpos order."""

    result: list[tuple[int, int]] = []
    for joint_id in range(model.njnt):
        start = int(model.jnt_qposadr[joint_id])
        joint_type = int(model.jnt_type[joint_id])
        if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
            result.append((start + 3, start + 7))
        elif joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
            result.append((start, start + 4))
    if not result:
        raise RolloutError("staged MuJoCo scene contains no quaternion joint")
    return tuple(result)


def _policy_snapshot(
    controller: SonicController | TeleopitController,
    step: Any,
    policy_seq: int,
) -> PolicySnapshot:
    """Build the held legacy-CSV snapshot without inventing SONIC fields."""

    common = {
        "policy_seq": int(policy_seq),
        "received_dof_pos": step.received_dof_pos,
    }
    if isinstance(controller, SonicController):
        return PolicySnapshot(
            **common,
            token=step.token,
            last_action=step.last_action,
            raw_action=step.raw_action,
        )
    return PolicySnapshot(**common)


def _controller_reference_signals(
    controller: SonicController | TeleopitController,
    step: Any,
    state: Any,
) -> dict[str, Any]:
    """Decode only reference signals that the network actually consumed."""

    if isinstance(controller, SonicController):
        orientations = controller.g1_anchor_orientation_from_encoder_input(
            step.encoder_input
        )
        relative_rot6d = np.asarray(orientations[0], dtype=np.float64).copy()
        return {
            "relative_anchor_rot6d": relative_rot6d,
            "orientation_error_rad": rot6d_orientation_error_rad(relative_rot6d),
            "height_error_m": None,
        }

    observation = np.asarray(step.observation, dtype=np.float64).reshape(-1)
    relative_rot6d = observation[58:64].copy()
    reference_height = float(observation[166])
    live_height = float(state.teleopit_torso_height_m)
    return {
        "relative_anchor_rot6d": relative_rot6d,
        "orientation_error_rad": rot6d_orientation_error_rad(relative_rot6d),
        "height_error_m": live_height - reference_height,
        "reference_torso_linear_velocity_body": observation[157:160].copy(),
        "reference_torso_angular_velocity_body": observation[160:163].copy(),
        "reference_projected_gravity": observation[163:166].copy(),
        "reference_torso_height_m": reference_height,
        "live_torso_height_m": live_height,
    }


def _telemetry_record(
    *,
    controller: Any,
    step: Any,
    reference: Any,
    state: Any,
    command: AppliedPdCommand,
    inference_index: int,
    source_row_index: int,
    phase_matched_source_qpos: np.ndarray,
    fallen: bool,
    controller_reference_signals: Mapping[str, Any],
) -> TelemetryRecord:
    warmup = inference_index < SELF_WARMUP_INFERENCES
    # ``reference`` in the native 50 Hz telemetry is always what the current
    # controller actually consumed.  In particular, SONIC orientation slots
    # depend on the live robot orientation and therefore cannot be represented
    # by blindly copying the recording's stored reference_motion columns.
    reference_native = (
        controller.reference_motion_from_encoder_input(step.encoder_input)
        if isinstance(controller, SonicController)
        else reference.teleopit_qpos36
    )
    extra: dict[str, Any] = {
        "desiredLeftHandTarget": reference.left_hand_target,
        "desiredRightHandTarget": reference.right_hand_target,
        "appliedLeftHandTarget": command.left_hand_target,
        "appliedRightHandTarget": command.right_hand_target,
        "leftHandAppliedTorque": command.left_hand_torque,
        "rightHandAppliedTorque": command.right_hand_torque,
        "leftHandTorqueSaturation": command.left_hand_torque_saturation,
        "rightHandTorqueSaturation": command.right_hand_torque_saturation,
        "receivedDofPos": step.received_dof_pos,
        "sourceRootPos": reference.source_root_pos,
        "sourceRootQuatWxyz": reference.source_root_quat_wxyz,
        "phaseMatchedSourceQpos": phase_matched_source_qpos,
        "phaseMatchedSourceRootPos": phase_matched_source_qpos[:3],
        "phaseMatchedSourceRootQuatWxyz": phase_matched_source_qpos[3:7],
        "sonicBaseAngVelQvel": state.sonic_base_ang_vel_qvel,
        "teleopitPelvisAngVelB": state.teleopit_pelvis_ang_vel_b,
        "controllerReferenceAnchorRelativeRot6d": (
            controller_reference_signals["relative_anchor_rot6d"]
        ),
        "controllerReferenceAnchorOrientationErrorRad": np.asarray(
            [controller_reference_signals["orientation_error_rad"]],
            dtype=np.float64,
        ),
        "controllerReferenceAbsoluteTranslationAvailable": np.asarray(
            [0.0], dtype=np.float64
        ),
        "preFallValid": np.asarray([not fallen], dtype=np.float64),
    }
    if isinstance(controller, TeleopitController):
        extra.update(
            {
                "controllerReferenceTorsoLinearVelocityBody": (
                    controller_reference_signals[
                        "reference_torso_linear_velocity_body"
                    ]
                ),
                "controllerReferenceTorsoAngularVelocityBody": (
                    controller_reference_signals[
                        "reference_torso_angular_velocity_body"
                    ]
                ),
                "controllerReferenceProjectedGravity": (
                    controller_reference_signals["reference_projected_gravity"]
                ),
                "controllerReferenceTorsoHeightM": np.asarray(
                    [controller_reference_signals["reference_torso_height_m"]]
                ),
                "liveTeleopitTorsoHeightM": np.asarray(
                    [controller_reference_signals["live_torso_height_m"]]
                ),
                "controllerReferenceTorsoHeightErrorM": np.asarray(
                    [controller_reference_signals["height_error_m"]]
                ),
            }
        )
    encoder_input = getattr(step, "encoder_input", None)
    if encoder_input is not None:
        extra["encoderInput"] = encoder_input
    qtarget_isaaclab = getattr(step, "q_target_isaaclab", None)
    if qtarget_isaaclab is not None:
        extra["qTargetIsaaclab"] = qtarget_isaaclab
    validation_error = getattr(step, "prefill_validation_max_abs", None)
    if validation_error is not None:
        extra["prefillValidationMaxAbs"] = np.asarray([validation_error])
    return TelemetryRecord(
        policy_seq=int(reference.policy_seq),
        policy_time_s=float(state.timestamp_s),
        source_row_index=int(source_row_index),
        warmup=warmup,
        evaluation=not warmup and not fallen,
        reference_index=int(reference.frame_index),
        observation=step.observation,
        history=step.history,
        token=getattr(step, "token", None),
        last_action=step.last_action,
        raw_action=step.raw_action,
        action=step.raw_action,
        q_target=step.q_target,
        torque=command.body_torque,
        torque_saturation=command.body_torque_saturation,
        reference=reference_native,
        robot_q=state.joint_pos,
        robot_dq=state.joint_vel,
        root_qpos=np.concatenate((state.root_pos, state.root_quat_wxyz)),
        kp=step.kp,
        kd=step.kd,
        extra_arrays=extra,
    )


class _DetachedViewer:
    """Render a cloned model/data pair that cannot perturb live physics."""

    def __init__(self, scene: DeterministicTaskScene) -> None:
        import mujoco
        import mujoco.viewer

        self._mujoco = mujoco
        self._model = mujoco.MjModel.from_xml_path(str(scene.scene_xml))
        self._data = mujoco.MjData(self._model)
        if self._model.nq != scene.model.nq or self._model.nv != scene.model.nv:
            raise RolloutError("detached viewer model does not match live simulation")
        self._handle = mujoco.viewer.launch_passive(
            self._model,
            self._data,
            show_left_ui=False,
            show_right_ui=True,
        )

    def is_running(self) -> bool:
        return bool(self._handle.is_running())

    def sync(self, scene: DeterministicTaskScene) -> None:
        self._data.qpos[:] = scene.data.qpos
        self._data.qvel[:] = scene.data.qvel
        self._data.time = scene.data.time
        if self._data.mocap_pos.shape == scene.data.mocap_pos.shape:
            self._data.mocap_pos[:] = scene.data.mocap_pos
            self._data.mocap_quat[:] = scene.data.mocap_quat
        self._mujoco.mj_forward(self._model, self._data)
        # Model/UI changes are kept on the display clone.  They can never be
        # copied back into the live controller-replacement simulation.
        self._handle.sync(state_only=True)

    def close(self) -> None:
        self._handle.close()


def _open_viewer(
    scene: DeterministicTaskScene, enabled: bool
) -> _DetachedViewer | None:
    if not enabled:
        return None
    return _DetachedViewer(scene)


def _close_viewer(viewer: _DetachedViewer | None) -> None:
    if viewer is not None:
        viewer.close()


def _run_rollout_impl(
    config: RolloutConfig,
    *,
    controller: SonicController | TeleopitController,
    history_context: Any,
    recording_dir: Path,
    source_csv: Path,
    output: Path,
    work_output: Path,
    controller_model_paths: Mapping[str, Path],
) -> RolloutResult:
    """Generate one complete rollout inside a transaction work directory.

    ``history_context`` is intentionally duck-typed to keep the timing engine
    independent of source-history construction.  It must expose the exact
    phase-matched initial state and the selected takeover policy sequence.
    """

    if not isinstance(controller, (SonicController, TeleopitController)):
        raise TypeError("controller must be SonicController or TeleopitController")
    reference = config.reference
    if not len(reference):
        raise RolloutError("reference sequence is empty")
    if int(reference.policy_seq[0]) != int(history_context.selected_policy_seq):
        raise RolloutError(
            "reference first policy_seq does not match source-history takeover"
        )
    if source_csv.resolve() != reference.source_csv_path.resolve():
        raise RolloutError("config recording and reference source CSV differ")
    source_csv_sha256_before = sha256_file(source_csv)
    if source_csv_sha256_before != reference.source_csv_sha256:
        raise RolloutError(
            "source data.csv no longer matches the bytes used to build the reference"
        )
    history_source_sha256 = str(
        history_context.sonic_prefill_payload.get("source_csv_sha256", "")
    )
    if source_csv_sha256_before != history_source_sha256:
        raise RolloutError(
            "source data.csv no longer matches the source-history prefill"
        )
    if (
        config.expected_source_csv_sha256 is not None
        and source_csv_sha256_before != config.expected_source_csv_sha256
    ):
        raise RolloutError(
            "source data.csv changed after the formal launcher loaded its inputs"
        )
    behavior_source_hashes_before = _behavior_source_hashes()
    if (
        config.expected_behavior_source_hashes is not None
        and behavior_source_hashes_before
        != {
            str(directory): {
                str(path): str(digest) for path, digest in hashes.items()
            }
            for directory, hashes in config.expected_behavior_source_hashes.items()
        }
    ):
        raise RolloutError(
            "behavior-bearing source files changed after the formal launcher "
            "loaded the runtime"
    )
    staged = stage_recording_snapshot(
        recording_dir,
        work_output,
        asset_model_root=config.asset_model_root,
    )
    published_scene_path = output / staged.scene_path.relative_to(work_output)
    scene = DeterministicTaskScene(
        staged.scene_path,
        root_assist=config.root_assist,
        hand_torque_profile=config.hand_torque_profile,
    )
    timeline = CsvTimeline.from_source_history(
        source_csv,
        history_context,
        quaternion_qpos_slices=_model_quaternion_qpos_slices(scene.model),
    )
    if timeline.qpos_width != scene.model.nq:
        raise RolloutError(
            f"source qpos width {timeline.qpos_width} does not match "
            f"staged MuJoCo nq={scene.model.nq}"
        )
    reference, reference_horizon = _fit_reference_to_timeline(
        reference,
        available_log_rows=timeline.available_log_rows,
    )
    compiled_model_fingerprint = compiled_mujoco_model_fingerprint(scene.model)
    if config.automatic_output:
        _validate_automatic_scene_reuse(output, compiled_model_fingerprint)
    initial_qpos, initial_qvel = timeline.state()
    scene.initialize(initial_qpos, initial_qvel)
    if isinstance(controller, SonicController):
        controller.reset(history_context.sonic_prefill_payload)
        controller_metadata = controller.metadata()
        controller_name = controller.name
        controller_family = controller.controller_family
    else:
        if history_context.teleopit_prefill is None:
            raise RolloutError("formal Teleopit rollout requires Teleopit source history")
        controller.reset(history_context.teleopit_prefill)
        controller_metadata = controller.metadata()
        controller_name = controller.name
        controller_family = controller.controller_family
    controller_model_hashes_before = _path_hashes(controller_model_paths)
    loaded_controller_model_hashes = _loaded_controller_model_hashes(controller)
    if controller_model_hashes_before != loaded_controller_model_hashes:
        raise RolloutError(
            "controller model files no longer match the bytes used to create "
            "the inference/FK sessions"
        )

    with source_csv.open("r", newline="", encoding="utf-8-sig") as source:
        source_header = next(csv.reader(source))
    data_path = work_output / "data.csv"
    telemetry_path = work_output / "policy_telemetry.npz"
    metrics_path = work_output / "metrics.json"
    csv_writer = ReplayCsvWriter(
        data_path,
        source_csv,
        controller_family=controller_family,
        # The work directory is atomically renamed on success, so record the
        # final published path rather than the unique transaction work path.
        scene_path=published_scene_path,
        logging_hz=LOG_HZ,
    )
    telemetry_writer = ControllerTelemetryWriter(
        telemetry_path,
        controller_name=controller_name,
        controller_family=controller_family,
        metadata={
            "controller": controller_metadata,
            "source_history": history_context.metadata(),
            "reference": reference.metadata(),
            "reference_horizon": reference_horizon,
            "applied_torque_sampling": (
                "first 200 Hz PD update at each 50 Hz policy boundary only; "
                "use metrics.json for all-200-Hz body-torque aggregates"
            ),
            "hand_target_fields": {
                "desired": (
                    "desiredLeftHandTarget/desiredRightHandTarget are the raw "
                    "50 Hz reference targets"
                ),
                "applied": (
                    "appliedLeftHandTarget/appliedRightHandTarget are the "
                    "rate-limited targets used by that PD update"
                ),
            },
            "state_signal_contract": {
                "sonic_base_ang_vel_qvel": (
                    "MuJoCo floating-joint qvel[3:6], matching released SONIC bridge"
                ),
                "teleopit_pelvis_ang_vel_b": (
                    "MuJoCo pelvis body angular velocity from mj_objectVelocity "
                    "with local-frame flag"
                ),
                "controller_reference_anchor_orientation": (
                    "geodesic angle decoded from the exact relative rot6d "
                    "network input"
                ),
                "controller_reference_absolute_root_xy": "unavailable/not consumed",
            },
            "hand_control": scene.hand_control_metadata(),
        },
    )
    metrics = RolloutMetrics(
        body_joint_names=controller.joint_names,
        task_qpos_labels=scene.task_qpos_labels,
        warmup_inferences=SELF_WARMUP_INFERENCES,
        fall_height_m=config.fall_height_m,
    )
    viewer = _open_viewer(scene, config.viewer)
    start_wall = time.monotonic()
    inference_count = 0
    log_index = 0
    source_row_indices: list[int] = []
    source_control_times: list[float] = []
    source_qpos_samples: list[np.ndarray] = []
    reference_indices: list[int] = []
    pre_fall_valid_rows: list[bool] = []
    contact_rows: list[Mapping[str, float | int]] = []
    current_step: Any | None = None
    current_frame: Any | None = None
    current_snapshot: PolicySnapshot | None = None
    command: AppliedPdCommand | None = None
    current_policy_warmup = True
    source_time_zero = float(history_context.timeline_start_control_time_s)
    total_policy_frames = reference.num_frames
    expected_log_rows = total_policy_frames * LOGS_PER_POLICY
    completed = False
    try:
        while log_index < expected_log_rows:
            source_qpos, source_qvel = timeline.state()
            if log_index % LOGS_PER_PD == 0:
                scene.apply_root_assist(
                    source_qpos,
                    source_qvel,
                    source_velocity_active=not timeline.exhausted,
                )
            fallen_at_row = metrics.observe_fall(
                qpos=scene.data.qpos,
                time_s=float(scene.data.time),
                log_row_exclusive=log_index,
                source_row_index=timeline.current_row_index,
            )
            if log_index % LOGS_PER_POLICY == 0:
                reference_index = log_index // LOGS_PER_POLICY
                current_policy_warmup = (
                    inference_count < SELF_WARMUP_INFERENCES
                )
                current_frame = reference.frame(reference_index)
                state_before = scene.state()
                current_step = controller.infer(state_before, current_frame)
                scene.set_controller_command(
                    q_target=current_step.q_target,
                    kp=current_step.kp,
                    kd=current_step.kd,
                    torque_limit=current_step.torque_limit,
                    left_hand_target=current_frame.left_hand_target,
                    right_hand_target=current_frame.right_hand_target,
                )
                current_snapshot = _policy_snapshot(
                    controller, current_step, int(current_frame.policy_seq)
                )
            if log_index % LOGS_PER_PD == 0:
                command = scene.update_pd()
                metrics.record_pd_control(
                    body_torque=command.body_torque,
                    body_torque_saturation=command.body_torque_saturation,
                    evaluation=(
                        not current_policy_warmup and not metrics.fallen
                    ),
                )
                if log_index % LOGS_PER_POLICY == 0:
                    assert current_step is not None and current_frame is not None
                    controller_reference_signals = (
                        _controller_reference_signals(
                            controller, current_step, state_before
                        )
                    )
                    metrics.record_policy(
                        robot_joint_pos=state_before.joint_pos,
                        controller_reference_joint_pos=(
                            _reference_joint_pos(current_frame)
                        ),
                        q_target=current_step.q_target,
                        robot_root_pos=state_before.root_pos,
                        phase_matched_source_root_pos=source_qpos[:3],
                        robot_root_quat_wxyz=state_before.root_quat_wxyz,
                        phase_matched_source_root_quat_wxyz=source_qpos[3:7],
                        controller_reference_orientation_error_rad=(
                            controller_reference_signals[
                                "orientation_error_rad"
                            ]
                        ),
                        controller_reference_height_error_m=(
                            controller_reference_signals["height_error_m"]
                        ),
                        warmup=current_policy_warmup,
                    )
                    telemetry_writer.append(
                        _telemetry_record(
                            controller=controller,
                            step=current_step,
                            reference=current_frame,
                            state=state_before,
                            command=command,
                            inference_index=inference_count,
                            source_row_index=timeline.current_row_index,
                            phase_matched_source_qpos=source_qpos,
                            fallen=fallen_at_row,
                            controller_reference_signals=(
                                controller_reference_signals
                            ),
                        )
                    )
                    inference_count += 1
            assert (
                command is not None
                and current_step is not None
                and current_frame is not None
                and current_snapshot is not None
            )
            csv_writer.write_frame(
                source_row_index=timeline.current_row_index,
                sample_index=log_index,
                control_time_s=log_index / LOG_HZ,
                mujoco_time_s=float(scene.data.time),
                qpos=scene.data.qpos,
                qvel=scene.data.qvel,
                command_fields=_command_fields(command),
                policy_snapshot=(
                    current_snapshot
                    if log_index % LOGS_PER_POLICY == 0
                    else None
                ),
                reference_motion=(
                    controller.reference_motion_from_encoder_input(
                        current_step.encoder_input
                    )
                    if isinstance(controller, SonicController)
                    else None
                ),
                clear_reference_motion=not isinstance(controller, SonicController),
            )
            contact_summary = scene.contact_summary()
            metrics.record_log(
                qpos=scene.data.qpos,
                task_qpos=scene.task_qpos(),
                source_row_index=timeline.current_row_index,
                contact_summary=contact_summary,
            )
            source_row_indices.append(int(timeline.current_row_index))
            source_control_times.append(float(timeline.current_time_s))
            source_qpos_samples.append(source_qpos.copy())
            reference_indices.append(int(current_frame.frame_index))
            pre_fall_valid_rows.append(not fallen_at_row)
            contact_rows.append(contact_summary.as_dict())
            if viewer is not None:
                if not viewer.is_running():
                    raise KeyboardInterrupt("viewer closed")
                if log_index % LOGS_PER_POLICY == 0:
                    viewer.sync(scene)
            for _ in range(PHYSICS_STEPS_PER_LOG):
                scene.physics_step(1)
                metrics.observe_fall(
                    qpos=scene.data.qpos,
                    time_s=float(scene.data.time),
                    log_row_exclusive=log_index + 1,
                    source_row_index=timeline.current_row_index,
                )
            scene.validate_state()
            log_index += 1
            if log_index < expected_log_rows:
                if not timeline.advance():
                    break
            if viewer is not None:
                # Display pacing may make the run take longer, but never
                # chooses a reference/action or changes the fixed step count.
                delay = start_wall + log_index / LOG_HZ - time.monotonic()
                if delay > 0.0:
                    time.sleep(delay)
        if log_index != expected_log_rows or inference_count != total_policy_frames:
            raise RolloutError(
                "rollout stopped before the fixed-step schedule completed: "
                f"rows={log_index}/{expected_log_rows}, "
                f"inferences={inference_count}/{total_policy_frames}"
            )
        # The final physics step is not another CSV row, but a terminal fall at
        # that exact boundary must still disqualify task success.
        metrics.observe_fall(
            qpos=scene.data.qpos,
            time_s=float(scene.data.time),
            log_row_exclusive=log_index,
            source_row_index=timeline.current_row_index,
        )
        metrics.record_terminal(
            qpos=scene.data.qpos,
            task_qpos=scene.task_qpos(),
            time_s=float(scene.data.time),
            contact_summary=scene.contact_summary(),
        )
        completed = True
    finally:
        _close_viewer(viewer)
        timeline.close()
        if completed:
            csv_writer.close()
            telemetry_writer.close()
        else:
            csv_writer.abort()

    wall_seconds = time.monotonic() - start_wall
    simulated_seconds = log_index / LOG_HZ
    metrics_payload = metrics.report()
    metrics_payload["reference_horizon"] = reference_horizon
    metrics_payload["root_assist"] = scene.root_assist.metadata()
    metrics_payload["hand_control"] = scene.hand_control_metadata()
    metrics_payload["state_signal_contract"] = {
        "sonic_base_ang_vel_qvel": {
            "provider": "MuJoCo free-joint qvel[3:6]",
            "consumer": "SONIC decoder observation",
        },
        "teleopit_pelvis_ang_vel_b": {
            "provider": "mj_objectVelocity(pelvis, local=1)[0:3]",
            "consumer": "Teleopit observation",
            "source_history_provider": (
                "same call reconstructed in pinned Teleopit robot model"
            ),
        },
        "controller_reference_anchor_orientation": {
            "provider": (
                (
                    "SONIC encoder_input"
                    f"[{controller.spec.g1_orientation_offset}:"
                    f"{controller.spec.g1_orientation_offset + 6}]"
                )
                if isinstance(controller, SonicController)
                else "Teleopit observation[58:64]"
            ),
            "metric": "orthonormalized rot6d geodesic angle",
            "absolute_root_xy_available": False,
        },
    }
    metrics_payload["runtime"] = {
        "wall_seconds": wall_seconds,
        "simulated_seconds": simulated_seconds,
        "wall_clock_timing_used_for_control": False,
        "realtime_factor_diagnostic_only": (
            simulated_seconds / wall_seconds if wall_seconds > 0.0 else None
        ),
    }
    write_json_atomic(metrics_path, metrics_payload)
    with (work_output / "source_timeline.npz").open("wb") as timeline_output:
        np.savez_compressed(
            timeline_output,
            source_row_index=np.asarray(source_row_indices, dtype=np.int64),
            source_control_time_s=np.asarray(source_control_times, dtype=np.float64),
            source_qpos=np.asarray(source_qpos_samples, dtype=np.float64),
            reference_index=np.asarray(reference_indices, dtype=np.int64),
            pre_fall_valid_mask=np.asarray(
                pre_fall_valid_rows, dtype=np.bool_
            ),
            metadata_json=np.asarray(
                json.dumps(
                    {
                        "format_version": 2,
                        "protocol_revision": 2,
                        "rate_hz": LOG_HZ,
                        "mapping": (
                            "phase-matched source qpos interpolated at each "
                            "exact 400 Hz rollout sample; source_row_index is "
                            "the nearest provenance row"
                        ),
                        "source_csv": str(source_csv),
                        "source_csv_sha256": source_csv_sha256_before,
                        "pre_fall_valid_mask": (
                            "true only for rows strictly before the first "
                            "latched root-height fall"
                        ),
                    },
                    sort_keys=True,
                )
            ),
        )
    contact_names = tuple(contact_rows[0]) if contact_rows else ()
    with (work_output / "contact_telemetry.npz").open("wb") as contact_output:
        np.savez_compressed(
            contact_output,
            **{
                name: np.asarray([row[name] for row in contact_rows])
                for name in contact_names
            },
            metadata_json=np.asarray(
                json.dumps(
                    {
                        "rate_hz": LOG_HZ,
                        "classification": (
                            "robot_environment excludes robot self-contact and "
                            "the MuJoCo world body; it can include task objects, "
                            "furniture, and fixtures"
                        ),
                    },
                    sort_keys=True,
                )
            ),
        )
    prepared_reference_path = reference.save_prepared_npz(
        work_output / "prepared_reference.npz"
    )
    # Loading the just-written archive with pickle disabled re-runs every
    # ReferenceSequence invariant and catches incomplete serialization before
    # a result can be published.
    ReferenceSequence.load_prepared_npz(
        prepared_reference_path, source_csv_path=source_csv
    )
    write_data_schema(
        work_output / "data_schema.json",
        header=source_header,
        controller_family=controller_family,
        source_csv=source_csv,
        logging_hz=LOG_HZ,
        generated_command_fields=tuple(_command_fields(command)),
        extra={
            "sonic_only_policy_fields": (
                "native values" if controller_family == "sonic" else "all zero"
            ),
            "controller_neutral_policy_received_dof_pos": (
                "real body29+left7+right7 state for every controller"
            ),
            "policy_telemetry_applied_torque_sampling": (
                "first 200 Hz PD update at each 50 Hz policy boundary only"
            ),
        },
    )
    # Compatibility sidecars let the existing compare viewer find the source
    # recording.  The controller_replacement replay wrapper consumes the exact
    # high-rate source-row mapping instead of assuming raw policy group sizes.
    write_json_atomic(
        work_output / "run_metadata.json",
        {
            "source_recording": str(recording_dir),
            "source_csv": str(source_csv),
            "initial_row_index": int(source_row_indices[0]),
            "source_rows_per_control": 1,
            "control_dt": 1.0 / LOG_HZ,
            "source_dt": 1.0 / LOG_HZ,
            "samples": log_index,
            "source_timeline_path": "source_timeline.npz",
            "phase_matched_start_time_s": source_time_zero,
        },
    )
    write_json_atomic(
        work_output / "launch_manifest.json",
        {
            "recording": str(recording_dir),
            "controller": controller_name,
            "reference_mode": reference.mode.value,
            "root_assist": config.root_assist,
            "hand_torque_profile": config.hand_torque_profile,
            "fall_height_m": config.fall_height_m,
            "reference_horizon": reference_horizon,
            "viewer": {
                "enabled": config.viewer,
                "detached_model_and_data": True,
                "can_modify_live_physics": False,
            },
            "initialization": {
                "actual_start_source_row_index": int(source_row_indices[0]),
                "selected_policy_seq": history_context.selected_policy_seq,
                "raw_policy_group_offset": (
                    history_context.raw_policy_group_offset
                ),
            },
        },
    )
    source_history_paths = write_source_history_artifacts(
        history_context, work_output
    )

    source_csv_sha256_after = sha256_file(source_csv)
    if source_csv_sha256_after != source_csv_sha256_before:
        raise RolloutError(
            "source data.csv changed while the rollout was running; refusing "
            "to publish a mixed-provenance result"
        )
    behavior_source_hashes_after = _behavior_source_hashes()
    if behavior_source_hashes_after != behavior_source_hashes_before:
        raise RolloutError(
            "behavior-bearing source files changed while the rollout was "
            "running; refusing to publish mixed code provenance"
        )
    controller_model_hashes_after = _path_hashes(controller_model_paths)
    if controller_model_hashes_after != controller_model_hashes_before:
        raise RolloutError(
            "controller model artifacts changed while the rollout was running; "
            "refusing to publish mixed model provenance"
        )
    provenance_payload = {
        "invocation": {
            "argv": [str(Path(sys.executable).resolve()), *sys.argv],
            "working_directory": str(Path.cwd().resolve()),
        },
        "git": git_provenance(REPO_ROOT),
        "runtime": runtime_environment(),
        "source_recording": {
            "data_csv": str(source_csv),
            "sha256_before": source_csv_sha256_before,
            "sha256_after": source_csv_sha256_after,
            "unchanged_during_rollout": True,
        },
        "behavior_source": {
            "sha256": behavior_source_hashes_before,
            "unchanged_during_rollout": True,
        },
        "controller_model_sha256_before_and_after": {
            "sha256": controller_model_hashes_before,
            "unchanged_during_rollout": True,
        },
        "staged_scene": {
            "scene_xml": str(published_scene_path),
            "snapshot_xml_sha256": snapshot_xml_hashes(staged.snapshot_root),
            "asset_symlink_targets": snapshot_symlink_targets(
                staged.snapshot_root
            ),
            "asset_model_root": str(staged.asset_model_root),
            "compiled_mujoco_model": compiled_model_fingerprint,
        },
    }
    write_run_manifest(
        work_output / "run_manifest.json",
        controller_name=controller_name,
        controller_family=controller_family,
        source_recording=recording_dir,
        reference_mode=reference.mode.value,
        root_assist=config.root_assist,
        rates_hz={
            "physics": PHYSICS_HZ,
            "pd": PD_HZ,
            "policy": POLICY_HZ,
            "csv_logging": LOG_HZ,
        },
        model_paths=controller_model_paths,
        provenance_extra=provenance_payload,
        extra={
            "deterministic_single_process": True,
            "dds_or_zmq": False,
            "wall_clock_controls_reference": False,
            "viewer": {
                "enabled": config.viewer,
                "detached_model_and_data": True,
                "can_modify_live_physics": False,
            },
            "source_history": history_context.metadata(),
            "reference": reference.metadata(),
            "reference_horizon": reference_horizon,
            "controller": controller_metadata,
            "state_signal_contract": metrics_payload["state_signal_contract"],
            "hand_control": scene.hand_control_metadata(),
            "fall_detection": metrics_payload["stability"]["fall_definition"],
            "self_warmup_inferences": SELF_WARMUP_INFERENCES,
            "tracking_begins_at_inference": SELF_WARMUP_INFERENCES + 1,
            "tracking_stops_after_latched_fall": True,
            "warmup_saved_and_affects_physics": True,
            "post_fall_physics_continues": True,
            "semantic_task_success": (
                "false on fall; otherwise manual/unassigned; see metrics.json"
            ),
            "inference_backend": "onnxruntime",
        },
    )
    generated_artifacts = [
        data_path.name,
        telemetry_path.name,
        metrics_path.name,
        "source_timeline.npz",
        "contact_telemetry.npz",
        prepared_reference_path.name,
        "data_schema.json",
        "run_manifest.json",
        "run_metadata.json",
        "launch_manifest.json",
        *(path.name for path in source_history_paths.values()),
    ]
    generated_artifact_hashes = artifact_hashes(
        work_output, generated_artifacts
    )
    if sha256_file(source_csv) != source_csv_sha256_before:
        raise RolloutError(
            "source data.csv changed while final artifacts were certified"
        )
    if _behavior_source_hashes() != behavior_source_hashes_before:
        raise RolloutError(
            "behavior-bearing source files changed while final artifacts were certified"
        )
    if _path_hashes(controller_model_paths) != controller_model_hashes_before:
        raise RolloutError(
            "controller models changed while final artifacts were certified"
        )
    # This is deliberately the final generated file.  Its presence certifies
    # that all scheduled simulation steps, validations and artifact hashes
    # completed before the atomic directory publication.
    write_json_atomic(
        work_output / "run_complete.json",
        {
            "protocol_revision": 2,
            "complete": True,
            "fixed_step_schedule_complete": True,
            "finite_state_complete": True,
            "logged_rows": log_index,
            "policy_inferences": inference_count,
            "requested_reference_frames": reference_horizon[
                "requested_reference_frames"
            ],
            "executed_reference_frames": reference_horizon[
                "executed_reference_frames"
            ],
            "wall_seconds": wall_seconds,
            "simulated_seconds": simulated_seconds,
            "source_time_zero_s": source_time_zero,
            "source_recording_unchanged": True,
            "behavior_source_unchanged": True,
            "controller_models_unchanged": True,
            "hidden_terminal_endpoint_in_metrics": True,
            "fallen": metrics.fallen,
            "task_success_eligible": metrics.task_success_eligible,
            "artifact_sha256": generated_artifact_hashes,
        },
    )
    return RolloutResult(
        output_directory=output,
        data_csv=output / data_path.name,
        policy_telemetry=output / telemetry_path.name,
        metrics_json=output / metrics_path.name,
        policy_inferences=inference_count,
        logged_rows=log_index,
        wall_seconds=wall_seconds,
        simulated_seconds=simulated_seconds,
    )


def run_rollout(
    config: RolloutConfig,
    *,
    controller: SonicController | TeleopitController,
    history_context: Any,
) -> RolloutResult:
    """Run and failure-atomically publish a formal deterministic rollout.

    The transaction context deliberately surrounds staging, model loading,
    simulation, validation, sidecar generation, and final publication.  Thus
    every exception after a work directory is created removes only that unique
    work directory and always releases the per-output advisory lock.
    """

    if not isinstance(controller, (SonicController, TeleopitController)):
        raise TypeError("controller must be SonicController or TeleopitController")
    reference = config.reference
    if not len(reference):
        raise RolloutError("reference sequence is empty")
    if int(reference.policy_seq[0]) != int(history_context.selected_policy_seq):
        raise RolloutError(
            "reference first policy_seq does not match source-history takeover"
        )

    recording_dir, source_csv = resolve_recording(config.recording)
    if source_csv.resolve() != reference.source_csv_path.resolve():
        raise RolloutError("config recording and reference source CSV differ")
    raw_output = config.output_directory.expanduser()
    if raw_output.is_symlink():
        raise RolloutError(
            f"refusing to use a symbolic link as the final output: {raw_output}"
        )
    output = raw_output.resolve()
    controller_model_paths = _controller_model_paths(controller)
    _validate_output_path_safety(
        output,
        recording_dir=recording_dir,
        source_csv=source_csv,
        model_paths=controller_model_paths,
        asset_model_root=config.asset_model_root,
    )

    with _OutputTransaction(
        output,
        existing_output_validator=config.existing_output_validator,
    ) as transaction:
        if transaction.work_output is None:  # pragma: no cover - invariant
            raise RolloutError("output transaction did not create a work directory")
        result = _run_rollout_impl(
            config,
            controller=controller,
            history_context=history_context,
            recording_dir=recording_dir,
            source_csv=source_csv,
            output=output,
            work_output=transaction.work_output,
            controller_model_paths=controller_model_paths,
        )
        transaction.publish()
        return result


__all__ = [
    "LOGS_PER_PD",
    "LOGS_PER_POLICY",
    "PHYSICS_STEPS_PER_LOG",
    "RolloutConfig",
    "RolloutError",
    "RolloutResult",
    "SELF_WARMUP_INFERENCES",
    "run_rollout",
]
