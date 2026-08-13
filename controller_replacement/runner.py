"""Single-process deterministic controller-replacement rollout engine."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import shutil
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from change_ckpt_track.task_sim_io import (
    resolve_recording,
    stage_recording_snapshot,
)
from controller_replacement.controllers import SonicController, TeleopitController
from controller_replacement.metrics import RolloutMetrics
from controller_replacement.history import write_source_history_artifacts
from controller_replacement.output import (
    ControllerTelemetryWriter,
    PolicySnapshot,
    ReplayCsvWriter,
    TelemetryRecord,
    write_data_schema,
    write_json_atomic,
    write_run_manifest,
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
    if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != "1":
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


@dataclass(frozen=True)
class RolloutConfig:
    recording: Path
    output_directory: Path
    reference: ReferenceSequence
    root_assist: str = "none"
    viewer: bool = False
    asset_model_root: Path | None = None
    self_warmup_inferences: int = SELF_WARMUP_INFERENCES

    def __post_init__(self) -> None:
        if self.root_assist not in {"none", "xy"}:
            raise ValueError("root_assist must be 'none' or 'xy'")
        if self.self_warmup_inferences != SELF_WARMUP_INFERENCES:
            raise ValueError(
                "the formal protocol fixes self-warmup at ten controller inferences"
            )


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


def _command_fields(command: AppliedPdCommand) -> dict[str, np.ndarray]:
    return command.csv_command_fields()


def _sonic_snapshot(step: Any, policy_seq: int) -> PolicySnapshot:
    return PolicySnapshot(
        policy_seq=policy_seq,
        token=step.token,
        last_action=step.last_action,
        raw_action=step.raw_action,
        received_dof_pos=step.received_dof_pos,
    )


def _telemetry_record(
    *,
    controller: Any,
    step: Any,
    reference: Any,
    state: Any,
    command: AppliedPdCommand,
    inference_index: int,
    source_row_index: int,
) -> TelemetryRecord:
    warmup = inference_index < SELF_WARMUP_INFERENCES
    # ``reference`` in the native 50 Hz telemetry is always what the current
    # controller actually consumed.  In particular, SONIC orientation slots
    # depend on the live robot orientation and therefore cannot be represented
    # by blindly copying the recording's stored reference_motion columns.
    reference_native = (
        step.encoder_input[4:644]
        if isinstance(controller, SonicController)
        else reference.teleopit_qpos36
    )
    extra: dict[str, Any] = {
        "leftHandTarget": reference.left_hand_target,
        "rightHandTarget": reference.right_hand_target,
        "leftHandAppliedTorque": command.left_hand_torque,
        "rightHandAppliedTorque": command.right_hand_torque,
        "receivedDofPos": step.received_dof_pos,
        "sourceRootPos": reference.source_root_pos,
        "sourceRootQuatWxyz": reference.source_root_quat_wxyz,
    }
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
        evaluation=not warmup,
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


def _open_viewer(scene: DeterministicTaskScene, enabled: bool) -> Any | None:
    if not enabled:
        return None
    import mujoco.viewer

    return mujoco.viewer.launch_passive(
        scene.model,
        scene.data,
        show_left_ui=False,
        show_right_ui=True,
    )


def _close_viewer(viewer: Any | None) -> None:
    if viewer is not None:
        viewer.close()


def run_rollout(
    config: RolloutConfig,
    *,
    controller: SonicController | TeleopitController,
    history_context: Any,
) -> RolloutResult:
    """Run a formal deterministic rollout and atomically publish its outputs.

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
    recording_dir, source_csv = resolve_recording(config.recording)
    if source_csv.resolve() != reference.source_csv_path.resolve():
        raise RolloutError("config recording and reference source CSV differ")
    output = config.output_directory.expanduser().resolve()
    # Validate ownership now, but keep the previous complete result intact
    # until the replacement run has itself completed successfully.
    _validate_owned_output(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    work_output = output.with_name(output.name + ".incomplete")
    _remove_owned_output(work_output)
    work_output.mkdir(parents=True)
    (work_output / _OWNED_OUTPUT_MARKER).write_text("1\n", encoding="utf-8")

    staged = stage_recording_snapshot(
        recording_dir,
        work_output,
        asset_model_root=config.asset_model_root,
    )
    published_scene_path = output / staged.scene_path.relative_to(work_output)
    timeline = CsvTimeline.from_source_history(source_csv, history_context)
    scene = DeterministicTaskScene(staged.scene_path, root_assist=config.root_assist)
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
        # final published path rather than the temporary ``.incomplete`` path.
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
        },
    )
    metrics = RolloutMetrics(
        body_joint_names=controller.joint_names,
        task_qpos_labels=scene.task_qpos_labels,
        warmup_inferences=SELF_WARMUP_INFERENCES,
    )
    viewer = _open_viewer(scene, config.viewer)
    start_wall = time.monotonic()
    inference_count = 0
    log_index = 0
    source_row_indices: list[int] = []
    source_control_times: list[float] = []
    source_qpos_samples: list[np.ndarray] = []
    reference_indices: list[int] = []
    contact_rows: list[Mapping[str, float | int]] = []
    current_step: Any | None = None
    command: AppliedPdCommand | None = None
    source_time_zero = float(history_context.timeline_start_control_time_s)
    total_policy_frames = min(
        reference.num_frames,
        timeline.available_log_rows // LOGS_PER_POLICY,
    )
    if total_policy_frames <= 0:
        raise RolloutError(
            "source timeline has fewer than eight 400 Hz rows after takeover"
        )
    expected_log_rows = total_policy_frames * LOGS_PER_POLICY
    completed = False
    try:
        while log_index < expected_log_rows:
            if log_index % LOGS_PER_PD == 0:
                source_qpos, source_qvel = timeline.state()
                scene.apply_root_assist(
                    source_qpos,
                    source_qvel,
                    source_velocity_active=not timeline.exhausted,
                )
            if log_index % LOGS_PER_POLICY == 0:
                reference_index = log_index // LOGS_PER_POLICY
                frame = reference.frame(reference_index)
                state_before = scene.state()
                current_step = controller.infer(state_before, frame)
                scene.set_controller_command(
                    q_target=current_step.q_target,
                    kp=current_step.kp,
                    kd=current_step.kd,
                    torque_limit=current_step.torque_limit,
                    left_hand_target=frame.left_hand_target,
                    right_hand_target=frame.right_hand_target,
                )
            if log_index % LOGS_PER_PD == 0:
                command = scene.update_pd()
                if log_index % LOGS_PER_POLICY == 0:
                    assert current_step is not None
                    frame = reference.frame(log_index // LOGS_PER_POLICY)
                    state = scene.state()
                    metrics.record_policy(
                        robot_joint_pos=state.joint_pos,
                        reference_joint_pos=_reference_joint_pos(frame),
                        q_target=current_step.q_target,
                        robot_root_pos=state.root_pos,
                        reference_root_pos=frame.source_root_pos,
                        robot_root_quat_wxyz=state.root_quat_wxyz,
                        reference_root_quat_wxyz=frame.teleopit_qpos36[3:7],
                        body_torque=command.body_torque,
                        body_torque_saturation=command.body_torque_saturation,
                        warmup=inference_count < SELF_WARMUP_INFERENCES,
                    )
                    telemetry_writer.append(
                        _telemetry_record(
                            controller=controller,
                            step=current_step,
                            reference=frame,
                            state=state,
                            command=command,
                            inference_index=inference_count,
                            source_row_index=timeline.current_row_index,
                        )
                    )
                    inference_count += 1
            assert command is not None and current_step is not None
            state = scene.state()
            snapshot = (
                _sonic_snapshot(current_step, int(frame.policy_seq))
                if isinstance(controller, SonicController)
                else PolicySnapshot(policy_seq=int(frame.policy_seq))
            )
            csv_writer.write_frame(
                source_row_index=timeline.current_row_index,
                sample_index=log_index,
                control_time_s=log_index / LOG_HZ,
                mujoco_time_s=float(scene.data.time),
                qpos=scene.data.qpos,
                qvel=scene.data.qvel,
                command_fields=_command_fields(command),
                policy_snapshot=snapshot if log_index % LOGS_PER_POLICY == 0 else None,
                reference_motion=(
                    current_step.encoder_input[4:644]
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
            source_qpos_samples.append(timeline.state()[0])
            reference_indices.append(int(frame.frame_index))
            contact_rows.append(contact_summary.as_dict())
            if viewer is not None:
                if not viewer.is_running():
                    raise KeyboardInterrupt("viewer closed")
                if log_index % LOGS_PER_POLICY == 0:
                    viewer.sync()
            scene.physics_step(PHYSICS_STEPS_PER_LOG)
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
    metrics_payload["root_assist"] = scene.root_assist.metadata()
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
            metadata_json=np.asarray(
                json.dumps(
                    {
                        "rate_hz": LOG_HZ,
                        "mapping": (
                            "phase-matched source qpos interpolated at each "
                            "exact 400 Hz rollout sample; source_row_index is "
                            "the nearest provenance row"
                        ),
                        "source_csv": str(source_csv),
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
    with (work_output / "prepared_reference.npz").open("wb") as reference_output:
        np.savez_compressed(
            reference_output,
            policy_seq=np.asarray(reference.policy_seq, dtype=np.int64),
            source_row_index=np.asarray(reference.source_row_index, dtype=np.int64),
            control_time_s=np.asarray(reference.control_time_s, dtype=np.float64),
            metadata_json=np.asarray(
                json.dumps(reference.metadata(), ensure_ascii=False, sort_keys=True)
            ),
        )
    write_data_schema(
        work_output / "data_schema.json",
        header=source_header,
        controller_family=controller_family,
        source_csv=source_csv,
        logging_hz=LOG_HZ,
        extra={
            "sonic_only_policy_fields": (
                "native values" if controller_family == "sonic" else "all zero"
            )
        },
    )
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
        model_paths=_controller_model_paths(controller),
        extra={
            "deterministic_single_process": True,
            "dds_or_zmq": False,
            "wall_clock_controls_reference": False,
            "source_history": history_context.metadata(),
            "reference": reference.metadata(),
            "controller": controller_metadata,
            "self_warmup_inferences": SELF_WARMUP_INFERENCES,
            "tracking_begins_at_inference": SELF_WARMUP_INFERENCES + 1,
            "warmup_saved_and_affects_physics": True,
            "semantic_task_success": "manual/unassigned; see metrics.json",
        },
    )
    write_json_atomic(
        work_output / "run_complete.json",
        {
            "complete": True,
            "logged_rows": log_index,
            "policy_inferences": inference_count,
            "wall_seconds": wall_seconds,
            "simulated_seconds": simulated_seconds,
            "source_time_zero_s": source_time_zero,
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
            "initialization": {
                "actual_start_source_row_index": int(source_row_indices[0]),
                "selected_policy_seq": history_context.selected_policy_seq,
            },
        },
    )
    write_source_history_artifacts(history_context, work_output)
    _remove_owned_output(output)
    work_output.replace(output)
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
