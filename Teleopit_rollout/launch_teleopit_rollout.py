#!/usr/bin/env python3
"""Run Teleopit v0.5.0 on qpos references inside a recorded task scene.

This is a single-process, fixed-simulation-clock experiment.  It does not use
DDS, ZMQ or a wall-clock reference publisher, so viewer performance cannot
change which reference frame reaches the policy.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback
from typing import Any, Sequence

import mujoco
import numpy as np
import onnxruntime


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from .constants import (
        LOG_HZ,
        PD_HZ,
        POLICY_HZ,
        TELEOPIT_COMMIT,
        TELEOPIT_VERSION,
    )
    from .download_assets import (
        ROBOT_ASSETS_SHA256,
        ROBOT_XML_SHA256,
        TRACK_G1_SHA256,
        sha256_file,
    )
    from .reference_data import PreparedReference, load_prepared_reference
    from .task_simulator import TaskSceneController
    from .teleopit_policy import (
        ReferenceFeatures,
        TeleopitObservationBuilder,
        TeleopitOnnxPolicy,
    )
except ImportError:  # pragma: no cover - direct script execution
    from constants import (
        LOG_HZ,
        PD_HZ,
        POLICY_HZ,
        TELEOPIT_COMMIT,
        TELEOPIT_VERSION,
    )
    from download_assets import (
        ROBOT_ASSETS_SHA256,
        ROBOT_XML_SHA256,
        TRACK_G1_SHA256,
        sha256_file,
    )
    from reference_data import PreparedReference, load_prepared_reference
    from task_simulator import TaskSceneController
    from teleopit_policy import (
        ReferenceFeatures,
        TeleopitObservationBuilder,
        TeleopitOnnxPolicy,
    )

from change_ckpt_track.task_sim_io import (  # noqa: E402
    CsvTimeline,
    ReplayCsvWriter,
    resolve_recording,
    stage_recording_snapshot,
    write_metadata,
)


DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "data"
DEFAULT_ASSET_MANIFEST = SCRIPT_DIR / "assets" / "manifest.json"
RUN_MARKER = ".teleopit_rollout_run"


@dataclass(frozen=True)
class AssetPaths:
    checkpoint: Path
    robot_xml: Path
    checkpoint_sha256: str
    robot_xml_sha256: str
    manifest: Path | None
    official_pinned_checkpoint: bool


@dataclass(frozen=True)
class RolloutResult:
    reason: str
    samples: int
    policy_ticks: int
    simulated_seconds: float
    wall_seconds: float
    fallen: bool
    invalid_state: bool
    source_exhausted: bool


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def resolve_assets(args: argparse.Namespace) -> AssetPaths:
    manifest_path = args.asset_manifest.expanduser().resolve()
    manifest: dict[str, Any] | None = None
    if manifest_path.is_file():
        manifest = _load_json(manifest_path)
    elif args.checkpoint is None or args.robot_xml is None:
        raise FileNotFoundError(
            f"Teleopit asset manifest not found: {manifest_path}. Run "
            "Teleopit_rollout/setup_env.sh first, or pass both --checkpoint "
            "and --robot-xml."
        )

    if args.checkpoint is not None:
        checkpoint = args.checkpoint.expanduser().resolve()
        official = False
    else:
        assert manifest is not None
        entry = manifest.get("assets", {}).get("track_g1_onnx", {})
        checkpoint = (manifest_path.parent / str(entry.get("path", ""))).resolve()
        if str(entry.get("expected_sha256", "")) != TRACK_G1_SHA256:
            raise ValueError("asset manifest does not describe the pinned track_g1 model")
        official = True

    if args.robot_xml is not None:
        robot_xml = args.robot_xml.expanduser().resolve()
    else:
        assert manifest is not None
        entry = manifest.get("assets", {}).get("robot_xml", {})
        robot_xml = (manifest_path.parent / str(entry.get("path", ""))).resolve()

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Teleopit checkpoint not found: {checkpoint}")
    if not robot_xml.is_file():
        raise FileNotFoundError(f"Teleopit FK robot XML not found: {robot_xml}")
    checkpoint_hash = sha256_file(checkpoint)
    robot_hash = sha256_file(robot_xml)
    if official and checkpoint_hash != TRACK_G1_SHA256:
        raise ValueError(
            f"pinned track_g1 checkpoint hash mismatch: {checkpoint_hash} != "
            f"{TRACK_G1_SHA256}"
        )
    if manifest is not None and args.robot_xml is None:
        expected_robot_hash = str(
            manifest.get("assets", {}).get("robot_xml", {}).get("sha256", "")
        )
        if not expected_robot_hash or robot_hash != expected_robot_hash:
            raise ValueError(
                "official robot XML no longer matches the verified asset manifest; "
                "rerun download_assets.py --offline"
            )
        if robot_hash != ROBOT_XML_SHA256:
            raise ValueError(
                f"pinned robot XML hash mismatch: {robot_hash} != {ROBOT_XML_SHA256}"
            )
        archive_hash = str(
            manifest.get("assets", {})
            .get("robot_assets_archive", {})
            .get("expected_sha256", "")
        )
        if archive_hash != ROBOT_ASSETS_SHA256:
            raise ValueError("asset manifest does not describe the pinned robot archive")
    return AssetPaths(
        checkpoint=checkpoint,
        robot_xml=robot_xml,
        checkpoint_sha256=checkpoint_hash,
        robot_xml_sha256=robot_hash,
        manifest=manifest_path if manifest is not None else None,
        official_pinned_checkpoint=official,
    )


def _default_run_name(recording: Path, root_assist: str) -> str:
    suffix = (
        "teleopit_no_root_assist"
        if root_assist == "none"
        else f"teleopit_root_assist_{root_assist}"
    )
    return f"{recording.name}_{suffix}"


def prepare_run_directory(
    output_root: Path,
    run_name: str,
    *,
    overwrite: bool,
    recording_dir: Path,
    root_assist: str,
    explicit_run_name: bool,
) -> Path:
    if not run_name or run_name in (".", "..") or Path(run_name).name != run_name:
        raise ValueError("--run-name must be one plain directory name")
    root = output_root.expanduser().resolve()
    run_dir = (root / run_name).resolve()
    if run_dir.parent != root:
        raise ValueError("run directory escaped --output-dir")
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"output already exists: {run_dir}; pass --overwrite to replace it"
            )
        marker = run_dir / RUN_MARKER
        if not marker.is_file():
            raise ValueError(
                f"refusing to overwrite unmarked directory: {run_dir}"
            )
        if not explicit_run_name:
            manifest_path = run_dir / "launch_manifest.json"
            if not manifest_path.is_file():
                raise ValueError(
                    "refusing to overwrite an incomplete automatically named run; "
                    "inspect it and use an explicit --run-name if replacement is intended"
                )
            previous = _load_json(manifest_path)
            previous_recording = Path(str(previous.get("recording", ""))).resolve()
            if (
                previous_recording != recording_dir.resolve()
                or previous.get("root_assist") != root_assist
            ):
                raise ValueError(
                    "the automatic output name belongs to a different recording/config; "
                    "use an explicit --run-name to avoid a basename collision"
                )
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    (run_dir / RUN_MARKER).write_text(
        "Created by Teleopit_rollout/launch_teleopit_rollout.py\n",
        encoding="utf-8",
    )
    return run_dir


class PolicyTelemetry:
    def __init__(self) -> None:
        self.policy_index: list[int] = []
        self.reference_frame_index: list[int] = []
        self.reference_is_hold: list[bool] = []
        self.policy_seq: list[int] = []
        self.source_row_index: list[int] = []
        self.sim_time_s: list[float] = []
        self.reference_qpos36: list[np.ndarray] = []
        self.reference_joint_vel: list[np.ndarray] = []
        self.reference_anchor_lin_vel_w: list[np.ndarray] = []
        self.reference_anchor_ang_vel_w: list[np.ndarray] = []
        self.observation: list[np.ndarray] = []
        self.observation_history: list[np.ndarray] = []
        self.raw_action: list[np.ndarray] = []
        self.q_target: list[np.ndarray] = []
        self.actual_joint_pos: list[np.ndarray] = []
        self.actual_joint_vel: list[np.ndarray] = []
        self.actual_root_qpos: list[np.ndarray] = []

    def append(
        self,
        *,
        frame_index: int,
        reference: PreparedReference,
        features: ReferenceFeatures,
        observation: np.ndarray,
        history: np.ndarray,
        raw_action: np.ndarray,
        q_target: np.ndarray,
        scene: TaskSceneController,
        reference_is_hold: bool,
    ) -> None:
        state = scene.robot_state()
        self.policy_index.append(len(self.policy_index))
        self.reference_frame_index.append(frame_index)
        self.reference_is_hold.append(reference_is_hold)
        self.policy_seq.append(int(reference.policy_seq[frame_index]))
        self.source_row_index.append(int(reference.source_row_index[frame_index]))
        self.sim_time_s.append(float(scene.data.time))
        self.reference_qpos36.append(features.qpos36.copy())
        self.reference_joint_vel.append(features.joint_vel.copy())
        self.reference_anchor_lin_vel_w.append(features.anchor_lin_vel_w.copy())
        self.reference_anchor_ang_vel_w.append(features.anchor_ang_vel_w.copy())
        self.observation.append(np.asarray(observation, dtype=np.float32).copy())
        self.observation_history.append(np.asarray(history, dtype=np.float32).copy())
        self.raw_action.append(np.asarray(raw_action, dtype=np.float32).copy())
        self.q_target.append(np.asarray(q_target, dtype=np.float32).copy())
        self.actual_joint_pos.append(np.asarray(state.joint_pos, dtype=np.float32))
        self.actual_joint_vel.append(np.asarray(state.joint_vel, dtype=np.float32))
        self.actual_root_qpos.append(
            np.concatenate((state.root_pos, state.root_quat_wxyz)).astype(np.float32)
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.policy_index:
            raise RuntimeError("cannot save empty Teleopit policy telemetry")
        np.savez_compressed(
            path,
            policy_index=np.asarray(self.policy_index, dtype=np.int64),
            reference_frame_index=np.asarray(
                self.reference_frame_index, dtype=np.int64
            ),
            reference_is_hold=np.asarray(self.reference_is_hold, dtype=np.bool_),
            policy_seq=np.asarray(self.policy_seq, dtype=np.int64),
            source_row_index=np.asarray(self.source_row_index, dtype=np.int64),
            sim_time_s=np.asarray(self.sim_time_s, dtype=np.float64),
            reference_qpos36=np.stack(self.reference_qpos36).astype(np.float32),
            reference_joint_vel=np.stack(self.reference_joint_vel).astype(np.float32),
            reference_anchor_lin_vel_w=np.stack(
                self.reference_anchor_lin_vel_w
            ).astype(np.float32),
            reference_anchor_ang_vel_w=np.stack(
                self.reference_anchor_ang_vel_w
            ).astype(np.float32),
            observation=np.stack(self.observation).astype(np.float32),
            observation_history=np.stack(self.observation_history).astype(np.float32),
            raw_action=np.stack(self.raw_action).astype(np.float32),
            q_target=np.stack(self.q_target).astype(np.float32),
            actual_joint_pos=np.stack(self.actual_joint_pos).astype(np.float32),
            actual_joint_vel=np.stack(self.actual_joint_vel).astype(np.float32),
            actual_root_qpos=np.stack(self.actual_root_qpos).astype(np.float32),
            observation_layout=np.asarray(
                "Teleopit v0.5.0 velcmd_history: obs[167], history[10,167]"
            ),
        )


class DeterministicRollout:
    LOG_DT = 1.0 / LOG_HZ
    PD_DT = 1.0 / PD_HZ
    POLICY_DT = 1.0 / POLICY_HZ
    LOGS_PER_PD = int(round(LOG_HZ / PD_HZ))
    LOGS_PER_POLICY = int(round(LOG_HZ / POLICY_HZ))

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        reference: PreparedReference,
        timeline: CsvTimeline,
        scene: TaskSceneController,
        observation_builder: TeleopitObservationBuilder,
        policy: TeleopitOnnxPolicy,
        writer: ReplayCsvWriter,
    ) -> None:
        self.args = args
        self.reference = reference
        self.timeline = timeline
        self.scene = scene
        self.observation_builder = observation_builder
        self.policy = policy
        self.writer = writer
        self.telemetry = PolicyTelemetry()
        ratio = self.LOG_DT / float(scene.model.opt.timestep)
        self.physics_steps_per_log = int(round(ratio))
        if self.physics_steps_per_log <= 0 or not np.isclose(
            ratio, self.physics_steps_per_log, rtol=0.0, atol=1e-10
        ):
            raise ValueError(
                f"scene timestep {scene.model.opt.timestep:g} does not divide "
                f"the fixed 400 Hz log dt {self.LOG_DT:g}"
            )
        if not np.isclose(scene.model.opt.timestep, 0.0005, rtol=0.0, atol=1e-12):
            raise ValueError(
                "recorded task physics must remain at XML timestep 0.0005 s; "
                f"got {scene.model.opt.timestep:g}"
            )
        if self.LOGS_PER_PD != 2 or self.LOGS_PER_POLICY != 8:
            raise AssertionError("unexpected fixed-clock ratios")
        self.reference_clock_steps = (
            self.reference.num_frames * self.LOGS_PER_POLICY
        )
        self.source_clock_steps = (
            self.timeline.last_row_index - self.timeline.start_row_index
        )
        self.total_reference_steps = min(
            self.reference_clock_steps, self.source_clock_steps
        )
        self.post_steps = int(round(self.args.post_rollout_seconds * LOG_HZ))
        self.previous_raw_action = np.zeros(29, dtype=np.float32)
        self.current_frame = 0
        self.source_active = True

    def _policy_tick(self, frame_index: int, *, hold_reference: bool = False) -> None:
        previous = (
            self.reference.qpos36[frame_index]
            if hold_reference
            else (
                None
                if frame_index == 0
                else self.reference.qpos36[frame_index - 1]
            )
        )
        features = self.observation_builder.reference_features(
            self.reference.qpos36[frame_index], previous
        )
        observation = self.observation_builder.build(
            self.scene.robot_state(), features, self.previous_raw_action
        )
        raw_action, q_target, history = self.policy.infer(observation)
        self.scene.set_policy_command(
            q_target,
            self.reference.left_hand_target[frame_index],
            self.reference.right_hand_target[frame_index],
        )
        self.telemetry.append(
            frame_index=frame_index,
            reference=self.reference,
            features=features,
            observation=observation,
            history=history,
            raw_action=raw_action,
            q_target=q_target,
            scene=self.scene,
            reference_is_hold=hold_reference,
        )
        self.previous_raw_action = raw_action.copy()
        self.current_frame = frame_index

    def _write(self, sample_index: int) -> None:
        self.writer.write(
            source_row_index=self.timeline.current_row_index,
            sample_index=sample_index,
            control_time=sample_index * self.LOG_DT,
            mujoco_time=float(self.scene.data.time),
            qpos=self.scene.data.qpos,
            qvel=self.scene.data.qvel,
            command=self.scene.command_snapshot(),
        )

    def run(self) -> RolloutResult:
        self.policy.reset()
        self._policy_tick(0)
        self.scene.update_pd_command()
        self._write(0)

        # A sliced smoke test ends after its selected policy frames.  A full
        # recording instead ends at its final 400 Hz source sample; this is
        # important for recordings whose observed 50 Hz groups contain 7/9
        # rows and whose final policy hold is shorter than a full 20 ms.
        total_steps = self.total_reference_steps + self.post_steps
        viewer = None
        viewer_module = None
        if self.args.viewer:
            import mujoco.viewer as viewer_module

            viewer = viewer_module.launch_passive(self.scene.model, self.scene.data)

        reason = "reference completed"
        fallen = False
        invalid = False
        samples = 1
        wall_started = time.monotonic()
        try:
            for step in range(1, total_steps + 1):
                if viewer is not None and not viewer.is_running():
                    reason = "viewer closed"
                    break

                in_recording_phase = step <= self.total_reference_steps
                if not in_recording_phase:
                    # A sliced run must hold the source row at the slice end;
                    # it must not combine the last selected policy command
                    # with root/task state from later, unselected recording rows.
                    self.source_active = False
                # The same torque is held through two 2.5 ms log intervals,
                # giving one exact 200 Hz PD interval.
                self.scene.physics_step(self.physics_steps_per_log)
                if in_recording_phase and self.source_active:
                    self.source_active = self.timeline.advance()
                source_qpos, source_qvel = self.timeline.state()

                pd_boundary = step % self.LOGS_PER_PD == 0
                policy_boundary = step % self.LOGS_PER_POLICY == 0
                if pd_boundary:
                    self.scene.apply_root_assist(
                        source_qpos,
                        source_qvel,
                        source_velocity_active=self.source_active,
                    )
                    if policy_boundary and step < total_steps:
                        next_frame = step // self.LOGS_PER_POLICY
                        if in_recording_phase and next_frame < self.reference.num_frames:
                            self._policy_tick(next_frame)
                        else:
                            # The policy keeps running at 50 Hz after the source
                            # ends, just as a deployed controller keeps consuming
                            # robot state.  Repeating the final reference pose
                            # makes reference q/dq and torso velocities exactly
                            # zero instead of freezing a stale, moving action.
                            self._policy_tick(
                                self.current_frame, hold_reference=True
                            )
                    self.scene.update_pd_command()

                self.scene.validate_state()
                invalid = False
                self._write(step)
                samples += 1

                if pd_boundary and float(self.scene.data.qpos[2]) < self.args.fall_height:
                    fallen = True
                    if self.args.stop_on_fall:
                        reason = (
                            f"robot fell below {self.args.fall_height:g} m at "
                            f"t={self.scene.data.time:.4f} s"
                        )
                        break

                if viewer is not None and (
                    policy_boundary or step == total_steps
                ):
                    viewer.sync()
                if self.args.pace_real_time:
                    deadline = wall_started + float(self.scene.data.time)
                    delay = deadline - time.monotonic()
                    if delay > 0.0:
                        time.sleep(delay)
        except FloatingPointError:
            invalid = True
            reason = "non-finite simulation state"
        finally:
            if viewer is not None:
                viewer.close()

        wall_seconds = time.monotonic() - wall_started
        return RolloutResult(
            reason=reason,
            samples=samples,
            policy_ticks=len(self.telemetry.policy_index),
            simulated_seconds=float(self.scene.data.time),
            wall_seconds=wall_seconds,
            fallen=fallen,
            invalid_state=invalid,
            source_exhausted=(
                self.timeline.current_row_index >= self.timeline.last_row_index
            ),
        )


def _orientation_error_degrees(actual: np.ndarray, reference: np.ndarray) -> np.ndarray:
    dots = np.sum(actual[:, 3:7] * reference[:, 3:7], axis=1)
    return np.degrees(2.0 * np.arccos(np.clip(np.abs(dots), 0.0, 1.0)))


def telemetry_summary(telemetry: PolicyTelemetry) -> dict[str, object]:
    actual_q = np.stack(telemetry.actual_joint_pos).astype(np.float64)
    reference_qpos = np.stack(telemetry.reference_qpos36).astype(np.float64)
    actual_root = np.stack(telemetry.actual_root_qpos).astype(np.float64)
    joint_rmse_by_tick = np.sqrt(np.mean((actual_q - reference_qpos[:, 7:36]) ** 2, axis=1))
    root_xy_error = np.linalg.norm(actual_root[:, :2] - reference_qpos[:, :2], axis=1)
    root_z_error = np.abs(actual_root[:, 2] - reference_qpos[:, 2])
    orientation_error = _orientation_error_degrees(actual_root, reference_qpos)
    return {
        "joint_position_rmse_rad_mean": float(np.mean(joint_rmse_by_tick)),
        "joint_position_rmse_rad_max": float(np.max(joint_rmse_by_tick)),
        "root_xy_error_m_mean": float(np.mean(root_xy_error)),
        "root_xy_error_m_max": float(np.max(root_xy_error)),
        "root_z_error_m_mean": float(np.mean(root_z_error)),
        "root_z_error_m_max": float(np.max(root_z_error)),
        "root_orientation_error_deg_mean": float(np.mean(orientation_error)),
        "root_orientation_error_deg_max": float(np.max(orientation_error)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path, help="recording directory or data.csv")
    parser.add_argument("--checkpoint", type=Path, help="custom track_g1 ONNX")
    parser.add_argument("--robot-xml", type=Path, help="custom Teleopit FK robot XML")
    parser.add_argument(
        "--asset-manifest", type=Path, default=DEFAULT_ASSET_MANIFEST
    )
    parser.add_argument("--asset-model-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--policy-offset", type=int, default=0)
    parser.add_argument("--policy-count", type=int)
    parser.add_argument(
        "--root-assist",
        choices=("none", "xy", "xyz"),
        default="none",
        help=(
            "oracle source-root hard alignment at 200 Hz; assisted comparisons "
            "must use the identical mode for SONIC and Teleopit"
        ),
    )
    parser.add_argument("--device", choices=("cpu", "auto"), default="cpu")
    parser.add_argument(
        "--post-rollout-seconds",
        type=float,
        default=1.0,
        help=(
            "hold the final policy/source pose after the recording (default: "
            "1.0 s, matching the existing SONIC qpos-track experiment)"
        ),
    )
    parser.add_argument("--fall-height", type=float, default=0.2)
    parser.add_argument(
        "--stop-on-fall", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--viewer", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--pace-real-time",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="pace by simulation time (default: on with viewer, off headless)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="load reference/model/ONNX and run one inference without writing output",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.policy_offset < 0:
        raise ValueError("--policy-offset must be non-negative")
    if args.policy_count is not None and args.policy_count <= 0:
        raise ValueError("--policy-count must be positive")
    if args.post_rollout_seconds < 0.0:
        raise ValueError("--post-rollout-seconds must be non-negative")
    if args.fall_height <= 0.0:
        raise ValueError("--fall-height must be positive")
    nonstandard: list[str] = []
    if args.checkpoint is not None:
        nonstandard.append("--checkpoint")
    if args.robot_xml is not None:
        nonstandard.append("--robot-xml")
    if args.asset_manifest.expanduser().resolve() != DEFAULT_ASSET_MANIFEST.resolve():
        nonstandard.append("--asset-manifest")
    if args.asset_model_root is not None:
        nonstandard.append("--asset-model-root")
    if args.policy_offset != 0:
        nonstandard.append("--policy-offset")
    if args.policy_count is not None:
        nonstandard.append("--policy-count")
    if not np.isclose(args.post_rollout_seconds, 1.0):
        nonstandard.append("--post-rollout-seconds")
    if not np.isclose(args.fall_height, 0.2):
        nonstandard.append("--fall-height")
    if not args.stop_on_fall:
        nonstandard.append("--no-stop-on-fall")
    if args.device != "cpu":
        nonstandard.append("--device")
    if args.viewer:
        nonstandard.append("--viewer")
    if nonstandard and args.run_name is None and not args.validate_only:
        raise ValueError(
            "non-default runs require an explicit --run-name to prevent output "
            f"collisions (changed: {', '.join(nonstandard)})"
        )
    if args.pace_real_time is None:
        args.pace_real_time = bool(args.viewer)


def _adapter_source_hashes() -> dict[str, str]:
    local_names = (
        "constants.py",
        "reference_data.py",
        "task_simulator.py",
        "teleopit_policy.py",
        "launch_teleopit_rollout.py",
    )
    result = {name: sha256_file(SCRIPT_DIR / name) for name in local_names}
    for relative in (
        "change_ckpt_track/qpos_reference_data.py",
        "change_ckpt_track/task_sim_io.py",
    ):
        result[relative] = sha256_file(REPO_ROOT / relative)
    return result


def _snapshot_xml_hashes(snapshot_root: Path) -> dict[str, str]:
    return {
        path.relative_to(snapshot_root).as_posix(): sha256_file(path)
        for path in sorted(snapshot_root.rglob("*.xml"))
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    recording_dir, source_csv = resolve_recording(args.recording)
    source_csv_sha256_start = sha256_file(source_csv)
    adapter_source_hashes = _adapter_source_hashes()
    reference = load_prepared_reference(
        recording_dir,
        policy_offset=args.policy_offset,
        policy_count=args.policy_count,
    )
    assets = resolve_assets(args)
    timeline = CsvTimeline(
        source_csv,
        policy_seq=None,
        row_index=reference.first_source_row_index,
        policy_offset=None,
    )

    run_dir: Path | None = None
    staged = None
    writer = None
    runner = None
    result = None
    error_text = None
    source_csv_sha256_end: str | None = None
    initial_state_qpos: np.ndarray | None = None
    try:
        # Complete scene/FK/ONNX preflight in a temporary snapshot before an
        # existing --overwrite target is touched.
        staged = stage_recording_snapshot(
            recording_dir,
            None,
            asset_model_root=args.asset_model_root,
        )
        scene = TaskSceneController(staged.scene_path, root_assist=args.root_assist)
        initial_qpos, initial_qvel = timeline.state()
        initial_state_qpos = initial_qpos.copy()
        scene.initialize(initial_qpos, initial_qvel)
        observation_builder = TeleopitObservationBuilder(assets.robot_xml)
        policy = TeleopitOnnxPolicy(assets.checkpoint, device=args.device)

        # Validate the entire adapter, not just ONNX file readability.
        initial_features = observation_builder.reference_features(
            reference.qpos36[0], None
        )
        initial_observation = observation_builder.build(
            scene.robot_state(), initial_features, np.zeros(29, dtype=np.float32)
        )
        policy.reset()
        policy.infer(initial_observation)
        policy.reset()

        print(f"[teleopit] source: {source_csv}")
        print(
            f"[teleopit] reference: frames={reference.num_frames}, "
            f"policy_seq={reference.policy_seq[0]}..{reference.policy_seq[-1]}, "
            f"source_row={reference.first_source_row_index}"
        )
        print(f"[teleopit] checkpoint: {assets.checkpoint}")
        print(f"[teleopit] FK model: {assets.robot_xml}")
        print(
            f"[teleopit] clocks: physics={1 / scene.model.opt.timestep:.0f} Hz, "
            f"PD={PD_HZ:.0f} Hz, policy={POLICY_HZ:.0f} Hz, log={LOG_HZ:.0f} Hz"
        )
        if args.root_assist != "none":
            print(
                "[teleopit] WARNING: oracle root assist is enabled; report this "
                "as an assisted controller-replacement result"
            )
        if args.validate_only:
            print("[teleopit] validation passed (167D obs, 10x167 history, 29D action)")
            return 0

        staged.close()
        staged = None
        run_name = args.run_name or _default_run_name(
            recording_dir, args.root_assist
        )
        run_dir = prepare_run_directory(
            args.output_dir,
            run_name,
            overwrite=args.overwrite,
            recording_dir=recording_dir,
            root_assist=args.root_assist,
            explicit_run_name=args.run_name is not None,
        )
        staged = stage_recording_snapshot(
            recording_dir,
            run_dir,
            asset_model_root=args.asset_model_root,
        )
        scene = TaskSceneController(staged.scene_path, root_assist=args.root_assist)
        scene.initialize(initial_qpos, initial_qvel)
        policy.reset()

        assert run_dir is not None
        reference.save_npz(run_dir / "prepared_reference.npz")
        writer = ReplayCsvWriter(
            run_dir / "data.csv",
            source_csv,
            timeline.header,
            staged.scene_path,
            timeline.qpos_columns,
            timeline.qvel_columns,
        )
        launch_manifest = {
            "schema_version": 1,
            "experiment_kind": "teleopit_qpos_track_in_recorded_task_scene",
            "created_at_unix_s": time.time(),
            "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
            "recording": str(recording_dir),
            "source_csv": str(source_csv),
            "source_csv_sha256_start": source_csv_sha256_start,
            "run_directory": str(run_dir),
            "scene_xml_sha256": sha256_file(staged.scene_path),
            "model_snapshot_xml_sha256": _snapshot_xml_hashes(
                staged.snapshot_root
            ),
            "external_asset_model_root": str(staged.asset_model_root),
            "external_asset_tree_hashed": False,
            "adapter_source_sha256": adapter_source_hashes,
            "root_assist": args.root_assist,
            "rollout_config": {
                "policy_offset": args.policy_offset,
                "policy_count": reference.num_frames,
                "post_rollout_seconds": args.post_rollout_seconds,
                "fall_height_m": args.fall_height,
                "stop_on_fall": args.stop_on_fall,
                "viewer": args.viewer,
                "pace_real_time": args.pace_real_time,
                "device_request": args.device,
                "onnx_execution_providers": list(policy.providers),
            },
            "teleopit": {
                "version": TELEOPIT_VERSION,
                "source_commit": TELEOPIT_COMMIT,
                "checkpoint": str(assets.checkpoint),
                "checkpoint_sha256": assets.checkpoint_sha256,
                "robot_xml": str(assets.robot_xml),
                "robot_xml_sha256": assets.robot_xml_sha256,
                "official_pinned_checkpoint": assets.official_pinned_checkpoint,
                "asset_manifest": str(assets.manifest) if assets.manifest else None,
                "asset_manifest_sha256": (
                    sha256_file(assets.manifest) if assets.manifest else None
                ),
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "mujoco": mujoco.__version__,
                "onnxruntime": onnxruntime.__version__,
            },
            "initialization": {
                "actual_start_source_row_index": reference.first_source_row_index,
                "policy_offset": args.policy_offset,
                "policy_count": reference.num_frames,
            },
        }
        write_metadata(run_dir / "launch_manifest.json", launch_manifest)
        runner = DeterministicRollout(
            args=args,
            reference=reference,
            timeline=timeline,
            scene=scene,
            observation_builder=observation_builder,
            policy=policy,
            writer=writer,
        )
        result = runner.run()
    except Exception:
        error_text = traceback.format_exc()
        raise
    finally:
        close_error = None
        if writer is not None:
            try:
                writer.close()
            except Exception:
                close_error = traceback.format_exc()
                if error_text is None:
                    error_text = close_error
            try:
                source_csv_sha256_end = sha256_file(source_csv)
                if source_csv_sha256_end != source_csv_sha256_start:
                    raise RuntimeError(
                        "source data.csv changed while the rollout was running: "
                        f"{source_csv_sha256_start} -> {source_csv_sha256_end}"
                    )
            except Exception:
                hash_error = traceback.format_exc()
                close_error = (
                    f"{close_error}\n{hash_error}" if close_error else hash_error
                )
                if error_text is None:
                    error_text = hash_error
        if run_dir is not None and runner is not None and runner.telemetry.policy_index:
            runner.telemetry.save(run_dir / "teleopit_policy.npz")
        if run_dir is not None and staged is not None and runner is not None:
            scene = runner.scene
            assert initial_state_qpos is not None
            source_qpos, _ = timeline.state()
            task_initial = initial_state_qpos[scene.task_qpos_start :]
            task_final = scene.data.qpos[scene.task_qpos_start :].copy()
            task_source = source_qpos[scene.task_qpos_start :]
            payload: dict[str, Any] = {
                "schema_version": 1,
                "experiment_kind": "teleopit_qpos_track_in_recorded_task_scene",
                "source_recording": str(recording_dir),
                "source_csv": str(source_csv),
                "source_csv_sha256_start": source_csv_sha256_start,
                "source_csv_sha256_end": source_csv_sha256_end,
                "scene_path": str(staged.scene_path),
                "scene_xml_sha256": sha256_file(staged.scene_path),
                "asset_model_root": str(staged.asset_model_root),
                "initial_row_index": reference.first_source_row_index,
                "initial_policy_seq": int(reference.policy_seq[0]),
                "physics_dt": float(scene.model.opt.timestep),
                "pd_dt": 1.0 / PD_HZ,
                "policy_dt": 1.0 / POLICY_HZ,
                "log_dt": 1.0 / LOG_HZ,
                "csv_sample_dt": 1.0 / LOG_HZ,
                "source_dt": 1.0 / LOG_HZ,
                "source_rows_per_control": 1,
                "reference_frames": reference.num_frames,
                "reference_hold_policy_ticks": int(
                    np.count_nonzero(runner.telemetry.reference_is_hold)
                ),
                "reference_clock_steps_planned": runner.reference_clock_steps,
                "source_clock_steps_available": runner.source_clock_steps,
                "recording_steps_run": runner.total_reference_steps,
                "post_rollout_seconds": args.post_rollout_seconds,
                "post_steps_planned": runner.post_steps,
                "physics_steps_per_log": runner.physics_steps_per_log,
                "root_assist": scene.root_assist.as_dict(),
                "task_qpos_start": scene.task_qpos_start,
                "task_qpos_labels": scene.task_qpos_labels,
                "initial_task_qpos": task_initial.tolist(),
                "final_task_qpos": task_final.tolist(),
                "current_source_task_qpos": task_source.tolist(),
                "task_qpos_motion_l2_from_initial": float(
                    np.linalg.norm(task_final - task_initial)
                ),
                "source_task_qpos_motion_l2_from_initial": float(
                    np.linalg.norm(task_source - task_initial)
                ),
                "task_qpos_max_abs_error_to_current_source": float(
                    np.max(np.abs(task_final - task_source))
                    if task_final.size
                    else 0.0
                ),
                "task_qpos_aggregate_metrics_warning": (
                    "The aggregate task-qpos values mix metres, radians and "
                    "quaternion components and are diagnostics, not a task score."
                ),
                "policy_telemetry_valid": bool(runner.telemetry.policy_index),
                "policy_telemetry_path": "teleopit_policy.npz",
                "prepared_reference_path": "prepared_reference.npz",
                "controller_adapter_note": (
                    "Teleopit observation/history/action mapping and gains are "
                    "used inside the original recorded task XML with explicit "
                    "200 Hz torque PD. Teleopit's standalone hand-free 5 ms "
                    "built-in-PD simulator is intentionally not substituted."
                ),
                "csv_reference_fields_note": (
                    "reference_motion/token fields in data.csv retain source-schema "
                    "provenance only and are not Teleopit observations. Exact Teleopit "
                    "observations/actions are in teleopit_policy.npz."
                ),
                "error": error_text,
                "csv_close_error": close_error,
            }
            if runner.telemetry.policy_index:
                payload["tracking_metrics_at_policy_rate"] = telemetry_summary(
                    runner.telemetry
                )
            if result is not None:
                payload.update(
                    {
                        "stop_reason": result.reason,
                        "samples": result.samples,
                        "policy_ticks": result.policy_ticks,
                        "simulated_seconds": result.simulated_seconds,
                        "wall_seconds": result.wall_seconds,
                        "real_time_factor": result.simulated_seconds
                        / max(result.wall_seconds, 1e-9),
                        "fallen": result.fallen,
                        "invalid_state": result.invalid_state,
                        "source_exhausted": result.source_exhausted,
                        "user_aborted": result.reason == "viewer closed",
                        "final_base_height_m": float(scene.data.qpos[2]),
                    }
                )
            write_metadata(run_dir / "run_metadata.json", payload)
        timeline.close()
        if staged is not None:
            staged.close()
        if close_error is not None and result is not None:
            raise RuntimeError(f"failed to finalize replay CSV/source:\n{close_error}")

    assert result is not None and run_dir is not None
    user_aborted = result.reason == "viewer closed"
    if not user_aborted:
        artifact_hashes = {
            name: sha256_file(run_dir / name)
            for name in (
                "data.csv",
                "prepared_reference.npz",
                "teleopit_policy.npz",
                "launch_manifest.json",
                "run_metadata.json",
            )
        }
        write_metadata(
            run_dir / "run_complete.json",
            {
                "schema_version": 1,
                "complete": True,
                "source_csv_sha256": source_csv_sha256_start,
                "samples": result.samples,
                "policy_ticks": result.policy_ticks,
                "artifact_sha256": artifact_hashes,
            },
        )
    print(
        f"[teleopit] stopped: {result.reason}; samples={result.samples}, "
        f"policy_ticks={result.policy_ticks}, sim={result.simulated_seconds:.3f}s, "
        f"wall={result.wall_seconds:.3f}s, RTF="
        f"{result.simulated_seconds / max(result.wall_seconds, 1e-9):.3f}"
    )
    print(f"[teleopit] output: {run_dir}")
    if user_aborted:
        print(
            "[teleopit] output is intentionally incomplete because the viewer "
            "was closed; run_complete.json was not written",
            file=sys.stderr,
        )
        return 6
    if result.invalid_state:
        return 5
    if result.fallen:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
