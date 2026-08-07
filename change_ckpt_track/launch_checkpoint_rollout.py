#!/usr/bin/env python3
"""Preflight and launch the qpos-track SONIC task-scene experiment.

This file deliberately lives outside the original simulator/deployment sources.
It launches only adapters copied into ``change_ckpt_track`` plus the existing
deploy executable, while keeping every generated artifact under
``change_ckpt_track/data`` by default.

Examples
--------
Check the low-latency model and all adapters without launching anything::

    python change_ckpt_track/launch_checkpoint_rollout.py preflight \
        --checkpoint low_latency

Run the real-time viewer and save a replay CSV (both are defaults)::

    python change_ckpt_track/launch_checkpoint_rollout.py run \
        --checkpoint low_latency

Run only the C++ deployment process (useful with three terminals)::

    python change_ckpt_track/launch_checkpoint_rollout.py deploy \
        --checkpoint low_latency

The streamed input contract is intentionally strict: protocol v1 motion data,
no ``token_state`` field, and C++ ``zmq_manager`` input.  A protocol-v4/external
token message is treated as a fatal error because it bypasses the local encoder.
The body values come only from recorded qpos/qvel/pelvis quaternion.  In the
optional regular recorded-window experiment, ``reference_motion`` is inspected
only to infer the ten temporal lags; none of its values are streamed.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
CHANGE_ROOT = REPO_ROOT / "change_ckpt_track"
CHECKPOINT_SOURCE_ROOT = REPO_ROOT / "change_ckpt"
DEFAULT_RECORDING = (
    REPO_ROOT / "sample_data/ztj/20260612/20260612_144117_g1_sim"
)
DEFAULT_DATA_ROOT = CHANGE_ROOT / "data"
DEPLOY_ROOT = REPO_ROOT / "gear_sonic_deploy"
DEPLOY_BINARY = DEPLOY_ROOT / "target/release/g1_deploy_onnx_ref"
SIM_SCRIPT = CHANGE_ROOT / "run_task_sim_loop.py"
PUBLISHER_SCRIPT = CHANGE_ROOT / "csv_reference_publisher.py"
SIM_PYTHON = REPO_ROOT / ".venv_sim/bin/python"
REFERENCE_FALLBACK = DEPLOY_ROOT / "reference/example"
PLANNER_MODEL = (
    CHECKPOINT_SOURCE_ROOT / "models/planner/target_vel/V2/planner_sonic.onnx"
)

# Dimensions are the current C++ observation-registry dimensions.  Keeping the
# table here makes preflight catch stale YAML aliases before TensorRT is started.
OBSERVATION_DIMS: dict[str, int] = {
    "token_state": 64,
    "his_base_angular_velocity_10frame_step1": 30,
    "his_body_joint_positions_10frame_step1": 290,
    "his_body_joint_velocities_10frame_step1": 290,
    "his_last_actions_10frame_step1": 290,
    "his_gravity_dir_10frame_step1": 30,
}

ENCODER_OBSERVATION_DIMS: dict[str, int] = {
    "encoder_mode": 3,
    "encoder_mode_4": 4,
    "motion_joint_positions_10frame_step5": 290,
    "motion_joint_velocities_10frame_step5": 290,
    "motion_anchor_orientation_10frame_step5": 60,
    "motion_joint_positions_10frame_step1": 290,
    "motion_joint_velocities_10frame_step1": 290,
    "motion_anchor_orientation_10frame_step1": 60,
    "motion_anchor_orientation_heading_10frame_step5": 60,
    "motion_anchor_orientation_heading_10frame_step1": 60,
    "motion_anchor_orientation_heading": 6,
    "motion_anchor_orientation": 6,
    "motion_root_z_position": 1,
    "motion_root_z_position_10frame_step5": 10,
    "motion_root_z_position_10frame_step1": 10,
    "motion_root_z_position_3frame_step1": 3,
    "motion_joint_positions_lowerbody_10frame_step5": 120,
    "motion_joint_velocities_lowerbody_10frame_step5": 120,
    "motion_joint_positions_lowerbody_10frame_step1": 120,
    "motion_joint_velocities_lowerbody_10frame_step1": 120,
    "vr_3point_local_target": 9,
    "vr_3point_local_orn_target": 12,
    "smpl_joints_10frame_step1": 720,
    "smpl_anchor_orientation_10frame_step1": 60,
    "smpl_anchor_orientation_heading_10frame_step1": 60,
    "motion_joint_positions_wrists_10frame_step1": 60,
    "smpl_joints_4frame_step1": 288,
    "smpl_anchor_orientation_4frame_step1": 24,
    "motion_joint_positions_wrists_4frame_step1": 24,
}

FORBIDDEN_DEPLOY_MESSAGES = (
    "protocol v4",
    "protocol version 4",
    "copied external token",
    "received external token",
    "using external token",
)

DEPLOY_READY_MESSAGES = (
    "g1deploy object created successfully",
)

DEPLOY_STREAM_READY_MESSAGES = (
    "merged streamed data:",
)

DEPLOY_CONSOLE_NOISE_PREFIXES = (
    "[ZMQEndpointInterface] Received ZMQ message",
    "[ZMQEndpointInterface] *** Starting ZMQ processing",
    "[ZMQEndpointInterface] *** End of ZMQ decoding processing",
    "[ZMQEndpointInterface] Protocol version:",
    "[ZMQEndpointInterface] Decoded body quaternions",
    "[ZMQEndpointInterface] Decoded left_hand_joints",
    "[ZMQEndpointInterface] Decoded right_hand_joints",
    "[ZMQEndpointInterface] Raw message field",
    "[ZMQEndpointInterface] catch_up field",
    "[ZMQEndpointInterface] Decoded data",
    "[ZMQEndpointInterface] Left hand joints set",
    "[ZMQEndpointInterface] Right hand joints set",
    "[ZMQEndpointInterface] Decode interval",
    "[ZMQEndpointInterface] active_protocol_version_",
    "[ZMQEndpointInterface] result.motion",
    "[ZMQEndpointInterface] motion name:",
    "[StreamedMotionMerger] Processing",
    "[StreamedMotionMerger] incoming_frame_start:",
    "[StreamedMotionMerger] Copying old data:",
    "[StreamedMotionMerger] Merged motion:",
    "[ZMQEndpointInterface] Merged streamed data:",
    "Motion streamed completed and waiting following motion",
)


class PreflightError(RuntimeError):
    """A deterministic configuration error found before launching processes."""


@dataclass(frozen=True)
class ModelFiles:
    name: str
    encoder: Path
    decoder: Path
    obs_config: Path
    expected_encoder_input: int
    expected_decoder_input: int = 994
    expected_token_dim: int = 64
    expected_action_dim: int = 29


def _normalise_checkpoint(name: str) -> str:
    aliases = {
        "regular": "regular",
        "release": "regular",
        "sonic_release": "regular",
        "low": "low_latency",
        "low-latency": "low_latency",
        "low_latency": "low_latency",
        "sonic_v1_1": "sonic_v1_1",
        "sonic-v1-1": "sonic_v1_1",
        "v1.1": "sonic_v1_1",
        "v1_1": "sonic_v1_1",
    }
    try:
        return aliases[name]
    except KeyError as exc:
        raise PreflightError(f"Unsupported checkpoint name: {name}") from exc


def _publisher_checkpoint_layout(checkpoint: str) -> str:
    """Return the publisher layout for a checkpoint family.

    SONIC v1.1 has its own explicit publisher layout so diagnostics retain the
    tested controller name.  It still streams world-frame quaternions and uses
    the canonical regular-style 10-frame, step-5 gather window; the C++
    observation registry performs the heading normalization.
    """
    return _normalise_checkpoint(checkpoint)


def _required_future_frames(checkpoint: str) -> int:
    checkpoint = _normalise_checkpoint(checkpoint)
    return 46 if checkpoint in ("regular", "sonic_v1_1") else 10


def resolve_model_files(args: argparse.Namespace) -> ModelFiles:
    checkpoint = _normalise_checkpoint(args.checkpoint)
    if checkpoint == "low_latency":
        model_root = CHECKPOINT_SOURCE_ROOT / "models/low_latency"
        default_config = model_root / "observation_config.yaml"
        encoder_input = 1247
    elif checkpoint == "sonic_v1_1":
        model_root = CHECKPOINT_SOURCE_ROOT / "models/v1.1"
        default_config = model_root / "observation_config.yaml"
        encoder_input = 1751
    else:
        model_root = CHECKPOINT_SOURCE_ROOT / "models/regular"
        default_config = (
            CHECKPOINT_SOURCE_ROOT / "observation_config_sonic_release.yaml"
        )
        encoder_input = 1751

    encoder = (
        Path(args.encoder).expanduser()
        if args.encoder
        else model_root / "model_encoder.onnx"
    )
    decoder = (
        Path(args.decoder).expanduser()
        if args.decoder
        else model_root / "model_decoder.onnx"
    )
    obs_config = (
        Path(args.obs_config).expanduser()
        if args.obs_config
        else default_config
    )
    return ModelFiles(
        checkpoint,
        _absolute(encoder),
        _absolute(decoder),
        _absolute(obs_config),
        encoder_input,
    )


def _absolute(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _recording_csv(recording: Path) -> Path:
    recording = _absolute(recording.expanduser())
    return recording / "data.csv" if recording.is_dir() else recording


@lru_cache(maxsize=8)
def _load_processed_reference(
    recording: str, drop_truncated_edges: bool
) -> Any:
    try:
        from change_ckpt_track.qpos_reference_data import load_qpos_reference
    except (ImportError, ModuleNotFoundError) as exc:
        try:
            # Direct script execution places change_ckpt_track at sys.path[0].
            from qpos_reference_data import load_qpos_reference
        except (ImportError, ModuleNotFoundError) as direct_exc:
            raise PreflightError(
                "change_ckpt_track/qpos_reference_data.py is missing or cannot be imported"
            ) from direct_exc
    try:
        return load_qpos_reference(
            recording, drop_truncated_edges=drop_truncated_edges
        )
    except Exception as exc:  # noqa: BLE001 - provide one preflight error type
        raise PreflightError(f"Cannot build qpos-derived reference: {exc}") from exc


def _processed_reference(args: argparse.Namespace) -> Any:
    recording = str(_absolute(Path(args.recording).expanduser()))
    return _load_processed_reference(
        recording, bool(getattr(args, "drop_truncated_edges", True))
    )


def _recorded_slot_lag_report(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from change_ckpt_track.recorded_slot_lags import (
            infer_recorded_slot_lags,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        try:
            from recorded_slot_lags import infer_recorded_slot_lags
        except (ImportError, ModuleNotFoundError) as direct_exc:
            raise PreflightError(
                "change_ckpt_track/recorded_slot_lags.py is missing or "
                "cannot be imported"
            ) from direct_exc
    try:
        return infer_recorded_slot_lags(
            str(_absolute(Path(args.recording).expanduser()))
        )
    except ValueError as exc:
        raise PreflightError(
            f"Cannot infer recorded regular future-window lags: {exc}"
        ) from exc


def _selected_reference_frame(args: argparse.Namespace) -> tuple[Any, int]:
    sequence = _processed_reference(args)
    offset = int(args.start_policy_offset)
    count = int(len(sequence.policy_seq))
    if offset < 0 or offset >= count:
        raise PreflightError(
            f"--start-policy-offset={offset} is outside the processed qpos "
            f"sequence [0, {count - 1}]"
        )
    return sequence, offset


def _regular_export_help() -> str:
    return f"""
The fixed-G1 regular export used by change_ckpt is expected at:
  {CHECKPOINT_SOURCE_ROOT / "models/regular"}
You may pass --encoder, --decoder, and --obs-config explicitly.
""".strip()


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        yaml = importlib.import_module("yaml")
    except ImportError as exc:
        raise PreflightError(
            f"PyYAML is required for model/config preflight; run this launcher from "
            f"the isaaclab environment or install PyYAML. ({exc})"
        ) from exc
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - report the source file cleanly
        raise PreflightError(f"Cannot parse observation config {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PreflightError(f"Observation config must contain a YAML mapping: {path}")
    return payload


def _enabled_names(items: Any, section: str) -> list[str]:
    if not isinstance(items, list):
        raise PreflightError(f"{section} must be a YAML list")
    names: list[str] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict) or "name" not in item:
            raise PreflightError(f"{section}[{index}] must contain a name")
        if item.get("enabled", True):
            names.append(str(item["name"]))
    return names


def _sum_known_dimensions(names: Iterable[str], dimensions: dict[str, int], section: str) -> int:
    total = 0
    unknown: list[str] = []
    for name in names:
        if name not in dimensions:
            unknown.append(name)
        else:
            total += dimensions[name]
    if unknown:
        joined = ", ".join(unknown)
        raise PreflightError(
            f"{section} uses names not supported by the current C++ observation registry: {joined}"
        )
    return total


def _inspect_onnx_in_process(path: Path) -> dict[str, list[list[Any]]]:
    onnx = importlib.import_module("onnx")
    model = onnx.load_model(str(path), load_external_data=False)

    def values(nodes: Any) -> list[list[Any]]:
        result: list[list[Any]] = []
        for node in nodes:
            shape: list[Any] = []
            for dim in node.type.tensor_type.shape.dim:
                shape.append(dim.dim_value or dim.dim_param or None)
            result.append([node.name, shape])
        return result

    return {"inputs": values(model.graph.input), "outputs": values(model.graph.output)}


def _onnx_python_candidates() -> list[Path]:
    candidates = [Path(sys.executable)]
    if os.environ.get("CONDA_PREFIX"):
        candidates.append(Path(os.environ["CONDA_PREFIX"]) / "bin/python")
    candidates.extend(
        [
            Path("/opt/conda/envs/isaaclab/bin/python"),
            Path("/usr/local/bin/python3"),
            Path("/usr/bin/python3"),
        ]
    )
    result: list[Path] = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.is_file() and resolved not in result:
            result.append(resolved)
    return result


def inspect_onnx(path: Path) -> dict[str, list[list[Any]]]:
    try:
        return _inspect_onnx_in_process(path)
    except ImportError:
        pass

    helper = r"""
import json, onnx, sys
m = onnx.load_model(sys.argv[1], load_external_data=False)
def vals(nodes):
    out = []
    for node in nodes:
        shape = []
        for d in node.type.tensor_type.shape.dim:
            shape.append(d.dim_value or d.dim_param or None)
        out.append([node.name, shape])
    return out
print(json.dumps({'inputs': vals(m.graph.input), 'outputs': vals(m.graph.output)}))
"""
    failures: list[str] = []
    for python in _onnx_python_candidates():
        completed = subprocess.run(
            [str(python), "-c", helper, str(path)],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            check=False,
        )
        if completed.returncode == 0:
            try:
                return json.loads(completed.stdout)
            except json.JSONDecodeError:
                failures.append(f"{python}: invalid JSON")
        else:
            last_line = completed.stderr.strip().splitlines()[-1:] or ["unknown error"]
            failures.append(f"{python}: {last_line[0]}")
    raise PreflightError(
        "Cannot inspect ONNX tensor dimensions. Tried: " + "; ".join(failures)
    )


def _feature_dim(io_items: list[list[Any]], kind: str, path: Path) -> tuple[list[Any], int]:
    if not io_items:
        raise PreflightError(f"ONNX {kind} list is empty: {path}")
    shape = io_items[0][1]
    if not isinstance(shape, list) or len(shape) != 2 or not isinstance(shape[1], int):
        raise PreflightError(f"Expected a rank-2 primary ONNX {kind}, got {shape}: {path}")
    return shape, int(shape[1])


def validate_model_and_config(model: ModelFiles) -> dict[str, Any]:
    missing = [path for path in (model.encoder, model.decoder, model.obs_config) if not path.is_file()]
    if missing:
        detail = "\n".join(f"  - {path}" for path in missing)
        suffix = f"\n\n{_regular_export_help()}" if model.name == "regular" else ""
        raise PreflightError(f"Missing model/config files:\n{detail}{suffix}")

    config = _load_yaml(model.obs_config)
    policy_names = _enabled_names(config.get("observations"), "observations")
    encoder_cfg = config.get("encoder")
    if not isinstance(encoder_cfg, dict):
        raise PreflightError("Observation config is missing encoder mapping")
    token_dim = int(encoder_cfg.get("dimension", 0))
    encoder_names = _enabled_names(
        encoder_cfg.get("encoder_observations"), "encoder.encoder_observations"
    )

    # token_state follows encoder.dimension; all other policy dimensions are fixed.
    policy_dimensions = dict(OBSERVATION_DIMS)
    policy_dimensions["token_state"] = token_dim
    policy_total = _sum_known_dimensions(policy_names, policy_dimensions, "observations")
    encoder_total = _sum_known_dimensions(
        encoder_names, ENCODER_OBSERVATION_DIMS, "encoder.encoder_observations"
    )

    encoder_io = inspect_onnx(model.encoder)
    decoder_io = inspect_onnx(model.decoder)
    encoder_input_shape, encoder_input = _feature_dim(
        encoder_io["inputs"], "input", model.encoder
    )
    encoder_output_shape, encoder_output = _feature_dim(
        encoder_io["outputs"], "output", model.encoder
    )
    decoder_input_shape, decoder_input = _feature_dim(
        decoder_io["inputs"], "input", model.decoder
    )
    decoder_output_shape, decoder_output = _feature_dim(
        decoder_io["outputs"], "output", model.decoder
    )

    checks = {
        "encoder config total": (encoder_total, encoder_input),
        "encoder expected input": (encoder_input, model.expected_encoder_input),
        "encoder token output": (encoder_output, model.expected_token_dim),
        "YAML encoder dimension": (token_dim, model.expected_token_dim),
        "policy config total": (policy_total, decoder_input),
        "decoder expected input": (decoder_input, model.expected_decoder_input),
        "decoder action output": (decoder_output, model.expected_action_dim),
    }
    mismatches = [f"{label}: {actual} != {expected}" for label, (actual, expected) in checks.items() if actual != expected]
    if mismatches:
        raise PreflightError("Model/config dimension mismatch:\n  - " + "\n  - ".join(mismatches))

    return {
        "checkpoint": model.name,
        "encoder": str(model.encoder),
        "encoder_io": {"input": encoder_input_shape, "output": encoder_output_shape},
        "decoder": str(model.decoder),
        "decoder_io": {"input": decoder_input_shape, "output": decoder_output_shape},
        "obs_config": str(model.obs_config),
        "encoder_observations": encoder_names,
        "policy_observations": policy_names,
    }


def _literal_module_constants(path: Path) -> dict[str, Any]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise PreflightError(f"Cannot parse publisher source {path}: {exc}") from exc
    constants: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value_node = node.value
        if value_node is None:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                try:
                    constants[target.id] = ast.literal_eval(value_node)
                except (ValueError, TypeError):
                    pass
    return constants


def validate_publisher_contract(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PreflightError(f"CSV reference publisher does not exist: {path}")
    constants = _literal_module_constants(path)
    protocol = constants.get("PROTOCOL_VERSION")
    publishes_token = constants.get("PUBLISHES_EXTERNAL_TOKEN")
    if protocol != 1:
        raise PreflightError(
            f"Publisher must expose PROTOCOL_VERSION = 1; found {protocol!r} in {path}"
        )
    if publishes_token is not False:
        raise PreflightError(
            "Publisher must expose PUBLISHES_EXTERNAL_TOKEN = False so preflight can "
            f"guarantee local encoder use; found {publishes_token!r}"
        )
    completed = subprocess.run(
        [str(SIM_PYTHON), str(path), "--describe-protocol"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        raise PreflightError(
            f"Publisher --describe-protocol failed with code {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    try:
        described = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise PreflightError("Publisher --describe-protocol did not return JSON") from exc
    fields = described.get("fields", {})
    required_fields = {"joint_pos", "joint_vel", "body_quat_w", "frame_index"}
    missing_fields = sorted(required_fields.difference(fields)) if isinstance(fields, dict) else sorted(required_fields)
    if described.get("version") != 1 or described.get("publishes_external_token") is not False:
        raise PreflightError(f"Publisher runtime protocol contract is unsafe: {described}")
    if not isinstance(fields, dict) or "token_state" in fields or missing_fields:
        raise PreflightError(
            "Publisher protocol fields are invalid: "
            f"missing={missing_fields}, token_state_present={isinstance(fields, dict) and 'token_state' in fields}"
        )
    return {
        "protocol_version": protocol,
        "publishes_external_token": publishes_token,
        "fields": sorted(fields),
        "publisher": str(path),
    }


def _probe_help(python: Path, script: Path, required_options: Sequence[str] = ()) -> None:
    completed = subprocess.run(
        [str(python), str(script), "--help"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-15:])
        raise PreflightError(f"{script.name} --help failed with code {completed.returncode}:\n{tail}")
    missing = [option for option in required_options if option not in completed.stdout]
    if missing:
        raise PreflightError(
            f"{script.name} is missing launcher-required CLI options: {', '.join(missing)}"
        )


def _validate_recording(recording: Path) -> dict[str, Any]:
    csv_path = _recording_csv(recording)
    if not csv_path.is_file():
        raise PreflightError(f"Recording data.csv does not exist: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
            first = next(reader)
            second = next(reader)
        except StopIteration as exc:
            raise PreflightError(
                f"Recording CSV needs at least two data rows: {csv_path}"
            ) from exc
    required = {
        "policy_seq": lambda name: name == "policy_seq",
        "qpos0": lambda name: name.startswith("qpos:") and "[qpos0]" in name,
        "qvel0": lambda name: name.startswith("qvel:") and "[qvel0]" in name,
        "left_hand_q[0]": lambda name: name == "left_hand_q[0]",
        "right_hand_q[0]": lambda name: name == "right_hand_q[0]",
        "control_time_s": lambda name: name == "control_time_s",
    }
    missing = [label for label, predicate in required.items() if not any(predicate(name) for name in header)]
    if missing:
        raise PreflightError(f"Recording CSV is missing required columns: {', '.join(missing)}")
    if len(first) != len(header):
        raise PreflightError(
            f"First CSV data row has {len(first)} values but header has {len(header)} columns"
        )
    if len(second) != len(header):
        raise PreflightError(
            f"Second CSV data row has {len(second)} values but header has {len(header)} columns"
        )
    time_column = header.index("control_time_s")
    try:
        source_row_dt = float(second[time_column]) - float(first[time_column])
    except ValueError as exc:
        raise PreflightError("Recording control_time_s is not numeric in its first two rows") from exc
    if not math.isfinite(source_row_dt) or source_row_dt <= 0:
        raise PreflightError(
            f"Recording has invalid first-row control_time_s delta: {source_row_dt!r}"
        )
    recording_root = csv_path.parent
    scene = recording_root / "model_snapshot/mujoco/model/g1/scene_43dof.xml"
    if not scene.is_file():
        raise PreflightError(f"Task snapshot scene does not exist: {scene}")
    sequence = _load_processed_reference(str(recording_root.resolve()), True)
    return {
        "recording": str(recording_root),
        "data_csv": str(csv_path),
        "csv_columns": len(header),
        "source_row_dt_s": source_row_dt,
        "scene": str(scene),
        "reference_kind": "recorded_robot_qpos_track",
        "processed_50hz_frames_default": int(len(sequence.policy_seq)),
        "processed_first_policy_seq_default": int(sequence.policy_seq[0]),
        "processed_last_policy_seq_default": int(sequence.policy_seq[-1]),
        "truncated_edges_dropped_by_default": True,
    }


def _check_deploy_binary() -> dict[str, Any]:
    if not DEPLOY_BINARY.is_file() or not os.access(DEPLOY_BINARY, os.X_OK):
        raise PreflightError(
            f"Deploy executable is missing: {DEPLOY_BINARY}\n"
            "Build it from gear_sonic_deploy before running this experiment."
        )
    if not PLANNER_MODEL.is_file():
        raise PreflightError(
            f"Planner model is required by the current zmq_manager even in streamed-motion "
            f"mode, but it is missing: {PLANNER_MODEL}"
        )
    completed = subprocess.run(
        ["ldd", str(DEPLOY_BINARY)],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
        check=False,
    )
    missing = [line.strip() for line in completed.stdout.splitlines() if "not found" in line]
    if completed.returncode != 0 or missing:
        raise PreflightError(
            f"Deploy executable has unresolved shared libraries:\n  " + "\n  ".join(missing or [completed.stdout.strip()])
        )
    return {
        "deploy_binary": str(DEPLOY_BINARY),
        "planner_model": str(PLANNER_MODEL),
        "shared_libraries": "resolved",
    }


def _check_port_available(port: int) -> None:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except PermissionError:
        # Some managed/container sandboxes deny socket creation even for a
        # read-only availability probe.  The publisher's bind remains the
        # authoritative check when the real run starts.
        print(
            f"WARNING: sandbox denied the port-{port} availability probe; "
            "the publisher will verify it at bind time.",
            file=sys.stderr,
        )
        return
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", port))
    except OSError as exc:
        raise PreflightError(
            f"TCP port {port} is already in use; stop an old CSV/VLA publisher first. ({exc})"
        ) from exc
    finally:
        sock.close()


def _runtime_conflicts() -> list[dict[str, Any]]:
    """Find old simulator/deploy processes that would share DDS or ZMQ state."""
    watched = {
        "g1_deploy_onnx_ref",
        "run_sim_loop.py",
        "run_task_sim_loop.py",
        "csv_reference_publisher.py",
    }
    conflicts: list[dict[str, Any]] = []
    own_pids = {os.getpid(), os.getppid()}
    proc = Path("/proc")
    if not proc.is_dir():
        return conflicts
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) in own_pids:
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        argv = [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]
        matches = sorted({Path(argument).name for argument in argv}.intersection(watched))
        if matches:
            conflicts.append(
                {
                    "pid": int(entry.name),
                    "matches": matches,
                    "command": shlex.join(argv),
                }
            )
    return sorted(conflicts, key=lambda item: int(item["pid"]))


def _check_runtime_conflicts() -> None:
    conflicts = _runtime_conflicts()
    if not conflicts:
        return
    rendered = "\n".join(
        f"  PID {item['pid']}: {item['command']}" for item in conflicts
    )
    raise PreflightError(
        "Existing SONIC simulator/deploy processes could contaminate the DDS/ZMQ run. "
        "Stop them first, or use --skip-process-check only after verifying isolation:\n"
        + rendered
    )


def _probe_recording_adapters(args: argparse.Namespace, model: ModelFiles) -> None:
    """Load the exact task XML/assets and prepare the exact reference selection."""
    sequence, offset = _selected_reference_frame(args)
    actual_policy_seq = int(sequence.policy_seq[offset])
    sim_command = [
        str(SIM_PYTHON),
        str(SIM_SCRIPT),
        str(_absolute(Path(args.recording))),
        "--policy-seq",
        str(actual_policy_seq),
        "--physics-dt",
        str(args.physics_dt),
        "--control-dt",
        str(args.control_dt),
        "--source-dt",
        str(args.source_dt),
        "--root-assist",
        args.root_assist,
        "--no-save-csv",
        "--no-viewer",
        "--no-wait-for-lowcmd",
        "--dry-run",
    ]
    if args.asset_model_root:
        sim_command.extend(
            ["--asset-model-root", str(_absolute(Path(args.asset_model_root)))]
        )
    publisher_command_line = [
        str(SIM_PYTHON),
        str(PUBLISHER_SCRIPT),
        "--input",
        str(_absolute(Path(args.recording))),
        "--rate",
        str(args.reference_rate),
        "--start-policy-offset",
        str(args.start_policy_offset),
        "--chunk-size",
        str(args.chunk_size),
        "--lookahead",
        str(args.lookahead),
        "--checkpoint-layout",
        _publisher_checkpoint_layout(model.name),
        "--regular-future-window",
        args.regular_future_window,
        (
            "--drop-truncated-edges"
            if args.drop_truncated_edges
            else "--no-drop-truncated-edges"
        ),
        "--dry-run",
    ]
    for label, command in (("simulator", sim_command), ("publisher", publisher_command_line)):
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=120,
            check=False,
        )
        if completed.returncode != 0:
            tail = "\n".join(completed.stdout.splitlines()[-30:])
            raise PreflightError(
                f"Exact {label} recording dry-run failed with code "
                f"{completed.returncode}:\n{tail}"
            )


def run_preflight(args: argparse.Namespace, *, include_adapters: bool = True) -> dict[str, Any]:
    model = resolve_model_files(args)
    report: dict[str, Any] = {"model": validate_model_and_config(model)}
    if not include_adapters:
        return report
    if (
        args.regular_future_window == "recorded"
        and model.name != "regular"
    ):
        raise PreflightError(
            "--regular-future-window recorded is valid only with "
            "--checkpoint regular"
        )

    recording = Path(args.recording)
    report["recording"] = _validate_recording(recording)
    if args.regular_future_window == "recorded":
        report["recorded_slot_lag_inference"] = (
            _recorded_slot_lag_report(args)
        )
    if not SIM_PYTHON.is_file():
        raise PreflightError(f"MuJoCo Python environment does not exist: {SIM_PYTHON}")
    if not SIM_SCRIPT.is_file():
        raise PreflightError(f"Task simulator copy does not exist: {SIM_SCRIPT}")
    report["publisher"] = validate_publisher_contract(PUBLISHER_SCRIPT)
    _probe_help(
        SIM_PYTHON,
        SIM_SCRIPT,
        (
            "--policy-offset",
            "--gate-status-file",
            "--run-dir",
            "--save-csv",
            "--viewer",
            "--root-assist",
        ),
    )
    _probe_help(
        SIM_PYTHON,
        PUBLISHER_SCRIPT,
        (
            "--start-policy-offset",
            "--checkpoint-layout",
            "--regular-future-window",
            "--gate-status-file",
            "--publisher-status-file",
            "--chunk-size",
            "--lookahead",
        ),
    )
    report["deploy"] = _check_deploy_binary()
    if hasattr(args, "control_dt"):
        _probe_recording_adapters(args, model)
        report["adapter_dry_run"] = {
            "simulator_model_and_assets": "passed",
            "publisher_reference_preparation": "passed",
        }
    if not args.skip_port_check:
        _check_port_available(args.port)
    report["runtime"] = {
        "sim_python": str(SIM_PYTHON),
        "sim_script": str(SIM_SCRIPT),
        "output_root": str(_absolute(Path(args.output_root))),
        "zmq": {"protocol": 1, "host": args.deploy_host, "port": args.port, "topic": args.topic},
        "local_encoder_required": True,
    }
    return report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _network_interface() -> str:
    # Reading sysfs works in managed sandboxes where socket.if_nameindex() may
    # be denied; the actual DDS processes still validate the interface later.
    if Path("/sys/class/net/lo").exists():
        return "lo"
    if Path("/sys/class/net/lo0").exists():
        return "lo0"
    try:
        names = {name for _, name in socket.if_nameindex()}
    except PermissionError as exc:
        raise PreflightError("Sandbox denied loopback-interface discovery") from exc
    if "lo" in names:
        return "lo"
    if "lo0" in names:
        return "lo0"
    raise PreflightError("No loopback network interface (lo/lo0) was found")


def deploy_command(args: argparse.Namespace, model: ModelFiles, logs_dir: Path) -> list[str]:
    locked = (
        "--input-type",
        "--encoder-file",
        "--planner-file",
        "--obs-config",
        "--zmq-host",
        "--zmq-port",
        "--zmq-topic",
    )
    if any(item == option or item.startswith(option + "=") for item in args.deploy_arg for option in locked):
        raise PreflightError(
            "--deploy-arg may not override input/model/ZMQ routing options; "
            "those are locked to preserve local-encoder and stream-alignment semantics"
        )
    command = [
        str(DEPLOY_BINARY),
        _network_interface(),
        str(model.decoder),
        str(REFERENCE_FALLBACK),
        "--obs-config",
        str(model.obs_config),
        "--encoder-file",
        str(model.encoder),
        # ZMQManager currently stops immediately when G1Deploy was constructed
        # without a planner, even if the first network command will switch it
        # to streamed-motion mode.  Match deploy.sh and provide the released
        # planner so the process remains alive long enough to receive that
        # switch command.
        "--planner-file",
        str(PLANNER_MODEL),
        "--input-type",
        "zmq_manager",
        "--output-type",
        "all",
        "--zmq-host",
        args.deploy_host,
        "--zmq-port",
        str(args.port),
        "--zmq-topic",
        args.topic,
        "--disable-crc-check",
        "--enable-csv-logs",
        "--logs-dir",
        str(logs_dir),
        "--target-motion-logfile",
        str(logs_dir.parent / "target_motion.csv"),
    ]
    command.extend(args.deploy_arg)
    return command


def simulator_command(args: argparse.Namespace, run_dir: Path) -> list[str]:
    locked = (
        "--run-dir",
        "--output-dir",
        "--policy-offset",
        "--policy-seq",
        "--row-index",
        "--gate-status-file",
        "--save-csv",
        "--no-save-csv",
        "--viewer",
        "--no-viewer",
        "--stop-at-source-end",
        "--no-stop-at-source-end",
        "--control-dt",
        "--source-dt",
        "--root-assist",
        "--physics-dt",
        "--viewer-dt",
        "--fall-height",
        "--wait-for-lowcmd",
        "--no-wait-for-lowcmd",
    )
    if any(item == option or item.startswith(option + "=") for item in args.sim_arg for option in locked):
        raise PreflightError(
            "--sim-arg may not override initialization, timing, gate, viewer, or output options"
        )
    command = [
        str(SIM_PYTHON),
        str(SIM_SCRIPT),
        str(_absolute(Path(args.recording))),
        "--run-dir",
        str(run_dir),
        "--control-dt",
        str(args.control_dt),
        "--source-dt",
        str(args.source_dt),
        "--root-assist",
        args.root_assist,
        "--physics-dt",
        str(args.physics_dt),
        "--viewer-dt",
        str(args.viewer_dt),
        "--fall-height",
        str(args.fall_height),
        "--save-csv" if args.save_csv else "--no-save-csv",
        "--viewer" if args.viewer else "--no-viewer",
        "--stop-at-source-end" if args.stop_at_source_end else "--no-stop-at-source-end",
        "--wait-for-lowcmd",
        "--gate-status-file",
        str(run_dir / "control_gate.status"),
    ]
    # The offset is defined inside the processed qpos sequence (after optional
    # edge trimming).  Resolve it back to the recording's actual policy_seq so
    # the 400 Hz simulator timeline and 50 Hz publisher start on the same row
    # without applying the trim/offset twice.
    sequence, offset = _selected_reference_frame(args)
    command.extend(["--policy-seq", str(int(sequence.policy_seq[offset]))])
    if args.asset_model_root:
        command.extend(["--asset-model-root", str(_absolute(Path(args.asset_model_root)))])
    command.extend(args.sim_arg)
    return command


def publisher_command(args: argparse.Namespace, run_dir: Path) -> list[str]:
    locked = (
        "--input",
        "--host",
        "--port",
        "--topic",
        "--rate",
        "--start-policy-offset",
        "--max-policy-frames",
        "--chunk-size",
        "--lookahead",
        "--checkpoint-layout",
        "--regular-future-window",
        "--base-sample",
        "--drop-truncated-edges",
        "--no-drop-truncated-edges",
        "--diagnostics-json",
        "--prepared-output",
        "--gate-status-file",
        "--gate-ready-value",
        "--publisher-status-file",
        "--initial-pose-retries",
        "--gate-poll-interval",
        "--apply-heading-correction",
        "--heading-correction-tick",
        "--no-command",
        "--no-hands",
        "--dry-run",
    )
    if any(item == option or item.startswith(option + "=") for item in args.publisher_arg for option in locked):
        raise PreflightError(
            "--publisher-arg may not override source, layout, timing, gate, or output options"
        )
    command = [
        str(SIM_PYTHON),
        str(PUBLISHER_SCRIPT),
        "--input",
        str(_absolute(Path(args.recording))),
        "--host",
        args.publish_host,
        "--port",
        str(args.port),
        "--topic",
        args.topic,
        "--rate",
        str(args.reference_rate),
        "--start-policy-offset",
        str(args.start_policy_offset),
        "--chunk-size",
        str(args.chunk_size),
        "--lookahead",
        str(args.lookahead),
        "--checkpoint-layout",
        _publisher_checkpoint_layout(args.checkpoint),
        "--regular-future-window",
        args.regular_future_window,
        (
            "--drop-truncated-edges"
            if args.drop_truncated_edges
            else "--no-drop-truncated-edges"
        ),
        "--diagnostics-json",
        str(run_dir / "reference_diagnostics.json"),
        "--prepared-output",
        str(run_dir / "prepared_reference.npz"),
        "--gate-status-file",
        str(run_dir / "control_gate.status"),
        "--publisher-status-file",
        str(run_dir / "publisher.status"),
    ]
    if args.heading_correction:
        command.extend(["--apply-heading-correction", "--heading-correction-tick", "1"])
    command.extend(args.publisher_arg)
    return command


def _new_run_dir(args: argparse.Namespace, checkpoint: str) -> Path:
    root = _absolute(Path(args.output_root))
    root.mkdir(parents=True, exist_ok=True)
    recording_name = _recording_csv(Path(args.recording)).parent.name
    if not recording_name:
        raise PreflightError(
            f"Cannot derive a recording directory name from {args.recording!r}"
        )

    suffix_parts = [checkpoint]
    if checkpoint == "regular" and args.regular_future_window == "recorded":
        suffix_parts.append("recorded_lags")
    if args.root_assist != "none":
        suffix_parts.extend(("root_assist", args.root_assist))
    result = root / f"{recording_name}_{'_'.join(suffix_parts)}"
    if result.exists() or result.is_symlink():
        print(f"[launcher] replacing existing run directory: {result}", flush=True)
        if result.is_dir() and not result.is_symlink():
            shutil.rmtree(result)
        else:
            result.unlink()
    result.mkdir(parents=False)
    return result


def _validate_launch_settings(
    args: argparse.Namespace,
    model: ModelFiles,
    report: dict[str, Any],
) -> None:
    output_root = _absolute(Path(args.output_root))
    if (
        args.regular_future_window == "recorded"
        and model.name != "regular"
    ):
        raise PreflightError(
            "--regular-future-window recorded is valid only with "
            "--checkpoint regular"
        )
    if args.regular_future_window == "recorded":
        lag_report = report.get("recorded_slot_lag_inference")
        lags = (
            lag_report.get("future_slot_policy_lags")
            if isinstance(lag_report, dict)
            else None
        )
        if (
            not isinstance(lags, list)
            or len(lags) != 10
            or lags[0] != 0
        ):
            raise PreflightError(
                "Recorded regular future-window preflight did not produce "
                f"a valid ten-slot lag vector: {lags!r}"
            )
    try:
        output_root.relative_to(DEFAULT_DATA_ROOT.resolve())
    except ValueError as exc:
        raise PreflightError(
            f"Run output must stay under {DEFAULT_DATA_ROOT.resolve()}, got {output_root}"
        ) from exc
    if not 1 <= args.port <= 65535:
        raise PreflightError(f"Invalid ZMQ port: {args.port}")
    if args.start_policy_offset < 0:
        raise PreflightError("--start-policy-offset must be non-negative")
    positive = {
        "--control-dt": args.control_dt,
        "--source-dt": args.source_dt,
        "--physics-dt": args.physics_dt,
        "--viewer-dt": args.viewer_dt,
        "--reference-rate": args.reference_rate,
        "--deploy-ready-timeout": args.deploy_ready_timeout,
        "--control-start-timeout": args.control_start_timeout,
    }
    invalid = [f"{name}={value}" for name, value in positive.items() if value <= 0]
    if invalid:
        raise PreflightError("Timing values must be positive: " + ", ".join(invalid))
    if not math.isclose(args.control_dt, 0.005, rel_tol=0.0, abs_tol=1e-12):
        raise PreflightError(
            "--control-dt must remain 0.005 s: the C++ deployment and repository "
            "sim2sim control loop run at 200 Hz"
        )
    if not math.isclose(args.reference_rate, 50.0, rel_tol=0.0, abs_tol=1e-9):
        raise PreflightError(
            "--reference-rate must remain 50 Hz to match the C++ streamed-motion clock"
        )
    recorded_source_dt = float(report["recording"]["source_row_dt_s"])
    if not math.isclose(args.source_dt, recorded_source_dt, rel_tol=0.0, abs_tol=1e-9):
        raise PreflightError(
            f"--source-dt={args.source_dt:g} does not match the recording's first-row "
            f"control_time_s delta ({recorded_source_dt:g} s)"
        )
    for numerator_name, denominator_name, ratio in (
        ("control_dt", "physics_dt", args.control_dt / args.physics_dt),
        ("control_dt", "source_dt", args.control_dt / args.source_dt),
    ):
        if round(ratio) < 1 or abs(ratio - round(ratio)) > 1e-9:
            raise PreflightError(
                f"{numerator_name} must be an integer multiple of {denominator_name}; "
                f"got ratio {ratio:g}"
            )
    if args.chunk_size <= 0 or args.lookahead <= 0:
        raise PreflightError("--chunk-size and --lookahead must be positive")
    required_future = _required_future_frames(model.name)
    available_future = min(args.chunk_size, args.lookahead)
    if available_future < required_future:
        raise PreflightError(
            f"{model.name} encoder needs at least {required_future} rolling reference frames, "
            f"but min(chunk-size, lookahead) is {available_future}"
        )
    # Also validates that the processed-sequence offset exists.  The packet is
    # tail-padded by the publisher, so even the last selected source frame is a
    # valid finite rollout start.
    _selected_reference_frame(args)


def _write_manifest(
    run_dir: Path,
    args: argparse.Namespace,
    model: ModelFiles,
    report: dict[str, Any],
    commands: dict[str, Sequence[str]],
) -> None:
    sequence, offset = _selected_reference_frame(args)
    recorded_window = args.regular_future_window == "recorded"
    if model.name in ("regular", "sonic_v1_1"):
        encoder_lags = (
            report["recorded_slot_lag_inference"][
                "future_slot_policy_lags"
            ]
            if model.name == "regular" and recorded_window
            else list(range(0, 50, 5))
        )
    else:
        encoder_lags = list(range(10))
    payload = {
        "created_at": datetime.now().astimezone().isoformat(),
        "checkpoint": model.name,
        "recording": str(_absolute(Path(args.recording))),
        "protocol": {"version": 1, "external_token": False},
        "experiment": {
            "kind": "qpos_track_in_recorded_task_scene",
            "body_reference": "recorded qpos/qvel/pelvis quaternion at 50 Hz",
            "regular_future_window": args.regular_future_window,
            "encoder_source_policy_lags": encoder_lags,
            "reference_motion_columns_used": recorded_window,
            "reference_motion_values_streamed": False,
            "reference_motion_columns_used_for_lag_inference": (
                recorded_window
            ),
            "recorded_slot_lag_inference": (
                report.get("recorded_slot_lag_inference")
                if recorded_window
                else None
            ),
            "root_position_streamed": False,
            "root_assist": args.root_assist,
            "root_position_note": (
                "Protocol v1 does not carry body_pos. When root_assist is not "
                "'none', the copied simulator applies an oracle hard alignment "
                "from the original recording after each physics tick; otherwise "
                "root_pos is audit-only. The active G1 mode-0 encoder inputs for "
                "the selected checkpoint do not require root-z."
            ),
            "task_object_state": (
                "initialized from the selected source row, then physics-driven"
            ),
            "hand_targets": (
                "the same recorded external targets are sent to both runs; "
                "they bypass the body encoder/decoder and use latest-value semantics"
            ),
        },
        "initialization": {
            "processed_sequence_offset": offset,
            "actual_start_policy_seq": int(sequence.policy_seq[offset]),
            "actual_start_source_row_index": int(
                sequence.source_row_index[offset]
            ),
            "drop_truncated_edges": bool(args.drop_truncated_edges),
            "gate_file": str(run_dir / "control_gate.status"),
            "publisher_gate_file": str(run_dir / "publisher.status"),
            "history_mode": "cpp_state_logger_zero_fill_until_10_control_frames_exist",
            "warmup_exclusion_s": 0.2,
            "first_control_tick_heading": "cpp_initial_alignment",
            "heading_correction_enabled": args.heading_correction,
            "heading_correction_tick": 1 if args.heading_correction else None,
            "sim_control_dt_s": args.control_dt,
            "source_row_dt_s": args.source_dt,
            "reference_rate_hz": args.reference_rate,
            "root_assist": args.root_assist,
        },
        "models": {
            "encoder": str(model.encoder),
            "encoder_sha256": _sha256(model.encoder),
            "decoder": str(model.decoder),
            "decoder_sha256": _sha256(model.decoder),
            "obs_config": str(model.obs_config),
            "obs_config_sha256": _sha256(model.obs_config),
        },
        "commands": {name: list(command) for name, command in commands.items()},
        "preflight": report,
    }
    (run_dir / "launch_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _print_command(label: str, command: Sequence[str]) -> None:
    print(f"[{label}] {shlex.join(command)}", flush=True)


def _signal_process_group(process: subprocess.Popen[str], sig: signal.Signals) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def _stop_processes(processes: Iterable[subprocess.Popen[str]]) -> None:
    active = [process for process in processes if process.poll() is None]
    for process in active:
        _signal_process_group(process, signal.SIGINT)
    deadline = time.monotonic() + 5.0
    while active and time.monotonic() < deadline:
        active = [process for process in active if process.poll() is None]
        time.sleep(0.1)
    for process in active:
        _signal_process_group(process, signal.SIGTERM)
    deadline = time.monotonic() + 3.0
    while active and time.monotonic() < deadline:
        active = [process for process in active if process.poll() is None]
        time.sleep(0.1)
    for process in active:
        _signal_process_group(process, signal.SIGKILL)


def _matches_deploy_ready_line(line: str) -> bool:
    """True only after the complete G1Deploy object, including ZMQ SUB, exists."""

    lower = line.lower()
    return any(message in lower for message in DEPLOY_READY_MESSAGES)


def _pump_output(
    label: str,
    process: subprocess.Popen[str],
    logfile: Path,
    fatal_event: threading.Event,
    ready_event: threading.Event | None = None,
    stream_ready_event: threading.Event | None = None,
) -> None:
    assert process.stdout is not None
    with logfile.open("w", encoding="utf-8", buffering=1) as stream:
        for line in process.stdout:
            stream.write(line)
            stripped = line.lstrip()
            noisy_deploy_line = label == "deploy" and (
                stripped.startswith("Frame[")
                or any(stripped.startswith(prefix) for prefix in DEPLOY_CONSOLE_NOISE_PREFIXES)
            )
            if not noisy_deploy_line:
                print(f"[{label}] {line}", end="", flush=True)
            lower = line.lower()
            if label == "deploy" and any(message in lower for message in FORBIDDEN_DEPLOY_MESSAGES):
                print(
                    "[safety] External token/protocol-v4 input detected; stopping because "
                    "it bypasses the local encoder.",
                    file=sys.stderr,
                    flush=True,
                )
                fatal_event.set()
            if ready_event and _matches_deploy_ready_line(line):
                ready_event.set()
            if stream_ready_event and any(
                message in lower for message in DEPLOY_STREAM_READY_MESSAGES
            ):
                stream_ready_event.set()


def _spawn(
    label: str,
    command: Sequence[str],
    run_dir: Path,
    fatal: threading.Event,
    ready: threading.Event | None = None,
    stream_ready: threading.Event | None = None,
    keep_deploy_stdin_pipe: bool = True,
) -> tuple[subprocess.Popen[str], threading.Thread]:
    # g1_deploy_onnx_ref treats EOF on stdin as an instruction to shut down.
    # A launcher started from a non-interactive shell often inherits an already
    # closed stdin, so give deploy an owned pipe and deliberately keep its write
    # end open for the lifetime of the process.  Signals are still used for the
    # coordinated shutdown below.
    child_stdin = (
        subprocess.PIPE if label == "deploy" and keep_deploy_stdin_pipe else None
    )
    process = subprocess.Popen(
        list(command),
        cwd=REPO_ROOT,
        stdin=child_stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    thread = threading.Thread(
        target=_pump_output,
        args=(label, process, run_dir / f"{label}.log", fatal, ready, stream_ready),
        name=f"{label}-log",
        daemon=True,
    )
    thread.start()
    return process, thread


def _send_stream_start_key(
    deploy: subprocess.Popen[str],
    publisher: subprocess.Popen[str],
    simulator: subprocess.Popen[str],
    gate_file: Path,
    publisher_status_file: Path,
    fatal: threading.Event,
    stream_ready: threading.Event,
    timeout_s: float,
) -> None:
    """Mirror deploy.sh's ``]`` key once INIT and the CSV stream are ready.

    The current C++ ``ZMQManager`` consumes the network ``start`` flag only in
    planner mode.  After it switches to streamed-motion mode, start is handled
    by the nested pose interface's keyboard path instead.  Keeping this small
    compatibility action in the copied launcher avoids modifying the original
    deploy source while preserving exactly the normal operator action.
    """

    deadline = time.monotonic() + timeout_s
    last_value = "<missing>"
    publisher_value = "<missing>"
    while time.monotonic() < deadline:
        if fatal.is_set():
            raise PreflightError("Deploy safety monitor rejected the run before control start")
        for label, process in (
            ("simulator", simulator),
            ("deploy", deploy),
            ("publisher", publisher),
        ):
            if process.poll() is not None:
                raise PreflightError(
                    f"{label} exited before the deploy INIT gate (code {process.returncode})"
                )
        try:
            last_value = gate_file.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            last_value = "<missing>"
        try:
            publisher_value = publisher_status_file.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            publisher_value = "<missing>"
        if (
            last_value == "deploy_init_ready"
            and publisher_value == "ready_for_control"
            and stream_ready.is_set()
        ):
            if deploy.stdin is None:
                raise PreflightError("Deploy stdin pipe is unavailable for streamed start")
            deploy.stdin.write("]")
            deploy.stdin.flush()
            print(
                "[launcher] INIT and first streamed-motion packet ready; sent the normal ']' start key",
                flush=True,
            )
            running_deadline = time.monotonic() + min(10.0, timeout_s)
            while time.monotonic() < running_deadline:
                if fatal.is_set() or deploy.poll() is not None or simulator.poll() is not None:
                    break
                try:
                    running_value = gate_file.read_text(encoding="utf-8").strip()
                except FileNotFoundError:
                    running_value = "<missing>"
                if running_value == "running":
                    print("[launcher] simulator confirmed policy control is running", flush=True)
                    return
                time.sleep(0.02)
            raise PreflightError(
                "Sent the streamed-control key, but the simulator did not confirm a "
                "non-INIT policy command within 10 seconds"
            )
        time.sleep(0.02)
    raise PreflightError(
        f"Timed out after {timeout_s:g}s waiting for simulator/publisher readiness; "
        f"simulator={last_value!r}, publisher={publisher_value!r}"
    )


def _run_deploy_foreground(args: argparse.Namespace, model: ModelFiles, run_dir: Path) -> int:
    command = deploy_command(args, model, run_dir / "deploy_csv")
    _print_command("deploy", command)
    fatal = threading.Event()
    # In the standalone three-terminal workflow, inherit the terminal so the
    # operator can use deploy's normal '[' / ']' controls.  The unified `run`
    # workflow keeps an owned pipe and injects `]` after both readiness gates.
    process, thread = _spawn(
        "deploy", command, run_dir, fatal, keep_deploy_stdin_pipe=False
    )
    try:
        while process.poll() is None and not fatal.is_set():
            time.sleep(0.1)
        if fatal.is_set():
            _stop_processes([process])
            return 3
        return int(process.returncode or 0)
    except KeyboardInterrupt:
        _stop_processes([process])
        return 130
    finally:
        thread.join(timeout=2)


def _run_all(args: argparse.Namespace, model: ModelFiles, report: dict[str, Any], run_dir: Path) -> int:
    # The deploy logger flushes many small CSV streams every 5 ms.  Doing that
    # on the workspace volume can stall the simulator and let the 50 Hz
    # wall-clock reference run ahead.  Keep those transient files on local
    # scratch storage, then copy complete artifacts into change_ckpt_track/data
    # after every process has stopped.
    deploy_spool: tempfile.TemporaryDirectory[str] | None = None
    deploy_output_root = run_dir
    if not args.dry_run:
        deploy_spool = tempfile.TemporaryDirectory(
            prefix="change_ckpt_track_deploy_logs_"
        )
        deploy_output_root = Path(deploy_spool.name)
    commands = {
        "sim": simulator_command(args, run_dir),
        "deploy": deploy_command(args, model, deploy_output_root / "deploy_csv"),
        "publisher": publisher_command(args, run_dir),
    }
    _write_manifest(run_dir, args, model, report, commands)
    for label, command in commands.items():
        _print_command(label, command)
    if args.dry_run:
        print(f"Dry run only; manifest: {run_dir / 'launch_manifest.json'}")
        return 0

    fatal = threading.Event()
    deploy_ready = threading.Event()
    stream_ready = threading.Event()
    processes: dict[str, subprocess.Popen[str]] = {}
    threads: list[threading.Thread] = []
    try:
        processes["sim"], thread = _spawn("sim", commands["sim"], run_dir, fatal)
        threads.append(thread)
        time.sleep(0.5)
        if processes["sim"].poll() is not None:
            raise PreflightError(
                f"Task simulator exited during startup with code {processes['sim'].returncode}"
            )

        processes["deploy"], thread = _spawn(
            "deploy", commands["deploy"], run_dir, fatal, deploy_ready, stream_ready
        )
        threads.append(thread)

        # Avoid dropping the beginning of the reference while TensorRT engines load.
        deadline = time.monotonic() + args.deploy_ready_timeout
        while not deploy_ready.is_set():
            if fatal.is_set():
                raise PreflightError("Deploy safety monitor rejected external-token input")
            if processes["deploy"].poll() is not None:
                raise PreflightError(
                    f"Deploy exited during startup with code {processes['deploy'].returncode}"
                )
            if time.monotonic() >= deadline:
                raise PreflightError(
                    f"Deploy did not report encoder initialization within {args.deploy_ready_timeout:g}s"
                )
            time.sleep(0.1)

        processes["publisher"], thread = _spawn(
            "publisher", commands["publisher"], run_dir, fatal
        )
        threads.append(thread)

        _send_stream_start_key(
            processes["deploy"],
            processes["publisher"],
            processes["sim"],
            run_dir / "control_gate.status",
            run_dir / "publisher.status",
            fatal,
            stream_ready,
            args.control_start_timeout,
        )

        publisher_finished = False
        deploy_finished = False
        while True:
            if fatal.is_set():
                return 3
            deploy_code = processes["deploy"].poll()
            sim_code = processes["sim"].poll()
            publisher_code = processes["publisher"].poll()
            if publisher_code is not None:
                if publisher_code != 0:
                    print(f"Publisher exited with code {publisher_code}", file=sys.stderr)
                    return int(publisher_code)
                if not publisher_finished:
                    publisher_finished = True
                    print(
                        "[launcher] publisher reached the source end; waiting for the "
                        "simulator to finish/flush (or for the viewer to close)",
                        flush=True,
                    )
            if deploy_code is not None and not deploy_finished:
                if deploy_code != 0 or not publisher_finished:
                    print(f"Deploy exited with code {deploy_code}", file=sys.stderr)
                    return int(deploy_code or 1)
                # After the source ends the simulator stops publishing LowState
                # before its buffered 1579-column replay has necessarily reached
                # disk.  The deploy process then exits normally.  Keep waiting for
                # the simulator instead of interrupting ReplayCsvWriter.close().
                deploy_finished = True
                print(
                    "[launcher] deploy exited normally after the source end; "
                    "waiting for the simulator to finish/flush",
                    flush=True,
                )
            if sim_code is not None:
                return int(sim_code or 0)
            time.sleep(0.1)
    except KeyboardInterrupt:
        return 130
    finally:
        _stop_processes(processes.values())
        for thread in threads:
            thread.join(timeout=2)
        if deploy_spool is not None:
            source_logs = deploy_output_root / "deploy_csv"
            if source_logs.is_dir():
                shutil.copytree(source_logs, run_dir / "deploy_csv", dirs_exist_ok=True)
            source_target = deploy_output_root / "target_motion.csv"
            if source_target.is_file():
                shutil.copy2(source_target, run_dir / "target_motion.csv")
            deploy_spool.cleanup()


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--checkpoint",
        default="low_latency",
        choices=(
            "regular",
            "sonic_release",
            "low_latency",
            "low",
            "low-latency",
            "sonic_v1_1",
            "sonic-v1-1",
            "v1.1",
            "v1_1",
        ),
        help="Checkpoint family (default: low_latency)",
    )
    parser.add_argument(
        "--recording",
        default=str(DEFAULT_RECORDING),
        help="Recording directory or its data.csv",
    )
    parser.add_argument("--encoder", help="Override encoder ONNX path")
    parser.add_argument("--decoder", help="Override decoder ONNX path")
    parser.add_argument("--obs-config", help="Override observation YAML path")
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_DATA_ROOT),
        help="All generated run directories are placed here",
    )
    parser.add_argument("--deploy-host", default="localhost", help="ZMQ host used by C++ subscriber")
    parser.add_argument("--publish-host", default="*", help="ZMQ bind host used by CSV publisher")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--topic", default="pose")
    parser.add_argument(
        "--skip-port-check",
        action="store_true",
        help="Allow preflight while another process owns the ZMQ port",
    )
    parser.add_argument(
        "--skip-process-check",
        action="store_true",
        help="Allow unified run while another SONIC simulator/deploy process exists",
    )


def _add_launch_arguments(parser: argparse.ArgumentParser) -> None:
    bool_action = argparse.BooleanOptionalAction
    parser.add_argument("--save-csv", action=bool_action, default=True)
    parser.add_argument("--viewer", action=bool_action, default=True)
    parser.add_argument("--stop-at-source-end", action=bool_action, default=True)
    parser.add_argument("--asset-model-root")
    parser.add_argument("--control-dt", type=float, default=0.005)
    parser.add_argument("--source-dt", type=float, default=0.0025)
    parser.add_argument(
        "--root-assist",
        choices=("none", "xy", "xyz"),
        default="none",
        help=(
            "oracle alignment of simulator root to the original recording: "
            "none (default), xy (recommended diagnostic), or xyz"
        ),
    )
    parser.add_argument(
        "--physics-dt",
        type=float,
        default=0.001,
        help="MuJoCo integration step (1 ms sustains real time on this host)",
    )
    parser.add_argument("--viewer-dt", type=float, default=0.02)
    parser.add_argument("--fall-height", type=float, default=0.2)
    parser.add_argument("--reference-rate", type=float, default=50.0)
    parser.add_argument(
        "--regular-future-window",
        choices=("canonical", "recorded"),
        default="canonical",
        help=(
            "Regular-only encoder timing. canonical makes C++ gather qpos "
            "lags 0,5,...,45; recorded infers this recording's ten slot lags "
            "from reference_motion and rearranges qpos/qvel/orientation"
        ),
    )
    parser.add_argument(
        "--heading-correction",
        action=bool_action,
        default=False,
        help=(
            "Send an explicit zero heading increment at tick 1. Normally disabled: "
            "the qpos reference and simulator use the same recorded start pelvis."
        ),
    )
    parser.add_argument(
        "--drop-truncated-edges",
        action=bool_action,
        default=True,
        help=(
            "Drop visibly partial first/last policy_seq groups before applying "
            "--start-policy-offset"
        ),
    )
    parser.add_argument(
        "--start-policy-offset",
        type=int,
        default=0,
        help="Offset inside the processed qpos sequence (after optional edge trimming)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help=(
            "Frames sent per pose packet. The 100-frame default leaves startup "
            "margin beyond regular encoder's 46-frame future window."
        ),
    )
    parser.add_argument("--lookahead", type=int, default=100)
    parser.add_argument("--deploy-ready-timeout", type=float, default=180.0)
    parser.add_argument(
        "--control-start-timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for deploy INIT before injecting its normal ']' start key",
    )
    parser.add_argument(
        "--sim-arg", action="append", default=[], metavar="ARG", help="Extra simulator arg; repeat as needed"
    )
    parser.add_argument(
        "--publisher-arg", action="append", default=[], metavar="ARG", help="Extra publisher arg; repeat as needed"
    )
    parser.add_argument(
        "--deploy-arg", action="append", default=[], metavar="ARG", help="Extra C++ deploy arg; repeat as needed"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="Validate models, config, recording, and adapters")
    _add_common_arguments(preflight)
    _add_launch_arguments(preflight)
    preflight.add_argument(
        "--models-only", action="store_true", help="Only check ONNX/config, not simulator/publisher/runtime"
    )

    deploy = subparsers.add_parser("deploy", help="Run only the local-encoder C++ deployment")
    _add_common_arguments(deploy)
    _add_launch_arguments(deploy)

    run = subparsers.add_parser("run", help="Run simulator, C++ deployment, and CSV publisher together")
    _add_common_arguments(run)
    _add_launch_arguments(run)
    run.add_argument("--dry-run", action="store_true", help="Write manifest and print commands only")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        include_adapters = not getattr(args, "models_only", False)
        report = run_preflight(args, include_adapters=include_adapters)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print("Preflight passed.", flush=True)
        model = resolve_model_files(args)
        if args.command == "preflight":
            if not args.models_only:
                _validate_launch_settings(args, model, report)
            return 0

        _validate_launch_settings(args, model, report)
        if args.command == "run" and not args.dry_run and not args.skip_process_check:
            _check_runtime_conflicts()
        run_dir = _new_run_dir(args, model.name)
        if args.command == "deploy":
            commands = {"deploy": deploy_command(args, model, run_dir / "deploy_csv")}
            _write_manifest(run_dir, args, model, report, commands)
            print(f"Run data directory: {run_dir}")
            return _run_deploy_foreground(args, model, run_dir)

        print(f"Run data directory: {run_dir}")
        return _run_all(args, model, report, run_dir)
    except PreflightError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except subprocess.TimeoutExpired as exc:
        print(f"ERROR: Preflight command timed out: {exc.cmd}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
