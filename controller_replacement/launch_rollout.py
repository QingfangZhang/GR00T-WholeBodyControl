#!/usr/bin/env python3
"""Launch one formal, deterministic controller-replacement experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "controller_replacement/data"
DEFAULT_TELEOPIT_CHECKPOINT = (
    REPO_ROOT / "Teleopit_rollout/assets/checkpoints/track_g1.onnx"
)
DEFAULT_TELEOPIT_ROBOT_XML = (
    REPO_ROOT
    / "Teleopit_rollout/assets/robot_assets/unitree_g1/g1_29dof.xml"
)
DEFAULT_HAND_TORQUE_PROFILE = "sonic_release"
DEFAULT_FALL_HEIGHT_M = 0.2
DEFAULT_RAW_POLICY_GROUP_OFFSET = 11
PROTOCOL_NAME = "protocol2"
BEHAVIOR_SOURCE_DIRECTORIES = (
    "controller_replacement",
    "change_ckpt",
    "change_ckpt_track",
    "Teleopit_rollout",
)


def _anchored(path: Path) -> Path:
    value = path.expanduser()
    return value if value.is_absolute() else REPO_ROOT / value


def _absolute(path: Path) -> Path:
    return _anchored(path).resolve()


def _controller_suffix(name: str) -> str:
    return {
        "regular": "regular",
        "low_latency": "low_latency",
        "sonic_v1_1": "sonic_v1_1",
        "teleopit": "teleopit_v0_5",
    }[name]


def default_output_directory(
    recording: Path,
    *,
    controller: str,
    reference_mode: str,
    root_assist: str,
    output_root: Path,
    hand_torque_profile: str = DEFAULT_HAND_TORQUE_PROFILE,
    fall_height_m: float = DEFAULT_FALL_HEIGHT_M,
    raw_policy_group_offset: int = DEFAULT_RAW_POLICY_GROUP_OFFSET,
    policy_count: int | None = None,
    device: str = "cpu",
    viewer: bool = False,
) -> Path:
    stem = recording.name if recording.is_dir() else recording.parent.name
    parts = [stem, _controller_suffix(controller), reference_mode, PROTOCOL_NAME]
    if root_assist == "xy":
        parts.append("root_assist_xy")
    if hand_torque_profile != DEFAULT_HAND_TORQUE_PROFILE:
        parts.append(f"hand_{hand_torque_profile}")
    if fall_height_m != DEFAULT_FALL_HEIGHT_M:
        rendered = format(float(fall_height_m), ".6g").replace(".", "p")
        parts.append(f"fall_z_{rendered}m")
    if raw_policy_group_offset != DEFAULT_RAW_POLICY_GROUP_OFFSET:
        parts.append(f"raw_offset_{int(raw_policy_group_offset)}")
    if policy_count is not None:
        parts.append(f"policy_count_{int(policy_count)}")
    if device != "cpu":
        parts.append(str(device))
    if viewer:
        parts.append("viewer")
    return output_root / "_".join(parts)


def _validate_automatic_output_reuse(
    output: Path,
    recording: Path,
    *,
    source_csv_sha256: str,
    model_sha256: dict[str, str],
    expected_condition: Mapping[str, Any],
) -> None:
    """Reject basename collisions instead of overwriting another recording."""

    if not output.exists():
        return
    manifest_path = output / "run_manifest.json"
    complete_path = output / "run_complete.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(
            "automatic output already exists without a run_manifest.json; "
            f"choose an explicit --output: {output}"
        )
    if complete_path.is_symlink() or not complete_path.is_file():
        raise ValueError(
            "automatic output already exists without a regular "
            "run_complete.json; choose an explicit --output: "
            f"{output}"
        )
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "automatic output has an unreadable manifest/completion "
            f"certificate; choose an explicit --output: {output}"
        ) from exc
    if not isinstance(manifest, dict) or not isinstance(complete, dict):
        raise ValueError(
            "automatic output manifest/completion certificate is malformed; "
            f"choose an explicit --output: {output}"
        )
    try:
        complete_protocol_revision = int(
            complete.get("protocol_revision", -1)
        )
    except (TypeError, ValueError):
        complete_protocol_revision = -1
    if complete.get("complete") is not True or complete_protocol_revision != 2:
        raise ValueError(
            "automatic output is not certified as a complete protocol-2 run; "
            f"choose an explicit --output: {output}"
        )
    certified_artifacts = complete.get("artifact_sha256", {})
    certified_manifest_sha256 = (
        certified_artifacts.get("run_manifest.json")
        if isinstance(certified_artifacts, dict)
        else None
    )
    actual_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if certified_manifest_sha256 != actual_manifest_sha256:
        raise ValueError(
            "automatic output run_manifest.json does not match its completion "
            f"certificate; choose an explicit --output: {output}"
        )
    expected_source = str(
        (recording if recording.is_dir() else recording.parent).resolve()
    )
    if manifest.get("source_recording") != expected_source:
        raise ValueError(
            "automatic output basename collides with a different recording; "
            f"choose an explicit --output: {output}"
        )
    try:
        manifest_protocol_revision = int(manifest.get("protocol_revision", -1))
    except (TypeError, ValueError):
        manifest_protocol_revision = -1
    if manifest_protocol_revision != 2:
        raise ValueError(
            "automatic output belongs to another protocol revision; choose an "
            f"explicit --output: {output}"
        )
    raw_provenance = manifest.get("provenance", {})
    raw_source_provenance = (
        raw_provenance.get("source_recording", {})
        if isinstance(raw_provenance, dict)
        else {}
    )
    recorded_source_sha256 = (
        raw_source_provenance.get("sha256_before")
        if isinstance(raw_source_provenance, dict)
        else None
    )
    if recorded_source_sha256 != source_csv_sha256:
        raise ValueError(
            "automatic output was produced from different source CSV bytes; "
            f"choose an explicit --output: {output}"
        )
    raw_models = manifest.get("models", {})
    if not isinstance(raw_models, dict):
        raise ValueError(
            "automatic output has malformed controller model provenance; "
            f"choose an explicit --output: {output}"
        )
    recorded_models = {
        str(name): str(details.get("sha256", ""))
        for name, details in raw_models.items()
        if isinstance(details, dict)
    }
    if recorded_models != model_sha256:
        raise ValueError(
            "automatic output was produced with different controller model "
            f"bytes; choose an explicit --output: {output}"
        )

    def nested(*keys: str) -> Any:
        value: Any = manifest
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                return None
            value = value[key]
        return value

    requested_reference_frames = nested(
        "extra", "reference_horizon", "requested_reference_frames"
    )
    if requested_reference_frames is None:
        # Compatibility with protocol-2 outputs produced before the horizon
        # contract was made explicit.  Those manifests stored only the
        # originally selected reference length.
        requested_reference_frames = nested(
            "extra", "reference", "selected_frames"
        )
    recorded_condition = {
        "controller_name": manifest.get("controller_name"),
        "controller_family": manifest.get("controller_family"),
        "reference_mode": manifest.get("reference_mode"),
        "root_assist": manifest.get("root_assist"),
        "hand_torque_profile": nested("extra", "hand_control", "profile"),
        "fall_height_m": nested(
            "extra", "fall_detection", "threshold_m"
        ),
        "raw_policy_group_offset": nested(
            "extra", "source_history", "raw_policy_group_offset"
        ),
        "selected_policy_count": requested_reference_frames,
        "requested_device": nested(
            "extra", "controller", "inference_runtime", "requested_device"
        ),
        "viewer": nested("extra", "viewer", "enabled"),
    }
    normalized_expected = dict(expected_condition)
    differing = sorted(
        key
        for key in set(recorded_condition) | set(normalized_expected)
        if recorded_condition.get(key) != normalized_expected.get(key)
    )
    if differing:
        raise ValueError(
            "automatic output was produced under different experiment "
            f"conditions ({', '.join(differing)}); choose an explicit "
            f"--output: {output}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path, help="recording folder or data.csv")
    parser.add_argument(
        "--controller",
        required=True,
        choices=("regular", "low_latency", "sonic_v1_1", "teleopit"),
    )
    parser.add_argument(
        "--reference-mode",
        default="reference_motion",
        choices=("reference_motion", "executed_qpos"),
        help="main intent input; executed_qpos is the auxiliary ablation",
    )
    parser.add_argument(
        "--raw-policy-group-offset",
        type=int,
        default=DEFAULT_RAW_POLICY_GROUP_OFFSET,
        help=(
            "zero-based raw CSV policy_seq-group index; default 11 always "
            "selects the 12th raw group, regardless of edge trimming"
        ),
    )
    parser.add_argument("--policy-count", type=int)
    parser.add_argument("--root-assist", choices=("none", "xy"), default="none")
    parser.add_argument(
        "--hand-torque-profile",
        choices=("sonic_release", "staged_xml"),
        default=DEFAULT_HAND_TORQUE_PROFILE,
        help=(
            "formal default uses released SONIC Dex3 limits [2.45,0.7x6] Nm; "
            "staged_xml is a sensitivity condition"
        ),
    )
    parser.add_argument(
        "--fall-height-m",
        type=float,
        default=DEFAULT_FALL_HEIGHT_M,
        help="latching root qpos[2] fall threshold (formal default: 0.2 m)",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--asset-model-root", type=Path)
    parser.add_argument(
        "--teleopit-checkpoint", type=Path, default=DEFAULT_TELEOPIT_CHECKPOINT
    )
    parser.add_argument(
        "--teleopit-robot-xml", type=Path, default=DEFAULT_TELEOPIT_ROBOT_XML
    )
    return parser


def run(args: argparse.Namespace) -> int:
    # Delayed imports make ``--help`` work even in a Python environment that
    # lacks MuJoCo/ONNX Runtime and permit a precise environment diagnostic.
    from controller_replacement.provenance import source_tree_hashes

    behavior_source_sha256_at_import = {
        relative: source_tree_hashes(REPO_ROOT / relative)
        for relative in BEHAVIOR_SOURCE_DIRECTORIES
        if (REPO_ROOT / relative).is_dir()
    }
    from controller_replacement.controllers import SonicController, TeleopitController
    from controller_replacement.history import (
        build_source_history_context,
        load_reference_from_raw_policy_group,
    )
    from controller_replacement.runner import RolloutConfig, run_rollout
    from controller_replacement.output import sha256_file
    from change_ckpt_track.task_sim_io import resolve_recording

    recording = _absolute(args.recording)
    recording_dir, source_csv = resolve_recording(recording)
    source_csv_sha256_at_load = sha256_file(source_csv)
    if args.output is None and args.asset_model_root is not None:
        raise ValueError(
            "a custom --asset-model-root requires an explicit --output so it "
            "cannot overwrite the default experiment condition"
        )
    teleopit_checkpoint = _absolute(args.teleopit_checkpoint)
    teleopit_robot_xml = _absolute(args.teleopit_robot_xml)
    if args.output is None and args.controller == "teleopit" and (
        teleopit_checkpoint != DEFAULT_TELEOPIT_CHECKPOINT.resolve()
        or teleopit_robot_xml != DEFAULT_TELEOPIT_ROBOT_XML.resolve()
    ):
        raise ValueError(
            "custom Teleopit checkpoint/XML paths require an explicit --output"
        )
    reference = load_reference_from_raw_policy_group(
        recording,
        mode=args.reference_mode,
        raw_policy_group_offset=args.raw_policy_group_offset,
        policy_count=args.policy_count,
    )
    if args.controller == "teleopit":
        controller = TeleopitController(
            checkpoint=teleopit_checkpoint,
            robot_xml=teleopit_robot_xml,
            device=args.device,
            require_source_history=True,
        )
        history = build_source_history_context(
            recording,
            selected_reference=reference,
            teleopit_observation_builder=controller.observation_builder,
        )
    else:
        controller = SonicController(
            args.controller,
            device=args.device,
            require_source_history=True,
        )
        history = build_source_history_context(
            recording, selected_reference=reference
        )
    source_csv_sha256_after_load = sha256_file(source_csv)
    if source_csv_sha256_after_load != source_csv_sha256_at_load:
        raise ValueError(
            "source data.csv changed while reference/history inputs were loaded"
        )
    behavior_source_sha256_after_load = {
        relative: source_tree_hashes(REPO_ROOT / relative)
        for relative in BEHAVIOR_SOURCE_DIRECTORIES
        if (REPO_ROOT / relative).is_dir()
    }
    if behavior_source_sha256_after_load != behavior_source_sha256_at_import:
        raise ValueError(
            "behavior-bearing source files changed while runtime/controller "
            "inputs were loaded"
        )
    output_root = _absolute(args.output_root)
    output = (
        _anchored(args.output)
        if args.output is not None
        else default_output_directory(
            recording,
            controller=args.controller,
            reference_mode=args.reference_mode,
            root_assist=args.root_assist,
            output_root=output_root,
            hand_torque_profile=args.hand_torque_profile,
            fall_height_m=args.fall_height_m,
            raw_policy_group_offset=args.raw_policy_group_offset,
            policy_count=args.policy_count,
            device=args.device,
            viewer=args.viewer,
        )
    )
    automatic_output_validator = None
    if args.output is None:
        expected_condition = {
            "controller_name": controller.name,
            "controller_family": controller.controller_family,
            "reference_mode": reference.mode.value,
            "root_assist": args.root_assist,
            "hand_torque_profile": args.hand_torque_profile,
            "fall_height_m": float(args.fall_height_m),
            "raw_policy_group_offset": int(history.raw_policy_group_offset),
            "selected_policy_count": int(reference.num_frames),
            "requested_device": controller.requested_device,
            "viewer": bool(args.viewer),
        }

        def automatic_output_validator(candidate: Path) -> None:
            _validate_automatic_output_reuse(
                candidate,
                recording_dir,
                source_csv_sha256=source_csv_sha256_at_load,
                model_sha256=dict(controller.loaded_model_sha256),
                expected_condition=expected_condition,
            )

        # Give fast feedback before staging.  The same validator is passed to
        # the runner and repeated after its per-output lock is held, closing
        # the stale-validation race with another process.
        automatic_output_validator(output)
    print(
        json.dumps(
            {
                "recording": str(recording),
                "controller": args.controller,
                "reference_mode": args.reference_mode,
                "requested_raw_policy_group_offset": args.raw_policy_group_offset,
                "takeover_policy_seq": history.selected_policy_seq,
                "raw_policy_group_offset": history.raw_policy_group_offset,
                "timeline_start_row_index": history.timeline_start_row_index,
                "root_assist": args.root_assist,
                "hand_torque_profile": args.hand_torque_profile,
                "fall_height_m": args.fall_height_m,
                "source_csv_sha256": source_csv_sha256_at_load,
                "output": str(output),
                "timing": {
                    "physics_hz": 2000,
                    "pd_hz": 200,
                    "policy_hz": 50,
                    "csv_logging_hz": 400,
                    "wall_clock_controls_simulation": False,
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    result = run_rollout(
        RolloutConfig(
            recording=recording,
            output_directory=output,
            reference=reference,
            root_assist=args.root_assist,
            viewer=args.viewer,
            asset_model_root=(
                None
                if args.asset_model_root is None
                else _absolute(args.asset_model_root)
            ),
            hand_torque_profile=args.hand_torque_profile,
            fall_height_m=args.fall_height_m,
            expected_source_csv_sha256=source_csv_sha256_at_load,
            expected_behavior_source_hashes=behavior_source_sha256_at_import,
            automatic_output=args.output is None,
            existing_output_validator=automatic_output_validator,
        ),
        controller=controller,
        history_context=history,
    )
    print(
        json.dumps(
            {
                "complete": True,
                "output": str(result.output_directory),
                "policy_inferences": result.policy_inferences,
                "logged_rows": result.logged_rows,
                "wall_seconds": result.wall_seconds,
                "simulated_seconds": result.simulated_seconds,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(build_parser().parse_args(argv))
    except KeyboardInterrupt:
        print("rollout interrupted; incomplete output was not published", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - one concise CLI diagnostic
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
