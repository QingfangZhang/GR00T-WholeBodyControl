#!/usr/bin/env python3
"""Launch one formal, deterministic controller-replacement experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


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


def _absolute(path: Path) -> Path:
    value = path.expanduser()
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


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
) -> Path:
    stem = recording.name if recording.is_dir() else recording.parent.name
    parts = [stem, _controller_suffix(controller), reference_mode]
    if root_assist == "xy":
        parts.append("root_assist_xy")
    return output_root / "_".join(parts)


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
        "--policy-offset",
        type=int,
        default=10,
        help=(
            "processed reference offset; default 10 guarantees nine source-history "
            "predecessors under the established takeover convention"
        ),
    )
    parser.add_argument("--policy-count", type=int)
    parser.add_argument("--root-assist", choices=("none", "xy"), default="none")
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
    from controller_replacement.controllers import SonicController, TeleopitController
    from controller_replacement.history import build_source_history_context
    from controller_replacement.references import load_reference
    from controller_replacement.runner import RolloutConfig, run_rollout

    recording = _absolute(args.recording)
    reference = load_reference(
        recording,
        mode=args.reference_mode,
        policy_offset=args.policy_offset,
        policy_count=args.policy_count,
    )
    if args.controller == "teleopit":
        if args.device == "cuda":
            raise ValueError(
                "the pinned Teleopit runtime supports --device cpu/auto; "
                "use cpu in the formal environment"
            )
        controller = TeleopitController(
            checkpoint=_absolute(args.teleopit_checkpoint),
            robot_xml=_absolute(args.teleopit_robot_xml),
            device="cpu",
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
    output_root = _absolute(args.output_root)
    output = (
        _absolute(args.output)
        if args.output is not None
        else default_output_directory(
            recording,
            controller=args.controller,
            reference_mode=args.reference_mode,
            root_assist=args.root_assist,
            output_root=output_root,
        )
    )
    print(
        json.dumps(
            {
                "recording": str(recording),
                "controller": args.controller,
                "reference_mode": args.reference_mode,
                "takeover_policy_seq": history.selected_policy_seq,
                "raw_policy_group_offset": history.raw_policy_group_offset,
                "timeline_start_row_index": history.timeline_start_row_index,
                "root_assist": args.root_assist,
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
