#!/usr/bin/env python3
"""Compare two completed task rollouts using physical task-state indicators."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np


CHANGE_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = CHANGE_ROOT / "data"


def _quat_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    first = first / np.linalg.norm(first)
    second = second / np.linalg.norm(second)
    dot = float(np.clip(abs(np.dot(first, second)), 0.0, 1.0))
    return math.degrees(2.0 * math.acos(dot))


def _csv_extrema(path: Path, column_name: str) -> tuple[float, float, int]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        try:
            column = header.index(column_name)
        except ValueError as exc:
            raise ValueError(f"missing {column_name!r} in {path}") from exc
        minimum = math.inf
        maximum = -math.inf
        rows = 0
        for row in reader:
            value = float(row[column])
            minimum = min(minimum, value)
            maximum = max(maximum, value)
            rows += 1
    return minimum, maximum, rows


def _qpos_column_name(path: Path, qpos_index: int) -> str:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        header = next(csv.reader(stream))
    suffix = f"[qpos{qpos_index}]"
    matches = [name for name in header if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"expected one {suffix} column in {path}, got {matches}")
    return matches[0]


def _numeric_log(path: Path, value_prefix: str) -> dict[str, Any]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        columns = [index for index, name in enumerate(header) if name.startswith(value_prefix)]
        rows = 0
        finite = True
        for row in reader:
            rows += 1
            finite = finite and all(math.isfinite(float(row[index])) for index in columns)
    return {"rows": rows, "dimension": len(columns), "all_finite": finite}


def _protocol_checks(run_dir: Path) -> dict[str, Any]:
    deploy_text = (run_dir / "deploy.log").read_text(encoding="utf-8", errors="replace")
    publisher_text = (run_dir / "publisher.log").read_text(
        encoding="utf-8", errors="replace"
    )
    field_counts = [int(value) for value in re.findall(r"num_fields: (\d+)", deploy_text)]
    late_matches = re.findall(r"source complete; late_ticks=(\d+)", publisher_text)
    return {
        "protocol_v1_established": "Protocol version 1 established" in deploy_text,
        "external_token_forbidden_messages_absent": not any(
            phrase in deploy_text.lower()
            for phrase in (
                "protocol v4",
                "copied external token",
                "received external token",
                "using external token",
            )
        ),
        "initial_catch_up_resets": deploy_text.count("Catch-up: Reset"),
        "seven_field_packets": field_counts.count(7),
        "one_time_eight_field_heading_packets": field_counts.count(8),
        "publisher_late_ticks": int(late_matches[-1]) if late_matches else None,
    }


def summarize(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "launch_manifest.json").read_text(encoding="utf-8"))
    initial = np.asarray(metadata["initial_task_qpos"], dtype=np.float64)
    final = np.asarray(metadata["final_task_qpos"], dtype=np.float64)
    source = np.asarray(metadata["current_source_task_qpos"], dtype=np.float64)
    labels = metadata["task_qpos_labels"]
    if len(initial) < 10 or "drawer" not in labels[0] or not labels[3].endswith("[0]"):
        raise ValueError(
            f"{run_dir} does not have the expected drawer + free scanner task-state layout"
        )

    source_drawer_motion = float(source[0] - initial[0])
    drawer_progress = (
        float((final[0] - initial[0]) / source_drawer_motion)
        if abs(source_drawer_motion) > 1e-12
        else None
    )
    source_scanner_motion = float(np.linalg.norm(source[3:6] - initial[3:6]))
    scanner_motion = float(np.linalg.norm(final[3:6] - initial[3:6]))
    scanner_motion_fraction = (
        scanner_motion / source_scanner_motion if source_scanner_motion > 1e-12 else None
    )
    replay_csv = run_dir / "data.csv"
    drawer_min, drawer_max, replay_rows = _csv_extrema(
        replay_csv, _qpos_column_name(replay_csv, 50)
    )

    stable = bool(
        metadata["wall_clock_timing_valid"]
        and not metadata["fallen"]
        and not metadata["invalid_state"]
    )
    task_state_screen_pass = bool(
        stable
        and drawer_progress is not None
        and drawer_progress >= 0.8
        and scanner_motion_fraction is not None
        and scanner_motion_fraction >= 0.7
    )
    protocol = _protocol_checks(run_dir)
    protocol["formal_handshake_valid"] = bool(
        protocol["protocol_v1_established"]
        and protocol["external_token_forbidden_messages_absent"]
        and protocol["initial_catch_up_resets"] == 1
        and protocol["one_time_eight_field_heading_packets"] == 1
        and protocol["publisher_late_ticks"] == 0
    )

    return {
        "run_dir": str(run_dir),
        "checkpoint": manifest["checkpoint"],
        "model_sha256": manifest["models"],
        "runtime": {
            "stable": stable,
            "timing_valid": metadata["wall_clock_timing_valid"],
            "real_time_factor": metadata["real_time_factor"],
            "max_schedule_lag_ms": metadata["max_schedule_lag_s"] * 1000.0,
            "fallen": metadata["fallen"],
            "invalid_state": metadata["invalid_state"],
            "final_base_height_m": metadata["final_base_height_m"],
            "hand_state_publish": metadata.get("hand_state_publish"),
        },
        "task_state": {
            "lower_drawer": {
                "initial": float(initial[0]),
                "final": float(final[0]),
                "source_final": float(source[0]),
                "final_abs_error_to_source": float(abs(final[0] - source[0])),
                "source_normalized_progress": drawer_progress,
                "trajectory_min": drawer_min,
                "trajectory_max": drawer_max,
            },
            "scanner": {
                "initial_position": initial[3:6].tolist(),
                "final_position": final[3:6].tolist(),
                "source_final_position": source[3:6].tolist(),
                "motion_from_initial_m": scanner_motion,
                "source_motion_from_initial_m": source_scanner_motion,
                "source_normalized_motion": scanner_motion_fraction,
                "final_position_error_to_source_m": float(
                    np.linalg.norm(final[3:6] - source[3:6])
                ),
                "final_orientation_error_to_source_deg": _quat_error_deg(
                    final[6:10], source[6:10]
                ),
            },
            "replay_rows": replay_rows,
            "screen_rule": (
                "diagnostic only: timing valid, no fall/nonfinite, lower drawer progress "
                ">= 0.8, and scanner displacement >= 0.7 of source"
            ),
            "task_state_screen_pass": task_state_screen_pass,
            "visual_confirmation_still_required": True,
        },
        "policy_logs": {
            "token_state": _numeric_log(run_dir / "deploy_csv/token_state.csv", "token_"),
            "action": _numeric_log(run_dir / "deploy_csv/action.csv", "act_"),
            "alignment_note": (
                "token_state[i] corresponds to action[i+1]; action[0] is zero startup history"
            ),
        },
        "protocol": protocol,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("regular_run", type=Path)
    parser.add_argument("low_latency_run", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="optional JSON output (must be below change_ckpt/data)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    regular = summarize(args.regular_run)
    low_latency = summarize(args.low_latency_run)
    report = {
        "regular": regular,
        "low_latency": low_latency,
        "comparison": {
            "both_runtime_valid": bool(
                regular["runtime"]["stable"] and low_latency["runtime"]["stable"]
            ),
            "regular_task_state_screen_pass": regular["task_state"][
                "task_state_screen_pass"
            ],
            "low_latency_task_state_screen_pass": low_latency["task_state"][
                "task_state_screen_pass"
            ],
            "interpretation": (
                "The screen compares physical object motion, not semantic task success. "
                "Confirm the result in the live viewer or replay."
            ),
        },
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        try:
            output.relative_to(DEFAULT_DATA_ROOT.resolve())
        except ValueError as exc:
            raise ValueError(f"output must be below {DEFAULT_DATA_ROOT}") from exc
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        print(f"wrote {output}")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
