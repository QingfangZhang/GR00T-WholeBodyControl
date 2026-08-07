#!/usr/bin/env python3
"""Summarize completed qpos-track rollouts without modifying them.

For every completed run this tool reports three distinct measurements:

* 29-DoF body tracking against the constructed 50 Hz ``target_motion.csv``;
* floating-base tracking against the original recording's qpos timeline;
* terminal task-object qpos from ``run_metadata.json``.

Task-object qpos are physical indicators, not an automatic task-success label.
Contact, grasp, placement, and stability success must be confirmed with a
task-specific criterion or by replaying the run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

import numpy as np

if __package__:
    # Package-style invocation: ``python -m change_ckpt_track...``.
    from .compare_qpos_track import ComparisonError, analyze_run
else:
    # Direct script invocation used by the README and shell examples.
    from compare_qpos_track import ComparisonError, analyze_run


TRACK_ROOT = Path(__file__).resolve().parent
DEFAULT_SCAN_ROOT = TRACK_ROOT / "data" / "sonic_v1_1_comparison"


class SummaryError(RuntimeError):
    """Raised when a completed run cannot be summarized safely."""


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SummaryError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SummaryError(f"expected a JSON object in {path}")
    return value


def _qpos_columns(header: Sequence[str], count: int = 7) -> list[int]:
    pattern = re.compile(r"\[qpos(\d+)\]$")
    indexed: dict[int, int] = {}
    for column, name in enumerate(header):
        match = pattern.search(name)
        if match:
            indexed[int(match.group(1))] = column
    missing = [index for index in range(count) if index not in indexed]
    if missing:
        raise SummaryError(f"CSV is missing floating-base qpos columns {missing}")
    return [indexed[index] for index in range(count)]


def _read_root_csv(
    path: Path, *, read_sample_index: bool
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray | None]:
    try:
        stream = path.open("r", newline="", encoding="utf-8-sig")
    except OSError as exc:
        raise SummaryError(f"cannot open CSV {path}: {exc}") from exc
    with stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise SummaryError(f"empty CSV: {path}") from exc
        qpos_columns = _qpos_columns(header)
        sample_column = (
            header.index("sample_index")
            if read_sample_index and "sample_index" in header
            else None
        )
        policy_seq_column = (
            header.index("policy_seq") if "policy_seq" in header else None
        )
        roots: list[list[float]] = []
        samples: list[int] = []
        policy_sequences: list[int] = []
        extra_columns = [
            column
            for column in (sample_column, policy_seq_column)
            if column is not None
        ]
        minimum_width = max(qpos_columns + extra_columns)
        for row_number, row in enumerate(reader, start=0):
            if len(row) <= minimum_width:
                raise SummaryError(
                    f"short row {row_number} in {path}: {len(row)} columns"
                )
            try:
                roots.append([float(row[column]) for column in qpos_columns])
                if sample_column is not None:
                    value = float(row[sample_column])
                    sample = int(value)
                    if value != sample:
                        raise ValueError("sample_index is not integral")
                    samples.append(sample)
                if policy_seq_column is not None:
                    value = float(row[policy_seq_column])
                    policy_sequence = int(value)
                    if value != policy_sequence:
                        raise ValueError("policy_seq is not integral")
                    policy_sequences.append(policy_sequence)
            except ValueError as exc:
                raise SummaryError(
                    f"invalid numeric value in data row {row_number} of {path}: {exc}"
                ) from exc
    if not roots:
        raise SummaryError(f"CSV has no data rows: {path}")
    root_array = np.asarray(roots, dtype=np.float64)
    sample_array = (
        np.asarray(samples, dtype=np.int64)
        if sample_column is not None
        else None
    )
    policy_seq_array = (
        np.asarray(policy_sequences, dtype=np.int64)
        if policy_seq_column is not None
        else None
    )
    return sample_array, root_array, policy_seq_array


def _rms(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return None
    return float(np.sqrt(np.mean(np.square(finite))))


def _maximum(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if finite.size else None


def _quat_errors_deg(actual: np.ndarray, source: np.ndarray) -> np.ndarray:
    actual_norm = np.linalg.norm(actual, axis=1)
    source_norm = np.linalg.norm(source, axis=1)
    valid = (
        np.isfinite(actual).all(axis=1)
        & np.isfinite(source).all(axis=1)
        & (actual_norm > 1e-12)
        & (source_norm > 1e-12)
    )
    result = np.full(len(actual), np.nan, dtype=np.float64)
    dots = np.abs(
        np.sum(
            actual[valid] / actual_norm[valid, None]
            * source[valid] / source_norm[valid, None],
            axis=1,
        )
    )
    result[valid] = np.degrees(2.0 * np.arccos(np.clip(dots, 0.0, 1.0)))
    return result


def _yaw_wxyz(quaternion: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quaternion, axis=1)
    valid = np.isfinite(quaternion).all(axis=1) & (norm > 1e-12)
    unit = np.full_like(quaternion, np.nan)
    unit[valid] = quaternion[valid] / norm[valid, None]
    w, x, y, z = unit.T
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrapped_angle_error_deg(actual: np.ndarray, source: np.ndarray) -> np.ndarray:
    error = _yaw_wxyz(actual) - _yaw_wxyz(source)
    return np.degrees(np.arctan2(np.sin(error), np.cos(error)))


def _vector_metrics(error: np.ndarray) -> dict[str, Any]:
    magnitude = np.linalg.norm(error, axis=1)
    return {
        "rms_magnitude": _rms(magnitude),
        "mean_magnitude": (
            float(np.mean(magnitude[np.isfinite(magnitude)]))
            if np.isfinite(magnitude).any()
            else None
        ),
        "max_magnitude": _maximum(magnitude),
        "final_error": error[-1].tolist() if np.isfinite(error[-1]).all() else None,
        "final_magnitude": (
            float(magnitude[-1]) if math.isfinite(float(magnitude[-1])) else None
        ),
    }


def root_tracking(
    run_dir: Path, metadata: dict[str, Any], warmup_s: float
) -> dict[str, Any]:
    """Compare rollout root qpos to exact source rows used by CsvTimeline.

    ``run_task_sim_loop.py`` writes sample 0 at ``initial_row_index`` and then
    advances ``source_rows_per_control`` source rows before every later sample.
    It holds the last source row after EOF.  Reconstructing those row IDs is
    therefore deterministic from run metadata and ``sample_index``.
    """

    source_value = metadata.get("source_csv")
    if not isinstance(source_value, str):
        raise SummaryError(f"{run_dir}/run_metadata.json has no source_csv")
    source_csv = Path(source_value).expanduser()
    if not source_csv.is_absolute():
        source_csv = (run_dir / source_csv).resolve()
    samples, actual_root, rollout_policy_seq = _read_root_csv(
        run_dir / "data.csv", read_sample_index=True
    )
    if samples is None:
        samples = np.arange(len(actual_root), dtype=np.int64)
        sample_method = "data-row ordinal fallback"
    else:
        sample_method = "data.csv sample_index"
    if len(samples) != len(actual_root):
        raise SummaryError(f"sample/root row count mismatch in {run_dir}/data.csv")
    if np.any(samples < 0) or np.any(np.diff(samples) < 0):
        raise SummaryError(f"sample_index is negative or non-monotonic in {run_dir}")

    _, source_root, source_policy_seq = _read_root_csv(
        source_csv, read_sample_index=False
    )
    start = metadata.get("initial_row_index")
    stride = metadata.get("source_rows_per_control")
    control_dt = metadata.get("control_dt")
    if not isinstance(start, int) or start < 0:
        raise SummaryError(f"invalid initial_row_index in {run_dir}/run_metadata.json")
    if not isinstance(stride, int) or stride < 1:
        raise SummaryError(
            f"invalid source_rows_per_control in {run_dir}/run_metadata.json"
        )
    if not isinstance(control_dt, (int, float)) or control_dt <= 0:
        raise SummaryError(f"invalid control_dt in {run_dir}/run_metadata.json")
    if start >= len(source_root):
        raise SummaryError(
            f"initial source row {start} exceeds {len(source_root)} rows in {source_csv}"
        )
    source_rows = np.minimum(start + samples * stride, len(source_root) - 1)
    selected_source = source_root[source_rows]
    policy_seq_match: bool | None = None
    if rollout_policy_seq is not None and source_policy_seq is not None:
        policy_seq_match = bool(
            np.array_equal(rollout_policy_seq, source_policy_seq[source_rows])
        )
        if not policy_seq_match:
            mismatch = int(
                np.flatnonzero(
                    rollout_policy_seq != source_policy_seq[source_rows]
                )[0]
            )
            raise SummaryError(
                "reconstructed source-row mapping disagrees with the copied "
                f"policy_seq at rollout row {mismatch} in {run_dir}"
            )
    # Exclude the deliberate post-rollout hold: it is useful for contact to
    # settle and therefore belongs in terminal task qpos, but it is not another
    # segment of the source trajectory.  The final unique source row is kept.
    final_active_sample = int(
        math.ceil((len(source_root) - 1 - start) / stride)
    )
    evaluation_mask = (
        (samples.astype(np.float64) * float(control_dt) + 1e-12 >= warmup_s)
        & (samples <= final_active_sample)
    )
    if not evaluation_mask.any():
        raise SummaryError(
            f"no root samples remain after {warmup_s:.3f}s warmup in {run_dir}"
        )
    actual = actual_root[evaluation_mask]
    source = selected_source[evaluation_mask]
    selected_rows = source_rows[evaluation_mask]
    position_error = actual[:, :3] - source[:, :3]
    xy_error = position_error[:, :2]
    orientation_error = _quat_errors_deg(actual[:, 3:7], source[:, 3:7])
    yaw_error = _wrapped_angle_error_deg(actual[:, 3:7], source[:, 3:7])
    absolute_yaw_error = np.abs(yaw_error)
    assist = metadata.get("root_assist", {})
    assist_mode = assist.get("mode", "unknown") if isinstance(assist, dict) else "unknown"
    intervention = (
        {
            key: assist.get(key)
            for key in (
                "ticks",
                "pre_alignment_error_rms_m",
                "pre_alignment_error_max_m",
                "post_alignment_error_rms_m",
                "post_alignment_error_max_m",
            )
            if key in assist
        }
        if isinstance(assist, dict)
        else {}
    )
    return {
        "reference": "original recording floating-base qpos at CsvTimeline source rows",
        "source_csv": str(source_csv.resolve()),
        "source_row_mapping": {
            "method": (
                f"{sample_method}: min(initial_row_index + sample_index * "
                "source_rows_per_control, final_source_row)"
            ),
            "initial_row_index": start,
            "source_rows_per_control": stride,
            "first_evaluated_source_row": int(selected_rows[0]),
            "last_evaluated_source_row": int(selected_rows[-1]),
            "copied_policy_seq_matches_all_rows": policy_seq_match,
        },
        "warmup_exclusion_s": warmup_s,
        "post_source_hold_excluded": True,
        "final_active_sample_index": final_active_sample,
        "samples": int(len(actual)),
        "all_finite": bool(
            np.isfinite(actual).all() and np.isfinite(source).all()
        ),
        "root_assist_mode": assist_mode,
        "root_assist_intervention": intervention,
        "root_assist_caveat": (
            "assisted coordinates are hard-overwritten from the source after every "
            "control tick; their near-zero errors are imposed, not policy performance"
            if assist_mode != "none"
            else None
        ),
        "xy_position_m": _vector_metrics(xy_error),
        "xyz_position_m": _vector_metrics(position_error),
        "z_position_m": {
            "rmse": _rms(position_error[:, 2]),
            "max_abs": _maximum(np.abs(position_error[:, 2])),
            "final_error": (
                float(position_error[-1, 2])
                if math.isfinite(float(position_error[-1, 2]))
                else None
            ),
        },
        "orientation_geodesic_deg": {
            "rms": _rms(orientation_error),
            "max": _maximum(orientation_error),
            "final": (
                float(orientation_error[-1])
                if math.isfinite(float(orientation_error[-1]))
                else None
            ),
        },
        "yaw_abs_error_deg": {
            "rms": _rms(absolute_yaw_error),
            "max": _maximum(absolute_yaw_error),
            "final": (
                float(abs(yaw_error[-1]))
                if math.isfinite(float(yaw_error[-1]))
                else None
            ),
        },
    }


def _root_assist_mode(metadata: dict[str, Any]) -> str:
    value = metadata.get("root_assist")
    if isinstance(value, dict) and isinstance(value.get("mode"), str):
        return value["mode"]
    return "unknown"


def summarize_run(run_dir: Path, warmup_s: float | None) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.is_file():
        raise SummaryError(f"completed-run metadata is missing: {metadata_path}")
    metadata = _load_object(metadata_path)
    try:
        analysis = analyze_run(run_dir, warmup_s)
    except ComparisonError as exc:
        raise SummaryError(f"cannot analyze body trajectory in {run_dir}: {exc}") from exc
    body = analysis.summary
    effective_warmup = float(body["playback_alignment"]["warmup_exclusion_s"])
    root = root_tracking(run_dir, metadata, effective_warmup)
    identity = body["identity"]
    recording_value = metadata.get("source_recording")
    recording = (
        Path(recording_value).name
        if isinstance(recording_value, str)
        else "unknown"
    )
    return {
        "run_dir": str(run_dir),
        "recording": recording,
        "checkpoint": identity.get("checkpoint", run_dir.name),
        "root_assist": _root_assist_mode(metadata),
        "runtime": body["runtime"],
        "runtime_validity": body["runtime_validity"],
        "body_target_tracking": {
            "reference": (
                "constructed 50 Hz qpos-derived target_motion.csv, aligned by "
                "prepared_reference frame IDs"
            ),
            "error_definition": body["tracking"]["error_definition"],
            "warmup_exclusion_s": effective_warmup,
            "evaluation_rows": body["playback_alignment"]["evaluation_rows"],
            "overall": body["tracking"]["metrics"]["overall"],
            "groups": body["tracking"]["metrics"]["groups"],
        },
        "root_source_tracking": root,
        "task_object_terminal_state": body["task_object_terminal_state"],
        "warnings": body.get("warnings", []),
    }


def discover_runs(scan_roots: Iterable[Path], explicit: Iterable[Path]) -> list[Path]:
    candidates = {path.expanduser().resolve() for path in explicit}
    for root in scan_roots:
        root = root.expanduser().resolve()
        if not root.is_dir():
            raise SummaryError(f"scan root does not exist: {root}")
        candidates.update(path.parent for path in root.rglob("run_metadata.json"))
    return sorted(candidates, key=lambda path: str(path))


def _number(value: Any, digits: int = 5) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "NA"


def print_table(summaries: Sequence[dict[str, Any]]) -> None:
    columns = (
        "recording",
        "checkpoint",
        "assist",
        "valid",
        "fallen",
        "body_rmse_rad",
        "root_xy_rms_m",
        "root_xyz_rms_m",
        "yaw_rms_deg",
        "task_final_l2_source",
    )
    print("\t".join(columns))
    for summary in summaries:
        body = summary["body_target_tracking"]["overall"]
        root = summary["root_source_tracking"]
        task = summary["task_object_terminal_state"]
        runtime = summary["runtime"]
        print(
            "\t".join(
                (
                    str(summary["recording"]),
                    str(summary["checkpoint"]),
                    str(summary["root_assist"]),
                    str(bool(summary["runtime_validity"]["valid"])),
                    str(runtime.get("fallen", "NA")),
                    _number(body.get("rmse_rad")),
                    _number(root["xy_position_m"].get("rms_magnitude")),
                    _number(root["xyz_position_m"].get("rms_magnitude")),
                    _number(root["yaw_abs_error_deg"].get("rms"), 3),
                    _number(task.get("final_l2_error_to_recorded_source")),
                )
            )
        )

    print("\nTerminal task qpos (descriptive physical state; not an automatic success label):")
    for summary in summaries:
        task = summary["task_object_terminal_state"]
        print(
            f"[{summary['recording']} | {summary['checkpoint']} | "
            f"assist={summary['root_assist']}]"
        )
        if not task.get("available"):
            print(f"  unavailable: {task.get('reason', 'unknown reason')}")
            continue
        for dof in task.get("dofs", []):
            print(
                f"  {dof['label']}: initial={_number(dof.get('initial'))}, "
                f"final={_number(dof.get('final'))}, "
                f"source_final={_number(dof.get('source_final'))}, "
                f"final-source={_number(dof.get('final_minus_source'))}"
            )
        if isinstance(task.get("drawer"), dict):
            drawer = task["drawer"]
            print(
                "  drawer indicator: "
                f"motion={_number(drawer.get('motion_from_initial'))}, "
                f"recorded_motion={_number(drawer.get('source_motion_from_initial'))}, "
                f"normalized_progress={_number(drawer.get('source_normalized_progress'))}"
            )
        if isinstance(task.get("scanner"), dict):
            scanner = task["scanner"]
            print(
                "  scanner indicator: "
                f"position_motion_m={_number(scanner.get('position_motion_m'))}, "
                "final_position_error_to_source_m="
                f"{_number(scanner.get('final_position_error_to_source_m'))}, "
                "final_orientation_error_to_source_deg="
                f"{_number(scanner.get('final_orientation_error_to_source_deg'), 2)}"
            )
    print(
        "\nInterpretation: body RMSE measures articulation tracking of the constructed "
        "50 Hz target. Root errors measure world-space deviation from original qpos. "
        "Root-assist coordinates are oracle-overwritten. Object terminal qpos alone "
        "cannot prove grasp/contact/placement/stability success; use replay or an "
        "explicit task-specific criterion."
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs",
        nargs="*",
        type=Path,
        help="explicit completed run directories",
    )
    parser.add_argument(
        "--scan-root",
        action="append",
        type=Path,
        help=(
            "recursively find run_metadata.json below this directory; may be "
            "repeated. Defaults to the v1.1 comparison output when no run is given"
        ),
    )
    parser.add_argument(
        "--warmup-s",
        type=float,
        help=(
            "override trajectory/root warmup; default uses each launch manifest "
            "warmup_exclusion_s (normally 0.2 s)"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print detailed machine-readable JSON instead of the compact table",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.warmup_s is not None and args.warmup_s < 0:
        raise SystemExit("ERROR: --warmup-s must be non-negative")
    scan_roots = list(args.scan_root or [])
    if not scan_roots and not args.runs:
        scan_roots = [DEFAULT_SCAN_ROOT]
    try:
        runs = discover_runs(scan_roots, args.runs)
        if not runs:
            raise SummaryError("no completed runs with run_metadata.json were found")
        summaries = [summarize_run(run, args.warmup_s) for run in runs]
    except SummaryError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    if args.json:
        print(
            json.dumps(
                _json_safe(
                    {
                        "schema_version": 1,
                        "task_success_semantics": (
                            "not inferred; terminal object qpos are descriptive indicators"
                        ),
                        "runs": summaries,
                    }
                ),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )
    else:
        print_table(summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
