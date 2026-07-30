#!/usr/bin/env python3
"""Compare regular and low-latency qpos-track rollouts.

The deploy logger writes one 50 Hz CSV per signal.  This script uses
``motion_playing.csv`` as the authoritative playback gate, maps the C++
``target_motion.csv`` rows back to exact ``prepared_reference.npz`` frame IDs,
and reports frame-aligned tracking, smoothness, numerical-validity, and
task-object terminal-state metrics.

No project source is imported so that completed runs remain analysable even
when the simulator/deploy Python environments are not active.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

import numpy as np


TRACK_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = TRACK_ROOT / "data"

# q.csv, dq.csv, target_motion.csv, and robot commands use this MuJoCo /
# hardware order.  action.csv is the raw policy output in IsaacLab order.
MJ_JOINT_NAMES = (
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "waist_yaw",
    "waist_roll",
    "waist_pitch",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
)

# For a MuJoCo joint i, select raw_action[MUJOCO_TO_ISAACLAB[i]].
# This is named isaaclab_to_mujoco in the C++ policy_parameters.hpp, but its
# actual use is exactly the indexing operation above.
MUJOCO_TO_ISAACLAB = np.asarray(
    [
        0,
        3,
        6,
        9,
        13,
        17,
        1,
        4,
        7,
        10,
        14,
        18,
        2,
        5,
        8,
        11,
        15,
        19,
        21,
        23,
        25,
        27,
        12,
        16,
        20,
        22,
        24,
        26,
        28,
    ],
    dtype=np.int64,
)

JOINT_GROUPS: dict[str, tuple[int, ...]] = {
    "left_leg": tuple(range(0, 6)),
    "right_leg": tuple(range(6, 12)),
    "waist": tuple(range(12, 15)),
    "left_arm": tuple(range(15, 22)),
    "right_arm": tuple(range(22, 29)),
}

_METADATA_COLUMNS = {
    "index",
    "frame",
    "frame_index",
    "time",
    "time_s",
    "time_ms",
    "time_realtime_ms",
    "time_monotonic_ms",
    "ros_timestamp",
}


class ComparisonError(RuntimeError):
    """Raised when a rollout cannot be compared safely."""


@dataclass(frozen=True)
class NumericLog:
    path: Path
    indices: np.ndarray
    time_s: np.ndarray
    values: np.ndarray
    value_names: tuple[str, ...]


@dataclass(frozen=True)
class TargetLog:
    path: Path
    joint_pos_mj: np.ndarray
    explicit_indices: np.ndarray | None
    has_header: bool


@dataclass(frozen=True)
class PreparedReference:
    path: Path
    joint_pos_mj: np.ndarray
    policy_seq: np.ndarray


@dataclass
class RunAnalysis:
    run_dir: Path
    summary: dict[str, Any]
    indices: np.ndarray
    time_s: np.ndarray
    reference_frame_indices: np.ndarray
    reference_policy_seq: np.ndarray
    has_prepared_reference_alignment: bool
    target_q_mj: np.ndarray
    measured_q_mj: np.ndarray
    error_mj: np.ndarray
    action_mj: np.ndarray


def _normalise_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def _numeric_suffix(name: str) -> int | None:
    match = re.search(r"(?:_|\[)(\d+)\]?$", name)
    return int(match.group(1)) if match else None


def _load_json_if_present(paths: Iterable[Path]) -> tuple[dict[str, Any], Path | None]:
    for path in paths:
        if path.is_file():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ComparisonError(f"cannot read JSON {path}: {exc}") from exc
            if not isinstance(loaded, dict):
                raise ComparisonError(f"expected a JSON object in {path}")
            return loaded, path
    return {}, None


def _line_sample(line_number: int, text: str) -> dict[str, Any]:
    return {"line": line_number, "text": text.strip()[:500]}


def _protocol_timing_audit(run_dir: Path) -> dict[str, Any]:
    """Audit publisher deadlines and non-initial streamed-motion catch-ups.

    The first packet necessarily constructs a new streamed MotionSequence, so
    C++ reports ``did_catchup=1`` followed by one ``Catch-up: Reset``.  Those
    two initial lines are classified separately and are not failures.
    """

    publisher_path = run_dir / "publisher.log"
    deploy_path = run_dir / "deploy.log"
    missing = [
        str(path.name)
        for path in (publisher_path, deploy_path)
        if not path.is_file()
    ]
    publisher_lines = (
        publisher_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if publisher_path.is_file()
        else []
    )
    deploy_lines = (
        deploy_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if deploy_path.is_file()
        else []
    )

    late_matches: list[tuple[int, int, str]] = []
    for line_number, line in enumerate(publisher_lines, start=1):
        match = re.search(r"\blate_ticks\s*=\s*(\d+)", line, flags=re.IGNORECASE)
        if match:
            late_matches.append((line_number, int(match.group(1)), line))
    late_ticks = late_matches[-1][1] if late_matches else None

    merged: list[tuple[int, int, str]] = []
    for line_number, line in enumerate(deploy_lines, start=1):
        if "merged streamed data:" not in line.lower():
            continue
        match = re.search(r"\bdid_catchup\s*=\s*(true|false|1|0)", line, re.I)
        if match:
            did_catchup = match.group(1).lower() in ("true", "1")
            merged.append((line_number, int(did_catchup), line))

    initial_packet_did_catchup = bool(merged and merged[0][1] == 1)
    non_initial_did_catchup = [
        item for packet, item in enumerate(merged) if packet > 0 and item[1] == 1
    ]
    forcing = [
        (line_number, line)
        for line_number, line in enumerate(deploy_lines, start=1)
        if "forcing catch-up" in line.lower()
    ]
    gap_catchup = [
        (line_number, line)
        for line_number, line in enumerate(deploy_lines, start=1)
        if "catch-up:" in line.lower()
        and "catch-up: reset" not in line.lower()
    ]
    reset_lines = [
        (line_number, line)
        for line_number, line in enumerate(deploy_lines, start=1)
        if "catch-up: reset" in line.lower()
    ]

    # At most one reset between the first and second merged-packet lines is the
    # expected first-packet reset.  Generic startup/safety "reset" messages are
    # deliberately ignored because they are not streamed catch-up events.
    normal_initial_reset: tuple[int, str] | None = None
    if initial_packet_did_catchup:
        first_merged_line = merged[0][0]
        next_merged_line = merged[1][0] if len(merged) > 1 else math.inf
        normal_initial_reset = next(
            (
                item
                for item in reset_lines
                if first_merged_line < item[0] < next_merged_line
            ),
            None,
        )
    non_initial_reset = [
        item for item in reset_lines if item != normal_initial_reset
    ]

    abnormal_by_line: dict[int, str] = {}
    for line_number, _flag, line in non_initial_did_catchup:
        abnormal_by_line[line_number] = line
    for line_number, line in (*forcing, *gap_catchup, *non_initial_reset):
        abnormal_by_line[line_number] = line
    abnormal_samples = [
        _line_sample(line_number, abnormal_by_line[line_number])
        for line_number in sorted(abnormal_by_line)[:8]
    ]
    complete = not missing and late_ticks is not None and bool(merged)
    valid = bool(
        complete
        and late_ticks == 0
        and not non_initial_did_catchup
        and not forcing
        and not gap_catchup
        and not non_initial_reset
    )
    warnings: list[str] = []
    if missing:
        warnings.append(
            "protocol timing audit incomplete; missing " + ", ".join(missing)
        )
    if not late_matches and publisher_path.is_file():
        warnings.append("publisher.log has no final late_ticks record")
    if not merged and deploy_path.is_file():
        warnings.append("deploy.log has no merged streamed packet record")
    if late_ticks not in (None, 0):
        warnings.append(f"publisher missed {late_ticks} 50 Hz deadlines")
    abnormal_count = len(abnormal_by_line)
    if abnormal_count:
        warnings.append(
            f"deploy reported {abnormal_count} non-initial catch-up/reset lines"
        )
    return {
        "valid": valid,
        "complete": complete,
        "publisher": {
            "late_ticks": late_ticks,
            "late_ticks_nonzero": (
                bool(late_ticks) if late_ticks is not None else None
            ),
            "records": len(late_matches),
            "sample": (
                _line_sample(late_matches[-1][0], late_matches[-1][2])
                if late_matches
                else None
            ),
        },
        "deploy_stream": {
            "merged_packets": len(merged),
            "initial_packet_did_catchup": initial_packet_did_catchup,
            "normal_initial_reset_count": int(normal_initial_reset is not None),
            "non_initial_did_catchup_count": len(non_initial_did_catchup),
            "forcing_catchup_count": len(forcing),
            "gap_catchup_count": len(gap_catchup),
            "non_initial_catchup_reset_count": len(non_initial_reset),
            "abnormal_related_line_count": abnormal_count,
            "abnormal_samples": abnormal_samples,
        },
        "classification_note": (
            "The first merged packet's did_catchup=1 and its immediately following "
            "Catch-up: Reset are expected initialization and are not failures."
        ),
        "warnings": warnings,
    }


def _select_value_columns(
    header: Sequence[str], prefix: str, expected_dim: int | None
) -> list[int]:
    normalised_prefix = _normalise_name(prefix)
    matched: list[tuple[int, int]] = []
    for column, raw_name in enumerate(header):
        name = _normalise_name(raw_name)
        if name == normalised_prefix or name.startswith(normalised_prefix + "_"):
            suffix = _numeric_suffix(raw_name)
            if suffix is not None:
                matched.append((suffix, column))
    if matched:
        matched.sort()
        columns = [column for _, column in matched]
    else:
        columns = [
            column
            for column, name in enumerate(header)
            if _normalise_name(name) not in _METADATA_COLUMNS
        ]
        if expected_dim is not None and len(columns) >= expected_dim:
            columns = columns[-expected_dim:]
    if expected_dim is not None and len(columns) != expected_dim:
        raise ComparisonError(
            f"{prefix} log has {len(columns)} value columns in header {list(header)!r}; "
            f"expected {expected_dim}"
        )
    if not columns:
        raise ComparisonError(f"no value columns found for {prefix}")
    return columns


def read_numeric_log(
    path: Path, prefix: str, expected_dim: int | None = None
) -> NumericLog:
    """Read a split deploy CSV using names first and dimensions as fallback."""

    if not path.is_file():
        raise ComparisonError(f"missing required log: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.reader(stream))
    if not rows:
        raise ComparisonError(f"empty CSV: {path}")
    header = [name.strip() for name in rows[0]]
    columns = _select_value_columns(header, prefix, expected_dim)
    normalised = [_normalise_name(name) for name in header]
    index_column = next(
        (normalised.index(name) for name in ("index", "frame_index", "frame") if name in normalised),
        None,
    )
    time_column: int | None = None
    time_scale = 1.0
    for candidate, scale in (
        ("time_s", 1.0),
        ("time_ms", 1e-3),
        ("time_monotonic_ms", 1e-3),
        ("time", 1.0),
    ):
        if candidate in normalised:
            time_column = normalised.index(candidate)
            time_scale = scale
            break

    parsed_indices: list[int] = []
    parsed_times: list[float] = []
    parsed_values: list[list[float]] = []
    for row_number, row in enumerate(rows[1:], start=2):
        if not row or not any(cell.strip() for cell in row):
            continue
        if len(row) < len(header):
            row = list(row) + [""] * (len(header) - len(row))
        try:
            index = (
                int(float(row[index_column]))
                if index_column is not None
                else len(parsed_indices)
            )
            time_value = (
                float(row[time_column]) * time_scale
                if time_column is not None and row[time_column].strip()
                else math.nan
            )
            values = [
                float(row[column]) if row[column].strip() else math.nan
                for column in columns
            ]
        except (ValueError, IndexError) as exc:
            raise ComparisonError(f"invalid numeric row {row_number} in {path}: {exc}") from exc
        parsed_indices.append(index)
        parsed_times.append(time_value)
        parsed_values.append(values)
    if not parsed_indices:
        raise ComparisonError(f"no data rows in {path}")
    if len(set(parsed_indices)) != len(parsed_indices):
        raise ComparisonError(f"duplicate indices in {path}")
    return NumericLog(
        path=path,
        indices=np.asarray(parsed_indices, dtype=np.int64),
        time_s=np.asarray(parsed_times, dtype=np.float64),
        values=np.asarray(parsed_values, dtype=np.float64),
        value_names=tuple(header[column] for column in columns),
    )


def _row_is_numeric(row: Sequence[str]) -> bool:
    nonempty = [cell.strip() for cell in row if cell.strip()]
    if not nonempty:
        return False
    try:
        for cell in nonempty:
            float(cell)
    except ValueError:
        return False
    return True


def _target_joint_columns(header: Sequence[str]) -> list[int]:
    indexed: list[tuple[int, int]] = []
    canonical = {_normalise_name(name): i for i, name in enumerate(MJ_JOINT_NAMES)}
    named: list[tuple[int, int]] = []
    for column, raw_name in enumerate(header):
        name = _normalise_name(raw_name)
        if name.endswith("_joint"):
            name = name[: -len("_joint")]
        if name in canonical:
            named.append((canonical[name], column))
            continue
        match = re.fullmatch(
            r"(?:joint_pos|joint_position|target_q|q)_(\d+)", name
        )
        if match:
            indexed.append((int(match.group(1)), column))
    selected = named if len(named) == 29 else indexed
    if len(selected) == 29 and sorted(index for index, _ in selected) == list(range(29)):
        return [column for _, column in sorted(selected)]

    non_metadata = [
        column
        for column, name in enumerate(header)
        if _normalise_name(name) not in _METADATA_COLUMNS
    ]
    if len(non_metadata) >= 36:
        # C++ layout is root xyz + root wxyz + 29 q values.
        return non_metadata[-29:]
    if len(non_metadata) == 29:
        return non_metadata
    raise ComparisonError(
        f"cannot identify 29 joint columns in target-motion header {list(header)!r}"
    )


def read_target_motion(path: Path) -> TargetLog:
    """Read C++ target motion, accepting both headerless and named variants."""

    if not path.is_file():
        raise ComparisonError(f"missing required target motion: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = [row for row in csv.reader(stream) if any(cell.strip() for cell in row)]
    if not rows:
        raise ComparisonError(f"empty target motion: {path}")
    has_header = not _row_is_numeric(rows[0])
    explicit_indices: list[int] | None = [] if has_header else None
    parsed: list[list[float]] = []

    if not has_header:
        for row_number, row in enumerate(rows, start=1):
            try:
                values = [float(cell) for cell in row if cell.strip()]
            except ValueError as exc:
                raise ComparisonError(
                    f"invalid target-motion row {row_number} in {path}: {exc}"
                ) from exc
            if len(values) < 36:
                raise ComparisonError(
                    f"target-motion row {row_number} in {path} has {len(values)} "
                    "numbers; expected root xyz + quaternion + 29 joints"
                )
            parsed.append(values[-29:])
    else:
        header = [cell.strip() for cell in rows[0]]
        normalised = [_normalise_name(name) for name in header]
        joint_columns = _target_joint_columns(header)
        index_column = next(
            (
                normalised.index(name)
                for name in ("index", "frame_index", "frame")
                if name in normalised
            ),
            None,
        )
        if index_column is None:
            explicit_indices = None
        for row_number, row in enumerate(rows[1:], start=2):
            if len(row) < len(header):
                row = list(row) + [""] * (len(header) - len(row))
            try:
                parsed.append([float(row[column]) for column in joint_columns])
                if explicit_indices is not None and index_column is not None:
                    explicit_indices.append(int(float(row[index_column])))
            except (ValueError, IndexError) as exc:
                raise ComparisonError(
                    f"invalid target-motion row {row_number} in {path}: {exc}"
                ) from exc
    if not parsed:
        raise ComparisonError(f"no target-motion data rows in {path}")
    return TargetLog(
        path=path,
        joint_pos_mj=np.asarray(parsed, dtype=np.float64),
        explicit_indices=(
            np.asarray(explicit_indices, dtype=np.int64)
            if explicit_indices is not None
            else None
        ),
        has_header=has_header,
    )


def _read_prepared_reference(path: Path) -> PreparedReference | None:
    """Read the exact qpos-derived reference saved by the publisher.

    ``joint_pos`` is stored in IsaacLab order.  The C++ target-motion log is in
    MuJoCo/hardware order, so convert here before matching individual playback
    rows back to their source reference frame.
    """

    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            if "joint_pos" not in archive:
                raise ComparisonError(
                    f"prepared reference is missing joint_pos: {path}"
                )
            joint_pos_isaac = np.asarray(
                archive["joint_pos"], dtype=np.float64
            )
            if joint_pos_isaac.ndim != 2 or joint_pos_isaac.shape[1] != 29:
                raise ComparisonError(
                    f"prepared reference joint_pos must be N x 29, got "
                    f"{joint_pos_isaac.shape}: {path}"
                )
            policy_seq = (
                np.asarray(archive["policy_seq"], dtype=np.int64)
                if "policy_seq" in archive
                else np.arange(len(joint_pos_isaac), dtype=np.int64)
            )
    except (OSError, ValueError) as exc:
        raise ComparisonError(f"cannot read prepared reference {path}: {exc}") from exc
    if policy_seq.shape != (len(joint_pos_isaac),):
        raise ComparisonError(
            f"prepared reference policy_seq has shape {policy_seq.shape}; "
            f"expected {(len(joint_pos_isaac),)}: {path}"
        )
    return PreparedReference(
        path=path,
        joint_pos_mj=joint_pos_isaac[:, MUJOCO_TO_ISAACLAB],
        policy_seq=policy_seq,
    )


def _match_target_rows_to_reference(
    target_q_mj: np.ndarray,
    reference: PreparedReference,
    *,
    tolerance_rad: float = 1e-5,
    local_lookahead: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Monotonically match C++ target rows to prepared-reference frames.

    Target values are direct copies of reference values, but the headerless C++
    logger omits the global frame index and prints with limited decimal
    precision.  A monotonic nearest-value match therefore recovers duplicates
    and skips without assuming that two independent 50 Hz loops stayed in the
    same phase.
    """

    target = np.asarray(target_q_mj, dtype=np.float64)
    source = np.asarray(reference.joint_pos_mj, dtype=np.float64)
    if target.ndim != 2 or target.shape[1] != 29:
        raise ComparisonError(f"target rows must be N x 29, got {target.shape}")
    if source.ndim != 2 or source.shape[1] != 29 or not len(source):
        raise ComparisonError(
            f"prepared-reference rows must be non-empty N x 29, got {source.shape}"
        )

    matched = np.full(len(target), -1, dtype=np.int64)
    residual = np.full(len(target), np.nan, dtype=np.float64)
    previous = 0
    for row, values in enumerate(target):
        if not np.isfinite(values).all():
            continue
        if row == 0:
            candidate_start = 0
            candidate_end = len(source)
            expected = 0
        else:
            candidate_start = previous
            candidate_end = min(len(source), previous + local_lookahead + 1)
            expected = min(previous + 1, len(source) - 1)

        distances = np.max(
            np.abs(source[candidate_start:candidate_end] - values), axis=1
        )
        best_distance = float(np.min(distances))
        if best_distance > tolerance_rad and candidate_end < len(source):
            candidate_end = len(source)
            distances = np.max(
                np.abs(source[candidate_start:candidate_end] - values), axis=1
            )
            best_distance = float(np.min(distances))

        # Prefer the nominal next frame when multiple nearly-identical source
        # poses are tied.  This avoids manufacturing duplicates in stationary
        # portions while retaining a real duplicate when it is measurably the
        # closest source frame.
        tied = np.flatnonzero(
            distances <= best_distance + max(1e-12, tolerance_rad * 1e-3)
        )
        candidates = candidate_start + tied
        best = int(candidates[np.argmin(np.abs(candidates - expected))])
        residual[row] = best_distance
        if best_distance <= tolerance_rad:
            matched[row] = best
            previous = best
    return matched, residual


def _reference_match_diagnostics(
    frame_indices: np.ndarray,
    residual_rad: np.ndarray,
    reference: PreparedReference,
) -> dict[str, Any]:
    valid_rows = np.flatnonzero(frame_indices >= 0)
    unmatched_rows = np.flatnonzero(frame_indices < 0)
    duplicate_rows: list[int] = []
    terminal_hold_rows: list[int] = []
    skipped_events: list[dict[str, int]] = []
    backward_events: list[dict[str, int]] = []
    for previous_row, row in zip(valid_rows[:-1], valid_rows[1:]):
        if row != previous_row + 1:
            continue
        previous_frame = int(frame_indices[previous_row])
        frame = int(frame_indices[row])
        delta = frame - previous_frame
        if delta == 0:
            if frame == len(reference.joint_pos_mj) - 1:
                terminal_hold_rows.append(int(row))
            else:
                duplicate_rows.append(int(row))
        elif delta > 1:
            skipped_events.append(
                {
                    "target_row": int(row),
                    "previous_reference_frame": previous_frame,
                    "reference_frame": frame,
                    "skipped_frames": delta - 1,
                }
            )
        elif delta < 0:
            backward_events.append(
                {
                    "target_row": int(row),
                    "previous_reference_frame": previous_frame,
                    "reference_frame": frame,
                }
            )
    finite_residual = residual_rad[np.isfinite(residual_rad)]
    first_frame = int(frame_indices[valid_rows[0]]) if len(valid_rows) else None
    last_frame = int(frame_indices[valid_rows[-1]]) if len(valid_rows) else None
    return {
        "available": True,
        "method": (
            "monotonic nearest joint-position match to prepared_reference.npz "
            "in MuJoCo order"
        ),
        "prepared_reference": str(reference.path),
        "reference_frames": int(len(reference.joint_pos_mj)),
        "target_rows": int(len(frame_indices)),
        "matched_rows": int(len(valid_rows)),
        "unmatched_rows": int(len(unmatched_rows)),
        "unmatched_row_sample": unmatched_rows[:8].astype(int).tolist(),
        "match_tolerance_rad": 1e-5,
        "match_max_abs_residual_rad": (
            float(np.max(finite_residual)) if finite_residual.size else None
        ),
        "first_reference_frame": first_frame,
        "last_reference_frame": last_frame,
        "first_policy_seq": (
            int(reference.policy_seq[first_frame]) if first_frame is not None else None
        ),
        "last_policy_seq": (
            int(reference.policy_seq[last_frame]) if last_frame is not None else None
        ),
        "unexpected_duplicate_count": len(duplicate_rows),
        "unexpected_duplicate_target_rows": duplicate_rows[:8],
        "terminal_hold_count": len(terminal_hold_rows),
        "skipped_event_count": len(skipped_events),
        "skipped_frame_count": int(
            sum(event["skipped_frames"] for event in skipped_events)
        ),
        "skipped_event_sample": skipped_events[:8],
        "backward_event_count": len(backward_events),
        "backward_event_sample": backward_events[:8],
        "strict_progression_before_terminal": bool(
            len(unmatched_rows) == 0
            and not duplicate_rows
            and not skipped_events
            and not backward_events
        ),
    }


def _true_segments(log: NumericLog) -> list[np.ndarray]:
    playing = np.isfinite(log.values[:, 0]) & (log.values[:, 0] > 0.5)
    true_rows = np.flatnonzero(playing)
    if not len(true_rows):
        raise ComparisonError(f"motion never entered playing state in {log.path}")
    boundaries = np.flatnonzero(np.diff(true_rows) > 1) + 1
    return [part for part in np.split(true_rows, boundaries) if len(part)]


def _infer_log_dt(log: NumericLog, fallback: float = 0.02) -> float:
    finite = np.isfinite(log.time_s)
    differences = np.diff(log.time_s[finite])
    differences = differences[differences > 0]
    if len(differences):
        return float(np.median(differences))
    return fallback


def _time_vector(log: NumericLog, fallback_dt: float) -> np.ndarray:
    if np.isfinite(log.time_s).all():
        return log.time_s.copy()
    return np.arange(len(log.indices), dtype=np.float64) * fallback_dt


def _map_target_to_indices(
    target: TargetLog,
    q_log: NumericLog,
    play_log: NumericLog,
    selected_play_rows: np.ndarray,
    warnings: list[str],
) -> dict[int, np.ndarray]:
    target_rows = target.joint_pos_mj
    if target.explicit_indices is not None:
        if len(target.explicit_indices) != len(target_rows):
            raise ComparisonError("target-motion explicit-index count does not match rows")
        return {
            int(index): target_rows[row]
            for row, index in enumerate(target.explicit_indices)
        }

    if len(target_rows) == len(q_log.indices):
        keys = q_log.indices
        method = "row-for-row with q.csv"
    elif len(target_rows) == len(play_log.indices):
        keys = play_log.indices
        method = "row-for-row with motion_playing.csv"
    else:
        all_playing_rows = np.flatnonzero(
            np.isfinite(play_log.values[:, 0]) & (play_log.values[:, 0] > 0.5)
        )
        if len(target_rows) == len(all_playing_rows):
            keys = play_log.indices[all_playing_rows]
            method = "row-for-row with all playing rows"
        elif len(target_rows) == len(selected_play_rows):
            keys = play_log.indices[selected_play_rows]
            method = "row-for-row with first playing segment"
        else:
            count = min(len(target_rows), len(q_log.indices))
            if count < 2:
                raise ComparisonError(
                    "cannot align target_motion.csv: row count matches neither q.csv "
                    "nor motion_playing.csv"
                )
            keys = q_log.indices[:count]
            target_rows = target_rows[:count]
            method = f"fallback truncation to first {count} q.csv rows"
            warnings.append(
                "target_motion.csv row count did not match q.csv or motion_playing.csv; "
                f"used {method}"
            )
    warnings.append(f"target alignment: {method}")
    return {int(index): target_rows[row] for row, index in enumerate(keys)}


def _align_log(log: NumericLog, indices: np.ndarray) -> np.ndarray:
    lookup = {int(index): row for row, index in enumerate(log.indices)}
    output = np.full((len(indices), log.values.shape[1]), np.nan, dtype=np.float64)
    for output_row, index in enumerate(indices):
        source_row = lookup.get(int(index))
        if source_row is not None:
            output[output_row] = log.values[source_row]
    return output


def _finite_number(value: float | np.floating[Any]) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _json_safe(value: Any) -> Any:
    """Replace NumPy scalars and non-finite floats before strict JSON output."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _array_metrics(values: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(values)
    finite_values = values[finite]
    return {
        "rows": int(values.shape[0]),
        "dimension": int(values.shape[1]) if values.ndim == 2 else 1,
        "all_finite": bool(finite.all()),
        "finite_fraction": float(finite.mean()) if finite.size else 0.0,
        "min": _finite_number(np.min(finite_values)) if finite_values.size else None,
        "max": _finite_number(np.max(finite_values)) if finite_values.size else None,
    }


def _error_metrics(error: np.ndarray) -> dict[str, Any]:
    if error.ndim != 2 or error.shape[1] != 29:
        raise ComparisonError(f"tracking error must be N x 29, got {error.shape}")
    absolute = np.abs(error)
    finite = np.isfinite(error)

    def aggregate(indices: Sequence[int]) -> dict[str, Any]:
        selected = error[:, indices]
        selected_abs = np.abs(selected)
        selected_finite = np.isfinite(selected)
        values = selected[selected_finite]
        abs_values = selected_abs[selected_finite]
        return {
            "samples": int(values.size),
            "all_finite": bool(selected_finite.all()),
            "rmse_rad": (
                float(np.sqrt(np.mean(np.square(values)))) if values.size else None
            ),
            "mae_rad": float(np.mean(abs_values)) if values.size else None,
            "p95_abs_rad": float(np.percentile(abs_values, 95)) if values.size else None,
            "max_abs_rad": float(np.max(abs_values)) if values.size else None,
        }

    per_joint: list[dict[str, Any]] = []
    for joint_index, joint_name in enumerate(MJ_JOINT_NAMES):
        stats = aggregate([joint_index])
        stats.update(
            {
                "joint_index_mj": joint_index,
                "joint_name": joint_name,
                "group": next(
                    name
                    for name, members in JOINT_GROUPS.items()
                    if joint_index in members
                ),
            }
        )
        per_joint.append(stats)
    return {
        "overall": aggregate(tuple(range(29))),
        "groups": {
            name: aggregate(indices) for name, indices in JOINT_GROUPS.items()
        },
        "per_joint": per_joint,
        "finite_elements": int(finite.sum()),
        "elements": int(finite.size),
        "max_error_location": (
            {
                "evaluation_row": int(np.unravel_index(np.nanargmax(absolute), absolute.shape)[0]),
                "joint_index_mj": int(np.unravel_index(np.nanargmax(absolute), absolute.shape)[1]),
                "joint_name": MJ_JOINT_NAMES[
                    int(np.unravel_index(np.nanargmax(absolute), absolute.shape)[1])
                ],
            }
            if finite.any()
            else None
        ),
    }


def _smoothness(values: np.ndarray, time_s: np.ndarray) -> dict[str, Any]:
    if len(values) < 2:
        return {
            "adjacent_pairs": 0,
            "step_delta_rms": None,
            "max_abs_step_delta": None,
            "rate_rms_per_s": None,
            "max_abs_rate_per_s": None,
            "second_difference_rms": None,
        }
    dt = np.diff(time_s)
    pair_finite = (
        np.isfinite(values[:-1]).all(axis=1)
        & np.isfinite(values[1:]).all(axis=1)
        & np.isfinite(dt)
        & (dt > 0)
    )
    delta = np.diff(values, axis=0)
    if not pair_finite.any():
        return {
            "adjacent_pairs": 0,
            "step_delta_rms": None,
            "max_abs_step_delta": None,
            "rate_rms_per_s": None,
            "max_abs_rate_per_s": None,
            "second_difference_rms": None,
        }
    valid_delta = delta[pair_finite]
    rate = valid_delta / dt[pair_finite, None]
    triple_finite = (
        np.isfinite(values[:-2]).all(axis=1)
        & np.isfinite(values[1:-1]).all(axis=1)
        & np.isfinite(values[2:]).all(axis=1)
    )
    second = np.diff(values, n=2, axis=0)[triple_finite]
    per_joint_step_rms = np.sqrt(np.mean(np.square(valid_delta), axis=0))
    return {
        "adjacent_pairs": int(pair_finite.sum()),
        "step_delta_rms": float(np.sqrt(np.mean(np.square(valid_delta)))),
        "max_abs_step_delta": float(np.max(np.abs(valid_delta))),
        "rate_rms_per_s": float(np.sqrt(np.mean(np.square(rate)))),
        "max_abs_rate_per_s": float(np.max(np.abs(rate))),
        "second_difference_rms": (
            float(np.sqrt(np.mean(np.square(second)))) if second.size else None
        ),
        "per_joint_step_delta_rms": per_joint_step_rms.tolist(),
    }


def _quat_error_deg(first: Sequence[float], second: Sequence[float]) -> float | None:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.shape != (4,) or b.shape != (4,) or not (
        np.isfinite(a).all() and np.isfinite(b).all()
    ):
        return None
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return None
    dot = float(np.clip(abs(np.dot(a / norm_a, b / norm_b)), 0.0, 1.0))
    return math.degrees(2.0 * math.acos(dot))


def _task_terminal_state(metadata: dict[str, Any]) -> dict[str, Any]:
    task_start = metadata.get("task_qpos_start")
    labels = metadata.get("task_qpos_labels")

    def task_array(task_key: str, full_key: str) -> np.ndarray:
        if task_key in metadata:
            return np.asarray(metadata[task_key], dtype=np.float64)
        if full_key in metadata and isinstance(task_start, int):
            return np.asarray(metadata[full_key], dtype=np.float64)[task_start:]
        return np.asarray([], dtype=np.float64)

    initial = task_array("initial_task_qpos", "initial_qpos")
    final = task_array("final_task_qpos", "final_qpos")
    source = task_array("current_source_task_qpos", "current_source_qpos")
    count = min(len(initial), len(final))
    if count == 0:
        return {
            "available": False,
            "reason": "run metadata has no initial/final task qpos",
        }
    initial = initial[:count]
    final = final[:count]
    source = source[:count] if len(source) >= count else np.full(count, np.nan)
    if not isinstance(labels, list) or len(labels) < count:
        base = int(task_start) if isinstance(task_start, int) else 0
        labels = [f"task_qpos[{base + index}]" for index in range(count)]
    labels = [str(name) for name in labels[:count]]
    entries = []
    for index, label in enumerate(labels):
        entries.append(
            {
                "task_index": index,
                "label": label,
                "initial": _finite_number(initial[index]),
                "final": _finite_number(final[index]),
                "delta": _finite_number(final[index] - initial[index]),
                "source_final": _finite_number(source[index]),
                "final_minus_source": _finite_number(final[index] - source[index]),
            }
        )
    result: dict[str, Any] = {
        "available": True,
        "dofs": entries,
        "all_finite": bool(np.isfinite(initial).all() and np.isfinite(final).all()),
        "motion_l2_from_initial": float(np.linalg.norm(final - initial)),
        "final_l2_error_to_recorded_source": (
            float(np.linalg.norm(final - source)) if np.isfinite(source).all() else None
        ),
        "interpretation": (
            "These are physical terminal qpos values. They do not by themselves "
            "prove semantic task success; confirm in the viewer or replay."
        ),
    }

    # Preserve useful task-specific indicators when names make them unambiguous.
    drawer_indices = [
        index for index, name in enumerate(labels) if "drawer" in name.lower()
    ]
    if drawer_indices:
        index = drawer_indices[0]
        source_motion = source[index] - initial[index]
        result["drawer"] = {
            "label": labels[index],
            "motion_from_initial": float(final[index] - initial[index]),
            "source_motion_from_initial": _finite_number(source_motion),
            "source_normalized_progress": (
                float((final[index] - initial[index]) / source_motion)
                if math.isfinite(float(source_motion)) and abs(source_motion) > 1e-12
                else None
            ),
        }
    scanner_start = next(
        (
            index
            for index, name in enumerate(labels)
            if "scanner" in name.lower() and name.endswith("[0]")
        ),
        None,
    )
    if scanner_start is not None and scanner_start + 7 <= count:
        initial_scanner = initial[scanner_start : scanner_start + 7]
        final_scanner = final[scanner_start : scanner_start + 7]
        source_scanner = source[scanner_start : scanner_start + 7]
        result["scanner"] = {
            "position_motion_m": float(
                np.linalg.norm(final_scanner[:3] - initial_scanner[:3])
            ),
            "source_position_motion_m": (
                float(np.linalg.norm(source_scanner[:3] - initial_scanner[:3]))
                if np.isfinite(source_scanner[:3]).all()
                else None
            ),
            "final_position_error_to_source_m": (
                float(np.linalg.norm(final_scanner[:3] - source_scanner[:3]))
                if np.isfinite(source_scanner[:3]).all()
                else None
            ),
            "final_orientation_error_to_source_deg": _quat_error_deg(
                final_scanner[3:7], source_scanner[3:7]
            ),
        }
    return result


def _manifest_warmup(manifest: dict[str, Any]) -> float:
    initialization = manifest.get("initialization")
    if isinstance(initialization, dict):
        value = initialization.get("warmup_exclusion_s")
        if isinstance(value, (int, float)) and value >= 0:
            return float(value)
    return 0.2


def _checkpoint_identity(
    run_dir: Path, manifest: dict[str, Any], deploy_metadata: dict[str, Any]
) -> dict[str, Any]:
    robot_config = deploy_metadata.get("robot_config", {})
    if not isinstance(robot_config, dict):
        robot_config = {}
    return {
        "run_dir": str(run_dir),
        "checkpoint": manifest.get("checkpoint", run_dir.name),
        "models": manifest.get("models", {}),
        "encoder_file": robot_config.get("encoder_file"),
        "decoder_file": robot_config.get("model_path"),
        "observation_config": robot_config.get("obs_config_path"),
    }


def analyze_run(run_dir: Path, warmup_s: float | None = None) -> RunAnalysis:
    run_dir = run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise ComparisonError(f"run directory does not exist: {run_dir}")
    deploy_dir = run_dir / "deploy_csv"
    q_log = read_numeric_log(deploy_dir / "q.csv", "q", 29)
    dq_log = read_numeric_log(deploy_dir / "dq.csv", "dq", 29)
    action_log = read_numeric_log(deploy_dir / "action.csv", "act", 29)
    token_log = read_numeric_log(deploy_dir / "token_state.csv", "token", None)
    play_log = read_numeric_log(deploy_dir / "motion_playing.csv", "playing", 1)
    target_log = read_target_motion(run_dir / "target_motion.csv")

    metadata, metadata_path = _load_json_if_present(
        [run_dir / "run_metadata.json", run_dir / "sim_metadata.json"]
    )
    manifest, manifest_path = _load_json_if_present(
        [run_dir / "launch_manifest.json", run_dir / "run_manifest.json"]
    )
    diagnostics, diagnostics_path = _load_json_if_present(
        [run_dir / "reference_diagnostics.json", run_dir / "conversion_report.json"]
    )
    deploy_metadata, deploy_metadata_path = _load_json_if_present(
        [deploy_dir / "metadata.json"]
    )
    protocol_timing = _protocol_timing_audit(run_dir)

    segments = _true_segments(play_log)
    selected_play_rows = segments[0]
    selected_indices = play_log.indices[selected_play_rows]
    warnings: list[str] = []
    warnings.extend(protocol_timing["warnings"])
    if len(segments) > 1:
        warnings.append(
            f"motion_playing contains {len(segments)} true segments; only the first "
            "segment is evaluated"
        )
    target_by_index = _map_target_to_indices(
        target_log, q_log, play_log, selected_play_rows, warnings
    )
    q_lookup = {int(index): row for row, index in enumerate(q_log.indices)}
    usable_indices = np.asarray(
        [
            int(index)
            for index in selected_indices
            if int(index) in q_lookup and int(index) in target_by_index
        ],
        dtype=np.int64,
    )
    if not len(usable_indices):
        raise ComparisonError(
            f"no common playing indices among q.csv and target_motion.csv in {run_dir}"
        )
    usable_target_values = np.stack(
        [target_by_index[int(index)] for index in usable_indices], axis=0
    )
    prepared_reference = _read_prepared_reference(
        run_dir / "prepared_reference.npz"
    )
    if prepared_reference is not None:
        usable_reference_frames, reference_match_residual = (
            _match_target_rows_to_reference(
                usable_target_values, prepared_reference
            )
        )
        unmatched = np.flatnonzero(usable_reference_frames < 0)
        if len(unmatched):
            finite_residual = reference_match_residual[
                np.isfinite(reference_match_residual)
            ]
            maximum = (
                float(np.max(finite_residual)) if finite_residual.size else math.nan
            )
            raise ComparisonError(
                "cannot map target_motion.csv to prepared_reference.npz within "
                f"1e-5 rad in {run_dir}; unmatched target rows "
                f"{unmatched[:8].astype(int).tolist()}, maximum nearest residual "
                f"{maximum:.9g} rad"
            )
        reference_match = _reference_match_diagnostics(
            usable_reference_frames,
            reference_match_residual,
            prepared_reference,
        )
        usable_reference_policy_seq = prepared_reference.policy_seq[
            usable_reference_frames
        ]
        if reference_match["unexpected_duplicate_count"]:
            warnings.append(
                "prepared-reference alignment found "
                f"{reference_match['unexpected_duplicate_count']} non-terminal "
                "duplicate target frame(s); ordinal cross-run alignment would be biased"
            )
        if reference_match["skipped_event_count"]:
            warnings.append(
                "prepared-reference alignment found "
                f"{reference_match['skipped_frame_count']} skipped target frame(s)"
            )
    else:
        usable_reference_frames = np.arange(len(usable_indices), dtype=np.int64)
        usable_reference_policy_seq = usable_reference_frames.copy()
        reference_match = {
            "available": False,
            "method": "evaluation ordinal fallback; prepared_reference.npz is absent",
            "prepared_reference": None,
            "reference_frames": None,
            "target_rows": int(len(usable_indices)),
            "matched_rows": int(len(usable_indices)),
            "unmatched_rows": 0,
            "strict_progression_before_terminal": None,
        }
        warnings.append(
            "prepared_reference.npz is absent; cross-run comparison falls back "
            "to playback ordinals and cannot correct a duplicate/drop mismatch"
        )

    deploy_logging = deploy_metadata.get("logging", {})
    fallback_dt = (
        float(deploy_logging.get("dt", 0.02))
        if isinstance(deploy_logging, dict)
        else 0.02
    )
    q_time_full = _time_vector(q_log, fallback_dt)
    q_time_lookup = {
        int(index): q_time_full[row] for row, index in enumerate(q_log.indices)
    }
    usable_time = np.asarray(
        [q_time_lookup[int(index)] for index in usable_indices], dtype=np.float64
    )
    effective_warmup = (
        _manifest_warmup(manifest) if warmup_s is None else float(warmup_s)
    )
    if effective_warmup < 0:
        raise ComparisonError("warmup must be non-negative")
    relative_time = usable_time - usable_time[0]
    evaluation_mask = relative_time + 1e-12 >= effective_warmup
    evaluation_indices = usable_indices[evaluation_mask]
    evaluation_time = usable_time[evaluation_mask]
    if len(evaluation_indices) < 2:
        raise ComparisonError(
            f"only {len(evaluation_indices)} evaluation rows remain after "
            f"{effective_warmup:.3f}s warmup in {run_dir}"
        )

    q_values = _align_log(q_log, evaluation_indices)
    dq_values = _align_log(dq_log, evaluation_indices)
    action_isaac = _align_log(action_log, evaluation_indices)
    action_mj = action_isaac[:, MUJOCO_TO_ISAACLAB]
    token_values = _align_log(token_log, evaluation_indices)
    target_values = usable_target_values[evaluation_mask]
    evaluation_reference_frames = usable_reference_frames[evaluation_mask]
    evaluation_reference_policy_seq = usable_reference_policy_seq[evaluation_mask]
    error = q_values - target_values
    evaluation_time = evaluation_time - evaluation_time[0]
    dt = _infer_log_dt(q_log, fallback_dt)

    segment_descriptions = []
    play_times = _time_vector(play_log, fallback_dt)
    for segment in segments:
        segment_descriptions.append(
            {
                "start_index": int(play_log.indices[segment[0]]),
                "end_index": int(play_log.indices[segment[-1]]),
                "rows": int(len(segment)),
                "start_time_s": float(play_times[segment[0]]),
                "end_time_s": float(play_times[segment[-1]]),
            }
        )

    runtime_keys = (
        "wall_clock_timing_valid",
        "real_time_factor",
        "max_schedule_lag_s",
        "fallen",
        "invalid_state",
        "stop_reason",
        "final_base_height_m",
    )
    signals_finite = bool(
        np.isfinite(q_values).all()
        and np.isfinite(dq_values).all()
        and np.isfinite(action_mj).all()
        and np.isfinite(token_values).all()
    )
    runtime_validity_warnings = list(protocol_timing["warnings"])
    if metadata.get("wall_clock_timing_valid") is False:
        runtime_validity_warnings.append("simulator wall-clock timing is invalid")
    if metadata.get("fallen") is True:
        runtime_validity_warnings.append("robot fell during rollout")
    if metadata.get("invalid_state") is True:
        runtime_validity_warnings.append("simulator reported an invalid state")
    if not signals_finite:
        runtime_validity_warnings.append(
            "q/dq/action/token evaluation signals contain non-finite values"
        )
    runtime_valid = bool(
        signals_finite
        and protocol_timing["valid"]
        and metadata.get("wall_clock_timing_valid") is not False
        and metadata.get("fallen") is not True
        and metadata.get("invalid_state") is not True
    )
    summary: dict[str, Any] = {
        "identity": _checkpoint_identity(run_dir, manifest, deploy_metadata),
        "files": {
            "run_metadata": str(metadata_path) if metadata_path else None,
            "manifest": str(manifest_path) if manifest_path else None,
            "reference_diagnostics": (
                str(diagnostics_path) if diagnostics_path else None
            ),
            "deploy_metadata": (
                str(deploy_metadata_path) if deploy_metadata_path else None
            ),
            "target_motion": str(target_log.path),
            "prepared_reference": (
                str(prepared_reference.path)
                if prepared_reference is not None
                else None
            ),
        },
        "runtime": {key: metadata.get(key) for key in runtime_keys if key in metadata},
        "protocol_timing_audit": protocol_timing,
        "runtime_validity": {
            "valid": runtime_valid,
            "signals_finite": signals_finite,
            "protocol_timing_valid": protocol_timing["valid"],
            "warnings": runtime_validity_warnings,
        },
        "playback_alignment": {
            "authority": "deploy_csv/motion_playing.csv",
            "selection": "first contiguous segment with playing > 0.5",
            "segments": segment_descriptions,
            "selected_rows_before_warmup": int(len(usable_indices)),
            "warmup_exclusion_s": effective_warmup,
            "evaluation_rows": int(len(evaluation_indices)),
            "evaluation_dt_s": dt,
            "evaluation_duration_s": (
                float(evaluation_time[-1]) if len(evaluation_time) else 0.0
            ),
            "first_evaluated_index": int(evaluation_indices[0]),
            "last_evaluated_index": int(evaluation_indices[-1]),
            "reference_frame_alignment": reference_match,
            "first_evaluated_reference_frame": int(
                evaluation_reference_frames[0]
            ),
            "last_evaluated_reference_frame": int(
                evaluation_reference_frames[-1]
            ),
            "first_evaluated_policy_seq": int(
                evaluation_reference_policy_seq[0]
            ),
            "last_evaluated_policy_seq": int(
                evaluation_reference_policy_seq[-1]
            ),
        },
        "tracking": {
            "joint_order": "MuJoCo/hardware",
            "joint_names": list(MJ_JOINT_NAMES),
            "error_definition": "deploy_csv/q.csv - target_motion.csv joint position",
            "metrics": _error_metrics(error),
        },
        "signals": {
            "q_mj": _array_metrics(q_values),
            "dq_mj": {
                **_array_metrics(dq_values),
                "smoothness": _smoothness(dq_values, evaluation_time),
            },
            "raw_action": {
                **_array_metrics(action_mj),
                "logged_order": "IsaacLab",
                "analysis_order": "MuJoCo (permutation only; values remain raw/unscaled)",
                "smoothness": _smoothness(action_mj, evaluation_time),
            },
            "token_state": {
                **_array_metrics(token_values),
                "smoothness": _smoothness(token_values, evaluation_time),
            },
        },
        "task_object_terminal_state": _task_terminal_state(metadata),
        "reference_diagnostics": {
            "joint_order": diagnostics.get("joint_order"),
            "joint_names": diagnostics.get("joint_names"),
            "source_row_indices_present": "source_row_indices" in diagnostics,
            "policy_seq_present": "policy_seq" in diagnostics,
        },
        "warnings": warnings,
    }
    return RunAnalysis(
        run_dir=run_dir,
        summary=summary,
        indices=evaluation_indices,
        time_s=evaluation_time,
        reference_frame_indices=evaluation_reference_frames,
        reference_policy_seq=evaluation_reference_policy_seq,
        has_prepared_reference_alignment=prepared_reference is not None,
        target_q_mj=target_values,
        measured_q_mj=q_values,
        error_mj=error,
        action_mj=action_mj,
    )


def _metric_delta(
    regular: dict[str, Any], low_latency: dict[str, Any], key: str
) -> float | None:
    first = regular.get(key)
    second = low_latency.get(key)
    if isinstance(first, (int, float)) and isinstance(second, (int, float)):
        return float(second - first)
    return None


def _comparison_rows(
    regular: RunAnalysis, low_latency: RunAnalysis
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Return pairwise rows, preferring exact prepared-reference frame IDs."""

    if (
        regular.has_prepared_reference_alignment
        and low_latency.has_prepared_reference_alignment
    ):
        regular_first: dict[int, int] = {}
        low_first: dict[int, int] = {}
        for row, frame in enumerate(regular.reference_frame_indices):
            regular_first.setdefault(int(frame), row)
        for row, frame in enumerate(low_latency.reference_frame_indices):
            low_first.setdefault(int(frame), row)
        common_frames = np.asarray(
            sorted(set(regular_first).intersection(low_first)), dtype=np.int64
        )
        if len(common_frames) < 2:
            raise ComparisonError(
                "runs have fewer than two common prepared-reference frames"
            )
        regular_rows = np.asarray(
            [regular_first[int(frame)] for frame in common_frames], dtype=np.int64
        )
        low_rows = np.asarray(
            [low_first[int(frame)] for frame in common_frames], dtype=np.int64
        )
        return (
            regular_rows,
            low_rows,
            common_frames,
            (
                "first occurrence of each common prepared_reference frame "
                "after per-run playing gate and warmup"
            ),
        )

    count = min(len(regular.indices), len(low_latency.indices))
    if count < 2:
        raise ComparisonError("runs have fewer than two common playback ordinals")
    rows = np.arange(count, dtype=np.int64)
    return (
        rows,
        rows.copy(),
        rows.copy(),
        (
            "playback ordinal fallback after each run's playing gate and warmup; "
            "prepared-reference alignment unavailable"
        ),
    )


def compare_analyses(
    regular: RunAnalysis, low_latency: RunAnalysis
) -> dict[str, Any]:
    regular_rows, low_rows, common_frames, alignment_method = _comparison_rows(
        regular, low_latency
    )
    count = len(common_frames)
    target_delta = (
        low_latency.target_q_mj[low_rows] - regular.target_q_mj[regular_rows]
    )
    finite_target_delta = target_delta[np.isfinite(target_delta)]
    ordinal_count = min(len(regular.indices), len(low_latency.indices))
    ordinal_target_delta = (
        low_latency.target_q_mj[:ordinal_count]
        - regular.target_q_mj[:ordinal_count]
    )
    finite_ordinal_target_delta = ordinal_target_delta[
        np.isfinite(ordinal_target_delta)
    ]
    ordinal_row_delta = np.nanmax(
        np.abs(ordinal_target_delta), axis=1
    )
    ordinal_mismatch_rows = np.flatnonzero(ordinal_row_delta > 1e-6)
    if finite_ordinal_target_delta.size:
        maximum_ordinal_location = np.unravel_index(
            np.nanargmax(np.abs(ordinal_target_delta)),
            ordinal_target_delta.shape,
        )
        maximum_ordinal_row = int(maximum_ordinal_location[0])
        maximum_ordinal_joint = int(maximum_ordinal_location[1])
        ordinal_maximum_detail: dict[str, Any] | None = {
            "comparison_ordinal": maximum_ordinal_row,
            "regular_logger_index": int(regular.indices[maximum_ordinal_row]),
            "low_latency_logger_index": int(
                low_latency.indices[maximum_ordinal_row]
            ),
            "regular_reference_frame": int(
                regular.reference_frame_indices[maximum_ordinal_row]
            ),
            "low_latency_reference_frame": int(
                low_latency.reference_frame_indices[maximum_ordinal_row]
            ),
            "regular_policy_seq": int(
                regular.reference_policy_seq[maximum_ordinal_row]
            ),
            "low_latency_policy_seq": int(
                low_latency.reference_policy_seq[maximum_ordinal_row]
            ),
            "joint_index_mj": maximum_ordinal_joint,
            "joint_name": MJ_JOINT_NAMES[maximum_ordinal_joint],
            "regular_target_rad": float(
                regular.target_q_mj[
                    maximum_ordinal_row, maximum_ordinal_joint
                ]
            ),
            "low_latency_target_rad": float(
                low_latency.target_q_mj[
                    maximum_ordinal_row, maximum_ordinal_joint
                ]
            ),
            "low_minus_regular_rad": float(
                ordinal_target_delta[
                    maximum_ordinal_row, maximum_ordinal_joint
                ]
            ),
        }
    else:
        ordinal_maximum_detail = None
    regular_tracking = _error_metrics(regular.error_mj[regular_rows])
    low_tracking = _error_metrics(low_latency.error_mj[low_rows])
    regular_overall = regular_tracking["overall"]
    low_overall = low_tracking["overall"]
    group_deltas = {}
    for group in JOINT_GROUPS:
        regular_group = regular_tracking["groups"][group]
        low_group = low_tracking["groups"][group]
        group_deltas[group] = {
            "regular_rmse_rad": regular_group["rmse_rad"],
            "low_latency_rmse_rad": low_group["rmse_rad"],
            "rmse_rad_low_minus_regular": _metric_delta(
                regular_group, low_group, "rmse_rad"
            ),
            "regular_max_abs_rad": regular_group["max_abs_rad"],
            "low_latency_max_abs_rad": low_group["max_abs_rad"],
            "max_abs_rad_low_minus_regular": _metric_delta(
                regular_group, low_group, "max_abs_rad"
            ),
        }
    regular_action = _smoothness(
        regular.action_mj[regular_rows], regular.time_s[regular_rows]
    )
    low_action = _smoothness(
        low_latency.action_mj[low_rows], low_latency.time_s[low_rows]
    )
    regular_validity = regular.summary["runtime_validity"]
    low_validity = low_latency.summary["runtime_validity"]
    validity_warnings = [
        *[
            f"regular: {warning}"
            for warning in regular_validity.get("warnings", [])
        ],
        *[
            f"low_latency: {warning}"
            for warning in low_validity.get("warnings", [])
        ],
    ]
    regular_reference_match = regular.summary["playback_alignment"][
        "reference_frame_alignment"
    ]
    low_reference_match = low_latency.summary["playback_alignment"][
        "reference_frame_alignment"
    ]
    schedule_matches = bool(
        len(regular.reference_frame_indices)
        == len(low_latency.reference_frame_indices)
        and np.array_equal(
            regular.reference_frame_indices,
            low_latency.reference_frame_indices,
        )
    )
    if not schedule_matches:
        validity_warnings.append(
            "reference playback schedules differed; tracking comparison was "
            "re-aligned by common prepared-reference frame"
        )
    return {
        "alignment": {
            "method": alignment_method,
            "common_rows": count,
            "common_reference_frames": count,
            "first_common_reference_frame": int(common_frames[0]),
            "last_common_reference_frame": int(common_frames[-1]),
            "regular_rows": int(len(regular.indices)),
            "low_latency_rows": int(len(low_latency.indices)),
            "regular_unique_reference_frames": int(
                len(np.unique(regular.reference_frame_indices))
            ),
            "low_latency_unique_reference_frames": int(
                len(np.unique(low_latency.reference_frame_indices))
            ),
            "same_reference_schedule_by_ordinal": schedule_matches,
            "regular_playback_diagnostics": regular_reference_match,
            "low_latency_playback_diagnostics": low_reference_match,
            "target_max_abs_difference_rad": (
                float(np.max(np.abs(finite_target_delta)))
                if finite_target_delta.size
                else None
            ),
            "same_target_within_1e-6": bool(
                finite_target_delta.size
                and np.max(np.abs(finite_target_delta)) <= 1e-6
            ),
            "ordinal_target_max_abs_difference_rad": (
                float(np.max(np.abs(finite_ordinal_target_delta)))
                if finite_ordinal_target_delta.size
                else None
            ),
            "ordinal_target_mismatch_rows_over_1e-6": int(
                len(ordinal_mismatch_rows)
            ),
            "first_ordinal_target_mismatch": (
                int(ordinal_mismatch_rows[0])
                if len(ordinal_mismatch_rows)
                else None
            ),
            "last_ordinal_target_mismatch": (
                int(ordinal_mismatch_rows[-1])
                if len(ordinal_mismatch_rows)
                else None
            ),
            "ordinal_target_maximum_detail": ordinal_maximum_detail,
        },
        "tracking": {
            "population": (
                "first occurrence of each common prepared-reference frame"
                if regular.has_prepared_reference_alignment
                and low_latency.has_prepared_reference_alignment
                else "common playback ordinals"
            ),
            "overall": {
                "regular_rmse_rad": regular_overall["rmse_rad"],
                "low_latency_rmse_rad": low_overall["rmse_rad"],
                "rmse_rad_low_minus_regular": _metric_delta(
                    regular_overall, low_overall, "rmse_rad"
                ),
                "regular_mae_rad": regular_overall["mae_rad"],
                "low_latency_mae_rad": low_overall["mae_rad"],
                "mae_rad_low_minus_regular": _metric_delta(
                    regular_overall, low_overall, "mae_rad"
                ),
                "regular_max_abs_rad": regular_overall["max_abs_rad"],
                "low_latency_max_abs_rad": low_overall["max_abs_rad"],
                "max_abs_rad_low_minus_regular": _metric_delta(
                    regular_overall, low_overall, "max_abs_rad"
                ),
            },
            "groups": group_deltas,
        },
        "action_smoothness": {
            "regular_step_delta_rms": regular_action["step_delta_rms"],
            "low_latency_step_delta_rms": low_action["step_delta_rms"],
            "step_delta_rms_low_minus_regular": _metric_delta(
                regular_action, low_action, "step_delta_rms"
            ),
            "regular_max_abs_step_delta": regular_action[
                "max_abs_step_delta"
            ],
            "low_latency_max_abs_step_delta": low_action[
                "max_abs_step_delta"
            ],
            "max_abs_step_delta_low_minus_regular": _metric_delta(
                regular_action, low_action, "max_abs_step_delta"
            ),
        },
        "runtime_validity": {
            "both_valid": bool(
                regular_validity["valid"] and low_validity["valid"]
            ),
            "regular_valid": regular_validity["valid"],
            "low_latency_valid": low_validity["valid"],
            "regular_finite": bool(
                regular.summary["signals"]["q_mj"]["all_finite"]
                and regular.summary["signals"]["raw_action"]["all_finite"]
            ),
            "low_latency_finite": bool(
                low_latency.summary["signals"]["q_mj"]["all_finite"]
                and low_latency.summary["signals"]["raw_action"]["all_finite"]
            ),
            "regular_protocol_timing_valid": regular_validity[
                "protocol_timing_valid"
            ],
            "low_latency_protocol_timing_valid": low_validity[
                "protocol_timing_valid"
            ],
            "regular_fallen": regular.summary["runtime"].get("fallen"),
            "low_latency_fallen": low_latency.summary["runtime"].get("fallen"),
            "warnings": validity_warnings,
        },
        "interpretation": (
            "This evaluates tracking of a robot qpos-derived reference in the "
            "recorded task scene. Because the reference came from a regular-policy "
            "trajectory, it is a diagnostic comparison and is biased toward regular. "
            "Object qpos metrics are physical indicators; visually confirm task success."
        ),
    }


def _joint_metric_rows(
    regular: RunAnalysis, low_latency: RunAnalysis
) -> list[dict[str, Any]]:
    regular_rows, low_rows, _, _ = _comparison_rows(regular, low_latency)
    regular_metrics = _error_metrics(regular.error_mj[regular_rows])["per_joint"]
    low_metrics = _error_metrics(low_latency.error_mj[low_rows])["per_joint"]
    regular_action = _smoothness(
        regular.action_mj[regular_rows], regular.time_s[regular_rows]
    ).get("per_joint_step_delta_rms", [None] * 29)
    low_action = _smoothness(
        low_latency.action_mj[low_rows], low_latency.time_s[low_rows]
    ).get("per_joint_step_delta_rms", [None] * 29)
    rows: list[dict[str, Any]] = []
    for index in range(29):
        first = regular_metrics[index]
        second = low_metrics[index]
        rows.append(
            {
                "joint_index_mj": index,
                "joint_name": MJ_JOINT_NAMES[index],
                "group": first["group"],
                "regular_rmse_rad": first["rmse_rad"],
                "low_latency_rmse_rad": second["rmse_rad"],
                "rmse_low_minus_regular_rad": (
                    second["rmse_rad"] - first["rmse_rad"]
                    if first["rmse_rad"] is not None
                    and second["rmse_rad"] is not None
                    else None
                ),
                "regular_mae_rad": first["mae_rad"],
                "low_latency_mae_rad": second["mae_rad"],
                "regular_max_abs_rad": first["max_abs_rad"],
                "low_latency_max_abs_rad": second["max_abs_rad"],
                "regular_action_step_rms_raw": regular_action[index],
                "low_latency_action_step_rms_raw": low_action[index],
            }
        )
    return rows


def _write_dict_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ComparisonError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_aligned_tracking(
    path: Path, regular: RunAnalysis, low_latency: RunAnalysis
) -> None:
    regular_rows, low_rows, common_frames, _ = _comparison_rows(
        regular, low_latency
    )
    header = [
        "playback_ordinal",
        "reference_frame_index",
        "regular_policy_seq",
        "low_latency_policy_seq",
        "regular_index",
        "low_latency_index",
        "regular_time_s",
        "low_latency_time_s",
    ]
    for name in MJ_JOINT_NAMES:
        header.extend(
            [
                f"{name}_regular_target",
                f"{name}_regular_q",
                f"{name}_regular_error",
                f"{name}_low_latency_target",
                f"{name}_low_latency_q",
                f"{name}_low_latency_error",
            ]
        )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for ordinal, (regular_row, low_row, reference_frame) in enumerate(
            zip(regular_rows, low_rows, common_frames)
        ):
            values: list[Any] = [
                ordinal,
                int(reference_frame),
                int(regular.reference_policy_seq[regular_row]),
                int(low_latency.reference_policy_seq[low_row]),
                int(regular.indices[regular_row]),
                int(low_latency.indices[low_row]),
                float(regular.time_s[regular_row]),
                float(low_latency.time_s[low_row]),
            ]
            for joint in range(29):
                values.extend(
                    [
                        float(regular.target_q_mj[regular_row, joint]),
                        float(regular.measured_q_mj[regular_row, joint]),
                        float(regular.error_mj[regular_row, joint]),
                        float(low_latency.target_q_mj[low_row, joint]),
                        float(low_latency.measured_q_mj[low_row, joint]),
                        float(low_latency.error_mj[low_row, joint]),
                    ]
                )
            writer.writerow(values)


def _write_plot(
    path: Path, regular: RunAnalysis, low_latency: RunAnalysis
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ComparisonError(
            "--plot requested, but matplotlib is not installed in this environment"
        ) from exc

    regular_rows, low_rows, common_frames, _ = _comparison_rows(
        regular, low_latency
    )
    time = (common_frames - common_frames[0]) * min(
        regular.summary["playback_alignment"]["evaluation_dt_s"],
        low_latency.summary["playback_alignment"]["evaluation_dt_s"],
    )
    regular_frame_rmse = np.sqrt(
        np.nanmean(np.square(regular.error_mj[regular_rows]), axis=1)
    )
    low_frame_rmse = np.sqrt(
        np.nanmean(np.square(low_latency.error_mj[low_rows]), axis=1)
    )
    regular_action_delta = np.sqrt(
        np.nanmean(
            np.square(np.diff(regular.action_mj[regular_rows], axis=0)), axis=1
        )
    )
    low_action_delta = np.sqrt(
        np.nanmean(
            np.square(np.diff(low_latency.action_mj[low_rows], axis=0)), axis=1
        )
    )
    regular_pair_metrics = _error_metrics(regular.error_mj[regular_rows])
    low_pair_metrics = _error_metrics(low_latency.error_mj[low_rows])

    figure, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)
    axes[0].plot(time, regular_frame_rmse, label="regular")
    axes[0].plot(time, low_frame_rmse, label="low_latency")
    axes[0].set_ylabel("joint tracking RMSE [rad]")
    axes[0].set_xlabel("evaluated playback time [s]")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    groups = list(JOINT_GROUPS)
    x = np.arange(len(groups))
    width = 0.38
    regular_groups = [
        regular_pair_metrics["groups"][name]["rmse_rad"]
        for name in groups
    ]
    low_groups = [
        low_pair_metrics["groups"][name]["rmse_rad"]
        for name in groups
    ]
    axes[1].bar(x - width / 2, regular_groups, width, label="regular")
    axes[1].bar(x + width / 2, low_groups, width, label="low_latency")
    axes[1].set_xticks(x, groups, rotation=15)
    axes[1].set_ylabel("group RMSE [rad]")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()

    axes[2].plot(time[1:], regular_action_delta, label="regular")
    axes[2].plot(time[1:], low_action_delta, label="low_latency")
    axes[2].set_ylabel("raw action step RMS")
    axes[2].set_xlabel("evaluated playback time [s]")
    axes[2].grid(alpha=0.25)
    axes[2].legend()
    figure.suptitle("qpos-track checkpoint comparison")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def infer_output_dir(regular_run: Path, low_latency_run: Path) -> Path:
    regular_name = regular_run.resolve().name
    low_name = low_latency_run.resolve().name
    regular_base = re.sub(r"_regular$", "", regular_name)
    low_base = re.sub(r"_low_latency$", "", low_name)
    recording = regular_base if regular_base == low_base else regular_base
    return DEFAULT_DATA_ROOT / f"{recording}_comparison"


def write_comparison(
    regular: RunAnalysis,
    low_latency: RunAnalysis,
    output_dir: Path,
    make_plot: bool = False,
) -> dict[str, Path]:
    output_dir = output_dir.expanduser().resolve()
    try:
        output_dir.relative_to(DEFAULT_DATA_ROOT.resolve())
    except ValueError as exc:
        raise ComparisonError(
            f"comparison output must be inside {DEFAULT_DATA_ROOT.resolve()}"
        ) from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = compare_analyses(regular, low_latency)
    report = {
        "schema_version": 2,
        "scope": "qpos-derived body-reference tracking in restored task scene",
        "regular": regular.summary,
        "low_latency": low_latency.summary,
        "comparison": comparison,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            _json_safe(report), ensure_ascii=False, indent=2, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    joint_path = output_dir / "per_joint_metrics.csv"
    _write_dict_csv(joint_path, _joint_metric_rows(regular, low_latency))
    aligned_path = output_dir / "aligned_tracking.csv"
    _write_aligned_tracking(aligned_path, regular, low_latency)
    paths = {
        "summary": summary_path,
        "per_joint_metrics": joint_path,
        "aligned_tracking": aligned_path,
    }
    if make_plot:
        plot_path = output_dir / "tracking_overview.png"
        _write_plot(plot_path, regular, low_latency)
        paths["plot"] = plot_path
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("regular_run", type=Path)
    parser.add_argument("low_latency_run", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "comparison directory below change_ckpt_track/data; default is "
            "<recording>_comparison"
        ),
    )
    parser.add_argument(
        "--warmup-s",
        type=float,
        help=(
            "seconds to exclude after motion_playing first becomes true; default "
            "reads manifest initialization.warmup_exclusion_s, otherwise 0.2"
        ),
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="also write tracking_overview.png (requires matplotlib)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        regular = analyze_run(args.regular_run, args.warmup_s)
        low_latency = analyze_run(args.low_latency_run, args.warmup_s)
        output_dir = args.output_dir or infer_output_dir(
            args.regular_run, args.low_latency_run
        )
        paths = write_comparison(
            regular, low_latency, output_dir, make_plot=args.plot
        )
    except ComparisonError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    comparison = compare_analyses(regular, low_latency)
    overall = comparison["tracking"]["overall"]
    print(
        f"regular common-frame tracking RMSE:    "
        f"{overall['regular_rmse_rad']:.6f} rad"
    )
    print(
        f"low-latency common-frame tracking RMSE: "
        f"{overall['low_latency_rmse_rad']:.6f} rad"
    )
    for label, path in paths.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
