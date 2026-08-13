#!/usr/bin/env python3
"""Build the decoder's startup history from a source recording.

The SONIC decoder consumes ten 50 Hz robot-state/action snapshots.  A fresh
deployment process normally zero-pads the missing snapshots.  This module
extracts the nine complete policy groups immediately before a selected start
offset and serialises them for the opt-in deploy variant in
``change_ckpt/source_history_deploy``.  The selected current group is retained
only for first-live-state validation and for its ``policy_last_action_in``;
the current q/dq/IMU state is still read from live MuJoCo DDS on the first
CONTROL tick.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np


HISTORY_FRAMES = 10
PREFILL_FRAMES = HISTORY_FRAMES - 1
POLICY_DT_S = 0.02
NUM_BODY_JOINTS = 29
PHASE_SEARCH_WINDOW_S = 0.02

# policy_parameters.hpp: MuJoCo/hardware index read for each IsaacLab index.
MUJOCO_TO_ISAACLAB = np.asarray(
    [
        0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
        16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
    ],
    dtype=np.int64,
)

# policy_parameters.hpp, in MuJoCo/hardware order.
DEFAULT_ANGLES = np.asarray(
    [
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        0.0, 0.0, 0.0,
        0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
        0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    ],
    dtype=np.float64,
)

# The 43-DoF XML inserts seven left-hand joints between the left and right arm.
BODY_QPOS_IDS = tuple(range(7, 29)) + tuple(range(36, 43))
BODY_QVEL_IDS = tuple(range(6, 28)) + tuple(range(35, 42))
ROBOT_QPOS_IDS_IN_RECEIVED_ORDER = (
    *BODY_QPOS_IDS,
    *range(29, 36),
    *range(43, 50),
)
ROBOT_QVEL_IDS_IN_RECEIVED_ORDER = (
    *BODY_QVEL_IDS,
    *range(28, 35),
    *range(42, 49),
)


class SourceHistoryError(ValueError):
    """The recording cannot safely provide the requested decoder history."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class _PolicyGroup:
    policy_seq: int
    first_row_index: int
    first_row: list[str]
    row_count: int = 1


@dataclass
class _SourceTable:
    rows: list[list[str]]
    qpos: np.ndarray
    qvel: np.ndarray
    control_time_s: np.ndarray


def _resolve_csv(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path = path / "data.csv"
    if not path.is_file():
        raise SourceHistoryError(f"data.csv not found: {path}")
    return path


def _as_int(row: Sequence[str], column: int, label: str, row_index: int) -> int:
    try:
        value = float(row[column])
    except (IndexError, TypeError, ValueError) as exc:
        raise SourceHistoryError(
            f"invalid {label} at CSV data row {row_index}"
        ) from exc
    if not math.isfinite(value) or not value.is_integer():
        raise SourceHistoryError(
            f"non-integer {label}={value!r} at CSV data row {row_index}"
        )
    return int(value)


def _indexed_columns(header: Sequence[str], kind: str) -> dict[int, int]:
    pattern = re.compile(rf"\[{kind}(\d+)\]$")
    result: dict[int, int] = {}
    for column, name in enumerate(header):
        match = pattern.search(name)
        if match:
            result[int(match.group(1))] = column
    if not result:
        raise SourceHistoryError(f"recording has no {kind} columns")
    actual = sorted(result)
    if actual != list(range(actual[-1] + 1)):
        raise SourceHistoryError(f"{kind} indices are not contiguous: {actual[:12]}...")
    return result


def _vector(row: Sequence[str], columns: Sequence[int], label: str) -> np.ndarray:
    try:
        values = np.asarray([float(row[column]) for column in columns], dtype=np.float64)
    except (IndexError, TypeError, ValueError) as exc:
        raise SourceHistoryError(f"invalid numeric values in {label}") from exc
    if not np.isfinite(values).all():
        raise SourceHistoryError(f"non-finite numeric values in {label}")
    return values


def _load_groups(
    csv_path: Path,
) -> tuple[list[str], list[_PolicyGroup], float, _SourceTable]:
    with csv_path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise SourceHistoryError(f"empty CSV: {csv_path}") from exc
        column = {name: index for index, name in enumerate(header)}
        for required in ("policy_seq", "policy_valid", "control_time_s"):
            if required not in column:
                raise SourceHistoryError(f"recording is missing {required!r}")

        groups: list[_PolicyGroup] = []
        rows: list[list[str]] = []
        previous_seq: int | None = None
        previous_row_time: float | None = None
        source_deltas: list[float] = []
        for row_index, row in enumerate(reader):
            if len(row) != len(header):
                raise SourceHistoryError(
                    f"CSV data row {row_index} has {len(row)} fields; expected {len(header)}"
                )
            rows.append(list(row))
            seq = _as_int(row, column["policy_seq"], "policy_seq", row_index)
            try:
                row_time = float(row[column["control_time_s"]])
            except ValueError as exc:
                raise SourceHistoryError(
                    f"invalid control_time_s at CSV data row {row_index}"
                ) from exc
            if previous_row_time is not None:
                delta = row_time - previous_row_time
                if delta > 0:
                    source_deltas.append(delta)
            previous_row_time = row_time

            if seq != previous_seq:
                if groups and seq != groups[-1].policy_seq + 1:
                    raise SourceHistoryError(
                        "policy_seq is not contiguous at data row "
                        f"{row_index}: {groups[-1].policy_seq} -> {seq}"
                    )
                groups.append(_PolicyGroup(seq, row_index, list(row)))
                previous_seq = seq
            else:
                groups[-1].row_count += 1

    if len(groups) < HISTORY_FRAMES + 1:
        raise SourceHistoryError(
            f"recording has only {len(groups)} policy groups; at least "
            f"{HISTORY_FRAMES + 1} are required"
        )
    if not source_deltas:
        raise SourceHistoryError("could not infer the source row period")
    source_dt = float(np.median(np.asarray(source_deltas, dtype=np.float64)))
    if not math.isfinite(source_dt) or source_dt <= 0:
        raise SourceHistoryError(f"invalid source row period: {source_dt!r}")
    qpos_columns = _indexed_columns(header, "qpos")
    qvel_columns = _indexed_columns(header, "qvel")
    table = _SourceTable(
        rows=rows,
        qpos=np.asarray(
            [[float(row[qpos_columns[index]]) for index in range(len(qpos_columns))]
             for row in rows],
            dtype=np.float64,
        ),
        qvel=np.asarray(
            [[float(row[qvel_columns[index]]) for index in range(len(qvel_columns))]
             for row in rows],
            dtype=np.float64,
        ),
        control_time_s=np.asarray(
            [float(row[column["control_time_s"]]) for row in rows],
            dtype=np.float64,
        ),
    )
    if not np.isfinite(table.qpos).all() or not np.isfinite(table.qvel).all():
        raise SourceHistoryError("source qpos/qvel contains non-finite values")
    return header, groups, source_dt, table


def _normalised_quaternion(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise SourceHistoryError("cannot interpolate an invalid source quaternion")
    return value / norm


def _slerp(q0: np.ndarray, q1: np.ndarray, fraction: float) -> np.ndarray:
    q0 = _normalised_quaternion(q0)
    q1 = _normalised_quaternion(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return _normalised_quaternion(q0 + fraction * (q1 - q0))
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    return (
        math.sin((1.0 - fraction) * theta) / sin_theta * q0
        + math.sin(fraction * theta) / sin_theta * q1
    )


def _interpolate_vector(
    values: np.ndarray, times: np.ndarray, sample_time: float
) -> np.ndarray:
    right = int(np.searchsorted(times, sample_time, side="right"))
    if right <= 0:
        return values[0].copy()
    if right >= len(times):
        return values[-1].copy()
    left = right - 1
    width = float(times[right] - times[left])
    fraction = 0.0 if width <= 0.0 else (sample_time - times[left]) / width
    return values[left] + fraction * (values[right] - values[left])


def _interpolate_quaternion(
    values: np.ndarray, times: np.ndarray, sample_time: float
) -> np.ndarray:
    right = int(np.searchsorted(times, sample_time, side="right"))
    if right <= 0:
        return _normalised_quaternion(values[0])
    if right >= len(times):
        return _normalised_quaternion(values[-1])
    left = right - 1
    width = float(times[right] - times[left])
    fraction = 0.0 if width <= 0.0 else (sample_time - times[left]) / width
    return _slerp(values[left], values[right], float(fraction))


def _match_received_state(
    group: _PolicyGroup,
    *,
    table: _SourceTable,
    received_columns: Sequence[int],
    source_dt: float,
) -> dict[str, Any]:
    received = _vector(group.first_row, received_columns, "policy_received_dof_pos")
    if received.size != len(ROBOT_QPOS_IDS_IN_RECEIVED_ORDER):
        raise SourceHistoryError(
            "policy_received_dof_pos must contain 43 body/hand joints"
        )
    received_body = received[:NUM_BODY_JOINTS]
    robot_qpos = table.qpos[:, ROBOT_QPOS_IDS_IN_RECEIVED_ORDER]
    robot_qvel = table.qvel[:, ROBOT_QVEL_IDS_IN_RECEIVED_ORDER]
    search_rows = max(1, int(math.ceil(PHASE_SEARCH_WINDOW_S / source_dt)))
    search_start = max(0, group.first_row_index - search_rows)
    search_end = group.first_row_index
    candidates: list[tuple[float, float, int, float]] = []
    half_source_dt = source_dt / 2.0
    for row_index in range(search_start, search_end + 1):
        difference = received - robot_qpos[row_index]
        velocity = robot_qvel[row_index]
        denominator = float(np.dot(velocity, velocity))
        alpha = (
            float(np.dot(velocity, difference) / denominator)
            if denominator > 1e-12
            else 0.0
        )
        alpha = float(np.clip(alpha, -half_source_dt, half_source_dt))
        residual = robot_qpos[row_index] + alpha * velocity - received
        candidates.append(
            (
                float(np.sqrt(np.mean(residual * residual))),
                float(np.max(np.abs(residual))),
                row_index,
                alpha,
            )
        )
    rmse, max_abs, matched_row, alpha = min(candidates)
    sample_time = float(table.control_time_s[matched_row] + alpha)
    boundary_time = float(table.control_time_s[group.first_row_index])
    if max_abs > 1e-4:
        raise SourceHistoryError(
            f"policy_received_dof_pos for seq {group.policy_seq} cannot be "
            f"matched to qpos (best max error {max_abs:g})"
        )
    return {
        "received_body_q_mujoco": received_body,
        "received_robot_q": received,
        "matched_row_index": matched_row,
        "interpolation_offset_s": alpha,
        "matched_control_time_s": sample_time,
        "policy_boundary_delay_s": boundary_time - sample_time,
        "received_q_fit_rmse": rmse,
        "received_q_fit_max_abs": max_abs,
    }


def _state_from_received_match(
    row: Sequence[str],
    match: dict[str, Any],
    *,
    table: _SourceTable,
    last_action_columns: Sequence[int],
) -> dict[str, list[float]]:
    sample_time = float(match["matched_control_time_s"])
    q_mujoco = np.asarray(match["received_body_q_mujoco"], dtype=np.float64)
    dq_mujoco = _interpolate_vector(
        table.qvel[:, BODY_QVEL_IDS], table.control_time_s, sample_time
    )
    base_quat = _interpolate_quaternion(
        table.qpos[:, 3:7], table.control_time_s, sample_time
    )
    base_ang_vel = _interpolate_vector(
        table.qvel[:, 3:6], table.control_time_s, sample_time
    )
    last_action = _vector(row, last_action_columns, "policy_last_action_in")
    return {
        "base_quat": base_quat.tolist(),
        "base_ang_vel": base_ang_vel.tolist(),
        "body_q": (
            q_mujoco[MUJOCO_TO_ISAACLAB]
            - DEFAULT_ANGLES[MUJOCO_TO_ISAACLAB]
        ).tolist(),
        "body_dq": dq_mujoco[MUJOCO_TO_ISAACLAB].tolist(),
        "last_action": last_action.tolist(),
    }


def build_source_history_prefill(
    recording: str | Path,
    *,
    start_policy_offset: int,
) -> dict[str, Any]:
    """Return a validated JSON-serialisable prefill payload.

    ``start_policy_offset`` has exactly the launcher's existing zero-based
    unique-policy-group semantics.  The payload contains nine prior groups and
    one current validation record; it never duplicates the current state in
    the logger ring.
    """

    if start_policy_offset < PREFILL_FRAMES:
        raise SourceHistoryError(
            f"start policy offset {start_policy_offset} has fewer than "
            f"{PREFILL_FRAMES} preceding groups"
        )
    csv_path = _resolve_csv(recording)
    source_csv_sha256 = _sha256(csv_path)
    header, groups, source_dt, table = _load_groups(csv_path)
    if _sha256(csv_path) != source_csv_sha256:
        raise SourceHistoryError(
            f"source CSV changed while its history was being read: {csv_path}"
        )
    if start_policy_offset >= len(groups):
        raise SourceHistoryError(
            f"start policy offset {start_policy_offset} is outside "
            f"0..{len(groups) - 1}"
        )

    expected_rows = int(round(POLICY_DT_S / source_dt))
    if expected_rows <= 0 or not math.isclose(
        expected_rows * source_dt, POLICY_DT_S, rel_tol=0.0, abs_tol=1e-7
    ):
        raise SourceHistoryError(
            f"source dt {source_dt:g}s does not divide the 50 Hz policy period"
        )
    previous_groups = groups[
        start_policy_offset - PREFILL_FRAMES : start_policy_offset
    ]
    current_group = groups[start_policy_offset]
    selected_groups = [*previous_groups, current_group]

    column = {name: index for index, name in enumerate(header)}
    invalid_groups = [
        group.policy_seq
        for group in selected_groups
        if _as_int(
            group.first_row,
            column["policy_valid"],
            "policy_valid",
            group.first_row_index,
        )
        == 0
    ]
    if invalid_groups:
        raise SourceHistoryError(
            "selected source-history groups have policy_valid=0: "
            + ", ".join(str(value) for value in invalid_groups)
        )
    try:
        last_action_columns = [
            column[f"policy_last_action_in[{index}]"]
            for index in range(NUM_BODY_JOINTS)
        ]
        received_columns = [
            column[f"policy_received_dof_pos[{index}]"]
            for index in range(len(ROBOT_QPOS_IDS_IN_RECEIVED_ORDER))
        ]
    except KeyError as exc:
        raise SourceHistoryError(f"recording is missing {exc.args[0]!r}") from exc

    matches = [
        _match_received_state(
            group,
            table=table,
            received_columns=received_columns,
            source_dt=source_dt,
        )
        for group in selected_groups
    ]

    entries: list[dict[str, Any]] = []
    for offset, (group, match) in enumerate(
        zip(previous_groups, matches[:-1], strict=True),
        start=start_policy_offset - PREFILL_FRAMES,
    ):
        entry: dict[str, Any] = {
            "policy_offset": offset,
            "policy_seq": group.policy_seq,
            "source_row_index": group.first_row_index,
            "source_group_row_count": group.row_count,
            "matched_source_row_index": match["matched_row_index"],
            "matched_control_time_s": match["matched_control_time_s"],
            "interpolation_offset_s": match["interpolation_offset_s"],
            "policy_boundary_delay_s": match["policy_boundary_delay_s"],
            "received_q_fit_rmse": match["received_q_fit_rmse"],
            "received_q_fit_max_abs": match["received_q_fit_max_abs"],
        }
        entry.update(
            _state_from_received_match(
                group.first_row,
                match,
                table=table,
                last_action_columns=last_action_columns,
            )
        )
        entries.append(entry)

    current_match = matches[-1]
    current: dict[str, Any] = {
        "policy_offset": start_policy_offset,
        "policy_seq": current_group.policy_seq,
        "source_row_index": current_group.first_row_index,
        "source_group_row_count": current_group.row_count,
        "matched_source_row_index": current_match["matched_row_index"],
        "matched_control_time_s": current_match["matched_control_time_s"],
        "interpolation_offset_s": current_match["interpolation_offset_s"],
        "policy_boundary_delay_s": current_match["policy_boundary_delay_s"],
        "received_q_fit_rmse": current_match["received_q_fit_rmse"],
        "received_q_fit_max_abs": current_match["received_q_fit_max_abs"],
    }
    current.update(
        _state_from_received_match(
            current_group.first_row,
            current_match,
            table=table,
            last_action_columns=last_action_columns,
        )
    )
    current["sim_initial_qpos"] = _interpolate_vector(
        table.qpos, table.control_time_s, current_match["matched_control_time_s"]
    ).tolist()
    current["sim_initial_qvel"] = _interpolate_vector(
        table.qvel, table.control_time_s, current_match["matched_control_time_s"]
    ).tolist()
    current["sim_initial_qpos"][3:7] = _interpolate_quaternion(
        table.qpos[:, 3:7],
        table.control_time_s,
        current_match["matched_control_time_s"],
    ).tolist()
    received_robot = np.asarray(
        current_match["received_robot_q"], dtype=np.float64
    )
    for received_index, qpos_index in enumerate(ROBOT_QPOS_IDS_IN_RECEIVED_ORDER):
        current["sim_initial_qpos"][qpos_index] = float(received_robot[received_index])

    action_link_error: float | None = None
    raw_names = [f"policy_raw_action_out[{index}]" for index in range(NUM_BODY_JOINTS)]
    if all(name in column for name in raw_names):
        previous_raw = _vector(
            previous_groups[-1].first_row,
            [column[name] for name in raw_names],
            "previous policy_raw_action_out",
        )
        action_link_error = float(
            np.max(np.abs(previous_raw - np.asarray(current["last_action"])))
        )
        if action_link_error > 1e-6:
            raise SourceHistoryError(
                "current policy_last_action_in does not match the previous "
                f"policy_raw_action_out (max error {action_link_error:g})"
            )

    return {
        "format": "g1_decoder_source_history_prefill",
        "version": 1,
        "source_csv": str(csv_path),
        "source_csv_sha256": source_csv_sha256,
        "start_policy_offset": start_policy_offset,
        "history_entry_count": len(entries),
        "history_order": "oldest_to_newest",
        "entries": entries,
        "current": current,
        "validation": {
            "source_dt_s": source_dt,
            "policy_dt_s": POLICY_DT_S,
            "expected_rows_per_policy_group": expected_rows,
            "selected_group_row_counts": [group.row_count for group in selected_groups],
            "policy_seq_contiguous": True,
            "received_state_phase_fit_max_abs": max(
                float(match["received_q_fit_max_abs"]) for match in matches
            ),
            "policy_boundary_delay_s": [
                float(match["policy_boundary_delay_s"]) for match in matches
            ],
            "current_last_action_vs_previous_raw_action_max_abs": action_link_error,
        },
        "field_sources": {
            "state_sample": (
                "policy_received_dof_pos phase-matched to the preceding 20 ms "
                "qpos/qvel timeline; base/dq interpolated at the matched time"
            ),
            "base_quat": "SLERP of qpos[3:7] at the phase-matched time, wxyz",
            "base_ang_vel": "linear interpolation of qvel[3:6] at matched time",
            "body_q": (
                "policy_received_dof_pos[0:29] (measured q); "
                "MuJoCo->IsaacLab reorder; default_angles subtracted"
            ),
            "body_dq": (
                "qvel ids 6..27 and 35..41 interpolated at matched time; "
                "MuJoCo->IsaacLab reorder"
            ),
            "last_action": (
                "policy_last_action_in[0:29], already unscaled IsaacLab order"
            ),
            "current_state": (
                "live DDS state on first CONTROL tick after MuJoCo is initialized "
                "to the phase-matched source state; strictly validated"
            ),
        },
    }


def write_source_history_prefill(payload: dict[str, Any], path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    return path


def summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_csv": payload["source_csv"],
        "source_csv_sha256": payload["source_csv_sha256"],
        "start_policy_offset": payload["start_policy_offset"],
        "current_policy_seq": payload["current"]["policy_seq"],
        "current_source_row_index": payload["current"]["source_row_index"],
        "sim_timeline_start_row_index": payload["current"][
            "matched_source_row_index"
        ],
        "current_matched_source_row_index": payload["current"][
            "matched_source_row_index"
        ],
        "current_matched_control_time_s": payload["current"][
            "matched_control_time_s"
        ],
        "history_policy_seq": [entry["policy_seq"] for entry in payload["entries"]],
        "history_source_row_indices": [
            entry["source_row_index"] for entry in payload["entries"]
        ],
        "history_matched_source_row_indices": [
            entry["matched_source_row_index"] for entry in payload["entries"]
        ],
        "history_entry_count": payload["history_entry_count"],
        "validation": payload["validation"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", help="recording directory or data.csv")
    parser.add_argument(
        "--start-policy-offset",
        type=int,
        default=10,
        help="raw CSV policy-group offset (not the launcher public offset)",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = build_source_history_prefill(
        args.recording, start_policy_offset=args.start_policy_offset
    )
    if args.output is not None:
        output = write_source_history_prefill(payload, args.output)
        print(f"wrote {output}")
    print(json.dumps(summary(payload), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
