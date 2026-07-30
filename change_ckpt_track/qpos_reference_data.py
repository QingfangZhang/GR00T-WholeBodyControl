"""Build a 50 Hz SONIC deploy reference from recorded MuJoCo state.

The source ``data.csv`` is sampled once per consecutive ``policy_seq`` group:
the first row of each group is retained.  Body joints are selected by name and
reordered into the 29-DOF G1 IsaacLab order; task joints and finger state in
MuJoCo ``qpos`` are deliberately excluded.  The recorded hand command fields
(``left_hand_q`` and ``right_hand_q``) are retained separately for the task
publisher.

By default, an incomplete first and/or last policy group is removed when its
row count is less than half the median group size.  Every retained frame keeps
the original policy sequence and CSV row numbers so the task simulator can be
initialized at exactly the same source boundary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


FRAME_RATE_HZ = 50.0
FRAME_DT_S = 1.0 / FRAME_RATE_HZ
NUM_BODY_JOINTS = 29
NUM_HAND_JOINTS = 7

# Kept locally rather than importing joint_utils: that module imports torch and
# IsaacLab-facing code, while this converter is intentionally NumPy-only.
G1_ISAACLAB_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)

G1_MUJOCO_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

_ROOT_POSITION_COLUMNS = (
    "qpos:pelvis.floating_base_joint.x[qpos0]",
    "qpos:pelvis.floating_base_joint.y[qpos1]",
    "qpos:pelvis.floating_base_joint.z[qpos2]",
)
_ROOT_QUATERNION_COLUMNS = (
    "qpos:pelvis.floating_base_joint.qw[qpos3]",
    "qpos:pelvis.floating_base_joint.qx[qpos4]",
    "qpos:pelvis.floating_base_joint.qy[qpos5]",
    "qpos:pelvis.floating_base_joint.qz[qpos6]",
)

_QPOS_STATE_RE = re.compile(r"\[qpos(?P<index>\d+)\]$")
_QVEL_STATE_RE = re.compile(r"\[qvel(?P<index>\d+)\]$")
_QPOS_JOINT_RE = re.compile(
    r"^qpos:(?P<body>.+)\.(?P<joint>[^.]+)\.angle\[qpos(?P<index>\d+)\]$"
)
_QVEL_JOINT_RE = re.compile(
    r"^qvel:(?P<body>.+)\.(?P<joint>[^.]+)\.omega\[qvel(?P<index>\d+)\]$"
)


class QposReferenceError(ValueError):
    """Raised when a recording cannot safely become a deploy reference."""


@dataclass(frozen=True)
class DroppedEdgeGroup:
    """Description of one incomplete policy group removed at an edge."""

    side: str
    policy_seq: int
    row_count: int
    source_row_index: int
    source_csv_row_number: int


@dataclass(frozen=True)
class _CsvLayout:
    header: tuple[str, ...]
    policy_seq_index: int
    control_time_index: int
    policy_valid_index: int | None
    joint_qpos_indices: tuple[int, ...]
    joint_qvel_indices: tuple[int, ...]
    root_position_indices: tuple[int, ...]
    root_quaternion_indices: tuple[int, ...]
    left_hand_indices: tuple[int, ...]
    right_hand_indices: tuple[int, ...]
    qpos_width: int
    qvel_width: int


@dataclass
class _PolicyGroup:
    policy_seq: int
    control_time_s: float
    source_row_index: int
    source_csv_row_number: int
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    root_pos: np.ndarray
    root_quat_wxyz: np.ndarray
    left_hand_target: np.ndarray
    right_hand_target: np.ndarray
    row_count: int = 1
    max_joint_pos_change_from_first: float = 0.0
    max_joint_vel_change_from_first: float = 0.0
    max_root_pos_change_from_first: float = 0.0
    max_hand_change_from_first: float = 0.0


@dataclass(frozen=True)
class QposReferenceSequence:
    """Validated policy-rate state and hand commands from one recording."""

    csv_path: Path
    joint_names: tuple[str, ...]
    policy_seq: np.ndarray
    control_time_s: np.ndarray
    source_row_indices: np.ndarray
    source_csv_row_numbers: np.ndarray
    group_row_counts: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    root_pos: np.ndarray
    root_quat_wxyz: np.ndarray
    left_hand_target: np.ndarray
    right_hand_target: np.ndarray
    joint_qpos_columns: tuple[str, ...]
    joint_qvel_columns: tuple[str, ...]
    root_position_columns: tuple[str, ...]
    root_quaternion_columns: tuple[str, ...]
    left_hand_columns: tuple[str, ...]
    right_hand_columns: tuple[str, ...]
    source_header_width: int
    source_qpos_width: int
    source_qvel_width: int
    source_total_rows: int
    source_valid_rows: int
    source_invalid_rows: int
    source_group_count: int
    median_group_rows: float
    edge_fraction: float
    dropped_edge_groups: tuple[DroppedEdgeGroup, ...]
    frame_offset: int = 0
    max_intragroup_joint_pos_change: float = 0.0
    max_intragroup_joint_vel_change: float = 0.0
    max_intragroup_root_pos_change: float = 0.0
    max_intragroup_hand_change: float = 0.0
    quaternion_max_norm_error_before_normalization: float = 0.0
    quaternion_sign_flips_fixed: int = 0

    def __post_init__(self) -> None:
        if self.joint_names != G1_ISAACLAB_JOINT_NAMES:
            raise QposReferenceError(
                "joint_names must exactly match the 29-DOF G1 IsaacLab order"
            )
        arrays = {
            "policy_seq": (self.policy_seq, np.int64, (None,)),
            "control_time_s": (self.control_time_s, np.float64, (None,)),
            "source_row_indices": (self.source_row_indices, np.int64, (None,)),
            "source_csv_row_numbers": (
                self.source_csv_row_numbers,
                np.int64,
                (None,),
            ),
            "group_row_counts": (self.group_row_counts, np.int32, (None,)),
            "joint_pos": (
                self.joint_pos,
                np.float32,
                (None, NUM_BODY_JOINTS),
            ),
            "joint_vel": (
                self.joint_vel,
                np.float32,
                (None, NUM_BODY_JOINTS),
            ),
            "root_pos": (self.root_pos, np.float32, (None, 3)),
            "root_quat_wxyz": (self.root_quat_wxyz, np.float32, (None, 4)),
            "left_hand_target": (
                self.left_hand_target,
                np.float32,
                (None, NUM_HAND_JOINTS),
            ),
            "right_hand_target": (
                self.right_hand_target,
                np.float32,
                (None, NUM_HAND_JOINTS),
            ),
        }
        frame_count: int | None = None
        for name, (raw, dtype, expected_shape) in arrays.items():
            value = np.ascontiguousarray(raw, dtype=dtype)
            if value.ndim != len(expected_shape):
                raise QposReferenceError(
                    f"{name} has shape {value.shape}; expected {expected_shape}"
                )
            for actual, expected in zip(value.shape, expected_shape, strict=True):
                if expected is not None and actual != expected:
                    raise QposReferenceError(
                        f"{name} has shape {value.shape}; expected {expected_shape}"
                    )
            if frame_count is None:
                frame_count = int(value.shape[0])
            elif value.shape[0] != frame_count:
                raise QposReferenceError(
                    f"{name} has {value.shape[0]} frames; expected {frame_count}"
                )
            if np.issubdtype(value.dtype, np.floating) and not np.all(
                np.isfinite(value)
            ):
                raise QposReferenceError(f"{name} contains non-finite values")
            value.setflags(write=False)
            object.__setattr__(self, name, value)

        if not frame_count:
            raise QposReferenceError("reference sequence is empty")
        if frame_count > 1 and np.any(np.diff(self.policy_seq) != 1):
            raise QposReferenceError("policy_seq must be consecutive with step one")
        if frame_count > 1 and np.any(np.diff(self.control_time_s) <= 0.0):
            raise QposReferenceError("control_time_s must be strictly increasing")
        if np.any(self.source_row_indices < 0):
            raise QposReferenceError("source_row_indices must be non-negative")
        if np.any(self.source_csv_row_numbers != self.source_row_indices + 2):
            raise QposReferenceError(
                "source_csv_row_numbers must equal source_row_indices + 2 "
                "(one header row and one-based CSV numbering)"
            )
        if np.any(self.group_row_counts <= 0):
            raise QposReferenceError("group_row_counts must all be positive")
        quaternion_norms = np.linalg.norm(self.root_quat_wxyz, axis=1)
        if np.max(np.abs(quaternion_norms - 1.0)) > 2e-5:
            raise QposReferenceError("root_quat_wxyz is not unit-normalized")

        column_sets = {
            "joint_qpos_columns": (
                self.joint_qpos_columns,
                NUM_BODY_JOINTS,
            ),
            "joint_qvel_columns": (
                self.joint_qvel_columns,
                NUM_BODY_JOINTS,
            ),
            "root_position_columns": (self.root_position_columns, 3),
            "root_quaternion_columns": (self.root_quaternion_columns, 4),
            "left_hand_columns": (self.left_hand_columns, NUM_HAND_JOINTS),
            "right_hand_columns": (self.right_hand_columns, NUM_HAND_JOINTS),
        }
        for name, (columns, expected) in column_sets.items():
            if len(columns) != expected or len(set(columns)) != expected:
                raise QposReferenceError(
                    f"{name} must contain {expected} unique column names"
                )

    @property
    def num_frames(self) -> int:
        return int(self.policy_seq.shape[0])

    @property
    def duration_s(self) -> float:
        """Playback duration when every retained frame is held for one tick."""

        return self.num_frames * FRAME_DT_S

    @property
    def sample_span_s(self) -> float:
        """Time from the first sample instant to the final sample instant."""

        return max(0, self.num_frames - 1) * FRAME_DT_S

    @property
    def start_policy_seq(self) -> int:
        return int(self.policy_seq[0])

    @property
    def start_source_row_index(self) -> int:
        return int(self.source_row_indices[0])

    @property
    def start_source_csv_row_number(self) -> int:
        return int(self.source_csv_row_numbers[0])

    # Singular aliases are convenient for callers written against an earlier
    # draft of this module.  The values remain arrays.
    @property
    def source_row_index(self) -> np.ndarray:
        return self.source_row_indices

    @property
    def source_csv_row_number(self) -> np.ndarray:
        return self.source_csv_row_numbers

    @property
    def left_hand_joints(self) -> np.ndarray:
        return self.left_hand_target

    @property
    def right_hand_joints(self) -> np.ndarray:
        return self.right_hand_target

    def slice(
        self, start: int = 0, count: int | None = None
    ) -> "QposReferenceSequence":
        """Return a frame slice; ``start`` is relative to this sequence."""

        if start < 0 or start >= self.num_frames:
            raise QposReferenceError(
                f"start offset {start} is outside [0, {self.num_frames - 1}]"
            )
        if count is not None and count <= 0:
            raise QposReferenceError("slice count must be positive")
        stop = self.num_frames if count is None else min(
            self.num_frames, start + count
        )
        array_names = (
            "policy_seq",
            "control_time_s",
            "source_row_indices",
            "source_csv_row_numbers",
            "group_row_counts",
            "joint_pos",
            "joint_vel",
            "root_pos",
            "root_quat_wxyz",
            "left_hand_target",
            "right_hand_target",
        )
        changes = {name: getattr(self, name)[start:stop] for name in array_names}
        changes["frame_offset"] = self.frame_offset + start
        return replace(self, **changes)


def _resolve_csv_path(path: str | Path) -> Path:
    source = Path(path).expanduser()
    if source.is_dir():
        source = source / "data.csv"
    if not source.is_file():
        raise QposReferenceError(f"recording CSV does not exist: {source}")
    return source.resolve()


def _finite_float(value: str, *, column: str, row_number: int) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise QposReferenceError(
            f"invalid numeric value in {column!r} at CSV row {row_number}: "
            f"{value!r}"
        ) from exc
    if not math.isfinite(result):
        raise QposReferenceError(
            f"non-finite value in {column!r} at CSV row {row_number}: "
            f"{value!r}"
        )
    return result


def _integer_value(value: str, *, column: str, row_number: int) -> int:
    number = _finite_float(value, column=column, row_number=row_number)
    rounded = round(number)
    if abs(number - rounded) > 1e-9:
        raise QposReferenceError(
            f"{column!r} must be integer-valued at CSV row {row_number}: "
            f"{value!r}"
        )
    return int(rounded)


def _state_width(header: Sequence[str], kind: str) -> int:
    pattern = _QPOS_STATE_RE if kind == "qpos" else _QVEL_STATE_RE
    found: dict[int, str] = {}
    for name in header:
        match = pattern.search(name)
        if match is None:
            continue
        index = int(match.group("index"))
        if index in found:
            raise QposReferenceError(
                f"duplicate {kind} state index {index}: "
                f"{found[index]!r} and {name!r}"
            )
        found[index] = name
    if not found:
        raise QposReferenceError(f"CSV contains no indexed {kind} columns")
    expected = set(range(max(found) + 1))
    missing = sorted(expected.difference(found))
    if missing:
        raise QposReferenceError(
            f"{kind} indexed columns are not contiguous; missing indices "
            f"{missing[:10]}"
        )
    return max(found) + 1


def _indexed_columns(
    header: Sequence[str], stem: str, count: int
) -> tuple[int, ...]:
    pattern = re.compile(rf"^{re.escape(stem)}\[(?P<index>\d+)\]$")
    found: dict[int, int] = {}
    for column, name in enumerate(header):
        match = pattern.match(name)
        if match is None:
            continue
        index = int(match.group("index"))
        if index in found:
            raise QposReferenceError(
                f"duplicate {stem}[{index}] columns at {found[index]} and {column}"
            )
        found[index] = column
    expected = set(range(count))
    if set(found) != expected:
        missing = sorted(expected.difference(found))
        extra = sorted(set(found).difference(expected))
        raise QposReferenceError(
            f"{stem} columns must be exactly [0:{count}]; "
            f"missing={missing}, extra={extra}"
        )
    return tuple(found[index] for index in range(count))


def _joint_columns(
    header: Sequence[str], pattern: re.Pattern[str], kind: str
) -> tuple[int, ...]:
    found: dict[str, int] = {}
    for column, name in enumerate(header):
        match = pattern.match(name)
        if match is None:
            continue
        joint_name = match.group("joint")
        if joint_name in found:
            raise QposReferenceError(
                f"duplicate {kind} mapping for joint {joint_name!r}: "
                f"{header[found[joint_name]]!r} and {name!r}"
            )
        found[joint_name] = column
    missing = [name for name in G1_ISAACLAB_JOINT_NAMES if name not in found]
    if missing:
        raise QposReferenceError(
            f"CSV is missing {kind} columns for G1 joints: {missing}"
        )
    columns = tuple(found[name] for name in G1_ISAACLAB_JOINT_NAMES)
    if len(set(columns)) != NUM_BODY_JOINTS:
        raise QposReferenceError(f"{kind} joint-name mapping is not one-to-one")
    return columns


def _build_layout(header: Sequence[str]) -> _CsvLayout:
    if not header:
        raise QposReferenceError("CSV header is empty")
    duplicates = sorted(
        name for name in set(header) if header.count(name) > 1
    )
    if duplicates:
        raise QposReferenceError(
            f"CSV header contains duplicate names: {duplicates[:10]}"
        )
    lookup = {name: index for index, name in enumerate(header)}
    missing_metadata = [
        name for name in ("policy_seq", "control_time_s") if name not in lookup
    ]
    if missing_metadata:
        raise QposReferenceError(
            f"CSV is missing required columns: {missing_metadata}"
        )
    missing_root = [
        name
        for name in (*_ROOT_POSITION_COLUMNS, *_ROOT_QUATERNION_COLUMNS)
        if name not in lookup
    ]
    if missing_root:
        raise QposReferenceError(
            f"CSV is missing pelvis qpos columns: {missing_root}"
        )
    return _CsvLayout(
        header=tuple(header),
        policy_seq_index=lookup["policy_seq"],
        control_time_index=lookup["control_time_s"],
        policy_valid_index=lookup.get("policy_valid"),
        joint_qpos_indices=_joint_columns(
            header, _QPOS_JOINT_RE, "qpos"
        ),
        joint_qvel_indices=_joint_columns(
            header, _QVEL_JOINT_RE, "qvel"
        ),
        root_position_indices=tuple(lookup[name] for name in _ROOT_POSITION_COLUMNS),
        root_quaternion_indices=tuple(
            lookup[name] for name in _ROOT_QUATERNION_COLUMNS
        ),
        left_hand_indices=_indexed_columns(
            header, "left_hand_q", NUM_HAND_JOINTS
        ),
        right_hand_indices=_indexed_columns(
            header, "right_hand_q", NUM_HAND_JOINTS
        ),
        qpos_width=_state_width(header, "qpos"),
        qvel_width=_state_width(header, "qvel"),
    )


def _values(
    row: Sequence[str],
    indices: Iterable[int],
    layout: _CsvLayout,
    row_number: int,
) -> np.ndarray:
    return np.asarray(
        [
            _finite_float(
                row[column],
                column=layout.header[column],
                row_number=row_number,
            )
            for column in indices
        ],
        dtype=np.float64,
    )


def _normalize_quaternion(
    quaternion: np.ndarray,
    *,
    tolerance: float,
    row_number: int,
) -> tuple[np.ndarray, float]:
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm < 1e-8:
        raise QposReferenceError(
            f"invalid pelvis quaternion at CSV row {row_number}: norm={norm}"
        )
    error = abs(norm - 1.0)
    if error > tolerance:
        raise QposReferenceError(
            f"pelvis quaternion norm error {error:.6g} at CSV row "
            f"{row_number} exceeds tolerance {tolerance:.6g}"
        )
    return quaternion / norm, error


def _continuous_quaternion_sign(
    quaternions: np.ndarray,
) -> tuple[np.ndarray, int]:
    result = np.array(quaternions, dtype=np.float64, copy=True)
    flips = 0
    for index in range(1, result.shape[0]):
        if float(np.dot(result[index - 1], result[index])) < 0.0:
            result[index] *= -1.0
            flips += 1
    return result, flips


def _read_groups(
    csv_path: Path,
    *,
    quaternion_norm_tolerance: float,
) -> tuple[_CsvLayout, list[_PolicyGroup], dict[str, float | int]]:
    groups: list[_PolicyGroup] = []
    total_rows = 0
    valid_rows = 0
    invalid_rows = 0
    max_quaternion_norm_error = 0.0
    previous_time: float | None = None

    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise QposReferenceError(f"empty CSV: {csv_path}") from exc
        layout = _build_layout(header)

        for source_row_index, row in enumerate(reader):
            row_number = source_row_index + 2
            total_rows += 1
            if len(row) != len(header):
                raise QposReferenceError(
                    f"CSV row {row_number} has {len(row)} fields; "
                    f"expected {len(header)}"
                )
            policy_seq = _integer_value(
                row[layout.policy_seq_index],
                column="policy_seq",
                row_number=row_number,
            )
            control_time = _finite_float(
                row[layout.control_time_index],
                column="control_time_s",
                row_number=row_number,
            )
            if layout.policy_valid_index is not None:
                policy_valid = _integer_value(
                    row[layout.policy_valid_index],
                    column="policy_valid",
                    row_number=row_number,
                )
                if policy_valid not in (0, 1):
                    raise QposReferenceError(
                        f"policy_valid must be 0 or 1 at CSV row {row_number}"
                    )
                if not policy_valid:
                    invalid_rows += 1
                    continue

            if previous_time is not None and control_time <= previous_time:
                raise QposReferenceError(
                    "control_time_s is not strictly increasing across valid "
                    f"rows at CSV row {row_number}: {previous_time} -> "
                    f"{control_time}"
                )
            previous_time = control_time
            valid_rows += 1

            joint_pos = _values(
                row, layout.joint_qpos_indices, layout, row_number
            )
            joint_vel = _values(
                row, layout.joint_qvel_indices, layout, row_number
            )
            root_pos = _values(
                row, layout.root_position_indices, layout, row_number
            )
            root_quat, norm_error = _normalize_quaternion(
                _values(
                    row, layout.root_quaternion_indices, layout, row_number
                ),
                tolerance=quaternion_norm_tolerance,
                row_number=row_number,
            )
            max_quaternion_norm_error = max(
                max_quaternion_norm_error, norm_error
            )
            left_hand = _values(
                row, layout.left_hand_indices, layout, row_number
            )
            right_hand = _values(
                row, layout.right_hand_indices, layout, row_number
            )

            if not groups or groups[-1].policy_seq != policy_seq:
                if groups and policy_seq != groups[-1].policy_seq + 1:
                    raise QposReferenceError(
                        "policy_seq is not consecutive at CSV row "
                        f"{row_number}: {groups[-1].policy_seq} -> {policy_seq}"
                    )
                groups.append(
                    _PolicyGroup(
                        policy_seq=policy_seq,
                        control_time_s=control_time,
                        source_row_index=source_row_index,
                        source_csv_row_number=row_number,
                        joint_pos=joint_pos,
                        joint_vel=joint_vel,
                        root_pos=root_pos,
                        root_quat_wxyz=root_quat,
                        left_hand_target=left_hand,
                        right_hand_target=right_hand,
                    )
                )
                continue

            group = groups[-1]
            group.row_count += 1
            group.max_joint_pos_change_from_first = max(
                group.max_joint_pos_change_from_first,
                float(np.max(np.abs(joint_pos - group.joint_pos))),
            )
            group.max_joint_vel_change_from_first = max(
                group.max_joint_vel_change_from_first,
                float(np.max(np.abs(joint_vel - group.joint_vel))),
            )
            group.max_root_pos_change_from_first = max(
                group.max_root_pos_change_from_first,
                float(np.max(np.abs(root_pos - group.root_pos))),
            )
            group.max_hand_change_from_first = max(
                group.max_hand_change_from_first,
                float(np.max(np.abs(left_hand - group.left_hand_target))),
                float(np.max(np.abs(right_hand - group.right_hand_target))),
            )

    if not groups:
        raise QposReferenceError(
            f"CSV has no policy-valid rows: {csv_path} "
            f"(invalid rows skipped: {invalid_rows})"
        )
    return layout, groups, {
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "invalid_rows": invalid_rows,
        "max_quaternion_norm_error": max_quaternion_norm_error,
    }


def load_qpos_reference(
    path: str | Path,
    *,
    drop_truncated_edges: bool = True,
    edge_fraction: float = 0.5,
    quaternion_norm_tolerance: float = 1e-3,
    expected_frame_dt_s: float = FRAME_DT_S,
    median_frame_dt_tolerance_s: float = 5e-3,
) -> QposReferenceSequence:
    """Load a recording folder or ``data.csv`` as a 50 Hz state reference.

    ``policy_seq`` and every frame array in the returned object already exclude
    default-dropped edge groups.  Consequently, ``sequence.policy_seq[offset]``
    is the exact value that a task simulator should receive for matching
    initialization.
    """

    if not 0.0 < edge_fraction < 1.0:
        raise QposReferenceError("edge_fraction must be between zero and one")
    if quaternion_norm_tolerance <= 0.0:
        raise QposReferenceError(
            "quaternion_norm_tolerance must be positive"
        )
    csv_path = _resolve_csv_path(path)
    layout, all_groups, source = _read_groups(
        csv_path,
        quaternion_norm_tolerance=quaternion_norm_tolerance,
    )
    median_group_rows = float(
        np.median([group.row_count for group in all_groups])
    )
    threshold = median_group_rows * edge_fraction
    start = 0
    stop = len(all_groups)
    dropped: list[DroppedEdgeGroup] = []
    if drop_truncated_edges and all_groups[0].row_count < threshold:
        group = all_groups[0]
        dropped.append(
            DroppedEdgeGroup(
                side="first",
                policy_seq=group.policy_seq,
                row_count=group.row_count,
                source_row_index=group.source_row_index,
                source_csv_row_number=group.source_csv_row_number,
            )
        )
        start += 1
    if (
        drop_truncated_edges
        and stop > start
        and all_groups[-1].row_count < threshold
    ):
        group = all_groups[-1]
        dropped.append(
            DroppedEdgeGroup(
                side="last",
                policy_seq=group.policy_seq,
                row_count=group.row_count,
                source_row_index=group.source_row_index,
                source_csv_row_number=group.source_csv_row_number,
            )
        )
        stop -= 1
    groups = all_groups[start:stop]
    if not groups:
        raise QposReferenceError(
            "edge trimming removed every policy group; use "
            "drop_truncated_edges=False only after inspecting the recording"
        )

    policy_seq = np.asarray(
        [group.policy_seq for group in groups], dtype=np.int64
    )
    if policy_seq.size > 1 and np.any(np.diff(policy_seq) != 1):
        raise QposReferenceError(
            "retained policy_seq values are not consecutive with step one"
        )
    control_time = np.asarray(
        [group.control_time_s for group in groups], dtype=np.float64
    )
    if control_time.size > 1:
        deltas = np.diff(control_time)
        median_delta = float(np.median(deltas))
        if abs(median_delta - expected_frame_dt_s) > median_frame_dt_tolerance_s:
            raise QposReferenceError(
                "first-row policy timing is not consistent with 50 Hz: "
                f"median dt={median_delta:.9g}s, expected "
                f"{expected_frame_dt_s:.9g}s"
            )

    root_quat, sign_flips = _continuous_quaternion_sign(
        np.stack([group.root_quat_wxyz for group in groups])
    )
    sequence = QposReferenceSequence(
        csv_path=csv_path,
        joint_names=G1_ISAACLAB_JOINT_NAMES,
        policy_seq=policy_seq,
        control_time_s=control_time,
        source_row_indices=np.asarray(
            [group.source_row_index for group in groups], dtype=np.int64
        ),
        source_csv_row_numbers=np.asarray(
            [group.source_csv_row_number for group in groups], dtype=np.int64
        ),
        group_row_counts=np.asarray(
            [group.row_count for group in groups], dtype=np.int32
        ),
        joint_pos=np.stack([group.joint_pos for group in groups]),
        joint_vel=np.stack([group.joint_vel for group in groups]),
        root_pos=np.stack([group.root_pos for group in groups]),
        root_quat_wxyz=root_quat,
        left_hand_target=np.stack(
            [group.left_hand_target for group in groups]
        ),
        right_hand_target=np.stack(
            [group.right_hand_target for group in groups]
        ),
        joint_qpos_columns=tuple(
            layout.header[index] for index in layout.joint_qpos_indices
        ),
        joint_qvel_columns=tuple(
            layout.header[index] for index in layout.joint_qvel_indices
        ),
        root_position_columns=tuple(
            layout.header[index] for index in layout.root_position_indices
        ),
        root_quaternion_columns=tuple(
            layout.header[index] for index in layout.root_quaternion_indices
        ),
        left_hand_columns=tuple(
            layout.header[index] for index in layout.left_hand_indices
        ),
        right_hand_columns=tuple(
            layout.header[index] for index in layout.right_hand_indices
        ),
        source_header_width=len(layout.header),
        source_qpos_width=layout.qpos_width,
        source_qvel_width=layout.qvel_width,
        source_total_rows=int(source["total_rows"]),
        source_valid_rows=int(source["valid_rows"]),
        source_invalid_rows=int(source["invalid_rows"]),
        source_group_count=len(all_groups),
        median_group_rows=median_group_rows,
        edge_fraction=edge_fraction,
        dropped_edge_groups=tuple(dropped),
        max_intragroup_joint_pos_change=max(
            group.max_joint_pos_change_from_first for group in all_groups
        ),
        max_intragroup_joint_vel_change=max(
            group.max_joint_vel_change_from_first for group in all_groups
        ),
        max_intragroup_root_pos_change=max(
            group.max_root_pos_change_from_first for group in all_groups
        ),
        max_intragroup_hand_change=max(
            group.max_hand_change_from_first for group in all_groups
        ),
        quaternion_max_norm_error_before_normalization=float(
            source["max_quaternion_norm_error"]
        ),
        quaternion_sign_flips_fixed=sign_flips,
    )
    return sequence


def _summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
        }
    return {
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
    }


def _quaternion_step_degrees(quaternions: np.ndarray) -> np.ndarray:
    if quaternions.shape[0] < 2:
        return np.empty(0, dtype=np.float64)
    dot = np.sum(quaternions[:-1] * quaternions[1:], axis=1)
    return np.degrees(2.0 * np.arccos(np.clip(np.abs(dot), 0.0, 1.0)))


def build_diagnostics(sequence: QposReferenceSequence) -> dict[str, Any]:
    """Return JSON-serializable provenance and numerical diagnostics."""

    frame_deltas = np.diff(sequence.control_time_s)
    if sequence.num_frames >= 2:
        edge_order = 2 if sequence.num_frames >= 3 else 1
        finite_difference_velocity = np.gradient(
            sequence.joint_pos.astype(np.float64),
            FRAME_DT_S,
            axis=0,
            edge_order=edge_order,
        )
        velocity_error = (
            finite_difference_velocity
            - sequence.joint_vel.astype(np.float64)
        )
        per_joint_rmse = np.sqrt(np.mean(velocity_error**2, axis=0))
        velocity_consistency: dict[str, Any] = {
            "method": "numpy.gradient at fixed deploy dt=0.02s",
            "aggregate_rmse_rad_s": float(
                np.sqrt(np.mean(velocity_error**2))
            ),
            "max_abs_error_rad_s": float(np.max(np.abs(velocity_error))),
            "per_joint_rmse_rad_s": {
                name: float(value)
                for name, value in zip(
                    sequence.joint_names, per_joint_rmse, strict=True
                )
            },
            "interpretation": (
                "diagnostic only; recorded qvel is exported unchanged"
            ),
        }
    else:
        velocity_consistency = {
            "method": "unavailable for a one-frame slice",
            "aggregate_rmse_rad_s": None,
            "max_abs_error_rad_s": None,
            "per_joint_rmse_rad_s": {},
            "interpretation": (
                "diagnostic only; recorded qvel is exported unchanged"
            ),
        }

    left_transitions = (
        np.max(np.abs(np.diff(sequence.left_hand_target, axis=0)), axis=1)
        if sequence.num_frames > 1
        else np.empty(0)
    )
    right_transitions = (
        np.max(np.abs(np.diff(sequence.right_hand_target, axis=0)), axis=1)
        if sequence.num_frames > 1
        else np.empty(0)
    )
    return {
        "schema_version": 1,
        "source": {
            "csv_path": str(sequence.csv_path),
            "header_width": sequence.source_header_width,
            "qpos_width": sequence.source_qpos_width,
            "qvel_width": sequence.source_qvel_width,
            "total_data_rows": sequence.source_total_rows,
            "policy_valid_rows": sequence.source_valid_rows,
            "policy_invalid_rows_skipped": sequence.source_invalid_rows,
        },
        "sampling": {
            "method": "first CSV row of each consecutive policy_seq group",
            "output_rate_hz": FRAME_RATE_HZ,
            "output_dt_s": FRAME_DT_S,
            "source_group_count": sequence.source_group_count,
            "retained_frame_count": sequence.num_frames,
            "frame_offset": sequence.frame_offset,
            "duration_s": sequence.duration_s,
            "sample_span_s": sequence.sample_span_s,
            "first_policy_seq": sequence.start_policy_seq,
            "last_policy_seq": int(sequence.policy_seq[-1]),
            "policy_seq_is_consecutive": bool(
                sequence.num_frames == 1
                or np.all(np.diff(sequence.policy_seq) == 1)
            ),
            "first_source_row_index": sequence.start_source_row_index,
            "first_source_csv_row_number": (
                sequence.start_source_csv_row_number
            ),
            "group_row_counts": _summary(sequence.group_row_counts),
            "source_median_group_rows": sequence.median_group_rows,
            "truncated_edge_rule": (
                "drop first/last group when row_count < "
                "median_group_rows * edge_fraction"
            ),
            "edge_fraction": sequence.edge_fraction,
            "edge_threshold_rows": (
                sequence.median_group_rows * sequence.edge_fraction
            ),
            "dropped_edge_groups": [
                {
                    "side": group.side,
                    "policy_seq": group.policy_seq,
                    "row_count": group.row_count,
                    "source_row_index": group.source_row_index,
                    "source_csv_row_number": group.source_csv_row_number,
                }
                for group in sequence.dropped_edge_groups
            ],
            "source_first_row_time_delta_s": _summary(frame_deltas),
        },
        "column_mapping": {
            "joint_order": "G1 IsaacLab 29-DOF",
            "joint_names": list(sequence.joint_names),
            "joint_qpos": dict(
                zip(
                    sequence.joint_names,
                    sequence.joint_qpos_columns,
                    strict=True,
                )
            ),
            "joint_qvel": dict(
                zip(
                    sequence.joint_names,
                    sequence.joint_qvel_columns,
                    strict=True,
                )
            ),
            "root_position": list(sequence.root_position_columns),
            "root_quaternion_wxyz": list(
                sequence.root_quaternion_columns
            ),
            "left_hand_target": list(sequence.left_hand_columns),
            "right_hand_target": list(sequence.right_hand_columns),
        },
        "validation": {
            "all_exported_values_finite": True,
            "all_source_rows_have_header_width": True,
            "joint_name_mapping_is_complete_and_one_to_one": True,
            "indexed_qpos_qvel_columns_are_contiguous": True,
            "quaternion_max_norm_error_before_normalization": (
                sequence.quaternion_max_norm_error_before_normalization
            ),
            "quaternion_max_norm_error_after_normalization": float(
                np.max(
                    np.abs(
                        np.linalg.norm(sequence.root_quat_wxyz, axis=1) - 1.0
                    )
                )
            ),
            "quaternion_sign_flips_fixed": (
                sequence.quaternion_sign_flips_fixed
            ),
            "root_orientation_step_degrees": _summary(
                _quaternion_step_degrees(sequence.root_quat_wxyz)
            ),
            "max_intragroup_joint_pos_change": (
                sequence.max_intragroup_joint_pos_change
            ),
            "max_intragroup_joint_vel_change": (
                sequence.max_intragroup_joint_vel_change
            ),
            "max_intragroup_root_pos_change": (
                sequence.max_intragroup_root_pos_change
            ),
            "max_intragroup_hand_change": (
                sequence.max_intragroup_hand_change
            ),
        },
        "value_ranges": {
            "joint_pos_rad": _summary(sequence.joint_pos),
            "joint_vel_rad_s": _summary(sequence.joint_vel),
            "root_pos_m": _summary(sequence.root_pos),
            "left_hand_target": _summary(sequence.left_hand_target),
            "right_hand_target": _summary(sequence.right_hand_target),
            "left_hand_transition_max_abs": _summary(left_transitions),
            "right_hand_transition_max_abs": _summary(right_transitions),
        },
        "joint_velocity_consistency": velocity_consistency,
    }


def _write_numeric_csv(
    path: Path, header: Sequence[str], values: np.ndarray
) -> None:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] != len(header):
        raise QposReferenceError(
            f"cannot write {path.name}: shape {array.shape}, "
            f"header width {len(header)}"
        )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        for row in array:
            writer.writerow([format(float(value), ".9g") for value in row])


def write_deploy_reference(
    sequence: QposReferenceSequence,
    output_dir: str | Path,
    *,
    motion_name: str | None = None,
    include_sidecars: bool = True,
) -> Path:
    """Write the five root-only deploy files and optional provenance sidecars.

    If ``motion_name`` is given, files are written to
    ``output_dir/motion_name`` so that ``output_dir`` can be passed directly to
    ``deploy.sh --motion-data``.  With no name, ``output_dir`` itself is the
    motion folder.
    """

    root = Path(output_dir).expanduser().resolve()
    if motion_name is not None:
        if (
            not motion_name
            or motion_name in {".", ".."}
            or Path(motion_name).name != motion_name
        ):
            raise QposReferenceError(
                f"motion_name must be one safe path component: {motion_name!r}"
            )
        motion_dir = root / motion_name
    else:
        motion_dir = root
        motion_name = motion_dir.name
    motion_dir.mkdir(parents=True, exist_ok=True)

    _write_numeric_csv(
        motion_dir / "joint_pos.csv",
        [f"joint_{index}" for index in range(NUM_BODY_JOINTS)],
        sequence.joint_pos,
    )
    _write_numeric_csv(
        motion_dir / "joint_vel.csv",
        [f"joint_vel_{index}" for index in range(NUM_BODY_JOINTS)],
        sequence.joint_vel,
    )
    _write_numeric_csv(
        motion_dir / "body_pos.csv",
        ("body_0_x", "body_0_y", "body_0_z"),
        sequence.root_pos,
    )
    _write_numeric_csv(
        motion_dir / "body_quat.csv",
        ("body_0_w", "body_0_x", "body_0_y", "body_0_z"),
        sequence.root_quat_wxyz,
    )
    metadata = (
        f"Metadata for: {motion_name}\n"
        "==============================\n\n"
        "Body part indexes:\n"
        "[0]\n\n"
        f"Total timesteps: {sequence.num_frames}\n"
    )
    (motion_dir / "metadata.txt").write_text(metadata, encoding="utf-8")

    if include_sidecars:
        with (motion_dir / "frame_map.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(
                (
                    "frame_index",
                    "policy_seq",
                    "control_time_s",
                    "source_row_index",
                    "source_csv_row_number",
                    "group_row_count",
                )
            )
            for frame in range(sequence.num_frames):
                writer.writerow(
                    (
                        frame,
                        int(sequence.policy_seq[frame]),
                        format(float(sequence.control_time_s[frame]), ".17g"),
                        int(sequence.source_row_indices[frame]),
                        int(sequence.source_csv_row_numbers[frame]),
                        int(sequence.group_row_counts[frame]),
                    )
                )
        report = build_diagnostics(sequence)
        report["output"] = {
            "motion_name": motion_name,
            "motion_dir": str(motion_dir),
            "deploy_files": [
                "joint_pos.csv",
                "joint_vel.csv",
                "body_pos.csv",
                "body_quat.csv",
                "metadata.txt",
            ],
            "sidecars": ["frame_map.csv", "conversion_report.json"],
        }
        (motion_dir / "conversion_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return motion_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert recording/data.csv qpos into a 50 Hz root-only SONIC "
            "deploy reference."
        )
    )
    parser.add_argument("recording", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--motion-name",
        help=(
            "optional child motion folder; with this option output-dir is a "
            "deploy dataset root"
        ),
    )
    parser.add_argument(
        "--keep-truncated-edges",
        action="store_true",
        help="retain incomplete first/last policy_seq groups",
    )
    parser.add_argument(
        "--no-sidecars",
        action="store_true",
        help="omit frame_map.csv and conversion_report.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    sequence = load_qpos_reference(
        args.recording,
        drop_truncated_edges=not args.keep_truncated_edges,
    )
    motion_dir = write_deploy_reference(
        sequence,
        args.output_dir,
        motion_name=args.motion_name,
        include_sidecars=not args.no_sidecars,
    )
    print(
        f"[qpos-reference] wrote {sequence.num_frames} frames "
        f"(policy_seq {sequence.policy_seq[0]}..{sequence.policy_seq[-1]}) "
        f"to {motion_dir}"
    )
    if sequence.dropped_edge_groups:
        summary = ", ".join(
            f"{group.side} seq={group.policy_seq} rows={group.row_count}"
            for group in sequence.dropped_edge_groups
        )
        print(f"[qpos-reference] dropped truncated edge groups: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
