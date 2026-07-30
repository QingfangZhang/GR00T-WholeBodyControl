"""Read and validate a recorded SONIC MuJoCo CSV as a 50 Hz reference.

Only the policy-level, exogenous data is extracted here.  In particular, this
module never treats the recorded robot ``qpos`` trajectory as a command.  The
recorded pelvis quaternion is used solely to undo the body-relative 6D anchor
orientation saved in ``reference_motion``.

The regular SONIC G1 encoder layout saved by this recording is::

    reference_motion[0:290]   = 10 x 29 reference joint positions
    reference_motion[290:580] = 10 x 29 reference joint velocities
    reference_motion[580:640] = 10 x 6 relative anchor orientations

For protocol-v1 streaming we recover a consecutive 50 Hz sequence by taking
the *current* frame (slot zero) from each distinct ``policy_seq``.  Low-latency
uses that sequence directly.  The regular checkpoint instead retains the ten
recorded slots verbatim, because this CSV's tail slots are source-clamped and
are not equivalent to sampling a newly reconstructed 46-frame future.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


REFERENCE_HEADER_SIZE = 1024
ACTIVE_REFERENCE_SIZE = 640
NUM_REFERENCE_FRAMES = 10
NUM_BODY_JOINTS = 29
ANCHOR_6D_SIZE = 6

_QPOS_INDEX_RE = re.compile(r"\[qpos(\d+)\]$")


class ReferenceDataError(RuntimeError):
    """Raised when a recording cannot safely be interpreted as SONIC input."""


@dataclass
class _PolicyGroup:
    """Raw values retained for one distinct policy sequence number."""

    policy_seq: int
    control_time_s: float
    reference_motion: np.ndarray
    left_hand: np.ndarray
    right_hand: np.ndarray
    base_quats: list[np.ndarray]
    row_count: int = 1
    max_reference_change: float = 0.0
    max_hand_change: float = 0.0


@dataclass(frozen=True)
class ReferenceSequence:
    """Consecutive policy-level data ready for protocol-v1 publication."""

    csv_path: Path
    policy_seq: np.ndarray
    control_time_s: np.ndarray
    group_row_counts: np.ndarray
    reference_motion: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    relative_anchor_6d: np.ndarray
    reference_anchor_quat_wxyz: np.ndarray
    policy_base_quat_wxyz: np.ndarray
    base_quat_samples_wxyz: np.ndarray
    base_sample_counts: np.ndarray
    left_hand_target: np.ndarray
    right_hand_target: np.ndarray
    base_sample_mode: str
    max_intragroup_reference_change: float
    max_intragroup_hand_change: float

    @property
    def num_frames(self) -> int:
        return int(self.policy_seq.shape[0])

    def slice(self, start: int = 0, count: int | None = None) -> "ReferenceSequence":
        if start < 0 or start >= self.num_frames:
            raise ReferenceDataError(
                f"start policy offset {start} is outside [0, {self.num_frames - 1}]"
            )
        stop = self.num_frames if count is None else min(self.num_frames, start + count)
        if stop <= start:
            raise ReferenceDataError("selected reference slice is empty")
        values: dict[str, Any] = {}
        for name in (
            "policy_seq",
            "control_time_s",
            "group_row_counts",
            "reference_motion",
            "joint_pos",
            "joint_vel",
            "relative_anchor_6d",
            "reference_anchor_quat_wxyz",
            "policy_base_quat_wxyz",
            "base_quat_samples_wxyz",
            "base_sample_counts",
            "left_hand_target",
            "right_hand_target",
        ):
            values[name] = getattr(self, name)[start:stop]
        return ReferenceSequence(
            csv_path=self.csv_path,
            base_sample_mode=self.base_sample_mode,
            max_intragroup_reference_change=self.max_intragroup_reference_change,
            max_intragroup_hand_change=self.max_intragroup_hand_change,
            **values,
        )


def _as_float(value: str, *, column: str, row_number: int) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ReferenceDataError(
            f"invalid numeric value in {column!r} at CSV row {row_number}: {value!r}"
        ) from exc
    if not math.isfinite(result):
        raise ReferenceDataError(
            f"non-finite value in {column!r} at CSV row {row_number}: {value!r}"
        )
    return result


def _normalise_quaternion(quat: np.ndarray, *, label: str) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if not math.isfinite(norm) or norm < 1e-8:
        raise ReferenceDataError(f"invalid quaternion for {label}: norm={norm}")
    return quat / norm


def quat_conjugate_wxyz(quat: np.ndarray) -> np.ndarray:
    result = np.array(quat, dtype=np.float64, copy=True)
    result[..., 1:] *= -1.0
    return result


def quat_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton product, accepting a final quaternion dimension of four."""

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    w1, x1, y1, z1 = np.moveaxis(left, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(right, -1, 0)
    values = np.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        axis=-1,
    )
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norms < 1e-8) or not np.all(np.isfinite(norms)):
        raise ReferenceDataError("quaternion product produced an invalid norm")
    return values / norms


def quat_to_matrix_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norms = np.linalg.norm(quat, axis=-1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ReferenceDataError("cannot convert a zero quaternion to a matrix")
    quat = quat / norms
    w, x, y, z = np.moveaxis(quat, -1, 0)
    return np.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(quat.shape[:-1] + (3, 3))


def matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Convert one proper 3x3 rotation matrix to a scalar-first quaternion."""

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ReferenceDataError(f"expected a 3x3 rotation matrix, got {matrix.shape}")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(max(0.0, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])) * 2.0
            quat = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif index == 1:
            scale = math.sqrt(max(0.0, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])) * 2.0
            quat = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = math.sqrt(max(0.0, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])) * 2.0
            quat = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    return _normalise_quaternion(quat, label="rotation matrix")


def rotation_6d_to_matrix(value: np.ndarray) -> tuple[np.ndarray, float]:
    """Recover a proper rotation from row-major ``R[:, :2]`` values.

    Returns the matrix and the Frobenius correction applied by Gram-Schmidt.
    The correction is diagnostic: recorded values should already be very close
    to orthonormal.
    """

    value = np.asarray(value, dtype=np.float64)
    if value.shape != (6,) or not np.all(np.isfinite(value)):
        raise ReferenceDataError(f"invalid anchor 6D value with shape {value.shape}")
    raw = value.reshape(3, 2)
    first = raw[:, 0]
    first_norm = float(np.linalg.norm(first))
    if first_norm < 1e-8:
        raise ReferenceDataError("anchor 6D first column has zero norm")
    first = first / first_norm
    second = raw[:, 1] - first * float(np.dot(first, raw[:, 1]))
    second_norm = float(np.linalg.norm(second))
    if second_norm < 1e-8:
        raise ReferenceDataError("anchor 6D columns are linearly dependent")
    second = second / second_norm
    third = np.cross(first, second)
    matrix = np.column_stack((first, second, third))
    correction = float(np.linalg.norm(matrix[:, :2] - raw))
    return matrix, correction


def _relative_quaternions(relative_6d: np.ndarray) -> tuple[np.ndarray, float]:
    flat = np.asarray(relative_6d, dtype=np.float64).reshape(-1, 6)
    quats = np.empty((flat.shape[0], 4), dtype=np.float64)
    max_correction = 0.0
    for index, value in enumerate(flat):
        matrix, correction = rotation_6d_to_matrix(value)
        quats[index] = matrix_to_quat_wxyz(matrix)
        max_correction = max(max_correction, correction)
    return quats.reshape(relative_6d.shape[:-1] + (4,)), max_correction


def _continuous_quaternion_sign(quaternions: np.ndarray) -> np.ndarray:
    result = np.array(quaternions, dtype=np.float64, copy=True)
    for index in range(1, result.shape[0]):
        if float(np.dot(result[index - 1], result[index])) < 0.0:
            result[index] *= -1.0
    return result


def _column_indices(header: Sequence[str], prefix: str, count: int) -> list[int]:
    lookup = {name: index for index, name in enumerate(header)}
    names = [f"{prefix}[{index}]" for index in range(count)]
    missing = [name for name in names if name not in lookup]
    if missing:
        preview = ", ".join(missing[:5])
        raise ReferenceDataError(f"CSV is missing {len(missing)} required columns: {preview}")
    return [lookup[name] for name in names]


def _qpos_quaternion_indices(header: Sequence[str]) -> list[int]:
    by_qpos_index: dict[int, int] = {}
    for column, name in enumerate(header):
        if not name.startswith("qpos:"):
            continue
        match = _QPOS_INDEX_RE.search(name)
        if match:
            by_qpos_index[int(match.group(1))] = column
    missing = [index for index in range(3, 7) if index not in by_qpos_index]
    if missing:
        raise ReferenceDataError(f"CSV is missing pelvis qpos quaternion indices: {missing}")
    return [by_qpos_index[index] for index in range(3, 7)]


def _resolve_csv_path(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path = path / "data.csv"
    if not path.is_file():
        raise ReferenceDataError(f"recording CSV does not exist: {path}")
    return path


def _base_from_group(groups: Sequence[_PolicyGroup], index: int, mode: str) -> np.ndarray:
    use_previous = mode.startswith("previous-")
    selector = mode[len("previous-") :] if use_previous else mode
    # No pre-recording row exists for the first (partial) policy group.  Its
    # first row is the closest causal fallback; all subsequent groups can use
    # the explicitly selected sample from the preceding group.
    if use_previous and index == 0:
        return groups[0].base_quats[0]
    group_index = index - 1 if use_previous else index
    values = groups[group_index].base_quats
    if selector == "first":
        chosen = values[0]
    elif selector == "last":
        chosen = values[-1]
    elif selector == "middle":
        chosen = values[len(values) // 2]
    elif selector.startswith("index"):
        raw_index = selector.removeprefix("index").lstrip(":=-_")
        if not raw_index.isdigit():
            raise ReferenceDataError(
                f"invalid base sample mode {mode!r}; expected e.g. previous-index5"
            )
        sample_index = int(raw_index)
        chosen = values[min(sample_index, len(values) - 1)]
    else:
        raise ReferenceDataError(
            f"invalid base sample mode {mode!r}; use first/last/middle/indexN, "
            "optionally prefixed by previous-"
        )
    return chosen


def _extract_groups(csv_path: Path) -> list[_PolicyGroup]:
    groups: list[_PolicyGroup] = []
    invalid_rows = 0
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ReferenceDataError(f"empty CSV: {csv_path}") from exc
        lookup = {name: index for index, name in enumerate(header)}
        for required in ("policy_seq", "control_time_s"):
            if required not in lookup:
                raise ReferenceDataError(f"CSV is missing required column {required!r}")
        valid_index = lookup.get("policy_valid")
        size_index = lookup.get("policy_reference_motion_size")
        reference_indices = _column_indices(header, "reference_motion", REFERENCE_HEADER_SIZE)
        left_indices = _column_indices(header, "left_hand_q", 7)
        right_indices = _column_indices(header, "right_hand_q", 7)
        base_indices = _qpos_quaternion_indices(header)

        for row_number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ReferenceDataError(
                    f"CSV row {row_number} has {len(row)} fields; expected {len(header)}"
                )
            if valid_index is not None and not bool(int(float(row[valid_index]))):
                invalid_rows += 1
                continue
            if size_index is not None:
                active_size = int(float(row[size_index]))
                if active_size != ACTIVE_REFERENCE_SIZE:
                    raise ReferenceDataError(
                        f"row {row_number} declares reference size {active_size}; "
                        f"expected {ACTIVE_REFERENCE_SIZE}"
                    )
            policy_seq = int(_as_float(row[lookup["policy_seq"]], column="policy_seq", row_number=row_number))
            control_time = _as_float(
                row[lookup["control_time_s"]], column="control_time_s", row_number=row_number
            )
            base_quat = _normalise_quaternion(
                np.array(
                    [
                        _as_float(row[column], column=header[column], row_number=row_number)
                        for column in base_indices
                    ]
                ),
                label=f"CSV row {row_number} pelvis",
            )
            reference = np.array(
                [
                    _as_float(row[column], column=header[column], row_number=row_number)
                    for column in reference_indices
                ],
                dtype=np.float64,
            )
            left = np.array(
                [
                    _as_float(row[column], column=header[column], row_number=row_number)
                    for column in left_indices
                ],
                dtype=np.float64,
            )
            right = np.array(
                [
                    _as_float(row[column], column=header[column], row_number=row_number)
                    for column in right_indices
                ],
                dtype=np.float64,
            )

            if not groups or groups[-1].policy_seq != policy_seq:
                if groups and policy_seq <= groups[-1].policy_seq:
                    raise ReferenceDataError(
                        f"policy_seq is not strictly increasing at CSV row {row_number}: "
                        f"{groups[-1].policy_seq} -> {policy_seq}"
                    )
                groups.append(
                    _PolicyGroup(
                        policy_seq=policy_seq,
                        control_time_s=control_time,
                        reference_motion=reference,
                        left_hand=left,
                        right_hand=right,
                        base_quats=[base_quat],
                    )
                )
            else:
                group = groups[-1]
                group.row_count += 1
                group.base_quats.append(base_quat)
                group.max_reference_change = max(
                    group.max_reference_change,
                    float(np.max(np.abs(reference - group.reference_motion))),
                )
                group.max_hand_change = max(
                    group.max_hand_change,
                    float(np.max(np.abs(left - group.left_hand))),
                    float(np.max(np.abs(right - group.right_hand))),
                )

    if not groups:
        suffix = f" ({invalid_rows} invalid rows skipped)" if invalid_rows else ""
        raise ReferenceDataError(f"CSV has no valid policy rows{suffix}: {csv_path}")
    return groups


def load_reference_sequence(
    path: str | Path,
    *,
    base_sample_mode: str = "previous-index5",
) -> ReferenceSequence:
    """Load one recording and recover its current-frame absolute reference."""

    csv_path = _resolve_csv_path(path)
    groups = _extract_groups(csv_path)
    policy_seq = np.asarray([group.policy_seq for group in groups], dtype=np.int64)
    gaps = np.diff(policy_seq)
    if gaps.size and np.any(gaps != 1):
        bad = np.flatnonzero(gaps != 1)[:5]
        examples = ", ".join(
            f"{policy_seq[index]}->{policy_seq[index + 1]}" for index in bad
        )
        raise ReferenceDataError(f"policy_seq contains gaps; examples: {examples}")

    reference = np.stack([group.reference_motion for group in groups])
    active = reference[:, :ACTIVE_REFERENCE_SIZE]
    joint_pos = active[:, : NUM_REFERENCE_FRAMES * NUM_BODY_JOINTS].reshape(
        -1, NUM_REFERENCE_FRAMES, NUM_BODY_JOINTS
    )[:, 0]
    joint_vel = active[
        :,
        NUM_REFERENCE_FRAMES * NUM_BODY_JOINTS : 2 * NUM_REFERENCE_FRAMES * NUM_BODY_JOINTS,
    ].reshape(-1, NUM_REFERENCE_FRAMES, NUM_BODY_JOINTS)[:, 0]
    relative_6d = active[:, 2 * NUM_REFERENCE_FRAMES * NUM_BODY_JOINTS :].reshape(
        -1, NUM_REFERENCE_FRAMES, ANCHOR_6D_SIZE
    )
    relative_quat, _ = _relative_quaternions(relative_6d[:, 0])
    base_quat = np.stack(
        [_base_from_group(groups, index, base_sample_mode) for index in range(len(groups))]
    )
    max_group_rows = max(len(group.base_quats) for group in groups)
    base_quat_samples = np.empty((len(groups), max_group_rows, 4), dtype=np.float64)
    base_sample_counts = np.empty(len(groups), dtype=np.int32)
    for index, group in enumerate(groups):
        samples = np.stack(group.base_quats)
        base_sample_counts[index] = samples.shape[0]
        base_quat_samples[index, : samples.shape[0]] = samples
        base_quat_samples[index, samples.shape[0] :] = samples[-1]
    absolute_quat = _continuous_quaternion_sign(
        quat_multiply_wxyz(base_quat, relative_quat)
    )

    return ReferenceSequence(
        csv_path=csv_path,
        policy_seq=policy_seq,
        control_time_s=np.asarray([group.control_time_s for group in groups]),
        group_row_counts=np.asarray([group.row_count for group in groups], dtype=np.int32),
        reference_motion=reference,
        joint_pos=joint_pos.astype(np.float32),
        joint_vel=joint_vel.astype(np.float32),
        relative_anchor_6d=relative_6d.astype(np.float32),
        reference_anchor_quat_wxyz=absolute_quat.astype(np.float32),
        policy_base_quat_wxyz=base_quat.astype(np.float32),
        base_quat_samples_wxyz=base_quat_samples.astype(np.float32),
        base_sample_counts=base_sample_counts,
        left_hand_target=np.stack([group.left_hand for group in groups]).astype(np.float32),
        right_hand_target=np.stack([group.right_hand for group in groups]).astype(np.float32),
        base_sample_mode=base_sample_mode,
        max_intragroup_reference_change=max(group.max_reference_change for group in groups),
        max_intragroup_hand_change=max(group.max_hand_change for group in groups),
    )


def _quaternion_angle_degrees(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    dot = np.sum(left * right, axis=-1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def _summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {"median": 0.0, "mean": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def _future_views(sequence: ReferenceSequence) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    active = sequence.reference_motion[:, :ACTIVE_REFERENCE_SIZE]
    positions = active[:, :290].reshape(-1, 10, 29)
    velocities = active[:, 290:580].reshape(-1, 10, 29)
    orientations = active[:, 580:640].reshape(-1, 10, 6)
    return positions, velocities, orientations


def reference_slot_views(
    sequence: ReferenceSequence,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return recorded q/dq slots and their recovered world anchor quaternions."""

    positions, velocities, relative_6d = _future_views(sequence)
    relative_quat, _ = _relative_quaternions(relative_6d)
    absolute_quat = quat_multiply_wxyz(
        sequence.policy_base_quat_wxyz[:, None, :], relative_quat
    )
    return (
        positions.astype(np.float32, copy=False),
        velocities.astype(np.float32, copy=False),
        absolute_quat.astype(np.float32),
    )


def _infer_future_lags(sequence: ReferenceSequence, max_lag: int = 60) -> list[dict[str, Any]]:
    positions, velocities, _ = _future_views(sequence)
    count = sequence.num_frames
    result: list[dict[str, Any]] = []
    # Ignore a short tail where the original motion source may have clamped all
    # future slots to its final frame.
    comparison_count = max(1, count - max_lag - 10)
    for slot in range(NUM_REFERENCE_FRAMES):
        candidates: list[tuple[float, float, int]] = []
        for lag in range(min(max_lag, count - 1) + 1):
            usable = min(comparison_count, count - lag)
            if usable <= 0:
                continue
            pos_error = np.abs(positions[:usable, slot] - positions[lag : lag + usable, 0])
            vel_error = np.abs(velocities[:usable, slot] - velocities[lag : lag + usable, 0])
            mean_error = float(np.mean(pos_error) + 0.01 * np.mean(vel_error))
            max_position_error = float(np.max(pos_error))
            candidates.append((mean_error, max_position_error, lag))
        score, max_error, lag = min(candidates)
        result.append(
            {
                "slot": slot,
                "inferred_policy_lag": lag,
                "score": score,
                "max_joint_position_error": max_error,
            }
        )
    return result


def _base_array_from_samples(sequence: ReferenceSequence, mode: str) -> np.ndarray:
    use_previous = mode.startswith("previous-")
    selector = mode[len("previous-") :] if use_previous else mode
    result = np.empty((sequence.num_frames, 4), dtype=np.float64)
    for index in range(sequence.num_frames):
        if use_previous and index == 0:
            result[index] = sequence.base_quat_samples_wxyz[0, 0]
            continue
        source = index - 1 if use_previous else index
        count = int(sequence.base_sample_counts[source])
        if selector == "first":
            sample = 0
        elif selector == "last":
            sample = count - 1
        elif selector == "middle":
            sample = count // 2
        elif selector.startswith("index"):
            raw_index = selector.removeprefix("index").lstrip(":=-_")
            if not raw_index.isdigit():
                raise ReferenceDataError(f"invalid base sample candidate: {mode}")
            sample = min(int(raw_index), count - 1)
        else:
            raise ReferenceDataError(f"invalid base sample candidate: {mode}")
        result[index] = sequence.base_quat_samples_wxyz[source, sample]
    return result


def _base_candidate_report(sequence: ReferenceSequence, future_lag: int) -> dict[str, Any]:
    modes = ["first", "previous-first", "previous-middle", "previous-last"]
    modes.extend(f"previous-index{index}" for index in range(9))
    report: dict[str, Any] = {}
    for mode in modes:
        base = _base_array_from_samples(sequence, mode)
        _, _, orientation_6d = _future_views(sequence)
        if future_lag <= 0 or sequence.num_frames <= future_lag:
            continue
        rel_future, _ = _relative_quaternions(orientation_6d[:-future_lag, 1])
        future_world = quat_multiply_wxyz(base[:-future_lag], rel_future)
        current_rel, _ = _relative_quaternions(orientation_6d[future_lag:, 0])
        current_world = quat_multiply_wxyz(base[future_lag:], current_rel)
        report[mode] = _summary(_quaternion_angle_degrees(future_world, current_world))
    return report


def build_diagnostics(
    sequence: ReferenceSequence,
    *,
    selected_start: int = 0,
    include_base_candidates: bool = True,
) -> dict[str, Any]:
    """Return JSON-serialisable evidence for every reconstruction assumption."""

    if selected_start < 0 or selected_start >= sequence.num_frames:
        raise ReferenceDataError(
            f"diagnostic start {selected_start} outside [0, {sequence.num_frames - 1}]"
        )
    time_delta = np.diff(sequence.control_time_s)
    future_lags = _infer_future_lags(sequence)
    _, _, orientation_6d = _future_views(sequence)
    overlap: list[dict[str, Any]] = []
    for item in future_lags:
        slot = int(item["slot"])
        lag = int(item["inferred_policy_lag"])
        entry = dict(item)
        if lag < sequence.num_frames:
            usable = sequence.num_frames - lag
            rel_quat, correction = _relative_quaternions(orientation_6d[:usable, slot])
            candidate_world = quat_multiply_wxyz(
                sequence.policy_base_quat_wxyz[:usable], rel_quat
            )
            target_world = sequence.reference_anchor_quat_wxyz[lag : lag + usable]
            entry["world_orientation_error_deg"] = _summary(
                _quaternion_angle_degrees(candidate_world, target_world)
            )
            entry["max_6d_orthonormalisation_correction"] = correction
        overlap.append(entry)

    current_rel_quat, correction = _relative_quaternions(
        sequence.relative_anchor_6d[:, 0]
    )
    reconstructed_rel = quat_multiply_wxyz(
        quat_conjugate_wxyz(sequence.policy_base_quat_wxyz),
        sequence.reference_anchor_quat_wxyz,
    )
    roundtrip_angle = _quaternion_angle_degrees(current_rel_quat, reconstructed_rel)
    start_rel_matrix, _ = rotation_6d_to_matrix(sequence.relative_anchor_6d[selected_start, 0])
    recorded_relative_yaw = math.atan2(start_rel_matrix[1, 0], start_rel_matrix[0, 0])
    simulator_start_base = sequence.base_quat_samples_wxyz[selected_start, 0]
    simulator_to_reference = quat_multiply_wxyz(
        quat_conjugate_wxyz(simulator_start_base),
        sequence.reference_anchor_quat_wxyz[selected_start],
    )
    simulator_relative_matrix = quat_to_matrix_wxyz(simulator_to_reference)
    simulator_relative_yaw = math.atan2(
        simulator_relative_matrix[1, 0], simulator_relative_matrix[0, 0]
    )

    group_counts = Counter(int(value) for value in sequence.group_row_counts)
    padding = sequence.reference_motion[:, ACTIVE_REFERENCE_SIZE:]
    left_changes = np.max(np.abs(np.diff(sequence.left_hand_target, axis=0)), axis=1)
    right_changes = np.max(np.abs(np.diff(sequence.right_hand_target, axis=0)), axis=1)
    first_future_lag = int(future_lags[1]["inferred_policy_lag"])
    result: dict[str, Any] = {
        "csv": str(sequence.csv_path),
        "policy_frames": sequence.num_frames,
        "policy_seq": {
            "first": int(sequence.policy_seq[0]),
            "last": int(sequence.policy_seq[-1]),
            "continuous": bool(np.all(np.diff(sequence.policy_seq) == 1)),
            "selected_start_offset": selected_start,
            "selected_start_seq": int(sequence.policy_seq[selected_start]),
        },
        "policy_period_s": _summary(time_delta),
        "rows_per_policy_seq": {str(key): value for key, value in sorted(group_counts.items())},
        "layout": {
            "active_reference_dimension": ACTIVE_REFERENCE_SIZE,
            "joint_position": [10, 29],
            "joint_velocity": [10, 29],
            "relative_anchor_orientation": [10, 6],
            "max_abs_padding_640_1024": float(np.max(np.abs(padding))) if padding.size else 0.0,
        },
        "deduplication": {
            "max_reference_change_inside_policy_seq": sequence.max_intragroup_reference_change,
            "max_hand_change_inside_policy_seq": sequence.max_intragroup_hand_change,
        },
        "absolute_orientation_recovery": {
            "formula": "q_reference_world = q_policy_base_world * q_robot_to_reference",
            "base_sample_mode": sequence.base_sample_mode,
            "max_6d_orthonormalisation_correction": correction,
            "roundtrip_error_deg": _summary(roundtrip_angle),
            "future_overlap": overlap,
        },
        "heading_reinitialisation": {
            "recorded_policy_relative_yaw_rad": recorded_relative_yaw,
            "recorded_policy_relative_yaw_deg": math.degrees(recorded_relative_yaw),
            "simulator_initial_state_relative_yaw_rad": simulator_relative_yaw,
            "simulator_initial_state_relative_yaw_deg": math.degrees(simulator_relative_yaw),
            "warning": (
                "C++ UpdateHeadingState aligns the first reference heading to the current base. "
                "That removes this initial yaw offset unless a separately verified one-time "
                "heading_increment is applied after reinitialisation."
            ),
        },
        "hands": {
            "left_target_transition_count": int(np.count_nonzero(left_changes > 1e-6)),
            "right_target_transition_count": int(np.count_nonzero(right_changes > 1e-6)),
            "source_fields": ["left_hand_q[0:7]", "right_hand_q[0:7]"],
        },
    }
    if include_base_candidates and first_future_lag > 0:
        result["absolute_orientation_recovery"]["base_sample_candidates"] = (
            _base_candidate_report(sequence, first_future_lag)
        )
        result["absolute_orientation_recovery"]["candidate_note"] = (
            "Candidate errors are diagnostics, not an automatic selector. "
            "The configured base_sample_mode remains authoritative."
        )
    return result


def write_prepared_npz(
    sequence: ReferenceSequence,
    path: str | Path,
    *,
    diagnostics: dict[str, Any] | None = None,
) -> Path:
    """Persist the compact 50 Hz exogenous stream for reproducible inspection."""

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source_csv": str(sequence.csv_path),
        "base_sample_mode": sequence.base_sample_mode,
        "protocol_version": 1,
        "contains_token_state": False,
    }
    if diagnostics is not None:
        metadata["diagnostics"] = diagnostics
    np.savez_compressed(
        path,
        policy_seq=sequence.policy_seq,
        control_time_s=sequence.control_time_s,
        reference_motion=sequence.reference_motion,
        joint_pos=sequence.joint_pos,
        joint_vel=sequence.joint_vel,
        body_quat_w=sequence.reference_anchor_quat_wxyz,
        left_hand_joints=sequence.left_hand_target,
        right_hand_joints=sequence.right_hand_target,
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    return path


def json_dump(payload: Any) -> str:
    """Stable pretty JSON used by both the CLI and tests."""

    return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
