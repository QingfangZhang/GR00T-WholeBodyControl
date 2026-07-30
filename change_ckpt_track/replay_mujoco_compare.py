#!/usr/bin/env python3
"""Replay a checkpoint rollout with the recorded qpos drawn as a ghost robot.

The original ``sample_data/ztj/replay_mujoco_csv`` binary is intentionally left
unchanged.  This viewer uses MuJoCo's passive Python viewer and its ``user_scn``
hook so that a second, visual-only G1 can be drawn without adding another
physical robot to the model.

By default the ghost follows the original recording at the simulator's
source-row clock (currently every second row of the 400 Hz recording).  This
works for both ``change_ckpt`` and ``change_ckpt_track`` rollouts.  Explicit
``reference`` mode instead samples original qpos at the 50 Hz policy boundaries
described by ``prepared_reference.npz`` and holds each frame at the nominal
reference clock.  Neither mode changes physics or reruns a policy.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any, Sequence

import mujoco
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from task_sim_io import (  # noqa: E402
    resolve_recording,
    stage_recording_snapshot,
)


class ReplayCompareError(RuntimeError):
    """Raised when the rollout/reference pair cannot be compared safely."""


@dataclass(frozen=True)
class CsvQpos:
    path: Path
    header: tuple[str, ...]
    qpos_names: tuple[str, ...]
    qpos: np.ndarray
    sample_index: np.ndarray
    control_time_s: np.ndarray
    mujoco_time_s: np.ndarray
    policy_seq: np.ndarray | None


@dataclass(frozen=True)
class PreparedReference:
    path: Path
    source_row_index: np.ndarray | None
    policy_seq: np.ndarray
    rate_hz: float
    metadata: dict[str, Any]
    control_time_s: np.ndarray | None = None


@dataclass(frozen=True)
class GhostTrack:
    qpos: np.ndarray
    source_row_index: np.ndarray
    reference_frame_index: np.ndarray
    policy_seq: np.ndarray
    mode: str
    description: str


@dataclass(frozen=True)
class RobotVisual:
    pelvis_body_id: int
    robot_body_ids: frozenset[int]
    hand_body_ids: frozenset[int]
    geom_ids: tuple[int, ...]
    root_qpos_indices: np.ndarray
    body_joint_qpos_indices: np.ndarray


_INDEXED_QPOS = re.compile(r"\[qpos(\d+)\]$")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayCompareError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReplayCompareError(f"expected a JSON object in {path}")
    return value


def _optional_float(row: Sequence[str], column: int | None) -> float:
    if column is None or column >= len(row) or not row[column].strip():
        return math.nan
    try:
        return float(row[column])
    except ValueError:
        return math.nan


def _optional_int(row: Sequence[str], column: int | None) -> int | None:
    value = _optional_float(row, column)
    if not math.isfinite(value):
        return None
    rounded = int(round(value))
    if not math.isclose(value, rounded, abs_tol=1e-6):
        return None
    return rounded


def read_qpos_csv(value: str | Path) -> CsvQpos:
    """Read only the qpos and alignment columns from a wide replay CSV."""

    path = Path(value).expanduser().resolve()
    if path.is_dir():
        path = path / "data.csv"
    if not path.is_file():
        raise ReplayCompareError(f"data.csv not found: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        try:
            header_list = next(reader)
        except StopIteration as exc:
            raise ReplayCompareError(f"empty CSV: {path}") from exc

        header = tuple(header_list)
        column = {name: index for index, name in enumerate(header)}
        indexed: list[tuple[int, int, str]] = []
        for csv_column, name in enumerate(header):
            match = _INDEXED_QPOS.search(name)
            if match:
                indexed.append((int(match.group(1)), csv_column, name))
        indexed.sort()
        actual_indices = [index for index, _, _ in indexed]
        if not indexed or actual_indices != list(range(len(indexed))):
            raise ReplayCompareError(
                f"qpos columns are missing or non-contiguous in {path}: "
                f"{actual_indices[:12]}"
            )

        qpos_columns = [csv_column for _, csv_column, _ in indexed]
        qpos_names = tuple(name for _, _, name in indexed)
        sample_column = column.get("sample_index")
        control_column = column.get("control_time_s")
        mujoco_column = column.get("mujoco_time_s")
        policy_column = column.get("policy_seq")

        qpos_rows: list[list[float]] = []
        sample_rows: list[int] = []
        control_rows: list[float] = []
        mujoco_rows: list[float] = []
        policy_rows: list[int] = []
        policy_available = policy_column is not None

        for data_row_index, row in enumerate(reader):
            if len(row) != len(header):
                raise ReplayCompareError(
                    f"CSV data row {data_row_index} in {path} has {len(row)} "
                    f"columns; expected {len(header)}"
                )
            try:
                qpos_rows.append([float(row[index]) for index in qpos_columns])
            except ValueError as exc:
                raise ReplayCompareError(
                    f"invalid qpos value in data row {data_row_index} of {path}: {exc}"
                ) from exc

            sample_value = _optional_int(row, sample_column)
            sample_rows.append(
                data_row_index if sample_value is None else sample_value
            )
            control_rows.append(_optional_float(row, control_column))
            mujoco_rows.append(_optional_float(row, mujoco_column))
            if policy_available:
                policy_value = _optional_int(row, policy_column)
                if policy_value is None:
                    raise ReplayCompareError(
                        f"invalid policy_seq in data row {data_row_index} of {path}"
                    )
                policy_rows.append(policy_value)

    if not qpos_rows:
        raise ReplayCompareError(f"CSV has a header but no data rows: {path}")

    return CsvQpos(
        path=path,
        header=header,
        qpos_names=qpos_names,
        qpos=np.asarray(qpos_rows, dtype=np.float64),
        sample_index=np.asarray(sample_rows, dtype=np.int64),
        control_time_s=np.asarray(control_rows, dtype=np.float64),
        mujoco_time_s=np.asarray(mujoco_rows, dtype=np.float64),
        policy_seq=(
            np.asarray(policy_rows, dtype=np.int64)
            if policy_available
            else None
        ),
    )


def _read_prepared_reference(path: Path) -> PreparedReference:
    if not path.is_file():
        raise ReplayCompareError(
            f"prepared reference is missing: {path}. "
            "--ghost-mode reference requires a rollout with "
            "prepared_reference.npz."
        )
    try:
        with np.load(path, allow_pickle=False) as archive:
            if "policy_seq" not in archive:
                raise ReplayCompareError(
                    f"{path} is missing required array 'policy_seq'"
                )
            source_rows = (
                np.asarray(archive["source_row_index"], dtype=np.int64)
                if "source_row_index" in archive
                else None
            )
            policy_seq = np.asarray(archive["policy_seq"], dtype=np.int64)
            metadata: dict[str, Any] = {}
            if "metadata_json" in archive:
                raw_metadata = str(archive["metadata_json"].item())
                decoded = json.loads(raw_metadata)
                if isinstance(decoded, dict):
                    metadata = decoded
            reference_time = (
                np.asarray(archive["control_time_s"], dtype=np.float64)
                if "control_time_s" in archive
                else None
            )
    except ReplayCompareError:
        raise
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ReplayCompareError(f"cannot read prepared reference {path}: {exc}") from exc

    if policy_seq.ndim != 1 or not len(policy_seq):
        raise ReplayCompareError(
            f"policy_seq must be a non-empty vector in {path}"
        )
    if len(policy_seq) > 1 and np.any(np.diff(policy_seq) <= 0):
        raise ReplayCompareError(
            f"policy_seq must be strictly increasing in {path}"
        )
    if source_rows is not None and (
        source_rows.ndim != 1 or not len(source_rows)
    ):
        raise ReplayCompareError(
            f"source_row_index must be a non-empty vector in {path}"
        )
    if source_rows is not None and policy_seq.shape != source_rows.shape:
        raise ReplayCompareError(
            f"policy_seq shape {policy_seq.shape} does not match "
            f"source_row_index shape {source_rows.shape} in {path}"
        )
    if reference_time is not None and reference_time.shape != policy_seq.shape:
        raise ReplayCompareError(
            f"control_time_s shape {reference_time.shape} does not match "
            f"policy_seq shape {policy_seq.shape} in {path}"
        )

    rate_hz = float(metadata.get("rate_hz", 0.0))
    if rate_hz <= 0.0 and reference_time is not None and len(reference_time) > 1:
        deltas = np.diff(reference_time)
        positive = deltas[np.isfinite(deltas) & (deltas > 0.0)]
        if positive.size:
            rate_hz = 1.0 / float(np.median(positive))
    if rate_hz <= 0.0:
        rate_hz = 50.0

    return PreparedReference(
        path=path,
        source_row_index=source_rows,
        policy_seq=policy_seq,
        rate_hz=rate_hz,
        metadata=metadata,
        control_time_s=reference_time,
    )


def _prepared_source_rows(
    prepared: PreparedReference,
    source: CsvQpos,
) -> tuple[np.ndarray, str]:
    """Return original-recording rows for prepared 50 Hz frames.

    Qpos-track artifacts persist these rows directly.  Legacy ``change_ckpt``
    artifacts persist only policy_seq/control_time_s because their actual
    encoder stream was the 1024-D reference_motion.  For visualization only,
    recover the original qpos sampled at each policy boundary.
    """

    if source.policy_seq is None:
        raise ReplayCompareError(
            f"source CSV has no policy_seq column: {source.path}"
        )

    if prepared.source_row_index is not None:
        source_rows = prepared.source_row_index
        mapping = "stored qpos-track source_row_index"
    else:
        if len(source.policy_seq) > 1 and np.any(np.diff(source.policy_seq) < 0):
            raise ReplayCompareError(
                f"source policy_seq is not monotonic in {source.path}; cannot "
                "recover legacy 50 Hz policy-boundary rows"
            )
        first_row_by_policy: dict[int, int] = {}
        for row_index, sequence in enumerate(source.policy_seq):
            first_row_by_policy.setdefault(int(sequence), row_index)
        missing = [
            int(sequence)
            for sequence in prepared.policy_seq
            if int(sequence) not in first_row_by_policy
        ]
        if missing:
            raise ReplayCompareError(
                "legacy prepared reference contains policy_seq values absent "
                f"from {source.path}; examples: {missing[:8]}"
            )
        source_rows = np.asarray(
            [
                first_row_by_policy[int(sequence)]
                for sequence in prepared.policy_seq
            ],
            dtype=np.int64,
        )
        mapping = "legacy policy_seq recovered at first source row"

        # Legacy control_time_s is the first row time of each policy group.
        # Check it when present so a different recording with coincidentally
        # overlapping policy_seq values is not silently accepted.
        if prepared.control_time_s is not None:
            source_times = source.control_time_s[source_rows]
            finite = np.isfinite(source_times) & np.isfinite(
                prepared.control_time_s
            )
            if np.any(finite):
                source_deltas = np.diff(source.control_time_s)
                positive = source_deltas[
                    np.isfinite(source_deltas) & (source_deltas > 0.0)
                ]
                tolerance = (
                    max(1e-6, 0.51 * float(np.median(positive)))
                    if positive.size
                    else 1e-6
                )
                errors = np.abs(
                    source_times[finite] - prepared.control_time_s[finite]
                )
                if np.any(errors > tolerance):
                    raise ReplayCompareError(
                        "legacy prepared control_time_s does not match the "
                        f"policy-boundary rows in {source.path}; max error "
                        f"{float(np.max(errors)):.6g}s exceeds "
                        f"{tolerance:.6g}s"
                    )

    if np.any(source_rows < 0) or np.any(source_rows >= len(source.qpos)):
        bad = int(
            np.flatnonzero(
                (source_rows < 0) | (source_rows >= len(source.qpos))
            )[0]
        )
        raise ReplayCompareError(
            f"prepared reference frame {bad} points to source data row "
            f"{source_rows[bad]}, outside [0, {len(source.qpos) - 1}]"
        )
    source_policy = source.policy_seq[source_rows]
    if not np.array_equal(source_policy, prepared.policy_seq):
        bad = int(np.flatnonzero(source_policy != prepared.policy_seq)[0])
        raise ReplayCompareError(
            f"prepared reference/source mismatch at frame {bad}: source row "
            f"{source_rows[bad]} has policy_seq {source_policy[bad]}, expected "
            f"{prepared.policy_seq[bad]}"
        )
    return source_rows, mapping


def _reference_path_from_sidecars(
    rollout_dir: Path,
    run_metadata: dict[str, Any],
    manifest: dict[str, Any],
    prepared: PreparedReference | None,
    *,
    relocation_roots: Sequence[Path] | None = None,
) -> Path | None:
    candidates: list[Any] = [
        run_metadata.get("source_recording"),
        run_metadata.get("source_csv"),
        manifest.get("recording"),
    ]
    if prepared is not None:
        candidates.extend(
            [
                prepared.metadata.get("source_csv"),
                (
                    prepared.metadata.get("source", {}).get("csv_path")
                    if isinstance(prepared.metadata.get("source"), dict)
                    else None
                ),
            ]
        )
    recording_names: set[str] = set()
    for candidate in candidates:
        if not candidate:
            continue
        raw_path = Path(str(candidate)).expanduser()
        recording_name = (
            raw_path.parent.name
            if raw_path.name.lower() == "data.csv"
            else raw_path.name
        )
        if recording_name:
            recording_names.add(recording_name)
        possible_paths = (
            [raw_path.resolve()]
            if raw_path.is_absolute()
            else [
                (rollout_dir / raw_path).resolve(),
                (REPO_ROOT / raw_path).resolve(),
                (Path.cwd() / raw_path).resolve(),
            ]
        )
        seen: set[Path] = set()
        for path in possible_paths:
            if path in seen:
                continue
            seen.add(path)
            csv_path = path / "data.csv" if path.is_dir() else path
            if csv_path.is_file():
                return path

    # Result directories are often kept after the source recordings have been
    # reorganized into date subdirectories.  Recover a stale sidecar path only
    # when its recording basename has one unambiguous match below sample_data.
    roots = (
        tuple(relocation_roots)
        if relocation_roots is not None
        else (REPO_ROOT / "sample_data",)
    )
    relocated: set[Path] = set()
    for root in roots:
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            continue
        for recording_name in recording_names:
            try:
                matches = root.rglob(recording_name)
                for match in matches:
                    if match.is_dir() and (match / "data.csv").is_file():
                        relocated.add(match.resolve())
            except OSError:
                continue
    if len(relocated) == 1:
        recovered = next(iter(relocated))
        print(
            "[ghost-replay] source recording sidecar path is stale; "
            f"using unique relocated match: {recovered}"
        )
        return recovered
    if len(relocated) > 1:
        rendered = "\n  - ".join(str(path) for path in sorted(relocated))
        print(
            "[ghost-replay] warning: source recording path is stale and its "
            "basename is ambiguous; pass --reference explicitly. Matches:\n  - "
            + rendered
        )
    return None


def _playback_times(target: CsvQpos, run_metadata: dict[str, Any]) -> np.ndarray:
    for candidate in (target.control_time_s, target.mujoco_time_s):
        if (
            np.isfinite(candidate).all()
            and len(candidate) > 1
            and np.all(np.diff(candidate) >= -1e-12)
            and float(candidate[-1] - candidate[0]) > 0.0
        ):
            return candidate.copy()

    control_dt = float(run_metadata.get("control_dt", 0.0))
    if control_dt <= 0.0:
        control_dt = 0.005
    return np.arange(len(target.qpos), dtype=np.float64) * control_dt


def _validate_source_layout(target: CsvQpos, source: CsvQpos) -> None:
    if target.qpos_names != source.qpos_names:
        mismatch = next(
            (
                index
                for index, (target_name, source_name) in enumerate(
                    zip(target.qpos_names, source.qpos_names)
                )
                if target_name != source_name
            ),
            None,
        )
        if mismatch is None:
            detail = (
                f"different widths ({len(target.qpos_names)} versus "
                f"{len(source.qpos_names)})"
            )
        else:
            detail = (
                f"qpos{mismatch}: {target.qpos_names[mismatch]!r} versus "
                f"{source.qpos_names[mismatch]!r}"
            )
        raise ReplayCompareError(
            "rollout and source CSV qpos layouts differ; refusing to draw a "
            f"mislabelled ghost ({detail})"
        )


def _validate_target_sample_indices(target: CsvQpos) -> None:
    expected = np.arange(len(target.sample_index), dtype=np.int64)
    if not np.array_equal(target.sample_index, expected):
        mismatch = int(np.flatnonzero(target.sample_index != expected)[0])
        raise ReplayCompareError(
            f"rollout sample_index is not contiguous at CSV row {mismatch}: "
            f"got {target.sample_index[mismatch]}, expected {mismatch}"
        )


def build_ghost_track(
    *,
    target: CsvQpos,
    source: CsvQpos,
    playback_time_s: np.ndarray,
    run_metadata: dict[str, Any],
    manifest: dict[str, Any],
    prepared: PreparedReference | None,
    mode: str,
    reference_time_offset_s: float,
) -> GhostTrack:
    """Construct one source qpos for every rollout replay sample."""

    _validate_source_layout(target, source)
    _validate_target_sample_indices(target)
    if source.policy_seq is None:
        raise ReplayCompareError(
            f"source CSV has no policy_seq column: {source.path}"
        )

    if mode == "reference":
        if prepared is None:
            raise ReplayCompareError(
                "--ghost-mode reference requires prepared_reference.npz"
            )
        source_rows, row_mapping = _prepared_source_rows(prepared, source)

        elapsed = playback_time_s - float(playback_time_s[0])
        reference_index = np.floor(
            (elapsed + reference_time_offset_s) * prepared.rate_hz + 1e-9
        ).astype(np.int64)
        reference_index = np.clip(reference_index, 0, len(source_rows) - 1)
        mapped_source_rows = source_rows[reference_index]
        policy_seq = prepared.policy_seq[reference_index]
        return GhostTrack(
            qpos=source.qpos[mapped_source_rows].copy(),
            source_row_index=mapped_source_rows,
            reference_frame_index=reference_index,
            policy_seq=policy_seq,
            mode=mode,
            description=(
                "constructed 50 Hz recorded qpos sampled at policy boundaries "
                f"(held between reference ticks; {row_mapping})"
            ),
        )

    if mode != "source":
        raise ReplayCompareError(f"unsupported ghost mode: {mode}")

    initial_row = run_metadata.get("initial_row_index")
    if initial_row is None:
        initialization = manifest.get("initialization", {})
        if isinstance(initialization, dict):
            initial_row = initialization.get("actual_start_source_row_index")
    if initial_row is None:
        raise ReplayCompareError(
            "--ghost-mode source requires initial_row_index in run_metadata.json"
        )
    initial_row = int(initial_row)

    rows_per_control = run_metadata.get("source_rows_per_control")
    if rows_per_control is None:
        control_dt = float(run_metadata.get("control_dt", 0.0))
        source_dt = float(run_metadata.get("source_dt", 0.0))
        if control_dt > 0.0 and source_dt > 0.0:
            ratio = control_dt / source_dt
            rounded = int(round(ratio))
            if math.isclose(ratio, rounded, rel_tol=0.0, abs_tol=1e-9):
                rows_per_control = rounded
    if rows_per_control is None or int(rows_per_control) <= 0:
        raise ReplayCompareError(
            "--ghost-mode source requires a positive source_rows_per_control "
            "in run_metadata.json"
        )
    rows_per_control = int(rows_per_control)

    mapped_source_rows = np.minimum(
        initial_row + target.sample_index * rows_per_control,
        len(source.qpos) - 1,
    )
    if np.any(mapped_source_rows < 0):
        raise ReplayCompareError(
            f"invalid initial source row {initial_row} for {source.path}"
        )
    policy_seq = source.policy_seq[mapped_source_rows]
    if target.policy_seq is not None:
        mismatch = np.flatnonzero(target.policy_seq != policy_seq)
        if len(mismatch):
            row = int(mismatch[0])
            raise ReplayCompareError(
                "source-row reconstruction disagrees with rollout policy_seq at "
                f"sample {row}: reconstructed {policy_seq[row]}, rollout "
                f"{target.policy_seq[row]}. Check that --reference points to the "
                "recording used for this rollout."
            )

    return GhostTrack(
        qpos=source.qpos[mapped_source_rows].copy(),
        source_row_index=mapped_source_rows,
        reference_frame_index=np.full(len(target.qpos), -1, dtype=np.int64),
        policy_seq=policy_seq,
        mode=mode,
        description=(
            "source qpos vector sampled at the simulator source-row clock "
            "(not the optional constructed 50 Hz sample-and-hold ghost)"
        ),
    )


def _joint_qpos_width(joint_type: int) -> int:
    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
        return 7
    if joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
        return 4
    return 1


def inspect_robot_visual(model: mujoco.MjModel, geom_group: int) -> RobotVisual:
    pelvis_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
    )
    if pelvis_id < 0:
        free_joints = np.flatnonzero(
            np.asarray(model.jnt_type) == int(mujoco.mjtJoint.mjJNT_FREE)
        )
        if not len(free_joints):
            raise ReplayCompareError(
                "cannot identify G1: model has neither a pelvis body nor a free joint"
            )
        pelvis_id = int(model.jnt_bodyid[int(free_joints[0])])

    robot_bodies: set[int] = {int(pelvis_id)}
    changed = True
    while changed:
        changed = False
        for body_id in range(1, model.nbody):
            if (
                body_id not in robot_bodies
                and int(model.body_parentid[body_id]) in robot_bodies
            ):
                robot_bodies.add(body_id)
                changed = True

    hand_bodies: set[int] = set()
    for body_id in robot_bodies:
        name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, body_id
        )
        if name and "_hand_" in name:
            hand_bodies.add(body_id)

    all_robot_geoms = [
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in robot_bodies
    ]
    geom_ids = [
        geom_id
        for geom_id in all_robot_geoms
        if int(model.geom_group[geom_id]) == geom_group
    ]
    if not geom_ids:
        geom_ids = all_robot_geoms
        print(
            f"[ghost-replay] warning: robot has no geom group {geom_group}; "
            "using every robot geom"
        )
    if not geom_ids:
        raise ReplayCompareError("identified G1 body tree contains no geoms")

    root_qpos: list[int] = []
    body_joint_qpos: list[int] = []
    for joint_id in range(model.njnt):
        body_id = int(model.jnt_bodyid[joint_id])
        if body_id not in robot_bodies:
            continue
        qpos_address = int(model.jnt_qposadr[joint_id])
        width = _joint_qpos_width(int(model.jnt_type[joint_id]))
        indices = list(range(qpos_address, qpos_address + width))
        if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
            root_qpos.extend(indices)
        elif body_id not in hand_bodies:
            body_joint_qpos.extend(indices)

    if len(root_qpos) != 7:
        raise ReplayCompareError(
            f"expected one 7-DOF G1 floating base, found {len(root_qpos)} qpos"
        )
    if len(body_joint_qpos) != 29:
        raise ReplayCompareError(
            f"expected 29 non-hand G1 joint qpos, found {len(body_joint_qpos)}"
        )

    return RobotVisual(
        pelvis_body_id=int(pelvis_id),
        robot_body_ids=frozenset(robot_bodies),
        hand_body_ids=frozenset(hand_bodies),
        geom_ids=tuple(geom_ids),
        root_qpos_indices=np.asarray(root_qpos, dtype=np.int64),
        body_joint_qpos_indices=np.asarray(body_joint_qpos, dtype=np.int64),
    )


def _scene_data_id(model: mujoco.MjModel, geom_id: int) -> int:
    """Return the render-time data id used by ``mjv_addGeoms``.

    MuJoCo stores the asset id in ``model.geom_dataid``.  For mesh and SDF
    scene geoms, however, the renderer uses ``2 * asset_id`` for the original
    mesh and ``2 * asset_id + 1`` for its convex hull.  The ghost always draws
    the original visual mesh, matching a normal group-1 model geom.
    """

    data_id = int(model.geom_dataid[geom_id])
    if data_id < 0:
        return -1
    geom_type = int(model.geom_type[geom_id])
    if geom_type in (
        int(mujoco.mjtGeom.mjGEOM_MESH),
        int(mujoco.mjtGeom.mjGEOM_SDF),
    ):
        return 2 * data_id
    return data_id


def _scene_uses_texture_coordinates(
    model: mujoco.MjModel, geom_id: int
) -> int:
    data_id = int(model.geom_dataid[geom_id])
    geom_type = int(model.geom_type[geom_id])
    if (
        data_id >= 0
        and geom_type
        in (
            int(mujoco.mjtGeom.mjGEOM_MESH),
            int(mujoco.mjtGeom.mjGEOM_SDF),
        )
        and int(model.mesh_texcoordadr[data_id]) >= 0
    ):
        return 1
    return 0


def fill_ghost_scene(
    *,
    scene: mujoco.MjvScene,
    model: mujoco.MjModel,
    ghost_data: mujoco.MjData,
    visual: RobotVisual,
    alpha: float,
    offset: np.ndarray,
    show_hands: bool,
    visible: bool,
) -> int:
    """Replace ``scene`` contents with visual-only robot mesh geoms."""

    scene.ngeom = 0
    if not visible:
        return 0

    body_rgba = np.asarray([0.05, 0.90, 1.00, alpha], dtype=np.float32)
    # The recorded hand qpos is useful for inspection, but it is not part of
    # the 29-DOF body encoder reference.  Use amber to make that distinction
    # visible instead of silently presenting it as the same signal.
    hand_rgba = np.asarray([1.00, 0.55, 0.05, alpha], dtype=np.float32)

    count = 0
    for geom_id in visual.geom_ids:
        body_id = int(model.geom_bodyid[geom_id])
        is_hand = body_id in visual.hand_body_ids
        if is_hand and not show_hands:
            continue
        if count >= int(scene.maxgeom):
            raise ReplayCompareError(
                f"user scene capacity {scene.maxgeom} is too small for ghost geoms"
            )

        geom = scene.geoms[count]
        position = np.asarray(
            ghost_data.geom_xpos[geom_id], dtype=np.float64
        ) + offset
        mujoco.mjv_initGeom(
            geom,
            int(model.geom_type[geom_id]),
            np.asarray(model.geom_size[geom_id], dtype=np.float64),
            position,
            np.asarray(ghost_data.geom_xmat[geom_id], dtype=np.float64),
            hand_rgba if is_hand else body_rgba,
        )
        # mjv_initGeom cannot infer which model asset a custom geom uses.
        # Do not copy model.geom_dataid verbatim for a mesh: mjv_addGeoms
        # encodes original/convex-hull mesh variants in the low bit.
        geom.dataid = _scene_data_id(model, geom_id)
        geom.matid = -1
        geom.texcoord = _scene_uses_texture_coordinates(model, geom_id)
        geom.objtype = int(mujoco.mjtObj.mjOBJ_UNKNOWN)
        geom.objid = -1
        geom.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
        geom.segid = count
        geom.emission = 0.15
        geom.specular = 0.15
        geom.modelrbound = float(model.geom_rbound[geom_id])
        geom.camdist = 0.0
        geom.transparent = 1 if alpha < 1.0 else 0
        count += 1

    scene.ngeom = count
    return count


class PlaybackControls:
    """Thread-safe keyboard state shared with the viewer callback."""

    _KEY_SPACE = 32
    _KEY_G = 71
    _KEY_R = 82
    _KEY_RIGHT = 262
    _KEY_LEFT = 263

    def __init__(self, paused: bool) -> None:
        self._lock = threading.Lock()
        self.paused = paused
        self.ghost_visible = True
        self.restart_requested = False
        self.step_delta = 0
        self.changed = True

    def on_key(self, keycode: int) -> None:
        with self._lock:
            if keycode == self._KEY_SPACE:
                self.paused = not self.paused
                self.changed = True
            elif keycode == self._KEY_G:
                self.ghost_visible = not self.ghost_visible
                self.changed = True
            elif keycode == self._KEY_R:
                self.restart_requested = True
                self.paused = False
                self.changed = True
            elif keycode == self._KEY_RIGHT:
                self.step_delta += 1
                self.paused = True
                self.changed = True
            elif keycode == self._KEY_LEFT:
                self.step_delta -= 1
                self.paused = True
                self.changed = True

    def consume(self) -> tuple[bool, bool, bool, int, bool]:
        with self._lock:
            result = (
                self.paused,
                self.ghost_visible,
                self.restart_requested,
                self.step_delta,
                self.changed,
            )
            self.restart_requested = False
            self.step_delta = 0
            self.changed = False
            return result

    def set_paused(self, value: bool) -> None:
        with self._lock:
            self.paused = value
            self.changed = True

    def has_pending_change(self) -> bool:
        with self._lock:
            return self.changed or self.restart_requested or self.step_delta != 0


def _apply_replay_sample(
    *,
    sample: int,
    model: mujoco.MjModel,
    actual_data: mujoco.MjData,
    ghost_data: mujoco.MjData,
    target: CsvQpos,
    ghost: GhostTrack,
    visual: RobotVisual,
    scene: mujoco.MjvScene,
    alpha: float,
    offset: np.ndarray,
    root_mode: str,
    show_hands: bool,
    ghost_visible: bool,
) -> int:
    actual_data.qpos[:] = target.qpos[sample]
    if math.isfinite(float(target.mujoco_time_s[sample])):
        actual_data.time = float(target.mujoco_time_s[sample])
    mujoco.mj_forward(model, actual_data)

    ghost_data.qpos[:] = ghost.qpos[sample]
    if root_mode == "actual":
        ghost_data.qpos[visual.root_qpos_indices] = actual_data.qpos[
            visual.root_qpos_indices
        ]
    mujoco.mj_forward(model, ghost_data)

    return fill_ghost_scene(
        scene=scene,
        model=model,
        ghost_data=ghost_data,
        visual=visual,
        alpha=alpha,
        offset=offset,
        show_hands=show_hands,
        visible=ghost_visible,
    )


def _selected_bounds(
    playback_time_s: np.ndarray,
    start_time_s: float,
    end_time_s: float | None,
) -> tuple[int, int]:
    relative = playback_time_s - float(playback_time_s[0])
    first = int(np.searchsorted(relative, start_time_s, side="left"))
    if end_time_s is None:
        last = len(relative) - 1
    else:
        last = int(np.searchsorted(relative, end_time_s, side="right") - 1)
    first = max(0, min(first, len(relative) - 1))
    last = max(0, min(last, len(relative) - 1))
    if first > last:
        raise ReplayCompareError(
            f"empty playback interval: start={start_time_s}, end={end_time_s}"
        )
    return first, last


def _print_summary(
    *,
    rollout_dir: Path,
    target: CsvQpos,
    source: CsvQpos,
    prepared: PreparedReference | None,
    ghost: GhostTrack,
    playback_time_s: np.ndarray,
    model: mujoco.MjModel,
    visual: RobotVisual,
    first: int,
    last: int,
    root_mode: str,
    show_hands: bool,
) -> None:
    duration = float(playback_time_s[last] - playback_time_s[first])
    body_error = (
        target.qpos[first:last + 1, visual.body_joint_qpos_indices]
        - ghost.qpos[first:last + 1, visual.body_joint_qpos_indices]
    )
    print(f"[ghost-replay] rollout: {rollout_dir}")
    print(f"[ghost-replay] replay CSV: {target.path}")
    print(f"[ghost-replay] source CSV: {source.path}")
    print(
        f"[ghost-replay] ghost mode: {ghost.mode} — {ghost.description}"
    )
    if prepared is not None:
        usage = "used by ghost" if ghost.mode == "reference" else "available; not used"
        row_mapping = (
            "stored source rows"
            if prepared.source_row_index is not None
            else "source rows recovered from legacy policy_seq"
        )
        print(
            f"[ghost-replay] prepared reference: {prepared.path} "
            f"({len(prepared.policy_seq)} frames @ {prepared.rate_hz:.3f} Hz; "
            f"{row_mapping}; {usage})"
        )
    print(
        f"[ghost-replay] model: nq={model.nq}, bodies={model.nbody}, "
        f"geoms={model.ngeom}"
    )
    print(
        f"[ghost-replay] display: solid/textured=rollout qpos; "
        f"cyan=recorded body qpos; "
        f"{'amber=recorded hand qpos' if show_hands else 'ghost hands hidden'}"
    )
    print(
        f"[ghost-replay] root mode: {root_mode}; robot ghost geoms="
        f"{len(visual.geom_ids)}"
    )
    print(
        f"[ghost-replay] playback samples: {first}..{last} "
        f"({last - first + 1} rows, duration≈{duration:.3f}s)"
    )
    print(
        "[ghost-replay] 29-body-joint difference over the selected interval: "
        f"RMSE={float(np.sqrt(np.mean(body_error * body_error))):.6f} rad, "
        f"max={float(np.max(np.abs(body_error))):.6f} rad"
    )
    if root_mode == "recorded":
        print(
            "[ghost-replay] semantic note: recorded root xyz is drawn for "
            "world-space comparison, but protocol v1 did not send root xyz to "
            "the encoder."
        )
    else:
        print(
            "[ghost-replay] semantic note: ghost root qpos is replaced by the "
            "rollout root for articulation-only comparison; recorded root "
            "tracking error is therefore hidden."
        )
    if show_hands:
        print(
            "[ghost-replay] semantic note: amber hands are recorded robot qpos, "
            "not the separate hand target command."
        )


def _dry_run(
    *,
    target: CsvQpos,
    ghost: GhostTrack,
    model: mujoco.MjModel,
    visual: RobotVisual,
    first: int,
    last: int,
    alpha: float,
    offset: np.ndarray,
    root_mode: str,
    show_hands: bool,
) -> None:
    actual_data = mujoco.MjData(model)
    ghost_data = mujoco.MjData(model)
    scene = mujoco.MjvScene(model, maxgeom=max(1000, len(visual.geom_ids) + 8))
    samples = sorted({first, (first + last) // 2, last})
    for sample in samples:
        count = _apply_replay_sample(
            sample=sample,
            model=model,
            actual_data=actual_data,
            ghost_data=ghost_data,
            target=target,
            ghost=ghost,
            visual=visual,
            scene=scene,
            alpha=alpha,
            offset=offset,
            root_mode=root_mode,
            show_hands=show_hands,
            ghost_visible=True,
        )
        error = target.qpos[sample, visual.body_joint_qpos_indices] - ghost.qpos[
            sample, visual.body_joint_qpos_indices
        ]
        print(
            f"[ghost-replay] dry-run sample={sample}, "
            f"reference_frame={ghost.reference_frame_index[sample]}, "
            f"source_row={ghost.source_row_index[sample]}, "
            f"policy_seq={ghost.policy_seq[sample]}, geoms={count}, "
            f"body_joint_max_error={float(np.max(np.abs(error))):.6f} rad"
        )
    print("[ghost-replay] dry-run passed; no viewer was opened")


def run_viewer(
    *,
    args: argparse.Namespace,
    target: CsvQpos,
    ghost: GhostTrack,
    playback_time_s: np.ndarray,
    model: mujoco.MjModel,
    visual: RobotVisual,
    first: int,
    last: int,
) -> None:
    import mujoco.viewer

    actual_data = mujoco.MjData(model)
    ghost_data = mujoco.MjData(model)
    actual_data.qpos[:] = target.qpos[first]
    mujoco.mj_forward(model, actual_data)

    controls = PlaybackControls(paused=args.paused)
    offset = np.asarray(args.ghost_offset, dtype=np.float64)
    with mujoco.viewer.launch_passive(
        model,
        actual_data,
        key_callback=controls.on_key,
        show_left_ui=True,
        show_right_ui=True,
    ) as viewer:
        if viewer.user_scn is None:
            raise ReplayCompareError(
                "MuJoCo passive viewer did not expose a user scene"
            )
        viewer.cam.lookat[:] = actual_data.qpos[:3]
        viewer.cam.distance = max(2.5, float(model.stat.extent) * 0.85)
        viewer.cam.azimuth = 140.0
        viewer.cam.elevation = -18.0

        current = first
        rendered = -1
        finished = False
        deadline = time.monotonic()
        previous_visible: bool | None = None
        print(
            "[ghost-replay] keys: Space pause/resume, G ghost on/off, "
            "R restart, Left/Right step while paused"
        )

        while viewer.is_running():
            paused, visible, restart, step_delta, changed = controls.consume()
            if visible != previous_visible:
                state = "ON" if visible else "OFF"
                print(f"[ghost-replay] ghost {state}")
                previous_visible = visible

            if restart:
                current = first
                rendered = -1
                finished = False
                deadline = time.monotonic()
                print("[ghost-replay] restarted")

            if step_delta:
                current = max(first, min(last, current + step_delta))
                rendered = -1
                finished = current >= last
                deadline = time.monotonic()

            if finished and not paused:
                # Space at the held final pose restarts playback.
                current = first
                rendered = -1
                finished = False
                deadline = time.monotonic()

            if current != rendered or changed:
                with viewer.lock():
                    _apply_replay_sample(
                        sample=current,
                        model=model,
                        actual_data=actual_data,
                        ghost_data=ghost_data,
                        target=target,
                        ghost=ghost,
                        visual=visual,
                        scene=viewer.user_scn,
                        alpha=args.ghost_alpha,
                        offset=offset,
                        root_mode=args.root_mode,
                        show_hands=not args.hide_ghost_hands,
                        ghost_visible=visible,
                    )
                viewer.sync()
                rendered = current

            if paused or finished:
                time.sleep(0.01)
                deadline = time.monotonic()
                continue

            if current >= last:
                if args.exit_at_end:
                    break
                finished = True
                controls.set_paused(True)
                print(
                    f"[ghost-replay] finished at sample {current}; "
                    "holding the final pose (R restarts)"
                )
                continue

            dt = float(
                playback_time_s[current + 1] - playback_time_s[current]
            )
            if not math.isfinite(dt) or dt <= 0.0 or dt > 1.0:
                dt = 0.005
            deadline += dt / args.speed
            interrupted = False
            while viewer.is_running() and time.monotonic() < deadline:
                # Keep the wait short so keyboard state is observed promptly.
                remaining = deadline - time.monotonic()
                time.sleep(max(0.0001, min(0.002, remaining)))
                if controls.has_pending_change():
                    interrupted = True
                    break
            if interrupted:
                deadline = time.monotonic()
                continue
            current += 1

        if args.exit_at_end and viewer.is_running():
            viewer.close()
    print("[ghost-replay] viewer exited")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a change_ckpt or qpos-track rollout while drawing recorded "
            "qpos as a semi-transparent, visual-only G1."
        )
    )
    parser.add_argument(
        "rollout",
        help="checkpoint rollout directory (or its data.csv)",
    )
    parser.add_argument(
        "--reference",
        help=(
            "original recording directory/data.csv; default: source path from "
            "run_metadata.json, launch_manifest.json or prepared_reference.npz"
        ),
    )
    parser.add_argument(
        "--ghost-mode",
        choices=("reference", "source"),
        default="source",
        help=(
            "source: original recording qpos at the simulator source-row clock "
            "(default, supports change_ckpt and qpos-track); reference: held "
            "50 Hz recorded qpos sampled at prepared policy boundaries"
        ),
    )
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--ghost-alpha", type=float, default=0.32)
    parser.add_argument(
        "--ghost-offset",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 0.0),
        help="world-space display-only offset for the ghost, in metres",
    )
    parser.add_argument(
        "--root-mode",
        choices=("recorded", "actual"),
        default="recorded",
        help=(
            "recorded: show the original root pose; actual: attach the ghost to "
            "the rollout root so articulation differences are easier to see"
        ),
    )
    parser.add_argument(
        "--reference-time-offset",
        type=float,
        default=0.0,
        help=(
            "advance the nominal 50 Hz ghost clock by this many seconds "
            "(negative values delay it)"
        ),
    )
    parser.add_argument(
        "--geom-group",
        type=int,
        default=1,
        help="MuJoCo robot geom group used for the ghost mesh (default: 1)",
    )
    parser.add_argument(
        "--hide-ghost-hands",
        action="store_true",
        help="do not draw the amber recorded-hand qpos",
    )
    parser.add_argument(
        "--asset-model-root",
        help="override the live mujoco/model asset directory",
    )
    parser.add_argument("--start-time", type=float, default=0.0)
    parser.add_argument("--end-time", type=float)
    parser.add_argument("--paused", action="store_true")
    parser.add_argument(
        "--exit-at-end",
        action="store_true",
        help="close instead of holding the final pose",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate data/model/ghost geoms without opening a GUI",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(
        f"[ghost-replay] MuJoCo Python version: "
        f"{getattr(mujoco, '__version__', mujoco.mj_versionString())}"
    )
    if not str(getattr(mujoco, "__version__", "")).startswith("3.2."):
        print(
            "[ghost-replay] warning: this viewer was validated with MuJoCo 3.2; "
            "run it with .venv_replay/bin/python for the isolated replay environment"
        )
    if args.speed <= 0.0 or not math.isfinite(args.speed):
        raise ReplayCompareError("--speed must be finite and positive")
    if not 0.0 < args.ghost_alpha <= 1.0:
        raise ReplayCompareError("--ghost-alpha must be in (0, 1]")
    if not math.isfinite(args.start_time) or args.start_time < 0.0:
        raise ReplayCompareError("--start-time must be finite and non-negative")
    if args.end_time is not None and (
        not math.isfinite(args.end_time) or args.end_time < args.start_time
    ):
        raise ReplayCompareError("--end-time must be >= --start-time")
    if not math.isfinite(args.reference_time_offset):
        raise ReplayCompareError("--reference-time-offset must be finite")
    if not np.isfinite(np.asarray(args.ghost_offset, dtype=np.float64)).all():
        raise ReplayCompareError("--ghost-offset values must be finite")
    if args.ghost_mode == "source" and args.reference_time_offset != 0.0:
        raise ReplayCompareError(
            "--reference-time-offset is only meaningful with "
            "--ghost-mode reference"
        )

    rollout_dir, rollout_csv = resolve_recording(args.rollout)
    run_metadata = _load_json(rollout_dir / "run_metadata.json")
    manifest = _load_json(rollout_dir / "launch_manifest.json")
    # Legacy change_ckpt NPZ files contain the encoder's 1024-D reference
    # stream but no source_row_index or root XYZ.  Source mode does not need
    # that file: it reconstructs the original qpos row directly from simulator
    # timeline metadata.  Reference mode reads it only for policy_seq/time and
    # recovers complete qpos from the original recording.
    prepared = (
        _read_prepared_reference(rollout_dir / "prepared_reference.npz")
        if args.ghost_mode == "reference"
        else None
    )

    reference_value: str | Path | None = args.reference
    if reference_value is None:
        reference_value = _reference_path_from_sidecars(
            rollout_dir, run_metadata, manifest, prepared
        )
    if reference_value is None:
        raise ReplayCompareError(
            "cannot locate the original recording from rollout sidecars; "
            "pass --reference <recording_dir>"
        )
    _, source_csv = resolve_recording(reference_value)

    print("[ghost-replay] loading wide CSV files...")
    target = read_qpos_csv(rollout_csv)
    source = read_qpos_csv(source_csv)
    expected_samples = run_metadata.get("samples")
    if expected_samples is not None and int(expected_samples) != len(target.qpos):
        raise ReplayCompareError(
            f"run_metadata samples={expected_samples}, but {target.path} has "
            f"{len(target.qpos)} data rows"
        )
    playback_time_s = _playback_times(target, run_metadata)
    ghost = build_ghost_track(
        target=target,
        source=source,
        playback_time_s=playback_time_s,
        run_metadata=run_metadata,
        manifest=manifest,
        prepared=prepared,
        mode=args.ghost_mode,
        reference_time_offset_s=args.reference_time_offset,
    )

    staged = stage_recording_snapshot(
        rollout_dir,
        destination_parent=None,
        asset_model_root=args.asset_model_root,
    )
    try:
        print(f"[ghost-replay] loading scene: {staged.scene_path}")
        try:
            model = mujoco.MjModel.from_xml_path(str(staged.scene_path))
        except ValueError as exc:
            raise ReplayCompareError(
                f"failed to load MuJoCo scene {staged.scene_path}: {exc}"
            ) from exc
        if model.nq != target.qpos.shape[1]:
            raise ReplayCompareError(
                f"model nq={model.nq}, but replay CSV has "
                f"{target.qpos.shape[1]} qpos columns"
            )
        visual = inspect_robot_visual(model, args.geom_group)
        first, last = _selected_bounds(
            playback_time_s, args.start_time, args.end_time
        )
        _print_summary(
            rollout_dir=rollout_dir,
            target=target,
            source=source,
            prepared=prepared,
            ghost=ghost,
            playback_time_s=playback_time_s,
            model=model,
            visual=visual,
            first=first,
            last=last,
            root_mode=args.root_mode,
            show_hands=not args.hide_ghost_hands,
        )
        offset = np.asarray(args.ghost_offset, dtype=np.float64)
        if args.dry_run:
            _dry_run(
                target=target,
                ghost=ghost,
                model=model,
                visual=visual,
                first=first,
                last=last,
                alpha=args.ghost_alpha,
                offset=offset,
                root_mode=args.root_mode,
                show_hands=not args.hide_ghost_hands,
            )
        else:
            run_viewer(
                args=args,
                target=target,
                ghost=ghost,
                playback_time_s=playback_time_s,
                model=model,
                visual=visual,
                first=first,
                last=last,
            )
    finally:
        staged.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ReplayCompareError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
