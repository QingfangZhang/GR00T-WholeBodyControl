"""I/O helpers for the qpos-track checkpoint-comparison MuJoCo simulator.

This module intentionally lives outside ``gear_sonic``.  It stages an XML-only
recording snapshot without modifying the recording, selects an initialization
row from its CSV, and writes a CSV with the same schema for the existing replay
viewer.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_recording(value: str | Path) -> tuple[Path, Path]:
    """Return ``(recording_dir, data_csv)`` for a directory or CSV argument."""
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        csv_path = path / "data.csv"
        recording_dir = path
    else:
        csv_path = path
        recording_dir = path.parent
    if not csv_path.is_file():
        raise FileNotFoundError(f"data.csv not found: {csv_path}")
    snapshot = recording_dir / "model_snapshot"
    if not snapshot.is_dir():
        raise FileNotFoundError(f"model_snapshot not found: {snapshot}")
    return recording_dir, csv_path


def find_snapshot_scene(recording_dir: Path) -> Path:
    snapshot = recording_dir / "model_snapshot"
    candidates = sorted(snapshot.rglob("scene*.xml"))
    if not candidates:
        candidates = sorted(snapshot.rglob("*.xml"))
    if not candidates:
        raise FileNotFoundError(f"no MuJoCo XML found below {snapshot}")
    exact = [path for path in candidates if path.name == "scene_43dof.xml"]
    return (exact or candidates)[0].resolve()


def _valid_asset_root(path: Path) -> bool:
    return (path / "g1" / "meshes").is_dir() and (path / "task_assets").is_dir()


def find_asset_model_root(recording_dir: Path, explicit: str | Path | None = None) -> Path:
    """Find the live ``mujoco/model`` tree used by XML-only snapshots."""
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit).expanduser())
    for ancestor in (recording_dir, *recording_dir.parents):
        candidates.append(ancestor / "mujoco" / "model")
    candidates.extend(
        [
            REPO_ROOT / "mujoco" / "model",
            REPO_ROOT / "sample_data" / "ztj" / "mujoco" / "model",
        ]
    )
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if _valid_asset_root(candidate):
            return candidate
    rendered = "\n  - ".join(str(path) for path in seen)
    raise FileNotFoundError(
        "could not find an asset model root containing g1/meshes and task_assets; "
        "pass --asset-model-root. Checked:\n  - " + rendered
    )


def _link_directory(target: Path, link: Path) -> None:
    if not target.is_dir() or link.exists() or link.is_symlink():
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target.resolve(), target_is_directory=True)


@dataclass
class StagedSnapshot:
    scene_path: Path
    snapshot_root: Path
    asset_model_root: Path
    temporary_directory: tempfile.TemporaryDirectory[str] | None = None

    def close(self) -> None:
        if self.temporary_directory is not None:
            self.temporary_directory.cleanup()
            self.temporary_directory = None


def stage_recording_snapshot(
    recording_dir: Path,
    destination_parent: Path | None,
    asset_model_root: str | Path | None = None,
) -> StagedSnapshot:
    """Copy snapshot XML and add asset links in a new output or temporary tree."""
    source_root = (recording_dir / "model_snapshot").resolve()
    source_scene = find_snapshot_scene(recording_dir)
    assets = find_asset_model_root(recording_dir, asset_model_root)

    temporary: tempfile.TemporaryDirectory[str] | None = None
    if destination_parent is None:
        temporary = tempfile.TemporaryDirectory(prefix="change_ckpt_track_task_sim_")
        destination_parent = Path(temporary.name)
    destination_parent.mkdir(parents=True, exist_ok=True)
    destination_root = destination_parent / "model_snapshot"
    destination_root.mkdir(parents=True, exist_ok=False)

    # Snapshots produced by the collector contain XML only.  Copy exactly those
    # files, then resolve large immutable assets through links.
    for source in source_root.rglob("*.xml"):
        destination = destination_root / source.relative_to(source_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    destination_scene = destination_root / source_scene.relative_to(source_root)
    scene_dir = destination_scene.parent
    mujoco_root = destination_root / "mujoco"
    model_root = mujoco_root / "model"

    _link_directory(assets / "g1" / "meshes", scene_dir / "meshes")
    _link_directory(assets / "g1" / "textures", scene_dir / "textures")
    _link_directory(assets / "g1" / "objects", scene_dir / "objects")
    _link_directory(assets / "g1" / "objects", model_root / "objects")
    for relative in (scene_dir / "task_assets", model_root / "task_assets", mujoco_root / "task_assets"):
        _link_directory(assets / "task_assets", relative)
    for name in ("adam_pro", "adam_pick", "robotwin_assets"):
        for relative in (scene_dir / name, model_root / name, mujoco_root / name):
            _link_directory(assets / name, relative)

    return StagedSnapshot(
        scene_path=destination_scene.resolve(),
        snapshot_root=destination_root.resolve(),
        asset_model_root=assets,
        temporary_directory=temporary,
    )


def _integer_cell(row: Sequence[str], index: int, label: str) -> int:
    try:
        return int(float(row[index]))
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid {label} value in CSV row: {row[index:index + 1]}") from exc


class CsvTimeline:
    """Indexed source timing/state beginning at one initialization point.

    The source has 1,579 columns.  Parsing two complete rows inside every
    200 Hz control tick costs roughly 0.4 ms and can make the wall-clock
    reference outrun MuJoCo.  Parse the file once before DDS starts, retain
    only qpos/qvel, and advance an integer during the real-time section.
    """

    def __init__(
        self,
        csv_path: Path,
        *,
        policy_seq: int | None,
        row_index: int | None,
        policy_offset: int | None = None,
    ) -> None:
        self.csv_path = csv_path
        self._file = csv_path.open("r", newline="", encoding="utf-8-sig")
        self._reader = csv.reader(self._file)
        try:
            self.header = next(self._reader)
        except StopIteration as exc:
            raise ValueError(f"empty CSV: {csv_path}") from exc
        self.column = {name: index for index, name in enumerate(self.header)}
        for required in ("policy_seq", "qpos:pelvis.floating_base_joint.x[qpos0]"):
            if required not in self.column:
                raise ValueError(f"required CSV column is missing: {required}")
        self.qpos_columns = self._indexed_columns("qpos")
        self.qvel_columns = self._indexed_columns("qvel")
        self.start_row_index, self.current = self._select(
            policy_seq, row_index, policy_offset
        )
        self.start_policy_seq = _integer_cell(
            self.current, self.column["policy_seq"], "policy_seq"
        )
        self.current_row_index = self.start_row_index
        self._position = 0
        qpos, qvel = self._state_from_row(self.current)
        qpos_states = [qpos]
        qvel_states = [qvel]
        for row in self._reader:
            if len(row) != len(self.header):
                raise ValueError(
                    f"CSV row after index {self.start_row_index} has {len(row)} fields; "
                    f"expected {len(self.header)}"
                )
            qpos, qvel = self._state_from_row(row)
            qpos_states.append(qpos)
            qvel_states.append(qvel)
        self._qpos_states = np.asarray(qpos_states, dtype=np.float64)
        self._qvel_states = np.asarray(qvel_states, dtype=np.float64)
        self.last_row_index = self.start_row_index + len(self._qpos_states) - 1
        self._file.close()
        self.exhausted = False

    def _indexed_columns(self, kind: str) -> list[int]:
        pattern = re.compile(rf"\[{kind}(\d+)\]$")
        indexed: list[tuple[int, int]] = []
        for column, name in enumerate(self.header):
            match = pattern.search(name)
            if match:
                indexed.append((int(match.group(1)), column))
        indexed.sort()
        actual = [index for index, _ in indexed]
        if actual != list(range(len(indexed))):
            raise ValueError(f"{kind} columns are not contiguous: {actual[:10]} ...")
        return [column for _, column in indexed]

    def _select(
        self,
        policy_seq: int | None,
        row_index: int | None,
        policy_offset: int | None,
    ) -> tuple[int, list[str]]:
        if sum(value is not None for value in (policy_seq, row_index, policy_offset)) > 1:
            raise ValueError(
                "--policy-seq, --policy-offset and --row-index are mutually exclusive"
            )
        if policy_offset is not None and policy_offset < 0:
            raise ValueError("--policy-offset must be non-negative")
        first_sequence: int | None = None
        unique_offset = -1
        previous_sequence: int | None = None
        for index, row in enumerate(self._reader):
            if len(row) != len(self.header):
                raise ValueError(
                    f"CSV row {index} has {len(row)} fields; expected {len(self.header)}"
                )
            sequence = _integer_cell(row, self.column["policy_seq"], "policy_seq")
            if sequence != previous_sequence:
                unique_offset += 1
                previous_sequence = sequence
            if row_index is not None and index == row_index:
                return index, row
            if policy_seq is not None and sequence == policy_seq:
                return index, row
            if policy_offset is not None and unique_offset == policy_offset:
                return index, row
            if policy_seq is None and row_index is None and policy_offset is None:
                # The recording begins part-way through a policy hold.  The first
                # sequence transition is the first complete policy boundary.
                if first_sequence is None:
                    first_sequence = sequence
                elif sequence != first_sequence:
                    return index, row
        if policy_seq is not None:
            target = f"policy_seq={policy_seq}"
        elif policy_offset is not None:
            target = f"policy_offset={policy_offset}"
        else:
            target = f"row={row_index}"
        raise ValueError(f"initialization point not found in {self.csv_path}: {target}")

    def _state_from_row(self, row: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        qpos = np.asarray([float(row[index]) for index in self.qpos_columns])
        qvel = np.asarray([float(row[index]) for index in self.qvel_columns])
        return qpos, qvel

    def state(self) -> tuple[np.ndarray, np.ndarray]:
        return self._qpos_states[self._position].copy(), self._qvel_states[self._position].copy()

    def advance(self) -> bool:
        """Advance one recorded 400 Hz row, returning false at end-of-file."""
        if self._position + 1 >= len(self._qpos_states):
            self.exhausted = True
            return False
        self._position += 1
        self.current_row_index += 1
        return True

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()


@dataclass
class _BufferedReplayRow:
    source_row_index: int
    sample_index: int
    control_time: float
    mujoco_time: float
    qpos: np.ndarray
    qvel: np.ndarray
    command: dict[str, np.ndarray]


class ReplayCsvWriter:
    """Write simulated state using the source CSV's replay-compatible schema.

    Policy internals are not available on DDS.  Consequently ``policy_valid``
    is set to zero and token/raw-action fields are cleared instead of silently
    copying stale values from the old checkpoint.  The reference motion and
    policy sequence remain copied from the source timeline.
    """

    def __init__(
        self,
        output_path: Path,
        source_csv: Path,
        header: Sequence[str],
        scene_path: Path,
        qpos_columns: Sequence[int],
        qvel_columns: Sequence[int],
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path = output_path
        self.source_csv = source_csv.resolve()
        # The workspace volume can pause for 5--15 ms on buffered writes, which
        # is enough to desynchronise a 200 Hz simulator from the wall-clock 50 Hz
        # publisher.  Spool on the local /tmp filesystem during control, then
        # copy the completed replay into the requested change_ckpt_track/data run.
        self._spool_directory = tempfile.TemporaryDirectory(
            prefix="change_ckpt_track_replay_csv_"
        )
        self._spool_path = Path(self._spool_directory.name) / output_path.name
        self._file = None
        self._writer = None
        self._buffer: list[_BufferedReplayRow] = []
        self.header = list(header)
        self.column = {name: index for index, name in enumerate(self.header)}
        self.qpos_columns = list(qpos_columns)
        self.qvel_columns = list(qvel_columns)
        self.scene_path = scene_path.resolve()
        self.rows_written = 0
        self._token_columns = self._prefix_columns("token_state[")
        self._last_action_columns = self._prefix_columns("policy_last_action_in[")
        self._raw_action_columns = self._prefix_columns("policy_raw_action_out[")
        self._received_columns = self._prefix_columns("policy_received_dof_pos[")
        self._hand_columns = {
            (side, field): self._prefix_columns(f"{side}_hand_{field}[")
            for side in ("left", "right")
            for field in ("q", "dq", "kp", "kd", "tau")
        }

    def _prefix_columns(self, prefix: str) -> list[int]:
        return [index for index, name in enumerate(self.header) if name.startswith(prefix)]

    @staticmethod
    def _assign(row: list[object], columns: Sequence[int], values: Sequence[float]) -> None:
        if len(columns) != len(values):
            raise ValueError(f"column/value length mismatch: {len(columns)} != {len(values)}")
        for column, value in zip(columns, values, strict=True):
            row[column] = format(float(value), ".17g")

    def write(
        self,
        *,
        source_row_index: int,
        sample_index: int,
        control_time: float,
        mujoco_time: float,
        qpos: np.ndarray,
        qvel: np.ndarray,
        command: dict[str, np.ndarray],
    ) -> None:
        self._buffer.append(
            _BufferedReplayRow(
                source_row_index=source_row_index,
                sample_index=sample_index,
                control_time=control_time,
                mujoco_time=mujoco_time,
                qpos=np.asarray(qpos, dtype=np.float64).copy(),
                qvel=np.asarray(qvel, dtype=np.float64).copy(),
                command={name: np.asarray(values, dtype=np.float64).copy() for name, values in command.items()},
            )
        )

    def _render(self, source_row: Sequence[str], buffered: _BufferedReplayRow) -> None:
        assert self._writer is not None
        row: list[object] = list(source_row)
        row[self.column["scene_path"]] = str(self.scene_path)
        row[self.column["sample_index"]] = buffered.sample_index
        row[self.column["control_time_s"]] = format(buffered.control_time, ".9f")
        row[self.column["mujoco_time_s"]] = format(buffered.mujoco_time, ".9f")
        self._assign(row, self.qpos_columns, buffered.qpos)
        self._assign(row, self.qvel_columns, buffered.qvel)

        # DDS exposes commanded joint targets but not encoder tokens or the raw
        # neural-network action.  Clear the unavailable fields explicitly.
        row[self.column["policy_valid"]] = 0
        row[self.column["policy_token_size"]] = 0
        for column in (
            *self._token_columns,
            *self._last_action_columns,
            *self._raw_action_columns,
        ):
            row[column] = "0"
        if self._received_columns:
            self._assign(row, self._received_columns, buffered.command["received_dof_pos"])

        for side in ("left", "right"):
            for field in ("q", "dq", "kp", "kd", "tau"):
                columns = self._hand_columns[(side, field)]
                if columns:
                    self._assign(row, columns, buffered.command[f"{side}_hand_{field}"])

        self._writer.writerow(row)
        self.rows_written += 1

    def close(self) -> None:
        if self._spool_directory is None:
            return
        self._file = self._spool_path.open(
            "w", newline="", encoding="utf-8", buffering=4 * 1024 * 1024
        )
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.header)
        with self.source_csv.open("r", newline="", encoding="utf-8-sig") as source:
            reader = csv.reader(source)
            source_header = next(reader)
            if source_header != self.header:
                raise ValueError("source CSV header changed while the rollout was running")
            source_index = -1
            source_row: list[str] | None = None
            for buffered in self._buffer:
                if buffered.source_row_index < source_index:
                    raise ValueError("buffered source row indices are not monotonic")
                while source_index < buffered.source_row_index:
                    try:
                        source_row = next(reader)
                    except StopIteration as exc:
                        raise ValueError(
                            f"source row {buffered.source_row_index} no longer exists"
                        ) from exc
                    source_index += 1
                assert source_row is not None
                self._render(source_row, buffered)
        self._file.flush()
        self._file.close()
        shutil.copy2(self._spool_path, self.output_path)
        self._buffer.clear()
        self._spool_directory.cleanup()
        self._spool_directory = None


def make_output_directory(
    output_root: Path, recording_dir: Path, policy_seq: int
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{timestamp}_{recording_dir.name}_seq{policy_seq}_sim"
    candidate = output_root.expanduser().resolve() / stem
    suffix = 1
    while candidate.exists():
        candidate = output_root.expanduser().resolve() / f"{stem}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def write_metadata(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
