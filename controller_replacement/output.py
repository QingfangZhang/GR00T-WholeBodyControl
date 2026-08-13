"""Output primitives for deterministic controller-replacement rollouts.

This module deliberately has no dependency on the simulator or controller
adapters.  It provides two complementary outputs:

* :class:`ReplayCsvWriter` preserves a source ``data.csv`` header and column
  order while replacing the state, timing, command, and policy fields produced
  by the new rollout.  A policy snapshot is held across all 400 Hz rows until
  the next 50 Hz inference updates it.
* :class:`ControllerTelemetryWriter` stores the controller-native 50 Hz data in
  a compressed NPZ.  Its ragged encoding permits different controllers (and
  diagnostics) to use different observation, history, and token shapes without
  object arrays or ``allow_pickle=True``.

``source_row_index`` always means a zero-based *data-row* index; the CSV header
is not counted.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


FORMAT_VERSION = 1
DEFAULT_LOGGING_HZ = 400.0


def _as_numeric_array(value: Any, *, name: str, dtype: np.dtype[Any] = np.dtype(np.float64)) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if dtype.kind in "fc" and not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array.copy()


def _format_float(value: Any) -> str:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError("CSV output contains a non-finite numeric value")
    return format(number, ".17g")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(_jsonable(payload), output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write a JSON object using the output module's type rules."""

    _write_json_atomic(Path(path), payload)


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 of ``path`` without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PolicySnapshot:
    """One controller inference, held over subsequent high-rate CSV rows.

    For a SONIC run, ``token``, ``last_action``, ``raw_action``, and
    ``received_dof_pos`` are mandatory.  For a non-SONIC run they are ignored
    by the legacy CSV writer, whose SONIC-only fields are explicitly zeroed.
    Controller-native non-SONIC values belong in ``policy_telemetry.npz``.
    """

    policy_seq: int
    token: np.ndarray | Sequence[float] | None = None
    last_action: np.ndarray | Sequence[float] | None = None
    raw_action: np.ndarray | Sequence[float] | None = None
    received_dof_pos: np.ndarray | Sequence[float] | None = None


def _indexed_columns(header: Sequence[str], stem: str) -> list[int]:
    pattern = re.compile(rf"^{re.escape(stem)}\[(\d+)\]$")
    indexed: dict[int, int] = {}
    for column, name in enumerate(header):
        match = pattern.match(name)
        if match is None:
            continue
        logical_index = int(match.group(1))
        if logical_index in indexed:
            raise ValueError(f"duplicate indexed column {stem}[{logical_index}]")
        indexed[logical_index] = column
    if not indexed:
        return []
    expected = set(range(max(indexed) + 1))
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        raise ValueError(f"non-contiguous {stem} columns; missing indices {missing}")
    return [indexed[index] for index in range(len(indexed))]


def _state_columns(header: Sequence[str], state: str) -> list[int]:
    pattern = re.compile(rf"^{re.escape(state)}:.*\[{re.escape(state)}(\d+)\]$")
    indexed: dict[int, int] = {}
    for column, name in enumerate(header):
        match = pattern.match(name)
        if match is None:
            continue
        logical_index = int(match.group(1))
        if logical_index in indexed:
            raise ValueError(f"duplicate {state}{logical_index} column")
        indexed[logical_index] = column
    if not indexed:
        raise ValueError(f"source CSV has no {state} columns")
    expected = set(range(max(indexed) + 1))
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        raise ValueError(f"non-contiguous {state} columns; missing indices {missing}")
    return [indexed[index] for index in range(len(indexed))]


class ReplayCsvWriter:
    """Write a replay-compatible CSV using the exact source schema.

    The source row supplies provenance fields such as ``reference_motion``.
    State, time, and caller-provided command fields are overwritten.  A SONIC
    :class:`PolicySnapshot` is copied into every logged row until a new snapshot
    is supplied (normally eight 400 Hz rows per 50 Hz policy inference).

    ``controller_family`` values beginning with ``"sonic"`` (case-insensitive)
    are treated as SONIC.  All other controllers get zero in the legacy SONIC
    policy fields and ``policy_valid=policy_token_size=0``.
    """

    def __init__(
        self,
        output_path: Path,
        source_csv: Path,
        *,
        controller_family: str,
        scene_path: Path | str | None = None,
        logging_hz: float = DEFAULT_LOGGING_HZ,
    ) -> None:
        self.output_path = Path(output_path)
        self.source_csv = Path(source_csv).expanduser().resolve()
        self.controller_family = str(controller_family)
        self.is_sonic = self.controller_family.strip().lower().startswith("sonic")
        self.scene_path = None if scene_path is None else str(Path(scene_path).expanduser().resolve())
        self.logging_hz = float(logging_hz)
        if not np.isfinite(self.logging_hz) or self.logging_hz <= 0:
            raise ValueError("logging_hz must be finite and positive")
        if self.output_path.resolve() == self.source_csv:
            raise ValueError("output_path must differ from source_csv")

        self._source_file = self.source_csv.open("r", newline="", encoding="utf-8-sig")
        self._source_reader = csv.reader(self._source_file)
        try:
            self.header = next(self._source_reader)
        except StopIteration as exc:
            self._source_file.close()
            raise ValueError("source CSV is empty") from exc
        if len(self.header) != len(set(self.header)):
            self._source_file.close()
            raise ValueError("source CSV contains duplicate column names")
        self.column = {name: index for index, name in enumerate(self.header)}
        for required in ("sample_index", "control_time_s", "mujoco_time_s"):
            if required not in self.column:
                self._source_file.close()
                raise ValueError(f"source CSV is missing required column {required!r}")

        self.qpos_columns = _state_columns(self.header, "qpos")
        self.qvel_columns = _state_columns(self.header, "qvel")
        self.token_columns = _indexed_columns(self.header, "token_state")
        self.last_action_columns = _indexed_columns(self.header, "policy_last_action_in")
        self.raw_action_columns = _indexed_columns(self.header, "policy_raw_action_out")
        self.received_columns = _indexed_columns(self.header, "policy_received_dof_pos")
        self.reference_motion_columns = _indexed_columns(
            self.header, "reference_motion"
        )
        self._indexed_cache: dict[str, list[int]] = {
            "token_state": self.token_columns,
            "policy_last_action_in": self.last_action_columns,
            "policy_raw_action_out": self.raw_action_columns,
            "policy_received_dof_pos": self.received_columns,
        }
        if self.is_sonic:
            required = ("policy_valid", "policy_seq", "policy_token_size")
            missing = [name for name in required if name not in self.column]
            if missing or not self.token_columns:
                self._source_file.close()
                raise ValueError(
                    "SONIC output requires policy_valid, policy_seq, "
                    "policy_token_size, and token_state columns; "
                    f"missing={missing}, token_capacity={len(self.token_columns)}"
                )

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{self.output_path.name}.", suffix=".tmp", dir=self.output_path.parent
        )
        self._temporary_path = Path(temporary_name)
        self._output_file = os.fdopen(fd, "w", newline="", encoding="utf-8", buffering=4 * 1024 * 1024)
        self._writer = csv.writer(self._output_file)
        self._writer.writerow(self.header)
        self._source_index = -1
        self._source_row: list[str] | None = None
        self._policy_snapshot: PolicySnapshot | None = None
        self.rows_written = 0
        self.closed = False

    @property
    def token_capacity(self) -> int:
        return len(self.token_columns)

    def __enter__(self) -> "ReplayCsvWriter":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()

    def _source_at(self, source_row_index: int) -> list[str]:
        source_row_index = int(source_row_index)
        if source_row_index < 0:
            raise ValueError("source_row_index must be non-negative")
        if source_row_index < self._source_index:
            raise ValueError("source_row_index must be monotonic (repeated indices are allowed)")
        while self._source_index < source_row_index:
            try:
                self._source_row = next(self._source_reader)
            except StopIteration as exc:
                raise IndexError(f"source data row {source_row_index} does not exist") from exc
            self._source_index += 1
        assert self._source_row is not None
        if len(self._source_row) != len(self.header):
            raise ValueError(
                f"source row {self._source_index} has {len(self._source_row)} columns, "
                f"expected {len(self.header)}"
            )
        return list(self._source_row)

    @staticmethod
    def _assign_vector(row: list[Any], columns: Sequence[int], values: Any, *, name: str, exact: bool = True) -> None:
        array = _as_numeric_array(values, name=name).reshape(-1)
        if (exact and array.size != len(columns)) or (not exact and array.size > len(columns)):
            relation = "exactly" if exact else "at most"
            raise ValueError(f"{name} has {array.size} values; schema accepts {relation} {len(columns)}")
        for column, value in zip(columns, array, strict=False):
            row[column] = _format_float(value)

    @staticmethod
    def _clear(row: list[Any], columns: Sequence[int]) -> None:
        for column in columns:
            row[column] = "0"

    def _columns_for_command(self, name: str) -> list[int]:
        if name not in self._indexed_cache:
            self._indexed_cache[name] = _indexed_columns(self.header, name)
        return self._indexed_cache[name]

    def _apply_command_fields(self, row: list[Any], command_fields: Mapping[str, Any]) -> None:
        for name, value in command_fields.items():
            if name in self.column:
                array = np.asarray(value)
                if array.ndim != 0:
                    raise ValueError(f"exact command column {name!r} requires a scalar")
                row[self.column[name]] = _format_float(array.item())
                continue
            columns = self._columns_for_command(name)
            if not columns:
                raise KeyError(f"command field {name!r} is absent from the source schema")
            self._assign_vector(row, columns, value, name=name)

    def set_policy_snapshot(self, snapshot: PolicySnapshot) -> None:
        """Set the inference snapshot held over subsequent CSV frames."""

        if int(snapshot.policy_seq) < 0:
            raise ValueError("policy_seq must be non-negative")
        if self.is_sonic:
            required = {
                "token": snapshot.token,
                "last_action": snapshot.last_action,
                "raw_action": snapshot.raw_action,
                "received_dof_pos": snapshot.received_dof_pos,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError(f"SONIC policy snapshot is missing {missing}")
            token = _as_numeric_array(snapshot.token, name="token").reshape(-1)
            if token.size > self.token_capacity:
                raise ValueError(
                    f"SONIC token has {token.size} values but data.csv capacity is {self.token_capacity}"
                )
            for name, value, columns in (
                ("last_action", snapshot.last_action, self.last_action_columns),
                ("raw_action", snapshot.raw_action, self.raw_action_columns),
                ("received_dof_pos", snapshot.received_dof_pos, self.received_columns),
            ):
                if not columns:
                    raise ValueError(f"SONIC source schema has no {name} columns")
                array = _as_numeric_array(value, name=name).reshape(-1)
                if array.size != len(columns):
                    raise ValueError(
                        f"SONIC {name} has {array.size} values but data.csv capacity is {len(columns)}"
                    )
            snapshot = PolicySnapshot(
                policy_seq=int(snapshot.policy_seq),
                token=token,
                last_action=_as_numeric_array(snapshot.last_action, name="last_action").reshape(-1),
                raw_action=_as_numeric_array(snapshot.raw_action, name="raw_action").reshape(-1),
                received_dof_pos=_as_numeric_array(
                    snapshot.received_dof_pos, name="received_dof_pos"
                ).reshape(-1),
            )
        else:
            snapshot = PolicySnapshot(policy_seq=int(snapshot.policy_seq))
        self._policy_snapshot = snapshot

    def _apply_policy(self, row: list[Any]) -> None:
        snapshot = self._policy_snapshot
        if snapshot is not None and "policy_seq" in self.column:
            row[self.column["policy_seq"]] = str(snapshot.policy_seq)

        policy_columns = (
            *self.token_columns,
            *self.last_action_columns,
            *self.raw_action_columns,
            *self.received_columns,
        )
        self._clear(row, policy_columns)
        if not self.is_sonic:
            if "policy_valid" in self.column:
                row[self.column["policy_valid"]] = "0"
            if "policy_token_size" in self.column:
                row[self.column["policy_token_size"]] = "0"
            return

        if snapshot is None:
            raise RuntimeError("set a SONIC policy snapshot before writing its first CSV frame")
        assert snapshot.token is not None
        assert snapshot.last_action is not None
        assert snapshot.raw_action is not None
        assert snapshot.received_dof_pos is not None
        row[self.column["policy_valid"]] = "1"
        row[self.column["policy_token_size"]] = str(np.asarray(snapshot.token).size)
        self._assign_vector(row, self.token_columns, snapshot.token, name="token", exact=False)
        self._assign_vector(row, self.last_action_columns, snapshot.last_action, name="last_action")
        self._assign_vector(row, self.raw_action_columns, snapshot.raw_action, name="raw_action")
        self._assign_vector(
            row, self.received_columns, snapshot.received_dof_pos, name="received_dof_pos"
        )

    def _apply_reference_motion(
        self,
        row: list[Any],
        reference_motion: Any | None,
        *,
        clear_when_none: bool,
    ) -> None:
        """Write the exact controller input reference or clear legacy fields."""

        if reference_motion is None and not clear_when_none:
            return
        self._clear(row, self.reference_motion_columns)
        if reference_motion is None:
            size = 0
        else:
            array = _as_numeric_array(
                reference_motion, name="reference_motion"
            ).reshape(-1)
            if array.size > len(self.reference_motion_columns):
                raise ValueError(
                    "reference_motion has "
                    f"{array.size} values but data.csv capacity is "
                    f"{len(self.reference_motion_columns)}"
                )
            self._assign_vector(
                row,
                self.reference_motion_columns,
                array,
                name="reference_motion",
                exact=False,
            )
            size = int(array.size)
        if "policy_reference_motion_size" in self.column:
            row[self.column["policy_reference_motion_size"]] = str(size)

    def write_frame(
        self,
        *,
        source_row_index: int,
        sample_index: int,
        control_time_s: float,
        mujoco_time_s: float,
        qpos: Any,
        qvel: Any,
        command_fields: Mapping[str, Any] | None = None,
        policy_snapshot: PolicySnapshot | None = None,
        reference_motion: Any | None = None,
        clear_reference_motion: bool = False,
    ) -> None:
        """Write one state sample; omitted ``policy_snapshot`` means hold last."""

        if self.closed:
            raise RuntimeError("cannot write to a closed ReplayCsvWriter")
        if policy_snapshot is not None:
            self.set_policy_snapshot(policy_snapshot)
        row: list[Any] = self._source_at(source_row_index)
        if self.scene_path is not None and "scene_path" in self.column:
            row[self.column["scene_path"]] = self.scene_path
        row[self.column["sample_index"]] = str(int(sample_index))
        row[self.column["control_time_s"]] = _format_float(control_time_s)
        row[self.column["mujoco_time_s"]] = _format_float(mujoco_time_s)
        self._assign_vector(row, self.qpos_columns, qpos, name="qpos")
        self._assign_vector(row, self.qvel_columns, qvel, name="qvel")
        if command_fields:
            self._apply_command_fields(row, command_fields)
        self._apply_reference_motion(
            row,
            reference_motion,
            clear_when_none=bool(clear_reference_motion),
        )
        # Apply policy last so a generic command cannot accidentally leave stale
        # SONIC internals in a non-SONIC run.
        self._apply_policy(row)
        self._writer.writerow(row)
        self.rows_written += 1

    def close(self) -> None:
        """Atomically publish the completed CSV; calling twice is harmless."""

        if self.closed:
            return
        self._output_file.flush()
        os.fsync(self._output_file.fileno())
        self._output_file.close()
        self._source_file.close()
        os.replace(self._temporary_path, self.output_path)
        self.closed = True

    def abort(self) -> None:
        """Discard an incomplete CSV."""

        if self.closed:
            return
        self._output_file.close()
        self._source_file.close()
        self._temporary_path.unlink(missing_ok=True)
        self.closed = True


@dataclass(frozen=True)
class TelemetryRecord:
    """Controller-native data for one policy inference."""

    policy_seq: int
    policy_time_s: float
    source_row_index: int
    warmup: bool
    evaluation: bool
    reference_index: int = -1
    observation: Any | None = None
    history: Any | None = None
    token: Any | None = None
    last_action: Any | None = None
    raw_action: Any | None = None
    action: Any | None = None
    q_target: Any | None = None
    torque: Any | None = None
    torque_saturation: Any | None = None
    reference: Any | None = None
    robot_q: Any | None = None
    robot_dq: Any | None = None
    root_qpos: Any | None = None
    kp: Any | None = None
    kd: Any | None = None
    extra_arrays: Mapping[str, Any] = field(default_factory=dict)


_TELEMETRY_ARRAY_FIELDS = tuple(
    item.name
    for item in fields(TelemetryRecord)
    if item.name
    not in {
        "policy_seq",
        "policy_time_s",
        "source_row_index",
        "warmup",
        "evaluation",
        "reference_index",
        "extra_arrays",
    }
)


def _encode_ragged(values: Sequence[np.ndarray | None], *, dtype: np.dtype[Any]) -> dict[str, np.ndarray]:
    data_parts: list[np.ndarray] = []
    offsets = [0]
    shape_data: list[int] = []
    shape_offsets = [0]
    present: list[bool] = []
    for value in values:
        if value is None:
            present.append(False)
            offsets.append(offsets[-1])
            shape_offsets.append(shape_offsets[-1])
            continue
        array = np.asarray(value, dtype=dtype)
        if dtype.kind in "fc" and not np.isfinite(array).all():
            raise ValueError("telemetry contains non-finite values")
        present.append(True)
        flat = array.reshape(-1)
        data_parts.append(flat)
        offsets.append(offsets[-1] + flat.size)
        shape_data.extend(array.shape)
        shape_offsets.append(shape_offsets[-1] + array.ndim)
    data = np.concatenate(data_parts) if data_parts else np.empty(0, dtype=dtype)
    return {
        "data": data,
        "offsets": np.asarray(offsets, dtype=np.int64),
        "shape_data": np.asarray(shape_data, dtype=np.int64),
        "shape_offsets": np.asarray(shape_offsets, dtype=np.int64),
        "present": np.asarray(present, dtype=np.bool_),
    }


def decode_ragged(archive: Mapping[str, np.ndarray], field_name: str, index: int) -> np.ndarray | None:
    """Decode one ragged item from a telemetry NPZ/archive mapping."""

    present = archive[f"{field_name}__present"]
    index = int(index)
    if index < 0:
        index += len(present)
    if index < 0 or index >= len(present):
        raise IndexError(index)
    if not bool(present[index]):
        return None
    offsets = archive[f"{field_name}__offsets"]
    shape_offsets = archive[f"{field_name}__shape_offsets"]
    shape_data = archive[f"{field_name}__shape_data"]
    shape = tuple(int(item) for item in shape_data[shape_offsets[index] : shape_offsets[index + 1]])
    data = archive[f"{field_name}__data"][offsets[index] : offsets[index + 1]]
    return data.reshape(shape)


class ControllerTelemetryWriter:
    """Collect and atomically save controller-native 50 Hz telemetry."""

    def __init__(
        self,
        output_path: Path,
        *,
        controller_name: str,
        controller_family: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.output_path = Path(output_path)
        self.controller_name = str(controller_name)
        self.controller_family = str(controller_family)
        self.metadata = dict(metadata or {})
        self._records: list[TelemetryRecord] = []
        self.closed = False

    def __enter__(self) -> "ControllerTelemetryWriter":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if exc_type is None:
            self.close()

    def append(self, record: TelemetryRecord) -> None:
        if self.closed:
            raise RuntimeError("cannot append to a closed ControllerTelemetryWriter")
        if record.warmup and record.evaluation:
            raise ValueError("warmup and evaluation masks cannot both be true")
        if int(record.policy_seq) < 0 or int(record.source_row_index) < 0:
            raise ValueError("policy_seq and source_row_index must be non-negative")
        if not np.isfinite(float(record.policy_time_s)):
            raise ValueError("policy_time_s must be finite")

        copied: dict[str, Any] = {}
        for name in _TELEMETRY_ARRAY_FIELDS:
            value = getattr(record, name)
            dtype = np.dtype(np.bool_) if name == "torque_saturation" else np.dtype(np.float64)
            copied[name] = None if value is None else _as_numeric_array(value, name=name, dtype=dtype)
        extra_arrays: dict[str, np.ndarray] = {}
        for name, value in record.extra_arrays.items():
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
                raise ValueError(
                    f"extra telemetry name {name!r} must match [A-Za-z][A-Za-z0-9_]*"
                )
            if name in _TELEMETRY_ARRAY_FIELDS:
                raise ValueError(f"extra telemetry name {name!r} collides with a standard field")
            extra_arrays[name] = _as_numeric_array(value, name=f"extra_arrays[{name}]")
        self._records.append(
            TelemetryRecord(
                policy_seq=int(record.policy_seq),
                policy_time_s=float(record.policy_time_s),
                source_row_index=int(record.source_row_index),
                warmup=bool(record.warmup),
                evaluation=bool(record.evaluation),
                reference_index=int(record.reference_index),
                extra_arrays=extra_arrays,
                **copied,
            )
        )

    def close(self) -> None:
        if self.closed:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        records = self._records
        payload: dict[str, np.ndarray] = {
            "format_version": np.asarray(FORMAT_VERSION, dtype=np.int64),
            "controller_name": np.asarray(self.controller_name),
            "controller_family": np.asarray(self.controller_family),
            "metadata_json": np.asarray(
                json.dumps(_jsonable(self.metadata), ensure_ascii=False, sort_keys=True)
            ),
            "policy_seq": np.asarray([record.policy_seq for record in records], dtype=np.int64),
            "policy_time_s": np.asarray(
                [record.policy_time_s for record in records], dtype=np.float64
            ),
            "source_row_index": np.asarray(
                [record.source_row_index for record in records], dtype=np.int64
            ),
            "reference_index": np.asarray(
                [record.reference_index for record in records], dtype=np.int64
            ),
            "warmup_mask": np.asarray([record.warmup for record in records], dtype=np.bool_),
            "evaluation_mask": np.asarray(
                [record.evaluation for record in records], dtype=np.bool_
            ),
        }
        for name in _TELEMETRY_ARRAY_FIELDS:
            dtype = np.dtype(np.bool_) if name == "torque_saturation" else np.dtype(np.float64)
            encoded = _encode_ragged([getattr(record, name) for record in records], dtype=dtype)
            for suffix, value in encoded.items():
                payload[f"{name}__{suffix}"] = value

        extra_names = sorted({name for record in records for name in record.extra_arrays})
        payload["extra_names"] = np.asarray(extra_names, dtype=np.str_)
        for name in extra_names:
            encoded = _encode_ragged(
                [record.extra_arrays.get(name) for record in records], dtype=np.dtype(np.float64)
            )
            for suffix, value in encoded.items():
                payload[f"extra_{name}__{suffix}"] = value

        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{self.output_path.name}.", suffix=".tmp", dir=self.output_path.parent
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as output:
                np.savez_compressed(output, **payload)
            os.replace(temporary_path, self.output_path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
        self.closed = True


def write_data_schema(
    path: Path,
    *,
    header: Sequence[str],
    controller_family: str,
    source_csv: Path | None = None,
    logging_hz: float = DEFAULT_LOGGING_HZ,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``data_schema.json`` describing the compatible CSV contract."""

    header = list(header)
    header_digest = hashlib.sha256("\x1f".join(header).encode("utf-8")).hexdigest()
    is_sonic = str(controller_family).strip().lower().startswith("sonic")
    groups = {
        "qpos": [name for name in header if name.startswith("qpos:")],
        "qvel": [name for name in header if name.startswith("qvel:")],
        "token_state": [name for name in header if name.startswith("token_state[")],
        "policy_last_action_in": [
            name for name in header if name.startswith("policy_last_action_in[")
        ],
        "policy_raw_action_out": [
            name for name in header if name.startswith("policy_raw_action_out[")
        ],
        "policy_received_dof_pos": [
            name for name in header if name.startswith("policy_received_dof_pos[")
        ],
    }
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "source_csv": None if source_csv is None else str(Path(source_csv).expanduser().resolve()),
        "logging_hz": float(logging_hz),
        "controller_family": str(controller_family),
        "column_count": len(header),
        "header_sha256": header_digest,
        "header": header,
        "groups": {name: {"size": len(columns), "columns": columns} for name, columns in groups.items()},
        "policy_columns": {
            "mode": "sonic_native" if is_sonic else "legacy_sonic_fields_zero",
            "high_rate_semantics": "latest 50 Hz policy snapshot held on 400 Hz state rows",
        },
    }
    if extra:
        payload["extra"] = dict(extra)
    _write_json_atomic(Path(path), payload)
    return payload


def write_run_manifest(
    path: Path,
    *,
    controller_name: str,
    controller_family: str,
    source_recording: Path,
    reference_mode: str,
    root_assist: str,
    rates_hz: Mapping[str, float],
    model_paths: Mapping[str, Path] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a reproducibility manifest, including model hashes."""

    if root_assist not in {"none", "xy"}:
        raise ValueError("root_assist must be 'none' or 'xy'")
    normalized_rates: dict[str, float] = {}
    for name, value in rates_hz.items():
        rate = float(value)
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError(f"rate {name!r} must be finite and positive")
        normalized_rates[str(name)] = rate
    models: dict[str, dict[str, Any]] = {}
    for name, model_path in (model_paths or {}).items():
        resolved = Path(model_path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        models[str(name)] = {
            "path": str(resolved),
            "size_bytes": resolved.stat().st_size,
            "sha256": sha256_file(resolved),
        }
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "controller_name": str(controller_name),
        "controller_family": str(controller_family),
        "source_recording": str(Path(source_recording).expanduser().resolve()),
        "reference_mode": str(reference_mode),
        "root_assist": root_assist,
        "rates_hz": normalized_rates,
        "models": models,
        "outputs": {
            "data_csv": "source-schema-compatible 400 Hz physical trajectory",
            "policy_telemetry_npz": "controller-native policy-rate telemetry",
            "source_timeline_npz": "phase-matched 400 Hz source qpos/provenance",
            "contact_telemetry_npz": "400 Hz contact counts and normal-force summary",
            "prepared_reference_npz": "controller-neutral 50 Hz reference provenance",
        },
    }
    if extra:
        payload["extra"] = dict(extra)
    _write_json_atomic(Path(path), payload)
    return payload


__all__ = [
    "ControllerTelemetryWriter",
    "DEFAULT_LOGGING_HZ",
    "FORMAT_VERSION",
    "PolicySnapshot",
    "ReplayCsvWriter",
    "TelemetryRecord",
    "decode_ragged",
    "sha256_file",
    "write_data_schema",
    "write_json_atomic",
    "write_run_manifest",
]
