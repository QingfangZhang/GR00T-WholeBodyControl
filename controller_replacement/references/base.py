"""Controller-neutral, policy-rate reference interfaces.

The reference providers expose both controller-specific views without hiding
where any component came from:

* ``sonic_regular_*`` is the ten-slot view used by a regular SONIC encoder;
* ``sonic_consecutive_*`` is the ten consecutive 50 Hz view used by the
  low-latency adapter (and is also the canonical current-frame stream); and
* ``teleopit_qpos36`` is Teleopit's ``root xyz + quat_wxyz + 29 joints`` pose.

All arrays are copied, validated, made C-contiguous, and marked read-only.  A
runner can therefore share one :class:`ReferenceSequence` across adapters
without one controller accidentally mutating another controller's input.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, ClassVar, Protocol, Sequence, runtime_checkable

import numpy as np

from change_ckpt_track.qpos_reference_data import G1_ISAACLAB_JOINT_NAMES
from Teleopit_rollout.constants import G1_JOINT_NAMES as TELEOPIT_G1_JOINT_NAMES


POLICY_RATE_HZ = 50.0
POLICY_DT_S = 1.0 / POLICY_RATE_HZ
NUM_BODY_JOINTS = 29
NUM_HAND_JOINTS = 7
NUM_SONIC_SLOTS = 10
SONIC_REFERENCE_WIDTH = 1024
SONIC_ACTIVE_REFERENCE_WIDTH = 640
TELEOPIT_QPOS_WIDTH = 36
CONSECUTIVE_SLOT_OFFSETS: tuple[int, ...] = tuple(range(NUM_SONIC_SLOTS))
REGULAR_QPOS_SLOT_OFFSETS: tuple[int, ...] = tuple(range(0, 50, 5))
PREPARED_REFERENCE_FORMAT_VERSION = 3

_REFERENCE_ARRAY_FIELDS: tuple[str, ...] = (
    "policy_seq",
    "control_time_s",
    "source_row_index",
    "source_csv_row_number",
    "group_row_counts",
    "source_root_pos",
    "source_root_quat_wxyz",
    "left_hand_target",
    "right_hand_target",
    "sonic_regular_joint_pos",
    "sonic_regular_joint_vel",
    "sonic_regular_anchor_quat_wxyz",
    "sonic_consecutive_joint_pos",
    "sonic_consecutive_joint_vel",
    "sonic_consecutive_anchor_quat_wxyz",
    "teleopit_qpos36",
    "teleopit_reference_joint_vel",
)
_OPTIONAL_REFERENCE_ARRAY_FIELDS: tuple[str, ...] = (
    "source_reference_motion",
    "source_recorded_relative_anchor_6d",
)


class ReferenceError(ValueError):
    """Raised when a recording cannot form an unambiguous reference."""


class ReferenceMode(str, Enum):
    """Reference intent used for one controller-replacement experiment."""

    REFERENCE_MOTION = "reference_motion"
    EXECUTED_QPOS = "executed_qpos"


def _scalar_archive_value(value: np.ndarray, *, name: str) -> Any:
    array = np.asarray(value)
    if array.shape != ():
        raise ReferenceError(f"prepared reference field {name!r} must be scalar")
    return array.item()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _readonly_array(
    value: Any,
    *,
    name: str,
    dtype: np.dtype[Any] | type[Any],
    shape: tuple[int | None, ...],
) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    if result.ndim != len(shape):
        raise ReferenceError(f"{name} has shape {result.shape}; expected {shape}")
    for actual, expected in zip(result.shape, shape, strict=True):
        if expected is not None and actual != expected:
            raise ReferenceError(f"{name} has shape {result.shape}; expected {shape}")
    if np.issubdtype(result.dtype, np.floating) and not np.all(np.isfinite(result)):
        raise ReferenceError(f"{name} contains NaN or infinity")
    result.setflags(write=False)
    return result


def _quaternion_orientation_error(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return sign-invariant quaternion orientation error in radians."""

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left / np.linalg.norm(left, axis=-1, keepdims=True)
    right = right / np.linalg.norm(right, axis=-1, keepdims=True)
    dot = np.sum(left * right, axis=-1)
    return 2.0 * np.arccos(np.clip(np.abs(dot), 0.0, 1.0))


@dataclass(frozen=True)
class ReferenceProvenance:
    """Immutable description of how every controller view was constructed."""

    mode: ReferenceMode
    reference_kind: str
    regular_slot_source: str
    regular_slot_offsets: tuple[int, ...] | None
    orientation_base_sample_mode: str | None
    component_sources: tuple[tuple[str, str], ...]
    source_frame_count: int
    selected_policy_offset: int
    drop_truncated_edges: bool
    policy_alignment_max_time_error_s: float = 0.0
    policy_alignment_max_hand_error: float = 0.0

    def __post_init__(self) -> None:
        try:
            mode = ReferenceMode(self.mode)
        except ValueError as exc:
            raise ReferenceError(f"unsupported reference mode {self.mode!r}") from exc
        object.__setattr__(self, "mode", mode)
        if not self.reference_kind.strip():
            raise ReferenceError("reference_kind must not be empty")
        if not self.regular_slot_source.strip():
            raise ReferenceError("regular_slot_source must not be empty")
        if self.source_frame_count <= 0:
            raise ReferenceError("source_frame_count must be positive")
        if self.selected_policy_offset < 0:
            raise ReferenceError("selected_policy_offset must be non-negative")
        if self.selected_policy_offset >= self.source_frame_count:
            raise ReferenceError(
                "selected_policy_offset must be smaller than source_frame_count"
            )
        if self.regular_slot_offsets is not None:
            offsets = tuple(int(value) for value in self.regular_slot_offsets)
            if len(offsets) != NUM_SONIC_SLOTS:
                raise ReferenceError(
                    f"regular_slot_offsets must contain {NUM_SONIC_SLOTS} values"
                )
            if offsets[0] != 0 or any(value < 0 for value in offsets):
                raise ReferenceError(
                    "regular_slot_offsets must start at zero and be non-negative"
                )
            if any(right < left for left, right in zip(offsets, offsets[1:])):
                raise ReferenceError("regular_slot_offsets must be non-decreasing")
            object.__setattr__(self, "regular_slot_offsets", offsets)
        entries = tuple((str(key), str(value)) for key, value in self.component_sources)
        keys = [key for key, _ in entries]
        if not entries or any(not key or not value for key, value in entries):
            raise ReferenceError("component_sources must contain non-empty entries")
        if len(keys) != len(set(keys)):
            raise ReferenceError("component_sources contains duplicate keys")
        object.__setattr__(self, "component_sources", entries)
        for name, value in (
            ("policy_alignment_max_time_error_s", self.policy_alignment_max_time_error_s),
            ("policy_alignment_max_hand_error", self.policy_alignment_max_hand_error),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ReferenceError(f"{name} must be finite and non-negative")

        if mode is ReferenceMode.REFERENCE_MOTION:
            if self.regular_slot_offsets is not None:
                raise ReferenceError(
                    "recorded reference_motion slots must not be assigned invented offsets"
                )
            if not self.orientation_base_sample_mode:
                raise ReferenceError(
                    "reference_motion requires orientation_base_sample_mode"
                )
        elif tuple(self.regular_slot_offsets or ()) != REGULAR_QPOS_SLOT_OFFSETS:
            raise ReferenceError(
                "executed_qpos regular slots must use the declared 0,5,...,45 offsets"
            )

    def metadata(self) -> dict[str, Any]:
        return {
            "reference_mode": self.mode.value,
            "reference_kind": self.reference_kind,
            "regular_slot_source": self.regular_slot_source,
            "regular_slot_offsets": (
                list(self.regular_slot_offsets)
                if self.regular_slot_offsets is not None
                else None
            ),
            "orientation_base_sample_mode": self.orientation_base_sample_mode,
            "component_sources": dict(self.component_sources),
            "source_frames_after_edge_trim": self.source_frame_count,
            "selected_policy_offset": self.selected_policy_offset,
            "drop_truncated_edges": self.drop_truncated_edges,
            "policy_alignment_max_time_error_s": self.policy_alignment_max_time_error_s,
            "policy_alignment_max_hand_error": self.policy_alignment_max_hand_error,
        }


@dataclass(frozen=True)
class ReferenceFrame:
    """One immutable 50 Hz frame returned by :meth:`ReferenceSequence.frame`."""

    frame_index: int
    policy_seq: int
    control_time_s: float
    source_row_index: int
    source_csv_row_number: int
    group_row_count: int
    source_root_pos: np.ndarray
    source_root_quat_wxyz: np.ndarray
    left_hand_target: np.ndarray
    right_hand_target: np.ndarray
    sonic_regular_joint_pos: np.ndarray
    sonic_regular_joint_vel: np.ndarray
    sonic_regular_anchor_quat_wxyz: np.ndarray
    sonic_consecutive_joint_pos: np.ndarray
    sonic_consecutive_joint_vel: np.ndarray
    sonic_consecutive_anchor_quat_wxyz: np.ndarray
    teleopit_qpos36: np.ndarray
    teleopit_reference_joint_vel: np.ndarray
    source_reference_motion: np.ndarray | None
    source_recorded_relative_anchor_6d: np.ndarray | None


@dataclass(frozen=True)
class ReferenceSequence:
    """Validated, controller-neutral reference sequence at exactly 50 Hz."""

    SCHEMA_VERSION: ClassVar[int] = 1

    source_csv_path: Path
    provenance: ReferenceProvenance
    policy_seq: np.ndarray
    control_time_s: np.ndarray
    source_row_index: np.ndarray
    source_csv_row_number: np.ndarray
    group_row_counts: np.ndarray
    source_root_pos: np.ndarray
    source_root_quat_wxyz: np.ndarray
    left_hand_target: np.ndarray
    right_hand_target: np.ndarray
    sonic_regular_joint_pos: np.ndarray
    sonic_regular_joint_vel: np.ndarray
    sonic_regular_anchor_quat_wxyz: np.ndarray
    sonic_consecutive_joint_pos: np.ndarray
    sonic_consecutive_joint_vel: np.ndarray
    sonic_consecutive_anchor_quat_wxyz: np.ndarray
    teleopit_qpos36: np.ndarray
    teleopit_reference_joint_vel: np.ndarray
    source_reference_motion: np.ndarray | None = None
    source_recorded_relative_anchor_6d: np.ndarray | None = None
    source_csv_sha256: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        source = Path(self.source_csv_path).expanduser().resolve()
        if not source.is_file():
            raise ReferenceError(f"source recording CSV does not exist: {source}")
        object.__setattr__(self, "source_csv_path", source)
        object.__setattr__(self, "source_csv_sha256", _sha256_file(source))
        if not isinstance(self.provenance, ReferenceProvenance):
            raise ReferenceError("provenance must be a ReferenceProvenance")

        arrays: dict[str, np.ndarray] = {
            "policy_seq": _readonly_array(
                self.policy_seq, name="policy_seq", dtype=np.int64, shape=(None,)
            ),
            "control_time_s": _readonly_array(
                self.control_time_s,
                name="control_time_s",
                dtype=np.float64,
                shape=(None,),
            ),
            "source_row_index": _readonly_array(
                self.source_row_index,
                name="source_row_index",
                dtype=np.int64,
                shape=(None,),
            ),
            "source_csv_row_number": _readonly_array(
                self.source_csv_row_number,
                name="source_csv_row_number",
                dtype=np.int64,
                shape=(None,),
            ),
            "group_row_counts": _readonly_array(
                self.group_row_counts,
                name="group_row_counts",
                dtype=np.int32,
                shape=(None,),
            ),
            "source_root_pos": _readonly_array(
                self.source_root_pos,
                name="source_root_pos",
                dtype=np.float32,
                shape=(None, 3),
            ),
            "source_root_quat_wxyz": _readonly_array(
                self.source_root_quat_wxyz,
                name="source_root_quat_wxyz",
                dtype=np.float32,
                shape=(None, 4),
            ),
            "left_hand_target": _readonly_array(
                self.left_hand_target,
                name="left_hand_target",
                dtype=np.float32,
                shape=(None, NUM_HAND_JOINTS),
            ),
            "right_hand_target": _readonly_array(
                self.right_hand_target,
                name="right_hand_target",
                dtype=np.float32,
                shape=(None, NUM_HAND_JOINTS),
            ),
            "sonic_regular_joint_pos": _readonly_array(
                self.sonic_regular_joint_pos,
                name="sonic_regular_joint_pos",
                dtype=np.float32,
                shape=(None, NUM_SONIC_SLOTS, NUM_BODY_JOINTS),
            ),
            "sonic_regular_joint_vel": _readonly_array(
                self.sonic_regular_joint_vel,
                name="sonic_regular_joint_vel",
                dtype=np.float32,
                shape=(None, NUM_SONIC_SLOTS, NUM_BODY_JOINTS),
            ),
            "sonic_regular_anchor_quat_wxyz": _readonly_array(
                self.sonic_regular_anchor_quat_wxyz,
                name="sonic_regular_anchor_quat_wxyz",
                dtype=np.float32,
                shape=(None, NUM_SONIC_SLOTS, 4),
            ),
            "sonic_consecutive_joint_pos": _readonly_array(
                self.sonic_consecutive_joint_pos,
                name="sonic_consecutive_joint_pos",
                dtype=np.float32,
                shape=(None, NUM_SONIC_SLOTS, NUM_BODY_JOINTS),
            ),
            "sonic_consecutive_joint_vel": _readonly_array(
                self.sonic_consecutive_joint_vel,
                name="sonic_consecutive_joint_vel",
                dtype=np.float32,
                shape=(None, NUM_SONIC_SLOTS, NUM_BODY_JOINTS),
            ),
            "sonic_consecutive_anchor_quat_wxyz": _readonly_array(
                self.sonic_consecutive_anchor_quat_wxyz,
                name="sonic_consecutive_anchor_quat_wxyz",
                dtype=np.float32,
                shape=(None, NUM_SONIC_SLOTS, 4),
            ),
            "teleopit_qpos36": _readonly_array(
                self.teleopit_qpos36,
                name="teleopit_qpos36",
                dtype=np.float32,
                shape=(None, TELEOPIT_QPOS_WIDTH),
            ),
            "teleopit_reference_joint_vel": _readonly_array(
                self.teleopit_reference_joint_vel,
                name="teleopit_reference_joint_vel",
                dtype=np.float32,
                shape=(None, NUM_BODY_JOINTS),
            ),
        }
        if self.source_reference_motion is not None:
            arrays["source_reference_motion"] = _readonly_array(
                self.source_reference_motion,
                name="source_reference_motion",
                dtype=np.float32,
                shape=(None, SONIC_REFERENCE_WIDTH),
            )
        if self.source_recorded_relative_anchor_6d is not None:
            arrays["source_recorded_relative_anchor_6d"] = _readonly_array(
                self.source_recorded_relative_anchor_6d,
                name="source_recorded_relative_anchor_6d",
                dtype=np.float32,
                shape=(None, NUM_SONIC_SLOTS, 6),
            )

        frame_count = int(arrays["policy_seq"].shape[0])
        if frame_count == 0:
            raise ReferenceError("reference sequence is empty")
        for name, value in arrays.items():
            if value.shape[0] != frame_count:
                raise ReferenceError(
                    f"{name} has {value.shape[0]} frames; expected {frame_count}"
                )
            object.__setattr__(self, name, value)

        if self.provenance.selected_policy_offset + frame_count > self.provenance.source_frame_count:
            raise ReferenceError(
                "selected reference extends beyond the validated source sequence"
            )
        if frame_count > 1:
            if np.any(np.diff(arrays["policy_seq"]) != 1):
                raise ReferenceError("policy_seq must be consecutive with step one")
            if np.any(np.diff(arrays["control_time_s"]) <= 0.0):
                raise ReferenceError("control_time_s must be strictly increasing")
            if np.any(np.diff(arrays["source_row_index"]) <= 0):
                raise ReferenceError("source_row_index must be strictly increasing")
        if np.any(arrays["source_row_index"] < 0):
            raise ReferenceError("source_row_index must be non-negative")
        if np.any(arrays["source_csv_row_number"] != arrays["source_row_index"] + 2):
            raise ReferenceError("source_csv_row_number must equal source_row_index + 2")
        if np.any(arrays["group_row_counts"] <= 0):
            raise ReferenceError("group_row_counts must all be positive")

        for name in (
            "source_root_quat_wxyz",
            "sonic_regular_anchor_quat_wxyz",
            "sonic_consecutive_anchor_quat_wxyz",
        ):
            norm_error = float(
                np.max(np.abs(np.linalg.norm(arrays[name], axis=-1) - 1.0))
            )
            if norm_error > 2e-5:
                raise ReferenceError(f"{name} is not unit-normalized")
        teleopit_root_quat = arrays["teleopit_qpos36"][:, 3:7]
        if float(np.max(np.abs(np.linalg.norm(teleopit_root_quat, axis=1) - 1.0))) > 2e-5:
            raise ReferenceError("teleopit_qpos36 root quaternion is not unit-normalized")

        if not np.array_equal(
            arrays["sonic_regular_joint_pos"][:, 0],
            arrays["sonic_consecutive_joint_pos"][:, 0],
        ):
            raise ReferenceError("regular and consecutive SONIC slot zero joint positions differ")
        if not np.array_equal(
            arrays["sonic_regular_joint_vel"][:, 0],
            arrays["sonic_consecutive_joint_vel"][:, 0],
        ):
            raise ReferenceError("regular and consecutive SONIC slot zero joint velocities differ")
        orientation_error = _quaternion_orientation_error(
            arrays["sonic_regular_anchor_quat_wxyz"][:, 0],
            arrays["sonic_consecutive_anchor_quat_wxyz"][:, 0],
        )
        if float(np.max(orientation_error)) > 2e-5:
            raise ReferenceError("regular and consecutive SONIC slot zero orientations differ")
        if not np.array_equal(arrays["teleopit_qpos36"][:, :3], arrays["source_root_pos"]):
            raise ReferenceError("Teleopit reference root xyz must equal declared source root xyz")
        orientation_error = _quaternion_orientation_error(
            arrays["teleopit_qpos36"][:, 3:7],
            arrays["sonic_consecutive_anchor_quat_wxyz"][:, 0],
        )
        if float(np.max(orientation_error)) > 2e-5:
            raise ReferenceError("Teleopit and SONIC current reference orientations differ")

        reorder = np.asarray(
            [G1_ISAACLAB_JOINT_NAMES.index(name) for name in TELEOPIT_G1_JOINT_NAMES],
            dtype=np.int64,
        )
        expected_teleopit_joints = arrays["sonic_consecutive_joint_pos"][:, 0, reorder]
        if not np.array_equal(arrays["teleopit_qpos36"][:, 7:], expected_teleopit_joints):
            raise ReferenceError(
                "Teleopit body joints do not match SONIC slot zero after name reorder"
            )

        if self.provenance.mode is ReferenceMode.REFERENCE_MOTION:
            if "source_reference_motion" not in arrays:
                raise ReferenceError("reference_motion mode requires source_reference_motion")
            if "source_recorded_relative_anchor_6d" not in arrays:
                raise ReferenceError(
                    "reference_motion mode requires recorded relative anchor 6D slots"
                )
            active = arrays["source_reference_motion"][:, :SONIC_ACTIVE_REFERENCE_WIDTH]
            expected_pos = active[:, :290].reshape(-1, 10, 29)
            expected_vel = active[:, 290:580].reshape(-1, 10, 29)
            expected_6d = active[:, 580:640].reshape(-1, 10, 6)
            if not np.array_equal(expected_pos, arrays["sonic_regular_joint_pos"]):
                raise ReferenceError("source_reference_motion joint positions disagree")
            if not np.array_equal(expected_vel, arrays["sonic_regular_joint_vel"]):
                raise ReferenceError("source_reference_motion joint velocities disagree")
            if not np.array_equal(
                expected_6d, arrays["source_recorded_relative_anchor_6d"]
            ):
                raise ReferenceError("source_reference_motion anchor 6D values disagree")
        elif self.source_reference_motion is not None or self.source_recorded_relative_anchor_6d is not None:
            raise ReferenceError(
                "executed_qpos mode must not claim recorded reference_motion as its input"
            )

    @property
    def mode(self) -> ReferenceMode:
        return self.provenance.mode

    @property
    def num_frames(self) -> int:
        return int(self.policy_seq.shape[0])

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, index: int) -> ReferenceFrame:
        return self.frame(index)

    @property
    def first_source_row_index(self) -> int:
        return int(self.source_row_index[0])

    @property
    def sonic_current_joint_pos(self) -> np.ndarray:
        """Canonical 50 Hz SONIC joint-position stream in IsaacLab order."""

        return self.sonic_consecutive_joint_pos[:, 0]

    @property
    def sonic_current_joint_vel(self) -> np.ndarray:
        """Canonical 50 Hz SONIC joint-velocity stream in IsaacLab order."""

        return self.sonic_consecutive_joint_vel[:, 0]

    @property
    def sonic_current_anchor_quat_wxyz(self) -> np.ndarray:
        """Canonical 50 Hz reference-anchor orientation in world coordinates."""

        return self.sonic_consecutive_anchor_quat_wxyz[:, 0]

    @property
    def teleopit_reference_root_pos(self) -> np.ndarray:
        return self.teleopit_qpos36[:, :3]

    @property
    def teleopit_reference_root_quat_wxyz(self) -> np.ndarray:
        return self.teleopit_qpos36[:, 3:7]

    @property
    def teleopit_reference_joint_pos(self) -> np.ndarray:
        return self.teleopit_qpos36[:, 7:]

    def frame(self, index: int) -> ReferenceFrame:
        """Return one policy frame; negative indexing is intentionally rejected."""

        if index < 0 or index >= self.num_frames:
            raise ReferenceError(
                f"frame index {index} is outside [0, {self.num_frames - 1}]"
            )
        source_reference_motion = (
            None
            if self.source_reference_motion is None
            else self.source_reference_motion[index]
        )
        relative_anchor = (
            None
            if self.source_recorded_relative_anchor_6d is None
            else self.source_recorded_relative_anchor_6d[index]
        )
        return ReferenceFrame(
            frame_index=index,
            policy_seq=int(self.policy_seq[index]),
            control_time_s=float(self.control_time_s[index]),
            source_row_index=int(self.source_row_index[index]),
            source_csv_row_number=int(self.source_csv_row_number[index]),
            group_row_count=int(self.group_row_counts[index]),
            source_root_pos=self.source_root_pos[index],
            source_root_quat_wxyz=self.source_root_quat_wxyz[index],
            left_hand_target=self.left_hand_target[index],
            right_hand_target=self.right_hand_target[index],
            sonic_regular_joint_pos=self.sonic_regular_joint_pos[index],
            sonic_regular_joint_vel=self.sonic_regular_joint_vel[index],
            sonic_regular_anchor_quat_wxyz=self.sonic_regular_anchor_quat_wxyz[index],
            sonic_consecutive_joint_pos=self.sonic_consecutive_joint_pos[index],
            sonic_consecutive_joint_vel=self.sonic_consecutive_joint_vel[index],
            sonic_consecutive_anchor_quat_wxyz=self.sonic_consecutive_anchor_quat_wxyz[index],
            teleopit_qpos36=self.teleopit_qpos36[index],
            teleopit_reference_joint_vel=self.teleopit_reference_joint_vel[index],
            source_reference_motion=source_reference_motion,
            source_recorded_relative_anchor_6d=relative_anchor,
        )

    def slice(
        self, policy_offset: int = 0, policy_count: int | None = None
    ) -> "ReferenceSequence":
        """Return a validated subsequence without reparsing the source CSV."""

        validate_policy_slice(policy_offset, policy_count)
        if policy_offset >= self.num_frames:
            raise ReferenceError(
                f"policy_offset {policy_offset} is outside [0, {self.num_frames - 1}]"
            )
        stop = self.num_frames if policy_count is None else min(
            self.num_frames, policy_offset + policy_count
        )
        selected = slice(policy_offset, stop)
        values: dict[str, Any] = {
            "source_csv_path": self.source_csv_path,
            "provenance": replace(
                self.provenance,
                selected_policy_offset=(
                    self.provenance.selected_policy_offset + policy_offset
                ),
            ),
        }
        for name in (
            "policy_seq",
            "control_time_s",
            "source_row_index",
            "source_csv_row_number",
            "group_row_counts",
            "source_root_pos",
            "source_root_quat_wxyz",
            "left_hand_target",
            "right_hand_target",
            "sonic_regular_joint_pos",
            "sonic_regular_joint_vel",
            "sonic_regular_anchor_quat_wxyz",
            "sonic_consecutive_joint_pos",
            "sonic_consecutive_joint_vel",
            "sonic_consecutive_anchor_quat_wxyz",
            "teleopit_qpos36",
            "teleopit_reference_joint_vel",
        ):
            values[name] = getattr(self, name)[selected]
        values["source_reference_motion"] = (
            None
            if self.source_reference_motion is None
            else self.source_reference_motion[selected]
        )
        values["source_recorded_relative_anchor_6d"] = (
            None
            if self.source_recorded_relative_anchor_6d is None
            else self.source_recorded_relative_anchor_6d[selected]
        )
        return ReferenceSequence(**values)

    def save_prepared_npz(self, path: str | Path) -> Path:
        """Atomically save every controller-neutral reference array.

        The archive deliberately contains no object arrays, so consumers can
        and should open it with ``allow_pickle=False``.  Optional source-only
        arrays are represented by an explicit scalar presence flag rather than
        a pickled ``None`` value.
        """

        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_csv = self.source_csv_path.expanduser().resolve()
        if not source_csv.is_file():
            raise ReferenceError(f"source CSV does not exist: {source_csv}")
        source_csv_sha256 = _sha256_file(source_csv)
        if source_csv_sha256 != self.source_csv_sha256:
            raise ReferenceError(
                "source CSV changed after this reference sequence was built"
            )
        payload: dict[str, np.ndarray] = {
            "prepared_reference_format_version": np.asarray(
                PREPARED_REFERENCE_FORMAT_VERSION, dtype=np.int64
            ),
            "sonic_joint_names": np.asarray(
                G1_ISAACLAB_JOINT_NAMES, dtype=np.str_
            ),
            "teleopit_joint_names": np.asarray(
                TELEOPIT_G1_JOINT_NAMES, dtype=np.str_
            ),
            "source_csv_sha256": np.asarray(
                source_csv_sha256, dtype=np.str_
            ),
        }
        for name in _REFERENCE_ARRAY_FIELDS:
            payload[name] = np.asarray(getattr(self, name))
        optional_presence: dict[str, bool] = {}
        for name in _OPTIONAL_REFERENCE_ARRAY_FIELDS:
            value = getattr(self, name)
            present = value is not None
            optional_presence[name] = present
            payload[f"{name}__present"] = np.asarray(present, dtype=np.bool_)
            if present:
                payload[name] = np.asarray(value)

        metadata = self.metadata()
        metadata["prepared_reference_format_version"] = (
            PREPARED_REFERENCE_FORMAT_VERSION
        )
        metadata["optional_arrays"] = optional_presence
        metadata["source_csv_sha256"] = source_csv_sha256
        payload["metadata_json"] = np.asarray(
            json.dumps(metadata, ensure_ascii=False, sort_keys=True), dtype=np.str_
        )
        for name, value in payload.items():
            if np.asarray(value).dtype.hasobject:
                raise ReferenceError(
                    f"prepared reference field {name!r} would require pickle"
                )

        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "wb") as output:
                np.savez_compressed(output, **payload)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_path, destination)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
        return destination

    @classmethod
    def load_prepared_npz(
        cls,
        path: str | Path,
        *,
        source_csv_path: str | Path | None = None,
    ) -> "ReferenceSequence":
        """Load and fully revalidate an archive written by
        :meth:`save_prepared_npz`.

        ``source_csv_path`` may point to a relocated copy of the source CSV.
        The regular :class:`ReferenceSequence` invariants are rerun after
        decoding, so this helper is also the archive round-trip validator.
        """

        archive_path = Path(path).expanduser().resolve()
        if not archive_path.is_file():
            raise ReferenceError(
                f"prepared reference archive does not exist: {archive_path}"
            )
        try:
            with np.load(archive_path, allow_pickle=False) as archive:
                for name in archive.files:
                    if archive[name].dtype.hasobject:
                        raise ReferenceError(
                            f"prepared reference field {name!r} requires pickle"
                        )
                required = {
                    "prepared_reference_format_version",
                    "sonic_joint_names",
                    "teleopit_joint_names",
                    "source_csv_sha256",
                    "metadata_json",
                    *_REFERENCE_ARRAY_FIELDS,
                    *(f"{name}__present" for name in _OPTIONAL_REFERENCE_ARRAY_FIELDS),
                }
                missing = sorted(required.difference(archive.files))
                if missing:
                    raise ReferenceError(
                        f"prepared reference archive is missing fields {missing}"
                    )
                version = _scalar_archive_value(
                    archive["prepared_reference_format_version"],
                    name="prepared_reference_format_version",
                )
                if int(version) != PREPARED_REFERENCE_FORMAT_VERSION:
                    raise ReferenceError(
                        "unsupported prepared reference format version "
                        f"{version!r}; expected {PREPARED_REFERENCE_FORMAT_VERSION}"
                    )
                metadata_text = _scalar_archive_value(
                    archive["metadata_json"], name="metadata_json"
                )
                try:
                    metadata = json.loads(str(metadata_text))
                except (TypeError, json.JSONDecodeError) as exc:
                    raise ReferenceError(
                        "prepared reference metadata_json is not valid JSON"
                    ) from exc
                if not isinstance(metadata, dict):
                    raise ReferenceError(
                        "prepared reference metadata_json must contain an object"
                    )
                archived_source_sha256 = str(
                    _scalar_archive_value(
                        archive["source_csv_sha256"],
                        name="source_csv_sha256",
                    )
                )
                if (
                    len(archived_source_sha256) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in archived_source_sha256
                    )
                ):
                    raise ReferenceError(
                        "prepared reference source_csv_sha256 is invalid"
                    )
                sonic_names = tuple(
                    str(value) for value in np.asarray(archive["sonic_joint_names"])
                )
                teleopit_names = tuple(
                    str(value) for value in np.asarray(archive["teleopit_joint_names"])
                )
                if sonic_names != tuple(G1_ISAACLAB_JOINT_NAMES):
                    raise ReferenceError(
                        "prepared reference SONIC joint names/order do not match"
                    )
                if teleopit_names != tuple(TELEOPIT_G1_JOINT_NAMES):
                    raise ReferenceError(
                        "prepared reference Teleopit joint names/order do not match"
                    )

                values: dict[str, Any] = {
                    name: np.array(archive[name], copy=True)
                    for name in _REFERENCE_ARRAY_FIELDS
                }
                optional_presence: dict[str, bool] = {}
                for name in _OPTIONAL_REFERENCE_ARRAY_FIELDS:
                    raw_present = _scalar_archive_value(
                        archive[f"{name}__present"],
                        name=f"{name}__present",
                    )
                    if not isinstance(raw_present, (bool, np.bool_)):
                        raise ReferenceError(
                            f"{name}__present must be a scalar boolean"
                        )
                    present = bool(raw_present)
                    optional_presence[name] = present
                    if present and name not in archive.files:
                        raise ReferenceError(
                            f"{name} is marked present but absent from the archive"
                        )
                    if not present and name in archive.files:
                        raise ReferenceError(
                            f"{name} is marked absent but is stored in the archive"
                        )
                    values[name] = (
                        np.array(archive[name], copy=True) if present else None
                    )
        except (OSError, ValueError) as exc:
            if isinstance(exc, ReferenceError):
                raise
            raise ReferenceError(
                f"failed to read prepared reference archive {archive_path}: {exc}"
            ) from exc

        metadata_optional = metadata.get("optional_arrays")
        if metadata_optional != optional_presence:
            raise ReferenceError(
                "prepared reference optional-array flags disagree with metadata_json"
            )
        source = (
            Path(str(metadata.get("source_csv", "")))
            if source_csv_path is None
            else Path(source_csv_path)
        )
        source = source.expanduser().resolve()
        if not source.is_file():
            raise ReferenceError(
                f"prepared reference source CSV does not exist: {source}"
            )
        actual_source_sha256 = _sha256_file(source)
        if actual_source_sha256 != archived_source_sha256:
            raise ReferenceError(
                "prepared reference source CSV SHA-256 mismatch: "
                f"expected {archived_source_sha256}, got {actual_source_sha256}"
            )
        if metadata.get("source_csv_sha256") != archived_source_sha256:
            raise ReferenceError(
                "prepared reference source SHA-256 disagrees with metadata_json"
            )
        component_sources = metadata.get("component_sources")
        if not isinstance(component_sources, dict):
            raise ReferenceError(
                "prepared reference metadata has no component_sources object"
            )
        try:
            provenance = ReferenceProvenance(
                mode=ReferenceMode(str(metadata["reference_mode"])),
                reference_kind=str(metadata["reference_kind"]),
                regular_slot_source=str(metadata["regular_slot_source"]),
                regular_slot_offsets=(
                    None
                    if metadata["regular_slot_offsets"] is None
                    else tuple(int(value) for value in metadata["regular_slot_offsets"])
                ),
                orientation_base_sample_mode=(
                    None
                    if metadata["orientation_base_sample_mode"] is None
                    else str(metadata["orientation_base_sample_mode"])
                ),
                component_sources=tuple(
                    (str(key), str(value))
                    for key, value in component_sources.items()
                ),
                source_frame_count=int(metadata["source_frames_after_edge_trim"]),
                selected_policy_offset=int(metadata["selected_policy_offset"]),
                drop_truncated_edges=bool(metadata["drop_truncated_edges"]),
                policy_alignment_max_time_error_s=float(
                    metadata["policy_alignment_max_time_error_s"]
                ),
                policy_alignment_max_hand_error=float(
                    metadata["policy_alignment_max_hand_error"]
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ReferenceError(
                "prepared reference metadata cannot reconstruct provenance"
            ) from exc
        result = cls(source_csv_path=source, provenance=provenance, **values)

        expected_metadata = result.metadata()
        for key, expected in expected_metadata.items():
            if key == "source_csv" and source_csv_path is not None:
                continue
            if metadata.get(key) != expected:
                raise ReferenceError(
                    f"prepared reference metadata field {key!r} disagrees with arrays"
                )
        if metadata.get("source_csv_sha256") != actual_source_sha256:
            raise ReferenceError(
                "prepared reference metadata source SHA-256 changed"
            )
        if metadata.get("prepared_reference_format_version") != (
            PREPARED_REFERENCE_FORMAT_VERSION
        ):
            raise ReferenceError(
                "prepared reference metadata has the wrong format version"
            )
        return result

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "source_csv": str(self.source_csv_path),
            "source_csv_sha256": self.source_csv_sha256,
            "rate_hz": POLICY_RATE_HZ,
            "policy_dt_s": POLICY_DT_S,
            "selected_frames": self.num_frames,
            "first_policy_seq": int(self.policy_seq[0]),
            "last_policy_seq": int(self.policy_seq[-1]),
            "first_source_row_index": int(self.source_row_index[0]),
            "last_source_row_index": int(self.source_row_index[-1]),
            "sonic_joint_order": "g1_isaaclab_29dof",
            "sonic_joint_names": list(G1_ISAACLAB_JOINT_NAMES),
            "sonic_regular_layout": "10 x (q[29], dq[29], anchor_quat_wxyz[4])",
            "sonic_consecutive_offsets": list(CONSECUTIVE_SLOT_OFFSETS),
            "teleopit_qpos36_layout": "root_xyz[3] + root_quat_wxyz[4] + body_q[29]",
            "teleopit_joint_order": "teleopit_g1_canonical",
            "teleopit_joint_names": list(TELEOPIT_G1_JOINT_NAMES),
            "hand_target_layout": "left_hand_q[0:7] / right_hand_q[0:7]",
            "source_reference_motion_present": self.source_reference_motion is not None,
            "source_recorded_relative_anchor_6d_present": (
                self.source_recorded_relative_anchor_6d is not None
            ),
            **self.provenance.metadata(),
        }


@runtime_checkable
class ReferenceProvider(Protocol):
    """Interface implemented by all reference sources."""

    mode: ClassVar[ReferenceMode]

    def load(
        self,
        recording: str | Path,
        *,
        policy_offset: int = 0,
        policy_count: int | None = None,
    ) -> ReferenceSequence:
        """Load and validate a policy-rate sequence from ``recording``."""


def validate_policy_slice(policy_offset: int, policy_count: int | None) -> None:
    if policy_offset < 0:
        raise ReferenceError("policy_offset must be non-negative")
    if policy_count is not None and policy_count <= 0:
        raise ReferenceError("policy_count must be positive")


def clamped_future_indices(
    *,
    start_indices: Sequence[int] | np.ndarray,
    offsets: Sequence[int],
    source_frame_count: int,
) -> np.ndarray:
    """Return ``(N, len(offsets))`` future indices with last-frame clamping."""

    if source_frame_count <= 0:
        raise ReferenceError("source_frame_count must be positive")
    starts = np.asarray(start_indices, dtype=np.int64)
    future_offsets = np.asarray(tuple(offsets), dtype=np.int64)
    if starts.ndim != 1 or future_offsets.ndim != 1 or future_offsets.size == 0:
        raise ReferenceError("start_indices and offsets must be non-empty 1-D values")
    if np.any(starts < 0) or np.any(starts >= source_frame_count):
        raise ReferenceError("start_indices are outside the source sequence")
    if np.any(future_offsets < 0) or np.any(np.diff(future_offsets) < 0):
        raise ReferenceError("offsets must be non-negative and non-decreasing")
    return np.minimum(
        starts[:, None] + future_offsets[None, :], source_frame_count - 1
    )


def reorder_isaaclab_to_teleopit(values: np.ndarray) -> np.ndarray:
    """Reorder a final 29-joint axis by name, preserving leading dimensions."""

    array = np.asarray(values)
    if array.shape[-1:] != (NUM_BODY_JOINTS,):
        raise ReferenceError(
            f"joint array has shape {array.shape}; final dimension must be 29"
        )
    indices = [G1_ISAACLAB_JOINT_NAMES.index(name) for name in TELEOPIT_G1_JOINT_NAMES]
    return array[..., indices]
