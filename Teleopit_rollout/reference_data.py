"""Prepare recorded G1 qpos as a Teleopit policy-rate reference.

The recording parser remains the validated implementation in
``change_ckpt_track.qpos_reference_data``.  This module only adds the adapter
needed by the Teleopit rollout:

* explicitly reorder the 29 body joints by name into Teleopit's canonical
  :data:`G1_JOINT_NAMES` order;
* form each 36-D reference pose as ``root_xyz + root_quat_wxyz + body_q``;
* retain hand targets and exact source ``policy_seq``/CSV-row mappings; and
* write a self-describing ``prepared_reference.npz`` artifact.

The recorded joint velocity is retained for diagnostics, but it is *not* the
Teleopit reference velocity.  The rollout computes Teleopit's reference joint
and torso velocities from consecutive 36-D poses at the fixed 50 Hz policy
rate, matching the upstream controller.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:  # Package import: ``from Teleopit_rollout.reference_data import ...``
    from .constants import ACTION_DIM, G1_JOINT_NAMES, POLICY_HZ
except ImportError:  # Script import: ``python Teleopit_rollout/reference_data.py``
    from constants import ACTION_DIM, G1_JOINT_NAMES, POLICY_HZ

try:
    from change_ckpt_track.qpos_reference_data import (
        QposReferenceError,
        QposReferenceSequence,
        load_qpos_reference,
    )
except ModuleNotFoundError as exc:
    # Python sets sys.path[0] to Teleopit_rollout when this file is executed as
    # a script, so the repository root is not necessarily importable.
    if exc.name not in {
        "change_ckpt_track",
        "change_ckpt_track.qpos_reference_data",
    }:
        raise
    _REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_REPOSITORY_ROOT))
    from change_ckpt_track.qpos_reference_data import (
        QposReferenceError,
        QposReferenceSequence,
        load_qpos_reference,
    )


ROOT_QPOS_DIM = 7
TELEOPIT_QPOS_DIM = ROOT_QPOS_DIM + ACTION_DIM
HAND_DIM = 7
SCHEMA_VERSION = 1


class PreparedReferenceError(ValueError):
    """Raised when a recording cannot form a safe Teleopit reference."""


def _readonly_array(
    value: Any,
    *,
    name: str,
    dtype: np.dtype[Any] | type[Any],
    shape: tuple[int | None, ...],
) -> np.ndarray:
    """Return a finite, contiguous, immutable array of the required shape."""

    # Always copy: marking the validated representation read-only must not
    # unexpectedly change the caller's array flags.
    result = np.array(value, dtype=dtype, order="C", copy=True)
    if result.ndim != len(shape):
        raise PreparedReferenceError(
            f"{name} has shape {result.shape}; expected {shape}"
        )
    for actual, expected in zip(result.shape, shape, strict=True):
        if expected is not None and actual != expected:
            raise PreparedReferenceError(
                f"{name} has shape {result.shape}; expected {shape}"
            )
    if np.issubdtype(result.dtype, np.floating) and not np.all(
        np.isfinite(result)
    ):
        raise PreparedReferenceError(f"{name} contains NaN or infinity")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PreparedReference:
    """Validated 50 Hz Teleopit reference plus source-frame provenance.

    ``qpos36`` has the exact layout consumed by
    :meth:`TeleopitObservationBuilder.reference_features`::

        [root_x, root_y, root_z, root_qw, root_qx, root_qy, root_qz,
         joint_0, ..., joint_28]

    where the body joints follow :data:`G1_JOINT_NAMES`.
    """

    source_csv_path: Path
    joint_names: tuple[str, ...]
    qpos36: np.ndarray
    recorded_joint_vel: np.ndarray
    left_hand_target: np.ndarray
    right_hand_target: np.ndarray
    policy_seq: np.ndarray
    control_time_s: np.ndarray
    source_row_index: np.ndarray
    source_csv_row_number: np.ndarray
    group_row_counts: np.ndarray
    policy_offset: int
    source_frame_count: int

    def __post_init__(self) -> None:
        source_path = Path(self.source_csv_path).expanduser().resolve()
        if not source_path.is_file():
            raise PreparedReferenceError(
                f"source recording CSV does not exist: {source_path}"
            )
        object.__setattr__(self, "source_csv_path", source_path)

        if tuple(self.joint_names) != G1_JOINT_NAMES:
            raise PreparedReferenceError(
                "joint_names must exactly match Teleopit's canonical "
                "29-DoF G1 order"
            )
        if self.policy_offset < 0:
            raise PreparedReferenceError("policy_offset must be non-negative")
        if self.source_frame_count <= 0:
            raise PreparedReferenceError("source_frame_count must be positive")

        arrays = {
            "qpos36": _readonly_array(
                self.qpos36,
                name="qpos36",
                dtype=np.float32,
                shape=(None, TELEOPIT_QPOS_DIM),
            ),
            "recorded_joint_vel": _readonly_array(
                self.recorded_joint_vel,
                name="recorded_joint_vel",
                dtype=np.float32,
                shape=(None, ACTION_DIM),
            ),
            "left_hand_target": _readonly_array(
                self.left_hand_target,
                name="left_hand_target",
                dtype=np.float32,
                shape=(None, HAND_DIM),
            ),
            "right_hand_target": _readonly_array(
                self.right_hand_target,
                name="right_hand_target",
                dtype=np.float32,
                shape=(None, HAND_DIM),
            ),
            "policy_seq": _readonly_array(
                self.policy_seq,
                name="policy_seq",
                dtype=np.int64,
                shape=(None,),
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
        }
        frame_count = int(arrays["policy_seq"].shape[0])
        if frame_count == 0:
            raise PreparedReferenceError("prepared reference is empty")
        for name, value in arrays.items():
            if value.shape[0] != frame_count:
                raise PreparedReferenceError(
                    f"{name} has {value.shape[0]} frames; expected {frame_count}"
                )
            object.__setattr__(self, name, value)

        if self.policy_offset + frame_count > self.source_frame_count:
            raise PreparedReferenceError(
                "selected frames extend beyond source_frame_count: "
                f"offset={self.policy_offset}, selected={frame_count}, "
                f"source={self.source_frame_count}"
            )
        if frame_count > 1:
            if np.any(np.diff(arrays["policy_seq"]) != 1):
                raise PreparedReferenceError(
                    "policy_seq must be consecutive with step one"
                )
            if np.any(np.diff(arrays["control_time_s"]) <= 0.0):
                raise PreparedReferenceError(
                    "control_time_s must be strictly increasing"
                )
            if np.any(np.diff(arrays["source_row_index"]) <= 0):
                raise PreparedReferenceError(
                    "source_row_index must be strictly increasing"
                )
        if np.any(arrays["source_row_index"] < 0):
            raise PreparedReferenceError(
                "source_row_index must contain non-negative data-row indices"
            )
        if np.any(
            arrays["source_csv_row_number"]
            != arrays["source_row_index"] + 2
        ):
            raise PreparedReferenceError(
                "source_csv_row_number must equal source_row_index + 2"
            )
        if np.any(arrays["group_row_counts"] <= 0):
            raise PreparedReferenceError("group_row_counts must all be positive")

        quaternion_norms = np.linalg.norm(arrays["qpos36"][:, 3:7], axis=1)
        if float(np.max(np.abs(quaternion_norms - 1.0))) > 2e-5:
            raise PreparedReferenceError(
                "root quaternion in qpos36 is not unit-normalized"
            )

    @property
    def num_frames(self) -> int:
        return int(self.policy_seq.shape[0])

    @property
    def root_qpos(self) -> np.ndarray:
        """Read-only ``(N, 7)`` root ``xyz + quaternion_wxyz`` view."""

        return self.qpos36[:, :ROOT_QPOS_DIM]

    @property
    def root_pos(self) -> np.ndarray:
        return self.qpos36[:, :3]

    @property
    def root_quat_wxyz(self) -> np.ndarray:
        return self.qpos36[:, 3:7]

    @property
    def joint_pos(self) -> np.ndarray:
        """Read-only body-q view in Teleopit's canonical joint order."""

        return self.qpos36[:, ROOT_QPOS_DIM:]

    @property
    def first_source_row_index(self) -> int:
        return int(self.source_row_index[0])

    @property
    def last_source_row_index(self) -> int:
        return int(self.source_row_index[-1])

    def metadata(self) -> dict[str, Any]:
        """Return JSON-serializable layout and provenance metadata."""

        return {
            "schema_version": SCHEMA_VERSION,
            "reference_kind": "recorded_robot_qpos_track_for_teleopit",
            "source_csv": str(self.source_csv_path),
            "rate_hz": POLICY_HZ,
            "policy_dt_s": 1.0 / POLICY_HZ,
            "source_frames_after_edge_trim": self.source_frame_count,
            "selected_policy_offset": self.policy_offset,
            "selected_frames": self.num_frames,
            "first_policy_seq": int(self.policy_seq[0]),
            "last_policy_seq": int(self.policy_seq[-1]),
            "first_source_row_index": self.first_source_row_index,
            "last_source_row_index": self.last_source_row_index,
            "qpos36_layout": "root_xyz[3] + root_quat_wxyz[4] + body_q[29]",
            "joint_order": "teleopit_g1_canonical",
            "joint_names": list(self.joint_names),
            "hand_target_layout": "left_hand_q[0:7] / right_hand_q[0:7]",
            "recorded_joint_vel_note": (
                "diagnostic source state only; Teleopit reference velocities "
                "are finite differences of qpos36 at 50 Hz"
            ),
        }

    def save_npz(self, path: str | Path) -> Path:
        """Save this reference as a compressed, pickle-free NPZ artifact.

        A directory argument is accepted as shorthand for
        ``<directory>/prepared_reference.npz``.
        """

        destination = Path(path).expanduser()
        if destination.exists() and destination.is_dir():
            destination = destination / "prepared_reference.npz"
        elif destination.suffix == "":
            destination = destination / "prepared_reference.npz"
        destination = destination.resolve()
        if destination.suffix.lower() != ".npz":
            raise PreparedReferenceError(
                f"prepared reference output must end in .npz: {destination}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            destination,
            qpos36=np.asarray(self.qpos36, dtype=np.float32),
            root_qpos=np.asarray(self.root_qpos, dtype=np.float32),
            joint_pos=np.asarray(self.joint_pos, dtype=np.float32),
            recorded_joint_vel=np.asarray(
                self.recorded_joint_vel, dtype=np.float32
            ),
            left_hand_target=np.asarray(
                self.left_hand_target, dtype=np.float32
            ),
            right_hand_target=np.asarray(
                self.right_hand_target, dtype=np.float32
            ),
            # Compatibility aliases used by the existing comparison viewer.
            left_hand_joints=np.asarray(
                self.left_hand_target, dtype=np.float32
            ),
            right_hand_joints=np.asarray(
                self.right_hand_target, dtype=np.float32
            ),
            policy_seq=np.asarray(self.policy_seq, dtype=np.int64),
            control_time_s=np.asarray(self.control_time_s, dtype=np.float64),
            source_row_index=np.asarray(
                self.source_row_index, dtype=np.int64
            ),
            source_csv_row_number=np.asarray(
                self.source_csv_row_number, dtype=np.int64
            ),
            group_row_counts=np.asarray(
                self.group_row_counts, dtype=np.int32
            ),
            joint_names=np.asarray(self.joint_names),
            metadata_json=np.asarray(
                json.dumps(
                    self.metadata(), ensure_ascii=False, sort_keys=True
                )
            ),
        )
        return destination


def _joint_reorder_indices(
    source_joint_names: Sequence[str],
) -> np.ndarray:
    names = tuple(source_joint_names)
    if len(names) != len(set(names)):
        raise PreparedReferenceError(
            "source reference contains duplicate body joint names"
        )
    missing = [name for name in G1_JOINT_NAMES if name not in names]
    extra = [name for name in names if name not in G1_JOINT_NAMES]
    if missing or extra:
        raise PreparedReferenceError(
            "source and Teleopit G1 joint sets differ: "
            f"missing={missing}, extra={extra}"
        )
    return np.asarray([names.index(name) for name in G1_JOINT_NAMES], dtype=np.int64)


def _adapt_sequence(
    selected: QposReferenceSequence,
    *,
    source_frame_count: int,
) -> PreparedReference:
    reorder = _joint_reorder_indices(selected.joint_names)
    root_qpos = np.concatenate(
        (selected.root_pos, selected.root_quat_wxyz), axis=1
    ).astype(np.float32, copy=False)
    joint_pos = np.asarray(selected.joint_pos[:, reorder], dtype=np.float32)
    qpos36 = np.concatenate((root_qpos, joint_pos), axis=1).astype(
        np.float32, copy=False
    )
    return PreparedReference(
        source_csv_path=selected.csv_path,
        joint_names=G1_JOINT_NAMES,
        qpos36=qpos36,
        recorded_joint_vel=np.asarray(
            selected.joint_vel[:, reorder], dtype=np.float32
        ),
        left_hand_target=np.asarray(
            selected.left_hand_target, dtype=np.float32
        ),
        right_hand_target=np.asarray(
            selected.right_hand_target, dtype=np.float32
        ),
        policy_seq=np.asarray(selected.policy_seq, dtype=np.int64),
        control_time_s=np.asarray(selected.control_time_s, dtype=np.float64),
        source_row_index=np.asarray(
            selected.source_row_indices, dtype=np.int64
        ),
        source_csv_row_number=np.asarray(
            selected.source_csv_row_numbers, dtype=np.int64
        ),
        group_row_counts=np.asarray(
            selected.group_row_counts, dtype=np.int32
        ),
        policy_offset=int(selected.frame_offset),
        source_frame_count=source_frame_count,
    )


def load_prepared_reference(
    recording: str | Path,
    *,
    policy_offset: int = 0,
    policy_count: int | None = None,
    drop_truncated_edges: bool = True,
) -> PreparedReference:
    """Load, slice, and reorder one recording for Teleopit.

    ``policy_offset`` is relative to the validated sequence after optional
    incomplete-edge trimming.  ``policy_count`` is a maximum; when it extends
    beyond the recording, the remaining frames are returned.  This matches the
    slicing behavior of the qpos-track pipeline.
    """

    if policy_offset < 0:
        raise PreparedReferenceError("policy_offset must be non-negative")
    if policy_count is not None and policy_count <= 0:
        raise PreparedReferenceError("policy_count must be positive")

    try:
        full = load_qpos_reference(
            recording, drop_truncated_edges=drop_truncated_edges
        )
    except QposReferenceError as exc:
        raise PreparedReferenceError(str(exc)) from exc

    if policy_offset >= full.num_frames:
        raise PreparedReferenceError(
            f"policy_offset {policy_offset} is outside the validated "
            f"reference with {full.num_frames} frames"
        )
    selected = full.slice(policy_offset, policy_count)
    return _adapt_sequence(selected, source_frame_count=full.num_frames)


# Concise alias for rollout code that treats preparation as a pure conversion.
prepare_reference = load_prepared_reference


def save_prepared_reference(
    reference: PreparedReference, path: str | Path
) -> Path:
    """Functional wrapper around :meth:`PreparedReference.save_npz`."""

    return reference.save_npz(path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a recorded G1 qpos track for Teleopit rollout."
    )
    parser.add_argument("recording", help="recording directory or data.csv")
    parser.add_argument(
        "--output", required=True, help="output .npz file or directory"
    )
    parser.add_argument("--policy-offset", type=int, default=0)
    parser.add_argument("--policy-count", type=int)
    parser.add_argument(
        "--keep-truncated-edges",
        action="store_true",
        help="do not drop short first/last policy groups",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    reference = load_prepared_reference(
        args.recording,
        policy_offset=args.policy_offset,
        policy_count=args.policy_count,
        drop_truncated_edges=not args.keep_truncated_edges,
    )
    destination = reference.save_npz(args.output)
    print(
        json.dumps(
            {"output": str(destination), **reference.metadata()},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
