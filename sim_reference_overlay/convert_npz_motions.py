#!/usr/bin/env python3
"""Convert one G1 reference-motion NPZ archive to the C++ CSV layout.

The deployment reader expects a dataset root containing one directory per
motion.  This converter therefore publishes::

    <output-root>/<motion-name>/
        joint_pos.csv
        joint_vel.csv
        body_pos.csv
        body_quat.csv
        body_lin_vel.csv
        body_ang_vel.csv
        metadata.txt
        info.txt

The input arrays use the full 30-body IsaacLab layout.  The output keeps the
same 14 bodies as the released reference examples and records their canonical
IsaacLab indexes in ``metadata.txt``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Callable, Sequence

import numpy as np


EXPECTED_FPS = 50
NUM_JOINTS = 29
NUM_SOURCE_BODIES = 30
CSV_FLOAT_FORMAT = "%.9g"  # Nine significant digits round-trip float32.
QUATERNION_NORM_TOLERANCE = 1.0e-4

# Matches every released reference/example/*/metadata.txt file.
RELEASE_BODY_INDEXES = np.asarray(
    [0, 4, 10, 18, 5, 11, 19, 9, 16, 22, 28, 17, 23, 29],
    dtype=np.int64,
)

# Duplicated here deliberately: importing the training configuration would
# require torch/IsaacLab in the lightweight deployment environment.
G1_ISAACLAB_JOINT_NAMES = (
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

G1_ISAACLAB_BODY_NAMES = (
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)

REQUIRED_ARRAYS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


class ConversionError(ValueError):
    """Raised when an input or generated reference violates the contract."""


@dataclass(frozen=True)
class PreparedMotion:
    """Validated, release-layout arrays ready for CSV serialization."""

    name: str
    source_path: Path
    source_sha256: str
    fps: int
    timesteps: int
    arrays: dict[str, np.ndarray]
    joint_order_validation: str
    body_order_validation: str
    quaternion_max_norm_deviation: float
    derivation_notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class CsvSpec:
    filename: str
    array_key: str
    headers: tuple[str, ...]
    reshape: Callable[[np.ndarray], np.ndarray]


def _identity_2d(array: np.ndarray) -> np.ndarray:
    return array


def _flatten_frames(array: np.ndarray) -> np.ndarray:
    return array.reshape(array.shape[0], -1)


def _joint_headers(prefix: str) -> tuple[str, ...]:
    return tuple(f"{prefix}{index}" for index in range(NUM_JOINTS))


def _body_headers(suffixes: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        f"body_{body_index}_{suffix}"
        for body_index in range(RELEASE_BODY_INDEXES.size)
        for suffix in suffixes
    )


CSV_SPECS = (
    CsvSpec("joint_pos.csv", "joint_pos", _joint_headers("joint_"), _identity_2d),
    CsvSpec("joint_vel.csv", "joint_vel", _joint_headers("joint_vel_"), _identity_2d),
    CsvSpec("body_pos.csv", "body_pos_w", _body_headers(("x", "y", "z")), _flatten_frames),
    CsvSpec(
        "body_quat.csv",
        "body_quat_w",
        _body_headers(("w", "x", "y", "z")),
        _flatten_frames,
    ),
    CsvSpec(
        "body_lin_vel.csv",
        "body_lin_vel_w",
        _body_headers(("vel_x", "vel_y", "vel_z")),
        _flatten_frames,
    ),
    CsvSpec(
        "body_ang_vel.csv",
        "body_ang_vel_w",
        _body_headers(("angvel_x", "angvel_y", "angvel_z")),
        _flatten_frames,
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_motion_name(value: str) -> str:
    if (
        value in {"", ".", ".."}
        or value.startswith(".")
        or re.fullmatch(r"[A-Za-z0-9._-]+", value) is None
    ):
        raise ConversionError(
            "motion name must be visible and contain only letters, digits, '.', '_' or '-'"
        )
    return value


def _copy_numeric_array(archive: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    if key not in archive.files:
        raise ConversionError(f"NPZ is missing required array {key!r}")
    array = np.asarray(archive[key])
    if not (
        np.issubdtype(array.dtype, np.floating)
        or np.issubdtype(array.dtype, np.integer)
    ):
        raise ConversionError(f"{key} must be real numeric data, got dtype {array.dtype}")
    if not np.isfinite(array).all():
        bad = int(array.size - np.count_nonzero(np.isfinite(array)))
        raise ConversionError(f"{key} contains {bad} NaN or infinite values")
    return array.copy()


def _validate_names(
    archive: np.lib.npyio.NpzFile,
    key: str,
    expected: Sequence[str],
    assume_isaaclab_order: bool,
) -> str:
    if key not in archive.files:
        if not assume_isaaclab_order:
            raise ConversionError(
                f"NPZ has no {key}; pass --assume-isaaclab-order only if the "
                "producer is known to use the canonical G1 IsaacLab layout"
            )
        return "absent in source; canonical IsaacLab order explicitly assumed"

    actual_array = np.asarray(archive[key])
    if actual_array.ndim != 1:
        raise ConversionError(f"{key} must be one-dimensional, got {actual_array.shape}")
    actual = tuple(str(item) for item in actual_array.tolist())
    expected_tuple = tuple(expected)
    if actual != expected_tuple:
        mismatch = next(
            (
                index,
                actual[index] if index < len(actual) else "<missing>",
                expected_tuple[index] if index < len(expected_tuple) else "<extra>",
            )
            for index in range(max(len(actual), len(expected_tuple)))
            if index >= len(actual)
            or index >= len(expected_tuple)
            or actual[index] != expected_tuple[index]
        )
        raise ConversionError(
            f"{key} is not in canonical IsaacLab order: index {mismatch[0]} is "
            f"{mismatch[1]!r}, expected {mismatch[2]!r}"
        )
    return f"validated {len(expected_tuple)} names against canonical IsaacLab order"


def load_and_prepare(
    source_path: str | Path,
    motion_name: str | None = None,
    *,
    assume_isaaclab_order: bool = False,
) -> PreparedMotion:
    """Load, validate, and reduce one NPZ archive to the release 14-body layout."""

    source = Path(source_path).expanduser().resolve()
    if not source.is_file():
        raise ConversionError(f"NPZ file does not exist: {source}")
    if source.suffix.lower() != ".npz":
        raise ConversionError(f"input must have a .npz suffix: {source}")
    name = _safe_motion_name(motion_name or source.stem)

    try:
        archive_context = np.load(source, allow_pickle=False)
    except Exception as exc:
        raise ConversionError(f"failed to load {source}: {exc}") from exc

    with archive_context as archive:
        if "fps" not in archive.files:
            raise ConversionError("NPZ is missing required array 'fps'")
        fps_array = np.asarray(archive["fps"])
        if fps_array.size != 1 or not np.issubdtype(fps_array.dtype, np.number):
            raise ConversionError(f"fps must contain one numeric value, got {fps_array!r}")
        fps_value = float(fps_array.reshape(-1)[0])
        if not np.isfinite(fps_value) or fps_value != EXPECTED_FPS:
            raise ConversionError(
                f"reference motion must be {EXPECTED_FPS} Hz, got fps={fps_value}"
            )

        source_arrays = {key: _copy_numeric_array(archive, key) for key in REQUIRED_ARRAYS}
        joint_order_validation = _validate_names(
            archive,
            "joint_names",
            G1_ISAACLAB_JOINT_NAMES,
            assume_isaaclab_order,
        )
        body_order_validation = _validate_names(
            archive,
            "body_names",
            G1_ISAACLAB_BODY_NAMES,
            assume_isaaclab_order,
        )

    joint_pos = source_arrays["joint_pos"]
    if joint_pos.ndim != 2 or joint_pos.shape[1] != NUM_JOINTS:
        raise ConversionError(
            f"joint_pos must have shape (T, {NUM_JOINTS}), got {joint_pos.shape}"
        )
    timesteps = int(joint_pos.shape[0])
    if timesteps <= 0:
        raise ConversionError("reference motion contains no frames")

    expected_shapes = {
        "joint_pos": (timesteps, NUM_JOINTS),
        "joint_vel": (timesteps, NUM_JOINTS),
        "body_pos_w": (timesteps, NUM_SOURCE_BODIES, 3),
        "body_quat_w": (timesteps, NUM_SOURCE_BODIES, 4),
        "body_lin_vel_w": (timesteps, NUM_SOURCE_BODIES, 3),
        "body_ang_vel_w": (timesteps, NUM_SOURCE_BODIES, 3),
    }
    for key, expected_shape in expected_shapes.items():
        if source_arrays[key].shape != expected_shape:
            raise ConversionError(
                f"{key} must have shape {expected_shape}, got {source_arrays[key].shape}"
            )

    quaternion_norms = np.linalg.norm(
        source_arrays["body_quat_w"].astype(np.float64), axis=-1
    )
    quaternion_max_deviation = float(np.max(np.abs(quaternion_norms - 1.0)))
    if quaternion_max_deviation > QUATERNION_NORM_TOLERANCE:
        raise ConversionError(
            "body_quat_w contains non-unit quaternions: maximum norm deviation is "
            f"{quaternion_max_deviation:.6g}, limit is {QUATERNION_NORM_TOLERANCE:g}"
        )

    prepared_arrays = {
        "joint_pos": source_arrays["joint_pos"],
        "joint_vel": source_arrays["joint_vel"],
        "body_pos_w": source_arrays["body_pos_w"][:, RELEASE_BODY_INDEXES, :],
        "body_quat_w": source_arrays["body_quat_w"][:, RELEASE_BODY_INDEXES, :],
        "body_lin_vel_w": source_arrays["body_lin_vel_w"][:, RELEASE_BODY_INDEXES, :],
        "body_ang_vel_w": source_arrays["body_ang_vel_w"][:, RELEASE_BODY_INDEXES, :],
    }

    motion = PreparedMotion(
        name=name,
        source_path=source,
        source_sha256=sha256_file(source),
        fps=EXPECTED_FPS,
        timesteps=timesteps,
        arrays=prepared_arrays,
        joint_order_validation=joint_order_validation,
        body_order_validation=body_order_validation,
        quaternion_max_norm_deviation=quaternion_max_deviation,
    )
    validate_prepared_motion(motion)
    return motion


def validate_prepared_motion(motion: PreparedMotion) -> None:
    """Validate a release-layout motion before it is serialized.

    ``load_and_prepare`` performs stricter validation of the original 30-body
    NPZ. This public validator covers the reduced 14-body contract as well so
    derived motions can safely use :func:`publish_prepared_motion`.
    """

    _safe_motion_name(motion.name)
    if motion.fps != EXPECTED_FPS:
        raise ConversionError(
            f"prepared motion must be {EXPECTED_FPS} Hz, got {motion.fps}"
        )
    if motion.timesteps <= 0:
        raise ConversionError("prepared motion contains no frames")
    if set(motion.arrays) != set(REQUIRED_ARRAYS):
        missing = sorted(set(REQUIRED_ARRAYS) - set(motion.arrays))
        extra = sorted(set(motion.arrays) - set(REQUIRED_ARRAYS))
        raise ConversionError(
            f"prepared motion has the wrong array keys; missing={missing}, extra={extra}"
        )

    body_count = int(RELEASE_BODY_INDEXES.size)
    expected_shapes = {
        "joint_pos": (motion.timesteps, NUM_JOINTS),
        "joint_vel": (motion.timesteps, NUM_JOINTS),
        "body_pos_w": (motion.timesteps, body_count, 3),
        "body_quat_w": (motion.timesteps, body_count, 4),
        "body_lin_vel_w": (motion.timesteps, body_count, 3),
        "body_ang_vel_w": (motion.timesteps, body_count, 3),
    }
    for key, expected_shape in expected_shapes.items():
        array = np.asarray(motion.arrays[key])
        if array.shape != expected_shape:
            raise ConversionError(
                f"prepared {key} must have shape {expected_shape}, got {array.shape}"
            )
        if not np.issubdtype(array.dtype, np.floating):
            raise ConversionError(
                f"prepared {key} must use a floating dtype, got {array.dtype}"
            )
        if not np.isfinite(array).all():
            bad = int(array.size - np.count_nonzero(np.isfinite(array)))
            raise ConversionError(
                f"prepared {key} contains {bad} NaN or infinite values"
            )

    quaternion_norms = np.linalg.norm(
        motion.arrays["body_quat_w"].astype(np.float64), axis=-1
    )
    quaternion_max_deviation = float(np.max(np.abs(quaternion_norms - 1.0)))
    if quaternion_max_deviation > QUATERNION_NORM_TOLERANCE:
        raise ConversionError(
            "prepared body_quat_w contains non-unit quaternions: maximum norm "
            f"deviation is {quaternion_max_deviation:.6g}, limit is "
            f"{QUATERNION_NORM_TOLERANCE:g}"
        )
    if not np.isfinite(motion.quaternion_max_norm_deviation):
        raise ConversionError("quaternion_max_norm_deviation must be finite")
    if (
        motion.quaternion_max_norm_deviation > QUATERNION_NORM_TOLERANCE
        or motion.quaternion_max_norm_deviation + 1.0e-7 < quaternion_max_deviation
    ):
        raise ConversionError(
            "quaternion_max_norm_deviation is inconsistent with body_quat_w: "
            f"recorded={motion.quaternion_max_norm_deviation:.9g}, "
            f"actual={quaternion_max_deviation:.9g}"
        )
    for note in motion.derivation_notes:
        if not note.strip() or "\n" in note or "\r" in note:
            raise ConversionError(
                "each derivation note must be non-empty and contain no newlines"
            )


def _write_csv(path: Path, array: np.ndarray, headers: Sequence[str]) -> None:
    if array.ndim != 2 or array.shape[1] != len(headers):
        raise ConversionError(
            f"internal CSV shape error for {path.name}: {array.shape} vs {len(headers)} headers"
        )
    np.savetxt(
        path,
        array,
        delimiter=",",
        fmt=CSV_FLOAT_FORMAT,
        header=",".join(headers),
        comments="",
    )


def _body_index_text() -> str:
    return "[ " + " ".join(str(int(value)) for value in RELEASE_BODY_INDEXES) + "]"


def _write_metadata(path: Path, motion: PreparedMotion) -> None:
    lines = [
        f"Metadata for: {motion.name}",
        "=" * 30,
        "",
        "Body part indexes:",
        _body_index_text(),
        "",
        f"Total timesteps: {motion.timesteps}",
        f"FPS: {motion.fps}",
        "",
        "Data arrays summary:",
    ]
    for key in REQUIRED_ARRAYS:
        array = motion.arrays[key]
        lines.append(f"  {key}: {array.shape} ({array.dtype})")
    lines.extend(
        [
            f"  _body_indexes: ({RELEASE_BODY_INDEXES.size},) ({RELEASE_BODY_INDEXES.dtype})",
            "  time_step_total: () (int64)",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_info(path: Path, motion: PreparedMotion) -> None:
    lines = [
        f"Motion Information: {motion.name}",
        "=" * 50,
        "",
        f"Source NPZ: {motion.source_path}",
        f"Source SHA-256: {motion.source_sha256}",
        f"FPS: {motion.fps}",
        f"Total timesteps: {motion.timesteps}",
        f"CSV float format: {CSV_FLOAT_FORMAT}",
        f"Body part indexes: {_body_index_text()}",
        f"Joint order: {motion.joint_order_validation}",
        f"Body order: {motion.body_order_validation}",
        "Quaternion convention: world-frame (w, x, y, z)",
        "Quaternion max norm deviation: "
        f"{motion.quaternion_max_norm_deviation:.9g}",
    ]
    if motion.derivation_notes:
        lines.extend(["", "Derivation:"])
        lines.extend(f"  {note}" for note in motion.derivation_notes)
    lines.append("")
    for key in REQUIRED_ARRAYS:
        array = motion.arrays[key]
        flat = array.reshape(-1)
        sample = np.array2string(flat[:5], precision=9, separator=" ")
        lines.extend(
            [
                f"{key}:",
                f"  Shape: {array.shape}",
                f"  Dtype: {array.dtype}",
                f"  Range: [{float(flat.min()):.9g}, {float(flat.max()):.9g}]",
                f"  Sample: {sample}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def verify_generated_motion(
    motion_dir: str | Path, motion: PreparedMotion
) -> dict[str, float]:
    """Read generated files back and verify exact float32 round-tripping."""

    directory = Path(motion_dir)
    maximum_errors: dict[str, float] = {}
    for spec in CSV_SPECS:
        path = directory / spec.filename
        if not path.is_file():
            raise ConversionError(f"generated CSV is missing: {path}")
        with path.open("r", encoding="utf-8") as stream:
            actual_header = stream.readline().rstrip("\r\n")
        expected_header = ",".join(spec.headers)
        if actual_header != expected_header:
            raise ConversionError(
                f"header mismatch in {path.name}: {actual_header!r} != {expected_header!r}"
            )

        generated = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
        expected = spec.reshape(motion.arrays[spec.array_key])
        if generated.shape != expected.shape:
            raise ConversionError(
                f"shape mismatch in {path.name}: {generated.shape} != {expected.shape}"
            )
        if not np.isfinite(generated).all():
            raise ConversionError(f"generated CSV contains NaN or infinity: {path}")
        maximum_error = float(
            np.max(np.abs(generated.astype(np.float64) - expected.astype(np.float64)))
        )
        maximum_errors[spec.filename] = maximum_error
        if not np.array_equal(generated.astype(np.float32), expected.astype(np.float32)):
            raise ConversionError(
                f"{path.name} does not round-trip exactly to the source float32 values; "
                f"maximum absolute error is {maximum_error:.9g}"
            )

    metadata = (directory / "metadata.txt").read_text(encoding="utf-8")
    if _body_index_text() not in metadata:
        raise ConversionError("metadata.txt does not contain the release body-index mapping")
    if f"Total timesteps: {motion.timesteps}" not in metadata:
        raise ConversionError("metadata.txt contains the wrong timestep count")
    if f"FPS: {motion.fps}" not in metadata:
        raise ConversionError("metadata.txt contains the wrong FPS")
    if not (directory / "info.txt").is_file():
        raise ConversionError("generated info.txt is missing")
    return maximum_errors


def publish_prepared_motion(
    motion: PreparedMotion,
    output_root: str | Path,
) -> tuple[Path, dict[str, float]]:
    """Validate and atomically publish one prepared reference dataset."""

    validate_prepared_motion(motion)
    destination = Path(output_root).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise ConversionError(
            f"output root already exists; refusing to overwrite it: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)

    work_root = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        work_motion_dir = work_root / motion.name
        work_motion_dir.mkdir()
        for spec in CSV_SPECS:
            _write_csv(
                work_motion_dir / spec.filename,
                spec.reshape(motion.arrays[spec.array_key]),
                spec.headers,
            )
        _write_metadata(work_motion_dir / "metadata.txt", motion)
        _write_info(work_motion_dir / "info.txt", motion)
        maximum_errors = verify_generated_motion(work_motion_dir, motion)
        os.replace(work_root, destination)
    except Exception:
        shutil.rmtree(work_root, ignore_errors=True)
        raise

    return destination / motion.name, maximum_errors


def convert_npz(
    source_path: str | Path,
    output_root: str | Path,
    motion_name: str | None = None,
    *,
    assume_isaaclab_order: bool = False,
) -> tuple[Path, dict[str, float]]:
    """Convert and atomically publish one NPZ reference dataset."""

    motion = load_and_prepare(
        source_path,
        motion_name,
        assume_isaaclab_order=assume_isaaclab_order,
    )
    return publish_prepared_motion(motion, output_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a 50 Hz, 29-DoF G1 NPZ motion to deployment CSV files."
    )
    parser.add_argument("npz_file", help="Source NPZ archive")
    parser.add_argument("--name", help="Motion folder name (default: NPZ filename stem)")
    parser.add_argument(
        "--output-root",
        required=True,
        help="Dataset root to create; the motion folder is created inside it",
    )
    parser.add_argument(
        "--assume-isaaclab-order",
        action="store_true",
        help=(
            "accept a source without joint_names/body_names and explicitly record "
            "that canonical G1 IsaacLab order was assumed"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output_dir, maximum_errors = convert_npz(
            args.npz_file,
            args.output_root,
            args.name,
            assume_isaaclab_order=args.assume_isaaclab_order,
        )
    except (ConversionError, OSError) as exc:
        print(f"Conversion failed: {exc}", file=sys.stderr)
        return 1

    print(f"Converted reference motion: {output_dir}")
    print("Verified generated CSV files (maximum absolute serialization error):")
    for filename, error in maximum_errors.items():
        print(f"  {filename}: {error:.9g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
