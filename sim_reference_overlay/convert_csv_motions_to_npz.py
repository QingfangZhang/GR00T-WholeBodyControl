#!/usr/bin/env python3
"""Reconstruct canonical 30-body G1 NPZ motions from deployment CSV data.

Deployment CSV files retain 29 joint signals but only 14 selected body signals.
This converter uses the same MuJoCo G1 scene as the safe-reference builder to
recover the other 16 body poses with forward kinematics.  The original 14 body
signals are then copied back verbatim so CSV -> NPZ -> CSV round-trips exactly.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Sequence

import numpy as np

import build_real_safe_reference as safe
import convert_npz_motions as csvio


BODY_COMPONENTS = {
    "body_pos_w": 3,
    "body_quat_w": 4,
    "body_lin_vel_w": 3,
    "body_ang_vel_w": 3,
}
NPZ_KEYS = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "joint_names",
    "body_names",
)


@dataclass(frozen=True)
class ReconstructionReport:
    timesteps: int
    fk_release_position_error: float
    fk_release_quaternion_error: float
    derived_release_linear_velocity_error: float
    derived_release_angular_velocity_error: float


def _required_csv_names() -> set[str]:
    return {spec.filename for spec in csvio.CSV_SPECS}


def resolve_motion_directory(path: str | Path) -> Path:
    """Accept either one motion directory or a root containing exactly one."""

    candidate = Path(path).expanduser().resolve()
    if not candidate.is_dir():
        raise csvio.ConversionError(f"CSV path is not a directory: {candidate}")
    required = _required_csv_names()
    if required.issubset({item.name for item in candidate.iterdir() if item.is_file()}):
        return candidate

    matches = [
        child
        for child in candidate.iterdir()
        if child.is_dir()
        and required.issubset({item.name for item in child.iterdir() if item.is_file()})
    ]
    if len(matches) != 1:
        raise csvio.ConversionError(
            "CSV dataset root must contain exactly one motion directory; "
            f"found {[item.name for item in matches]} in {candidate}"
        )
    return matches[0]


def _metadata_integer(metadata: str, label: str) -> int:
    match = re.search(rf"^{re.escape(label)}:\s*(\d+)\s*$", metadata, re.MULTILINE)
    if match is None:
        raise csvio.ConversionError(f"metadata.txt is missing integer field {label!r}")
    return int(match.group(1))


def _validate_metadata(motion_dir: Path, timesteps: int) -> None:
    path = motion_dir / "metadata.txt"
    if not path.is_file():
        raise csvio.ConversionError(f"CSV motion is missing metadata.txt: {path}")
    metadata = path.read_text(encoding="utf-8")
    if _metadata_integer(metadata, "Total timesteps") != timesteps:
        raise csvio.ConversionError("metadata.txt timestep count does not match the CSV files")
    if _metadata_integer(metadata, "FPS") != csvio.EXPECTED_FPS:
        raise csvio.ConversionError(
            f"CSV motion must be {csvio.EXPECTED_FPS} Hz"
        )

    index_match = re.search(r"Body part indexes:\s*\[([^\]]+)\]", metadata)
    if index_match is None:
        raise csvio.ConversionError("metadata.txt is missing the body-index mapping")
    indexes = np.fromstring(index_match.group(1), sep=" ", dtype=np.int64)
    if not np.array_equal(indexes, csvio.RELEASE_BODY_INDEXES):
        raise csvio.ConversionError(
            "metadata.txt body indexes do not match the deployment CSV contract: "
            f"{indexes.tolist()}"
        )


def read_release_csv_motion(path: str | Path) -> tuple[Path, dict[str, np.ndarray]]:
    motion_dir = resolve_motion_directory(path)
    arrays: dict[str, np.ndarray] = {}
    timesteps: int | None = None
    body_count = int(csvio.RELEASE_BODY_INDEXES.size)

    for spec in csvio.CSV_SPECS:
        csv_path = motion_dir / spec.filename
        with csv_path.open("r", encoding="utf-8") as stream:
            actual_header = stream.readline().rstrip("\r\n")
        expected_header = ",".join(spec.headers)
        if actual_header != expected_header:
            raise csvio.ConversionError(
                f"header mismatch in {csv_path.name}: "
                f"{actual_header!r} != {expected_header!r}"
            )

        flat = np.loadtxt(
            csv_path,
            delimiter=",",
            skiprows=1,
            dtype=np.float32,
            ndmin=2,
        )
        if flat.shape[1] != len(spec.headers):
            raise csvio.ConversionError(
                f"{csv_path.name} has {flat.shape[1]} columns, expected "
                f"{len(spec.headers)}"
            )
        if timesteps is None:
            timesteps = int(flat.shape[0])
        elif flat.shape[0] != timesteps:
            raise csvio.ConversionError(
                f"{csv_path.name} has {flat.shape[0]} frames, expected {timesteps}"
            )
        if not np.isfinite(flat).all():
            raise csvio.ConversionError(f"{csv_path.name} contains NaN or infinity")

        if spec.array_key in BODY_COMPONENTS:
            arrays[spec.array_key] = flat.reshape(
                flat.shape[0], body_count, BODY_COMPONENTS[spec.array_key]
            )
        else:
            arrays[spec.array_key] = flat

    if timesteps is None or timesteps < 3:
        raise csvio.ConversionError("CSV motion must contain at least three frames")
    _validate_metadata(motion_dir, timesteps)

    prepared = csvio.PreparedMotion(
        name=motion_dir.name,
        source_path=motion_dir,
        source_sha256="CSV reconstruction input",
        fps=csvio.EXPECTED_FPS,
        timesteps=timesteps,
        arrays=arrays,
        joint_order_validation="canonical deployment CSV order",
        body_order_validation="canonical 14-body deployment CSV order",
        quaternion_max_norm_deviation=float(
            np.max(
                np.abs(
                    np.linalg.norm(arrays["body_quat_w"].astype(np.float64), axis=-1)
                    - 1.0
                )
            )
        ),
    )
    csvio.validate_prepared_motion(prepared)
    return motion_dir, arrays


def _normalize_continuous_quaternions(quaternions: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternions, dtype=np.float64).copy()
    norms = np.linalg.norm(result, axis=-1, keepdims=True)
    if np.any(norms == 0.0):
        raise csvio.ConversionError("FK produced a zero quaternion")
    result /= norms
    for frame in range(1, result.shape[0]):
        flip = np.sum(result[frame - 1] * result[frame], axis=-1) < 0.0
        result[frame, flip] *= -1.0
    return result.astype(np.float32)


def reconstruct_npz_payload(
    release_arrays: dict[str, np.ndarray],
    scene_path: str | Path,
) -> tuple[dict[str, np.ndarray], ReconstructionReport]:
    timesteps = int(release_arrays["joint_pos"].shape[0])
    kinematics = safe.G1Kinematics(Path(scene_path))
    full_body_ids = np.asarray(
        [kinematics._body_id(name) for name in csvio.G1_ISAACLAB_BODY_NAMES],
        dtype=np.int64,
    )
    root_rotations = safe.rotation_from_wxyz(release_arrays["body_quat_w"][:, 0])

    full_pos = np.empty((timesteps, csvio.NUM_SOURCE_BODIES, 3), dtype=np.float64)
    full_quat = np.empty((timesteps, csvio.NUM_SOURCE_BODIES, 4), dtype=np.float64)
    for frame in range(timesteps):
        kinematics._set_pose(
            release_arrays["joint_pos"][frame],
            release_arrays["body_pos_w"][frame, 0],
            root_rotations[frame],
        )
        full_pos[frame] = kinematics.data.xpos[full_body_ids]
        full_quat[frame] = kinematics.data.xquat[full_body_ids]

    full_quat_f32 = _normalize_continuous_quaternions(full_quat)
    release_indexes = csvio.RELEASE_BODY_INDEXES
    fk_pos_error = float(
        np.max(
            np.abs(
                full_pos[:, release_indexes].astype(np.float32)
                - release_arrays["body_pos_w"]
            )
        )
    )

    predicted_quat = full_quat_f32[:, release_indexes].astype(np.float64)
    observed_quat = release_arrays["body_quat_w"].astype(np.float64)
    sign = np.where(
        np.sum(predicted_quat * observed_quat, axis=-1, keepdims=True) < 0.0,
        -1.0,
        1.0,
    )
    fk_quat_error = float(np.max(np.abs(predicted_quat * sign - observed_quat)))
    if fk_pos_error > 2.0e-5 or fk_quat_error > 2.0e-5:
        raise csvio.ConversionError(
            "CSV body poses are inconsistent with the configured MuJoCo G1 scene: "
            f"position error={fk_pos_error:.9g}, quaternion error={fk_quat_error:.9g}"
        )

    full_pos_f32 = full_pos.astype(np.float32)
    full_pos_f32[:, release_indexes] = release_arrays["body_pos_w"]
    full_quat_f32[:, release_indexes] = release_arrays["body_quat_w"]

    full_lin_vel = safe._gradient(full_pos_f32)
    full_ang_vel = safe._angular_velocity(full_quat_f32)
    lin_vel_error = float(
        np.max(
            np.abs(
                full_lin_vel[:, release_indexes]
                - release_arrays["body_lin_vel_w"]
            )
        )
    )
    ang_vel_error = float(
        np.max(
            np.abs(
                full_ang_vel[:, release_indexes]
                - release_arrays["body_ang_vel_w"]
            )
        )
    )
    full_lin_vel[:, release_indexes] = release_arrays["body_lin_vel_w"]
    full_ang_vel[:, release_indexes] = release_arrays["body_ang_vel_w"]

    payload = {
        "fps": np.asarray([csvio.EXPECTED_FPS], dtype=np.int64),
        "joint_pos": release_arrays["joint_pos"].astype(np.float32, copy=True),
        "joint_vel": release_arrays["joint_vel"].astype(np.float32, copy=True),
        "body_pos_w": full_pos_f32,
        "body_quat_w": full_quat_f32,
        "body_lin_vel_w": full_lin_vel.astype(np.float32, copy=False),
        "body_ang_vel_w": full_ang_vel.astype(np.float32, copy=False),
        "joint_names": np.asarray(csvio.G1_ISAACLAB_JOINT_NAMES, dtype="<U64"),
        "body_names": np.asarray(csvio.G1_ISAACLAB_BODY_NAMES, dtype="<U64"),
    }
    for key in csvio.REQUIRED_ARRAYS:
        if not np.isfinite(payload[key]).all():
            raise csvio.ConversionError(f"reconstructed {key} contains NaN or infinity")
    quaternion_deviation = float(
        np.max(
            np.abs(
                np.linalg.norm(payload["body_quat_w"].astype(np.float64), axis=-1)
                - 1.0
            )
        )
    )
    if quaternion_deviation > csvio.QUATERNION_NORM_TOLERANCE:
        raise csvio.ConversionError(
            "reconstructed full-body quaternions are not unit length: "
            f"maximum deviation={quaternion_deviation:.9g}"
        )

    return payload, ReconstructionReport(
        timesteps=timesteps,
        fk_release_position_error=fk_pos_error,
        fk_release_quaternion_error=fk_quat_error,
        derived_release_linear_velocity_error=lin_vel_error,
        derived_release_angular_velocity_error=ang_vel_error,
    )


def _verify_archive(
    path: Path,
    release_arrays: dict[str, np.ndarray],
) -> None:
    with np.load(path, allow_pickle=False) as archive:
        if tuple(archive.files) != NPZ_KEYS:
            raise csvio.ConversionError(
                f"generated NPZ keys are {archive.files}, expected {list(NPZ_KEYS)}"
            )
        expected_shapes = {
            "fps": (1,),
            "joint_pos": (release_arrays["joint_pos"].shape[0], csvio.NUM_JOINTS),
            "joint_vel": (release_arrays["joint_pos"].shape[0], csvio.NUM_JOINTS),
            "body_pos_w": (release_arrays["joint_pos"].shape[0], csvio.NUM_SOURCE_BODIES, 3),
            "body_quat_w": (release_arrays["joint_pos"].shape[0], csvio.NUM_SOURCE_BODIES, 4),
            "body_lin_vel_w": (release_arrays["joint_pos"].shape[0], csvio.NUM_SOURCE_BODIES, 3),
            "body_ang_vel_w": (release_arrays["joint_pos"].shape[0], csvio.NUM_SOURCE_BODIES, 3),
            "joint_names": (csvio.NUM_JOINTS,),
            "body_names": (csvio.NUM_SOURCE_BODIES,),
        }
        for key, shape in expected_shapes.items():
            if archive[key].shape != shape:
                raise csvio.ConversionError(
                    f"generated {key} has shape {archive[key].shape}, expected {shape}"
                )
        if archive["fps"].dtype != np.dtype(np.int64):
            raise csvio.ConversionError("generated fps does not use int64")
        for key in csvio.REQUIRED_ARRAYS:
            if archive[key].dtype != np.dtype(np.float32):
                raise csvio.ConversionError(f"generated {key} does not use float32")
        if archive["joint_names"].dtype != np.dtype("<U64"):
            raise csvio.ConversionError("generated joint_names does not use <U64")
        if archive["body_names"].dtype != np.dtype("<U64"):
            raise csvio.ConversionError("generated body_names does not use <U64")

    prepared = csvio.load_and_prepare(path, motion_name="reconstructed_verification")
    for key in csvio.REQUIRED_ARRAYS:
        if not np.array_equal(prepared.arrays[key], release_arrays[key]):
            raise csvio.ConversionError(
                f"generated NPZ does not round-trip {key} exactly to the source CSV"
            )


def write_npz_atomically(
    payload: dict[str, np.ndarray],
    release_arrays: dict[str, np.ndarray],
    output_path: str | Path,
) -> Path:
    destination = Path(output_path).expanduser().resolve()
    if destination.suffix.lower() != ".npz":
        raise csvio.ConversionError(f"output must have a .npz suffix: {destination}")
    if destination.exists() or destination.is_symlink():
        raise csvio.ConversionError(
            f"output already exists; refusing to overwrite it: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.tmp-",
        suffix=".npz",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as stream:
            np.savez(stream, **payload)
        _verify_archive(temporary, release_arrays)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def convert_csv_to_npz(
    csv_path: str | Path,
    output_path: str | Path,
    *,
    scene_path: str | Path | None = None,
) -> tuple[Path, ReconstructionReport]:
    _, release_arrays = read_release_csv_motion(csv_path)
    payload, report = reconstruct_npz_payload(
        release_arrays,
        scene_path or safe.default_scene_path(),
    )
    output = write_npz_atomically(payload, release_arrays, output_path)
    return output, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct a canonical 30-body G1 NPZ from one 14-body deployment "
            "CSV motion using MuJoCo forward kinematics."
        )
    )
    parser.add_argument("csv_path", help="Motion directory or single-motion dataset root")
    parser.add_argument("output_npz", help="New .npz file to create")
    parser.add_argument(
        "--scene",
        default=str(safe.default_scene_path()),
        help="MuJoCo G1 scene used to reconstruct the missing body signals",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output, report = convert_csv_to_npz(
            args.csv_path,
            args.output_npz,
            scene_path=args.scene,
        )
    except (csvio.ConversionError, safe.SafeMotionError, OSError, ValueError) as exc:
        print(f"CSV-to-NPZ conversion failed: {exc}", file=sys.stderr)
        return 1

    print(f"Generated canonical G1 NPZ: {output}")
    print(f"  frames={report.timesteps}, fps={csvio.EXPECTED_FPS}, bodies=30, joints=29")
    print(
        "  FK release-pose max errors [position/quaternion]="
        f"{report.fk_release_position_error:.9g}/"
        f"{report.fk_release_quaternion_error:.9g}"
    )
    print(
        "  derived release-velocity max errors [linear/angular]="
        f"{report.derived_release_linear_velocity_error:.9g}/"
        f"{report.derived_release_angular_velocity_error:.9g}"
    )
    print("  verified exact NPZ -> deployment CSV float32 round-trip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
