#!/usr/bin/env python3
"""Validate reconstructed regular-SONIC G1 inputs against recorded tokens.

This is a diagnostic only; the rollout publisher never sends ``token_state``.
It loads the trusted local ``sonic_release/last.pt`` checkpoint, rebuilds the
G1 MLP and FSQ bottleneck from its weights, and compares its 64-D output with
the first 64 ``token_state`` columns in the recording.

Two checks are reported:

* ``recorded_input`` feeds the CSV's active 640-D ``reference_motion`` exactly.
  This proves the MLP/FSQ implementation and CSV/token row alignment.
* ``base_sample_candidates`` reconstructs future anchor orientations from the
  slot-zero reference sequence for each pelvis sampling hypothesis.  It can
  distinguish hypotheses only through cross-frame orientation; slot zero alone
  cancels algebraically for every hypothesis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from change_ckpt.reference_data import (
        ACTIVE_REFERENCE_SIZE,
        ReferenceDataError,
        ReferenceSequence,
        _base_array_from_samples,
        _future_views,
        _infer_future_lags,
        _relative_quaternions,
        load_reference_sequence,
        quat_conjugate_wxyz,
        quat_multiply_wxyz,
        quat_to_matrix_wxyz,
    )
except ModuleNotFoundError:
    from reference_data import (  # type: ignore[no-redef]
        ACTIVE_REFERENCE_SIZE,
        ReferenceDataError,
        ReferenceSequence,
        _base_array_from_samples,
        _future_views,
        _infer_future_lags,
        _relative_quaternions,
        load_reference_sequence,
        quat_conjugate_wxyz,
        quat_multiply_wxyz,
        quat_to_matrix_wxyz,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORDING = (
    REPO_ROOT / "sample_data/ztj/20260612/20260720_144342_g1_sim"
)
DEFAULT_CHECKPOINT = REPO_ROOT / "sonic_release/last.pt"
TOKEN_DIMENSION = 64
FSQ_LEVELS = 32
FSQ_EPSILON = 1e-3


def _csv_path(path: str | Path) -> Path:
    result = Path(path).expanduser().resolve()
    if result.is_dir():
        result = result / "data.csv"
    if not result.is_file():
        raise ReferenceDataError(f"recording CSV does not exist: {result}")
    return result


def _read_recorded_tokens(
    path: str | Path, expected_policy_seq: np.ndarray
) -> tuple[np.ndarray, float]:
    """Read the first token per unique policy_seq and check hold invariance."""

    path = _csv_path(path)
    tokens: list[np.ndarray] = []
    sequences: list[int] = []
    max_change = 0.0
    previous_sequence: int | None = None
    previous_token: np.ndarray | None = None
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ReferenceDataError(f"empty CSV: {path}") from exc
        lookup = {name: index for index, name in enumerate(header)}
        if "policy_seq" not in lookup:
            raise ReferenceDataError("CSV has no policy_seq column")
        token_indices: list[int] = []
        for index in range(TOKEN_DIMENSION):
            name = f"token_state[{index}]"
            if name not in lookup:
                raise ReferenceDataError(f"CSV is missing {name}")
            token_indices.append(lookup[name])
        valid_index = lookup.get("policy_valid")

        for row_number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ReferenceDataError(
                    f"CSV row {row_number} has {len(row)} fields; expected {len(header)}"
                )
            if valid_index is not None and not bool(int(float(row[valid_index]))):
                continue
            sequence = int(float(row[lookup["policy_seq"]]))
            token = np.asarray([float(row[index]) for index in token_indices], dtype=np.float32)
            if not np.all(np.isfinite(token)):
                raise ReferenceDataError(f"non-finite token at CSV row {row_number}")
            if sequence != previous_sequence:
                sequences.append(sequence)
                tokens.append(token)
                previous_sequence = sequence
                previous_token = token
            else:
                assert previous_token is not None
                max_change = max(max_change, float(np.max(np.abs(token - previous_token))))

    actual = np.asarray(sequences, dtype=np.int64)
    if not np.array_equal(actual, expected_policy_seq):
        raise ReferenceDataError(
            "token policy_seq values do not align with the deduplicated reference sequence"
        )
    return np.stack(tokens), max_change


def _load_encoder_state(checkpoint_path: Path) -> dict[str, Any]:
    try:
        import torch
        import trl.trainer.utils
    except ImportError as exc:
        raise ReferenceDataError(
            "this diagnostic needs PyTorch and TRL; run it with the isaaclab Python"
        ) from exc

    # Old trusted checkpoints pickle this metadata class by its former TRL
    # module path.  No trainer code is needed for an inference-only state dict.
    if not hasattr(trl.trainer.utils, "OnlineTrainerState"):
        compatibility_type = type("OnlineTrainerState", (), {})
        compatibility_type.__module__ = "trl.trainer.utils"
        trl.trainer.utils.OnlineTrainerState = compatibility_type

    if not checkpoint_path.is_file():
        raise ReferenceDataError(f"regular checkpoint does not exist: {checkpoint_path}")
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except Exception as exc:  # noqa: BLE001 - report checkpoint compatibility cleanly
        raise ReferenceDataError(f"cannot load regular checkpoint: {exc}") from exc
    state = checkpoint.get("policy_state_dict") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict):
        raise ReferenceDataError("checkpoint has no policy_state_dict")
    return state


def _encode_regular_g1(inputs: np.ndarray, state: dict[str, Any], batch_size: int) -> np.ndarray:
    """Run the checkpoint's 640->64 SiLU MLP and its parameter-free FSQ."""

    import torch
    import torch.nn.functional as functional

    prefix = "actor_module.encoders.g1.module."
    linear_indices = (0, 2, 4, 6, 8)
    expected_shapes = ((2048, 640), (1024, 2048), (512, 1024), (512, 512), (64, 512))
    weights: list[Any] = []
    biases: list[Any] = []
    for index, shape in zip(linear_indices, expected_shapes, strict=True):
        weight_key = f"{prefix}{index}.weight"
        bias_key = f"{prefix}{index}.bias"
        if weight_key not in state or bias_key not in state:
            raise ReferenceDataError(f"checkpoint is missing {weight_key} or {bias_key}")
        if tuple(state[weight_key].shape) != shape:
            raise ReferenceDataError(
                f"unexpected {weight_key} shape {tuple(state[weight_key].shape)}; expected {shape}"
            )
        weights.append(state[weight_key].detach().to(device="cpu", dtype=torch.float32))
        biases.append(state[bias_key].detach().to(device="cpu", dtype=torch.float32))

    output_chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, inputs.shape[0], batch_size):
            value = torch.as_tensor(inputs[start : start + batch_size], dtype=torch.float32)
            for layer, (weight, bias) in enumerate(zip(weights, biases, strict=True)):
                value = functional.linear(value, weight, bias)
                if layer + 1 < len(weights):
                    value = functional.silu(value)

            # vector_quantize_pytorch.FSQ(levels=[32] * 32), default settings.
            value = value.reshape(-1, 2, 32)
            half_level = (FSQ_LEVELS - 1) * (1.0 + FSQ_EPSILON) / 2.0
            shift = math.atanh(0.5 / half_level)
            value = torch.round(torch.tanh(value + shift) * half_level - 0.5)
            value = value / (FSQ_LEVELS // 2)
            output_chunks.append(value.reshape(-1, TOKEN_DIMENSION).numpy())
    return np.concatenate(output_chunks, axis=0)


def _error_summary(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    difference = np.abs(np.asarray(actual) - np.asarray(expected))
    return {
        "mean_abs": float(np.mean(difference)),
        "max_abs": float(np.max(difference)),
        "elements_different_gt_1e-6": int(np.count_nonzero(difference > 1e-6)),
        "frames_exact_within_1e-6": int(np.count_nonzero(np.max(difference, axis=1) <= 1e-6)),
        "frames": int(difference.shape[0]),
    }


def _regular_encoder_layout(reference_motion: np.ndarray) -> np.ndarray:
    """Apply the legacy training-time temporal reshape used by the G1 MLP.

    ``command_multi_future`` is first built as ``[all q, all dq]`` (580-D),
    then ``non_flatten=True`` reshapes that flat buffer to ``[10, 58]``.  The
    6-D orientation is concatenated per temporal row afterwards.  This is not
    equivalent to feeding the canonical ``[all q, all dq, all ori]`` buffer
    directly; the behavior is documented as the historical observation bug in
    ``gear_sonic/trl/losses/token_losses.py``.
    """

    reference_motion = np.asarray(reference_motion, dtype=np.float32)
    if reference_motion.ndim != 2 or reference_motion.shape[1] != ACTIVE_REFERENCE_SIZE:
        raise ReferenceDataError(
            f"expected canonical reference input [N,640], got {reference_motion.shape}"
        )
    command = reference_motion[:, :580].reshape(-1, 10, 58)
    orientation = reference_motion[:, 580:640].reshape(-1, 10, 6)
    return np.concatenate((command, orientation), axis=-1).reshape(-1, 640)


def _candidate_input(sequence: ReferenceSequence, mode: str, lags: Sequence[int]) -> np.ndarray:
    """Rebuild the 640-D input using a consistent candidate robot sample clock."""

    positions, velocities, raw_orientation = _future_views(sequence)
    base = _base_array_from_samples(sequence, mode)
    slot_zero_relative, _ = _relative_quaternions(raw_orientation[:, 0])
    reference_world = quat_multiply_wxyz(base, slot_zero_relative)
    rebuilt_orientation = np.empty_like(raw_orientation, dtype=np.float64)
    for frame in range(sequence.num_frames):
        base_inverse = quat_conjugate_wxyz(base[frame])
        for slot, lag in enumerate(lags):
            target = min(frame + int(lag), sequence.num_frames - 1)
            relative = quat_multiply_wxyz(base_inverse, reference_world[target])
            rebuilt_orientation[frame, slot] = quat_to_matrix_wxyz(relative)[:, :2].reshape(6)
    return np.concatenate(
        (
            positions.reshape(sequence.num_frames, -1),
            velocities.reshape(sequence.num_frames, -1),
            rebuilt_orientation.reshape(sequence.num_frames, -1),
        ),
        axis=1,
    ).astype(np.float32)


def run(args: argparse.Namespace) -> dict[str, Any]:
    sequence = load_reference_sequence(args.recording, base_sample_mode="previous-last")
    recorded_tokens, token_hold_change = _read_recorded_tokens(
        args.recording, sequence.policy_seq
    )
    state = _load_encoder_state(Path(args.checkpoint).expanduser().resolve())
    recorded_input = sequence.reference_motion[:, :ACTIVE_REFERENCE_SIZE].astype(np.float32)
    reproduced = _encode_regular_g1(
        _regular_encoder_layout(recorded_input), state, args.batch_size
    )
    lags = [int(item["inferred_policy_lag"]) for item in _infer_future_lags(sequence)]

    result: dict[str, Any] = {
        "recording": str(sequence.csv_path),
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "policy_frames": sequence.num_frames,
        "future_slot_policy_lags": lags,
        "max_token_change_inside_policy_hold": token_hold_change,
        "recorded_input": _error_summary(reproduced, recorded_tokens),
        "base_sample_candidates": {},
        "interpretation": (
            "A candidate is meaningful only if recorded_input first validates the exact "
            "checkpoint/FSQ/token alignment. Lower candidate token and orientation error is better."
        ),
    }
    raw_orientation = recorded_input[:, 580:640]
    for mode in args.base_sample:
        candidate = _candidate_input(sequence, mode, lags)
        candidate_tokens = _encode_regular_g1(
            _regular_encoder_layout(candidate), state, args.batch_size
        )
        result["base_sample_candidates"][mode] = {
            "anchor_orientation": _error_summary(candidate[:, 580:640], raw_orientation),
            "token_vs_csv": _error_summary(candidate_tokens, recorded_tokens),
        }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", default=str(DEFAULT_RECORDING))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument(
        "--base-sample",
        action="append",
        default=None,
        help="Candidate mode; repeat as needed (default: previous-last and previous-index5)",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", help="Optional JSON output under change_ckpt/data")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.base_sample is None:
        args.base_sample = ["previous-last", "previous-index5"]
    if args.batch_size <= 0:
        print("ERROR: --batch-size must be positive", file=sys.stderr)
        return 2
    try:
        result = run(args)
        rendered = json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True)
        print(rendered)
        if args.output:
            output = Path(args.output).expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered + "\n", encoding="utf-8")
        return 0
    except (ReferenceDataError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
