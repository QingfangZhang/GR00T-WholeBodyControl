#!/usr/bin/env python3
"""Export the legacy regular SONIC G1 wrapper for checkpoint verification.

This exporter is intentionally limited to the G1 encoder mode used by the CSV
rollout experiment.  It reconstructs the two inference networks directly from
``sonic_release/last.pt`` and writes the historical G1-specialized models:

* ``model_encoder.onnx``: 1751 -> 64
* ``model_decoder.onnx``: 994 -> 29

The official release encoder now used by the rollout launchers has a 1762-D
multiplexed input.  Do not overwrite ``change_ckpt/models/regular`` with this
1751-D verification export or pair it with the official observation config.

The regular G1 observation has a historical training-time reshape that must be
preserved.  The canonical C++ slice ``[q(290), dq(290), orientation(60)]`` is
converted to ``concat(reshape(q+dq, 10, 58), reshape(orientation, 10, 6))``
before the 640-D encoder MLP.  Its 64-D output is then passed through the same
parameter-free FSQ operation as the released policy.

Only load checkpoints you trust: PyTorch checkpoint metadata is unpickled.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = REPO_ROOT / "sonic_release/last.pt"
# Keep this legacy, G1-specialized 1751-D export separate from the official
# 1762-D release files stored in ``change_ckpt/models/regular``.
DEFAULT_OUTPUT_DIR = REPO_ROOT / "change_ckpt/models/regular_legacy_export"

ENCODER_INPUT_DIM = 1751
ENCODER_OUTPUT_DIM = 64
DECODER_INPUT_DIM = 994
DECODER_OUTPUT_DIM = 29

MODE_PREFIX_DIM = 4
G1_COMMAND_DIM = 580
G1_ORIENTATION_DIM = 60
G1_INPUT_DIM = G1_COMMAND_DIM + G1_ORIENTATION_DIM
G1_INPUT_END = MODE_PREFIX_DIM + G1_INPUT_DIM

FSQ_LEVELS = 32
FSQ_EPSILON = 1e-3


class ExportError(RuntimeError):
    """Raised when checkpoint validation or ONNX export fails."""


def _absolute(path: str | Path) -> Path:
    result = Path(path).expanduser()
    if result.is_absolute():
        return result.resolve()
    return (Path.cwd() / result).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_policy_state(torch: Any, checkpoint_path: Path) -> dict[str, Any]:
    """Load the policy state while supporting the checkpoint's old TRL class."""

    try:
        import trl.trainer.utils
    except ImportError as exc:
        raise ExportError(
            "TRL is required to read checkpoint metadata; run this script with "
            "the isaaclab Python environment"
        ) from exc

    if not hasattr(trl.trainer.utils, "OnlineTrainerState"):
        compatibility_type = type("OnlineTrainerState", (), {})
        compatibility_type.__module__ = "trl.trainer.utils"
        trl.trainer.utils.OnlineTrainerState = compatibility_type

    if not checkpoint_path.is_file():
        raise ExportError(f"checkpoint does not exist: {checkpoint_path}")

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except Exception as exc:  # noqa: BLE001 - convert checkpoint errors to CLI errors
        raise ExportError(f"cannot load checkpoint {checkpoint_path}: {exc}") from exc

    if not isinstance(checkpoint, dict):
        raise ExportError("checkpoint root is not a mapping")
    state = checkpoint.get("policy_state_dict")
    if state is None:
        state = checkpoint.get("actor_model_state_dict")
    if not isinstance(state, dict):
        raise ExportError("checkpoint has neither policy_state_dict nor actor_model_state_dict")
    return state


def _load_mlp_state(
    module: Any,
    state: dict[str, Any],
    *,
    prefix: str,
    expected_shapes: Sequence[tuple[int, int]],
) -> None:
    own_state: dict[str, Any] = {}
    linear_indices = tuple(range(0, 2 * len(expected_shapes), 2))
    for index, expected_shape in zip(linear_indices, expected_shapes, strict=True):
        weight_name = f"{prefix}{index}.weight"
        bias_name = f"{prefix}{index}.bias"
        if weight_name not in state or bias_name not in state:
            raise ExportError(f"checkpoint is missing {weight_name} or {bias_name}")
        weight = state[weight_name]
        bias = state[bias_name]
        if tuple(weight.shape) != expected_shape:
            raise ExportError(
                f"unexpected {weight_name} shape {tuple(weight.shape)}; "
                f"expected {expected_shape}"
            )
        if tuple(bias.shape) != (expected_shape[0],):
            raise ExportError(
                f"unexpected {bias_name} shape {tuple(bias.shape)}; "
                f"expected {(expected_shape[0],)}"
            )
        own_state[f"{index}.weight"] = weight
        own_state[f"{index}.bias"] = bias
    module.load_state_dict(own_state, strict=True)


def _build_models(torch: Any, state: dict[str, Any]) -> tuple[Any, Any]:
    nn = torch.nn

    def make_mlp(dimensions: Sequence[int]) -> Any:
        layers: list[Any] = []
        for layer, (input_dim, output_dim) in enumerate(
            zip(dimensions[:-1], dimensions[1:], strict=True)
        ):
            layers.append(nn.Linear(input_dim, output_dim))
            if layer + 1 < len(dimensions) - 1:
                layers.append(nn.SiLU())
        return nn.Sequential(*layers)

    encoder_mlp = make_mlp((640, 2048, 1024, 512, 512, 64))
    _load_mlp_state(
        encoder_mlp,
        state,
        prefix="actor_module.encoders.g1.module.",
        expected_shapes=(
            (2048, 640),
            (1024, 2048),
            (512, 1024),
            (512, 512),
            (64, 512),
        ),
    )

    decoder_mlp = make_mlp((994, 2048, 2048, 1024, 1024, 512, 512, 29))
    _load_mlp_state(
        decoder_mlp,
        state,
        prefix="actor_module.decoders.g1_dyn.module.",
        expected_shapes=(
            (2048, 994),
            (2048, 2048),
            (1024, 2048),
            (1024, 1024),
            (512, 1024),
            (512, 512),
            (29, 512),
        ),
    )

    class RegularG1Encoder(nn.Module):
        def __init__(self, mlp: Any):
            super().__init__()
            self.mlp = mlp

        def forward(self, obs_dict: Any) -> Any:
            # C++ encoder layout:
            #   [mode selector + padding (4), q/dq (580), orientation (60), ...]
            # Preserve the legacy non-flat observation reshape used in training.
            command = obs_dict[..., MODE_PREFIX_DIM : MODE_PREFIX_DIM + G1_COMMAND_DIM]
            command = command.reshape(-1, 10, 58)
            orientation = obs_dict[
                ...,
                MODE_PREFIX_DIM + G1_COMMAND_DIM : G1_INPUT_END,
            ].reshape(-1, 10, 6)
            legacy_input = torch.cat((command, orientation), dim=-1).reshape(-1, 640)

            latent = self.mlp(legacy_input).reshape(-1, 2, 32)
            half_level = (FSQ_LEVELS - 1) * (1.0 + FSQ_EPSILON) / 2.0
            shift = math.atanh(0.5 / half_level)
            quantized = torch.round(torch.tanh(latent + shift) * half_level - 0.5)
            return (quantized / (FSQ_LEVELS // 2)).reshape(-1, ENCODER_OUTPUT_DIM)

    class RegularG1DynamicDecoder(nn.Module):
        def __init__(self, mlp: Any):
            super().__init__()
            self.mlp = mlp

        def forward(self, obs_dict: Any) -> Any:
            return self.mlp(obs_dict)

    return RegularG1Encoder(encoder_mlp).eval(), RegularG1DynamicDecoder(decoder_mlp).eval()


def _onnx_shape(value_info: Any) -> list[int | str]:
    return [
        dimension.dim_value or dimension.dim_param
        for dimension in value_info.type.tensor_type.shape.dim
    ]


def _export_and_check(
    torch: Any,
    onnx: Any,
    model: Any,
    *,
    path: Path,
    input_dim: int,
    output_name: str,
    expected_output_dim: int,
    opset: int,
) -> dict[str, Any]:
    example = torch.zeros((1, input_dim), dtype=torch.float32)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            example,
            str(path),
            input_names=["obs_dict"],
            output_names=[output_name],
            opset_version=opset,
            dynamo=False,
        )

    graph = onnx.load(path)
    onnx.checker.check_model(graph)
    input_shape = _onnx_shape(graph.graph.input[0])
    output_shape = _onnx_shape(graph.graph.output[0])
    if input_shape != [1, input_dim] or output_shape != [1, expected_output_dim]:
        raise ExportError(
            f"unexpected ONNX shapes for {path}: input={input_shape}, output={output_shape}"
        )
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "input": {"name": graph.graph.input[0].name, "shape": input_shape},
        "output": {"name": graph.graph.output[0].name, "shape": output_shape},
        "nodes": len(graph.graph.node),
    }


def _recording_csv(path: str | Path) -> Path:
    result = _absolute(path)
    if result.is_dir():
        result = result / "data.csv"
    if not result.is_file():
        raise ExportError(f"validation CSV does not exist: {result}")
    return result


def _read_validation_samples(path: Path, count: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    inputs: list[np.ndarray] = []
    tokens: list[np.ndarray] = []
    sequences: list[int] = []
    previous_sequence: int | None = None

    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ExportError(f"validation CSV is empty: {path}") from exc
        lookup = {name: index for index, name in enumerate(header)}
        required = ["policy_seq"]
        required.extend(f"reference_motion[{index}]" for index in range(G1_INPUT_DIM))
        required.extend(f"token_state[{index}]" for index in range(ENCODER_OUTPUT_DIM))
        missing = [name for name in required if name not in lookup]
        if missing:
            raise ExportError(f"validation CSV is missing columns: {', '.join(missing[:5])}")
        valid_index = lookup.get("policy_valid")
        size_index = lookup.get("policy_reference_motion_size")

        for row_number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ExportError(
                    f"CSV row {row_number} has {len(row)} columns; expected {len(header)}"
                )
            if valid_index is not None and not bool(int(float(row[valid_index]))):
                continue
            sequence = int(float(row[lookup["policy_seq"]]))
            if sequence == previous_sequence:
                continue
            previous_sequence = sequence
            if size_index is not None and int(float(row[size_index])) != G1_INPUT_DIM:
                raise ExportError(
                    f"CSV row {row_number} declares policy reference size "
                    f"{row[size_index]}; expected {G1_INPUT_DIM}"
                )

            reference = np.asarray(
                [float(row[lookup[f"reference_motion[{index}]"]]) for index in range(G1_INPUT_DIM)],
                dtype=np.float32,
            )
            token = np.asarray(
                [float(row[lookup[f"token_state[{index}]"]]) for index in range(ENCODER_OUTPUT_DIM)],
                dtype=np.float32,
            )
            if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(token)):
                raise ExportError(f"CSV row {row_number} contains a non-finite reference or token")

            full_input = np.zeros(ENCODER_INPUT_DIM, dtype=np.float32)
            full_input[0] = 0.0  # fixed G1 encoder mode
            full_input[MODE_PREFIX_DIM:G1_INPUT_END] = reference
            inputs.append(full_input)
            tokens.append(token)
            sequences.append(sequence)
            if len(inputs) == count:
                break

    if len(inputs) < count:
        raise ExportError(
            f"validation CSV contains only {len(inputs)} unique valid policy frames; requested {count}"
        )
    return np.stack(inputs), np.stack(tokens), sequences


def _error_summary(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    difference = np.abs(np.asarray(actual) - np.asarray(expected))
    return {
        "max_abs": float(np.max(difference)),
        "mean_abs": float(np.mean(difference)),
        "elements_different_gt_1e-6": int(np.count_nonzero(difference > 1e-6)),
    }


def _validate_recording(
    torch: Any,
    onnx: Any,
    encoder: Any,
    encoder_path: Path,
    recording: Path,
    frames: int,
    tolerance: float,
) -> dict[str, Any]:
    inputs, expected, sequences = _read_validation_samples(recording, frames)
    with torch.inference_mode():
        pytorch_output = encoder(torch.from_numpy(inputs)).cpu().numpy()

    # onnxruntime is not required by the training environment.  ONNX's bundled
    # reference evaluator is sufficient for this small optional correctness check.
    from onnx.reference import ReferenceEvaluator

    evaluator = ReferenceEvaluator(onnx.load(encoder_path))
    onnx_output = np.concatenate(
        [evaluator.run(None, {"obs_dict": item[None]})[0] for item in inputs],
        axis=0,
    )
    pytorch_vs_csv = _error_summary(pytorch_output, expected)
    onnx_vs_csv = _error_summary(onnx_output, expected)
    onnx_vs_pytorch = _error_summary(onnx_output, pytorch_output)
    worst = max(pytorch_vs_csv["max_abs"], onnx_vs_csv["max_abs"], onnx_vs_pytorch["max_abs"])
    if worst > tolerance:
        raise ExportError(
            f"recorded-token validation failed: max_abs={worst:.9g} exceeds {tolerance:.9g}"
        )
    return {
        "recording": str(recording),
        "policy_seq": sequences,
        "frames": frames,
        "tolerance": tolerance,
        "pytorch_vs_csv": pytorch_vs_csv,
        "onnx_vs_csv": onnx_vs_csv,
        "onnx_vs_pytorch": onnx_vs_pytorch,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument(
        "--validate-recording",
        help="Optional data.csv or recording directory whose real token_state is checked",
    )
    parser.add_argument(
        "--validate-frames",
        type=int,
        default=1,
        help="Number of unique policy_seq frames checked when --validate-recording is used",
    )
    parser.add_argument("--validation-tolerance", type=float, default=1e-6)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.opset < 13:
        raise ExportError("--opset must be at least 13 because the encoder uses Round")
    if args.validate_frames <= 0:
        raise ExportError("--validate-frames must be positive")
    if args.validation_tolerance < 0:
        raise ExportError("--validation-tolerance must be non-negative")

    # Enforce CPU-only behavior before importing PyTorch/TRL.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        import onnx
        import torch
    except ImportError as exc:
        raise ExportError(
            "PyTorch and ONNX are required; run this script with the isaaclab Python environment"
        ) from exc

    checkpoint = _absolute(args.checkpoint)
    output_dir = _absolute(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder_path = output_dir / "model_encoder.onnx"
    decoder_path = output_dir / "model_decoder.onnx"

    state = _load_policy_state(torch, checkpoint)
    encoder, decoder = _build_models(torch, state)
    result: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "output_dir": str(output_dir),
        "fixed_encoder_mode": "g1",
        "limitations": "teleop and SMPL encoder modes are intentionally not exported",
        "legacy_g1_layout": "reshape(q+dq,10,58); concat orientation(10,6); flatten",
        "models": {},
    }
    result["models"]["encoder"] = _export_and_check(
        torch,
        onnx,
        encoder,
        path=encoder_path,
        input_dim=ENCODER_INPUT_DIM,
        output_name="encoded_tokens",
        expected_output_dim=ENCODER_OUTPUT_DIM,
        opset=args.opset,
    )
    result["models"]["decoder"] = _export_and_check(
        torch,
        onnx,
        decoder,
        path=decoder_path,
        input_dim=DECODER_INPUT_DIM,
        output_name="action",
        expected_output_dim=DECODER_OUTPUT_DIM,
        opset=args.opset,
    )

    if args.validate_recording:
        result["recorded_token_validation"] = _validate_recording(
            torch,
            onnx,
            encoder,
            encoder_path,
            _recording_csv(args.validate_recording),
            args.validate_frames,
            args.validation_tolerance,
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except (ExportError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
