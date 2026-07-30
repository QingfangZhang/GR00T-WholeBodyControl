"""Infer the regular encoder's ten temporal slots from a recorded CSV.

The qpos-track experiment uses robot state values from ``qpos``.  In the
optional recorded-window experiment, the original 640 active
``reference_motion`` columns are read only to recover *when* each of the ten
recorded slots points.  Their joint/orientation values are never sent on the
wire.

The inference intentionally mirrors the overlap check used by
``change_ckpt/reference_data.py``: each recorded slot is compared with slot
zero of later ``policy_seq`` frames, while a short source-clamped tail is
excluded from the score.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


NUM_SLOTS = 10
NUM_BODY_JOINTS = 29
ACTIVE_REFERENCE_SIZE = 640
MAX_INFERRED_LAG = 60


class RecordedSlotLagError(ValueError):
    """The recording cannot provide an auditable ten-slot timing layout."""


def _load_recorded_reference(path: str | Path) -> Any:
    try:
        from change_ckpt.reference_data import load_reference_sequence
    except (ImportError, ModuleNotFoundError) as exc:
        raise RecordedSlotLagError(
            "cannot import change_ckpt.reference_data to inspect "
            "reference_motion slot timing"
        ) from exc
    try:
        return load_reference_sequence(path)
    except Exception as exc:  # noqa: BLE001 - expose one adapter error type
        raise RecordedSlotLagError(
            f"cannot load recorded reference_motion slot timing: {exc}"
        ) from exc


def _infer_from_active_reference(
    active_reference: np.ndarray,
    *,
    max_lag: int = MAX_INFERRED_LAG,
) -> dict[str, Any]:
    active = np.asarray(active_reference, dtype=np.float64)
    if active.ndim != 2 or active.shape[1] < ACTIVE_REFERENCE_SIZE:
        raise RecordedSlotLagError(
            "active reference must have shape [N,>=640], got "
            f"{active.shape}"
        )
    if active.shape[0] < 2:
        raise RecordedSlotLagError(
            "at least two policy frames are required to infer slot timing"
        )
    if not np.all(np.isfinite(active[:, :ACTIVE_REFERENCE_SIZE])):
        raise RecordedSlotLagError("reference_motion contains non-finite values")
    if max_lag < 0:
        raise RecordedSlotLagError("max_lag must be non-negative")

    positions = active[:, :290].reshape(-1, NUM_SLOTS, NUM_BODY_JOINTS)
    velocities = active[:, 290:580].reshape(
        -1, NUM_SLOTS, NUM_BODY_JOINTS
    )
    count = int(active.shape[0])
    comparison_count = max(1, count - max_lag - NUM_SLOTS)
    slots: list[dict[str, Any]] = []
    for slot in range(NUM_SLOTS):
        candidates: list[tuple[float, float, int]] = []
        for lag in range(min(max_lag, count - 1) + 1):
            usable = min(comparison_count, count - lag)
            if usable <= 0:
                continue
            position_error = np.abs(
                positions[:usable, slot]
                - positions[lag : lag + usable, 0]
            )
            velocity_error = np.abs(
                velocities[:usable, slot]
                - velocities[lag : lag + usable, 0]
            )
            score = float(
                np.mean(position_error) + 0.01 * np.mean(velocity_error)
            )
            candidates.append(
                (score, float(np.max(position_error)), lag)
            )
        if not candidates:
            raise RecordedSlotLagError(
                f"no lag candidate is available for slot {slot}"
            )
        score, max_position_error, lag = min(candidates)
        slots.append(
            {
                "slot": slot,
                "inferred_policy_lag": int(lag),
                "score": score,
                "max_joint_position_error": max_position_error,
            }
        )

    lags = [int(item["inferred_policy_lag"]) for item in slots]
    if lags[0] != 0:
        raise RecordedSlotLagError(
            f"recorded slot zero must resolve to lag 0, got {lags[0]}"
        )
    if any(right < left for left, right in zip(lags, lags[1:])):
        raise RecordedSlotLagError(
            f"recorded slot lags must be non-decreasing, got {lags}"
        )
    return {
        "method": (
            "global q/dq overlap against future reference_motion slot zero; "
            "last max_lag+10 frames excluded from scoring"
        ),
        "policy_frames_inspected": count,
        "max_lag_searched": min(max_lag, count - 1),
        "comparison_frames_per_candidate_cap": comparison_count,
        "future_slot_policy_lags": lags,
        "slots": slots,
    }


@lru_cache(maxsize=16)
def _infer_recorded_slot_lags_cached(
    resolved_path: str,
    max_lag: int,
) -> dict[str, Any]:
    sequence = _load_recorded_reference(resolved_path)
    result = _infer_from_active_reference(
        np.asarray(sequence.reference_motion)[:, :ACTIVE_REFERENCE_SIZE],
        max_lag=max_lag,
    )
    result["source_csv"] = str(sequence.csv_path)
    result["first_policy_seq"] = int(sequence.policy_seq[0])
    result["last_policy_seq"] = int(sequence.policy_seq[-1])
    return result


def infer_recorded_slot_lags(
    path: str | Path,
    *,
    max_lag: int = MAX_INFERRED_LAG,
) -> dict[str, Any]:
    """Return a JSON-serialisable ten-slot lag report for one recording."""

    source = Path(path).expanduser().resolve()
    # Return a fresh shallow/deep-enough structure so callers cannot mutate the
    # cached report used by a later preflight in the same launcher process.
    report = _infer_recorded_slot_lags_cached(str(source), int(max_lag))
    return {
        **report,
        "future_slot_policy_lags": list(
            report["future_slot_policy_lags"]
        ),
        "slots": [dict(item) for item in report["slots"]],
    }


__all__ = [
    "RecordedSlotLagError",
    "infer_recorded_slot_lags",
]
