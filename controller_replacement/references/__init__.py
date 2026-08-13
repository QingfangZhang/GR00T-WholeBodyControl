"""Public reference-provider API for deterministic controller replacement."""

from __future__ import annotations

from pathlib import Path

from .base import (
    CONSECUTIVE_SLOT_OFFSETS,
    POLICY_DT_S,
    POLICY_RATE_HZ,
    REGULAR_QPOS_SLOT_OFFSETS,
    ReferenceError,
    ReferenceFrame,
    ReferenceMode,
    ReferenceProvider,
    ReferenceProvenance,
    ReferenceSequence,
)
from .executed_qpos import ExecutedQposProvider, load_executed_qpos
from .reference_motion import (
    DEFAULT_BASE_SAMPLE_MODE,
    ReferenceMotionProvider,
    load_reference_motion,
)


def make_reference_provider(
    mode: ReferenceMode | str,
    *,
    drop_truncated_edges: bool = True,
    base_sample_mode: str = DEFAULT_BASE_SAMPLE_MODE,
) -> ReferenceProvider:
    """Create a provider without making runners depend on concrete classes."""

    try:
        parsed = ReferenceMode(mode)
    except ValueError as exc:
        choices = ", ".join(item.value for item in ReferenceMode)
        raise ReferenceError(
            f"unsupported reference mode {mode!r}; choose one of {choices}"
        ) from exc
    if parsed is ReferenceMode.REFERENCE_MOTION:
        return ReferenceMotionProvider(
            base_sample_mode=base_sample_mode,
            drop_truncated_edges=drop_truncated_edges,
        )
    return ExecutedQposProvider(drop_truncated_edges=drop_truncated_edges)


def load_reference(
    recording: str | Path,
    *,
    mode: ReferenceMode | str = ReferenceMode.REFERENCE_MOTION,
    policy_offset: int = 0,
    policy_count: int | None = None,
    drop_truncated_edges: bool = True,
    base_sample_mode: str = DEFAULT_BASE_SAMPLE_MODE,
) -> ReferenceSequence:
    """Load either reference mode through one stable public function."""

    provider = make_reference_provider(
        mode,
        drop_truncated_edges=drop_truncated_edges,
        base_sample_mode=base_sample_mode,
    )
    return provider.load(
        recording,
        policy_offset=policy_offset,
        policy_count=policy_count,
    )


__all__ = [
    "CONSECUTIVE_SLOT_OFFSETS",
    "DEFAULT_BASE_SAMPLE_MODE",
    "ExecutedQposProvider",
    "POLICY_DT_S",
    "POLICY_RATE_HZ",
    "REGULAR_QPOS_SLOT_OFFSETS",
    "ReferenceError",
    "ReferenceFrame",
    "ReferenceMode",
    "ReferenceMotionProvider",
    "ReferenceProvider",
    "ReferenceProvenance",
    "ReferenceSequence",
    "load_executed_qpos",
    "load_reference",
    "load_reference_motion",
    "make_reference_provider",
]
