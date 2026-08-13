"""Formal source-history handoff for deterministic controller replacement.

The first replacement-controller inference is made at an *actual*
``policy_seq`` selected by a :class:`~controller_replacement.references.ReferenceSequence`.
That sequence number is deliberately resolved back to the raw CSV policy-group
offset before the proven SONIC history reconstructor is called.  This avoids
silently changing the takeover frame when the reference provider removes a
truncated edge group.

Only the formal source-history path is implemented here.  Zero padding,
repeated-current observations, and state-only approximations are diagnostics,
not interchangeable experiment conditions, so this module provides no such
fallbacks.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from change_ckpt.source_history_prefill import (
    SourceHistoryError as ProvenSourceHistoryError,
    build_source_history_prefill as _build_proven_sonic_prefill,
    write_source_history_prefill as _write_proven_sonic_prefill,
)

from .controllers.teleopit import (
    TeleopitSourceHistoryPrefill,
    build_sonic_source_history_prefill,
)
from .references import (
    DEFAULT_BASE_SAMPLE_MODE,
    ReferenceMode,
    ReferenceSequence,
    load_reference,
)


SOURCE_HISTORY_PRIOR_FRAMES = 9


class SourceHistoryContextError(ValueError):
    """The selected takeover cannot be reconstructed without an assumption."""


def _resolve_csv(recording: str | Path) -> Path:
    value = Path(recording).expanduser().resolve()
    if value.is_dir():
        value = value / "data.csv"
    if not value.is_file():
        raise SourceHistoryContextError(f"data.csv not found: {value}")
    return value


@dataclass(frozen=True)
class _RawPolicyGroup:
    policy_seq: int
    first_source_row_index: int
    row_count: int


def _read_raw_policy_groups(csv_path: Path) -> tuple[_RawPolicyGroup, ...]:
    """Read raw groups without applying the reference provider's edge trim."""

    with csv_path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise SourceHistoryContextError(f"empty CSV: {csv_path}") from exc
        try:
            sequence_column = header.index("policy_seq")
        except ValueError as exc:
            raise SourceHistoryContextError(
                "recording is missing 'policy_seq'"
            ) from exc

        groups: list[_RawPolicyGroup] = []
        for source_row_index, row in enumerate(reader):
            if len(row) != len(header):
                raise SourceHistoryContextError(
                    f"CSV data row {source_row_index} has {len(row)} fields; "
                    f"expected {len(header)}"
                )
            try:
                raw = float(row[sequence_column])
            except (IndexError, ValueError) as exc:
                raise SourceHistoryContextError(
                    f"invalid policy_seq at CSV data row {source_row_index}"
                ) from exc
            if not math.isfinite(raw) or not raw.is_integer():
                raise SourceHistoryContextError(
                    f"non-integer policy_seq={raw!r} at CSV data row "
                    f"{source_row_index}"
                )
            sequence = int(raw)
            if groups and sequence == groups[-1].policy_seq:
                previous = groups[-1]
                groups[-1] = _RawPolicyGroup(
                    policy_seq=previous.policy_seq,
                    first_source_row_index=previous.first_source_row_index,
                    row_count=previous.row_count + 1,
                )
                continue
            if groups and sequence != groups[-1].policy_seq + 1:
                raise SourceHistoryContextError(
                    "policy_seq is not contiguous at CSV data row "
                    f"{source_row_index}: {groups[-1].policy_seq} -> {sequence}"
                )
            groups.append(
                _RawPolicyGroup(
                    policy_seq=sequence,
                    first_source_row_index=source_row_index,
                    row_count=1,
                )
            )
    if not groups:
        raise SourceHistoryContextError(f"recording has no data rows: {csv_path}")
    return tuple(groups)


def resolve_raw_policy_group_offset(
    recording: str | Path, *, selected_policy_seq: int
) -> int:
    """Map one exact policy sequence to its zero-based raw CSV group offset."""

    if isinstance(selected_policy_seq, bool) or not isinstance(
        selected_policy_seq, (int, np.integer)
    ):
        raise SourceHistoryContextError("selected_policy_seq must be an integer")
    selected = int(selected_policy_seq)
    groups = _read_raw_policy_groups(_resolve_csv(recording))
    first = groups[0].policy_seq
    candidate = selected - first
    if (
        candidate < 0
        or candidate >= len(groups)
        or groups[candidate].policy_seq != selected
    ):
        raise SourceHistoryContextError(
            f"policy_seq {selected} is not present in raw range "
            f"{groups[0].policy_seq}..{groups[-1].policy_seq}"
        )
    return candidate


def _readonly_vector(value: Any, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1).copy()
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise SourceHistoryContextError(f"{name} is empty or contains non-finite values")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class SourceHistoryContext:
    """Validated startup state and native controller-prefill artifacts."""

    source_csv_path: Path
    selected_policy_seq: int
    selected_reference_source_row_index: int
    raw_policy_group_offset: int
    raw_first_policy_seq: int
    raw_last_policy_seq: int
    raw_policy_group_count: int
    timeline_start_row_index: int
    timeline_start_control_time_s: float
    initial_qpos: np.ndarray
    initial_qvel: np.ndarray
    sonic_prefill_payload: Mapping[str, Any]
    reference_mode: ReferenceMode
    reference_history_first_policy_seq: int | None = None
    reference_history_last_policy_seq: int | None = None
    teleopit_prefill: TeleopitSourceHistoryPrefill | None = None

    def __post_init__(self) -> None:
        source = Path(self.source_csv_path).expanduser().resolve()
        if not source.is_file():
            raise SourceHistoryContextError(f"source CSV does not exist: {source}")
        object.__setattr__(self, "source_csv_path", source)
        object.__setattr__(
            self,
            "reference_mode",
            ReferenceMode(self.reference_mode),
        )
        for name in (
            "selected_policy_seq",
            "selected_reference_source_row_index",
            "raw_policy_group_offset",
            "raw_first_policy_seq",
            "raw_last_policy_seq",
            "raw_policy_group_count",
            "timeline_start_row_index",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise SourceHistoryContextError(f"{name} must be an integer")
            object.__setattr__(self, name, int(value))
        if self.raw_policy_group_offset < SOURCE_HISTORY_PRIOR_FRAMES:
            raise SourceHistoryContextError(
                "formal takeover requires nine complete preceding policy groups"
            )
        if self.raw_policy_group_count <= self.raw_policy_group_offset:
            raise SourceHistoryContextError("raw policy-group metadata is inconsistent")
        if self.timeline_start_row_index < 0:
            raise SourceHistoryContextError("timeline_start_row_index is negative")
        if not math.isfinite(self.timeline_start_control_time_s):
            raise SourceHistoryContextError(
                "timeline_start_control_time_s must be finite"
            )
        object.__setattr__(
            self,
            "initial_qpos",
            _readonly_vector(self.initial_qpos, name="initial_qpos"),
        )
        object.__setattr__(
            self,
            "initial_qvel",
            _readonly_vector(self.initial_qvel, name="initial_qvel"),
        )
        if not isinstance(self.sonic_prefill_payload, Mapping):
            raise SourceHistoryContextError("sonic_prefill_payload must be a mapping")
        payload = self.sonic_prefill_payload
        if payload.get("format") != "g1_decoder_source_history_prefill" or int(
            payload.get("version", -1)
        ) != 1:
            raise SourceHistoryContextError("unsupported SONIC source-history payload")
        current = payload.get("current")
        if not isinstance(current, Mapping):
            raise SourceHistoryContextError("source-history payload has no current state")
        if int(current.get("policy_seq", -1)) != self.selected_policy_seq:
            raise SourceHistoryContextError(
                "source-history current policy_seq disagrees with the selection"
            )
        if int(current.get("matched_source_row_index", -1)) != (
            self.timeline_start_row_index
        ):
            raise SourceHistoryContextError(
                "source-history matched row disagrees with the timeline start"
            )
        has_span = self.reference_history_first_policy_seq is not None
        if has_span != (self.reference_history_last_policy_seq is not None):
            raise SourceHistoryContextError(
                "reference-history span must provide both endpoints"
            )
        if self.teleopit_prefill is not None and not has_span:
            raise SourceHistoryContextError(
                "Teleopit prefill must retain its reference-history provenance"
            )

    def metadata(self) -> dict[str, Any]:
        """Return JSON-safe provenance without duplicating large state vectors."""

        payload = self.sonic_prefill_payload
        validation = payload.get("validation", {})
        return {
            "mode": "source_history_prefill",
            "formal_experiment_condition": True,
            "selection_semantics": "exact selected ReferenceSequence first policy_seq",
            "source_csv": str(self.source_csv_path),
            "source_csv_sha256": payload.get("source_csv_sha256"),
            "reference_mode": self.reference_mode.value,
            "selected_policy_seq": self.selected_policy_seq,
            "selected_reference_source_row_index": (
                self.selected_reference_source_row_index
            ),
            "resolved_raw_policy_group_offset": self.raw_policy_group_offset,
            "raw_first_policy_seq": self.raw_first_policy_seq,
            "raw_last_policy_seq": self.raw_last_policy_seq,
            "raw_policy_group_count": self.raw_policy_group_count,
            "history_policy_seq": [
                int(entry["policy_seq"]) for entry in payload["entries"]
            ],
            "history_entry_count": int(payload["history_entry_count"]),
            "timeline_start_row_index": self.timeline_start_row_index,
            "timeline_start_control_time_s": self.timeline_start_control_time_s,
            "initial_qpos_size": int(self.initial_qpos.size),
            "initial_qvel_size": int(self.initial_qvel.size),
            "phase_match_validation": validation,
            "teleopit_prefill_built": self.teleopit_prefill is not None,
            "teleopit_reference_history_policy_seq": (
                [
                    self.reference_history_first_policy_seq,
                    self.reference_history_last_policy_seq,
                ]
                if self.reference_history_first_policy_seq is not None
                else None
            ),
            "teleopit_prefill": (
                self.teleopit_prefill.metadata()
                if self.teleopit_prefill is not None
                else None
            ),
            "excluded_initialization_modes": [
                "zero_padding",
                "repeat_current",
                "state_only_prefill",
            ],
        }


def _reference_source_csv(reference: ReferenceSequence) -> Path:
    try:
        value = Path(reference.source_csv_path).expanduser().resolve()
    except (AttributeError, TypeError) as exc:
        raise SourceHistoryContextError(
            "selected_reference has no valid source_csv_path"
        ) from exc
    return value


def _first_reference_scalar(reference: ReferenceSequence, name: str) -> int:
    try:
        values = np.asarray(getattr(reference, name)).reshape(-1)
    except (AttributeError, TypeError) as exc:
        raise SourceHistoryContextError(
            f"selected_reference has no valid {name}"
        ) from exc
    if values.size == 0:
        raise SourceHistoryContextError("selected_reference is empty")
    raw = values[0]
    if not np.isfinite(raw) or float(raw) != int(raw):
        raise SourceHistoryContextError(
            f"selected_reference first {name} is not an integer"
        )
    return int(raw)


def build_source_history_context(
    recording: str | Path,
    *,
    selected_reference: ReferenceSequence,
    teleopit_observation_builder: Any | None = None,
    strict_teleopit_action_conversion: bool = True,
) -> SourceHistoryContext:
    """Build the only supported formal startup context.

    The selected reference's first ``policy_seq`` is authoritative.  A raw
    offset is an implementation detail resolved from that sequence, never a
    second user-facing selection coordinate.
    """

    csv_path = _resolve_csv(recording)
    reference_csv = _reference_source_csv(selected_reference)
    if csv_path != reference_csv:
        raise SourceHistoryContextError(
            "recording and selected_reference resolve to different CSV files: "
            f"{csv_path} != {reference_csv}"
        )
    selected_policy_seq = _first_reference_scalar(
        selected_reference, "policy_seq"
    )
    selected_source_row = _first_reference_scalar(
        selected_reference, "source_row_index"
    )
    groups = _read_raw_policy_groups(csv_path)
    raw_offset = resolve_raw_policy_group_offset(
        csv_path, selected_policy_seq=selected_policy_seq
    )
    if raw_offset < SOURCE_HISTORY_PRIOR_FRAMES:
        raise SourceHistoryContextError(
            f"selected policy_seq {selected_policy_seq} is raw group {raw_offset}; "
            f"at least {SOURCE_HISTORY_PRIOR_FRAMES} preceding groups are required"
        )
    raw_group = groups[raw_offset]
    if raw_group.first_source_row_index != selected_source_row:
        raise SourceHistoryContextError(
            "selected ReferenceSequence source row does not identify its raw "
            f"policy-group boundary: {selected_source_row} != "
            f"{raw_group.first_source_row_index}"
        )

    try:
        payload = _build_proven_sonic_prefill(
            csv_path, start_policy_offset=raw_offset
        )
    except ProvenSourceHistoryError as exc:
        raise SourceHistoryContextError(str(exc)) from exc
    current = payload["current"]
    if int(current["policy_seq"]) != selected_policy_seq:
        raise SourceHistoryContextError(
            "source CSV changed while resolving the takeover policy sequence"
        )
    if int(current["source_row_index"]) != selected_source_row:
        raise SourceHistoryContextError(
            "source CSV changed while resolving the takeover source row"
        )

    # Record both coordinate systems in the payload forwarded to SONIC and in
    # the separately written context metadata.
    payload["controller_replacement_selection"] = {
        "selection_mode": "exact_reference_policy_seq",
        "selected_policy_seq": selected_policy_seq,
        "selected_reference_source_row_index": selected_source_row,
        "resolved_raw_policy_group_offset": raw_offset,
        "raw_first_policy_seq": groups[0].policy_seq,
        "raw_last_policy_seq": groups[-1].policy_seq,
        "raw_policy_group_count": len(groups),
    }

    try:
        reference_mode = ReferenceMode(selected_reference.mode)
        provenance = selected_reference.provenance
        drop_edges = bool(provenance.drop_truncated_edges)
        base_mode = (
            provenance.orientation_base_sample_mode or DEFAULT_BASE_SAMPLE_MODE
        )
    except (AttributeError, ValueError) as exc:
        raise SourceHistoryContextError(
            "selected_reference has incomplete mode/provenance metadata"
        ) from exc

    teleopit_prefill: TeleopitSourceHistoryPrefill | None = None
    history_first: int | None = None
    history_last: int | None = None
    if teleopit_observation_builder is not None:
        # Loading from processed offset zero retains the predecessor of the
        # oldest (t-9) payload frame under the formal takeover convention.  We
        # validate that explicitly before asking the adapter to compute t-10 ->
        # t-9 reference velocity.
        broad_reference = load_reference(
            csv_path,
            mode=reference_mode,
            policy_offset=0,
            drop_truncated_edges=drop_edges,
            base_sample_mode=base_mode,
        )
        history_sequences = [
            int(entry["policy_seq"]) for entry in payload["entries"]
        ] + [selected_policy_seq]
        required_first = history_sequences[0] - 1
        available = np.asarray(broad_reference.policy_seq, dtype=np.int64)
        start_index = int(np.searchsorted(available, required_first))
        if (
            start_index >= available.size
            or int(available[start_index]) != required_first
        ):
            raise SourceHistoryContextError(
                "the edge-trimmed reference does not contain the policy "
                f"immediately before the oldest source-history frame "
                f"(required policy_seq {required_first})"
            )
        current_index = int(np.searchsorted(available, selected_policy_seq))
        if (
            current_index >= available.size
            or int(available[current_index]) != selected_policy_seq
        ):
            raise SourceHistoryContextError(
                "the broad reference does not contain the selected takeover frame"
            )
        expected = np.arange(
            required_first, selected_policy_seq + 1, dtype=np.int64
        )
        actual = available[start_index : current_index + 1]
        if not np.array_equal(actual, expected):
            raise SourceHistoryContextError(
                "the Teleopit reference-history policy span is not consecutive"
            )
        try:
            teleopit_prefill = build_sonic_source_history_prefill(
                payload=payload,
                reference_sequence=broad_reference,
                observation_builder=teleopit_observation_builder,
                strict_action_conversion=strict_teleopit_action_conversion,
            )
        except (ValueError, TypeError) as exc:
            raise SourceHistoryContextError(
                f"could not construct Teleopit source history: {exc}"
            ) from exc
        history_first = required_first
        history_last = selected_policy_seq

    return SourceHistoryContext(
        source_csv_path=csv_path,
        selected_policy_seq=selected_policy_seq,
        selected_reference_source_row_index=selected_source_row,
        raw_policy_group_offset=raw_offset,
        raw_first_policy_seq=groups[0].policy_seq,
        raw_last_policy_seq=groups[-1].policy_seq,
        raw_policy_group_count=len(groups),
        timeline_start_row_index=int(current["matched_source_row_index"]),
        timeline_start_control_time_s=float(current["matched_control_time_s"]),
        initial_qpos=current["sim_initial_qpos"],
        initial_qvel=current["sim_initial_qvel"],
        sonic_prefill_payload=payload,
        reference_mode=reference_mode,
        reference_history_first_policy_seq=history_first,
        reference_history_last_policy_seq=history_last,
        teleopit_prefill=teleopit_prefill,
    )


def _write_json_atomic(value: Mapping[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def write_source_history_artifacts(
    context: SourceHistoryContext, directory: str | Path
) -> dict[str, Path]:
    """Write the native payload and compact provenance as atomic JSON files."""

    if not isinstance(context, SourceHistoryContext):
        raise TypeError("context must be SourceHistoryContext")
    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    payload_path = _write_proven_sonic_prefill(
        dict(context.sonic_prefill_payload), root / "source_history_prefill.json"
    )
    metadata_path = _write_json_atomic(
        context.metadata(), root / "source_history_context.json"
    )
    return {
        "source_history_prefill": payload_path,
        "source_history_context": metadata_path,
    }


__all__ = [
    "SOURCE_HISTORY_PRIOR_FRAMES",
    "SourceHistoryContext",
    "SourceHistoryContextError",
    "build_source_history_context",
    "resolve_raw_policy_group_offset",
    "write_source_history_artifacts",
]
