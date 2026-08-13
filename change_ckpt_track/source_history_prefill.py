#!/usr/bin/env python3
"""Build a decoder startup history at an actual qpos-track policy frame.

The qpos-track launcher exposes a stable public offset whose origin is the
second raw ``policy_seq`` group, then resolves that selection through any edge
trimming to an actual ``policy_seq``.  Source-history reconstruction itself
must use raw consecutive groups.  This module resolves the selected actual
``policy_seq`` back to that raw group and delegates numerical phase matching
to the vendored, validated core in this directory.

The resulting payload always records both coordinate systems.  The selected
current group is used for its previous action and first-live-state validation;
the current q/dq/IMU observation is still supplied by live MuJoCo DDS.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

try:
    from . import source_history_prefill_core as _proven
except ImportError:  # Direct ``python change_ckpt_track/...py`` execution.
    import source_history_prefill_core as _proven  # type: ignore[no-redef]


# Re-export the contract constants and error type used by the launcher/tests.
HISTORY_FRAMES = _proven.HISTORY_FRAMES
PREFILL_FRAMES = _proven.PREFILL_FRAMES
POLICY_DT_S = _proven.POLICY_DT_S
NUM_BODY_JOINTS = _proven.NUM_BODY_JOINTS
MUJOCO_TO_ISAACLAB = _proven.MUJOCO_TO_ISAACLAB
DEFAULT_ANGLES = _proven.DEFAULT_ANGLES
BODY_QPOS_IDS = _proven.BODY_QPOS_IDS
BODY_QVEL_IDS = _proven.BODY_QVEL_IDS
ROBOT_QPOS_IDS_IN_RECEIVED_ORDER = _proven.ROBOT_QPOS_IDS_IN_RECEIVED_ORDER
ROBOT_QVEL_IDS_IN_RECEIVED_ORDER = _proven.ROBOT_QVEL_IDS_IN_RECEIVED_ORDER
SourceHistoryError = _proven.SourceHistoryError


def _resolve_csv(path: str | Path) -> Path:
    value = Path(path).expanduser().resolve()
    if value.is_dir():
        value = value / "data.csv"
    if not value.is_file():
        raise SourceHistoryError(f"data.csv not found: {value}")
    return value


def _policy_groups(csv_path: Path) -> list[tuple[int, int]]:
    """Return ``(policy_seq, first_data_row_index)`` in source order."""

    with csv_path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise SourceHistoryError(f"empty CSV: {csv_path}") from exc
        try:
            sequence_column = header.index("policy_seq")
        except ValueError as exc:
            raise SourceHistoryError("recording is missing 'policy_seq'") from exc

        groups: list[tuple[int, int]] = []
        previous: int | None = None
        for row_index, row in enumerate(reader):
            if len(row) != len(header):
                raise SourceHistoryError(
                    f"CSV data row {row_index} has {len(row)} fields; "
                    f"expected {len(header)}"
                )
            try:
                raw = float(row[sequence_column])
            except (IndexError, ValueError) as exc:
                raise SourceHistoryError(
                    f"invalid policy_seq at CSV data row {row_index}"
                ) from exc
            if not math.isfinite(raw) or not raw.is_integer():
                raise SourceHistoryError(
                    f"non-integer policy_seq={raw!r} at CSV data row {row_index}"
                )
            sequence = int(raw)
            if sequence == previous:
                continue
            if groups and sequence != groups[-1][0] + 1:
                raise SourceHistoryError(
                    "policy_seq is not contiguous at data row "
                    f"{row_index}: {groups[-1][0]} -> {sequence}"
                )
            groups.append((sequence, row_index))
            previous = sequence
    if not groups:
        raise SourceHistoryError(f"recording has no data rows: {csv_path}")
    return groups


def resolve_raw_policy_offset(
    recording: str | Path, *, start_policy_seq: int
) -> int:
    """Resolve an actual policy sequence to the source's raw group offset."""

    if isinstance(start_policy_seq, bool) or not isinstance(start_policy_seq, int):
        raise SourceHistoryError("start_policy_seq must be an integer")
    groups = _policy_groups(_resolve_csv(recording))
    first, last = groups[0][0], groups[-1][0]
    if start_policy_seq < first or start_policy_seq > last:
        raise SourceHistoryError(
            f"policy_seq {start_policy_seq} is outside {first}..{last}"
        )
    # Contiguity was validated above, so this is both explicit and O(1).
    offset = start_policy_seq - first
    if offset >= len(groups) or groups[offset][0] != start_policy_seq:
        raise SourceHistoryError(
            f"policy_seq {start_policy_seq} is not present in {recording}"
        )
    return offset


def build_source_history_prefill(
    recording: str | Path,
    *,
    start_policy_seq: int | None = None,
    start_policy_offset: int | None = None,
) -> dict[str, Any]:
    """Return a strictly phase-matched source-history payload.

    Prefer ``start_policy_seq`` for qpos-track launches.  It is invariant to
    the qpos converter dropping a truncated first/last group.  The raw
    ``start_policy_offset`` selector remains available for diagnostics and
    backward-compatible focused tests, but exactly one selector is required.
    """

    if (start_policy_seq is None) == (start_policy_offset is None):
        raise SourceHistoryError(
            "supply exactly one of start_policy_seq or start_policy_offset"
        )
    csv_path = _resolve_csv(recording)
    groups = _policy_groups(csv_path)
    if start_policy_seq is not None:
        raw_offset = resolve_raw_policy_offset(
            csv_path, start_policy_seq=start_policy_seq
        )
        selection_mode = "actual_policy_seq"
        requested_policy_seq = start_policy_seq
    else:
        assert start_policy_offset is not None
        if isinstance(start_policy_offset, bool) or not isinstance(
            start_policy_offset, int
        ):
            raise SourceHistoryError("start_policy_offset must be an integer")
        raw_offset = start_policy_offset
        selection_mode = "raw_policy_group_offset"
        requested_policy_seq = None

    payload = _proven.build_source_history_prefill(
        csv_path, start_policy_offset=raw_offset
    )
    actual_policy_seq = int(payload["current"]["policy_seq"])
    if requested_policy_seq is not None and actual_policy_seq != requested_policy_seq:
        # This also protects against the CSV changing between selector
        # resolution and the proven builder's own hash-protected read.
        raise SourceHistoryError(
            "resolved source group changed while building history: expected "
            f"policy_seq {requested_policy_seq}, got {actual_policy_seq}"
        )
    payload["selection"] = {
        "mode": selection_mode,
        "requested_policy_seq": requested_policy_seq,
        "resolved_actual_policy_seq": actual_policy_seq,
        "resolved_raw_policy_group_offset": raw_offset,
        "raw_first_policy_seq": int(groups[0][0]),
        "raw_last_policy_seq": int(groups[-1][0]),
        "raw_policy_group_count": len(groups),
        "qpos_track_note": (
            "the launcher maps its second-raw-group-based public offset to "
            "an actual policy_seq, then resolves that sequence through the "
            "processed qpos reference before this builder is called"
        ),
    }
    # Keep the legacy key's meaning explicit for downstream manifests.
    payload["raw_start_policy_offset"] = raw_offset
    return payload


def write_source_history_prefill(payload: dict[str, Any], path: str | Path) -> Path:
    return _proven.write_source_history_prefill(payload, path)


def summary(payload: dict[str, Any]) -> dict[str, Any]:
    result = _proven.summary(payload)
    result["selection"] = payload["selection"]
    result["raw_start_policy_offset"] = payload["raw_start_policy_offset"]
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", help="recording directory or data.csv")
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--start-policy-seq", type=int)
    selector.add_argument(
        "--start-policy-offset",
        type=int,
        help="raw CSV policy-group offset (not the launcher public offset)",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_source_history_prefill(
        args.recording,
        start_policy_seq=args.start_policy_seq,
        start_policy_offset=args.start_policy_offset,
    )
    if args.output is not None:
        output = write_source_history_prefill(payload, args.output)
        print(f"wrote {output}")
    print(json.dumps(summary(payload), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
