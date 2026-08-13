#!/usr/bin/env python3
"""Replay a controller-replacement result with its exact source-qpos ghost.

This is a small compatibility wrapper around the established visual-only ghost
viewer.  Controller-replacement rollouts use strict eight-row 50 Hz holds and
can begin at a phase-matched sub-row time, so raw ``initial_row + sample``
reconstruction is not reliable.  The wrapper uses the per-sample mapping saved
in ``source_timeline.npz`` instead.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from change_ckpt_track import replay_mujoco_compare as _viewer  # noqa: E402


_ORIGINAL_BUILD_GHOST_TRACK = _viewer.build_ghost_track


def _build_ghost_track(**kwargs: Any) -> Any:
    mode = kwargs["mode"]
    metadata = kwargs["run_metadata"]
    if mode != "source" or "source_timeline_path" not in metadata:
        return _ORIGINAL_BUILD_GHOST_TRACK(**kwargs)

    target = kwargs["target"]
    source = kwargs["source"]
    sidecar = target.path.parent / str(metadata["source_timeline_path"])
    if not sidecar.is_file():
        raise _viewer.ReplayCompareError(
            f"source timeline sidecar is missing: {sidecar}"
        )
    try:
        with np.load(sidecar, allow_pickle=False) as archive:
            source_rows = np.asarray(archive["source_row_index"], dtype=np.int64)
            source_qpos = (
                np.asarray(archive["source_qpos"], dtype=np.float64)
                if "source_qpos" in archive
                else None
            )
    except (OSError, KeyError, ValueError) as exc:
        raise _viewer.ReplayCompareError(
            f"cannot read source timeline {sidecar}: {exc}"
        ) from exc
    if source_rows.shape != (len(target.qpos),):
        raise _viewer.ReplayCompareError(
            f"source timeline shape {source_rows.shape} does not match "
            f"rollout rows {(len(target.qpos),)}"
        )
    if np.any(source_rows < 0) or np.any(source_rows >= len(source.qpos)):
        raise _viewer.ReplayCompareError("source timeline contains out-of-range rows")
    if np.any(np.diff(source_rows) < 0):
        raise _viewer.ReplayCompareError("source timeline rows are not monotonic")
    if source.policy_seq is None:
        raise _viewer.ReplayCompareError("source data.csv has no policy_seq")
    ghost_qpos = source.qpos[source_rows].copy()
    if source_qpos is not None:
        if source_qpos.shape != ghost_qpos.shape:
            raise _viewer.ReplayCompareError(
                f"source_qpos shape {source_qpos.shape} does not match "
                f"rollout/source qpos shape {ghost_qpos.shape}"
            )
        if not np.all(np.isfinite(source_qpos)):
            raise _viewer.ReplayCompareError("source_qpos contains NaN or infinity")
        ghost_qpos = source_qpos
    return _viewer.GhostTrack(
        qpos=ghost_qpos,
        source_row_index=source_rows,
        reference_frame_index=np.full(len(source_rows), -1, dtype=np.int64),
        policy_seq=source.policy_seq[source_rows],
        mode="source",
        description=(
            "original recording qpos interpolated on the exact per-sample "
            "phase-matched 400 Hz source clock saved by controller_replacement"
        ),
    )


def main() -> int:
    _viewer.build_ghost_track = _build_ghost_track
    return _viewer.main()


if __name__ == "__main__":
    raise SystemExit(main())
