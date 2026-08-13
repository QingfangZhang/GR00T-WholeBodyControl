"""Phase-aligned, deterministic 400 Hz sampling of a recording timeline.

The source-history matcher can place controller takeover a fraction of one CSV
row before or after its nearest recorded row.  Advancing that nearest row by
integer indices would retain the fractional phase error for the whole rollout.
This module instead samples at::

    matched_control_time_s + sample_index / 400

and interpolates from the original recording at every sample.  Only the first
state may be replaced by the exact state already computed by source-history
matching.  The next sample is again obtained from the absolute source clock,
so the override cannot introduce a permanent sub-row shift.
"""

from __future__ import annotations

import csv
from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np


SOURCE_HZ = 400
SOURCE_DT_S = 1.0 / SOURCE_HZ


class TimelineError(ValueError):
    """The recording cannot provide a valid phase-aligned source timeline."""


def _normalise_quaternion_wxyz(value: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        raise TimelineError("root quaternion is zero or non-finite")
    return quaternion / norm


def _slerp_wxyz(left: np.ndarray, right: np.ndarray, fraction: float) -> np.ndarray:
    """Shortest-arc quaternion interpolation in MuJoCo's wxyz convention."""

    q0 = _normalise_quaternion_wxyz(left)
    q1 = _normalise_quaternion_wxyz(right)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return _normalise_quaternion_wxyz(q0 + fraction * (q1 - q0))
    angle = float(np.arccos(dot))
    denominator = float(np.sin(angle))
    return (
        np.sin((1.0 - fraction) * angle) / denominator * q0
        + np.sin(fraction * angle) / denominator * q1
    )


class CsvTimeline:
    """Sample qpos/qvel from ``data.csv`` on an exact 400 Hz clock.

    Parameters
    ----------
    csv_path:
        Source recording's ``data.csv``.
    start_time_s:
        Phase-matched ``control_time_s`` at takeover.  This need not coincide
        with a source row.
    start_row_index:
        Optional row selected by the source-history matcher.  It is used as the
        first output provenance row when it is a valid nearest-row candidate;
        state sampling still uses ``start_time_s`` exactly.
    log_hz:
        Must remain 400 for the formal controller-replacement protocol.
    """

    def __init__(
        self,
        csv_path: str | Path,
        *,
        start_time_s: float,
        start_row_index: int | None = None,
        log_hz: int = SOURCE_HZ,
    ) -> None:
        if isinstance(log_hz, bool) or int(log_hz) != SOURCE_HZ:
            raise TimelineError(
                f"formal source timeline must be {SOURCE_HZ} Hz, got {log_hz!r}"
            )
        self.csv_path = Path(csv_path).expanduser().resolve()
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"data.csv not found: {self.csv_path}")
        self.log_hz = SOURCE_HZ
        self.dt_s = SOURCE_DT_S
        (
            self.header,
            self.column,
            self.qpos_columns,
            self.qvel_columns,
            self._times,
            self._qpos,
            self._qvel,
            self._policy_seq,
        ) = self._load(self.csv_path)
        self._validate_source_clock(self._times)

        self._start_time_s = float(start_time_s)
        if not np.isfinite(self._start_time_s):
            raise TimelineError("start_time_s must be finite")
        endpoint_tolerance = max(1.0e-10, self.dt_s * 1.0e-7)
        if self._start_time_s < self._times[0] - endpoint_tolerance or (
            self._start_time_s > self._times[-1] + endpoint_tolerance
        ):
            raise TimelineError(
                f"start_time_s {self._start_time_s:.9f} lies outside source range "
                f"[{self._times[0]:.9f}, {self._times[-1]:.9f}]"
            )
        self._start_time_s = float(
            np.clip(self._start_time_s, self._times[0], self._times[-1])
        )
        span = max(0.0, float(self._times[-1] - self._start_time_s))
        self.available_log_rows = int(
            np.floor((span + endpoint_tolerance) / self.dt_s)
        ) + 1
        self._position = 0
        self._initial_override: tuple[np.ndarray, np.ndarray] | None = None
        nearest = self._nearest_source_row(self._start_time_s)
        if start_row_index is not None:
            if isinstance(start_row_index, bool) or not isinstance(
                start_row_index, (int, np.integer)
            ):
                raise TimelineError("start_row_index must be an integer")
            hint = int(start_row_index)
            if not 0 <= hint < len(self._times):
                raise TimelineError(
                    f"start_row_index {hint} is outside [0, {len(self._times) - 1}]"
                )
            # Phase matching clips its offset to half a nominal source period.
            # Accept that exact provenance row, including the half-period tie.
            if abs(float(self._times[hint] - self._start_time_s)) > (
                self.dt_s / 2.0 + endpoint_tolerance
            ):
                raise TimelineError(
                    "start_row_index is not a nearest-row candidate for start_time_s"
                )
            nearest = hint
        self.start_row_index = nearest
        self.current_row_index = nearest
        self.last_row_index = len(self._times) - 1
        self.start_policy_seq = (
            int(self._policy_seq[nearest]) if self._policy_seq is not None else None
        )

    @classmethod
    def from_source_history(
        cls, csv_path: str | Path, context: Any
    ) -> "CsvTimeline":
        """Build and initialize from a ``SourceHistoryContext``-like object."""

        required = (
            "timeline_start_control_time_s",
            "timeline_start_row_index",
            "initial_qpos",
            "initial_qvel",
        )
        missing = [name for name in required if not hasattr(context, name)]
        if missing:
            raise TimelineError(
                "source-history context is missing: " + ", ".join(missing)
            )
        timeline = cls(
            csv_path,
            start_time_s=float(context.timeline_start_control_time_s),
            start_row_index=int(context.timeline_start_row_index),
        )
        timeline.override_initial_state(context.initial_qpos, context.initial_qvel)
        return timeline

    @staticmethod
    def _indexed_columns(header: Sequence[str], kind: str) -> list[int]:
        pattern = re.compile(rf"\[{kind}(\d+)\]$")
        indexed: list[tuple[int, int]] = []
        for column_index, name in enumerate(header):
            match = pattern.search(name)
            if match:
                indexed.append((int(match.group(1)), column_index))
        indexed.sort()
        actual = [item[0] for item in indexed]
        if not actual:
            raise TimelineError(f"recording has no {kind} columns")
        if actual != list(range(len(actual))):
            raise TimelineError(f"{kind} columns are not contiguous: {actual[:10]} ...")
        return [item[1] for item in indexed]

    @classmethod
    def _load(
        cls, csv_path: Path
    ) -> tuple[
        list[str],
        dict[str, int],
        list[int],
        list[int],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
    ]:
        with csv_path.open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.reader(stream)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise TimelineError(f"empty CSV: {csv_path}") from exc
            if len(set(header)) != len(header):
                raise TimelineError("CSV header contains duplicate column names")
            column = {name: index for index, name in enumerate(header)}
            if "control_time_s" not in column:
                raise TimelineError("required CSV column is missing: control_time_s")
            qpos_columns = cls._indexed_columns(header, "qpos")
            qvel_columns = cls._indexed_columns(header, "qvel")
            if len(qpos_columns) < 7:
                raise TimelineError(
                    "qpos must include root position and quaternion (at least 7 values)"
                )
            times: list[float] = []
            qpos: list[list[float]] = []
            qvel: list[list[float]] = []
            policy_seq: list[int] | None = [] if "policy_seq" in column else None
            for row_index, row in enumerate(reader):
                if len(row) != len(header):
                    raise TimelineError(
                        f"CSV row {row_index} has {len(row)} fields; expected "
                        f"{len(header)}"
                    )
                try:
                    times.append(float(row[column["control_time_s"]]))
                    qpos.append([float(row[index]) for index in qpos_columns])
                    qvel.append([float(row[index]) for index in qvel_columns])
                    if policy_seq is not None:
                        policy_seq.append(int(float(row[column["policy_seq"]])))
                except ValueError as exc:
                    raise TimelineError(
                        f"CSV row {row_index} contains a non-numeric timeline value"
                    ) from exc
        if not times:
            raise TimelineError(f"recording has no data rows: {csv_path}")
        time_array = np.asarray(times, dtype=np.float64)
        qpos_array = np.asarray(qpos, dtype=np.float64)
        qvel_array = np.asarray(qvel, dtype=np.float64)
        if not (
            np.all(np.isfinite(time_array))
            and np.all(np.isfinite(qpos_array))
            and np.all(np.isfinite(qvel_array))
        ):
            raise TimelineError("source control_time_s/qpos/qvel contains non-finite values")
        # Validate every source quaternion early instead of failing mid-rollout.
        quaternion_norms = np.linalg.norm(qpos_array[:, 3:7], axis=1)
        if np.any(~np.isfinite(quaternion_norms)) or np.any(quaternion_norms <= 1.0e-12):
            raise TimelineError("source contains a zero or non-finite root quaternion")
        policy_array = (
            np.asarray(policy_seq, dtype=np.int64) if policy_seq is not None else None
        )
        return (
            header,
            column,
            qpos_columns,
            qvel_columns,
            time_array,
            qpos_array,
            qvel_array,
            policy_array,
        )

    @staticmethod
    def _validate_source_clock(times: np.ndarray) -> None:
        if times.size == 1:
            return
        differences = np.diff(times)
        if np.any(differences <= 0.0):
            bad = int(np.flatnonzero(differences <= 0.0)[0])
            raise TimelineError(
                "control_time_s must be strictly increasing; violation between "
                f"rows {bad} and {bad + 1}"
            )
        median = float(np.median(differences))
        if abs(median - SOURCE_DT_S) > SOURCE_DT_S * 0.05:
            raise TimelineError(
                f"source median period is {median:.9f}s, expected approximately "
                f"{SOURCE_DT_S:.9f}s (400 Hz)"
            )
        # Small acquisition jitter is expected.  A gap outside 0.5--1.5 source
        # periods indicates a missing/duplicated row and is not safe to conceal
        # with interpolation in a formal comparison.
        lower = SOURCE_DT_S * 0.5
        upper = SOURCE_DT_S * 1.5
        outside = np.flatnonzero((differences < lower) | (differences > upper))
        if outside.size:
            index = int(outside[0])
            raise TimelineError(
                f"source period {differences[index]:.9f}s between rows {index} and "
                f"{index + 1} is outside the accepted 400 Hz jitter range"
            )

    def _nearest_source_row(self, sample_time_s: float) -> int:
        right = int(np.searchsorted(self._times, sample_time_s, side="left"))
        if right <= 0:
            return 0
        if right >= len(self._times):
            return len(self._times) - 1
        left = right - 1
        left_distance = abs(sample_time_s - float(self._times[left]))
        right_distance = abs(float(self._times[right]) - sample_time_s)
        # Prefer the later row at an exact tie, matching the policy boundary
        # convention used by source-history phase matching.
        return right if right_distance <= left_distance else left

    def _sample(self, sample_time_s: float) -> tuple[np.ndarray, np.ndarray]:
        right = int(np.searchsorted(self._times, sample_time_s, side="left"))
        if right <= 0:
            return self._qpos[0].copy(), self._qvel[0].copy()
        if right >= len(self._times):
            return self._qpos[-1].copy(), self._qvel[-1].copy()
        if abs(float(self._times[right] - sample_time_s)) <= 1.0e-12:
            qpos = self._qpos[right].copy()
            qpos[3:7] = _normalise_quaternion_wxyz(qpos[3:7])
            return qpos, self._qvel[right].copy()
        left = right - 1
        width = float(self._times[right] - self._times[left])
        fraction = float((sample_time_s - self._times[left]) / width)
        qpos = self._qpos[left] + fraction * (self._qpos[right] - self._qpos[left])
        qvel = self._qvel[left] + fraction * (self._qvel[right] - self._qvel[left])
        qpos = np.asarray(qpos, dtype=np.float64)
        qpos[3:7] = _slerp_wxyz(
            self._qpos[left, 3:7], self._qpos[right, 3:7], fraction
        )
        return qpos, np.asarray(qvel, dtype=np.float64)

    @property
    def current_time_s(self) -> float:
        """Exact source-clock time of the current 400 Hz sample."""

        return self._start_time_s + self._position * self.dt_s

    @property
    def exhausted(self) -> bool:
        """Whether no later exact 400 Hz source sample is available."""

        return self._position + 1 >= self.available_log_rows

    def state(self) -> tuple[np.ndarray, np.ndarray]:
        """Return copies of current phase-aligned qpos and qvel."""

        if self._position == 0 and self._initial_override is not None:
            qpos, qvel = self._initial_override
            return qpos.copy(), qvel.copy()
        return self._sample(self.current_time_s)

    def override_initial_state(
        self, qpos: Sequence[float], qvel: Sequence[float]
    ) -> None:
        """Use an exact source-history state for sample zero only."""

        if self._position != 0:
            raise TimelineError("initial state cannot be overridden after advancing")
        qpos_array = np.asarray(qpos, dtype=np.float64)
        qvel_array = np.asarray(qvel, dtype=np.float64)
        if qpos_array.shape != self._qpos[0].shape:
            raise TimelineError(
                f"initial qpos override shape {qpos_array.shape}; expected "
                f"{self._qpos[0].shape}"
            )
        if qvel_array.shape != self._qvel[0].shape:
            raise TimelineError(
                f"initial qvel override shape {qvel_array.shape}; expected "
                f"{self._qvel[0].shape}"
            )
        if not np.all(np.isfinite(qpos_array)) or not np.all(np.isfinite(qvel_array)):
            raise TimelineError("initial state override contains non-finite values")
        # Keep the exact phase-match quaternion, but reject invalid state rather
        # than silently normalising data that the matcher promised was exact.
        if float(np.linalg.norm(qpos_array[3:7])) <= 1.0e-12:
            raise TimelineError("initial state override has a zero root quaternion")
        self._initial_override = (qpos_array.copy(), qvel_array.copy())

    def advance(self) -> bool:
        """Advance one exact 1/400-second sample; return false at source end."""

        if self.exhausted:
            return False
        self._position += 1
        candidate = self._nearest_source_row(self.current_time_s)
        self.current_row_index = max(self.current_row_index, candidate)
        return True

    def close(self) -> None:
        """Compatibility no-op; the CSV is loaded and closed at construction."""


__all__ = ["CsvTimeline", "SOURCE_DT_S", "SOURCE_HZ", "TimelineError"]
