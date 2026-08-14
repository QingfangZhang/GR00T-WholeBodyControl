"""Online metrics for deterministic controller-replacement experiments.

The report keeps three targets separate:

* the phase-matched state that the source robot actually executed;
* the reference pose consumed by the replacement controller; and
* the joint target emitted by that controller.

It also latches a fall event without resetting or shortening the rollout.  A
fall disqualifies semantic task success, while task-object/contact diagnostics
continue over the common fixed horizon.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


METRICS_FORMAT_VERSION = 3
DEFAULT_FALL_HEIGHT_M = 0.2


def _finite(value: Any, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (size,):
        raise ValueError(f"{name} has shape {array.shape}; expected {(size,)}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def _orientation_error_rad(left_wxyz: Any, right_wxyz: Any) -> float:
    left = _finite(left_wxyz, 4, "left quaternion")
    right = _finite(right_wxyz, 4, "right quaternion")
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        raise ValueError("metric quaternion must have non-zero norm")
    left = left / left_norm
    right = right / right_norm
    return float(
        2.0
        * np.arccos(np.clip(abs(float(np.dot(left, right))), 0.0, 1.0))
    )


def rot6d_orientation_error_rad(value: Any) -> float:
    """Return the geodesic angle represented by a 6D rotation input.

    Both SONIC and Teleopit flatten the first two rotation-matrix columns as
    ``[r00,r01,r10,r11,r20,r21]``.  Orthonormalising those columns makes this
    diagnostic robust to float32 round-off while measuring the exact relative
    orientation supplied to the controller.
    """

    rot6 = _finite(value, 6, "relative orientation rot6d")
    first = np.asarray([rot6[0], rot6[2], rot6[4]], dtype=np.float64)
    second = np.asarray([rot6[1], rot6[3], rot6[5]], dtype=np.float64)
    first_norm = float(np.linalg.norm(first))
    if first_norm <= 1e-12:
        raise ValueError("relative orientation rot6d has a zero first column")
    first /= first_norm
    second -= first * float(np.dot(first, second))
    second_norm = float(np.linalg.norm(second))
    if second_norm <= 1e-12:
        raise ValueError("relative orientation rot6d columns are collinear")
    second /= second_norm
    third = np.cross(first, second)
    rotation = np.column_stack((first, second, third))
    cosine = np.clip((float(np.trace(rotation)) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(cosine))


@dataclass
class _VectorMoments:
    width: int

    def __post_init__(self) -> None:
        self.count = 0
        self.sum_sq = np.zeros(self.width, dtype=np.float64)
        self.sum_abs = np.zeros(self.width, dtype=np.float64)
        self.max_abs = np.zeros(self.width, dtype=np.float64)

    def add_error(self, error: Any) -> None:
        value = _finite(error, self.width, "metric error")
        self.count += 1
        self.sum_sq += value * value
        self.sum_abs += np.abs(value)
        self.max_abs = np.maximum(self.max_abs, np.abs(value))

    def report(self) -> dict[str, Any]:
        if not self.count:
            rmse = np.zeros(self.width, dtype=np.float64)
            mae = np.zeros(self.width, dtype=np.float64)
        else:
            rmse = np.sqrt(self.sum_sq / self.count)
            mae = self.sum_abs / self.count
        return {
            "sample_count": self.count,
            "aggregate_rmse": float(np.sqrt(np.mean(rmse * rmse))),
            "aggregate_mae": float(np.mean(mae)),
            "aggregate_max_abs": (
                float(np.max(self.max_abs)) if self.width else 0.0
            ),
            "per_dimension_rmse": rmse.tolist(),
            "per_dimension_mae": mae.tolist(),
            "per_dimension_max_abs": self.max_abs.tolist(),
        }


@dataclass
class _ScalarMoments:
    unit: str = "rad"

    def __post_init__(self) -> None:
        if not self.unit or not self.unit.replace("_", "").isalnum():
            raise ValueError("scalar metric unit must be a simple identifier")
        self.count = 0
        self.sum_sq = 0.0
        self.sum_abs = 0.0
        self.max_abs = 0.0

    def add(self, value: float) -> None:
        number = float(value)
        if not np.isfinite(number):
            raise ValueError("scalar metric contains NaN or infinity")
        magnitude = abs(number)
        self.count += 1
        self.sum_sq += number * number
        self.sum_abs += magnitude
        self.max_abs = max(self.max_abs, magnitude)

    def report(self) -> dict[str, Any]:
        return {
            "sample_count": self.count,
            f"rmse_{self.unit}": (
                float(np.sqrt(self.sum_sq / self.count)) if self.count else 0.0
            ),
            f"mae_{self.unit}": self.sum_abs / self.count if self.count else 0.0,
            f"max_{self.unit}": self.max_abs,
        }


class RolloutMetrics:
    """Accumulate policy-rate tracking and 400 Hz task diagnostics."""

    def __init__(
        self,
        *,
        body_joint_names: Sequence[str],
        task_qpos_labels: Sequence[str],
        warmup_inferences: int = 10,
        fall_height_m: float = DEFAULT_FALL_HEIGHT_M,
    ) -> None:
        if len(body_joint_names) != 29:
            raise ValueError("body_joint_names must contain 29 names")
        if warmup_inferences < 0:
            raise ValueError("warmup_inferences must be non-negative")
        if not np.isfinite(fall_height_m) or fall_height_m <= 0.0:
            raise ValueError("fall_height_m must be finite and positive")
        self.body_joint_names = tuple(body_joint_names)
        self.task_qpos_labels = tuple(task_qpos_labels)
        self.warmup_inferences = int(warmup_inferences)
        self.fall_height_m = float(fall_height_m)

        self.total_policy = 0
        self.warmup_policy = 0
        self.evaluation_policy = 0
        self.post_fall_policy = 0
        self.total_log_rows = 0

        self._joint_error = _VectorMoments(29)
        self._target_error = _VectorMoments(29)
        self._source_root_translation_error = _VectorMoments(3)
        self._source_root_orientation_error = _ScalarMoments()
        self._controller_reference_orientation_error = _ScalarMoments()
        self._controller_reference_height_error = _ScalarMoments("m")

        self.total_pd_updates = 0
        self.evaluation_pd_updates = 0
        self._saturation_count = np.zeros(29, dtype=np.int64)
        self._any_saturation_pd_updates = 0
        self._torque_abs_max = np.zeros(29, dtype=np.float64)

        self._task_start: np.ndarray | None = None
        self._task_end: np.ndarray | None = None
        self._task_min: np.ndarray | None = None
        self._task_max: np.ndarray | None = None
        self._root_z_min = np.inf
        self._root_z_max = -np.inf
        self._first_source_row: int | None = None
        self._last_source_row: int | None = None

        self._contact_frames = 0
        self._robot_world_contact_frames = 0
        self._robot_environment_contact_frames = 0
        self._robot_environment_contact_count_sum = 0
        self._robot_environment_force_sum = 0.0
        self._robot_environment_force_max = 0.0
        self._terminal_endpoint_observed = False
        self._terminal_time_s: float | None = None
        self._terminal_contact: dict[str, float | int] | None = None

        self.fallen = False
        self.initially_fallen = False
        self.first_fall_time_s: float | None = None
        self.first_fall_log_row_exclusive: int | None = None
        self.first_fall_source_row_index: int | None = None
        self._last_fall_observation_time_s: float | None = None

    def observe_fall(
        self,
        *,
        qpos: Any,
        time_s: float,
        log_row_exclusive: int,
        source_row_index: int,
    ) -> bool:
        """Latch the first root-height fall and return current fall state.

        ``log_row_exclusive`` is the number of high-rate rows strictly before
        this state.  A fall first seen at saved row ``k`` therefore makes
        ``[0, k)`` the pre-fall interval.  Passing the final row count checks
        the hidden post-step endpoint without inventing another CSV sample.
        """

        state = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if state.size < 3 or not np.all(np.isfinite(state)):
            raise ValueError("fall observation qpos is invalid")
        timestamp = float(time_s)
        row = int(log_row_exclusive)
        source_row = int(source_row_index)
        if not np.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError("fall observation time must be finite and non-negative")
        if row < 0 or source_row < 0:
            raise ValueError("fall row indices must be non-negative")
        self._last_fall_observation_time_s = (
            timestamp
            if self._last_fall_observation_time_s is None
            else max(self._last_fall_observation_time_s, timestamp)
        )
        # Fall is latched at every 2 kHz physics substep, so extrema reported
        # beside that decision must use the same observation clock.  Otherwise
        # a brief threshold crossing between 400 Hz CSV rows could yield the
        # contradictory combination ``fallen=true`` and ``root_z_min>threshold``.
        self._root_z_min = min(self._root_z_min, float(state[2]))
        self._root_z_max = max(self._root_z_max, float(state[2]))
        if not self.fallen and float(state[2]) < self.fall_height_m:
            self.fallen = True
            self.initially_fallen = row == 0
            self.first_fall_time_s = timestamp
            self.first_fall_log_row_exclusive = row
            self.first_fall_source_row_index = source_row
        return self.fallen

    @property
    def task_success_eligible(self) -> bool:
        return not self.fallen

    def record_policy(
        self,
        *,
        robot_joint_pos: Any,
        controller_reference_joint_pos: Any,
        q_target: Any,
        robot_root_pos: Any,
        phase_matched_source_root_pos: Any,
        robot_root_quat_wxyz: Any,
        phase_matched_source_root_quat_wxyz: Any,
        controller_reference_orientation_error_rad: float,
        controller_reference_height_error_m: float | None,
        warmup: bool,
    ) -> None:
        robot_q = _finite(robot_joint_pos, 29, "robot_joint_pos")
        reference_q = _finite(
            controller_reference_joint_pos,
            29,
            "controller_reference_joint_pos",
        )
        target = _finite(q_target, 29, "q_target")
        root = _finite(robot_root_pos, 3, "robot_root_pos")
        source_root = _finite(
            phase_matched_source_root_pos,
            3,
            "phase_matched_source_root_pos",
        )

        self.total_policy += 1
        if warmup:
            self.warmup_policy += 1
            return
        if self.fallen:
            self.post_fall_policy += 1
            return

        self.evaluation_policy += 1
        self._joint_error.add_error(robot_q - reference_q)
        self._target_error.add_error(robot_q - target)
        self._source_root_translation_error.add_error(root - source_root)
        self._source_root_orientation_error.add(
            _orientation_error_rad(
                robot_root_quat_wxyz,
                phase_matched_source_root_quat_wxyz,
            )
        )
        self._controller_reference_orientation_error.add(
            float(controller_reference_orientation_error_rad)
        )
        if controller_reference_height_error_m is not None:
            self._controller_reference_height_error.add(
                float(controller_reference_height_error_m)
            )

    def record_pd_control(
        self,
        *,
        body_torque: Any,
        body_torque_saturation: Any,
        evaluation: bool,
    ) -> None:
        """Record one applied 200 Hz body-PD command.

        The caller marks whether this update belongs to the post-warmup,
        pre-fall evaluation interval.  All PD updates reach this method so the
        report can distinguish schedule coverage from the filtered metric
        denominator.
        """

        torque = _finite(body_torque, 29, "body_torque")
        saturation = np.asarray(body_torque_saturation, dtype=np.bool_).reshape(-1)
        if saturation.shape != (29,):
            raise ValueError("body_torque_saturation must have shape (29,)")
        if not isinstance(evaluation, (bool, np.bool_)):
            raise ValueError("evaluation must be boolean")

        self.total_pd_updates += 1
        if not bool(evaluation):
            return
        self.evaluation_pd_updates += 1
        self._saturation_count += saturation.astype(np.int64)
        self._any_saturation_pd_updates += int(np.any(saturation))
        self._torque_abs_max = np.maximum(self._torque_abs_max, np.abs(torque))

    def record_log(
        self,
        *,
        qpos: Any,
        task_qpos: Any,
        source_row_index: int,
        contact_summary: Any | None = None,
    ) -> None:
        state = np.asarray(qpos, dtype=np.float64).reshape(-1)
        task = _finite(task_qpos, len(self.task_qpos_labels), "task_qpos")
        if state.size < 3 or not np.all(np.isfinite(state)):
            raise ValueError("logged qpos is invalid")
        source_row_index = int(source_row_index)
        if source_row_index < 0:
            raise ValueError("source_row_index must be non-negative")
        if (
            self._last_source_row is not None
            and source_row_index < self._last_source_row
        ):
            raise ValueError("source_row_index must be monotonic")
        self.total_log_rows += 1
        self._first_source_row = (
            source_row_index
            if self._first_source_row is None
            else self._first_source_row
        )
        self._last_source_row = source_row_index
        self._root_z_min = min(self._root_z_min, float(state[2]))
        self._root_z_max = max(self._root_z_max, float(state[2]))
        if self._task_start is None:
            self._task_start = task.copy()
            self._task_min = task.copy()
            self._task_max = task.copy()
        self._task_end = task.copy()
        assert self._task_min is not None and self._task_max is not None
        self._task_min = np.minimum(self._task_min, task)
        self._task_max = np.maximum(self._task_max, task)
        if contact_summary is not None:
            world_contacts = int(contact_summary.robot_world_contacts)
            environment_contacts = int(contact_summary.robot_environment_contacts)
            environment_force_sum = float(
                contact_summary.robot_environment_normal_force_sum_n
            )
            environment_force_max = float(
                contact_summary.robot_environment_normal_force_max_n
            )
            if min(world_contacts, environment_contacts) < 0 or not np.isfinite(
                [environment_force_sum, environment_force_max]
            ).all():
                raise ValueError("contact summary is invalid")
            self._contact_frames += 1
            self._robot_world_contact_frames += int(world_contacts > 0)
            self._robot_environment_contact_frames += int(
                environment_contacts > 0
            )
            self._robot_environment_contact_count_sum += environment_contacts
            self._robot_environment_force_sum += environment_force_sum
            self._robot_environment_force_max = max(
                self._robot_environment_force_max, environment_force_max
            )

    def record_terminal(
        self,
        *,
        qpos: Any,
        task_qpos: Any,
        time_s: float,
        contact_summary: Any | None = None,
    ) -> None:
        """Include the hidden final physics endpoint without inventing a CSV row."""

        if self._terminal_endpoint_observed:
            raise ValueError("terminal endpoint was already recorded")
        state = np.asarray(qpos, dtype=np.float64).reshape(-1)
        task = _finite(task_qpos, len(self.task_qpos_labels), "terminal task_qpos")
        timestamp = float(time_s)
        if state.size < 3 or not np.all(np.isfinite(state)):
            raise ValueError("terminal qpos is invalid")
        if not np.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError("terminal time must be finite and non-negative")
        self._terminal_endpoint_observed = True
        self._terminal_time_s = timestamp
        self._root_z_min = min(self._root_z_min, float(state[2]))
        self._root_z_max = max(self._root_z_max, float(state[2]))
        if self._task_start is None:
            self._task_start = task.copy()
            self._task_min = task.copy()
            self._task_max = task.copy()
        self._task_end = task.copy()
        assert self._task_min is not None and self._task_max is not None
        self._task_min = np.minimum(self._task_min, task)
        self._task_max = np.maximum(self._task_max, task)
        if contact_summary is not None:
            self._terminal_contact = dict(contact_summary.as_dict())

    def report(self) -> dict[str, Any]:
        task_start = (
            np.zeros(len(self.task_qpos_labels), dtype=np.float64)
            if self._task_start is None
            else self._task_start
        )
        task_end = task_start if self._task_end is None else self._task_end
        task_min = task_start if self._task_min is None else self._task_min
        task_max = task_start if self._task_max is None else self._task_max
        saturation_denominator = max(1, self.evaluation_pd_updates)

        source_translation = self._source_root_translation_error.report()
        source_orientation = self._source_root_orientation_error.report()
        controller_orientation = (
            self._controller_reference_orientation_error.report()
        )
        controller_height = self._controller_reference_height_error.report()
        controller_height_available = controller_height["sample_count"] > 0

        fall_time = self.first_fall_time_s
        valid_rows = (
            self.total_log_rows
            if self.first_fall_log_row_exclusive is None
            else self.first_fall_log_row_exclusive
        )
        semantic_success: bool | None = False if self.fallen else None
        semantic_rule = "fall_disqualifier" if self.fallen else None
        observed_horizon_end = max(
            value
            for value in (
                self._last_fall_observation_time_s,
                self._terminal_time_s,
                0.0,
            )
            if value is not None
        )
        post_fall_physics_continued = bool(
            self.fallen
            and fall_time is not None
            and observed_horizon_end > fall_time + 1e-12
        )

        return {
            "format_version": METRICS_FORMAT_VERSION,
            "protocol_revision": 2,
            "evaluation_protocol": {
                "source_history_prefill": True,
                "self_warmup_inferences": self.warmup_inferences,
                "warmup_saved_and_affects_physics": True,
                "tracking_metrics_begin_at_inference": (
                    self.warmup_inferences + 1
                ),
                "tracking_stops_after_latched_fall": True,
            },
            "counts": {
                "policy_inferences": self.total_policy,
                "warmup_policy_inferences": self.warmup_policy,
                "evaluation_policy_inferences": self.evaluation_policy,
                "post_fall_policy_inferences": self.post_fall_policy,
                "body_pd_updates": self.total_pd_updates,
                "evaluation_body_pd_updates": self.evaluation_pd_updates,
                "logged_400hz_rows": self.total_log_rows,
                "first_source_row_index": self._first_source_row,
                "last_source_row_index": self._last_source_row,
            },
            "tracking": {
                "sampling": "50 Hz controller-input boundary",
                "body_joint_names": list(self.body_joint_names),
                "robot_vs_reference_rad": self._joint_error.report(),
                "robot_vs_controller_target_rad": self._target_error.report(),
                "root": {
                    "robot_vs_phase_matched_source": {
                        "translation_xyz_m": source_translation,
                        "orientation_error_rad": source_orientation,
                        "target_semantics": (
                            "interpolated source executed qpos at the exact "
                            "rollout policy boundary"
                        ),
                    },
                    "robot_vs_controller_reference": {
                        "anchor_orientation_error_rad": controller_orientation,
                        "anchor_height_error_m": (
                            controller_height
                            if controller_height_available
                            else None
                        ),
                        "anchor_height_available": controller_height_available,
                        "absolute_translation_xyz_m": None,
                        "absolute_translation_available": False,
                        "target_semantics": (
                            "derived from the controller's exact relative-anchor "
                            "rot6d input; Teleopit additionally observes torso "
                            "anchor height, while neither controller observes "
                            "absolute reference root xyz"
                        ),
                    },
                },
                # One-revision compatibility aliases.  They intentionally keep
                # the v1 mixed semantics visible rather than pretending they
                # formed one coherent root-pose target.
                "root_xyz_m": source_translation,
                "root_orientation": controller_orientation,
                "deprecated_root_aliases": {
                    "remove_after_protocol_revision": 2,
                    "root_xyz_m": (
                        "alias of robot_vs_phase_matched_source.translation_xyz_m"
                    ),
                    "root_orientation": (
                        "alias of robot_vs_controller_reference."
                        "anchor_orientation_error_rad"
                    ),
                },
            },
            "control": {
                "sampling": (
                    "every 200 Hz body-PD update in the post-warmup, "
                    "pre-fall evaluation interval"
                ),
                "per_joint_torque_saturation_fraction": (
                    self._saturation_count / saturation_denominator
                ).tolist(),
                "joint_sample_torque_saturation_fraction": float(
                    np.sum(self._saturation_count)
                    / (saturation_denominator * len(self._saturation_count))
                ),
                "pd_updates_with_any_torque_saturation_fraction": (
                    self._any_saturation_pd_updates / saturation_denominator
                ),
                "per_joint_max_abs_applied_torque": (
                    self._torque_abs_max.tolist()
                ),
            },
            "stability": {
                "root_z_min_m": (
                    None if not self.total_log_rows else self._root_z_min
                ),
                "root_z_max_m": (
                    None if not self.total_log_rows else self._root_z_max
                ),
                "finite_logged_states": self.total_log_rows > 0,
                "root_z_extrema_sampling": (
                    "every 2000 Hz fall observation, including the hidden "
                    "terminal endpoint"
                ),
                "fall_definition": {
                    "signal": "qpos[2]",
                    "comparison": "<",
                    "threshold_m": self.fall_height_m,
                    "sampling": (
                        "every 2000 Hz MuJoCo physics substep, including the "
                        "hidden terminal endpoint"
                    ),
                    "latched": True,
                    "reset_on_fall": False,
                    "stop_on_fall": False,
                },
                "fallen": self.fallen,
                "initially_fallen": self.initially_fallen,
                "first_fall_time_s": fall_time,
                "first_fall_log_row_exclusive": (
                    self.first_fall_log_row_exclusive
                ),
                "first_fall_source_row_index": (
                    self.first_fall_source_row_index
                ),
                "post_fall_physics_continued": post_fall_physics_continued,
            },
            "evaluation_validity": {
                "valid_until_time_s": fall_time,
                "valid_log_rows_exclusive": valid_rows,
                "task_success_eligible": self.task_success_eligible,
            },
            "task_object_qpos": {
                "labels": list(self.task_qpos_labels),
                "start": task_start.tolist(),
                "end": task_end.tolist(),
                "minimum": task_min.tolist(),
                "maximum": task_max.tolist(),
                "maximum_excursion_from_start": np.maximum(
                    np.abs(task_min - task_start),
                    np.abs(task_max - task_start),
                ).tolist(),
                "includes_post_fall_physics": post_fall_physics_continued,
                "includes_hidden_terminal_endpoint": (
                    self._terminal_endpoint_observed
                ),
            },
            "contacts": {
                "classification": (
                    "robot-world means MuJoCo world body (typically floor); "
                    "world-attached static fixtures can also be included; "
                    "robot-environment means a non-robot, non-world body and "
                    "may include task objects, furniture, or fixtures"
                ),
                "logged_frames": self._contact_frames,
                "terminal_endpoint": self._terminal_contact,
                "terminal_endpoint_time_s": self._terminal_time_s,
                "robot_world_contact_frame_fraction": (
                    self._robot_world_contact_frames / self._contact_frames
                    if self._contact_frames
                    else 0.0
                ),
                "robot_environment_contact_frame_fraction": (
                    self._robot_environment_contact_frames / self._contact_frames
                    if self._contact_frames
                    else 0.0
                ),
                "mean_robot_environment_contact_count": (
                    self._robot_environment_contact_count_sum
                    / self._contact_frames
                    if self._contact_frames
                    else 0.0
                ),
                "mean_robot_environment_normal_force_sum_n": (
                    self._robot_environment_force_sum / self._contact_frames
                    if self._contact_frames
                    else 0.0
                ),
                "maximum_robot_environment_normal_force_n": (
                    self._robot_environment_force_max
                ),
                "includes_post_fall_physics": post_fall_physics_continued,
            },
            "semantic_task_success": {
                "value": semantic_success,
                "automatic_rule": semantic_rule,
                "manual_video_or_task_specific_evaluator_required": (
                    not self.fallen
                ),
            },
        }


__all__ = [
    "DEFAULT_FALL_HEIGHT_M",
    "METRICS_FORMAT_VERSION",
    "RolloutMetrics",
]
