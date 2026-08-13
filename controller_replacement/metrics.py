"""Online metrics for deterministic controller-replacement experiments.

The automatic report deliberately does not invent task-specific success rules.
It measures tracking, stability, actuator saturation, and every task-object qpos
excursion.  Whether the semantic task succeeded remains a visual/manual label
unless a task-specific evaluator is supplied later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


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
    left /= np.linalg.norm(left)
    right /= np.linalg.norm(right)
    return float(2.0 * np.arccos(np.clip(abs(float(np.dot(left, right))), 0.0, 1.0)))


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
            zero = np.zeros(self.width, dtype=np.float64)
            rmse = mae = zero
        else:
            rmse = np.sqrt(self.sum_sq / self.count)
            mae = self.sum_abs / self.count
        return {
            "sample_count": self.count,
            "aggregate_rmse": float(np.sqrt(np.mean(rmse * rmse))),
            "aggregate_mae": float(np.mean(mae)),
            "aggregate_max_abs": float(np.max(self.max_abs)) if self.width else 0.0,
            "per_dimension_rmse": rmse.tolist(),
            "per_dimension_mae": mae.tolist(),
            "per_dimension_max_abs": self.max_abs.tolist(),
        }


class RolloutMetrics:
    """Accumulate policy-rate tracking and 400 Hz task-state diagnostics."""

    def __init__(
        self,
        *,
        body_joint_names: Sequence[str],
        task_qpos_labels: Sequence[str],
        warmup_inferences: int = 10,
    ) -> None:
        if len(body_joint_names) != 29:
            raise ValueError("body_joint_names must contain 29 names")
        if warmup_inferences < 0:
            raise ValueError("warmup_inferences must be non-negative")
        self.body_joint_names = tuple(body_joint_names)
        self.task_qpos_labels = tuple(task_qpos_labels)
        self.warmup_inferences = int(warmup_inferences)
        self.total_policy = 0
        self.warmup_policy = 0
        self.evaluation_policy = 0
        self.total_log_rows = 0
        self._joint_error = _VectorMoments(29)
        self._target_error = _VectorMoments(29)
        self._root_error = _VectorMoments(3)
        self._orientation_error_sq = 0.0
        self._orientation_error_abs = 0.0
        self._orientation_error_max = 0.0
        self._saturation_count = np.zeros(29, dtype=np.int64)
        self._any_saturation_policy_steps = 0
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

    def record_policy(
        self,
        *,
        robot_joint_pos: Any,
        reference_joint_pos: Any,
        q_target: Any,
        robot_root_pos: Any,
        reference_root_pos: Any,
        robot_root_quat_wxyz: Any,
        reference_root_quat_wxyz: Any,
        body_torque: Any,
        body_torque_saturation: Any,
        warmup: bool,
    ) -> None:
        robot_q = _finite(robot_joint_pos, 29, "robot_joint_pos")
        reference_q = _finite(reference_joint_pos, 29, "reference_joint_pos")
        target = _finite(q_target, 29, "q_target")
        root = _finite(robot_root_pos, 3, "robot_root_pos")
        reference_root = _finite(reference_root_pos, 3, "reference_root_pos")
        torque = _finite(body_torque, 29, "body_torque")
        saturation = np.asarray(body_torque_saturation, dtype=np.bool_).reshape(-1)
        if saturation.shape != (29,):
            raise ValueError("body_torque_saturation must have shape (29,)")

        self.total_policy += 1
        if warmup:
            self.warmup_policy += 1
            return
        self.evaluation_policy += 1
        self._joint_error.add_error(robot_q - reference_q)
        self._target_error.add_error(robot_q - target)
        self._root_error.add_error(root - reference_root)
        orientation_error = _orientation_error_rad(
            robot_root_quat_wxyz, reference_root_quat_wxyz
        )
        self._orientation_error_sq += orientation_error * orientation_error
        self._orientation_error_abs += abs(orientation_error)
        self._orientation_error_max = max(
            self._orientation_error_max, abs(orientation_error)
        )
        self._saturation_count += saturation.astype(np.int64)
        self._any_saturation_policy_steps += int(np.any(saturation))
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
        if self._last_source_row is not None and source_row_index < self._last_source_row:
            raise ValueError("source_row_index must be monotonic")
        self.total_log_rows += 1
        self._first_source_row = (
            source_row_index if self._first_source_row is None else self._first_source_row
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

    def report(self) -> dict[str, Any]:
        task_start = (
            np.zeros(len(self.task_qpos_labels), dtype=np.float64)
            if self._task_start is None
            else self._task_start
        )
        task_end = task_start if self._task_end is None else self._task_end
        task_min = task_start if self._task_min is None else self._task_min
        task_max = task_start if self._task_max is None else self._task_max
        orientation_count = self.evaluation_policy
        saturation_denominator = max(1, orientation_count)
        return {
            "evaluation_protocol": {
                "source_history_prefill": True,
                "self_warmup_inferences": self.warmup_inferences,
                "warmup_saved_and_affects_physics": True,
                "tracking_metrics_begin_at_inference": self.warmup_inferences + 1,
            },
            "counts": {
                "policy_inferences": self.total_policy,
                "warmup_policy_inferences": self.warmup_policy,
                "evaluation_policy_inferences": self.evaluation_policy,
                "logged_400hz_rows": self.total_log_rows,
                "first_source_row_index": self._first_source_row,
                "last_source_row_index": self._last_source_row,
            },
            "tracking": {
                "body_joint_names": list(self.body_joint_names),
                "robot_vs_reference_rad": self._joint_error.report(),
                "robot_vs_controller_target_rad": self._target_error.report(),
                "root_xyz_m": self._root_error.report(),
                "root_orientation": {
                    "sample_count": orientation_count,
                    "rmse_rad": float(
                        np.sqrt(self._orientation_error_sq / orientation_count)
                    ) if orientation_count else 0.0,
                    "mae_rad": self._orientation_error_abs / orientation_count
                    if orientation_count else 0.0,
                    "max_rad": self._orientation_error_max,
                },
                "root_target_semantics": {
                    "xyz": (
                        "recorded executed root translation; SONIC reference_motion "
                        "does not contain root xyz"
                    ),
                    "orientation": "controller-neutral reference anchor orientation",
                },
            },
            "control": {
                "sampling": (
                    "evaluation policy boundaries after the new command is "
                    "installed; not every 200 Hz PD update"
                ),
                "per_joint_torque_saturation_fraction": (
                    self._saturation_count / saturation_denominator
                ).tolist(),
                "joint_sample_torque_saturation_fraction": float(
                    np.sum(self._saturation_count) /
                    (saturation_denominator * len(self._saturation_count))
                ),
                "policy_steps_with_any_torque_saturation_fraction": (
                    self._any_saturation_policy_steps / saturation_denominator
                ),
                "per_joint_max_abs_applied_torque": self._torque_abs_max.tolist(),
            },
            "stability": {
                "root_z_min_m": None if not self.total_log_rows else self._root_z_min,
                "root_z_max_m": None if not self.total_log_rows else self._root_z_max,
                "finite_state_completed": self.total_log_rows > 0,
            },
            "task_object_qpos": {
                "labels": list(self.task_qpos_labels),
                "start": task_start.tolist(),
                "end": task_end.tolist(),
                "minimum": task_min.tolist(),
                "maximum": task_max.tolist(),
                "maximum_excursion_from_start": np.maximum(
                    np.abs(task_min - task_start), np.abs(task_max - task_start)
                ).tolist(),
            },
            "contacts": {
                "classification": (
                    "robot-world means MuJoCo world body (typically floor); "
                    "world-attached static fixtures can also be included; "
                    "robot-environment means a non-robot, non-world body and "
                    "may include task objects, furniture, or fixtures"
                ),
                "logged_frames": self._contact_frames,
                "robot_world_contact_frame_fraction": (
                    self._robot_world_contact_frames / self._contact_frames
                    if self._contact_frames else 0.0
                ),
                "robot_environment_contact_frame_fraction": (
                    self._robot_environment_contact_frames / self._contact_frames
                    if self._contact_frames else 0.0
                ),
                "mean_robot_environment_contact_count": (
                    self._robot_environment_contact_count_sum / self._contact_frames
                    if self._contact_frames else 0.0
                ),
                "mean_robot_environment_normal_force_sum_n": (
                    self._robot_environment_force_sum / self._contact_frames
                    if self._contact_frames else 0.0
                ),
                "maximum_robot_environment_normal_force_n": (
                    self._robot_environment_force_max
                ),
            },
            "semantic_task_success": {
                "value": None,
                "automatic_rule": None,
                "manual_video_or_task_specific_evaluator_required": True,
            },
        }


__all__ = ["RolloutMetrics"]
