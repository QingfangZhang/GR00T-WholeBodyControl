"""Render the live SONIC reference target as a ghost in the physics viewer.

This module is intentionally independent of the official simulator sources. It
subscribes to the C++ deployment program's existing ``g1_debug`` ZMQ output,
evaluates the target pose in a second :class:`mujoco.MjData`, and copies only
the robot's visual geoms into the passive viewer's ``user_scn``. The ghost is
therefore visual-only: it has no contacts, actuators, mass, or effect on the
physical simulation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import time
from typing import Any, Literal, Protocol

import mujoco
import numpy as np

from gear_sonic.utils.data_collection.zmq_state_subscriber import (
    DEFAULT_STATE_ZMQ_PORT,
    STATE_ZMQ_TOPIC,
    ZMQStateSubscriber,
)


RootMode = Literal["actual", "reference"]
NUM_BODY_JOINTS = 29
MAX_QUATERNION_NORM_DEVIATION = 5.0e-2
MAX_ABS_ROOT_POSITION_M = 1.0e4
MAX_ABS_JOINT_POSITION_RAD = 100.0
START_FRAME_JOINT_RMS_TOLERANCE_RAD = 1.0e-3
START_FRAME_ROOT_Z_TOLERANCE_M = 1.0e-3
START_FRAME_CONFIRMATION_MESSAGES = 3
GHOST_RGB = np.asarray([0.10, 0.95, 0.25], dtype=np.float32)

# This is the order produced by C++ ``body_q_target`` after its
# IsaacLab-to-MuJoCo remapping. Validate names as well as width so a different
# 29-DoF model cannot silently render a plausible but incorrect ghost.
G1_MUJOCO_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


class ReferenceGhostError(RuntimeError):
    """Raised when target data or the MuJoCo model violates the overlay contract."""


class StateSubscriber(Protocol):
    """Small injectable interface used by the live overlay and unit tests."""

    def get_msg(self, clear: bool = True) -> Mapping[str, Any] | None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class GhostOverlayConfig:
    """Runtime settings for the reference ghost."""

    root_mode: RootMode = "reference"
    alpha: float = 0.30
    host: str = "localhost"
    port: int = DEFAULT_STATE_ZMQ_PORT
    topic: str = STATE_ZMQ_TOPIC
    stale_timeout_s: float = 1.0
    visual_geom_group: int = 1

    def __post_init__(self) -> None:
        if self.root_mode not in {"actual", "reference"}:
            raise ReferenceGhostError(
                f"root_mode must be 'actual' or 'reference', got {self.root_mode!r}"
            )
        if not np.isfinite(self.alpha) or not 0.0 < self.alpha <= 1.0:
            raise ReferenceGhostError("ghost alpha must be finite and in (0, 1]")
        if not self.host.strip():
            raise ReferenceGhostError("ghost ZMQ host must not be empty")
        if not 1 <= self.port <= 65535:
            raise ReferenceGhostError("ghost ZMQ port must be in [1, 65535]")
        if not self.topic:
            raise ReferenceGhostError("ghost ZMQ topic must not be empty")
        if not np.isfinite(self.stale_timeout_s) or self.stale_timeout_s <= 0.0:
            raise ReferenceGhostError("ghost stale timeout must be positive and finite")
        if self.visual_geom_group < 0:
            raise ReferenceGhostError("ghost visual geom group must be non-negative")


def _read_vector(
    message: Mapping[str, Any],
    key: str,
    width: int,
    *,
    maximum_absolute_value: float | None = None,
) -> np.ndarray:
    if key not in message:
        raise ReferenceGhostError(f"g1_debug message is missing {key!r}")
    try:
        value = np.asarray(message[key], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ReferenceGhostError(f"{key} is not real numeric data") from exc
    if value.shape != (width,):
        raise ReferenceGhostError(f"{key} must have shape ({width},), got {value.shape}")
    if not np.isfinite(value).all():
        raise ReferenceGhostError(f"{key} contains NaN or infinity")
    if (
        maximum_absolute_value is not None
        and float(np.max(np.abs(value))) > maximum_absolute_value
    ):
        raise ReferenceGhostError(
            f"{key} exceeds the visualization safety limit "
            f"{maximum_absolute_value:g}"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        scene_value = value.astype(np.float32)
    if not np.isfinite(scene_value).all():
        raise ReferenceGhostError(f"{key} cannot be represented by MuJoCo scene floats")
    result = np.ascontiguousarray(value)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class TargetPose:
    """Validated target pose published by the SONIC C++ controller."""

    root_position: np.ndarray
    root_quaternion_wxyz: np.ndarray
    body_joint_positions: np.ndarray
    source_index: int | None = None

    @classmethod
    def from_debug_message(cls, message: Mapping[str, Any]) -> "TargetPose":
        if not isinstance(message, Mapping):
            raise ReferenceGhostError(
                f"g1_debug payload must be a mapping, got {type(message).__name__}"
            )
        root_position = _read_vector(
            message,
            "base_trans_target",
            3,
            maximum_absolute_value=MAX_ABS_ROOT_POSITION_M,
        )
        quaternion = _read_vector(message, "base_quat_target", 4).copy()
        norm = float(np.linalg.norm(quaternion))
        if abs(norm - 1.0) > MAX_QUATERNION_NORM_DEVIATION:
            raise ReferenceGhostError(
                "base_quat_target is not a unit quaternion: "
                f"norm={norm:.9g}, allowed deviation={MAX_QUATERNION_NORM_DEVIATION:g}"
            )
        quaternion /= norm
        quaternion.setflags(write=False)
        body_joint_positions = _read_vector(
            message,
            "body_q_target",
            NUM_BODY_JOINTS,
            maximum_absolute_value=MAX_ABS_JOINT_POSITION_RAD,
        )
        source_index: int | None = None
        raw_index = message.get("index")
        if (
            isinstance(raw_index, (int, np.integer))
            and not isinstance(raw_index, (bool, np.bool_))
            and int(raw_index) >= 0
        ):
            source_index = int(raw_index)
        return cls(root_position, quaternion, body_joint_positions, source_index)


class ReferenceRootAlignment:
    """Anchor a reference trajectory to the physical root at its start frame.

    The C++ ``g1_debug`` protocol does not expose ``current_frame`` or ``play``.
    For replay of one preloaded motion, returning to its start is therefore
    detected from three consecutive new target messages matching the first
    frame's joint signature and root height. This is deterministic for the
    converted ``dance`` and ``dun`` motions, whose first-frame signatures are
    unique, while avoiding a re-anchor when a motion crosses the pose once.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._reference_origin: np.ndarray | None = None
        self._actual_origin: np.ndarray | None = None
        self._start_joint_signature: np.ndarray | None = None
        self._start_root_z: float | None = None
        self._at_start_frame = False
        self._return_match_count = 0
        self._last_source_index: int | None = None

    @property
    def anchored(self) -> bool:
        return self._reference_origin is not None and self._actual_origin is not None

    @staticmethod
    def _copy_actual_root(actual_root_position: np.ndarray) -> np.ndarray:
        value = np.asarray(actual_root_position, dtype=np.float64)
        if value.shape != (3,):
            raise ReferenceGhostError(
                f"actual MuJoCo root position must have shape (3,), got {value.shape}"
            )
        if not np.isfinite(value).all():
            raise ReferenceGhostError("actual MuJoCo root position contains NaN or infinity")
        result = np.array(value, copy=True)
        result.setflags(write=False)
        return result

    def _capture(self, target: TargetPose, actual_root_position: np.ndarray) -> None:
        reference_origin = np.array(target.root_position, copy=True)
        reference_origin.setflags(write=False)
        self._reference_origin = reference_origin
        self._actual_origin = self._copy_actual_root(actual_root_position)

    def _set_start_frame(
        self, target: TargetPose, actual_root_position: np.ndarray
    ) -> None:
        signature = np.array(target.body_joint_positions, copy=True)
        signature.setflags(write=False)
        self._start_joint_signature = signature
        self._start_root_z = float(target.root_position[2])
        self._at_start_frame = True
        self._return_match_count = 0
        self._capture(target, actual_root_position)

    def _matches_start_frame(self, target: TargetPose) -> bool:
        if self._start_joint_signature is None or self._start_root_z is None:
            return False
        difference = target.body_joint_positions - self._start_joint_signature
        joint_rms = float(np.sqrt(np.mean(np.square(difference))))
        return (
            joint_rms <= START_FRAME_JOINT_RMS_TOLERANCE_RAD
            and abs(float(target.root_position[2]) - self._start_root_z)
            <= START_FRAME_ROOT_Z_TOLERANCE_M
        )

    def observe(
        self, target: TargetPose, actual_root_position: np.ndarray
    ) -> Literal["anchored", "reanchored", "controller-restarted"] | None:
        """Observe one new controller message and update the root anchor state."""

        source_index = target.source_index
        if (
            source_index is not None
            and self._last_source_index is not None
            and source_index < self._last_source_index
        ):
            self.reset()
            self._last_source_index = source_index
            self._set_start_frame(target, actual_root_position)
            return "controller-restarted"
        if source_index is not None:
            if source_index == self._last_source_index:
                return None
            self._last_source_index = source_index

        if not self.anchored:
            self._set_start_frame(target, actual_root_position)
            return "anchored"

        matches_start = self._matches_start_frame(target)
        if self._at_start_frame:
            if matches_start:
                # While playback is paused at frame 0, keep both roots exactly
                # together. The final pre-play message becomes the fixed anchor.
                self._capture(target, actual_root_position)
            else:
                self._at_start_frame = False
                self._return_match_count = 0
            return None

        if matches_start:
            self._return_match_count += 1
            if self._return_match_count >= START_FRAME_CONFIRMATION_MESSAGES:
                self._at_start_frame = True
                self._return_match_count = 0
                self._capture(target, actual_root_position)
                return "reanchored"
        else:
            self._return_match_count = 0
        return None

    def aligned_position(self, target: TargetPose) -> np.ndarray:
        if self._reference_origin is None or self._actual_origin is None:
            raise ReferenceGhostError("reference root trajectory has not been anchored")
        result = self._actual_origin + (target.root_position - self._reference_origin)
        if not np.isfinite(result).all():
            raise ReferenceGhostError("aligned reference root position is not finite")
        if float(np.max(np.abs(result))) > MAX_ABS_ROOT_POSITION_M:
            raise ReferenceGhostError(
                "aligned reference root position exceeds the visualization safety limit"
            )
        return result


@dataclass(frozen=True)
class RobotVisual:
    """Indexes needed to pose and draw the G1 visual body tree."""

    pelvis_body_id: int
    robot_body_ids: frozenset[int]
    hand_body_ids: frozenset[int]
    geom_ids: tuple[int, ...]
    root_qpos_indices: np.ndarray
    body_joint_qpos_indices: np.ndarray
    body_joint_names: tuple[str, ...]


def _body_descendants(model: mujoco.MjModel, roots: set[int]) -> set[int]:
    descendants = set(roots)
    changed = True
    while changed:
        changed = False
        for body_id in range(1, model.nbody):
            if (
                body_id not in descendants
                and int(model.body_parentid[body_id]) in descendants
            ):
                descendants.add(body_id)
                changed = True
    return descendants


def _joint_qpos_width(joint_type: int) -> int:
    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
        return 7
    if joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
        return 4
    return 1


def inspect_robot_visual(model: mujoco.MjModel, geom_group: int = 1) -> RobotVisual:
    """Resolve the G1 body, joint-qpos, and visual-geom indexes from a model."""

    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis_id < 0:
        raise ReferenceGhostError("MuJoCo model does not contain a 'pelvis' body")

    robot_bodies = _body_descendants(model, {int(pelvis_id)})
    hand_roots: set[int] = set()
    for body_id in robot_bodies:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if name and "hand" in name.lower():
            hand_roots.add(body_id)
    hand_bodies = _body_descendants(model, hand_roots) if hand_roots else set()
    hand_bodies.intersection_update(robot_bodies)

    all_robot_geoms = tuple(
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in robot_bodies
    )
    visual_geoms = tuple(
        geom_id
        for geom_id in all_robot_geoms
        if int(model.geom_group[geom_id]) == geom_group
    )
    if not visual_geoms:
        visual_geoms = all_robot_geoms
        print(
            f"[reference-ghost] warning: no robot geoms in group {geom_group}; "
            "using every robot geom"
        )
    if not visual_geoms:
        raise ReferenceGhostError("G1 pelvis body tree contains no geoms")

    root_qpos: list[int] = []
    body_joint_qpos: list[int] = []
    body_joint_names: list[str] = []
    for joint_id in range(model.njnt):
        body_id = int(model.jnt_bodyid[joint_id])
        if body_id not in robot_bodies:
            continue
        address = int(model.jnt_qposadr[joint_id])
        joint_type = int(model.jnt_type[joint_id])
        width = _joint_qpos_width(joint_type)
        indices = list(range(address, address + width))
        if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
            root_qpos.extend(indices)
        elif body_id not in hand_bodies:
            body_joint_qpos.extend(indices)
            joint_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
            )
            body_joint_names.append(joint_name or "<unnamed>")

    if len(root_qpos) != 7:
        raise ReferenceGhostError(
            f"expected one 7-value G1 floating root, found {len(root_qpos)} values"
        )
    if len(body_joint_qpos) != NUM_BODY_JOINTS:
        raise ReferenceGhostError(
            f"expected {NUM_BODY_JOINTS} non-hand G1 joint qpos values, "
            f"found {len(body_joint_qpos)}"
        )
    if len(body_joint_names) != NUM_BODY_JOINTS:
        raise ReferenceGhostError(
            f"expected {NUM_BODY_JOINTS} non-hand G1 scalar joints, "
            f"found {len(body_joint_names)} named joints"
        )
    if tuple(body_joint_names) != G1_MUJOCO_JOINT_NAMES:
        mismatch = next(
            index
            for index, (actual, expected) in enumerate(
                zip(body_joint_names, G1_MUJOCO_JOINT_NAMES)
            )
            if actual != expected
        )
        raise ReferenceGhostError(
            "G1 joint order does not match body_q_target: "
            f"index {mismatch} is {body_joint_names[mismatch]!r}, expected "
            f"{G1_MUJOCO_JOINT_NAMES[mismatch]!r}"
        )

    root_indexes = np.asarray(root_qpos, dtype=np.int64)
    body_indexes = np.asarray(body_joint_qpos, dtype=np.int64)
    root_indexes.setflags(write=False)
    body_indexes.setflags(write=False)
    return RobotVisual(
        pelvis_body_id=int(pelvis_id),
        robot_body_ids=frozenset(robot_bodies),
        hand_body_ids=frozenset(hand_bodies),
        geom_ids=visual_geoms,
        root_qpos_indices=root_indexes,
        body_joint_qpos_indices=body_indexes,
        body_joint_names=tuple(body_joint_names),
    )


def apply_target_pose(
    *,
    model: mujoco.MjModel,
    actual_data: mujoco.MjData,
    ghost_data: mujoco.MjData,
    visual: RobotVisual,
    target: TargetPose,
    root_mode: RootMode,
    aligned_reference_root_position: np.ndarray | None = None,
) -> None:
    """Pose ``ghost_data`` without stepping physics or changing ``actual_data``."""

    if root_mode not in {"actual", "reference"}:
        raise ReferenceGhostError(f"unsupported ghost root mode: {root_mode!r}")
    if actual_data.qpos.shape != ghost_data.qpos.shape:
        raise ReferenceGhostError(
            "actual and ghost MjData qpos shapes differ: "
            f"{actual_data.qpos.shape} vs {ghost_data.qpos.shape}"
        )

    ghost_data.qpos[:] = actual_data.qpos
    ghost_data.qvel[:] = 0.0
    ghost_data.time = actual_data.time
    if model.nmocap:
        ghost_data.mocap_pos[:] = actual_data.mocap_pos
        ghost_data.mocap_quat[:] = actual_data.mocap_quat

    root = visual.root_qpos_indices
    if root_mode == "reference":
        if aligned_reference_root_position is None:
            raise ReferenceGhostError(
                "reference mode requires an initial-aligned root position"
            )
        aligned_root = np.asarray(aligned_reference_root_position, dtype=np.float64)
        if aligned_root.shape != (3,) or not np.isfinite(aligned_root).all():
            raise ReferenceGhostError(
                "aligned reference root position must be a finite shape-(3,) vector"
            )
        ghost_data.qpos[root[:3]] = aligned_root
    else:
        # Align translation only. Keeping the reference quaternion makes root
        # orientation error remain visible in the overlaid silhouettes.
        ghost_data.qpos[root[:3]] = actual_data.qpos[root[:3]]
    ghost_data.qpos[root[3:7]] = target.root_quaternion_wxyz
    ghost_data.qpos[visual.body_joint_qpos_indices] = target.body_joint_positions
    mujoco.mj_forward(model, ghost_data)


def _scene_data_id(model: mujoco.MjModel, geom_id: int) -> int:
    """Convert a model geom asset id to MuJoCo's user-scene mesh encoding."""

    data_id = int(model.geom_dataid[geom_id])
    if data_id < 0:
        return -1
    geom_type = int(model.geom_type[geom_id])
    if geom_type in (
        int(mujoco.mjtGeom.mjGEOM_MESH),
        int(mujoco.mjtGeom.mjGEOM_SDF),
    ):
        return 2 * data_id
    return data_id


def _scene_uses_texture_coordinates(model: mujoco.MjModel, geom_id: int) -> int:
    data_id = int(model.geom_dataid[geom_id])
    geom_type = int(model.geom_type[geom_id])
    if (
        data_id >= 0
        and geom_type
        in (
            int(mujoco.mjtGeom.mjGEOM_MESH),
            int(mujoco.mjtGeom.mjGEOM_SDF),
        )
        and int(model.mesh_texcoordadr[data_id]) >= 0
    ):
        return 1
    return 0


def fill_reference_ghost_scene(
    *,
    scene: mujoco.MjvScene,
    model: mujoco.MjModel,
    ghost_data: mujoco.MjData,
    visual: RobotVisual,
    alpha: float,
) -> int:
    """Replace the user scene with transparent, visual-only G1 mesh geoms."""

    scene.ngeom = 0
    rgba = np.concatenate((GHOST_RGB, np.asarray([alpha], dtype=np.float32)))
    count = 0
    for geom_id in visual.geom_ids:
        if int(model.geom_bodyid[geom_id]) in visual.hand_body_ids:
            # The reference contains the 29 body joints but no dexterous-hand
            # targets. Hiding the hands avoids presenting actual hand qpos as a
            # reference signal.
            continue
        if count >= int(scene.maxgeom):
            raise ReferenceGhostError(
                f"viewer user-scene capacity {scene.maxgeom} is too small for ghost geoms"
            )

        geom = scene.geoms[count]
        mujoco.mjv_initGeom(
            geom,
            int(model.geom_type[geom_id]),
            np.asarray(model.geom_size[geom_id], dtype=np.float64),
            np.asarray(ghost_data.geom_xpos[geom_id], dtype=np.float64),
            np.asarray(ghost_data.geom_xmat[geom_id], dtype=np.float64),
            rgba,
        )
        # mjv_initGeom cannot infer model-backed mesh assets for custom geoms.
        geom.dataid = _scene_data_id(model, geom_id)
        geom.matid = -1
        geom.texcoord = _scene_uses_texture_coordinates(model, geom_id)
        geom.objtype = int(mujoco.mjtObj.mjOBJ_UNKNOWN)
        geom.objid = -1
        geom.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
        geom.segid = count
        geom.emission = 0.15
        geom.specular = 0.15
        geom.modelrbound = float(model.geom_rbound[geom_id])
        geom.camdist = 0.0
        geom.transparent = 1 if alpha < 1.0 else 0
        count += 1

    scene.ngeom = count
    return count


class ReferenceGhostOverlay:
    """Own the ZMQ target cache and transparent ghost rendering state."""

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        actual_data: mujoco.MjData,
        viewer: Any,
        config: GhostOverlayConfig,
        subscriber: StateSubscriber | None = None,
    ) -> None:
        if viewer is None or viewer.user_scn is None:
            raise ReferenceGhostError(
                "reference ghost requires an onscreen passive MuJoCo viewer"
            )
        self.model = model
        self.actual_data = actual_data
        self.viewer = viewer
        self.config = config
        self.visual = inspect_robot_visual(model, config.visual_geom_group)
        self.ghost_data = mujoco.MjData(model)
        self.reference_root_alignment = ReferenceRootAlignment()
        self.subscriber: StateSubscriber | None = subscriber or ZMQStateSubscriber(
            host=config.host,
            port=config.port,
            topic=config.topic,
        )
        self._target: TargetPose | None = None
        self._last_message_time: float | None = None
        self._closed = False
        self._visible = False
        self._invalid_messages = 0
        self._reported_first_target = False
        self._reported_stale = False
        print(
            "[reference-ghost] ready: "
            f"mode={config.root_mode}, alpha={config.alpha:.3g}, "
            f"geoms={len(self.visual.geom_ids)}"
        )

    def _clear_scene(self) -> None:
        if self.viewer is None or self.viewer.user_scn is None:
            return
        with self.viewer.lock():
            self.viewer.user_scn.ngeom = 0
        self._visible = False

    def _warn_invalid_message(self, exc: BaseException) -> None:
        self._invalid_messages += 1
        if self._invalid_messages == 1 or self._invalid_messages % 100 == 0:
            print(
                "[reference-ghost] ignored invalid g1_debug message "
                f"#{self._invalid_messages}: {exc}"
            )

    def update(self) -> int:
        """Poll the newest target and redraw the ghost; returns drawn geom count."""

        if self._closed or self.subscriber is None:
            return 0
        now = time.monotonic()
        try:
            message = self.subscriber.get_msg(clear=True)
        except Exception as exc:  # Keep a visualization failure out of physics.
            self._warn_invalid_message(exc)
            message = None

        if message is not None:
            try:
                target = TargetPose.from_debug_message(message)
            except ReferenceGhostError as exc:
                self._warn_invalid_message(exc)
            else:
                self._target = target
                self._last_message_time = now
                self._reported_stale = False
                if self.config.root_mode == "reference":
                    actual_root = self.actual_data.qpos[
                        self.visual.root_qpos_indices[:3]
                    ]
                    anchor_event = self.reference_root_alignment.observe(
                        target, actual_root
                    )
                    if anchor_event is not None:
                        aligned = self.reference_root_alignment.aligned_position(target)
                        print(
                            f"[reference-ghost] {anchor_event} reference root at "
                            f"XYZ={np.array2string(aligned, precision=4)}"
                        )
                if not self._reported_first_target:
                    print("[reference-ghost] received first valid reference target")
                    self._reported_first_target = True

        if self._target is None or self._last_message_time is None:
            return 0
        if now - self._last_message_time > self.config.stale_timeout_s:
            if self._visible:
                self._clear_scene()
            if not self._reported_stale:
                if self.config.root_mode == "reference":
                    # A local controller normally publishes every policy tick.
                    # After a full stale timeout, treat the next valid target as
                    # a new session even if its logger index has already caught
                    # up with the previous process before ZMQ reconnects.
                    self.reference_root_alignment.reset()
                print(
                    "[reference-ghost] target stream is stale; hiding ghost until "
                    "g1_debug resumes"
                )
                self._reported_stale = True
            return 0

        with self.viewer.lock():
            aligned_reference_root_position = None
            if self.config.root_mode == "reference":
                aligned_reference_root_position = (
                    self.reference_root_alignment.aligned_position(self._target)
                )
            apply_target_pose(
                model=self.model,
                actual_data=self.actual_data,
                ghost_data=self.ghost_data,
                visual=self.visual,
                target=self._target,
                root_mode=self.config.root_mode,
                aligned_reference_root_position=aligned_reference_root_position,
            )
            count = fill_reference_ghost_scene(
                scene=self.viewer.user_scn,
                model=self.model,
                ghost_data=self.ghost_data,
                visual=self.visual,
                alpha=self.config.alpha,
            )
        self._visible = count > 0
        return count

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.reference_root_alignment.reset()
        try:
            self._clear_scene()
        except Exception:
            pass
        subscriber, self.subscriber = self.subscriber, None
        if subscriber is not None:
            try:
                subscriber.close()
            except Exception:
                pass
