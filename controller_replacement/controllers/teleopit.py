"""Pinned Teleopit v0.5 controller adapter for deterministic rollouts.

The observation and ONNX execution paths are reused from the already audited
``Teleopit_rollout`` implementation, which tracks Teleopit tag ``v0.5.0`` at
commit ``f9263865c581802ad531854b8e547e2403a945f3``.  This module adds the
controller-neutral lifecycle needed by :mod:`controller_replacement`:

* explicit source-history initialization (nine source observations followed
  by the first live observation, exactly matching Teleopit's ONNX history
  buffer semantics);
* an auditable conversion of a source SONIC raw action through physical joint
  target space into Teleopit's native raw-action coordinates; and
* one result object containing the exact observation, ONNX history, raw action,
  joint target, and native low-level controller constants used by a rollout.

No zero-filled or repeated-current history is silently substituted.  A runner
may explicitly reset without a prefill for diagnostics, but metadata exposes
that choice and the formal experiment launcher is expected to require a
:class:`TeleopitSourceHistoryPrefill`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from Teleopit_rollout.constants import (
    ACTION_DIM,
    ACTION_SCALE,
    DEFAULT_DOF_POS,
    G1_JOINT_NAMES,
    HISTORY_LENGTH,
    KDS,
    KPS,
    OBSERVATION_DIM,
    TELEOPIT_COMMIT,
    TELEOPIT_VERSION,
    TORQUE_LIMITS,
)
from Teleopit_rollout.teleopit_policy import (
    RobotState,
    TeleopitObservationBuilder,
    TeleopitOnnxPolicy,
)


FloatArray = np.ndarray
SOURCE_HISTORY_PRIOR_FRAMES = HISTORY_LENGTH - 1
SOURCE_ACTION_CLIP = (-10.0, 10.0)

# ``change_ckpt/source_history_prefill.py`` calls this mapping
# ``MUJOCO_TO_ISAACLAB``.  More explicitly, element ``j`` is the MuJoCo joint
# index containing IsaacLab joint ``j``.  Scattering an IsaacLab vector at
# these indices therefore produces Teleopit's MuJoCo/canonical joint order.
SONIC_MUJOCO_INDEX_FOR_ISAACLAB = np.asarray(
    [
        0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
        16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
    ],
    dtype=np.int64,
)

# SONIC ``policy_parameters.hpp`` values in MuJoCo/hardware order.  They are
# computed there as ``0.25 * effort_limit / stiffness``.  Keeping the values
# here makes the cross-controller previous-action conversion independent of a
# C++ parser while the tests verify the complete physical-target round trip.
SONIC_ACTION_SCALE_MUJOCO = np.asarray(
    [
        0.350661466378824, 0.350661466378824, 0.547546465214230,
        0.350661466378824, 0.438577313923367, 0.438577313923367,
        0.350661466378824, 0.350661466378824, 0.547546465214230,
        0.350661466378824, 0.438577313923367, 0.438577313923367,
        0.547546465214230, 0.438577313923367, 0.438577313923367,
        0.438577313923367, 0.438577313923367, 0.438577313923367,
        0.438577313923367, 0.438577313923367, 0.0745008703295071,
        0.0745008703295071, 0.438577313923367, 0.438577313923367,
        0.438577313923367, 0.438577313923367, 0.438577313923367,
        0.0745008703295071, 0.0745008703295071,
    ],
    dtype=np.float32,
)
SONIC_DEFAULT_DOF_POS_MUJOCO = np.asarray(DEFAULT_DOF_POS, dtype=np.float32).copy()


class TeleopitAdapterError(ValueError):
    """Raised when an input cannot be represented without hidden assumptions."""


@runtime_checkable
class TeleopitStateLike(Protocol):
    """Minimal state surface consumed by the Teleopit observation builder."""

    joint_pos: FloatArray
    joint_vel: FloatArray
    root_pos: FloatArray
    root_quat_wxyz: FloatArray
    root_ang_vel_b: FloatArray
    timestamp_s: float


@runtime_checkable
class TeleopitReferenceLike(Protocol):
    """Reference-frame surface shared by both reference providers."""

    policy_seq: int
    teleopit_qpos36: FloatArray


def _finite_vector(value: Any, size: int, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (size,):
        raise TeleopitAdapterError(f"{name} has shape {array.shape}; expected {(size,)}")
    if not np.all(np.isfinite(array)):
        raise TeleopitAdapterError(f"{name} contains NaN or infinity")
    return array.copy()


def _readonly(value: Any, shape: tuple[int, ...], name: str) -> FloatArray:
    array = np.array(value, dtype=np.float32, order="C", copy=True)
    if array.shape != shape:
        raise TeleopitAdapterError(f"{name} has shape {array.shape}; expected {shape}")
    if not np.all(np.isfinite(array)):
        raise TeleopitAdapterError(f"{name} contains NaN or infinity")
    array.setflags(write=False)
    return array


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PreviousActionConversion:
    """One SONIC-to-Teleopit previous-action conversion with provenance."""

    source_raw_action_isaaclab: FloatArray
    source_q_target_mujoco: FloatArray
    teleopit_raw_action_unclipped: FloatArray
    teleopit_raw_action: FloatArray
    reconstructed_q_target_mujoco: FloatArray
    q_target_residual: FloatArray
    clipped: bool

    def __post_init__(self) -> None:
        for name in (
            "source_raw_action_isaaclab",
            "source_q_target_mujoco",
            "teleopit_raw_action_unclipped",
            "teleopit_raw_action",
            "reconstructed_q_target_mujoco",
            "q_target_residual",
        ):
            object.__setattr__(
                self, name, _readonly(getattr(self, name), (ACTION_DIM,), name)
            )

    @property
    def max_abs_q_target_residual(self) -> float:
        return float(np.max(np.abs(self.q_target_residual)))

    def metadata(self) -> dict[str, Any]:
        return {
            "source_controller": "sonic",
            "source_action_order": "isaaclab_g1_29dof",
            "conversion": (
                "sonic raw action -> SONIC physical q_target in MuJoCo order -> "
                "inverse Teleopit default pose/action scale"
            ),
            "semantic_scope": (
                "counterfactual Teleopit previous-action coordinate representing "
                "the source SONIC physical target; not a Teleopit-generated action"
            ),
            "teleopit_clip_range": list(SOURCE_ACTION_CLIP),
            "clipped": self.clipped,
            "max_abs_q_target_residual_rad": self.max_abs_q_target_residual,
        }


def convert_sonic_previous_action(
    source_raw_action_isaaclab: Any,
    *,
    strict: bool = True,
    residual_tolerance: float = 2e-6,
) -> PreviousActionConversion:
    """Convert a source SONIC action through the physical ``q_target`` space.

    Teleopit's observation contains its *native raw previous action*.  Copying
    SONIC's raw vector would be wrong because the order and per-joint scales
    differ.  The only controller-neutral bridge is the physical joint target.
    If clipping prevents Teleopit from representing that target, strict mode
    rejects the prefill rather than silently claiming an exact conversion.
    """

    source = _finite_vector(
        source_raw_action_isaaclab, ACTION_DIM, "source_raw_action_isaaclab"
    )
    source_mujoco = np.empty(ACTION_DIM, dtype=np.float32)
    source_mujoco[SONIC_MUJOCO_INDEX_FOR_ISAACLAB] = source
    source_target = (
        SONIC_DEFAULT_DOF_POS_MUJOCO + SONIC_ACTION_SCALE_MUJOCO * source_mujoco
    )
    teleopit_unclipped = (
        source_target - np.asarray(DEFAULT_DOF_POS, dtype=np.float32)
    ) / np.asarray(ACTION_SCALE, dtype=np.float32)
    teleopit_action = np.clip(
        teleopit_unclipped, SOURCE_ACTION_CLIP[0], SOURCE_ACTION_CLIP[1]
    ).astype(np.float32)
    reconstructed = (
        np.asarray(DEFAULT_DOF_POS, dtype=np.float32)
        + np.asarray(ACTION_SCALE, dtype=np.float32) * teleopit_action
    )
    residual = reconstructed - source_target
    clipped = bool(np.any(teleopit_action != teleopit_unclipped))
    max_residual = float(np.max(np.abs(residual)))
    if strict and max_residual > residual_tolerance:
        raise TeleopitAdapterError(
            "source SONIC q_target is outside Teleopit's representable action "
            f"range: max residual {max_residual:.9g} rad"
        )
    return PreviousActionConversion(
        source_raw_action_isaaclab=source,
        source_q_target_mujoco=source_target,
        teleopit_raw_action_unclipped=teleopit_unclipped,
        teleopit_raw_action=teleopit_action,
        reconstructed_q_target_mujoco=reconstructed,
        q_target_residual=residual,
        clipped=clipped,
    )


@dataclass(frozen=True)
class TeleopitSourceHistoryPrefill:
    """Nine prior observations plus first-live validation information.

    Teleopit's official dual-input policy appends the current observation to a
    ten-entry deque *before* ONNX inference.  Consequently a correct takeover
    prefill contains nine prior source observations; the first live
    observation becomes entry ten.  ``expected_current_observation`` is never
    inserted into the deque—it is retained only to validate phase alignment.
    """

    prior_observations: FloatArray
    expected_current_observation: FloatArray
    takeover_previous_raw_action: FloatArray
    previous_reference_qpos36: FloatArray
    source_policy_seq: np.ndarray
    action_conversions: tuple[PreviousActionConversion, ...]
    source_csv: str
    source_csv_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "prior_observations",
            _readonly(
                self.prior_observations,
                (SOURCE_HISTORY_PRIOR_FRAMES, OBSERVATION_DIM),
                "prior_observations",
            ),
        )
        object.__setattr__(
            self,
            "expected_current_observation",
            _readonly(
                self.expected_current_observation,
                (OBSERVATION_DIM,),
                "expected_current_observation",
            ),
        )
        object.__setattr__(
            self,
            "takeover_previous_raw_action",
            _readonly(
                self.takeover_previous_raw_action,
                (ACTION_DIM,),
                "takeover_previous_raw_action",
            ),
        )
        object.__setattr__(
            self,
            "previous_reference_qpos36",
            _readonly(
                self.previous_reference_qpos36,
                (36,),
                "previous_reference_qpos36",
            ),
        )
        sequences = np.asarray(self.source_policy_seq, dtype=np.int64).reshape(-1)
        if sequences.shape != (HISTORY_LENGTH,) or np.any(np.diff(sequences) != 1):
            raise TeleopitAdapterError(
                "source_policy_seq must contain ten consecutive policy groups"
            )
        sequences = sequences.copy()
        sequences.setflags(write=False)
        object.__setattr__(self, "source_policy_seq", sequences)
        conversions = tuple(self.action_conversions)
        if len(conversions) != HISTORY_LENGTH or not all(
            isinstance(item, PreviousActionConversion) for item in conversions
        ):
            raise TeleopitAdapterError(
                "action_conversions must contain one conversion for each source group"
            )
        object.__setattr__(self, "action_conversions", conversions)
        try:
            hash_is_hex = (
                len(self.source_csv_sha256) == 64
                and int(self.source_csv_sha256, 16) >= 0
            )
        except ValueError:
            hash_is_hex = False
        if not self.source_csv or not hash_is_hex:
            raise TeleopitAdapterError("source CSV provenance is incomplete")

    @property
    def takeover_policy_seq(self) -> int:
        return int(self.source_policy_seq[-1])

    def metadata(self) -> dict[str, Any]:
        residuals = [item.max_abs_q_target_residual for item in self.action_conversions]
        return {
            "mode": "source_history_prefill",
            "history_semantics": "nine prior source observations plus first live observation",
            "history_order": "oldest_to_newest",
            "prior_observation_count": SOURCE_HISTORY_PRIOR_FRAMES,
            "onnx_history_length": HISTORY_LENGTH,
            "source_policy_seq": self.source_policy_seq.tolist(),
            "takeover_policy_seq": self.takeover_policy_seq,
            "source_csv": self.source_csv,
            "source_csv_sha256": self.source_csv_sha256,
            "previous_action_conversion": (
                "SONIC raw -> SONIC physical q_target -> Teleopit raw"
            ),
            "previous_action_provenance": (
                "source-controller command converted through physical q_target; "
                "never labelled as an action generated by Teleopit"
            ),
            "previous_action_any_clipped": any(
                item.clipped for item in self.action_conversions
            ),
            "previous_action_max_q_target_residual_rad": max(residuals),
        }


def _source_snapshot_state(snapshot: Mapping[str, Any], policy_seq: int) -> RobotState:
    """Recover Teleopit-order measured state from a validated SONIC snapshot."""

    body_q_relative_isaaclab = _finite_vector(
        snapshot.get("body_q"), ACTION_DIM, "source_history.body_q"
    )
    body_dq_isaaclab = _finite_vector(
        snapshot.get("body_dq"), ACTION_DIM, "source_history.body_dq"
    )
    q_isaaclab = body_q_relative_isaaclab + SONIC_DEFAULT_DOF_POS_MUJOCO[
        SONIC_MUJOCO_INDEX_FOR_ISAACLAB
    ]
    q_mujoco = np.empty(ACTION_DIM, dtype=np.float32)
    dq_mujoco = np.empty(ACTION_DIM, dtype=np.float32)
    q_mujoco[SONIC_MUJOCO_INDEX_FOR_ISAACLAB] = q_isaaclab
    dq_mujoco[SONIC_MUJOCO_INDEX_FOR_ISAACLAB] = body_dq_isaaclab
    return RobotState(
        joint_pos=q_mujoco,
        joint_vel=dq_mujoco,
        # World translation has no effect on any Teleopit observation block:
        # only the robot torso orientation is read from robot FK.  The source
        # SONIC prefill payload intentionally omits root xyz, so use an explicit
        # zero with this invariant documented rather than borrowing reference xy.
        root_pos=np.zeros(3, dtype=np.float32),
        root_quat_wxyz=_finite_vector(
            snapshot.get("base_quat"), 4, "source_history.base_quat"
        ),
        root_ang_vel_b=_finite_vector(
            snapshot.get("base_ang_vel"), 3, "source_history.base_ang_vel"
        ),
        timestamp_s=float(policy_seq) / 50.0,
    )


def build_sonic_source_history_prefill(
    *,
    payload: Mapping[str, Any],
    reference_sequence: Any,
    observation_builder: TeleopitObservationBuilder,
    strict_action_conversion: bool = True,
) -> TeleopitSourceHistoryPrefill:
    """Build Teleopit's exact takeover history from a validated SONIC payload.

    ``payload`` is the format emitted by
    :func:`change_ckpt.source_history_prefill.build_source_history_prefill`.
    ``reference_sequence`` must include the policy group immediately before
    the oldest payload entry so the oldest reference velocity can be computed
    without zero padding.
    """

    if payload.get("format") != "g1_decoder_source_history_prefill" or int(
        payload.get("version", -1)
    ) != 1:
        raise TeleopitAdapterError("unsupported SONIC source-history payload")
    if payload.get("history_order") != "oldest_to_newest":
        raise TeleopitAdapterError("source-history entries must be oldest_to_newest")
    if int(payload.get("history_entry_count", -1)) != SOURCE_HISTORY_PRIOR_FRAMES:
        raise TeleopitAdapterError("source-history entry count metadata is inconsistent")
    entries_value = payload.get("entries")
    current_value = payload.get("current")
    if not isinstance(entries_value, Sequence) or isinstance(
        entries_value, (str, bytes)
    ):
        raise TeleopitAdapterError("source-history entries must be a sequence")
    entries = list(entries_value)
    if len(entries) != SOURCE_HISTORY_PRIOR_FRAMES or not isinstance(
        current_value, Mapping
    ):
        raise TeleopitAdapterError(
            "source-history payload must contain nine entries and one current record"
        )
    snapshots: list[Mapping[str, Any]] = []
    for index, value in enumerate([*entries, current_value]):
        if not isinstance(value, Mapping):
            raise TeleopitAdapterError(f"source-history snapshot {index} is not an object")
        snapshots.append(value)

    policy_seq = np.asarray(
        [int(snapshot.get("policy_seq", -1)) for snapshot in snapshots],
        dtype=np.int64,
    )
    if policy_seq.shape != (HISTORY_LENGTH,) or np.any(np.diff(policy_seq) != 1):
        raise TeleopitAdapterError("source-history policy_seq is not consecutive")

    reference_policy_seq = np.asarray(reference_sequence.policy_seq, dtype=np.int64)
    reference_qpos36 = np.asarray(
        reference_sequence.teleopit_qpos36, dtype=np.float32
    )
    if reference_qpos36.shape != (reference_policy_seq.size, 36):
        raise TeleopitAdapterError("reference_sequence Teleopit view has invalid shape")
    if reference_policy_seq.size != np.unique(reference_policy_seq).size:
        raise TeleopitAdapterError("reference_sequence contains duplicate policy_seq")
    index_by_policy = {
        int(sequence): index for index, sequence in enumerate(reference_policy_seq)
    }
    missing = [int(sequence) for sequence in policy_seq if int(sequence) not in index_by_policy]
    if missing:
        raise TeleopitAdapterError(
            f"reference_sequence is missing source-history policies {missing}"
        )
    first_reference_index = index_by_policy[int(policy_seq[0])]
    if first_reference_index == 0 or int(
        reference_policy_seq[first_reference_index - 1]
    ) != int(policy_seq[0] - 1):
        raise TeleopitAdapterError(
            "reference_sequence must include the policy immediately before the "
            "oldest source-history entry"
        )

    observations: list[FloatArray] = []
    conversions: list[PreviousActionConversion] = []
    for snapshot, sequence in zip(snapshots, policy_seq, strict=True):
        reference_index = index_by_policy[int(sequence)]
        if reference_index == 0 or int(reference_policy_seq[reference_index - 1]) != int(
            sequence - 1
        ):
            raise TeleopitAdapterError(
                f"reference predecessor for policy_seq {int(sequence)} is missing"
            )
        current_reference = reference_qpos36[reference_index]
        previous_reference = reference_qpos36[reference_index - 1]
        features = observation_builder.reference_features(
            current_reference, previous_reference
        )
        conversion = convert_sonic_previous_action(
            snapshot.get("last_action"), strict=strict_action_conversion
        )
        observation = observation_builder.build(
            _source_snapshot_state(snapshot, int(sequence)),
            features,
            conversion.teleopit_raw_action,
        )
        observations.append(observation)
        conversions.append(conversion)

    source_csv = str(payload.get("source_csv", ""))
    source_csv_sha256 = str(payload.get("source_csv_sha256", ""))
    return TeleopitSourceHistoryPrefill(
        prior_observations=np.stack(observations[:-1], axis=0),
        expected_current_observation=observations[-1],
        takeover_previous_raw_action=conversions[-1].teleopit_raw_action,
        previous_reference_qpos36=reference_qpos36[
            index_by_policy[int(policy_seq[-1])] - 1
        ],
        source_policy_seq=policy_seq,
        action_conversions=tuple(conversions),
        source_csv=source_csv,
        source_csv_sha256=source_csv_sha256,
    )


class _SourcePrefillableTeleopitPolicy(TeleopitOnnxPolicy):
    """Expose a checked initializer for the audited policy's history deque."""

    def load_precurrent_history(self, observations: Any) -> None:
        history = np.asarray(observations, dtype=np.float32)
        expected = (SOURCE_HISTORY_PRIOR_FRAMES, OBSERVATION_DIM)
        if history.shape != expected or not np.all(np.isfinite(history)):
            raise TeleopitAdapterError(
                f"Teleopit pre-current history must be finite {expected}, got {history.shape}"
            )
        # TeleopitOnnxPolicy owns this deque.  The subclassed write is kept in
        # this one audited method so the old validated module remains untouched.
        self._history = deque(
            (row.copy() for row in history), maxlen=HISTORY_LENGTH
        )
        self.last_history = None


@dataclass(frozen=True)
class TeleopitStep:
    """Controller-native telemetry for one 50 Hz inference."""

    policy_seq: int
    observation: FloatArray
    observation_history: FloatArray
    previous_raw_action: FloatArray
    raw_action: FloatArray
    q_target: FloatArray
    reference_qpos36: FloatArray
    kp: FloatArray
    kd: FloatArray
    torque_limit: FloatArray
    received_dof_pos: FloatArray
    prefill_validation_max_abs: float | None

    def __post_init__(self) -> None:
        for name, shape in (
            ("observation", (OBSERVATION_DIM,)),
            ("observation_history", (HISTORY_LENGTH, OBSERVATION_DIM)),
            ("previous_raw_action", (ACTION_DIM,)),
            ("raw_action", (ACTION_DIM,)),
            ("q_target", (ACTION_DIM,)),
            ("reference_qpos36", (36,)),
            ("kp", (ACTION_DIM,)),
            ("kd", (ACTION_DIM,)),
            ("torque_limit", (ACTION_DIM,)),
            ("received_dof_pos", (ACTION_DIM,)),
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name), shape, name))

    @property
    def history(self) -> FloatArray:
        """Alias matching :class:`SonicInference` telemetry naming."""

        return self.observation_history

    @property
    def last_action(self) -> FloatArray:
        """Alias matching :class:`SonicInference` telemetry naming."""

        return self.previous_raw_action


class TeleopitController:
    """Deterministic controller adapter using Teleopit's native stack."""

    name = "teleopit_v0.5"
    family = "teleopit"
    controller_family = "teleopit"
    joint_names = G1_JOINT_NAMES
    is_sonic = False
    policy_hz = 50.0
    pd_hz = 200.0
    observation_dim = OBSERVATION_DIM
    history_length = HISTORY_LENGTH
    action_dim = ACTION_DIM

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        robot_xml: str | Path,
        device: str = "cpu",
        prefill_validation_atol: float = 2e-4,
        require_source_history: bool = True,
    ) -> None:
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.robot_xml = Path(robot_xml).expanduser().resolve()
        if prefill_validation_atol < 0.0 or not np.isfinite(prefill_validation_atol):
            raise ValueError("prefill_validation_atol must be finite and non-negative")
        self.prefill_validation_atol = float(prefill_validation_atol)
        self.require_source_history = bool(require_source_history)
        self.observation_builder = TeleopitObservationBuilder(self.robot_xml)
        self.policy = _SourcePrefillableTeleopitPolicy(self.checkpoint, device=device)
        self._previous_raw_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._previous_reference_qpos36: FloatArray | None = None
        self._expected_current_observation: FloatArray | None = None
        self._first_step = True
        self.history_initialization = "not_reset"

    @property
    def native_kp(self) -> FloatArray:
        return np.asarray(KPS, dtype=np.float64).copy()

    @property
    def native_kd(self) -> FloatArray:
        return np.asarray(KDS, dtype=np.float64).copy()

    @property
    def native_torque_limits(self) -> FloatArray:
        return np.asarray(TORQUE_LIMITS, dtype=np.float64).copy()

    @property
    def kp(self) -> FloatArray:
        return self.native_kp

    @property
    def kd(self) -> FloatArray:
        return self.native_kd

    @property
    def torque_limit(self) -> FloatArray:
        return self.native_torque_limits

    @property
    def previous_raw_action(self) -> FloatArray:
        return self._previous_raw_action.copy()

    def reset(self, prefill: TeleopitSourceHistoryPrefill | None = None) -> None:
        """Reset controller state, optionally installing the formal prefill.

        ``prefill=None`` is deliberately labelled ``repeat_current_diagnostic``:
        the inherited official policy will repeat the first live observation
        ten times.  With the default ``require_source_history=True``, inference
        rejects that diagnostic initialization.  It must be enabled explicitly
        at construction time and is never a formal controller-replacement run.
        """

        self.policy.reset()
        self._previous_raw_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._previous_reference_qpos36 = None
        self._expected_current_observation = None
        self._first_step = True
        if prefill is None:
            self.history_initialization = "repeat_current_diagnostic"
            return
        if not isinstance(prefill, TeleopitSourceHistoryPrefill):
            raise TypeError("prefill must be TeleopitSourceHistoryPrefill or None")
        self.policy.load_precurrent_history(prefill.prior_observations)
        self._previous_raw_action = prefill.takeover_previous_raw_action.copy()
        self._previous_reference_qpos36 = prefill.previous_reference_qpos36.copy()
        self._expected_current_observation = (
            prefill.expected_current_observation.copy()
        )
        self.history_initialization = "source_history_prefill"

    def build_observation(
        self,
        robot_state: TeleopitStateLike,
        reference: TeleopitReferenceLike,
    ) -> FloatArray:
        qpos36 = _finite_vector(reference.teleopit_qpos36, 36, "teleopit_qpos36")
        features = self.observation_builder.reference_features(
            qpos36, self._previous_reference_qpos36
        )
        state = RobotState(
            joint_pos=_finite_vector(robot_state.joint_pos, ACTION_DIM, "joint_pos"),
            joint_vel=_finite_vector(robot_state.joint_vel, ACTION_DIM, "joint_vel"),
            root_pos=_finite_vector(robot_state.root_pos, 3, "root_pos"),
            root_quat_wxyz=_finite_vector(
                robot_state.root_quat_wxyz, 4, "root_quat_wxyz"
            ),
            root_ang_vel_b=_finite_vector(
                robot_state.root_ang_vel_b, 3, "root_ang_vel_b"
            ),
            timestamp_s=float(robot_state.timestamp_s),
        )
        return self.observation_builder.build(
            state, features, self._previous_raw_action
        )

    def infer(
        self,
        robot_state: TeleopitStateLike,
        reference: TeleopitReferenceLike,
    ) -> TeleopitStep:
        """Run one 50 Hz policy inference and advance native history state."""

        if self.history_initialization == "not_reset":
            raise TeleopitAdapterError("reset() must be called before inference")
        if self.require_source_history and self.history_initialization != (
            "source_history_prefill"
        ):
            raise TeleopitAdapterError(
                "formal Teleopit inference requires source-history prefill; "
                "construct with require_source_history=False only for diagnostics"
            )
        observation = self.build_observation(robot_state, reference)
        validation_error: float | None = None
        if self._first_step and self._expected_current_observation is not None:
            validation_error = float(
                np.max(np.abs(observation - self._expected_current_observation))
            )
            if validation_error > self.prefill_validation_atol:
                raise TeleopitAdapterError(
                    "first live Teleopit observation does not match the phase-aligned "
                    f"source prefill: max error {validation_error:.9g} exceeds "
                    f"{self.prefill_validation_atol:.9g}"
                )
        previous_action = self._previous_raw_action.copy()
        raw_action, q_target, history = self.policy.infer(observation)
        qpos36 = _finite_vector(reference.teleopit_qpos36, 36, "teleopit_qpos36")
        result = TeleopitStep(
            policy_seq=int(reference.policy_seq),
            observation=observation,
            observation_history=history,
            previous_raw_action=previous_action,
            raw_action=raw_action,
            q_target=q_target,
            reference_qpos36=qpos36,
            kp=self.kp,
            kd=self.kd,
            torque_limit=self.torque_limit,
            received_dof_pos=np.asarray(robot_state.joint_pos, dtype=np.float32),
            prefill_validation_max_abs=validation_error,
        )
        self._previous_raw_action = np.asarray(raw_action, dtype=np.float32).copy()
        self._previous_reference_qpos36 = qpos36.copy()
        self._expected_current_observation = None
        self._first_step = False
        return result

    def compute_pd_torque(
        self, joint_pos: Any, joint_vel: Any, q_target: Any
    ) -> FloatArray:
        """Apply Teleopit's native PD gains and torque limits."""

        q = _finite_vector(joint_pos, ACTION_DIM, "joint_pos")
        dq = _finite_vector(joint_vel, ACTION_DIM, "joint_vel")
        target = _finite_vector(q_target, ACTION_DIM, "q_target")
        torque = np.asarray(KPS, dtype=np.float64) * (target - q) - np.asarray(
            KDS, dtype=np.float64
        ) * dq
        return np.clip(
            torque,
            -np.asarray(TORQUE_LIMITS, dtype=np.float64),
            np.asarray(TORQUE_LIMITS, dtype=np.float64),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "controller": self.name,
            "family": self.family,
            "teleopit_version": TELEOPIT_VERSION,
            "teleopit_commit": TELEOPIT_COMMIT,
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": _sha256(self.checkpoint),
            "fk_robot_xml": str(self.robot_xml),
            "fk_robot_xml_sha256": _sha256(self.robot_xml),
            "observation_dim": OBSERVATION_DIM,
            "observation_history_length": HISTORY_LENGTH,
            "history_initialization": self.history_initialization,
            "source_history_required": self.require_source_history,
            "action_dim": ACTION_DIM,
            "action_mapping": "default_dof_pos + action_scale * clip(raw,-10,10)",
            "pd": "Teleopit v0.5 native Kp/Kd and torque limits",
            "policy_hz": self.policy_hz,
            "pd_hz": self.pd_hz,
        }


__all__ = [
    "PreviousActionConversion",
    "SOURCE_HISTORY_PRIOR_FRAMES",
    "TeleopitAdapterError",
    "TeleopitController",
    "TeleopitSourceHistoryPrefill",
    "TeleopitStateLike",
    "TeleopitStep",
    "build_sonic_source_history_prefill",
    "convert_sonic_previous_action",
]
