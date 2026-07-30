#!/usr/bin/env python3
"""Publish a qpos-derived G1 reference through SONIC ZMQ protocol v1.

The streamed values always come from the continuous 50 Hz sequence built by
``qpos_reference_data.load_qpos_reference``:

* 29 body joint positions in IsaacLab order;
* 29 body joint velocities in IsaacLab order;
* the recorded pelvis quaternion in wxyz order;
* the recorded seven-joint target for each hand.

The default ``canonical`` window sends consecutive qpos frames.  An optional
regular-only ``recorded`` window reads ``reference_motion`` solely to infer
the original ten temporal lags (for example ``0,5,9,...,9``), then rearranges
qpos/qvel/pelvis-quaternion values so the unchanged C++ step-5 gatherer sees
those lags.  No ``reference_motion`` value or external token is sent.

The regular checkpoint needs at least 46 packet frames because its G1 encoder
samples offsets ``0, 5, ..., 45``; low-latency needs at least 10 frames because
it samples offsets ``0, 1, ..., 9``.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np


# These literals are inspected statically by launch_checkpoint_rollout.py.
PROTOCOL_VERSION = 1
PUBLISHES_EXTERNAL_TOKEN = False
HEADER_SIZE = 1280

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_INPUT = (
    REPO_ROOT / "sample_data/ztj/20260612/20260612_144117_g1_sim"
)
MIN_PACKET_FRAMES = {"regular": 46, "low_latency": 10}
REGULAR_GATHER_STEP = 5
REGULAR_NUM_SLOTS = 10
CANONICAL_REGULAR_LAGS = tuple(
    index * REGULAR_GATHER_STEP for index in range(REGULAR_NUM_SLOTS)
)


class TrackPublisherError(RuntimeError):
    """A qpos-track input or runtime configuration is invalid."""


def _import_reference_api() -> tuple[Any, Any, Any]:
    try:
        from change_ckpt_track.qpos_reference_data import (
            QposReferenceSequence,
            build_diagnostics,
            load_qpos_reference,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        raise TrackPublisherError(
            "cannot import change_ckpt_track.qpos_reference_data; "
            "the qpos reference builder must be present"
        ) from exc
    return QposReferenceSequence, build_diagnostics, load_qpos_reference


def _import_wire_helpers() -> tuple[Any, Any]:
    try:
        from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
            build_command_message,
            pack_pose_message,
        )
    except Exception as exc:  # noqa: BLE001
        raise TrackPublisherError(
            f"cannot import the repository ZMQ wire helpers: {exc}"
        ) from exc
    return build_command_message, pack_pose_message


def _import_recorded_lag_api() -> Any:
    try:
        from change_ckpt_track.recorded_slot_lags import (
            infer_recorded_slot_lags,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        raise TrackPublisherError(
            "cannot import change_ckpt_track.recorded_slot_lags"
        ) from exc
    return infer_recorded_slot_lags


def _normalise_layout(value: str) -> str:
    aliases = {
        "regular": "regular",
        "release": "regular",
        "sonic_release": "regular",
        "low": "low_latency",
        "low-latency": "low_latency",
        "low_latency": "low_latency",
    }
    try:
        return aliases[value]
    except KeyError as exc:
        raise TrackPublisherError(f"unsupported checkpoint layout: {value}") from exc


def describe_protocol() -> dict[str, Any]:
    return {
        "version": PROTOCOL_VERSION,
        "topic": "pose",
        "reference_source": (
            "recorded qpos/qvel; reference_motion may be inspected only to "
            "infer regular temporal lags"
        ),
        "publishes_external_token": PUBLISHES_EXTERNAL_TOKEN,
        "fields": {
            "joint_pos": (
                "f32[N,29] qpos-derived frames (IsaacLab order; optionally "
                "regular recorded-lag rearranged)"
            ),
            "joint_vel": (
                "f32[N,29] qvel-derived frames (IsaacLab order; optionally "
                "regular recorded-lag rearranged)"
            ),
            "body_quat_w": "f32[N,4] recorded pelvis quaternion (w,x,y,z)",
            "frame_index": (
                "i64[N] consecutive synthetic policy_seq timeline; never "
                "rearranged or repeated"
            ),
            "catch_up": "bool[1]",
            "left_hand_joints": "optional f32[7], synchronized recorded target",
            "right_hand_joints": "optional f32[7], synchronized recorded target",
        },
        "minimum_packet_frames": MIN_PACKET_FRAMES,
        "forbidden_fields": ["token_state", "reference_motion"],
        "wire_layout": (
            "topic prefix + 1280-byte JSON header + concatenated little-endian fields"
        ),
    }


def _num_frames(sequence: Any) -> int:
    return int(np.asarray(sequence.policy_seq).shape[0])


def _window(array: np.ndarray, current: int, count: int) -> np.ndarray:
    source = np.asarray(array)
    stop = min(source.shape[0], current + count)
    result = np.asarray(source[current:stop], dtype=np.float32)
    if result.shape[0] == 0:
        raise TrackPublisherError(
            f"packet start {current} is outside a {source.shape[0]}-frame sequence"
        )
    if result.shape[0] < count:
        result = np.concatenate(
            (result, np.repeat(result[-1:], count - result.shape[0], axis=0)),
            axis=0,
        )
    return np.ascontiguousarray(result, dtype=np.float32)


def _regular_recorded_lag_window(
    array: np.ndarray,
    current: int,
    count: int,
    lags: Sequence[int],
) -> np.ndarray:
    """Weave qpos-derived values for the C++ ten-frame step-5 gatherer.

    C++ may consume any of the five packet residues before the next 50 Hz
    network update.  Packet row ``r + 5*s`` therefore carries source frame
    ``current + r + lags[s]``.  Rows after slot nine repeat that last slot,
    matching the existing recorded-reference publisher's safety margin.
    """

    source = np.asarray(array)
    lag_array = np.asarray(lags, dtype=np.int64)
    if lag_array.shape != (REGULAR_NUM_SLOTS,):
        raise TrackPublisherError(
            "regular recorded future lags must contain exactly "
            f"{REGULAR_NUM_SLOTS} values; got {lag_array.tolist()}"
        )
    if lag_array[0] != 0 or np.any(lag_array < 0):
        raise TrackPublisherError(
            "regular recorded future lags must start at zero and be non-negative"
        )
    if np.any(np.diff(lag_array) < 0):
        raise TrackPublisherError(
            "regular recorded future lags must be non-decreasing"
        )
    local = np.arange(count, dtype=np.int64)
    slot = np.minimum(local // REGULAR_GATHER_STEP, REGULAR_NUM_SLOTS - 1)
    source_indices = (
        current + local % REGULAR_GATHER_STEP + lag_array[slot]
    )
    source_indices = np.minimum(source_indices, source.shape[0] - 1)
    return np.ascontiguousarray(source[source_indices], dtype=np.float32)


def _packet_indices(sequence: Any, current: int, count: int) -> np.ndarray:
    values = np.asarray(sequence.policy_seq, dtype=np.int64)
    stop = min(values.shape[0], current + count)
    result = values[current:stop].copy()
    if result.shape[0] == 0:
        raise TrackPublisherError(
            f"packet start {current} is outside a {values.shape[0]}-frame sequence"
        )
    if result.shape[0] < count:
        result = np.concatenate(
            (
                result,
                np.arange(
                    int(result[-1]) + 1,
                    int(result[-1]) + 1 + count - result.shape[0],
                    dtype=np.int64,
                ),
            )
        )
    return np.ascontiguousarray(result, dtype=np.int64)


def build_pose_payload(
    sequence: Any,
    current: int,
    packet_frames: int,
    *,
    include_hands: bool,
    checkpoint_layout: str,
    regular_future_lags: Sequence[int] | None = None,
    heading_increment: float | None = None,
) -> dict[str, np.ndarray]:
    """Build one rolling packet from qpos-derived 50 Hz frames."""

    layout = _normalise_layout(checkpoint_layout)
    required = MIN_PACKET_FRAMES[layout]
    if packet_frames < required:
        raise TrackPublisherError(
            f"{layout} needs at least {required} consecutive packet frames; "
            f"got {packet_frames}"
        )

    if layout == "regular" and regular_future_lags is not None:
        joint_pos = _regular_recorded_lag_window(
            sequence.joint_pos, current, packet_frames, regular_future_lags
        )
        joint_vel = _regular_recorded_lag_window(
            sequence.joint_vel, current, packet_frames, regular_future_lags
        )
        body_quat = _regular_recorded_lag_window(
            sequence.root_quat_wxyz,
            current,
            packet_frames,
            regular_future_lags,
        )
    else:
        joint_pos = _window(sequence.joint_pos, current, packet_frames)
        joint_vel = _window(sequence.joint_vel, current, packet_frames)
        body_quat = _window(
            sequence.root_quat_wxyz, current, packet_frames
        )

    payload: dict[str, np.ndarray] = {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_quat_w": body_quat,
        "frame_index": _packet_indices(sequence, current, packet_frames),
        "catch_up": np.asarray([False], dtype=bool),
    }
    if include_hands:
        payload["left_hand_joints"] = np.ascontiguousarray(
            np.asarray(sequence.left_hand_target[current], dtype=np.float32)
        )
        payload["right_hand_joints"] = np.ascontiguousarray(
            np.asarray(sequence.right_hand_target[current], dtype=np.float32)
        )
    if heading_increment is not None:
        payload["heading_increment"] = np.asarray(
            [heading_increment], dtype=np.float32
        )
    if "token_state" in payload or "reference_motion" in payload:
        raise AssertionError("qpos-track protocol v1 must use the local encoder")
    return payload


def _send_repeated(socket: Any, message: bytes, repeats: int, interval_s: float) -> None:
    for index in range(repeats):
        socket.send(message)
        if index + 1 < repeats:
            time.sleep(interval_s)


def _write_status(path_value: str | None, status: str) -> None:
    if not path_value:
        return
    path = Path(path_value).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(status + "\n", encoding="utf-8")


def _wait_for_gate(
    args: argparse.Namespace,
    should_stop: Any,
    ready_values: set[str] | None = None,
) -> None:
    if not args.gate_status_file:
        return
    expected = ready_values or {args.gate_ready_value}
    path = Path(args.gate_status_file).expanduser().resolve()
    deadline = time.monotonic() + args.gate_timeout
    print(
        f"[qpos-track] waiting for gate {path} in {sorted(expected)!r}; "
        "reference tick 0 remains frozen",
        flush=True,
    )
    last_value = "<missing>"
    while True:
        if should_stop():
            raise KeyboardInterrupt
        try:
            last_value = path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            last_value = "<missing>"
        if last_value in expected:
            print(f"[qpos-track] gate ready: {last_value}", flush=True)
            return
        if time.monotonic() >= deadline:
            raise TrackPublisherError(
                f"timed out after {args.gate_timeout:g}s waiting for {path} in "
                f"{sorted(expected)!r}; last value was {last_value!r}"
            )
        time.sleep(args.gate_poll_interval)


def _sequence_diagnostics(
    sequence: Any,
    args: argparse.Namespace,
    *,
    full_frames: int,
) -> dict[str, Any]:
    counts = np.asarray(sequence.group_row_counts, dtype=np.int64)
    quaternions = np.asarray(sequence.root_quat_wxyz, dtype=np.float64)
    norms = np.linalg.norm(quaternions, axis=1)
    recorded_window = args.regular_future_window == "recorded"
    if args.checkpoint_layout == "regular":
        encoder_lags = (
            list(args.regular_future_lags)
            if recorded_window
            else list(CANONICAL_REGULAR_LAGS)
        )
    else:
        encoder_lags = list(range(REGULAR_NUM_SLOTS))
    return {
        "reference_kind": "recorded_robot_qpos_track",
        "source_csv": str(sequence.csv_path),
        "drop_truncated_edges": bool(args.drop_truncated_edges),
        "processed_frames_before_slice": full_frames,
        "selected_frames": _num_frames(sequence),
        "selected_processed_offset": args.start_policy_offset,
        "first_policy_seq": int(sequence.policy_seq[0]),
        "last_policy_seq": int(sequence.policy_seq[-1]),
        "first_source_row_index": int(sequence.source_row_index[0]),
        "last_source_row_index": int(sequence.source_row_index[-1]),
        "rate_hz": args.rate,
        "checkpoint_layout": args.checkpoint_layout,
        "regular_future_window": args.regular_future_window,
        "encoder_source_policy_lags": encoder_lags,
        "regular_packet_rearrangement": (
            (
                "packet row r+5*s reads qpos source "
                "current+r+encoder_source_policy_lags[s]"
            )
            if recorded_window
            else "none"
        ),
        "packet_frames": min(args.chunk_size, args.lookahead),
        "joint_pos_shape": list(np.asarray(sequence.joint_pos).shape),
        "joint_vel_shape": list(np.asarray(sequence.joint_vel).shape),
        "root_pos_shape": list(np.asarray(sequence.root_pos).shape),
        "root_quat_shape": list(quaternions.shape),
        "root_quat_norm_max_error": float(np.max(np.abs(norms - 1.0))),
        "rows_per_policy_seq": {
            "min": int(counts.min()),
            "max": int(counts.max()),
            "median": float(np.median(counts)),
        },
        "joint_order": "isaaclab",
        "joint_names": list(sequence.joint_names),
        "policy_seq": [
            int(value) for value in np.asarray(sequence.policy_seq)
        ],
        "source_row_indices": [
            int(value) for value in np.asarray(sequence.source_row_index)
        ],
        "source_csv_row_numbers": [
            int(value) for value in np.asarray(sequence.source_csv_row_number)
        ],
        "hands_published": not args.no_hands,
        "external_token": False,
        "reference_motion_columns_used": recorded_window,
        "reference_motion_values_streamed": False,
        "reference_motion_columns_used_for_lag_inference": recorded_window,
        "recorded_slot_lag_inference": (
            args.recorded_slot_lag_inference if recorded_window else None
        ),
        "root_position_wire_note": (
            "root_pos is retained in prepared_reference.npz for inspection; "
            "SONIC protocol-v1 streamed motion carries root quaternion but no body position"
        ),
    }


def _write_prepared_npz(
    sequence: Any,
    path_value: str,
    diagnostics: dict[str, Any],
) -> Path:
    path = Path(path_value).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        policy_seq=np.asarray(sequence.policy_seq, dtype=np.int64),
        control_time_s=np.asarray(sequence.control_time_s, dtype=np.float64),
        source_row_index=np.asarray(sequence.source_row_index, dtype=np.int64),
        source_csv_row_number=np.asarray(
            sequence.source_csv_row_number, dtype=np.int64
        ),
        group_row_counts=np.asarray(sequence.group_row_counts, dtype=np.int32),
        joint_pos=np.asarray(sequence.joint_pos, dtype=np.float32),
        joint_vel=np.asarray(sequence.joint_vel, dtype=np.float32),
        root_pos=np.asarray(sequence.root_pos, dtype=np.float32),
        body_quat_w=np.asarray(sequence.root_quat_wxyz, dtype=np.float32),
        left_hand_joints=np.asarray(sequence.left_hand_target, dtype=np.float32),
        right_hand_joints=np.asarray(sequence.right_hand_target, dtype=np.float32),
        encoder_source_policy_lags=np.asarray(
            diagnostics["encoder_source_policy_lags"], dtype=np.int64
        ),
        joint_names=np.asarray(sequence.joint_names),
        metadata_json=np.asarray(
            json.dumps(diagnostics, ensure_ascii=False, sort_keys=True)
        ),
    )
    return path


def run_publisher(args: argparse.Namespace, sequence: Any) -> int:
    try:
        import zmq
    except ImportError as exc:
        raise TrackPublisherError(
            "pyzmq is required; run this publisher with .venv_sim/bin/python"
        ) from exc
    build_command_message, pack_pose_message = _import_wire_helpers()

    packet_frames = min(args.chunk_size, args.lookahead)
    required = MIN_PACKET_FRAMES[args.checkpoint_layout]
    if packet_frames < required:
        raise TrackPublisherError(
            f"{args.checkpoint_layout} needs at least {required} consecutive frames; "
            f"min(chunk-size, lookahead)={packet_frames}"
        )

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.SNDHWM, 3)
    socket.setsockopt(zmq.LINGER, 0)
    endpoint = f"tcp://{args.host}:{args.port}"
    try:
        socket.bind(endpoint)
    except zmq.ZMQError as exc:
        socket.close(0)
        context.term()
        raise TrackPublisherError(f"cannot bind publisher to {endpoint}: {exc}") from exc

    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    print(
        f"[qpos-track] bound {endpoint}; protocol=v{PROTOCOL_VERSION}, "
        f"layout={args.checkpoint_layout}, frames={_num_frames(sequence)}, "
        f"packet={packet_frames}, source=qpos, "
        f"future_window={args.regular_future_window}, token_state=never",
        flush=True,
    )
    try:
        # PUB/SUB has no delivery acknowledgement and drops messages while the
        # SUB connection is still joining.  Keep the PUB socket bound throughout
        # deploy INIT, but do not spend the one-shot protocol-v1 pose until the
        # simulator confirms deploy has reached WAIT_FOR_CONTROL.  This also
        # avoids repeating frame_index[0], which the C++ merger interprets as a
        # catch-up/reset.
        time.sleep(args.subscriber_warmup)
        _wait_for_gate(args, lambda: stop_requested)

        if not args.no_command:
            switch_message = build_command_message(
                start=False, stop=False, planner=False
            )
            _send_repeated(
                socket,
                switch_message,
                args.command_retries,
                args.command_retry_interval,
            )
            time.sleep(args.mode_switch_delay)

        initial_payload = build_pose_payload(
            sequence,
            0,
            packet_frames,
            include_hands=not args.no_hands,
            checkpoint_layout=args.checkpoint_layout,
            regular_future_lags=getattr(
                args, "regular_future_lags", None
            ),
        )
        initial_message = pack_pose_message(
            initial_payload, topic=args.topic, version=PROTOCOL_VERSION
        )
        _send_repeated(
            socket,
            initial_message,
            args.initial_pose_retries,
            args.command_retry_interval,
        )
        time.sleep(args.initial_pose_delay)

        _write_status(args.publisher_status_file, "ready_for_control")

        if not args.no_command:
            start_message = build_command_message(
                start=True, stop=False, planner=False
            )
            _send_repeated(
                socket,
                start_message,
                args.command_retries,
                args.command_retry_interval,
            )

        if args.gate_status_file:
            _wait_for_gate(args, lambda: stop_requested, {"running"})
        _write_status(args.publisher_status_file, "streaming")

        period = 1.0 / args.rate
        first_timed_tick = 1
        start_clock = time.monotonic()
        late_count = 0
        for current in range(first_timed_tick, _num_frames(sequence)):
            if stop_requested:
                return 130
            deadline = start_clock + (current - first_timed_tick) * period
            remaining = deadline - time.monotonic()
            if remaining > 0.0:
                time.sleep(remaining)
            elif remaining < -period:
                late_count += 1

            # With qpos-derived reference the simulator and reference pelvis are
            # identical at tick zero, so the old relative-heading correction is
            # exactly zero.  Keep the compatibility flag explicit and auditable.
            heading = (
                0.0
                if args.apply_heading_correction
                and current == args.heading_correction_tick
                else None
            )
            payload = build_pose_payload(
                sequence,
                current,
                packet_frames,
                include_hands=not args.no_hands,
                checkpoint_layout=args.checkpoint_layout,
                regular_future_lags=getattr(
                    args, "regular_future_lags", None
                ),
                heading_increment=heading,
            )
            socket.send(
                pack_pose_message(
                    payload, topic=args.topic, version=PROTOCOL_VERSION
                )
            )
            if (
                current % max(1, int(args.rate)) == 0
                or current + 1 == _num_frames(sequence)
            ):
                print(
                    f"[qpos-track] tick={current}/{_num_frames(sequence) - 1} "
                    f"policy_seq={int(sequence.policy_seq[current])} "
                    f"source_row={int(sequence.source_row_index[current])} "
                    f"hands={'off' if args.no_hands else 'on'}",
                    flush=True,
                )

        _write_status(args.publisher_status_file, "completed")
        print(
            f"[qpos-track] source complete; late_ticks={late_count}. "
            "Simulator may now flush its replay CSV.",
            flush=True,
        )
        return 3 if late_count else 0
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        socket.close(0)
        context.term()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Original recording directory or its data.csv",
    )
    parser.add_argument("--host", default="*", help="ZMQ PUB bind host")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--topic", default="pose")
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument(
        "--start-policy-offset",
        type=int,
        default=0,
        help=(
            "Zero-based offset within the processed qpos sequence, after optional "
            "truncated-edge removal"
        ),
    )
    parser.add_argument("--max-policy-frames", type=int)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--lookahead", type=int, default=100)
    parser.add_argument(
        "--checkpoint-layout",
        choices=(
            "regular",
            "release",
            "sonic_release",
            "low_latency",
            "low",
            "low-latency",
            "auto",
        ),
        default="auto",
    )
    parser.add_argument(
        "--regular-future-window",
        choices=("canonical", "recorded"),
        default="canonical",
        help=(
            "Regular-only temporal layout: canonical sends consecutive qpos "
            "so C++ sees 0,5,...,45; recorded infers the ten lags from this "
            "recording's reference_motion and rearranges qpos values"
        ),
    )
    parser.add_argument(
        "--drop-truncated-edges",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop the visibly partial first/last policy_seq groups",
    )
    parser.add_argument(
        "--base-sample",
        default="unused-for-qpos-track",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--no-hands", action="store_true")
    parser.add_argument("--no-command", action="store_true")
    parser.add_argument("--subscriber-warmup", type=float, default=1.0)
    parser.add_argument("--mode-switch-delay", type=float, default=0.25)
    parser.add_argument("--initial-pose-delay", type=float, default=0.20)
    parser.add_argument("--gate-status-file")
    parser.add_argument("--gate-ready-value", default="deploy_init_ready")
    parser.add_argument("--publisher-status-file")
    parser.add_argument("--gate-timeout", type=float, default=30.0)
    parser.add_argument("--gate-poll-interval", type=float, default=0.001)
    parser.add_argument("--command-retries", type=int, default=3)
    parser.add_argument("--initial-pose-retries", type=int, default=1)
    parser.add_argument("--command-retry-interval", type=float, default=0.05)
    parser.add_argument("--apply-heading-correction", action="store_true")
    parser.add_argument("--heading-correction-tick", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--diagnose-orientation", action="store_true")
    parser.add_argument("--diagnostics-json")
    parser.add_argument("--prepared-output")
    parser.add_argument("--describe-protocol", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.checkpoint_layout == "auto":
        raise TrackPublisherError(
            "--checkpoint-layout must be regular or low_latency for qpos-track"
        )
    args.checkpoint_layout = _normalise_layout(args.checkpoint_layout)
    if (
        args.regular_future_window == "recorded"
        and args.checkpoint_layout != "regular"
    ):
        raise TrackPublisherError(
            "--regular-future-window recorded is valid only with "
            "--checkpoint-layout regular"
        )
    if args.rate <= 0.0:
        raise TrackPublisherError("--rate must be positive")
    if args.start_policy_offset < 0:
        raise TrackPublisherError("--start-policy-offset cannot be negative")
    if args.max_policy_frames is not None and args.max_policy_frames <= 0:
        raise TrackPublisherError("--max-policy-frames must be positive")
    for name in (
        "chunk_size",
        "lookahead",
        "command_retries",
        "initial_pose_retries",
    ):
        if getattr(args, name) <= 0:
            raise TrackPublisherError(
                f"--{name.replace('_', '-')} must be positive"
            )
    for name in (
        "subscriber_warmup",
        "mode_switch_delay",
        "initial_pose_delay",
        "command_retry_interval",
    ):
        if getattr(args, name) < 0.0:
            raise TrackPublisherError(
                f"--{name.replace('_', '-')} cannot be negative"
            )
    if args.gate_timeout <= 0.0 or args.gate_poll_interval <= 0.0:
        raise TrackPublisherError("gate timeout and poll interval must be positive")


def _configure_regular_future_window(args: argparse.Namespace) -> None:
    args.regular_future_lags = None
    args.recorded_slot_lag_inference = None
    if args.regular_future_window != "recorded":
        return
    infer_recorded_slot_lags = _import_recorded_lag_api()
    try:
        report = infer_recorded_slot_lags(args.input)
    except ValueError as exc:
        raise TrackPublisherError(
            f"cannot infer recorded regular future window: {exc}"
        ) from exc
    lags = tuple(int(value) for value in report["future_slot_policy_lags"])
    if len(lags) != REGULAR_NUM_SLOTS:
        raise TrackPublisherError(
            "recorded lag inference did not return exactly ten slots: "
            f"{list(lags)}"
        )
    args.regular_future_lags = lags
    args.recorded_slot_lag_inference = report
    print(
        "[qpos-track] inferred recorded regular encoder lags: "
        f"{list(lags)}",
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.describe_protocol:
            print(json.dumps(describe_protocol(), indent=2, ensure_ascii=False))
            return 0
        _validate_args(args)
        _configure_regular_future_window(args)
        (
            _sequence_type,
            build_reference_diagnostics,
            load_qpos_reference,
        ) = _import_reference_api()
        full_sequence = load_qpos_reference(
            args.input, drop_truncated_edges=args.drop_truncated_edges
        )
        full_frames = _num_frames(full_sequence)
        sequence = full_sequence.slice(
            args.start_policy_offset, count=args.max_policy_frames
        )
        policy_diff = np.diff(np.asarray(sequence.policy_seq, dtype=np.int64))
        if policy_diff.size and np.any(policy_diff != 1):
            bad = np.flatnonzero(policy_diff != 1)[:5]
            examples = ", ".join(
                f"{int(sequence.policy_seq[index])}->{int(sequence.policy_seq[index + 1])}"
                for index in bad
            )
            raise TrackPublisherError(
                f"qpos reference policy_seq must be consecutive; examples: {examples}"
            )
        diagnostics = _sequence_diagnostics(
            sequence, args, full_frames=full_frames
        )
        diagnostics["reference_builder_diagnostics"] = (
            build_reference_diagnostics(sequence)
        )
        if args.diagnostics_json:
            path = Path(args.diagnostics_json).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        if args.prepared_output:
            _write_prepared_npz(sequence, args.prepared_output, diagnostics)
        print(json.dumps(diagnostics, indent=2, ensure_ascii=False), flush=True)
        if args.dry_run or args.diagnose_orientation:
            return 0
        return run_publisher(args, sequence)
    except (TrackPublisherError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
