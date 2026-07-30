#!/usr/bin/env python3
"""Publish a recorded SONIC G1 reference through ZMQ protocol v1.

This process intentionally publishes motion fields, not ``token_state``.  The
C++ deployment therefore runs its local encoder and decoder exactly as in the
normal sim2sim path.

Typical use (the subscriber/deploy process must already be running)::

    .venv_sim/bin/python change_ckpt/csv_reference_publisher.py \
        --input sample_data/ztj/20260720_144342_g1_sim

Use ``--dry-run`` first to inspect policy deduplication, future-frame lags and
the absolute-orientation reconstruction without opening a network socket.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from change_ckpt.reference_data import (
        ReferenceDataError,
        ReferenceSequence,
        build_diagnostics,
        json_dump,
        load_reference_sequence,
        quat_conjugate_wxyz,
        quat_multiply_wxyz,
        quat_to_matrix_wxyz,
        reference_slot_views,
        write_prepared_npz,
    )
except ModuleNotFoundError:
    # Direct ``python change_ckpt/csv_reference_publisher.py`` execution puts
    # change_ckpt rather than the repository root at sys.path[0].
    from reference_data import (  # type: ignore[no-redef]
        ReferenceDataError,
        ReferenceSequence,
        build_diagnostics,
        json_dump,
        load_reference_sequence,
        quat_conjugate_wxyz,
        quat_multiply_wxyz,
        quat_to_matrix_wxyz,
        reference_slot_views,
        write_prepared_npz,
    )


# Parsed statically by launch_checkpoint_rollout.py.  Do not turn these into
# computed values: preflight must be able to prove that the local encoder is not
# bypassed without importing this process.
PROTOCOL_VERSION = 1
PUBLISHES_EXTERNAL_TOKEN = False
HEADER_SIZE = 1280

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "sample_data/ztj/20260720_144342_g1_sim"
REGULAR_PACKET_SLOT_OFFSETS = np.arange(10, dtype=np.int64) * 5


def _import_wire_helpers() -> tuple[Any, Any]:
    try:
        from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
            build_command_message,
            pack_pose_message,
        )
    except Exception as exc:  # noqa: BLE001 - turn optional runtime dependency into CLI error
        raise ReferenceDataError(f"cannot import the repository ZMQ wire helpers: {exc}") from exc
    return build_command_message, pack_pose_message


def describe_protocol() -> dict[str, Any]:
    return {
        "version": PROTOCOL_VERSION,
        "topic": "pose",
        "publishes_external_token": PUBLISHES_EXTERNAL_TOKEN,
        "fields": {
            "joint_pos": "f32[N,29] (IsaacLab order)",
            "joint_vel": "f32[N,29] (IsaacLab order)",
            "body_quat_w": "f32[N,4] (w,x,y,z)",
            "frame_index": "i64[N]",
            "catch_up": "bool[1]",
            "left_hand_joints": "optional f32[7]",
            "right_hand_joints": "optional f32[7]",
            "heading_increment": "optional one-shot f32[1]",
        },
        "forbidden_fields": ["token_state"],
        "wire_layout": "topic prefix + 1280-byte JSON header + concatenated little-endian fields",
    }


def _packet_indices(sequence: ReferenceSequence, current: int, count: int) -> np.ndarray:
    stop = min(sequence.num_frames, current + count)
    values = sequence.policy_seq[current:stop].astype(np.int64, copy=True)
    if values.shape[0] < count:
        start = int(values[-1]) + 1
        padding = np.arange(start, start + count - values.shape[0], dtype=np.int64)
        values = np.concatenate((values, padding))
    return values


def _window(array: np.ndarray, current: int, count: int) -> np.ndarray:
    stop = min(array.shape[0], current + count)
    result = np.asarray(array[current:stop], dtype=np.float32)
    if result.shape[0] < count:
        padding = np.repeat(result[-1:], count - result.shape[0], axis=0)
        result = np.concatenate((result, padding), axis=0)
    return np.ascontiguousarray(result, dtype=np.float32)


def build_pose_payload(
    sequence: ReferenceSequence,
    current: int,
    packet_frames: int,
    *,
    include_hands: bool,
    checkpoint_layout: str = "low_latency",
    regular_slots: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    heading_increment: float | None = None,
) -> dict[str, np.ndarray]:
    """Build one rolling, local-encoder protocol-v1 payload."""

    if checkpoint_layout == "regular":
        if packet_frames <= int(REGULAR_PACKET_SLOT_OFFSETS[-1]):
            raise ReferenceDataError(
                "regular recorded-slot replay needs packet_frames >= 46"
            )
        positions, velocities, absolute_quat = (
            reference_slot_views(sequence) if regular_slots is None else regular_slots
        )
        # C++ samples current+[0,5,...,45].  Fill all five modulo classes so
        # this packet exactly serves rows current..current+4 even if the next
        # network packet is a few milliseconds late.  Each source row retains
        # its ten recorded slots, including the source-clamped tail.
        local = np.arange(packet_frames, dtype=np.int64)
        source_rows = np.minimum(current + local % 5, sequence.num_frames - 1)
        source_slots = np.minimum(local // 5, 9)
        joint_pos = np.ascontiguousarray(
            positions[source_rows, source_slots], dtype=np.float32
        )
        joint_vel = np.ascontiguousarray(
            velocities[source_rows, source_slots], dtype=np.float32
        )
        body_quat = np.ascontiguousarray(
            absolute_quat[source_rows, source_slots], dtype=np.float32
        )
    elif checkpoint_layout != "low_latency":
        raise ReferenceDataError(f"unsupported checkpoint layout: {checkpoint_layout}")
    else:
        joint_pos = _window(sequence.joint_pos, current, packet_frames)
        joint_vel = _window(sequence.joint_vel, current, packet_frames)
        body_quat = _window(sequence.reference_anchor_quat_wxyz, current, packet_frames)

    payload: dict[str, np.ndarray] = {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_quat_w": body_quat,
        "frame_index": _packet_indices(sequence, current, packet_frames),
        # pack_pose_message recognises numpy bool explicitly.  Using uint8 here
        # would be silently cast to f32 by that shared helper, while the C++
        # catch_up decoder intentionally accepts bool/u8/i32/i64 (not f32).
        "catch_up": np.asarray([False], dtype=bool),
    }
    if include_hands:
        # ZMQEndpointInterface reads seven values from the first row of a hand
        # field.  A rank-one field makes that message-level semantics explicit.
        payload["left_hand_joints"] = np.ascontiguousarray(
            sequence.left_hand_target[current], dtype=np.float32
        )
        payload["right_hand_joints"] = np.ascontiguousarray(
            sequence.right_hand_target[current], dtype=np.float32
        )
    if heading_increment is not None:
        payload["heading_increment"] = np.asarray([heading_increment], dtype=np.float32)
    if "token_state" in payload:
        raise AssertionError("protocol-v1 publisher must never include token_state")
    return payload


def _send_repeated(socket: Any, message: bytes, repeats: int, interval_s: float) -> None:
    for index in range(repeats):
        socket.send(message)
        if index + 1 < repeats:
            time.sleep(interval_s)


def _write_publisher_status(args: argparse.Namespace, status: str) -> None:
    if not args.publisher_status_file:
        return
    path = Path(args.publisher_status_file).expanduser().resolve()
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
        f"[csv-reference] waiting for gate {path} in {sorted(expected)!r}; "
        "formal reference time is still frozen at tick 0",
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
            print(f"[csv-reference] gate ready: {last_value}", flush=True)
            return
        if time.monotonic() >= deadline:
            raise ReferenceDataError(
                f"timed out after {args.gate_timeout:g}s waiting for {path} in "
                f"{sorted(expected)!r}; last value was {last_value!r}"
            )
        time.sleep(args.gate_poll_interval)


def run_publisher(args: argparse.Namespace, sequence: ReferenceSequence) -> int:
    try:
        import zmq
    except ImportError as exc:
        raise ReferenceDataError(
            "pyzmq is required for publication; run with .venv_sim/bin/python"
        ) from exc
    build_command_message, pack_pose_message = _import_wire_helpers()

    packet_frames = min(args.chunk_size, args.lookahead)
    if packet_frames < 10:
        raise ReferenceDataError(
            f"rolling packet has only {packet_frames} frames; both checkpoints need at least 10"
        )
    if args.checkpoint_layout == "regular" and packet_frames < 46:
        raise ReferenceDataError(
            "regular step-5 encoder needs current..current+45; use --chunk-size and "
            "--lookahead values of at least 46"
        )
    regular_slots = (
        reference_slot_views(sequence) if args.checkpoint_layout == "regular" else None
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
        raise ReferenceDataError(f"cannot bind publisher to {endpoint}: {exc}") from exc

    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    print(
        f"[csv-reference] bound {endpoint}; protocol=v{PROTOCOL_VERSION}, "
        f"frames={sequence.num_frames}, packet={packet_frames}, token_state=never",
        flush=True,
    )
    try:
        time.sleep(args.subscriber_warmup)

        if not args.no_command:
            switch_message = build_command_message(start=False, stop=False, planner=False)
            _send_repeated(
                socket, switch_message, args.command_retries, args.command_retry_interval
            )
            time.sleep(args.mode_switch_delay)

        initial_payload = build_pose_payload(
            sequence,
            0,
            packet_frames,
            include_hands=not args.no_hands,
            checkpoint_layout=args.checkpoint_layout,
            regular_slots=regular_slots,
        )
        initial_message = pack_pose_message(
            initial_payload, topic=args.topic, version=PROTOCOL_VERSION
        )
        _send_repeated(
            socket, initial_message, args.initial_pose_retries, args.command_retry_interval
        )
        time.sleep(args.initial_pose_delay)

        # The deploy process spends roughly three seconds in INIT, during which
        # q_target is merely ramped to default_angles.  Advancing the 50 Hz CSV
        # clock in that state would silently skip the beginning of the task.
        # The copied simulator writes deploy_init_ready only after observing the
        # completed INIT target; keep tick zero frozen until then.
        _wait_for_gate(args, lambda: stop_requested)
        _write_publisher_status(args, "ready_for_control")

        if not args.no_command:
            start_message = build_command_message(start=True, stop=False, planner=False)
            _send_repeated(
                socket, start_message, args.command_retries, args.command_retry_interval
            )

        # CONTROL starts only after the launcher sees ready_for_control.  Start
        # the 50 Hz clock after the simulator confirms the first policy command,
        # so tick zero occupies exactly the first reference interval.
        if args.gate_status_file:
            _wait_for_gate(args, lambda: stop_requested, {"running"})
        _write_publisher_status(args, "streaming")

        # Tick zero was already installed above while deploy was in INIT.  Do
        # not publish that same frame-start again after the gate: the C++
        # StreamedMotionMerger interprets a non-increasing frame-start as a
        # catch-up/reinitialisation and would reset heading at CONTROL entry.
        # The first timed update is therefore tick one, one reference period
        # after control starts.
        first_timed_tick = 1
        period = 1.0 / args.rate
        start_clock = time.monotonic()
        simulator_to_reference = quat_multiply_wxyz(
            quat_conjugate_wxyz(sequence.base_quat_samples_wxyz[0, 0]),
            sequence.reference_anchor_quat_wxyz[0],
        )
        simulator_relative_matrix = quat_to_matrix_wxyz(simulator_to_reference)
        heading_delta = float(
            np.arctan2(simulator_relative_matrix[1, 0], simulator_relative_matrix[0, 0])
        )
        late_count = 0
        for current in range(first_timed_tick, sequence.num_frames):
            if stop_requested:
                return 130
            # P1 is installed immediately after tick0 produced its first
            # command; P2 follows one period later.  This keeps the next local
            # encoder window ready before the next 50 Hz policy tick.
            deadline = start_clock + (current - first_timed_tick) * period
            remaining = deadline - time.monotonic()
            if remaining > 0.0:
                time.sleep(remaining)
            elif remaining < -period:
                late_count += 1

            one_time_heading: float | None = None
            if args.apply_heading_correction and current == args.heading_correction_tick:
                one_time_heading = heading_delta
                print(
                    "[csv-reference] applying explicitly requested one-time "
                    f"heading_increment={heading_delta:+.8f} rad at publisher tick {current}; "
                    "verify this against encoder-token diagnostics",
                    flush=True,
                )
            payload = build_pose_payload(
                sequence,
                current,
                packet_frames,
                include_hands=not args.no_hands,
                checkpoint_layout=args.checkpoint_layout,
                regular_slots=regular_slots,
                heading_increment=one_time_heading,
            )
            message = pack_pose_message(payload, topic=args.topic, version=PROTOCOL_VERSION)
            socket.send(message)
            if current % max(1, int(args.rate)) == 0 or current + 1 == sequence.num_frames:
                print(
                    f"[csv-reference] tick={current}/{sequence.num_frames - 1} "
                    f"policy_seq={int(sequence.policy_seq[current])} "
                    f"hands={'off' if args.no_hands else 'on'}",
                    flush=True,
                )

        print(
            f"[csv-reference] source complete; late_ticks={late_count}. "
            "Leaving control active so the simulator can flush its replay CSV.",
            flush=True,
        )
        return 3 if late_count else 0
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        socket.close(0)
        context.term()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Deduplicate a MuJoCo recording by policy_seq, recover the absolute G1 "
            "reference and publish rolling protocol-v1 motion packets at 50 Hz."
        )
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Recording directory or its data.csv",
    )
    parser.add_argument("--host", default="*", help="ZMQ PUB bind host")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--topic", default="pose")
    parser.add_argument("--rate", type=float, default=50.0, help="Policy publication rate")
    parser.add_argument(
        "--start-policy-offset",
        type=int,
        default=1,
        help=(
            "Zero-based offset in the deduplicated policy sequence. The recording's "
            "offset 0 is a partial policy hold, so the first complete boundary is 1."
        ),
    )
    parser.add_argument(
        "--max-policy-frames",
        type=int,
        help="Optional number of policy frames to publish after the start offset",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="Maximum frames in every rolling pose packet",
    )
    parser.add_argument(
        "--lookahead",
        type=int,
        default=50,
        help="Requested rolling look-ahead; packet size is min(chunk-size, lookahead)",
    )
    parser.add_argument(
        "--checkpoint-layout",
        choices=("regular", "low_latency", "auto"),
        default="auto",
        help="Use regular to enforce its 46-frame step-5 look-ahead requirement",
    )
    parser.add_argument(
        "--base-sample",
        default="previous-index5",
        help=(
            "Pelvis sample used to undo relative orientation: first/last/middle/indexN, "
            "optionally previous-* (default: previous-index5, selected by regular-token validation)"
        ),
    )
    parser.add_argument("--no-hands", action="store_true", help="Do not publish hand targets")
    parser.add_argument(
        "--no-command",
        action="store_true",
        help="Do not switch ZMQ manager mode or send the start-control command",
    )
    parser.add_argument("--subscriber-warmup", type=float, default=1.0)
    parser.add_argument("--mode-switch-delay", type=float, default=0.25)
    parser.add_argument("--initial-pose-delay", type=float, default=0.20)
    parser.add_argument(
        "--gate-status-file",
        help=(
            "Optional simulator/deploy readiness file. The reference stays at tick 0 "
            "until it contains --gate-ready-value."
        ),
    )
    parser.add_argument("--gate-ready-value", default="deploy_init_ready")
    parser.add_argument(
        "--publisher-status-file",
        help="Optional launcher handshake file (ready_for_control -> streaming).",
    )
    parser.add_argument("--gate-timeout", type=float, default=30.0)
    parser.add_argument("--gate-poll-interval", type=float, default=0.001)
    parser.add_argument("--command-retries", type=int, default=3)
    # The launcher starts this publisher only after the subscriber is ready and
    # gives the PUB/SUB connection a one-second warm-up.  A pose packet must be
    # sent exactly once: StreamedMotionMerger treats a repeated frame-start as
    # catch-up and reinitialises heading, even when catch_up=false.
    parser.add_argument("--initial-pose-retries", type=int, default=1)
    parser.add_argument("--command-retry-interval", type=float, default=0.05)
    parser.add_argument(
        "--apply-heading-correction",
        action="store_true",
        help=(
            "Opt in to one heading_increment equal to the recorded start-frame relative yaw. "
            "This is off by default because its ordering versus C++ heading reset must be "
            "verified in the integrated run."
        ),
    )
    parser.add_argument(
        "--heading-correction-tick",
        type=int,
        default=1,
        help="Publisher tick for the opt-in one-time heading correction",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and diagnose only; do not open a ZMQ socket",
    )
    parser.add_argument(
        "--diagnose-orientation",
        action="store_true",
        help="Alias for --dry-run with all base-sample candidate comparisons",
    )
    parser.add_argument(
        "--diagnostics-json",
        help="Optional diagnostics JSON path (use a path under change_ckpt/data)",
    )
    parser.add_argument(
        "--prepared-output",
        help="Optional compact .npz output containing the prepared 50 Hz stream",
    )
    parser.add_argument(
        "--describe-protocol",
        action="store_true",
        help="Print the machine-readable protocol contract and exit",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.rate <= 0.0:
        raise ReferenceDataError("--rate must be positive")
    for name in ("chunk_size", "lookahead", "command_retries", "initial_pose_retries"):
        if getattr(args, name) <= 0:
            raise ReferenceDataError(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "subscriber_warmup",
        "mode_switch_delay",
        "initial_pose_delay",
        "command_retry_interval",
    ):
        if getattr(args, name) < 0.0:
            raise ReferenceDataError(f"--{name.replace('_', '-')} cannot be negative")
    for name in ("gate_timeout", "gate_poll_interval"):
        if getattr(args, name) <= 0.0:
            raise ReferenceDataError(f"--{name.replace('_', '-')} must be positive")
    if args.heading_correction_tick < 0:
        raise ReferenceDataError("--heading-correction-tick cannot be negative")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.describe_protocol:
            print(json_dump(describe_protocol()))
            return 0
        _validate_args(args)
        full_sequence = load_reference_sequence(args.input, base_sample_mode=args.base_sample)
        diagnostics = build_diagnostics(
            full_sequence,
            selected_start=args.start_policy_offset,
            include_base_candidates=args.diagnose_orientation or args.dry_run,
        )
        sequence = full_sequence.slice(
            args.start_policy_offset,
            count=args.max_policy_frames,
        )
        diagnostics["publication"] = {
            "selected_frames": sequence.num_frames,
            "first_policy_seq": int(sequence.policy_seq[0]),
            "last_policy_seq": int(sequence.policy_seq[-1]),
            "rate_hz": args.rate,
            "packet_frames": min(args.chunk_size, args.lookahead),
            "hands": not args.no_hands,
            "protocol_version": PROTOCOL_VERSION,
            "publishes_external_token": PUBLISHES_EXTERNAL_TOKEN,
            "checkpoint_layout": args.checkpoint_layout,
            "reference_window_semantics": (
                "recorded CSV slots tiled by five row residues; each row is sampled "
                "at packet-local offsets 0,5,...,45"
                if args.checkpoint_layout == "regular"
                else "consecutive deduplicated policy frames at offsets 0..9"
            ),
        }
        print(json_dump(diagnostics), flush=True)
        if args.diagnostics_json:
            output = Path(args.diagnostics_json).expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json_dump(diagnostics) + "\n", encoding="utf-8")
            print(f"[csv-reference] wrote diagnostics: {output}", flush=True)
        if args.prepared_output:
            output = write_prepared_npz(sequence, args.prepared_output, diagnostics=diagnostics)
            print(f"[csv-reference] wrote prepared stream: {output}", flush=True)
        if args.dry_run or args.diagnose_orientation:
            return 0
        return run_publisher(args, sequence)
    except (ReferenceDataError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("[csv-reference] interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
