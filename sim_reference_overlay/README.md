# SONIC reference overlay for MuJoCo

This custom launcher adds a transparent reference-motion robot to the official
SONIC physics viewer without modifying any existing `gear_sonic` or
`gear_sonic_deploy` source file.

The solid robot remains the real MuJoCo body controlled through the normal
closed-loop path:

```text
SONIC policy -> DDS LowCmd -> PD torque -> MuJoCo physics
```

The transparent green robot is visual-only. It subscribes to the C++ deploy
program's existing `g1_debug` output (`tcp://localhost:5557`), runs forward
kinematics in a separate `MjData`, and copies only visual geoms into the main
viewer. It cannot create contacts or affect control.

## Run

Terminal 1, from the repository root:

```bash
.venv_sim/bin/python sim_reference_overlay/run_sim_loop.py \
  --ghost-alpha 0.3
```

This uses the first-frame-aligned global reference trajectory by default. To
lock the ghost translation to the physical robot on every frame instead, add
`--ghost-root-mode actual` to the same command.

Terminal 2 runs the normal `g1_deploy_onnx_ref` command. Its output type already
defaults to ZMQ port `5557` and topic `g1_debug`; no extra flags are required.
The ghost appears after the deployment process starts publishing valid target
frames. Keyboard focus matters: type `]`, `T`, `R`, `N`, `P`, and `O` in
**Terminal 2**, where `g1_deploy_onnx_ref` is running. Only press `9` in the
MuJoCo window to release the elastic band. The MuJoCo window and Terminal 1 do
not forward `T` to the deployment process.

After a motion reaches its last frame, the deployment process automatically
pauses and resets it to frame 0. Focus Terminal 2 and press `T` to replay it;
no `R` is required. A successful key press prints `Playing motion ... from
frame 0` in Terminal 2. If `--input-type manager` has been switched away from
keyboard input, press `!` in Terminal 2 to select keyboard again, then press
`R` and `T`.

No third terminal or second MuJoCo window is required. This implementation does
not use `visualize_motion.py` or `lxml`.

## Root modes

The modes are mutually exclusive and selected at startup:

- `--ghost-root-mode actual` aligns only the ghost XYZ translation to the
  physical MuJoCo robot. Reference root orientation and all 29 reference joint
  angles are preserved, making silhouette/posture errors easy to see.
- `--ghost-root-mode reference` aligns the first reference frame's XYZ to the
  physical MuJoCo root, then preserves the reference's relative XYZ trajectory:

  ```text
  ghost_xyz(t) = actual_xyz(t0) + target_xyz(t) - target_xyz(t0)
  ```

  Reference orientation and joints are unchanged. While playback is paused at
  frame 0, the anchor follows the physical root; it freezes when the motion
  leaves frame 0. This removes arbitrary dataset/world-origin offsets without
  discarding the reference displacement trajectory.

  The official `g1_debug` message does not contain the motion frame, play state,
  or motion name. For the single-motion `dance` and `dun` datasets, replay is
  detected when three consecutive new controller messages return to the unique
  first-frame joint signature; the root is then re-anchored. A restarted C++
  controller is also detected when its monotonic message index rolls backward.
  Start this overlay before pressing `T`; if it joins mid-motion, its first seen
  target necessarily becomes the initial anchor.

The default is `reference`.

## Other options

```text
--ghost-root-mode {actual,reference} root handling, default reference
--ghost-alpha FLOAT             opacity in (0, 1], default 0.30
--ghost-zmq-host HOST           default localhost
--ghost-zmq-port PORT           default 5557
--ghost-zmq-topic TOPIC         default g1_debug
--ghost-stale-timeout SECONDS   hide stale targets, default 1.0
```

If the deploy command uses a custom output port or topic, pass matching values
to this launcher. If the C++ program is stopped, the ghost is hidden after the
stale timeout while the physics simulator continues running. In `reference`
mode, the first valid target after that timeout establishes a fresh XYZ anchor.

## Test

The tests load the real G1 MuJoCo scene but do not open a GUI or initialize DDS:

```bash
.venv_sim/bin/python sim_reference_overlay/test_ghost_overlay.py -v
```
