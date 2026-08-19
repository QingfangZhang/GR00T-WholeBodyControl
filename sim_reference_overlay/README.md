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

## Staged real-robot reference

`build_real_safe_reference.py` creates a guarded entry/exit wrapper without
modifying the source NPZ or any official SONIC file. For the first staged
`dun` experiment, run from the repository root:

```bash
.venv_sim/bin/python sim_reference_overlay/build_real_safe_reference.py \
  sim_reference_overlay/data/npz/dun.npz \
  --output-root sim_reference_overlay/data/csv_dun_real_safe \
  --name dun_real_safe_stage1 \
  --assume-isaaclab-order
```

The command refuses to overwrite an existing output directory. The default
stage-1 interval is source frame 12 through 1158, inclusive. It adds:

```text
5 s policy-default hold
5 s minimum-jerk entry
1 s source-start settle
source frames 13..1158
0.5 s smooth brake
1 s stopped hold
6 s minimum-jerk return to policy-default joints
5 s final hold
```

The output is intentionally named `dun_real_safe_stage1`, not complete `dun`:
it retains 74.5 percent of the usable source frames. The excluded final quarter
contains another high-dynamic return sequence. More importantly, the retained
source still reaches about `12.97 rad/s` lower-body joint speed and a pelvis
height of about `0.523 m`. "Safe" here means that the entry, exit, and held
frame 0 were constructed and checked; it is not a safety certification for the
source motion.

After stage 1 has passed the required suspended tests, generate the fourth-stage
near-full candidate with:

```bash
.venv_sim/bin/python sim_reference_overlay/build_real_safe_reference.py \
  sim_reference_overlay/data/npz/dun.npz \
  --output-root sim_reference_overlay/data/csv_dun_real_near_full \
  --name dun_real_near_full \
  --source-start-frame 12 \
  --source-end-frame 1525 \
  --brake-joint-speed-limit 1.5 \
  --brake-non-arm-speed-limit 0.7 \
  --assume-isaaclab-order
```

This keeps source frames 12 through 1525 (98.376 percent from the selected
start to source end) and excludes the final 25 source frames. Frame 1525 was
chosen because both feet are nearly stationary there. Its remaining motion is
primarily in the arms, so the 0.5 s C2 brake is allowed up to `1.5 rad/s` for
all joints while a separate `0.7 rad/s` guard remains enforced for every waist
and leg joint. The generated brake measures about `1.337 rad/s` overall and
`0.265 rad/s` for non-arm joints. This explicit exception does not apply to the
default stage-1 command.

The near-full return-to-default fit can move an individual foot marker by up to
about `8.57 cm`. Inspect that return in closed-loop MuJoCo and repeat the
suspended progression before considering supported playback. Near-full is
still a staged test artifact, not a safety certification of the source motion.

Before any real playback, complete the closed-loop MuJoCo test. The next
real-robot stage should enter CONTROL while paused at frame 0 and must not press
`T`. Only after that suspended test passes should staged playback be considered.

## Reconstruct full NPZ files

Deployment CSV stores only 14 selected body signals, while the source
`dun.npz` schema contains all 30 G1 bodies. The reverse converter uses the same
MuJoCo scene to reconstruct the missing 16 body poses and velocities, then
copies the original 14 CSV body signals back exactly:

```bash
.venv_sim/bin/python sim_reference_overlay/convert_csv_motions_to_npz.py \
  sim_reference_overlay/data/csv_dun_real_safe \
  sim_reference_overlay/data/npz/dun_real_safe_stage1.npz

.venv_sim/bin/python sim_reference_overlay/convert_csv_motions_to_npz.py \
  sim_reference_overlay/data/csv_dun_real_near_full \
  sim_reference_overlay/data/npz/dun_real_near_full.npz
```

The converter refuses to overwrite an existing file. Each output has the same
nine keys, canonical 29-joint/30-body name arrays, float32 signal dtypes, and
uncompressed NPZ layout as `dun.npz`. It verifies that loading the result back
through the existing NPZ-to-CSV path reproduces all six source CSV arrays
exactly at float32 precision.

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
