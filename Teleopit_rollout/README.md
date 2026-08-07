# Teleopit controller-replacement rollout

This directory runs the pinned Teleopit `track_g1` controller on the same
recorded MuJoCo task scenes and the same robot-qpos references used by the
SONIC qpos-track experiment.  It does not modify `gear_sonic`,
`gear_sonic_deploy`, `change_ckpt`, or `change_ckpt_track`.

## What is being compared

The body reference is constructed from the first valid row of each recorded
`policy_seq` group:

```text
recorded root xyz + root quaternion wxyz + recorded 29-DoF body qpos
                                  │
                                  ▼ 50 Hz
Teleopit official 167-D observation + 10-frame history (including current)
                                  │
                                  ▼ track_g1.onnx
29-D raw action → Teleopit q target → 200 Hz PD → original task MuJoCo scene
```

The 29 joints are reordered by name into Teleopit's canonical order.  The
recorded qvel is retained only as a diagnostic; Teleopit reference joint and
torso velocities are recomputed from adjacent 50 Hz reference poses, as in
Teleopit v0.5.0.

Although `qpos36` stores root xyz, the 167-D tracker observation does not carry
an absolute reference x/y target.  Root translation contributes through the
reference torso velocity, while reference torso height, orientation, and joint
state are represented explicitly.  The tracker can therefore accumulate
world-frame x/y drift even when the qpos reference itself contains x/y.

Hands are not produced by `track_g1.onnx`.  The source recording's
`left_hand_q[0:7]` and `right_hand_q[0:7]` targets are restored through the
separate Dex3-style PD path matched to SONIC's target/gain/limit semantics.
The entire initial MuJoCo qpos/qvel is read from the recording, so drawers,
objects, the trash pedal, and the trash lid begin at the recorded task state.
Non-robot task actuators remain zero.

The fixed clocks are:

| Component | Rate | Meaning |
|---|---:|---|
| MuJoCo physics | 2000 Hz | original XML timestep `0.0005 s` |
| body/hand PD | 200 Hz | one update per two log samples |
| Teleopit ONNX | 50 Hz | one action held for four PD updates |
| replay CSV | 400 Hz | one original source row per output sample |

There is no DDS/ZMQ or wall-clock reference publisher.  A slow viewer or CPU
therefore makes the run take longer in wall time but cannot advance the
reference ahead of MuJoCo.

During the default one-second post-rollout hold, the final reference pose and
source row are repeated, their commanded reference velocities become zero,
and Teleopit continues to infer at 50 Hz from the evolving robot state/history.
The final neural action is not simply frozen.

Teleopit's standalone sim uses its hand-free base XML and MuJoCo built-in PD
with a 5 ms model step.  This adapter instead keeps the recorded task XML,
Dex3 hand dynamics, contact solver, 0.5 ms physics step, and explicit 200 Hz
PD.  The result should therefore be described as deploying Teleopit's native
tracker observation/history/action mapping in the recorded SONIC task
environment, not as a bitwise reproduction of Teleopit's standalone simulator.

The SONIC and Teleopit adapters also have different native temporal inputs:
SONIC regular receives the canonical future slots `[0,5,10,...,45]` (up to
0.9 s ahead), whereas Teleopit receives the current reference state plus a
ten-frame observation history that includes the current observation (the
first observation is repeated ten times at initialization).  This experiment
compares complete controller stacks fed from the same qpos pose source; it is
not merely an ONNX checkpoint swap with identical information.

### Interpreting root assist

The user's prior SONIC regular qpos-track run also fails without root assist.
Consequently, `none` is retained to compare native drift, but it is not the
primary task-success comparison.  A task comparison must use the same assist
mode on both controllers:

- `xy`: after each 200 Hz PD interval, overwrite root world x/y and matching
  linear velocity from the same-time original recording;
- `xyz`: also overwrite root z and z velocity;
- neither mode overwrites root orientation or angular velocity.

Assisted success means *controller replacement under a common external root
stabilizer*.  It is not evidence that the tracker independently recovers the
recorded root trajectory.  In particular, `xyz` directly overwrites vertical
position/velocity and therefore changes foot-contact dynamics; treat it as an
oracle-assisted contact diagnostic, not a native-controller success.

For the drawer task, SONIC regular and Teleopit should be compared with common
`xy` assist.  For the trash task, the user's replay inspection found that the
existing SONIC `xy` run misses the bin while SONIC `xyz` reaches/contacts it in
the user's replay inspection, so the primary contact comparison is common
`xyz`; `none` and `xy` remain ablations.  Whether a foot actually steps onto
the bin and remains stable must be judged from replay/contact metrics.
Pedal/lid qpos alone is not a valid binary label.
Do not compare Teleopit qpos-track directly with `change_ckpt`: that pipeline
uses a different reference construction.

## One-time setup

The setup creates only `Teleopit_rollout/.venv`; it does not touch
`.venv_sim`, `.venv_replay`, or IsaacLab.  It installs fixed versions of NumPy,
MuJoCo, ONNX Runtime, and all transitive dependencies from
`requirements.lock`, downloads Teleopit v0.5.0 assets, and verifies their
SHA-256 hashes.

```bash
bash Teleopit_rollout/setup_env.sh
```

Pinned provenance:

- Teleopit tag `v0.5.0`, source commit
  `f9263865c581802ad531854b8e547e2403a945f3`;
- model repository tag `v0.5.0`, commit
  `94cf996444fea6894b87c28e86606cd4c2f1408f`;
- `track_g1.onnx` SHA-256
  `1ebd341d9193e1c49a986450f6043ba1a9473ad46636ce0bcb1c7755c856e0de`;
- robot asset archive SHA-256
  `fb5f1aeec3c57be6b26533c9a9aad0d095048a5fe8aa3c6915d8f6a67c35d4fc`;
- `g1_29dof.xml` SHA-256
  `512ccfe8b811e2bfaec0c2fc57960941371e189b365db5c1ff874474166800a3`.

If the verified downloads already exist, use `--offline`:

```bash
bash Teleopit_rollout/setup_env.sh --offline
```

## Validation and execution

Validate the recording, task model, exact 167-D observation, ONNX signature,
history, and 29-D output without writing a run:

```bash
Teleopit_rollout/.venv/bin/python \
  Teleopit_rollout/launch_teleopit_rollout.py \
  sample_data/ztj/20260612/20260612_144127_g1_sim \
  --policy-count 3 --root-assist xy --validate-only
```

Short, writable smoke test:

```bash
Teleopit_rollout/.venv/bin/python \
  Teleopit_rollout/launch_teleopit_rollout.py \
  sample_data/ztj/20260612/20260612_144127_g1_sim \
  --policy-count 10 --post-rollout-seconds 0 \
  --root-assist xy --run-name smoke_trash_10f
```

Run the agreed full matrix in sequence.  The script deliberately continues if
a rollout reports a fall, so one experimental outcome does not stop later
runs:

```bash
bash Teleopit_rollout/run_experiments.sh
```

Repeat and replace the same deterministic run directories:

```bash
bash Teleopit_rollout/run_experiments.sh --overwrite
```

One full pinned run can also be launched directly.  Its default output name is
derived from the recording and assist mode:

```bash
Teleopit_rollout/.venv/bin/python \
  Teleopit_rollout/launch_teleopit_rollout.py \
  sample_data/ztj/20260612/20260720_144342_g1_sim \
  --root-assist xy --overwrite
```

For a custom checkpoint/XML, a sliced run, or changed rollout duration, the
launcher requires an explicit `--run-name`; this prevents a smoke test from
overwriting a formal result with the same recording basename.

For a live solid-robot view, add `--viewer` and an explicit viewer-only name,
for example `--run-name drawer_xy_viewer`.  Closing the viewer early returns
exit code 6 and deliberately omits `run_complete.json`.  Headless generation
followed by replay is recommended here because CPU ONNX inference plus the
heavy contact scene runs below real time; simulation/reference synchronization
remains deterministic either way.

## Output and replay

Each run is written below `Teleopit_rollout/data/<run-name>/`:

```text
data.csv                 400 Hz replay-compatible complete qpos/qvel
model_snapshot/          task XML plus links to external referenced assets
prepared_reference.npz   exact 50 Hz qpos reference and source-row mapping
teleopit_policy.npz      obs[167], history[10,167], raw action[29], q target[29]
run_metadata.json        clocks, root-assist disclosure, tracking/task metrics
launch_manifest.json     checkpoint/XML hashes, command and environment
run_complete.json        completion marker plus output artifact SHA-256 hashes
```

The wide CSV keeps the source schema for compatibility.  Its old
`reference_motion` columns are provenance, not Teleopit input; exact Teleopit
inputs and outputs are in `teleopit_policy.npz`.

Replay with the original same-time qpos as the default cyan/amber ghost:

```bash
.venv_replay/bin/python change_ckpt_track/replay_mujoco_compare.py \
  Teleopit_rollout/data/20260720_144342_g1_sim_teleopit_root_assist_xy
```

Use the constructed 50 Hz sample-and-hold qpos reference as the ghost instead:

```bash
.venv_replay/bin/python change_ckpt_track/replay_mujoco_compare.py \
  Teleopit_rollout/data/20260720_144342_g1_sim_teleopit_root_assist_xy \
  --ghost-mode reference
```

In the viewer, the textured robot is Teleopit rollout qpos, cyan is the
recorded body qpos, and amber is recorded hand qpos.  `--root-mode actual`
attaches the ghost root to the rollout root when the goal is to inspect only
articulation error.

## Results generated in this workspace

Five formal directories have been generated with a one-second final hold:

The policy-rate tracking means below include both the recording phase and all
50 ticks of that final hold; `teleopit_policy.npz` marks the hold ticks in
`reference_is_hold` so motion-phase-only statistics can be computed separately.

| Recording | Assist | Main numerical outcome |
|---|---|---|
| trash `20260612_144127` | none | policy-rate root-xy mean error `0.1147 m`; pedal/lid source motion not reproduced |
| trash `20260612_144127` | xy | policy-rate root-xy mean/max error `0.00011/0.00181 m`; pedal/lid source motion not reproduced |
| trash `20260612_144127` | xyz | policy-rate root-z mean/max error `0.00002/0.00040 m`; pedal/lid source motion not reproduced |
| drawer `20260720_144342` | none | object final-position error to source `0.797 m` |
| drawer `20260720_144342` | xy | object final-position error `0.055 m`; drawer-0 final `-0.0002 rad` versus source `-0.0128 rad` |

For drawer + `xy`, the object's maximum height is `0.839 m` versus the source
`0.860 m`, and its world-y range is `0.757 m` versus `0.746 m`.  These are
coarse kinematic signals consistent with task interaction; they do not prove a
contact sequence or semantic success.  Use the replay video to verify grasp
quality, placement, and drawer closure.  For trash, the unchanged pedal/lid
shows that their recorded joint motion was not reproduced, but replay or
foot-bin contact/stability metrics are still required to decide whether the
robot stepped onto the bin.

The closest currently available SONIC qpos-track folders under
`change_ckpt_track` support the user's root-assist observation:

| Drawer controller | Assist | Final object-position error to source | Final drawer-0 qpos |
|---|---|---:|---:|
| SONIC regular qpos-track | none | `0.809 m` | `-0.0005 rad` |
| Teleopit qpos-track | none | `0.797 m` | `-0.0399 rad` |
| SONIC regular qpos-track | xy | `0.061 m` | `-0.0032 rad` |
| Teleopit qpos-track | xy | `0.055 m` | `-0.0002 rad` |

The original final drawer-0 value is `-0.0128 rad`.  Thus the numerical task
state comparison is encouraging under a common `xy` stabilizer, while both
unassisted trackers miss the object placement badly.  This comparison still
needs the two replay videos before assigning a binary task-success label.

This is not yet a strictly matched controller-only table: the existing SONIC
folders used 1000 Hz physics, while these Teleopit runs use the recorded XML's
2000 Hz physics.  SONIC also runs through its DDS/ZMQ deployment process,
while Teleopit is a deterministic single-process adapter.  The 200 Hz versus
400 Hz CSV difference changes sampling resolution, not the dynamics by itself.
A publication-quality paired comparison should match physics/control/assist
clocks, log at 400 Hz or resample both outputs to common timestamps, and report
the runtime/input-horizon differences explicitly.

Run the regression tests with:

```bash
Teleopit_rollout/.venv/bin/python -m unittest \
  Teleopit_rollout.test_adapter -v
```

The tests lock the two recording boundaries, name-based joint mapping, first
167-D observation, ONNX action transform, and ten-frame history behavior.
