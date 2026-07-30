#!/bin/bash



source .venv_sim/bin/activate

# EXPNAME="20260720_144342_g1_sim"
EXPNAME="20260612_144127_g1_sim"

# 思路一
python change_ckpt/launch_checkpoint_rollout.py run \
  --checkpoint regular \
  --recording sample_data/ztj/20260612/"$EXPNAME" \
  --root-assist none \
  --no-viewer

python change_ckpt/launch_checkpoint_rollout.py run \
  --checkpoint low_latency \
  --recording sample_data/ztj/20260612/"$EXPNAME" \
  --root-assist none \
  --no-viewer

python change_ckpt/launch_checkpoint_rollout.py run \
  --checkpoint regular \
  --recording sample_data/ztj/20260612/"$EXPNAME" \
  --root-assist xyz \
  --no-viewer

python change_ckpt/launch_checkpoint_rollout.py run \
  --checkpoint low_latency \
  --recording sample_data/ztj/20260612/"$EXPNAME" \
  --root-assist xyz \
  --no-viewer

python change_ckpt/launch_checkpoint_rollout.py run \
  --checkpoint regular \
  --recording sample_data/ztj/20260612/"$EXPNAME" \
  --root-assist xy \
  --no-viewer

python change_ckpt/launch_checkpoint_rollout.py run \
  --checkpoint low_latency \
  --recording sample_data/ztj/20260612/"$EXPNAME" \
  --root-assist xy \
  --no-viewer

# 思路二
python change_ckpt_track/launch_checkpoint_rollout.py run \
  --checkpoint regular \
  --recording sample_data/ztj/20260612/"$EXPNAME" \
  --regular-future-window canonical \
  --root-assist none \
  --no-viewer

python change_ckpt_track/launch_checkpoint_rollout.py run \
  --checkpoint low_latency \
  --recording sample_data/ztj/20260612/"$EXPNAME" \
  --regular-future-window canonical \
  --root-assist none \
  --no-viewer

python change_ckpt_track/launch_checkpoint_rollout.py run \
  --checkpoint regular \
  --recording sample_data/ztj/20260612/"$EXPNAME" \
  --regular-future-window canonical \
  --root-assist xyz \
  --no-viewer

python change_ckpt_track/launch_checkpoint_rollout.py run \
  --checkpoint low_latency \
  --recording sample_data/ztj/20260612/"$EXPNAME" \
  --regular-future-window canonical \
  --root-assist xyz \
  --no-viewer

python change_ckpt_track/launch_checkpoint_rollout.py run \
  --checkpoint regular \
  --recording sample_data/ztj/20260612/"$EXPNAME" \
  --regular-future-window canonical \
  --root-assist xy \
  --no-viewer

python change_ckpt_track/launch_checkpoint_rollout.py run \
  --checkpoint low_latency \
  --recording sample_data/ztj/20260612/"$EXPNAME" \
  --regular-future-window canonical \
  --root-assist xy \
  --no-viewer
