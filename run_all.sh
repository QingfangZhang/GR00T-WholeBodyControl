#!/bin/bash



source .venv_sim/bin/activate

# Both launchers default to source-history prefill. Rebuild the isolated
# wrappers once before the batch so a source/hash change cannot stop run 1.
python change_ckpt/build_source_history_deploy.py
python change_ckpt_track/build_source_history_deploy.py

# EXPNAME="20260720_144342_g1_sim"
# EXPNAME="20260612_144127_g1_sim"
# EXPNAME="20260612_144154_g1_sim"
# EXPNAME="20260612_144214_g1_sim"
EXPNAME="20260722_154958_g1_sim"

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
