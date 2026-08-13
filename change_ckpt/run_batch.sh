#!/usr/bin/env bash
set -e

# Source-history prefill is the launcher default; keep its isolated wrapper in
# sync with the current deployment sources before starting a long batch.
.venv_sim/bin/python change_ckpt/build_source_history_deploy.py

recordings=(
  20260612_144154_g1_sim
  20260612_144214_g1_sim
  20260722_145020_g1_sim
  20260722_160121_g1_sim
)

for name in "${recordings[@]}"; do
  .venv_sim/bin/python change_ckpt/launch_checkpoint_rollout.py run \
    --checkpoint sonic_v1_1 \
    --recording "sample_data/ztj/20260612/${name}" \
    --root-assist none \
    --no-viewer

  .venv_sim/bin/python change_ckpt/launch_checkpoint_rollout.py run \
    --checkpoint sonic_v1_1 \
    --recording "sample_data/ztj/20260612/${name}" \
    --root-assist xy \
    --no-viewer
done
