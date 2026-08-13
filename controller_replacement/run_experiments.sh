#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 RECORDING [RECORDING ...]" >&2
  exit 2
fi

PYTHON="${PYTHON:-Teleopit_rollout/.venv/bin/python}"
REFERENCE_MODE="${REFERENCE_MODE:-reference_motion}"
ROOT_ASSIST="${ROOT_ASSIST:-none}"

for recording in "$@"; do
  for controller in regular low_latency sonic_v1_1 teleopit; do
    if ! "$PYTHON" controller_replacement/launch_rollout.py \
        "$recording" \
        --controller "$controller" \
        --reference-mode "$REFERENCE_MODE" \
        --root-assist "$ROOT_ASSIST"; then
      echo "FAILED: recording=$recording controller=$controller" >&2
      status=1
    fi
  done
done

exit "${status:-0}"
