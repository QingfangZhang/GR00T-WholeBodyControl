#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 RECORDING [RECORDING ...]" >&2
  exit 2
fi

PYTHON="${PYTHON:-$REPO_ROOT/Teleopit_rollout/.venv/bin/python}"
REFERENCE_MODE="${REFERENCE_MODE:-reference_motion}"
ROOT_ASSIST="${ROOT_ASSIST:-none}"
HAND_TORQUE_PROFILE="${HAND_TORQUE_PROFILE:-sonic_release}"
FALL_HEIGHT_M="${FALL_HEIGHT_M:-0.2}"

for recording in "$@"; do
  for controller in regular low_latency sonic_v1_1 teleopit; do
    if ! "$PYTHON" "$SCRIPT_DIR/launch_rollout.py" \
        "$recording" \
        --controller "$controller" \
        --reference-mode "$REFERENCE_MODE" \
        --root-assist "$ROOT_ASSIST" \
        --hand-torque-profile "$HAND_TORQUE_PROFILE" \
        --fall-height-m "$FALL_HEIGHT_M"; then
      echo "FAILED: recording=$recording controller=$controller" >&2
      status=1
    fi
  done
done

exit "${status:-0}"
