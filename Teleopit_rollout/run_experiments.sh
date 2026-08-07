#!/usr/bin/env bash
# Run the agreed Teleopit qpos-track matrix without stopping on task failures.

set -uo pipefail

trap 'echo "experiment matrix interrupted" >&2; exit 130' INT TERM

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${SCRIPT_DIR}/.venv/bin/python"
LAUNCHER="${SCRIPT_DIR}/launch_teleopit_rollout.py"
DATA_ROOT="${REPO_ROOT}/sample_data/ztj/20260612"

if [[ ! -x "${PYTHON}" ]]; then
    echo "error: run ${SCRIPT_DIR}/setup_env.sh first" >&2
    exit 2
fi

extra_args=()
if [[ "${1:-}" == "--overwrite" ]]; then
    extra_args+=(--overwrite)
elif [[ $# -gt 0 ]]; then
    echo "usage: $0 [--overwrite]" >&2
    exit 2
fi

unexpected=0
run_one() {
    local recording="$1"
    local assist="$2"
    echo
    echo "== ${recording} / root-assist ${assist} =="
    "${PYTHON}" "${LAUNCHER}" \
        "${DATA_ROOT}/${recording}" \
        --root-assist "${assist}" \
        --post-rollout-seconds 1.0 \
        "${extra_args[@]}"
    local status=$?
    # Exit 4 is a valid observed fall.  Exit 5 is still collected so later
    # conditions run, but makes the matrix return non-zero because a non-finite
    # rollout cannot be treated as a usable experimental result.
    if [[ ${status} -eq 130 ]]; then
        exit 130
    elif [[ ${status} -eq 5 ]]; then
        echo "non-finite rollout; matrix will finish with failure" >&2
        unexpected=1
    elif [[ ${status} -ne 0 && ${status} -ne 4 ]]; then
        echo "unexpected launcher exit ${status}" >&2
        unexpected=1
    fi
}

# run_one 20260612_144127_g1_sim none
# run_one 20260612_144127_g1_sim xy
# run_one 20260612_144127_g1_sim xyz
run_one 20260612_144104_g1_sim none
run_one 20260612_144104_g1_sim xy

run_one 20260612_144154_g1_sim none
run_one 20260612_144154_g1_sim xy

run_one 20260612_144214_g1_sim none
run_one 20260612_144214_g1_sim xy

run_one 20260612_144714_g1_sim none
run_one 20260612_144714_g1_sim xy

run_one 20260612_144732_g1_sim none
run_one 20260612_144732_g1_sim xy

# run_one 20260720_144342_g1_sim none
# run_one 20260720_144342_g1_sim xy
# run_one 20260722_145020_g1_sim none
# run_one 20260722_145020_g1_sim xy

# run_one 20260722_150051_g1_sim none
# run_one 20260722_150051_g1_sim xy

# run_one 20260722_153551_g1_sim none
# run_one 20260722_153551_g1_sim xy

# run_one 20260722_154958_g1_sim none
# run_one 20260722_154958_g1_sim xy

# run_one 20260722_160121_g1_sim none
# run_one 20260722_160121_g1_sim xy

exit "${unexpected}"
