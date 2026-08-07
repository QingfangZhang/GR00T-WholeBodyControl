#!/usr/bin/env bash
# Create an isolated Teleopit rollout environment and fetch pinned assets.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${TELEOPIT_VENV_DIR:-${SCRIPT_DIR}/.venv}"
ASSETS_DIR="${TELEOPIT_ASSETS_DIR:-${SCRIPT_DIR}/assets}"
LOCK_FILE="${SCRIPT_DIR}/requirements.lock"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${SCRIPT_DIR}/.uv-cache}"

offline_args=()
download_args=()
if [[ "${1:-}" == "--offline" ]]; then
    offline_args+=(--offline)
    download_args+=(--offline)
elif [[ $# -gt 0 ]]; then
    echo "usage: $0 [--offline]" >&2
    exit 2
fi

if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
elif [[ -x /opt/conda/envs/isaaclab/bin/uv ]]; then
    UV_BIN=/opt/conda/envs/isaaclab/bin/uv
else
    echo "error: uv was not found in PATH or /opt/conda/envs/isaaclab/bin/uv" >&2
    exit 1
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    "${UV_BIN}" venv "${offline_args[@]}" --python 3.10 "${VENV_DIR}"
fi

PYTHON_VERSION="$("${VENV_DIR}/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${PYTHON_VERSION}" != "3.10" ]]; then
    echo "error: ${VENV_DIR} uses Python ${PYTHON_VERSION}; Python 3.10 is required" >&2
    exit 1
fi

"${UV_BIN}" pip sync --python "${VENV_DIR}/bin/python" \
    "${offline_args[@]}" "${LOCK_FILE}"

"${VENV_DIR}/bin/python" "${SCRIPT_DIR}/download_assets.py" \
    --assets-dir "${ASSETS_DIR}" "${download_args[@]}"

echo
echo "Teleopit rollout environment is ready."
echo "Python: ${VENV_DIR}/bin/python"
echo "Assets: ${ASSETS_DIR}/manifest.json"
