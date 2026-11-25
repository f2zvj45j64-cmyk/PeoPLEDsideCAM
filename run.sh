#!/usr/bin/env bash
set -e

# Always run this copy of the application with its isolated environment.
APP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$APP_DIR/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "Missing project environment: $PYTHON" >&2
    echo "Create .venv and install the project dependencies first." >&2
    exit 1
fi

cd "$APP_DIR"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
# Local-only retention policy; no Network_Backup writes are performed.
export LOCAL_STORAGE_MAX_GB="${LOCAL_STORAGE_MAX_GB:-60}"
export LOCAL_STORAGE_DELETE_GB="${LOCAL_STORAGE_DELETE_GB:-30}"
exec "$PYTHON" JetsonDualCamPy.py "$@"
