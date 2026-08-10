#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEBUI_PATH="$ROOT_DIR/webUI.py"

if [ ! -f "$WEBUI_PATH" ]; then
    echo "Cannot find WebUI entry point: $WEBUI_PATH" >&2
    exit 1
fi

PYTHON="$ROOT_DIR/python_env/bin/python"
if [ ! -x "$PYTHON" ]; then
    PYTHON=""
    for candidate in python3 python; do
        if command -v "$candidate" >/dev/null 2>&1; then
            PYTHON="$candidate"
            break
        fi
    done
fi

if [ -z "$PYTHON" ]; then
    echo "Cannot find a Python interpreter. Install Python 3.9-3.10 or provide python_env/bin/python." >&2
    exit 1
fi

cd "$ROOT_DIR"
echo "Starting So-VITS-SVC WebUI..."
exec "$PYTHON" "$WEBUI_PATH" "$@"
