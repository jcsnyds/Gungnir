#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# Use the already-active environment's interpreter explicitly when possible.
if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    exec "${VIRTUAL_ENV}/bin/python" setup.py "$@"
fi
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    exec "${CONDA_PREFIX}/bin/python" setup.py "$@"
fi

# In an activated venv, `python` is normally the venv interpreter; checking it
# before python3 avoids accidentally selecting a system interpreter.
if command -v python >/dev/null 2>&1; then
    exec python setup.py "$@"
elif command -v python3 >/dev/null 2>&1; then
    exec python3 setup.py "$@"
else
    echo "Python 3 was not found on PATH." >&2
    echo "Activate your existing virtual environment first, then run setup.sh again." >&2
    exit 1
fi
