#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if command -v python3 >/dev/null 2>&1; then
    exec python3 setup.py "$@"
elif command -v python >/dev/null 2>&1; then
    exec python setup.py "$@"
else
    echo "Python 3 was not found on PATH." >&2
    echo "Activate your existing virtual environment first, then run setup.sh again." >&2
    exit 1
fi
