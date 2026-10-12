#!/bin/sh
set -eu
SQLTENSOR_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if ! command -v python3 >/dev/null 2>&1; then
    echo 'Python 3.10+ is required. On Debian/Ubuntu: sudo apt-get install python3 python3-venv' >&2
    exit 1
fi
exec python3 "$SQLTENSOR_DIR/install.py" "$@"
