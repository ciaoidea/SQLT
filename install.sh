#!/bin/sh
# Linux bootstrap. Keep application setup unprivileged; elevate only apt/systemd.
set -eu
cd "$(dirname "$0")"
if ! command -v python3 >/dev/null 2>&1; then
    if ! command -v apt-get >/dev/null 2>&1; then
        echo 'Install Python 3.11 or newer, then run this installer again.' >&2
        exit 1
    fi
    if [ "$(id -u)" -eq 0 ]; then
        apt-get update
        apt-get install -y python3 python3-venv ca-certificates
    else
        sudo apt-get update
        sudo apt-get install -y python3 python3-venv ca-certificates
    fi
fi
exec python3 installer.py "$@"
