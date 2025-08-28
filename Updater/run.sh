#!/usr/bin/env bash
set -euo pipefail
APP="script_main_gui.py"
# Use the Python from your default311 env
PY="$HOME/miniforge3/envs/default311/bin/python"

exec "$PY" "$APP"

