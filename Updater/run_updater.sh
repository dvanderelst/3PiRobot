#!/usr/bin/env bash
set -Eeuo pipefail

# --- config ---
CONDA_HOME="/home/dieter/miniforge3"
ENV_NAME="default311"
APP_DIR="/home/dieter/Dropbox/PythonRepos/3PiRobot/Updater"
PY_SCRIPT="script_main_gui.py"
# --------------

# Initialize conda for non-interactive shells
if [ -f "$CONDA_HOME/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  . "$CONDA_HOME/etc/profile.d/conda.sh"
else
  echo "ERROR: conda.sh not found at $CONDA_HOME/etc/profile.d/conda.sh" >&2
  exit 1
fi

# Activate the environment (adds .../envs/default311/bin to PATH)
conda activate "$ENV_NAME"

# Optional diagnostics (comment out once happy)
echo "[debug] python:  $(command -v python)"
echo "[debug] rshell:  $(command -v rshell)"

# Run the app
cd "$APP_DIR"
exec python "$PY_SCRIPT"
