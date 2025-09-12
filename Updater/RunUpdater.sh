#!/usr/bin/env bash
set -Eeuo pipefail

# --- config ---
CONDA_HOME="${CONDA_HOME:-/home/dieter/miniforge3}"
ENV_NAMES=("default" "default311")  # list in priority order
APP_DIR="/home/dieter/Dropbox/PythonRepos/3PiRobot/Updater"
PY_SCRIPT="script_main_gui.py"
export MAMBA_NO_BANNER=1
# --------------

# Initialize mamba for non-interactive shells
if [ -f "$CONDA_HOME/etc/profile.d/mamba.sh" ]; then
  # shellcheck source=/dev/null
  . "$CONDA_HOME/etc/profile.d/mamba.sh"
elif command -v mamba >/dev/null 2>&1; then
  # Fallback if mamba.sh isn't present
  eval "$(mamba shell hook -s bash)"
else
  echo "ERROR: mamba not found. Try: 'conda install -n base mamba'." >&2
  exit 1
fi

# Helper: activate with nounset disabled to avoid activate.d script breakage
activate_env() {
  local env_name="$1"
  set +u
  mamba activate "$env_name" >/dev/null 2>&1 || return 1
  set -u
}

# Try to activate the first available environment
ENV_FOUND=""
# Use mamba to list envs; awk grabs the first column (env names)
ENV_LIST="$(mamba env list 2>/dev/null | awk 'NF>0{print $1}')"

for env in "${ENV_NAMES[@]}"; do
  if echo "$ENV_LIST" | grep -qx "$env"; then
    if activate_env "$env"; then
      ENV_FOUND="$env"
      break
    fi
  fi
done

if [ -z "$ENV_FOUND" ]; then
  echo "ERROR: None of the specified environments found: ${ENV_NAMES[*]}" >&2
  exit 1
fi

# Optional diagnostics (comment out once happy)
echo "[debug] Activated env: $ENV_FOUND"
echo "[debug] python:  $(command -v python)"
echo "[debug] rshell:  $(command -v rshell)"

# Run the app
cd "$APP_DIR"
exec python "$PY_SCRIPT"
