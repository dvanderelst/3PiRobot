#!/bin/bash

# Navigate to the project directory (if not already there)
cd "$(dirname "$0")"

# Check if the .venv virtual environment exists
if [ -d ".venv" ]; then
    echo "Activating .venv virtual environment..."
    source .venv/bin/activate
else
    echo "Error: .venv virtual environment not found in the current directory."
    exit 1
fi

# Check if 'vibe' command is available
if command -v vibe &> /dev/null; then
    echo "Starting Mistral Vibe with project-local configurations..."
    # Set VIBE_HOME to a project-local directory
    export VIBE_HOME="$(pwd)/.vibe"
    # Create .vibe directory structure if it doesn't exist
    mkdir -p "$VIBE_HOME/prompts" "$VIBE_HOME/agents"
    vibe
else
    echo "Error: 'vibe' command not found. Make sure Mistral Vibe is installed in this virtual environment."
    exit 1
fi
