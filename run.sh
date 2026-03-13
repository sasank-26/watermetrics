#!/bin/zsh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
MAIN_FILE="$SCRIPT_DIR/main.py"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Error: Python interpreter not found at $PYTHON_BIN"
  echo "Create the virtual environment first, then install dependencies."
  exit 1
fi

exec "$PYTHON_BIN" "$MAIN_FILE" "$@"
