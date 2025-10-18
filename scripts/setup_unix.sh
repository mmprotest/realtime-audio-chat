#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON:-python3}
VENV_DIR=.venv

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
# Torch may require specific wheels; pip will pick the right build for your platform.
pip install -r requirements.txt
pytest -q
