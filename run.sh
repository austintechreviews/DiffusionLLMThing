#!/bin/bash
#
# Run script for Discrete Diffusion Language Model
# Activates venv and runs commands
#
# Usage:
#   ./run.sh python scripts/train.py --config configs/default.yaml
#   ./run.sh pytest tests/ -v
#   ./run.sh python -c "import diffusionllm; print(diffusionllm.__version__)"
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

# Check if venv exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Error: Virtual environment not found at $VENV_DIR"
    echo "Run './setup.sh' first to create it."
    exit 1
fi

# Activate venv and run command
source "$VENV_DIR/bin/activate"
exec "$@"
