#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-/workspace/robotic-sim-dishwash}"
ENV_DIR="$PROJECT_DIR/.env"
PYTHON_BIN="${PYTHON_BIN:-python3}"
KERNEL_NAME="robotic-sim-dishwash-env"
KERNEL_DISPLAY_NAME="Python (.env robotic-sim-dishwash)"

if [[ ! -d "$PROJECT_DIR" ]]; then
  echo "[ERROR] Project directory not found: $PROJECT_DIR"
  echo "Usage: bash scripts/runpod_setup.sh /workspace/robotic-sim-dishwash"
  exit 1
fi

cd "$PROJECT_DIR"

echo "[1/6] Creating virtual environment at $ENV_DIR"
$PYTHON_BIN -m venv "$ENV_DIR"

echo "[2/6] Activating environment"
source "$ENV_DIR/bin/activate"

echo "[3/6] Upgrading pip"
python -m pip install --upgrade pip

echo "[4/6] Installing project dependencies"
python -m pip install -r requirements.runpod.txt

echo "[5/6] Registering Jupyter kernel"
python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY_NAME"

echo "[6/6] Running smoke test"
python scripts/runpod_verify.py

echo
echo "✅ RunPod environment is ready"
echo "Activate with: source .env/bin/activate"
echo "Start Jupyter with: jupyter lab --ip 0.0.0.0 --port 8888 --no-browser --allow-root"
