#!/bin/bash
# Run from reid_check/live_cam_test/
# Creates .venv and installs all dependencies for TransReID testing

set -e
cd "$(dirname "$0")/.."

echo "=== Creating virtualenv ==="
python3 -m venv .venv
source .venv/bin/activate

echo "=== Installing PyTorch (CUDA 12.4) ==="
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo "=== Installing TransReID dependencies ==="
pip install -r TransReID/requirements.txt

echo "=== Installing pipeline extras ==="
pip install ultralytics gdown tqdm scipy pandas matplotlib

echo "=== Verifying CUDA ==="
python3 -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"

echo ""
echo "Done. Activate with: source live_cam_test/.venv/bin/activate"
