#!/bin/bash
# ==============================================
#  CartPole RL - Auto Install Script
#  Auto-detect GPU and install the appropriate
#  version of PyTorch (CUDA or CPU)
# ==============================================

set -e

echo "=============================================="
echo "  CartPole RL - Auto Install Script"
echo "=============================================="

# Detect OS
OS="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    REQ_FILE="requirements_linux.txt"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="mac"
    REQ_FILE="requirements_linux.txt"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    OS="windows"
    REQ_FILE="requirements_win.txt"
else
    echo "[WARN] Unrecognized OS: $OSTYPE, falling back to Linux requirements"
    REQ_FILE="requirements_linux.txt"
fi

echo "[INFO] Detected OS: $OS"
echo "[INFO] Using requirements file: $REQ_FILE"

# Check for NVIDIA GPU
HAS_GPU=false
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        HAS_GPU=true
        echo "[INFO] NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || true
    fi
fi

if [ "$HAS_GPU" = true ]; then
    echo ""
    echo "[INFO] >>> Installing GPU version of PyTorch (CUDA) ..."
    pip install -r "$REQ_FILE"
else
    echo ""
    echo "[INFO] No NVIDIA GPU detected, installing CPU version of PyTorch ..."
    echo "[INFO] >>> Step 1/2: Installing CPU-only PyTorch ..."
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    echo "[INFO] >>> Step 2/2: Installing remaining dependencies ..."
    pip install -r "$REQ_FILE"
fi

echo ""
echo "=============================================="
echo "  Installation complete! Verifying ..."
echo "=============================================="
python -c "
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'  PyTorch version: {torch.__version__}')
print(f'  Device: {device}')
if device == 'cuda':
    print(f'  GPU name: {torch.cuda.get_device_name(0)}')
else:
    print(f'  (CPU mode - the model in this project is small, CPU training is perfectly fine)')
import gymnasium
print(f'  Gymnasium version: {gymnasium.__version__}')
print()
print('  All dependencies installed successfully!')
"
echo "=============================================="
