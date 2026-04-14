#!/bin/bash
# GPU RL Test Script
# 200 diffusion steps, 5 batches x 8 molecules, QED reward

set -e

# Activate conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate DiffSBDD

# Set OpenBabel plugin directory
export BABEL_LIBDIR=~/miniforge3/envs/dif/lib/openbabel/3.1.0

echo "=========================================="
echo "GPU RL Test Configuration"
echo "=========================================="
echo "Diffusion steps: 200"
echo "Batches: 5"
echo "Molecules per batch: 8"
echo "Total molecules: 40"
echo "Reward: QED"
echo "Device: CUDA"
echo "=========================================="
echo ""

# Check GPU availability
echo "Checking GPU..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# Run training
echo "Starting RL training..."
python ddpo/main_diffsbdd.py \
    --config ddpo/configs/gpu_rl_test.yaml \
    --verbose

echo ""
echo "=========================================="
echo "Test completed!"
echo "=========================================="
