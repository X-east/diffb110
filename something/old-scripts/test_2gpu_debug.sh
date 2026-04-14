#!/bin/bash
# 2 GPU Debug Test Script
# Usage: bash scripts/test_2gpu_debug.sh

set -e

echo "=========================================="
echo "Starting 2 GPU Lightning DDP Debug Test"
echo "=========================================="

# Activate conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate dif

# Set OpenBabel plugin directory
export BABEL_LIBDIR=~/miniforge3/envs/dif/lib/openbabel/3.1.0

# Use only 2 GPUs for debugging
export CUDA_VISIBLE_DEVICES=0,1

# Display GPU info
echo "Available GPUs:"
nvidia-smi --list-gpus

# Display CPU info
echo ""
echo "CPU cores: $(nproc)"
echo ""

# Add debug output
export PYTHONUNBUFFERED=1

# Run training with verbose output
python -u ddpo/main_diffsbdd_lightning.py \
    --config_path ddpo/configs/ddp_2gpu_debug.yaml \
    --seed 42 \
    --log_level INFO 2>&1 | tee ./log/2gpu_debug_test.log

echo "=========================================="
echo "2 GPU debug test completed!"
echo "=========================================="
