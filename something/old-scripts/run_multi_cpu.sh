#!/bin/bash
# Run training with multi-CPU parallelization
# Uses 16 CPU cores for parallel reward computation

set -e

echo "=========================================="
echo "Multi-CPU Training Script"
echo "=========================================="
echo "Configuration:"
echo "  - GPU: 1x CUDA"
echo "  - CPU: 16 cores (Ray parallel)"
echo "  - Batch size: 8"
echo "  - Epochs: 5"
echo "=========================================="
echo ""

# Activate conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate dif

# Run training
python ddpo/main_diffsbdd.py \
    --config_path ddpo/configs/gpu_rl_test.yaml \
    --log_level INFO

echo ""
echo "=========================================="
echo "Training completed!"
echo "Check logs in: ./log/gpu-rl-test/"
echo "=========================================="
