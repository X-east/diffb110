#!/bin/bash
# 8 GPU Training Script with PyTorch Lightning
# Usage: bash scripts/train_8gpu.sh

set -e  # Exit on error

echo "=========================================="
echo "Starting 8 GPU Lightning DDP Training"
echo "=========================================="

# Activate conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate dif

# Set OpenBabel plugin directory
export BABEL_LIBDIR=~/miniforge3/envs/dif/lib/openbabel/3.1.0

# Use all 8 GPUs (no CUDA_VISIBLE_DEVICES restriction)
# If you want to use specific GPUs, uncomment and modify:
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Display GPU info
echo "Available GPUs:"
nvidia-smi --list-gpus

# Display CPU info
echo ""
echo "CPU cores: $(nproc)"
echo "Using 24 CPU cores for Ray parallel docking"
echo ""

# Run training
python ddpo/main_diffsbdd_lightning.py \
    --config_path ddpo/configs/ddp_8gpu_training.yaml \
    --seed 42 \
    --log_level INFO

echo "=========================================="
echo "8 GPU training completed!"
echo "=========================================="
