#!/bin/bash
# 8 GPU Training Script with PyTorch Lightning
# Usage: bash scripts/run_8gpu_lightning.sh

set -e  # Exit on error

echo "=========================================="
echo "Starting 8 GPU Lightning DDP Training"
echo "=========================================="

# Activate conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate dif

# Set OpenBabel plugin directory
export BABEL_LIBDIR=~/miniforge3/envs/dif/lib/openbabel/3.1.0

# Set CUDA devices (all 8 GPUs)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Verify GPUs
echo "Available GPUs:"
nvidia-smi --list-gpus

# Run training with Lightning
python ddpo/main_diffsbdd_lightning.py \
    --config_path ddpo/configs/ddp_8gpu.yaml \
    --seed 42 \
    --log_level INFO

echo "=========================================="
echo "Training completed!"
echo "=========================================="
