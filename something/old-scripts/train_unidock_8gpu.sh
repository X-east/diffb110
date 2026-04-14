#!/bin/bash

# UniDock 8-GPU Production Training Script
# Full-scale training with GPU-accelerated docking

set -e  # Exit on error

echo "========================================="
echo "UniDock 8-GPU Production Training"
echo "========================================="
echo ""

# Activate conda environment
echo "[1/6] Activating conda environment..."
source ~/miniforge3/etc/profile.d/conda.sh
conda activate dif

# Set environment variables
echo "[2/6] Setting environment variables..."
export BABEL_LIBDIR=~/miniforge3/envs/dif/lib/openbabel/3.1.0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # All 8 GPUs

# Verify installations
echo "[3/6] Verifying installations..."
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA devices: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "UniDock: $(unidock --version)"
echo ""

# GPU status
echo "[4/6] GPU Status:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

# Check required files
echo "[5/6] Checking required files..."
if [ ! -f "DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt" ]; then
    echo "ERROR: DiffSBDD checkpoint not found!"
    exit 1
fi

if [ ! -f "data/8GJJ-C-Cip.pdb" ]; then
    echo "ERROR: Receptor PDB file not found!"
    exit 1
fi

echo "All required files present."
echo ""

# Create output directories
mkdir -p exp/unidock-8gpu-production
mkdir -p log/unidock-8gpu-production

# Run training
echo "[6/6] Starting 8-GPU training with UniDock..."
echo "Config: ddpo/configs/unidock_8gpu_training.yaml"
echo "Log: log/unidock-8gpu-production/training_$(date +%Y%m%d_%H%M%S).log"
echo ""
echo "GPU Allocation:"
echo "  - GPU 0: Model training"
echo "  - GPU 1-7: UniDock docking (Ray parallel)"
echo ""

python ddpo/main_diffsbdd.py \
    --config ddpo/configs/unidock_8gpu_training.yaml \
    2>&1 | tee log/unidock-8gpu-production/training_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "========================================="
echo "Training completed!"
echo "========================================="
echo ""
echo "Check results in:"
echo "  - exp/unidock-8gpu-production/"
echo "  - log/unidock-8gpu-production/"
