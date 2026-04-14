#!/bin/bash
# CPU Parallel Test Script
# 500 diffusion steps, Ray parallel smina docking on 8 CPUs

set -e

# Activate conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate dif

# Set OpenBabel plugin directory
export BABEL_LIBDIR=~/miniforge3/envs/dif/lib/openbabel/3.1.0

# Add smina.static to PATH
export PATH=~/bin:$PATH

echo "=========================================="
echo "GPU Training + CPU Parallel Docking Test"
echo "=========================================="
echo "Diffusion steps: 500"
echo "Batches: 3"
echo "Molecules per batch: 8"
echo "Total molecules: 24"
echo "Reward: Pure Docking (Smina)"
echo "Model Device: GPU (CUDA)"
echo "Docking: CPU Parallel (Ray 8 workers)"
echo "=========================================="
echo ""

# Check smina installation
echo "Checking smina installation..."
if command -v smina.static &> /dev/null; then
    echo "✓ smina.static found: $(which smina.static)"
else
    echo "✗ smina.static not found in PATH"
    echo "Please ensure smina.static is in ~/bin/ or add it to PATH"
    exit 1
fi
echo ""

# Check GPU availability
echo "Checking GPU..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# Check Ray
echo "Checking Ray..."
python -c "import ray; print(f'Ray version: {ray.__version__}')"
echo ""

# Check receptor file
echo "Checking receptor file..."
if [ -f "data/8GJJ-C-Cip.pdb" ]; then
    echo "✓ Receptor file found: data/8GJJ-C-Cip.pdb"
else
    echo "✗ Receptor file not found: data/8GJJ-C-Cip.pdb"
    exit 1
fi
echo ""

# Run training with timing
echo "Starting GPU training with CPU parallel docking..."
echo "This will test:"
echo "  1. 500-step diffusion on GPU"
echo "  2. Ray parallel smina docking (8 CPU workers)"
echo "  3. Molecule building and validation"
echo ""

time python ddpo/main_diffsbdd.py \
    --config_path ddpo/configs/cpu_parallel_test.yaml

echo ""
echo "=========================================="
echo "Test completed!"
echo "=========================================="
echo ""
echo "Check the log file for:"
echo "  - Ray parallel docking timing"
echo "  - Docking scores for each molecule"
echo "  - Molecule validity statistics"
echo ""
echo "Log directory: ./log/cpu-parallel-test/"

