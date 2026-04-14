#!/bin/bash
# Test 8 GPU configurations specifically
# Environment: dif (conda)
# Created: 2026-03-06

set -e

# Activate conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate dif

# Set OpenBabel plugin path
export BABEL_LIBDIR=~/miniforge3/envs/dif/lib/openbabel/3.1.0

# Create log directory
mkdir -p log/8gpu-tests

echo "=========================================="
echo "8 GPU Configuration Tests"
echo "Started at: $(date)"
echo "=========================================="

# Test 1: ddp_8gpu_training.yaml
echo ""
echo "Test 1: ddp_8gpu_training.yaml"
echo "----------------------------------------"
python ddpo/main_diffsbdd_lightning.py \
    --config ddpo/configs/ddp_8gpu_training.yaml \
    2>&1 | tee log/8gpu-tests/ddp_8gpu_training_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "✅ Test 1 completed"
echo ""

# Test 2: ddp_8gpu.yaml
echo "Test 2: ddp_8gpu.yaml"
echo "----------------------------------------"
python ddpo/main_diffsbdd_lightning.py \
    --config ddpo/configs/ddp_8gpu.yaml \
    2>&1 | tee log/8gpu-tests/ddp_8gpu_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "✅ Test 2 completed"
echo ""

# Test 3: ddp_8gpu_production.yaml
echo "Test 3: ddp_8gpu_production.yaml"
echo "----------------------------------------"
python ddpo/main_diffsbdd_lightning.py \
    --config ddpo/configs/ddp_8gpu_production.yaml \
    2>&1 | tee log/8gpu-tests/ddp_8gpu_production_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "✅ Test 3 completed"
echo ""

echo "=========================================="
echo "All 8 GPU tests completed!"
echo "Finished at: $(date)"
echo "=========================================="
