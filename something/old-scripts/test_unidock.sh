#!/bin/bash
# UniDock RL Training Test
# Tests UniDock GPU-accelerated docking in RL training loop

set -e  # Exit on error

echo "========================================================================"
echo "UniDock RL Training Test"
echo "========================================================================"
echo ""

# Activate conda environment
echo "[1/5] Activating conda environment..."
source ~/miniforge3/etc/profile.d/conda.sh
conda activate dif
export BABEL_LIBDIR=~/miniforge3/envs/dif/lib/openbabel/3.1.0
echo "✓ Environment activated"
echo ""

# Verify UniDock installation
echo "[2/5] Verifying UniDock installation..."
if ! command -v unidock &> /dev/null; then
    echo "✗ UniDock not found. Please install via: conda install -c conda-forge unidock"
    exit 1
fi
unidock --version
echo "✓ UniDock available"
echo ""

# Verify receptor file
echo "[3/5] Verifying receptor file..."
if [ ! -f "data/8GJJ-C-Cip.pdbqt" ]; then
    echo "✗ Receptor PDBQT file not found: data/8GJJ-C-Cip.pdbqt"
    echo "  Please run: bash scripts/prepare_receptor.sh"
    exit 1
fi
echo "✓ Receptor file exists"
echo ""

# Clean up previous runs
echo "[4/5] Cleaning up previous runs..."
rm -rf exp/unidock-test/training_* 2>/dev/null || true
rm -rf log/unidock-test/training_* 2>/dev/null || true
echo "✓ Cleanup complete"
echo ""

# Run training test
echo "[5/5] Starting RL training test..."
echo "  Config: ddpo/configs/unidock_test.yaml"
echo "  Batch size: 16 molecules"
echo "  Epochs: 1"
echo "  Backend: UniDock GPU"
echo ""
echo "========================================================================"
echo ""

# Set Ray environment variables for better error handling
export RAY_DEDUP_LOGS=0
export RAY_IGNORE_UNHANDLED_ERRORS=1

# Run training with timeout
timeout 600 python ddpo/main_diffsbdd.py \
    --config ddpo/configs/unidock_test.yaml \
    2>&1 | tee log/unidock-test/training_$(date +%Y%m%d_%H%M%S).log

EXIT_CODE=$?

echo ""
echo "========================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Training test completed successfully"
    echo ""
    echo "Check results in:"
    echo "  - exp/unidock-test/"
    echo "  - log/unidock-test/"
elif [ $EXIT_CODE -eq 124 ]; then
    echo "✗ Training test timed out (10 minutes)"
    echo "  This may indicate the training is stuck"
else
    echo "✗ Training test failed with exit code: $EXIT_CODE"
fi
echo "========================================================================"

exit $EXIT_CODE
