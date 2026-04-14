#!/bin/bash
# UniDock Minimal RL Training Test
# Ultra-minimal test with only 4 molecules

set -e

echo "========================================================================"
echo "UniDock Minimal RL Training Test"
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
    echo "✗ UniDock not found"
    exit 1
fi
unidock --version
echo "✓ UniDock available"
echo ""

# Verify receptor file
echo "[3/5] Verifying receptor file..."
if [ ! -f "data/8GJJ-C-Cip.pdbqt" ]; then
    echo "✗ Receptor PDBQT file not found"
    exit 1
fi
echo "✓ Receptor file exists"
echo ""

# Clean up previous runs
echo "[4/5] Cleaning up previous runs..."
rm -rf exp/unidock-minimal-test 2>/dev/null || true
rm -rf log/unidock-minimal-test 2>/dev/null || true
mkdir -p log/unidock-minimal-test
echo "✓ Cleanup complete"
echo ""

# Run training test
echo "[5/5] Starting MINIMAL RL training test..."
echo "  Config: ddpo/configs/unidock_minimal_test.yaml"
echo "  Batch size: 4 molecules (MINIMAL)"
echo "  Chunk size: 2 molecules"
echo "  Ray workers: 2"
echo "  Epochs: 1"
echo ""
echo "========================================================================"
echo ""

# Set Ray environment variables
export RAY_DEDUP_LOGS=0
export RAY_IGNORE_UNHANDLED_ERRORS=1

# Run training with timeout
timeout 300 python ddpo/main_diffsbdd.py \
    --config ddpo/configs/unidock_minimal_test.yaml \
    2>&1 | tee log/unidock-minimal-test/training_$(date +%Y%m%d_%H%M%S).log

EXIT_CODE=$?

echo ""
echo "========================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Minimal training test completed successfully!"
    echo ""
    echo "Check results in:"
    echo "  - exp/unidock-minimal-test/"
    echo "  - log/unidock-minimal-test/"
    echo ""
    echo "If this test passed, you can try the full test:"
    echo "  bash scripts/test_unidock.sh"
elif [ $EXIT_CODE -eq 124 ]; then
    echo "✗ Training test timed out (5 minutes)"
else
    echo "✗ Training test failed with exit code: $EXIT_CODE"
fi
echo "========================================================================"

exit $EXIT_CODE
