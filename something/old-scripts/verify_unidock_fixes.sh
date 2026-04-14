#!/bin/bash
# Verify UniDock RL training fixes

set -e

echo "========================================================================"
echo "UniDock RL Training Fixes - Verification"
echo "========================================================================"
echo ""

# Activate conda environment
echo "[1/2] Activating conda environment..."
source ~/miniforge3/etc/profile.d/conda.sh
conda activate dif
export BABEL_LIBDIR=~/miniforge3/envs/dif/lib/openbabel/3.1.0
echo "✓ Environment activated"
echo ""

# Run verification test
echo "[2/2] Running verification tests..."
python scripts/test_unidock_fixes.py

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "========================================================================"
    echo "✓ All fixes verified successfully!"
    echo "========================================================================"
    echo ""
    echo "You can now run the full training test:"
    echo "  bash scripts/test_unidock.sh"
else
    echo "========================================================================"
    echo "✗ Verification failed"
    echo "========================================================================"
fi

exit $EXIT_CODE
