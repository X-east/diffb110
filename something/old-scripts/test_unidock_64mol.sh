#!/bin/bash
# Test UniDock with 64 molecules to check docking success rate

set -e

echo "========================================================================"
echo "UniDock 64 Molecules Test"
echo "========================================================================"

# Activate conda environment
echo ""
echo "[1/5] Activating conda environment..."
source ~/miniforge3/etc/profile.d/conda.sh
conda activate dif
export BABEL_LIBDIR=~/miniforge3/envs/dif/lib/openbabel/3.1.0
echo "✓ Environment activated"

# Verify UniDock installation
echo ""
echo "[2/5] Verifying UniDock installation..."
unidock --version
echo "✓ UniDock available"

# Verify receptor file
echo ""
echo "[3/5] Verifying receptor file..."
if [ ! -f "data/8GJJ-C-Cip.pdbqt" ]; then
    echo "❌ Receptor PDBQT file not found!"
    echo "   Run: bash scripts/prepare_receptor.sh"
    exit 1
fi
echo "✓ Receptor file exists"

# Clean up previous runs
echo ""
echo "[4/5] Cleaning up previous runs..."
rm -rf exp/unidock-64mol-test
rm -rf log/unidock-64mol-test
echo "✓ Cleanup complete"

# Start test
echo ""
echo "[5/5] Starting 64-molecule test..."
echo "  Config: ddpo/configs/unidock_64mol_test.yaml"
echo "  Batch size: 64 molecules"
echo "  Chunk size: 8 molecules"
echo "  Ray workers: 8"
echo "  Epochs: 1"
echo ""
echo "========================================================================"
echo ""

# Run training
python ddpo/main_diffsbdd.py --config ddpo/configs/unidock_64mol_test.yaml

echo ""
echo "========================================================================"
echo "Test completed!"
echo "========================================================================"
