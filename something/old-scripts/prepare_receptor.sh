#!/bin/bash

# Prepare receptor PDB file for UniDock (convert to PDBQT)

set -e

echo "========================================="
echo "Receptor Preparation for UniDock"
echo "========================================="
echo ""

# Input and output files
INPUT_PDB="data/8GJJ-C-Cip.pdb"
OUTPUT_PDBQT="data/8GJJ-C-Cip.pdbqt"

# Check if input exists
if [ ! -f "$INPUT_PDB" ]; then
    echo "ERROR: Input PDB file not found: $INPUT_PDB"
    exit 1
fi

echo "Input: $INPUT_PDB"
echo "Output: $OUTPUT_PDBQT"
echo ""

# Check if obabel is available
if command -v obabel &> /dev/null; then
    echo "Using OpenBabel (obabel)..."
    obabel -ipdb "$INPUT_PDB" -opdbqt -O "$OUTPUT_PDBQT" -xr
    echo "✓ Conversion successful!"
elif command -v babel &> /dev/null; then
    echo "Using Babel..."
    babel -ipdb "$INPUT_PDB" -opdbqt "$OUTPUT_PDBQT" -xr
    echo "✓ Conversion successful!"
else
    echo "WARNING: OpenBabel not found!"
    echo ""
    echo "UniDock can work with PDB files directly, but PDBQT is recommended."
    echo "To install OpenBabel:"
    echo "  conda install -c conda-forge openbabel"
    echo ""
    echo "For now, we'll use the PDB file directly."
    exit 0
fi

echo ""
echo "Receptor ready for UniDock: $OUTPUT_PDBQT"
