#!/bin/bash

# Test UniDock with raw command to see actual output

set -e

echo "Testing UniDock raw command..."
echo ""

# Create a simple test ligand (benzene)
cat > /tmp/test_ligand.pdbqt << 'EOF'
REMARK  Name = benzene
REMARK  1 active torsions:
REMARK  status: ('A' for Active; 'I' for Inactive)
ROOT
ATOM      1  C   UNL     1       0.000   0.000   0.000  1.00  0.00     0.000 C
ATOM      2  C   UNL     1       1.400   0.000   0.000  1.00  0.00     0.000 C
ATOM      3  C   UNL     1       2.100   1.200   0.000  1.00  0.00     0.000 C
ATOM      4  C   UNL     1       1.400   2.400   0.000  1.00  0.00     0.000 C
ATOM      5  C   UNL     1       0.000   2.400   0.000  1.00  0.00     0.000 C
ATOM      6  C   UNL     1      -0.700   1.200   0.000  1.00  0.00     0.000 C
ENDROOT
TORSDOF 0
EOF

echo "Created test ligand: /tmp/test_ligand.pdbqt"
echo ""

# Run UniDock
echo "Running UniDock..."
echo "Command: unidock --receptor data/8GJJ-C-Cip.pdbqt --gpu_batch /tmp/test_ligand.pdbqt --search_mode fast --center_x 0 --center_y 0 --center_z 0 --size_x 20 --size_y 20 --size_z 20 --dir /tmp/unidock_test"
echo ""

mkdir -p /tmp/unidock_test

unidock \
    --receptor data/8GJJ-C-Cip.pdbqt \
    --gpu_batch /tmp/test_ligand.pdbqt \
    --search_mode fast \
    --center_x 0 \
    --center_y 0 \
    --center_z 0 \
    --size_x 20 \
    --size_y 20 \
    --size_z 20 \
    --dir /tmp/unidock_test

echo ""
echo "UniDock completed!"
echo ""
echo "Output files:"
ls -lh /tmp/unidock_test/
