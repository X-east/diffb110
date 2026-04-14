#!/usr/bin/env python
"""
Test UniDock docking with a simple molecule.
"""

import sys
import tempfile
from pathlib import Path

# Add DiffSBDD to path
sys.path.insert(0, str(Path(__file__).parent.parent / "DiffSBDD"))

from analysis.molecule_builder import build_molecule, process_molecule
from constants import dataset_params
import utils
import torch


def create_test_molecule():
    """Create a simple test molecule for docking."""
    print("Creating test molecule...")
    
    dataset_info = dataset_params['crossdock']
    
    # Create a simple benzene-like molecule (6 carbons in a ring)
    positions = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.4, 0.0, 0.0],
        [2.1, 1.2, 0.0],
        [1.4, 2.4, 0.0],
        [0.0, 2.4, 0.0],
        [-0.7, 1.2, 0.0],
    ], dtype=torch.float32)
    
    # All carbon atoms
    atom_types = torch.tensor([0, 0, 0, 0, 0, 0])
    
    mol = build_molecule(
        positions=positions,
        atom_types=atom_types,
        dataset_info=dataset_info,
        add_coords=True,
        use_openbabel=True
    )
    
    if mol is None:
        print("✗ Failed to build molecule")
        return None
    
    # Process molecule
    mol = process_molecule(mol, add_hydrogens=False, sanitize=True)
    
    if mol is None:
        print("✗ Failed to process molecule")
        return None
    
    print(f"✓ Created test molecule: {mol.GetNumAtoms()} atoms")
    return mol


def save_molecule_as_sdf(mol, filename):
    """Save molecule as SDF file."""
    utils.write_sdf_file(filename, [mol])
    print(f"✓ Saved molecule to: {filename}")


def convert_sdf_to_pdbqt(sdf_file, pdbqt_file):
    """Convert SDF to PDBQT using obabel."""
    import subprocess
    
    try:
        result = subprocess.run(
            ['obabel', '-isdf', sdf_file, '-opdbqt', '-O', pdbqt_file],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✓ Converted to PDBQT: {pdbqt_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Conversion failed: {e.stderr}")
        return False


def test_unidock_docking(ligand_pdbqt, receptor_pdbqt):
    """Test UniDock docking."""
    from ddpo.rewards.docking_backends import UniDockBackend
    
    print("\nTesting UniDock docking...")
    
    try:
        # Initialize backend
        backend = UniDockBackend(gpu_id=0, search_mode="fast")
        print(f"✓ UniDock backend initialized")
        
        # Perform docking with docking box centered on receptor
        # RE-CmeB protein is around (183, 180, 230)
        print(f"Docking ligand: {ligand_pdbqt}")
        print(f"Receptor: {receptor_pdbqt}")
        print(f"Docking box: center=(183, 180, 230), size=(25, 25, 25)")
        
        score = backend.dock_single(
            ligand_file=ligand_pdbqt,
            receptor_file=receptor_pdbqt,
            center_x=183.0,
            center_y=180.0,
            center_z=230.0,
            size_x=25.0,
            size_y=25.0,
            size_z=25.0
        )
        
        print(f"\n✓ Docking completed!")
        print(f"  Binding affinity: {score:.2f} kcal/mol")
        
        if score != 0.0:
            print(f"  Status: Valid docking score obtained")
        else:
            print(f"  Status: Warning - zero score may indicate parsing issue")
        
        return score
        
    except Exception as e:
        print(f"\n✗ Docking failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run UniDock docking test."""
    print("=" * 60)
    print("UniDock Docking Test")
    print("=" * 60)
    print()
    
    # Check receptor file
    receptor_pdbqt = "data/8GJJ-C-Cip.pdbqt"
    if not Path(receptor_pdbqt).exists():
        print(f"✗ Receptor PDBQT not found: {receptor_pdbqt}")
        print("Run: bash scripts/prepare_receptor.sh")
        return 1
    
    print(f"✓ Receptor ready: {receptor_pdbqt}")
    print()
    
    # Create test molecule
    mol = create_test_molecule()
    if mol is None:
        return 1
    
    # Save as SDF
    with tempfile.NamedTemporaryFile(suffix='.sdf', delete=False) as tmp_sdf:
        sdf_file = tmp_sdf.name
        save_molecule_as_sdf(mol, sdf_file)
    
    # Convert to PDBQT
    with tempfile.NamedTemporaryFile(suffix='.pdbqt', delete=False) as tmp_pdbqt:
        pdbqt_file = tmp_pdbqt.name
    
    if not convert_sdf_to_pdbqt(sdf_file, pdbqt_file):
        return 1
    
    print()
    
    # Test docking
    score = test_unidock_docking(pdbqt_file, receptor_pdbqt)
    
    if score is not None:
        print("\n" + "=" * 60)
        print("✓ UniDock docking test PASSED")
        print("=" * 60)
        return 0
    else:
        print("\n" + "=" * 60)
        print("✗ UniDock docking test FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
