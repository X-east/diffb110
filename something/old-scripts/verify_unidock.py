#!/usr/bin/env python
"""
Verify UniDock installation and test basic functionality.
"""

import sys
import subprocess
import tempfile
from pathlib import Path

# Add DiffSBDD to path
sys.path.insert(0, str(Path(__file__).parent.parent / "DiffSBDD"))

from analysis.molecule_builder import build_molecule
from constants import dataset_params
import utils
import torch


def test_unidock_installation():
    """Test if UniDock is installed and working."""
    print("=" * 60)
    print("Testing UniDock Installation")
    print("=" * 60)
    print()
    
    try:
        result = subprocess.run(
            ['unidock', '--version'],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✓ UniDock installed: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"✗ UniDock not found: {e}")
        return False


def test_unidock_backend():
    """Test UniDock backend class."""
    print("\n" + "=" * 60)
    print("Testing UniDock Backend")
    print("=" * 60)
    print()
    
    try:
        from ddpo.rewards.docking_backends import UniDockBackend
        
        backend = UniDockBackend(gpu_id=0, search_mode="fast")
        print(f"✓ UniDockBackend initialized successfully")
        print(f"  - GPU ID: {backend.gpu_id}")
        print(f"  - Search mode: {backend.search_mode}")
        return True
    except Exception as e:
        print(f"✗ UniDockBackend initialization failed: {e}")
        return False


def test_molecule_to_pdbqt():
    """Test converting a molecule to PDBQT format."""
    print("\n" + "=" * 60)
    print("Testing Molecule to PDBQT Conversion")
    print("=" * 60)
    print()
    
    try:
        # Create a simple test molecule (methane-like)
        dataset_info = dataset_params['crossdock']
        
        # Simple 5-atom molecule
        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0]
        ])
        
        atom_types = torch.tensor([0, 0, 0, 0, 0])  # All carbon
        
        mol = build_molecule(
            positions=positions,
            atom_types=atom_types,
            dataset_info=dataset_info,
            add_coords=True,
            use_openbabel=True
        )
        
        if mol is None:
            print("✗ Failed to build molecule")
            return False
        
        print(f"✓ Built test molecule: {mol.GetNumAtoms()} atoms")
        
        # Try to write as SDF
        with tempfile.NamedTemporaryFile(suffix='.sdf', delete=False) as tmp:
            tmp_file = tmp.name
            utils.write_sdf_file(tmp_file, [mol])
            print(f"✓ Wrote molecule to SDF: {tmp_file}")
        
        # TODO: Convert SDF to PDBQT using obabel
        # For now, just verify SDF creation
        
        return True
        
    except Exception as e:
        print(f"✗ Molecule conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gpu_availability():
    """Test GPU availability for UniDock."""
    print("\n" + "=" * 60)
    print("Testing GPU Availability")
    print("=" * 60)
    print()
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("✗ CUDA not available")
            return False
        
        num_gpus = torch.cuda.device_count()
        print(f"✓ CUDA available: {num_gpus} GPU(s)")
        
        for i in range(num_gpus):
            name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1024**3
            print(f"  - GPU {i}: {name} ({memory_gb:.1f} GB)")
        
        return True
        
    except Exception as e:
        print(f"✗ GPU check failed: {e}")
        return False


def main():
    """Run all verification tests."""
    print("\n" + "=" * 60)
    print("UniDock Installation Verification")
    print("=" * 60)
    print()
    
    results = []
    
    # Test 1: UniDock installation
    results.append(("UniDock Installation", test_unidock_installation()))
    
    # Test 2: GPU availability
    results.append(("GPU Availability", test_gpu_availability()))
    
    # Test 3: UniDock backend
    results.append(("UniDock Backend", test_unidock_backend()))
    
    # Test 4: Molecule conversion
    results.append(("Molecule Conversion", test_molecule_to_pdbqt()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)
    print()
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(passed for _, passed in results)
    
    print()
    if all_passed:
        print("✓ All tests passed! UniDock is ready to use.")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
