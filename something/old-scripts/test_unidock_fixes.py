#!/usr/bin/env python
"""
Test script to verify UniDock RL training fixes.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "DiffSBDD"))

# Change to project root
os.chdir(project_root)

def test_imports():
    """Test that all imports work."""
    print("[1/4] Testing imports...")
    try:
        from ddpo.rewards.docking import DockingReward
        from ddpo.rewards.docking_backends import UniDockBackend
        print("✓ Imports successful")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def test_unidock_backend():
    """Test UniDock backend initialization."""
    print("\n[2/4] Testing UniDock backend...")
    try:
        from ddpo.rewards.docking_backends import UniDockBackend
        backend = UniDockBackend(gpu_id=0, search_mode='fast')
        print("✓ UniDock backend initialized")
        print(f"  GPU ID: {backend.gpu_id}")
        print(f"  Search mode: {backend.search_mode}")
        return True
    except Exception as e:
        print(f"✗ Backend error: {e}")
        return False

def test_docking_reward():
    """Test DockingReward initialization."""
    print("\n[3/4] Testing DockingReward...")
    try:
        from ddpo.rewards.docking import DockingReward
        reward = DockingReward(
            backend='unidock',
            receptor_file='data/8GJJ-C-Cip.pdbqt',
            gpu_id=0,
            search_mode='fast',
            chunk_size=8,
            center_x=183.0,
            center_y=180.0,
            center_z=230.0,
            size_x=25.0,
            size_y=25.0,
            size_z=25.0
        )
        print("✓ DockingReward initialized")
        print(f"  Backend: {reward.backend}")
        print(f"  GPU ID: {reward.gpu_id}")
        print(f"  Chunk size: {reward.chunk_size}")
        print(f"  Receptor: {Path(reward.receptor_file).name}")
        return True
    except Exception as e:
        print(f"✗ Reward error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config():
    """Test configuration file."""
    print("\n[4/4] Testing configuration...")
    try:
        import yaml
        with open('ddpo/configs/unidock_test.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Check key parameters
        assert config['train']['batch_size'] == 16, "Batch size should be 16"
        assert config['reward']['docking']['backend'] == 'unidock', "Backend should be unidock"
        assert config['reward']['docking']['gpu_id'] == 0, "GPU ID should be 0"
        assert config['reward']['docking']['chunk_size'] == 8, "Chunk size should be 8"
        assert config['ray']['num_cpus'] == 8, "CPU count should be 8"
        
        print("✓ Configuration valid")
        print(f"  Batch size: {config['train']['batch_size']}")
        print(f"  Chunk size: {config['reward']['docking']['chunk_size']}")
        print(f"  Ray CPUs: {config['ray']['num_cpus']}")
        return True
    except Exception as e:
        print(f"✗ Config error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 70)
    print("UniDock RL Training Fixes - Verification Test")
    print("=" * 70)
    print()
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("UniDock Backend", test_unidock_backend()))
    results.append(("DockingReward", test_docking_reward()))
    results.append(("Configuration", test_config()))
    
    print()
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:20s}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 70)
    
    if all_passed:
        print("\n✓ All tests passed! Ready to run training test.")
        print("\nNext step:")
        print("  bash scripts/test_unidock.sh")
        return 0
    else:
        print("\n✗ Some tests failed. Please fix errors before running training.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
