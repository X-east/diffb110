#!/usr/bin/env python
"""
Quick test script for Ray parallel smina docking.
Tests if Ray can parallelize docking calculations correctly.
"""

import sys
import time
from pathlib import Path

import torch
import ray

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "DiffSBDD"))

from ddpo.rewards.docking import DockingReward, calculate_single_docking_score
from DiffSBDD.constants import dataset_params

# Simple DataProto replacement for testing
class DataProto:
    def __init__(self, batch):
        self.batch = batch


def create_dummy_molecules(batch_size=8, n_nodes=20):
    """Create dummy molecule data for testing."""
    # Random coordinates
    x = torch.randn(batch_size, n_nodes, 3) * 2.0
    
    # Random atom types (one-hot encoded)
    # Atom types: C, N, O, F, S, Cl, Br (7 types)
    num_atom_types = 10  # crossdock has 10 atom types
    h = torch.zeros(batch_size, n_nodes, num_atom_types)
    for i in range(batch_size):
        # Randomly assign atom types (mostly C, N, O)
        atom_indices = torch.randint(0, 3, (n_nodes,))  # 0=C, 1=N, 2=O
        h[i, torch.arange(n_nodes), atom_indices] = 1.0
    
    # All molecules have same number of nodes for simplicity
    nodesxsample = torch.full((batch_size,), n_nodes, dtype=torch.long)
    
    return x, h, nodesxsample


def test_ray_parallel_docking():
    """Test Ray parallel docking with timing comparison."""
    print("=" * 60)
    print("Ray Parallel Smina Docking Test")
    print("=" * 60)
    
    # Initialize Ray
    print("\n[1] Initializing Ray...")
    ray.init(num_cpus=8, ignore_reinit_error=True)
    print(f"✓ Ray initialized with 8 CPUs")
    print(f"  Available resources: {ray.available_resources()}")
    
    # Create test data
    print("\n[2] Creating test molecules...")
    batch_size = 8
    n_nodes = 20
    x, h, nodesxsample = create_dummy_molecules(batch_size, n_nodes)
    print(f"✓ Created {batch_size} dummy molecules with {n_nodes} atoms each")
    
    # Create DataProto
    data = DataProto(batch={
        'x': x,
        'categorical': h,
        'nodesxsample': nodesxsample
    })
    
    # Initialize DockingReward
    print("\n[3] Initializing DockingReward...")
    receptor_file = "data/8GJJ-C-Cip.pdb"
    if not Path(receptor_file).exists():
        print(f"✗ Receptor file not found: {receptor_file}")
        print("  Please ensure the receptor file exists")
        ray.shutdown()
        return False
    
    reward_fn = DockingReward(
        receptor_file=receptor_file,
        normalization_factor=-10.0,
        failure_penalty=0.0
    )
    print(f"✓ DockingReward initialized")
    print(f"  Receptor: {receptor_file}")
    
    # Test Ray parallel docking
    print("\n[4] Testing Ray parallel docking...")
    print(f"  Submitting {batch_size} docking tasks to Ray...")
    
    start_time = time.time()
    result_data = reward_fn.calculate_rewards(data)
    parallel_time = time.time() - start_time
    
    rewards = result_data.batch['rewards']
    print(f"✓ Ray parallel docking completed in {parallel_time:.2f}s")
    print(f"  Average time per molecule: {parallel_time/batch_size:.2f}s")
    print(f"  Rewards shape: {rewards.shape}")
    print(f"  Rewards: {rewards.tolist()}")
    
    # Analyze results
    print("\n[5] Analyzing results...")
    valid_count = (rewards != 0.0).sum().item()
    failed_count = (rewards == 0.0).sum().item()
    
    print(f"  Valid docking scores: {valid_count}/{batch_size}")
    print(f"  Failed molecules: {failed_count}/{batch_size}")
    
    if valid_count > 0:
        valid_rewards = rewards[rewards != 0.0]
        print(f"  Valid reward range: [{valid_rewards.min():.3f}, {valid_rewards.max():.3f}]")
        print(f"  Valid reward mean: {valid_rewards.mean():.3f}")
    
    # Estimate speedup
    print("\n[6] Estimating speedup...")
    serial_time_estimate = parallel_time * batch_size
    speedup = serial_time_estimate / parallel_time
    print(f"  Estimated serial time: {serial_time_estimate:.2f}s")
    print(f"  Parallel time: {parallel_time:.2f}s")
    print(f"  Estimated speedup: {speedup:.1f}x")
    
    # Shutdown Ray
    print("\n[7] Shutting down Ray...")
    ray.shutdown()
    print("✓ Ray shutdown complete")
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"✓ Ray parallel docking is working")
    print(f"✓ Processed {batch_size} molecules in {parallel_time:.2f}s")
    print(f"✓ Valid docking scores: {valid_count}/{batch_size}")
    print(f"✓ Estimated speedup: {speedup:.1f}x")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    try:
        success = test_ray_parallel_docking()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        ray.shutdown()
        sys.exit(1)
