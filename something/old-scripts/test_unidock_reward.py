#!/usr/bin/env python
"""
Test UniDock integration with DockingReward.
"""

import sys
import torch
import ray
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ddpo.rewards.docking import DockingReward
from DiffSBDD.constants import dataset_params


def create_test_batch(batch_size=4):
    """Create a test batch of molecules."""
    print(f"Creating test batch of {batch_size} molecules...")
    
    # Simple test molecules (benzene-like rings)
    positions = torch.randn(batch_size, 10, 3) * 2.0
    atom_types = torch.zeros(batch_size, 10, 9)
    atom_types[:, :, 0] = 1.0  # All carbon
    nodesxsample = torch.full((batch_size,), 6, dtype=torch.long)
    
    # Create DataProto-like structure
    class DataProto:
        def __init__(self):
            self.batch = {
                'x': positions,
                'categorical': atom_types,
                'nodesxsample': nodesxsample
            }
    
    return DataProto()


def test_unidock_reward():
    """Test UniDock reward function."""
    print("=" * 60)
    print("Testing UniDock Reward Integration")
    print("=" * 60)
    print()
    
    # Initialize Ray
    print("Initializing Ray...")
    ray.init(ignore_reinit_error=True, num_cpus=4, num_gpus=1)
    print("✓ Ray initialized")
    print()
    
    try:
        # Create reward function
        print("Creating DockingReward with UniDock backend...")
        reward = DockingReward(
            backend="unidock",
            receptor_file="data/8GJJ-C-Cip.pdbqt",
            gpu_id=0,
            search_mode="fast",
            center_x=183.0,
            center_y=180.0,
            center_z=230.0,
            size_x=25.0,
            size_y=25.0,
            size_z=25.0,
            chunk_size=4
        )
        print()
        
        # Create test batch
        data = create_test_batch(batch_size=4)
        print(f"✓ Test batch created: {data.batch['x'].shape}")
        print()
        
        # Calculate rewards
        print("Calculating rewards...")
        data = reward.calculate_rewards(data)
        
        rewards = data.batch['rewards']
        print(f"\n✓ Rewards calculated!")
        print(f"  Shape: {rewards.shape}")
        print(f"  Values: {rewards.tolist()}")
        print(f"  Mean: {rewards.mean():.3f}")
        print(f"  Std: {rewards.std():.3f}")
        
        # Validate
        assert rewards.shape[0] == 4, "Wrong batch size"
        assert not torch.isnan(rewards).any(), "NaN rewards"
        
        print("\n" + "=" * 60)
        print("✓ UniDock Reward Test PASSED")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        ray.shutdown()


if __name__ == "__main__":
    sys.exit(test_unidock_reward())
