#!/usr/bin/env python
"""
Diagnostic script to identify issues in the training pipeline.
"""

import sys
from pathlib import Path

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "RLPF-verl_diffusion"))
sys.path.insert(0, str(PROJECT_ROOT / "DiffSBDD"))

import torch
from verl_diffusion.protocol import DataProto, TensorDict

print("=" * 80)
print("DIAGNOSTIC SCRIPT - Testing Training Pipeline Components")
print("=" * 80)

# Test 1: DataProto basic operations
print("\n[Test 1] DataProto basic operations")
try:
    batch_data = {
        'x': torch.randn(4, 20, 3),
        'categorical': torch.randn(4, 20, 10),
        'nodesxsample': torch.tensor([15, 18, 12, 20]),
        'rewards': torch.randn(4),
    }
    
    data = DataProto(
        batch=TensorDict(batch_data, batch_size=4),
        meta_info={}
    )
    
    print(f"✓ DataProto created: batch_size={data.batch.batch_size}")
    print(f"  Keys: {list(data.batch.keys())}")
    print(f"  Rewards shape: {data.batch['rewards'].shape}")
    
    # Test select_idxs
    indices = [0, 2]
    filtered = data.select_idxs(indices)
    print(f"✓ select_idxs works: filtered batch_size={filtered.batch.batch_size}")
    
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 2: DiffSBDDFilter
print("\n[Test 2] DiffSBDDFilter")
try:
    from ddpo.utils import DiffSBDDFilter
    from ddpo.configs.dataset_info import get_crossdock_info
    
    dataset_info = get_crossdock_info()
    filter_obj = DiffSBDDFilter(dataset_info=dataset_info)
    
    # Create test data
    batch_data = {
        'x': torch.randn(4, 20, 3),
        'categorical': torch.randn(4, 20, 10),
        'nodesxsample': torch.tensor([15, 18, 12, 20]),
        'rewards': torch.randn(4),
    }
    
    data = DataProto(
        batch=TensorDict(batch_data, batch_size=4),
        meta_info={'max_n_nodes': 50}
    )
    
    print(f"✓ Filter created: {filter_obj}")
    print(f"  Input batch_size: {data.batch.batch_size}")
    
    # Test filter
    filtered_data, filter_ratio, novelty_ratio = filter_obj.filter(data)
    print(f"✓ Filter works: filter_ratio={filter_ratio:.2f}, novelty_ratio={novelty_ratio:.2f}")
    print(f"  Output batch_size: {filtered_data.batch.batch_size}")
    
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Reward calculation
print("\n[Test 3] DruglikenessReward")
try:
    from ddpo.rewards import DruglikenessReward
    from ddpo.configs.dataset_info import get_crossdock_info
    
    dataset_info = get_crossdock_info()
    reward = DruglikenessReward(dataset_info=dataset_info)
    
    # Create test data with proper one-hot encoding
    batch_size = 4
    max_nodes = 20
    num_atom_types = 10
    
    # Create one-hot categorical
    categorical = torch.zeros(batch_size, max_nodes, num_atom_types)
    categorical[:, :, 0] = 1  # All carbon atoms
    
    batch_data = {
        'x': torch.randn(batch_size, max_nodes, 3),
        'categorical': categorical,
        'nodesxsample': torch.tensor([15, 18, 12, 20]),
    }
    
    data = DataProto(
        batch=TensorDict(batch_data, batch_size=batch_size),
        meta_info={'max_n_nodes': max_nodes}
    )
    
    print(f"✓ Reward created: {reward}")
    print(f"  Input shapes: x={data.batch['x'].shape}, h={data.batch['categorical'].shape}")
    
    # Calculate rewards
    data = reward.calculate_rewards(data)
    print(f"✓ Rewards calculated: shape={data.batch['rewards'].shape}")
    print(f"  Reward values: {data.batch['rewards']}")
    
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Dimension compatibility
print("\n[Test 4] Tensor dimension compatibility")
try:
    # Test concat operations
    data1 = DataProto(
        batch=TensorDict({
            'x': torch.randn(2, 20, 3),
            'categorical': torch.randn(2, 20, 10),
            'nodesxsample': torch.tensor([15, 18]),
        }, batch_size=2),
        meta_info={}
    )
    
    data2 = DataProto(
        batch=TensorDict({
            'x': torch.randn(2, 20, 3),
            'categorical': torch.randn(2, 20, 10),
            'nodesxsample': torch.tensor([12, 20]),
        }, batch_size=2),
        meta_info={}
    )
    
    print(f"✓ Created two DataProto objects")
    print(f"  data1 batch_size: {data1.batch.batch_size}")
    print(f"  data2 batch_size: {data2.batch.batch_size}")
    
    # Test concat
    combined = DataProto.concat([data1, data2])
    print(f"✓ Concat works: combined batch_size={combined.batch.batch_size}")
    
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Union operation
print("\n[Test 5] DataProto union operation")
try:
    # Create sample data
    samples = DataProto(
        batch=TensorDict({
            'x': torch.randn(4, 20, 3),
            'categorical': torch.randn(4, 20, 10),
            'nodesxsample': torch.tensor([15, 18, 12, 20]),
        }, batch_size=4),
        meta_info={}
    )
    
    # Create reward data
    rewards = DataProto(
        batch=TensorDict({
            'rewards': torch.randn(4),
        }, batch_size=4),
        meta_info={}
    )
    
    print(f"✓ Created samples and rewards")
    print(f"  samples keys: {list(samples.batch.keys())}")
    print(f"  rewards keys: {list(rewards.batch.keys())}")
    
    # Test union
    combined = samples.union(rewards)
    print(f"✓ Union works: combined keys={list(combined.batch.keys())}")
    print(f"  Combined batch_size: {combined.batch.batch_size}")
    
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)
