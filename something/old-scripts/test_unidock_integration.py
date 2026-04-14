#!/usr/bin/env python
"""
Test UniDock integration in the full RL training pipeline.
This script tests the complete workflow: sampling → filtering → docking → reward calculation.
"""

import sys
import torch
from pathlib import Path

# Add DiffSBDD to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "DiffSBDD"))
sys.path.insert(0, str(project_root))

from ddpo.integration.diffsbdd_model import DiffSBDDModel
from ddpo.rewards.docking import DockingReward
from ddpo.utils.diffsbdd_filter import DiffSBDDFilter
from constants import dataset_params

def test_unidock_integration():
    """Test the complete UniDock integration workflow."""
    
    print("=" * 70)
    print("UniDock Integration Test")
    print("=" * 70)
    print()
    
    # 1. Initialize model
    print("[1/5] Initializing DiffSBDDModel...")
    model = DiffSBDDModel(
        checkpoint_path="DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt",
        pocket_pdb_path="data/8GJJ-C-Cip.pdb",
        ref_ligand_path="C:1101",  # Reference ligand for pocket definition
        device="cuda:0"
    )
    print(f"✓ Model loaded: T={model.T}, atom_nf={model.atom_nf}")
    print()
    
    # 2. Sample molecules
    print("[2/5] Sampling molecules...")
    batch_size = 4  # Small batch for testing
    device = "cuda:0"
    
    # Prepare masks for sampling
    max_n_nodes = 30
    n_nodes = torch.randint(15, max_n_nodes, (batch_size,)).to(device)
    node_mask = torch.zeros(batch_size, max_n_nodes, 1).to(device)
    for i, n in enumerate(n_nodes):
        node_mask[i, :n, :] = 1
    edge_mask = torch.ones(batch_size * max_n_nodes * max_n_nodes, 1).to(device)
    
    # Sample
    x, h, latents, logps, timesteps, mus, sigmas = model.sample(
        n_samples=batch_size,
        n_nodes=n_nodes,
        node_mask=node_mask,
        edge_mask=edge_mask
    )
    
    # Create DataProto-like structure
    from verl_diffusion.protocol import DataProto
    from tensordict import TensorDict
    
    data = DataProto(batch=TensorDict({
        'x': x,
        'categorical': h['categorical'],
        'nodesxsample': n_nodes
    }, batch_size=[batch_size]))
    
    print(f"✓ Sampled {batch_size} molecules")
    print(f"  x shape: {data.batch['x'].shape}")
    print(f"  categorical shape: {data.batch['categorical'].shape}")
    print(f"  nodesxsample: {data.batch['nodesxsample']}")
    print()
    
    # 3. Filter molecules
    print("[3/5] Filtering molecules...")
    filter_fn = DiffSBDDFilter(dataset_info=dataset_params['crossdock'])
    data, filtering_ratio, novelty_penalty_ratio = filter_fn.filter(data)
    print(f"✓ Filtered molecules")
    print(f"  filtering_ratio: {filtering_ratio:.2f}")
    print(f"  novelty_penalty_ratio: {novelty_penalty_ratio:.2f}")
    print()
    
    # 4. Calculate rewards with UniDock
    print("[4/5] Calculating rewards with UniDock...")
    reward_fn = DockingReward(
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
        normalization_factor=-10.0,
        chunk_size=4
    )
    
    data = reward_fn.calculate_rewards(data)
    print(f"✓ Rewards calculated")
    print(f"  rewards shape: {data.batch['rewards'].shape}")
    print(f"  rewards: {data.batch['rewards']}")
    print(f"  mean reward: {data.batch['rewards'].mean().item():.4f}")
    print(f"  max reward: {data.batch['rewards'].max().item():.4f}")
    print(f"  min reward: {data.batch['rewards'].min().item():.4f}")
    print()
    
    # 5. Summary
    print("[5/5] Test Summary")
    print("=" * 70)
    print(f"✓ Model sampling: PASSED")
    print(f"✓ Molecule filtering: PASSED (ratio={filtering_ratio:.2f})")
    print(f"✓ UniDock docking: PASSED")
    print(f"✓ Reward calculation: PASSED")
    print()
    print("=" * 70)
    print("UniDock Integration Test: ALL TESTS PASSED")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    try:
        success = test_unidock_integration()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test FAILED with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
