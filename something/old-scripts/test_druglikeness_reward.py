#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test DruglikenessReward using generated molecules.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np


def main():
    print("\n" + "="*60)
    print("  DRUGLIKENESS REWARD TEST")
    print("="*60)
    
    # Load generated molecule data
    batch_dir = Path("generated_molecules/batch_data")
    if not batch_dir.exists():
        print("Error: Run generate_and_visualize.py first!")
        return
    
    print("\n[1] Loading generated molecules...", flush=True)
    x_final = np.load(batch_dir / "x_final.npy")
    h_categorical = np.load(batch_dir / "h_categorical.npy")
    node_counts = np.load(batch_dir / "node_counts.npy")
    
    print(f"    Loaded {len(node_counts)} molecules")
    
    # Convert to tensors
    x = torch.from_numpy(x_final).float()
    h = torch.from_numpy(h_categorical).float()
    nodesxsample = torch.from_numpy(node_counts).long()
    
    # Create mock DataProto
    class MockDataProto:
        def __init__(self):
            self.batch = {}
    
    data = MockDataProto()
    data.batch = {
        'x': x,
        'categorical': h,
        'nodesxsample': nodesxsample,
    }
    
    # Initialize DruglikenessReward
    print("\n[2] Initializing DruglikenessReward...", flush=True)
    from ddpo.rewards.druglikeness import DruglikenessReward
    
    reward_fn = DruglikenessReward(
        qed_weight=0.5,
        sa_weight=0.5,
    )
    print(f"    {reward_fn}")
    
    # Calculate rewards
    print("\n[3] Calculating drug-likeness rewards...", flush=True)
    result = reward_fn.calculate_rewards(data)
    
    rewards = result.batch['rewards']
    qed_scores = result.batch['qed_scores']
    sa_scores = result.batch['sa_scores']
    
    # Print results
    print("\n[4] Results:")
    print("-" * 50)
    print(f"{'Mol':<5} {'QED':<8} {'SA':<8} {'Reward':<8}")
    print("-" * 50)
    
    for i in range(len(rewards)):
        print(f"{i+1:<5} {qed_scores[i].item():<8.4f} {sa_scores[i].item():<8.4f} {rewards[i].item():<8.4f}")
    
    print("-" * 50)
    print(f"Mean QED:    {qed_scores.mean().item():.4f}")
    print(f"Mean SA:     {sa_scores.mean().item():.4f}")
    print(f"Mean Reward: {rewards.mean().item():.4f}")
    
    # Check success
    n_valid = (rewards > 0).sum().item()
    print(f"\nValid molecules: {n_valid}/{len(rewards)}")
    
    print("\n" + "="*60)
    print("  TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
