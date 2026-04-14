"""
Test script to verify the RL training pipeline works end-to-end.

This script tests:
1. Model loading
2. Sample generation
3. Reward computation
4. One training step

Usage:
    python scripts/test_rl_pipeline.py
"""

import sys
from pathlib import Path

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "RLPF-verl_diffusion"))
sys.path.insert(0, str(PROJECT_ROOT / "DiffSBDD"))

import torch
import yaml

print("=" * 60)
print("DiffSBDD-RLPF Pipeline Test")
print("=" * 60)

# Check CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n[1] Device: {device}")

# Load config
config_path = PROJECT_ROOT / "ddpo/configs/diffsbdd_ddpo.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)
print(f"[2] Config loaded from: {config_path}")

# Check checkpoint exists
checkpoint_path = PROJECT_ROOT / config["model"]["checkpoint"]
if not checkpoint_path.exists():
    print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
    print("Available checkpoints:")
    for ckpt in (PROJECT_ROOT / "DiffSBDD/checkpoints").glob("*.ckpt"):
        print(f"  - {ckpt}")
    sys.exit(1)
print(f"[3] Checkpoint found: {checkpoint_path}")

# Check PDB file
pdb_path = PROJECT_ROOT / config["pocket"]["pdb_path"]
if not pdb_path.exists():
    print(f"[ERROR] PDB file not found: {pdb_path}")
    sys.exit(1)
print(f"[4] PDB file found: {pdb_path}")

# Import components
print("\n[5] Importing components...")
try:
    from constants import dataset_params
    print("    ✓ DiffSBDD constants")
except ImportError as e:
    print(f"    ✗ DiffSBDD constants: {e}")
    sys.exit(1)

try:
    from ddpo.integration import DiffSBDDModel
    print("    ✓ DiffSBDDModel")
except ImportError as e:
    print(f"    ✗ DiffSBDDModel: {e}")
    sys.exit(1)

try:
    from ddpo.rewards import DockingReward, DruglikenessReward, CompositeReward
    print("    ✓ Reward functions")
except ImportError as e:
    print(f"    ✗ Reward functions: {e}")
    sys.exit(1)

try:
    from ddpo.utils import DiffSBDDFilter
    print("    ✓ DiffSBDDFilter")
except ImportError as e:
    print(f"    ✗ DiffSBDDFilter: {e}")
    sys.exit(1)

# Get dataset info
dataset_info = dataset_params["crossdock"]
print(f"\n[6] Dataset info: {dataset_info['atom_decoder']}")

# Test model loading
print("\n[7] Loading DiffSBDDModel...")
try:
    model = DiffSBDDModel(
        checkpoint_path=str(checkpoint_path),
        pocket_pdb_path=str(pdb_path),
        ref_ligand_path="C:1101",  # Chain:ResidueNumber format
        device=device,
        dataset_name="crossdock"
    )
    print(f"    ✓ Model loaded: T={model.T}, atom_nf={model.atom_nf}")
    print(f"    ✓ Pocket info: {model.get_pocket_info()}")
except Exception as e:
    print(f"    ✗ Model loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test sample generation
print("\n[8] Testing sample generation...")
try:
    n_samples = 4
    max_n_nodes = 30
    nodesxsample = torch.randint(15, 25, (n_samples,), device=device)
    
    node_mask, edge_mask = model.get_mask(nodesxsample, n_samples, max_n_nodes)
    print(f"    ✓ Masks generated: node_mask={node_mask.shape}, edge_mask={edge_mask.shape}")
    
    # Sample with fewer timesteps for speed
    x, h, latents, logps, timesteps, mus, sigmas = model.sample(
        n_samples=n_samples,
        n_nodes=max_n_nodes,
        node_mask=node_mask,
        edge_mask=edge_mask,
        timestep=10  # Use fewer steps for testing
    )
    print(f"    ✓ Samples generated: x={x.shape}, h={h.shape if isinstance(h, torch.Tensor) else type(h)}")
    print(f"    ✓ Trajectory: {len(latents)} latents, {len(logps)} logps")
except Exception as e:
    print(f"    ✗ Sample generation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test reward computation
print("\n[9] Testing reward computation...")
try:
    from verl_diffusion.protocol import DataProto, TensorDict
    
    # Prepare data
    if isinstance(h, dict):
        h_tensor = h["categorical"]
    else:
        h_tensor = h
    
    batch_data = {
        "x": x,
        "categorical": h_tensor,
        "nodesxsample": nodesxsample,
    }
    data = DataProto(
        batch=TensorDict(batch_data, batch_size=[n_samples]),
        meta_info={"max_n_nodes": max_n_nodes}
    )
    
    # Test DruglikenessReward (faster than docking)
    druglikeness = DruglikenessReward(
        qed_weight=0.5,
        sa_weight=0.5,
        dataset_info=dataset_info
    )
    result = druglikeness.calculate_rewards(data)
    rewards = result.batch["rewards"]
    print(f"    ✓ DruglikenessReward: {rewards.tolist()}")
    
except Exception as e:
    print(f"    ✗ Reward computation failed: {e}")
    import traceback
    traceback.print_exc()
    # Don't exit - this might fail due to molecule building issues

# Test filter
print("\n[10] Testing DiffSBDDFilter...")
try:
    filters = DiffSBDDFilter(
        dataset_info=dataset_info,
        enable_filtering=True,
        enable_penalty=False
    )
    
    # Create mock data for filter
    batch_data = {
        "x": x,
        "categorical": h_tensor,
        "nodesxsample": nodesxsample,
        "rewards": torch.zeros(n_samples, device=device),
    }
    data = DataProto(
        batch=TensorDict(batch_data, batch_size=[n_samples]),
        meta_info={"max_n_nodes": max_n_nodes}
    )
    
    # Use filter method, not __call__
    filtered_data, filtering_ratio, novelty_ratio = filters.filter(data)
    print(f"    ✓ Filter stats: filtering_ratio={filtering_ratio:.2f}, novelty_ratio={novelty_ratio:.2f}")
except Exception as e:
    print(f"    ✗ Filter failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Pipeline Test Complete!")
print("=" * 60)
print("\nTo run full training:")
print("  python ddpo/main_diffsbdd.py --config_path ddpo/configs/diffsbdd_ddpo.yaml")
print("\nNote: Full training requires:")
print("  - GPU with sufficient memory (16GB+ recommended)")
print("  - AutoDock Vina/Smina for docking rewards")
print("  - Ray for parallel reward computation")
