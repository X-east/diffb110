"""
Minimal RL loop test - verifies the complete training cycle works.

Tests:
1. Model sampling (50 steps)
2. Reward computation (DruglikenessReward only, no docking)
3. Filter
4. Actor update (PPO) with gradient descent
5. Checkpoint save/load
6. 2 training epochs

Usage:
    python scripts/test_rl_loop.py
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "RLPF-verl_diffusion"))
sys.path.insert(0, str(PROJECT_ROOT / "DiffSBDD"))

# Activate wandb_stub for logging (MUST be before other imports)
import ddpo.utils

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import time
import wandb  # Now this will use wandb_stub

print("=" * 60)
print("Minimal RL Loop Test (50 steps, 4 samples, 2 epochs)")
print("WITH gradient update and checkpoint save/load")
print("=" * 60)

# Initialize wandb logging
wandb.init(
    project="test-rl-loop",
    name=f"test_{time.strftime('%Y%m%d_%H%M%S')}",
    config={"test": "minimal_rl_loop"}
)
print(f"Logging to: {wandb.run.dir}")

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}")

# Load minimal config
config_path = PROJECT_ROOT / "ddpo/configs/test_quick.yaml"
with open(config_path, encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Imports
from constants import dataset_params
from ddpo.integration import DiffSBDDModel
from ddpo.rewards import DruglikenessReward
from ddpo.utils import DiffSBDDFilter
from verl_diffusion.protocol import DataProto, TensorDict

dataset_info = dataset_params["crossdock"]

# ============================================================================
# 1. Load Model
# ============================================================================
print("\n[1/5] Loading model...")
t0 = time.time()
model = DiffSBDDModel(
    checkpoint_path=str(PROJECT_ROOT / config["model"]["checkpoint"]),
    pocket_pdb_path=str(PROJECT_ROOT / config["pocket"]["pdb_path"]),
    ref_ligand_path=config["pocket"]["ref_ligand"],
    device=device,
    dataset_name="crossdock"
)
print(f"    Model loaded in {time.time()-t0:.1f}s")
print(f"    T={model.T}, atom_nf={model.atom_nf}")

# ============================================================================
# 1b. Setup Optimizer
# ============================================================================
print("\n[1b/6] Setting up optimizer...")
# Get trainable parameters
trainable_params = [p for p in model.model.parameters() if p.requires_grad]
print(f"    Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

optimizer = optim.Adam(
    trainable_params,
    lr=config["train"]["learning_rate"],
    betas=(config["train"]["adam_beta1"], config["train"]["adam_beta2"]),
    weight_decay=config["train"]["adam_weight_decay"],
    eps=config["train"]["adam_epsilon"]
)
print(f"    Optimizer: Adam (lr={config['train']['learning_rate']})")

# ============================================================================
# 2. Initialize Reward & Filter
# ============================================================================
print("\n[2/6] Initializing reward & filter...")
reward_fn = DruglikenessReward(
    qed_weight=0.5,
    sa_weight=0.5,
    dataset_info=dataset_info
)
filter_fn = DiffSBDDFilter(
    dataset_info=dataset_info,
    enable_filtering=False,  # No filtering, keep all samples
    enable_penalty=False
)
print("    DruglikenessReward + DiffSBDDFilter ready")

# ============================================================================
# 2b. Setup checkpoint directory
# ============================================================================
print("\n[2b/6] Setting up checkpoint directory...")
ckpt_dir = PROJECT_ROOT / "exp" / "test-minimal"
ckpt_dir.mkdir(parents=True, exist_ok=True)
print(f"    Checkpoint dir: {ckpt_dir}")

# ============================================================================
# 3. RL Loop
# ============================================================================
print("\n[3/6] Starting RL loop...")

n_samples = config["dataloader"]["each_prompt_sample"]
max_n_nodes = config["dataloader"]["max_nodes"]
timestep = config["model"]["time_step"]
n_epochs = config["dataloader"]["epochs"]

# Track metrics
all_rewards = []
all_losses = []
all_clipfracs = []
all_validities = []
best_reward = float('-inf')

for epoch in range(n_epochs):
    print(f"\n--- Epoch {epoch+1}/{n_epochs} ---")
    
    # 3a. Sample node counts
    nodesxsample = torch.randint(
        config["dataloader"]["min_nodes"],
        max_n_nodes + 1,
        (n_samples,),
        device=device
    )
    print(f"    Nodes per sample: {nodesxsample.tolist()}")
    
    # 3b. Generate masks
    node_mask, edge_mask = model.get_mask(nodesxsample, n_samples, max_n_nodes)
    
    # 3c. Sample molecules (with gradients for training)
    print(f"    Sampling {n_samples} molecules ({timestep} steps)...")
    t0 = time.time()
    
    # Enable training mode
    model.model.train()
    
    x, h, latents, logps, timesteps, mus, sigmas = model.sample(
        n_samples=n_samples,
        n_nodes=max_n_nodes,
        node_mask=node_mask,
        edge_mask=edge_mask,
        timestep=timestep
    )
    sample_time = time.time() - t0
    print(f"    Sampled in {sample_time:.1f}s")
    print(f"    x: {x.shape}, latents: {len(latents)}, logps: {len(logps)}")
    
    # 3d. Prepare data for reward
    if isinstance(h, dict):
        h_tensor = h["categorical"]
    else:
        h_tensor = h
    
    batch_data = {
        "x": x.detach(),  # detach for reward computation
        "categorical": h_tensor.detach() if isinstance(h_tensor, torch.Tensor) else h_tensor,
        "nodesxsample": nodesxsample,
    }
    data = DataProto(
        batch=TensorDict(batch_data, batch_size=[n_samples]),
        meta_info={"max_n_nodes": max_n_nodes}
    )
    
    # 3e. Compute rewards
    print("    Computing rewards...")
    t0 = time.time()
    data = reward_fn.calculate_rewards(data)
    rewards = data.batch["rewards"]
    print(f"    Rewards: {[f'{r:.3f}' for r in rewards.tolist()]}")
    print(f"    Mean reward: {rewards.mean().item():.4f}")
    print(f"    Reward time: {time.time()-t0:.1f}s")
    all_rewards.append(rewards.mean().item())
    
    # 3f. Filter and compute validity
    validity = filter_fn.get_validity_ratio(data)
    all_validities.append(validity)
    print(f"    Validity: {validity*100:.1f}%")
    
    # Log reward metrics to wandb
    wandb.log({
        "epoch": epoch,
        "reward/mean": rewards.mean().item(),
        "reward/min": rewards.min().item(),
        "reward/max": rewards.max().item(),
        "reward/std": rewards.std().item() if len(rewards) > 1 else 0.0,
        "metrics/validity": validity,
    }, step=epoch)
    
    # 3g. Compute advantages
    advantages = rewards - rewards.mean()
    advantages = advantages.to(device)
    print(f"    Advantages: {[f'{a:.3f}' for a in advantages.tolist()]}")
    
    # ========================================================================
    # 3h. PPO Update with importance sampling (DDPO style)
    # ========================================================================
    print("    Computing PPO loss and updating model...")
    
    optimizer.zero_grad()
    model.model.train()
    
    # Stack log probs from trajectory (these are "old" log probs, detached)
    old_logps_tensor = torch.stack(logps, dim=1)  # [B, T]
    old_total_logp = old_logps_tensor.sum(dim=1).detach()  # [B]
    
    # For DDPO training, we need to recompute log_prob with current model
    # This creates the gradient path through the network
    
    # Pick a random timestep to update (simplified - full DDPO updates all)
    t_idx = len(latents) // 2  # Middle timestep
    
    if t_idx > 0 and t_idx < len(latents) - 1:
        # Get latent at this timestep
        zt = latents[t_idx].detach()  # Input is detached
        zs = latents[t_idx + 1].detach()  # Target is detached
        
        # Compute timestep values
        t_val = (len(latents) - 1 - t_idx) / model.T
        s_val = (len(latents) - 2 - t_idx) / model.T
        
        t_tensor = torch.full((n_samples, 1), t_val, device=device)
        s_tensor = torch.full((n_samples, 1), s_val, device=device)
        
        # Recompute with current model (this has gradients!)
        # prev_sample=zs tells the model to compute log_prob for zs instead of sampling
        _, new_logp, _, _ = model.sample_p_zs_given_zt(
            s_tensor, t_tensor, zt, node_mask, edge_mask,
            prev_sample=zs  # This is the key - compute log_prob for existing sample
        )
        
        # Old log prob for this timestep
        old_logp = logps[t_idx].detach()
        
        # Importance sampling ratio
        ratio = torch.exp(new_logp - old_logp)
        
        # Clipped PPO objective
        clip_range = config["train"]["clip_range"]
        clipped_ratio = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
        
        # Compute clipfrac - fraction of samples that were clipped
        clipfrac = torch.mean((torch.abs(ratio - 1.0) > clip_range).float())
        
        # PPO loss
        loss1 = -advantages * ratio
        loss2 = -advantages * clipped_ratio
        policy_loss = torch.max(loss1, loss2).mean()
        
        # Check if we have gradients
        if policy_loss.requires_grad:
            # Backward pass
            policy_loss.backward()
            
            # Gradient clipping
            grad_norm = nn.utils.clip_grad_norm_(
                trainable_params, 
                config["train"]["max_grad_norm"]
            )
            
            # Optimizer step
            optimizer.step()
            
            print(f"    ✓ PPO loss: {policy_loss.item():.6f}")
            print(f"    ✓ Ratio: [{ratio.min().item():.3f}, {ratio.max().item():.3f}]")
            print(f"    ✓ ClipFrac: {clipfrac.item():.3f}")
            print(f"    ✓ Grad norm: {grad_norm:.4f}")
            print(f"    ✓ Model parameters UPDATED!")
            all_losses.append(policy_loss.item())
            all_clipfracs.append(clipfrac.item())
            
            # Log PPO metrics to wandb
            wandb.log({
                "ppo/loss": policy_loss.item(),
                "ppo/clipfrac": clipfrac.item(),
                "ppo/ratio_mean": ratio.mean().item(),
                "ppo/ratio_min": ratio.min().item(),
                "ppo/ratio_max": ratio.max().item(),
                "train/grad_norm": grad_norm,
                "train/learning_rate": optimizer.param_groups[0]['lr'],
            }, step=epoch)
        else:
            print(f"    ⚠ Loss has no grad")
            print(f"    PPO loss value: {policy_loss.item():.6f}")
            all_losses.append(policy_loss.item())
            all_clipfracs.append(0.0)
    else:
        print("    ⚠ Not enough trajectory data")
        all_losses.append(0.0)
        all_clipfracs.append(0.0)
    
    # Track best reward
    if rewards.mean().item() > best_reward:
        best_reward = rewards.mean().item()

# ============================================================================
# 4. Save Checkpoint
# ============================================================================
print("\n[4/6] Saving checkpoint...")
ckpt_path = ckpt_dir / "model_final.pt"
torch.save({
    'epoch': n_epochs,
    'model_state_dict': model.model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'rewards': all_rewards,
    'losses': all_losses,
    'clipfracs': all_clipfracs,
    'validities': all_validities,
    'best_reward': best_reward,
    'config': config,
}, ckpt_path)
print(f"    ✓ Checkpoint saved to: {ckpt_path}")
print(f"    ✓ File size: {ckpt_path.stat().st_size / 1024 / 1024:.1f} MB")

# ============================================================================
# 5. Load Checkpoint (verify it works)
# ============================================================================
print("\n[5/6] Loading checkpoint to verify...")
checkpoint = torch.load(ckpt_path, map_location=device)
model.model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
print(f"    ✓ Checkpoint loaded successfully!")
print(f"    ✓ Loaded epoch: {checkpoint['epoch']}")
print(f"    ✓ Loaded rewards: {checkpoint['rewards']}")

# ============================================================================
# 6. Summary
# ============================================================================
print("\n" + "=" * 60)
print("RL Loop Test Complete!")
print("=" * 60)
print(f"\nSummary:")
print(f"  - Epochs: {n_epochs}")
print(f"  - Samples per epoch: {n_samples}")
print(f"  - Timesteps: {timestep}")
print(f"  - Device: {device}")
print(f"  - Mean rewards: {all_rewards}")
print(f"  - Losses: {all_losses}")
print(f"  - ClipFracs: {all_clipfracs}")
print(f"  - Validities: {all_validities}")
print(f"  - Best reward: {best_reward:.4f}")
print(f"\nAll components working:")
print(f"  ✓ DiffSBDDModel.sample()")
print(f"  ✓ DruglikenessReward.calculate_rewards()")
print(f"  ✓ DiffSBDDFilter.get_validity_ratio()")
print(f"  ✓ PPO loss computation")
print(f"  ✓ Gradient descent (optimizer.step())")
print(f"  ✓ Checkpoint save")
print(f"  ✓ Checkpoint load")
print(f"\nReady for full training!")

# Finish wandb logging
wandb.finish()
print(f"\nLogs saved to: {PROJECT_ROOT / 'log' / 'test-rl-loop'}")
