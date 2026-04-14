#!/usr/bin/env python
"""
QED + SA only RL smoke test.

Purpose
-------
Run a minimal reinforcement-learning-style smoke test for the current
DiffSBDD + DDPO integration while using only DruglikenessReward
(QED + SA) and no docking backend.

What it tests
-------------
1. Load DiffSBDD checkpoint and pocket.
2. Sample a small batch of molecules.
3. Compute QED + SA reward only.
4. Measure validity with DiffSBDDFilter.
5. Perform one minimal PPO-style gradient update using a single diffusion step.

Notes
-----
- This is a smoke test, not a production trainer.
- It intentionally keeps the update small and local.
- Reward semantics are unchanged: reward comes only from QED + SA.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.optim as optim
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "RLPF-verl_diffusion"))
sys.path.insert(0, str(PROJECT_ROOT / "DiffSBDD"))

# Must happen before imports that may import wandb
import ddpo.utils  # noqa: F401

from constants import dataset_params
from ddpo.integration import DiffSBDDModel
from ddpo.rewards import DruglikenessReward
from ddpo.utils import DiffSBDDFilter
from verl_diffusion.protocol import DataProto, TensorDict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QED+SA only RL smoke test")
    parser.add_argument(
        "--config_path",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "ddp_gpu_druglikeness_only.yaml"),
        help="Path to YAML config",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use: cuda or cpu",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="Number of molecules to sample in the smoke test",
    )
    parser.add_argument(
        "--min_nodes",
        type=int,
        default=None,
        help="Override minimum node count",
    )
    parser.add_argument(
        "--max_nodes",
        type=int,
        default=20,
        help="Maximum node count / padding size for the smoke test",
    )
    parser.add_argument(
        "--timestep",
        type=int,
        default=20,
        help="Number of reverse diffusion steps for sampling",
    )
    parser.add_argument(
        "--ppo_clip",
        type=float,
        default=0.2,
        help="PPO clip epsilon for the smoke update",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(device_str: str) -> str:
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU")
        return "cpu"
    return device_str


def build_dataproto(
    x: torch.Tensor,
    h: torch.Tensor,
    nodesxsample: torch.Tensor,
    max_n_nodes: int,
) -> DataProto:
    batch_data = {
        "x": x.detach(),
        "categorical": h.detach(),
        "nodesxsample": nodesxsample,
    }
    return DataProto(
        batch=TensorDict(batch_data, batch_size=[x.shape[0]]),
        meta_info={"max_n_nodes": max_n_nodes},
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    config = load_config(args.config_path)
    device = resolve_device(args.device)
    dataset_info = dataset_params["crossdock"]

    checkpoint_path = PROJECT_ROOT / config["model"]["checkpoint"]
    pocket_pdb_path = PROJECT_ROOT / config["pocket"]["pdb_path"]
    ref_ligand = config["pocket"]["ref_ligand"]

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not pocket_pdb_path.exists():
        raise FileNotFoundError(f"Pocket PDB not found: {pocket_pdb_path}")

    max_n_nodes = args.max_nodes
    min_nodes = args.min_nodes
    if min_nodes is None:
        min_nodes = min(config["dataloader"].get("min_nodes", 10), max_n_nodes)
    if min_nodes > max_n_nodes:
        raise ValueError("min_nodes cannot be greater than max_nodes")

    print("=" * 70)
    print("QED + SA Only RL Smoke Test")
    print("=" * 70)
    print(f"Config:      {args.config_path}")
    print(f"Device:      {device}")
    print(f"Checkpoint:  {checkpoint_path}")
    print(f"Pocket PDB:  {pocket_pdb_path}")
    print(f"Ref ligand:  {ref_ligand}")
    print(f"Samples:     {args.n_samples}")
    print(f"Node range:  [{min_nodes}, {max_n_nodes}]")
    print(f"Timesteps:   {args.timestep}")
    print()

    print("[1/5] Loading DiffSBDD model...")
    model = DiffSBDDModel(
        checkpoint_path=str(checkpoint_path),
        pocket_pdb_path=str(pocket_pdb_path),
        ref_ligand_path=ref_ligand,
        device=device,
        dataset_name="crossdock",
    )
    print(f"  Loaded model: T={model.T}, atom_nf={model.atom_nf}")
    print(f"  Pocket info: {model.get_pocket_info()}")

    print("[2/5] Building QED+SA reward and validity filter...")
    reward_fn = DruglikenessReward(
        qed_weight=0.5,
        sa_weight=0.5,
        dataset_info=dataset_info,
    )
    filter_fn = DiffSBDDFilter(
        dataset_info=dataset_info,
        enable_filtering=False,
        enable_penalty=False,
    )
    print(f"  Reward: {reward_fn}")
    print(f"  Filter: {filter_fn}")

    print("[3/5] Sampling molecules...")
    nodesxsample = torch.randint(
        low=min_nodes,
        high=max_n_nodes + 1,
        size=(args.n_samples,),
        device=device,
    )
    node_mask, edge_mask = model.get_mask(nodesxsample, args.n_samples, max_n_nodes)

    x, h, latents, logps, timesteps, mus, sigmas = model.sample(
        n_samples=args.n_samples,
        n_nodes=max_n_nodes,
        node_mask=node_mask.to(device),
        edge_mask=edge_mask.to(device),
        timestep=args.timestep,
    )

    if isinstance(h, dict):
        h_categorical = h["categorical"]
    else:
        h_categorical = h

    print(f"  x shape:          {tuple(x.shape)}")
    print(f"  h shape:          {tuple(h_categorical.shape)}")
    print(f"  trajectory steps: {len(latents)} latents / {len(logps)} logps")
    print(f"  nodesxsample:     {nodesxsample.tolist()}")

    print("[4/5] Computing QED+SA rewards and validity...")
    data = build_dataproto(x, h_categorical, nodesxsample, max_n_nodes)
    data = reward_fn.calculate_rewards(data)
    rewards = data.batch["rewards"]
    qed_scores = data.batch["qed_scores"]
    sa_scores = data.batch["sa_scores"]
    validity = filter_fn.get_validity_ratio(data)

    print("  Per-sample metrics:")
    for i in range(args.n_samples):
        print(
            "   "
            f"sample={i:02d} "
            f"nodes={int(nodesxsample[i].item()):02d} "
            f"qed={float(qed_scores[i]):.4f} "
            f"sa_raw={float(sa_scores[i]):.4f} "
            f"reward={float(rewards[i]):.4f}"
        )

    print(f"  Mean reward: {float(rewards.mean()):.4f}")
    print(f"  Mean QED:    {float(qed_scores.mean()):.4f}")
    print(f"  Mean SA raw: {float(sa_scores.mean()):.4f}")
    print(f"  Validity:    {validity * 100:.2f}%")

    print("[5/5] Running one minimal PPO-style update...")
    trainable_params = [p for p in model.model.parameters() if p.requires_grad]
    optimizer = optim.Adam(
        trainable_params,
        lr=float(config["train"].get("learning_rate", 5.0e-4)),
        betas=(
            float(config["train"].get("adam_beta1", 0.9)),
            float(config["train"].get("adam_beta2", 0.999)),
        ),
        weight_decay=float(config["train"].get("adam_weight_decay", 1.0e-4)),
        eps=float(config["train"].get("adam_epsilon", 1.0e-8)),
    )

    if len(latents) < 3 or len(logps) < 2:
        raise RuntimeError(
            "Trajectory too short for PPO-style smoke update; increase --timestep"
        )

    # Use an interior diffusion step to recompute a transition log-prob.
    t_idx = max(1, len(logps) // 2)
    zt = latents[t_idx].detach()
    zs = latents[t_idx + 1].detach()

    total_steps = len(latents) - 1
    t_val = (total_steps - t_idx) / float(model.T)
    s_val = (total_steps - t_idx - 1) / float(model.T)
    t_tensor = torch.full((args.n_samples, 1), t_val, device=device)
    s_tensor = torch.full((args.n_samples, 1), s_val, device=device)

    model.model.train()
    optimizer.zero_grad()

    _, new_logp, _, _ = model.sample_p_zs_given_zt(
        s=s_tensor,
        t=t_tensor,
        zt=zt,
        node_mask=node_mask.to(device),
        edge_mask=edge_mask.to(device),
        prev_sample=zs,
    )

    old_logp = logps[t_idx].detach()
    advantages = rewards.detach() - rewards.detach().mean()
    advantages = advantages / (advantages.std(unbiased=False) + 1e-8)

    ratio = torch.exp(new_logp - old_logp)
    clipped_ratio = torch.clamp(ratio, 1.0 - args.ppo_clip, 1.0 + args.ppo_clip)
    surrogate_1 = ratio * advantages
    surrogate_2 = clipped_ratio * advantages
    loss = -torch.min(surrogate_1, surrogate_2).mean()

    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite PPO loss: {float(loss)}")

    loss.backward()

    grad_sq_sum = 0.0
    for param in trainable_params:
        if param.grad is not None:
            grad_sq_sum += float(param.grad.detach().pow(2).sum().item())
    grad_norm = grad_sq_sum ** 0.5

    optimizer.step()

    clip_frac = ((ratio - clipped_ratio).abs() > 1e-8).float().mean().item()

    print(f"  old_logp mean: {float(old_logp.mean()):.6f}")
    print(f"  new_logp mean: {float(new_logp.mean()):.6f}")
    print(f"  ratio mean:    {float(ratio.mean()):.6f}")
    print(f"  clip fraction: {clip_frac:.6f}")
    print(f"  grad norm:     {grad_norm:.6f}")
    print(f"  PPO loss:      {float(loss):.6f}")

    print()
    print("Smoke test passed: sampling, QED+SA reward, validity, and one RL update all succeeded.")


if __name__ == "__main__":
    main()
