from __future__ import annotations

import argparse
from typing import Any

import torch
import yaml

from ddpo.integration import DiffSBDDModel
from ddpo.rewards.normalization import RewardNormalizationConfig, RewardNormalizer
from ddpo.runtime.data_loader import DiffSBDDDataLoader
from ddpo.utils import DiffSBDDFilter
from verl_diffusion.worker.actor.edm_actor import EDMActor


def load_config(config_path: str) -> dict[str, Any]:
    """Load YAML configuration from disk."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)



def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Apply CLI overrides while preserving config as the source of truth."""
    if args.seed is not None:
        config.setdefault("train", {})["seed"] = args.seed
    return config



def resolve_device(config: dict[str, Any]) -> str:
    """Resolve runtime device from config with CUDA-aware fallback."""
    return config.get("model", {}).get(
        "device",
        "cuda" if torch.cuda.is_available() else "cpu",
    )



def save_config_snapshot(config: dict[str, Any], snapshot_path: str) -> None:
    """Persist the resolved config snapshot for reproducibility."""
    with open(snapshot_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)



def build_model(config: dict[str, Any], device: str, dataset_name: str) -> DiffSBDDModel:
    """Construct the DiffSBDD model wrapper."""
    return DiffSBDDModel(
        checkpoint_path=config["model"]["checkpoint"],
        pocket_pdb_path=config["pocket"]["pdb_path"],
        ref_ligand_path=config["pocket"]["ref_ligand"],
        device=device,
        dataset_name=dataset_name,
    )



def build_dataloader(
    config: dict[str, Any],
    dataset_info: dict[str, Any],
    device: str,
    model_impl=None,
) -> DiffSBDDDataLoader:
    """Construct the single-GPU prompt dataloader.

    Args:
        config: Full YAML config dict.
        dataset_info: Dataset constants (from ``dataset_params[dataset_name]``).
        device: Target device string (e.g. ``"cuda"``).
        model_impl: Optional :class:`DiffSBDDModel` instance.  When provided *and*
            the config requests ``node_sampling: size_distribution``, the
            loader will extract ``size_distribution`` and pocket-size tensor
            from the model so that ligand node counts are drawn from
            P(N_lig | N_pocket) instead of a uniform range.
    """
    size_dist = None
    pocket_size = None

    sampling_method = config.get("dataloader", {}).get("node_sampling", "uniform")
    if sampling_method == "size_distribution" and model_impl is not None:
        # Extract learned conditional distribution from the loaded checkpoint
        ddpm = getattr(model_impl, "ddpm", None)
        if ddpm is not None and hasattr(ddpm, "size_distribution"):
            size_dist = ddpm.size_distribution
        # Extract current pocket's residue count for conditioning
        pocket_single = getattr(model_impl, "_pocket_single", None)
        if pocket_single is not None and "size" in pocket_single:
            pocket_size = pocket_single["size"]  # shape [1] or [n_residues]

    return DiffSBDDDataLoader(
        config=config,
        dataset_info=dataset_info,
        device=device,
        num_batches=config.get("dataloader", {}).get("epochs"),
        size_distribution=size_dist,
        pocket_size_tensor=pocket_size,
    )



def build_filter(config: dict[str, Any], dataset_info: dict[str, Any]) -> DiffSBDDFilter:
    """Construct the canonical molecule filter."""
    return DiffSBDDFilter(
        dataset_info=dataset_info,
        enable_filtering=bool(config.get("filter", {}).get("enable_filtering", False)),
        enable_penalty=bool(config.get("filter", {}).get("enable_penalty", False)),
        novelty_penalty_coef=float(
            config.get("filter", {}).get("novelty_penalty_coef", 0.1)
        ),
    )



def build_actor(model, config: dict[str, Any], dataset_info: dict[str, Any], model_impl=None):
    """Construct the PPO actor and inject explicit max-node semantics."""
    actor = EDMActor(model, config, model_impl=model_impl)
    actor.max_n_nodes = config.get("dataloader", {}).get(
        "max_n_nodes",
        config.get("dataloader", {}).get("max_nodes", dataset_info.get("max_n_nodes", 50)),
    )
    return actor



def build_reward_normalizer(config: dict[str, Any], logger=None) -> RewardNormalizer:
    """Construct the YAML-driven total reward normalizer."""
    norm_cfg = RewardNormalizationConfig.from_config(config)
    return RewardNormalizer(config=norm_cfg, logger=logger)
