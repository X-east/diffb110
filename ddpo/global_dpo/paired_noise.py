"""DiffSBDD-native paired endpoint noising for Global DPO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .endpoint_data import EndpointSample


@dataclass(frozen=True)
class GlobalNoisyBatch:
    """One paired noisy endpoint batch in DiffSBDD flat tensor format."""

    z_t: torch.Tensor
    epsilon: torch.Tensor
    xh0_pocket: torch.Tensor
    ligand_mask: torch.Tensor
    pocket_mask: torch.Tensor
    timestep: torch.Tensor
    sizes: torch.Tensor


def _paired_noise(
    sizes: torch.Tensor,
    feature_dim: int,
    *,
    device: torch.device,
    generator: torch.Generator,
    coupling: Literal["common_random", "independent"],
) -> torch.Tensor:
    """Create endpoint noise using AliDiff-compatible or paired semantics."""
    epsilon = torch.randn(
        int(sizes.sum().item()), feature_dim, device=device, generator=generator
    )
    if coupling == "independent":
        return epsilon
    if coupling != "common_random":
        raise ValueError(f"unsupported noise coupling: {coupling}")
    offset = 0
    for pair_index in range(0, len(sizes), 2):
        if pair_index + 1 >= len(sizes):
            raise ValueError("endpoint batch must contain complete winner/loser pairs")
        left_nodes = int(sizes[pair_index].item())
        right_nodes = int(sizes[pair_index + 1].item())
        if left_nodes != right_nodes:
            raise ValueError(
                "common_random noise requires equal winner/loser node counts; "
                "use coupling='independent' for AliDiff-style variable-size pairs"
            )
        epsilon[offset + left_nodes : offset + 2 * left_nodes] = epsilon[offset : offset + left_nodes]
        offset += 2 * left_nodes
    return epsilon


def build_paired_noisy_batch(
    model: object,
    items: list[EndpointSample],
    timestep: int,
    *,
    timestep_grid: int = 1000,
    noise_seed: int,
    noise_coupling: Literal["common_random", "independent"] = "common_random",
) -> GlobalNoisyBatch:
    """Construct common-noise paired endpoints using DiffSBDD native semantics."""
    if not items or len(items) % 2 != 0:
        raise ValueError("items must contain one or more complete endpoint pairs")
    if not 1 <= timestep <= timestep_grid:
        raise ValueError(f"timestep={timestep} outside [1, {timestep_grid}]")
    device = torch.device(model.device)
    sizes = torch.tensor([item.n_nodes for item in items], device=device, dtype=torch.long)
    ligand_mask = torch.repeat_interleave(torch.arange(len(items), device=device), sizes).long()
    ligand = {
        "x": torch.cat([item.x for item in items], dim=0).to(device),
        "one_hot": torch.cat([item.v for item in items], dim=0).to(device),
        "size": sizes,
        "mask": ligand_mask,
    }
    pocket = model._replicate_pocket(len(items))
    ligand, pocket = model.ddpm.normalize(ligand, pocket)
    xh0_lig = torch.cat([ligand["x"], ligand["one_hot"]], dim=1)
    xh0_pocket = torch.cat([pocket["x"], pocket["one_hot"]], dim=1)
    lig_coords, pocket_coords = model.ddpm.remove_mean_batch(
        xh0_lig[:, : model.ddpm.n_dims],
        xh0_pocket[:, : model.ddpm.n_dims],
        ligand["mask"],
        pocket["mask"],
    )
    xh0_lig = torch.cat([lig_coords, xh0_lig[:, model.ddpm.n_dims :]], dim=1)
    xh0_pocket = torch.cat([pocket_coords, xh0_pocket[:, model.ddpm.n_dims :]], dim=1)
    t = torch.full(
        (len(items), 1),
        float(timestep) / float(timestep_grid),
        device=device,
        dtype=xh0_lig.dtype,
    )
    gamma_t = model.ddpm.inflate_batch_array(model.ddpm.gamma(t), ligand["x"])
    alpha_t = model.ddpm.alpha(gamma_t, xh0_lig)
    sigma_t = model.ddpm.sigma(gamma_t, xh0_lig)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(noise_seed))
    epsilon = _paired_noise(
        sizes,
        xh0_lig.shape[-1],
        device=device,
        generator=generator,
        coupling=noise_coupling,
    )
    z_t = alpha_t[ligand["mask"]] * xh0_lig + sigma_t[ligand["mask"]] * epsilon
    z_coords, pocket_coords = model.ddpm.remove_mean_batch(
        z_t[:, : model.ddpm.n_dims],
        xh0_pocket[:, : model.ddpm.n_dims],
        ligand["mask"],
        pocket["mask"],
    )
    z_t = torch.cat([z_coords, z_t[:, model.ddpm.n_dims :]], dim=1)
    xh0_pocket = torch.cat([pocket_coords, xh0_pocket[:, model.ddpm.n_dims :]], dim=1)
    return GlobalNoisyBatch(
        z_t=z_t,
        epsilon=epsilon,
        xh0_pocket=xh0_pocket,
        ligand_mask=ligand["mask"],
        pocket_mask=pocket["mask"],
        timestep=t,
        sizes=sizes,
    )
