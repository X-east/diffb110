"""Gaussian reverse-transition scores for DiffSBDD continuous states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


Reduction = Literal["sum", "mean_per_atom", "mean_per_element"]


@dataclass(frozen=True)
class TransitionScore:
    """Total and component-wise Gaussian log-probability contributions."""

    total: torch.Tensor
    coordinate: torch.Tensor
    atom_feature: torch.Tensor


def _broadcast_sigma(sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Broadcast a scalar or batch-wise standard deviation to ``target``."""
    sigma_value = torch.as_tensor(sigma, device=target.device, dtype=target.dtype)
    while sigma_value.ndim < target.ndim:
        sigma_value = sigma_value.unsqueeze(-1)
    return torch.broadcast_to(sigma_value, target.shape)


def _reduce(values: torch.Tensor, mask: torch.Tensor, reduction: Reduction) -> torch.Tensor:
    """Apply a documented reduction over node-feature values."""
    masked = values * mask
    summed = masked.sum(dim=tuple(range(1, masked.ndim)))
    if reduction == "sum":
        return summed
    if reduction == "mean_per_atom":
        free = (mask.any(dim=-1).float()).sum(dim=tuple(range(1, mask.ndim - 1)))
    else:
        free = mask.sum(dim=tuple(range(1, mask.ndim)))
    return summed / free.clamp_min(1.0)


def gaussian_transition_score(
    z_s: torch.Tensor,
    mean: torch.Tensor,
    sigma: torch.Tensor,
    node_mask: torch.Tensor,
    *,
    reduction: Reduction = "sum",
    include_normal_constant: bool = True,
    include_sigma_term: bool = True,
) -> TransitionScore:
    """Compute an exact diagonal-Gaussian transition score.

    ``z_s`` is detached because saved trajectory states are observations;
    gradients flow only through ``mean`` and ``sigma`` produced by the model.
    The final decoder transition ``p(x,h|z_0)`` is outside this function.
    """
    if z_s.shape != mean.shape:
        raise ValueError(f"z_s and mean shapes differ: {z_s.shape} vs {mean.shape}")
    if z_s.ndim != 3 or z_s.shape[-1] < 4:
        raise ValueError(f"expected [B,N,D] with D>=4, got {z_s.shape}")
    if node_mask.shape not in {(z_s.shape[0], z_s.shape[1], 1), z_s.shape}:
        raise ValueError(f"invalid node_mask shape {node_mask.shape} for {z_s.shape}")
    expanded_mask = node_mask.expand_as(z_s) if node_mask.shape[-1] == 1 else node_mask
    sigma_full = _broadcast_sigma(sigma, z_s).clamp_min(torch.finfo(z_s.dtype).tiny)
    if torch.any(sigma_full <= torch.finfo(z_s.dtype).tiny):
        raise ValueError("Gaussian transition score received a degenerate sigma")
    residual = (z_s.detach() - mean).square()
    values = -0.5 * residual / sigma_full.square()
    if include_sigma_term:
        values = values - torch.log(sigma_full)
    if include_normal_constant:
        values = values - 0.5 * torch.log(
            torch.as_tensor(2.0 * torch.pi, device=z_s.device, dtype=z_s.dtype)
        )
    coordinate = _reduce(values[..., :3], expanded_mask[..., :3], reduction)
    atom_feature = _reduce(values[..., 3:], expanded_mask[..., 3:], reduction)
    return TransitionScore(
        total=coordinate + atom_feature,
        coordinate=coordinate,
        atom_feature=atom_feature,
    )
