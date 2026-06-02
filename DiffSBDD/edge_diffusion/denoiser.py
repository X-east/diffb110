"""Small edge denoiser modules for halfedge bond-type overfit experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from DiffSBDD.edge_diffusion.dataset import EdgeBatch


@dataclass(frozen=True)
class EdgeDenoiserConfig:
    """Configuration for the lightweight MLP edge denoiser."""

    num_atom_types: int
    num_edge_types: int = 5
    num_timesteps: int = 1000
    hidden_dim: int = 128
    num_layers: int = 3


class EdgeDenoiserMLP(nn.Module):
    """Predict clean halfedge bond types from noised labels and local geometry."""

    def __init__(self, config: EdgeDenoiserConfig) -> None:
        """Initialize embeddings and MLP."""
        super().__init__()
        if config.num_layers < 2:
            raise ValueError("num_layers must be at least 2")
        self.config = config
        self.atom_embedding = nn.Embedding(config.num_atom_types, config.hidden_dim)
        self.edge_embedding = nn.Embedding(config.num_edge_types, config.hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(4, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        input_dim = config.hidden_dim * 3 + 4
        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(config.num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, config.hidden_dim),
                nn.SiLU(),
            ])
            current_dim = config.hidden_dim
        layers.append(nn.Linear(current_dim, config.num_edge_types))
        self.mlp = nn.Sequential(*layers)

    def forward(self, batch: EdgeBatch, noised_edge_type: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return logits with shape ``[num_halfedges, num_edge_types]``."""
        if noised_edge_type.shape != batch.target_edge_type.shape:
            raise ValueError("noised_edge_type shape must match target_edge_type")
        src, dst = batch.halfedge_index
        atom_pair = (
            self.atom_embedding(batch.ligand_atom_type[src])
            + self.atom_embedding(batch.ligand_atom_type[dst])
        )
        edge_features = self.edge_embedding(noised_edge_type.long())
        time_features = self.time_mlp(_time_features(t, batch.edge_batch, self.config.num_timesteps))
        delta = batch.ligand_pos[src] - batch.ligand_pos[dst]
        distance = torch.linalg.norm(delta, dim=-1, keepdim=True)
        geometry = torch.cat([delta, distance], dim=-1)
        return self.mlp(torch.cat([atom_pair, edge_features, time_features, geometry], dim=-1))


def edge_focal_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weight: torch.Tensor | None = None,
    gamma: float = 0.0,
) -> torch.Tensor:
    """Return weighted focal cross entropy for edge labels."""
    log_prob = F.log_softmax(logits, dim=-1)
    nll = -log_prob.gather(dim=-1, index=target.long().unsqueeze(-1)).squeeze(-1)
    if gamma > 0:
        prob = torch.exp(-nll)
        nll = ((1.0 - prob).clamp_min(0.0) ** gamma) * nll
    if class_weight is not None:
        nll = class_weight.to(logits.device)[target.long()] * nll
    return nll.mean()


def make_inverse_sqrt_class_weight(counts: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Return inverse-sqrt class weights normalized to mean one."""
    weights = 1.0 / torch.sqrt(counts.float() + eps)
    return weights / weights.mean().clamp_min(1e-8)


def _time_features(t: torch.Tensor, edge_batch: torch.Tensor, num_timesteps: int) -> torch.Tensor:
    """Return simple sinusoidal timestep features per edge."""
    if t.ndim == 0 or t.numel() == 1:
        t_per_edge = t.reshape(()).expand_as(edge_batch)
    else:
        t_per_edge = t[edge_batch.long()]
    t_float = t_per_edge.float() / float(max(1, num_timesteps - 1))
    return torch.stack([
        t_float,
        torch.sin(torch.pi * t_float),
        torch.cos(torch.pi * t_float),
        torch.ones_like(t_float),
    ], dim=-1)
