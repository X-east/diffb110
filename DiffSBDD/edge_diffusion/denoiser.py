"""Small edge denoiser modules for halfedge bond-type overfit experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math

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
    distance_basis_dim: int = 32
    dropout: float = 0.0


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


class EdgeDenoiserNodeEdge(nn.Module):
    """DiffGui/MolSnapper-style node-edge denoiser for halfedge bond labels."""

    def __init__(self, config: EdgeDenoiserConfig) -> None:
        """Initialize atom/edge/time embeddings and message-passing blocks."""
        super().__init__()
        if config.num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if config.distance_basis_dim < 4:
            raise ValueError("distance_basis_dim must be at least 4")
        self.config = config
        hidden_dim = config.hidden_dim
        self.atom_embedding = nn.Embedding(config.num_atom_types, hidden_dim)
        self.edge_embedding = nn.Embedding(config.num_edge_types, hidden_dim)
        self.node_time_mlp = _make_mlp(4, hidden_dim, hidden_dim, num_layers=2, dropout=config.dropout)
        self.edge_time_mlp = _make_mlp(4, hidden_dim, hidden_dim, num_layers=2, dropout=config.dropout)
        self.distance_expansion = GaussianDistanceExpansion(
            num_basis=config.distance_basis_dim,
            start=0.0,
            stop=6.0,
        )
        self.edge_init = _make_mlp(
            hidden_dim * 4 + config.distance_basis_dim + 1,
            hidden_dim,
            hidden_dim,
            num_layers=2,
            dropout=config.dropout,
        )
        self.blocks = nn.ModuleList([
            NodeEdgeBlock(hidden_dim, config.distance_basis_dim, config.dropout)
            for _ in range(config.num_layers)
        ])
        self.edge_decoder = _make_mlp(
            hidden_dim * 2 + config.distance_basis_dim + 1,
            config.num_edge_types,
            hidden_dim,
            num_layers=3,
            dropout=config.dropout,
        )

    def forward(self, batch: EdgeBatch, noised_edge_type: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return logits with shape ``[num_halfedges, num_edge_types]``."""
        if noised_edge_type.shape != batch.target_edge_type.shape:
            raise ValueError("noised_edge_type shape must match target_edge_type")
        num_halfedges = int(noised_edge_type.numel())
        if num_halfedges == 0:
            return batch.ligand_pos.new_zeros((0, self.config.num_edge_types))

        halfedge_index = batch.halfedge_index.long()
        directed_edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
        directed_edge_batch = torch.cat([batch.edge_batch, batch.edge_batch], dim=0)
        directed_edge_type = torch.cat([noised_edge_type, noised_edge_type], dim=0).long()

        node_h = self.atom_embedding(batch.ligand_atom_type.long())
        node_h = node_h + self.node_time_mlp(
            _time_features(t, batch.atom_batch, self.config.num_timesteps)
        )

        src, dst = directed_edge_index
        distance_features, distances = self._edge_geometry(batch.ligand_pos, directed_edge_index)
        edge_h = self.edge_init(torch.cat([
            self.edge_embedding(directed_edge_type),
            node_h[src],
            node_h[dst],
            self.edge_time_mlp(_time_features(t, directed_edge_batch, self.config.num_timesteps)),
            distance_features,
            distances,
        ], dim=-1))

        for block in self.blocks:
            node_h, edge_h = block(
                node_h=node_h,
                edge_h=edge_h,
                edge_index=directed_edge_index,
                distance_features=distance_features,
                distances=distances,
            )

        half_src, half_dst = halfedge_index
        half_distance_features, half_distances = self._edge_geometry(batch.ligand_pos, halfedge_index)
        paired_edge_h = edge_h[:num_halfedges] + edge_h[num_halfedges:]
        paired_node_h = node_h[half_src] + node_h[half_dst]
        return self.edge_decoder(torch.cat([
            paired_edge_h,
            paired_node_h,
            half_distance_features,
            half_distances,
        ], dim=-1))

    def _edge_geometry(
        self,
        ligand_pos: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return Gaussian distance features and raw distances for edges."""
        src, dst = edge_index
        distances = torch.linalg.norm(ligand_pos[src] - ligand_pos[dst], dim=-1, keepdim=True)
        return self.distance_expansion(distances.squeeze(-1)), distances


class NodeEdgeBlock(nn.Module):
    """Residual node-edge message passing block adapted from modern SBDD denoisers."""

    def __init__(self, hidden_dim: int, distance_basis_dim: int, dropout: float = 0.0) -> None:
        """Initialize edge update, node message, and normalization layers."""
        super().__init__()
        self.edge_mlp = _make_mlp(
            hidden_dim * 3 + distance_basis_dim + 1,
            hidden_dim,
            hidden_dim,
            num_layers=2,
            dropout=dropout,
        )
        self.message_mlp = _make_mlp(
            hidden_dim * 2,
            hidden_dim,
            hidden_dim,
            num_layers=2,
            dropout=dropout,
        )
        self.node_mlp = _make_mlp(
            hidden_dim * 2,
            hidden_dim,
            hidden_dim,
            num_layers=2,
            dropout=dropout,
        )
        self.edge_norm = nn.LayerNorm(hidden_dim)
        self.node_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_h: torch.Tensor,
        edge_h: torch.Tensor,
        edge_index: torch.Tensor,
        distance_features: torch.Tensor,
        distances: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update node and edge states with directed edge messages."""
        src, dst = edge_index
        edge_update = self.edge_mlp(torch.cat([
            edge_h,
            node_h[src],
            node_h[dst],
            distance_features,
            distances,
        ], dim=-1))
        edge_h = self.edge_norm(edge_h + edge_update)

        messages = self.message_mlp(torch.cat([edge_h, node_h[src]], dim=-1))
        aggregated = torch.zeros_like(node_h)
        aggregated.index_add_(0, dst, messages)
        degree = torch.zeros((node_h.shape[0], 1), device=node_h.device, dtype=node_h.dtype)
        degree.index_add_(0, dst, torch.ones((messages.shape[0], 1), device=node_h.device, dtype=node_h.dtype))
        aggregated = aggregated / degree.clamp_min(1.0)
        node_update = self.node_mlp(torch.cat([node_h, aggregated], dim=-1))
        node_h = self.node_norm(node_h + node_update)
        return node_h, edge_h


class GaussianDistanceExpansion(nn.Module):
    """Expand scalar distances into fixed Gaussian basis features."""

    def __init__(self, num_basis: int, start: float, stop: float) -> None:
        """Initialize Gaussian centers and width."""
        super().__init__()
        centers = torch.linspace(start, stop, num_basis)
        self.register_buffer("centers", centers)
        self.coefficient = -0.5 / ((centers[1] - centers[0]).item() ** 2)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """Return Gaussian basis features for distance tensor ``[num_edges]``."""
        return torch.exp(self.coefficient * (distances.unsqueeze(-1) - self.centers) ** 2)


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
        torch.sin(math.pi * t_float),
        torch.cos(math.pi * t_float),
        torch.ones_like(t_float),
    ], dim=-1)


def _make_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float = 0.0,
) -> nn.Sequential:
    """Build a compact SiLU MLP."""
    if num_layers < 1:
        raise ValueError("num_layers must be at least 1")
    layers: list[nn.Module] = []
    current_dim = input_dim
    for _ in range(num_layers - 1):
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.SiLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)
