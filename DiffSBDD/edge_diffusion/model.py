"""Standalone x/h/e denoiser skeleton for optional edge diffusion experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn

from DiffSBDD.edge_diffusion.dataset import EdgeBatch
from DiffSBDD.edge_diffusion.denoiser import GaussianDistanceExpansion, NodeEdgeBlock
from DiffSBDD.edge_diffusion.schema import EdgeDiffusionInputs, EdgeDiffusionOutput


@dataclass(frozen=True)
class EdgeDiffusionModelConfig:
    """Configuration for the standalone ligand x/h/e denoiser."""

    num_atom_types: int
    num_edge_types: int = 5
    num_timesteps: int = 1000
    hidden_dim: int = 128
    num_layers: int = 4
    distance_basis_dim: int = 32
    dropout: float = 0.0
    coordinate_scale: float = 0.1


class EdgeDiffusionModel(nn.Module):
    """Predict clean ligand coordinates, atom types, and halfedge bond types."""

    def __init__(self, config: EdgeDiffusionModelConfig) -> None:
        """Initialize node/edge embeddings, message passing blocks, and decoders."""
        super().__init__()
        if config.num_atom_types <= 0:
            raise ValueError("num_atom_types must be positive")
        if config.num_edge_types <= 1:
            raise ValueError("num_edge_types must be greater than 1")
        if config.num_timesteps <= 0:
            raise ValueError("num_timesteps must be positive")
        if config.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
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
        self.coordinate_mlp = _make_mlp(
            hidden_dim + config.distance_basis_dim + 1,
            1,
            hidden_dim,
            num_layers=2,
            dropout=config.dropout,
        )
        self.atom_decoder = _make_mlp(
            hidden_dim,
            config.num_atom_types,
            hidden_dim,
            num_layers=2,
            dropout=config.dropout,
        )
        self.edge_decoder = _make_mlp(
            hidden_dim * 2 + config.distance_basis_dim + 1,
            config.num_edge_types,
            hidden_dim,
            num_layers=3,
            dropout=config.dropout,
        )

    def forward(self, batch: EdgeBatch, inputs: EdgeDiffusionInputs) -> EdgeDiffusionOutput:
        """Return clean-state predictions for one collated ligand edge batch."""
        self._validate_inputs(batch, inputs)

        pos_t = inputs.ligand_pos_t
        halfedge_index = batch.halfedge_index.long()
        num_halfedges = int(halfedge_index.shape[1])
        directed_edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
        directed_edge_batch = torch.cat([batch.edge_batch, batch.edge_batch], dim=0)
        directed_edge_type = torch.cat([
            inputs.ligand_edge_type_t,
            inputs.ligand_edge_type_t,
        ], dim=0).long()

        node_h = self.atom_embedding(inputs.ligand_atom_type_t.long())
        node_h = node_h + self.node_time_mlp(
            _time_features(inputs.t, batch.atom_batch, self.config.num_timesteps)
        )

        distance_features, distances = self._edge_geometry(pos_t, directed_edge_index)
        if directed_edge_type.numel() == 0:
            edge_h = pos_t.new_zeros((0, self.config.hidden_dim))
        else:
            src, dst = directed_edge_index
            edge_h = self.edge_init(torch.cat([
                self.edge_embedding(directed_edge_type),
                node_h[src],
                node_h[dst],
                self.edge_time_mlp(_time_features(
                    inputs.t,
                    directed_edge_batch,
                    self.config.num_timesteps,
                )),
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

        pred_pos = self._predict_clean_pos(pos_t, directed_edge_index, edge_h, distance_features, distances)
        pred_atom_logits = self.atom_decoder(node_h)
        pred_edge_logits = self._predict_clean_edges(
            node_h=node_h,
            edge_h=edge_h,
            halfedge_index=halfedge_index,
            pos=pos_t,
            num_halfedges=num_halfedges,
        )
        return EdgeDiffusionOutput(
            pred_pos=pred_pos,
            pred_atom_logits=pred_atom_logits,
            pred_edge_logits=pred_edge_logits,
        )

    def _predict_clean_pos(
        self,
        pos_t: torch.Tensor,
        directed_edge_index: torch.Tensor,
        edge_h: torch.Tensor,
        distance_features: torch.Tensor,
        distances: torch.Tensor,
    ) -> torch.Tensor:
        """Predict x0 coordinates through an equivariant residual update."""
        if directed_edge_index.shape[1] == 0:
            return pos_t
        src, dst = directed_edge_index
        coord_gate = torch.tanh(self.coordinate_mlp(torch.cat([
            edge_h,
            distance_features,
            distances,
        ], dim=-1)))
        direction = (pos_t[src] - pos_t[dst]) / distances.clamp_min(1e-6)
        messages = coord_gate * direction
        aggregated = torch.zeros_like(pos_t)
        aggregated.index_add_(0, dst, messages)
        degree = torch.zeros((pos_t.shape[0], 1), device=pos_t.device, dtype=pos_t.dtype)
        degree.index_add_(0, dst, torch.ones((messages.shape[0], 1), device=pos_t.device, dtype=pos_t.dtype))
        aggregated = aggregated / degree.clamp_min(1.0)
        return pos_t + float(self.config.coordinate_scale) * aggregated

    def _predict_clean_edges(
        self,
        node_h: torch.Tensor,
        edge_h: torch.Tensor,
        halfedge_index: torch.Tensor,
        pos: torch.Tensor,
        num_halfedges: int,
    ) -> torch.Tensor:
        """Predict clean halfedge bond logits from paired directed edge states."""
        if num_halfedges == 0:
            return pos.new_zeros((0, self.config.num_edge_types))
        half_src, half_dst = halfedge_index
        distance_features, distances = self._edge_geometry(pos, halfedge_index)
        paired_edge_h = edge_h[:num_halfedges] + edge_h[num_halfedges:]
        paired_node_h = node_h[half_src] + node_h[half_dst]
        return self.edge_decoder(torch.cat([
            paired_edge_h,
            paired_node_h,
            distance_features,
            distances,
        ], dim=-1))

    def _edge_geometry(
        self,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return Gaussian distance features and raw distances for edge indices."""
        if edge_index.shape[1] == 0:
            return (
                pos.new_zeros((0, self.config.distance_basis_dim)),
                pos.new_zeros((0, 1)),
            )
        src, dst = edge_index
        distances = torch.linalg.norm(pos[src] - pos[dst], dim=-1, keepdim=True)
        return self.distance_expansion(distances.squeeze(-1)), distances

    def _validate_inputs(self, batch: EdgeBatch, inputs: EdgeDiffusionInputs) -> None:
        """Validate tensor contracts before model execution."""
        if inputs.ligand_pos_t.shape != batch.ligand_pos.shape:
            raise ValueError("ligand_pos_t shape must match batch.ligand_pos")
        if inputs.ligand_atom_type_t.shape != batch.ligand_atom_type.shape:
            raise ValueError("ligand_atom_type_t shape must match batch.ligand_atom_type")
        if inputs.ligand_edge_type_t.shape != batch.target_edge_type.shape:
            raise ValueError("ligand_edge_type_t shape must match batch.target_edge_type")
        if inputs.ligand_pos_t.shape[-1] != 3:
            raise ValueError("ligand_pos_t must have shape [num_atoms, 3]")
        if batch.halfedge_index.ndim != 2 or batch.halfedge_index.shape[0] != 2:
            raise ValueError("halfedge_index must have shape [2, num_halfedges]")


def _time_features(t: torch.Tensor, batch_index: torch.Tensor, num_timesteps: int) -> torch.Tensor:
    """Return simple sinusoidal timestep features per atom or edge."""
    if t.ndim == 0 or t.numel() == 1:
        t_per_item = t.reshape(()).expand_as(batch_index)
    else:
        t_per_item = t[batch_index.long()]
    t_float = t_per_item.float() / float(max(1, num_timesteps - 1))
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
