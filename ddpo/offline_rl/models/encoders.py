"""Distance-based invariant graph encoders for DiffSBDD critic networks.

The encoder borrows the scientific invariance principle used by DiffSBDD's
EGNN: scalar messages depend on node features and inter-node distances rather
than raw Cartesian components. Unlike the generative dynamics, critic
coordinates are not updated because only an E(3)-invariant scalar value is
required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from ..critic_adapter import GeometricCriticInput
from .graph_builder import (
    ACTION_ROLE,
    NUM_EDGE_TYPES,
    POCKET_ROLE,
    STATE_ROLE,
    RadiusGraphBuilder,
)


@dataclass(frozen=True)
class GeometricCriticConfig:
    """Architecture and radius-graph settings shared by Q/V implementations."""

    ligand_feature_dim: int
    pocket_feature_dim: int
    hidden_dim: int = 64
    num_message_layers: int = 3
    num_rbf: int = 16
    time_embedding_dim: int = 32
    pocket_cutoff: float = 6.0
    ligand_cutoff: float = 6.0
    interaction_cutoff: float = 8.0
    max_neighbors: int | None = 32
    dropout: float = 0.0

    def __post_init__(self) -> None:
        """Validate all architecture fields."""
        for name, value in (
            ("ligand_feature_dim", self.ligand_feature_dim),
            ("pocket_feature_dim", self.pocket_feature_dim),
            ("hidden_dim", self.hidden_dim),
            ("num_message_layers", self.num_message_layers),
            ("num_rbf", self.num_rbf),
            ("time_embedding_dim", self.time_embedding_dim),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.time_embedding_dim % 2 != 0:
            raise ValueError("time_embedding_dim must be even")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")


@dataclass(frozen=True)
class GeometricGraphEncoding:
    """Role-wise invariant graph pools and the time/final context embedding."""

    pocket: torch.Tensor
    state: torch.Tensor
    action: torch.Tensor | None
    context: torch.Tensor


class SinusoidalTimeEmbedding(nn.Module):
    """Embed normalized diffusion time with fixed sinusoidal frequencies."""

    def __init__(self, embedding_dim: int) -> None:
        """Initialize a fixed-frequency embedding."""
        super().__init__()
        if embedding_dim <= 0 or embedding_dim % 2 != 0:
            raise ValueError("embedding_dim must be a positive even integer")
        half_dim = embedding_dim // 2
        frequencies = torch.exp(
            torch.linspace(
                0.0,
                math.log(1000.0),
                half_dim,
                dtype=torch.float32,
            )
        )
        self.register_buffer("frequencies", frequencies)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        """Return sine/cosine features with shape ``[B, embedding_dim]``."""
        frequencies = self.frequencies.to(dtype=timestep.dtype)
        angles = timestep.unsqueeze(-1) * frequencies.unsqueeze(0)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class RadialBasisEmbedding(nn.Module):
    """Expand invariant edge distances with Gaussian radial basis functions."""

    def __init__(self, num_rbf: int, max_distance: float) -> None:
        """Initialize evenly spaced Gaussian centers."""
        super().__init__()
        if num_rbf <= 0:
            raise ValueError("num_rbf must be positive")
        if max_distance <= 0:
            raise ValueError("max_distance must be positive")
        centers = torch.linspace(0.0, float(max_distance), num_rbf)
        spacing = float(max_distance) / max(num_rbf - 1, 1)
        gamma = 1.0 / max(spacing, 1e-6) ** 2
        self.register_buffer("centers", centers)
        self.gamma = gamma

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        """Return RBF features with shape ``[E, num_rbf]``."""
        centers = self.centers.to(dtype=distance.dtype)
        difference = distance.unsqueeze(-1) - centers.unsqueeze(0)
        return torch.exp(-self.gamma * difference.square())


class InvariantMessageLayer(nn.Module):
    """One residual scalar message-passing layer over invariant edge features."""

    def __init__(
        self,
        hidden_dim: int,
        edge_feature_dim: int,
        dropout: float,
    ) -> None:
        """Initialize edge and node update networks."""
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.edge_gate = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Update flattened node features while preserving padded zeros."""
        source, target = edge_index
        if int(source.numel()) == 0:
            return node_features * node_mask
        message_input = torch.cat(
            [
                node_features[source],
                node_features[target],
                edge_features,
            ],
            dim=-1,
        )
        messages = self.edge_mlp(message_input)
        messages = messages * self.edge_gate(messages)
        aggregated = torch.zeros_like(node_features)
        aggregated.index_add_(0, target, messages)
        degree = node_features.new_zeros((int(node_features.shape[0]), 1))
        degree.index_add_(
            0,
            target,
            node_features.new_ones((int(target.numel()), 1)),
        )
        aggregated = aggregated / degree.clamp_min(1.0)
        update = self.node_mlp(torch.cat([node_features, aggregated], dim=-1))
        return self.layer_norm(node_features + update) * node_mask


class GeometricGraphEncoder(nn.Module):
    """Encode state/pocket or state/action/pocket graphs into invariant pools."""

    def __init__(self, config: GeometricCriticConfig) -> None:
        """Initialize role encoders, radius builder, and message layers."""
        super().__init__()
        self.config = config
        self.ligand_projection = nn.Linear(
            config.ligand_feature_dim,
            config.hidden_dim,
        )
        self.pocket_projection = nn.Linear(
            config.pocket_feature_dim,
            config.hidden_dim,
        )
        self.role_embedding = nn.Embedding(3, config.hidden_dim)
        self.time_embedding = SinusoidalTimeEmbedding(
            config.time_embedding_dim
        )
        self.context_projection = nn.Sequential(
            nn.Linear(config.time_embedding_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.final_embedding = nn.Embedding(2, config.hidden_dim)
        max_distance = max(
            config.pocket_cutoff,
            config.ligand_cutoff,
            config.interaction_cutoff,
        )
        self.rbf = RadialBasisEmbedding(config.num_rbf, max_distance)
        self.edge_type_embedding = nn.Embedding(
            NUM_EDGE_TYPES,
            config.hidden_dim,
        )
        edge_feature_dim = config.num_rbf + config.hidden_dim
        self.message_layers = nn.ModuleList(
            InvariantMessageLayer(
                config.hidden_dim,
                edge_feature_dim,
                config.dropout,
            )
            for _ in range(config.num_message_layers)
        )
        self.graph_builder = RadiusGraphBuilder(
            pocket_cutoff=config.pocket_cutoff,
            ligand_cutoff=config.ligand_cutoff,
            interaction_cutoff=config.interaction_cutoff,
            max_neighbors=config.max_neighbors,
        )

    def forward(
        self,
        inputs: GeometricCriticInput,
        *,
        include_action: bool,
    ) -> GeometricGraphEncoding:
        """Return role-pooled invariant features for V or Q."""
        self._validate_feature_dimensions(inputs)
        batch_size = inputs.batch_size
        pocket_nodes = int(inputs.pocket_coordinates.shape[1])
        ligand_nodes = int(inputs.state_coordinates.shape[1])
        context = self.context_projection(
            self.time_embedding(inputs.normalized_timestep)
        )
        context = context + self.final_embedding(
            inputs.is_final_decode.to(dtype=torch.long)
        )

        pocket_features = self.pocket_projection(inputs.pocket_features)
        state_features = self.ligand_projection(inputs.state_features)
        coordinates = [
            inputs.pocket_coordinates,
            inputs.state_coordinates,
        ]
        features = [pocket_features, state_features]
        masks = [inputs.pocket_mask, inputs.ligand_mask]
        role_parts = [
            torch.full(
                (pocket_nodes,),
                POCKET_ROLE,
                device=inputs.state_coordinates.device,
                dtype=torch.long,
            ),
            torch.full(
                (ligand_nodes,),
                STATE_ROLE,
                device=inputs.state_coordinates.device,
                dtype=torch.long,
            ),
        ]
        identity_pairs: torch.Tensor | None = None
        if include_action:
            action_features = self.ligand_projection(inputs.action_features)
            coordinates.append(inputs.action_coordinates)
            features.append(action_features)
            masks.append(inputs.ligand_mask)
            role_parts.append(
                torch.full(
                    (ligand_nodes,),
                    ACTION_ROLE,
                    device=inputs.state_coordinates.device,
                    dtype=torch.long,
                )
            )
            state_indices = (
                torch.arange(
                    ligand_nodes,
                    device=inputs.state_coordinates.device,
                    dtype=torch.long,
                )
                + pocket_nodes
            )
            action_indices = state_indices + ligand_nodes
            identity_pairs = torch.stack(
                [state_indices, action_indices],
                dim=-1,
            )

        coordinates_tensor = torch.cat(coordinates, dim=1)
        node_features = torch.cat(features, dim=1)
        node_mask = torch.cat(masks, dim=1)
        node_roles = torch.cat(role_parts)
        role_features = self.role_embedding(node_roles)
        node_features = (
            node_features
            + role_features.unsqueeze(0)
            + context.unsqueeze(1)
        )
        node_features = node_features * node_mask
        graph = self.graph_builder.build(
            coordinates_tensor,
            node_mask,
            node_roles,
            identity_pairs=identity_pairs,
        )
        edge_features = torch.cat(
            [
                self.rbf(graph.edge_distance),
                self.edge_type_embedding(graph.edge_type),
            ],
            dim=-1,
        )
        flattened = node_features.reshape(-1, self.config.hidden_dim)
        flattened_mask = node_mask.reshape(-1, 1)
        for layer in self.message_layers:
            flattened = layer(
                flattened,
                graph.edge_index,
                edge_features,
                flattened_mask,
            )
        node_features = flattened.reshape(
            batch_size,
            int(node_features.shape[1]),
            self.config.hidden_dim,
        )

        pocket_pool = self._masked_mean(
            node_features[:, :pocket_nodes],
            inputs.pocket_mask,
        )
        state_start = pocket_nodes
        state_end = state_start + ligand_nodes
        state_pool = self._masked_mean(
            node_features[:, state_start:state_end],
            inputs.ligand_mask,
        )
        action_pool: torch.Tensor | None = None
        if include_action:
            action_pool = self._masked_mean(
                node_features[:, state_end:],
                inputs.ligand_mask,
            )
        return GeometricGraphEncoding(
            pocket=pocket_pool,
            state=state_pool,
            action=action_pool,
            context=context,
        )

    def _validate_feature_dimensions(
        self,
        inputs: GeometricCriticInput,
    ) -> None:
        """Reject feature dimensions inconsistent with the network config."""
        ligand_dim = int(inputs.state_features.shape[-1])
        pocket_dim = int(inputs.pocket_features.shape[-1])
        if ligand_dim != self.config.ligand_feature_dim:
            raise ValueError(
                "ligand feature dimension mismatch: "
                f"expected {self.config.ligand_feature_dim}, got {ligand_dim}"
            )
        if pocket_dim != self.config.pocket_feature_dim:
            raise ValueError(
                "pocket feature dimension mismatch: "
                f"expected {self.config.pocket_feature_dim}, got {pocket_dim}"
            )

    @staticmethod
    def _masked_mean(
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-graph masked mean pooling."""
        numerator = torch.sum(values * mask, dim=1)
        denominator = torch.sum(mask, dim=1).clamp_min(1.0)
        return numerator / denominator
