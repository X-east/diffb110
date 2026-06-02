"""Pair-wise neural model for local ligand bond prediction."""

from __future__ import annotations

import torch
from torch import nn


class PairBondPredictor(nn.Module):
    """Predict ligand halfedge bond type from atom-pair features."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        num_edge_types: int = 5,
    ) -> None:
        """Initialize a lightweight MLP bond predictor."""
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, num_edge_types))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return bond-type logits for each atom pair."""
        return self.network(features)
