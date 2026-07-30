"""State-value network for DiffSBDD denoising offline IQL."""

from __future__ import annotations

import torch
from torch import nn

from ..critic_adapter import GeometricCriticInput
from .encoders import GeometricCriticConfig, GeometricGraphEncoder


class GeometricValueNetwork(nn.Module):
    """Predict an invariant scalar value from pocket and current ligand state."""

    def __init__(self, config: GeometricCriticConfig) -> None:
        """Initialize an independent state encoder and scalar head."""
        super().__init__()
        self.encoder = GeometricGraphEncoder(config)
        hidden_dim = config.hidden_dim
        self.head = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, inputs: GeometricCriticInput) -> torch.Tensor:
        """Return one value per transition state with shape ``[B]``."""
        encoded = self.encoder(inputs, include_action=False)
        head_input = torch.cat(
            [encoded.pocket, encoded.state, encoded.context],
            dim=-1,
        )
        return self.head(head_input).squeeze(-1)
