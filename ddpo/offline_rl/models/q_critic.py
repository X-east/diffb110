"""Twin action-value networks for DiffSBDD denoising offline IQL."""

from __future__ import annotations

import torch
from torch import nn

from ..critic_adapter import GeometricCriticInput
from .encoders import GeometricCriticConfig, GeometricGraphEncoder


class GeometricQNetwork(nn.Module):
    """Predict invariant Q(s,a) from pocket, state, and sampled next latent."""

    def __init__(self, config: GeometricCriticConfig) -> None:
        """Initialize one independent geometric Q network."""
        super().__init__()
        self.encoder = GeometricGraphEncoder(config)
        hidden_dim = config.hidden_dim
        self.head = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, inputs: GeometricCriticInput) -> torch.Tensor:
        """Return one action value per transition with shape ``[B]``."""
        encoded = self.encoder(inputs, include_action=True)
        if encoded.action is None:
            raise RuntimeError("Q encoder did not return an action pool")
        head_input = torch.cat(
            [
                encoded.pocket,
                encoded.state,
                encoded.action,
                encoded.context,
            ],
            dim=-1,
        )
        return self.head(head_input).squeeze(-1)


class TwinQCritic(nn.Module):
    """Two fully independent geometric Q networks for conservative targets."""

    def __init__(self, config: GeometricCriticConfig) -> None:
        """Initialize independent Q1 and Q2 parameters."""
        super().__init__()
        self.q1 = GeometricQNetwork(config)
        self.q2 = GeometricQNetwork(config)

    def forward(
        self,
        inputs: GeometricCriticInput,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(Q1,Q2)`` with one scalar from each network per row."""
        return self.q1(inputs), self.q2(inputs)
