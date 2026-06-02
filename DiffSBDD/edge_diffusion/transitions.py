"""Categorical transition helpers for ligand halfedge bond labels."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class EdgeTransitionConfig:
    """Configuration for a lightweight categorical edge noising process."""

    num_classes: int = 5
    num_timesteps: int = 1000
    prior: str = "uniform"


class EdgeCategoricalTransition:
    """Diffusion-style categorical noising for halfedge bond types.

    This class is intentionally small: it is used to verify the edge-data chain
    before a full x/h/e denoiser is introduced. The transition corrupts labels
    with a timestep-dependent probability and preserves tensor shape/device.
    """

    def __init__(self, config: EdgeTransitionConfig | None = None) -> None:
        """Initialize the transition."""
        self.config = config or EdgeTransitionConfig()
        if self.config.num_classes <= 1:
            raise ValueError("num_classes must be greater than 1")
        if self.config.num_timesteps <= 0:
            raise ValueError("num_timesteps must be positive")
        if self.config.prior != "uniform":
            raise ValueError(f"Unsupported edge prior: {self.config.prior}")

    def noise_probability(self, t: torch.Tensor) -> torch.Tensor:
        """Return per-item noising probabilities for timestep tensor ``t``."""
        t_float = t.to(dtype=torch.float32)
        return ((t_float + 1.0) / float(self.config.num_timesteps)).clamp(0.0, 1.0)

    def q_sample(
        self,
        edge_type: torch.Tensor,
        t: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample noised halfedge labels with the same shape as ``edge_type``."""
        if edge_type.dtype != torch.long:
            edge_type = edge_type.long()
        if edge_type.numel() == 0:
            return edge_type.clone()
        if edge_type.min().item() < 0 or edge_type.max().item() >= self.config.num_classes:
            raise ValueError("edge_type contains labels outside configured classes")

        t_expanded = self._expand_timestep(t.to(edge_type.device), edge_type)
        noise_prob = self.noise_probability(t_expanded).to(edge_type.device)
        random_labels = torch.randint(
            low=0,
            high=self.config.num_classes,
            size=edge_type.shape,
            device=edge_type.device,
            generator=generator,
        )
        replace_mask = torch.rand(
            edge_type.shape,
            device=edge_type.device,
            generator=generator,
        ) < noise_prob
        return torch.where(replace_mask, random_labels, edge_type)

    def one_hot(self, edge_type: torch.Tensor) -> torch.Tensor:
        """Return one-hot encoded edge labels."""
        return F.one_hot(edge_type.long(), num_classes=self.config.num_classes).float()

    def negative_log_likelihood(
        self,
        pred_logits: torch.Tensor,
        target_edge_type: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-halfedge categorical negative log likelihood."""
        if pred_logits.ndim != 2:
            raise ValueError("pred_logits must have shape [num_halfedges, num_classes]")
        if pred_logits.shape[-1] != self.config.num_classes:
            raise ValueError("pred_logits last dimension does not match num_classes")
        return F.cross_entropy(pred_logits, target_edge_type.long(), reduction="none")

    @staticmethod
    def _expand_timestep(t: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        """Broadcast a scalar or per-edge timestep tensor to edge shape."""
        if t.ndim == 0:
            return t.expand_as(edge_type)
        if t.shape == edge_type.shape:
            return t
        if t.numel() == 1:
            return t.reshape(()).expand_as(edge_type)
        raise ValueError("t must be scalar, single-valued, or the same shape as edge_type")
