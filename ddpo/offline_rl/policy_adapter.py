"""DiffSBDD policy-probability routing for offline denoising transitions.

The adapter keeps IQL/AWR code independent from the DiffSBDD implementation.
It groups ordinary transitions by timestep because the frozen DiffSBDD
dynamics consumes a scalar time, and it routes final decode transitions to the
coordinate-only decoder probability.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .protocol import DenoisingTransitionBatch


@dataclass(frozen=True)
class PolicyLogProbOutput:
    """Differentiable current-policy statistics in input transition order."""

    log_prob: torch.Tensor
    mean: torch.Tensor
    sigma: torch.Tensor
    ordinary_mask: torch.Tensor
    final_decode_mask: torch.Tensor

    def __post_init__(self) -> None:
        """Validate output shapes without detaching the training graph."""
        batch_size = int(self.log_prob.shape[0])
        if self.log_prob.ndim != 1:
            raise ValueError("log_prob must have shape [B]")
        if self.mean.ndim != 3 or int(self.mean.shape[0]) != batch_size:
            raise ValueError("mean must have shape [B,N,D]")
        for name, tensor in (
            ("sigma", self.sigma),
            ("ordinary_mask", self.ordinary_mask),
            ("final_decode_mask", self.final_decode_mask),
        ):
            if tuple(tensor.shape) != (batch_size,):
                raise ValueError(f"{name} must have shape [B]")
        if self.ordinary_mask.dtype != torch.bool:
            raise TypeError("ordinary_mask must be torch.bool")
        if self.final_decode_mask.dtype != torch.bool:
            raise TypeError("final_decode_mask must be torch.bool")
        if not torch.equal(self.ordinary_mask, ~self.final_decode_mask):
            raise ValueError("ordinary and final decode masks must be complementary")


def _per_sample_sigma(sigma: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Normalize scalar or broadcastable sigma to one diagnostic value per row."""
    if sigma.ndim == 0:
        return sigma.expand(batch_size)
    if int(sigma.shape[0]) == 1 and batch_size != 1:
        return sigma.reshape(1, -1)[:, 0].expand(batch_size)
    if int(sigma.shape[0]) != batch_size:
        raise ValueError(
            f"sigma leading dimension must be 1 or {batch_size}, "
            f"got {tuple(sigma.shape)}"
        )
    return sigma.reshape(batch_size, -1)[:, 0]


class DiffSBDDPolicyAdapter:
    """Route offline transition batches through the correct DiffSBDD branch."""

    def __init__(self, policy: nn.Module, *, num_diffusion_steps: int) -> None:
        """Initialize the adapter around a bare or DDP-wrapped policy module."""
        if num_diffusion_steps <= 0:
            raise ValueError("num_diffusion_steps must be positive")
        self.policy = policy
        self.num_diffusion_steps = int(num_diffusion_steps)

    def log_prob(
        self,
        transitions: DenoisingTransitionBatch,
    ) -> PolicyLogProbOutput:
        """Evaluate dataset actions under the current DiffSBDD policy.

        Ordinary transitions are grouped by target timestep before model
        evaluation. Final transitions use the separate coordinate-only decoder
        branch. Results are restored to the caller's original row order.
        """
        batch_size = transitions.batch_size
        device = transitions.device
        log_prob = transitions.state_latent.new_zeros(batch_size)
        mean = transitions.state_latent.new_zeros(
            transitions.state_latent.shape
        )
        sigma = transitions.state_latent.new_zeros(batch_size)
        final_mask = transitions.is_final_decode
        ordinary_mask = ~final_mask

        ordinary_indices = torch.nonzero(ordinary_mask, as_tuple=False).flatten()
        if int(ordinary_indices.numel()) > 0:
            target_times = transitions.target_diffusion_timestep[ordinary_indices]
            for target_time in torch.unique(target_times, sorted=True):
                group_mask = target_times == target_time
                indices = ordinary_indices[group_mask]
                group_log_prob, group_mean, group_sigma = self._ordinary_log_prob(
                    transitions,
                    indices,
                )
                log_prob = log_prob.index_copy(0, indices, group_log_prob)
                mean = mean.index_copy(0, indices, group_mean)
                sigma = sigma.index_copy(
                    0,
                    indices,
                    _per_sample_sigma(group_sigma, int(indices.numel())),
                )

        final_indices = torch.nonzero(final_mask, as_tuple=False).flatten()
        if int(final_indices.numel()) > 0:
            group_log_prob, group_mean, group_sigma = self._final_log_prob(
                transitions,
                final_indices,
            )
            log_prob = log_prob.index_copy(0, final_indices, group_log_prob)
            mean = mean.index_copy(0, final_indices, group_mean)
            sigma = sigma.index_copy(
                0,
                final_indices,
                _per_sample_sigma(group_sigma, int(final_indices.numel())),
            )

        return PolicyLogProbOutput(
            log_prob=log_prob,
            mean=mean,
            sigma=sigma,
            ordinary_mask=ordinary_mask.to(device=device),
            final_decode_mask=final_mask.to(device=device),
        )

    def _ordinary_log_prob(
        self,
        transitions: DenoisingTransitionBatch,
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate one homogeneous ordinary-timestep group."""
        target = transitions.target_diffusion_timestep[indices]
        source = transitions.source_diffusion_timestep[indices]
        s = target.to(dtype=transitions.state_latent.dtype).unsqueeze(-1)
        t = source.to(dtype=transitions.state_latent.dtype).unsqueeze(-1)
        s = s / float(self.num_diffusion_steps)
        t = t / float(self.num_diffusion_steps)
        edge_mask = transitions.state_latent.new_empty((0, 1))
        return self.policy(
            mode="offline_reverse_log_prob",
            s=s,
            t=t,
            zt=transitions.state_latent[indices],
            node_mask=transitions.node_mask[indices],
            edge_mask=edge_mask,
            prev_sample=transitions.action_latent[indices],
            action_mask=transitions.action_mask[indices],
            context=None,
        )

    def _final_log_prob(
        self,
        transitions: DenoisingTransitionBatch,
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate final coordinate actions under ``p(x | z0)``."""
        return self.policy(
            mode="offline_final_decode_log_prob",
            z0=transitions.state_latent[indices],
            node_mask=transitions.node_mask[indices],
            prev_sample=transitions.action_latent[indices],
            action_mask=transitions.action_mask[indices],
            context=None,
        )
