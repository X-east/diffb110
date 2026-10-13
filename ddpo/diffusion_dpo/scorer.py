"""DiffSBDD adapter for exact Gaussian reverse-transition scores."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch

from ddpo.integration.diffsbdd_model import DiffSBDDModel

from .logprob import Reduction, TransitionScore, gaussian_transition_score


@dataclass(frozen=True)
class ReverseStatistics:
    """Model-produced Gaussian parameters for one reverse transition."""

    mean: torch.Tensor
    sigma: torch.Tensor


class DiffSBDDGaussianTransitionScorer:
    """Score saved normalized DiffSBDD states under one model checkpoint."""

    def __init__(
        self,
        checkpoint_path: str,
        pocket_pdb_path: str,
        ref_ligand_path: str,
        *,
        device: str = "cuda",
        dataset_name: str = "crossdock",
        num_timesteps: int = 1000,
    ) -> None:
        """Load one DiffSBDD model and bind it to an explicit timestep grid."""
        if num_timesteps <= 0:
            raise ValueError(f"num_timesteps must be positive, got {num_timesteps}")
        self.num_timesteps = num_timesteps
        self.model = DiffSBDDModel(
            checkpoint_path=checkpoint_path,
            pocket_pdb_path=pocket_pdb_path,
            ref_ligand_path=ref_ligand_path,
            device=device,
            dataset_name=dataset_name,
        )
        self.model.T = num_timesteps
        self.model.model.T = num_timesteps

    def transition_sigma(self, timestep_t: int) -> float:
        """Return the native schedule sigma for one integer reverse step."""
        if not 1 <= timestep_t <= self.num_timesteps:
            raise ValueError(f"timestep t={timestep_t} outside [1, {self.num_timesteps}]")
        device = torch.device(self.model.device)
        t = torch.tensor([[float(timestep_t) / self.num_timesteps]], device=device)
        s = torch.tensor([[float(timestep_t - 1) / self.num_timesteps]], device=device)
        gamma_t = self.model.ddpm.gamma(t)
        gamma_s = self.model.ddpm.gamma(s)
        target = torch.zeros(1, 13, device=device)
        _sigma2, sigma, _alpha = self.model.ddpm.sigma_and_alpha_t_given_s(
            gamma_t, gamma_s, target
        )
        return float(sigma.max().detach().cpu())

    def valid_transition_timesteps(self, *, tolerance: float = 1.0e-8) -> list[int]:
        """Return T=1000 steps with non-degenerate native Gaussian variance."""
        return self._valid_transition_timesteps(tolerance)

    @lru_cache(maxsize=8)
    def _valid_transition_timesteps(self, tolerance: float) -> list[int]:
        """Cache the native valid-step audit for repeated trainer/eval calls."""
        return [
            timestep
            for timestep in range(1, self.num_timesteps + 1)
            if self.transition_sigma(timestep) > tolerance
        ]

    def reverse_statistics(
        self,
        z_t: torch.Tensor,
        timestep_t: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> ReverseStatistics:
        """Compute DiffSBDD's native ``mu`` and ``sigma`` for ``p(z_s|z_t)``."""
        if timestep_t.ndim == 2:
            timestep_t = timestep_t.squeeze(-1)
        if timestep_t.ndim != 1 or not torch.all(timestep_t == timestep_t[0]):
            raise ValueError("a score batch must use one common timestep")
        t_value = int(timestep_t[0].item())
        if not 1 <= t_value <= self.num_timesteps:
            raise ValueError(f"timestep t={t_value} outside [1, {self.num_timesteps}]")
        t = torch.full(
            (z_t.shape[0], 1),
            float(t_value) / float(self.num_timesteps),
            device=z_t.device,
            dtype=z_t.dtype,
        )
        s = torch.full(
            (z_t.shape[0], 1),
            float(t_value - 1) / float(self.num_timesteps),
            device=z_t.device,
            dtype=z_t.dtype,
        )
        mean_flat, sigma, ligand_mask, _pocket_mask, _pocket = self.model._compute_reverse_transition_stats(
            s=s,
            t=t,
            zt=z_t,
            node_mask=node_mask,
        )
        mean = self.model._diffsbdd_to_padded_format(
            mean_flat,
            ligand_mask,
            z_t.shape[1],
        )
        return ReverseStatistics(mean=mean, sigma=sigma)

    def score(
        self,
        z_t: torch.Tensor,
        z_s: torch.Tensor,
        timestep_t: torch.Tensor,
        node_mask: torch.Tensor,
        *,
        reduction: Reduction = "sum",
        include_normal_constant: bool = True,
        include_sigma_term: bool = True,
    ) -> TransitionScore:
        """Return exact diagonal-Gaussian component scores for one transition."""
        statistics = self.reverse_statistics(z_t, timestep_t, node_mask)
        return gaussian_transition_score(
            z_s,
            statistics.mean,
            statistics.sigma,
            node_mask,
            reduction=reduction,
            include_normal_constant=include_normal_constant,
            include_sigma_term=include_sigma_term,
        )
