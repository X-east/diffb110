"""Categorical transition helpers for ligand halfedge bond labels."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class EdgeTransitionConfig:
    """Configuration for a DiffGui-style categorical edge transition."""

    num_classes: int = 5
    num_timesteps: int = 1000
    prior: str = "uniform"
    beta_schedule: str = "linear"
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    custom_prior: tuple[float, ...] | None = None
    eps: float = 1e-30


class EdgeCategoricalTransition:
    """Categorical diffusion transition for halfedge bond types.

    The class keeps the early smoke-test API (`q_sample`, `one_hot`,
    `negative_log_likelihood`) while adding the posterior operations needed by
    full edge diffusion:

    - `q_vt_pred`: q(e_t | e_0)
    - `q_v_posterior`: q(e_{t-1} | e_t, e_0)
    - `compute_v_Lt`: KL/NLL discrete diffusion loss
    - `sample_init`: draw e_T from the configured prior
    """

    def __init__(
        self,
        config: EdgeTransitionConfig | None = None,
        *,
        init_prob: Sequence[float] | torch.Tensor | None = None,
    ) -> None:
        """Initialize transition matrices."""
        self.config = config or EdgeTransitionConfig()
        if self.config.num_classes <= 1:
            raise ValueError("num_classes must be greater than 1")
        if self.config.num_timesteps <= 0:
            raise ValueError("num_timesteps must be positive")

        prior = self._make_prior(init_prob)
        betas = self._make_betas()
        q_onestep = self._build_one_step_mats(betas, prior)
        q_mats = self._build_cumulative_mats(q_onestep)

        self.num_classes = int(self.config.num_classes)
        self.num_timesteps = int(self.config.num_timesteps)
        self.prior = prior
        self.betas = betas
        self.alphas_bar = torch.cumprod(1.0 - betas, dim=0)
        self.q_onestep_mats = q_onestep
        self.q_mats = q_mats
        self.transpose_q_onestep_mats = q_onestep.transpose(-1, -2)

    def noise_probability(self, t: torch.Tensor) -> torch.Tensor:
        """Return approximate cumulative corruption probability for timestep t."""
        t_flat = self._clamp_timestep(t.to(dtype=torch.long).reshape(-1))
        values = 1.0 - self.alphas_bar.index_select(0, t_flat).to(t.device)
        return values.reshape(t.shape)

    def q_sample(
        self,
        edge_type: torch.Tensor,
        t: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample e_t from q(e_t | e_0) and preserve the label shape."""
        edge_type = self._validate_labels(edge_type)
        if edge_type.numel() == 0:
            return edge_type.clone()
        log_v0 = self.index_to_log_onehot(edge_type)
        log_q_vt = self.q_vt_pred(log_v0, t.to(edge_type.device))
        return self.log_sample_categorical(log_q_vt, generator=generator)

    def q_vt_sample(
        self,
        log_v0: torch.Tensor,
        t: torch.Tensor,
        batch: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample categorical labels and log one-hot labels from q(e_t | e_0)."""
        log_q_vt = self.q_vt_pred(log_v0, t, batch)
        sampled = self.log_sample_categorical(log_q_vt, generator=generator)
        return sampled, self.index_to_log_onehot(sampled)

    def add_noise(
        self,
        edge_type: torch.Tensor,
        t: torch.Tensor,
        batch: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return one-hot e_t, log e_t, and log e_0 for training loops."""
        edge_type = self._validate_labels(edge_type)
        log_v0 = self.index_to_log_onehot(edge_type)
        sampled, log_vt = self.q_vt_sample(log_v0, t.to(edge_type.device), batch, generator)
        return self.one_hot(sampled), log_vt, log_v0

    def q_vt_pred(
        self,
        log_v0: torch.Tensor,
        t: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return log q(e_t | e_0)."""
        if log_v0.shape[-1] != self.num_classes:
            raise ValueError("log_v0 last dimension does not match num_classes")
        t_per_item = self._t_per_item(t, log_v0.shape[0], log_v0.device, batch)
        q_mats = self.q_mats.to(device=log_v0.device, dtype=log_v0.dtype)
        q_t = q_mats.index_select(0, t_per_item)
        probs = torch.einsum("...i,...ij->...j", log_v0.exp(), q_t)
        return torch.log(probs.clamp_min(self.config.eps))

    def q_v_posterior(
        self,
        log_v0: torch.Tensor,
        log_vt: torch.Tensor,
        t: torch.Tensor,
        batch: torch.Tensor | None = None,
        v0_prob: bool = True,
    ) -> torch.Tensor:
        """Return log q(e_{t-1} | e_t, e_0).

        When `v0_prob=True`, `log_v0` may contain a predicted probability
        distribution over clean edge classes. This is the mode used when the
        denoiser predicts e_0 logits.
        """
        if log_v0.shape != log_vt.shape:
            raise ValueError("log_v0 and log_vt must have the same shape")
        if log_v0.shape[-1] != self.num_classes:
            raise ValueError("edge distributions must end with num_classes")

        device = log_v0.device
        dtype = log_v0.dtype
        t_per_item = self._t_per_item(t, log_v0.shape[0], device, batch)
        t_minus_one = (t_per_item - 1).clamp_min(0)

        q_onestep_t = self.transpose_q_onestep_mats.to(device=device, dtype=dtype)
        q_mats = self.q_mats.to(device=device, dtype=dtype)

        fact1 = torch.einsum(
            "bi,bij->bj",
            log_vt.exp(),
            q_onestep_t.index_select(0, t_per_item),
        )
        q_t_minus_one = q_mats.index_select(0, t_minus_one)
        if v0_prob:
            fact2 = torch.einsum("bi,bij->bj", log_v0.exp(), q_t_minus_one)
        else:
            class_v0 = log_v0.argmax(dim=-1)
            fact2 = q_t_minus_one[torch.arange(len(class_v0), device=device), class_v0]

        log_out = torch.log(fact1.clamp_min(self.config.eps))
        log_out = log_out + torch.log(fact2.clamp_min(self.config.eps))
        log_out = log_out - torch.logsumexp(log_out, dim=-1, keepdim=True)
        return torch.where((t_per_item == 0).unsqueeze(-1), log_v0, log_out)

    def compute_v_Lt(
        self,
        log_v_post_true: torch.Tensor,
        log_v_post_pred: torch.Tensor,
        log_v0: torch.Tensor,
        t: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return per-item discrete diffusion KL/NLL loss."""
        if log_v_post_true.shape != log_v_post_pred.shape:
            raise ValueError("posterior tensors must have the same shape")
        kl_v = self.categorical_kl(log_v_post_true, log_v_post_pred)
        decoder_nll_v = -self.log_categorical(log_v0, log_v_post_pred)
        t_per_item = self._t_per_item(t, log_v0.shape[0], log_v0.device, batch)
        mask_t0 = (t_per_item == 0).to(dtype=kl_v.dtype)
        return mask_t0 * decoder_nll_v + (1.0 - mask_t0) * kl_v

    def sample_init(
        self,
        n: int,
        *,
        device: torch.device | str | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample initial e_T labels from the configured prior."""
        if n < 0:
            raise ValueError("n must be non-negative")
        target_device = torch.device("cpu") if device is None else torch.device(device)
        log_prior = torch.log(
            self.prior.to(target_device).clamp_min(self.config.eps)
        ).unsqueeze(0).repeat(n, 1)
        sampled = self.log_sample_categorical(log_prior, generator=generator)
        return sampled, self.one_hot(sampled), self.index_to_log_onehot(sampled)

    def one_hot(self, edge_type: torch.Tensor) -> torch.Tensor:
        """Return one-hot encoded edge labels."""
        return F.one_hot(edge_type.long(), num_classes=self.num_classes).float()

    def index_to_log_onehot(self, edge_type: torch.Tensor) -> torch.Tensor:
        """Return log one-hot encoded edge labels."""
        edge_type = self._validate_labels(edge_type)
        one_hot = F.one_hot(edge_type.long(), num_classes=self.num_classes)
        return torch.log(one_hot.float().clamp_min(self.config.eps))

    def negative_log_likelihood(
        self,
        pred_logits: torch.Tensor,
        target_edge_type: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-halfedge categorical negative log likelihood."""
        if pred_logits.ndim != 2:
            raise ValueError("pred_logits must have shape [num_halfedges, num_classes]")
        if pred_logits.shape[-1] != self.num_classes:
            raise ValueError("pred_logits last dimension does not match num_classes")
        return F.cross_entropy(pred_logits, target_edge_type.long(), reduction="none")

    def log_sample_categorical(
        self,
        logits: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample class ids from log probabilities with Gumbel noise."""
        uniform = torch.rand(
            logits.shape,
            device=logits.device,
            dtype=logits.dtype,
            generator=generator,
        )
        gumbel_noise = -torch.log(-torch.log(uniform.clamp_min(self.config.eps)))
        return (gumbel_noise + logits).argmax(dim=-1)

    @staticmethod
    def categorical_kl(log_prob1: torch.Tensor, log_prob2: torch.Tensor) -> torch.Tensor:
        """Return KL(log_prob1 || log_prob2) over the final class dimension."""
        return (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=-1)

    @staticmethod
    def log_categorical(log_x_start: torch.Tensor, log_prob: torch.Tensor) -> torch.Tensor:
        """Return log probability assigned to one-hot/log-prob encoded labels."""
        return (log_x_start.exp() * log_prob).sum(dim=-1)

    def _make_prior(
        self,
        init_prob: Sequence[float] | torch.Tensor | None,
    ) -> torch.Tensor:
        """Build a normalized categorical prior."""
        if init_prob is not None:
            prior = torch.as_tensor(init_prob, dtype=torch.float64)
        elif self.config.custom_prior is not None:
            prior = torch.as_tensor(self.config.custom_prior, dtype=torch.float64)
        elif self.config.prior == "uniform":
            prior = torch.ones(self.config.num_classes, dtype=torch.float64)
        elif self.config.prior == "absorb":
            prior = torch.full((self.config.num_classes,), 0.01, dtype=torch.float64)
            prior[0] = 1.0
        elif self.config.prior == "tomask":
            prior = torch.full((self.config.num_classes,), 0.001, dtype=torch.float64)
            prior[-1] = 1.0
        else:
            raise ValueError(
                "Unsupported edge prior. Expected uniform, absorb, tomask, or custom_prior."
            )
        if prior.shape != (self.config.num_classes,):
            raise ValueError("edge prior length must equal num_classes")
        if (prior < 0).any() or float(prior.sum()) <= 0:
            raise ValueError("edge prior must be non-negative and have positive mass")
        return (prior / prior.sum()).float()

    def _make_betas(self) -> torch.Tensor:
        """Return beta schedule as a float tensor."""
        if self.config.beta_schedule == "linear":
            betas = torch.linspace(
                float(self.config.beta_start),
                float(self.config.beta_end),
                int(self.config.num_timesteps),
                dtype=torch.float64,
            )
        elif self.config.beta_schedule == "cosine":
            betas = _cosine_beta_schedule(int(self.config.num_timesteps)).to(torch.float64)
        else:
            raise ValueError("Unsupported beta_schedule. Expected linear or cosine.")
        if (betas <= 0).any() or (betas >= 1).any():
            raise ValueError("all beta values must be in (0, 1)")
        return betas.float()

    def _build_one_step_mats(
        self,
        betas: torch.Tensor,
        prior: torch.Tensor,
    ) -> torch.Tensor:
        """Construct q(e_t | e_{t-1}) matrices."""
        identity = torch.eye(self.config.num_classes, dtype=torch.float64)
        prior_row = prior.to(dtype=torch.float64).unsqueeze(0).repeat(self.config.num_classes, 1)
        matrices = []
        for beta in betas.to(dtype=torch.float64):
            matrices.append((1.0 - beta) * identity + beta * prior_row)
        return torch.stack(matrices, dim=0).float()

    @staticmethod
    def _build_cumulative_mats(q_onestep: torch.Tensor) -> torch.Tensor:
        """Construct q(e_t | e_0) matrices."""
        q_bar = q_onestep[0]
        matrices = [q_bar]
        for step_idx in range(1, q_onestep.shape[0]):
            q_bar = q_bar @ q_onestep[step_idx]
            matrices.append(q_bar)
        return torch.stack(matrices, dim=0)

    def _validate_labels(self, edge_type: torch.Tensor) -> torch.Tensor:
        """Validate and return long edge labels."""
        labels = edge_type.long()
        if labels.numel() == 0:
            return labels
        if labels.min().item() < 0 or labels.max().item() >= self.num_classes:
            raise ValueError("edge_type contains labels outside configured classes")
        return labels

    def _t_per_item(
        self,
        t: torch.Tensor,
        num_items: int,
        device: torch.device,
        batch: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return one timestep id per item."""
        t_device = t.to(device=device, dtype=torch.long)
        if batch is not None:
            batch_device = batch.to(device=device, dtype=torch.long)
            if t_device.ndim == 0 or t_device.numel() == 1:
                return self._clamp_timestep(t_device.reshape(()).expand(num_items))
            return self._clamp_timestep(t_device.index_select(0, batch_device))
        if t_device.ndim == 0 or t_device.numel() == 1:
            return self._clamp_timestep(t_device.reshape(()).expand(num_items))
        if t_device.numel() == num_items:
            return self._clamp_timestep(t_device.reshape(num_items))
        raise ValueError("t must be scalar, single-valued, per-item, or indexed by batch")

    def _clamp_timestep(self, t: torch.Tensor) -> torch.Tensor:
        """Clamp timestep tensor to valid transition matrix indices."""
        return t.long().clamp(0, self.num_timesteps - 1)


def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Return the cosine beta schedule used by diffusion baselines."""
    steps = timesteps + 1
    x = torch.linspace(0, steps, steps, dtype=torch.float64)
    alphas_bar = torch.cos(((x / steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1.0 - (alphas_bar[1:] / alphas_bar[:-1])
    return betas.clamp(1e-8, 0.999)
