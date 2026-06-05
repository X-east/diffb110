"""Sampled log-probability KL approximations for diffusion PPO."""

from __future__ import annotations

import torch


def compute_logprob_k3_kl(
    log_prob_current: torch.Tensor,
    log_prob_ref: torch.Tensor,
    logratio_clip: float | None = 30.0,
) -> torch.Tensor:
    """Estimate KL(current || ref) with Schulman's non-negative K3 estimator.

    This mirrors the DePPA-style pretrained policy regularizer:

        logratio = log p_ref(action) - log p_current(action)
        ratio = exp(logratio)
        k3 = ratio - 1 - logratio

    For samples drawn from the current policy, ``mean(k3)`` estimates
    ``KL(current || ref)``. In PPO we use rollout samples from the old policy,
    which is the standard close-policy approximation used during updates.

    Args:
        log_prob_current: Log probability under the trainable policy.
        log_prob_ref: Log probability under the frozen pretrained policy.
        logratio_clip: Optional symmetric clamp applied before ``exp`` for
            numerical safety. Set to ``None`` to use the unclipped estimator.

    Returns:
        Scalar finite KL penalty. If every value is non-finite, returns zero on
        the same device and dtype so the training step can continue safely.
    """
    logratio = log_prob_ref - log_prob_current
    if logratio_clip is not None:
        clip = abs(float(logratio_clip))
        logratio = torch.clamp(logratio, min=-clip, max=clip)

    k3 = torch.exp(logratio) - 1.0 - logratio
    finite_mask = torch.isfinite(k3)
    if finite_mask.any():
        return k3[finite_mask].mean()
    return torch.zeros((), device=log_prob_current.device, dtype=log_prob_current.dtype)
