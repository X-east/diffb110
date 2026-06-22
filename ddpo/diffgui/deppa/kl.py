"""DePPA KL helper formulas for DiffGui experiments."""

from __future__ import annotations

import torch


def compute_deppa_k3_kl(
    reference_logprob: torch.Tensor,
    current_logprob: torch.Tensor,
    *,
    clip: float | None = 30.0,
) -> torch.Tensor:
    """Return DePPA K3 KL estimate ``exp(log r) - 1 - log r``.

    Here ``log r = log p_ref(a|s) - log p_theta(a|s)`` for the same DiffGui
    transition.  The helper is deliberately model-agnostic; callers are
    responsible for producing aligned reference/current transition logprobs.
    """
    logratio = reference_logprob - current_logprob
    if clip is not None:
        limit = abs(float(clip))
        logratio = logratio.clamp(min=-limit, max=limit)
    return torch.exp(logratio) - 1.0 - logratio
