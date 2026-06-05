"""
KL penalty toolkit for DDPO training.

Decoupled from any business logic — this module only contains
stateless mathematical functions operating on raw tensors.

Usage:
    from ddpo.kl import compute_gaussian_kl
    kl = compute_gaussian_kl(mu_ref, sigma_ref, mu_new, sigma_new, mask=node_mask)
"""

from .gaussian_kl import compute_gaussian_kl
from .logprob_kl import compute_logprob_k3_kl

__all__ = ["compute_gaussian_kl", "compute_logprob_k3_kl"]
