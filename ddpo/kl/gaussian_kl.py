"""
Gaussian KL divergence for diffusion policy regularization.

Stateless utility — zero dependencies on ddpo / verl_diffusion business objects.
Only operates on raw torch tensors.
"""

import torch


def compute_gaussian_kl(
    mu_ref: torch.Tensor,
    sigma_ref: torch.Tensor,
    mu_new: torch.Tensor,
    sigma_new: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute element-wise KL(N(mu_ref, sigma_ref^2) || N(mu_new, sigma_new^2)).

    For diffusion reverse transitions, tensors typically have shape [B, N, D]:
        B   – batch size
        N   – max nodes (padded)
        D   – feature dimension (coords + atom types)

    sigma may be broadcastable (e.g. scalar, [B, 1, 1], or [B, N, 1]).

    Args:
        mu_ref: Mean from the frozen reference model (pretrained).
        sigma_ref: Std from the frozen reference model.
        mu_new: Mean from the current (training) model.
        sigma_new: Std from the current (training) model.
        mask: Optional [B, N, 1] boolean/float mask; when provided,
              KL is averaged over valid (unmasked) positions only.
        eps: Floor for sigma to guarantee numerical stability.

    Returns:
        Scalar tensor — mean KL penalty over batch and valid nodes.

    Formula (analytic Gaussian KL):
        KL = log(sigma_new / sigma_ref)
            + (sigma_ref^2 + (mu_ref - mu_new)^2) / (2 * sigma_new^2)
            - 0.5
    """
    s_new = torch.clamp(sigma_new, min=eps)
    s_ref = torch.clamp(sigma_ref, min=eps)

    # Ensure sigma broadcasts with mu [B, N, D].
    # DiffSBDD returns sigma as [B, 1] (per-sample scalar std);
    # right-pad dims so [B,1] → [B,1,1] which broadcasts with [B,N,D].
    target_ndim = mu_new.dim()
    while s_ref.dim() < target_ndim:
        s_ref = s_ref.unsqueeze(-1)
    while s_new.dim() < target_ndim:
        s_new = s_new.unsqueeze(-1)

    kl_elementwise = (
        torch.log(s_new / s_ref)
        + (s_ref ** 2 + (mu_ref - mu_new) ** 2) / (2.0 * s_new ** 2)
        - 0.5
    )

    if mask is not None:
        return (kl_elementwise * mask).sum() / (mask.sum() + eps)

    return kl_elementwise.mean()
