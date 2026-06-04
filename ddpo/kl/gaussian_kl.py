"""
Gaussian KL divergence for diffusion policy regularization.

Stateless utility with zero dependencies on higher-level ddpo business objects.
Only operates on raw torch tensors.
"""

import torch


VALID_GAUSSIAN_KL_DIRECTIONS = frozenset({"ref_current", "current_ref"})


def compute_gaussian_kl(
    mu_ref: torch.Tensor,
    sigma_ref: torch.Tensor,
    mu_new: torch.Tensor,
    sigma_new: torch.Tensor,
    mask: torch.Tensor | None = None,
    direction: str = "ref_current",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute analytic KL between pretrained and current Gaussian transitions.

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
        direction: KL direction. ``"ref_current"`` preserves the historical
            behavior KL(ref || current). ``"current_ref"`` computes the RL
            prior regularizer KL(current || ref).
        eps: Floor for sigma to guarantee numerical stability.

    Returns:
        Scalar tensor — mean KL penalty over batch and valid nodes.

    Historical formula:
        KL = log(sigma_new / sigma_ref)
            + (sigma_ref^2 + (mu_ref - mu_new)^2) / (2 * sigma_new^2)
            - 0.5
    """
    if direction not in VALID_GAUSSIAN_KL_DIRECTIONS:
        raise ValueError(
            f"Unsupported Gaussian KL direction={direction!r}; "
            f"expected one of {sorted(VALID_GAUSSIAN_KL_DIRECTIONS)}"
        )
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

    diff_sq = (mu_ref - mu_new) ** 2
    if direction == "ref_current":
        kl_elementwise = (
            torch.log(s_new / s_ref)
            + (s_ref ** 2 + diff_sq) / (2.0 * s_new ** 2)
            - 0.5
        )
    else:
        kl_elementwise = (
            torch.log(s_ref / s_new)
            + (s_new ** 2 + diff_sq) / (2.0 * s_ref ** 2)
            - 0.5
        )

    if mask is not None:
        return (kl_elementwise * mask).sum() / (mask.sum() + eps)

    return kl_elementwise.mean()
