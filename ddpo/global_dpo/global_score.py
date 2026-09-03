"""Global DiffSBDD endpoint denoising scores."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .paired_noise import GlobalNoisyBatch


@dataclass(frozen=True)
class GlobalDenoisingScore:
    """Per-endpoint x/v losses and weighted global preference score."""

    score: torch.Tensor
    x_loss: torch.Tensor
    v_loss: torch.Tensor
    prediction: torch.Tensor


def component_epsilon_losses(
    prediction: torch.Tensor,
    epsilon: torch.Tensor,
    ligand_mask: torch.Tensor,
    sizes: torch.Tensor,
    *,
    coordinate_dim: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-sample coordinate and atom-feature epsilon MSE."""
    if prediction.shape != epsilon.shape or prediction.ndim != 2:
        raise ValueError(f"prediction/epsilon shape mismatch: {prediction.shape} vs {epsilon.shape}")
    if ligand_mask.ndim != 1 or ligand_mask.shape[0] != prediction.shape[0]:
        raise ValueError("ligand_mask must index every flat endpoint atom")
    if sizes.ndim != 1 or int(sizes.sum().item()) != prediction.shape[0]:
        raise ValueError("sizes do not match flat endpoint atom count")
    if not 0 < coordinate_dim < prediction.shape[1]:
        raise ValueError("coordinate_dim must split the prediction feature dimension")
    squared = (prediction - epsilon).square()
    batch_size = int(sizes.shape[0])
    x_sum = torch.zeros(batch_size, device=prediction.device, dtype=prediction.dtype)
    v_sum = torch.zeros_like(x_sum)
    x_sum.index_add_(0, ligand_mask, squared[:, :coordinate_dim].sum(dim=1))
    v_sum.index_add_(0, ligand_mask, squared[:, coordinate_dim:].sum(dim=1))
    x_loss = x_sum / (float(coordinate_dim) * sizes.to(prediction.dtype)).clamp_min(1.0)
    v_loss = v_sum / (float(prediction.shape[1] - coordinate_dim) * sizes.to(prediction.dtype)).clamp_min(1.0)
    return x_loss, v_loss


def score_from_component_losses(
    x_loss: torch.Tensor,
    v_loss: torch.Tensor,
    *,
    x_weight: float,
    v_weight: float,
) -> torch.Tensor:
    """Convert component MSEs into a larger-is-better global denoising score."""
    if x_loss.shape != v_loss.shape:
        raise ValueError("x_loss and v_loss shapes must match")
    if x_weight < 0 or v_weight < 0 or x_weight + v_weight <= 0:
        raise ValueError("x_weight and v_weight must be non-negative and not both zero")
    return -(float(x_weight) * x_loss + float(v_weight) * v_loss)


def compute_global_score(
    model: object,
    batch: GlobalNoisyBatch,
    *,
    x_weight: float,
    v_weight: float,
) -> GlobalDenoisingScore:
    """Run native DiffSBDD denoising and compute global endpoint score."""
    prediction, _ = model.ddpm.dynamics(
        batch.z_t,
        batch.xh0_pocket,
        batch.timestep,
        batch.ligand_mask,
        batch.pocket_mask,
    )
    x_loss, v_loss = component_epsilon_losses(
        prediction,
        batch.epsilon,
        batch.ligand_mask,
        batch.sizes,
        coordinate_dim=int(model.ddpm.n_dims),
    )
    return GlobalDenoisingScore(
        score=score_from_component_losses(
            x_loss,
            v_loss,
            x_weight=x_weight,
            v_weight=v_weight,
        ),
        x_loss=x_loss,
        v_loss=v_loss,
        prediction=prediction,
    )
