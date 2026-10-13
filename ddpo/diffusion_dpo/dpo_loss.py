"""Pairwise DPO objective for saved DiffSBDD transition scores."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DPOFacts:
    """DPO loss and diagnostics for a batch of preference pairs."""

    loss: torch.Tensor
    current_margin: torch.Tensor
    reference_margin: torch.Tensor
    relative_margin: torch.Tensor
    chosen_probability: torch.Tensor


def compute_dpo_loss(
    current_winner: torch.Tensor,
    current_loser: torch.Tensor,
    reference_winner: torch.Tensor,
    reference_loser: torch.Tensor,
    *,
    beta: float,
) -> DPOFacts:
    """Compute the reference-relative pairwise DPO loss."""
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")
    shapes = {tuple(value.shape) for value in (current_winner, current_loser, reference_winner, reference_loser)}
    if len(shapes) != 1:
        raise ValueError(f"DPO score shapes must match, got {sorted(shapes)}")
    current_margin = current_winner - current_loser
    reference_margin = reference_winner - reference_loser
    relative_margin = current_margin - reference_margin
    scaled = float(beta) * relative_margin
    loss = -F.logsigmoid(scaled).mean()
    return DPOFacts(
        loss=loss,
        current_margin=current_margin,
        reference_margin=reference_margin,
        relative_margin=relative_margin,
        chosen_probability=torch.sigmoid(scaled),
    )
