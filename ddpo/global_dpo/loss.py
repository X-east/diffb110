"""Pairwise binary-DPO and reward-weighted E2PO losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class GlobalDPOFacts:
    """DPO loss and pairwise diagnostics."""

    loss: torch.Tensor
    current_margin: torch.Tensor
    reference_margin: torch.Tensor
    relative_margin: torch.Tensor
    chosen_probability: torch.Tensor
    winner_weight: torch.Tensor
    loser_weight: torch.Tensor


def _validate_score_tensors(
    current_scores: torch.Tensor,
    reference_scores: torch.Tensor,
) -> None:
    """Validate the common score shape used by binary DPO and E2PO."""
    if current_scores.shape != reference_scores.shape:
        raise ValueError("current and reference score shapes must match")
    if current_scores.ndim != 2 or current_scores.shape[1] != 2:
        raise ValueError("scores must have shape [batch, 2]")
    if not torch.isfinite(current_scores).all() or not torch.isfinite(reference_scores).all():
        raise FloatingPointError("current/reference scores contain non-finite values")


def _build_facts(
    current_scores: torch.Tensor,
    reference_scores: torch.Tensor,
    *,
    beta: float,
    winner_weight: torch.Tensor,
    loser_weight: torch.Tensor,
) -> GlobalDPOFacts:
    """Build pairwise diagnostics from score tensors and soft labels."""
    current_margin = current_scores[:, 0] - current_scores[:, 1]
    reference_margin = reference_scores[:, 0] - reference_scores[:, 1]
    relative_margin = current_margin - reference_margin
    scaled = float(beta) * relative_margin
    winner_loss = -F.logsigmoid(scaled)
    loser_loss = -F.logsigmoid(-scaled)
    loss = (winner_weight * winner_loss + loser_weight * loser_loss).mean()
    return GlobalDPOFacts(
        loss=loss,
        current_margin=current_margin,
        reference_margin=reference_margin,
        relative_margin=relative_margin,
        chosen_probability=torch.sigmoid(scaled),
        winner_weight=winner_weight,
        loser_weight=loser_weight,
    )


def compute_global_dpo_loss(
    current_scores: torch.Tensor,
    reference_scores: torch.Tensor,
    *,
    beta: float,
) -> GlobalDPOFacts:
    """Compute DPO from score tensors shaped ``[batch, 2]``."""
    if beta <= 0:
        raise ValueError("beta must be positive")
    _validate_score_tensors(current_scores, reference_scores)
    winner_weight = torch.ones(
        current_scores.shape[0],
        device=current_scores.device,
        dtype=current_scores.dtype,
    )
    loser_weight = torch.zeros_like(winner_weight)
    return _build_facts(
        current_scores,
        reference_scores,
        beta=beta,
        winner_weight=winner_weight,
        loser_weight=loser_weight,
    )


def compute_global_e2po_loss(
    current_scores: torch.Tensor,
    reference_scores: torch.Tensor,
    winner_rewards: torch.Tensor,
    loser_rewards: torch.Tensor,
    *,
    beta: float,
    reward_temperature: float,
) -> GlobalDPOFacts:
    """Compute AliDiff-style reward-weighted preference optimization.

    ``current_scores`` and ``reference_scores`` are larger-is-better scores
    shaped ``[batch, 2]`` in winner/loser order.  The reward temperatures are
    applied only to the frozen endpoint rewards; they do not alter the model
    score scale.
    """
    if beta <= 0:
        raise ValueError("beta must be positive")
    if reward_temperature <= 0:
        raise ValueError("reward_temperature must be positive")
    _validate_score_tensors(current_scores, reference_scores)
    expected_shape = (current_scores.shape[0],)
    if tuple(winner_rewards.shape) != expected_shape or tuple(loser_rewards.shape) != expected_shape:
        raise ValueError("winner_rewards and loser_rewards must have shape [batch]")
    if (
        winner_rewards.device != current_scores.device
        or loser_rewards.device != current_scores.device
    ):
        raise ValueError("reward tensors must be on the same device as score tensors")
    if not torch.isfinite(winner_rewards).all() or not torch.isfinite(loser_rewards).all():
        raise FloatingPointError("endpoint rewards contain non-finite values")

    reward_logits = torch.stack([winner_rewards, loser_rewards], dim=1)
    reward_weights = torch.softmax(float(reward_temperature) * reward_logits, dim=1)
    return _build_facts(
        current_scores,
        reference_scores,
        beta=beta,
        winner_weight=reward_weights[:, 0],
        loser_weight=reward_weights[:, 1],
    )
