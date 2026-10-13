"""Unit tests for the DiffSBDD Gaussian-DPO mathematical primitives."""

from __future__ import annotations

import math

import pytest
import torch

from ddpo.diffusion_dpo.dpo_loss import compute_dpo_loss
from ddpo.diffusion_dpo.logprob import gaussian_transition_score
from ddpo.diffusion_dpo.aggregation import aggregate_transition_scores


def test_full_standard_normal_score_matches_manual_value() -> None:
    """The full score includes the Gaussian normalizing constant."""
    z_s = torch.zeros(1, 1, 4)
    mean = torch.zeros_like(z_s)
    mask = torch.ones(1, 1, 1)
    score = gaussian_transition_score(z_s, mean, torch.ones(()), mask, reduction="sum")
    assert score.total.item() == pytest.approx(-4.0 * 0.5 * math.log(2.0 * math.pi))


def test_mean_per_atom_does_not_depend_on_feature_dimension() -> None:
    """Mean-per-atom divides by atoms, not by atom feature dimensions."""
    z_s = torch.zeros(1, 2, 4)
    mean = torch.ones_like(z_s)
    mask = torch.ones(1, 2, 1)
    score = gaussian_transition_score(
        z_s,
        mean,
        torch.ones(()),
        mask,
        reduction="mean_per_atom",
        include_normal_constant=False,
        include_sigma_term=False,
    )
    assert score.coordinate.item() == pytest.approx(-1.5)
    assert score.atom_feature.item() == pytest.approx(-0.5)
    assert score.total.item() == pytest.approx(-2.0)


def test_padding_does_not_change_score() -> None:
    """Masked padding must not contribute to the transition likelihood."""
    z_s = torch.zeros(1, 2, 4)
    mean = torch.ones_like(z_s)
    mask = torch.ones(1, 2, 1)
    padded_z = torch.cat([z_s, torch.full((1, 1, 4), 99.0)], dim=1)
    padded_mean = torch.cat([mean, torch.full((1, 1, 4), -99.0)], dim=1)
    padded_mask = torch.cat([mask, torch.zeros(1, 1, 1)], dim=1)
    base = gaussian_transition_score(
        z_s, mean, torch.ones(()), mask, include_normal_constant=False
    )
    padded = gaussian_transition_score(
        padded_z, padded_mean, torch.ones(()), padded_mask, include_normal_constant=False
    )
    assert torch.allclose(base.total, padded.total)


def test_identical_current_and_reference_has_zero_relative_margin() -> None:
    """A current/reference identity must produce the neutral DPO loss."""
    score = torch.tensor([1.0, -2.0])
    facts = compute_dpo_loss(score, score.flip(0), score, score.flip(0), beta=0.1)
    assert torch.allclose(facts.relative_margin, torch.zeros_like(facts.relative_margin))
    assert facts.loss.item() == pytest.approx(math.log(2.0))


def test_dpo_requires_positive_beta() -> None:
    """Invalid beta values are rejected instead of silently changing the objective."""
    with pytest.raises(ValueError, match="beta"):
        compute_dpo_loss(torch.ones(1), torch.zeros(1), torch.ones(1), torch.zeros(1), beta=0.0)


def test_degenerate_gaussian_transition_is_rejected() -> None:
    """A zero-variance step is not silently treated as a Gaussian density."""
    with pytest.raises(ValueError, match="degenerate sigma"):
        gaussian_transition_score(
            torch.zeros(1, 1, 4),
            torch.zeros(1, 1, 4),
            torch.zeros(()),
            torch.ones(1, 1, 1),
        )


def test_multistep_score_aggregation_preserves_pair_axis() -> None:
    """Mean and sum aggregate timesteps while keeping winner/loser columns."""
    scores = [torch.tensor([[1.0, 3.0], [5.0, 7.0]]), torch.tensor([[3.0, 5.0], [7.0, 9.0]])]
    assert torch.equal(
        aggregate_transition_scores(scores, "mean"),
        torch.tensor([[2.0, 4.0], [6.0, 8.0]]),
    )
    assert torch.equal(
        aggregate_transition_scores(scores, "sum"),
        torch.tensor([[4.0, 8.0], [12.0, 16.0]]),
    )


def test_multistep_score_aggregation_rejects_empty_input() -> None:
    """An empty timestep list must not silently become a zero score."""
    with pytest.raises(ValueError, match="at least one timestep"):
        aggregate_transition_scores([], "mean")
