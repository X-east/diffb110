"""Unit tests for stateless PyTorch IQL losses."""

from __future__ import annotations

import math

import pytest
import torch

from ddpo.offline_rl.losses import (
    compute_actor_loss,
    compute_advantage,
    compute_advantage_weight,
    compute_q_loss,
    compute_td_target,
    compute_value_loss,
    effective_sample_size,
    expectile_loss,
)


def test_expectile_loss_matches_asymmetric_squared_error() -> None:
    """Positive and non-positive differences should use opposite weights."""
    difference = torch.tensor([2.0, -3.0, 0.0])

    loss = expectile_loss(difference, expectile=0.8)

    assert loss.tolist() == pytest.approx([3.2, 1.8, 0.0])


@pytest.mark.parametrize("expectile", [-0.1, 0.0, 1.0, 1.1, math.nan])
def test_expectile_loss_rejects_invalid_coefficients(expectile: float) -> None:
    """Expectile regression requires a finite open-interval coefficient."""
    with pytest.raises(ValueError, match="expectile"):
        expectile_loss(torch.ones(1), expectile=expectile)


def test_value_loss_detaches_twin_q_target() -> None:
    """The value update must not backpropagate into either target critic."""
    q1 = torch.tensor([2.0, 0.0], requires_grad=True)
    q2 = torch.tensor([3.0, 1.0], requires_grad=True)
    value = torch.tensor([1.0, 1.0], requires_grad=True)

    loss = compute_value_loss(q1, q2, value, expectile=0.8)
    loss.backward()

    assert loss.item() == pytest.approx(0.5)
    assert q1.grad is None
    assert q2.grad is None
    assert value.grad.tolist() == pytest.approx([-0.8, 0.2])


def test_td_target_masks_terminal_transition_and_detaches_inputs() -> None:
    """Terminal reward must not bootstrap and the target graph must be frozen."""
    reward = torch.tensor([1.0, 2.0], requires_grad=True)
    done = torch.tensor([False, True])
    next_value = torch.tensor([10.0, 10.0], requires_grad=True)

    target = compute_td_target(reward, done, next_value, discount=1.0)

    assert target.tolist() == pytest.approx([11.0, 2.0])
    assert not target.requires_grad


def test_td_target_accepts_binary_numeric_done() -> None:
    """Numeric zero/one terminal facts should preserve the same semantics."""
    reward = torch.tensor([1.0, 2.0])
    done = torch.tensor([0.0, 1.0])
    next_value = torch.tensor([10.0, 10.0])

    target = compute_td_target(reward, done, next_value, discount=0.5)

    assert target.tolist() == pytest.approx([6.0, 2.0])


def test_td_target_rejects_nonbinary_done() -> None:
    """Ambiguous numeric terminal masks must fail instead of being truncated."""
    with pytest.raises(ValueError, match="only 0 or 1"):
        compute_td_target(
            reward=torch.ones(2),
            done=torch.tensor([0.0, 0.5]),
            next_value=torch.ones(2),
            discount=1.0,
        )


@pytest.mark.parametrize("discount", [-0.1, 1.1, math.inf])
def test_td_target_rejects_invalid_discount(discount: float) -> None:
    """The core loss should reject invalid return conventions early."""
    with pytest.raises(ValueError, match="discount"):
        compute_td_target(
            reward=torch.ones(1),
            done=torch.zeros(1, dtype=torch.bool),
            next_value=torch.ones(1),
            discount=discount,
        )


def test_q_loss_matches_reference_mse_and_detaches_target() -> None:
    """Twin-Q MSE should sum both mean regressions without target gradients."""
    q1 = torch.tensor([1.0, 3.0], requires_grad=True)
    q2 = torch.tensor([2.0, 4.0], requires_grad=True)
    target = torch.tensor([2.0, 2.0], requires_grad=True)

    loss = compute_q_loss(q1, q2, target, loss_type="mse")
    loss.backward()

    assert loss.item() == pytest.approx(3.0)
    assert q1.grad.tolist() == pytest.approx([-1.0, 1.0])
    assert q2.grad.tolist() == pytest.approx([0.0, 2.0])
    assert target.grad is None


def test_q_loss_supports_explicit_smooth_l1_ablation() -> None:
    """The robust Q-loss option should remain explicit and deterministic."""
    q1 = torch.tensor([0.0])
    q2 = torch.tensor([4.0])
    target = torch.tensor([2.0])

    loss = compute_q_loss(q1, q2, target, loss_type="smooth_l1")

    assert loss.item() == pytest.approx(3.0)


def test_q_loss_rejects_unknown_type() -> None:
    """A misspelled Q-loss type must not silently change the objective."""
    with pytest.raises(ValueError, match="loss_type"):
        compute_q_loss(
            torch.ones(1),
            torch.ones(1),
            torch.ones(1),
            loss_type="mae",  # type: ignore[arg-type]
        )


def test_advantage_uses_minimum_twin_q_and_is_detached() -> None:
    """Policy extraction should use a pessimistic detached advantage."""
    q1 = torch.tensor([3.0, 1.0], requires_grad=True)
    q2 = torch.tensor([2.0, 4.0], requires_grad=True)
    value = torch.tensor([1.0, 2.0], requires_grad=True)

    advantage = compute_advantage(q1, q2, value)

    assert advantage.tolist() == pytest.approx([1.0, -1.0])
    assert not advantage.requires_grad


def test_advantage_weight_clips_before_exp_to_prevent_overflow() -> None:
    """Huge positive advantages should stay finite at the configured cap."""
    advantage = torch.tensor([1.0e6, 0.0, -1.0], requires_grad=True)

    weight = compute_advantage_weight(
        advantage,
        temperature=1.0,
        max_weight=100.0,
    )

    assert weight.tolist() == pytest.approx([100.0, 1.0, math.exp(-1.0)])
    assert torch.isfinite(weight).all()
    assert not weight.requires_grad


@pytest.mark.parametrize(
    ("temperature", "max_weight"),
    [
        (0.0, 100.0),
        (-1.0, 100.0),
        (1.0, 0.0),
        (1.0, math.inf),
    ],
)
def test_advantage_weight_rejects_invalid_hyperparameters(
    temperature: float,
    max_weight: float,
) -> None:
    """Actor weighting requires finite positive scale controls."""
    with pytest.raises(ValueError):
        compute_advantage_weight(
            torch.ones(1),
            temperature=temperature,
            max_weight=max_weight,
        )


def test_actor_loss_only_backpropagates_into_log_prob() -> None:
    """Detached weights must not create a critic-to-actor gradient path."""
    log_prob = torch.tensor([-2.0, -1.0], requires_grad=True)
    weight = torch.tensor([3.0, 1.0], requires_grad=True)

    loss = compute_actor_loss(log_prob, weight)
    loss.backward()

    assert loss.item() == pytest.approx(3.5)
    assert log_prob.grad.tolist() == pytest.approx([-1.5, -0.5])
    assert weight.grad is None


def test_effective_sample_size_matches_definition() -> None:
    """ESS should expose how strongly actor weights concentrate."""
    weight = torch.tensor([1.0, 1.0, 2.0])

    ess = effective_sample_size(weight)

    assert ess.item() == pytest.approx(16.0 / 6.0)


def test_effective_sample_size_is_zero_for_zero_weights() -> None:
    """A fully suppressed actor batch should have zero effective samples."""
    ess = effective_sample_size(torch.zeros(4))

    assert ess.item() == 0.0


def test_effective_sample_size_rejects_negative_weights() -> None:
    """Negative values are not valid importance-style actor weights."""
    with pytest.raises(ValueError, match="non-negative"):
        effective_sample_size(torch.tensor([1.0, -1.0]))


def test_iql_losses_reject_accidental_broadcasting() -> None:
    """Per-transition shapes must match instead of broadcasting silently."""
    with pytest.raises(ValueError, match="identical shapes"):
        compute_actor_loss(
            log_prob=torch.ones(2),
            weight=torch.ones(2, 1),
        )
