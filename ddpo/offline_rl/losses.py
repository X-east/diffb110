"""Stateless PyTorch losses for Implicit Q-Learning.

The equations follow Kostrikov, Nair, and Levine, "Offline Reinforcement
Learning with Implicit Q-Learning" (2021):

* https://arxiv.org/abs/2110.06169
* https://github.com/ikostrikov/implicit_q_learning

This is a native PyTorch adaptation for DiffB110. It does not depend on the
original JAX runtime or on TorchRL, and it intentionally has no knowledge of
DiffSBDD trajectories, graph models, optimizers, or distributed training.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch.nn import functional as F


Q_LOSS_TYPES = ("mse", "smooth_l1")
QLossType = Literal["mse", "smooth_l1"]


def _require_same_shape(**tensors: torch.Tensor) -> None:
    """Reject implicit broadcasting across per-transition tensors."""
    shapes = {name: tuple(tensor.shape) for name, tensor in tensors.items()}
    if len(set(shapes.values())) != 1:
        rendered = ", ".join(f"{name}={shape}" for name, shape in shapes.items())
        raise ValueError(f"IQL tensors must have identical shapes; got {rendered}")


def _validate_expectile(expectile: float) -> None:
    """Validate the open interval required by expectile regression."""
    if not math.isfinite(expectile) or not 0.0 < expectile < 1.0:
        raise ValueError(f"expectile must be finite and in (0, 1), got {expectile!r}")


def _validate_discount(discount: float) -> None:
    """Validate the finite discounted-return range."""
    if not math.isfinite(discount) or not 0.0 <= discount <= 1.0:
        raise ValueError(f"discount must be finite and in [0, 1], got {discount!r}")


def expectile_loss(difference: torch.Tensor, expectile: float) -> torch.Tensor:
    """Return elementwise asymmetric squared error for expectile regression.

    Args:
        difference: Typically ``Q(s, a) - V(s)``.
        expectile: Upper expectile coefficient in the open interval ``(0, 1)``.

    Returns:
        A tensor with the same shape as ``difference``.
    """
    _validate_expectile(expectile)
    weight = torch.where(
        difference > 0,
        difference.new_tensor(expectile),
        difference.new_tensor(1.0 - expectile),
    )
    return weight * difference.square()


def compute_td_target(
    reward: torch.Tensor,
    done: torch.Tensor,
    next_value: torch.Tensor,
    discount: float,
) -> torch.Tensor:
    """Build a detached one-step IQL target.

    The target is ``r + gamma * (1 - done) * V(s')``. ``done`` may be boolean
    or numeric, but must use one for terminal transitions and zero otherwise.
    The result is detached so critic updates cannot accidentally train the
    reward or value-producing graph.

    Args:
        reward: Per-transition rewards.
        done: Per-transition terminal flags with the same shape as ``reward``.
        next_value: Value estimates for successor states.
        discount: Discount factor in ``[0, 1]``. DiffSBDD's finite 1001-action
            denoising MDP is expected to start with ``1.0``.

    Returns:
        A detached TD target with the same shape as ``reward``.
    """
    _validate_discount(discount)
    _require_same_shape(reward=reward, done=done, next_value=next_value)

    if done.dtype == torch.bool:
        not_done = ~done
    else:
        valid_done = torch.logical_or(done == 0, done == 1)
        if not bool(torch.all(valid_done)):
            raise ValueError("numeric done flags must contain only 0 or 1")
        not_done = done == 0

    target = reward + discount * not_done.to(dtype=reward.dtype) * next_value
    return target.detach()


def compute_value_loss(
    q1: torch.Tensor,
    q2: torch.Tensor,
    value: torch.Tensor,
    expectile: float,
) -> torch.Tensor:
    """Return the scalar IQL value loss using a detached minimum twin-Q target.

    Args:
        q1: First target critic estimate for dataset actions.
        q2: Second target critic estimate for dataset actions.
        value: Current value-network estimate.
        expectile: Upper expectile coefficient.

    Returns:
        Mean expectile-regression loss.
    """
    _require_same_shape(q1=q1, q2=q2, value=value)
    target_q = torch.minimum(q1, q2).detach()
    return expectile_loss(target_q - value, expectile).mean()


def compute_q_loss(
    q1: torch.Tensor,
    q2: torch.Tensor,
    td_target: torch.Tensor,
    loss_type: QLossType = "mse",
) -> torch.Tensor:
    """Return the summed twin-Q regression loss.

    Args:
        q1: First online critic estimate.
        q2: Second online critic estimate.
        td_target: Detached target from :func:`compute_td_target`.
        loss_type: ``"mse"`` for the reference IQL objective or
            ``"smooth_l1"`` for a robust pilot ablation.

    Returns:
        Sum of the mean Q1 and Q2 regression losses.
    """
    _require_same_shape(q1=q1, q2=q2, td_target=td_target)
    if loss_type not in Q_LOSS_TYPES:
        raise ValueError(f"loss_type must be one of {Q_LOSS_TYPES}, got {loss_type!r}")

    target = td_target.detach()
    if loss_type == "mse":
        return F.mse_loss(q1, target) + F.mse_loss(q2, target)
    return F.smooth_l1_loss(q1, target) + F.smooth_l1_loss(q2, target)


def compute_advantage(
    q1: torch.Tensor,
    q2: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Return detached IQL advantages from the minimum twin-Q estimate.

    Args:
        q1: First target critic estimate for dataset actions.
        q2: Second target critic estimate for dataset actions.
        value: Value-network estimate for the same states.

    Returns:
        Detached ``min(Q1, Q2) - V`` values.
    """
    _require_same_shape(q1=q1, q2=q2, value=value)
    return (torch.minimum(q1, q2) - value).detach()


def compute_advantage_weight(
    advantage: torch.Tensor,
    temperature: float,
    max_weight: float,
) -> torch.Tensor:
    """Exponentiate detached advantages with an overflow-safe upper bound.

    Args:
        advantage: Per-transition advantages.
        temperature: Positive inverse temperature multiplying the advantage.
        max_weight: Finite upper bound for the exponentiated weight.

    Returns:
        Detached non-negative actor weights with the same shape as
        ``advantage``.
    """
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"temperature must be finite and positive, got {temperature!r}")
    if not math.isfinite(max_weight) or max_weight <= 0.0:
        raise ValueError(f"max_weight must be finite and positive, got {max_weight!r}")

    max_exponent = math.log(max_weight)
    exponent = (advantage.detach() * temperature).clamp(max=max_exponent)
    return torch.exp(exponent)


def compute_actor_loss(
    log_prob: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Return the advantage-weighted behavior-cloning loss.

    Args:
        log_prob: Current-policy log probability of each dataset action.
        weight: Non-negative per-transition weights, normally produced by
            :func:`compute_advantage_weight`.

    Returns:
        Mean negative weighted log probability. The weights are detached so
        actor updates cannot train the critics or value network.
    """
    _require_same_shape(log_prob=log_prob, weight=weight)
    return -(weight.detach() * log_prob).mean()


def effective_sample_size(weight: torch.Tensor) -> torch.Tensor:
    """Return the scalar effective sample size of non-negative actor weights.

    Args:
        weight: Non-negative weights of any shape.

    Returns:
        ``(sum(w) ** 2) / sum(w ** 2)``. Returns zero when all weights are
        zero.
    """
    if bool(torch.any(weight < 0)):
        raise ValueError("effective sample size requires non-negative weights")

    detached = weight.detach()
    numerator = detached.sum().square()
    denominator = detached.square().sum()
    zero = denominator.new_zeros(())
    return torch.where(denominator > 0, numerator / denominator, zero)
