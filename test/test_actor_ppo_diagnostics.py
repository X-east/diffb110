"""Tests for PPO/KL diagnostics emitted by the DDPO actor."""

from __future__ import annotations

import math

import pytest
import torch

from ddpo.runtime.actor import EDMActor


def _make_actor(kl_coef: float = 0.2, num_timesteps: int = 1000) -> EDMActor:
    """Create a minimal actor instance without constructing the full model stack."""
    actor = EDMActor.__new__(EDMActor)
    actor.kl_coef = kl_coef
    actor.num_timesteps = num_timesteps
    return actor


def test_ppo_diagnostics_filter_nonfinite_values() -> None:
    """Diagnostic metrics should stay scalar and finite when inputs contain NaN/Inf."""
    actor = _make_actor(kl_coef=0.2)
    logratio = torch.tensor([0.0, 0.2, float("nan"), -0.1, float("inf")])
    ratio = torch.exp(logratio)
    advantages = torch.tensor([0.0, 1.0, float("nan"), -2.0, float("inf")])
    kl_penalty = torch.tensor(3.0)

    diagnostics = actor._ppo_diagnostics(
        logratio=logratio,
        ratio=ratio,
        advantages=advantages,
        kl_penalty=kl_penalty,
        device=torch.device("cpu"),
    )

    expected_approx_kl = 0.5 * ((0.0**2 + 0.2**2 + (-0.1) ** 2) / 3.0)
    assert diagnostics["PPO_ApproxKL"].item() == pytest.approx(expected_approx_kl)
    assert diagnostics["PPO_Ratio_Max"].item() == pytest.approx(math.exp(0.2))
    assert diagnostics["Advantage_NonZeroFrac"].item() == pytest.approx(2.0 / 3.0)
    assert diagnostics["KL_Coef"].item() == pytest.approx(0.2)
    assert diagnostics["KL_Penalty_Weighted"].item() == pytest.approx(0.6)

    for value in diagnostics.values():
        assert value.ndim == 0
        assert torch.isfinite(value)


def test_timestep_bucket_uses_coarse_diffusion_regions() -> None:
    """Timestep buckets should separate high/mid/low diffusion regions."""
    actor = _make_actor(num_timesteps=1000)

    assert actor._timestep_bucket(torch.tensor([900.0, 800.0])) == "HighT"
    assert actor._timestep_bucket(torch.tensor([500.0, 400.0])) == "MidT"
    assert actor._timestep_bucket(torch.tensor([100.0, 200.0])) == "LowT"
