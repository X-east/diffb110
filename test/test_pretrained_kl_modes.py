"""Tests for pretrained KL regularization modes."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from ddpo.kl import compute_gaussian_kl, compute_logprob_k3_kl
from ddpo.runtime.actor import EDMActor


class _DummyPolicy:
    """Minimal callable policy returning fixed DDPO logprob, mu, and sigma."""

    def __init__(self, log_prob: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> None:
        self.log_prob = log_prob
        self.mu = mu
        self.sigma = sigma

    def __call__(self, **_: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return fixed diffusion transition statistics."""
        return self.log_prob, self.mu, self.sigma


class _DummyMaskHelper:
    """Minimal helper exposing DiffSBDD-style mask construction."""

    def get_mask(
        self,
        nodesxsample: torch.Tensor,
        batch_size: int,
        max_n_nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return valid-node and all-edge masks for a padded batch."""
        node_ids = torch.arange(max_n_nodes).unsqueeze(0)
        node_mask = (node_ids < nodesxsample.cpu().unsqueeze(1)).float().unsqueeze(-1)
        edge_mask = torch.ones(batch_size, max_n_nodes, max_n_nodes)
        return node_mask, edge_mask


def _make_actor(kl_mode: str, kl_k3_logratio_clip: float | None = None) -> EDMActor:
    """Create a minimal actor instance for KL helper tests."""
    actor = EDMActor.__new__(EDMActor)
    actor.kl_mode = kl_mode
    actor.kl_k3_logratio_clip = kl_k3_logratio_clip
    actor.kl_coef = 0.2
    actor.clip_range = 0.2
    actor.num_timesteps = 100
    actor.max_n_nodes = 1
    return actor


def _minimal_actor_config(kl_mode: str) -> dict[str, object]:
    """Build the smallest config needed by EDMActor.__init__."""
    return {
        "train": {
            "kl_coef": 0.2,
            "kl_mode": kl_mode,
            "kl_k3_logratio_clip": 12.0,
            "learning_rate": 1e-4,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_weight_decay": 0.0,
            "adam_epsilon": 1e-8,
            "train_micro_batch_size": 1,
            "max_grad_norm": 1.0,
            "clip_range": 0.2,
        },
        "model": {"time_step": 100},
    }


def test_actor_reads_kl_mode_from_config() -> None:
    """EDMActor should expose KL mode and K3 clamp as config-driven knobs."""
    actor = EDMActor(
        model=nn.Linear(1, 1),
        config=_minimal_actor_config("pretrained_logprob_k3"),
        model_impl=_DummyMaskHelper(),
    )

    assert actor.kl_mode == "pretrained_logprob_k3"
    assert actor.kl_k3_logratio_clip == pytest.approx(12.0)


def test_actor_rejects_unknown_kl_mode() -> None:
    """A typo in train.kl_mode should fail early instead of silently changing KL."""
    with pytest.raises(ValueError, match="Unsupported train.kl_mode"):
        EDMActor(
            model=nn.Linear(1, 1),
            config=_minimal_actor_config("bad_mode"),
            model_impl=_DummyMaskHelper(),
        )


def test_gaussian_kl_supports_both_directions() -> None:
    """Analytic KL direction should change the sigma denominator as expected."""
    mu_ref = torch.tensor([[[0.0]]])
    sigma_ref = torch.tensor([[[1.0]]])
    mu_new = torch.tensor([[[1.0]]])
    sigma_new = torch.tensor([[[2.0]]])

    ref_current = compute_gaussian_kl(
        mu_ref=mu_ref,
        sigma_ref=sigma_ref,
        mu_new=mu_new,
        sigma_new=sigma_new,
        direction="ref_current",
    )
    current_ref = compute_gaussian_kl(
        mu_ref=mu_ref,
        sigma_ref=sigma_ref,
        mu_new=mu_new,
        sigma_new=sigma_new,
        direction="current_ref",
    )

    expected_ref_current = math.log(2.0) + (1.0 + 1.0) / (2.0 * 4.0) - 0.5
    expected_current_ref = math.log(0.5) + (4.0 + 1.0) / 2.0 - 0.5
    assert ref_current.item() == pytest.approx(expected_ref_current)
    assert current_ref.item() == pytest.approx(expected_current_ref)
    assert current_ref.item() > ref_current.item()


def test_gaussian_kl_mask_keeps_historical_valid_node_average() -> None:
    """Masked Gaussian KL should sum feature terms and average by valid nodes."""
    mu_ref = torch.zeros(1, 2, 2)
    sigma_ref = torch.ones(1, 1, 1)
    mu_new = torch.ones(1, 2, 2)
    sigma_new = torch.ones(1, 1, 1)
    mask = torch.tensor([[[1.0], [0.0]]])

    kl = compute_gaussian_kl(
        mu_ref=mu_ref,
        sigma_ref=sigma_ref,
        mu_new=mu_new,
        sigma_new=sigma_new,
        mask=mask,
        direction="current_ref",
    )

    assert kl.item() == pytest.approx(1.0)


def test_logprob_k3_matches_deppa_style_formula() -> None:
    """K3 logprob KL should compute exp(ref-current) - 1 - (ref-current)."""
    log_prob_current = torch.tensor([-1.0, -2.0])
    log_prob_ref = torch.tensor([-1.2, -1.5])

    kl = compute_logprob_k3_kl(log_prob_current, log_prob_ref, logratio_clip=None)

    logratio = log_prob_ref - log_prob_current
    expected = (torch.exp(logratio) - 1.0 - logratio).mean()
    assert kl.item() == pytest.approx(expected.item())


def test_logprob_k3_clip_keeps_penalty_finite() -> None:
    """Optional clipping should prevent overflow in the sampled K3 estimator."""
    log_prob_current = torch.tensor([-1000.0])
    log_prob_ref = torch.tensor([0.0])

    kl = compute_logprob_k3_kl(log_prob_current, log_prob_ref, logratio_clip=10.0)

    assert torch.isfinite(kl)
    assert kl.item() == pytest.approx(math.exp(10.0) - 11.0)


@pytest.mark.parametrize(
    ("kl_mode", "expected"),
    [
        ("analytic_ref_current", math.log(2.0) + (1.0 + 1.0) / (2.0 * 4.0) - 0.5),
        ("analytic_current_ref", math.log(0.5) + (4.0 + 1.0) / 2.0 - 0.5),
        ("pretrained_logprob_k3", math.exp(0.5) - 1.5),
    ],
)
def test_actor_dispatches_pretrained_kl_modes(kl_mode: str, expected: float) -> None:
    """EDMActor should route each configured KL mode to the correct helper."""
    actor = _make_actor(kl_mode=kl_mode, kl_k3_logratio_clip=None)
    mu_ref = torch.tensor([[[0.0]]])
    sigma_ref = torch.tensor([[[1.0]]])
    mu_new = torch.tensor([[[1.0]]])
    sigma_new = torch.tensor([[[2.0]]])
    node_mask = torch.ones(1, 1, 1)
    log_prob_current = torch.tensor([-2.0])
    log_prob_ref = torch.tensor([-1.5])

    kl = actor._pretrained_kl_penalty(
        log_prob_current=log_prob_current,
        log_prob_ref=log_prob_ref,
        mu_new=mu_new,
        sigma_new=sigma_new,
        mu_ref=mu_ref,
        sigma_ref=sigma_ref,
        node_mask=node_mask,
    )

    assert kl.item() == pytest.approx(expected)


@pytest.mark.parametrize(
    ("kl_mode", "expected_kl"),
    [
        ("analytic_ref_current", math.log(2.0) + (1.0 + 1.0) / (2.0 * 4.0) - 0.5),
        ("analytic_current_ref", math.log(0.5) + (4.0 + 1.0) / 2.0 - 0.5),
        ("pretrained_logprob_k3", math.exp(0.5) - 1.5),
    ],
)
def test_calculate_loss_uses_configured_pretrained_kl_mode(kl_mode: str, expected_kl: float) -> None:
    """calculate_loss should execute the configured KL branch when kl_coef > 0."""
    actor = _make_actor(kl_mode=kl_mode, kl_k3_logratio_clip=None)
    mu_ref = torch.tensor([[[0.0]]])
    sigma_ref = torch.tensor([[[1.0]]])
    mu_new = torch.tensor([[[1.0]]])
    sigma_new = torch.tensor([[[2.0]]])
    actor.model = _DummyPolicy(log_prob=torch.tensor([-2.0], requires_grad=True), mu=mu_new, sigma=sigma_new)
    actor.ref_model = _DummyPolicy(log_prob=torch.tensor([-1.5]), mu=mu_ref, sigma=sigma_ref)
    actor.model_impl = _DummyMaskHelper()

    _, _, _, kl_penalty, diagnostics = actor.calculate_loss(
        latents=torch.zeros(1, 1, 1),
        timesteps=torch.tensor([1.0]),
        next_latents=torch.zeros(1, 1, 1),
        log_prob_old=torch.tensor([-2.0]),
        advantages=torch.tensor([1.0]),
        nodesxsample=torch.tensor([1]),
        context=None,
    )

    assert kl_penalty.item() == pytest.approx(expected_kl)
    assert diagnostics["KL_Penalty_Weighted"].item() == pytest.approx(0.2 * expected_kl)
