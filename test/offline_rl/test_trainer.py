"""Tests for BC/AWR actor updates and the complete IQL train step."""

from __future__ import annotations

import math
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
from torch import nn

from ddpo.offline_rl.critic_adapter import PocketContext
from ddpo.offline_rl.models import (
    GeometricCriticConfig,
    GeometricValueNetwork,
    TwinQCritic,
)
from ddpo.offline_rl.policy_adapter import DiffSBDDPolicyAdapter
from ddpo.offline_rl.protocol import DenoisingTransitionBatch
from ddpo.offline_rl.trainer import (
    ActorUpdateConfig,
    IQLTrainerConfig,
    OfflineActorUpdater,
    OfflineIQLTrainer,
)


class _TinyPolicy(nn.Module):
    """Small differentiable policy implementing both offline adapter modes."""

    def __init__(self) -> None:
        """Initialize one trainable scalar offset."""
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(0.15))

    def forward(self, mode: str, **kwargs: torch.Tensor):
        """Return masked Gaussian pseudo log-probability and finite stats."""
        if mode == "offline_reverse_log_prob":
            state = kwargs["zt"]
        elif mode == "offline_final_decode_log_prob":
            state = kwargs["z0"]
        else:
            raise ValueError(mode)
        action = kwargs["prev_sample"]
        action_mask = kwargs["action_mask"]
        mean = state + self.offset
        squared_error = ((action - mean) * action_mask).square()
        denominator = action_mask.sum(dim=(1, 2)).clamp_min(1.0)
        log_prob = -squared_error.sum(dim=(1, 2)) / denominator
        sigma = state.new_ones((int(state.shape[0]), 1, 1))
        return log_prob, mean, sigma


def _make_transitions(*, scored: bool = True) -> DenoisingTransitionBatch:
    """Build three ordinary transitions and one final decode transition."""
    torch.manual_seed(3)
    batch_size, max_nodes, latent_dim = 4, 3, 5
    state = torch.randn(batch_size, max_nodes, latent_dim)
    action = state + 0.3 * torch.randn_like(state)
    nodes = torch.tensor([2, 3, 2, 3], dtype=torch.long)
    node_mask = (
        torch.arange(max_nodes).unsqueeze(0) < nodes.unsqueeze(1)
    ).float().unsqueeze(-1)
    state = state * node_mask
    action = action * node_mask
    final = torch.tensor([False, False, False, True])
    action_mask = node_mask.expand_as(action).clone()
    action_mask[final, :, 3:] = 0
    return DenoisingTransitionBatch(
        state_latent=state,
        action_latent=action,
        next_state_latent=action,
        action_mask=action_mask,
        recorded_behavior_score=torch.zeros(batch_size),
        recorded_timestep=torch.tensor([2, 1, 0, 0]),
        source_diffusion_timestep=torch.tensor([3, 2, 1, 0]),
        target_diffusion_timestep=torch.tensor([2, 1, 0, -1]),
        decision_index=torch.tensor([0, 1, 2, 3]),
        episode_index=torch.arange(batch_size),
        nodesxsample=nodes,
        node_mask=node_mask,
        done=final,
        is_final_decode=final,
        sample_ids=tuple(f"sample-{index}" for index in range(batch_size)),
        reward=torch.tensor([0.0, 0.0, 0.0, 2.0]) if scored else None,
        episode_return=(
            torch.tensor([-1.0, 0.0, 1.0, 2.0]) if scored else None
        ),
    )


def _make_pocket() -> PocketContext:
    """Build a fixed finite pocket context."""
    torch.manual_seed(4)
    return PocketContext(
        coordinates=torch.randn(5, 3),
        features=torch.randn(5, 4),
        node_mask=torch.ones(5, 1),
    )


def _make_actor_updater(
    objective: str = "iql",
) -> tuple[_TinyPolicy, OfflineActorUpdater]:
    """Build a policy adapter, optimizer, and actor updater."""
    policy = _TinyPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
    config = ActorUpdateConfig(
        objective=objective,  # type: ignore[arg-type]
        temperature=1.0,
        max_weight=10.0,
        max_grad_norm=5.0,
        awr_reward_mean=0.5 if objective == "awr" else None,
        awr_reward_std=1.0 if objective == "awr" else None,
    )
    updater = OfflineActorUpdater(
        policy_adapter=DiffSBDDPolicyAdapter(
            policy,
            num_diffusion_steps=3,
        ),
        optimizer=optimizer,
        config=config,
    )
    return policy, updater


def _make_iql_trainer(
    *,
    actor_warmup_steps: int = 0,
) -> tuple[OfflineIQLTrainer, _TinyPolicy]:
    """Build a small geometric IQL trainer."""
    torch.manual_seed(5)
    critic_config = GeometricCriticConfig(
        ligand_feature_dim=2,
        pocket_feature_dim=4,
        hidden_dim=12,
        num_message_layers=1,
        num_rbf=6,
        time_embedding_dim=8,
        max_neighbors=8,
    )
    online_q = TwinQCritic(critic_config)
    value = GeometricValueNetwork(critic_config)
    policy, actor_updater = _make_actor_updater("iql")
    trainer = OfflineIQLTrainer(
        online_q=online_q,
        value=value,
        q_optimizer=torch.optim.Adam(online_q.parameters(), lr=3e-3),
        value_optimizer=torch.optim.Adam(value.parameters(), lr=3e-3),
        actor_updater=actor_updater,
        pocket=_make_pocket(),
        config=IQLTrainerConfig(
            num_diffusion_steps=3,
            discount=1.0,
            expectile=0.7,
            target_tau=0.1,
            q_loss_type="mse",
            critic_max_grad_norm=5.0,
            actor_warmup_steps=actor_warmup_steps,
        ),
    )
    return trainer, policy


def _parameter_snapshot(module: nn.Module) -> tuple[torch.Tensor, ...]:
    """Clone all module parameters."""
    return tuple(parameter.detach().clone() for parameter in module.parameters())


def _any_parameter_changed(
    before: tuple[torch.Tensor, ...],
    module: nn.Module,
) -> bool:
    """Return whether any module parameter differs from a snapshot."""
    return any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, module.parameters())
    )


def _assert_metrics_finite(payload: object) -> None:
    """Recursively require all emitted float metrics to be finite."""
    if isinstance(payload, dict):
        for value in payload.values():
            _assert_metrics_finite(value)
    elif isinstance(payload, float):
        assert math.isfinite(payload)


def _assert_nested_close(actual: object, expected: object) -> None:
    """Recursively compare nested metric dictionaries and scalar values."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_close(actual[key], expected[key])
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-6, abs=1e-7)
    else:
        assert actual == expected


def test_bc_and_awr_actor_updates_use_distinct_weight_semantics() -> None:
    """BC should be uniform while AWR should use repeated episode return."""
    transitions = _make_transitions()
    bc_policy, bc = _make_actor_updater("bc")
    awr_policy, awr = _make_actor_updater("awr")
    bc_before = bc_policy.offset.detach().clone()
    awr_before = awr_policy.offset.detach().clone()

    bc_metrics = bc.step(transitions)
    awr_metrics = awr.step(transitions)

    assert bc_metrics.weight_mean == pytest.approx(1.0)
    assert bc_metrics.effective_sample_size == pytest.approx(4.0)
    assert awr_metrics.weight_max > awr_metrics.weight_mean
    assert awr_metrics.effective_sample_size < 4.0
    assert not torch.equal(bc_policy.offset.detach(), bc_before)
    assert not torch.equal(awr_policy.offset.detach(), awr_before)


def test_awr_requires_episode_return_and_iql_requires_advantage() -> None:
    """Actor objectives must not silently substitute the sparse MDP reward."""
    _, awr = _make_actor_updater("awr")
    _, iql = _make_actor_updater("iql")

    with pytest.raises(ValueError, match="episode_return"):
        awr.step(_make_transitions(scored=False))
    with pytest.raises(ValueError, match="requires detached advantage"):
        iql.step(_make_transitions())


def test_iql_step_updates_v_q_target_and_actor_with_finite_metrics() -> None:
    """One step should respect all parameter boundaries and emit diagnostics."""
    trainer, policy = _make_iql_trainer()
    q_before = _parameter_snapshot(trainer.online_q)
    target_before = _parameter_snapshot(trainer.target_q)
    value_before = _parameter_snapshot(trainer.value)
    actor_before = _parameter_snapshot(policy)

    metrics = trainer.step(_make_transitions())

    assert trainer.global_step == 1
    assert metrics.global_step == 0
    assert metrics.actor_updated is True
    assert metrics.actor is not None
    assert _any_parameter_changed(q_before, trainer.online_q)
    assert _any_parameter_changed(target_before, trainer.target_q)
    assert _any_parameter_changed(value_before, trainer.value)
    assert _any_parameter_changed(actor_before, policy)
    assert all(
        not parameter.requires_grad for parameter in trainer.target_q.parameters()
    )
    _assert_metrics_finite(asdict(metrics))


def test_iql_warmup_updates_critics_without_actor() -> None:
    """Critic warm-up should leave policy parameters unchanged."""
    trainer, policy = _make_iql_trainer(actor_warmup_steps=2)
    actor_before = _parameter_snapshot(policy)

    first = trainer.step(_make_transitions())
    second = trainer.step(_make_transitions())
    third = trainer.step(_make_transitions())

    assert first.actor_updated is False
    assert second.actor_updated is False
    assert third.actor_updated is True
    assert _any_parameter_changed(actor_before, policy)


def test_iql_rejects_unscored_batches() -> None:
    """Critic updates must never treat missing sidecar reward as zero."""
    trainer, _ = _make_iql_trainer()

    with pytest.raises(ValueError, match="requires sparse reward"):
        trainer.step(_make_transitions(scored=False))


def test_iql_checkpoint_round_trip_reproduces_next_update(
    tmp_path: Path,
) -> None:
    """Models, optimizers, target, actor, and counters should resume exactly."""
    trainer, _ = _make_iql_trainer()
    transitions = _make_transitions()
    trainer.step(transitions)
    checkpoint_path = trainer.save_checkpoint(
        tmp_path / "epoch_0000.pt",
        epoch=0,
        sampler_state={"format": "fixture_sampler"},
        extra={"note": "test"},
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_metrics = trainer.step(transitions)

    restored, _ = _make_iql_trainer()
    restored.load_state_dict(payload["trainer"])
    actual_metrics = restored.step(transitions)

    assert payload["epoch"] == 0
    assert payload["sampler"] == {"format": "fixture_sampler"}
    assert actual_metrics.global_step == expected_metrics.global_step
    _assert_nested_close(
        asdict(actual_metrics),
        asdict(expected_metrics),
    )
    for expected, actual in zip(
        trainer.online_q.parameters(),
        restored.online_q.parameters(),
    ):
        torch.testing.assert_close(actual, expected)


def test_optimizer_parameter_coverage_mismatch_fails_fast() -> None:
    """An optimizer wired to the wrong module must not produce a fake update."""
    policy = _TinyPolicy()
    wrong_parameter = nn.Parameter(torch.tensor(0.0))
    wrong_optimizer = torch.optim.Adam([wrong_parameter], lr=1e-3)

    with pytest.raises(ValueError, match="coverage mismatch"):
        OfflineActorUpdater(
            policy_adapter=DiffSBDDPolicyAdapter(
                policy,
                num_diffusion_steps=3,
            ),
            optimizer=wrong_optimizer,
            config=ActorUpdateConfig(objective="bc"),
        )


def test_config_rejects_awr_without_frozen_train_statistics() -> None:
    """AWR must not normalize returns from each sampled minibatch."""
    with pytest.raises(ValueError, match="fixed train-split"):
        ActorUpdateConfig(objective="awr")
    with pytest.raises(ValueError, match="positive"):
        replace(
            ActorUpdateConfig(
                objective="awr",
                awr_reward_mean=0.0,
                awr_reward_std=1.0,
            ),
            awr_reward_std=0.0,
        )
