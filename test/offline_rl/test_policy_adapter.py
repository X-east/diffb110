"""Tests for DiffSBDD offline policy-probability routing."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn

from ddpo.offline_rl.policy_adapter import DiffSBDDPolicyAdapter
from ddpo.offline_rl.protocol import DenoisingTransitionBatch


class _FakeDiffSBDDPolicy(nn.Module):
    """Small differentiable policy exposing the two offline forward modes."""

    def __init__(self) -> None:
        super().__init__()
        self.location = nn.Parameter(torch.tensor(0.25))
        self.calls: list[dict[str, object]] = []

    def forward(self, mode: str, **kwargs):
        """Return masked Gaussian-like scores and record routing facts."""
        action = kwargs["prev_sample"]
        action_mask = kwargs["action_mask"]
        if mode == "offline_reverse_log_prob":
            self.calls.append(
                {
                    "mode": mode,
                    "s": kwargs["s"].detach().cpu().clone(),
                    "t": kwargs["t"].detach().cpu().clone(),
                }
            )
            state = kwargs["zt"]
            sigma_value = 1.0
        elif mode == "offline_final_decode_log_prob":
            self.calls.append({"mode": mode})
            state = kwargs["z0"]
            sigma_value = 2.0
        else:
            raise ValueError(f"unexpected mode {mode!r}")

        mean = torch.zeros_like(state) + self.location
        valid_atoms = (action_mask.sum(dim=2) > 0).sum(dim=1)
        score = (
            -0.5
            * ((action - mean).square() * action_mask).sum(dim=(1, 2))
            / valid_atoms
        )
        sigma = action.new_full((int(action.shape[0]), 1), sigma_value)
        return score, mean, sigma


def _make_transitions() -> DenoisingTransitionBatch:
    """Build a mixed-timestep batch with one final decode transition."""
    batch_size, max_nodes, features = 4, 2, 5
    state = torch.zeros(batch_size, max_nodes, features)
    action = torch.arange(
        batch_size * max_nodes * features,
        dtype=torch.float32,
    ).reshape(batch_size, max_nodes, features) / 10.0
    nodes = torch.tensor([2, 1, 2, 2], dtype=torch.long)
    node_mask = (
        torch.arange(max_nodes).unsqueeze(0) < nodes.unsqueeze(1)
    ).float().unsqueeze(-1)
    final = torch.tensor([False, False, True, False])
    action_mask = node_mask.expand_as(action).clone()
    action_mask[final, :, 3:] = 0
    return DenoisingTransitionBatch(
        state_latent=state,
        action_latent=action,
        next_state_latent=action,
        action_mask=action_mask,
        recorded_behavior_score=torch.zeros(batch_size),
        recorded_timestep=torch.tensor([2, 0, 0, 2]),
        source_diffusion_timestep=torch.tensor([3, 1, 0, 3]),
        target_diffusion_timestep=torch.tensor([2, 0, -1, 2]),
        decision_index=torch.tensor([0, 2, 3, 0]),
        episode_index=torch.arange(batch_size),
        nodesxsample=nodes,
        node_mask=node_mask,
        done=final,
        is_final_decode=final,
        sample_ids=tuple(f"sample-{index}" for index in range(batch_size)),
    )


def _expected_score(
    policy: _FakeDiffSBDDPolicy,
    transitions: DenoisingTransitionBatch,
) -> torch.Tensor:
    """Compute the fake policy objective without adapter routing."""
    mean = torch.zeros_like(transitions.action_latent) + policy.location
    valid_atoms = (transitions.action_mask.sum(dim=2) > 0).sum(dim=1)
    return (
        -0.5
        * (
            (transitions.action_latent - mean).square()
            * transitions.action_mask
        ).sum(dim=(1, 2))
        / valid_atoms
    )


def test_adapter_routes_mixed_timesteps_and_restores_input_order() -> None:
    """Ordinary time groups and final decode must use separate model branches."""
    policy = _FakeDiffSBDDPolicy()
    transitions = _make_transitions()
    adapter = DiffSBDDPolicyAdapter(policy, num_diffusion_steps=4)

    output = adapter.log_prob(transitions)

    torch.testing.assert_close(output.log_prob, _expected_score(policy, transitions))
    assert output.ordinary_mask.tolist() == [True, True, False, True]
    assert output.final_decode_mask.tolist() == [False, False, True, False]
    assert output.sigma.tolist() == pytest.approx([1.0, 1.0, 2.0, 1.0])
    assert [call["mode"] for call in policy.calls] == [
        "offline_reverse_log_prob",
        "offline_reverse_log_prob",
        "offline_final_decode_log_prob",
    ]
    assert policy.calls[0]["s"].flatten().tolist() == pytest.approx([0.0])
    assert policy.calls[0]["t"].flatten().tolist() == pytest.approx([0.25])
    assert policy.calls[1]["s"].flatten().tolist() == pytest.approx([0.5, 0.5])
    assert policy.calls[1]["t"].flatten().tolist() == pytest.approx([0.75, 0.75])


def test_adapter_final_score_ignores_recorded_h_noise() -> None:
    """Final h channels are provenance, not effective decoder actions."""
    policy = _FakeDiffSBDDPolicy()
    transitions = _make_transitions()
    adapter = DiffSBDDPolicyAdapter(policy, num_diffusion_steps=4)
    baseline = adapter.log_prob(transitions).log_prob
    changed_action = transitions.action_latent.clone()
    changed_action[2, :, 3:] += 1.0e4

    changed = adapter.log_prob(
        replace(
            transitions,
            action_latent=changed_action,
            next_state_latent=changed_action,
        )
    ).log_prob

    torch.testing.assert_close(changed, baseline)


def test_adapter_log_prob_backpropagates_to_policy() -> None:
    """Reordering and grouped forwards must preserve actor gradients."""
    policy = _FakeDiffSBDDPolicy()
    adapter = DiffSBDDPolicyAdapter(policy, num_diffusion_steps=4)

    loss = -adapter.log_prob(_make_transitions()).log_prob.mean()
    loss.backward()

    assert policy.location.grad is not None
    assert torch.isfinite(policy.location.grad)
    assert policy.location.grad.abs().item() > 0


def test_adapter_rejects_invalid_diffusion_horizon() -> None:
    """Normalized diffusion times require a positive horizon."""
    with pytest.raises(ValueError, match="positive"):
        DiffSBDDPolicyAdapter(_FakeDiffSBDDPolicy(), num_diffusion_steps=0)
