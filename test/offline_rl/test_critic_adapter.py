"""Tests for the DiffSBDD transition-to-critic input boundary."""

from __future__ import annotations

from dataclasses import replace

import torch

from ddpo.offline_rl.critic_adapter import (
    PocketContext,
    build_geometric_critic_input,
)
from ddpo.offline_rl.protocol import DenoisingTransitionBatch


def _make_transitions() -> DenoisingTransitionBatch:
    """Build one ordinary and one final transition with padding."""
    batch_size, max_nodes, latent_dim = 2, 3, 6
    state = torch.randn(batch_size, max_nodes, latent_dim)
    action = torch.randn_like(state)
    nodes = torch.tensor([2, 3], dtype=torch.long)
    node_mask = (
        torch.arange(max_nodes).unsqueeze(0) < nodes.unsqueeze(1)
    ).float().unsqueeze(-1)
    state = state * node_mask
    action = action * node_mask
    final = torch.tensor([False, True])
    action_mask = node_mask.expand_as(action).clone()
    action_mask[final, :, 3:] = 0
    return DenoisingTransitionBatch(
        state_latent=state,
        action_latent=action,
        next_state_latent=action,
        action_mask=action_mask,
        recorded_behavior_score=torch.zeros(batch_size),
        recorded_timestep=torch.tensor([2, 0]),
        source_diffusion_timestep=torch.tensor([3, 0]),
        target_diffusion_timestep=torch.tensor([2, -1]),
        decision_index=torch.tensor([0, 3]),
        episode_index=torch.arange(batch_size),
        nodesxsample=nodes,
        node_mask=node_mask,
        done=final,
        is_final_decode=final,
        sample_ids=("ordinary", "final"),
    )


def _make_pocket() -> PocketContext:
    """Build a small static pocket."""
    return PocketContext(
        coordinates=torch.randn(4, 3),
        features=torch.randn(4, 5),
        node_mask=torch.ones(4, 1),
    )


def test_critic_adapter_splits_latents_and_normalizes_time() -> None:
    """The boundary should expose centered pocket xyz and normalized time."""
    transitions = _make_transitions()

    inputs = build_geometric_critic_input(
        transitions,
        _make_pocket(),
        num_diffusion_steps=4,
    )

    assert tuple(inputs.state_coordinates.shape) == (2, 3, 3)
    assert tuple(inputs.state_features.shape) == (2, 3, 3)
    assert tuple(inputs.pocket_coordinates.shape) == (2, 4, 3)
    assert inputs.normalized_timestep.tolist() == [0.75, 0.0]
    assert inputs.is_final_decode.tolist() == [False, True]
    torch.testing.assert_close(
        inputs.pocket_coordinates.mean(dim=1),
        torch.zeros(2, 3),
        atol=1e-6,
        rtol=0,
    )


def test_final_action_features_never_reach_q_input() -> None:
    """The final decoder's sampled h channels are not effective MDP actions."""
    transitions = _make_transitions()
    changed_action = transitions.action_latent.clone()
    changed_action[1, :, 3:] += 1.0e5

    baseline = build_geometric_critic_input(
        transitions,
        _make_pocket(),
        num_diffusion_steps=4,
    )
    changed = build_geometric_critic_input(
        replace(
            transitions,
            action_latent=changed_action,
            next_state_latent=changed_action,
        ),
        _make_pocket(),
        num_diffusion_steps=4,
    )

    assert torch.count_nonzero(baseline.action_features[1]) == 0
    assert torch.count_nonzero(changed.action_features[1]) == 0
    torch.testing.assert_close(changed.action_features, baseline.action_features)


def test_pocket_context_understands_diffsbdd_single_pocket_mapping() -> None:
    """DiffSBDD graph assignment masks must not be mistaken for padding."""
    coordinates = torch.randn(3, 3)
    features = torch.nn.functional.one_hot(
        torch.tensor([0, 1, 2]),
        num_classes=4,
    )

    context = PocketContext.from_diffsbdd_mapping(
        {
            "x": coordinates,
            "one_hot": features,
            "mask": torch.zeros(3, dtype=torch.long),
        }
    )

    torch.testing.assert_close(context.coordinates, coordinates)
    torch.testing.assert_close(context.features, features.float())
    assert context.node_mask.tolist() == [[1.0], [1.0], [1.0]]
