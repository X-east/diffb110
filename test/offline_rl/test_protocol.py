"""Tests for typed offline denoising transition boundaries."""

from __future__ import annotations

import pytest
import torch

from ddpo.offline_rl.protocol import DenoisingTransitionBatch


def _make_batch() -> DenoisingTransitionBatch:
    """Build a minimal valid transition batch."""
    batch_size = 2
    nodes = 3
    features = 4
    state = torch.zeros(batch_size, nodes, features)
    action = torch.ones_like(state)
    return DenoisingTransitionBatch(
        state_latent=state,
        action_latent=action,
        next_state_latent=action,
        behavior_log_prob=torch.zeros(batch_size),
        diffusion_timestep=torch.tensor([2, 0], dtype=torch.long),
        decision_index=torch.tensor([0, 3], dtype=torch.long),
        episode_index=torch.tensor([0, 1], dtype=torch.long),
        nodesxsample=torch.tensor([2, 3], dtype=torch.long),
        node_mask=torch.tensor(
            [
                [[1.0], [1.0], [0.0]],
                [[1.0], [1.0], [1.0]],
            ]
        ),
        done=torch.tensor([False, True]),
        is_final_decode=torch.tensor([False, True]),
        sample_ids=("sample-0", "sample-1"),
        reward=torch.tensor([0.0, 5.0]),
    )


def test_transition_batch_reports_size_device_and_moves_tensors() -> None:
    """The protocol should preserve metadata while moving all tensors."""
    batch = _make_batch()

    moved = batch.to("cpu")

    assert batch.batch_size == 2
    assert batch.device.type == "cpu"
    assert moved.batch_size == 2
    assert moved.sample_ids == batch.sample_ids
    assert moved.reward is not None
    assert moved.reward.tolist() == pytest.approx([0.0, 5.0])


def test_transition_batch_rejects_terminal_semantic_mismatch() -> None:
    """Final decoder facts and terminal facts must agree in the v1 MDP."""
    batch = _make_batch()

    with pytest.raises(ValueError, match="done == is_final_decode"):
        DenoisingTransitionBatch(
            **{
                **batch.__dict__,
                "is_final_decode": torch.tensor([False, False]),
            }
        )


def test_transition_batch_rejects_shape_drift() -> None:
    """Latent and per-transition tensors must not broadcast silently."""
    batch = _make_batch()

    with pytest.raises(ValueError, match="action_latent"):
        DenoisingTransitionBatch(
            **{
                **batch.__dict__,
                "action_latent": torch.ones(2, 2, 4),
            }
        )
    with pytest.raises(ValueError, match="sample_ids length"):
        DenoisingTransitionBatch(
            **{
                **batch.__dict__,
                "sample_ids": ("only-one",),
            }
        )


def test_transition_batch_requires_finite_reward() -> None:
    """Scored transition batches must never carry NaN/Inf rewards."""
    batch = _make_batch()

    with pytest.raises(ValueError, match="finite"):
        DenoisingTransitionBatch(
            **{
                **batch.__dict__,
                "reward": torch.tensor([0.0, float("nan")]),
            }
        )
