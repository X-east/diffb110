"""Invariance, masking, shape, and gradient tests for geometric IQL critics."""

from __future__ import annotations

from dataclasses import replace

import torch

from ddpo.offline_rl.critic_adapter import GeometricCriticInput
from ddpo.offline_rl.models import (
    GeometricCriticConfig,
    GeometricValueNetwork,
    TwinQCritic,
)


def _make_inputs(
    *,
    batch_size: int = 2,
    ligand_nodes: int = 5,
    pocket_nodes: int = 4,
    valid_ligand_nodes: int = 4,
) -> GeometricCriticInput:
    """Build deterministic synthetic geometric critic inputs."""
    generator = torch.Generator().manual_seed(37)
    ligand_mask = torch.zeros(batch_size, ligand_nodes, 1)
    ligand_mask[:, :valid_ligand_nodes] = 1
    pocket_mask = torch.ones(batch_size, pocket_nodes, 1)
    state_coordinates = torch.randn(
        batch_size,
        ligand_nodes,
        3,
        generator=generator,
    )
    action_coordinates = torch.randn(
        batch_size,
        ligand_nodes,
        3,
        generator=generator,
    )
    state_features = torch.randn(
        batch_size,
        ligand_nodes,
        3,
        generator=generator,
    )
    action_features = torch.randn(
        batch_size,
        ligand_nodes,
        3,
        generator=generator,
    )
    pocket_coordinates = torch.randn(
        batch_size,
        pocket_nodes,
        3,
        generator=generator,
    )
    pocket_features = torch.randn(
        batch_size,
        pocket_nodes,
        5,
        generator=generator,
    )
    return GeometricCriticInput(
        state_coordinates=state_coordinates * ligand_mask,
        state_features=state_features * ligand_mask,
        action_coordinates=action_coordinates * ligand_mask,
        action_features=action_features * ligand_mask,
        pocket_coordinates=pocket_coordinates,
        pocket_features=pocket_features,
        ligand_mask=ligand_mask,
        pocket_mask=pocket_mask,
        normalized_timestep=torch.tensor([0.75, 0.0])[:batch_size],
        is_final_decode=torch.tensor([False, True])[:batch_size],
    )


def _make_models() -> tuple[TwinQCritic, GeometricValueNetwork]:
    """Build small deterministic critics for CPU tests."""
    torch.manual_seed(11)
    config = GeometricCriticConfig(
        ligand_feature_dim=3,
        pocket_feature_dim=5,
        hidden_dim=16,
        num_message_layers=2,
        num_rbf=8,
        time_embedding_dim=8,
        pocket_cutoff=100.0,
        ligand_cutoff=100.0,
        interaction_cutoff=100.0,
        max_neighbors=None,
    )
    twin_q = TwinQCritic(config).eval()
    value = GeometricValueNetwork(config).eval()
    return twin_q, value


def _rotation_matrix() -> torch.Tensor:
    """Return a fixed proper rotation matrix."""
    raw = torch.tensor(
        [
            [0.2, -0.9, 0.3],
            [0.7, 0.1, -0.6],
            [0.5, 0.4, 0.8],
        ],
        dtype=torch.float32,
    )
    orthogonal, _ = torch.linalg.qr(raw)
    if torch.linalg.det(orthogonal) < 0:
        orthogonal[:, 0] *= -1
    return orthogonal


def _evaluate(
    twin_q: TwinQCritic,
    value: GeometricValueNetwork,
    inputs: GeometricCriticInput,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate both Q networks and V without building an autograd graph."""
    with torch.no_grad():
        q1, q2 = twin_q(inputs)
        v = value(inputs)
    return q1, q2, v


def test_q_and_v_are_rotation_translation_and_reflection_invariant() -> None:
    """Rigid transforms and reflection must not change scalar critic outputs."""
    inputs = _make_inputs()
    twin_q, value = _make_models()
    baseline = _evaluate(twin_q, value, inputs)
    rotation = _rotation_matrix()
    reflection = torch.diag(torch.tensor([-1.0, 1.0, 1.0]))
    transform = rotation @ reflection
    translation = torch.tensor([3.0, -2.0, 5.0])

    transformed = replace(
        inputs,
        state_coordinates=inputs.state_coordinates @ transform + translation,
        action_coordinates=inputs.action_coordinates @ transform + translation,
        pocket_coordinates=inputs.pocket_coordinates @ transform + translation,
    )
    observed = _evaluate(twin_q, value, transformed)

    for actual, expected in zip(observed, baseline):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_q_and_v_are_ligand_and_pocket_permutation_invariant() -> None:
    """Consistent node permutations must preserve all scalar values."""
    inputs = _make_inputs()
    twin_q, value = _make_models()
    baseline = _evaluate(twin_q, value, inputs)
    ligand_permutation = torch.tensor([2, 0, 3, 1, 4])
    pocket_permutation = torch.tensor([3, 1, 0, 2])

    permuted = replace(
        inputs,
        state_coordinates=inputs.state_coordinates[:, ligand_permutation],
        state_features=inputs.state_features[:, ligand_permutation],
        action_coordinates=inputs.action_coordinates[:, ligand_permutation],
        action_features=inputs.action_features[:, ligand_permutation],
        ligand_mask=inputs.ligand_mask[:, ligand_permutation],
        pocket_coordinates=inputs.pocket_coordinates[:, pocket_permutation],
        pocket_features=inputs.pocket_features[:, pocket_permutation],
        pocket_mask=inputs.pocket_mask[:, pocket_permutation],
    )
    observed = _evaluate(twin_q, value, permuted)

    for actual, expected in zip(observed, baseline):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_random_padding_does_not_change_q_or_v() -> None:
    """Masked ligand padding must not create nodes, edges, or pooled signal."""
    inputs = _make_inputs(ligand_nodes=4, valid_ligand_nodes=4)
    twin_q, value = _make_models()
    baseline = _evaluate(twin_q, value, inputs)
    generator = torch.Generator().manual_seed(91)
    padding_coordinates = torch.randn(2, 2, 3, generator=generator) * 100
    padding_features = torch.randn(2, 2, 3, generator=generator) * 100
    zero_mask = torch.zeros(2, 2, 1)
    padded = replace(
        inputs,
        state_coordinates=torch.cat(
            [inputs.state_coordinates, padding_coordinates],
            dim=1,
        ),
        state_features=torch.cat(
            [inputs.state_features, padding_features],
            dim=1,
        ),
        action_coordinates=torch.cat(
            [inputs.action_coordinates, -padding_coordinates],
            dim=1,
        ),
        action_features=torch.cat(
            [inputs.action_features, -padding_features],
            dim=1,
        ),
        ligand_mask=torch.cat([inputs.ligand_mask, zero_mask], dim=1),
    )
    observed = _evaluate(twin_q, value, padded)

    for actual, expected in zip(observed, baseline):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_value_ignores_action_but_q_responds_to_valid_action_change() -> None:
    """V is state-only while Q consumes the effective action."""
    inputs = _make_inputs()
    twin_q, value = _make_models()
    baseline_q1, baseline_q2, baseline_v = _evaluate(twin_q, value, inputs)
    changed = replace(
        inputs,
        action_coordinates=inputs.action_coordinates + inputs.ligand_mask * 2.0,
        action_features=inputs.action_features + inputs.ligand_mask * 3.0,
    )
    changed_q1, changed_q2, changed_v = _evaluate(twin_q, value, changed)

    torch.testing.assert_close(changed_v, baseline_v)
    assert not torch.allclose(changed_q1, baseline_q1)
    assert not torch.allclose(changed_q2, baseline_q2)


def test_twin_q_and_value_have_finite_independent_gradients() -> None:
    """All three critics should produce trainable finite scalar objectives."""
    inputs = _make_inputs()
    twin_q, value = _make_models()
    twin_q.train()
    value.train()

    q1, q2 = twin_q(inputs)
    v = value(inputs)
    loss = q1.square().mean() + q2.square().mean() + v.square().mean()
    loss.backward()

    q1_parameter = next(twin_q.q1.parameters())
    q2_parameter = next(twin_q.q2.parameters())
    assert q1_parameter.data_ptr() != q2_parameter.data_ptr()
    for model in (twin_q.q1, twin_q.q2, value):
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        assert gradients
        assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
        assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0
