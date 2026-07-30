"""Boundary tests for role-aware sparse critic graph construction."""

from __future__ import annotations

import torch

from ddpo.offline_rl.models.graph_builder import (
    IDENTITY_EDGE_TYPE,
    STATE_ROLE,
    RadiusGraphBuilder,
)


def test_radius_builder_caps_non_identity_neighbors_per_target() -> None:
    """The configured cap should bound ordinary incoming radius edges."""
    builder = RadiusGraphBuilder(
        pocket_cutoff=10.0,
        ligand_cutoff=10.0,
        interaction_cutoff=10.0,
        max_neighbors=1,
    )
    coordinates = torch.tensor(
        [[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]]
    )
    node_mask = torch.ones(1, 4, 1)
    node_roles = torch.full((4,), STATE_ROLE, dtype=torch.long)

    graph = builder.build(coordinates, node_mask, node_roles)

    targets = graph.edge_index[1]
    counts = torch.bincount(targets, minlength=4)
    assert counts.tolist() == [1, 1, 1, 1]


def test_identity_edges_survive_radius_cutoff_and_padding() -> None:
    """Valid state-action pairs remain connected even when spatially distant."""
    builder = RadiusGraphBuilder(
        pocket_cutoff=0.1,
        ligand_cutoff=0.1,
        interaction_cutoff=0.1,
        max_neighbors=1,
    )
    coordinates = torch.tensor(
        [[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [200.0, 0.0, 0.0],
        ]]
    )
    node_mask = torch.tensor([[[1.0], [0.0], [1.0], [0.0]]])
    node_roles = torch.full((4,), STATE_ROLE, dtype=torch.long)
    identity_pairs = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)

    graph = builder.build(
        coordinates,
        node_mask,
        node_roles,
        identity_pairs=identity_pairs,
    )

    assert graph.edge_index.tolist() == [[0, 2], [2, 0]]
    assert graph.edge_type.tolist() == [
        IDENTITY_EDGE_TYPE,
        IDENTITY_EDGE_TYPE,
    ]
    assert graph.edge_distance.tolist() == [100.0, 100.0]
