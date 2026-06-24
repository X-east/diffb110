"""Tests for DiffGui dual-R slot and halfedge constraints."""

from __future__ import annotations

import torch

from ddpo.diffgui.constraints import (
    DiffGuiDualRSlotLayout,
    apply_dual_r_constraints_to_log_probs,
    apply_dual_r_constraints_to_types,
    batch_diffgui_dual_r_constraint_specs,
    build_diffgui_dual_r_constraint_spec,
    force_onehot_from_types,
)
from ddpo.diffgui.runtime.rollout import make_diffgui_ligand_placeholder


def _make_spec(r_slots_per_side: int = 2):
    layout = DiffGuiDualRSlotLayout.from_core(
        core_size=4,
        left_anchor_idx=0,
        right_anchor_idx=3,
        r_slots_per_side=r_slots_per_side,
    )
    _ligand_batch, halfedge_index, _halfedge_batch = make_diffgui_ligand_placeholder([layout.total_nodes])
    spec = build_diffgui_dual_r_constraint_spec(
        layout=layout,
        halfedge_index=halfedge_index,
        num_node_types=6,
        num_edge_types=5,
        core_bond_types={(0, 1): 4, (1, 2): 4, (2, 3): 4},
        attachment_bond_type=1,
        allowed_node_types=(2, 3, 4),
    )
    return layout, halfedge_index, spec


def test_slot_layout_uses_core_left_right_order() -> None:
    """The slot layout should keep core, left R, and right R indices stable."""
    layout = DiffGuiDualRSlotLayout.from_core(
        core_size=5,
        left_anchor_idx=1,
        right_anchor_idx=4,
        r_slots_per_side=3,
    )

    assert layout.total_nodes == 11
    assert layout.left_r_indices == (5, 6, 7)
    assert layout.right_r_indices == (8, 9, 10)
    assert layout.paired_node_indices == ((5, 8), (6, 9), (7, 10))
    assert layout.left_root_idx == 5
    assert layout.right_root_idx == 8


def test_halfedge_spec_forces_attachment_and_forbids_core_leakage() -> None:
    """The halfedge spec should identify forced and forbidden edge ids."""
    layout, halfedge_index, spec = _make_spec(r_slots_per_side=2)
    edge_lookup = {
        tuple(sorted((int(halfedge_index[0, i]), int(halfedge_index[1, i])))): i
        for i in range(halfedge_index.shape[1])
    }

    required = dict(spec.halfedge_map.required_edges)
    assert required[edge_lookup[(0, 1)]] == 4
    assert required[edge_lookup[(0, layout.left_root_idx)]] == 1
    assert required[edge_lookup[(3, layout.right_root_idx)]] == 1
    assert edge_lookup[(0, 2)] in spec.halfedge_map.forbidden_edges
    assert edge_lookup[(1, layout.left_root_idx)] in spec.halfedge_map.forbidden_edges
    assert edge_lookup[(2, layout.right_root_idx)] in spec.halfedge_map.forbidden_edges
    assert edge_lookup[(layout.left_root_idx, layout.right_root_idx)] in spec.halfedge_map.forbidden_edges


def test_log_prob_constraints_sync_nodes_edges_and_force_masks() -> None:
    """Paired consensus and hard masks should be visible in constrained log probabilities."""
    layout, halfedge_index, spec = _make_spec(r_slots_per_side=2)
    edge_lookup = {
        tuple(sorted((int(halfedge_index[0, i]), int(halfedge_index[1, i])))): i
        for i in range(halfedge_index.shape[1])
    }
    torch.manual_seed(11)
    node_log_probs = torch.log_softmax(torch.randn(layout.total_nodes, 6), dim=-1)
    edge_log_probs = torch.log_softmax(torch.randn(halfedge_index.shape[1], 5), dim=-1)

    constrained_nodes, constrained_edges = apply_dual_r_constraints_to_log_probs(
        node_log_probs,
        edge_log_probs,
        spec,
    )

    for left_idx, right_idx in layout.paired_node_indices:
        assert torch.allclose(constrained_nodes[left_idx], constrained_nodes[right_idx])
        assert constrained_nodes[left_idx].argmax().item() in {2, 3, 4}

    left_internal = edge_lookup[tuple(sorted(layout.left_r_indices))]
    right_internal = edge_lookup[tuple(sorted(layout.right_r_indices))]
    assert torch.allclose(constrained_edges[left_internal], constrained_edges[right_internal])

    forbidden_edge = edge_lookup[(1, layout.left_root_idx)]
    assert constrained_edges[forbidden_edge].argmax().item() == 0

    required = dict(spec.halfedge_map.required_edges)
    for edge_id, edge_type in required.items():
        assert constrained_edges[edge_id].argmax().item() == edge_type


def test_type_constraints_are_deterministic_after_sampling() -> None:
    """Already sampled categorical ids should be overwritten deterministically."""
    layout, halfedge_index, spec = _make_spec(r_slots_per_side=2)
    edge_lookup = {
        tuple(sorted((int(halfedge_index[0, i]), int(halfedge_index[1, i])))): i
        for i in range(halfedge_index.shape[1])
    }
    node_types = torch.arange(layout.total_nodes) % 6
    edge_types = torch.arange(halfedge_index.shape[1]) % 5

    constrained_nodes, constrained_edges = apply_dual_r_constraints_to_types(node_types, edge_types, spec)

    for left_idx, right_idx in layout.paired_node_indices:
        assert constrained_nodes[right_idx].item() == constrained_nodes[left_idx].item()
    for left_edge, right_edge in spec.halfedge_map.paired_edges:
        assert constrained_edges[right_edge].item() == constrained_edges[left_edge].item()
    assert constrained_edges[edge_lookup[(1, layout.left_root_idx)]].item() == 0
    assert constrained_edges[edge_lookup[(0, layout.left_root_idx)]].item() == 1


def test_force_onehot_from_types() -> None:
    """Forced type indices should convert back to one-hot categorical states."""
    onehot = force_onehot_from_types(torch.tensor([0, 2, 1]), num_classes=4)

    assert onehot.shape == (3, 4)
    assert onehot.dtype == torch.float32
    assert onehot.tolist() == [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]


def test_batch_constraint_specs_shift_graph_indices() -> None:
    """Batched specs should point at each graph's shifted node and halfedge ids."""
    layout = DiffGuiDualRSlotLayout.from_core(
        core_size=4,
        left_anchor_idx=0,
        right_anchor_idx=3,
        r_slots_per_side=1,
    )
    _ligand_batch, halfedge_index, _halfedge_batch = make_diffgui_ligand_placeholder(
        [layout.total_nodes, layout.total_nodes]
    )

    specs = batch_diffgui_dual_r_constraint_specs(
        layout=layout,
        halfedge_index=halfedge_index,
        num_graphs=2,
        num_node_types=6,
        num_edge_types=5,
        core_bond_types={(0, 1): 4},
        attachment_bond_type=1,
    )

    assert len(specs) == 2
    assert specs[0].layout.left_anchor_idx == 0
    assert specs[1].layout.left_anchor_idx == layout.total_nodes
    assert specs[1].layout.left_root_idx == layout.total_nodes + layout.left_root_idx
    assert specs[0].halfedge_map.required_edges != specs[1].halfedge_map.required_edges


def test_force_r_chain_edges_marks_consecutive_slots_required() -> None:
    """Optional R-chain constraints should force same-side consecutive R slots."""
    layout = DiffGuiDualRSlotLayout.from_core(
        core_size=4,
        left_anchor_idx=0,
        right_anchor_idx=3,
        r_slots_per_side=3,
    )
    _ligand_batch, halfedge_index, _halfedge_batch = make_diffgui_ligand_placeholder([layout.total_nodes])
    spec = build_diffgui_dual_r_constraint_spec(
        layout=layout,
        halfedge_index=halfedge_index,
        num_node_types=6,
        num_edge_types=5,
        core_bond_types={(0, 1): 4},
        attachment_bond_type=1,
        require_r_chain_edges=True,
        r_chain_bond_type=1,
    )
    edge_lookup = {
        tuple(sorted((int(halfedge_index[0, i]), int(halfedge_index[1, i])))): i
        for i in range(halfedge_index.shape[1])
    }
    required = dict(spec.halfedge_map.required_edges)

    for pair in ((4, 5), (5, 6), (7, 8), (8, 9)):
        assert required[edge_lookup[pair]] == 1

    node_types = torch.arange(layout.total_nodes) % 6
    edge_types = torch.zeros(halfedge_index.shape[1], dtype=torch.long)
    _constrained_nodes, constrained_edges = apply_dual_r_constraints_to_types(node_types, edge_types, spec)
    for pair in ((4, 5), (5, 6), (7, 8), (8, 9)):
        assert constrained_edges[edge_lookup[pair]].item() == 1
