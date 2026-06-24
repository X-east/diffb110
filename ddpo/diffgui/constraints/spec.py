"""Slot and halfedge specifications for DiffGui dual-R constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


DIFFGUI_NO_BOND_TYPE = 0
DIFFGUI_SINGLE_BOND_TYPE = 1


@dataclass(frozen=True)
class DiffGuiDualRSlotLayout:
    """Deterministic ligand slot layout for a fixed core plus paired R slots."""

    core_size: int
    left_anchor_idx: int
    right_anchor_idx: int
    r_slots_per_side: int
    left_r_indices: tuple[int, ...]
    right_r_indices: tuple[int, ...]

    @classmethod
    def from_core(
        cls,
        *,
        core_size: int,
        left_anchor_idx: int,
        right_anchor_idx: int,
        r_slots_per_side: int,
    ) -> "DiffGuiDualRSlotLayout":
        """Build the canonical slot order: core, left R slots, right R slots."""
        core_size = int(core_size)
        r_slots_per_side = int(r_slots_per_side)
        if core_size <= 0:
            raise ValueError("core_size must be positive.")
        if r_slots_per_side <= 0:
            raise ValueError("r_slots_per_side must be positive.")
        if left_anchor_idx < 0 or left_anchor_idx >= core_size:
            raise ValueError("left_anchor_idx must point to a core slot.")
        if right_anchor_idx < 0 or right_anchor_idx >= core_size:
            raise ValueError("right_anchor_idx must point to a core slot.")
        if left_anchor_idx == right_anchor_idx:
            raise ValueError("left_anchor_idx and right_anchor_idx must differ.")
        left_start = core_size
        right_start = core_size + r_slots_per_side
        return cls(
            core_size=core_size,
            left_anchor_idx=int(left_anchor_idx),
            right_anchor_idx=int(right_anchor_idx),
            r_slots_per_side=r_slots_per_side,
            left_r_indices=tuple(range(left_start, left_start + r_slots_per_side)),
            right_r_indices=tuple(range(right_start, right_start + r_slots_per_side)),
        )

    @property
    def total_nodes(self) -> int:
        """Return total ligand slots for one graph."""
        return int(self.core_size + 2 * self.r_slots_per_side)

    @property
    def core_start_idx(self) -> int:
        """Return the first fixed-core slot index, including batch offsets."""
        return int(self.left_r_indices[0] - self.core_size)

    @property
    def max_node_index(self) -> int:
        """Return the largest node index referenced by this layout."""
        return max((*self.core_indices(), *self.r_indices()))

    @property
    def paired_node_indices(self) -> tuple[tuple[int, int], ...]:
        """Return paired left/right R node indices."""
        return tuple(zip(self.left_r_indices, self.right_r_indices))

    @property
    def left_root_idx(self) -> int:
        """Return the first left R slot used as attachment root."""
        return self.left_r_indices[0]

    @property
    def right_root_idx(self) -> int:
        """Return the first right R slot used as attachment root."""
        return self.right_r_indices[0]

    def core_indices(self) -> tuple[int, ...]:
        """Return fixed core slot indices."""
        start = self.core_start_idx
        return tuple(range(start, start + self.core_size))

    def r_indices(self) -> tuple[int, ...]:
        """Return all generated R slot indices."""
        return self.left_r_indices + self.right_r_indices

    def for_graph(self, graph_index: int) -> "DiffGuiDualRSlotLayout":
        """Return this layout shifted to a batched graph offset."""
        offset = int(graph_index) * self.total_nodes
        return DiffGuiDualRSlotLayout(
            core_size=self.core_size,
            left_anchor_idx=self.left_anchor_idx + offset,
            right_anchor_idx=self.right_anchor_idx + offset,
            r_slots_per_side=self.r_slots_per_side,
            left_r_indices=tuple(index + offset for index in self.left_r_indices),
            right_r_indices=tuple(index + offset for index in self.right_r_indices),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable metadata."""
        return {
            "core_size": int(self.core_size),
            "left_anchor_idx": int(self.left_anchor_idx),
            "right_anchor_idx": int(self.right_anchor_idx),
            "r_slots_per_side": int(self.r_slots_per_side),
            "left_r_indices": list(self.left_r_indices),
            "right_r_indices": list(self.right_r_indices),
            "total_nodes": int(self.total_nodes),
            "core_start_idx": int(self.core_start_idx),
            "max_node_index": int(self.max_node_index),
        }


@dataclass(frozen=True)
class DiffGuiDualRHalfedgeMap:
    """Halfedge ids for required, forbidden, and paired dual-R constraints."""

    required_edges: tuple[tuple[int, int], ...]
    required_edge_types: tuple[int, ...]
    forbidden_edges: tuple[int, ...]
    paired_edges: tuple[tuple[int, int], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable metadata."""
        return {
            "required_edges": [list(pair) for pair in self.required_edges],
            "required_edge_types": [int(value) for value in self.required_edge_types],
            "forbidden_edges": [int(value) for value in self.forbidden_edges],
            "paired_edges": [list(pair) for pair in self.paired_edges],
        }


@dataclass(frozen=True)
class DiffGuiDualRConstraintSpec:
    """Complete tensor-level constraint specification for one DiffGui graph layout."""

    layout: DiffGuiDualRSlotLayout
    halfedge_map: DiffGuiDualRHalfedgeMap
    num_node_types: int
    num_edge_types: int
    allowed_node_types: tuple[int, ...] = ()
    no_bond_type: int = DIFFGUI_NO_BOND_TYPE

    def __post_init__(self) -> None:
        """Validate categorical dimensions and forced type ids."""
        if self.num_node_types <= 0:
            raise ValueError("num_node_types must be positive.")
        if self.num_edge_types <= 1:
            raise ValueError("num_edge_types must include no-bond plus at least one bond type.")
        for node_type in self.allowed_node_types:
            if node_type < 0 or node_type >= self.num_node_types:
                raise ValueError("allowed_node_types contains an out-of-range id.")
        for edge_type in self.halfedge_map.required_edge_types:
            if edge_type < 0 or edge_type >= self.num_edge_types:
                raise ValueError("required_edge_types contains an out-of-range id.")
        if self.no_bond_type < 0 or self.no_bond_type >= self.num_edge_types:
            raise ValueError("no_bond_type is out of range.")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable metadata."""
        return {
            "layout": self.layout.to_dict(),
            "halfedge_map": self.halfedge_map.to_dict(),
            "num_node_types": int(self.num_node_types),
            "num_edge_types": int(self.num_edge_types),
            "allowed_node_types": [int(value) for value in self.allowed_node_types],
            "no_bond_type": int(self.no_bond_type),
        }


def build_diffgui_dual_r_constraint_spec(
    *,
    layout: DiffGuiDualRSlotLayout,
    halfedge_index: torch.Tensor,
    num_node_types: int,
    num_edge_types: int,
    core_bond_types: dict[tuple[int, int], int] | None = None,
    attachment_bond_type: int = DIFFGUI_SINGLE_BOND_TYPE,
    allowed_node_types: tuple[int, ...] = (),
    forbid_cross_side_edges: bool = True,
    forbid_core_nonbond_edges: bool = True,
    require_r_chain_edges: bool = False,
    r_chain_bond_type: int = DIFFGUI_SINGLE_BOND_TYPE,
) -> DiffGuiDualRConstraintSpec:
    """Build hard edge and paired-node constraints from a slot layout."""
    if halfedge_index.ndim != 2 or halfedge_index.shape[0] != 2:
        raise ValueError("halfedge_index must have shape [2, E].")
    edge_lookup = _halfedge_lookup(halfedge_index)

    required: list[int] = []
    required_types: list[int] = []
    for edge_pair, edge_type in (core_bond_types or {}).items():
        edge_id = _lookup_edge(edge_lookup, edge_pair)
        required.append(edge_id)
        required_types.append(int(edge_type))
    required.append(_lookup_edge(edge_lookup, (layout.left_anchor_idx, layout.left_root_idx)))
    required_types.append(int(attachment_bond_type))
    required.append(_lookup_edge(edge_lookup, (layout.right_anchor_idx, layout.right_root_idx)))
    required_types.append(int(attachment_bond_type))
    if require_r_chain_edges:
        for edge_pair in _r_chain_edge_pairs(layout):
            required.append(_lookup_edge(edge_lookup, edge_pair))
            required_types.append(int(r_chain_bond_type))

    paired_edges = _paired_internal_edges(layout, edge_lookup)
    required_set = set(required)
    paired_edge_set = {edge_id for pair in paired_edges for edge_id in pair}
    forbidden = _forbidden_edges(
        layout,
        edge_lookup=edge_lookup,
        required_set=required_set,
        paired_edge_set=paired_edge_set,
        core_bond_pairs={tuple(sorted((int(a), int(b)))) for a, b in (core_bond_types or {})},
        forbid_cross_side_edges=forbid_cross_side_edges,
        forbid_core_nonbond_edges=forbid_core_nonbond_edges,
    )
    return DiffGuiDualRConstraintSpec(
        layout=layout,
        halfedge_map=DiffGuiDualRHalfedgeMap(
            required_edges=tuple((int(edge_id), int(edge_type)) for edge_id, edge_type in zip(required, required_types)),
            required_edge_types=tuple(int(value) for value in required_types),
            forbidden_edges=tuple(sorted(forbidden)),
            paired_edges=tuple(paired_edges),
        ),
        num_node_types=int(num_node_types),
        num_edge_types=int(num_edge_types),
        allowed_node_types=tuple(int(value) for value in allowed_node_types),
    )


def batch_diffgui_dual_r_constraint_specs(
    *,
    layout: DiffGuiDualRSlotLayout,
    halfedge_index: torch.Tensor,
    num_graphs: int,
    num_node_types: int,
    num_edge_types: int,
    core_bond_types: dict[tuple[int, int], int] | None = None,
    attachment_bond_type: int = DIFFGUI_SINGLE_BOND_TYPE,
    allowed_node_types: tuple[int, ...] = (),
    forbid_cross_side_edges: bool = True,
    forbid_core_nonbond_edges: bool = True,
    require_r_chain_edges: bool = False,
    r_chain_bond_type: int = DIFFGUI_SINGLE_BOND_TYPE,
) -> tuple[DiffGuiDualRConstraintSpec, ...]:
    """Build one shifted constraint spec per graph in a batched placeholder."""
    specs: list[DiffGuiDualRConstraintSpec] = []
    for graph_index in range(int(num_graphs)):
        offset = graph_index * layout.total_nodes
        shifted_core_bonds = {
            (int(a) + offset, int(b) + offset): int(edge_type)
            for (a, b), edge_type in (core_bond_types or {}).items()
        }
        specs.append(
            build_diffgui_dual_r_constraint_spec(
                layout=layout.for_graph(graph_index),
                halfedge_index=halfedge_index,
                num_node_types=num_node_types,
                num_edge_types=num_edge_types,
                core_bond_types=shifted_core_bonds,
                attachment_bond_type=attachment_bond_type,
                allowed_node_types=allowed_node_types,
                forbid_cross_side_edges=forbid_cross_side_edges,
                forbid_core_nonbond_edges=forbid_core_nonbond_edges,
                require_r_chain_edges=require_r_chain_edges,
                r_chain_bond_type=r_chain_bond_type,
            )
        )
    return tuple(specs)


def _halfedge_lookup(halfedge_index: torch.Tensor) -> dict[tuple[int, int], int]:
    """Return a lookup from unordered local node pair to halfedge id."""
    lookup: dict[tuple[int, int], int] = {}
    for edge_id in range(int(halfedge_index.shape[1])):
        a = int(halfedge_index[0, edge_id].item())
        b = int(halfedge_index[1, edge_id].item())
        if a == b:
            raise ValueError("halfedge_index must not contain self edges.")
        key = (a, b) if a < b else (b, a)
        if key in lookup:
            raise ValueError(f"Duplicate halfedge for node pair {key}.")
        lookup[key] = edge_id
    return lookup


def _lookup_edge(edge_lookup: dict[tuple[int, int], int], pair: tuple[int, int]) -> int:
    """Look up an unordered node pair in a halfedge table."""
    a, b = int(pair[0]), int(pair[1])
    key = (a, b) if a < b else (b, a)
    try:
        return int(edge_lookup[key])
    except KeyError as exc:
        raise ValueError(f"Missing halfedge for node pair {key}.") from exc


def _paired_internal_edges(
    layout: DiffGuiDualRSlotLayout,
    edge_lookup: dict[tuple[int, int], int],
) -> list[tuple[int, int]]:
    """Return corresponding internal edge ids for R1 and R2 slots."""
    pairs: list[tuple[int, int]] = []
    for first in range(layout.r_slots_per_side):
        for second in range(first + 1, layout.r_slots_per_side):
            left = (layout.left_r_indices[first], layout.left_r_indices[second])
            right = (layout.right_r_indices[first], layout.right_r_indices[second])
            pairs.append((_lookup_edge(edge_lookup, left), _lookup_edge(edge_lookup, right)))
    return pairs


def _r_chain_edge_pairs(layout: DiffGuiDualRSlotLayout) -> list[tuple[int, int]]:
    """Return consecutive left/right R-slot pairs for a linear sidechain scaffold."""
    pairs: list[tuple[int, int]] = []
    for indices in (layout.left_r_indices, layout.right_r_indices):
        for idx in range(len(indices) - 1):
            pairs.append((indices[idx], indices[idx + 1]))
    return pairs


def _forbidden_edges(
    layout: DiffGuiDualRSlotLayout,
    *,
    edge_lookup: dict[tuple[int, int], int],
    required_set: set[int],
    paired_edge_set: set[int],
    core_bond_pairs: set[tuple[int, int]],
    forbid_cross_side_edges: bool,
    forbid_core_nonbond_edges: bool,
) -> set[int]:
    """Return no-bond halfedges that prevent core leakage and side crossing."""
    forbidden: set[int] = set()
    core_indices = set(layout.core_indices())
    r_indices = set(layout.r_indices())
    attachment_allowed = {
        frozenset((layout.left_anchor_idx, layout.left_root_idx)),
        frozenset((layout.right_anchor_idx, layout.right_root_idx)),
    }
    for core_idx in core_indices:
        for r_idx in r_indices:
            if frozenset((core_idx, r_idx)) in attachment_allowed:
                continue
            forbidden.add(_lookup_edge(edge_lookup, (core_idx, r_idx)))

    if forbid_core_nonbond_edges:
        ordered_core = sorted(core_indices)
        for left_pos, left_idx in enumerate(ordered_core):
            for right_idx in ordered_core[left_pos + 1 :]:
                pair = tuple(sorted((left_idx, right_idx)))
                if pair in core_bond_pairs:
                    continue
                forbidden.add(_lookup_edge(edge_lookup, pair))

    if forbid_cross_side_edges:
        for left_idx in layout.left_r_indices:
            for right_idx in layout.right_r_indices:
                forbidden.add(_lookup_edge(edge_lookup, (left_idx, right_idx)))

    return forbidden.difference(required_set).difference(paired_edge_set)
