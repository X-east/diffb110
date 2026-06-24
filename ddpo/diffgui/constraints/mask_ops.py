"""Pure tensor operations for DiffGui dual-R categorical constraints."""

from __future__ import annotations

import torch

from ddpo.diffgui.constraints.spec import DiffGuiDualRConstraintSpec


def apply_dual_r_constraints_to_log_probs(
    node_log_probs: torch.Tensor,
    halfedge_log_probs: torch.Tensor,
    spec: DiffGuiDualRConstraintSpec,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply paired consensus and hard edge masks to categorical log probabilities."""
    _validate_log_probs(node_log_probs, halfedge_log_probs, spec)
    node_out = node_log_probs.clone()
    halfedge_out = halfedge_log_probs.clone()

    allowed_node_types = spec.allowed_node_types or tuple(range(spec.num_node_types))
    for left_idx, right_idx in spec.layout.paired_node_indices:
        consensus = _consensus_log_probs(
            node_out[left_idx],
            node_out[right_idx],
            allowed_types=allowed_node_types,
        )
        node_out[left_idx] = consensus
        node_out[right_idx] = consensus

    for left_edge_id, right_edge_id in spec.halfedge_map.paired_edges:
        consensus = _consensus_log_probs(
            halfedge_out[left_edge_id],
            halfedge_out[right_edge_id],
            allowed_types=tuple(range(spec.num_edge_types)),
        )
        halfedge_out[left_edge_id] = consensus
        halfedge_out[right_edge_id] = consensus

    for edge_id in spec.halfedge_map.forbidden_edges:
        halfedge_out[int(edge_id)] = _forced_log_probs(
            spec.num_edge_types,
            int(spec.no_bond_type),
            device=halfedge_out.device,
            dtype=halfedge_out.dtype,
        )
    for edge_id, edge_type in spec.halfedge_map.required_edges:
        halfedge_out[int(edge_id)] = _forced_log_probs(
            spec.num_edge_types,
            int(edge_type),
            device=halfedge_out.device,
            dtype=halfedge_out.dtype,
        )
    return node_out, halfedge_out


def apply_dual_r_constraints_to_types(
    node_types: torch.Tensor,
    halfedge_types: torch.Tensor,
    spec: DiffGuiDualRConstraintSpec,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply dual-R constraints to already sampled categorical type indices."""
    if node_types.ndim != 1:
        raise ValueError("node_types must be a 1D tensor.")
    if halfedge_types.ndim != 1:
        raise ValueError("halfedge_types must be a 1D tensor.")
    if node_types.shape[0] <= spec.layout.max_node_index:
        raise ValueError("node_types is shorter than the largest node index referenced by the layout.")
    out_nodes = node_types.clone()
    out_edges = halfedge_types.clone()

    for left_idx, right_idx in spec.layout.paired_node_indices:
        chosen = out_nodes[int(left_idx)]
        out_nodes[int(right_idx)] = chosen

    for left_edge_id, right_edge_id in spec.halfedge_map.paired_edges:
        chosen = out_edges[int(left_edge_id)]
        out_edges[int(right_edge_id)] = chosen

    for edge_id in spec.halfedge_map.forbidden_edges:
        out_edges[int(edge_id)] = int(spec.no_bond_type)
    for edge_id, edge_type in spec.halfedge_map.required_edges:
        out_edges[int(edge_id)] = int(edge_type)
    return out_nodes, out_edges


def force_onehot_from_types(types: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Return one-hot categorical states from forced type ids."""
    if types.ndim != 1:
        raise ValueError("types must be a 1D tensor.")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")
    if types.numel() and (types.min() < 0 or types.max() >= num_classes):
        raise ValueError("types contains ids outside num_classes.")
    return torch.nn.functional.one_hot(types.long(), num_classes=int(num_classes)).to(dtype=torch.float32)


def _validate_log_probs(
    node_log_probs: torch.Tensor,
    halfedge_log_probs: torch.Tensor,
    spec: DiffGuiDualRConstraintSpec,
) -> None:
    """Validate log-probability tensor shapes."""
    if node_log_probs.ndim != 2:
        raise ValueError("node_log_probs must have shape [N, C_node].")
    if halfedge_log_probs.ndim != 2:
        raise ValueError("halfedge_log_probs must have shape [E, C_edge].")
    if node_log_probs.shape[0] <= spec.layout.max_node_index:
        raise ValueError("node_log_probs is shorter than the largest node index referenced by the layout.")
    if node_log_probs.shape[1] != spec.num_node_types:
        raise ValueError("node_log_probs class dimension does not match spec.num_node_types.")
    if halfedge_log_probs.shape[1] != spec.num_edge_types:
        raise ValueError("halfedge_log_probs class dimension does not match spec.num_edge_types.")


def _consensus_log_probs(
    left_log_probs: torch.Tensor,
    right_log_probs: torch.Tensor,
    *,
    allowed_types: tuple[int, ...],
) -> torch.Tensor:
    """Combine two categorical log-probability vectors into one normalized posterior."""
    if left_log_probs.shape != right_log_probs.shape:
        raise ValueError("paired log-probability vectors must have the same shape.")
    logits = left_log_probs + right_log_probs
    mask = torch.full_like(logits, torch.finfo(logits.dtype).min)
    allowed = torch.tensor(allowed_types, dtype=torch.long, device=logits.device)
    if allowed.numel() == 0:
        raise ValueError("allowed_types must not be empty.")
    mask[allowed] = 0
    masked = logits + mask
    return torch.log_softmax(masked, dim=-1)


def _forced_log_probs(
    num_classes: int,
    forced_type: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a log-probability vector with probability one at forced_type."""
    if forced_type < 0 or forced_type >= num_classes:
        raise ValueError("forced_type is outside num_classes.")
    out = torch.full((int(num_classes),), torch.finfo(dtype).min, device=device, dtype=dtype)
    out[int(forced_type)] = 0.0
    return out
