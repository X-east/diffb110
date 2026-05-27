"""
Mask format conversion utilities between DiffSBDD and padded DDPO batches.

DiffSBDD uses batch_mask format: [total_nodes] tensor with batch indices
The local actor uses node_mask format: [B, N, 1] tensor with 0/1 values

This module provides bidirectional conversion between these formats.
"""

import torch
from typing import Optional


def batch_mask_to_node_mask(batch_mask: torch.Tensor) -> torch.Tensor:
    """
    Convert DiffSBDD batch_mask to padded node_mask.
    
    Args:
        batch_mask: [total_nodes] tensor with batch indices (0, 0, 0, 1, 1, 2, 2, 2, ...)
                   Each value indicates which batch the node belongs to.
    
    Returns:
        node_mask: [B, N, 1] tensor with 0/1 values
                  B = number of unique batch indices
                  N = maximum node count across batches
                  1 indicates valid node, 0 indicates padding
    
    Raises:
        ValueError: If batch_mask contains negative values or is empty
    
    Example:
        >>> batch_mask = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2])
        >>> node_mask = batch_mask_to_node_mask(batch_mask)
        >>> node_mask.shape
        torch.Size([3, 4, 1])  # 3 batches, max 4 nodes
    """
    if batch_mask.numel() == 0:
        raise ValueError("batch_mask cannot be empty")
    
    if batch_mask.dim() != 1:
        raise ValueError(f"batch_mask must be 1D, got shape {batch_mask.shape}")
    
    # Check for negative values
    if (batch_mask < 0).any():
        invalid_val = batch_mask[batch_mask < 0][0].item()
        raise ValueError(
            f"Invalid batch_mask value {invalid_val}, expected non-negative values"
        )
    
    # Get batch size (number of unique batches)
    batch_size = batch_mask.max().item() + 1
    
    # Check for gaps in batch indices
    unique_batches = torch.unique(batch_mask)
    expected_batches = torch.arange(batch_size, device=batch_mask.device)
    if not torch.equal(unique_batches, expected_batches):
        raise ValueError(
            f"batch_mask has gaps in batch indices. Found {unique_batches.tolist()}, "
            f"expected continuous range [0, {batch_size - 1}]"
        )
    
    # Count nodes per batch
    nodes_per_batch = torch.bincount(batch_mask, minlength=batch_size)
    max_nodes = nodes_per_batch.max().item()
    
    # Create node_mask [B, N, 1]
    node_mask = torch.zeros(batch_size, max_nodes, 1, 
                           dtype=torch.float32, device=batch_mask.device)
    
    # Fill in valid positions
    for b in range(batch_size):
        n_nodes = nodes_per_batch[b].item()
        node_mask[b, :n_nodes, 0] = 1.0
    
    return node_mask


def node_mask_to_batch_mask(node_mask: torch.Tensor) -> torch.Tensor:
    """
    Convert padded node_mask to DiffSBDD batch_mask.
    
    Args:
        node_mask: [B, N, 1] tensor with 0/1 values
                  B = batch size
                  N = max nodes per sample
                  1 indicates valid node, 0 indicates padding
    
    Returns:
        batch_mask: [total_valid_nodes] tensor with batch indices
                   Each value indicates which batch the node belongs to.
    
    Raises:
        ValueError: If node_mask contains values other than 0 or 1
        ValueError: If node_mask has invalid shape
    
    Example:
        >>> node_mask = torch.tensor([[[1], [1], [1], [0]],
        ...                           [[1], [1], [0], [0]],
        ...                           [[1], [1], [1], [1]]])
        >>> batch_mask = node_mask_to_batch_mask(node_mask)
        >>> batch_mask
        tensor([0, 0, 0, 1, 1, 2, 2, 2, 2])
    """
    if node_mask.dim() != 3 or node_mask.shape[2] != 1:
        raise ValueError(
            f"node_mask must have shape [B, N, 1], got {node_mask.shape}"
        )
    
    # Squeeze last dimension for easier processing
    mask_2d = node_mask.squeeze(-1)  # [B, N]
    
    # Check for valid values (0 or 1)
    unique_vals = torch.unique(mask_2d)
    valid_vals = torch.tensor([0.0, 1.0], device=mask_2d.device)
    
    for val in unique_vals:
        if val not in valid_vals:
            raise ValueError(
                f"Invalid node_mask values, expected 0 or 1, found {val.item()}"
            )
    
    batch_size, max_nodes = mask_2d.shape
    
    # Create batch indices for all positions
    batch_indices = torch.arange(batch_size, device=node_mask.device)
    batch_indices = batch_indices.unsqueeze(1).expand(-1, max_nodes)  # [B, N]
    
    # Select only valid nodes
    valid_mask = mask_2d.bool()  # [B, N]
    batch_mask = batch_indices[valid_mask]  # [total_valid_nodes]
    
    return batch_mask.long()


def get_nodes_per_batch(batch_mask: torch.Tensor) -> torch.Tensor:
    """
    Get the number of nodes in each batch from batch_mask.
    
    Args:
        batch_mask: [total_nodes] tensor with batch indices
    
    Returns:
        nodes_per_batch: [B] tensor with node counts per batch
    
    Example:
        >>> batch_mask = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2])
        >>> get_nodes_per_batch(batch_mask)
        tensor([3, 2, 4])
    """
    if batch_mask.numel() == 0:
        return torch.tensor([], dtype=torch.long, device=batch_mask.device)
    
    batch_size = batch_mask.max().item() + 1
    return torch.bincount(batch_mask, minlength=batch_size)


def create_batch_mask_from_counts(nodes_per_batch: torch.Tensor) -> torch.Tensor:
    """
    Create batch_mask from node counts per batch.
    
    Args:
        nodes_per_batch: [B] tensor with node counts per batch
    
    Returns:
        batch_mask: [total_nodes] tensor with batch indices
    
    Example:
        >>> nodes_per_batch = torch.tensor([3, 2, 4])
        >>> create_batch_mask_from_counts(nodes_per_batch)
        tensor([0, 0, 0, 1, 1, 2, 2, 2, 2])
    """
    batch_mask_list = []
    for b, n_nodes in enumerate(nodes_per_batch):
        batch_mask_list.append(torch.full((n_nodes.item(),), b, dtype=torch.long))
    
    if not batch_mask_list:
        return torch.tensor([], dtype=torch.long)
    
    return torch.cat(batch_mask_list, dim=0)
