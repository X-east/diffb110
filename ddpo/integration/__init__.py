"""
Integration module for DiffSBDD model wrapper and utilities.

Contains:
- mask_utils: Mask format conversion between DiffSBDD and the local DDPO actor
- DiffSBDDModel: Adapter wrapping DiffSBDD for the local DDPO interface
"""

from .mask_utils import (
    batch_mask_to_node_mask,
    node_mask_to_batch_mask,
    get_nodes_per_batch,
    create_batch_mask_from_counts,
)

# Lazy import DiffSBDDModel to avoid heavy dependencies (wandb, etc.)
# when only mask_utils is needed
def __getattr__(name):
    if name == "DiffSBDDModel":
        from .diffsbdd_model import DiffSBDDModel
        return DiffSBDDModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "batch_mask_to_node_mask",
    "node_mask_to_batch_mask",
    "get_nodes_per_batch",
    "create_batch_mask_from_counts",
    "DiffSBDDModel",
]
