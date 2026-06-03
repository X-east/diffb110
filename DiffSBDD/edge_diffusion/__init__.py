"""Prototype utilities for DiffSBDD edge diffusion experiments."""

from .dataset import EdgeBatch, EdgeBatchCollator, EdgeBundleDataset
from .denoiser import (
    EdgeDenoiserConfig,
    EdgeDenoiserMLP,
    EdgeDenoiserNodeEdge,
    make_edge_class_weight,
)
from .evaluation import (
    EdgeReconstructionEvalConfig,
    batch_confusion,
    classification_metrics,
    compare_halfedge_bond_maps,
    evaluate_edge_reconstruction,
    reconstruction_fieldnames,
    reconstruction_metric_defaults,
    reconstruction_summary_for_training,
)
from .losses import (
    EdgeDiffusionLossConfig,
    EdgeDiffusionLossResult,
    compute_edge_diffusion_loss,
)
from .model import EdgeDiffusionModel, EdgeDiffusionModelConfig
from .oracle_reconstruction import OracleReconstructionResult, build_oracle_mol
from .schema import EdgeDiffusionInputs, EdgeDiffusionOutput
from .semantic_oracle_reconstruction import build_semantic_oracle_mol
from .transitions import EdgeCategoricalTransition, EdgeTransitionConfig

__all__ = [
    "EdgeBatch",
    "EdgeBatchCollator",
    "EdgeBundleDataset",
    "EdgeCategoricalTransition",
    "EdgeDenoiserConfig",
    "EdgeDenoiserMLP",
    "EdgeDenoiserNodeEdge",
    "EdgeDiffusionInputs",
    "EdgeDiffusionLossConfig",
    "EdgeDiffusionLossResult",
    "EdgeDiffusionModel",
    "EdgeDiffusionModelConfig",
    "EdgeDiffusionOutput",
    "EdgeReconstructionEvalConfig",
    "EdgeTransitionConfig",
    "OracleReconstructionResult",
    "batch_confusion",
    "build_oracle_mol",
    "build_semantic_oracle_mol",
    "classification_metrics",
    "compute_edge_diffusion_loss",
    "compare_halfedge_bond_maps",
    "evaluate_edge_reconstruction",
    "make_edge_class_weight",
    "reconstruction_fieldnames",
    "reconstruction_metric_defaults",
    "reconstruction_summary_for_training",
]
