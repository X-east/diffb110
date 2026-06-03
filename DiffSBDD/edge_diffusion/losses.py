"""Loss helpers for standalone ligand x/h/e edge diffusion experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from DiffSBDD.edge_diffusion.dataset import EdgeBatch
from DiffSBDD.edge_diffusion.schema import EdgeDiffusionOutput


@dataclass(frozen=True)
class EdgeDiffusionLossConfig:
    """Weights for independent x/h/e training objectives."""

    pos_weight: float = 1.0
    atom_weight: float = 1.0
    edge_weight: float = 1.0
    bond_length_weight: float = 0.2


@dataclass(frozen=True)
class EdgeDiffusionLossResult:
    """Structured edge diffusion loss terms."""

    total: torch.Tensor
    pos_loss: torch.Tensor
    atom_loss: torch.Tensor
    edge_loss: torch.Tensor
    bond_length_loss: torch.Tensor

    def as_log_dict(self, prefix: str = "edge_diffusion") -> dict[str, float]:
        """Return detached scalar losses for logging."""
        return {
            f"{prefix}/total": float(self.total.detach().cpu()),
            f"{prefix}/pos": float(self.pos_loss.detach().cpu()),
            f"{prefix}/atom": float(self.atom_loss.detach().cpu()),
            f"{prefix}/edge": float(self.edge_loss.detach().cpu()),
            f"{prefix}/bond_length": float(self.bond_length_loss.detach().cpu()),
        }


def compute_edge_diffusion_loss(
    output: EdgeDiffusionOutput,
    batch: EdgeBatch,
    config: EdgeDiffusionLossConfig | None = None,
    atom_class_weight: torch.Tensor | None = None,
    edge_class_weight: torch.Tensor | None = None,
) -> EdgeDiffusionLossResult:
    """Compute decoupled x/h/e denoising losses for a collated edge batch."""
    loss_config = config or EdgeDiffusionLossConfig()
    _validate_output_shapes(output, batch)

    pos_loss = F.mse_loss(output.pred_pos, batch.ligand_pos)
    atom_loss = F.cross_entropy(
        output.pred_atom_logits,
        batch.ligand_atom_type.long(),
        weight=None if atom_class_weight is None else atom_class_weight.to(output.pred_atom_logits.device),
    )
    if batch.target_edge_type.numel() == 0:
        edge_loss = output.pred_pos.new_zeros(())
    else:
        edge_loss = F.cross_entropy(
            output.pred_edge_logits,
            batch.target_edge_type.long(),
            weight=None if edge_class_weight is None else edge_class_weight.to(output.pred_edge_logits.device),
        )
    bond_length_loss = _bond_length_loss(output.pred_pos, batch)
    total = (
        float(loss_config.pos_weight) * pos_loss
        + float(loss_config.atom_weight) * atom_loss
        + float(loss_config.edge_weight) * edge_loss
        + float(loss_config.bond_length_weight) * bond_length_loss
    )
    return EdgeDiffusionLossResult(
        total=total,
        pos_loss=pos_loss,
        atom_loss=atom_loss,
        edge_loss=edge_loss,
        bond_length_loss=bond_length_loss,
    )


def _bond_length_loss(pred_pos: torch.Tensor, batch: EdgeBatch) -> torch.Tensor:
    """Return smooth-L1 distance loss on true bonded halfedges only."""
    is_bond = batch.target_edge_type > 0
    if not bool(is_bond.any()):
        return pred_pos.new_zeros(())
    edge_index = batch.halfedge_index[:, is_bond]
    src, dst = edge_index
    pred_distance = torch.linalg.norm(pred_pos[src] - pred_pos[dst], dim=-1)
    target_distance = torch.linalg.norm(batch.ligand_pos[src] - batch.ligand_pos[dst], dim=-1)
    return F.smooth_l1_loss(pred_distance, target_distance)


def _validate_output_shapes(output: EdgeDiffusionOutput, batch: EdgeBatch) -> None:
    """Validate output tensors against edge batch targets."""
    if output.pred_pos.shape != batch.ligand_pos.shape:
        raise ValueError("pred_pos shape must match batch.ligand_pos")
    if output.pred_atom_logits.ndim != 2:
        raise ValueError("pred_atom_logits must have shape [num_atoms, num_atom_types]")
    if output.pred_atom_logits.shape[0] != batch.ligand_atom_type.shape[0]:
        raise ValueError("pred_atom_logits first dimension must match ligand atoms")
    if output.pred_edge_logits.ndim != 2:
        raise ValueError("pred_edge_logits must have shape [num_halfedges, num_edge_types]")
    if output.pred_edge_logits.shape[0] != batch.target_edge_type.shape[0]:
        raise ValueError("pred_edge_logits first dimension must match target_edge_type")
