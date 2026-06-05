"""Typed containers for the standalone edge diffusion backend."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EdgeDiffusionInputs:
    """Noised x/h/e state consumed by the edge diffusion denoiser."""

    ligand_pos_t: torch.Tensor
    ligand_atom_type_t: torch.Tensor
    ligand_edge_type_t: torch.Tensor
    t: torch.Tensor

    def to(self, device: torch.device | str) -> EdgeDiffusionInputs:
        """Move tensor fields to device."""
        return EdgeDiffusionInputs(
            ligand_pos_t=self.ligand_pos_t.to(device),
            ligand_atom_type_t=self.ligand_atom_type_t.to(device),
            ligand_edge_type_t=self.ligand_edge_type_t.to(device),
            t=self.t.to(device),
        )


@dataclass(frozen=True)
class EdgeDiffusionOutput:
    """Predicted clean x/h/e state from the edge diffusion denoiser."""

    pred_pos: torch.Tensor
    pred_atom_logits: torch.Tensor
    pred_edge_logits: torch.Tensor

    @property
    def pred_atom_type(self) -> torch.Tensor:
        """Return argmax atom type predictions."""
        return self.pred_atom_logits.argmax(dim=-1)

    @property
    def pred_edge_type(self) -> torch.Tensor:
        """Return argmax halfedge bond type predictions."""
        return self.pred_edge_logits.argmax(dim=-1)
