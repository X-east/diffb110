"""Dataset and collation helpers for standalone edge diffusion bundles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class EdgeBatch:
    """Mini-batch of ligand halfedge labels for edge denoising."""

    ligand_pos: torch.Tensor
    ligand_atom_type: torch.Tensor
    halfedge_index: torch.Tensor
    target_edge_type: torch.Tensor
    atom_batch: torch.Tensor
    edge_batch: torch.Tensor
    names: list[str]

    def to(self, device: torch.device | str) -> EdgeBatch:
        """Move tensor fields to device."""
        return EdgeBatch(
            ligand_pos=self.ligand_pos.to(device),
            ligand_atom_type=self.ligand_atom_type.to(device),
            halfedge_index=self.halfedge_index.to(device),
            target_edge_type=self.target_edge_type.to(device),
            atom_batch=self.atom_batch.to(device),
            edge_batch=self.edge_batch.to(device),
            names=self.names,
        )


class EdgeBundleDataset(Dataset):
    """Read variable-sized ligand edge bundles produced by build_edge_data_chain."""

    def __init__(
        self,
        edge_npz: Path | str,
        semantic_npz: Path | str | None = None,
        max_molecules: int | None = None,
    ) -> None:
        """Load edge arrays and optionally validate semantic sidecar alignment."""
        self.edge_npz = Path(edge_npz)
        self.semantic_npz = None if semantic_npz is None else Path(semantic_npz)
        with np.load(self.edge_npz, allow_pickle=True) as handle:
            self._edge_data = {key: handle[key] for key in handle.files}
        self._validate_required_fields()
        self._length = len(self._edge_data["names"])
        if max_molecules is not None:
            self._length = min(self._length, int(max_molecules))
        if self.semantic_npz is not None:
            self._validate_semantic_alignment()

    def __len__(self) -> int:
        """Return number of molecules exposed by this dataset."""
        return self._length

    def __getitem__(self, idx: int) -> dict[str, object]:
        """Return one molecule edge sample."""
        if idx < 0 or idx >= self._length:
            raise IndexError(idx)
        return {
            "name": str(self._edge_data["names"][idx]),
            "ligand_pos": torch.as_tensor(
                self._edge_data["ligand_pos"][idx],
                dtype=torch.float32,
            ),
            "ligand_atom_type": torch.as_tensor(
                self._edge_data["ligand_atom_type"][idx],
                dtype=torch.long,
            ),
            "halfedge_index": torch.as_tensor(
                self._edge_data["ligand_halfedge_index"][idx],
                dtype=torch.long,
            ),
            "target_edge_type": torch.as_tensor(
                self._edge_data["ligand_halfedge_type"][idx],
                dtype=torch.long,
            ),
        }

    def _validate_required_fields(self) -> None:
        """Ensure the edge bundle contains fields required for denoising."""
        required_fields = {
            "names",
            "ligand_pos",
            "ligand_atom_type",
            "ligand_halfedge_index",
            "ligand_halfedge_type",
        }
        missing = sorted(required_fields - set(self._edge_data))
        if missing:
            raise KeyError(f"edge bundle missing required fields: {missing}")

    def _validate_semantic_alignment(self) -> None:
        """Validate that semantic sidecar names match edge bundle names."""
        with np.load(self.semantic_npz, allow_pickle=True) as handle:
            semantic_names = handle["names"]
        if len(semantic_names) < self._length:
            raise ValueError("semantic sidecar has fewer molecules than edge bundle")
        for idx in range(self._length):
            if str(semantic_names[idx]) != str(self._edge_data["names"][idx]):
                raise ValueError(f"semantic sidecar name mismatch at index {idx}")


class EdgeBatchCollator:
    """Collate variable-sized edge samples into one concatenated EdgeBatch."""

    def __call__(self, samples: list[dict[str, object]]) -> EdgeBatch:
        """Collate samples with atom index offsets."""
        ligand_pos = []
        ligand_atom_type = []
        halfedge_index = []
        target_edge_type = []
        atom_batch = []
        edge_batch = []
        names = []
        atom_offset = 0
        for sample_idx, sample in enumerate(samples):
            pos = sample["ligand_pos"]
            atom_type = sample["ligand_atom_type"]
            edge_index = sample["halfedge_index"]
            edge_type = sample["target_edge_type"]
            ligand_pos.append(pos)
            ligand_atom_type.append(atom_type)
            halfedge_index.append(edge_index + atom_offset)
            target_edge_type.append(edge_type)
            atom_batch.append(torch.full((len(atom_type),), sample_idx, dtype=torch.long))
            edge_batch.append(torch.full((len(edge_type),), sample_idx, dtype=torch.long))
            names.append(str(sample["name"]))
            atom_offset += int(len(atom_type))

        return EdgeBatch(
            ligand_pos=torch.cat(ligand_pos, dim=0),
            ligand_atom_type=torch.cat(ligand_atom_type, dim=0),
            halfedge_index=torch.cat(halfedge_index, dim=1),
            target_edge_type=torch.cat(target_edge_type, dim=0),
            atom_batch=torch.cat(atom_batch, dim=0),
            edge_batch=torch.cat(edge_batch, dim=0),
            names=names,
        )
