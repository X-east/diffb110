"""Checkpoint-backed D-MPNN predictor used by training and reward code."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from rdkit import Chem

from chembl_classifier.dmpnn.model import DMPNNClassifier, mols_to_graph_tensors

logger = logging.getLogger(__name__)


class DMPNNClassifierPredictor:
    """Thin inference wrapper around a trained DMPNNClassifier checkpoint."""

    def __init__(
        self,
        model: DMPNNClassifier,
        *,
        device: str | torch.device = "cpu",
        cache_smiles: bool = True,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.cache_smiles = bool(cache_smiles)
        self._cache: dict[str, float] = {}

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device | None = None,
        cache_smiles: bool = True,
    ) -> "DMPNNClassifierPredictor":
        """Load predictor from a checkpoint path."""
        resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = DMPNNClassifier.from_checkpoint(str(checkpoint_path), device=resolved_device)
        return cls(model, device=resolved_device, cache_smiles=cache_smiles)

    @torch.no_grad()
    def predict_proba_mols(self, mols: list[Any], batch_size: int = 128) -> list[float | None]:
        """Return P(good molecule) for each RDKit molecule, preserving failures."""
        output: list[float | None] = [None] * len(mols)
        pending_mols: list[Any] = []
        pending_indices: list[int] = []

        for index, mol in enumerate(mols):
            if mol is None:
                continue
            smiles = _safe_smiles(mol)
            if self.cache_smiles and smiles and smiles in self._cache:
                output[index] = self._cache[smiles]
                continue
            pending_mols.append(mol)
            pending_indices.append(index)

        for start in range(0, len(pending_mols), batch_size):
            chunk = pending_mols[start : start + batch_size]
            chunk_indices = pending_indices[start : start + batch_size]
            try:
                graph = mols_to_graph_tensors(chunk, device=self.device)
                logits = self.model(graph)
                probs = torch.sigmoid(logits).detach().cpu().reshape(-1).tolist()
            except Exception:
                logger.debug("D-MPNN inference failed for a molecule chunk", exc_info=True)
                continue
            for index, mol, prob in zip(chunk_indices, chunk, probs):
                value = float(prob)
                output[index] = value
                smiles = _safe_smiles(mol)
                if self.cache_smiles and smiles:
                    self._cache[smiles] = value
        return output

    @property
    def n_params(self) -> int:
        """Number of trainable model parameters."""
        return self.model.n_params


def _safe_smiles(mol: Any) -> str:
    """Return canonical SMILES, or an empty string if RDKit conversion fails."""
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return ""
