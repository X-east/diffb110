"""MoleculeWriter: per-rank molecule CSV + SDF writing and epoch merging.

Responsibility:
    Write one CSV row per generated molecule (per rank), then merge
    all rank CSVs into a single epoch molecules.csv on rank-0.
    Also write per-rank multi-molecule SDF files (original 3D coordinates)
    and merge them into molecules.sdf on rank-0.

Note:
    FIELDNAMES uses 'rdkit_valid' (canonical field name, Bug4 Fix-6).
    Old code used 'valid' — do not revert.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

from rdkit import Chem

from ddpo.runtime.recording.run_context import RunContext
from ddpo.runtime.recording.schemas import BatchFacts


class MoleculeWriter:
    """Write per-rank molecule rows and merge per epoch."""

    FIELDNAMES = [
        "epoch", "batch_idx", "rank", "molecule_id",
        "smiles", "rdkit_valid",
        "qed_raw", "sa_raw", "docking_raw", "similarity_raw", "diversity_raw",
        "qed_reward", "sa_reward", "docking_reward", "diversity_reward",
        "chembl_prior_reward", "chembl_mean_similarity",
        "reward_total", "reward_total_raw", "reward_total_norm",
    ]

    def __init__(self, run_context: RunContext) -> None:
        self.run_context = run_context

    def _epoch_dir(self, epoch: int) -> Path:
        epoch_dir = Path(self.run_context.epochs_dir) / f"epoch_{epoch:04d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        return epoch_dir

    def write_rank_batch(self, facts: BatchFacts) -> None:
        """Append molecule rows from one batch to the rank-local CSV."""
        epoch_dir = self._epoch_dir(facts.epoch)
        path = epoch_dir / f"molecules_rank{facts.rank}.csv"
        file_exists = path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            if not file_exists:
                writer.writeheader()
            for row in facts.molecule_rows:
                writer.writerow({key: str(row.get(key, "")) for key in self.FIELDNAMES})

    def write_rank_sdf(self, facts: BatchFacts, rdmols: list) -> None:
        """Write per-rank molecules as multi-molecule SDF (original 3D coordinates).

        Only valid molecules (rdmols[i] is not None) are written.
        """
        epoch_dir = self._epoch_dir(facts.epoch)
        path = epoch_dir / f"molecules_rank{facts.rank}.sdf"
        valid_mols = [mol for mol in rdmols if mol is not None]
        if not valid_mols:
            return
        writer = Chem.SDWriter(str(path))
        try:
            for mol in valid_mols:
                writer.write(mol)
        finally:
            writer.close()

    def merge_epoch(self, epoch: int, world_size: int) -> Optional[Path]:
        """Merge all rank CSVs into a single molecules.csv for the epoch.

        Called on rank-0 only after all ranks have written their rows.
        Also merges per-rank SDFs into molecules.sdf.

        Returns:
            Path to merged CSV, or None if no rank files found.
        """
        epoch_dir = self._epoch_dir(epoch)
        merged_path = epoch_dir / "molecules.csv"
        merged_rows = []
        for rank in range(world_size):
            rank_path = epoch_dir / f"molecules_rank{rank}.csv"
            if rank_path.exists():
                with open(rank_path, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    merged_rows.extend(list(reader))
        if not merged_rows:
            return None
        with open(merged_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writeheader()
            for row in merged_rows:
                writer.writerow({key: str(row.get(key, "")) for key in self.FIELDNAMES})

        self._merge_sdf(epoch, world_size)
        return merged_path

    def _merge_sdf(self, epoch: int, world_size: int) -> None:
        """Merge per-rank SDF files into a single molecules.sdf."""
        epoch_dir = self._epoch_dir(epoch)
        merged_path = epoch_dir / "molecules.sdf"
        all_mols = []
        for rank in range(world_size):
            rank_path = epoch_dir / f"molecules_rank{rank}.sdf"
            if rank_path.exists() and rank_path.stat().st_size > 0:
                try:
                    suppl = Chem.SDMolSupplier(str(rank_path))
                    all_mols.extend(m for m in suppl if m is not None)
                except Exception:
                    continue
        if not all_mols:
            return
        writer = Chem.SDWriter(str(merged_path))
        try:
            for mol in all_mols:
                writer.write(mol)
        finally:
            writer.close()
