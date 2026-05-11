"""MoleculeWriter: per-rank molecule CSV writing and epoch merging.

Responsibility:
    Write one CSV row per generated molecule (per rank), then merge
    all rank CSVs into a single epoch molecules.csv on rank-0.

Note:
    FIELDNAMES uses 'rdkit_valid' (canonical field name, Bug4 Fix-6).
    Old code used 'valid' — do not revert.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

from ddpo.runtime.recording.run_context import RunContext
from ddpo.runtime.recording.schemas import BatchFacts


class MoleculeWriter:
    """Write per-rank molecule rows and merge per epoch."""

    FIELDNAMES = [
        "epoch", "batch_idx", "rank", "molecule_id",
        "smiles", "rdkit_valid",
        "qed_raw", "sa_raw", "docking_raw",
        "qed_reward", "sa_reward", "docking_reward",
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

    def merge_epoch(self, epoch: int, world_size: int) -> Optional[Path]:
        """Merge all rank CSVs into a single molecules.csv for the epoch.

        Called on rank-0 only after all ranks have written their rows.

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
        return merged_path
