"""
BatchRecorder: Per-batch molecule and reward recorder for DDPO training.

Writes two files under <output_dir>/logs/:
  molecules_<ts>.csv   - one row per molecule per batch
  batch_stats_<ts>.csv - one row per batch (aggregate stats)

Design rules:
- Never fabricate scores; use NaN for missing values.
- Never silently drop failed molecules; record them with is_valid=False.
- Flush after every batch so partial runs are readable.
- Only called from rank-0.
"""

import csv
import logging
from pathlib import Path
from typing import Dict, Optional

import torch

log = logging.getLogger("DiffSBDD-DDPO")

REWARD_COMPONENT_KEYS = [
    ("reward_total", "rewards"),
    ("reward_docking", "docking_reward"),
    ("reward_druglikeness", "druglikeness_reward"),
    ("score_qed", "qed_scores"),
    ("score_sa", "sa_scores"),
]

MOL_HEADER = (
    ["epoch", "batch", "mol_idx", "n_atoms", "smiles", "is_valid"]
    + [col for col, _ in REWARD_COMPONENT_KEYS]
)

STAT_HEADER = (
    ["epoch", "batch", "n_total", "n_valid", "valid_ratio", "filtering_ratio", "novelty_penalty_ratio"]
    + [f"{col}_mean" for col, _ in REWARD_COMPONENT_KEYS]
    + [f"{col}_std" for col, _ in REWARD_COMPONENT_KEYS]
)


class BatchRecorder:
    """Records per-molecule and per-batch reward statistics to CSV files."""

    def __init__(self, logs_dir: str, dataset_info: dict, timestamp: str):
        self.logs_dir = Path(logs_dir)
        self.dataset_info = dataset_info
        self.timestamp = timestamp

        mol_path = self.logs_dir / f"molecules_{timestamp}.csv"
        stat_path = self.logs_dir / f"batch_stats_{timestamp}.csv"

        self._mol_fh = open(mol_path, "w", newline="", encoding="utf-8")
        self._stat_fh = open(stat_path, "w", newline="", encoding="utf-8")
        self._mol_w = csv.DictWriter(self._mol_fh, fieldnames=MOL_HEADER)
        self._stat_w = csv.DictWriter(self._stat_fh, fieldnames=STAT_HEADER)
        self._mol_w.writeheader()
        self._stat_w.writeheader()
        self._mol_fh.flush()
        self._stat_fh.flush()

        log.info(f"BatchRecorder: molecules   -> {mol_path}")
        log.info(f"BatchRecorder: batch_stats -> {stat_path}")

    def record_batch(
        self,
        epoch: int,
        batch_idx: int,
        batch: dict,
        filtering_ratio: float = 0.0,
        novelty_penalty_ratio: float = 0.0,
    ) -> None:
        nodesxsample = batch.get("nodesxsample")
        if nodesxsample is None:
            log.warning("BatchRecorder: 'nodesxsample' missing; skipping record.")
            return

        batch_size = int(nodesxsample.shape[0])
        smiles_list = batch.get("smiles")
        # Bug4 Fix-5: primary field is 'rdkit_valid'; 'valid' fallback is legacy-only.
        # Remove batch.get('valid') fallback once all upstream code sets 'rdkit_valid'.
        valid_list = batch.get("rdkit_valid", batch.get("valid"))
        if smiles_list is None or valid_list is None:
            log.warning("BatchRecorder: canonical molecule semantics missing; skipping record.")
            return

        component_arrays: Dict[str, Optional[list]] = {}
        for col, key in REWARD_COMPONENT_KEYS:
            t = batch.get(key)
            component_arrays[col] = (
                t.detach().cpu().tolist() if (t is not None and isinstance(t, torch.Tensor)) else None
            )

        valid_flags = []
        for i in range(batch_size):
            n_atoms = int(nodesxsample[i].item())
            smiles, is_valid = self._try_get_smiles(batch, i)
            valid_flags.append(is_valid)

            row = {
                "epoch": epoch,
                "batch": batch_idx,
                "mol_idx": i,
                "n_atoms": n_atoms,
                "smiles": smiles,
                "is_valid": int(is_valid),
            }
            for col, _ in REWARD_COMPONENT_KEYS:
                arr = component_arrays[col]
                row[col] = f"{arr[i]:.6f}" if arr is not None else "nan"

            self._mol_w.writerow(row)
        self._mol_fh.flush()

        n_valid = sum(valid_flags)
        stat_row = {
            "epoch": epoch,
            "batch": batch_idx,
            "n_total": batch_size,
            "n_valid": n_valid,
            "valid_ratio": f"{n_valid / max(batch_size, 1):.4f}",
            "filtering_ratio": f"{filtering_ratio:.4f}",
            "novelty_penalty_ratio": f"{novelty_penalty_ratio:.4f}",
        }
        for col, _ in REWARD_COMPONENT_KEYS:
            arr = component_arrays[col]
            if arr is not None:
                t = torch.tensor(arr, dtype=torch.float32)
                stat_row[f"{col}_mean"] = f"{t.mean().item():.6f}"
                stat_row[f"{col}_std"] = f"{t.std().item():.6f}"
            else:
                stat_row[f"{col}_mean"] = "nan"
                stat_row[f"{col}_std"] = "nan"

        self._stat_w.writerow(stat_row)
        self._stat_fh.flush()

        total_arr = component_arrays.get("reward_total")
        if total_arr is not None:
            t = torch.tensor(total_arr, dtype=torch.float32)
            reward_str = (
                f"reward mean={t.mean():.4f}  std={t.std():.4f}  "
                f"min={t.min():.4f}  max={t.max():.4f}"
            )
        else:
            reward_str = "reward=N/A"

        comp_parts = []
        for col, _ in REWARD_COMPONENT_KEYS:
            if col == "reward_total":
                continue
            arr = component_arrays[col]
            if arr is not None:
                comp_parts.append(f"{col}={sum(arr)/len(arr):.4f}")

        log.info(
            f"[E{epoch} B{batch_idx}] "
            f"n={batch_size} valid={n_valid} "
            f"({n_valid/max(batch_size,1)*100:.0f}%) | "
            + reward_str
            + ("  |  " + "  ".join(comp_parts) if comp_parts else "")
        )

    def close(self) -> None:
        for fh in (self._mol_fh, self._stat_fh):
            fh.flush()
            fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _try_get_smiles(self, batch: dict, i: int):
        """Read canonical smiles/validity from batch semantics."""
        smiles_list = batch.get("smiles", [])
        # Bug4 Fix-5: primary field is 'rdkit_valid'; 'valid' fallback is legacy-only.
        # Remove batch.get('valid') fallback once all upstream code sets 'rdkit_valid'.
        valid_list = batch.get("rdkit_valid", batch.get("valid", []))

        smiles = smiles_list[i] if i < len(smiles_list) else None
        is_valid = int(valid_list[i]) == 1 if i < len(valid_list) else False
        return (str(smiles) if smiles is not None else "N/A", is_valid)
