"""
DiffSBDDFilter: dedup / novelty filter consuming canonical molecule semantics.

Canonical semantics source:
- `ddpo.utils.molecule_semantics.annotate_batch_molecule_semantics`
- `rdkit_valid` is the canonical RDKit legality field
- `smiles` is auxiliary only for logging / dedup / novelty

This filter no longer maintains an independent molecule reconstruction path.
"""

import random
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import torch

from constants import dataset_params
from ddpo.utils.molecule_semantics import annotate_batch_molecule_semantics

if TYPE_CHECKING:
    from verl_diffusion.protocol import DataProto


# ============================================================================
# USER CONFIGURABLE PARAMETERS
# ============================================================================

DEFAULT_ENABLE_FILTERING = True
DEFAULT_ENABLE_PENALTY = False
NOVELTY_PENALTY_COEF = 0.1


class DiffSBDDFilter:
    """
    Dedup / novelty filter operating only on canonical molecule semantics.

    Important semantics:
    - `rdkit_valid` is the canonical legality field
    - invalid molecules are NOT dropped when `enable_filtering=False`
    - this allows downstream reward code to penalize invalid samples explicitly
    """

    @staticmethod
    def _normalize_smiles_value(value) -> Optional[str]:
        """Normalize DataProto non-tensor metadata into `str | None`.

        `smiles` may arrive as a wrapped non-tensor object depending on the
        underlying batch container implementation. Novelty and dedup logic need
        stable, hashable Python strings.
        """
        if value is None:
            return None
        if isinstance(value, str):
            return value

        raw_value = getattr(value, "_value", value)
        if raw_value is None:
            return None
        if isinstance(raw_value, str):
            return raw_value

        try:
            if hasattr(raw_value, "item"):
                raw_value = raw_value.item()
        except Exception:
            pass

        if raw_value is None:
            return None
        return str(raw_value)

    def __init__(
        self,
        dataset_info: Optional[dict] = None,
        dataset_smiles_list: Optional[List[str]] = None,
        enable_filtering: bool = DEFAULT_ENABLE_FILTERING,
        enable_penalty: bool = DEFAULT_ENABLE_PENALTY,
        novelty_penalty_coef: float = NOVELTY_PENALTY_COEF,
    ):
        self.dataset_info = dataset_info or dataset_params['crossdock']
        self.dataset_smiles_list = set(dataset_smiles_list) if dataset_smiles_list else set()
        self.enable_filtering = enable_filtering
        self.enable_penalty = enable_penalty
        self.novelty_penalty_coef = novelty_penalty_coef

    def filter(self, data: "DataProto") -> tuple["DataProto", float, float]:
        """
        Filter molecules and optionally apply novelty penalty.
        """
        ntb = data.non_tensor_batch if hasattr(data, "non_tensor_batch") else None
        src = ntb if isinstance(ntb, dict) else data.batch
        if "rdkit_valid" not in src or "smiles" not in src:
            data = annotate_batch_molecule_semantics(data, dataset_info=self.dataset_info)
            ntb = data.non_tensor_batch if hasattr(data, "non_tensor_batch") else None
            src = ntb if isinstance(ntb, dict) else data.batch

        valid_flags = list(src.get("rdkit_valid", []))
        raw_smiles = list(src.get("smiles", []))
        all_smiles = [self._normalize_smiles_value(smiles) for smiles in raw_smiles]

        smiles_indices = {}
        none_idx = []
        novelty_penalty = []

        for idx, smiles in enumerate(all_smiles):
            novelty_penalty.append(-1 if smiles is not None and smiles in self.dataset_smiles_list else 0)
            if smiles is not None:
                smiles_indices.setdefault(smiles, []).append(idx)
            else:
                none_idx.append(idx)

        novelty_penalty_ratio = 1 + sum(novelty_penalty) / len(novelty_penalty) if novelty_penalty else 1.0

        keep_mask = [False] * len(all_smiles)
        if self.enable_filtering:
            for indices in smiles_indices.values():
                if indices:
                    keep_mask[random.choice(indices)] = True
            for idx in none_idx:
                keep_mask[idx] = True
            indices_to_keep = np.where(keep_mask)[0]
        else:
            indices_to_keep = np.arange(len(all_smiles))

        if self.enable_penalty and 'reward_total' in data.batch:
            novelty_penalty_tensor = torch.tensor(
                novelty_penalty,
                device=data.batch["reward_total"].device,
                dtype=data.batch["reward_total"].dtype,
            )
            data.batch["reward_total"] = data.batch["reward_total"] + novelty_penalty_tensor * self.novelty_penalty_coef

        filtered_data = data.select_idxs(indices_to_keep) if self.enable_filtering else data

        total_samples = len(all_smiles)
        kept_samples = len(indices_to_keep)
        filtering_ratio = kept_samples / total_samples if total_samples > 0 else 0.0

        filtered_rdkit_valid = [valid_flags[i] for i in indices_to_keep]
        filtered_data.batch['smiles'] = [all_smiles[i] for i in indices_to_keep]
        filtered_data.batch['rdkit_valid'] = filtered_rdkit_valid
        # Bug4 Fix-5: 'valid' kept as legacy alias for downstream compatibility only.
        # All internal logic reads from 'rdkit_valid'. Do not use 'valid' as primary field.
        # This alias will be removed once all downstream consumers are migrated.
        filtered_data.batch['valid'] = filtered_rdkit_valid

        return filtered_data, filtering_ratio, novelty_penalty_ratio

    def get_validity_ratio(self, data: "DataProto") -> float:
        """Compute validity ratio from canonical semantics."""
        if "rdkit_valid" not in data.batch:
            data = annotate_batch_molecule_semantics(data, dataset_info=self.dataset_info)
        valid_flags = list(data.batch.get("rdkit_valid", []))
        total = len(valid_flags)
        if total == 0:
            return 0.0
        return float(sum(int(v) for v in valid_flags)) / float(total)

    def __repr__(self) -> str:
        return (
            f"DiffSBDDFilter("
            f"enable_filtering={self.enable_filtering}, "
            f"enable_penalty={self.enable_penalty})"
        )
