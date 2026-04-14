"""DruglikenessReward: composite of QED + SA rewards.

.. deprecated::
    DruglikenessReward is a legacy combined reward.
    Prefer using QEDReward and SAReward independently via factory:
        reward.enabled: [qed, sa]
    This class is kept only for backward compatibility with old configs
    that use ``reward.enabled: [druglikeness]``.
    It will be removed in a future cleanup pass.

Responsibility:
    Combine QEDReward and SAReward into a single weighted reward.
    This is a thin composition layer — all actual computation
    lives in qed_reward.py and sa_reward.py.

If you want only QED or only SA, use QEDReward / SAReward directly.

Usage:
    from ddpo.rewards.druglikeness import DruglikenessReward
    reward = DruglikenessReward(qed_weight=0.5, sa_weight=0.5)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "DiffSBDD"))

from analysis.metrics import MoleculeProperties
from constants import dataset_params

from ddpo.rewards.interfaces import BaseReward
from ddpo.utils.molecule_semantics import (
    FAILURE_DRUGLIKELINESS,
    FAILURE_SANITIZE,
    FAILURE_SUCCESS,
)
from ddpo.utils.suppress_warnings import redirect_stderr_to_log

if TYPE_CHECKING:
    from ddpo.runtime.protocol import DataProto


DEFAULT_QED_WEIGHT: float = 0.5
DEFAULT_SA_WEIGHT: float = 0.5
DEFAULT_FAILURE_PENALTY: float = -1.0


class DruglikenessReward(BaseReward):
    """Weighted combination of QED and SA rewards.

    reward = qed_weight * QED + sa_weight * norm_SA
    where norm_SA = (10 - SA) / 9

    Failed molecules receive failure_penalty.

    Writes to batch:
        'rewards'              [B]  -- combined reward or failure_penalty
        'qed_scores'           [B]  -- raw QED (0.0 for invalid)
        'sa_scores'            [B]  -- raw SA  (0.0 for invalid)
        'druglikeness_success' [B]  -- 1 if both computed, 0 otherwise
    """

    def __init__(
        self,
        qed_weight: float = DEFAULT_QED_WEIGHT,
        sa_weight: float = DEFAULT_SA_WEIGHT,
        failure_penalty: float = DEFAULT_FAILURE_PENALTY,
        dataset_info: Optional[dict] = None,
    ) -> None:
        """
        Args:
            qed_weight:      Weight for QED component (default 0.5).
            sa_weight:       Weight for normalised SA component (default 0.5).
            failure_penalty: Reward for molecules that fail build or scoring.
            dataset_info:    Dataset info dict (defaults to crossdock).
        """
        self.qed_weight = qed_weight
        self.sa_weight = sa_weight
        self.failure_penalty = failure_penalty
        self.dataset_info = dataset_info or dataset_params["crossdock"]
        self._mol_props = MoleculeProperties()

    def calculate_rewards(self, data: "DataProto") -> "DataProto":
        """Calculate combined QED+SA rewards for a batch of molecules.

        Expects batch['rdkit_valid'] set by annotate_batch_molecule_semantics().
        Molecules with rdkit_valid==0 immediately receive failure_penalty.

        Args:
            data: DataProto with pre-annotated molecule semantics.

        Returns:
            DataProto with batch['rewards'], batch['qed_scores'],
            batch['sa_scores'], batch['druglikeness_success'] added.
        """
        batch = data.batch
        rdkit_valid = batch.get("rdkit_valid")
        failure_type = batch.get("failure_type", None)

        if rdkit_valid is None:
            raise ValueError(
                "[DruglikenessReward] rdkit_valid missing from batch. "
                "annotate_batch_molecule_semantics() must be called before calculate_rewards()."
            )

        # Reuse pre-built mol objects if available (set by annotate_batch_molecule_semantics).
        # Falls back to None-list if somehow missing, which will be caught per-molecule.
        rdmols = batch.get("rdmols", [])

        batch_size = len(rdkit_valid)
        # Derive device from rewards tensor or fallback to cpu
        _ref = batch.get("x")
        device = _ref.device if _ref is not None and isinstance(_ref, torch.Tensor) else torch.device("cpu")

        rewards = torch.full((batch_size,), self.failure_penalty, device=device)
        qed_scores = torch.zeros(batch_size, device=device)
        sa_scores = torch.zeros(batch_size, device=device)
        druglikeness_success = torch.zeros(batch_size, dtype=torch.int64, device=device)

        for i in range(batch_size):
            if int(rdkit_valid[i]) == 0:
                continue

            mol = rdmols[i] if i < len(rdmols) else None
            if mol is None:
                # rdmols slot missing despite rdkit_valid==1 — treat as build failure
                if failure_type is not None and str(failure_type[i]) == FAILURE_SUCCESS:
                    failure_type[i] = FAILURE_SANITIZE
                continue

            try:
                with redirect_stderr_to_log():
                    qed = self._mol_props.calculate_qed(mol)
                    sa_raw = self._mol_props.calculate_sa(mol)

                norm_sa = (10.0 - sa_raw) / 9.0
                qed_scores[i] = qed
                sa_scores[i] = sa_raw
                rewards[i] = self.qed_weight * qed + self.sa_weight * norm_sa
                druglikeness_success[i] = 1
                if failure_type is not None:
                    failure_type[i] = FAILURE_SUCCESS

            except Exception as exc:
                warnings.warn(f"[DruglikenessReward] molecule {i}: {exc}")
                if failure_type is not None and str(failure_type[i]) == FAILURE_SUCCESS:
                    failure_type[i] = FAILURE_DRUGLIKELINESS

        data.batch["rewards"] = rewards
        data.batch["qed_scores"] = qed_scores
        data.batch["sa_scores"] = sa_scores
        data.batch["druglikeness_success"] = druglikeness_success
        if failure_type is not None:
            data.batch["failure_type"] = failure_type

        return data

    def __repr__(self) -> str:
        return (
            f"DruglikenessReward("
            f"qed_weight={self.qed_weight}, "
            f"sa_weight={self.sa_weight})"
        )
