"""QEDReward: reward function based on Quantitative Estimate of Drug-likeness.

Single responsibility: compute QED for each molecule in the batch.
Range: [0, 1], higher is better.
No normalization needed (QED is already in [0, 1]).

Usage:
    from ddpo.rewards.qed_reward import QEDReward
    reward = QEDReward(failure_penalty=-1.0)
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
    build_canonical_rdmol_from_sample,
)
from ddpo.utils.suppress_warnings import redirect_stderr_to_log

if TYPE_CHECKING:
    from ddpo.runtime.protocol import DataProto


DEFAULT_FAILURE_PENALTY: float = -1.0


class QEDReward(BaseReward):
    """Reward function based on QED (Quantitative Estimate of Drug-likeness).

    QED ranges from 0 (not drug-like) to 1 (very drug-like).
    Failed molecules receive failure_penalty.

    Writes to batch:
        'rewards'         [B]  — QED score or failure_penalty
        'qed_scores'      [B]  — QED score (0.0 for invalid molecules)
        'qed_success'     [B]  — 1 if QED computed, 0 otherwise
    """

    def __init__(
        self,
        failure_penalty: float = DEFAULT_FAILURE_PENALTY,
        dataset_info: Optional[dict] = None,
    ) -> None:
        """
        Args:
            failure_penalty: Reward assigned to molecules that fail
                             canonical build or QED computation.
            dataset_info: Dataset info dict (defaults to crossdock).
        """
        self.failure_penalty = failure_penalty
        self.dataset_info = dataset_info or dataset_params["crossdock"]
        self._mol_props = MoleculeProperties()

    def calculate_rewards(self, data: "DataProto") -> "DataProto":
        """Calculate QED rewards for a batch of molecules.

        Expects batch['rdkit_valid'] to be set by annotate_batch_molecule_semantics()
        upstream. Molecules with rdkit_valid==0 immediately receive failure_penalty.

        Args:
            data: DataProto with pre-annotated molecule semantics.

        Returns:
            DataProto with batch['rewards'], batch['qed_scores'],
            batch['qed_success'] added.
        """
        batch = data.batch
        x = batch["x"]
        h = batch["categorical"]
        nodesxsample = batch["nodesxsample"]
        rdkit_valid = batch.get("rdkit_valid")
        failure_type = batch.get("failure_type", None)

        if rdkit_valid is None:
            raise ValueError(
                "[QEDReward] rdkit_valid missing from batch. "
                "annotate_batch_molecule_semantics() must be called before calculate_rewards()."
            )

        batch_size = x.shape[0]
        device = x.device

        rewards = torch.full((batch_size,), self.failure_penalty, device=device)
        qed_scores = torch.zeros(batch_size, device=device)
        qed_success = torch.zeros(batch_size, dtype=torch.int64, device=device)

        for i in range(batch_size):
            if int(rdkit_valid[i]) == 0:
                continue

            try:
                mol, _, _, is_valid, build_failure = build_canonical_rdmol_from_sample(
                    x=x,
                    categorical=h,
                    nodesxsample=nodesxsample,
                    sample_idx=i,
                    dataset_info=self.dataset_info,
                )
                if not is_valid or mol is None:
                    if failure_type is not None and str(failure_type[i]) == FAILURE_SUCCESS:
                        failure_type[i] = build_failure or FAILURE_SANITIZE
                    continue

                with redirect_stderr_to_log():
                    qed = self._mol_props.calculate_qed(mol)

                qed_scores[i] = qed
                rewards[i] = qed
                qed_success[i] = 1
                if failure_type is not None:
                    failure_type[i] = FAILURE_SUCCESS

            except Exception as exc:
                warnings.warn(f"[QEDReward] molecule {i}: {exc}")
                if failure_type is not None and str(failure_type[i]) == FAILURE_SUCCESS:
                    failure_type[i] = FAILURE_DRUGLIKELINESS

        data.batch["rewards"] = rewards
        data.batch["qed_scores"] = qed_scores
        data.batch["qed_success"] = qed_success
        if failure_type is not None:
            data.batch["failure_type"] = failure_type

        return data

    def __repr__(self) -> str:
        return f"QEDReward(failure_penalty={self.failure_penalty})"
