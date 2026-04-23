"""SAReward: reward function based on Synthetic Accessibility score.

Single responsibility: compute SA score for each molecule in the batch.
Raw SA range: [1, 10], lower is better (1 = easiest to synthesise).
Normalized to [0, 1]: norm_sa = (10 - SA) / 9  (higher = more accessible).

Usage:
    from ddpo.rewards.sa_reward import SAReward
    reward = SAReward(failure_penalty=-1.0, normalize=True)
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
_SA_MIN: float = 1.0
_SA_MAX: float = 10.0


def _normalize_sa(sa_raw: float) -> float:
    """Map SA score [1, 10] to reward [0, 1] (higher = more synthesisable)."""
    return (_SA_MAX - sa_raw) / (_SA_MAX - _SA_MIN)


class SAReward(BaseReward):
    """Reward function based on SA (Synthetic Accessibility) score.

    SA ranges from 1 (easy to synthesise) to 10 (very hard).
    When normalize=True (default), reward = (10 - SA) / 9 in [0, 1].
    When normalize=False, reward = raw SA score.

    Failed molecules receive failure_penalty.

    Writes to batch:
        'rewards'    [B]  -- normalised SA reward or failure_penalty
        'sa_scores'  [B]  -- raw SA score (0.0 for invalid molecules)
        'sa_success' [B]  -- 1 if SA computed, 0 otherwise
    """

    def __init__(
        self,
        failure_penalty: float = DEFAULT_FAILURE_PENALTY,
        normalize: bool = True,
        dataset_info: Optional[dict] = None,
    ) -> None:
        """
        Args:
            failure_penalty: Reward assigned to molecules that fail
                             canonical build or SA computation.
            normalize:       If True (default), return (10-SA)/9 in [0,1].
                             If False, return raw SA score.
            dataset_info:    Dataset info dict (defaults to crossdock).
        """
        self.failure_penalty = failure_penalty
        self.normalize = normalize
        self.dataset_info = dataset_info or dataset_params["crossdock"]
        self._mol_props = MoleculeProperties()

    def calculate_rewards(self, data: "DataProto") -> "DataProto":
        """Calculate SA rewards for a batch of molecules.

        Expects batch['rdkit_valid'] to be set by annotate_batch_molecule_semantics()
        upstream. Molecules with rdkit_valid==0 immediately receive failure_penalty.

        Args:
            data: DataProto with pre-annotated molecule semantics.

        Returns:
            DataProto with batch['rewards'], batch['sa_scores'],
            batch['sa_success'] added.
        """
        batch = data.batch
        x = batch["x"]
        h = batch["categorical"]
        nodesxsample = batch["nodesxsample"]
        rdkit_valid = batch.get("rdkit_valid")
        failure_type = batch.get("failure_type", None)

        if rdkit_valid is None:
            raise ValueError(
                "[SAReward] rdkit_valid missing from batch. "
                "annotate_batch_molecule_semantics() must be called before calculate_rewards()."
            )

        batch_size = x.shape[0]
        device = x.device

        rewards = torch.full((batch_size,), self.failure_penalty, device=device)
        sa_scores = torch.zeros(batch_size, device=device)
        sa_success = torch.zeros(batch_size, dtype=torch.int64, device=device)

        for i in range(batch_size):
            if int(rdkit_valid[i]) == 0:
                continue

            try:
                # Reuse canonical mol from mainline annotate_batch_molecule_semantics()
                # Stored in non_tensor_batch to avoid TensorDict wrapping RDKit objects.
                rdmols = data.non_tensor_batch.get("rdmols") \
                    if hasattr(data, "non_tensor_batch") and data.non_tensor_batch else None
                if rdmols is not None and i < len(rdmols) and rdmols[i] is not None:
                    mol = rdmols[i]
                    is_valid = True
                    build_failure = None
                else:
                    # Fallback: build locally (backward compat)
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
                    sa_raw = self._mol_props.calculate_sa(mol)

                sa_scores[i] = sa_raw
                rewards[i] = _normalize_sa(sa_raw) if self.normalize else sa_raw
                sa_success[i] = 1
                if failure_type is not None:
                    failure_type[i] = FAILURE_SUCCESS

            except Exception as exc:
                warnings.warn(f"[SAReward] molecule {i}: {exc}")
                if failure_type is not None and str(failure_type[i]) == FAILURE_SUCCESS:
                    failure_type[i] = FAILURE_DRUGLIKELINESS

        data.batch["rewards"] = rewards
        data.batch["sa_scores"] = sa_scores
        data.batch["sa_success"] = sa_success
        if failure_type is not None:
            data.batch["failure_type"] = failure_type

        return data

    def __repr__(self) -> str:
        return (
            f"SAReward(failure_penalty={self.failure_penalty}, "
            f"normalize={self.normalize})"
        )
