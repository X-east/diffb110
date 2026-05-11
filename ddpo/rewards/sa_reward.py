"""SAReward: reward function based on Synthetic Accessibility score.

Raw SA from sascorer ranges [1, 10], lower = easier synthesis.
We write both raw and normalised forms:
    'sa_raw'    [B]  -- raw SA score (1-10, 0.0 for invalid)
    'sa_reward' [B]  -- normalised SA (10-SA)/9 in [0, 1], higher = easier

Usage:
    from ddpo.rewards.sa_reward import SAReward
    reward = SAReward(failure_penalty=-1.0)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "DiffSBDD"))

from analysis.SA_Score.sascorer import calculateScore as _sascorer
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


class SAReward(BaseReward):
    """Reward function based on SA (Synthetic Accessibility) score.

    SA ranges from 1 (easy to synthesise) to 10 (very hard).
    Reward = (10 - raw_SA) / 9, mapping to [0, 1] where higher = easier synthesis.

    Writes to batch:
        'sa_raw'    [B]  -- raw SA score (1-10, 0.0 for invalid molecules)
        'sa_reward' [B]  -- normalised SA reward or failure_penalty
        'sa_success' [B]  -- 1 if SA computed, 0 otherwise
    """

    def __init__(
        self,
        failure_penalty: float = DEFAULT_FAILURE_PENALTY,
        dataset_info: Optional[dict] = None,
    ) -> None:
        """
        Args:
            failure_penalty: Reward assigned to molecules that fail
                             canonical build or SA computation.
            dataset_info:    Dataset info dict (defaults to crossdock).
        """
        self.failure_penalty = failure_penalty
        self.dataset_info = dataset_info or dataset_params["crossdock"]

    def calculate_rewards(self, data: "DataProto") -> "DataProto":
        """Calculate SA rewards for a batch of molecules.

        Expects batch['rdkit_valid'] to be set by annotate_batch_molecule_semantics()
        upstream. Molecules with rdkit_valid==0 immediately receive failure_penalty.

        Args:
            data: DataProto with pre-annotated molecule semantics.

        Returns:
            DataProto with batch['sa_raw'], batch['sa_reward'],
            batch['sa_success'] added.
        """
        batch = data.batch
        x = batch["x"]
        h = batch["categorical"]
        nodesxsample = batch["nodesxsample"]
        # Read Python-native metadata from non_tensor_batch to avoid TensorDict wrapping.
        ntb = data.non_tensor_batch if hasattr(data, "non_tensor_batch") else None
        rdkit_valid = ntb.get("rdkit_valid") if ntb else batch.get("rdkit_valid")
        failure_type = ntb.get("failure_type") if ntb else batch.get("failure_type")

        if rdkit_valid is None:
            raise ValueError(
                "[SAReward] rdkit_valid missing from batch. "
                "annotate_batch_molecule_semantics() must be called before calculate_rewards()."
            )

        batch_size = x.shape[0]
        device = x.device

        sa_reward = torch.full((batch_size,), self.failure_penalty, device=device)
        sa_raw = torch.zeros(batch_size, device=device)
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
                    raw_val = _sascorer(mol)           # raw SA [1, 10], lower = easier

                # Write both raw and normalised forms
                sa_raw[i] = round(raw_val, 2)
                sa_reward[i] = round((_SA_MAX - raw_val) / (_SA_MAX - _SA_MIN), 2)
                sa_success[i] = 1
                if failure_type is not None:
                    failure_type[i] = FAILURE_SUCCESS

            except Exception as exc:
                warnings.warn(f"[SAReward] molecule {i}: {exc}")
                if failure_type is not None and str(failure_type[i]) == FAILURE_SUCCESS:
                    failure_type[i] = FAILURE_DRUGLIKELINESS

        data.batch["sa_raw"] = sa_raw
        data.batch["sa_reward"] = sa_reward
        data.batch["sa_success"] = sa_success
        if failure_type is not None:
            data.batch["failure_type"] = failure_type

        return data

    def __repr__(self) -> str:
        return f"SAReward(failure_penalty={self.failure_penalty})"
