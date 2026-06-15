"""D-MPNN classifier reward for ChEMBL-like molecule quality."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from ddpo.field_registry import FieldSpec
from ddpo.rewards.interfaces.base_reward import BaseReward

if TYPE_CHECKING:
    from ddpo.runtime.protocol import DataProto
    from chembl_classifier.dmpnn.predictor import DMPNNClassifierPredictor

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class ChemblDMPNNReward(BaseReward):
    """Reward signal from a trained D-MPNN binary classifier."""

    def __init__(
        self,
        model_path: str | None = None,
        failure_penalty: float = -1.0,
        good_threshold: float = 0.5,
        bad_threshold: float | None = None,
        cap_high: float = 0.8,
        cap_low: float = 0.2,
        max_reward: float = 1.0,
        penalty: float = -1.0,
        device: str | None = None,
        batch_size: int = 128,
        cache_smiles: bool = True,
    ) -> None:
        super().__init__()
        if model_path is None:
            model_path = str(_PROJECT_ROOT / "chembl_classifier" / "models" / "chembl_dmpnn_v0_1_scaffold_strongpoly.pt")
        self._model_path = str(model_path)
        self._failure_penalty = float(failure_penalty)
        self._good_threshold = float(good_threshold)
        self._bad_threshold = float(bad_threshold) if bad_threshold is not None else self._good_threshold
        self._cap_high = float(cap_high)
        self._cap_low = float(cap_low)
        self._max_reward = float(max_reward)
        self._penalty = float(penalty)
        self._device = device
        self._batch_size = int(batch_size)
        self._cache_smiles = bool(cache_smiles)
        self._predictor: DMPNNClassifierPredictor | None = None

        if self._batch_size <= 0:
            raise ValueError("ChemblDMPNNReward batch_size must be positive")
        if not Path(self._model_path).exists():
            logger.warning(
                "D-MPNN classifier model not found at %s. Reward will use failure_penalty=%.3f.",
                self._model_path,
                self._failure_penalty,
            )

    def calculate_rewards(self, data: "DataProto") -> "DataProto":
        """Compute classifier probabilities and mapped rewards for a DataProto."""
        batch = data.batch
        ntb = data.non_tensor_batch
        batch_size = batch["x"].shape[0]
        output_device = batch["x"].device

        rewards = torch.full((batch_size,), self._failure_penalty, dtype=torch.float32, device=output_device)
        probs_tensor = torch.zeros((batch_size,), dtype=torch.float32, device=output_device)
        if not Path(self._model_path).exists():
            data.batch["chembl_dmpnn_reward"] = rewards
            data.batch["chembl_dmpnn_prob"] = probs_tensor
            return data

        self._ensure_predictor_loaded()
        if self._predictor is None:
            data.batch["chembl_dmpnn_reward"] = rewards
            data.batch["chembl_dmpnn_prob"] = probs_tensor
            return data

        rdkit_valid = ntb.get("rdkit_valid")
        rdmols = ntb.get("rdmols")
        if rdmols is None:
            logger.warning("D-MPNN reward received no rdmols field")
            data.batch["chembl_dmpnn_reward"] = rewards
            data.batch["chembl_dmpnn_prob"] = probs_tensor
            return data

        candidate_mols: list[object] = []
        candidate_indices: list[int] = []
        for index in range(batch_size):
            if rdkit_valid is not None and int(rdkit_valid[index]) == 0:
                continue
            mol = rdmols[index]
            if mol is None:
                continue
            candidate_mols.append(mol)
            candidate_indices.append(index)

        if not candidate_mols:
            data.batch["chembl_dmpnn_reward"] = rewards
            data.batch["chembl_dmpnn_prob"] = probs_tensor
            return data

        probabilities = self._predictor.predict_proba_mols(candidate_mols, batch_size=self._batch_size)
        valid_probs: list[float] = []
        valid_indices: list[int] = []
        for index, probability in zip(candidate_indices, probabilities):
            if probability is None:
                continue
            valid_indices.append(index)
            valid_probs.append(float(probability))

        if valid_probs:
            probs = torch.tensor(valid_probs, dtype=torch.float32, device=output_device)
            mapped = self._map_probabilities(probs)
            for index, prob, reward in zip(valid_indices, probs, mapped):
                probs_tensor[index] = prob
                rewards[index] = reward

        data.batch["chembl_dmpnn_prob"] = probs_tensor
        data.batch["chembl_dmpnn_reward"] = rewards
        return data

    def field_specs(self) -> list[FieldSpec]:
        """Declare molecule and epoch fields for recording."""
        return [
            FieldSpec(
                name="chembl_dmpnn_prob",
                scope="molecule",
                source="batch",
                reduce="none",
                default=0.0,
                order=610,
            ),
            FieldSpec(
                name="chembl_dmpnn_reward",
                scope="molecule",
                source="batch",
                reduce="none",
                order=611,
            ),
            FieldSpec(
                name="avg_chembl_dmpnn_prob",
                scope="epoch",
                source="batch",
                source_key="chembl_dmpnn_prob",
                reduce="valid_mean",
                order=610,
            ),
            FieldSpec(
                name="avg_chembl_dmpnn_reward",
                scope="epoch",
                source="batch",
                source_key="chembl_dmpnn_reward",
                reduce="generated_mean",
                order=611,
            ),
        ]

    @property
    def numel(self) -> int:
        """Number of parameters in the loaded D-MPNN model."""
        return self._predictor.n_params if self._predictor is not None else 0

    def _ensure_predictor_loaded(self) -> None:
        if self._predictor is not None:
            return
        from chembl_classifier.dmpnn.predictor import DMPNNClassifierPredictor

        self._predictor = DMPNNClassifierPredictor.from_checkpoint(
            self._model_path,
            device=self._device,
            cache_smiles=self._cache_smiles,
        )

    def _map_probabilities(self, probs: torch.Tensor) -> torch.Tensor:
        good_t = self._good_threshold
        bad_t = self._bad_threshold
        cap_h = self._cap_high
        cap_l = self._cap_low

        high_mask = probs >= good_t
        low_mask = probs < bad_t
        mapped = torch.zeros_like(probs)

        scale_h = max(cap_h - good_t, 1e-8)
        high_scaled = torch.clamp((probs[high_mask] - good_t) / scale_h, 0.0, 1.0)
        mapped[high_mask] = self._max_reward * high_scaled

        scale_l = max(bad_t - cap_l, 1e-8)
        low_scaled = torch.clamp((bad_t - probs[low_mask]) / scale_l, 0.0, 1.0)
        mapped[low_mask] = self._penalty * low_scaled
        return mapped
