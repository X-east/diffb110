"""
ChEMBLClassifierReward: uses the trained ChEMBL classifier as a reward signal.

Featurizes each generated molecule via MolSkill's morgan_count_rdkit_2d
(2048 Morgan count FP + 172 RDKit 2D descriptors), runs forward pass through
the 3-layer MLP, and outputs P(ChEMBL-like) ∈ [0,1] as the reward signal.

The classifier penalizes carbon-collapse PAHs, all-sulfur garbage,
and extreme outliers while rewarding drug-like chemical space.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from ddpo.runtime.protocol import DataProto

# ── Add molskill to Python path ──────────────────────────────────────────
_MOLSKILL_ROOT = Path(__file__).resolve().parent.parent.parent / "reference project" / "molskill"
if str(_MOLSKILL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MOLSKILL_ROOT))

from molskill.data.featurizers import get_featurizer, Featurizer  # noqa: E402

# ── Add project root ─────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from chembl_classifier.model import ChemblClassifier  # noqa: E402
from ddpo.rewards.interfaces.base_reward import BaseReward  # noqa: E402
from ddpo.field_registry import FieldSpec  # noqa: E402

logger = logging.getLogger(__name__)


class ChemblClassifierReward(BaseReward):
    """Reward signal from a ChEMBL binary classifier.

    The classifier was trained on:
      - Positive: 100k random ChEMBL molecules
      - Negative: 18K property-rule negatives (all-carbon, too-hydrophilic, etc.)

    Reward mapping (threshold-based, capped):
      - prob >= good_threshold  → positive reward, plateau at cap_high
      - prob <  good_threshold  → penalty, floor at cap_low
      When bad_threshold == good_threshold (default), there is no neutral zone.

    Args:
        model_path: Path to the trained classifier .pt file.
        failure_penalty: Reward for invalid molecules (rdkit_valid=0).
        good_threshold: P below this gets penalty, above gets reward.
        bad_threshold: Penalty threshold (default = good_threshold).
        cap_high: P above which reward plateaus at max_reward.
        cap_low: P below which penalty floors at penalty.
        max_reward: Maximum positive reward (at prob >= cap_high).
        penalty: Maximum penalty (at prob <= cap_low).
        device: Inference device. Default auto-detect.
    """

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
    ) -> None:
        super().__init__()

        if model_path is None:
            model_path = str(
                _PROJECT_ROOT / "chembl_classifier" / "models" / "chembl_classifier.pt"
            )

        if device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        self._failure_penalty = float(failure_penalty)
        self._good_threshold = float(good_threshold)
        self._bad_threshold = float(bad_threshold) if bad_threshold is not None else self._good_threshold
        self._cap_high = float(cap_high)
        self._cap_low = float(cap_low)
        self._max_reward = float(max_reward)
        self._penalty = float(penalty)
        self._model_path = model_path

        self._featurizer: Featurizer | None = None
        self._model: ChemblClassifier | None = None

        if not Path(model_path).exists():
            logger.warning(
                f"Classifier model not found at {model_path}. "
                f"Reward will use failure_penalty={failure_penalty} for all molecules."
            )
        else:
            self._ensure_model_loaded()

    # ── Lazy loading ──────────────────────────────────────────────────────

    def _ensure_model_loaded(self) -> None:
        """Load featurizer + model on first use. Safe to call multiple times."""
        if self._featurizer is None:
            logger.info("Initializing ChemblClassifierReward featurizer...")
            self._featurizer = get_featurizer("morgan_count_rdkit_2d")

        if self._model is None and Path(self._model_path).exists():
            logger.info(f"Loading ChEMBL classifier from {self._model_path}")
            self._model = ChemblClassifier.from_checkpoint(
                self._model_path, device=self._device
            )

    # ── Reward calculation ────────────────────────────────────────────────

    def calculate_rewards(self, data: DataProto) -> DataProto:
        """Compute P(ChEMBL-like) for each valid molecule.

        Args:
            data: DataProto with non_tensor_batch containing 'rdmols' and 'rdkit_valid'.

        Returns:
            DataProto with batch['chembl_classifier_reward'] added [B].
        """
        self._ensure_model_loaded()

        batch = data.batch
        ntb = data.non_tensor_batch
        rdkit_valid = ntb.get("rdkit_valid")
        rdmols = ntb.get("rdmols")

        batch_size = batch["x"].shape[0]
        device = batch["x"].device

        rewards = torch.full(
            (batch_size,), self._failure_penalty, dtype=torch.float32, device=device
        )

        # Fast path: no model available
        if self._model is None:
            data.batch["chembl_classifier_reward"] = rewards
            return data

        # Featurize valid molecules
        features_list: list[np.ndarray] = []
        valid_indices: list[int] = []

        for i in range(batch_size):
            if rdkit_valid is not None and int(rdkit_valid[i]) == 0:
                continue
            mol = rdmols[i]
            if mol is None:
                continue
            try:
                feat = self._featurizer.get_feat(mol)
                if feat is not None:
                    features_list.append(feat)
                    valid_indices.append(i)
            except Exception:
                logger.debug(f"Featurization failed for molecule {i}", exc_info=True)

        if not features_list:
            logger.warning("No valid molecules in batch for ChEMBL classifier")
            data.batch["chembl_classifier_reward"] = rewards
            return data

        # Batch inference
        X = torch.tensor(np.stack(features_list, axis=0), dtype=torch.float32).to(
            self._device
        )
        probs = self._model.predict_proba(X).flatten()

        # Threshold-based reward mapping with caps:
        #   prob >= good_t → positive, [0, max_reward], plateau at cap_high
        #   prob <  bad_t  → penalty, [penalty, 0], floor at cap_low
        probs = probs.to(device)
        good_t = self._good_threshold
        bad_t = self._bad_threshold
        cap_h = self._cap_high
        cap_l = self._cap_low

        high_mask = probs >= good_t
        low_mask = probs < bad_t
        mid_mask = ~(high_mask | low_mask)

        mapped = torch.zeros_like(probs)
        # High: linear [good_t, cap_h] → [0, max_reward], then plateau
        scale_h = max(cap_h - good_t, 1e-8)
        high_scaled = torch.clamp((probs[high_mask] - good_t) / scale_h, 0.0, 1.0)
        mapped[high_mask] = self._max_reward * high_scaled
        # Low: linear [cap_l, bad_t] → [penalty, 0], then floor
        scale_l = max(bad_t - cap_l, 1e-8)
        low_scaled = torch.clamp((bad_t - probs[low_mask]) / scale_l, 0.0, 1.0)
        mapped[low_mask] = self._penalty * low_scaled
        # Mid: neutral = 0.0 (already zeros)

        for idx, reward_val in zip(valid_indices, mapped):
            rewards[idx] = reward_val

        data.batch["chembl_classifier_reward"] = rewards

        return data

    # ── Field declaration ─────────────────────────────────────────────────

    def field_specs(self) -> list[FieldSpec]:
        """Declare molecule-level reward field."""
        return [
            FieldSpec(
                name="chembl_classifier_reward",
                scope="molecule",
                source="batch",
                reduce="none",
                order=600,
            )
        ]

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def numel(self) -> int:
        """Number of parameters in the underlying classifier."""
        if self._model is not None:
            return self._model.n_params
        return 0
