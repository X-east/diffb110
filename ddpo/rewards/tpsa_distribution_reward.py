"""TPSA distribution reward based on ChEMBL bin-proportion matching."""

from __future__ import annotations

import csv
import gzip
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch
from rdkit.Chem import Descriptors

from ddpo.field_registry import FieldSpec
from ddpo.rewards.interfaces import BaseReward

if TYPE_CHECKING:
    from ddpo.runtime.protocol import DataProto

logger = logging.getLogger(__name__)

DEFAULT_FAILURE_PENALTY: float = -1.0
DEFAULT_MIN_REWARD: float = -2.0
DEFAULT_MAX_REWARD: float = 2.0
DEFAULT_SCALE: float = 12.0
DEFAULT_EMA_ALPHA: float = 0.0
DEFAULT_REWARD_PREFIX: str = "tpsa_distribution"
DEFAULT_CHEMBL_CSV: str = "data/chembl/chembl_36_properties.csv.gz"
CHEMBL_TPSA_COLUMN: str = "tpsa"

DEFAULT_TPSA_BINS: list[float] = [
    -float("inf"),
    20.0,
    40.0,
    60.0,
    80.0,
    100.0,
    120.0,
    160.0,
    float("inf"),
]

BIN_LABELS: list[str] = [
    "(-inf,20)",
    "[20,40)",
    "[40,60)",
    "[60,80)",
    "[80,100)",
    "[100,120)",
    "[120,160)",
    "[160,+inf)",
]


class TPSADistributionReward(BaseReward):
    """Reward molecules whose TPSA bin is under-represented versus ChEMBL.

    This is intentionally distributional instead of monotonic. A monotonic TPSA
    reward would encourage very polar molecules; this reward only encourages
    under-sampled TPSA bins and penalizes over-sampled bins.
    """

    def __init__(
        self,
        chembl_csv: str = DEFAULT_CHEMBL_CSV,
        tpsa_bins: Sequence[float] | None = None,
        failure_penalty: float = DEFAULT_FAILURE_PENALTY,
        min_reward: float = DEFAULT_MIN_REWARD,
        max_reward: float = DEFAULT_MAX_REWARD,
        scale: float = DEFAULT_SCALE,
        ema_alpha: float = DEFAULT_EMA_ALPHA,
        reward_prefix: str = DEFAULT_REWARD_PREFIX,
    ) -> None:
        super().__init__()
        if min_reward > max_reward:
            raise ValueError("min_reward must be <= max_reward")
        if scale <= 0:
            raise ValueError("scale must be positive")
        if not 0.0 <= ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in [0, 1]")

        self.chembl_csv = chembl_csv
        self.failure_penalty = float(failure_penalty)
        self.min_reward = float(min_reward)
        self.max_reward = float(max_reward)
        self.scale = float(scale)
        self.ema_alpha = float(ema_alpha)
        self.reward_prefix = reward_prefix

        bins = tpsa_bins if tpsa_bins is not None else DEFAULT_TPSA_BINS
        self.tpsa_bins = np.array(bins, dtype=np.float64)
        self.n_bins = len(self.tpsa_bins) - 1
        if self.n_bins < 2:
            raise ValueError(f"Need at least 2 bins, got {self.n_bins}")

        self._chmbl_props = self._compute_chembl_proportions()
        self._ema_props: np.ndarray | None = None

        logger.info(
            "[TPSADistributionReward] Initialised: %d bins, alpha=%.3f, scale=%.1f, "
            "range=[%.1f, %.1f]",
            self.n_bins,
            self.ema_alpha,
            self.scale,
            self.min_reward,
            self.max_reward,
        )

    def _compute_chembl_proportions(self) -> np.ndarray:
        csv_path = Path(self.chembl_csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"ChEMBL CSV not found: {csv_path}")

        counts = np.zeros(self.n_bins, dtype=np.int64)
        with gzip.open(csv_path, "rt") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    tpsa = float(row[CHEMBL_TPSA_COLUMN])
                except (KeyError, ValueError):
                    continue
                idx = int(np.digitize(tpsa, self.tpsa_bins)) - 1
                if 0 <= idx < self.n_bins:
                    counts[idx] += 1

        total = int(counts.sum())
        if total == 0:
            raise RuntimeError("No valid TPSA values found in ChEMBL CSV")

        props = counts.astype(np.float64) / total
        logger.info("[TPSADistributionReward] ChEMBL reference (%d molecules):", total)
        for i, label in enumerate(BIN_LABELS[: self.n_bins]):
            logger.info("  %s: %.1f%%", label, props[i] * 100)
        return props

    def _compute_batch_proportions(self, tpsa_values: Sequence[float]) -> np.ndarray:
        counts = np.zeros(self.n_bins, dtype=np.float64)
        for tpsa in tpsa_values:
            idx = int(np.digitize(tpsa, self.tpsa_bins)) - 1
            if 0 <= idx < self.n_bins:
                counts[idx] += 1
        return (counts + 1.0) / (counts.sum() + self.n_bins)

    def _update_ema(self, batch_props: np.ndarray) -> np.ndarray:
        if self.ema_alpha <= 0.0:
            return batch_props.copy()
        if self._ema_props is None:
            self._ema_props = batch_props.copy()
        else:
            a = self.ema_alpha
            self._ema_props = (1.0 - a) * self._ema_props + a * batch_props
        return self._ema_props.copy()

    def calculate_rewards(self, data: "DataProto") -> "DataProto":
        batch = data.batch
        ntb = getattr(data, "non_tensor_batch", None)
        rdkit_valid = ntb.get("rdkit_valid") if ntb is not None else batch.get("rdkit_valid")
        rdmols = ntb.get("rdmols") if ntb is not None else None

        if "x" in batch:
            batch_size = int(batch["x"].shape[0])
            device = batch["x"].device
        elif "nodesxsample" in batch:
            batch_size = int(batch["nodesxsample"].shape[0])
            device = batch["nodesxsample"].device
        else:
            raise ValueError("[TPSADistributionReward] cannot determine batch size")

        rewards = torch.full((batch_size,), self.failure_penalty, dtype=torch.float32, device=device)
        success = torch.zeros(batch_size, dtype=torch.int64, device=device)
        bin_idx_t = torch.full((batch_size,), -1, dtype=torch.int64, device=device)
        gap_t = torch.zeros(batch_size, dtype=torch.float32, device=device)

        if rdmols is None or rdkit_valid is None:
            self._write_batch_fields(data, rewards, success, bin_idx_t, gap_t)
            return data

        valid_tpsa: list[float] = []
        mol_to_bin: dict[int, int] = {}
        for idx in range(batch_size):
            if int(rdkit_valid[idx]) == 0 or idx >= len(rdmols) or rdmols[idx] is None:
                continue
            try:
                tpsa = float(Descriptors.TPSA(rdmols[idx]))
            except Exception:
                continue
            bin_idx = int(np.digitize(tpsa, self.tpsa_bins)) - 1
            if 0 <= bin_idx < self.n_bins:
                valid_tpsa.append(tpsa)
                mol_to_bin[idx] = bin_idx

        if valid_tpsa:
            batch_props = self._compute_batch_proportions(valid_tpsa)
            ema_props = self._update_ema(batch_props)
            gaps = self._chmbl_props - ema_props
        else:
            gaps = np.zeros(self.n_bins, dtype=np.float64)

        for idx, bin_idx in mol_to_bin.items():
            gap_val = float(gaps[bin_idx])
            reward = self.scale * gap_val
            reward = max(self.min_reward, min(self.max_reward, reward))
            rewards[idx] = reward
            success[idx] = 1
            bin_idx_t[idx] = bin_idx
            gap_t[idx] = gap_val

        self._write_batch_fields(data, rewards, success, bin_idx_t, gap_t)
        return data

    def _write_batch_fields(
        self,
        data: "DataProto",
        rewards: torch.Tensor,
        success: torch.Tensor,
        bin_idx: torch.Tensor,
        gap: torch.Tensor,
    ) -> None:
        prefix = self.reward_prefix
        data.batch[f"{prefix}_reward"] = rewards
        data.batch[f"{prefix}_success"] = success
        data.batch[f"{prefix}_bin_idx"] = bin_idx
        data.batch[f"{prefix}_gap"] = gap

    def field_specs(self) -> list[FieldSpec]:
        prefix = self.reward_prefix
        return [
            FieldSpec(
                name=f"avg_{prefix}_reward",
                scope="epoch",
                source="batch",
                source_key=f"{prefix}_reward",
                reduce="generated_mean",
                order=105,
            ),
        ]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"prefix={self.reward_prefix!r}, bins={self.n_bins}, "
            f"alpha={self.ema_alpha}, scale={self.scale}, "
            f"range=[{self.min_reward}, {self.max_reward}])"
        )
