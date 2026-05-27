"""LogP Distribution Reward: per-molecule reward based on ChEMBL bin-proportion matching.

Divides LogP into N bins, compares per-batch EMA-smoothed proportions to
ChEMBL reference proportions, and assigns reward/punishment to each molecule
based on whether its bin is under- or over-represented.

Design:
    - 8 bins: [-inf, 0.5, 1.5, 2.5, 3.0, 3.5, 4.5, 5.5, +inf]
      chosen based on ChEMBL 1.55M LogP distribution analysis.
    - Reference proportions pre-computed from ChEMBL CSV at __init__.
    - EMA (exponential moving average) smooths per-batch bin proportions
      across training steps. Disabled by default (alpha=0.0).
    - Reward formula:
        gap[k] = ref_proportion[k] - ema_proportion[k]
        reward = clip(scale * gap[k], min_reward, max_reward)

Decoupling:
    - Pure Python + RDKit. No dependency on training infra, PPO, or GPU.
    - Does not touch schemas, metrics_writer, or train_helpers.
    - Only contract: implement BaseReward.calculate_rewards(DataProto) → DataProto.

Usage (YAML):
    reward:
      weights:
        logp_distribution: 0.15
      logp_distribution:
        chembl_csv: ./data/chembl/chembl_36_properties.csv.gz
        logp_bins: [-inf, 0.5, 1.5, 2.5, 3.0, 3.5, 4.5, 5.5, inf]
        failure_penalty: -1.0
        min_reward: -2.0
        max_reward: 2.0
        scale: 12.0
        ema_alpha: 0.0          # 0.0 = EMA disabled
        reward_prefix: logp_distribution
"""
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

# ── Module-level constants ─────────────────────────────────────────────────
DEFAULT_FAILURE_PENALTY: float = -1.0
DEFAULT_MIN_REWARD: float = -2.0
DEFAULT_MAX_REWARD: float = 2.0
DEFAULT_SCALE: float = 12.0
DEFAULT_EMA_ALPHA: float = 0.0  # EMA disabled by default
DEFAULT_REWARD_PREFIX: str = "logp_distribution"
DEFAULT_CHEMBL_CSV: str = "data/chembl/chembl_36_properties.csv.gz"
CHEMBL_LOGP_COLUMN: str = "mol_log_p"

# Recommended 8-bin edges based on ChEMBL 1.55M LogP distribution analysis.
# Proportions: [4.4%, 9.3%, 20.6%, 14.0%, 14.3%, 22.7%, 10.8%, 3.9%]
DEFAULT_LOGP_BINS: list[float] = [
    -float("inf"), 0.5, 1.5, 2.5, 3.0, 3.5, 4.5, 5.5, float("inf"),
]

BIN_LABELS: list[str] = [
    "(-∞,0.5)", "[0.5,1.5)", "[1.5,2.5)", "[2.5,3.0)",
    "[3.0,3.5)", "[3.5,4.5)", "[4.5,5.5)", "(5.5,+∞)",
]


class LogPDistributionReward(BaseReward):
    """Per-molecule reward matching batch LogP distribution to ChEMBL reference.

    The reward is proportional to the gap between ChEMBL reference proportion
    and EMA-smoothed batch proportion for the molecule's LogP bin.

    Bin under-representation → positive reward (encourage more).
    Bin over-representation  → negative reward (penalize excess).

    Expects ``data.non_tensor_batch['rdmols']`` and ``rdkit_valid`` to be
    set by upstream molecule semantics annotation.

    Writes to ``data.batch`` (where ``{prefix}`` = ``reward_prefix``):
        ``{prefix}_reward``   [B]  — per-molecule reward
        ``{prefix}_success``  [B]  — validity flag
        ``{prefix}_bin_idx``  [B]  — LogP bin index (0..n_bins-1, -1=invalid)
        ``{prefix}_gap``      [B]  — gap used for reward computation
    """

    def __init__(
        self,
        chembl_csv: str = DEFAULT_CHEMBL_CSV,
        logp_bins: Sequence[float] | None = None,
        failure_penalty: float = DEFAULT_FAILURE_PENALTY,
        min_reward: float = DEFAULT_MIN_REWARD,
        max_reward: float = DEFAULT_MAX_REWARD,
        scale: float = DEFAULT_SCALE,
        ema_alpha: float = DEFAULT_EMA_ALPHA,
        reward_prefix: str = DEFAULT_REWARD_PREFIX,
    ) -> None:
        """Initialise LogPDistributionReward.

        Args:
            chembl_csv: Path to gzipped ChEMBL properties CSV
                (must contain ``mol_log_p`` column).
            logp_bins: Bin edge sequence of length n_bins+1.
                Defaults to the recommended 8-bin scheme.
            failure_penalty: Reward for invalid / non-RDKit molecules.
            min_reward: Floor for per-molecule reward.
            max_reward: Ceiling for per-molecule reward.
            scale: Multiplier mapping gap → reward
                (reward = clip(scale * gap, min, max)).
            ema_alpha: EMA smoothing factor in (0, 1].
                0.0 = EMA disabled (use raw batch proportions).
                ~0.15 = smoothing window ≈ 7 batches.
            reward_prefix: Prefix for batch field names.
        """
        super().__init__()
        # ── Validate inputs ──
        if min_reward > max_reward:
            raise ValueError("min_reward must be <= max_reward")
        if scale <= 0:
            raise ValueError("scale must be positive")
        if not 0.0 <= ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in [0, 1]")

        self.chembl_csv: str = chembl_csv
        self.failure_penalty: float = float(failure_penalty)
        self.min_reward: float = float(min_reward)
        self.max_reward: float = float(max_reward)
        self.scale: float = float(scale)
        self.ema_alpha: float = float(ema_alpha)
        self.reward_prefix: str = reward_prefix

        # ── Bin configuration ──
        _bins = logp_bins if logp_bins is not None else DEFAULT_LOGP_BINS
        self.logp_bins: np.ndarray = np.array(_bins, dtype=np.float64)
        self.n_bins: int = len(self.logp_bins) - 1

        if self.n_bins < 2:
            raise ValueError(f"Need at least 2 bins, got {self.n_bins} from edges {self.logp_bins}")

        # ── Pre-compute ChEMBL reference proportions ──
        self._chmbl_props: np.ndarray = self._compute_chembl_proportions()

        # ── EMA state (reset on every DDP fork / new process) ──
        self._ema_props: np.ndarray | None = None

        logger.info(
            "[LogPDistributionReward] Initialised: %d bins, alpha=%.3f, scale=%.1f, "
            "range=[%.1f, %.1f]",
            self.n_bins, self.ema_alpha, self.scale,
            self.min_reward, self.max_reward,
        )

    # ── Reference computation ──────────────────────────────────────────

    def _compute_chembl_proportions(self) -> np.ndarray:
        """Read ChEMBL CSV once, bin LogP values, return reference proportions.

        Returns:
            Shape (n_bins,), sums to 1.0.
        """
        csv_path = Path(self.chembl_csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"ChEMBL CSV not found: {csv_path}")

        counts = np.zeros(self.n_bins, dtype=np.int64)
        with gzip.open(csv_path, "rt") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    logp = float(row[CHEMBL_LOGP_COLUMN])
                except (KeyError, ValueError):
                    continue
                idx = int(np.digitize(logp, self.logp_bins)) - 1
                if 0 <= idx < self.n_bins:
                    counts[idx] += 1

        total = int(counts.sum())
        if total == 0:
            raise RuntimeError("No valid LogP values found in ChEMBL CSV")

        props = counts.astype(np.float64) / total

        logger.info("[LogPDistributionReward] ChEMBL reference (%d molecules):", total)
        for i, label in enumerate(BIN_LABELS[:self.n_bins]):
            logger.info("  %s: %.1f%%", label, props[i] * 100)

        return props

    @property
    def chmbl_proportions(self) -> np.ndarray:
        """ChEMBL reference bin proportions (read-only)."""
        return self._chmbl_props

    # ── Per-batch computation ──────────────────────────────────────────

    def _compute_batch_proportions(self, logp_values: Sequence[float]) -> np.ndarray:
        """Compute Laplace-smoothed bin proportions for a single batch.

        Args:
            logp_values: LogP floats from valid RDKit molecules in this batch.

        Returns:
            Shape (n_bins,), Laplace-smoothed proportions summing to 1.0.
        """
        counts = np.zeros(self.n_bins, dtype=np.float64)
        for logp in logp_values:
            idx = int(np.digitize(logp, self.logp_bins)) - 1
            if 0 <= idx < self.n_bins:
                counts[idx] += 1
        # Laplace smoothing: +1 pseudocount per bin
        smoothed = (counts + 1.0) / (counts.sum() + self.n_bins)
        return smoothed

    def _update_ema(self, batch_props: np.ndarray) -> np.ndarray:
        """Update and return EMA-smoothed bin proportions.

        If ema_alpha == 0.0, EMA is disabled → returns raw batch_props.

        Args:
            batch_props: Current batch's smoothed bin proportions.

        Returns:
            EMA proportions for this batch (shape n_bins).
        """
        if self.ema_alpha <= 0.0:
            return batch_props.copy()

        if self._ema_props is None:
            self._ema_props = batch_props.copy()
        else:
            a = self.ema_alpha
            self._ema_props = (1.0 - a) * self._ema_props + a * batch_props
        return self._ema_props.copy()

    # ── Reward interface ───────────────────────────────────────────────

    def calculate_rewards(self, data: "DataProto") -> "DataProto":
        """Calculate LogP-distribution rewards for a batch of generated molecules.

        Args:
            data: DataProto with ``non_tensor_batch.rdmols`` and
                  ``non_tensor_batch.rdkit_valid`` (or batch equivalents).

        Returns:
            DataProto with batch fields added:
                ``{prefix}_reward``, ``{prefix}_success``,
                ``{prefix}_bin_idx``, ``{prefix}_gap``.
        """
        batch = data.batch
        ntb = getattr(data, "non_tensor_batch", None)
        rdkit_valid = ntb.get("rdkit_valid") if ntb is not None else batch.get("rdkit_valid")
        rdmols = ntb.get("rdmols") if ntb is not None else None

        # ── Determine batch size and device ──
        if "x" in batch:
            B = int(batch["x"].shape[0])
            device = batch["x"].device
        elif "nodesxsample" in batch:
            B = int(batch["nodesxsample"].shape[0])
            device = batch["nodesxsample"].device
        else:
            raise ValueError("[LogPDistributionReward] cannot determine batch size")

        # ── Allocate output tensors ──
        rewards = torch.full((B,), self.failure_penalty, dtype=torch.float32, device=device)
        success = torch.zeros(B, dtype=torch.int64, device=device)
        bin_idx_t = torch.full((B,), -1, dtype=torch.int64, device=device)
        gap_t = torch.zeros(B, dtype=torch.float32, device=device)

        # ── Early exit if no RDKit mols ──
        if rdmols is None:
            self._write_batch_fields(data, rewards, success, bin_idx_t, gap_t)
            return data

        # ── Phase 1: collect LogP for all valid molecules ──
        valid_logps: list[float] = []
        mol_to_bin: dict[int, int] = {}  # molecule index → bin index
        for i in range(B):
            if int(rdkit_valid[i]) == 0 or i >= len(rdmols) or rdmols[i] is None:
                continue
            try:
                logp = float(Descriptors.MolLogP(rdmols[i]))
            except Exception:
                continue
            idx = int(np.digitize(logp, self.logp_bins)) - 1
            if 0 <= idx < self.n_bins:
                valid_logps.append(logp)
                mol_to_bin[i] = idx

        # ── Phase 2: batch proportions → EMA → gaps ──
        if valid_logps:
            batch_props = self._compute_batch_proportions(valid_logps)
            ema_props = self._update_ema(batch_props)
            # gap[bin] = ref - ema: positive = under-sampled (reward), negative = over (penalize)
            gaps = self._chmbl_props - ema_props
        else:
            gaps = np.zeros(self.n_bins, dtype=np.float64)

        # ── Phase 3: per-molecule reward assignment ──
        for i, bin_idx in mol_to_bin.items():
            gap_val = float(gaps[bin_idx])
            r = self.scale * gap_val
            r = max(self.min_reward, min(self.max_reward, r))
            rewards[i] = r
            success[i] = 1
            bin_idx_t[i] = bin_idx
            gap_t[i] = gap_val

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
        """Write reward-related tensors to data.batch."""
        p = self.reward_prefix
        data.batch[f"{p}_reward"] = rewards
        data.batch[f"{p}_success"] = success
        data.batch[f"{p}_bin_idx"] = bin_idx
        data.batch[f"{p}_gap"] = gap

    # ── Dynamic field declaration ──────────────────────────────────────

    def field_specs(self) -> list[FieldSpec]:
        """Declare epoch-level aggregate for this reward.

        ``{prefix}_reward`` is averaged over all generated molecules
        (including failures, which receive failure_penalty), matching
        how other epoch-level reward averages are computed.
        """
        return [
            FieldSpec(
                name=f"avg_{self.reward_prefix}_reward",
                scope="epoch",
                source="batch",
                source_key=f"{self.reward_prefix}_reward",
                reduce="generated_mean",
                order=100,
            ),
        ]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"prefix={self.reward_prefix!r}, "
            f"bins={self.n_bins}, "
            f"alpha={self.ema_alpha}, "
            f"scale={self.scale}, "
            f"range=[{self.min_reward}, {self.max_reward}])"
        )
