"""ChEMBLPriorReward: bidirectional reward based on ChEMBL fingerprint similarity.

Reward semantics (bidirectional):
    - neutral point at ``low_similarity_threshold`` (e.g., 0.25)
    - positive reward when top-k mean similarity > threshold::
        reward = low_weight * (sim - threshold)
    - capped at ``max_reward_similarity`` (e.g., sim >= 0.50 gives max reward)
    - negative reward when similarity < threshold (same linear slope)
    - molecules without a valid canonical RDKit mol receive failure_penalty
    - excessive max similarity can receive a small copy/novelty penalty
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import torch
from rdkit.Chem import AllChem

from ddpo.rewards.chembl_similarity import (
    ChEMBLFingerprintCache,
    compute_topk_tanimoto,
    load_chembl_fingerprint_cache,
)
from ddpo.rewards.interfaces import BaseReward

if TYPE_CHECKING:
    from ddpo.runtime.protocol import DataProto


DEFAULT_FAILURE_PENALTY: float = -1.0
DEFAULT_TOP_K: int = 5
DEFAULT_CHUNK_SIZE: int = 100000
DEFAULT_LOW_SIMILARITY_THRESHOLD: float = 0.25
DEFAULT_LOW_SIMILARITY_WEIGHT: float = 4.0
DEFAULT_HIGH_SIMILARITY_THRESHOLD: float = 0.85
DEFAULT_HIGH_SIMILARITY_WEIGHT: float = 0.1
DEFAULT_USE_FEATURES: bool = False
DEFAULT_REWARD_PREFIX: str = "chembl_prior"
DEFAULT_MAX_REWARD_SIMILARITY: float = 0.50


class ChEMBLPriorReward(BaseReward):
    """ChEMBL fingerprint prior reward for canonical RDKit molecules.

    Supports both ECFP4 (use_features=False, default) and FCFP4
    (use_features=True) fingerprint types.  The fingerprint cache must match
    the feature type — ECFP4 fingerprints require use_features=False,
    FCFP4 fingerprints require use_features=True.

    Expects ``data.non_tensor_batch['rdmols']`` and ``rdkit_valid`` to be set by
    upstream molecule semantics annotation. The reward reuses those RDKit mols;
    it does not reconstruct molecules independently.

    Writes to ``data.batch``:
        ``chembl_max_similarity`` [B]
        ``chembl_top{k}_mean_similarity`` [B]
        ``chembl_low_similarity_penalty_raw`` [B]
        ``chembl_high_similarity_penalty_raw`` [B]
        ``chembl_prior_reward`` [B]
        ``chembl_prior_success`` [B]
    """

    def __init__(
        self,
        fingerprint_cache: str,
        failure_penalty: float = DEFAULT_FAILURE_PENALTY,
        top_k: int = DEFAULT_TOP_K,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        low_similarity_threshold: float = DEFAULT_LOW_SIMILARITY_THRESHOLD,
        low_similarity_weight: float = DEFAULT_LOW_SIMILARITY_WEIGHT,
        high_similarity_threshold: float = DEFAULT_HIGH_SIMILARITY_THRESHOLD,
        high_similarity_weight: float = DEFAULT_HIGH_SIMILARITY_WEIGHT,
        max_reward_similarity: float = DEFAULT_MAX_REWARD_SIMILARITY,
        use_features: bool = DEFAULT_USE_FEATURES,
        reward_prefix: str = DEFAULT_REWARD_PREFIX,
        cache: ChEMBLFingerprintCache | None = None,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if low_similarity_weight < 0.0 or high_similarity_weight < 0.0:
            raise ValueError("similarity penalty weights must be non-negative")
        if max_reward_similarity <= low_similarity_threshold:
            raise ValueError("max_reward_similarity must be > low_similarity_threshold")

        self.fingerprint_cache = fingerprint_cache
        self.failure_penalty = float(failure_penalty)
        self.top_k = int(top_k)
        self.chunk_size = int(chunk_size)
        self.low_similarity_threshold = float(low_similarity_threshold)
        self.low_similarity_weight = float(low_similarity_weight)
        self.high_similarity_threshold = float(high_similarity_threshold)
        self.high_similarity_weight = float(high_similarity_weight)
        self.max_reward_similarity = float(max_reward_similarity)
        self.use_features = bool(use_features)
        self.reward_prefix = reward_prefix
        self._cache = cache

    @property
    def cache(self) -> ChEMBLFingerprintCache:
        if self._cache is None:
            self._cache = load_chembl_fingerprint_cache(self.fingerprint_cache)
        return self._cache

    @property
    def topk_mean_field(self) -> str:
        return f"{self.reward_prefix}_top{self.top_k}_mean_similarity"

    def _reward_from_similarity(self, max_similarity: float, topk_mean_similarity: float) -> tuple[float, float, float]:
        low_penalty = max(0.0, self.low_similarity_threshold - topk_mean_similarity)
        high_penalty = max(0.0, max_similarity - self.high_similarity_threshold)
        # Bidirectional reward: positive above threshold, negative below
        reward = self.low_similarity_weight * (topk_mean_similarity - self.low_similarity_threshold)
        # Cap the maximum reward at max_reward_similarity
        max_reward = self.low_similarity_weight * (self.max_reward_similarity - self.low_similarity_threshold)
        reward = min(max_reward, reward)
        reward -= self.high_similarity_weight * high_penalty
        return max(self.failure_penalty, reward), low_penalty, high_penalty

    def calculate_rewards(self, data: "DataProto") -> "DataProto":
        batch = data.batch
        ntb = getattr(data, "non_tensor_batch", None)
        rdkit_valid = ntb.get("rdkit_valid") if ntb else batch.get("rdkit_valid")
        rdmols = ntb.get("rdmols") if ntb else None

        if rdkit_valid is None:
            raise ValueError(
                "[ChEMBLPriorReward] rdkit_valid missing from batch. "
                "annotate_batch_molecule_semantics() must be called before calculate_rewards()."
            )

        if "x" in batch:
            batch_size = batch["x"].shape[0]
            device = batch["x"].device
        elif "nodesxsample" in batch:
            batch_size = batch["nodesxsample"].shape[0]
            device = batch["nodesxsample"].device
        else:
            raise ValueError("[ChEMBLPriorReward] cannot determine batch size")

        max_similarity = torch.zeros(batch_size, dtype=torch.float32, device=device)
        topk_mean_similarity = torch.zeros(batch_size, dtype=torch.float32, device=device)
        low_penalty_raw = torch.zeros(batch_size, dtype=torch.float32, device=device)
        high_penalty_raw = torch.zeros(batch_size, dtype=torch.float32, device=device)
        prior_reward = torch.full((batch_size,), self.failure_penalty, dtype=torch.float32, device=device)
        success = torch.zeros(batch_size, dtype=torch.int64, device=device)

        if rdmols is None:
            warnings.warn("[ChEMBLPriorReward] rdmols not found in non_tensor_batch")
            self._write_batch_fields(data, max_similarity, topk_mean_similarity, low_penalty_raw, high_penalty_raw, prior_reward, success)
            return data

        cache = self.cache
        for i in range(batch_size):
            if int(rdkit_valid[i]) == 0 or i >= len(rdmols) or rdmols[i] is None:
                continue

            try:
                query_fp = AllChem.GetMorganFingerprintAsBitVect(
                    rdmols[i],
                    radius=cache.radius,
                    nBits=cache.n_bits,
                    useFeatures=self.use_features,
                )
                max_sim, topk_mean, _ = compute_topk_tanimoto(
                    query_fp,
                    cache.fingerprints,
                    top_k=self.top_k,
                    chunk_size=self.chunk_size,
                )
                reward, low_penalty, high_penalty = self._reward_from_similarity(max_sim, topk_mean)

                max_similarity[i] = float(max_sim)
                topk_mean_similarity[i] = float(topk_mean)
                low_penalty_raw[i] = float(low_penalty)
                high_penalty_raw[i] = float(high_penalty)
                prior_reward[i] = float(reward)
                success[i] = 1
            except Exception as exc:
                warnings.warn(f"[ChEMBLPriorReward] molecule {i}: {exc}")

        self._write_batch_fields(
            data,
            max_similarity,
            topk_mean_similarity,
            low_penalty_raw,
            high_penalty_raw,
            prior_reward,
            success,
        )
        return data

    def _write_batch_fields(
        self,
        data: "DataProto",
        max_similarity: torch.Tensor,
        topk_mean_similarity: torch.Tensor,
        low_penalty_raw: torch.Tensor,
        high_penalty_raw: torch.Tensor,
        prior_reward: torch.Tensor,
        success: torch.Tensor,
    ) -> None:
        p = self.reward_prefix
        data.batch[f"{p}_max_similarity"] = max_similarity
        data.batch[self.topk_mean_field] = topk_mean_similarity
        data.batch[f"{p}_low_similarity_penalty_raw"] = low_penalty_raw
        data.batch[f"{p}_high_similarity_penalty_raw"] = high_penalty_raw
        data.batch[f"{p}_reward"] = prior_reward
        data.batch[f"{p}_success"] = success

    def __repr__(self) -> str:
        return (
            "ChEMBLPriorReward("
            f"prefix={self.reward_prefix!r}, "
            f"fingerprint_cache={self.fingerprint_cache!r}, "
            f"use_features={self.use_features}, "
            f"top_k={self.top_k}, "
            f"low_similarity_threshold={self.low_similarity_threshold}, "
            f"low_similarity_weight={self.low_similarity_weight}, "
            f"max_reward_similarity={self.max_reward_similarity}, "
            f"high_similarity_threshold={self.high_similarity_threshold}, "
            f"high_similarity_weight={self.high_similarity_weight})"
        )
