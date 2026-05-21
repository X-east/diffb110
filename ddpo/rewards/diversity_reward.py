"""DiversityReward: discourages molecular mode collapse within a batch.

For each valid molecule, computes the average Tanimoto similarity to all other
valid molecules in the same batch. By default this is a hinge penalty:
valid molecules are only penalized when their mean similarity is above a
configured target.

This is a batch-dependent reward — the same molecule may get different scores
in different batches. This is consistent with PPO advantage computation which
is also per-batch.

Use diversity in conjunction with other rewards (QED, SA, docking) via
CompositeReward to balance quality vs. diversity.

Fingerprint type is configurable via ``use_features``:
    - ``use_features=False`` → ECFP4 (default)
    - ``use_features=True``  → FCFP4

All batch fields are prefixed by ``reward_prefix`` (default ``"diversity"``):

    ``{prefix}_similarity_raw``  [B]  — avg pairwise Tanimoto (0.0 for invalid)
    ``{prefix}_diversity_raw``   [B]  — 1.0 - avg pairwise Tanimoto (0.0 for invalid)
    ``{prefix}_reward``          [B]  — hinge penalty reward, or failure_penalty for invalid
    ``{prefix}_success``         [B]  — 1 if diversity computed, 0 otherwise

Never hardcode a fingerprint type — always use the ``use_features`` parameter.
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from rdkit.Chem import AllChem

from ddpo.rewards.interfaces import BaseReward

if TYPE_CHECKING:
    from ddpo.runtime.protocol import DataProto


DEFAULT_FAILURE_PENALTY: float = -1.0
DEFAULT_TARGET_SIMILARITY: float = 0.5
DEFAULT_PENALTY_SCALE: float = 1.0
DEFAULT_MODE: str = "hinge_penalty"
DEFAULT_USE_FEATURES: bool = False  # False=ECFP4, True=FCFP4
DEFAULT_REWARD_PREFIX: str = "diversity"
DEFAULT_FINGERPRINT_RADIUS: int = 2
DEFAULT_FINGERPRINT_BITS: int = 2048


class DiversityReward(BaseReward):
    """Reward that penalizes excessive similarity to batch-mates.

    Default mode:
        mean_similarity <= target_similarity -> reward = 0
        mean_similarity >  target_similarity -> negative hinge penalty

    Fingerprint type controlled by ``use_features``:
        False → ECFP4, True → FCFP4.
    """

    def __init__(
        self,
        failure_penalty: float = DEFAULT_FAILURE_PENALTY,
        dist_ctx: Any | None = None,
        global_batch: bool = True,
        mode: str = DEFAULT_MODE,
        target_similarity: float = DEFAULT_TARGET_SIMILARITY,
        penalty_scale: float = DEFAULT_PENALTY_SCALE,
        use_features: bool = DEFAULT_USE_FEATURES,
        reward_prefix: str = DEFAULT_REWARD_PREFIX,
    ) -> None:
        self.failure_penalty = failure_penalty
        self.dist_ctx = dist_ctx
        self.global_batch = global_batch
        self.mode = mode
        self.target_similarity = target_similarity
        self.penalty_scale = penalty_scale
        self.use_features = use_features
        self.reward_prefix = reward_prefix

    # ------------------------------------------------------------------
    # Fingerprint helpers (stateless, separate for testability)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_fingerprints(mols: list, use_features: bool = False) -> list:
        """Compute Morgan fingerprints (radius=2, 2048 bits).

        Args:
            mols: list of RDKit Mol objects.
            use_features: False → ECFP4, True → FCFP4 (feature-based).

        Never hardcode the fingerprint type — always pass ``use_features``.
        """
        fps = []
        for mol in mols:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol,
                radius=DEFAULT_FINGERPRINT_RADIUS,
                nBits=DEFAULT_FINGERPRINT_BITS,
                useFeatures=use_features,
            )
            fps.append(fp)
        return fps

    @staticmethod
    def fingerprint_on_bits(fp: Any) -> set[int]:
        """Convert an RDKit bit vector to a pickle-friendly set of on bits."""
        return set(int(bit) for bit in fp.GetOnBits())

    @staticmethod
    def tanimoto_similarity(a: set[int], b: set[int]) -> float:
        """Compute Tanimoto similarity for two binary fingerprint bit sets."""
        union_size = len(a | b)
        if union_size == 0:
            return 0.0
        return float(len(a & b) / union_size)

    @classmethod
    def compute_pairwise_diversity(cls, fps: list) -> list[float]:
        """Compute diversity score = 1 - mean_sim for each fingerprint.

        Returns one score per input fingerprint.  When 0 or 1 inputs,
        returns [0.0] (neutral) because pairwise diversity is undefined.
        """
        bit_sets = [cls.fingerprint_on_bits(fp) for fp in fps]
        similarities = cls.compute_mean_similarities_from_bit_sets(bit_sets)
        return [1.0 - sim for sim in similarities] if len(bit_sets) > 1 else [0.0] * len(bit_sets)

    @classmethod
    def compute_mean_similarities_from_bit_sets(
        cls,
        local_bit_sets: list[set[int]],
        reference_bit_sets: list[set[int]] | None = None,
        local_reference_offsets: list[int] | None = None,
    ) -> list[float]:
        """Compute mean similarity for local fingerprints against a reference batch.

        ``reference_bit_sets`` may be the local batch or a gathered global batch.
        ``local_reference_offsets`` marks each local fingerprint's own position in
        the reference batch so self-similarity can be excluded.
        """
        local_reference = reference_bit_sets is None
        refs = local_bit_sets if local_reference else reference_bit_sets
        n_ref = len(refs)
        if n_ref <= 1:
            return [0.0] * len(local_bit_sets)

        similarities: list[float] = []
        for local_idx, bits in enumerate(local_bit_sets):
            own_ref_idx = local_idx if local_reference else None
            if local_reference_offsets is not None and local_idx < len(local_reference_offsets):
                own_ref_idx = local_reference_offsets[local_idx]

            sims = [
                cls.tanimoto_similarity(bits, ref_bits)
                for ref_idx, ref_bits in enumerate(refs)
                if ref_idx != own_ref_idx
            ]
            mean_sim = sum(sims) / len(sims) if sims else 0.0
            similarities.append(mean_sim)
        return similarities

    @classmethod
    def compute_diversity_from_bit_sets(
        cls,
        local_bit_sets: list[set[int]],
        reference_bit_sets: list[set[int]] | None = None,
        local_reference_offsets: list[int] | None = None,
    ) -> list[float]:
        """Compute diversity = 1 - mean similarity for local fingerprints."""
        similarities = cls.compute_mean_similarities_from_bit_sets(
            local_bit_sets,
            reference_bit_sets=reference_bit_sets,
            local_reference_offsets=local_reference_offsets,
        )
        refs = reference_bit_sets if reference_bit_sets is not None else local_bit_sets
        if len(refs) <= 1:
            return [0.0] * len(local_bit_sets)
        return [1.0 - sim for sim in similarities]

    def _reward_from_similarity(self, mean_similarity: float) -> float:
        mode = self.mode.lower()
        if mode in {"linear", "linear_diversity"}:
            return 1.0 - mean_similarity
        if mode in {"hinge", "hinge_penalty", "threshold", "collapse_penalty"}:
            penalty = max(0.0, mean_similarity - self.target_similarity)
            reward = -self.penalty_scale * penalty
            return max(self.failure_penalty, reward)
        raise ValueError(
            "[DiversityReward] unknown mode "
            f"{self.mode!r}; expected 'hinge_penalty' or 'linear'."
        )

    def set_distributed_context(self, dist_ctx: Any | None) -> None:
        """Bind a runtime distributed context after construction."""
        self.dist_ctx = dist_ctx

    def _all_gather_object(self, obj: Any) -> list[Any]:
        ctx = self.dist_ctx
        if (
            not self.global_batch
            or ctx is None
            or not getattr(ctx, "enabled", False)
            or not dist.is_available()
            or not dist.is_initialized()
        ):
            return [obj]
        output = [None for _ in range(int(getattr(ctx, "world_size", 1)))]
        dist.all_gather_object(output, obj)
        return output

    def _compute_similarity_for_local_valid_mols(self, valid_mols: list) -> tuple[list[float], int]:
        fps = self.compute_fingerprints(valid_mols, use_features=self.use_features)
        local_bit_sets = [self.fingerprint_on_bits(fp) for fp in fps]
        gathered = self._all_gather_object(local_bit_sets)

        reference_bit_sets: list[set[int]] = []
        local_reference_offsets: list[int] | None = None
        ctx = self.dist_ctx
        rank = int(getattr(ctx, "rank", 0)) if ctx is not None and len(gathered) > 1 else 0
        for gathered_rank, rank_bit_sets in enumerate(gathered):
            rank_bit_sets = rank_bit_sets or []
            if gathered_rank == rank:
                start = len(reference_bit_sets)
                local_reference_offsets = list(range(start, start + len(rank_bit_sets)))
            reference_bit_sets.extend(rank_bit_sets)

        similarities = self.compute_mean_similarities_from_bit_sets(
            local_bit_sets,
            reference_bit_sets=reference_bit_sets,
            local_reference_offsets=local_reference_offsets,
        )
        return similarities, len(reference_bit_sets)

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def calculate_rewards(self, data: "DataProto") -> "DataProto":
        """Calculate diversity rewards for a batch of molecules.

        Expects ``data.non_tensor_batch['rdmols']`` and
        ``data.non_tensor_batch['rdkit_valid']`` to be set by
        :func:`~ddpo.utils.molecule_semantics.annotate_batch_molecule_semantics`
        upstream.
        """
        batch = data.batch
        ntb = getattr(data, "non_tensor_batch", None)
        rdkit_valid = ntb.get("rdkit_valid") if ntb else batch.get("rdkit_valid")

        if rdkit_valid is None:
            raise ValueError(
                "[DiversityReward] rdkit_valid missing from batch. "
                "annotate_batch_molecule_semantics() must be called before calculate_rewards()."
            )

        batch_size = batch["x"].shape[0]
        device = batch["x"].device

        div_reward = torch.full((batch_size,), self.failure_penalty, device=device)
        sim_raw = torch.zeros(batch_size, device=device)
        div_raw = torch.zeros(batch_size, device=device)
        div_success = torch.zeros(batch_size, dtype=torch.int64, device=device)

        # Gather valid RDKit mol objects
        rdmols = ntb.get("rdmols") if ntb else None
        if rdmols is None:
            warnings.warn("[DiversityReward] rdmols not found in non_tensor_batch")
            p = self.reward_prefix
            data.batch[f"{p}_similarity_raw"] = sim_raw
            data.batch[f"{p}_diversity_raw"] = div_raw
            data.batch[f"{p}_reward"] = div_reward
            data.batch[f"{p}_success"] = div_success
            return data

        valid_indices = []
        valid_mols = []
        for i in range(batch_size):
            if int(rdkit_valid[i]) and i < len(rdmols) and rdmols[i] is not None:
                valid_indices.append(i)
                valid_mols.append(rdmols[i])

        try:
            # In DDP every rank must enter all_gather_object, even ranks with
            # zero valid molecules, otherwise ranks with valid molecules hang.
            similarities, reference_count = self._compute_similarity_for_local_valid_mols(valid_mols)
            for idx, mean_similarity in zip(valid_indices, similarities):
                sim_raw[idx] = mean_similarity
                div_raw[idx] = 0.0 if reference_count <= 1 else 1.0 - mean_similarity
                div_reward[idx] = 0.0 if reference_count <= 1 else self._reward_from_similarity(mean_similarity)
                div_success[idx] = 1
        except Exception as exc:
            warnings.warn(f"[DiversityReward] fingerprint/similarity failed: {exc}")

        p = self.reward_prefix
        data.batch[f"{p}_similarity_raw"] = sim_raw
        data.batch[f"{p}_diversity_raw"] = div_raw
        data.batch[f"{p}_reward"] = div_reward
        data.batch[f"{p}_success"] = div_success

        return data

    def __repr__(self) -> str:
        fp_type = "FCFP4" if self.use_features else "ECFP4"
        return (
            "DiversityReward("
            f"prefix={self.reward_prefix!r}, "
            f"fp_type={fp_type}, "
            f"failure_penalty={self.failure_penalty}, "
            f"global_batch={self.global_batch}, "
            f"mode={self.mode!r}, "
            f"target_similarity={self.target_similarity}, "
            f"penalty_scale={self.penalty_scale})"
        )
