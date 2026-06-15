"""Ring-pathology reward for generated RDKit molecules.

This reward targets specific suspicious ring motifs observed in DiffGui runs:
too many rings, large rings, large rings fused to six-member rings, and
pathological three-member-ring topologies.  It intentionally does not penalize
multiple aromatic heterocycles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from rdkit import Chem

from ddpo.field_registry import FieldSpec
from ddpo.rewards.interfaces import BaseReward

if TYPE_CHECKING:
    from ddpo.runtime.protocol import DataProto


@dataclass(frozen=True)
class RingPathologyFacts:
    """Per-molecule ring-pathology facts."""

    ring_count: int = 0
    large_ring_count: int = 0
    large_ring_fused_six_count: int = 0
    three_member_ring_count: int = 0
    three_member_fused_or_spiro_count: int = 0
    penalty: float = 0.0


class RingPathologyReward(BaseReward):
    """Penalize selected pathological ring motifs while allowing normal rings."""

    def __init__(
        self,
        failure_penalty: float = -1.0,
        base_reward: float = 1.0,
        min_reward: float = -1.0,
        max_reward: float = 1.0,
        max_allowed_ring_count: int = 3,
        min_large_ring_size: int = 7,
        max_allowed_three_member_ring_count: int = 1,
        ring_count_penalty: float = 0.25,
        large_ring_penalty: float = 0.45,
        large_ring_fused_six_penalty: float = 0.65,
        extra_three_member_penalty: float = 0.45,
        three_member_fused_or_spiro_penalty: float = 0.65,
    ) -> None:
        """Initialize ring pathology reward parameters."""
        if min_reward > max_reward:
            raise ValueError("min_reward must be <= max_reward")
        self.failure_penalty = float(failure_penalty)
        self.base_reward = float(base_reward)
        self.min_reward = float(min_reward)
        self.max_reward = float(max_reward)
        self.max_allowed_ring_count = int(max_allowed_ring_count)
        self.min_large_ring_size = int(min_large_ring_size)
        self.max_allowed_three_member_ring_count = int(max_allowed_three_member_ring_count)
        self.ring_count_penalty = float(ring_count_penalty)
        self.large_ring_penalty = float(large_ring_penalty)
        self.large_ring_fused_six_penalty = float(large_ring_fused_six_penalty)
        self.extra_three_member_penalty = float(extra_three_member_penalty)
        self.three_member_fused_or_spiro_penalty = float(three_member_fused_or_spiro_penalty)

    def calculate_rewards(self, data: "DataProto") -> "DataProto":
        """Calculate ring-pathology rewards and write molecule facts."""
        batch = data.batch
        batch_size = int(batch["nodesxsample"].shape[0]) if "nodesxsample" in batch else int(batch["x"].shape[0])
        device = batch["nodesxsample"].device if "nodesxsample" in batch else batch["x"].device
        ntb = data.non_tensor_batch if hasattr(data, "non_tensor_batch") else None
        rdmols = ntb.get("rdmols") if isinstance(ntb, dict) else None
        rdkit_valid = ntb.get("rdkit_valid") if isinstance(ntb, dict) else batch.get("rdkit_valid")

        if rdkit_valid is None:
            raise ValueError(
                "[RingPathologyReward] rdkit_valid missing. "
                "annotate_batch_molecule_semantics() must run before rewards."
            )

        rewards = torch.full((batch_size,), self.failure_penalty, dtype=torch.float32, device=device)
        success = torch.zeros(batch_size, dtype=torch.int64, device=device)
        facts = [RingPathologyFacts() for _ in range(batch_size)]

        for idx in range(batch_size):
            if int(rdkit_valid[idx]) == 0:
                continue
            mol = rdmols[idx] if rdmols is not None and idx < len(rdmols) else None
            if mol is None:
                continue
            fact = self._compute_facts(mol)
            facts[idx] = fact
            reward = self.base_reward - fact.penalty
            rewards[idx] = torch.clamp(
                torch.tensor(reward, dtype=rewards.dtype, device=device),
                min=self.min_reward,
                max=self.max_reward,
            )
            success[idx] = 1

        self._write_fields(data, rewards, success, facts, device=device)
        return data

    def _compute_facts(self, mol: Chem.Mol) -> RingPathologyFacts:
        atom_rings = [tuple(ring) for ring in mol.GetRingInfo().AtomRings()]
        ring_sets = [set(ring) for ring in atom_rings]
        ring_count = len(atom_rings)
        large_ring_indices = [idx for idx, ring in enumerate(atom_rings) if len(ring) >= self.min_large_ring_size]
        three_member_indices = [idx for idx, ring in enumerate(atom_rings) if len(ring) == 3]

        large_ring_fused_six_count = 0
        for idx in large_ring_indices:
            large = ring_sets[idx]
            if any(j != idx and len(atom_rings[j]) == 6 and len(large & ring_sets[j]) >= 2 for j in range(len(atom_rings))):
                large_ring_fused_six_count += 1

        three_member_fused_or_spiro_count = 0
        for idx in three_member_indices:
            tri = ring_sets[idx]
            if any(j != idx and tri & ring_sets[j] for j in range(len(atom_rings))):
                three_member_fused_or_spiro_count += 1

        penalty = 0.0
        penalty += self.ring_count_penalty * max(0, ring_count - self.max_allowed_ring_count)
        penalty += self.large_ring_penalty * len(large_ring_indices)
        penalty += self.large_ring_fused_six_penalty * large_ring_fused_six_count
        penalty += self.extra_three_member_penalty * max(
            0,
            len(three_member_indices) - self.max_allowed_three_member_ring_count,
        )
        penalty += self.three_member_fused_or_spiro_penalty * three_member_fused_or_spiro_count

        return RingPathologyFacts(
            ring_count=ring_count,
            large_ring_count=len(large_ring_indices),
            large_ring_fused_six_count=large_ring_fused_six_count,
            three_member_ring_count=len(three_member_indices),
            three_member_fused_or_spiro_count=three_member_fused_or_spiro_count,
            penalty=float(penalty),
        )

    def _write_fields(
        self,
        data: "DataProto",
        rewards: torch.Tensor,
        success: torch.Tensor,
        facts: list[RingPathologyFacts],
        *,
        device: torch.device,
    ) -> None:
        float_dtype = rewards.dtype
        data.batch["ring_pathology_reward"] = rewards
        data.batch["ring_pathology_success"] = success
        data.batch["ring_pathology_ring_count"] = _tensor([f.ring_count for f in facts], device=device, dtype=float_dtype)
        data.batch["ring_pathology_large_ring_count"] = _tensor(
            [f.large_ring_count for f in facts],
            device=device,
            dtype=float_dtype,
        )
        data.batch["ring_pathology_large_ring_fused_six_count"] = _tensor(
            [f.large_ring_fused_six_count for f in facts],
            device=device,
            dtype=float_dtype,
        )
        data.batch["ring_pathology_three_member_ring_count"] = _tensor(
            [f.three_member_ring_count for f in facts],
            device=device,
            dtype=float_dtype,
        )
        data.batch["ring_pathology_three_member_fused_or_spiro_count"] = _tensor(
            [f.three_member_fused_or_spiro_count for f in facts],
            device=device,
            dtype=float_dtype,
        )
        data.batch["ring_pathology_penalty"] = _tensor([f.penalty for f in facts], device=device, dtype=float_dtype)

    def field_specs(self) -> list[FieldSpec]:
        return [
            FieldSpec(name="ring_pathology_reward", scope="molecule", source="batch", reduce="none", order=692),
            FieldSpec(
                name="avg_ring_pathology_reward",
                scope="epoch",
                source="batch",
                source_key="ring_pathology_reward",
                reduce="generated_mean",
                order=692,
            ),
        ]


def _tensor(values: list[float | int], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(values, device=device, dtype=dtype)
