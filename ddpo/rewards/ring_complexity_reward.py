"""Ring-complexity reward for generated RDKit molecules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from ddpo.field_registry import FieldSpec
from ddpo.rewards.interfaces import BaseReward

if TYPE_CHECKING:
    from ddpo.runtime.protocol import DataProto


@dataclass(frozen=True)
class RingComplexityFacts:
    """Per-molecule ring complexity facts."""

    ring_count: int = 0
    aromatic_ring_count: int = 0
    three_member_ring_count: int = 0
    max_fused_ring_system_size: int = 0
    spiro_atom_count: int = 0
    bridgehead_atom_count: int = 0
    penalty: float = 0.0


class RingComplexityReward(BaseReward):
    """Softly penalize excessive ring and fused-ring complexity."""

    def __init__(
        self,
        failure_penalty: float = -1.0,
        base_reward: float = 1.0,
        min_reward: float = -1.0,
        max_reward: float = 1.0,
        target_ring_count: int = 4,
        target_aromatic_ring_count: int = 2,
        target_fused_ring_system_size: int = 2,
        max_three_member_ring_count: int = 0,
        max_spiro_atom_count: int = 1,
        max_bridgehead_atom_count: int = 2,
        ring_penalty: float = 0.18,
        aromatic_penalty: float = 0.12,
        fused_penalty: float = 0.20,
        three_member_penalty: float = 0.35,
        spiro_penalty: float = 0.10,
        bridgehead_penalty: float = 0.08,
    ) -> None:
        """Initialize the ring-complexity reward.

        Args:
            failure_penalty: Reward for invalid molecules.
            base_reward: Starting reward before subtracting complexity penalties.
            min_reward: Lower clamp for valid molecule reward.
            max_reward: Upper clamp for valid molecule reward.
            target_ring_count: Rings above this count are softly penalized.
            target_aromatic_ring_count: Aromatic rings above this count are penalized.
            target_fused_ring_system_size: Largest fused-ring system above this is penalized.
            max_three_member_ring_count: Three-member rings above this are penalized.
            max_spiro_atom_count: Spiro atoms above this are penalized.
            max_bridgehead_atom_count: Bridgehead atoms above this are penalized.
            ring_penalty: Penalty per excess ring.
            aromatic_penalty: Penalty per excess aromatic ring.
            fused_penalty: Penalty per excess fused-ring-system ring.
            three_member_penalty: Penalty per excess three-member ring.
            spiro_penalty: Penalty per excess spiro atom.
            bridgehead_penalty: Penalty per excess bridgehead atom.
        """
        if min_reward > max_reward:
            raise ValueError("min_reward must be <= max_reward")

        self.failure_penalty = float(failure_penalty)
        self.base_reward = float(base_reward)
        self.min_reward = float(min_reward)
        self.max_reward = float(max_reward)
        self.target_ring_count = int(target_ring_count)
        self.target_aromatic_ring_count = int(target_aromatic_ring_count)
        self.target_fused_ring_system_size = int(target_fused_ring_system_size)
        self.max_three_member_ring_count = int(max_three_member_ring_count)
        self.max_spiro_atom_count = int(max_spiro_atom_count)
        self.max_bridgehead_atom_count = int(max_bridgehead_atom_count)
        self.ring_penalty = float(ring_penalty)
        self.aromatic_penalty = float(aromatic_penalty)
        self.fused_penalty = float(fused_penalty)
        self.three_member_penalty = float(three_member_penalty)
        self.spiro_penalty = float(spiro_penalty)
        self.bridgehead_penalty = float(bridgehead_penalty)

    def calculate_rewards(self, data: "DataProto") -> "DataProto":
        """Calculate ring-complexity reward and facts."""
        batch = data.batch
        batch_size = int(batch["nodesxsample"].shape[0]) if "nodesxsample" in batch else int(batch["x"].shape[0])
        device = batch["nodesxsample"].device if "nodesxsample" in batch else batch["x"].device
        ntb = data.non_tensor_batch if hasattr(data, "non_tensor_batch") else None
        rdmols = ntb.get("rdmols") if isinstance(ntb, dict) else None
        rdkit_valid = ntb.get("rdkit_valid") if isinstance(ntb, dict) else batch.get("rdkit_valid")

        if rdkit_valid is None:
            raise ValueError(
                "[RingComplexityReward] rdkit_valid missing. "
                "annotate_batch_molecule_semantics() must run before rewards."
            )

        rewards = torch.full((batch_size,), self.failure_penalty, dtype=torch.float32, device=device)
        facts = [RingComplexityFacts() for _ in range(batch_size)]
        success = torch.zeros(batch_size, dtype=torch.int64, device=device)

        for idx in range(batch_size):
            if int(rdkit_valid[idx]) == 0:
                continue
            mol = rdmols[idx] if rdmols is not None and idx < len(rdmols) else None
            if mol is None:
                continue
            fact = self.compute_facts(mol)
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

    def compute_facts(self, mol: Chem.Mol) -> RingComplexityFacts:
        """Compute ring-complexity facts for one valid RDKit molecule."""
        ring_info = mol.GetRingInfo()
        atom_rings = [tuple(ring) for ring in ring_info.AtomRings()]
        ring_count = len(atom_rings)
        aromatic_ring_count = int(rdMolDescriptors.CalcNumAromaticRings(mol))
        three_member_ring_count = sum(1 for ring in atom_rings if len(ring) == 3)
        max_fused_size = _max_fused_ring_system_size(atom_rings)
        spiro_atom_count = int(rdMolDescriptors.CalcNumSpiroAtoms(mol))
        bridgehead_atom_count = int(rdMolDescriptors.CalcNumBridgeheadAtoms(mol))

        penalty = 0.0
        penalty += self.ring_penalty * max(0, ring_count - self.target_ring_count)
        penalty += self.aromatic_penalty * max(0, aromatic_ring_count - self.target_aromatic_ring_count)
        penalty += self.fused_penalty * max(0, max_fused_size - self.target_fused_ring_system_size)
        penalty += self.three_member_penalty * max(0, three_member_ring_count - self.max_three_member_ring_count)
        penalty += self.spiro_penalty * max(0, spiro_atom_count - self.max_spiro_atom_count)
        penalty += self.bridgehead_penalty * max(0, bridgehead_atom_count - self.max_bridgehead_atom_count)

        return RingComplexityFacts(
            ring_count=ring_count,
            aromatic_ring_count=aromatic_ring_count,
            three_member_ring_count=three_member_ring_count,
            max_fused_ring_system_size=max_fused_size,
            spiro_atom_count=spiro_atom_count,
            bridgehead_atom_count=bridgehead_atom_count,
            penalty=float(penalty),
        )

    def _write_fields(
        self,
        data: "DataProto",
        rewards: torch.Tensor,
        success: torch.Tensor,
        facts: list[RingComplexityFacts],
        *,
        device: torch.device,
    ) -> None:
        float_dtype = rewards.dtype
        data.batch["ring_complexity_reward"] = rewards
        data.batch["ring_complexity_success"] = success
        data.batch["ring_complexity_ring_count"] = _tensor([f.ring_count for f in facts], device=device, dtype=float_dtype)
        data.batch["ring_complexity_aromatic_ring_count"] = _tensor(
            [f.aromatic_ring_count for f in facts],
            device=device,
            dtype=float_dtype,
        )
        data.batch["ring_complexity_three_member_ring_count"] = _tensor(
            [f.three_member_ring_count for f in facts],
            device=device,
            dtype=float_dtype,
        )
        data.batch["ring_complexity_max_fused_ring_system_size"] = _tensor(
            [f.max_fused_ring_system_size for f in facts],
            device=device,
            dtype=float_dtype,
        )
        data.batch["ring_complexity_spiro_atom_count"] = _tensor(
            [f.spiro_atom_count for f in facts],
            device=device,
            dtype=float_dtype,
        )
        data.batch["ring_complexity_bridgehead_atom_count"] = _tensor(
            [f.bridgehead_atom_count for f in facts],
            device=device,
            dtype=float_dtype,
        )
        data.batch["ring_complexity_penalty"] = _tensor([f.penalty for f in facts], device=device, dtype=float_dtype)

    def field_specs(self) -> list[FieldSpec]:
        return [
            FieldSpec(name="ring_complexity_reward", scope="molecule", source="batch", reduce="none", order=680),
            FieldSpec(name="avg_ring_complexity_reward", scope="epoch", source="batch", source_key="ring_complexity_reward", reduce="generated_mean", order=680),
        ]


def _max_fused_ring_system_size(atom_rings: list[tuple[int, ...]]) -> int:
    """Return the largest fused-ring connected component size."""
    if not atom_rings:
        return 0
    ring_sets = [set(ring) for ring in atom_rings]
    adjacency: list[set[int]] = [set() for _ in ring_sets]
    for i, left in enumerate(ring_sets):
        for j in range(i + 1, len(ring_sets)):
            if len(left & ring_sets[j]) >= 2:
                adjacency[i].add(j)
                adjacency[j].add(i)

    seen: set[int] = set()
    largest = 1
    for start in range(len(ring_sets)):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for nxt in adjacency[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        largest = max(largest, size)
    return largest


def _tensor(values: list[float | int], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(values, device=device, dtype=dtype)
