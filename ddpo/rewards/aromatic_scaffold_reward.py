"""Aromatic scaffold reward for generated RDKit molecules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from rdkit import Chem

from ddpo.field_registry import FieldSpec
from ddpo.rewards.interfaces import BaseReward

if TYPE_CHECKING:
    from ddpo.runtime.protocol import DataProto


HALOGEN_ATOMIC_NUMBERS = frozenset({9, 17, 35, 53})


@dataclass(frozen=True)
class AromaticScaffoldFacts:
    """Per-molecule aromatic scaffold facts used for reward shaping."""

    benzene_ring_count: int = 0
    heterocycle_ring_count: int = 0
    aromatic_heterocycle_ring_count: int = 0
    aliphatic_carbocycle_ring_count: int = 0
    max_fused_ring_system_size: int = 0
    three_member_ring_count: int = 0
    total_halogen_count: int = 0
    aromatic_halogen_count: int = 0
    bonus: float = 0.0
    penalty: float = 0.0


class AromaticScaffoldReward(BaseReward):
    """Reward a single benzene ring while penalizing suspicious ring patterns."""

    def __init__(
        self,
        failure_penalty: float = -1.0,
        base_reward: float = 0.0,
        min_reward: float = -1.0,
        max_reward: float = 1.0,
        exact_single_benzene_reward: float = 1.0,
        missing_benzene_penalty: float = 0.25,
        max_benzene_ring_count: int = 2,
        extra_benzene_penalty: float = 0.25,
        max_heterocycle_ring_count: int = 1,
        multi_heterocycle_penalty: float = 0.55,
        max_fused_ring_system_size: int = 3,
        fused_penalty: float = 0.30,
        max_three_member_ring_count: int = 0,
        three_member_penalty: float = 0.45,
        max_total_halogen_count: int = 2,
        total_halogen_penalty: float = 0.20,
        max_aromatic_halogen_count: int = 2,
        aromatic_halogen_penalty: float = 0.20,
    ) -> None:
        """Initialize aromatic scaffold reward parameters."""
        if min_reward > max_reward:
            raise ValueError("min_reward must be <= max_reward")

        self.failure_penalty = float(failure_penalty)
        self.base_reward = float(base_reward)
        self.min_reward = float(min_reward)
        self.max_reward = float(max_reward)
        self.exact_single_benzene_reward = float(exact_single_benzene_reward)
        self.missing_benzene_penalty = float(missing_benzene_penalty)
        self.max_benzene_ring_count = int(max_benzene_ring_count)
        self.extra_benzene_penalty = float(extra_benzene_penalty)
        self.max_heterocycle_ring_count = int(max_heterocycle_ring_count)
        self.multi_heterocycle_penalty = float(multi_heterocycle_penalty)
        self.max_fused_ring_system_size = int(max_fused_ring_system_size)
        self.fused_penalty = float(fused_penalty)
        self.max_three_member_ring_count = int(max_three_member_ring_count)
        self.three_member_penalty = float(three_member_penalty)
        self.max_total_halogen_count = int(max_total_halogen_count)
        self.total_halogen_penalty = float(total_halogen_penalty)
        self.max_aromatic_halogen_count = int(max_aromatic_halogen_count)
        self.aromatic_halogen_penalty = float(aromatic_halogen_penalty)

    def calculate_rewards(self, data: "DataProto") -> "DataProto":
        """Calculate aromatic scaffold reward and write molecule facts."""
        batch = data.batch
        batch_size = int(batch["nodesxsample"].shape[0]) if "nodesxsample" in batch else int(batch["x"].shape[0])
        device = batch["nodesxsample"].device if "nodesxsample" in batch else batch["x"].device
        ntb = data.non_tensor_batch if hasattr(data, "non_tensor_batch") else None
        rdmols = ntb.get("rdmols") if isinstance(ntb, dict) else None
        rdkit_valid = ntb.get("rdkit_valid") if isinstance(ntb, dict) else batch.get("rdkit_valid")

        if rdkit_valid is None:
            raise ValueError(
                "[AromaticScaffoldReward] rdkit_valid missing. "
                "annotate_batch_molecule_semantics() must run before rewards."
            )

        rewards = torch.full((batch_size,), self.failure_penalty, dtype=torch.float32, device=device)
        success = torch.zeros(batch_size, dtype=torch.int64, device=device)
        facts = [AromaticScaffoldFacts() for _ in range(batch_size)]

        for idx in range(batch_size):
            if int(rdkit_valid[idx]) == 0:
                continue
            mol = rdmols[idx] if rdmols is not None and idx < len(rdmols) else None
            if mol is None:
                continue
            fact = self._compute_facts(mol)
            facts[idx] = fact
            reward = self.base_reward + fact.bonus - fact.penalty
            rewards[idx] = torch.clamp(
                torch.tensor(reward, dtype=rewards.dtype, device=device),
                min=self.min_reward,
                max=self.max_reward,
            )
            success[idx] = 1

        self._write_fields(data, rewards, success, facts, device=device)
        return data

    def _compute_facts(self, mol: Chem.Mol) -> AromaticScaffoldFacts:
        ring_info = mol.GetRingInfo()
        atom_rings = [tuple(ring) for ring in ring_info.AtomRings()]

        benzene_ring_count = 0
        heterocycle_ring_count = 0
        aromatic_heterocycle_ring_count = 0
        aliphatic_carbocycle_ring_count = 0
        three_member_ring_count = 0

        for ring in atom_rings:
            atoms = [mol.GetAtomWithIdx(atom_idx) for atom_idx in ring]
            is_aromatic = all(atom.GetIsAromatic() for atom in atoms)
            is_all_carbon = all(atom.GetAtomicNum() == 6 for atom in atoms)
            has_hetero = any(atom.GetAtomicNum() not in (1, 6) for atom in atoms)
            if len(ring) == 3:
                three_member_ring_count += 1
            if is_aromatic and len(ring) == 6 and is_all_carbon:
                benzene_ring_count += 1
            if has_hetero:
                heterocycle_ring_count += 1
                aromatic_heterocycle_ring_count += int(is_aromatic)
            elif not is_aromatic:
                aliphatic_carbocycle_ring_count += 1

        max_fused_size = _max_fused_ring_system_size(atom_rings)
        total_halogen_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() in HALOGEN_ATOMIC_NUMBERS)
        aromatic_halogen_count = _aromatic_halogen_count(mol)

        bonus = self.exact_single_benzene_reward if benzene_ring_count == 1 else 0.0
        penalty = 0.0
        penalty += self.missing_benzene_penalty * int(benzene_ring_count == 0)
        penalty += self.extra_benzene_penalty * max(0, benzene_ring_count - self.max_benzene_ring_count)
        penalty += self.multi_heterocycle_penalty * max(
            0,
            heterocycle_ring_count - self.max_heterocycle_ring_count,
        )
        penalty += self.fused_penalty * max(0, max_fused_size - self.max_fused_ring_system_size)
        penalty += self.three_member_penalty * max(0, three_member_ring_count - self.max_three_member_ring_count)
        penalty += self.total_halogen_penalty * max(0, total_halogen_count - self.max_total_halogen_count)
        penalty += self.aromatic_halogen_penalty * max(
            0,
            aromatic_halogen_count - self.max_aromatic_halogen_count,
        )

        return AromaticScaffoldFacts(
            benzene_ring_count=benzene_ring_count,
            heterocycle_ring_count=heterocycle_ring_count,
            aromatic_heterocycle_ring_count=aromatic_heterocycle_ring_count,
            aliphatic_carbocycle_ring_count=aliphatic_carbocycle_ring_count,
            max_fused_ring_system_size=max_fused_size,
            three_member_ring_count=three_member_ring_count,
            total_halogen_count=total_halogen_count,
            aromatic_halogen_count=aromatic_halogen_count,
            bonus=float(bonus),
            penalty=float(penalty),
        )

    def _write_fields(
        self,
        data: "DataProto",
        rewards: torch.Tensor,
        success: torch.Tensor,
        facts: list[AromaticScaffoldFacts],
        *,
        device: torch.device,
    ) -> None:
        float_dtype = rewards.dtype
        data.batch["aromatic_scaffold_reward"] = rewards
        data.batch["aromatic_scaffold_success"] = success
        data.batch["aromatic_scaffold_benzene_ring_count"] = _tensor(
            [f.benzene_ring_count for f in facts],
            device=device,
            dtype=float_dtype,
        )
        data.batch["aromatic_scaffold_heterocycle_ring_count"] = _tensor(
            [f.heterocycle_ring_count for f in facts],
            device=device,
            dtype=float_dtype,
        )
        data.batch["aromatic_scaffold_aromatic_heterocycle_ring_count"] = _tensor(
            [f.aromatic_heterocycle_ring_count for f in facts],
            device=device,
            dtype=float_dtype,
        )
        data.batch["aromatic_scaffold_aliphatic_carbocycle_ring_count"] = _tensor(
            [f.aliphatic_carbocycle_ring_count for f in facts],
            device=device,
            dtype=float_dtype,
        )
        data.batch["aromatic_scaffold_max_fused_ring_system_size"] = _tensor(
            [f.max_fused_ring_system_size for f in facts],
            device=device,
            dtype=float_dtype,
        )
        data.batch["aromatic_scaffold_three_member_ring_count"] = _tensor(
            [f.three_member_ring_count for f in facts],
            device=device,
            dtype=float_dtype,
        )
        data.batch["aromatic_scaffold_total_halogen_count"] = _tensor(
            [f.total_halogen_count for f in facts],
            device=device,
            dtype=float_dtype,
        )
        data.batch["aromatic_scaffold_aromatic_halogen_count"] = _tensor(
            [f.aromatic_halogen_count for f in facts],
            device=device,
            dtype=float_dtype,
        )
        data.batch["aromatic_scaffold_bonus"] = _tensor(
            [f.bonus for f in facts],
            device=device,
            dtype=float_dtype,
        )
        data.batch["aromatic_scaffold_penalty"] = _tensor(
            [f.penalty for f in facts],
            device=device,
            dtype=float_dtype,
        )

    def field_specs(self) -> list[FieldSpec]:
        return [
            FieldSpec(name="aromatic_scaffold_reward", scope="molecule", source="batch", reduce="none", order=690),
            FieldSpec(
                name="avg_aromatic_scaffold_reward",
                scope="epoch",
                source="batch",
                source_key="aromatic_scaffold_reward",
                reduce="generated_mean",
                order=690,
            ),
        ]


def _aromatic_halogen_count(mol: Chem.Mol) -> int:
    count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in HALOGEN_ATOMIC_NUMBERS:
            continue
        neighbors = atom.GetNeighbors()
        if any(neighbor.GetIsAromatic() for neighbor in neighbors):
            count += 1
    return count


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
