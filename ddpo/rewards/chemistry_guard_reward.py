from __future__ import annotations

"""Chemistry guard reward and facts for generated RDKit molecules.

This reward is intentionally conservative and decoupled: it reuses
``non_tensor_batch['rdmols']`` and ``rdkit_valid`` produced by the canonical
molecule semantics step, writes guard facts into ``data.batch``, and optionally
turns those facts into a finite reward.  The same fields can be consumed by
``GoodMoleculePool.quality_filters`` before trajectory replay amplifies samples.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from ddpo.field_registry import FieldSpec
from ddpo.rewards.interfaces import BaseReward

if TYPE_CHECKING:
    from ddpo.runtime.protocol import DataProto


HALOGENS = frozenset({"F", "Cl", "Br", "I"})


@dataclass(frozen=True)
class ChemistryGuardFacts:
    """Molecule-level chemistry guard facts used for reward and filtering."""

    heavy_atom_count: int = 0
    carbon_fraction: float = 0.0
    hetero_atom_count: int = 0
    hetero_atom_fraction: float = 0.0
    aromatic_ring_count: int = 0
    three_member_ring_count: int = 0
    fragment_count: int = 0
    bad_bond_length_fraction: float = 0.0
    halogen_bad_position_count: int = 0
    violation_count: int = 0


class ChemistryGuardReward(BaseReward):
    """Reward and monitoring fields for basic chemical geometry guardrails."""

    def __init__(
        self,
        failure_penalty: float = -1.0,
        base_reward: float = 1.0,
        violation_penalty: float = 0.2,
        min_reward: float = -1.0,
        max_reward: float = 1.0,
        max_carbon_fraction: float = 0.85,
        min_hetero_atom_fraction: float = 0.0,
        max_three_member_ring_count: int = 0,
        max_fragment_count: int = 1,
        max_bad_bond_length_fraction: float = 0.05,
        max_halogen_bad_position_count: int = 0,
        bond_length_relative_tolerance: float = 0.30,
        bond_length_absolute_tolerance: float = 0.20,
        max_halogen_neighbor_heavy_degree: int | None = None,
    ) -> None:
        """Initialize the guard reward.

        Args:
            failure_penalty: Reward assigned to invalid molecules.
            base_reward: Reward before subtracting violation penalties.
            violation_penalty: Penalty per violated guard.
            min_reward: Lower clamp for valid-molecule reward.
            max_reward: Upper clamp for valid-molecule reward.
            max_carbon_fraction: Upper allowed heavy-atom carbon fraction.
            min_hetero_atom_fraction: Lower allowed hetero-heavy-atom fraction.
            max_three_member_ring_count: Maximum allowed 3-member rings.
            max_fragment_count: Maximum allowed disconnected fragments.
            max_bad_bond_length_fraction: Maximum allowed bad bond-length ratio.
            max_halogen_bad_position_count: Maximum allowed suspicious halogens.
            bond_length_relative_tolerance: Relative tolerance around covalent radii sum.
            bond_length_absolute_tolerance: Minimum absolute tolerance in Angstrom.
            max_halogen_neighbor_heavy_degree: Optional threshold for flagging
                halogens attached to highly substituted heavy atoms.
        """
        if min_reward > max_reward:
            raise ValueError("min_reward must be <= max_reward")
        if bond_length_relative_tolerance < 0.0:
            raise ValueError("bond_length_relative_tolerance must be >= 0")
        if bond_length_absolute_tolerance < 0.0:
            raise ValueError("bond_length_absolute_tolerance must be >= 0")

        self.failure_penalty = float(failure_penalty)
        self.base_reward = float(base_reward)
        self.violation_penalty = float(violation_penalty)
        self.min_reward = float(min_reward)
        self.max_reward = float(max_reward)
        self.max_carbon_fraction = float(max_carbon_fraction)
        self.min_hetero_atom_fraction = float(min_hetero_atom_fraction)
        self.max_three_member_ring_count = int(max_three_member_ring_count)
        self.max_fragment_count = int(max_fragment_count)
        self.max_bad_bond_length_fraction = float(max_bad_bond_length_fraction)
        self.max_halogen_bad_position_count = int(max_halogen_bad_position_count)
        self.bond_length_relative_tolerance = float(bond_length_relative_tolerance)
        self.bond_length_absolute_tolerance = float(bond_length_absolute_tolerance)
        self.max_halogen_neighbor_heavy_degree = (
            None
            if max_halogen_neighbor_heavy_degree is None
            else int(max_halogen_neighbor_heavy_degree)
        )

    def calculate_rewards(self, data: "DataProto") -> "DataProto":
        """Calculate guard facts and reward for a batch."""
        batch = data.batch
        batch_size = int(batch["nodesxsample"].shape[0]) if "nodesxsample" in batch else int(batch["x"].shape[0])
        device = batch["x"].device if "x" in batch else batch["nodesxsample"].device
        ntb = data.non_tensor_batch if hasattr(data, "non_tensor_batch") else None
        rdmols = ntb.get("rdmols") if isinstance(ntb, dict) else None
        rdkit_valid = ntb.get("rdkit_valid") if isinstance(ntb, dict) else batch.get("rdkit_valid")

        if rdkit_valid is None:
            raise ValueError(
                "[ChemistryGuardReward] rdkit_valid missing. "
                "annotate_batch_molecule_semantics() must run before rewards."
            )

        rewards = torch.full((batch_size,), self.failure_penalty, device=device)
        success = torch.zeros(batch_size, dtype=torch.int64, device=device)
        fact_rows = [ChemistryGuardFacts() for _ in range(batch_size)]

        for idx in range(batch_size):
            if int(rdkit_valid[idx]) == 0:
                continue
            mol = rdmols[idx] if rdmols is not None and idx < len(rdmols) else None
            if mol is None:
                continue
            facts = self._compute_facts(mol)
            fact_rows[idx] = facts
            reward = self.base_reward - self.violation_penalty * float(facts.violation_count)
            rewards[idx] = torch.clamp(
                torch.tensor(reward, dtype=rewards.dtype, device=device),
                min=self.min_reward,
                max=self.max_reward,
            )
            success[idx] = 1

        self._write_fields(data, rewards, success, fact_rows, device=device)
        return data

    def _compute_facts(self, mol: Chem.Mol) -> ChemistryGuardFacts:
        """Compute guard facts for one valid RDKit molecule."""
        heavy_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
        heavy_atom_count = len(heavy_atoms)
        if heavy_atom_count == 0:
            return ChemistryGuardFacts(fragment_count=1, violation_count=1)

        carbon_count = sum(1 for atom in heavy_atoms if atom.GetSymbol() == "C")
        hetero_atom_count = sum(1 for atom in heavy_atoms if atom.GetSymbol() != "C")
        carbon_fraction = carbon_count / heavy_atom_count
        hetero_atom_fraction = hetero_atom_count / heavy_atom_count
        ring_info = mol.GetRingInfo()
        three_member_ring_count = sum(1 for ring in ring_info.AtomRings() if len(ring) == 3)
        aromatic_ring_count = int(rdMolDescriptors.CalcNumAromaticRings(mol))
        fragment_count = len(Chem.GetMolFrags(mol, sanitizeFrags=False))
        bad_bond_length_fraction = self._bad_bond_length_fraction(mol)
        halogen_bad_position_count = self._halogen_bad_position_count(mol)

        violation_count = 0
        violation_count += int(carbon_fraction > self.max_carbon_fraction)
        violation_count += int(hetero_atom_fraction < self.min_hetero_atom_fraction)
        violation_count += int(three_member_ring_count > self.max_three_member_ring_count)
        violation_count += int(fragment_count > self.max_fragment_count)
        violation_count += int(bad_bond_length_fraction > self.max_bad_bond_length_fraction)
        violation_count += int(halogen_bad_position_count > self.max_halogen_bad_position_count)

        return ChemistryGuardFacts(
            heavy_atom_count=heavy_atom_count,
            carbon_fraction=carbon_fraction,
            hetero_atom_count=hetero_atom_count,
            hetero_atom_fraction=hetero_atom_fraction,
            aromatic_ring_count=aromatic_ring_count,
            three_member_ring_count=three_member_ring_count,
            fragment_count=fragment_count,
            bad_bond_length_fraction=bad_bond_length_fraction,
            halogen_bad_position_count=halogen_bad_position_count,
            violation_count=violation_count,
        )

    def _bad_bond_length_fraction(self, mol: Chem.Mol) -> float:
        """Return fraction of bonds outside covalent-radius length tolerance."""
        if mol.GetNumBonds() == 0 or mol.GetNumConformers() == 0:
            return 0.0
        conformer = mol.GetConformer()
        table = Chem.GetPeriodicTable()
        bad_count = 0
        checked_count = 0
        for bond in mol.GetBonds():
            begin = bond.GetBeginAtom()
            end = bond.GetEndAtom()
            if begin.GetAtomicNum() <= 1 or end.GetAtomicNum() <= 1:
                continue
            expected = table.GetRcovalent(begin.GetAtomicNum()) + table.GetRcovalent(end.GetAtomicNum())
            tolerance = max(self.bond_length_absolute_tolerance, expected * self.bond_length_relative_tolerance)
            distance = conformer.GetAtomPosition(begin.GetIdx()).Distance(conformer.GetAtomPosition(end.GetIdx()))
            checked_count += 1
            if distance < expected - tolerance or distance > expected + tolerance:
                bad_count += 1
        return float(bad_count / checked_count) if checked_count > 0 else 0.0

    def _halogen_bad_position_count(self, mol: Chem.Mol) -> int:
        """Count suspicious halogen placements using RDKit graph degree."""
        count = 0
        for atom in mol.GetAtoms():
            if atom.GetSymbol() not in HALOGENS:
                continue
            heavy_neighbors = [nbr for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() > 1]
            if len(heavy_neighbors) != 1:
                count += 1
                continue
            if self.max_halogen_neighbor_heavy_degree is None:
                continue
            neighbor_heavy_degree = sum(1 for nbr in heavy_neighbors[0].GetNeighbors() if nbr.GetAtomicNum() > 1)
            if neighbor_heavy_degree > self.max_halogen_neighbor_heavy_degree:
                count += 1
        return count

    @staticmethod
    def _tensor(values: list[float | int], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Create a tensor from Python fact values."""
        return torch.tensor(values, device=device, dtype=dtype)

    def _write_fields(
        self,
        data: "DataProto",
        rewards: torch.Tensor,
        success: torch.Tensor,
        fact_rows: list[ChemistryGuardFacts],
        *,
        device: torch.device,
    ) -> None:
        """Write reward and fact fields into ``data.batch``."""
        float_dtype = rewards.dtype
        data.batch["chemistry_guard_reward"] = rewards
        data.batch["chemistry_guard_success"] = success
        data.batch["heavy_atom_count"] = self._tensor([f.heavy_atom_count for f in fact_rows], device=device, dtype=float_dtype)
        data.batch["carbon_fraction"] = self._tensor([f.carbon_fraction for f in fact_rows], device=device, dtype=float_dtype)
        data.batch["hetero_atom_count"] = self._tensor([f.hetero_atom_count for f in fact_rows], device=device, dtype=float_dtype)
        data.batch["hetero_atom_fraction"] = self._tensor([f.hetero_atom_fraction for f in fact_rows], device=device, dtype=float_dtype)
        data.batch["aromatic_ring_count"] = self._tensor([f.aromatic_ring_count for f in fact_rows], device=device, dtype=float_dtype)
        data.batch["three_member_ring_count"] = self._tensor([f.three_member_ring_count for f in fact_rows], device=device, dtype=float_dtype)
        data.batch["fragment_count"] = self._tensor([f.fragment_count for f in fact_rows], device=device, dtype=float_dtype)
        data.batch["bad_bond_length_fraction"] = self._tensor([f.bad_bond_length_fraction for f in fact_rows], device=device, dtype=float_dtype)
        data.batch["halogen_bad_position_count"] = self._tensor([f.halogen_bad_position_count for f in fact_rows], device=device, dtype=float_dtype)
        data.batch["chemistry_guard_violation_count"] = self._tensor([f.violation_count for f in fact_rows], device=device, dtype=float_dtype)

    def field_specs(self) -> list[FieldSpec]:
        """Declare molecule and epoch fields produced by chemistry guard."""
        molecule_fields = [
            "chemistry_guard_reward",
            "chemistry_guard_success",
            "heavy_atom_count",
            "carbon_fraction",
            "hetero_atom_count",
            "hetero_atom_fraction",
            "aromatic_ring_count",
            "three_member_ring_count",
            "fragment_count",
            "bad_bond_length_fraction",
            "halogen_bad_position_count",
            "chemistry_guard_violation_count",
        ]
        specs = [
            FieldSpec(name=name, scope="molecule", source="batch", reduce="none", order=700)
            for name in molecule_fields
        ]
        specs.extend(
            [
                FieldSpec(
                    name=f"avg_{name}",
                    scope="epoch",
                    source="batch",
                    source_key=name,
                    reduce="valid_mean" if name != "chemistry_guard_reward" else "generated_mean",
                    order=700,
                )
                for name in molecule_fields
            ]
        )
        return specs

    def __repr__(self) -> str:
        """Return concise config summary."""
        return (
            "ChemistryGuardReward("
            f"max_carbon_fraction={self.max_carbon_fraction}, "
            f"max_three_member_ring_count={self.max_three_member_ring_count}, "
            f"max_bad_bond_length_fraction={self.max_bad_bond_length_fraction})"
        )
