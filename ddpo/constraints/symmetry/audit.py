"""RDKit-layer chemical audit for paired sidechain symmetry.

Verifies that R1/R2 atom types, bond topology, ring counts, and attachment
bonds are consistent after OpenBabel reconstruction.  Pure functions — does
not depend on training infrastructure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rdkit import Chem
from rdkit.Chem import rdmolops


@dataclass
class SymmetryAuditResult:
    """Per-molecule symmetry audit outcome."""

    molecule_index: int = 0
    rdkit_valid: bool = False
    r1_atom_types: list[str] = field(default_factory=list)
    r2_atom_types: list[str] = field(default_factory=list)
    atom_types_match: bool = False
    atom_types_correct_ratio: float = 0.0
    r1_heavy_count: int = 0
    r2_heavy_count: int = 0
    heavy_counts_match: bool = False
    r1_ring_count: int = 0
    r2_ring_count: int = 0
    ring_counts_match: bool = False
    subgraph_iso: bool = False
    r1_smiles: str = ""
    r2_smiles: str = ""
    attachment_bond_consistent: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def fully_consistent(self) -> bool:
        """Whether all atomic and topological checks passed."""
        return (
            self.rdkit_valid
            and self.atom_types_match
            and self.heavy_counts_match
            and self.ring_counts_match
            and self.subgraph_iso
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "molecule_index": self.molecule_index,
            "rdkit_valid": self.rdkit_valid,
            "atom_types_match": self.atom_types_match,
            "atom_types_correct_ratio": round(self.atom_types_correct_ratio, 4),
            "heavy_counts_match": self.heavy_counts_match,
            "ring_counts_match": self.ring_counts_match,
            "subgraph_iso": self.subgraph_iso,
            "attachment_bond_consistent": self.attachment_bond_consistent,
            "r1_heavy_count": self.r1_heavy_count,
            "r2_heavy_count": self.r2_heavy_count,
            "r1_ring_count": self.r1_ring_count,
            "r2_ring_count": self.r2_ring_count,
            "r1_smiles": self.r1_smiles,
            "r2_smiles": self.r2_smiles,
            "errors": self.errors,
        }


def _atom_symbol(mol: Chem.Mol, atom_idx: int) -> str:
    """Return the atomic symbol for a given atom index."""
    atom = mol.GetAtomWithIdx(atom_idx)
    return atom.GetSymbol()


def _heavy_atom_indices(mol: Chem.Mol) -> list[int]:
    """Return indices of non-hydrogen atoms."""
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]


def _count_rings_in_subset(mol: Chem.Mol, indices: set[int]) -> int:
    """Count unique rings that contain at least one atom in *indices*."""
    ri = mol.GetRingInfo()
    count = 0
    for ring in ri.AtomRings():
        if any(idx in indices for idx in ring):
            count += 1
    return count


def _subgraph_smiles(mol: Chem.Mol, indices: list[int]) -> str:
    """Extract a subgraph SMILES for the given atom indices."""
    sub = Chem.PathToSubmol(mol, [mol.GetBondBetweenAtoms(i, j).GetIdx()
                                   for i in indices for j in indices
                                   if i < j and mol.GetBondBetweenAtoms(i, j) is not None])
    return Chem.MolToSmiles(sub) if sub else ""


def audit_symmetry(
    mol: Chem.Mol,
    r1_indices: list[int],
    r2_indices: list[int],
    molecule_index: int = 0,
) -> SymmetryAuditResult:
    """Audit a single RDKit molecule for R1/R2 sidechain symmetry.

    Args:
        mol: RDKit Mol object (assumed sanitized).
        r1_indices: Atom indices belonging to R1 sidechain.
        r2_indices: Atom indices belonging to R2 sidechain.
        molecule_index: Index for traceability.

    Returns:
        SymmetryAuditResult with all checks populated.
    """
    result = SymmetryAuditResult(molecule_index=molecule_index)

    if mol is None:
        result.errors.append("mol is None")
        return result

    try:
        Chem.SanitizeMol(mol)
        result.rdkit_valid = True
    except Exception as e:
        result.errors.append(f"sanitize: {e}")
        return result

    # 1. Atom type check
    result.r1_atom_types = [_atom_symbol(mol, i) for i in r1_indices]
    result.r2_atom_types = [_atom_symbol(mol, i) for i in r2_indices]
    if len(r1_indices) == len(r2_indices):
        match_count = sum(
            1 for a, b in zip(result.r1_atom_types, result.r2_atom_types) if a == b
        )
        result.atom_types_correct_ratio = match_count / len(r1_indices) if r1_indices else 1.0
        result.atom_types_match = (match_count == len(r1_indices))
    else:
        result.errors.append("R1/R2 index count mismatch")

    # 2. Heavy atom count
    r1_heavy = [i for i in r1_indices if _atom_symbol(mol, i) != "H"]
    r2_heavy = [i for i in r2_indices if _atom_symbol(mol, i) != "H"]
    result.r1_heavy_count = len(r1_heavy)
    result.r2_heavy_count = len(r2_heavy)
    result.heavy_counts_match = (result.r1_heavy_count == result.r2_heavy_count)

    # 3. Ring count
    r1_set = set(r1_indices)
    r2_set = set(r2_indices)
    result.r1_ring_count = _count_rings_in_subset(mol, r1_set)
    result.r2_ring_count = _count_rings_in_subset(mol, r2_set)
    result.ring_counts_match = (result.r1_ring_count == result.r2_ring_count)

    # 4. Subgraph isomorphism (R1 vs R2)
    try:
        r1_smi = _subgraph_smiles(mol, r1_indices)
        r2_smi = _subgraph_smiles(mol, r2_indices)
        result.r1_smiles = r1_smi
        result.r2_smiles = r2_smi
        if r1_smi and r2_smi:
            r1_mol = Chem.MolFromSmiles(r1_smi)
            r2_mol = Chem.MolFromSmiles(r2_smi)
            if r1_mol and r2_mol:
                result.subgraph_iso = r1_mol.HasSubstructMatch(r2_mol) and r2_mol.HasSubstructMatch(r1_mol)
    except Exception as e:
        result.errors.append(f"subgraph: {e}")

    # 5. Attachment bond check — R1/R2 each connect to core via same bond type?
    # (Simplified: check that each side has at least one bond to non-R atoms)
    try:
        r1_bonds_to_core = sum(
            1 for i in r1_indices
            for nb in mol.GetAtomWithIdx(i).GetNeighbors()
            if nb.GetIdx() not in r1_set
        )
        r2_bonds_to_core = sum(
            1 for i in r2_indices
            for nb in mol.GetAtomWithIdx(i).GetNeighbors()
            if nb.GetIdx() not in r2_set
        )
        result.attachment_bond_consistent = (r1_bonds_to_core > 0 and r2_bonds_to_core > 0)
    except Exception as e:
        result.errors.append(f"attachment: {e}")

    return result


def audit_batch(
    mols: list[Chem.Mol],
    r1_indices: list[int],
    r2_indices: list[int],
) -> list[SymmetryAuditResult]:
    """Audit a batch of RDKit molecules and return results.

    Args:
        mols: List of RDKit Mol objects.
        r1_indices: Atom indices for R1 sidechain.
        r2_indices: Atom indices for R2 sidechain.

    Returns:
        List of SymmetryAuditResult, one per molecule.
    """
    return [
        audit_symmetry(mol, r1_indices, r2_indices, molecule_index=i)
        for i, mol in enumerate(mols)
    ]


def audit_summary(results: list[SymmetryAuditResult]) -> dict[str, Any]:
    """Compute aggregate statistics from a list of audit results.

    Returns a dictionary suitable for CSV writing or terminal display.
    """
    n = len(results)
    if n == 0:
        return {"total": 0}
    valid = sum(1 for r in results if r.rdkit_valid)
    atom_ok = sum(1 for r in results if r.atom_types_match)
    iso_ok = sum(1 for r in results if r.subgraph_iso)
    ring_ok = sum(1 for r in results if r.ring_counts_match)
    all_ok = sum(1 for r in results if r.fully_consistent)

    return {
        "total": n,
        "rdkit_valid": valid,
        "rdkit_valid_pct": round(100 * valid / n, 1),
        "atom_types_match": atom_ok,
        "atom_types_match_pct": round(100 * atom_ok / n, 1),
        "subgraph_iso": iso_ok,
        "subgraph_iso_pct": round(100 * iso_ok / n, 1),
        "ring_counts_match": ring_ok,
        "ring_counts_match_pct": round(100 * ring_ok / n, 1),
        "fully_consistent": all_ok,
        "fully_consistent_pct": round(100 * all_ok / n, 1),
    }
