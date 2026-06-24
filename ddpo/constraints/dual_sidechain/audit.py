"""RDKit audit utilities for dual aromatic-sidechain substitution sites."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from rdkit import Chem

from ddpo.constraints.dual_sidechain.spec import DualSubstitutionSpec, PhenolicOHSite


BOND_TYPE_NAMES = {
    Chem.BondType.SINGLE: "single",
    Chem.BondType.DOUBLE: "double",
    Chem.BondType.TRIPLE: "triple",
    Chem.BondType.AROMATIC: "aromatic",
}


@dataclass(frozen=True)
class SidechainBranchFacts:
    """Chemical facts for one substituent branch attached to an aromatic core."""

    label: str
    core_atom_idx: int
    root_atom_idx: int | None
    atom_indices: tuple[int, ...] = ()
    atom_symbols: tuple[str, ...] = ()
    internal_bonds: tuple[tuple[int, int, str, bool], ...] = ()
    attachment_bond_type: str = ""
    attachment_is_aromatic: bool = False
    ring_count: int = 0
    aromatic_ring_count: int = 0
    three_member_ring_count: int = 0
    max_fused_ring_system_size: int = 0
    signature: str = ""
    extractable: bool = False
    failure_type: str = ""

    @property
    def atom_count(self) -> int:
        """Return number of atoms in the branch."""
        return len(self.atom_indices)

    def to_dict(self, prefix: str) -> dict[str, Any]:
        """Return flat fields for CSV/JSON export."""
        return {
            f"{prefix}_core_atom_idx": self.core_atom_idx,
            f"{prefix}_root_atom_idx": self.root_atom_idx,
            f"{prefix}_atom_count": self.atom_count,
            f"{prefix}_atom_indices": " ".join(str(i) for i in self.atom_indices),
            f"{prefix}_atom_symbols": " ".join(self.atom_symbols),
            f"{prefix}_attachment_bond_type": self.attachment_bond_type,
            f"{prefix}_attachment_is_aromatic": int(self.attachment_is_aromatic),
            f"{prefix}_ring_count": self.ring_count,
            f"{prefix}_aromatic_ring_count": self.aromatic_ring_count,
            f"{prefix}_three_member_ring_count": self.three_member_ring_count,
            f"{prefix}_max_fused_ring_system_size": self.max_fused_ring_system_size,
            f"{prefix}_signature": self.signature,
            f"{prefix}_extractable": int(self.extractable),
            f"{prefix}_failure_type": self.failure_type,
        }


@dataclass(frozen=True)
class DualSidechainAuditResult:
    """Audit result for one molecule with two substitution sites."""

    molecule_index: int
    rdkit_valid: bool
    left: SidechainBranchFacts
    right: SidechainBranchFacts
    atom_count_match: bool = False
    atom_symbols_match: bool = False
    attachment_match: bool = False
    graph_signature_match: bool = False
    ring_profile_match: bool = False
    dual_success: bool = False
    failure_type: str = ""
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return flat fields for CSV/JSON export."""
        row: dict[str, Any] = {
            "molecule_index": self.molecule_index,
            "rdkit_valid": int(self.rdkit_valid),
            "atom_count_match": int(self.atom_count_match),
            "atom_symbols_match": int(self.atom_symbols_match),
            "attachment_match": int(self.attachment_match),
            "graph_signature_match": int(self.graph_signature_match),
            "ring_profile_match": int(self.ring_profile_match),
            "dual_success": int(self.dual_success),
            "failure_type": self.failure_type,
            "errors": ";".join(self.errors),
        }
        row.update(self.left.to_dict("left"))
        row.update(self.right.to_dict("right"))
        return row


def find_phenolic_oh_sites(mol: Chem.Mol) -> list[PhenolicOHSite]:
    """Find phenolic OH groups as aromatic-carbon--oxygen--hydrogen sites."""
    sites: list[PhenolicOHSite] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 8:
            continue
        aromatic_carbons = [
            neighbor
            for neighbor in atom.GetNeighbors()
            if neighbor.GetAtomicNum() == 6 and neighbor.GetIsAromatic()
        ]
        if len(aromatic_carbons) != 1:
            continue
        hydrogen_neighbors = [neighbor for neighbor in atom.GetNeighbors() if neighbor.GetAtomicNum() == 1]
        implicit_or_explicit_h = atom.GetTotalNumHs() > 0
        if not hydrogen_neighbors and not implicit_or_explicit_h:
            continue
        hydrogen_idx = hydrogen_neighbors[0].GetIdx() if hydrogen_neighbors else None
        sites.append(
            PhenolicOHSite(
                core_atom_idx=aromatic_carbons[0].GetIdx(),
                oxygen_atom_idx=atom.GetIdx(),
                hydrogen_atom_idx=hydrogen_idx,
            )
        )
    return sorted(sites, key=lambda site: (site.core_atom_idx, site.oxygen_atom_idx))


def build_dual_substitution_spec_from_phenolic_sites(
    sites: list[PhenolicOHSite],
    *,
    left_index: int = 0,
    right_index: int = 1,
    max_branch_atoms: int | None = None,
) -> DualSubstitutionSpec:
    """Build a dual-site spec from two detected phenolic OH sites."""
    if len(sites) < 2:
        raise ValueError(f"At least two phenolic OH sites are required, found {len(sites)}")
    if left_index == right_index:
        raise ValueError("left_index and right_index must differ")
    try:
        left_site = sites[left_index]
        right_site = sites[right_index]
    except IndexError as exc:
        raise ValueError(
            f"Phenolic site indices out of range: left={left_index}, right={right_index}, found={len(sites)}"
        ) from exc
    return DualSubstitutionSpec.from_phenolic_sites(
        left_site,
        right_site,
        max_branch_atoms=max_branch_atoms,
    )


def audit_dual_sidechain(
    mol: Chem.Mol | None,
    spec: DualSubstitutionSpec,
    *,
    molecule_index: int = 0,
    ignore_leaving_atoms: bool = False,
) -> DualSidechainAuditResult:
    """Audit whether two aromatic substituent branches have matching chemistry."""
    if mol is None:
        empty_left = _failed_branch("R1", spec.left_core_idx, "mol_none")
        empty_right = _failed_branch("R2", spec.right_core_idx, "mol_none")
        return DualSidechainAuditResult(
            molecule_index=molecule_index,
            rdkit_valid=False,
            left=empty_left,
            right=empty_right,
            failure_type="mol_none",
            errors=("mol is None",),
        )

    errors: list[str] = []
    rdkit_valid = True
    working = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(working)
    except Exception as exc:  # noqa: BLE001 - audit records RDKit failures.
        rdkit_valid = False
        errors.append(f"sanitize:{type(exc).__name__}:{exc}")

    left = extract_sidechain_branch(
        working,
        core_atom_idx=spec.left_core_idx,
        label=spec.left_label,
        leaving_atom_idx=spec.left_leaving_atom_idx,
        ignore_leaving_atom=ignore_leaving_atoms,
        max_branch_atoms=spec.max_branch_atoms,
    )
    right = extract_sidechain_branch(
        working,
        core_atom_idx=spec.right_core_idx,
        label=spec.right_label,
        leaving_atom_idx=spec.right_leaving_atom_idx,
        ignore_leaving_atom=ignore_leaving_atoms,
        max_branch_atoms=spec.max_branch_atoms,
    )

    atom_count_match = left.extractable and right.extractable and left.atom_count == right.atom_count
    atom_symbols_match = left.extractable and right.extractable and left.atom_symbols == right.atom_symbols
    attachment_match = (
        left.extractable
        and right.extractable
        and left.attachment_bond_type == right.attachment_bond_type
        and left.attachment_is_aromatic == right.attachment_is_aromatic
    )
    graph_signature_match = left.extractable and right.extractable and left.signature == right.signature
    ring_profile_match = (
        left.extractable
        and right.extractable
        and left.ring_count == right.ring_count
        and left.aromatic_ring_count == right.aromatic_ring_count
        and left.three_member_ring_count == right.three_member_ring_count
        and left.max_fused_ring_system_size == right.max_fused_ring_system_size
    )

    failure_type = _audit_failure_type(
        rdkit_valid=rdkit_valid,
        left=left,
        right=right,
        atom_symbols_match=atom_symbols_match,
        attachment_match=attachment_match,
        graph_signature_match=graph_signature_match,
        ring_profile_match=ring_profile_match,
    )
    dual_success = failure_type == "success"
    return DualSidechainAuditResult(
        molecule_index=molecule_index,
        rdkit_valid=rdkit_valid,
        left=left,
        right=right,
        atom_count_match=atom_count_match,
        atom_symbols_match=atom_symbols_match,
        attachment_match=attachment_match,
        graph_signature_match=graph_signature_match,
        ring_profile_match=ring_profile_match,
        dual_success=dual_success,
        failure_type=failure_type,
        errors=tuple(errors),
    )


def extract_sidechain_branch(
    mol: Chem.Mol,
    *,
    core_atom_idx: int,
    label: str,
    leaving_atom_idx: int | None = None,
    ignore_leaving_atom: bool = False,
    max_branch_atoms: int | None = None,
) -> SidechainBranchFacts:
    """Extract the non-aromatic substituent branch directly attached to one core atom."""
    if core_atom_idx < 0 or core_atom_idx >= mol.GetNumAtoms():
        return _failed_branch(label, core_atom_idx, "core_out_of_range")
    core_atom = mol.GetAtomWithIdx(core_atom_idx)
    candidate_roots = _candidate_substituent_roots(
        core_atom,
        leaving_atom_idx=leaving_atom_idx,
        ignore_leaving_atom=ignore_leaving_atom,
    )
    if not candidate_roots:
        return _failed_branch(label, core_atom_idx, "attachment_missing")
    if len(candidate_roots) > 1:
        return _failed_branch(label, core_atom_idx, "multiple_substituent_roots")

    root_idx = candidate_roots[0]
    branch_atoms = _collect_branch_atoms(mol, core_atom_idx=core_atom_idx, root_atom_idx=root_idx)
    if max_branch_atoms is not None and len(branch_atoms) > max_branch_atoms:
        return _failed_branch(label, core_atom_idx, "branch_too_large", root_atom_idx=root_idx)

    atom_indices = tuple(sorted(branch_atoms))
    atom_symbols = tuple(mol.GetAtomWithIdx(idx).GetSymbol() for idx in atom_indices)
    attachment_bond = mol.GetBondBetweenAtoms(core_atom_idx, root_idx)
    attachment_bond_type = _bond_type_name(attachment_bond)
    attachment_is_aromatic = bool(attachment_bond.GetIsAromatic()) if attachment_bond is not None else False
    internal_bonds = _internal_bond_signature(mol, atom_indices)
    ring_facts = _ring_facts(mol, set(atom_indices))
    signature = _branch_signature(
        mol=mol,
        atom_indices=atom_indices,
        root_idx=root_idx,
        internal_bonds=internal_bonds,
    )
    return SidechainBranchFacts(
        label=label,
        core_atom_idx=core_atom_idx,
        root_atom_idx=root_idx,
        atom_indices=atom_indices,
        atom_symbols=atom_symbols,
        internal_bonds=internal_bonds,
        attachment_bond_type=attachment_bond_type,
        attachment_is_aromatic=attachment_is_aromatic,
        ring_count=ring_facts["ring_count"],
        aromatic_ring_count=ring_facts["aromatic_ring_count"],
        three_member_ring_count=ring_facts["three_member_ring_count"],
        max_fused_ring_system_size=ring_facts["max_fused_ring_system_size"],
        signature=signature,
        extractable=True,
        failure_type="success",
    )


def summarize_audit_results(results: list[DualSidechainAuditResult]) -> dict[str, Any]:
    """Summarize a batch of dual-sidechain audit results."""
    total = len(results)
    if total == 0:
        return {"total": 0}
    failure_counts: dict[str, int] = {}
    for result in results:
        failure_counts[result.failure_type] = failure_counts.get(result.failure_type, 0) + 1
    return {
        "total": total,
        "rdkit_valid": sum(int(result.rdkit_valid) for result in results),
        "rdkit_valid_rate": _rate(sum(int(result.rdkit_valid) for result in results), total),
        "left_extractable": sum(int(result.left.extractable) for result in results),
        "right_extractable": sum(int(result.right.extractable) for result in results),
        "atom_symbols_match": sum(int(result.atom_symbols_match) for result in results),
        "attachment_match": sum(int(result.attachment_match) for result in results),
        "graph_signature_match": sum(int(result.graph_signature_match) for result in results),
        "ring_profile_match": sum(int(result.ring_profile_match) for result in results),
        "dual_success": sum(int(result.dual_success) for result in results),
        "dual_success_rate": _rate(sum(int(result.dual_success) for result in results), total),
        "failure_counts": failure_counts,
    }


def _candidate_substituent_roots(
    core_atom: Chem.Atom,
    *,
    leaving_atom_idx: int | None,
    ignore_leaving_atom: bool,
) -> list[int]:
    """Return direct non-ring substituent roots attached to an aromatic core."""
    roots: list[int] = []
    for neighbor in core_atom.GetNeighbors():
        neighbor_idx = neighbor.GetIdx()
        if ignore_leaving_atom and leaving_atom_idx is not None and neighbor_idx == leaving_atom_idx:
            continue
        if neighbor.GetAtomicNum() == 1:
            continue
        bond = core_atom.GetOwningMol().GetBondBetweenAtoms(core_atom.GetIdx(), neighbor_idx)
        if neighbor.GetIsAromatic() and bond is not None and bond.GetIsAromatic():
            continue
        roots.append(neighbor_idx)
    return sorted(roots)


def _collect_branch_atoms(mol: Chem.Mol, *, core_atom_idx: int, root_atom_idx: int) -> set[int]:
    """Collect atoms reachable from root without crossing back through the aromatic core."""
    visited: set[int] = set()
    queue: deque[int] = deque([root_atom_idx])
    while queue:
        atom_idx = queue.popleft()
        if atom_idx in visited:
            continue
        visited.add(atom_idx)
        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx == core_atom_idx or neighbor_idx in visited:
                continue
            queue.append(neighbor_idx)
    return visited


def _internal_bond_signature(
    mol: Chem.Mol,
    atom_indices: tuple[int, ...],
) -> tuple[tuple[int, int, str, bool], ...]:
    """Return sorted internal bond facts using local atom ordering."""
    index_to_local = {atom_idx: local_idx for local_idx, atom_idx in enumerate(atom_indices)}
    bonds: list[tuple[int, int, str, bool]] = []
    atom_set = set(atom_indices)
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if begin not in atom_set or end not in atom_set:
            continue
        local_begin = index_to_local[begin]
        local_end = index_to_local[end]
        if local_begin > local_end:
            local_begin, local_end = local_end, local_begin
        bonds.append((local_begin, local_end, _bond_type_name(bond), bool(bond.GetIsAromatic())))
    return tuple(sorted(bonds))


def _branch_signature(
    *,
    mol: Chem.Mol,
    atom_indices: tuple[int, ...],
    root_idx: int,
    internal_bonds: tuple[tuple[int, int, str, bool], ...],
) -> str:
    """Build a deterministic rooted branch signature."""
    try:
        return Chem.MolFragmentToSmiles(
            mol,
            atomsToUse=list(atom_indices),
            rootedAtAtom=int(root_idx),
            canonical=True,
            isomericSmiles=False,
        )
    except Exception:  # noqa: BLE001 - fall back to an explicit structural signature.
        pass
    atom_parts = []
    for local_idx, atom_idx in enumerate(atom_indices):
        atom = mol.GetAtomWithIdx(atom_idx)
        root_flag = "root" if atom_idx == root_idx else "node"
        atom_parts.append(
            f"{local_idx}:{root_flag}:{atom.GetSymbol()}:{int(atom.GetFormalCharge())}:"
            f"{int(atom.GetIsAromatic())}"
        )
    bond_parts = [
        f"{begin}-{end}:{bond_type}:{int(is_aromatic)}"
        for begin, end, bond_type, is_aromatic in internal_bonds
    ]
    return "|".join(atom_parts) + "||" + "|".join(bond_parts)


def _ring_facts(mol: Chem.Mol, atom_indices: set[int]) -> dict[str, int]:
    """Return simple ring descriptors for rings fully contained in a branch."""
    rings = [tuple(ring) for ring in mol.GetRingInfo().AtomRings() if set(ring).issubset(atom_indices)]
    aromatic_ring_count = 0
    for ring in rings:
        if all(mol.GetAtomWithIdx(atom_idx).GetIsAromatic() for atom_idx in ring):
            aromatic_ring_count += 1
    return {
        "ring_count": len(rings),
        "aromatic_ring_count": aromatic_ring_count,
        "three_member_ring_count": sum(1 for ring in rings if len(ring) == 3),
        "max_fused_ring_system_size": _max_fused_ring_system_size(rings),
    }


def _max_fused_ring_system_size(rings: list[tuple[int, ...]]) -> int:
    """Return the maximum number of fused rings in one connected ring system."""
    if not rings:
        return 0
    ring_sets = [set(ring) for ring in rings]
    adjacency = {idx: set() for idx in range(len(ring_sets))}
    for i, ring_i in enumerate(ring_sets):
        for j in range(i + 1, len(ring_sets)):
            if ring_i.intersection(ring_sets[j]):
                adjacency[i].add(j)
                adjacency[j].add(i)
    seen: set[int] = set()
    max_size = 0
    for idx in range(len(ring_sets)):
        if idx in seen:
            continue
        queue: deque[int] = deque([idx])
        size = 0
        while queue:
            current = queue.popleft()
            if current in seen:
                continue
            seen.add(current)
            size += 1
            queue.extend(adjacency[current] - seen)
        max_size = max(max_size, size)
    return max_size


def _audit_failure_type(
    *,
    rdkit_valid: bool,
    left: SidechainBranchFacts,
    right: SidechainBranchFacts,
    atom_symbols_match: bool,
    attachment_match: bool,
    graph_signature_match: bool,
    ring_profile_match: bool,
) -> str:
    """Return the first failure reason in priority order."""
    if not rdkit_valid:
        return "invalid_mol"
    if not left.extractable:
        return f"left_{left.failure_type}"
    if not right.extractable:
        return f"right_{right.failure_type}"
    if not atom_symbols_match:
        return "atom_symbols_mismatch"
    if not attachment_match:
        return "attachment_mismatch"
    if not graph_signature_match:
        return "graph_signature_mismatch"
    if not ring_profile_match:
        return "ring_profile_mismatch"
    return "success"


def _failed_branch(
    label: str,
    core_atom_idx: int,
    failure_type: str,
    *,
    root_atom_idx: int | None = None,
) -> SidechainBranchFacts:
    """Create a failed branch fact row."""
    return SidechainBranchFacts(
        label=label,
        core_atom_idx=core_atom_idx,
        root_atom_idx=root_atom_idx,
        extractable=False,
        failure_type=failure_type,
    )


def _bond_type_name(bond: Chem.Bond | None) -> str:
    """Return a stable bond type name."""
    if bond is None:
        return ""
    return BOND_TYPE_NAMES.get(bond.GetBondType(), str(bond.GetBondType()).lower())


def _rate(count: int, total: int) -> float:
    """Return a rounded percentage-like fraction."""
    if total <= 0:
        return 0.0
    return round(float(count) / float(total), 6)
