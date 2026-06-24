"""Core mapping helpers for generated dual-R candidate audits."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rdkit import Chem
from rdkit.Chem import rdFMCS

from ddpo.constraints.dual_sidechain.spec import DualSubstitutionSpec


_ORIGINAL_ATOM_INDEX_PROP = "_dual_sidechain_original_atom_idx"


@dataclass(frozen=True)
class CoreMappingQuery:
    """Heavy-atom scaffold query used to locate honokiol core atoms in targets."""

    query_mol: Chem.Mol
    site_reference_spec: DualSubstitutionSpec
    left_query_core_idx: int
    right_query_core_idx: int
    reference_to_query_atom_indices: dict[int, int] = field(default_factory=dict)
    query_to_reference_atom_indices: dict[int, int] = field(default_factory=dict)
    removed_atom_indices: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "query_num_atoms": int(self.query_mol.GetNumAtoms()),
            "query_num_bonds": int(self.query_mol.GetNumBonds()),
            "left_query_core_idx": int(self.left_query_core_idx),
            "right_query_core_idx": int(self.right_query_core_idx),
            "site_reference_spec": self.site_reference_spec.to_dict(),
            "removed_atom_indices": [int(idx) for idx in self.removed_atom_indices],
            "reference_to_query_atom_indices": {
                str(key): int(value) for key, value in self.reference_to_query_atom_indices.items()
            },
            "query_to_reference_atom_indices": {
                str(key): int(value) for key, value in self.query_to_reference_atom_indices.items()
            },
        }


@dataclass(frozen=True)
class CoreMappingResult:
    """Result of mapping one target molecule to a dual-R scaffold query."""

    molecule_index: int
    rdkit_valid: bool
    core_found: bool
    target_spec: DualSubstitutionSpec | None = None
    target_left_core_idx: int | None = None
    target_right_core_idx: int | None = None
    match_count: int = 0
    selected_match_index: int | None = None
    selected_match: tuple[int, ...] = ()
    failure_type: str = ""
    errors: tuple[str, ...] = ()
    mcs_num_atoms: int | None = None
    mcs_num_bonds: int | None = None
    mcs_atom_fraction: float | None = None
    mcs_canceled: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return flat fields for CSV/JSON export."""
        return {
            "molecule_index": self.molecule_index,
            "mapping_rdkit_valid": int(self.rdkit_valid),
            "core_found": int(self.core_found),
            "target_left_core_idx": "" if self.target_left_core_idx is None else int(self.target_left_core_idx),
            "target_right_core_idx": "" if self.target_right_core_idx is None else int(self.target_right_core_idx),
            "core_match_count": int(self.match_count),
            "selected_match_index": "" if self.selected_match_index is None else int(self.selected_match_index),
            "selected_match": " ".join(str(idx) for idx in self.selected_match),
            "mapping_failure_type": self.failure_type,
            "mapping_errors": ";".join(self.errors),
            "mcs_num_atoms": "" if self.mcs_num_atoms is None else int(self.mcs_num_atoms),
            "mcs_num_bonds": "" if self.mcs_num_bonds is None else int(self.mcs_num_bonds),
            "mcs_atom_fraction": "" if self.mcs_atom_fraction is None else float(self.mcs_atom_fraction),
            "mcs_canceled": "" if self.mcs_canceled is None else int(self.mcs_canceled),
        }


def build_core_mapping_query(
    reference_mol: Chem.Mol,
    spec: DualSubstitutionSpec,
    *,
    sanitize: bool = True,
) -> CoreMappingQuery:
    """Build a heavy-atom core query by deleting leaving groups and hydrogens."""
    working = Chem.Mol(reference_mol)
    if sanitize:
        Chem.SanitizeMol(working)
    heavy_mol, reference_to_heavy = _remove_hs_with_original_index_map(working)
    remove_heavy_indices = {
        reference_to_heavy[leaving_idx]
        for leaving_idx in (spec.left_leaving_atom_idx, spec.right_leaving_atom_idx)
        if leaving_idx is not None and leaving_idx in reference_to_heavy
    }
    heavy_to_query = _old_to_new_index_map(heavy_mol.GetNumAtoms(), remove_heavy_indices)
    reference_to_query = {
        reference_idx: heavy_to_query[heavy_idx]
        for reference_idx, heavy_idx in reference_to_heavy.items()
        if heavy_idx in heavy_to_query
    }
    try:
        left_query_core_idx = reference_to_query[spec.left_core_idx]
        right_query_core_idx = reference_to_query[spec.right_core_idx]
    except KeyError as exc:
        raise ValueError("A core atom was removed while building the mapping query") from exc

    rw_mol = Chem.RWMol(heavy_mol)
    for atom_idx in sorted(remove_heavy_indices, reverse=True):
        rw_mol.RemoveAtom(atom_idx)
    query_mol = rw_mol.GetMol()
    _relax_attachment_core_atoms(query_mol, (left_query_core_idx, right_query_core_idx))
    if sanitize:
        Chem.SanitizeMol(query_mol)
    query_to_reference = {query_idx: ref_idx for ref_idx, query_idx in reference_to_query.items()}
    removed_reference_indices = _core_query_removed_atoms(working, spec)
    return CoreMappingQuery(
        query_mol=query_mol,
        site_reference_spec=spec,
        left_query_core_idx=left_query_core_idx,
        right_query_core_idx=right_query_core_idx,
        reference_to_query_atom_indices=reference_to_query,
        query_to_reference_atom_indices=query_to_reference,
        removed_atom_indices=tuple(sorted(removed_reference_indices)),
    )


def map_core_to_target(
    target_mol: Chem.Mol | None,
    query: CoreMappingQuery,
    *,
    molecule_index: int = 0,
    sanitize: bool = True,
    max_matches: int = 1000,
    compute_mcs: bool = False,
    mcs_timeout: int = 2,
) -> CoreMappingResult:
    """Map the reference core query onto one target molecule."""
    if target_mol is None:
        return CoreMappingResult(
            molecule_index=molecule_index,
            rdkit_valid=False,
            core_found=False,
            failure_type="mol_none",
            errors=("mol is None",),
        )

    errors: list[str] = []
    rdkit_valid = True
    working = Chem.Mol(target_mol)
    if sanitize:
        try:
            Chem.SanitizeMol(working)
        except Exception as exc:  # noqa: BLE001 - audit-style result keeps RDKit failure text.
            rdkit_valid = False
            errors.append(f"sanitize:{type(exc).__name__}:{exc}")
            return _failed_mapping_result(
                molecule_index=molecule_index,
                rdkit_valid=False,
                failure_type="invalid_mol",
                errors=tuple(errors),
                query=query,
                target_mol=working,
                compute_mcs=compute_mcs,
                mcs_timeout=mcs_timeout,
            )

    target_heavy, heavy_to_target = _remove_hs_with_original_index_map(working)
    matches = target_heavy.GetSubstructMatches(
        query.query_mol,
        useChirality=False,
        uniquify=True,
        maxMatches=int(max_matches),
    )
    if not matches:
        return _failed_mapping_result(
            molecule_index=molecule_index,
            rdkit_valid=rdkit_valid,
            failure_type="core_no_match",
            errors=tuple(errors),
            query=query,
            target_mol=target_heavy,
            compute_mcs=compute_mcs,
            mcs_timeout=mcs_timeout,
        )

    selected_index, selected_match = _select_core_match(matches, query)
    selected_match_target = tuple(int(heavy_to_target[idx]) for idx in selected_match)
    left_core = int(heavy_to_target[selected_match[query.left_query_core_idx]])
    right_core = int(heavy_to_target[selected_match[query.right_query_core_idx]])
    target_spec = DualSubstitutionSpec(
        left_core_idx=left_core,
        right_core_idx=right_core,
        left_leaving_atom_idx=None,
        right_leaving_atom_idx=None,
        left_label=query.site_reference_spec.left_label,
        right_label=query.site_reference_spec.right_label,
        max_branch_atoms=query.site_reference_spec.max_branch_atoms,
    )
    return CoreMappingResult(
        molecule_index=molecule_index,
        rdkit_valid=rdkit_valid,
        core_found=True,
        target_spec=target_spec,
        target_left_core_idx=left_core,
        target_right_core_idx=right_core,
        match_count=len(matches),
        selected_match_index=selected_index,
        selected_match=selected_match_target,
        failure_type="success",
        errors=tuple(errors),
    )


def _failed_mapping_result(
    *,
    molecule_index: int,
    rdkit_valid: bool,
    failure_type: str,
    errors: tuple[str, ...],
    query: CoreMappingQuery,
    target_mol: Chem.Mol,
    compute_mcs: bool,
    mcs_timeout: int,
) -> CoreMappingResult:
    """Create a failed mapping result with optional MCS diagnostics."""
    mcs_summary = _mcs_summary(query.query_mol, target_mol, timeout=mcs_timeout) if compute_mcs else {}
    return CoreMappingResult(
        molecule_index=molecule_index,
        rdkit_valid=rdkit_valid,
        core_found=False,
        failure_type=failure_type,
        errors=errors,
        mcs_num_atoms=mcs_summary.get("mcs_num_atoms"),
        mcs_num_bonds=mcs_summary.get("mcs_num_bonds"),
        mcs_atom_fraction=mcs_summary.get("mcs_atom_fraction"),
        mcs_canceled=mcs_summary.get("mcs_canceled"),
    )


def _core_query_removed_atoms(mol: Chem.Mol, spec: DualSubstitutionSpec) -> set[int]:
    """Return hydrogens and leaving atoms removed from the mapping query."""
    remove_indices = {atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 1}
    for leaving_idx in (spec.left_leaving_atom_idx, spec.right_leaving_atom_idx):
        if leaving_idx is not None and 0 <= leaving_idx < mol.GetNumAtoms():
            remove_indices.add(leaving_idx)
    return remove_indices


def _remove_hs_with_original_index_map(mol: Chem.Mol) -> tuple[Chem.Mol, dict[int, int]]:
    """Remove hydrogens while preserving original heavy-atom index mapping."""
    tagged = Chem.Mol(mol)
    for atom in tagged.GetAtoms():
        atom.SetIntProp(_ORIGINAL_ATOM_INDEX_PROP, int(atom.GetIdx()))
    heavy_mol = Chem.RemoveHs(tagged, sanitize=False)
    original_to_heavy = {}
    for atom in heavy_mol.GetAtoms():
        if atom.HasProp(_ORIGINAL_ATOM_INDEX_PROP):
            original_to_heavy[int(atom.GetIntProp(_ORIGINAL_ATOM_INDEX_PROP))] = int(atom.GetIdx())
        else:
            original_to_heavy[int(atom.GetIdx())] = int(atom.GetIdx())
    return heavy_mol, original_to_heavy


def _relax_attachment_core_atoms(query_mol: Chem.Mol, core_atom_indices: tuple[int, int]) -> None:
    """Reset attachment atoms after deleting leaving groups so they behave as normal scaffold atoms."""
    for atom_idx in core_atom_indices:
        if atom_idx < 0 or atom_idx >= query_mol.GetNumAtoms():
            continue
        atom = query_mol.GetAtomWithIdx(atom_idx)
        atom.SetNoImplicit(False)
        atom.SetNumExplicitHs(0)
        atom.SetNumRadicalElectrons(0)
    query_mol.UpdatePropertyCache(strict=False)


def _old_to_new_index_map(num_atoms: int, remove_indices: set[int]) -> dict[int, int]:
    """Map atom indices after deleting a set of atoms."""
    mapping: dict[int, int] = {}
    removed_before = 0
    for old_idx in range(num_atoms):
        if old_idx in remove_indices:
            removed_before += 1
            continue
        mapping[old_idx] = old_idx - removed_before
    return mapping


def _select_core_match(
    matches: tuple[tuple[int, ...], ...],
    query: CoreMappingQuery,
) -> tuple[int, tuple[int, ...]]:
    """Select one deterministic core match from possible symmetry matches."""
    indexed_matches = list(enumerate(matches))
    indexed_matches.sort(
        key=lambda item: (
            item[1][query.left_query_core_idx],
            item[1][query.right_query_core_idx],
            item[1],
        )
    )
    return indexed_matches[0]


def _mcs_summary(query_mol: Chem.Mol, target_mol: Chem.Mol, *, timeout: int) -> dict[str, Any]:
    """Return a lightweight MCS diagnostic for core mapping failures."""
    try:
        target_no_h = Chem.RemoveHs(target_mol, sanitize=False)
        result = rdFMCS.FindMCS(
            [query_mol, target_no_h],
            timeout=max(1, int(timeout)),
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
            matchValences=False,
        )
    except Exception:  # noqa: BLE001 - MCS is diagnostic only.
        return {}
    query_atoms = max(1, int(query_mol.GetNumAtoms()))
    return {
        "mcs_num_atoms": int(result.numAtoms),
        "mcs_num_bonds": int(result.numBonds),
        "mcs_atom_fraction": round(float(result.numAtoms) / float(query_atoms), 6),
        "mcs_canceled": bool(result.canceled),
    }
