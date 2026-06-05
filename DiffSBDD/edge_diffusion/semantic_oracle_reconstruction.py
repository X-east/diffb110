"""Semantic-aware oracle reconstruction from atom, halfedge, and SDF metadata."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from rdkit import Chem
from rdkit.Geometry import Point3D

from DiffSBDD.edge_diffusion.oracle_reconstruction import (
    OracleReconstructionResult,
    sanitize_with_repairs,
)
from DiffSBDD.edge_utils import RDKIT_BOND_TYPE_BY_ID


def build_semantic_oracle_mol(
    atom_symbols: list[str],
    positions: torch.Tensor,
    halfedge_index: torch.Tensor,
    halfedge_type: torch.Tensor,
    formal_charges: Sequence[int] | torch.Tensor | None = None,
    total_hs: Sequence[int] | torch.Tensor | None = None,
    explicit_hs: Sequence[int] | torch.Tensor | None = None,
    no_implicit_flags: Sequence[int] | torch.Tensor | None = None,
    aromatic_atom_flags: Sequence[int] | torch.Tensor | None = None,
    chiral_tags: Sequence[int] | torch.Tensor | None = None,
    bond_pairs: Sequence[Sequence[int]] | torch.Tensor | None = None,
    bond_stereo: Sequence[int] | torch.Tensor | None = None,
) -> OracleReconstructionResult:
    """Build an RDKit molecule using true edges plus original SDF atom semantics."""
    if any(symbol == "others" for symbol in atom_symbols):
        return OracleReconstructionResult(
            mol=None,
            sanitize_error="unsupported_atom_type",
            initial_sanitize_error="unsupported_atom_type",
            repair_strategy="unsupported_atom_type",
        )
    metadata = _prepare_metadata(
        num_atoms=len(atom_symbols),
        formal_charges=formal_charges,
        total_hs=total_hs,
        explicit_hs=explicit_hs,
        no_implicit_flags=no_implicit_flags,
        aromatic_atom_flags=aromatic_atom_flags,
        chiral_tags=chiral_tags,
        bond_pairs=bond_pairs,
        bond_stereo=bond_stereo,
    )
    mol = _build_semantic_raw_mol(
        atom_symbols=atom_symbols,
        positions=positions,
        halfedge_index=halfedge_index,
        halfedge_type=halfedge_type,
        metadata=metadata,
    )
    return sanitize_with_repairs(mol)


def _build_semantic_raw_mol(
    atom_symbols: list[str],
    positions: torch.Tensor,
    halfedge_index: torch.Tensor,
    halfedge_type: torch.Tensor,
    metadata: dict[str, object],
) -> Chem.Mol:
    """Build an unsanitized RDKit molecule with SDF-derived atom semantics."""
    rw_mol = Chem.RWMol()
    aromatic_from_edges = _aromatic_atoms_from_edges(halfedge_index, halfedge_type)
    aromatic_atom_flags = metadata["aromatic_atom_flags"]
    formal_charges = metadata["formal_charges"]
    total_hs = metadata["total_hs"]
    explicit_hs = metadata["explicit_hs"]
    no_implicit_flags = metadata["no_implicit_flags"]
    chiral_tags = metadata["chiral_tags"]

    for atom_idx, symbol in enumerate(atom_symbols):
        atom = Chem.Atom(symbol)
        atom.SetFormalCharge(int(formal_charges[atom_idx]))
        if int(aromatic_atom_flags[atom_idx]) or atom_idx in aromatic_from_edges:
            atom.SetIsAromatic(True)

        explicit_h_count = int(explicit_hs[atom_idx])
        total_h_count = int(total_hs[atom_idx])
        no_implicit = bool(int(no_implicit_flags[atom_idx]))
        if explicit_h_count > 0:
            atom.SetNumExplicitHs(explicit_h_count)
        elif no_implicit and total_h_count > 0:
            atom.SetNumExplicitHs(total_h_count)
        atom.SetNoImplicit(no_implicit)
        _set_chiral_tag(atom, int(chiral_tags[atom_idx]))
        rw_mol.AddAtom(atom)

    bond_stereo_by_pair = metadata["bond_stereo_by_pair"]
    for edge_idx, edge_type_tensor in enumerate(halfedge_type):
        edge_type = int(edge_type_tensor)
        if edge_type <= 0:
            continue
        atom_i = int(halfedge_index[0, edge_idx])
        atom_j = int(halfedge_index[1, edge_idx])
        rdkit_bond_type = RDKIT_BOND_TYPE_BY_ID[edge_type]
        rw_mol.AddBond(atom_i, atom_j, rdkit_bond_type)
        bond = rw_mol.GetBondBetweenAtoms(atom_i, atom_j)
        if bond is not None:
            if edge_type == 4:
                bond.SetIsAromatic(True)
            stereo_value = bond_stereo_by_pair.get(tuple(sorted((atom_i, atom_j))))
            if stereo_value is not None:
                _set_bond_stereo(bond, int(stereo_value))

    mol = rw_mol.GetMol()
    conformer = Chem.Conformer(len(atom_symbols))
    for atom_idx, coord in enumerate(positions.cpu().tolist()):
        conformer.SetAtomPosition(atom_idx, Point3D(*map(float, coord)))
    mol.AddConformer(conformer)
    return mol


def _prepare_metadata(
    num_atoms: int,
    formal_charges: Sequence[int] | torch.Tensor | None,
    total_hs: Sequence[int] | torch.Tensor | None,
    explicit_hs: Sequence[int] | torch.Tensor | None,
    no_implicit_flags: Sequence[int] | torch.Tensor | None,
    aromatic_atom_flags: Sequence[int] | torch.Tensor | None,
    chiral_tags: Sequence[int] | torch.Tensor | None,
    bond_pairs: Sequence[Sequence[int]] | torch.Tensor | None,
    bond_stereo: Sequence[int] | torch.Tensor | None,
) -> dict[str, object]:
    """Coerce optional metadata arrays into validated integer lists."""
    bond_pair_list = _coerce_bond_pairs(bond_pairs)
    bond_stereo_list = _coerce_ints(bond_stereo, len(bond_pair_list), default=0)
    return {
        "formal_charges": _coerce_ints(formal_charges, num_atoms, default=0),
        "total_hs": _coerce_ints(total_hs, num_atoms, default=0),
        "explicit_hs": _coerce_ints(explicit_hs, num_atoms, default=0),
        "no_implicit_flags": _coerce_ints(no_implicit_flags, num_atoms, default=0),
        "aromatic_atom_flags": _coerce_ints(aromatic_atom_flags, num_atoms, default=0),
        "chiral_tags": _coerce_ints(chiral_tags, num_atoms, default=0),
        "bond_stereo_by_pair": {
            tuple(sorted(pair)): int(stereo)
            for pair, stereo in zip(bond_pair_list, bond_stereo_list)
        },
    }


def _coerce_ints(
    values: Sequence[int] | torch.Tensor | None,
    expected_length: int,
    default: int,
) -> list[int]:
    """Coerce optional int metadata into a fixed-length list."""
    if values is None:
        return [default for _ in range(expected_length)]
    if isinstance(values, torch.Tensor):
        raw_values = values.detach().cpu().reshape(-1).tolist()
    else:
        raw_values = list(values)
    if len(raw_values) != expected_length:
        raise ValueError(
            f"metadata length mismatch: expected {expected_length}, got {len(raw_values)}"
        )
    return [int(value) for value in raw_values]


def _coerce_bond_pairs(
    bond_pairs: Sequence[Sequence[int]] | torch.Tensor | None,
) -> list[tuple[int, int]]:
    """Coerce optional bond pair metadata into sorted integer pairs."""
    if bond_pairs is None:
        return []
    if isinstance(bond_pairs, torch.Tensor):
        raw_pairs = bond_pairs.detach().cpu().reshape(-1, 2).tolist()
    else:
        raw_pairs = list(bond_pairs)
    return [tuple(sorted((int(pair[0]), int(pair[1])))) for pair in raw_pairs]


def _aromatic_atoms_from_edges(
    halfedge_index: torch.Tensor,
    halfedge_type: torch.Tensor,
) -> set[int]:
    """Return atoms attached to aromatic halfedges."""
    aromatic_atoms = set()
    for edge_idx, edge_type_tensor in enumerate(halfedge_type):
        if int(edge_type_tensor) != 4:
            continue
        aromatic_atoms.add(int(halfedge_index[0, edge_idx]))
        aromatic_atoms.add(int(halfedge_index[1, edge_idx]))
    return aromatic_atoms


def _set_chiral_tag(atom: Chem.Atom, tag_value: int) -> None:
    """Set an RDKit chiral tag when the integer value is recognized."""
    if tag_value == 0:
        return
    try:
        atom.SetChiralTag(Chem.rdchem.ChiralType.values[tag_value])
    except KeyError:
        return


def _set_bond_stereo(bond: Chem.Bond, stereo_value: int) -> None:
    """Set RDKit bond stereo when the integer value is recognized."""
    if stereo_value == 0:
        return
    try:
        bond.SetStereo(Chem.rdchem.BondStereo.values[stereo_value])
    except KeyError:
        return
