"""RDKit oracle reconstruction from ligand atom and halfedge labels."""

from __future__ import annotations

import itertools
import re
from copy import deepcopy
from dataclasses import dataclass

import torch
from rdkit import Chem, RDLogger
from rdkit.Geometry import Point3D

from DiffSBDD.edge_utils import RDKIT_BOND_TYPE_BY_ID

RDLogger.DisableLog("rdApp.*")


@dataclass(frozen=True)
class OracleReconstructionResult:
    """Result of reconstructing one ligand from oracle halfedge labels."""

    mol: Chem.Mol | None
    sanitize_error: str | None
    initial_sanitize_error: str | None
    repair_strategy: str

    @property
    def sanitize_success(self) -> bool:
        """Return whether final RDKit sanitization succeeded."""
        return self.sanitize_error is None


def build_oracle_mol(
    atom_symbols: list[str],
    positions: torch.Tensor,
    halfedge_index: torch.Tensor,
    halfedge_type: torch.Tensor,
) -> OracleReconstructionResult:
    """Build and repair an RDKit molecule using true halfedge labels."""
    if any(symbol == "others" for symbol in atom_symbols):
        return OracleReconstructionResult(
            mol=None,
            sanitize_error="unsupported_atom_type",
            initial_sanitize_error="unsupported_atom_type",
            repair_strategy="unsupported_atom_type",
        )

    mol = _build_raw_mol(atom_symbols, positions, halfedge_index, halfedge_type)
    return _sanitize_with_repairs(mol)


def _build_raw_mol(
    atom_symbols: list[str],
    positions: torch.Tensor,
    halfedge_index: torch.Tensor,
    halfedge_type: torch.Tensor,
) -> Chem.Mol:
    """Build an unsanitized RDKit molecule with atom coordinates and bonds."""
    rw_mol = Chem.RWMol()
    aromatic_atoms = set()
    bonds_to_add = []
    for edge_idx, edge_type_tensor in enumerate(halfedge_type):
        edge_type = int(edge_type_tensor)
        if edge_type <= 0:
            continue
        atom_i = int(halfedge_index[0, edge_idx])
        atom_j = int(halfedge_index[1, edge_idx])
        bonds_to_add.append((atom_i, atom_j, edge_type))
        if edge_type == 4:
            aromatic_atoms.update((atom_i, atom_j))

    for atom_idx, symbol in enumerate(atom_symbols):
        atom = Chem.Atom(symbol)
        if atom_idx in aromatic_atoms:
            atom.SetIsAromatic(True)
        rw_mol.AddAtom(atom)

    for atom_i, atom_j, edge_type in bonds_to_add:
        rdkit_bond_type = RDKIT_BOND_TYPE_BY_ID[edge_type]
        rw_mol.AddBond(atom_i, atom_j, rdkit_bond_type)
        if edge_type == 4:
            bond = rw_mol.GetBondBetweenAtoms(atom_i, atom_j)
            if bond is not None:
                bond.SetIsAromatic(True)

    mol = rw_mol.GetMol()
    conformer = Chem.Conformer(len(atom_symbols))
    for atom_idx, coord in enumerate(positions.cpu().tolist()):
        conformer.SetAtomPosition(atom_idx, Point3D(*map(float, coord)))
    mol.AddConformer(conformer)
    return mol


def _sanitize_with_repairs(mol: Chem.Mol) -> OracleReconstructionResult:
    """Sanitize an RDKit molecule, trying DiffGui-style repairs when needed."""
    success, error_name, _ = _try_sanitize(mol)
    if success:
        return OracleReconstructionResult(
            mol=mol,
            sanitize_error=None,
            initial_sanitize_error=None,
            repair_strategy="none",
        )

    initial_error = error_name
    repair_attempts = [
        ("fix_valence", _fix_valence),
        ("fix_aromatic", lambda candidate: _fix_aromatic(candidate, strict=False)),
        ("fix_aromatic_strict", lambda candidate: _fix_aromatic(candidate, strict=True)),
        ("fix_valence_after_aromatic", _fix_valence),
    ]
    current = mol
    for strategy, repair_fn in repair_attempts:
        repaired_mol, fixed = repair_fn(current)
        if fixed:
            return OracleReconstructionResult(
                mol=repaired_mol,
                sanitize_error=None,
                initial_sanitize_error=initial_error,
                repair_strategy=strategy,
            )
        current = repaired_mol

    _, final_error, _ = _try_sanitize(current)
    return OracleReconstructionResult(
        mol=current,
        sanitize_error=final_error or initial_error,
        initial_sanitize_error=initial_error,
        repair_strategy="failed",
    )


def _try_sanitize(mol: Chem.Mol) -> tuple[bool, str | None, str | None]:
    """Try RDKit sanitization and return success plus error details."""
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:  # noqa: BLE001 - RDKit exposes multiple exception subclasses
        return False, type(exc).__name__, str(exc)
    return True, None, None


def _find_ring_systems(mol: Chem.Mol) -> list[list[int]]:
    """Return merged ring systems with shared atoms."""
    all_rings = [list(map(int, ring)) for ring in Chem.GetSymmSSSR(mol)]
    if not all_rings:
        return []

    ring_systems = [all_rings[0]]
    for ring in all_rings[1:]:
        merged = False
        for previous in ring_systems:
            if set(ring).intersection(previous):
                previous.extend(ring)
                merged = True
                break
        if not merged:
            ring_systems.append(ring)
    return [sorted(set(ring)) for ring in ring_systems]


def _fix_valence(mol: Chem.Mol) -> tuple[Chem.Mol, bool]:
    """Repair common RDKit N valence errors by assigning formal charge."""
    candidate = deepcopy(mol)
    fixed = False
    valence_pattern = re.compile(
        r"Explicit valence for atom # ([0-9]{1,}) N, 4, is greater than permitted"
    )
    for _ in range(100):
        success, error_name, error_message = _try_sanitize(candidate)
        if success:
            fixed = True
            break
        if error_name != "AtomValenceException" or error_message is None:
            return candidate, False
        matches = valence_pattern.findall(error_message)
        if not matches:
            return candidate, False
        atom = candidate.GetAtomWithIdx(int(matches[0]))
        atom.SetFormalCharge(1)
    return candidate, fixed


def _all_subsets(atom_indices: list[int]) -> list[tuple[int, ...]]:
    """Return all subsets of atom indices."""
    subsets: list[tuple[int, ...]] = []
    for subset_size in range(len(atom_indices) + 1):
        subsets.extend(itertools.combinations(atom_indices, subset_size))
    return subsets


def _fix_aromatic(mol: Chem.Mol, strict: bool = False) -> tuple[Chem.Mol, bool]:
    """Repair aromatic N/S rings by trying explicit H or formal charges."""
    original = mol
    aromatic_atoms = {atom.GetIdx() for atom in mol.GetAromaticAtoms()}
    n_ring_options: list[list[int]] = []
    s_ring_options: list[list[int]] = []
    for ring_system in _find_ring_systems(mol):
        if not set(ring_system).intersection(aromatic_atoms):
            continue
        n_atoms = [
            atom_idx for atom_idx in ring_system
            if mol.GetAtomWithIdx(atom_idx).GetSymbol() == "N"
        ]
        s_atoms = [
            atom_idx for atom_idx in ring_system
            if mol.GetAtomWithIdx(atom_idx).GetSymbol() == "S"
        ]
        if n_atoms:
            n_ring_options.append(n_atoms + [-1])
        if s_atoms:
            s_ring_options.append(s_atoms + [-1])

    fixed = False
    s_permutations = (
        _all_subsets([idx for ring in s_ring_options for idx in ring if idx != -1])
        if strict else itertools.product(*s_ring_options)
    )
    for permutation in s_permutations:
        candidate = deepcopy(original)
        for atom_idx in [idx for idx in permutation if idx != -1]:
            candidate.GetAtomWithIdx(int(atom_idx)).SetFormalCharge(1)
        if strict:
            candidate, fixed = _fix_valence(candidate)
            if fixed:
                return candidate, True
        success, _, _ = _try_sanitize(candidate)
        if success:
            return candidate, True

    n_permutations = (
        _all_subsets([idx for ring in n_ring_options for idx in ring if idx != -1])
        if strict else itertools.product(*n_ring_options)
    )
    for permutation in n_permutations:
        active_atoms = [int(idx) for idx in permutation if idx != -1]
        for actions in itertools.product([0, 1], repeat=len(active_atoms)):
            candidate = deepcopy(original)
            for atom_idx, action in zip(active_atoms, actions):
                atom = candidate.GetAtomWithIdx(atom_idx)
                if action == 0:
                    atom.SetNumExplicitHs(1)
                    atom.SetNoImplicit(True)
                else:
                    atom.SetFormalCharge(1)
            if strict:
                candidate, fixed = _fix_valence(candidate)
                if fixed:
                    return candidate, True
            success, _, _ = _try_sanitize(candidate)
            if success:
                return candidate, True

    return original, False
