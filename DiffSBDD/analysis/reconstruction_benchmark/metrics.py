"""Metric extraction for reconstruction benchmark rows."""

from __future__ import annotations

from typing import Any

import torch
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors

from analysis.SA_Score.sascorer import calculateScore
from edge_data_oracle_report import _lookup_reference_length
from .methods import ReconstructionAttempt

RDKIT_BOND_TYPE_TO_ID = {
    Chem.BondType.SINGLE: 1,
    Chem.BondType.DOUBLE: 2,
    Chem.BondType.TRIPLE: 3,
    Chem.BondType.AROMATIC: 4,
}


def _safe_sanitize(mol: Chem.Mol | None) -> tuple[Chem.Mol | None, str]:
    """Sanitize a copy of an RDKit molecule."""
    if mol is None:
        return None, "no_mol"
    copied = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(copied)
    except Exception as exc:  # noqa: BLE001 - RDKit exposes multiple exception types.
        return copied, type(exc).__name__
    return copied, ""


def _mol_edges(mol: Chem.Mol | None) -> dict[tuple[int, int], int]:
    """Return undirected molecule bonds as compact bond type ids."""
    if mol is None:
        return {}
    edges: dict[tuple[int, int], int] = {}
    for bond in mol.GetBonds():
        edge_type = RDKIT_BOND_TYPE_TO_ID.get(bond.GetBondType(), 0)
        if edge_type <= 0:
            continue
        atom_i = int(bond.GetBeginAtomIdx())
        atom_j = int(bond.GetEndAtomIdx())
        edges[tuple(sorted((atom_i, atom_j)))] = edge_type
    return edges


def _bond_length_metrics(
    mol: Chem.Mol | None,
    atom_symbols: list[str],
    coords: torch.Tensor,
    tolerance: float,
) -> dict[str, float | int]:
    """Compare inferred bond lengths against element/bond-order references."""
    checked = 0
    bad = 0
    abs_errors: list[float] = []
    for (atom_i, atom_j), edge_type in _mol_edges(mol).items():
        ref_length = _lookup_reference_length(atom_symbols[atom_i], atom_symbols[atom_j], edge_type)
        if ref_length is None:
            continue
        length = float(torch.linalg.norm(coords[atom_i] - coords[atom_j]).item())
        abs_error = abs(length - ref_length)
        checked += 1
        bad += int(abs_error > tolerance)
        abs_errors.append(abs_error)
    return {
        "checked_bond_lengths": checked,
        "bad_bond_lengths": bad,
        "bad_bond_length_rate": bad / max(1, checked),
        "mean_bond_length_abs_error": sum(abs_errors) / len(abs_errors) if abs_errors else 0.0,
    }


def _chemistry_problem_metrics(mol: Chem.Mol | None) -> dict[str, int]:
    """Count RDKit chemistry problems without throwing on malformed molecules."""
    if mol is None:
        return {
            "chemistry_problem_count": 0,
            "valence_problem_count": 0,
        }
    try:
        problems = list(Chem.DetectChemistryProblems(mol))
    except Exception:  # noqa: BLE001 - malformed molecules can trigger RDKit internals.
        return {
            "chemistry_problem_count": 1,
            "valence_problem_count": 1,
        }
    return {
        "chemistry_problem_count": len(problems),
        "valence_problem_count": sum(int("Valence" in type(problem).__name__) for problem in problems),
    }


def _topology_metrics(mol: Chem.Mol | None, atom_symbols: list[str]) -> dict[str, float | int | str]:
    """Calculate topology and composition metrics."""
    if mol is None or mol.GetNumAtoms() == 0:
        return {
            "num_bonds": 0,
            "num_fragments": 0,
            "largest_fragment_atom_frac": 0.0,
            "ring_count": 0,
            "aromatic_ring_count": 0,
            "three_member_ring_count": 0,
            "fused_ring_pair_count": 0,
            "carbon_fraction": 0.0,
            "hetero_atom_count": 0,
            "halogen_count": 0,
            "formal_charge_abs_sum": 0,
            "smiles": "",
        }

    fragments = Chem.GetMolFrags(mol, sanitizeFrags=False)
    largest_fragment_atoms = max((len(fragment) for fragment in fragments), default=0)
    ring_info = mol.GetRingInfo()
    atom_rings = [set(ring) for ring in ring_info.AtomRings()]
    aromatic_rings = 0
    for ring in atom_rings:
        if all(mol.GetAtomWithIdx(atom_idx).GetIsAromatic() for atom_idx in ring):
            aromatic_rings += 1
    fused_ring_pairs = 0
    for left_idx, left_ring in enumerate(atom_rings):
        for right_ring in atom_rings[left_idx + 1:]:
            fused_ring_pairs += int(len(left_ring & right_ring) > 0)

    halogens = {"F", "Cl", "Br", "I"}
    carbon_count = sum(int(symbol == "C") for symbol in atom_symbols)
    hetero_count = sum(int(symbol not in {"C", "H"}) for symbol in atom_symbols)
    halogen_count = sum(int(symbol in halogens) for symbol in atom_symbols)
    formal_charge_abs_sum = sum(abs(int(atom.GetFormalCharge())) for atom in mol.GetAtoms())
    try:
        smiles = Chem.MolToSmiles(mol)
    except Exception:  # noqa: BLE001 - SMILES conversion can fail on malformed molecules.
        smiles = ""
    return {
        "num_bonds": int(mol.GetNumBonds()),
        "num_fragments": len(fragments),
        "largest_fragment_atom_frac": largest_fragment_atoms / max(1, mol.GetNumAtoms()),
        "ring_count": len(atom_rings),
        "aromatic_ring_count": aromatic_rings,
        "three_member_ring_count": sum(int(len(ring) == 3) for ring in atom_rings),
        "fused_ring_pair_count": fused_ring_pairs,
        "carbon_fraction": carbon_count / max(1, len(atom_symbols)),
        "hetero_atom_count": hetero_count,
        "halogen_count": halogen_count,
        "formal_charge_abs_sum": formal_charge_abs_sum,
        "smiles": smiles,
    }


def _druglike_metrics(mol: Chem.Mol | None, sanitize_success: bool) -> dict[str, float | int]:
    """Calculate drug-like descriptors on sanitized molecules only."""
    if mol is None or not sanitize_success:
        return {
            "drug_metrics_available": 0,
            "qed": 0.0,
            "sa": 0.0,
            "logp": 0.0,
            "tpsa": 0.0,
            "hba": 0,
            "hbd": 0,
            "rotatable_bonds": 0,
            "mol_weight": 0.0,
            "lipinski": 0,
        }
    try:
        logp = Crippen.MolLogP(mol)
        lipinski = sum([
            int(Descriptors.ExactMolWt(mol) < 500),
            int(Lipinski.NumHDonors(mol) <= 5),
            int(Lipinski.NumHAcceptors(mol) <= 10),
            int(-2 <= logp <= 5),
            int(rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10),
        ])
        return {
            "drug_metrics_available": 1,
            "qed": float(QED.qed(mol)),
            "sa": float(round((10 - calculateScore(mol)) / 9, 4)),
            "logp": float(logp),
            "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
            "hba": int(Lipinski.NumHAcceptors(mol)),
            "hbd": int(Lipinski.NumHDonors(mol)),
            "rotatable_bonds": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
            "mol_weight": float(Descriptors.ExactMolWt(mol)),
            "lipinski": int(lipinski),
        }
    except Exception:  # noqa: BLE001 - descriptors can fail on edge-case molecules.
        return {
            "drug_metrics_available": 0,
            "qed": 0.0,
            "sa": 0.0,
            "logp": 0.0,
            "tpsa": 0.0,
            "hba": 0,
            "hbd": 0,
            "rotatable_bonds": 0,
            "mol_weight": 0.0,
            "lipinski": 0,
        }


def evaluate_reconstruction_attempt(
    *,
    sample_idx: int,
    attempt: ReconstructionAttempt,
    coords: torch.Tensor,
    atom_symbols: list[str],
    length_tolerance: float,
) -> tuple[dict[str, Any], Chem.Mol | None]:
    """Evaluate one reconstruction attempt and return a CSV row plus sanitized molecule."""
    sanitized_mol, sanitize_error = _safe_sanitize(attempt.raw_mol)
    final_error = attempt.build_error or sanitize_error
    sanitize_success = final_error == ""
    mol_for_metrics = sanitized_mol if sanitize_success else attempt.raw_mol

    row: dict[str, Any] = {
        "sample_idx": int(sample_idx),
        "method": attempt.method,
        "num_atoms": len(atom_symbols),
        "sanitize_success": int(sanitize_success),
        "error": final_error,
        "reconstruction_seconds": attempt.seconds,
    }
    row.update(_bond_length_metrics(mol_for_metrics, atom_symbols, coords, length_tolerance))
    row.update(_chemistry_problem_metrics(attempt.raw_mol))
    row.update(_topology_metrics(mol_for_metrics, atom_symbols))
    row.update(_druglike_metrics(sanitized_mol, sanitize_success))
    return row, sanitized_mol if sanitize_success else None
