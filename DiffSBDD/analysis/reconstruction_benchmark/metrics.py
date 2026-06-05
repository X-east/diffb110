"""Metric extraction for reconstruction benchmark rows."""

from __future__ import annotations

from typing import Any

import torch
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors

from analysis.SA_Score.sascorer import calculateScore
from edge_data_oracle_report import _lookup_reference_length
from .methods import ReconstructionAttempt

ELEMENT_FRACTION_ELEMENTS = ("C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "others")

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
            "six_member_ring_count": 0,
            "aromatic_six_ring_count": 0,
            "benzene_like_ring_count": 0,
            "non_six_aromatic_ring_count": 0,
            "abnormal_aromatic_ring_count": 0,
            "three_member_ring_count": 0,
            "fused_ring_pair_count": 0,
            "carbon_fraction": 0.0,
            "hetero_atom_count": 0,
            "halogen_count": 0,
            **{f"{_element_metric_name(element)}_fraction": 0.0 for element in ELEMENT_FRACTION_ELEMENTS},
            "formal_charge_abs_sum": 0,
            "smiles": "",
        }

    fragments = Chem.GetMolFrags(mol, sanitizeFrags=False)
    largest_fragment_atoms = max((len(fragment) for fragment in fragments), default=0)
    atom_rings = _safe_atom_rings(mol)
    aromatic_stats = _aromatic_ring_metrics(mol, atom_rings)
    fused_ring_pairs = 0
    for left_idx, left_ring in enumerate(atom_rings):
        for right_ring in atom_rings[left_idx + 1:]:
            fused_ring_pairs += int(len(left_ring & right_ring) > 0)

    halogens = {"F", "Cl", "Br", "I"}
    carbon_count = sum(int(symbol == "C") for symbol in atom_symbols)
    hetero_count = sum(int(symbol not in {"C", "H"}) for symbol in atom_symbols)
    halogen_count = sum(int(symbol in halogens) for symbol in atom_symbols)
    element_fractions = _element_fraction_metrics(atom_symbols)
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
        **aromatic_stats,
        "three_member_ring_count": sum(int(len(ring) == 3) for ring in atom_rings),
        "fused_ring_pair_count": fused_ring_pairs,
        "carbon_fraction": carbon_count / max(1, len(atom_symbols)),
        "hetero_atom_count": hetero_count,
        "halogen_count": halogen_count,
        **element_fractions,
        "formal_charge_abs_sum": formal_charge_abs_sum,
        "smiles": smiles,
    }


def _element_metric_name(element: str) -> str:
    """Return a stable metric-safe element name."""
    return "others" if element == "others" else element.lower()


def _element_fraction_metrics(atom_symbols: list[str]) -> dict[str, float]:
    """Return per-element fractions from generated atom symbols."""
    denominator = max(1, len(atom_symbols))
    known = set(ELEMENT_FRACTION_ELEMENTS) - {"others"}
    rows: dict[str, float] = {}
    for element in ELEMENT_FRACTION_ELEMENTS:
        if element == "others":
            count = sum(int(symbol not in known) for symbol in atom_symbols)
        else:
            count = sum(int(symbol == element) for symbol in atom_symbols)
        rows[f"{_element_metric_name(element)}_fraction"] = count / denominator
    return rows


def _safe_atom_rings(mol: Chem.Mol) -> list[set[int]]:
    """Return atom rings without letting unsanitized molecules break metrics."""
    try:
        return [set(ring) for ring in mol.GetRingInfo().AtomRings()]
    except Exception:  # noqa: BLE001 - unsanitized molecules may lack ring info.
        try:
            Chem.FastFindRings(mol)
            return [set(ring) for ring in mol.GetRingInfo().AtomRings()]
        except Exception:  # noqa: BLE001 - keep benchmark robust.
            return []


def _ring_bonds(mol: Chem.Mol, ring: set[int]) -> list[Chem.Bond]:
    """Return bonds whose two endpoints both lie in one ring."""
    bonds = []
    for bond in mol.GetBonds():
        begin_idx = int(bond.GetBeginAtomIdx())
        end_idx = int(bond.GetEndAtomIdx())
        if begin_idx in ring and end_idx in ring:
            bonds.append(bond)
    return bonds


def _aromatic_ring_metrics(mol: Chem.Mol, atom_rings: list[set[int]]) -> dict[str, int]:
    """Count strict aromatic and benzene-like rings."""
    aromatic_ring_count = 0
    six_member_ring_count = 0
    aromatic_six_ring_count = 0
    benzene_like_ring_count = 0
    non_six_aromatic_ring_count = 0
    abnormal_aromatic_ring_count = 0

    for ring in atom_rings:
        ring_atoms = [mol.GetAtomWithIdx(atom_idx) for atom_idx in sorted(ring)]
        ring_bonds = _ring_bonds(mol, ring)
        any_aromatic = any(atom.GetIsAromatic() for atom in ring_atoms) or any(
            bond.GetIsAromatic() for bond in ring_bonds
        )
        strict_aromatic = (
            len(ring_bonds) >= len(ring)
            and all(atom.GetIsAromatic() for atom in ring_atoms)
            and all(bond.GetIsAromatic() for bond in ring_bonds)
        )
        if len(ring) == 6:
            six_member_ring_count += 1
        if strict_aromatic:
            aromatic_ring_count += 1
            if len(ring) == 6:
                aromatic_six_ring_count += 1
            else:
                non_six_aromatic_ring_count += 1
        elif any_aromatic:
            abnormal_aromatic_ring_count += 1

        is_benzene_like = (
            len(ring) == 6
            and strict_aromatic
            and all(atom.GetSymbol() == "C" for atom in ring_atoms)
            and all(int(atom.GetFormalCharge()) == 0 for atom in ring_atoms)
        )
        benzene_like_ring_count += int(is_benzene_like)

    return {
        "aromatic_ring_count": aromatic_ring_count,
        "six_member_ring_count": six_member_ring_count,
        "aromatic_six_ring_count": aromatic_six_ring_count,
        "benzene_like_ring_count": benzene_like_ring_count,
        "non_six_aromatic_ring_count": non_six_aromatic_ring_count,
        "abnormal_aromatic_ring_count": abnormal_aromatic_ring_count,
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
