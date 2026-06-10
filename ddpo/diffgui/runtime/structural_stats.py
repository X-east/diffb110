"""RDKit-based structural statistics for DiffGui epoch monitoring.

Provides per-molecule bond-order distribution, valence violations,
fragment counts, and ring composition — all computable from RDKit
without model access.
"""

from __future__ import annotations

import statistics
from typing import Any

from rdkit import Chem


# Per-element max valence (simplified).
_MAX_VALENCE: dict[int, int] = {
    6: 4, 7: 4, 8: 2, 9: 1, 15: 5, 16: 6, 17: 1, 35: 1, 53: 1,
}


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _std(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def _fraction(values: list[float], predicate: Any) -> float:
    return float(sum(1 for v in values if predicate(v)) / len(values)) if values else 0.0


def compute_structural_metrics(mols: list[Chem.Mol]) -> dict[str, float]:
    """Compute bond-order, valence, fragment, and ring statistics."""
    if not mols:
        return _empty_structural_metrics()

    bond_single: list[float] = []
    bond_double: list[float] = []
    bond_triple: list[float] = []
    bond_aromatic: list[float] = []
    bond_total: list[float] = []
    valence_violations: list[float] = []
    fragment_counts: list[float] = []
    ring_counts: list[float] = []
    three_member_rings: list[float] = []

    for mol in mols:
        n_s = n_d = n_t = n_a = 0
        for bond in mol.GetBonds():
            btype = bond.GetBondType()
            if btype == Chem.BondType.SINGLE:
                n_s += 1
            elif btype == Chem.BondType.DOUBLE:
                n_d += 1
            elif btype == Chem.BondType.TRIPLE:
                n_t += 1
            elif btype == Chem.BondType.AROMATIC:
                n_a += 1
        bond_single.append(float(n_s))
        bond_double.append(float(n_d))
        bond_triple.append(float(n_t))
        bond_aromatic.append(float(n_a))
        bond_total.append(float(n_s + n_d + n_t + n_a))

        violations = 0
        for atom in mol.GetAtoms():
            if atom.GetExplicitValence() > _MAX_VALENCE.get(atom.GetAtomicNum(), 6):
                violations += 1
        valence_violations.append(float(violations))

        frags = Chem.GetMolFrags(mol, asMols=False)
        fragment_counts.append(float(len(frags)))

        ssr = Chem.GetSymmSSSR(mol)
        ring_counts.append(float(len(ssr)))
        three_member_rings.append(float(sum(1 for r in ssr if len(r) == 3)))

    return {
        "bond_order_single_mean": _mean(bond_single),
        "bond_order_double_mean": _mean(bond_double),
        "bond_order_triple_mean": _mean(bond_triple),
        "bond_order_aromatic_mean": _mean(bond_aromatic),
        "bond_order_total_mean": _mean(bond_total),
        "valence_violation_mean": _mean(valence_violations),
        "valence_violation_total": float(sum(valence_violations)),
        "fragment_count_mean": _mean(fragment_counts),
        "fragment_count_std": _std(fragment_counts),
        "fragment_multi_ratio": _fraction(fragment_counts, lambda x: x > 1),
        "ring_count_mean": _mean(ring_counts),
        "ring_count_std": _std(ring_counts),
        "three_member_ring_mean": _mean(three_member_rings),
        "three_member_ring_total": float(sum(three_member_rings)),
    }


def _empty_structural_metrics() -> dict[str, float]:
    """Return zero-valued structural metrics when no valid molecules exist."""
    return {
        "bond_order_single_mean": 0.0,
        "bond_order_double_mean": 0.0,
        "bond_order_triple_mean": 0.0,
        "bond_order_aromatic_mean": 0.0,
        "bond_order_total_mean": 0.0,
        "valence_violation_mean": 0.0,
        "valence_violation_total": 0.0,
        "fragment_count_mean": 0.0,
        "fragment_count_std": 0.0,
        "fragment_multi_ratio": 0.0,
        "ring_count_mean": 0.0,
        "ring_count_std": 0.0,
        "three_member_ring_mean": 0.0,
        "three_member_ring_total": 0.0,
    }
