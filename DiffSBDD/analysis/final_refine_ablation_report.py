"""Analyze final-only geometry refinement and atom-bias ablation SDF files."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from constants import bonds1, bonds2, bonds3  # noqa: E402

HALOGEN_ELEMENTS = {"F", "Cl", "Br", "I"}


@dataclass(frozen=True)
class MethodInput:
    """One named SDF input for ablation analysis."""

    name: str
    path: Path
    elapsed_sec: float


def parse_method(value: str) -> MethodInput:
    """Parse NAME:PATH:SECONDS method spec."""
    parts = value.split(":")
    if len(parts) < 3:
        raise ValueError("Method spec must be NAME:PATH:SECONDS.")
    name = parts[0]
    elapsed = float(parts[-1])
    path = Path(":".join(parts[1:-1]))
    return MethodInput(name=name, path=path, elapsed_sec=elapsed)


def expected_bond_length(atom_a: str, atom_b: str, bond: Chem.Bond) -> float | None:
    """Return expected bond length in Angstrom for common bond orders."""
    tables = {
        Chem.rdchem.BondType.SINGLE: bonds1,
        Chem.rdchem.BondType.DOUBLE: bonds2,
        Chem.rdchem.BondType.TRIPLE: bonds3,
        Chem.rdchem.BondType.AROMATIC: bonds1,
    }
    table = tables.get(bond.GetBondType())
    if table is None:
        return None
    value = table.get(atom_a, {}).get(atom_b)
    if value is None:
        value = table.get(atom_b, {}).get(atom_a)
    if value is None or value <= 0:
        return None
    return float(value) / 100.0


def iter_mols(path: Path) -> Iterable[Chem.Mol]:
    """Yield SDF molecules without sanitizing at load time."""
    supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
    for mol in supplier:
        if mol is not None:
            yield mol


def sanitize_copy(mol: Chem.Mol) -> Chem.Mol | None:
    """Return a sanitized molecule copy or None."""
    copied = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(copied)
    except Exception:
        return None
    return copied


def nearest_neighbor_distances(mol: Chem.Mol) -> list[float]:
    """Return per-atom nearest-neighbor distances."""
    if mol.GetNumAtoms() < 2 or mol.GetNumConformers() == 0:
        return []
    pos = np.asarray(mol.GetConformer().GetPositions(), dtype=float)
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    d[d == 0] = np.inf
    return np.min(d, axis=1).tolist()


def bond_length_errors(mol: Chem.Mol) -> tuple[list[float], int, int]:
    """Return bond length abs errors and bad/total counts."""
    if mol.GetNumConformers() == 0:
        return [], 0, 0
    pos = np.asarray(mol.GetConformer().GetPositions(), dtype=float)
    errors: list[float] = []
    bad = 0
    total = 0
    for bond in mol.GetBonds():
        atom_a = bond.GetBeginAtom().GetSymbol()
        atom_b = bond.GetEndAtom().GetSymbol()
        expected = expected_bond_length(atom_a, atom_b, bond)
        if expected is None:
            continue
        distance = float(np.linalg.norm(pos[bond.GetBeginAtomIdx()] - pos[bond.GetEndAtomIdx()]))
        error = abs(distance - expected)
        errors.append(error)
        total += 1
        if error > 0.20:
            bad += 1
    return errors, bad, total


def analyze_method(method: MethodInput) -> dict[str, float | str]:
    """Compute molecule, geometry, and descriptor metrics for one method."""
    mols = list(iter_mols(method.path))
    atom_counter: Counter[str] = Counter()
    sanitize_count = 0
    connected_count = 0
    three_rings: list[int] = []
    ring_counts: list[int] = []
    aromatic_rings: list[int] = []
    qed_values: list[float] = []
    logp_values: list[float] = []
    tpsa_values: list[float] = []
    weights: list[float] = []
    nn_values: list[float] = []
    bond_errors: list[float] = []
    bad_bonds = 0
    total_bonds = 0
    halogen_atoms = 0
    halogen_degree_violations = 0

    for mol in mols:
        atom_counter.update(atom.GetSymbol() for atom in mol.GetAtoms())
        for atom in mol.GetAtoms():
            if atom.GetSymbol() in HALOGEN_ELEMENTS:
                halogen_atoms += 1
                halogen_degree_violations += int(atom.GetDegree() > 1)
        nn_values.extend(nearest_neighbor_distances(mol))
        sanitized = sanitize_copy(mol)
        if sanitized is None:
            continue
        sanitize_count += 1
        connected_count += int(len(Chem.GetMolFrags(sanitized)) == 1)
        ring_info = sanitized.GetRingInfo()
        atom_rings = ring_info.AtomRings()
        three_rings.append(sum(1 for ring in atom_rings if len(ring) == 3))
        ring_counts.append(len(atom_rings))
        aromatic_rings.append(
            sum(
                1
                for ring in atom_rings
                if all(sanitized.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring)
            )
        )
        try:
            qed_values.append(float(QED.qed(sanitized)))
        except Exception:
            pass
        logp_values.append(float(Crippen.MolLogP(sanitized)))
        tpsa_values.append(float(rdMolDescriptors.CalcTPSA(sanitized)))
        weights.append(float(Descriptors.MolWt(sanitized)))
        errors, bad, total = bond_length_errors(sanitized)
        bond_errors.extend(errors)
        bad_bonds += bad
        total_bonds += total

    total_atoms = sum(atom_counter.values())
    hetero_atoms = total_atoms - atom_counter.get("C", 0)
    n_mols = len(mols)
    row: dict[str, float | str] = {
        "method": method.name,
        "path": str(method.path),
        "elapsed_sec": method.elapsed_sec,
        "sec_per_requested_mol": method.elapsed_sec / n_mols if n_mols else 0.0,
        "n_mols": float(n_mols),
        "sanitize_rate": sanitize_count / len(mols) if mols else 0.0,
        "connected_rate": connected_count / sanitize_count if sanitize_count else 0.0,
        "total_atoms": float(total_atoms),
        "carbon_fraction": atom_counter.get("C", 0) / total_atoms if total_atoms else 0.0,
        "hetero_fraction": hetero_atoms / total_atoms if total_atoms else 0.0,
        "bad_bond_length_rate": bad_bonds / total_bonds if total_bonds else 0.0,
        "mean_bond_length_abs_error": float(np.mean(bond_errors)) if bond_errors else 0.0,
        "halogen_atoms": float(halogen_atoms),
        "halogen_degree_violations": float(halogen_degree_violations),
        "halogen_degree_violation_rate": (
            halogen_degree_violations / halogen_atoms if halogen_atoms else 0.0
        ),
        "nearest_neighbor_mean": float(np.mean(nn_values)) if nn_values else 0.0,
        "nearest_neighbor_q10": float(np.quantile(nn_values, 0.10)) if nn_values else 0.0,
        "nearest_neighbor_q50": float(np.quantile(nn_values, 0.50)) if nn_values else 0.0,
        "nearest_neighbor_q90": float(np.quantile(nn_values, 0.90)) if nn_values else 0.0,
        "three_member_rings_mean": float(np.mean(three_rings)) if three_rings else 0.0,
        "ring_count_mean": float(np.mean(ring_counts)) if ring_counts else 0.0,
        "aromatic_ring_count_mean": float(np.mean(aromatic_rings)) if aromatic_rings else 0.0,
        "qed_mean": float(np.mean(qed_values)) if qed_values else 0.0,
        "logp_mean": float(np.mean(logp_values)) if logp_values else 0.0,
        "tpsa_mean": float(np.mean(tpsa_values)) if tpsa_values else 0.0,
        "molwt_mean": float(np.mean(weights)) if weights else 0.0,
    }
    for element in ["C", "N", "O", "S", "F", "Cl", "P", "Br", "I", "B"]:
        row[f"frac_{element}"] = atom_counter.get(element, 0) / total_atoms if total_atoms else 0.0
    return row


def write_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    """Write metrics rows to CSV."""
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(rows: list[dict[str, float | str]], outdir: Path) -> None:
    """Save concise metric plots."""
    methods = [str(row["method"]) for row in rows]
    metrics = [
        "sanitize_rate",
        "carbon_fraction",
        "hetero_fraction",
        "bad_bond_length_rate",
        "three_member_rings_mean",
        "nearest_neighbor_mean",
        "qed_mean",
        "elapsed_sec",
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, metric in zip(axes.flatten(), metrics):
        values = [float(row[metric]) for row in rows]
        ax.bar(methods, values, color="#4C78A8")
        ax.set_title(metric)
        ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    fig.savefig(outdir / "summary_metrics.png", dpi=200)
    plt.close(fig)

    elements = ["C", "N", "O", "S", "F", "Cl", "P"]
    x = np.arange(len(elements))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, row in enumerate(rows):
        values = [float(row[f"frac_{element}"]) for element in elements]
        ax.bar(x + (idx - 1.5) * width, values, width=width, label=str(row["method"]))
    ax.set_xticks(x)
    ax.set_xticklabels(elements)
    ax.set_ylabel("fraction")
    ax.set_title("Element fractions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "element_fractions.png", dpi=200)
    plt.close(fig)


def main() -> None:
    """Run analysis CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", action="append", required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    methods = [parse_method(value) for value in args.method]
    rows = [analyze_method(method) for method in methods]
    write_csv(rows, args.outdir / "metrics.csv")
    plot_summary(rows, args.outdir)


if __name__ == "__main__":
    main()
