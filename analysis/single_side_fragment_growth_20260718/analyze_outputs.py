"""Audit single-side fragment-growth outputs from three generative models."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem


def _read_sdf(path: Path) -> list[Chem.Mol]:
    """Read all non-null molecules from one SDF file."""
    return [mol for mol in Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False) if mol]


def _collect(run_dir: Path) -> dict[str, list[Chem.Mol]]:
    """Collect model outputs from their native layouts."""
    diffsbdd = _read_sdf(run_dir / "diffsbdd/diffsbdd_raw_20.sdf")
    pmdm = []
    for path in sorted((run_dir / "pmdm/input/frag_gen").glob("*.sdf")):
        pmdm.extend(_read_sdf(path))
    diffgui_dirs = sorted((run_dir / "diffgui").glob("diffgui_sample_*"))
    diffgui = []
    if diffgui_dirs:
        for path in sorted(diffgui_dirs[-1].glob("*_SDF/*.sdf")):
            if not path.name.startswith("traj_"):
                diffgui.extend(_read_sdf(path))
    return {"DiffSBDD": diffsbdd, "DiffGui": diffgui, "PMDM": pmdm}


def _coordinate_core_map(
    mol: Chem.Mol,
    core: Chem.Mol,
    tolerance: float = 0.20,
) -> tuple[int, ...] | None:
    """Greedily map fixed core atoms by element and 3D position."""
    if mol.GetNumConformers() == 0 or core.GetNumConformers() == 0:
        return None
    mol_pos = np.asarray(mol.GetConformer().GetPositions(), dtype=float)
    core_pos = np.asarray(core.GetConformer().GetPositions(), dtype=float)
    unused = set(range(mol.GetNumAtoms()))
    mapping: list[int] = []
    for core_idx, atom in enumerate(core.GetAtoms()):
        candidates = [
            idx
            for idx in unused
            if mol.GetAtomWithIdx(idx).GetSymbol() == atom.GetSymbol()
        ]
        if not candidates:
            return None
        candidate = min(candidates, key=lambda idx: float(np.linalg.norm(mol_pos[idx] - core_pos[core_idx])))
        if float(np.linalg.norm(mol_pos[candidate] - core_pos[core_idx])) > tolerance:
            return None
        mapping.append(candidate)
        unused.remove(candidate)
    return tuple(mapping)


def _aligned_slot_core_map(
    mol: Chem.Mol,
    core: Chem.Mol,
    tolerance: float = 0.20,
) -> tuple[tuple[int, ...] | None, float]:
    """Map the leading fixed slots after optimal rigid alignment."""
    n_core = core.GetNumAtoms()
    if mol.GetNumAtoms() < n_core or mol.GetNumConformers() == 0:
        return None, float("nan")
    if any(
        mol.GetAtomWithIdx(idx).GetSymbol() != core.GetAtomWithIdx(idx).GetSymbol()
        for idx in range(n_core)
    ):
        return None, float("nan")
    moving = np.asarray(mol.GetConformer().GetPositions()[:n_core], dtype=float)
    target = np.asarray(core.GetConformer().GetPositions(), dtype=float)
    moving_centered = moving - moving.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    covariance = moving_centered.T @ target_centered
    left, _, right = np.linalg.svd(covariance)
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    aligned = moving_centered @ rotation
    rmsd = float(np.sqrt(np.mean(np.sum((aligned - target_centered) ** 2, axis=1))))
    if rmsd > tolerance:
        return None, rmsd
    return tuple(range(n_core)), rmsd


def _audit_one(model: str, index: int, mol: Chem.Mol, core: Chem.Mol) -> dict[str, object]:
    """Compute legality, connectivity, and exact-core attachment facts."""
    row: dict[str, object] = {
        "model": model,
        "index": index,
        "atom_count": mol.GetNumAtoms(),
        "heavy_atom_count": mol.GetNumHeavyAtoms(),
        "sanitize_valid": False,
        "connected": False,
        "exact_core_match": False,
        "coordinate_core_match": False,
        "core_bonds_present": False,
        "slot_core_geometry_match": False,
        "slot_core_rmsd": float("nan"),
        "slot_core_bonds_present": False,
        "c0_extension": False,
        "c11_extension": False,
        "other_core_extension": False,
        "external_neighbor_count": 0,
        "smiles": "",
    }
    try:
        checked = Chem.Mol(mol)
        Chem.SanitizeMol(checked)
        row["sanitize_valid"] = True
        row["connected"] = len(Chem.GetMolFrags(checked)) == 1
        row["smiles"] = Chem.MolToSmiles(checked)
    except Exception:
        checked = mol

    matches = checked.GetSubstructMatches(core, uniquify=False)
    if matches:
        row["exact_core_match"] = True
    slot_match, slot_rmsd = _aligned_slot_core_map(checked, core)
    row["slot_core_rmsd"] = slot_rmsd
    if slot_match is not None:
        row["slot_core_geometry_match"] = True
        row["slot_core_bonds_present"] = all(
            checked.GetBondBetweenAtoms(
                slot_match[bond.GetBeginAtomIdx()], slot_match[bond.GetEndAtomIdx()]
            )
            is not None
            for bond in core.GetBonds()
        )
    coordinate_match = _coordinate_core_map(checked, core)
    if coordinate_match is not None:
        row["coordinate_core_match"] = True
        row["core_bonds_present"] = all(
            checked.GetBondBetweenAtoms(
                coordinate_match[bond.GetBeginAtomIdx()], coordinate_match[bond.GetEndAtomIdx()]
            )
            is not None
            for bond in core.GetBonds()
        )
    match = slot_match if slot_match is not None else coordinate_match
    if match is None:
        return row
    core_atoms = set(match)
    external_by_core: dict[int, int] = {}
    for core_idx, mol_idx in enumerate(match):
        count = sum(1 for nbr in checked.GetAtomWithIdx(mol_idx).GetNeighbors() if nbr.GetIdx() not in core_atoms)
        external_by_core[core_idx] = count
    row["c0_extension"] = external_by_core.get(0, 0) > 0
    row["c11_extension"] = external_by_core.get(11, 0) > 0
    row["other_core_extension"] = any(
        count > 0 for idx, count in external_by_core.items() if idx not in {0, 11}
    )
    row["external_neighbor_count"] = sum(external_by_core.values())
    return row


def main() -> None:
    """Write per-molecule CSV, model summary JSON/CSV, and a comparison plot."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    core = _read_sdf(args.run_dir / "inputs/fixed_core_no_oh.sdf")[0]
    try:
        Chem.SanitizeMol(core)
    except Exception:
        pass

    rows = [
        _audit_one(model, index, mol, core)
        for model, mols in _collect(args.run_dir).items()
        for index, mol in enumerate(mols)
    ]
    fieldnames = list(rows[0]) if rows else []
    with (args.run_dir / "molecule_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summaries = []
    for model in ("DiffSBDD", "DiffGui", "PMDM"):
        subset = [row for row in rows if row["model"] == model]
        n = len(subset)
        summaries.append(
            {
                "model": model,
                "written": n,
                "sanitize_valid": sum(bool(row["sanitize_valid"]) for row in subset),
                "connected": sum(bool(row["connected"]) for row in subset),
                "exact_core_match": sum(bool(row["exact_core_match"]) for row in subset),
                "coordinate_core_match": sum(bool(row["coordinate_core_match"]) for row in subset),
                "core_bonds_present": sum(bool(row["core_bonds_present"]) for row in subset),
                "slot_core_geometry_match": sum(bool(row["slot_core_geometry_match"]) for row in subset),
                "slot_core_bonds_present": sum(bool(row["slot_core_bonds_present"]) for row in subset),
                "c0_extension": sum(bool(row["c0_extension"]) for row in subset),
                "c11_extension": sum(bool(row["c11_extension"]) for row in subset),
                "other_core_extension": sum(bool(row["other_core_extension"]) for row in subset),
                "atom_count_distribution": dict(Counter(int(row["atom_count"]) for row in subset)),
            }
        )
    (args.run_dir / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        keys = [key for key in summaries[0] if key != "atom_count_distribution"]
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: row[key] for key in keys} for row in summaries)

    labels = ["written", "sanitize_valid", "slot_core_geometry_match", "slot_core_bonds_present", "c0_extension"]
    x = range(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for offset, summary in enumerate(summaries):
        ax.bar(
            [value + (offset - 1) * width for value in x],
            [summary[label] for label in labels],
            width=width,
            label=summary["model"],
        )
    ax.set_xticks(list(x), labels, rotation=15)
    ax.set_ylabel("Molecule count")
    ax.set_title("Single-side fragment growth audit")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.run_dir / "fragment_growth_audit.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
