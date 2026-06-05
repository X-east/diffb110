"""Summarize generated molecule files for reference-project smoke tests."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors


def _iter_sdf_paths(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(path.rglob("*.sdf"))
    return [path]


def _read_mols(paths: list[Path]) -> tuple[list[Chem.Mol], int]:
    mols: list[Chem.Mol] = []
    records = 0
    for path in paths:
        suppl = Chem.SDMolSupplier(str(path), sanitize=True, removeHs=False)
        for mol in suppl:
            records += 1
            if mol is not None:
                mols.append(mol)
    return mols, records


def _bad_bond_fraction(mol: Chem.Mol) -> float:
    if mol.GetNumConformers() == 0 or mol.GetNumBonds() == 0:
        return 0.0
    conf = mol.GetConformer()
    table = Chem.GetPeriodicTable()
    bad = 0
    total = 0
    for bond in mol.GetBonds():
        a = bond.GetBeginAtom()
        b = bond.GetEndAtom()
        if a.GetAtomicNum() == 1 or b.GetAtomicNum() == 1:
            continue
        pa = conf.GetAtomPosition(a.GetIdx())
        pb = conf.GetAtomPosition(b.GetIdx())
        dist = pa.Distance(pb)
        ideal = table.GetRcovalent(a.GetAtomicNum()) + table.GetRcovalent(b.GetAtomicNum())
        if ideal <= 0:
            continue
        ratio = dist / ideal
        total += 1
        if ratio < 0.70 or ratio > 1.45:
            bad += 1
    return bad / total if total else 0.0


def _mol_row(mol: Chem.Mol, idx: int) -> dict[str, object]:
    heavy = mol.GetNumHeavyAtoms()
    atom_counts: dict[str, int] = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        sym = atom.GetSymbol()
        atom_counts[sym] = atom_counts.get(sym, 0) + 1
    carbon = atom_counts.get("C", 0)
    hetero = heavy - carbon
    ring_info = mol.GetRingInfo()
    ring_sizes = [len(r) for r in ring_info.AtomRings()]
    aromatic_ring_count = rdMolDescriptors.CalcNumAromaticRings(mol)
    smiles = Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True)
    return {
        "idx": idx,
        "smiles": smiles,
        "heavy_atoms": heavy,
        "qed": QED.qed(mol),
        "logp": Crippen.MolLogP(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "hba": Lipinski.NumHAcceptors(mol),
        "hbd": Lipinski.NumHDonors(mol),
        "ring_count": rdMolDescriptors.CalcNumRings(mol),
        "aromatic_ring_count": aromatic_ring_count,
        "has_aromatic_ring": aromatic_ring_count > 0,
        "three_member_ring_count": sum(1 for size in ring_sizes if size == 3),
        "has_three_member_ring": any(size == 3 for size in ring_sizes),
        "carbon_fraction": carbon / heavy if heavy else 0.0,
        "hetero_atoms": hetero,
        "atom_counts": json.dumps(atom_counts, sort_keys=True),
        "bad_bond_length_fraction": _bad_bond_fraction(mol),
        "mw": Descriptors.MolWt(mol),
    }


def _safe_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _safe_median(values: list[float]) -> float | None:
    return median(values) if values else None


def _summary(rows: list[dict[str, object]], records: int, path_count: int) -> dict[str, object]:
    smiles = [str(row["smiles"]) for row in rows]
    summary: dict[str, object] = {
        "path_count": path_count,
        "sdf_records": records,
        "rdkit_valid": len(rows),
        "valid_rate": len(rows) / records if records else 0.0,
        "unique_smiles": len(set(smiles)),
        "unique_rate_valid": len(set(smiles)) / len(smiles) if smiles else 0.0,
    }
    for key in [
        "heavy_atoms",
        "qed",
        "logp",
        "tpsa",
        "hba",
        "hbd",
        "ring_count",
        "aromatic_ring_count",
        "three_member_ring_count",
        "carbon_fraction",
        "hetero_atoms",
        "bad_bond_length_fraction",
        "mw",
    ]:
        vals = [float(row[key]) for row in rows]
        summary[f"{key}_mean"] = _safe_mean(vals)
        summary[f"{key}_median"] = _safe_median(vals)
    if rows:
        summary["has_aromatic_ring_rate"] = mean(float(row["has_aromatic_ring"]) for row in rows)
        summary["has_three_member_ring_rate"] = mean(float(row["has_three_member_ring"]) for row in rows)
    else:
        summary["has_aromatic_ring_rate"] = None
        summary["has_three_member_ring_rate"] = None
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    paths = _iter_sdf_paths(args.input)
    mols, records = _read_mols(paths)
    rows = [_mol_row(mol, idx) for idx, mol in enumerate(mols)]
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    with args.out_json.open("w", encoding="utf-8") as handle:
        json.dump(_summary(rows, records, len(paths)), handle, indent=2, sort_keys=True)

    fieldnames = list(rows[0].keys()) if rows else ["idx", "smiles"]
    with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
