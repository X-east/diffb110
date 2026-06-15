"""Evaluate benchmark SDF files with one RDKit metric panel."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors


RDLogger.DisableLog("rdApp.*")


ELEMENT_KEYS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B"]


def _safe_mean(values: list[float]) -> float | None:
    clean = [value for value in values if not math.isnan(value)]
    return mean(clean) if clean else None


def _safe_median(values: list[float]) -> float | None:
    clean = [value for value in values if not math.isnan(value)]
    return median(clean) if clean else None


def _bad_bond_fraction(mol: Chem.Mol) -> tuple[int, int, float, float]:
    if mol.GetNumConformers() == 0 or mol.GetNumBonds() == 0:
        return 0, 0, 0.0, 0.0
    conf = mol.GetConformer()
    table = Chem.GetPeriodicTable()
    bad = 0
    checked = 0
    errors: list[float] = []
    for bond in mol.GetBonds():
        atom_a = bond.GetBeginAtom()
        atom_b = bond.GetEndAtom()
        if atom_a.GetAtomicNum() == 1 or atom_b.GetAtomicNum() == 1:
            continue
        pos_a = conf.GetAtomPosition(atom_a.GetIdx())
        pos_b = conf.GetAtomPosition(atom_b.GetIdx())
        dist = pos_a.Distance(pos_b)
        ideal = table.GetRcovalent(atom_a.GetAtomicNum()) + table.GetRcovalent(atom_b.GetAtomicNum())
        if ideal <= 0:
            continue
        ratio = dist / ideal
        checked += 1
        errors.append(abs(dist - ideal))
        if ratio < 0.70 or ratio > 1.45:
            bad += 1
    return bad, checked, bad / checked if checked else 0.0, mean(errors) if errors else 0.0


def _sanitize_copy(mol: Chem.Mol) -> Chem.Mol | None:
    copy = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(copy)
    except Exception:
        return None
    return copy


def _count_fragments(mol: Chem.Mol) -> tuple[int, float]:
    fragments = Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
    if not fragments:
        return 0, 0.0
    heavy = max(1, mol.GetNumHeavyAtoms())
    largest = max(sum(1 for idx in frag if mol.GetAtomWithIdx(idx).GetAtomicNum() != 1) for frag in fragments)
    return len(fragments), largest / heavy


def _atom_fractions(mol: Chem.Mol) -> tuple[dict[str, int], dict[str, float]]:
    counts: Counter[str] = Counter()
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        counts[atom.GetSymbol()] += 1
    heavy = sum(counts.values())
    fractions = {f"{key.lower()}_fraction": counts.get(key, 0) / heavy if heavy else 0.0 for key in ELEMENT_KEYS}
    known = sum(counts.get(key, 0) for key in ELEMENT_KEYS)
    fractions["other_fraction"] = (heavy - known) / heavy if heavy else 0.0
    return dict(counts), fractions


def _mol_row(model: str, index: int, mol: Chem.Mol) -> dict[str, object]:
    sanitized = _sanitize_copy(mol)
    metric_mol = sanitized if sanitized is not None else mol
    smiles = ""
    if sanitized is not None:
        smiles = Chem.MolToSmiles(Chem.RemoveHs(sanitized), isomericSmiles=True)
    bad, checked, bad_fraction, mean_error = _bad_bond_fraction(metric_mol)
    fragments, largest_fragment = _count_fragments(metric_mol)
    atom_counts, fractions = _atom_fractions(metric_mol)
    heavy = metric_mol.GetNumHeavyAtoms()
    carbon = atom_counts.get("C", 0)
    ring_sizes: list[int] = []
    ring_count = 0
    aromatic_ring_count = 0
    if sanitized is not None:
        ring_info = metric_mol.GetRingInfo()
        ring_sizes = [len(ring) for ring in ring_info.AtomRings()]
        ring_count = rdMolDescriptors.CalcNumRings(metric_mol)
        aromatic_ring_count = rdMolDescriptors.CalcNumAromaticRings(metric_mol)
    row: dict[str, object] = {
        "model": model,
        "idx": index,
        "sanitize_success": sanitized is not None,
        "smiles": smiles,
        "heavy_atoms": heavy,
        "num_bonds": metric_mol.GetNumBonds(),
        "bad_bond_lengths": bad,
        "checked_bond_lengths": checked,
        "bad_bond_length_fraction": bad_fraction,
        "mean_bond_length_abs_error": mean_error,
        "num_fragments": fragments,
        "largest_fragment_atom_frac": largest_fragment,
        "ring_count": ring_count,
        "aromatic_ring_count": aromatic_ring_count,
        "has_aromatic_ring": aromatic_ring_count > 0,
        "three_member_ring_count": sum(1 for size in ring_sizes if size == 3),
        "has_three_member_ring": any(size == 3 for size in ring_sizes),
        "hetero_atoms": heavy - carbon,
        "carbon_fraction": carbon / heavy if heavy else 0.0,
        "atom_counts": json.dumps(atom_counts, sort_keys=True),
        "qed": QED.qed(metric_mol) if sanitized is not None else math.nan,
        "logp": Crippen.MolLogP(metric_mol) if sanitized is not None else math.nan,
        "tpsa": rdMolDescriptors.CalcTPSA(metric_mol) if sanitized is not None else math.nan,
        "hba": Lipinski.NumHAcceptors(metric_mol) if sanitized is not None else math.nan,
        "hbd": Lipinski.NumHDonors(metric_mol) if sanitized is not None else math.nan,
        "rotatable_bonds": Lipinski.NumRotatableBonds(metric_mol) if sanitized is not None else math.nan,
        "mw": Descriptors.MolWt(metric_mol) if sanitized is not None else math.nan,
    }
    row.update(fractions)
    return row


def _read_rows(model: str, sdf_path: Path) -> tuple[list[dict[str, object]], int]:
    rows: list[dict[str, object]] = []
    records = 0
    supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False)
    for mol in supplier:
        records += 1
        if mol is not None:
            rows.append(_mol_row(model, len(rows), mol))
    return rows, records


def _summarize(model: str, rows: list[dict[str, object]], records: int, sdf_path: Path) -> dict[str, object]:
    sanitized_rows = [row for row in rows if row["sanitize_success"]]
    smiles = [str(row["smiles"]) for row in sanitized_rows if row["smiles"]]
    summary: dict[str, object] = {
        "model": model,
        "sdf_path": str(sdf_path),
        "sdf_records": records,
        "rdkit_readable": len(rows),
        "sanitize_valid": len(sanitized_rows),
        "readable_rate": len(rows) / records if records else 0.0,
        "sanitize_rate_records": len(sanitized_rows) / records if records else 0.0,
        "unique_smiles": len(set(smiles)),
        "unique_rate_valid": len(set(smiles)) / len(smiles) if smiles else 0.0,
    }
    numeric_keys = [
        "heavy_atoms",
        "num_bonds",
        "bad_bond_length_fraction",
        "mean_bond_length_abs_error",
        "num_fragments",
        "largest_fragment_atom_frac",
        "ring_count",
        "aromatic_ring_count",
        "three_member_ring_count",
        "hetero_atoms",
        "carbon_fraction",
        "qed",
        "logp",
        "tpsa",
        "hba",
        "hbd",
        "rotatable_bonds",
        "mw",
        *[f"{key.lower()}_fraction" for key in ELEMENT_KEYS],
        "other_fraction",
    ]
    for key in numeric_keys:
        values = [float(row[key]) for row in rows]
        summary[f"{key}_mean"] = _safe_mean(values)
        summary[f"{key}_median"] = _safe_median(values)
    if rows:
        summary["has_aromatic_ring_rate"] = mean(float(row["has_aromatic_ring"]) for row in rows)
        summary["has_three_member_ring_rate"] = mean(float(row["has_three_member_ring"]) for row in rows)
    else:
        summary["has_aromatic_ring_rate"] = None
        summary["has_three_member_ring_rate"] = None
    return summary


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["model", "idx"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_bar(summary_rows: list[dict[str, object]], key: str, ylabel: str, path: Path) -> None:
    labels = [str(row["model"]) for row in summary_rows]
    values = [float(row[key]) if row.get(key) is not None else 0.0 for row in summary_rows]
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(labels)), 4.5))
    ax.bar(labels, values, color="#4C78A8")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_scatter(rows: list[dict[str, object]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    models = sorted({str(row["model"]) for row in rows})
    for model in models:
        xs = [float(row["logp"]) for row in rows if row["model"] == model and not math.isnan(float(row["logp"]))]
        ys = [float(row["qed"]) for row in rows if row["model"] == model and not math.isnan(float(row["qed"]))]
        ax.scatter(xs, ys, s=12, alpha=0.55, label=model)
    ax.set_xlabel("logP")
    ax.set_ylabel("QED")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def evaluate(model_sdfs: list[str], out_dir: Path) -> dict[str, object]:
    """Evaluate all model=SDF specs and write tables/plots."""

    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for spec in model_sdfs:
        if "=" not in spec:
            raise ValueError(f"Expected model=sdf_path, got {spec}")
        model, sdf_raw = spec.split("=", 1)
        sdf_path = Path(sdf_raw)
        rows, records = _read_rows(model, sdf_path)
        all_rows.extend(rows)
        summaries.append(_summarize(model, rows, records, sdf_path))

    _write_csv(out_dir / "per_molecule_metrics.csv", all_rows)
    _write_csv(out_dir / "model_summary.csv", summaries)
    (out_dir / "model_summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if summaries:
        _plot_bar(summaries, "sanitize_rate_records", "Sanitize valid / SDF records", out_dir / "sanitize_rate.png")
        _plot_bar(summaries, "bad_bond_length_fraction_mean", "Mean bad bond length fraction", out_dir / "bad_bond_length.png")
        _plot_bar(summaries, "carbon_fraction_mean", "Mean carbon fraction", out_dir / "carbon_fraction.png")
        _plot_bar(summaries, "has_aromatic_ring_rate", "Aromatic ring molecule rate", out_dir / "aromatic_rate.png")
    if all_rows:
        _plot_scatter(all_rows, out_dir / "qed_vs_logp.png")
    return {"models": len(summaries), "molecules": len(all_rows), "out_dir": str(out_dir)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sdf", action="append", required=True, help="model=/path/to/file.sdf")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.sdf, args.out_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
