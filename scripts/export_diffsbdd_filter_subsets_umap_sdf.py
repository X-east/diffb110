#!/usr/bin/env python3
"""Export each frozen DiffSBDD chemistry-filter subset as UMAP data and SDF."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

FILTERS = {
    "qed_ge_0_4": "QED >= 0.4",
    "sa_le_6": "SA <= 6",
    "logp_in_neg1_to_5": "-1 <= LogP <= 5",
    "tpsa_in_20_to_140": "20 <= TPSA <= 140",
    "three_member_rings_eq_0": "3-member rings = 0",
    "fused_rings_le_4": "Fused rings <= 4",
    "complex_rings_le_1": "Complex rings <= 1",
    "brenk_eq_0": "Brenk alerts = 0",
    "chemistry_only_actual_good": "All original chemistry gates",
    "single_fragment": "Single molecular fragment",
    "allowed_organic_elements": "Allowed organic elements only",
    "mol_wt_in_120_to_600": "120 <= MW <= 600",
    "heavy_atoms_in_10_to_35": "10 <= heavy atoms <= 35",
    "hba_le_10": "HBA <= 10",
    "hbd_le_5": "HBD <= 5",
    "formal_charge_abs_le_1": "|Formal charge| <= 1",
    "lipinski_violations_le_1": "Lipinski violations <= 1",
    "carbon_fraction_le_0_85": "Carbon fraction <= 0.85",
    "normality_pre_chembl": "All normality gates before ChEMBL similarity",
}


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def plot_subset(reference: pd.DataFrame, valid: pd.DataFrame, selected: pd.DataFrame, title: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 8), dpi=180)
    axis.scatter(reference["umap1"], reference["umap2"], s=0.45, c="#A8ADB3", alpha=0.14, linewidths=0, label="ChEMBL100k")
    axis.scatter(valid["umap1"], valid["umap2"], s=0.5, c="#D95F59", alpha=0.08, linewidths=0, label="All DiffSBDD valid")
    axis.scatter(selected["umap1"], selected["umap2"], s=1.4, c="#16803C", alpha=0.38, linewidths=0, label="Selected")
    axis.set(title=f"DiffSBDD filter: {title}\npass {len(selected):,} / 100,000 attempts", xlabel="UMAP 1", ylabel="UMAP 2")
    axis.legend(markerscale=5, frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def main() -> int:
    parsed = args()
    parsed.output_dir.mkdir(parents=True, exist_ok=True)
    facts = pd.read_csv(parsed.facts, low_memory=False)
    reference = pd.read_csv(parsed.reference, low_memory=False)
    reference = reference[reference["model"].eq("ChEMBL36")][["umap1", "umap2"]]
    if len(facts) != 67_694 or len(reference) != 100_000:
        raise ValueError(f"unexpected counts facts={len(facts)} reference={len(reference)}")
    manifest = []
    writers: dict[str, Chem.SDWriter] = {}
    handles: dict[str, object] = {}
    try:
        for filter_id, description in FILTERS.items():
            subset_dir = parsed.output_dir / filter_id
            subset_dir.mkdir(parents=True, exist_ok=True)
            selected = facts[facts[filter_id].astype(bool)].copy()
            selected.to_csv(subset_dir / "selected_coordinates.csv.gz", index=False, compression="gzip")
            plot_subset(reference, facts, selected, description, subset_dir / "umap_projection.png")
            (subset_dir / "condition.txt").write_text(
                f"filter_id: {filter_id}\ncondition: {description}\n"
                f"fixed_attempts: 100000\nrdkit_valid_denominator: 67694\npass_count: {len(selected)}\n"
                "coordinate_system: old fixed five-model ChEMBL100k ECFP4 UMAP; generated molecules transformed only\n"
                "note: UMAP is descriptive and is not the positive-label criterion.\n",
                encoding="utf-8",
            )
            sdf_path = subset_dir / "selected_molecules.sdf"
            writers[filter_id] = Chem.SDWriter(str(sdf_path))
            writers[filter_id].SetKekulize(False)
            manifest.append({"filter_id": filter_id, "condition": description, "pass_count": int(len(selected)), "sdf": str(sdf_path), "coordinates": str(subset_dir / "selected_coordinates.csv.gz"), "plot": str(subset_dir / "umap_projection.png")})
        wanted = {filter_id: set(facts.loc[facts[filter_id].astype(bool), "trajectory_id"].astype(str)) for filter_id in FILTERS}
        matched = {filter_id: 0 for filter_id in FILTERS}
        for sdf_name in sorted(facts["sdf_path"].dropna().astype(str).unique()):
            supplier = Chem.SDMolSupplier(sdf_name, removeHs=False, sanitize=False)
            for molecule in supplier:
                if molecule is None or not molecule.HasProp("sample_id"):
                    continue
                molecule_id = molecule.GetProp("sample_id")
                for filter_id, ids in wanted.items():
                    if molecule_id in ids:
                        molecule.SetProp("filter_id", filter_id)
                        writers[filter_id].write(molecule)
                        matched[filter_id] += 1
        for writer in writers.values():
            writer.close()
    finally:
        for writer in writers.values():
            try:
                writer.close()
            except Exception:
                pass
    for row in manifest:
        row["sdf_molecule_count"] = matched[row["filter_id"]]
        if row["sdf_molecule_count"] != row["pass_count"]:
            raise RuntimeError(f"SDF count mismatch for {row['filter_id']}: {row['sdf_molecule_count']} != {row['pass_count']}")
    (parsed.output_dir / "manifest.json").write_text(json.dumps({"status": "complete", "coordinate_system": "old fixed five-model ChEMBL100k ECFP4 UMAP", "filters": manifest}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
