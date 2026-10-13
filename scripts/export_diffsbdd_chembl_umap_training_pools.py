#!/usr/bin/env python3
"""Export six fixed-coordinate ChEMBL-UMAP DiffSBDD training pools."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

SCHEMES = {
    "normality_nearest_chembl_le_0_01": ("normality_pre_chembl + nearest ChEMBL UMAP distance <= 0.01", "umap_nearest_chembl_distance", 0.01),
    "normality_nearest_chembl_le_0_02": ("normality_pre_chembl + nearest ChEMBL UMAP distance <= 0.02", "umap_nearest_chembl_distance", 0.02),
    "normality_nearest_chembl_le_0_03": ("normality_pre_chembl + nearest ChEMBL UMAP distance <= 0.03", "umap_nearest_chembl_distance", 0.03),
    "normality_chembl_k20_le_0_08": ("normality_pre_chembl + 20th ChEMBL-neighbor UMAP distance <= 0.08", "umap_chembl_k20_distance", 0.08),
    "normality_chembl_k20_le_0_10": ("normality_pre_chembl + 20th ChEMBL-neighbor UMAP distance <= 0.10", "umap_chembl_k20_distance", 0.10),
    "normality_chembl_k20_le_0_12": ("normality_pre_chembl + 20th ChEMBL-neighbor UMAP distance <= 0.12", "umap_chembl_k20_distance", 0.12),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--umap-metrics", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def write_plot(chembl: pd.DataFrame, generated: pd.DataFrame, selected: pd.DataFrame, title: str, output: Path) -> None:
    """Render one fixed-coordinate UMAP overlay."""
    fig, ax = plt.subplots(figsize=(10, 8), dpi=180)
    ax.scatter(chembl.umap1, chembl.umap2, s=0.45, c="#A8ADB3", alpha=0.14, linewidths=0, label="ChEMBL100k")
    ax.scatter(generated.umap1, generated.umap2, s=0.45, c="#D95F59", alpha=0.06, linewidths=0, label="All DiffSBDD valid")
    ax.scatter(selected.umap1, selected.umap2, s=1.6, c="#16803C", alpha=0.48, linewidths=0, label="Selected pool")
    ax.set(title=f"{title}\npass {len(selected):,} / 100,000 attempts", xlabel="UMAP 1", ylabel="UMAP 2")
    ax.legend(frameon=False, markerscale=5)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    facts = pd.read_csv(args.facts, low_memory=False)
    metrics = pd.read_csv(args.umap_metrics, low_memory=False)
    generated = facts.merge(metrics, on=["trajectory_id", "canonical_smiles", "umap1", "umap2"], how="inner", validate="one_to_one")
    ref = pd.read_csv(args.reference, low_memory=False)
    chembl = ref[ref.model.eq("ChEMBL36")][["umap1", "umap2"]]
    if len(generated) != 67_694 or len(chembl) != 100_000:
        raise ValueError(f"unexpected inputs: generated={len(generated)}, chembl={len(chembl)}")
    wanted: dict[str, set[str]] = {}
    manifest = []
    for scheme_id, (condition, column, threshold) in SCHEMES.items():
        subset_dir = args.output_dir / scheme_id
        subset_dir.mkdir(parents=True, exist_ok=True)
        selected = generated[generated.normality_pre_chembl.astype(bool) & generated[column].le(threshold)].copy()
        selected.to_csv(subset_dir / "selected_coordinates_and_facts.csv.gz", index=False, compression="gzip")
        write_plot(chembl, generated, selected, condition, subset_dir / "umap_projection.png")
        (subset_dir / "condition.txt").write_text(
            f"scheme_id: {scheme_id}\ncondition: {condition}\nfixed_attempts: 100000\n"
            f"rdkit_valid: 67694\npass_count: {len(selected)}\n"
            "umap: existing old five-model ChEMBL100k coordinate system; no fit or transform was run for this export.\n"
            "interpretation: candidate training pool, not formal actual-good labels.\n", encoding="utf-8")
        wanted[scheme_id] = set(selected.trajectory_id.astype(str))
        manifest.append({"scheme_id": scheme_id, "condition": condition, "pass_count": int(len(selected)), "sdf": str(subset_dir / "selected_molecules.sdf"), "coordinates": str(subset_dir / "selected_coordinates_and_facts.csv.gz"), "plot": str(subset_dir / "umap_projection.png")})
    writers = {scheme_id: Chem.SDWriter(str(args.output_dir / scheme_id / "selected_molecules.sdf")) for scheme_id in SCHEMES}
    for writer in writers.values():
        writer.SetKekulize(False)
    written = dict.fromkeys(SCHEMES, 0)
    try:
        for source_sdf in sorted(generated.sdf_path.dropna().astype(str).unique()):
            for mol in Chem.SDMolSupplier(source_sdf, removeHs=False, sanitize=False):
                if mol is None or not mol.HasProp("sample_id"):
                    continue
                molecule_id = mol.GetProp("sample_id")
                for scheme_id, selected_ids in wanted.items():
                    if molecule_id in selected_ids:
                        mol.SetProp("training_pool_scheme", scheme_id)
                        writers[scheme_id].write(mol)
                        written[scheme_id] += 1
    finally:
        for writer in writers.values():
            writer.close()
    for row in manifest:
        row["sdf_molecule_count"] = written[row["scheme_id"]]
        if row["pass_count"] != row["sdf_molecule_count"]:
            raise RuntimeError(f"SDF mismatch for {row['scheme_id']}")
    (args.output_dir / "manifest.json").write_text(json.dumps({"status": "complete", "coordinate_system": "existing old five-model ChEMBL100k UMAP", "schemes": manifest}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
