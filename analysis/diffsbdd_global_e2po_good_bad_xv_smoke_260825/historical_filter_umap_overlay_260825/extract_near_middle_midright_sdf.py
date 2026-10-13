#!/usr/bin/env python3
"""Extract ChEMBL-near molecules from the visually sparse middle UMAP region."""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/diffb110_near_region_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw


NEAR_DISTANCE = 0.01
REGION_X_LEFT = -1.0
REGION_X_RIGHT = 4.0
RIGHT_ISLAND_X = 4.0
SOURCE_RELATIVE = Path("analysis/offline_iql_100k_reasonableness_schemes_260819/expanded_reasonableness_facts.csv.gz")
REFERENCE_RELATIVE = Path("analysis/diffsbdd_global_e2po_good_bad_xv_smoke_260825/umap_original_model_projection_260825/chembl_reference_umap_coordinates.npy")


def configure_plotting() -> None:
    """Configure the Python/matplotlib plotting backend."""
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def set_property(molecule: Chem.Mol, key: str, value: object) -> None:
    """Write a finite or textual value as an SDF property."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        value = "NA"
    molecule.SetProp(key, str(value))


def load_sdf_by_smiles(path: Path) -> dict[str, list[tuple[int, Chem.Mol]]]:
    """Index successful SDF records by canonical SMILES.

    ``mol_index`` refers to the original fixed-attempt slot. Failed attempts
    are absent from the SDF, so it cannot be used as the SDF record index.
    """
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
    by_smiles: dict[str, list[tuple[int, Chem.Mol]]] = {}
    for record_index, molecule in enumerate(supplier):
        if molecule is None:
            continue
        canonical = Chem.MolToSmiles(molecule, canonical=True)
        by_smiles.setdefault(canonical, []).append((record_index, molecule))
    return by_smiles


def load_and_select(repo: Path) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Load historical facts and return near, selected, and reference coordinates."""
    source = repo / SOURCE_RELATIVE
    facts = pd.read_csv(source, low_memory=False)
    required = {"molecule_id", "trajectory_id", "sdf_path", "mol_index", "canonical_smiles", "umap1", "umap2", "umap_nearest_chembl_distance"}
    missing = sorted(required - set(facts.columns))
    if missing:
        raise ValueError(f"facts table is missing columns: {missing}")
    near = facts[facts["umap_nearest_chembl_distance"].le(NEAR_DISTANCE)].copy()
    selected = near[near["umap1"].ge(REGION_X_LEFT) & near["umap1"].lt(REGION_X_RIGHT)].copy()
    if selected["molecule_id"].duplicated().any():
        raise ValueError("selected molecule IDs are not unique")
    reference = np.load(repo / REFERENCE_RELATIVE)
    if reference.ndim != 2 or reference.shape[1] != 2 or not np.isfinite(reference).all():
        raise ValueError("reference coordinates are invalid")
    return near, selected, reference


def export_sdf(selected: pd.DataFrame, output: Path) -> tuple[int, list[dict[str, object]]]:
    """Recover selected structures from source SDF shards and write one SDF."""
    sdf_path = output / "chembl_near_middle_midright_umap1_m1_to_4.sdf"
    writer = Chem.SDWriter(str(sdf_path))
    if writer is None:
        raise RuntimeError(f"could not open SDF writer: {sdf_path}")
    exported = 0
    failures: list[dict[str, object]] = []
    for source_path, group in selected.groupby("sdf_path", sort=False):
        path = Path(str(source_path))
        if not path.exists():
            failures.append({"source_sdf": str(path), "reason": "missing_source_sdf", "n": int(len(group))})
            continue
        by_smiles = load_sdf_by_smiles(path)
        for row in group.itertuples(index=False):
            fact_index = int(row.mol_index)
            matches = by_smiles.get(str(row.canonical_smiles), [])
            if not matches:
                failures.append({"source_sdf": str(path), "molecule_id": str(row.molecule_id), "mol_index": fact_index, "reason": "canonical_smiles_not_found"})
                continue
            record_index, molecule = matches.pop(0)
            set_property(molecule, "source_molecule_id", row.molecule_id)
            set_property(molecule, "trajectory_id", row.trajectory_id)
            set_property(molecule, "source_sdf", source_path)
            set_property(molecule, "source_mol_index", fact_index)
            set_property(molecule, "source_sdf_record_index", record_index)
            set_property(molecule, "umap1", row.umap1)
            set_property(molecule, "umap2", row.umap2)
            set_property(molecule, "umap_nearest_chembl_distance", row.umap_nearest_chembl_distance)
            set_property(molecule, "umap_region", "chembl_near_middle_midright_umap1_m1_to_4")
            for column in ["qed", "sa_raw", "logp", "tpsa", "mol_wt", "heavy_atom_count", "carbon_fraction", "fragment_count", "fused_ring_count", "complex_ring_count", "brenk_count"]:
                if hasattr(row, column):
                    set_property(molecule, column, getattr(row, column))
            writer.write(molecule)
            exported += 1
    writer.close()
    return exported, failures


def write_region_summary(near: pd.DataFrame, selected: pd.DataFrame, output: Path) -> pd.DataFrame:
    """Write horizontal-region counts to make the visual selection auditable."""
    bins = [-8.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0]
    labels = [f"[{left:g},{right:g})" for left, right in zip(bins[:-1], bins[1:])]
    region = pd.cut(near["umap1"], bins=bins, labels=labels, right=False, include_lowest=True)
    summary = near.assign(umap1_region=region).groupby("umap1_region", observed=False).agg(
        n=("molecule_id", "size"),
        fraction_of_near=("molecule_id", lambda values: len(values) / len(near)),
        median_umap2=("umap2", "median"),
        median_distance=("umap_nearest_chembl_distance", "median"),
    ).reset_index()
    selected_label = f"selected [{REGION_X_LEFT:g},{REGION_X_RIGHT:g})"
    summary = pd.concat(
        [
            summary,
            pd.DataFrame(
                [{
                    "umap1_region": selected_label,
                    "n": len(selected),
                    "fraction_of_near": len(selected) / len(near),
                    "median_umap2": selected["umap2"].median(),
                    "median_distance": selected["umap_nearest_chembl_distance"].median(),
                }]
            ),
        ],
        ignore_index=True,
    )
    summary.to_csv(output / "near_umap1_region_summary.csv", index=False)
    return summary


def plot_region(near: pd.DataFrame, selected: pd.DataFrame, reference: np.ndarray, output: Path) -> None:
    """Plot all near points and highlight the extracted middle/mid-right region."""
    figure, axes = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [1.7, 1]})
    axes[0].scatter(reference[:, 0], reference[:, 1], s=1, color="#C7CCD1", alpha=0.10, linewidths=0, rasterized=True)
    axes[0].scatter(near["umap1"], near["umap2"], s=3, color="#0F766E", alpha=0.16, linewidths=0, rasterized=True, label="all ChEMBL-near")
    axes[0].scatter(selected["umap1"], selected["umap2"], s=8, color="#DC2626", alpha=0.62, linewidths=0, rasterized=True, label=f"extracted region (n={len(selected):,})")
    axes[0].axvspan(REGION_X_LEFT, REGION_X_RIGHT, color="#DC2626", alpha=0.05)
    axes[0].axvline(REGION_X_LEFT, color="#DC2626", linestyle="--", linewidth=1)
    axes[0].axvline(REGION_X_RIGHT, color="#DC2626", linestyle="--", linewidth=1)
    axes[0].set_title("ChEMBL-near points and extracted region")
    axes[0].set_xlabel("UMAP1")
    axes[0].set_ylabel("UMAP2")
    axes[0].legend(frameon=False, fontsize=10)
    axes[0].grid(alpha=0.13)

    axes[1].hist(near["umap1"], bins=60, color="#0F766E", alpha=0.35, label="all ChEMBL-near")
    axes[1].hist(selected["umap1"], bins=60, color="#DC2626", alpha=0.72, label="extracted region")
    axes[1].axvline(REGION_X_LEFT, color="#DC2626", linestyle="--", linewidth=1)
    axes[1].axvline(REGION_X_RIGHT, color="#DC2626", linestyle="--", linewidth=1)
    axes[1].set_title("UMAP1 distribution within ChEMBL-near")
    axes[1].set_xlabel("UMAP1")
    axes[1].set_ylabel("Molecules")
    axes[1].legend(frameon=False, fontsize=10)
    axes[1].grid(axis="y", alpha=0.15)
    figure.suptitle("Inspection of sparse middle/mid-right ChEMBL-near molecules", fontsize=16, y=1.01)
    figure.tight_layout()
    figure.savefig(output / "near_middle_midright_region.png", dpi=320, bbox_inches="tight")
    figure.savefig(output / "near_middle_midright_region.svg", bbox_inches="tight")
    plt.close(figure)


def write_structure_grid(selected: pd.DataFrame, output: Path, max_molecules: int = 36) -> tuple[int, list[str]]:
    """Draw a small stratified structure sample from the exported region."""
    if len(selected) <= max_molecules:
        sample = selected.copy()
    else:
        selected = selected.copy()
        selected["x_bin"] = pd.cut(selected["umap1"], bins=[-1, 0, 1, 2, 3, 4], right=False, include_lowest=True)
        pieces = []
        per_bin = max_molecules // 5
        for _, group in selected.groupby("x_bin", observed=False):
            pieces.append(group.sample(n=min(per_bin, len(group)), random_state=42))
        sample = pd.concat(pieces, ignore_index=True).head(max_molecules)
    molecules: list[Chem.Mol] = []
    legends: list[str] = []
    failures: list[str] = []
    for source_path, group in sample.groupby("sdf_path", sort=False):
        by_smiles = load_sdf_by_smiles(Path(str(source_path)))
        for row in group.itertuples(index=False):
            matches = by_smiles.get(str(row.canonical_smiles), [])
            if not matches:
                failures.append(str(row.molecule_id))
                continue
            _, molecule = matches.pop(0)
            molecules.append(molecule)
            legends.append(f"{row.molecule_id}\nx={row.umap1:.2f}, y={row.umap2:.2f}")
    image = Draw.MolsToGridImage(molecules, molsPerRow=6, subImgSize=(260, 220), legends=legends, useSVG=False)
    image.save(output / "near_middle_midright_structure_examples.png")
    return len(molecules), failures


def main() -> int:
    """Extract, visualize, and audit the requested ChEMBL-near UMAP region."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    configure_plotting()
    near, selected, reference = load_and_select(args.repo)
    exported, failures = export_sdf(selected, args.output)
    selected.to_csv(args.output / "chembl_near_middle_midright_umap1_m1_to_4.csv.gz", index=False)
    summary = write_region_summary(near, selected, args.output)
    plot_region(near, selected, reference, args.output)
    grid_count, grid_failures = write_structure_grid(selected, args.output)
    manifest = {
        "status": "complete" if exported == len(selected) and not failures else "completed_with_failures",
        "source_facts": str((args.repo / SOURCE_RELATIVE).resolve()),
        "near_definition": "umap_nearest_chembl_distance <= 0.01",
        "selected_region": f"{REGION_X_LEFT} <= UMAP1 < {REGION_X_RIGHT}",
        "near_count": int(len(near)),
        "selected_count": int(len(selected)),
        "selected_fraction_of_near": float(len(selected) / len(near)),
        "exported_sdf_count": int(exported),
        "sdf_failures": failures,
        "structure_grid_count": int(grid_count),
        "structure_grid_failures": grid_failures,
        "right_island_definition": f"UMAP1 >= {RIGHT_ISLAND_X}",
        "right_island_count": int((near["umap1"] >= RIGHT_ISLAND_X).sum()),
        "outputs": [
            "chembl_near_middle_midright_umap1_m1_to_4.sdf",
            "chembl_near_middle_midright_umap1_m1_to_4.csv.gz",
            "near_umap1_region_summary.csv",
            "near_middle_midright_region.png",
            "near_middle_midright_region.svg",
            "near_middle_midright_structure_examples.png",
        ],
    }
    (args.output / "near_middle_midright_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"near_count": len(near), "selected_count": len(selected), "exported_sdf_count": exported, "failures": len(failures), "right_island_count": manifest["right_island_count"]}, ensure_ascii=False))
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
