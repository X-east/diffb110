#!/usr/bin/env python3
"""Project the 53 x 100 RND-nearest DiffSBDD sets into a frozen UMAP."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/diffb110_rnd53_nearest100_umap_mpl")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/diffb110_rnd53_nearest100_umap_numba")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

RDLogger.DisableLog("rdApp.*")
LOGGER = logging.getLogger(__name__)


RND_ORDER = [f"RND-{index:02d}" for index in range(1, 54)]
BACKGROUND_COLOR = "#A8ADB3"
SIGNAL_COLOR = "#D95F59"
REFERENCE_COLOR = "#111111"


def sha256(path: Path) -> str:
    """Return a file SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bitvectors_to_csr(bit_vectors: list[object], n_bits: int = 2048) -> sparse.csr_matrix:
    """Convert RDKit bit vectors to the sparse matrix used by the UMAP model."""
    indptr = np.zeros(len(bit_vectors) + 1, dtype=np.int64)
    for index, bit_vector in enumerate(bit_vectors):
        indptr[index + 1] = indptr[index] + bit_vector.GetNumOnBits()
    indices = np.empty(int(indptr[-1]), dtype=np.int32)
    for index, bit_vector in enumerate(bit_vectors):
        start, stop = int(indptr[index]), int(indptr[index + 1])
        indices[start:stop] = np.fromiter(bit_vector.GetOnBits(), dtype=np.int32, count=stop - start)
    return sparse.csr_matrix(
        (np.ones(len(indices), dtype=np.float32), indices, indptr),
        shape=(len(bit_vectors), n_bits),
    )


def smiles_to_matrix(smiles: list[str]) -> sparse.csr_matrix:
    """Build ECFP4 fingerprints for canonical SMILES."""
    fingerprints: list[object] = []
    for value in smiles:
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"cannot parse canonical SMILES: {value}")
        fingerprints.append(AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=2048))
    return bitvectors_to_csr(fingerprints)


def read_pairs(input_dir: Path) -> pd.DataFrame:
    """Read and validate all 53 ranking tables."""
    paths = sorted(input_dir.glob("RND-*_nearest100.csv.gz"))
    if len(paths) != 53:
        raise ValueError(f"expected 53 ranking CSVs, found {len(paths)}")
    frames = [pd.read_csv(path, low_memory=False).assign(source_ranking_csv=str(path.resolve())) for path in paths]
    pairs = pd.concat(frames, ignore_index=True)
    required = {"epi_id", "epi_rank", "trajectory_id", "canonical_smiles", "ecfp4_tanimoto", "ecfp4_distance"}
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"ranking tables lack columns: {sorted(missing)}")
    if len(pairs) != 5_300:
        raise ValueError(f"expected 5,300 pair rows, found {len(pairs):,}")
    if sorted(pairs["epi_id"].unique()) != RND_ORDER:
        raise ValueError("RND IDs are incomplete or out of order")
    counts = pairs.groupby("epi_id", sort=False).size()
    if not counts.eq(100).all():
        raise ValueError(f"each RND ID must have 100 rows: {counts.to_dict()}")
    if pairs["canonical_smiles"].isna().any():
        raise ValueError("nearest100 tables contain missing canonical SMILES")
    return pairs


def read_references(path: Path, model: object) -> pd.DataFrame:
    """Read the 53 reference inhibitors and project them with the same model."""
    rows: list[dict[str, object]] = []
    supplier = Chem.SDMolSupplier(str(path), removeHs=True, sanitize=True)
    if len(supplier) != 53:
        raise ValueError(f"expected 53 reference molecules, found {len(supplier)}")
    smiles: list[str] = []
    for index, molecule in enumerate(supplier, start=1):
        if molecule is None:
            raise ValueError(f"reference RND-{index:02d} failed RDKit parsing")
        smiles.append(Chem.MolToSmiles(molecule, canonical=True))
        rows.append(
            {
                "epi_id": f"RND-{index:02d}",
                "preferred_name": molecule.GetProp("Preferred_Name") if molecule.HasProp("Preferred_Name") else "",
                "canonical_smiles": smiles[-1],
            }
        )
    coordinates = model.transform(smiles_to_matrix(smiles))
    references = pd.DataFrame(rows)
    references["umap1"] = coordinates[:, 0]
    references["umap2"] = coordinates[:, 1]
    return references


def project_pairs(pairs: pd.DataFrame, model: object) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Project each unique selected molecule once and join coordinates to pairs."""
    unique = pairs[["canonical_smiles"]].drop_duplicates().reset_index(drop=True)
    LOGGER.info("projecting %d unique molecules from %d pair rows", len(unique), len(pairs))
    coordinates = model.transform(smiles_to_matrix(unique["canonical_smiles"].astype(str).tolist()))
    unique["umap1"] = coordinates[:, 0]
    unique["umap2"] = coordinates[:, 1]
    pairs_with_coordinates = pairs.merge(unique, on="canonical_smiles", how="left", validate="many_to_one")
    if pairs_with_coordinates[["umap1", "umap2"]].isna().any().any():
        raise ValueError("some nearest100 rows did not receive UMAP coordinates")
    return pairs_with_coordinates, unique


def limits(reference_coordinates: np.ndarray, points: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    """Use shared limits based on the ChEMBL background and projected points."""
    coordinates = np.vstack([reference_coordinates, points[["umap1", "umap2"]].to_numpy(float)])
    low = np.quantile(coordinates, 0.001, axis=0)
    high = np.quantile(coordinates, 0.999, axis=0)
    margin = np.maximum((high - low) * 0.04, 0.1)
    return (float(low[0] - margin[0]), float(high[0] + margin[0])), (float(low[1] - margin[1]), float(high[1] + margin[1]))


def configure_plotting() -> None:
    """Configure readable Python/matplotlib output."""
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
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def draw_background(axis: plt.Axes, reference_coordinates: np.ndarray) -> None:
    """Draw only the gray frozen ChEMBL background."""
    axis.scatter(
        reference_coordinates[:, 0],
        reference_coordinates[:, 1],
        s=0.7,
        color=BACKGROUND_COLOR,
        alpha=0.14,
        linewidths=0,
        rasterized=True,
    )


def save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    """Save the standard PNG, SVG, and PDF variants."""
    figure.savefig(output_dir / f"{stem}.png", dpi=320, bbox_inches="tight")
    figure.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_overview(pairs: pd.DataFrame, references: pd.DataFrame, background: np.ndarray, output_dir: Path, plot_limits: tuple[tuple[float, float], tuple[float, float]]) -> None:
    """Plot all 5,300 pair rows and the 53 reference inhibitors."""
    figure, axis = plt.subplots(figsize=(10.5, 8.2))
    draw_background(axis, background)
    axis.scatter(pairs["umap1"], pairs["umap2"], s=5.0, color=SIGNAL_COLOR, alpha=0.28, linewidths=0, rasterized=True, label="RND53 nearest100 pair rows (n=5,300)")
    axis.scatter(references["umap1"], references["umap2"], s=42, marker="D", facecolor=REFERENCE_COLOR, edgecolor="white", linewidth=0.45, zorder=5, label="53 reference inhibitors")
    axis.set_xlim(*plot_limits[0])
    axis.set_ylim(*plot_limits[1])
    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2")
    axis.set_title("RND53 nearest100 DiffSBDD molecules on the frozen ChEMBL UMAP")
    axis.legend(frameon=False, markerscale=1.5, loc="best")
    axis.grid(alpha=0.13)
    save_figure(figure, output_dir, "rnd53_nearest100_overview_umap")


def plot_unique(pairs: pd.DataFrame, references: pd.DataFrame, background: np.ndarray, output_dir: Path, plot_limits: tuple[tuple[float, float], tuple[float, float]]) -> None:
    """Plot the 2,068 unique selected molecules without pair duplication."""
    unique = pairs.drop_duplicates("canonical_smiles")
    figure, axis = plt.subplots(figsize=(10.5, 8.2))
    draw_background(axis, background)
    axis.scatter(unique["umap1"], unique["umap2"], s=6.0, color=SIGNAL_COLOR, alpha=0.45, linewidths=0, rasterized=True, label=f"Unique nearest100 molecules (n={len(unique):,})")
    axis.scatter(references["umap1"], references["umap2"], s=42, marker="D", facecolor=REFERENCE_COLOR, edgecolor="white", linewidth=0.45, zorder=5, label="53 reference inhibitors")
    axis.set_xlim(*plot_limits[0])
    axis.set_ylim(*plot_limits[1])
    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2")
    axis.set_title("Unique molecules selected as ECFP4 nearest neighbors to RND53")
    axis.legend(frameon=False, markerscale=1.5, loc="best")
    axis.grid(alpha=0.13)
    save_figure(figure, output_dir, "rnd53_nearest100_unique_umap")


def plot_reference_facets(pairs: pd.DataFrame, background: np.ndarray, output_dir: Path, plot_limits: tuple[tuple[float, float], tuple[float, float]]) -> None:
    """Plot one panel per reference, with only that reference's 100 points colored."""
    figure, axes = plt.subplots(7, 8, figsize=(21, 18), sharex=True, sharey=True)
    for axis, epi_id in zip(axes.ravel(), RND_ORDER):
        draw_background(axis, background)
        group = pairs[pairs["epi_id"].eq(epi_id)]
        axis.scatter(group["umap1"], group["umap2"], s=5.5, color=SIGNAL_COLOR, alpha=0.7, linewidths=0, rasterized=True)
        axis.set_title(f"{epi_id} | n={len(group)}", fontsize=11)
        axis.set_xlim(*plot_limits[0])
        axis.set_ylim(*plot_limits[1])
        axis.grid(alpha=0.12)
    for axis in axes[-1, :]:
        axis.set_xlabel("UMAP 1")
    for axis in axes[:, 0]:
        axis.set_ylabel("UMAP 2")
    for axis in axes.ravel()[53:]:
        axis.axis("off")
    figure.suptitle("Each RND reference's 100 ECFP4-nearest DiffSBDD molecules", fontsize=19, y=0.995)
    figure.text(0.5, 0.012, "Gray = frozen ChEMBL100k background; red = only the named RND nearest100 set in that panel", ha="center", fontsize=12)
    figure.tight_layout(rect=(0, 0.025, 1, 0.975))
    save_figure(figure, output_dir, "rnd53_nearest100_by_reference_facets")


def plot_rank_facets(pairs: pd.DataFrame, background: np.ndarray, output_dir: Path, plot_limits: tuple[tuple[float, float], tuple[float, float]]) -> None:
    """Plot four rank bins to show how far nearest100 members spread in UMAP."""
    bins = [(1, 10, "Ranks 1-10"), (11, 25, "Ranks 11-25"), (26, 50, "Ranks 26-50"), (51, 100, "Ranks 51-100")]
    figure, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=True)
    colors = ["#7F1D1D", "#B91C1C", "#D95F59", "#F3A6A0"]
    for axis, (low_rank, high_rank, label), color in zip(axes.ravel(), bins, colors):
        draw_background(axis, background)
        group = pairs[pairs["epi_rank"].between(low_rank, high_rank)]
        axis.scatter(group["umap1"], group["umap2"], s=4.4, color=color, alpha=0.34, linewidths=0, rasterized=True)
        axis.set_title(f"{label} | pair rows n={len(group):,}")
        axis.set_xlim(*plot_limits[0])
        axis.set_ylim(*plot_limits[1])
        axis.grid(alpha=0.13)
    axes[1, 0].set_xlabel("UMAP 1")
    axes[1, 1].set_xlabel("UMAP 1")
    axes[0, 0].set_ylabel("UMAP 2")
    axes[1, 0].set_ylabel("UMAP 2")
    figure.suptitle("UMAP distribution by ECFP4 rank within each RND nearest100 set", fontsize=18, y=0.995)
    figure.text(0.5, 0.012, "Gray = frozen ChEMBL100k background; each panel contains only one rank interval", ha="center", fontsize=12)
    figure.tight_layout(rect=(0, 0.03, 1, 0.965))
    save_figure(figure, output_dir, "rnd53_nearest100_by_rank_facets")


def write_summaries(pairs: pd.DataFrame, unique: pd.DataFrame, references: pd.DataFrame, background: np.ndarray, output_dir: Path) -> None:
    """Write per-reference and global UMAP summary tables."""
    nearest = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=18).fit(background)
    distances = nearest.kneighbors(unique[["umap1", "umap2"]].to_numpy(), return_distance=True)[0][:, 0]
    unique = unique.copy()
    unique["nearest_chembl_umap_distance"] = distances
    unique.to_csv(output_dir / "rnd53_nearest100_unique_coordinates.csv.gz", index=False, compression="gzip")
    pairs = pairs.merge(unique[["canonical_smiles", "nearest_chembl_umap_distance"]], on="canonical_smiles", how="left", validate="many_to_one")
    pairs.to_csv(output_dir / "rnd53_nearest100_pair_coordinates.csv.gz", index=False, compression="gzip")

    rows: list[dict[str, object]] = []
    for epi_id in RND_ORDER:
        group = pairs[pairs["epi_id"].eq(epi_id)]
        unique_group = group.drop_duplicates("canonical_smiles")
        rows.append(
            {
                "epi_id": epi_id,
                "preferred_name": str(references.loc[references["epi_id"].eq(epi_id), "preferred_name"].iloc[0]),
                "pair_rows": int(len(group)),
                "unique_molecules": int(len(unique_group)),
                "median_ecfp4_tanimoto": float(group["ecfp4_tanimoto"].median()),
                "min_ecfp4_tanimoto": float(group["ecfp4_tanimoto"].min()),
                "median_nearest_chembl_umap_distance": float(unique_group["nearest_chembl_umap_distance"].median()),
                "umap1_mean": float(unique_group["umap1"].mean()),
                "umap2_mean": float(unique_group["umap2"].mean()),
                "umap1_std": float(unique_group["umap1"].std()),
                "umap2_std": float(unique_group["umap2"].std()),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "rnd53_nearest100_by_reference_summary.csv", index=False)
    global_unique = unique
    global_summary = {
        "pair_rows": int(len(pairs)),
        "unique_molecules": int(len(global_unique)),
        "unique_trajectory_ids": int(pairs["trajectory_id"].nunique()),
        "reference_count": 53,
        "nearest100_per_reference": 100,
        "median_nearest_chembl_umap_distance": float(global_unique["nearest_chembl_umap_distance"].median()),
        "p90_nearest_chembl_umap_distance": float(global_unique["nearest_chembl_umap_distance"].quantile(0.90)),
        "umap1_mean": float(global_unique["umap1"].mean()),
        "umap2_mean": float(global_unique["umap2"].mean()),
        "umap1_std": float(global_unique["umap1"].std()),
        "umap2_std": float(global_unique["umap2"].std()),
        "mean_ecfp4_tanimoto_pair_rows": float(pairs["ecfp4_tanimoto"].mean()),
        "median_ecfp4_tanimoto_pair_rows": float(pairs["ecfp4_tanimoto"].median()),
    }
    (output_dir / "summary.json").write_text(json.dumps(global_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    """Run the frozen-UMAP projection and render evidence figures."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference-coordinates", type=Path, required=True)
    parser.add_argument("--reference-sdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler(args.output_dir / "projection.log", mode="w", encoding="utf-8")])
    started = time.time()
    configure_plotting()
    pairs = read_pairs(args.input_dir)
    LOGGER.info("loaded pair rows=%d unique canonical=%d unique trajectories=%d", len(pairs), pairs["canonical_smiles"].nunique(), pairs["trajectory_id"].nunique())
    model = joblib.load(args.model)
    background = np.load(args.reference_coordinates)
    if background.shape != (100_000, 2) or not np.isfinite(background).all():
        raise ValueError(f"expected finite ChEMBL background coordinates (100000, 2), found {background.shape}")
    references = read_references(args.reference_sdf, model)
    projected_pairs, unique = project_pairs(pairs, model)
    plot_limits = limits(background, projected_pairs)
    write_summaries(projected_pairs, unique, references, background, args.output_dir)
    plot_overview(projected_pairs, references, background, args.output_dir, plot_limits)
    plot_unique(projected_pairs, references, background, args.output_dir, plot_limits)
    plot_reference_facets(projected_pairs, background, args.output_dir, plot_limits)
    plot_rank_facets(projected_pairs, background, args.output_dir, plot_limits)
    manifest = {
        "status": "complete",
        "fit_performed": False,
        "transform_performed": True,
        "coordinate_system": "existing frozen old ChEMBL100k UMAP",
        "model": str(args.model.resolve()),
        "model_sha256": sha256(args.model),
        "reference_coordinates": str(args.reference_coordinates.resolve()),
        "reference_coordinates_shape": list(background.shape),
        "reference_sdf": str(args.reference_sdf.resolve()),
        "reference_sdf_sha256": sha256(args.reference_sdf),
        "nearest100_input_dir": str(args.input_dir.resolve()),
        "reference_count": 53,
        "nearest100_per_reference": 100,
        "pair_rows": int(len(projected_pairs)),
        "unique_canonical_molecules": int(projected_pairs["canonical_smiles"].nunique()),
        "unique_trajectory_ids": int(projected_pairs["trajectory_id"].nunique()),
        "fingerprint": {"type": "Morgan/ECFP4", "radius": 2, "n_bits": 2048},
        "interpretation_boundary": "UMAP positions and nearest-background distances are descriptive frozen chemical-space context, not activity or molecular-quality labels.",
        "elapsed_seconds": round(time.time() - started, 3),
        "outputs": [
            "rnd53_nearest100_pair_coordinates.csv.gz",
            "rnd53_nearest100_unique_coordinates.csv.gz",
            "rnd53_nearest100_by_reference_summary.csv",
            "summary.json",
            "rnd53_nearest100_overview_umap.png",
            "rnd53_nearest100_unique_umap.png",
            "rnd53_nearest100_by_reference_facets.png",
            "rnd53_nearest100_by_rank_facets.png",
        ],
    }
    (args.output_dir / "projection_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("complete pair_rows=%d unique=%d elapsed=%.1fs", len(projected_pairs), len(unique), time.time() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
