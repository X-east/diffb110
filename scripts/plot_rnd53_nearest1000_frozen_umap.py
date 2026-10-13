#!/usr/bin/env python3
"""Plot the RND53 nearest1000 aggregates on the existing frozen UMAP."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


RND_ORDER = [f"RND-{index:02d}" for index in range(1, 54)]
BACKGROUND_COLOR = "#A8ADB3"
ALL_COLOR = "#D95F59"
STRICT_COLOR = "#16803C"
REFERENCE_COLOR = "#111111"


def sha256(path: Path) -> str:
    """Return a file SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def load_inputs(input_dir: Path, background_path: Path, references_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Load pair, aggregate, strict, and frozen-background coordinates."""
    files = sorted(input_dir.glob("RND-*_nearest1000.csv.gz"))
    if len(files) != 53:
        raise ValueError(f"expected 53 nearest1000 files, found {len(files)}")
    pairs = pd.concat([pd.read_csv(path, low_memory=False) for path in files], ignore_index=True)
    unique = pd.read_csv(input_dir / "rnd53_nearest1000_unique.csv.gz", low_memory=False)
    strict = pd.read_csv(input_dir / "rnd53_nearest1000_strict_unique.csv.gz", low_memory=False)
    references = pd.read_csv(references_path, low_memory=False)
    background = np.load(background_path)
    if len(pairs) != 53_000 or len(unique) != 16_186 or len(strict) != 2_477:
        raise ValueError(f"unexpected counts: pairs={len(pairs)}, unique={len(unique)}, strict={len(strict)}")
    if sorted(pairs["epi_id"].unique()) != RND_ORDER or not pairs.groupby("epi_id").size().eq(1000).all():
        raise ValueError("nearest1000 per-reference counts are invalid")
    for frame, name in [(pairs, "pairs"), (unique, "unique"), (strict, "strict")]:
        if frame[["umap1", "umap2"]].isna().any().any():
            raise ValueError(f"{name} has missing UMAP coordinates")
    if background.shape != (100_000, 2) or not np.isfinite(background).all():
        raise ValueError(f"unexpected frozen background shape: {background.shape}")
    if len(references) != 53 or references["rnd_id"].tolist() != RND_ORDER:
        raise ValueError("reference coordinate table is not the expected RND53 table")
    return pairs, unique, strict, references, background


def add_background(axis: plt.Axes, background: np.ndarray) -> None:
    """Draw only the gray frozen ChEMBL reference background."""
    axis.scatter(background[:, 0], background[:, 1], s=0.7, color=BACKGROUND_COLOR, alpha=0.14, linewidths=0, rasterized=True)


def save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    """Save PNG, SVG, and PDF variants."""
    figure.savefig(output_dir / f"{stem}.png", dpi=320, bbox_inches="tight")
    figure.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def source_reference_count(frame: pd.DataFrame) -> int:
    """Count distinct RND references represented by provenance fields."""
    if "source_epi_ids" not in frame.columns:
        return int(frame["epi_id"].nunique())
    references: set[str] = set()
    for value in frame["source_epi_ids"].dropna().astype(str):
        references.update(item for item in value.split(";") if item)
    return len(references)


def set_limits(axes: np.ndarray, background: np.ndarray, points: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    """Set identical limits on all panels from the plotted coordinate range."""
    coordinates = np.vstack([background, points[["umap1", "umap2"]].to_numpy(float)])
    low = np.quantile(coordinates, 0.001, axis=0)
    high = np.quantile(coordinates, 0.999, axis=0)
    margin = np.maximum((high - low) * 0.04, 0.1)
    limits = ((float(low[0] - margin[0]), float(high[0] + margin[0])), (float(low[1] - margin[1]), float(high[1] + margin[1])))
    for axis in np.asarray(axes).ravel():
        axis.set_xlim(*limits[0])
        axis.set_ylim(*limits[1])
    return limits


def plot_overview(pairs: pd.DataFrame, unique: pd.DataFrame, strict: pd.DataFrame, references: pd.DataFrame, background: np.ndarray, output_dir: Path) -> None:
    """Plot aggregate and strict unique sets in separate panels."""
    figure, axes = plt.subplots(1, 2, figsize=(15, 6.8), sharex=True, sharey=True)
    for axis, frame, color, title in [
        (axes[0], unique, ALL_COLOR, f"All nearest1000 aggregate\nn={len(unique):,} trajectories; {unique.canonical_smiles.nunique():,} canonical"),
        (axes[1], strict, STRICT_COLOR, f"Strict reasonable subset\nn={len(strict):,} trajectories; {strict.canonical_smiles.nunique():,} canonical"),
    ]:
        add_background(axis, background)
        axis.scatter(frame["umap1"], frame["umap2"], s=5.5, color=color, alpha=0.45, linewidths=0, rasterized=True)
        axis.scatter(references["umap1"], references["umap2"], s=33, marker="D", facecolor=REFERENCE_COLOR, edgecolor="white", linewidth=0.4, zorder=4)
        axis.set_title(title)
        axis.grid(alpha=0.13)
        axis.set_xlabel("UMAP 1")
        axis.set_ylabel("UMAP 2")
    set_limits(axes, background, pd.concat([unique, strict], ignore_index=True))
    figure.suptitle("RND53 nearest1000 DiffSBDD molecules on the frozen ChEMBL UMAP", fontsize=18, y=0.995)
    figure.text(0.5, 0.01, "Gray = ChEMBL100k; colored points = only the set named in each panel; black diamonds = 53 external reference inhibitors", ha="center", fontsize=11)
    figure.tight_layout(rect=(0, 0.035, 1, 0.96))
    save_figure(figure, output_dir, "rnd53_nearest1000_aggregate_vs_strict_umap")


def plot_pair_and_unique(pairs: pd.DataFrame, unique: pd.DataFrame, references: pd.DataFrame, background: np.ndarray, output_dir: Path) -> None:
    """Show the effect of pair duplication versus trajectory aggregation."""
    figure, axes = plt.subplots(1, 2, figsize=(15, 6.8), sharex=True, sharey=True)
    for axis, frame, title, alpha in [
        (axes[0], pairs, f"All pair rows\nn={len(pairs):,}", 0.16),
        (axes[1], unique, f"Trajectory-aggregated\nn={len(unique):,}", 0.45),
    ]:
        add_background(axis, background)
        axis.scatter(frame["umap1"], frame["umap2"], s=5.0, color=ALL_COLOR, alpha=alpha, linewidths=0, rasterized=True)
        axis.scatter(references["umap1"], references["umap2"], s=33, marker="D", facecolor=REFERENCE_COLOR, edgecolor="white", linewidth=0.4, zorder=4)
        axis.set_title(title)
        axis.set_xlabel("UMAP 1")
        axis.set_ylabel("UMAP 2")
        axis.grid(alpha=0.13)
    set_limits(axes, background, pairs)
    figure.suptitle("Pair-level versus aggregated RND53 nearest1000 distribution", fontsize=18, y=0.995)
    figure.text(0.5, 0.01, "The left panel intentionally shows repeated selections; the right panel counts each trajectory once", ha="center", fontsize=11)
    figure.tight_layout(rect=(0, 0.035, 1, 0.96))
    save_figure(figure, output_dir, "rnd53_nearest1000_pair_vs_aggregate_umap")


def plot_reference_facets(pairs: pd.DataFrame, background: np.ndarray, output_dir: Path, strict: bool = False) -> None:
    """Plot one colored category per RND reference panel."""
    frame = pairs[pairs["strict_screen_pass"].eq(True)].copy() if strict else pairs
    title_prefix = "Strict nearest1000" if strict else "Nearest1000"
    figure, axes = plt.subplots(7, 8, figsize=(21, 18), sharex=True, sharey=True)
    for axis, epi_id in zip(axes.ravel(), RND_ORDER):
        add_background(axis, background)
        group = frame[frame["epi_id"].eq(epi_id)]
        axis.scatter(group["umap1"], group["umap2"], s=4.2, color=STRICT_COLOR if strict else ALL_COLOR, alpha=0.55, linewidths=0, rasterized=True)
        axis.set_title(f"{epi_id} | n={len(group):,}", fontsize=11)
        axis.grid(alpha=0.12)
    for axis in axes[-1, :]:
        axis.set_xlabel("UMAP 1")
    for axis in axes[:, 0]:
        axis.set_ylabel("UMAP 2")
    for axis in axes.ravel()[53:]:
        axis.axis("off")
    all_points = frame if len(frame) else pairs
    set_limits(axes, background, all_points)
    figure.suptitle(f"Each RND reference's {title_prefix} set", fontsize=19, y=0.995)
    figure.text(0.5, 0.012, "Gray = frozen ChEMBL100k; each panel contains only the named RND reference's selected molecules", ha="center", fontsize=12)
    figure.tight_layout(rect=(0, 0.025, 1, 0.975))
    save_figure(figure, output_dir, f"rnd53_nearest1000_{'strict_' if strict else ''}by_reference_facets")


def plot_rank_facets(pairs: pd.DataFrame, background: np.ndarray, output_dir: Path) -> None:
    """Plot rank intervals to show how the aggregate expands away from each reference."""
    bins = [(1, 10, "Ranks 1-10"), (11, 50, "Ranks 11-50"), (51, 200, "Ranks 51-200"), (201, 1000, "Ranks 201-1000")]
    colors = ["#7F1D1D", "#B91C1C", "#D95F59", "#F3A6A0"]
    figure, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=True)
    for axis, (low, high, label), color in zip(axes.ravel(), bins, colors):
        group = pairs[pairs["epi_rank"].between(low, high)]
        add_background(axis, background)
        axis.scatter(group["umap1"], group["umap2"], s=4.2, color=color, alpha=0.28, linewidths=0, rasterized=True)
        axis.set_title(f"{label} | n={len(group):,}")
        axis.set_xlabel("UMAP 1")
        axis.set_ylabel("UMAP 2")
        axis.grid(alpha=0.13)
    set_limits(axes, background, pairs)
    figure.suptitle("Frozen-UMAP distribution by ECFP4 rank within each nearest1000 set", fontsize=18, y=0.995)
    figure.text(0.5, 0.01, "Each panel contains only one rank interval; gray = ChEMBL100k background", ha="center", fontsize=11)
    figure.tight_layout(rect=(0, 0.035, 1, 0.96))
    save_figure(figure, output_dir, "rnd53_nearest1000_by_rank_facets")


def write_summary(pairs: pd.DataFrame, unique: pd.DataFrame, strict: pd.DataFrame, references: pd.DataFrame, background: np.ndarray, output_dir: Path) -> None:
    """Write nearest-background distances and per-reference summaries."""
    tree = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=18).fit(background)
    for frame in [unique, strict]:
        distances = tree.kneighbors(frame[["umap1", "umap2"]].to_numpy(), return_distance=True)[0][:, 0]
        frame["nearest_chembl_umap_distance"] = distances
    unique.to_csv(output_dir / "rnd53_nearest1000_unique_coordinates.csv.gz", index=False, compression="gzip")
    strict.to_csv(output_dir / "rnd53_nearest1000_strict_unique_coordinates.csv.gz", index=False, compression="gzip")
    pair_summary = pairs.groupby("epi_id", sort=False).agg(
        pair_rows=("trajectory_id", "size"), unique_trajectory_ids=("trajectory_id", "nunique"),
        median_ecfp4_tanimoto=("ecfp4_tanimoto", "median"), max_ecfp4_tanimoto=("ecfp4_tanimoto", "max"),
        median_umap1=("umap1", "median"), median_umap2=("umap2", "median"),
    ).reset_index()
    strict_by_epi = pairs[pairs["strict_screen_pass"].eq(True)].groupby("epi_id", sort=False).agg(
        strict_pair_rows=("trajectory_id", "size"), strict_unique_trajectory_ids=("trajectory_id", "nunique")
    ).reset_index()
    pair_summary = pair_summary.merge(strict_by_epi, on="epi_id", how="left").fillna(0)
    pair_summary = pair_summary.merge(references[["rnd_id", "preferred_name"]], left_on="epi_id", right_on="rnd_id", how="left").drop(columns=["rnd_id"])
    pair_summary.to_csv(output_dir / "rnd53_nearest1000_by_reference_summary.csv", index=False)
    rows = []
    for name, frame in [("all_pair_rows", pairs), ("all_trajectory_aggregate", unique), ("strict_pair_rows", pairs[pairs["strict_screen_pass"].eq(True)]), ("strict_trajectory_aggregate", strict)]:
        rows.append({
            "group": name,
            "rows": int(len(frame)),
            "unique_trajectory_ids": int(frame["trajectory_id"].nunique()),
            "unique_canonical_smiles": int(frame["canonical_smiles"].nunique()),
            "reference_count": source_reference_count(frame),
            "median_nearest_chembl_umap_distance": float(frame["nearest_chembl_umap_distance"].median()) if "nearest_chembl_umap_distance" in frame else np.nan,
            "umap1_mean": float(frame["umap1"].mean()),
            "umap2_mean": float(frame["umap2"].mean()),
        })
    pd.DataFrame(rows).to_csv(output_dir / "rnd53_nearest1000_umap_group_summary.csv", index=False)


def main() -> int:
    """Render frozen-UMAP plots and write the audit manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_plotting()
    pairs, unique, strict, references, background = load_inputs(args.input_dir, args.background, args.references)
    strict_ids = set(strict["trajectory_id"].astype(str))
    pairs["strict_screen_pass"] = pairs["trajectory_id"].astype(str).isin(strict_ids)
    write_summary(pairs, unique, strict, references, background, args.output_dir)
    plot_overview(pairs, unique, strict, references, background, args.output_dir)
    plot_pair_and_unique(pairs, unique, references, background, args.output_dir)
    plot_reference_facets(pairs, background, args.output_dir, strict=False)
    plot_reference_facets(pairs, background, args.output_dir, strict=True)
    plot_rank_facets(pairs, background, args.output_dir)
    manifest = {
        "status": "complete",
        "coordinate_system": "existing frozen old ChEMBL100k UMAP",
        "fit_performed": False,
        "transform_performed": False,
        "source_coordinates": "nearest1000 CSV coordinates were inherited from the verified frozen-coordinate facts table",
        "background": str(args.background.resolve()),
        "background_sha256": sha256(args.background),
        "references": str(args.references.resolve()),
        "input_dir": str(args.input_dir.resolve()),
        "input_summary": str((args.input_dir / "summary.json").resolve()),
        "counts": {
            "reference_count": 53,
            "nearest_per_reference": 1000,
            "pair_rows": len(pairs),
            "unique_trajectory_ids": int(unique["trajectory_id"].nunique()),
            "unique_canonical_smiles": int(unique["canonical_smiles"].nunique()),
            "strict_pair_rows": int(pairs["strict_screen_pass"].sum()),
            "strict_unique_trajectory_ids": len(strict),
            "strict_unique_canonical_smiles": int(strict["canonical_smiles"].nunique()),
            "all_aggregate_reference_count": source_reference_count(unique),
            "strict_aggregate_reference_count": source_reference_count(strict),
        },
        "interpretation_boundary": "Frozen UMAP positions and nearest-background distances are descriptive chemical-space context, not activity, docking, or formal positive labels.",
        "outputs": [
            "rnd53_nearest1000_aggregate_vs_strict_umap.png",
            "rnd53_nearest1000_pair_vs_aggregate_umap.png",
            "rnd53_nearest1000_by_reference_facets.png",
            "rnd53_nearest1000_strict_by_reference_facets.png",
            "rnd53_nearest1000_by_rank_facets.png",
            "rnd53_nearest1000_umap_group_summary.csv",
            "rnd53_nearest1000_by_reference_summary.csv",
        ],
    }
    (args.output_dir / "umap_overlay_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
