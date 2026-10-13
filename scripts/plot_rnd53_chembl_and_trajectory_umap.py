#!/usr/bin/env python3
"""Plot RND53 against fixed ChEMBL100k and DiffSBDD trajectory UMAP coordinates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, Delaunay, cKDTree


CHEMBL_COLOR = "#A8ADB3"
DIFFSBDD_COLOR = "#D95F59"
RND_COLOR = "#111111"


def parse_args() -> argparse.Namespace:
    """Parse input and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-coordinates", type=Path, required=True)
    parser.add_argument("--trajectory-coordinates", type=Path, required=True)
    parser.add_argument("--rnd53-coordinates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def configure_plotting() -> None:
    """Configure readable, portable static-figure typography."""
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 14,
            "axes.titlesize": 19,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and validate the fixed-coordinate input tables."""
    reference = pd.read_csv(args.reference_coordinates, low_memory=False)
    trajectory = pd.read_csv(args.trajectory_coordinates, low_memory=False)
    rnd53 = pd.read_csv(args.rnd53_coordinates, low_memory=False)

    chembl = reference[reference["model"].eq("ChEMBL36")].copy()
    if len(chembl) != 100_000:
        raise ValueError(f"expected 100,000 ChEMBL rows, found {len(chembl):,}")
    if len(trajectory) != 67_694 or trajectory["trajectory_id"].duplicated().any():
        raise ValueError("trajectory table must contain 67,694 unique valid trajectory IDs")
    if len(rnd53) != 53 or rnd53["rnd_id"].duplicated().any():
        raise ValueError("RND table must contain 53 unique RND IDs")
    for name, frame in (("ChEMBL", chembl), ("trajectory", trajectory), ("RND53", rnd53)):
        if frame[["umap1", "umap2"]].isna().any().any():
            raise ValueError(f"{name} coordinates contain missing values")
    return chembl, trajectory, rnd53


def add_axes(axis: plt.Axes, title: str) -> None:
    """Apply common labels and bounds without changing the coordinate system."""
    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2")
    axis.set_title(title)


def plot_chembl_rnd53(chembl: pd.DataFrame, rnd53: pd.DataFrame, output_path: Path) -> None:
    """Plot the 53 external references against exactly ChEMBL100k."""
    figure, axis = plt.subplots(figsize=(9.6, 7.6), dpi=220)
    axis.scatter(chembl["umap1"], chembl["umap2"], s=0.75, c=CHEMBL_COLOR, alpha=0.20, linewidths=0, label="ChEMBL36 (n=100,000)")
    axis.scatter(rnd53["umap1"], rnd53["umap2"], s=46, c=RND_COLOR, marker="D", alpha=0.95, linewidths=0.45, edgecolors="white", label="RND-family references (n=53)", zorder=3)
    add_axes(axis, "RND-family references in the ChEMBL100k chemical space")
    axis.legend(markerscale=1.35, loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def plot_total(chembl: pd.DataFrame, trajectory: pd.DataFrame, rnd53: pd.DataFrame, output_path: Path) -> None:
    """Plot RND53, ChEMBL100k, and all valid trajectory molecules together."""
    figure, axis = plt.subplots(figsize=(9.6, 7.6), dpi=220)
    axis.scatter(chembl["umap1"], chembl["umap2"], s=0.70, c=CHEMBL_COLOR, alpha=0.14, linewidths=0, label="ChEMBL36 (n=100,000)")
    axis.scatter(trajectory["umap1"], trajectory["umap2"], s=1.55, c=DIFFSBDD_COLOR, alpha=0.18, linewidths=0, label="DiffSBDD trajectory-valid (n=67,694 / 100,000 attempts)")
    axis.scatter(rnd53["umap1"], rnd53["umap2"], s=46, c=RND_COLOR, marker="D", alpha=0.95, linewidths=0.45, edgecolors="white", label="RND-family references (n=53)", zorder=4)
    add_axes(axis, "RND53, ChEMBL100k, and DiffSBDD 100,000-attempt generation")
    axis.legend(markerscale=1.35, loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def distribution_comparison(reference: pd.DataFrame, chembl: pd.DataFrame, trajectory: pd.DataFrame) -> dict[str, float | int]:
    """Summarize old and current DiffSBDD locations in the shared 2D display space."""
    historical = reference[reference["model"].eq("DiffSBDD")]
    if len(historical) != 1_999:
        raise ValueError(f"expected 1,999 historical DiffSBDD rows, found {len(historical):,}")
    old_xy = historical[["umap1", "umap2"]].to_numpy()
    current_xy = trajectory[["umap1", "umap2"]].to_numpy()
    chembl_xy = chembl[["umap1", "umap2"]].to_numpy()
    old_centroid = old_xy.mean(axis=0)
    current_centroid = current_xy.mean(axis=0)
    old_cov = np.cov(old_xy, rowvar=False)
    current_cov = np.cov(current_xy, rowvar=False)
    hull = ConvexHull(old_xy)
    simplex = Delaunay(old_xy[hull.vertices])
    current_inside = simplex.find_simplex(current_xy) >= 0
    chembl_tree = cKDTree(chembl_xy)
    old_nearest = chembl_tree.query(old_xy, k=1)[0]
    current_nearest = chembl_tree.query(current_xy, k=1)[0]
    return {
        "historical_diffsbdd_n": int(len(historical)),
        "current_diffsbdd_valid_n": int(len(trajectory)),
        "centroid_distance": float(np.linalg.norm(current_centroid - old_centroid)),
        "historical_centroid_umap1": float(old_centroid[0]),
        "historical_centroid_umap2": float(old_centroid[1]),
        "current_centroid_umap1": float(current_centroid[0]),
        "current_centroid_umap2": float(current_centroid[1]),
        "historical_covariance_trace": float(np.trace(old_cov)),
        "current_covariance_trace": float(np.trace(current_cov)),
        "historical_convex_hull_area": float(hull.volume),
        "current_fraction_inside_historical_hull": float(current_inside.mean()),
        "historical_nearest_chembl_median": float(np.median(old_nearest)),
        "current_nearest_chembl_median": float(np.median(current_nearest)),
        "historical_nearest_chembl_p90": float(np.quantile(old_nearest, 0.90)),
        "current_nearest_chembl_p90": float(np.quantile(current_nearest, 0.90)),
    }


def main() -> int:
    """Create the two requested fixed-reference UMAP figures and source table."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_plotting()
    chembl, trajectory, rnd53 = load_inputs(args)
    reference = pd.read_csv(args.reference_coordinates, low_memory=False)

    plot_chembl_rnd53(chembl, rnd53, args.output_dir / "rnd53_vs_chembl100k_umap.png")
    plot_total(chembl, trajectory, rnd53, args.output_dir / "rnd53_chembl100k_diffsbdd_100k_attempts_umap.png")

    pd.concat(
        [
            chembl.assign(plot_group="ChEMBL36 100k"),
            trajectory.assign(plot_group="DiffSBDD trajectory-valid"),
            rnd53.assign(plot_group="RND-family reference"),
        ],
        ignore_index=True,
        sort=False,
    ).to_csv(args.output_dir / "plot_coordinates.csv.gz", index=False, compression="gzip")
    comparison = distribution_comparison(reference, chembl, trajectory)
    (args.output_dir / "historical_diffsbdd_vs_current_trajectory_umap_metrics.json").write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "status": "completed",
        "coordinate_system": "fixed old five-model ECFP4 UMAP reference",
        "umap_interpretation": "visualization only; coordinates were not refit for these figures",
        "groups": {
            "ChEMBL36_100k": len(chembl),
            "DiffSBDD_attempts": 100_000,
            "DiffSBDD_rdkit_valid": len(trajectory),
            "DiffSBDD_rdkit_invalid": 100_000 - len(trajectory),
            "RND_family_references": len(rnd53),
        },
        "historical_diffsbdd_comparison": comparison,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
