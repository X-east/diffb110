#!/usr/bin/env python3
"""Overlay historical ChEMBL filters and current generation batches on one frozen UMAP."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/diffb110_historical_umap_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = ["baseline_original", "x_only", "v_only", "native_joint"]
METHOD_LABELS = {
    "baseline_original": "Original",
    "x_only": "x_only",
    "v_only": "v_only",
    "native_joint": "native_joint",
}
METHOD_COLORS = {
    "baseline_original": "#4B5563",
    "x_only": "#2563EB",
    "v_only": "#D97706",
    "native_joint": "#B91C1C",
}

POOL_ORDER = [
    "normality_nearest_chembl_le_0_01",
    "normality_nearest_chembl_le_0_02",
    "normality_nearest_chembl_le_0_03",
    "normality_chembl_k20_le_0_08",
    "normality_chembl_k20_le_0_10",
    "normality_chembl_k20_le_0_12",
]
POOL_LABELS = {
    "normality_nearest_chembl_le_0_01": "Normality + nearest d <= 0.01",
    "normality_nearest_chembl_le_0_02": "Normality + nearest d <= 0.02",
    "normality_nearest_chembl_le_0_03": "Normality + nearest d <= 0.03",
    "normality_chembl_k20_le_0_08": "Normality + k20 d <= 0.08",
    "normality_chembl_k20_le_0_10": "Normality + k20 d <= 0.10",
    "normality_chembl_k20_le_0_12": "Normality + k20 d <= 0.12",
}
POOL_COLORS = ["#0F766E", "#14B8A6", "#5EEAD4", "#7C3AED", "#A78BFA", "#DDD6FE"]

SEMANTIC_ORDER = ["near", "mid", "far"]
SEMANTIC_LABELS = {"near": "ChEMBL-near (d <= 0.01)", "mid": "ChEMBL-mid (0.01 < d <= 0.03)", "far": "ChEMBL-far (d > 0.03)"}
SEMANTIC_COLORS = {"near": "#0F766E", "mid": "#D97706", "far": "#B91C1C"}
COMPOSITE_ORDER = ["good", "mid", "bad"]
COMPOSITE_LABELS = {"good": "Good exploratory (near + clean)", "mid": "Mid exploratory (mid + clean)", "bad": "Bad exploratory (far or hard-bad)"}
COMPOSITE_COLORS = {"good": "#15803D", "mid": "#D97706", "bad": "#B91C1C"}


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_plotting() -> None:
    """Configure the single Python/matplotlib plotting backend."""
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def read_inputs(repo: Path) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, dict[str, object]]:
    """Read current coordinates, historical facts, and the frozen reference map."""
    current_dir = repo / "analysis/diffsbdd_global_e2po_good_bad_xv_smoke_260825/umap_original_model_projection_260825"
    historical_source = repo / "analysis/offline_iql_100k_reasonableness_schemes_260819/expanded_reasonableness_facts.csv.gz"
    partition_path = repo / "analysis/offline_iql_chembl_distance_distribution_260821/chembl_distance_partition_manifest.csv.gz"
    current = pd.read_csv(current_dir / "per_molecule_portable_umap_metrics.csv.gz")
    historical = pd.read_csv(historical_source, low_memory=False)
    partitions = pd.read_csv(partition_path, low_memory=False)
    reference = np.load(current_dir / "chembl_reference_umap_coordinates.npy")

    required_current = {"batch", "umap1", "umap2", "nearest_umap_reference_distance"}
    required_history = {"molecule_id", "umap1", "umap2", "umap_nearest_chembl_distance"}
    if not required_current.issubset(current.columns):
        raise ValueError(f"current coordinate table lacks {sorted(required_current - set(current.columns))}")
    if not required_history.issubset(historical.columns):
        raise ValueError(f"historical facts table lacks {sorted(required_history - set(historical.columns))}")
    if historical["molecule_id"].duplicated().any():
        raise ValueError("historical molecule_id is not unique")
    if partitions["molecule_id"].duplicated().any():
        raise ValueError("partition molecule_id is not unique")
    if not set(partitions["molecule_id"]).issubset(set(historical["molecule_id"])):
        raise ValueError("partition manifest contains IDs absent from historical facts")
    if not np.isfinite(reference).all() or reference.shape[1] != 2:
        raise ValueError("reference coordinates are not a finite N x 2 array")

    partition_columns = [
        "molecule_id",
        "semantic_partition",
        "equal_mass_partition",
        "structural_hard_bad",
        "structural_hard_bad_reasons",
        "composite_partition",
    ]
    historical = historical.merge(partitions[partition_columns], on="molecule_id", how="left", validate="one_to_one")
    if historical["semantic_partition"].isna().any():
        raise ValueError("some historical facts lack partition labels")

    pool_frames: list[pd.DataFrame] = []
    for pool in POOL_ORDER:
        path = repo / "analysis/offline_iql_100k_chembl_training_pools_260819" / pool / "selected_coordinates_and_facts.csv.gz"
        frame = pd.read_csv(path, usecols=["molecule_id", "umap1", "umap2", "umap_nearest_chembl_distance"])
        frame["pool"] = pool
        pool_frames.append(frame)
        source = historical.set_index("molecule_id").loc[frame["molecule_id"]]
        coordinate_delta = np.abs(source[["umap1", "umap2"]].to_numpy(float) - frame[["umap1", "umap2"]].to_numpy(float))
        if len(frame) == 0 or float(coordinate_delta.max(initial=0.0)) > 1e-8:
            raise ValueError(f"coordinate mismatch in historical pool {pool}")
        distance_delta = np.abs(source["umap_nearest_chembl_distance"].to_numpy(float) - frame["umap_nearest_chembl_distance"].to_numpy(float))
        if float(distance_delta.max(initial=0.0)) > 1e-8:
            raise ValueError(f"distance mismatch in historical pool {pool}")
    pools = pd.concat(pool_frames, ignore_index=True)

    model_path = repo / "analysis/benchmark/benchmark5/rnd53_umap_overlay_20260817/rebuild_compat_20260820_run9/old100k_portable_exact_umap_model.joblib"
    projection_manifest = json.loads((current_dir / "projection_analysis_manifest.json").read_text(encoding="utf-8"))
    expected_hash = projection_manifest.get("model_sha256")
    actual_hash = sha256(model_path)
    if expected_hash and expected_hash != actual_hash:
        raise ValueError("frozen UMAP model hash differs from current projection manifest")
    audit = {
        "current_rows": int(len(current)),
        "current_valid_rows_by_batch": current.groupby("batch").size().astype(int).to_dict(),
        "historical_valid_rows": int(len(historical)),
        "reference_rows": int(len(reference)),
        "pool_rows_with_overlap": int(len(pools)),
        "unique_pool_molecules": int(pools["molecule_id"].nunique()),
        "frozen_model": str(model_path.resolve()),
        "frozen_model_sha256": actual_hash,
        "coordinate_system": "old five-model ChEMBL100k UMAP; no fit or transform in this overlay",
        "coordinate_checks": "all six pool coordinate and nearest-distance fields match the historical facts table within 1e-8",
    }
    return current, historical, reference, {"pools": pools, "audit": audit}


def shared_limits(reference: np.ndarray, historical: pd.DataFrame, current: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    """Set identical limits for every facet from the complete plotted coordinate range."""
    coordinates = np.vstack(
        [
            reference,
            historical[["umap1", "umap2"]].to_numpy(float),
            current[["umap1", "umap2"]].to_numpy(float),
        ]
    )
    low = np.quantile(coordinates, 0.001, axis=0)
    high = np.quantile(coordinates, 0.999, axis=0)
    margin = np.maximum((high - low) * 0.04, 0.1)
    return (float(low[0] - margin[0]), float(high[0] + margin[0])), (float(low[1] - margin[1]), float(high[1] + margin[1]))


def draw_base(ax: plt.Axes, reference: np.ndarray) -> None:
    """Draw only the gray ChEMBL reference background."""
    ax.scatter(reference[:, 0], reference[:, 1], s=1.0, color="#C7CCD1", alpha=0.12, linewidths=0, rasterized=True)


def save_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    """Save PNG, SVG, and PDF variants from matplotlib."""
    fig.savefig(output / f"{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(output / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_pool_facets(pools: pd.DataFrame, reference: np.ndarray, current: pd.DataFrame, limits: tuple[tuple[float, float], tuple[float, float]], output: Path) -> None:
    """Plot each historical training pool in its own shared-coordinate panel."""
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), sharex=True, sharey=True)
    for ax, pool, color in zip(axes.ravel(), POOL_ORDER, POOL_COLORS):
        draw_base(ax, reference)
        frame = pools[pools["pool"].eq(pool)]
        ax.scatter(frame["umap1"], frame["umap2"], s=5, color=color, alpha=0.32, linewidths=0, rasterized=True)
        ax.set_title(f"{POOL_LABELS[pool]}\nn={len(frame):,}")
        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])
        ax.grid(alpha=0.13)
    axes[1, 0].set_xlabel("UMAP1")
    axes[1, 1].set_xlabel("UMAP1")
    axes[1, 2].set_xlabel("UMAP1")
    axes[0, 0].set_ylabel("UMAP2")
    axes[1, 0].set_ylabel("UMAP2")
    fig.suptitle("Historical ChEMBL-filtered training pools on the frozen UMAP", fontsize=17, y=0.995)
    fig.text(0.5, 0.012, "Gray = ChEMBL100k reference; colored points = only the historical pool named in each panel", ha="center", fontsize=11)
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    save_figure(fig, output, "historical_training_pool_overlay_facets")


def plot_partition_facets(historical: pd.DataFrame, reference: np.ndarray, current: pd.DataFrame, limits: tuple[tuple[float, float], tuple[float, float]], output: Path) -> None:
    """Plot semantic and structure-aware historical partitions in six panels."""
    panels = [("semantic_partition", x, SEMANTIC_LABELS[x], SEMANTIC_COLORS[x]) for x in SEMANTIC_ORDER]
    panels += [("composite_partition", x, COMPOSITE_LABELS[x], COMPOSITE_COLORS[x]) for x in COMPOSITE_ORDER]
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), sharex=True, sharey=True)
    for ax, (column, value, label, color) in zip(axes.ravel(), panels):
        draw_base(ax, reference)
        frame = historical[historical[column].eq(value)]
        ax.scatter(frame["umap1"], frame["umap2"], s=3.2, color=color, alpha=0.28, linewidths=0, rasterized=True)
        ax.set_title(f"{label}\nn={len(frame):,}")
        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])
        ax.grid(alpha=0.13)
    axes[1, 0].set_xlabel("UMAP1")
    axes[1, 1].set_xlabel("UMAP1")
    axes[1, 2].set_xlabel("UMAP1")
    axes[0, 0].set_ylabel("UMAP2")
    axes[1, 0].set_ylabel("UMAP2")
    fig.suptitle("Historical ChEMBL-distance partitions on the frozen UMAP", fontsize=17, y=0.995)
    fig.text(0.5, 0.012, "Gray = ChEMBL100k reference; colored points = only the partition named in each panel", ha="center", fontsize=11)
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    save_figure(fig, output, "historical_partition_overlay_facets")


def classify_current(current: pd.DataFrame) -> pd.DataFrame:
    """Assign the historical semantic distance partition to current generated points."""
    current = current.copy()
    distance = current["nearest_umap_reference_distance"]
    current["semantic_partition"] = np.select([distance <= 0.01, distance <= 0.03], ["near", "mid"], default="far")
    return current


def write_tables(historical: pd.DataFrame, current: pd.DataFrame, pools: pd.DataFrame, output: Path) -> None:
    """Write point-level overlay data and group summaries."""
    pool_flags = pools.assign(selected=1).pivot_table(index="molecule_id", columns="pool", values="selected", fill_value=0, aggfunc="max")
    pool_flags.columns = [f"in_{x}" for x in pool_flags.columns]
    history_columns = ["molecule_id", "trajectory_id", "umap1", "umap2", "umap_nearest_chembl_distance", "semantic_partition", "equal_mass_partition", "structural_hard_bad", "structural_hard_bad_reasons", "composite_partition"]
    history_out = historical[history_columns].set_index("molecule_id").join(pool_flags, how="left").fillna({x: 0 for x in pool_flags.columns}).reset_index()
    history_out.to_csv(output / "historical_filter_overlay_coordinates.csv.gz", index=False)
    current_out = current[["batch", "attempt", "sdf_record", "umap1", "umap2", "nearest_umap_reference_distance", "semantic_partition"]].copy()
    current_out.to_csv(output / "current_generation_overlay_coordinates.csv.gz", index=False)

    rows: list[dict[str, object]] = []
    for name, frame, distance_column in [("historical_all_valid", historical, "umap_nearest_chembl_distance")]:
        rows.append({"dataset": name, "group": "all", "n": len(frame), "fraction": 1.0, "median_nearest_distance": float(frame[distance_column].median()), "mean_umap1": float(frame["umap1"].mean()), "mean_umap2": float(frame["umap2"].mean())})
        for category in SEMANTIC_ORDER:
            subset = frame[frame["semantic_partition"].eq(category)]
            rows.append({"dataset": name, "group": category, "n": len(subset), "fraction": len(subset) / len(frame), "median_nearest_distance": float(subset[distance_column].median()), "mean_umap1": float(subset["umap1"].mean()), "mean_umap2": float(subset["umap2"].mean())})
    for batch in METHOD_ORDER:
        frame = current[current["batch"].eq(batch)]
        rows.append({"dataset": "current_generation", "group": METHOD_LABELS[batch] + ":all", "n": len(frame), "fraction": 1.0, "median_nearest_distance": float(frame["nearest_umap_reference_distance"].median()), "mean_umap1": float(frame["umap1"].mean()), "mean_umap2": float(frame["umap2"].mean())})
        for category in SEMANTIC_ORDER:
            subset = frame[frame["semantic_partition"].eq(category)]
            rows.append({"dataset": "current_generation", "group": METHOD_LABELS[batch] + ":" + category, "n": len(subset), "fraction": len(subset) / len(frame), "median_nearest_distance": float(subset["nearest_umap_reference_distance"].median()) if len(subset) else np.nan, "mean_umap1": float(subset["umap1"].mean()) if len(subset) else np.nan, "mean_umap2": float(subset["umap2"].mean()) if len(subset) else np.nan})
    for pool in POOL_ORDER:
        frame = pools[pools["pool"].eq(pool)]
        rows.append({"dataset": "historical_training_pool", "group": pool, "n": len(frame), "fraction": len(frame) / len(historical), "median_nearest_distance": float(frame["umap_nearest_chembl_distance"].median()), "mean_umap1": float(frame["umap1"].mean()), "mean_umap2": float(frame["umap2"].mean())})
    pd.DataFrame(rows).to_csv(output / "overlay_group_summary.csv", index=False)


def main() -> int:
    """Run coordinate audits, render overlays, and write machine-readable evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    configure_plotting()
    current, historical, reference, extras = read_inputs(args.repo)
    current = classify_current(current)
    pools = extras["pools"]
    limits = shared_limits(reference, historical, current)
    plot_pool_facets(pools, reference, current, limits, args.output)
    plot_partition_facets(historical, reference, current, limits, args.output)
    write_tables(historical, current, pools, args.output)
    manifest = {
        "status": "complete",
        "fit_performed": False,
        "transform_performed": False,
        "coordinate_system": "existing old five-model ChEMBL100k frozen UMAP",
        "panel_display_policy": "each panel shows only its named historical category in color; gray points are the ChEMBL100k reference; current generation batches are intentionally omitted from these category panels",
        "semantic_partition": {"near": "d <= 0.01", "mid": "0.01 < d <= 0.03", "far": "d > 0.03"},
        "composite_partition": "good = near and structurally clean; mid = mid and structurally clean; bad = far or structural_hard_bad",
        "audit": extras["audit"],
        "outputs": [
            "historical_training_pool_overlay_facets.png",
            "historical_training_pool_overlay_facets.svg",
            "historical_partition_overlay_facets.png",
            "historical_partition_overlay_facets.svg",
            "historical_filter_overlay_coordinates.csv.gz",
            "current_generation_overlay_coordinates.csv.gz",
            "overlay_group_summary.csv",
        ],
    }
    (args.output / "overlay_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
