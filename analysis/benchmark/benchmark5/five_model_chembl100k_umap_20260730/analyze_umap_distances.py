#!/usr/bin/env python3
"""Quantify distances in the fixed five-model/ChEMBL100k UMAP coordinates."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/diffb110_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


LOGGER = logging.getLogger("five_model_umap_distance_analysis")
MODEL_ORDER = ["DiffGui", "DiffSBDD", "PMDM", "GCDM-SBDD", "TargetDiff"]
GROUP_ORDER = ["ChEMBL36", *MODEL_ORDER]
MODEL_COLORS = {
    "DiffGui": "#3B82C4",
    "DiffSBDD": "#D95F59",
    "PMDM": "#D9A441",
    "GCDM-SBDD": "#8E78B7",
    "TargetDiff": "#3D9B72",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    default_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coordinates",
        type=Path,
        default=default_dir / "joint_umap_coordinates.csv.gz",
    )
    parser.add_argument("--output-dir", type=Path, default=default_dir)
    return parser.parse_args()


def _configure_logging(output_dir: Path) -> None:
    """Configure terminal and durable file logging."""
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(
        output_dir / "distance_analysis.log", mode="w", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root.addHandler(stream)
    root.addHandler(file_handler)


def _configure_plotting() -> None:
    """Apply slide-readable Python plotting defaults."""
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
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


def _load_coordinates(path: Path) -> pd.DataFrame:
    """Load and validate the fixed joint-UMAP coordinate table."""
    frame = pd.read_csv(path, low_memory=False)
    required = {
        "dataset",
        "model",
        "repeat",
        "molecule_id",
        "canonical_smiles",
        "umap1",
        "umap2",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"coordinate table is missing columns: {sorted(missing)}")
    if len(frame) != 110_910:
        raise ValueError(f"expected 110,910 coordinate rows, found {len(frame):,}")
    if not np.isfinite(frame[["umap1", "umap2"]].to_numpy()).all():
        raise ValueError("coordinate table contains non-finite UMAP coordinates")
    observed = frame.groupby("model").size().to_dict()
    expected = {
        "ChEMBL36": 100_000,
        "DiffGui": 2_641,
        "DiffSBDD": 1_999,
        "PMDM": 1_726,
        "GCDM-SBDD": 1_570,
        "TargetDiff": 2_974,
    }
    if observed != expected:
        raise ValueError(f"unexpected group counts: {observed}")
    return frame


def _distance_summaries(
    coordinates: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Calculate pooled and repeat-level distances in the fixed 2D map."""
    chembl = coordinates[coordinates["model"] == "ChEMBL36"].copy()
    generated = coordinates[coordinates["dataset"] == "Generated"].copy()
    chembl_xy = chembl[["umap1", "umap2"]].to_numpy()
    tree = cKDTree(chembl_xy)

    # The second query neighbor excludes each ChEMBL point itself. Other points
    # may still have identical 2D coordinates, which is a property of the map.
    chembl_self_distances = tree.query(chembl_xy, k=2, workers=-1)[0][:, 1]
    chembl_reference = {
        "median": float(np.median(chembl_self_distances)),
        "p90": float(np.quantile(chembl_self_distances, 0.90)),
        "p95": float(np.quantile(chembl_self_distances, 0.95)),
    }
    generated_distances = tree.query(
        generated[["umap1", "umap2"]].to_numpy(), k=1, workers=-1
    )[0]
    generated["nearest_chembl_umap_distance"] = generated_distances

    chembl_centroid = chembl_xy.mean(axis=0)
    pooled_rows: list[dict[str, Any]] = []
    repeat_rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        group = generated[generated["model"] == model]
        xy = group[["umap1", "umap2"]].to_numpy()
        centroid = xy.mean(axis=0)
        distances = group["nearest_chembl_umap_distance"].to_numpy()
        pooled_rows.append(
            {
                "model": model,
                "n": int(len(group)),
                "centroid_umap1": float(centroid[0]),
                "centroid_umap2": float(centroid[1]),
                "centroid_distance_to_chembl": float(np.linalg.norm(centroid - chembl_centroid)),
                "within_group_rms_radius": float(
                    np.sqrt(np.mean(np.sum((xy - centroid) ** 2, axis=1)))
                ),
                "nearest_chembl_distance_mean": float(np.mean(distances)),
                "nearest_chembl_distance_median": float(np.median(distances)),
                "nearest_chembl_distance_p90": float(np.quantile(distances, 0.90)),
                "nearest_chembl_distance_p95": float(np.quantile(distances, 0.95)),
                "fraction_within_chembl_self_nn_p95": float(
                    np.mean(distances <= chembl_reference["p95"])
                ),
                "chembl_self_nn_median_reference": chembl_reference["median"],
                "chembl_self_nn_p90_reference": chembl_reference["p90"],
                "chembl_self_nn_p95_reference": chembl_reference["p95"],
            }
        )
        for repeat in sorted(group["repeat"].dropna().unique()):
            repeat_group = group[group["repeat"] == repeat]
            repeat_xy = repeat_group[["umap1", "umap2"]].to_numpy()
            repeat_centroid = repeat_xy.mean(axis=0)
            repeat_distances = repeat_group["nearest_chembl_umap_distance"].to_numpy()
            repeat_rows.append(
                {
                    "model": model,
                    "repeat": repeat,
                    "n": int(len(repeat_group)),
                    "centroid_umap1": float(repeat_centroid[0]),
                    "centroid_umap2": float(repeat_centroid[1]),
                    "centroid_distance_to_chembl": float(
                        np.linalg.norm(repeat_centroid - chembl_centroid)
                    ),
                    "within_group_rms_radius": float(
                        np.sqrt(
                            np.mean(
                                np.sum(
                                    (repeat_xy - repeat_centroid) ** 2,
                                    axis=1,
                                )
                            )
                        )
                    ),
                    "nearest_chembl_distance_mean": float(np.mean(repeat_distances)),
                    "nearest_chembl_distance_median": float(
                        np.median(repeat_distances)
                    ),
                    "nearest_chembl_distance_p90": float(
                        np.quantile(repeat_distances, 0.90)
                    ),
                    "nearest_chembl_distance_p95": float(
                        np.quantile(repeat_distances, 0.95)
                    ),
                    "fraction_within_chembl_self_nn_p95": float(
                        np.mean(repeat_distances <= chembl_reference["p95"])
                    ),
                    "chembl_self_nn_median_reference": chembl_reference["median"],
                    "chembl_self_nn_p90_reference": chembl_reference["p90"],
                    "chembl_self_nn_p95_reference": chembl_reference["p95"],
                }
            )
    pooled_summary = pd.DataFrame(pooled_rows)
    repeat_summary = pd.DataFrame(repeat_rows)
    repeat_metrics = [
        "n",
        "centroid_umap1",
        "centroid_umap2",
        "centroid_distance_to_chembl",
        "within_group_rms_radius",
        "nearest_chembl_distance_mean",
        "nearest_chembl_distance_median",
        "nearest_chembl_distance_p90",
        "nearest_chembl_distance_p95",
        "fraction_within_chembl_self_nn_p95",
    ]
    summary = (
        repeat_summary.groupby("model", sort=False)[repeat_metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        column
        if isinstance(column, str)
        else column[0]
        if not column[1]
        else f"{column[0]}_{column[1]}"
        for column in summary.columns
    ]
    summary.insert(
        1,
        "repeats",
        repeat_summary.groupby("model", sort=False)["repeat"].nunique().to_numpy(),
    )
    summary.insert(
        2,
        "n_total",
        repeat_summary.groupby("model", sort=False)["n"].sum().to_numpy(),
    )
    for key, value in chembl_reference.items():
        summary[f"chembl_self_nn_{key}_reference"] = value

    centroids = {
        group: coordinates[coordinates["model"] == group][["umap1", "umap2"]]
        .mean()
        .to_numpy()
        for group in GROUP_ORDER
    }
    matrix = pd.DataFrame(index=GROUP_ORDER, columns=GROUP_ORDER, dtype=float)
    long_rows: list[dict[str, Any]] = []
    for source in GROUP_ORDER:
        for target in GROUP_ORDER:
            distance = float(np.linalg.norm(centroids[source] - centroids[target]))
            matrix.loc[source, target] = distance
            if GROUP_ORDER.index(source) < GROUP_ORDER.index(target):
                long_rows.append(
                    {"group_a": source, "group_b": target, "centroid_distance": distance}
                )
    matrix.index.name = "group"
    long_table = pd.DataFrame(long_rows)
    return (
        summary,
        repeat_summary,
        pooled_summary,
        generated,
        matrix,
        long_table,
    )


def _save_figure(fig: plt.Figure, stem: Path) -> None:
    """Export one Python-rendered distance figure in four formats."""
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def _plot_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    """Plot repeat means and sample standard deviations for distance metrics."""
    frame = summary.set_index("model").loc[MODEL_ORDER].reset_index()
    x = np.arange(len(frame))
    colors = [MODEL_COLORS[model] for model in frame["model"]]
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.2))

    axes[0].bar(
        x,
        frame["centroid_distance_to_chembl_mean"],
        yerr=frame["centroid_distance_to_chembl_std"],
        color=colors,
        width=0.72,
        capsize=4,
        error_kw={"elinewidth": 1.2, "capthick": 1.2},
    )
    for index, value in enumerate(frame["centroid_distance_to_chembl_mean"]):
        axes[0].text(
            index,
            value + frame["centroid_distance_to_chembl_std"].iloc[index] + 0.08,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    axes[0].set_title("a  Centroid distance to ChEMBL", loc="left")
    axes[0].set_ylabel("Centroid distance")
    axes[0].set_xticks(x, frame["model"], rotation=25, ha="right")
    axes[0].set_ylim(
        0,
        (
            frame["centroid_distance_to_chembl_mean"]
            + frame["centroid_distance_to_chembl_std"]
        ).max()
        * 1.20,
    )

    axes[1].errorbar(
        x - 0.11,
        frame["nearest_chembl_distance_median_mean"],
        yerr=frame["nearest_chembl_distance_median_std"],
        color="#2563A6",
        marker="o",
        markersize=7,
        linestyle="none",
        capsize=4,
        label="Median",
    )
    axes[1].errorbar(
        x + 0.11,
        frame["nearest_chembl_distance_p90_mean"],
        yerr=frame["nearest_chembl_distance_p90_std"],
        color="#B43C39",
        marker="D",
        markersize=6,
        linestyle="none",
        capsize=4,
        label="90th percentile",
    )
    axes[1].set_title("b  Nearest-ChEMBL distance", loc="left")
    axes[1].set_ylabel("Nearest distance")
    axes[1].set_xticks(x, frame["model"], rotation=25, ha="right")
    axes[1].legend(loc="upper right")

    coverage_mean = frame["fraction_within_chembl_self_nn_p95_mean"] * 100.0
    coverage_std = frame["fraction_within_chembl_self_nn_p95_std"] * 100.0
    axes[2].bar(
        x,
        coverage_mean,
        yerr=coverage_std,
        color=colors,
        width=0.72,
        capsize=4,
        error_kw={"elinewidth": 1.2, "capthick": 1.2},
    )
    for index, value in enumerate(coverage_mean):
        axes[2].text(
            index,
            value + coverage_std.iloc[index] + 1.2,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    axes[2].set_title("c  Local-neighborhood coverage", loc="left")
    axes[2].set_ylabel("Coverage (%)")
    axes[2].set_xticks(x, frame["model"], rotation=25, ha="right")
    axes[2].set_ylim(0, min(100, (coverage_mean + coverage_std).max() * 1.22))

    fig.text(
        0.5,
        0.008,
        "Mean ± sample SD across three independent generation batches (n=3); "
        "distances describe this fixed 2D UMAP projection only.",
        ha="center",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save_figure(fig, output_dir / "five_model_chembl100k_umap_distance_summary")


def _update_manifest(output_dir: Path, summary: pd.DataFrame) -> None:
    """Append distance definitions and output lineage to the existing manifest."""
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["distance_analysis"] = {
        "coordinate_space": "the fixed 2D joint UMAP coordinates",
        "centroid_distance": "Euclidean distance between group centroids",
        "nearest_chembl_distance": "Euclidean distance from each generated point to its nearest ChEMBL36 point",
        "chembl_self_reference": "second-neighbor distance among ChEMBL36 points, excluding each query point itself",
        "repeat_summary": (
            "Each metric is calculated separately for rep1, rep2, and rep3; "
            "the main summary and figure report the repeat mean and sample SD (ddof=1, n=3)"
        ),
        "chembl_self_nn_p95": float(summary["chembl_self_nn_p95_reference"].iloc[0]),
        "outputs": [
            "distance_to_chembl_summary.csv",
            "distance_to_chembl_by_repeat.csv",
            "distance_to_chembl_pooled_summary.csv",
            "nearest_chembl_distance_per_generated_molecule.csv.gz",
            "centroid_distance_matrix.csv",
            "centroid_distance_pairs.csv",
            "five_model_chembl100k_umap_distance_summary.png",
            "five_model_chembl100k_umap_distance_summary.svg",
            "five_model_chembl100k_umap_distance_summary.pdf",
            "five_model_chembl100k_umap_distance_summary.tiff",
        ],
        "interpretation_boundary": (
            "All reported distances depend on this fixed nonlinear 2D projection and must not be "
            "interpreted as molecular efficacy, target activity, or absolute chemical similarity."
        ),
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)


def main() -> None:
    """Run distance analysis and export all supporting data."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _configure_logging(args.output_dir)
    _configure_plotting()
    LOGGER.info("loading coordinates: %s", args.coordinates.resolve())
    coordinates = _load_coordinates(args.coordinates)
    (
        summary,
        repeat_summary,
        pooled_summary,
        molecule_distances,
        matrix,
        pairs,
    ) = _distance_summaries(coordinates)

    summary.to_csv(args.output_dir / "distance_to_chembl_summary.csv", index=False)
    repeat_summary.to_csv(
        args.output_dir / "distance_to_chembl_by_repeat.csv",
        index=False,
    )
    pooled_summary.to_csv(
        args.output_dir / "distance_to_chembl_pooled_summary.csv",
        index=False,
    )
    molecule_distances.to_csv(
        args.output_dir / "nearest_chembl_distance_per_generated_molecule.csv.gz",
        index=False,
        compression="gzip",
    )
    matrix.to_csv(args.output_dir / "centroid_distance_matrix.csv")
    pairs.to_csv(args.output_dir / "centroid_distance_pairs.csv", index=False)
    _plot_summary(summary, args.output_dir)
    _update_manifest(args.output_dir, summary)
    LOGGER.info("repeat-level distance summary:\n%s", repeat_summary.to_string(index=False))
    LOGGER.info("repeat mean and sample SD:\n%s", summary.to_string(index=False))
    LOGGER.info("completed successfully")


if __name__ == "__main__":
    main()
