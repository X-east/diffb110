#!/usr/bin/env python3
"""Summarize frozen full-ChEMBL UMAP projections for five benchmark models."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


MODELS = ["DiffGui", "DiffSBDD", "PMDM", "GCDM-SBDD", "TargetDiff"]
COLORS = {"DiffGui": "#3B82C4", "DiffSBDD": "#D95F59", "PMDM": "#D9A441", "GCDM-SBDD": "#8E78B7", "TargetDiff": "#3D9B72"}


def density_grid(points: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts, x_edges, y_edges = np.histogram2d(points[:, 0], points[:, 1], bins=bins)
    return counts.T, x_edges, y_edges


def hdr_membership(points: np.ndarray, density: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray, mass: float) -> np.ndarray:
    ordered = np.sort(density.ravel())[::-1]
    threshold = ordered[min(np.searchsorted(np.cumsum(ordered) / ordered.sum(), mass), len(ordered) - 1)]
    x_bin = np.searchsorted(x_edges, points[:, 0], side="right") - 1
    y_bin = np.searchsorted(y_edges, points[:, 1], side="right") - 1
    valid = (x_bin >= 0) & (x_bin < density.shape[1]) & (y_bin >= 0) & (y_bin < density.shape[0])
    result = np.zeros(len(points), dtype=bool)
    result[valid] = density[y_bin[valid], x_bin[valid]] >= threshold
    return result


def draw_density(ax: plt.Axes, density: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray) -> None:
    nonzero = density[density > 0]
    ax.imshow(density, origin="lower", extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]), cmap="Greys", norm=colors.LogNorm(vmin=max(1, np.quantile(nonzero, 0.05)), vmax=density.max()), alpha=0.78, aspect="auto", rasterized=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinates", type=Path, required=True)
    parser.add_argument("--chembl-xy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=28)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated = pd.read_csv(args.coordinates)
    chembl = np.load(args.chembl_xy, mmap_mode="r")
    density, x_edges, y_edges = density_grid(chembl, 500)
    generated_xy = generated[["umap1", "umap2"]].to_numpy(np.float32)
    for mass in [0.5, 0.9, 0.95, 0.99]:
        generated[f"inside_chembl_hdr_{int(mass * 100)}"] = hdr_membership(generated_xy, density, x_edges, y_edges, mass)
    neighbors = NearestNeighbors(n_neighbors=1, algorithm="kd_tree", n_jobs=args.workers).fit(chembl)
    generated["nearest_chembl_umap_distance"] = neighbors.kneighbors(generated_xy, return_distance=True)[0][:, 0]
    generated.to_csv(args.output_dir / "five_model_projected_metrics.csv.gz", index=False, compression="gzip")
    by_repeat = generated.groupby(["model", "repeat"]).agg(n=("row_id", "size"), inside_hdr95=("inside_chembl_hdr_95", "mean"), median_umap_distance=("nearest_chembl_umap_distance", "median"), centroid_umap1=("umap1", "mean"), centroid_umap2=("umap2", "mean")).reset_index()
    by_repeat.to_csv(args.output_dir / "five_model_umap_by_repeat.csv", index=False)
    summary = by_repeat.groupby("model").agg(n_mean=("n", "mean"), n_sd=("n", "std"), hdr95_mean=("inside_hdr95", "mean"), hdr95_sd=("inside_hdr95", "std"), median_distance_mean=("median_umap_distance", "mean"), median_distance_sd=("median_umap_distance", "std")).reindex(MODELS).reset_index()
    summary.to_csv(args.output_dir / "five_model_umap_summary.csv", index=False)
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), sharex=True, sharey=True)
    for ax, model in zip(axes.flat, [*MODELS, "ChEMBL36"]):
        draw_density(ax, density, x_edges, y_edges)
        if model != "ChEMBL36":
            group = generated[generated.model.eq(model)]
            ax.scatter(group.umap1, group.umap2, s=9, color=COLORS[model], alpha=0.62, linewidths=0, rasterized=True)
            ax.set_title(f"{model} | n={len(group):,}")
        else:
            ax.set_title(f"Frozen ChEMBL36 density | n={len(chembl):,}")
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    fig.suptitle("Five models in one frozen full-ChEMBL36 UMAP coordinate system", fontsize=16)
    fig.text(0.5, 0.01, "UMAP is descriptive chemical-space context, not an activity, docking, or IQL-label criterion.", ha="center")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(args.output_dir / "five_model_full_chembl_umap_facets.png", dpi=350); fig.savefig(args.output_dir / "five_model_full_chembl_umap_facets.svg"); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8))
    x = np.arange(len(MODELS)); subset = summary.set_index("model").reindex(MODELS)
    axes[0].bar(x, subset.n_mean, yerr=subset.n_sd, color=[COLORS[m] for m in MODELS], capsize=4)
    axes[0].set_ylabel("Legal molecules per 3,000 attempts"); axes[0].set_title("Embedding yield")
    axes[1].bar(x, subset.hdr95_mean * 100, yerr=subset.hdr95_sd * 100, color=[COLORS[m] for m in MODELS], capsize=4)
    axes[1].set_ylabel("Inside ChEMBL 95% HDR (%)"); axes[1].set_title("High-density ChEMBL overlap")
    axes[2].bar(x, subset.median_distance_mean, yerr=subset.median_distance_sd, color=[COLORS[m] for m in MODELS], capsize=4)
    axes[2].set_ylabel("Nearest ChEMBL 2D distance"); axes[2].set_title("Local UMAP neighborhood")
    for ax in axes:
        ax.set_xticks(x, MODELS, rotation=25, ha="right")
    fig.text(0.5, 0.01, "Bars show mean across three repeats; error bars are sample SD. 2D distances are descriptive only.", ha="center")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(args.output_dir / "five_model_full_chembl_umap_summary.png", dpi=350); fig.savefig(args.output_dir / "five_model_full_chembl_umap_summary.svg"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 7))
    draw_density(ax, density, x_edges, y_edges)
    for model in MODELS:
        group = by_repeat[by_repeat.model.eq(model)]
        ax.scatter(group.centroid_umap1, group.centroid_umap2, s=120, color=COLORS[model], label=model)
        for _, row in group.iterrows():
            ax.plot([row.centroid_umap1, group.centroid_umap1.mean()], [row.centroid_umap2, group.centroid_umap2.mean()], color=COLORS[model], alpha=.45, linewidth=1)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2"); ax.set_title("Repeat-level centroid stability in frozen ChEMBL map"); ax.legend()
    fig.tight_layout(); fig.savefig(args.output_dir / "five_model_full_chembl_umap_repeat_stability.png", dpi=350); fig.savefig(args.output_dir / "five_model_full_chembl_umap_repeat_stability.svg"); plt.close(fig)


if __name__ == "__main__":
    main()
