#!/usr/bin/env python3
"""Plot a joint-fit full-ChEMBL and five-model UMAP experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS = ["DiffGui", "DiffSBDD", "PMDM", "GCDM-SBDD", "TargetDiff"]
COLORS = {"DiffGui": "#3B82C4", "DiffSBDD": "#D95F59", "PMDM": "#D9A441", "GCDM-SBDD": "#8E78B7", "TargetDiff": "#3D9B72"}


def draw_background(ax: plt.Axes, coordinates: np.ndarray) -> None:
    """Draw full ChEMBL as a log-scaled density map."""
    counts, x_edges, y_edges = np.histogram2d(coordinates[:, 0], coordinates[:, 1], bins=500)
    density = counts.T
    positive = density[density > 0]
    ax.imshow(density, origin="lower", extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]), cmap="Greys", norm=colors.LogNorm(vmin=max(1, np.quantile(positive, .05)), vmax=density.max()), alpha=.78, aspect="auto", rasterized=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-npy", type=Path, required=True)
    parser.add_argument("--generated-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    xy = np.load(args.joint_npy, mmap_mode="r")
    generated = pd.read_csv(args.generated_csv)
    generated = generated[generated.sanitize_success.eq(1) & generated.canonical_smiles.notna()].copy().reset_index(drop=True)
    if len(generated) != 10910 or len(xy) - len(generated) != 2734034:
        raise ValueError("joint coordinate ordering/count mismatch")
    chembl_xy, generated_xy = xy[:2734034], xy[2734034:]
    generated["umap1"], generated["umap2"] = generated_xy[:, 0], generated_xy[:, 1]
    generated.to_csv(args.output_dir / "joint_fit_five_model_coordinates.csv.gz", index=False, compression="gzip")
    fig, ax = plt.subplots(figsize=(12, 9))
    draw_background(ax, chembl_xy)
    for model in MODELS:
        group = generated[generated.model.eq(model)]
        ax.scatter(group.umap1, group.umap2, s=10, alpha=.62, linewidths=0, c=COLORS[model], label=f"{model} (n={len(group):,})", rasterized=True)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title("Joint UMAP fit: full ChEMBL36 + five-model legal molecules")
    ax.legend(markerscale=2, fontsize=9)
    fig.text(.5, .01, "All five models and ChEMBL were jointly fitted. Coordinates are descriptive, not quality labels.", ha="center")
    fig.tight_layout(rect=(0, .03, 1, 1))
    fig.savefig(args.output_dir / "joint_full_chembl_five_model_umap_overview.png", dpi=350)
    fig.savefig(args.output_dir / "joint_full_chembl_five_model_umap_overview.svg")
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), sharex=True, sharey=True)
    for ax, model in zip(axes.flat, [*MODELS, "ChEMBL36"]):
        draw_background(ax, chembl_xy)
        if model != "ChEMBL36":
            group = generated[generated.model.eq(model)]
            ax.scatter(group.umap1, group.umap2, s=9, alpha=.62, linewidths=0, c=COLORS[model], rasterized=True)
            ax.set_title(f"{model} | n={len(group):,}")
        else:
            ax.set_title("ChEMBL36 density")
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    fig.suptitle("Joint-fit model-specific views in full ChEMBL chemical space", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(args.output_dir / "joint_full_chembl_five_model_umap_facets.png", dpi=350)
    fig.savefig(args.output_dir / "joint_full_chembl_five_model_umap_facets.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
