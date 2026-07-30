#!/usr/bin/env python3
"""Regenerate the facet UMAP figure from existing coordinates (skip UMAP fit)."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/diffb110_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
OUTPUT_DIR = Path(__file__).resolve().parent
COORDINATES_PATH = OUTPUT_DIR / "joint_umap_coordinates.csv.gz"

MODEL_ORDER = ["DiffGui", "DiffSBDD", "PMDM", "GCDM-SBDD", "TargetDiff"]
MODEL_COLORS = {
    "DiffGui": "#3B82C4",
    "DiffSBDD": "#D95F59",
    "PMDM": "#D9A441",
    "GCDM-SBDD": "#8E78B7",
    "TargetDiff": "#3D9B72",
}
CHEMBL_COLOR = "#A8ADB3"


def _configure_plotting() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 11,
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


def _save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(
        stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight",
        facecolor="white", pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def _plot_facets(coordinates: pd.DataFrame, output_dir: Path) -> None:
    chembl = coordinates[coordinates["model"] == "ChEMBL36"]
    generated = coordinates[coordinates["dataset"] == "Generated"]
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 8.2), sharex=True, sharey=True)
    panels = [*MODEL_ORDER, "ChEMBL36"]
    for axis, panel in zip(axes.ravel(), panels):
        if panel == "ChEMBL36":
            axis.scatter(
                chembl["umap1"], chembl["umap2"], s=0.6, color=CHEMBL_COLOR,
                alpha=0.14, linewidths=0, rasterized=True,
            )
            axis.set_title(f"ChEMBL36  |  n={len(chembl):,}", fontsize=15)
        else:
            group = generated[generated["model"] == panel]
            axis.scatter(
                chembl["umap1"], chembl["umap2"], s=0.5, color=CHEMBL_COLOR,
                alpha=0.09, linewidths=0, rasterized=True,
            )
            axis.scatter(
                group["umap1"], group["umap2"], s=7.5, color=MODEL_COLORS[panel],
                alpha=0.55, linewidths=0, rasterized=True,
            )
            axis.set_title(f"{panel}  |  n={len(group):,}", fontsize=15)
        axis.set_xlabel("UMAP 1", fontsize=13)
        axis.set_ylabel("UMAP 2", fontsize=13)
        axis.tick_params(labelsize=11)
    fig.suptitle(
        "Model-specific coverage and ChEMBL36 reference in one shared chemical-space map",
        fontsize=18,
    )
    fig.text(
        0.5, 0.008,
        "Coordinates are descriptive neighborhood projections; distance is not efficacy or molecular quality.",
        ha="center", fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.965))
    _save_figure(fig, output_dir / "five_model_chembl100k_umap_facets")


def main() -> None:
    _configure_plotting()
    coordinates = pd.read_csv(COORDINATES_PATH, low_memory=False)
    print(f"loaded coordinates: n={len(coordinates):,}")
    _plot_facets(coordinates, OUTPUT_DIR)
    print(f"saved facets to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
