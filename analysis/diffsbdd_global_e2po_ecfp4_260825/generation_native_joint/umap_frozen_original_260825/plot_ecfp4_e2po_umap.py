#!/usr/bin/env python3
"""Plot the current ECFP4-E2PO and Original batches on the frozen UMAP."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    """Render a correctly labeled frozen-UMAP overlay and anomaly marker."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--anomaly-sdf-record",
        type=int,
        default=956,
        help="Zero-based SDF record for docking attempt 957 (+2006.438).",
    )
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    reference = np.load(args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    fig, axis = plt.subplots(figsize=(11, 9), constrained_layout=True)
    axis.scatter(
        reference[:, 0],
        reference[:, 1],
        s=3,
        c="#bdbdbd",
        alpha=0.12,
        linewidths=0,
        rasterized=True,
        label="ChEMBL reference",
    )
    colors = {
        "baseline_original": ("Original", "#1f77b4"),
        "ecfp4_e2po": ("ECFP4-E2PO", "#ff7f0e"),
    }
    for batch, (label, color) in colors.items():
        subset = data.loc[data["batch"].eq(batch)]
        axis.scatter(
            subset["umap1"],
            subset["umap2"],
            s=28,
            facecolors="none",
            edgecolors=color,
            linewidths=0.9,
            alpha=0.72,
            label=f"{label} (n={len(subset)})",
        )

    anomaly = data.loc[
        data["batch"].eq("ecfp4_e2po")
        & data["sdf_record"].eq(args.anomaly_sdf_record)
    ]
    if len(anomaly) != 1:
        raise ValueError(
            f"Expected one anomaly row for sdf_record={args.anomaly_sdf_record}, got {len(anomaly)}"
        )
    point = anomaly.iloc[0]
    axis.scatter(
        [point["umap1"]],
        [point["umap2"]],
        s=150,
        marker="x",
        c="#d62728",
        linewidths=2.2,
        label="attempt 957, +2006.438",
        zorder=5,
    )
    axis.annotate(
        "+2006.438\n(attempt 957)",
        (point["umap1"], point["umap2"]),
        xytext=(10, 12),
        textcoords="offset points",
        color="#b22222",
        fontsize=11,
        arrowprops={"arrowstyle": "-", "color": "#b22222", "lw": 0.8},
    )
    axis.set_xlabel("UMAP1")
    axis.set_ylabel("UMAP2")
    axis.set_title("Original and ECFP4-E2PO on frozen 100k ChEMBL UMAP")
    axis.legend(frameon=True, loc="upper right")
    axis.grid(alpha=0.15)
    fig.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
