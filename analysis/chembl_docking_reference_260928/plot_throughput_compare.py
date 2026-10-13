#!/usr/bin/env python
"""Bar chart: UniDock docking throughput, old vs new submission strategy.

Data are measured on this machine (single GPU per measurement):
- old  : dock_sdf_unidock.py, fixed batch_size=128, one process per batch
- sm244: dock_shard_index.py, --ligand_index, 245 ligands per call
- pipe1k: dock_worker.py pipeline, 1000 ligands per shard (2 runs)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

HERE = Path(__file__).resolve().parent

LABELS = ["old\n(batch=128)", "index call\n(245/shard)", "pipeline\n(1000/shard) A",
          "pipeline\n(1000/shard) B"]
VALUES = [4.75, 6.00, 8.72, 10.53]
COLORS = ["#95A5A6", "#F39C12", "#2ECC71", "#27AE60"]


def main() -> None:
    """Render and save the throughput comparison."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.35)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.edgecolor"] = "#d4d4d4"

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(np.arange(len(VALUES)), VALUES, color=COLORS,
                  edgecolor="white", linewidth=1.0, width=0.66)
    for rect, value in zip(bars, VALUES):
        ax.annotate(f"{value:.2f}", (rect.get_x() + rect.get_width() / 2, value),
                    ha="center", va="bottom", fontsize=13, fontweight="bold",
                    xytext=(0, 3), textcoords="offset points")
    ax.set_xticks(np.arange(len(VALUES)))
    ax.set_xticklabels(LABELS, fontsize=12)
    ax.set_ylabel("Throughput (molecules / s / GPU)", fontsize=14, fontweight="bold")
    ax.set_title("UniDock throughput: submission strategy comparison (RE-CmeB, search_mode=fast)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(VALUES) * 1.25)
    sns.despine(ax=ax, top=True, right=True)

    fig.tight_layout()
    fig.savefig(HERE / "throughput_compare.png", dpi=300, bbox_inches="tight")
    fig.savefig(HERE / "throughput_compare.svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote throughput_compare.png/.svg")


if __name__ == "__main__":
    main()
