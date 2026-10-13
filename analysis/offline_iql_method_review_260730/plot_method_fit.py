"""Plot the qualitative method-fit matrix for DiffSBDD offline RL."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "method_fit_scores.csv"
OUTPUT_PATH = ROOT / "method_fit_matrix.png"


def main() -> None:
    """Render the method-fit scores as a presentation-readable heatmap."""
    with CSV_PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    methods = [row["method"] for row in rows]
    fields = [
        "mdp_granularity_match",
        "uses_saved_denoising_transitions",
        "direct_generator_update",
        "geometric_action_compatibility",
        "engineering_feasibility",
    ]
    labels = [
        "MDP granularity\nmatch",
        "Uses saved denoising\ntransitions",
        "Direct generator\nupdate",
        "Geometric action\ncompatibility",
        "Engineering\nfeasibility",
    ]
    values = np.asarray(
        [[int(row[field]) for field in fields] for row in rows],
        dtype=np.int64,
    )

    fig, axis = plt.subplots(figsize=(13.5, 8.2))
    image = axis.imshow(values, cmap="YlGn", vmin=0, vmax=3, aspect="auto")
    axis.set_xticks(np.arange(len(labels)), labels=labels, fontsize=13)
    axis.set_yticks(np.arange(len(methods)), labels=methods, fontsize=13)
    axis.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    axis.set_title(
        "Fit to DiffSBDD Denoising-Trajectory Offline RL",
        fontsize=20,
        fontweight="bold",
        pad=24,
    )

    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = int(values[row_index, column_index])
            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                fontsize=15,
                fontweight="bold",
                color="white" if value >= 2 else "#303030",
            )

    axis.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(methods), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=2)
    axis.tick_params(which="minor", bottom=False, left=False)
    colorbar = fig.colorbar(image, ax=axis, fraction=0.035, pad=0.03)
    colorbar.set_ticks([0, 1, 2, 3])
    colorbar.set_label("0 = poor fit, 3 = strong fit", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
