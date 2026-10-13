"""Plot the method-coverage matrix for application-oriented molecule generation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap


ROOT = Path(__file__).resolve().parent
INPUT_CSV = ROOT / "literature_matrix.csv"
OUTPUT_PNG = ROOT / "literature_landscape.png"
OUTPUT_SVG = ROOT / "literature_landscape.svg"

FEATURES = {
    "hard_filter_evaluation": "Hard-filter\nevaluation",
    "soft_desirability": "Soft\ndesirability",
    "pareto_or_rank_reward": "Pareto /\nrank reward",
    "synthesis_signal": "Synthesis\nsignal",
    "docking_signal": "Docking\nsignal",
    "specificity_signal": "Specificity\nsignal",
    "raw_3d_pose_qc": "Raw 3D\npose QC",
    "rl_or_guidance": "RL /\nguidance",
    "prospective_validation": "Prospective\nvalidation",
}


def main() -> None:
    """Create PNG and SVG heatmaps from the literature evidence matrix."""
    frame = pd.read_csv(INPUT_CSV)
    matrix = frame[list(FEATURES)].to_numpy()

    fig_height = max(6.5, 0.42 * len(frame))
    fig, ax = plt.subplots(figsize=(11.5, fig_height))
    ax.imshow(matrix, aspect="auto", cmap=ListedColormap(["#eceff1", "#1b7f5a"]), vmin=0, vmax=1)

    ax.set_xticks(range(len(FEATURES)), labels=FEATURES.values())
    ax.set_yticks(range(len(frame)), labels=[f"{name} ({year})" for name, year in zip(frame["paper"], frame["year"])])
    ax.tick_params(axis="x", labelrotation=0, labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_xticks([x - 0.5 for x in range(1, len(FEATURES))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(frame))], minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, "Yes" if matrix[row, col] else "", ha="center", va="center", color="white", fontsize=8)

    ax.set_title("Soft-constraint and application-oriented molecular generation methods", fontsize=14, pad=16)
    ax.set_xlabel("Method component")
    ax.set_ylabel("Paper")
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=240, bbox_inches="tight")
    fig.savefig(OUTPUT_SVG, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
