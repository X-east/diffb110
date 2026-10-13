"""Plot the audited reinforcement-learning attempt timeline and outcome matrix."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def plot_timeline(df: pd.DataFrame) -> None:
    """Plot completed versus planned epochs for every audited attempt."""
    plot_df = df.copy()
    plot_df["completion"] = plot_df["completed_epochs"] / plot_df["planned_epochs"]
    colors = {
        "completed": "#2A9D8F",
        "completed matrix": "#2A9D8F",
        "completed continuation": "#2A9D8F",
        "short completed": "#E9C46A",
        "23 complete plus partial epoch 24": "#E9C46A",
        "stopped": "#E76F51",
        "stopped by user": "#E76F51",
        "stopped/incomplete": "#E76F51",
        "incomplete continuation": "#E76F51",
        "failed during epoch 90 merge; latest checkpoint 85": "#D1495B",
    }
    fig, ax = plt.subplots(figsize=(12, 8.2))
    y = np.arange(len(plot_df))
    ax.barh(y, np.ones(len(plot_df)), color="#E8EAED", height=0.66)
    ax.barh(
        y,
        plot_df["completion"].clip(upper=1),
        color=[colors.get(x, "#8D99AE") for x in plot_df["status"]],
        height=0.66,
    )
    labels = [f"{row.id}  {row.model}: {row.method}" for row in plot_df.itertuples()]
    ax.set_yticks(y, labels=labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Fraction of planned epochs with complete summaries")
    ax.set_title("Audited RL attempts: execution completeness is not efficacy")
    ax.grid(axis="x", alpha=0.2)
    for yi, row in enumerate(plot_df.itertuples()):
        ax.text(
            min(row.completion, 1) + 0.01,
            yi,
            f"{row.completed_epochs}/{row.planned_epochs}",
            va="center",
            fontsize=7.5,
        )
    fig.tight_layout()
    fig.savefig(ROOT / "training_attempt_timeline.png", dpi=220)
    fig.savefig(ROOT / "training_attempt_timeline.svg")
    plt.close(fig)


def plot_outcome_matrix() -> None:
    """Plot evidence-coded outcomes for representative completed strategies."""
    rows = [
        "DS dock+classifier",
        "DS strong guard",
        "DG ring-last3",
        "DG Next6-04 ep50",
        "DG DePPA t200 ep89",
        "DG PB-fast ep100",
        "Original DePPA ep100",
    ]
    cols = ["Target proxy", "RDKit/recon", "3D geometry", "Properties", "PPO stability"]
    # Ordinal evidence code: +2 strong gain, +1 gain, 0 mixed/unknown, -1 cost, -2 severe cost.
    values = np.array(
        [
            [1, 1, 0, -2, 0],
            [-2, 0, 0, 1, 0],
            [0, 2, 0, -1, -1],
            [1, -1, 0, -2, -1],
            [1, 1, -2, -1, -2],
            [-1, 1, -1, -2, -2],
            [2, 1, 0, -2, 2],
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(10.2, 5.3))
    image = ax.imshow(values, cmap="RdYlGn", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(np.arange(len(cols)), labels=cols)
    ax.set_yticks(np.arange(len(rows)), labels=rows)
    ax.set_title("Evidence-coded outcome matrix (not a common-scale benchmark)")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:+.0f}", ha="center", va="center", fontsize=9)
    cbar = fig.colorbar(image, ax=ax, shrink=0.82)
    cbar.set_ticks([-2, -1, 0, 1, 2], labels=["severe cost", "cost", "mixed/NA", "gain", "strong gain"])
    fig.tight_layout()
    fig.savefig(ROOT / "representative_outcome_matrix.png", dpi=220)
    fig.savefig(ROOT / "representative_outcome_matrix.svg")
    plt.close(fig)


def main() -> None:
    """Generate all review plots."""
    attempts = pd.read_csv(ROOT / "training_attempts.csv")
    plot_timeline(attempts)
    plot_outcome_matrix()


if __name__ == "__main__":
    main()
