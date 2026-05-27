#!/usr/bin/env python3
"""Plot the reward loophole risk register."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-diffb110")

import matplotlib.pyplot as plt
import pandas as pd


ANALYSIS_DIR = Path(__file__).resolve().parent


def main() -> None:
    risks = pd.read_csv(ANALYSIS_DIR / "risk_register.tsv", sep="\t")
    colors = {
        "confirmed": "#c2410c",
        "likely": "#d97706",
        "methodological-risk": "#7c3aed",
        "possible": "#2563eb",
    }

    fig, ax = plt.subplots(figsize=(10, 7))
    for status, group in risks.groupby("status"):
        ax.scatter(
            group["likelihood"],
            group["impact"],
            s=group["priority"] * 18,
            color=colors.get(status, "#4b5563"),
            alpha=0.78,
            label=status,
            edgecolor="white",
            linewidth=1.0,
        )
        for _, row in group.iterrows():
            ax.text(
                row["likelihood"] + 0.04,
                row["impact"] + 0.035,
                row["risk_id"],
                fontsize=9,
                weight="bold",
            )

    ax.set_xlim(2.5, 5.45)
    ax.set_ylim(2.5, 5.45)
    ax.set_xticks([3, 4, 5])
    ax.set_yticks([3, 4, 5])
    ax.set_xlabel("Likelihood")
    ax.set_ylabel("Impact")
    ax.set_title("Reward loophole risk priority matrix")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "risk_priority_matrix.png", dpi=220)
    plt.close()

    top = risks.sort_values(["priority", "likelihood"], ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(11, 6))
    labels = top["risk_id"] + " " + top["loophole"]
    ax.barh(labels, top["priority"], color="#c2410c")
    ax.invert_yaxis()
    ax.set_xlabel("Priority = likelihood x impact")
    ax.set_title("Highest-priority reward loopholes")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "top_reward_loopholes.png", dpi=220)
    plt.close()


if __name__ == "__main__":
    main()
