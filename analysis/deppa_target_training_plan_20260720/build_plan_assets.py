"""Build a compact workflow figure for the DePPA target-training plan."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT_DIR = Path(__file__).resolve().parent
PHASES = [
    ("P0", "Target pack", "PDB/PDBQT/SDF/residues"),
    ("P1", "Prototype hardening", "logging/resume/optimizer"),
    ("P2", "Frozen baseline", "fixed-seed generation"),
    ("P3", "2-rollout smoke", "reward + PPO correctness"),
    ("P4", "10-rollout pilot", "signal and drift gate"),
    ("P5", "100-rollout run", "checkpoint every rollout"),
    ("P6", "Held-out audit", "sanitize/docking/diversity"),
]


def main() -> None:
    """Write the phase table and workflow figure."""
    with (OUT_DIR / "training_phases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["phase", "name", "deliverable"])
        writer.writerows(PHASES)

    fig, ax = plt.subplots(figsize=(14.0, 3.2), constrained_layout=True)
    ax.set_xlim(-0.4, len(PHASES) - 0.6)
    ax.set_ylim(-0.8, 1.0)
    ax.axis("off")
    colors = ["#457b9d", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#6d597a"]
    for index, ((phase, name, detail), color) in enumerate(zip(PHASES, colors)):
        box = FancyBboxPatch(
            (index - 0.42, -0.25),
            0.84,
            0.72,
            boxstyle="round,pad=0.03,rounding_size=0.05",
            facecolor=color,
            edgecolor="white",
            linewidth=1.5,
        )
        ax.add_patch(box)
        ax.text(index, 0.28, phase, ha="center", va="center", color="white", weight="bold")
        ax.text(index, 0.04, name, ha="center", va="center", color="white", fontsize=9, weight="bold")
        ax.text(index, -0.47, detail, ha="center", va="top", fontsize=8)
        if index < len(PHASES) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (index + 0.43, 0.11),
                    (index + 0.57, 0.11),
                    arrowstyle="-|>",
                    mutation_scale=12,
                    linewidth=1.4,
                    color="#444444",
                )
            )
    ax.set_title("DePPA single-target training: gated execution plan", fontsize=15, pad=15)
    fig.savefig(OUT_DIR / "deppa_target_training_plan.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
