"""Plot implementation plan figures for dual-side sidechain soft coupling."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


def _draw_box(ax, xy, text, width=0.18, height=0.12, color="#eef3f7") -> None:
    rect = plt.Rectangle(xy, width, height, facecolor=color, edgecolor="#39434d", linewidth=1)
    ax.add_patch(rect)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=8.5)


def _arrow(ax, start, end) -> None:
    ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.15, "color": "#39434d"})


def plot_dataflow(out_dir: Path) -> None:
    """Plot the end-to-end dataflow."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    boxes = [
        ((0.03, 0.73), "MappingSpec\nR1/R2/root/core", "#f4f1de"),
        ((0.25, 0.73), "DiffSBDD\nx+h sampling", "#eef3f7"),
        ((0.47, 0.73), "Soft coupling\nh-copy / h-consensus", "#e0f2f1"),
        ((0.70, 0.73), "OpenBabel/RDKit\ncanonical mol", "#eef3f7"),
        ((0.70, 0.45), "SidechainGraphAudit\niso + attachment + rings", "#fce8d5"),
        ((0.47, 0.20), "Consistency gate\nsuccess / failure", "#fce8d5"),
        ((0.25, 0.20), "Reward/filter\nfinite penalty or discard", "#e8edf7"),
        ((0.03, 0.20), "PPO train\nmasked logprob/KL", "#e8edf7"),
    ]
    for xy, text, color in boxes:
        _draw_box(ax, xy, text, color=color)

    _arrow(ax, (0.21, 0.79), (0.25, 0.79))
    _arrow(ax, (0.43, 0.79), (0.47, 0.79))
    _arrow(ax, (0.65, 0.79), (0.70, 0.79))
    _arrow(ax, (0.79, 0.73), (0.79, 0.57))
    _arrow(ax, (0.70, 0.51), (0.65, 0.32))
    _arrow(ax, (0.47, 0.26), (0.43, 0.26))
    _arrow(ax, (0.25, 0.26), (0.21, 0.26))
    _arrow(ax, (0.12, 0.32), (0.12, 0.73))

    ax.text(
        0.03,
        0.04,
        "Principle: generation-time coupling is optional and soft; chemical identity is judged after canonical reconstruction.",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "dual_sidechain_dataflow.png", dpi=180)
    fig.savefig(out_dir / "dual_sidechain_dataflow.svg")
    plt.close(fig)


def plot_stage_gates(out_dir: Path) -> None:
    """Plot stage gate timeline from the CSV matrix."""
    rows = list(csv.DictReader((out_dir / "stage_gate_matrix.csv").open()))
    fig, ax = plt.subplots(figsize=(11, 5.2))
    y = list(range(len(rows)))
    readiness = [5, 4, 3, 2, 1, 1]
    risk = [1, 2, 3, 3, 3, 5]
    scientific_value = [3, 3, 2, 4, 2, 5]
    labels = [f"{row['stage']} {row['name'].replace('_', ' ')}" for row in rows]

    ax.barh([i + 0.22 for i in y], readiness, height=0.2, color="#4c78a8", label="readiness")
    ax.barh(y, scientific_value, height=0.2, color="#2f9e8f", label="scientific value")
    ax.barh([i - 0.22 for i in y], risk, height=0.2, color="#e45756", label="risk")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, 5.2)
    ax.set_xlabel("qualitative score, 1 low / 5 high")
    ax.set_title("Stage gates for dual-side soft coupling")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "stage_gate_scores.png", dpi=180)
    fig.savefig(out_dir / "stage_gate_scores.svg")
    plt.close(fig)


def main() -> None:
    """Generate all figures."""
    out_dir = Path(__file__).resolve().parent
    plot_dataflow(out_dir)
    plot_stage_gates(out_dir)


if __name__ == "__main__":
    main()
