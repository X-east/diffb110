"""Create route feasibility figures for the dual-side soft-coupling report."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    """Generate qualitative feasibility and workflow figures."""
    out_dir = Path(__file__).resolve().parent
    rows = list(csv.DictReader((out_dir / "feasibility_matrix.csv").open()))

    labels = [row["route"].replace("_", "\n") for row in rows]
    complexity = [1, 2, 2, 4, 5]
    fit = [1, 3, 3, 4, 5]
    readiness = [5, 4, 3, 2, 1]

    fig, ax = plt.subplots(figsize=(10, 5.2))
    y_positions = list(range(len(rows)))
    ax.barh([y + 0.22 for y in y_positions], fit, height=0.2, label="target fit", color="#2f9e8f")
    ax.barh(y_positions, readiness, height=0.2, label="implementation readiness", color="#4c78a8")
    ax.barh([y - 0.22 for y in y_positions], complexity, height=0.2, label="engineering complexity", color="#f58518")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, 5.2)
    ax.set_xlabel("qualitative score, 1 low / 5 high")
    ax.set_title("Dual-side sidechain routes: target fit vs readiness vs complexity")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "route_feasibility_scores.png", dpi=180)
    fig.savefig(out_dir / "route_feasibility_scores.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axis("off")
    boxes = [
        ("Pocket L/R\nconditioning", 0.05, 0.62),
        ("DiffSBDD x+h\nreverse sampling", 0.25, 0.62),
        ("Optional soft coupling\nh-copy / consensus / reward", 0.48, 0.62),
        ("OpenBabel/RDKit\nreconstruction", 0.72, 0.62),
        ("Consistency gate\nR1 graph == R2 graph?", 0.48, 0.20),
        ("Success pool / score\nvalid + dock + consistency", 0.72, 0.20),
    ]
    for text, x_coord, y_coord in boxes:
        rect = plt.Rectangle(
            (x_coord, y_coord),
            0.18,
            0.18,
            fill=True,
            facecolor="#eef3f7",
            edgecolor="#3b4a54",
            linewidth=1,
        )
        ax.add_patch(rect)
        ax.text(x_coord + 0.09, y_coord + 0.09, text, ha="center", va="center", fontsize=9)

    arrows = [
        ((0.23, 0.71), (0.25, 0.71)),
        ((0.43, 0.71), (0.48, 0.71)),
        ((0.66, 0.71), (0.72, 0.71)),
        ((0.81, 0.62), (0.57, 0.38)),
        ((0.66, 0.29), (0.72, 0.29)),
    ]
    for start, end in arrows:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#3b4a54"},
        )
    ax.text(
        0.05,
        0.04,
        "After clarification: failure is allowed; identical reconstructed sidechains are an acceptance metric, not a guaranteed latent invariant.",
        ha="left",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "soft_coupling_acceptance_flow.png", dpi=180)
    fig.savefig(out_dir / "soft_coupling_acceptance_flow.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
