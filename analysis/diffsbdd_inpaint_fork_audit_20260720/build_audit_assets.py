"""Build the compact capability audit for the local DiffSBDD inpaint fork."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


OUT_DIR = Path(__file__).resolve().parent
ROWS = [
    ("Restore fixed coordinates/types", 3, "Implemented before reconstruction"),
    ("Preserve fixed-core graph", 3, "Copies fixed SDF bonds after reconstruction"),
    ("Selectable reconstruction backend", 3, "Useful and isolated"),
    ("Paired atom-type synchronization", 1, "Prototype; slot ordering assumption"),
    ("Root radial-distance nudge", 1, "No exit direction or bond guarantee"),
    ("All-pair radial-distance nudge", 0, "Radial shells are not R-group topology"),
    ("Explicit attachment bond", 0, "Not represented by DiffSBDD state"),
    ("Variable-size dual-R sampling", 0, "Constraint requires explicit add_n_nodes"),
]


def main() -> None:
    """Write CSV and a horizontal capability plot."""
    with (OUT_DIR / "component_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["component", "suitability_score", "reason"])
        writer.writerows(ROWS)

    labels = [row[0] for row in ROWS][::-1]
    scores = [row[1] for row in ROWS][::-1]
    colors = ["#2a9d8f" if score == 3 else "#e9c46a" if score > 0 else "#e76f51" for score in scores]
    fig, ax = plt.subplots(figsize=(9.4, 5.2), constrained_layout=True)
    bars = ax.barh(labels, scores, color=colors)
    ax.set_xlim(0, 3.25)
    ax.set_xticks([0, 1, 2, 3], ["absent/unsuitable", "prototype", "partial", "suitable"])
    ax.set_title("Local DiffSBDD inpaint fork: component suitability")
    ax.grid(axis="x", alpha=0.25)
    for bar, score in zip(bars, scores):
        ax.text(score + 0.06, bar.get_y() + bar.get_height() / 2, str(score), va="center")
    fig.savefig(OUT_DIR / "component_suitability.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
