"""Plot global metrics from the DePPA four-GPU resume test."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
METRICS_PATH = ROOT / "output" / "8GJL" / "metrics" / "metrics.csv"
OUTPUT_PATH = ROOT / "deppa_4gpu_resume_metrics.png"


def main() -> None:
    """Render the two-epoch correctness metrics."""
    with METRICS_PATH.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    epochs = [int(row["rollout"]) for row in rows]
    panels = (
        ("valid_rate", "Valid fraction", (0.0, 1.05)),
        ("qed", "Mean QED", (0.0, 1.0)),
        ("vina_score", "Mean QVina reward", None),
        ("distance", "Global fingerprint diversity", (0.0, 1.0)),
    )

    plt.rcParams.update({"font.family": "Arial", "font.size": 10})
    figure, axes = plt.subplots(2, 2, figsize=(8.4, 5.6), constrained_layout=True)
    for axis, (field, title, limits) in zip(axes.flat, panels):
        values = [float(row[field]) for row in rows]
        axis.plot(epochs, values, marker="o", linewidth=1.8, color="#176b87")
        axis.set_title(title)
        axis.set_xlabel("Completed epoch")
        axis.set_xticks(epochs)
        if limits is not None:
            axis.set_ylim(*limits)
        axis.grid(axis="y", color="#d9d9d9", linewidth=0.7)
        for epoch, value in zip(epochs, values):
            axis.annotate(
                f"{value:.3f}",
                (epoch, value),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
            )

    figure.suptitle("DePPA 8GJL four-GPU checkpoint/resume correctness")
    figure.savefig(OUTPUT_PATH, dpi=180)


if __name__ == "__main__":
    main()
