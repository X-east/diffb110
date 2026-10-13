"""Plot the two-rollout DePPA official-configuration smoke metrics."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
METRICS_PATH = ROOT / "output" / "8GJL" / "metrics" / "metrics.csv"


def main() -> None:
    """Create compact PNG and SVG summaries from the global metrics CSV."""
    with METRICS_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    rollouts = [int(row["rollout"]) for row in rows]
    valid_rates = [100.0 * float(row["valid_rate"]) for row in rows]
    qed = [float(row["qed"]) for row in rows]
    sa = [float(row["sa"]) for row in rows]
    vina = [float(row["vina_score"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    axes[0].plot(rollouts, valid_rates, marker="o", color="#277da1", linewidth=2)
    axes[0].set(title="Attempt-level validity", xlabel="Rollout", ylabel="Valid (%)")
    axes[0].set_xticks(rollouts)
    axes[0].set_ylim(0, 100)
    axes[0].grid(alpha=0.25)

    axes[1].plot(rollouts, qed, marker="o", label="QED", color="#43aa8b")
    axes[1].plot(rollouts, sa, marker="s", label="SA reward", color="#f9c74f")
    vina_axis = axes[1].twinx()
    vina_axis.plot(rollouts, vina, marker="^", label="QVina reward", color="#f94144")
    axes[1].set(title="Global rollout metrics", xlabel="Rollout", ylabel="QED / SA")
    vina_axis.set_ylabel("QVina reward")
    axes[1].set_xticks(rollouts)
    axes[1].grid(alpha=0.25)
    lines = axes[1].lines + vina_axis.lines
    axes[1].legend(lines, [line.get_label() for line in lines], loc="best")

    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(ROOT / f"official_smoke_metrics.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
