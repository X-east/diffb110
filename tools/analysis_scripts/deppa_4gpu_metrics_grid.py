"""Plot corrected training metrics for the DePPA 4-GPU RE-CmeB run.

The training pipeline stores ``vina_score`` as the negated QuickVina2
``--score_only`` affinity so that a larger reward is better. This figure
converts it back to the conventional affinity sign: more negative is better.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
METRICS_DIR = (
    ROOT
    / "OUT"
    / "deppa-8gjl-official-4gpu-20260720"
    / "output"
    / "8GJL"
    / "metrics"
)
GLOBAL_METRICS = METRICS_DIR / "metrics.csv"
OUTPUT_STEM = METRICS_DIR / "deppa_4gpu_training_metrics_grid"
ROLLING_WINDOW = 10

COLORS = {
    "affinity": "#165A72",
    "qed": "#D9822B",
    "sa": "#25877A",
    "diversity": "#4A8F58",
    "validity": "#3B78A8",
    "logp": "#8A5AA8",
    "decline": "#B4453F",
    "neutral": "#6B6B6B",
    "grid": "#E3E3E3",
}


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.labelsize": 8.5,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def _load_metrics() -> pd.DataFrame:
    """Load global metrics and add rank-mean LogP from per-rank files."""
    metrics = pd.read_csv(GLOBAL_METRICS).sort_values("rollout")
    rank_frames = []
    for path in sorted(METRICS_DIR.glob("rank_*.csv")):
        rank_frames.append(pd.read_csv(path)[["rollout", "LogP"]])
    if not rank_frames:
        raise FileNotFoundError(f"No rank metrics found under {METRICS_DIR}")
    rank_logp = (
        pd.concat(rank_frames, ignore_index=True)
        .groupby("rollout", as_index=False)["LogP"]
        .mean()
        .rename(columns={"LogP": "rank_mean_logp"})
    )
    metrics = metrics.merge(rank_logp, on="rollout", how="left")
    metrics["qvina_affinity"] = -metrics["vina_score"]
    metrics["valid_percent"] = 100.0 * metrics["valid_rate"]
    return metrics


def _rolling(values: pd.Series) -> pd.Series:
    """Return a centered rolling mean for visual trend estimation."""
    return values.rolling(ROLLING_WINDOW, center=True, min_periods=1).mean()


def _style_axis(ax: plt.Axes, panel_label: str) -> None:
    """Apply shared panel styling."""
    ax.set_xlim(1, 100)
    ax.set_xticks([1, 20, 40, 60, 80, 100])
    ax.set_xlabel("Rollout")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.6)
    ax.tick_params(length=3, width=0.7, color="#777777")
    ax.text(
        -0.12,
        1.06,
        panel_label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )


def _plot_metric(
    ax: plt.Axes,
    metrics: pd.DataFrame,
    *,
    column: str,
    title: str,
    ylabel: str,
    color: str,
    panel_label: str,
    decimals: int = 3,
    annotation_color: str | None = None,
) -> None:
    """Plot raw rollout values, a rolling mean, and early-to-late change."""
    x = metrics["rollout"]
    y = metrics[column]
    smooth = _rolling(y)
    early = float(y.iloc[:10].mean())
    late = float(y.iloc[-10:].mean())
    delta = late - early

    ax.plot(x, y, color=color, alpha=0.22, linewidth=0.8)
    ax.plot(x, smooth, color=color, linewidth=2.0)
    ax.scatter([x.iloc[-1]], [smooth.iloc[-1]], color=color, s=18, zorder=3)
    ax.set_title(title, loc="left", fontweight="bold", pad=6)
    ax.set_ylabel(ylabel)
    _style_axis(ax, panel_label)

    number = f"{{:.{decimals}f}}"
    ax.text(
        0.98,
        0.96,
        f"early {number.format(early)}\nlate {number.format(late)}\n"
        f"delta {number.format(delta)}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.2,
        color=annotation_color or color,
    )


def main() -> None:
    """Generate corrected PNG, SVG, and PDF training-metric figures."""
    metrics = _load_metrics()
    if len(metrics) != 100:
        raise ValueError(f"Expected 100 rollouts, found {len(metrics)}")
    numeric = metrics.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Metrics contain NaN or infinite values")

    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.2), constrained_layout=True)
    panels = axes.ravel()

    _plot_metric(
        panels[0],
        metrics,
        column="qvina_affinity",
        title="QuickVina2 score-only affinity",
        ylabel="Affinity (kcal/mol; more negative is better)",
        color=COLORS["affinity"],
        panel_label="a",
        decimals=2,
    )
    _plot_metric(
        panels[1],
        metrics,
        column="qed",
        title="Drug-likeness",
        ylabel="QED",
        color=COLORS["qed"],
        panel_label="b",
        annotation_color=COLORS["decline"],
    )
    _plot_metric(
        panels[2],
        metrics,
        column="sa",
        title="Synthetic accessibility",
        ylabel="SA reward (higher is better)",
        color=COLORS["sa"],
        panel_label="c",
    )
    _plot_metric(
        panels[3],
        metrics,
        column="distance",
        title="Molecular diversity",
        ylabel="RDK diversity (1 - mean Tanimoto)",
        color=COLORS["diversity"],
        panel_label="d",
    )
    _plot_metric(
        panels[4],
        metrics,
        column="valid_percent",
        title="Attempt-level validity",
        ylabel="Valid molecules / 32 attempts (%)",
        color=COLORS["validity"],
        panel_label="e",
        decimals=1,
    )
    panels[4].set_ylim(79, 101)
    _plot_metric(
        panels[5],
        metrics,
        column="rank_mean_logp",
        title="Lipophilicity",
        ylabel="Mean LogP across four ranks",
        color=COLORS["logp"],
        panel_label="f",
        annotation_color=COLORS["decline"],
    )

    fig.suptitle(
        "DePPA reinforcement learning on RE-CmeB (8GJL)",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.965,
        "100 rollouts; 32 attempts per rollout; thin lines show rollout values and thick lines show 10-rollout means",
        ha="center",
        va="top",
        fontsize=7.5,
        color=COLORS["neutral"],
    )

    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_STEM.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPUT_STEM}.png/.svg/.pdf")


if __name__ == "__main__":
    main()
