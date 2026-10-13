"""Analyze and plot the completed 100-rollout DePPA RE-CmeB training run."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "OUT" / "deppa-8gjl-official-4gpu-20260720"
OUTPUT_DIR = Path(__file__).resolve().parent
GLOBAL_METRICS = RUN_DIR / "output" / "8GJL" / "metrics" / "metrics.csv"
RANK_METRICS_GLOB = RUN_DIR / "output" / "8GJL" / "metrics"

BLUE = "#0F4D92"
BLUE_SOFT = "#7AA6D1"
RED = "#B64342"
GREEN = "#42949E"
GOLD = "#D69E2E"
VIOLET = "#9A4D8E"
GREY = "#767676"
WINDOW = 10


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 12
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["legend.frameon"] = False


def _rolling(series: pd.Series) -> pd.Series:
    return series.rolling(WINDOW, min_periods=1).mean()


def _plot_raw_and_smooth(
    ax: plt.Axes,
    x: pd.Series,
    y: pd.Series,
    *,
    color: str,
    label: str,
) -> None:
    ax.plot(x, y, color=color, alpha=0.20, linewidth=0.8)
    ax.plot(x, _rolling(y), color=color, linewidth=2.0, label=label)


def _style_axis(ax: plt.Axes, ylabel: str) -> None:
    ax.set_xlabel("Rollout")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.6, alpha=0.65)
    ax.set_xlim(1, 100)


def load_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load global metrics and rank-mean diagnostic metrics."""
    global_df = pd.read_csv(GLOBAL_METRICS)
    rank_frames = []
    for path in sorted(RANK_METRICS_GLOB.glob("rank_*.csv")):
        frame = pd.read_csv(path)
        frame["rank"] = int(path.stem.split("_")[-1])
        rank_frames.append(frame)
    rank_df = pd.concat(rank_frames, ignore_index=True)
    rank_mean = (
        rank_df.groupby("rollout", as_index=False)
        .agg(
            diversity=("diversity", "mean"),
            logp=("LogP", "mean"),
            rank_valid_rate=("valid_rate", "mean"),
        )
        .sort_values("rollout")
    )
    return global_df, rank_mean


def write_tables(global_df: pd.DataFrame, rank_mean: pd.DataFrame) -> dict[str, float]:
    """Write 20-rollout stage summaries and early/late comparisons."""
    merged = global_df.merge(rank_mean, on="rollout", how="left")
    stage_rows = []
    stage_metrics = [
        "valid_rate",
        "vina_score",
        "qed",
        "sa",
        "distance",
        "mean_size",
        "connectivity",
        "diversity",
        "logp",
        "combined_reward_std",
        "approx_kl",
        "clipfrac",
    ]
    for start in range(1, 101, 20):
        stop = start + 19
        subset = merged[merged["rollout"].between(start, stop)]
        row = {"stage": f"{start}-{stop}", "n_rollouts": len(subset)}
        row.update({metric: float(subset[metric].mean()) for metric in stage_metrics})
        row["approx_kl_median"] = float(subset["approx_kl"].median())
        row["approx_kl_max"] = float(subset["approx_kl"].max())
        stage_rows.append(row)
    pd.DataFrame(stage_rows).to_csv(OUTPUT_DIR / "stage_metrics.csv", index=False)

    early = merged.iloc[:10]
    late = merged.iloc[-10:]
    comparison_rows = []
    for metric in stage_metrics:
        early_mean = float(early[metric].mean())
        late_mean = float(late[metric].mean())
        comparison_rows.append(
            {
                "metric": metric,
                "early_1_10_mean": early_mean,
                "late_91_100_mean": late_mean,
                "absolute_change": late_mean - early_mean,
                "relative_change_percent": (
                    100.0 * (late_mean - early_mean) / abs(early_mean)
                    if early_mean != 0.0
                    else np.nan
                ),
            }
        )
    pd.DataFrame(comparison_rows).to_csv(
        OUTPUT_DIR / "early_late_comparison.csv", index=False
    )

    summary = {
        "rollouts": int(len(global_df)),
        "attempts": int(global_df["attempts"].sum()),
        "written": int(global_df["written"].sum()),
        "overall_attempt_valid_rate": float(
            global_df["written"].sum() / global_df["attempts"].sum()
        ),
        "early_vina_reward": float(early["vina_score"].mean()),
        "late_vina_reward": float(late["vina_score"].mean()),
        "early_qed": float(early["qed"].mean()),
        "late_qed": float(late["qed"].mean()),
        "early_sa": float(early["sa"].mean()),
        "late_sa": float(late["sa"].mean()),
        "early_diversity": float(early["diversity"].mean()),
        "late_diversity": float(late["diversity"].mean()),
        "early_logp": float(early["logp"].mean()),
        "late_logp": float(late["logp"].mean()),
        "early_valid_rate": float(early["valid_rate"].mean()),
        "late_valid_rate": float(late["valid_rate"].mean()),
        "early_connectivity": float(early["connectivity"].mean()),
        "late_connectivity": float(late["connectivity"].mean()),
        "vina_qed_correlation": float(merged["vina_score"].corr(merged["qed"])),
        "max_approx_kl": float(global_df["approx_kl"].max()),
        "max_approx_kl_rollout": int(
            global_df.loc[global_df["approx_kl"].idxmax(), "rollout"]
        ),
        "nonzero_clipfrac_rollouts": int((global_df["clipfrac"] > 0).sum()),
    }
    (OUTPUT_DIR / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def plot_outcomes(global_df: pd.DataFrame, rank_mean: pd.DataFrame) -> None:
    """Plot reward, chemistry, validity, diversity, and lipophilicity trends."""
    x = global_df["rollout"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), constrained_layout=True)

    ax = axes[0, 0]
    _plot_raw_and_smooth(ax, x, global_df["vina_score"], color=BLUE, label="10-rollout mean")
    _style_axis(ax, "QVina reward (higher is better)")
    ax.legend(loc="lower right")
    ax.text(-0.12, 1.05, "a", transform=ax.transAxes, fontweight="bold", fontsize=15)

    ax = axes[0, 1]
    _plot_raw_and_smooth(ax, x, global_df["qed"], color=GREEN, label="QED")
    _plot_raw_and_smooth(ax, x, global_df["sa"], color=GOLD, label="SA reward")
    _style_axis(ax, "Drug-likeness reward")
    ax.legend(loc="best")
    ax.text(-0.12, 1.05, "b", transform=ax.transAxes, fontweight="bold", fontsize=15)

    ax = axes[1, 0]
    _plot_raw_and_smooth(
        ax, x, 100.0 * global_df["valid_rate"], color=BLUE, label="Attempt-valid"
    )
    _plot_raw_and_smooth(
        ax, x, 100.0 * global_df["connectivity"], color=VIOLET, label="Connectivity"
    )
    _style_axis(ax, "Rate (%)")
    ax.set_ylim(75, 102)
    ax.legend(loc="lower right")
    ax.text(-0.12, 1.05, "c", transform=ax.transAxes, fontweight="bold", fontsize=15)

    ax = axes[1, 1]
    _plot_raw_and_smooth(
        ax,
        rank_mean["rollout"],
        rank_mean["diversity"],
        color=GREEN,
        label="Diversity",
    )
    _style_axis(ax, "Rank-mean RDK diversity")
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(
        rank_mean["rollout"],
        rank_mean["logp"],
        color=RED,
        alpha=0.18,
        linewidth=0.8,
    )
    ax2.plot(
        rank_mean["rollout"],
        _rolling(rank_mean["logp"]),
        color=RED,
        linewidth=2.0,
        label="LogP",
    )
    ax2.set_ylabel("Rank-mean LogP", color=RED)
    lines = ax.lines[-1:] + ax2.lines[-1:]
    ax.legend(lines, [line.get_label() for line in lines], loc="best")
    ax.text(-0.12, 1.05, "d", transform=ax.transAxes, fontweight="bold", fontsize=15)

    for suffix in ("png", "svg"):
        fig.savefig(OUTPUT_DIR / f"training_outcome_curves.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ppo_stability(global_df: pd.DataFrame) -> None:
    """Plot PPO diagnostics and highlight the isolated rollout-68 excursion."""
    x = global_df["rollout"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), constrained_layout=True)

    panels = [
        ("combined_reward_std", "Reward standard deviation", BLUE, "linear"),
        ("approx_kl", "Approximate KL", RED, "log"),
        ("clipfrac", "Clipped update fraction", VIOLET, "log_floor"),
        ("loss", "PPO loss", GREY, "symlog"),
    ]
    for label, (metric, ylabel, color, scale) in zip("abcd", panels):
        ax = axes.flat[ord(label) - ord("a")]
        y = global_df[metric]
        if scale == "log_floor":
            y = y.clip(lower=1e-5)
        ax.plot(x, y, color=color, linewidth=1.2)
        ax.axvline(68, color=RED, linestyle="--", linewidth=0.9, alpha=0.75)
        rollout_68_value = global_df.loc[
            global_df["rollout"] == 68, metric
        ].iloc[0]
        if scale == "log_floor":
            rollout_68_value = max(rollout_68_value, 1e-5)
        ax.scatter(
            [68],
            [rollout_68_value],
            color=RED,
            s=24,
            zorder=3,
        )
        _style_axis(ax, ylabel)
        if scale == "log":
            ax.set_yscale("log")
        elif scale == "log_floor":
            ax.set_yscale("log")
            ax.text(
                0.02,
                0.06,
                "zeros shown at 1e-5",
                transform=ax.transAxes,
                color=GREY,
                fontsize=9,
            )
        elif scale == "symlog":
            ax.set_yscale("symlog", linthresh=1e-4)
        ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontweight="bold", fontsize=15)
    axes[0, 1].annotate(
        "rollout 68\nrank-2 excursion",
        xy=(68, global_df.loc[global_df["rollout"] == 68, "approx_kl"].iloc[0]),
        xytext=(48, 0.01),
        arrowprops={"arrowstyle": "->", "color": RED, "linewidth": 0.8},
        color=RED,
        fontsize=10,
    )

    for suffix in ("png", "svg"):
        fig.savefig(OUTPUT_DIR / f"ppo_stability_curves.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run the complete training-metric analysis."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    global_df, rank_mean = load_metrics()
    if len(global_df) != 100:
        raise ValueError(f"Expected 100 global metric rows, found {len(global_df)}")
    required = global_df.select_dtypes(include=[np.number])
    if not np.isfinite(required.to_numpy()).all():
        raise ValueError("Global metrics contain NaN or infinite values")
    write_tables(global_df, rank_mean)
    plot_outcomes(global_df, rank_mean)
    plot_ppo_stability(global_df)


if __name__ == "__main__":
    main()
