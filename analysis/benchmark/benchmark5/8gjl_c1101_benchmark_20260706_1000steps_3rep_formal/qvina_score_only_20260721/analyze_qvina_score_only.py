"""Summarize and plot five-model DePPA-style QVina score-only results."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
FORMAL_DIR = SCRIPT_DIR.parent
MODEL_ORDER = ["DiffGui", "DiffSBDD", "PMDM", "GCDM-SBDD", "TargetDiff"]
COLORS = {
    "DiffGui": "#0F4D92",
    "DiffSBDD": "#42949E",
    "PMDM": "#D69E2E",
    "GCDM-SBDD": "#9A4D8E",
    "TargetDiff": "#B64342",
}


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 11
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["legend.frameon"] = False


def _root_count(path: str) -> int:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    return len(re.findall(r"(?m)^ROOT\s*$", text))


def annotate_failures(scores: pd.DataFrame) -> pd.DataFrame:
    """Classify raw QuickVina failures without changing the source result CSV."""
    annotated = scores.copy()
    annotated["pdbqt_root_count"] = 1
    annotated["failure_class"] = ""
    failed = ~annotated["success"]
    annotated.loc[failed, "pdbqt_root_count"] = annotated.loc[
        failed, "pdbqt_path"
    ].map(_root_count)
    multi_root = failed & (annotated["pdbqt_root_count"] > 1)
    annotated.loc[multi_root, "failure_class"] = "multiple_disconnected_fragments"
    annotated.loc[failed & ~multi_root, "failure_class"] = annotated.loc[
        failed & ~multi_root, "failure_reason"
    ]
    annotated.to_csv(RESULTS_DIR / "per_ligand_scores_annotated.csv", index=False)
    return annotated


def build_analysis_summary(scores: pd.DataFrame) -> pd.DataFrame:
    """Build robust model-level statistics with explicit denominators."""
    c1101 = float(
        scores.loc[(scores["model"] == "C1101") & scores["success"], "affinity_kcal_mol"].iloc[0]
    )
    rows = []
    for model in MODEL_ORDER:
        all_rows = scores[scores["model"] == model]
        values = all_rows.loc[all_rows["success"], "affinity_kcal_mol"].astype(float)
        rep_means = (
            all_rows.loc[all_rows["success"]]
            .groupby("repeat")["affinity_kcal_mol"]
            .mean()
        )
        affinity_std_between_repeats = (
            rep_means.std(ddof=0) if len(rep_means) > 1 else float("nan")
        )
        rows.append(
            {
                "model": model,
                "generation_attempts": 3000,
                "sanitize_valid_inputs": len(all_rows),
                "score_success": len(values),
                "sanitize_valid_per_attempt": len(all_rows) / 3000,
                "score_success_per_input": len(values) / len(all_rows),
                "score_success_per_attempt": len(values) / 3000,
                "affinity_mean": values.mean(),
                "affinity_median": values.median(),
                "affinity_q25": values.quantile(0.25),
                "affinity_q75": values.quantile(0.75),
                "affinity_std": affinity_std_between_repeats,
                "affinity_std_per_molecule": values.std(ddof=0),
                "affinity_min_best": values.min(),
                "affinity_best_5pct_mean": values.nsmallest(max(1, int(np.ceil(0.05 * len(values))))).mean(),
                "rate_affinity_le_minus10_per_scored": (values <= -10).mean(),
                "rate_affinity_better_than_c1101_per_scored": (values < c1101).mean(),
                "rate_positive_unfavorable_per_scored": (values > 0).mean(),
                "multi_fragment_failure_count": int(
                    (all_rows["failure_class"] == "multiple_disconnected_fragments").sum()
                ),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS_DIR / "analysis_model_summary.csv", index=False)

    per_rep = (
        scores[scores["success"]]
        .groupby(["model", "repeat"])["affinity_kcal_mol"]
        .agg(["mean", "median", "std", "min", "count"])
        .reset_index()
    )
    per_rep.to_csv(RESULTS_DIR / "per_repeat_summary.csv", index=False)
    return summary


def build_docking_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    """Join score-only medians to the existing full-docking summary."""
    docking = pd.read_csv(FORMAL_DIR / "formal_3rep_docking_model_summary.csv")
    comparison = summary[["model", "affinity_median", "affinity_mean"]].merge(
        docking[["model", "score_median_mean", "score_mean_mean"]], on="model", how="inner"
    )
    comparison["score_only_vs_docking_median_delta"] = (
        comparison["affinity_median"] - comparison["score_median_mean"]
    )
    comparison["score_only_median_rank"] = comparison["affinity_median"].rank(
        method="average", ascending=True
    )
    comparison["full_docking_median_rank"] = comparison["score_median_mean"].rank(
        method="average", ascending=True
    )
    comparison.to_csv(RESULTS_DIR / "score_only_vs_full_docking.csv", index=False)
    return comparison


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.04, label, transform=ax.transAxes, fontweight="bold", fontsize=14)


def plot_overview(
    scores: pd.DataFrame,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    """Create a four-panel quantitative overview."""
    successful = scores[(scores["success"]) & (scores["model"].isin(MODEL_ORDER))]
    c1101 = float(
        scores.loc[(scores["model"] == "C1101") & scores["success"], "affinity_kcal_mol"].iloc[0]
    )
    fig = plt.figure(figsize=(8.8, 6.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.12, 1.0])

    ax = fig.add_subplot(grid[0, 0])
    data = [
        successful.loc[successful["model"] == model, "affinity_kcal_mol"].to_numpy()
        for model in MODEL_ORDER
    ]
    box = ax.boxplot(
        data,
        tick_labels=MODEL_ORDER,
        patch_artist=True,
        showfliers=False,
        widths=0.62,
        medianprops={"color": "#272727", "linewidth": 1.2},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
    )
    for patch, model in zip(box["boxes"], MODEL_ORDER):
        patch.set_facecolor(COLORS[model])
        patch.set_alpha(0.78)
    ax.axhline(c1101, color="#606060", linestyle="--", linewidth=1.0, label=f"C1101 ({c1101:.2f})")
    ax.axhline(0, color="#A8A8A8", linewidth=0.7)
    ax.set_ylabel("QVina score-only affinity (kcal/mol)\nlower is better")
    ax.tick_params(axis="x", rotation=22)
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.5, alpha=0.7)
    ax.legend(loc="upper left")
    ax.text(
        0.98,
        0.03,
        "boxes: IQR; outlier points omitted",
        transform=ax.transAxes,
        ha="right",
        color="#606060",
        fontsize=9,
    )
    _panel_label(ax, "a")

    ax = fig.add_subplot(grid[0, 1])
    repeats = ["rep1", "rep2", "rep3"]
    x = np.arange(len(MODEL_ORDER))
    offsets = [-0.18, 0.0, 0.18]
    markers = ["o", "s", "^"]
    for repeat, offset, marker in zip(repeats, offsets, markers):
        medians = [
            successful.loc[
                (successful["model"] == model) & (successful["repeat"] == repeat),
                "affinity_kcal_mol",
            ].median()
            for model in MODEL_ORDER
        ]
        ax.scatter(x + offset, medians, marker=marker, s=24, label=repeat, color="#272727")
    ax.axhline(c1101, color="#606060", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, rotation=22)
    ax.set_ylabel("Per-repeat median affinity (kcal/mol)")
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.5, alpha=0.7)
    ax.legend(title="Repeat", ncol=3, loc="upper left")
    _panel_label(ax, "b")

    ax = fig.add_subplot(grid[1, 0])
    x = np.arange(len(MODEL_ORDER))
    ordered = summary.set_index("model").loc[MODEL_ORDER]
    width = 0.34
    ax.bar(
        x - width / 2,
        100 * ordered["sanitize_valid_per_attempt"],
        width,
        color="#B4C0E4",
        label="Sanitize-valid / 1000 attempts",
    )
    ax.bar(
        x + width / 2,
        100 * ordered["score_success_per_attempt"],
        width,
        color="#0F4D92",
        label="Score success / 1000 attempts",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, rotation=22)
    ax.set_ylabel("Attempt-level rate (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.5, alpha=0.7)
    ax.legend(loc="upper left", fontsize=10)
    _panel_label(ax, "c")

    ax = fig.add_subplot(grid[1, 1])
    label_offsets = {
        "DiffGui": (-4, 10),
        "TargetDiff": (5, -13),
        "DiffSBDD": (-2, 12),
        "PMDM": (5, -13),
        "GCDM-SBDD": (5, 4),
    }
    for _, row in comparison.iterrows():
        model = row["model"]
        ax.scatter(
            row["affinity_median"],
            row["score_median_mean"],
            s=42,
            color=COLORS[model],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        ax.annotate(
            model,
            (row["affinity_median"], row["score_median_mean"]),
            xytext=label_offsets[model],
            textcoords="offset points",
            fontsize=10,
            ha="left" if label_offsets[model][0] >= 0 else "right",
        )
    limits = [
        min(comparison["affinity_median"].min(), comparison["score_median_mean"].min()) - 1,
        max(comparison["affinity_median"].max(), comparison["score_median_mean"].max()) + 1,
    ]
    ax.plot(limits, limits, linestyle="--", color="#A8A8A8", linewidth=0.8)
    correlation = comparison["affinity_median"].corr(comparison["score_median_mean"])
    rank_correlation = comparison["score_only_median_rank"].corr(
        comparison["full_docking_median_rank"]
    )
    ax.text(
        0.04,
        0.94,
        f"Pearson r = {correlation:.3f}\nSpearman rho = {rank_correlation:.2f}",
        transform=ax.transAxes,
        va="top",
    )
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("Score-only pooled median (kcal/mol)")
    ax.set_ylabel("Full-docking repeat-mean median (kcal/mol)")
    ax.grid(color="#D8D8D8", linewidth=0.5, alpha=0.7)
    _panel_label(ax, "d")

    for suffix in ("png", "svg"):
        fig.savefig(RESULTS_DIR / f"qvina_score_only_overview.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run failure annotation, robust summaries, and plotting."""
    scores = pd.read_csv(RESULTS_DIR / "per_ligand_scores.csv")
    scores["success"] = scores["success"].astype(bool)
    annotated = annotate_failures(scores)
    summary = build_analysis_summary(annotated)
    comparison = build_docking_comparison(summary)
    plot_overview(annotated, summary, comparison)


if __name__ == "__main__":
    main()
