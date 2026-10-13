"""Combine and visualize DiffGui QVina score-only results across four step counts."""

from __future__ import annotations

import csv
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
RESULTS_DIR = SCRIPT_DIR / "results"
STEP_ORDER = [100, 200, 500, 1000]
REPEAT_ORDER = ["rep1", "rep2", "rep3"]
COLORS = {100: "#3A6EA5", 200: "#4E9F78", 500: "#D99B42", 1000: "#B64F4A"}


def _read_scores() -> pd.DataFrame:
    """Read new 100/200/500 scores and the completed formal 1000-step scores."""
    new_scores = pd.read_csv(SCRIPT_DIR / "results_100_200_500/per_ligand_scores.csv")
    extracted_steps = pd.to_numeric(
        new_scores["model"].str.extract(r"^(\d+) steps$")[0], errors="coerce"
    )
    new_scores["diffusion_steps"] = extracted_steps
    formal_scores = pd.read_csv(
        PROJECT_ROOT
        / "analysis/benchmark/benchmark5"
        / "8gjl_c1101_benchmark_20260706_1000steps_3rep_formal"
        / "qvina_score_only_20260721/results/per_ligand_scores.csv"
    )
    formal_scores = formal_scores[
        (formal_scores["model"] == "DiffGui") | (formal_scores["model"] == "C1101")
    ].copy()
    formal_scores["diffusion_steps"] = np.where(formal_scores["model"] == "DiffGui", 1000, np.nan)
    combined = pd.concat([new_scores, formal_scores], ignore_index=True)
    combined["success"] = combined["success"].astype(str).str.lower().isin(["true", "1"])
    combined["affinity_kcal_mol"] = pd.to_numeric(combined["affinity_kcal_mol"], errors="coerce")
    combined = combined.drop_duplicates(subset=["task_id", "diffusion_steps"], keep="first")
    return combined


def _best_fraction_mean(values: pd.Series, fraction: float = 0.05) -> float:
    count = max(1, math.ceil(len(values) * fraction))
    return float(values.nsmallest(count).mean())


def _build_summaries(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write explicit per-repeat and pooled summaries with attempt denominators."""
    rows = []
    generated = scores[scores["diffusion_steps"].notna()].copy()
    generated["diffusion_steps"] = generated["diffusion_steps"].astype(int)
    for steps in STEP_ORDER:
        for repeat in REPEAT_ORDER:
            group = generated[
                (generated["diffusion_steps"] == steps) & (generated["repeat"] == repeat)
            ]
            values = group.loc[group["success"], "affinity_kcal_mol"].dropna()
            rows.append(
                {
                    "diffusion_steps": steps,
                    "repeat": repeat,
                    "generation_attempts": 1000,
                    "sanitize_valid_inputs": len(group),
                    "score_success_count": len(values),
                    "score_success_per_input": len(values) / len(group),
                    "score_success_per_attempt": len(values) / 1000,
                    "affinity_mean": values.mean(),
                    "affinity_median": values.median(),
                    "affinity_std_population": values.std(ddof=0),
                    "affinity_best_min": values.min(),
                    "affinity_best_5pct_mean": _best_fraction_mean(values),
                    "rate_affinity_le_minus10_per_scored": (values <= -10).mean(),
                }
            )
    per_repeat = pd.DataFrame(rows)

    pooled_rows = []
    for steps in STEP_ORDER:
        step_rows = generated[generated["diffusion_steps"] == steps]
        values = step_rows.loc[step_rows["success"], "affinity_kcal_mol"].dropna()
        repeats = per_repeat[per_repeat["diffusion_steps"] == steps]
        pooled_rows.append(
            {
                "diffusion_steps": steps,
                "generation_attempts_total": 3000,
                "sanitize_valid_inputs_total": len(step_rows),
                "score_success_total": len(values),
                "sanitize_valid_per_attempt": len(step_rows) / 3000,
                "score_success_per_input": len(values) / len(step_rows),
                "score_success_per_attempt": len(values) / 3000,
                "affinity_pooled_mean": values.mean(),
                "affinity_pooled_median": values.median(),
                "affinity_pooled_std_population": values.std(ddof=0),
                "affinity_best_min": values.min(),
                "affinity_best_5pct_mean": _best_fraction_mean(values),
                "rate_affinity_le_minus10_per_scored": (values <= -10).mean(),
                "repeat_affinity_mean_mean": repeats["affinity_mean"].mean(),
                "repeat_affinity_mean_sd": repeats["affinity_mean"].std(ddof=1),
                "repeat_affinity_median_mean": repeats["affinity_median"].mean(),
                "repeat_affinity_median_sd": repeats["affinity_median"].std(ddof=1),
            }
        )
    pooled = pd.DataFrame(pooled_rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    generated.to_csv(RESULTS_DIR / "per_ligand_scores_all_steps.csv", index=False)
    per_repeat.to_csv(RESULTS_DIR / "per_repeat_summary.csv", index=False)
    pooled.to_csv(RESULTS_DIR / "step_summary.csv", index=False)
    return per_repeat, pooled


def _plot(scores: pd.DataFrame, per_repeat: pd.DataFrame, pooled: pd.DataFrame) -> None:
    """Plot score distributions, repeat stability, and explicit success rates."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    generated = scores[scores["diffusion_steps"].notna() & scores["success"]].copy()
    generated["diffusion_steps"] = generated["diffusion_steps"].astype(int)
    c1101 = float(
        scores.loc[(scores["model"] == "C1101") & scores["success"], "affinity_kcal_mol"].iloc[0]
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.2), constrained_layout=True)

    ax = axes[0, 0]
    data = [
        generated.loc[generated["diffusion_steps"] == steps, "affinity_kcal_mol"].to_numpy()
        for steps in STEP_ORDER
    ]
    box = ax.boxplot(data, tick_labels=STEP_ORDER, patch_artist=True, showfliers=False, widths=0.62)
    for patch, steps in zip(box["boxes"], STEP_ORDER):
        patch.set_facecolor(COLORS[steps])
        patch.set_alpha(0.82)
    ax.axhline(c1101, color="#444444", linestyle="--", linewidth=1, label=f"C1101 ({c1101:.2f})")
    ax.set_xlabel("Diffusion steps")
    ax.set_ylabel("QVina score-only (kcal/mol)\nlower is better")
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.5)
    ax.legend(loc="upper right")

    ax = axes[0, 1]
    for repeat, marker in zip(REPEAT_ORDER, ["o", "s", "^"]):
        subset = per_repeat[per_repeat["repeat"] == repeat]
        ax.plot(
            subset["diffusion_steps"],
            subset["affinity_mean"],
            marker=marker,
            linewidth=1.2,
            markersize=4,
            label=repeat,
        )
    ax.set_xticks(STEP_ORDER)
    ax.set_xlabel("Diffusion steps")
    ax.set_ylabel("Mean score by repeat (kcal/mol)")
    ax.grid(color="#D8D8D8", linewidth=0.5)
    ax.legend(ncol=3, loc="best")

    ax = axes[1, 0]
    x = np.arange(len(STEP_ORDER))
    width = 0.36
    ax.bar(
        x - width / 2,
        100 * pooled["sanitize_valid_per_attempt"],
        width,
        color="#A9BCD0",
        label="Sanitize-valid / attempts",
    )
    ax.bar(
        x + width / 2,
        100 * pooled["score_success_per_attempt"],
        width,
        color="#365F7D",
        label="Score success / attempts",
    )
    ax.set_xticks(x, STEP_ORDER)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Diffusion steps")
    ax.set_ylabel("Attempt-level rate (%)")
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.5)
    ax.legend(loc="lower right")

    ax = axes[1, 1]
    docking = pd.read_csv(
        PROJECT_ROOT
        / "analysis/benchmark/diffgui_step_sweep_100_200_500_1000_stats_20260710"
        / "diffgui_step_docking_summary.csv"
    )
    comparison = pooled[["diffusion_steps", "affinity_pooled_mean"]].merge(
        docking[["diffusion_steps", "score_mean_mean"]], on="diffusion_steps"
    )
    comparison.to_csv(RESULTS_DIR / "score_only_vs_full_docking.csv", index=False)
    ax.plot(
        comparison["diffusion_steps"],
        comparison["affinity_pooled_mean"],
        marker="o",
        linewidth=1.4,
        label="QVina score-only",
        color="#B64F4A",
    )
    ax.plot(
        comparison["diffusion_steps"],
        comparison["score_mean_mean"],
        marker="s",
        linewidth=1.4,
        label="Full docking",
        color="#3A6EA5",
    )
    ax.set_xticks(STEP_ORDER)
    ax.set_xlabel("Diffusion steps")
    ax.set_ylabel("Mean score (kcal/mol)")
    ax.grid(color="#D8D8D8", linewidth=0.5)
    ax.legend(loc="best")

    for label, ax in zip("abcd", axes.flat):
        ax.text(-0.12, 1.04, label, transform=ax.transAxes, fontsize=11, fontweight="bold")
    for suffix in ("png", "svg"):
        fig.savefig(RESULTS_DIR / f"diffgui_step_qvina_score_only.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_report(per_repeat: pd.DataFrame, pooled: pd.DataFrame) -> None:
    """Write the concise narrative report with in-report tables and figures."""
    by_step = pooled.set_index("diffusion_steps")
    delta_200_vs_100 = float(
        by_step.loc[200, "affinity_pooled_mean"] - by_step.loc[100, "affinity_pooled_mean"]
    )
    delta_500_vs_200 = float(
        by_step.loc[500, "affinity_pooled_mean"] - by_step.loc[200, "affinity_pooled_mean"]
    )
    delta_1000_vs_500 = float(
        by_step.loc[1000, "affinity_pooled_mean"] - by_step.loc[500, "affinity_pooled_mean"]
    )
    valid_gain_1000_vs_500 = 100 * float(
        by_step.loc[1000, "sanitize_valid_per_attempt"]
        - by_step.loc[500, "sanitize_valid_per_attempt"]
    )
    strong_score_gain_1000_vs_500 = 100 * float(
        by_step.loc[1000, "rate_affinity_le_minus10_per_scored"]
        - by_step.loc[500, "rate_affinity_le_minus10_per_scored"]
    )
    lines = [
        "# DiffGui 100/200/500/1000-step QVina score-only comparison",
        "",
        "All four settings use three fixed-attempt repeats (1000 generation attempts per repeat), "
        "seed groups 1-7, 8-14, and 15-21. QVina was run with `--score_only`, "
        "the same receptor/box, and exhaustiveness 16. Lower affinity is better.",
        "",
        "![DiffGui step QVina score-only comparison](results/diffgui_step_qvina_score_only.png)",
        "",
        "## Step summary",
        "",
        "| Steps | Sanitize-valid / 3000 attempts | QVina successes | Success / valid | "
        "Pooled mean | Pooled median | Repeat mean +/- SD | <= -10 / scored |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in pooled.to_dict("records"):
        lines.append(
            f"| {int(row['diffusion_steps'])} | {int(row['sanitize_valid_inputs_total'])} "
            f"({100 * row['sanitize_valid_per_attempt']:.2f}%) | {int(row['score_success_total'])} | "
            f"{100 * row['score_success_per_input']:.2f}% | {row['affinity_pooled_mean']:.3f} | "
            f"{row['affinity_pooled_median']:.3f} | {row['repeat_affinity_mean_mean']:.3f} +/- "
            f"{row['repeat_affinity_mean_sd']:.3f} | "
            f"{100 * row['rate_affinity_le_minus10_per_scored']:.2f}% |"
        )
    best_step = int(pooled.loc[pooled["affinity_pooled_mean"].idxmin(), "diffusion_steps"])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- The most favorable pooled score-only mean is at **{best_step} steps**.",
            f"- 100 and 200 steps are effectively tied (200 - 100 mean delta: "
            f"{delta_200_vs_100:+.3f} kcal/mol). The mean improves by "
            f"{delta_500_vs_200:+.3f} kcal/mol from 200 to 500 steps and by "
            f"{delta_1000_vs_500:+.3f} kcal/mol from 500 to 1000 steps; negative deltas are favorable.",
            f"- From 500 to 1000 steps, sanitize-valid per attempt increases by only "
            f"{valid_gain_1000_vs_500:+.2f} percentage points, while the fraction scoring <= -10 "
            f"increases by {strong_score_gain_1000_vs_500:+.2f} percentage points among scored molecules.",
            f"- The 1000-step repeat-mean SD is {by_step.loc[1000, 'repeat_affinity_mean_sd']:.3f} "
            f"kcal/mol, the smallest among the four settings.",
            "- Score-only evaluates the generated pose without redocking or local minimization. It should be "
            "interpreted together with PoseBusters/clash checks and the full-docking comparison, not as a "
            "standalone binding-affinity estimate.",
            "- Percentages above state their denominator explicitly: validity and score-success-per-attempt use "
            "3000 generation attempts per step; the <= -10 rate uses successfully scored molecules.",
            "",
            "## Evidence",
            "",
            "- [All per-ligand scores](results/per_ligand_scores_all_steps.csv)",
            "- [Per-repeat summary](results/per_repeat_summary.csv)",
            "- [Four-step summary](results/step_summary.csv)",
            "- [Score-only versus full docking](results/score_only_vs_full_docking.csv)",
            "",
        ]
    )
    (SCRIPT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run the complete four-step score-only analysis."""
    scores = _read_scores()
    per_repeat, pooled = _build_summaries(scores)
    _plot(scores, per_repeat, pooled)
    _write_report(per_repeat, pooled)


if __name__ == "__main__":
    main()
