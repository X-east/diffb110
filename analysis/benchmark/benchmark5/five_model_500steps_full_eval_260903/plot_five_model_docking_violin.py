"""Rebuild the five-model docking-score violin plot on the Table 2.1 batch.

Why this script exists
----------------------
An earlier violin plot (analysis/benchmark/model_selection_why_diffgui_20260625/
docking_score_violin.png) was drawn from a single 1000-molecule batch
(8gjl_c1101_benchmark_20260624_1000mol_500steps) using a sanitize-valid
denominator (DiffGui 879 / DiffSBDD 666 / PMDM 548 / GCDM-SBDD 533 /
TargetDiff 438). Thesis Table 2.1 and Table 3.2 instead report the formal
3-repeat protocol (rep4 / rep5 / rep6, 1000 attempts each) with n_valid
2604 / 2007 / 1489 / 1557 / 1361. The two are different batches at different
scales, so the old figure cannot sit next to Table 2.1.

This script rebuilds the figure from the per-molecule docking results of
exactly the Table 2.1 batch, so that figure and table share one data source
in the same way fig3_1_model_pathology.png shares model_r1r5_rates_500steps_3rep.csv.

Non-physical scores
-------------------
Of the 8017 ligands in this batch, two carry a positive score, which for a
binding free energy means a pose-level clash rather than a real binding mode:

    rep5/DiffSBDD/diffsbdd_000104    = +345.218 kcal/mol
    rep4/GCDM-SBDD/gcdm-sbdd_000108  =  +12.865 kcal/mol

They are excluded from the plotted distributions (they carry no binding
information) but are still counted by the consistency check, so the script
still proves that the batch reproduces Table 2.1 exactly before dropping them.
Table 2.1 itself needs no correction: it reports robust mean/median statistics
over the full 8017 ligands.

Input : docking/rep{4,5,6}/{model}/docking_results.csv
Output: five_model_500steps_docking_violin.png / .svg  (this directory)

Environment: needs seaborn >= 0.13 (use the `dif_umap_compat` conda env here)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# (no axis-external text is drawn any more; keep imports minimal)
import seaborn as sns

HERE = Path(__file__).resolve().parent
DOCKING_DIR = HERE / "docking"
REPEATS = ("rep4", "rep5", "rep6")

# Display order follows thesis Table 2.1.
MODEL_ORDER = ["DiffGui", "DiffSBDD", "GCDM-SBDD", "PMDM", "TargetDiff"]

# On-disk directory name for each model.
MODEL_DIR = {
    "DiffGui": "diffgui",
    "DiffSBDD": "diffsbdd",
    "GCDM-SBDD": "gcdm_sbdd",
    "PMDM": "pmdm",
    "TargetDiff": "targetdiff",
}

# thesis Table 2.1 n_valid, used as a hard consistency assertion
EXPECTED_N = {
    "DiffGui": 2604,
    "DiffSBDD": 2007,
    "GCDM-SBDD": 1557,
    "PMDM": 1489,
    "TargetDiff": 1361,
}

# A positive docking score means the pose is rejected by the scoring function,
# not a weak binder, so such ligands are dropped from the plotted distribution.
NON_PHYSICAL_MAX = 0.0

# The high-affinity reference threshold used across all benchmark tables.
SCORE_THRESHOLD = -10.0

PALETTE = {
    "DiffGui": "#c0392b",
    "DiffSBDD": "#2e86c1",
    "GCDM-SBDD": "#7f8c8d",
    "PMDM": "#e67e22",
    "TargetDiff": "#27ae60",
}

OUT_STEM = "five_model_500steps_docking_violin"


def load_docking_scores() -> pd.DataFrame:
    """Load per-molecule UniDock scores for all three repeats and five models."""
    frames: list[pd.DataFrame] = []
    for repeat in REPEATS:
        for model in MODEL_ORDER:
            path = DOCKING_DIR / repeat / MODEL_DIR[model] / "docking_results.csv"
            if not path.is_file():
                raise FileNotFoundError(path)
            df = pd.read_csv(path)
            keep = df["success"].astype(bool) & df["score"].notna()
            frames.append(
                pd.DataFrame(
                    {
                        "model": model,
                        "repeat": repeat,
                        "score": df.loc[keep, "score"].astype(float),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def verify_counts(data: pd.DataFrame) -> None:
    """Fail loudly if the batch does not reproduce thesis Table 2.1."""
    counts = data.groupby("model")["score"].size()
    print("Consistency check against thesis Table 2.1 (n_valid, before exclusion):")
    bad = []
    for model in MODEL_ORDER:
        got = int(counts.get(model, 0))
        want = EXPECTED_N[model]
        flag = "OK" if got == want else "MISMATCH"
        if got != want:
            bad.append(model)
        print(f"  {model:<12s} {got:>6d}  (table: {want:>6d})  -> {flag}")
    if bad:
        raise SystemExit(f"Refusing to plot: n_valid mismatch for {bad}")
    print(f"  total      {len(data):>6d}")


def drop_non_physical(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split off ligands whose score is not a binding free energy."""
    mask = data["score"] > NON_PHYSICAL_MAX
    return data.loc[~mask].copy(), data.loc[mask].copy()


def load_validity() -> dict[str, float]:
    """Read the per-model generation validity (valid molecules / attempts), %.

    source: model_selection_summary.csv, column valid_per_attempt_mean,
    the same quantity reported in thesis Table 2.1 (生成有效率).
    """
    path = HERE / "model_selection_summary.csv"
    df = pd.read_csv(path).set_index("model")
    rates: dict[str, float] = {}
    for model in MODEL_ORDER:
        rates[model] = float(df.loc[model, "valid_per_attempt_mean"]) * 100.0
    print("\nGeneration validity (valid_per_attempt_mean, %):")
    for model in MODEL_ORDER:
        print(f"  {model:<12s} {rates[model]:.2f}%")
    return rates


def style_axes(ax: plt.Axes) -> None:
    """Hide top/right spines and soften the remaining ones."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")


def main() -> None:
    raw = load_docking_scores()
    verify_counts(raw)

    data, dropped = drop_non_physical(raw)
    print(f"\nExcluded {len(dropped)} ligand(s) with positive (non-physical) scores:")
    for row in dropped.sort_values("score", ascending=False).itertuples():
        print(f"  {row.repeat}/{row.model:<11s} {row.score:+.3f} kcal/mol")

    # cross-check only: prints the 有效率 column of the companion summary table
    load_validity()

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.edgecolor"] = "#d4d4d4"
    plt.rcParams["axes.linewidth"] = 1.2

    fig, ax = plt.subplots(figsize=(9.6, 5.9), dpi=300)

    sns.violinplot(
        data=data,
        x="model",
        y="score",
        order=MODEL_ORDER,
        hue="model",
        hue_order=MODEL_ORDER,
        palette=PALETTE,
        inner=None,
        cut=0,
        linewidth=0.9,
        saturation=1.0,
        width=0.82,
        legend=False,
        ax=ax,
    )

    # only the mean is marked, drawn inside the violin body
    for i, model in enumerate(MODEL_ORDER):
        vals = data.loc[data["model"] == model, "score"].to_numpy()
        mean = float(np.mean(vals))
        ax.hlines(mean, i - 0.30, i + 0.30, color="#2c3e50", linewidth=3.2, zorder=4)
        ax.scatter(
            [i], [mean], s=54, color="white", edgecolor="#2c3e50",
            linewidth=1.8, zorder=5,
        )
        ax.text(
            i, mean + 0.20, f"{mean:.2f}", ha="center", va="bottom",
            fontsize=13, fontweight="bold", color="#2c3e50", zorder=6,
            bbox=dict(facecolor="white", alpha=0.80, edgecolor="none", pad=1.4),
        )

    # the x tick labels carry the model name and its generation validity
    # model names on the tick axis, large and bold
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels(MODEL_ORDER, fontsize=19.0, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("UniDock score (kcal/mol)", fontsize=16)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_ylim(float(data["score"].min()) - 0.6, float(data["score"].max()) + 0.85)
    ax.set_xlim(-0.62, len(MODEL_ORDER) - 1 + 0.72)
    style_axes(ax)

    # the two excluded ligands are reported on stdout by drop_non_physical()
    fig.subplots_adjust(left=0.095, right=0.985, top=0.975, bottom=0.145)

    png = HERE / f"{OUT_STEM}.png"
    svg = HERE / f"{OUT_STEM}.svg"
    fig.savefig(png, dpi=300)
    fig.savefig(svg)
    plt.close(fig)
    print(f"\nWrote {png}")
    print(f"Wrote {svg}")


if __name__ == "__main__":
    main()
