"""Generate publication-quality core molecular property plots from a CSV.

Plots six properties:
  - MolWt  (histogram + KDE)
  - LogP   (violin + stripplot)
  - TPSA   (violin + stripplot)
  - RotBonds (count bar chart)
  - QED    (violin + stripplot)
  - SA_Score (violin + stripplot)

Outputs:
  - Single plots: {property}.png
  - Combined grid: core_properties_combined.png
  - Summary CSV: core_properties_summary.csv

Usage:
    conda run -n dif python scripts/plotting/core_properties.py \\
        --input per_molecule_metrics.csv \\
        --outdir evaluation/beautiful_plots/

The input CSV must contain columns matching the property names above.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

# ============================================================
# Global aesthetics
# ============================================================
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.edgecolor"] = "#d4d4d4"
plt.rcParams["axes.linewidth"] = 1.2

# ============================================================
# Tuneable defaults
# ============================================================
SINGLE_FIGSIZE = (10, 7.5)
COMBINED_FIGSIZE = (22, 14)
COMBINED_LAYOUT = (2, 3)
FIG_DPI = 300

COMBINED_TITLE = "Core Molecular Property Distributions"
COMBINED_TITLE_FONTSIZE = 26
PANEL_TITLE_FONTSIZE = 16
PANEL_TITLE_WEIGHT = "bold"
AXIS_LABEL_FONTSIZE = 14
TICK_LABELSIZE = 12

THRESHOLD_LINE_COLOR = "#E74C3C"
THRESHOLD_LINE_WIDTH = 2.0
MOLWT_BIN_WIDTH = 40

CRITERIA = {
    "MolWt":    {"type": "lt",    "value": 500, "low": 0,   "high": 0,  "label": "< 500"},
    "LogP":     {"type": "range", "value": 0,   "low": 0,   "high": 3,  "label": "0 - 3"},
    "TPSA":     {"type": "lt",    "value": 120, "low": 0,   "high": 0,  "label": "< 120"},
    "RotBonds": {"type": "le",    "value": 10,  "low": 0,   "high": 0,  "label": "\u2264 10"},
    "QED":      {"type": "gt",    "value": 0.5, "low": 0,   "high": 0,  "label": "> 0.5"},
    "SA_Score": {"type": "lt",    "value": 4,   "low": 0,   "high": 0,  "label": "< 4"},
}

PLOT_ORDER = ["MolWt", "LogP", "TPSA", "RotBonds", "QED", "SA_Score"]

PLOT_CONFIGS = {
    "MolWt": {
        "title": "Molecular Weight", "xlabel": "MolWt (Da)",
        "ylabel": "Count / Density", "kind": "hist", "color": "#4A90E2",
    },
    "LogP": {
        "title": "LogP (Lipophilicity)", "xlabel": "",
        "ylabel": "LogP", "kind": "violin", "color": "#F39C12",
        "ylim": (-2, 8),
    },
    "TPSA": {
        "title": "Topological Polar Surface Area", "xlabel": "",
        "ylabel": "TPSA (\u00c5\u00b2)", "kind": "violin", "color": "#2ECC71",
        "ylim": (0, 300),
    },
    "RotBonds": {
        "title": "Rotatable Bonds", "xlabel": "Number of RotBonds",
        "ylabel": "Count", "kind": "count", "color": "#9B59B6",
    },
    "QED": {
        "title": "Quantitative Estimate of Druglikeness", "xlabel": "",
        "ylabel": "QED Score", "kind": "violin", "color": "#E74C3C",
        "ylim": (0, 1),
    },
    "SA_Score": {
        "title": "Synthetic Accessibility Score", "xlabel": "",
        "ylabel": "SA Score", "kind": "violin", "color": "#1ABC9C",
        "ylim": (1, 10),
    },
}


# ------------------------------------------------------------------
# Axis styling
# ------------------------------------------------------------------
def apply_common_axis_style(ax) -> None:
    ax.tick_params(axis="both", labelsize=TICK_LABELSIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")


def add_threshold_lines(ax, column: str, orientation: str) -> None:
    crit = CRITERIA[column]
    if crit["type"] in {"lt", "le", "gt"}:
        values = [crit["value"]]
    else:
        values = [crit["low"], crit["high"]]

    for value in values:
        if orientation == "vertical":
            ax.axvline(
                value, color=THRESHOLD_LINE_COLOR, linestyle="--",
                linewidth=THRESHOLD_LINE_WIDTH, alpha=0.85, zorder=4,
            )
            y_pos = ax.get_ylim()[1] * 0.95
            ax.text(
                value + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.015,
                y_pos, crit["label"],
                color=THRESHOLD_LINE_COLOR, fontweight="bold", fontsize=11,
                ha="left", va="top", zorder=5,
            )
        else:
            ax.axhline(
                value, color=THRESHOLD_LINE_COLOR, linestyle="--",
                linewidth=THRESHOLD_LINE_WIDTH, alpha=0.85, zorder=4,
            )
            x_pos = ax.get_xlim()[1] * 0.95
            ax.text(
                x_pos,
                value + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02,
                crit["label"],
                color=THRESHOLD_LINE_COLOR, fontweight="bold", fontsize=11,
                ha="right", va="bottom", zorder=5,
            )


# ------------------------------------------------------------------
# Plot functions
# ------------------------------------------------------------------
def plot_hist(ax, series: pd.Series, column: str, cfg: dict) -> None:
    values = series.dropna().astype(float)
    vmin = np.floor(values.min() / MOLWT_BIN_WIDTH) * MOLWT_BIN_WIDTH
    vmax = np.ceil(values.max() / MOLWT_BIN_WIDTH) * MOLWT_BIN_WIDTH + MOLWT_BIN_WIDTH
    bins = np.arange(vmin, vmax + 1e-9, MOLWT_BIN_WIDTH)
    sns.histplot(
        values, bins=bins, kde=True, color=cfg["color"], ax=ax,
        edgecolor="white", alpha=0.7, linewidth=1.2,
    )
    add_threshold_lines(ax, column, orientation="vertical")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.set_title(cfg["title"], fontsize=PANEL_TITLE_FONTSIZE, fontweight=PANEL_TITLE_WEIGHT, pad=12)
    ax.set_xlabel(cfg["xlabel"], fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax.set_ylabel(cfg["ylabel"], fontsize=AXIS_LABEL_FONTSIZE)
    apply_common_axis_style(ax)


def plot_count(ax, series: pd.Series, column: str, cfg: dict) -> None:
    values = series.dropna().astype(int)
    counts = values.value_counts().sort_index()
    x_numeric = counts.index.to_numpy(dtype=float)
    palette = sns.color_palette("mako", n_colors=len(counts))
    ax.bar(
        x_numeric, counts.values, width=0.8, color=palette,
        edgecolor="#333", linewidth=0.8, alpha=0.9,
    )
    add_threshold_lines(ax, column, orientation="vertical")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    ax.set_xlim(x_numeric.min() - 0.8, x_numeric.max() + 0.8)
    ax.set_title(cfg["title"], fontsize=PANEL_TITLE_FONTSIZE, fontweight=PANEL_TITLE_WEIGHT, pad=12)
    ax.set_xlabel(cfg["xlabel"], fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax.set_ylabel(cfg["ylabel"], fontsize=AXIS_LABEL_FONTSIZE)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    apply_common_axis_style(ax)


def plot_violin(ax, series: pd.Series, column: str, cfg: dict) -> None:
    values = series.dropna().astype(float)
    sns.violinplot(
        y=values, ax=ax, color=cfg["color"], inner="quartile",
        linewidth=1.5, alpha=0.6, width=0.6,
    )
    sns.stripplot(
        y=values, ax=ax, color="#444444", alpha=0.15, size=2.5,
        jitter=0.2, zorder=0,
    )
    add_threshold_lines(ax, column, orientation="horizontal")
    ax.set_title(cfg["title"], fontsize=PANEL_TITLE_FONTSIZE, fontweight=PANEL_TITLE_WEIGHT, pad=12)
    ax.set_xlabel(cfg["xlabel"], fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(cfg["ylabel"], fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax.set_xticks([])
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    if "ylim" in cfg:
        ax.set_ylim(cfg["ylim"])
    apply_common_axis_style(ax)


def draw_one(ax, df: pd.DataFrame, column: str) -> None:
    cfg = PLOT_CONFIGS[column]
    if cfg["kind"] == "hist":
        plot_hist(ax, df[column], column, cfg)
    elif cfg["kind"] == "count":
        plot_count(ax, df[column], column, cfg)
    else:
        plot_violin(ax, df[column], column, cfg)


# ------------------------------------------------------------------
# Output
# ------------------------------------------------------------------
def save_single_plots(df: pd.DataFrame, outdir: Path) -> None:
    for column in PLOT_ORDER:
        if column not in df.columns:
            print(f"  Skip {column}: not in CSV columns")
            continue
        fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE, dpi=FIG_DPI)
        draw_one(ax, df, column)
        fig.tight_layout(pad=1.5)
        fig.savefig(outdir / f"{column}.png", bbox_inches="tight", facecolor="white")
        plt.close(fig)


def save_combined_plot(df: pd.DataFrame, outdir: Path) -> None:
    nrows, ncols = COMBINED_LAYOUT
    fig, axes = plt.subplots(nrows, ncols, figsize=COMBINED_FIGSIZE, dpi=FIG_DPI)
    axes = np.array(axes).flatten()
    for ax, column in zip(axes, PLOT_ORDER):
        if column in df.columns:
            draw_one(ax, df, column)
        else:
            ax.set_visible(False)
    fig.suptitle(COMBINED_TITLE, fontsize=COMBINED_TITLE_FONTSIZE, fontweight="bold", y=1.02)
    fig.tight_layout(pad=2.0)
    fig.savefig(outdir / "core_properties_combined.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def evaluate_criterion(series: pd.Series, column: str) -> pd.Series:
    crit = CRITERIA[column]
    values = pd.to_numeric(series, errors="coerce")
    if crit["type"] == "lt":
        return values < crit["value"]
    if crit["type"] == "le":
        return values <= crit["value"]
    if crit["type"] == "gt":
        return values > crit["value"]
    return (values >= crit["low"]) & (values <= crit["high"])


def save_summary_table(df: pd.DataFrame, outdir: Path) -> None:
    rows: list[dict] = []
    all_pass_mask = pd.Series(True, index=df.index)
    for column in PLOT_ORDER:
        if column not in df.columns:
            continue
        series = pd.to_numeric(df[column], errors="coerce")
        valid_mask = series.notna()
        valid_values = series[valid_mask]
        pass_mask = evaluate_criterion(series, column).fillna(False)
        all_pass_mask &= pass_mask
        total_count = int(valid_mask.sum())
        pass_count = int(pass_mask.sum())
        pass_ratio = 0.0 if total_count == 0 else pass_count / total_count
        rows.append({
            "性质名称": column,
            "判别标准": CRITERIA[column]["label"],
            "中位数": round(float(valid_values.median()), 4),
            "最大值": round(float(valid_values.max()), 4),
            "最小值": round(float(valid_values.min()), 4),
            "平均值": round(float(valid_values.mean()), 4),
            "通过分子数": pass_count,
            "通过分子比例": round(pass_ratio, 4),
            "总分子数": total_count,
        })
    rows.append({
        "性质名称": "ALL_CRITERIA",
        "判别标准": "同时满足全部核心标准",
        "中位数": "",
        "最大值": "", "最小值": "", "平均值": "",
        "通过分子数": int(all_pass_mask.sum()),
        "通过分子比例": round(float(all_pass_mask.mean()), 4),
        "总分子数": int(len(df)),
    })
    pd.DataFrame(rows).to_csv(
        outdir / "core_properties_summary.csv", index=False, encoding="utf-8-sig",
    )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate core molecular property plots from a CSV."
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="Input property CSV (must contain MolWt, LogP, TPSA, RotBonds, QED, SA_Score)")
    parser.add_argument("--outdir", type=Path, required=True,
                        help="Output directory for plots and summary CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    print("Generating single property plots...")
    save_single_plots(df, outdir)
    print("Generating combined plot...")
    save_combined_plot(df, outdir)
    print("Writing summary table...")
    save_summary_table(df, outdir)
    print(f"Done → {outdir}")


if __name__ == "__main__":
    main()
