#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对 6 个核心分子性质作图，并导出核心性质汇总表。

与 plot_core_properties.py 的区别：
- 将 guacamol.py 归一化后的 SA_Score 反变换回原始值 (1~10)
- 字体整体放大，适合论文/答辩展示

核心性质：
- MolWt
- LogP
- TPSA
- RotBonds
- QED
- SA_Score (原始值)
"""

from __future__ import annotations

import argparse
import colorsys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


# ============================================================
# 用户可调参数
# ============================================================
DEFAULT_INPUT = "/home/md666/data/disk6/DiffB110/runs/260407_generate_ligands_10000/guacamol_eval_260407generated_10000-2_first1000/260407generated_10000-2_first1000_properties.csv"
DEFAULT_OUTDIR = "/home/md666/data/disk6/DiffB110/bishe/RE-out/analysis"
SUMMARY_CSV_NAME = None  # None 时自动按输出目录名生成

SINGLE_FIGSIZE = (10, 7.5)
COMBINED_FIGSIZE = (24, 16)
COMBINED_LAYOUT = (2, 3)
FIG_DPI = 180

COMBINED_TITLE = "Core Property Distributions"
COMBINED_TITLE_FONTSIZE = 0
PANEL_TITLE_FONTSIZE = 32
PANEL_TITLE_WEIGHT = "bold"
AXIS_LABEL_FONTSIZE = 32
TICK_LABELSIZE = 32

GRID_ALPHA = 0.22
HIST_ALPHA = 0.90
BAR_ALPHA = 0.92
VIOLIN_ALPHA = 0.58
SCATTER_ALPHA = 0.25
SCATTER_SIZE = 20
SCATTER_COLOR = "#666666"
SCATTER_JITTER_SCALE = 0.025
SCATTER_RANDOM_SEED = 42

HIST_EDGE_COLOR = "#2E2E2E"
BAR_EDGE_COLOR = "#2E2E2E"
VIOLIN_EDGE_COLOR = "#444444"
MEDIAN_LINE_COLOR = "#2C7A4B"
THRESHOLD_LINE_COLOR = "#A23B3B"
SUMMARY_LINE_WIDTH = 2.0
THRESHOLD_LINE_WIDTH = 2.0
VIOLIN_LINE_WIDTH = 1.2

ROTBONDS_MAX_YTICKS = 7
ROTBONDS_MAX_XTICKS = 10
MOLWT_BIN_WIDTH = 40

CRITERIA = {
    "MolWt": {"type": "lt", "value": 500, "label": "< 500"},
    "LogP": {"type": "range", "low": 0, "high": 3, "label": "0 - 3"},
    "TPSA": {"type": "lt", "value": 120, "label": "< 120"},
    "RotBonds": {"type": "le", "value": 10, "label": "≤ 10"},
    "QED": {"type": "gt", "value": 0.5, "label": "> 0.5"},
    "SA_Score": {"type": "lt", "value": 4, "label": "< 4"},
}

PLOT_ORDER = ["MolWt", "LogP", "TPSA", "RotBonds", "QED", "SA_Score"]

PLOT_CONFIGS = {
    "MolWt": {
        "title": "Molecular Weight",
        "xlabel": "MolWt",
        "ylabel": "Count",
        "kind": "hist",
        "color": "#4C78A8",
    },
    "LogP": {
        "title": "LogP",
        "xlabel": "",
        "ylabel": "",
        "kind": "violin",
        "color": "#F58518",
        "ylim": (-4, 10),
    },
    "TPSA": {
        "title": "TPSA",
        "xlabel": "",
        "ylabel": "",
        "kind": "violin",
        "color": "#54A24B",
        "ylim": (0, 300),
    },
    "RotBonds": {
        "title": "Rotatable Bonds",
        "xlabel": "RotBonds",
        "ylabel": "Count",
        "kind": "count",
        "color": "#B279A2",
    },
    "QED": {
        "title": "QED",
        "xlabel": "",
        "ylabel": "",
        "kind": "violin",
        "color": "#E45756",
        "ylim": (0, 1),
    },
    "SA_Score": {
        "title": "SA Score",
        "xlabel": "",
        "ylabel": "",
        "kind": "violin",
        "color": "#72B7B2",
        "ylim": (1, 10),
    },
}


def denormalize_sa_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    guacamol.py 中 SA_Score 被归一化为 (10 - raw) / 9，范围 0~1。
    这里反变换回原始值：raw = 10 - 9 * normalized
    """
    if "SA_Score" in df.columns:
        df = df.copy()
        normalized = pd.to_numeric(df["SA_Score"], errors="coerce")
        # 只在值明显处于归一化范围 (0~1) 时才反变换
        mask = normalized.notna() & (normalized >= 0) & (normalized <= 1)
        df.loc[mask, "SA_Score"] = 10 - 9 * normalized[mask]
    return df


def lighten_color(color: str, amount: float = 0.25) -> tuple[float, float, float]:
    import matplotlib.colors as mc

    r, g, b = mc.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = min(1.0, l + amount * (1 - l))
    return colorsys.hls_to_rgb(h, l, s)


def apply_common_axis_style(ax) -> None:
    ax.tick_params(axis="both", labelsize=TICK_LABELSIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_threshold_lines(ax, column: str, orientation: str) -> None:
    crit = CRITERIA[column]
    if crit["type"] in {"lt", "le", "gt"}:
        values = [crit["value"]]
    else:
        values = [crit["low"], crit["high"]]

    for value in values:
        if orientation == "vertical_line":
            ax.axvline(value, color=THRESHOLD_LINE_COLOR, linestyle="--", linewidth=THRESHOLD_LINE_WIDTH, alpha=0.9)
        else:
            ax.axhline(value, color=THRESHOLD_LINE_COLOR, linestyle="--", linewidth=THRESHOLD_LINE_WIDTH, alpha=0.9)


def plot_hist(ax, series: pd.Series, column: str, cfg: dict) -> None:
    values = series.dropna().astype(float)
    color = lighten_color(cfg["color"], 0.18)
    vmin = np.floor(values.min() / MOLWT_BIN_WIDTH) * MOLWT_BIN_WIDTH
    vmax = np.ceil(values.max() / MOLWT_BIN_WIDTH) * MOLWT_BIN_WIDTH + MOLWT_BIN_WIDTH
    bins = np.arange(vmin, vmax + 1e-9, MOLWT_BIN_WIDTH)

    ax.hist(values, bins=bins, color=color, edgecolor=HIST_EDGE_COLOR, alpha=HIST_ALPHA)
    add_threshold_lines(ax, column, orientation="vertical_line")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))

    ax.set_title(cfg["title"], fontsize=PANEL_TITLE_FONTSIZE, fontweight=PANEL_TITLE_WEIGHT)
    ax.set_xlabel(cfg["xlabel"], fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(cfg["ylabel"], fontsize=AXIS_LABEL_FONTSIZE)
    ax.grid(axis="y", alpha=GRID_ALPHA)
    apply_common_axis_style(ax)


def plot_count(ax, series: pd.Series, column: str, cfg: dict) -> None:
    values = series.dropna().astype(int)
    counts = values.value_counts().sort_index()
    color = lighten_color(cfg["color"], 0.22)
    x_numeric = counts.index.to_numpy(dtype=float)

    ax.bar(x_numeric, counts.values, width=0.82, color=color, edgecolor=BAR_EDGE_COLOR, alpha=BAR_ALPHA)

    threshold = CRITERIA[column]["value"]
    ax.axvline(threshold, color=THRESHOLD_LINE_COLOR, linestyle="--", linewidth=THRESHOLD_LINE_WIDTH, alpha=0.9)

    ax.yaxis.set_major_locator(MaxNLocator(nbins=ROTBONDS_MAX_YTICKS, integer=True, prune=None))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=ROTBONDS_MAX_XTICKS, integer=True, prune=None))
    ax.set_xlim(x_numeric.min() - 0.8, x_numeric.max() + 0.8)

    ax.set_title(cfg["title"], fontsize=PANEL_TITLE_FONTSIZE, fontweight=PANEL_TITLE_WEIGHT)
    ax.set_xlabel(cfg["xlabel"], fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(cfg["ylabel"], fontsize=AXIS_LABEL_FONTSIZE)
    ax.grid(axis="y", alpha=GRID_ALPHA)
    apply_common_axis_style(ax)


def plot_violin(ax, series: pd.Series, column: str, cfg: dict) -> None:
    values = series.dropna().astype(float).to_numpy()
    color = lighten_color(cfg["color"], 0.24)

    parts = ax.violinplot(values, showmeans=False, showmedians=True, showextrema=False, vert=True, points=300)
    for body in parts["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor(VIOLIN_EDGE_COLOR)
        body.set_alpha(VIOLIN_ALPHA)
    if "cmedians" in parts:
        parts["cmedians"].set_color(MEDIAN_LINE_COLOR)
        parts["cmedians"].set_linewidth(VIOLIN_LINE_WIDTH + 0.4)

    rng = np.random.default_rng(SCATTER_RANDOM_SEED)
    x = rng.normal(loc=1.0, scale=SCATTER_JITTER_SCALE, size=len(values))
    ax.scatter(x, values, s=SCATTER_SIZE, alpha=SCATTER_ALPHA, color=SCATTER_COLOR, linewidths=0)

    add_threshold_lines(ax, column, orientation="horizontal_line")

    ax.set_title(cfg["title"], fontsize=PANEL_TITLE_FONTSIZE, fontweight=PANEL_TITLE_WEIGHT)
    ax.set_xlabel(cfg["xlabel"], fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(cfg["ylabel"], fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xticks([])
    ax.grid(axis="y", alpha=GRID_ALPHA)
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


def save_single_plots(df: pd.DataFrame, outdir: Path) -> None:
    for column in PLOT_ORDER:
        fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE, dpi=FIG_DPI)
        draw_one(ax, df, column)
        fig.tight_layout(pad=1.4)
        fig.savefig(outdir / f"{column}.png", bbox_inches="tight")
        plt.close(fig)


def save_combined_plot(df: pd.DataFrame, outdir: Path) -> None:
    nrows, ncols = COMBINED_LAYOUT
    fig, axes = plt.subplots(nrows, ncols, figsize=COMBINED_FIGSIZE, dpi=FIG_DPI)
    axes = np.array(axes).flatten()

    for ax, column in zip(axes, PLOT_ORDER):
        draw_one(ax, df, column)

    fig.suptitle(COMBINED_TITLE, fontsize=COMBINED_TITLE_FONTSIZE, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97), pad=1.6)
    fig.savefig(outdir / "core_properties_combined.png", bbox_inches="tight")
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


def default_summary_csv_name(outdir: Path) -> str:
    return f"core_properties_{outdir.name}.csv"


def save_summary_table(df: pd.DataFrame, outdir: Path) -> None:
    rows: list[dict] = []
    all_pass_mask = pd.Series(True, index=df.index)

    for column in PLOT_ORDER:
        series = pd.to_numeric(df[column], errors="coerce")
        valid_mask = series.notna()
        valid_values = series[valid_mask]
        pass_mask = evaluate_criterion(series, column).fillna(False)
        all_pass_mask &= pass_mask

        total_count = int(valid_mask.sum())
        pass_count = int(pass_mask.sum())
        pass_ratio = 0.0 if total_count == 0 else pass_count / total_count

        rows.append(
            {
                "性质名称": column,
                "判别标准": CRITERIA[column]["label"],
                "中位数": round(float(valid_values.median()), 4),
                "最大值": round(float(valid_values.max()), 4),
                "最小值": round(float(valid_values.min()), 4),
                "平均值": round(float(valid_values.mean()), 4),
                "通过分子数": pass_count,
                "通过分子比例": round(pass_ratio, 4),
                "总分子数": total_count,
            }
        )

    rows.append(
        {
            "性质名称": "ALL_CRITERIA",
            "判别标准": "同时满足全部核心标准",
            "中位数": "",
            "最大值": "",
            "最小值": "",
            "平均值": "",
            "通过分子数": int(all_pass_mask.sum()),
            "通过分子比例": round(float(all_pass_mask.mean()), 4),
            "总分子数": int(len(df)),
        }
    )

    summary_name = SUMMARY_CSV_NAME or default_summary_csv_name(outdir)
    pd.DataFrame(rows).to_csv(outdir / summary_name, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input property CSV path")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Output directory")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    save_single_plots(df, outdir)
    save_combined_plot(df, outdir)
    save_summary_table(df, outdir)
    print(f"Done. Plots saved to: {outdir}")


if __name__ == "__main__":
    main()
