#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对全部性质自动选择合适图形并逐项出图 (美化版)。

输入：analyze_valid_sdf_properties.py 导出的分子性质 CSV
输出：
- 每个性质单独一张图
- 一个综合汇总图
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

# 全局美学设置
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#e0e0e0'
plt.rcParams['axes.linewidth'] = 1.2

DEFAULT_INPUT = "/home/md666/data/disk6/DiffB110/runs/260407_generate_ligands_10000/260407generated_10000_filtered_properties_all.csv"
DEFAULT_OUTDIR = "/home/md666/data/disk6/DiffB110/runs/260407_generate_ligands_10000/260410-10000-generate1-5829mol/all_properties_plots"
FIG_DPI = 300  # 提高DPI使图像更清晰
SINGLE_FIGSIZE = (10, 7)
TITLE_SIZE = 16
LABEL_SIZE = 13
RNG_SEED = 123

CATEGORICAL_COLUMNS = ["Murcko_Scaffold"]
BINARY_COLUMNS = ["Lipinski_Pass"]
COUNT_LIKE_COLUMNS = [
    "HBA", "HBD", "RotBonds", "HeavyAtomCount", 
    "NumRings", "NumAromaticRings", "FormalCharge"
]
CONTINUOUS_COLUMNS = [
    "MolWt", "ExactMolWt", "LogP", "TPSA", 
    "FractionCSP3", "QED", "SA_Score"
]

def apply_style(ax) -> None:
    """去除顶部和右侧的边框线，使图表更清爽"""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.grid(axis='y', linestyle='--', alpha=0.6)

def infer_kind(column: str) -> str:
    if column in CATEGORICAL_COLUMNS: return "top_categories"
    if column in BINARY_COLUMNS:      return "binary_donut"
    if column in COUNT_LIKE_COLUMNS:  return "count_bar"
    if column in CONTINUOUS_COLUMNS:  return "dist_violin"
    return "dist_violin"

def plot_dist_violin(ax_hist, ax_violin, values: pd.Series, title: str) -> None:
    """连续变量：KDE直方图 + 小提琴图"""
    arr = pd.to_numeric(values, errors="coerce").dropna()
    
    # 顶部：直方图 + KDE 密度曲线
    sns.histplot(arr, kde=True, ax=ax_hist, color="#4C9181", 
                 edgecolor="white", alpha=0.7, linewidth=1.5)
    ax_hist.axvline(arr.median(), color="#E67E22", linestyle="--", linewidth=2, label=f'Median: {arr.median():.2f}')
    ax_hist.set_title(title, fontsize=TITLE_SIZE, fontweight="bold", pad=15)
    ax_hist.set_ylabel("Count / Density", fontsize=LABEL_SIZE)
    ax_hist.set_xlabel("")
    ax_hist.legend(loc='upper right', frameon=True, shadow=True)
    apply_style(ax_hist)

    # 底部：水平小提琴图 (展示分布形态及四分位数)
    sns.violinplot(x=arr, ax=ax_violin, color="#85C1E9", inner="quartile", linewidth=1.2)
    ax_violin.set_xlabel(title, fontsize=LABEL_SIZE, fontweight="bold")
    ax_violin.set_ylabel("")
    apply_style(ax_violin)
    ax_violin.grid(axis='x', linestyle='--', alpha=0.6)
    ax_violin.grid(axis='y', visible=False)

def plot_count_bar(ax, values: pd.Series, title: str) -> None:
    """计数变量：渐变色条形图"""
    counts = pd.to_numeric(values, errors="coerce").dropna().astype(int).value_counts().sort_index()
    
    # 使用渐变色板
    palette = sns.color_palette("crest", n_colors=len(counts))
    sns.barplot(x=counts.index, y=counts.values, ax=ax, palette=palette, edgecolor="#333", linewidth=0.8)
    
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold", pad=15)
    ax.set_xlabel(title, fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_ylabel("Count", fontsize=LABEL_SIZE)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
    apply_style(ax)

def plot_binary_donut(ax, values: pd.Series, title: str) -> None:
    """二元变量：高级圆环图 (Donut Chart)"""
    counts = pd.to_numeric(values, errors="coerce").dropna().astype(int).value_counts().sort_index()
    counts = counts.reindex([0, 1], fill_value=0)
    
    labels = ["Fail (0)", "Pass (1)"]
    colors = ["#E74C3C", "#2ECC71"] # 红色代表Fail，绿色代表Pass
    explode = (0.05, 0.05)
    
    ax.pie(counts.values, labels=labels, colors=colors, autopct='%1.1f%%', 
           startangle=140, pctdistance=0.85, explode=explode,
           wedgeprops=dict(width=0.4, edgecolor='w', linewidth=2),
           textprops={'fontsize': LABEL_SIZE, 'fontweight': 'bold'})
    
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold", pad=20)
    # 圆环图不需要坐标轴网格
    ax.axis('equal') 

def plot_top_categories(ax, values: pd.Series, title: str, top_n: int = 15) -> None:
    """分类变量：水平条形图"""
    counts = values.fillna("<NA>").astype(str).value_counts().head(top_n)
    
    palette = sns.color_palette("mako", n_colors=len(counts))
    sns.barplot(x=counts.values, y=counts.index, ax=ax, palette=palette, edgecolor="#333", linewidth=0.5)
    
    ax.set_title(f"{title} (Top {top_n})", fontsize=TITLE_SIZE, fontweight="bold", pad=15)
    ax.set_xlabel("Count", fontsize=LABEL_SIZE)
    ax.set_ylabel("")
    apply_style(ax)

def save_single_plot(df: pd.DataFrame, column: str, outdir: Path) -> None:
    kind = infer_kind(column)
    
    if kind == "dist_violin":
        # 连续变量使用上下结构的画布
        fig, axes = plt.subplots(2, 1, figsize=SINGLE_FIGSIZE, dpi=FIG_DPI, gridspec_kw={"height_ratios": [4, 1.5]})
        plot_dist_violin(axes[0], axes[1], df[column], column)
    else:
        fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE, dpi=FIG_DPI)
        if kind == "count_bar":
            plot_count_bar(ax, df[column], column)
        elif kind == "binary_donut":
            plot_binary_donut(ax, df[column], column)
        else:
            plot_top_categories(ax, df[column], column)
            
    fig.tight_layout(pad=1.5)
    fig.savefig(outdir / f"all_properties_{column}.png", bbox_inches="tight", transparent=False, facecolor='white')
    plt.close(fig)

def save_overview(df: pd.DataFrame, outdir: Path) -> None:
    columns = list(df.columns)
    ncols = 4  # 调整为4列，使总览图更加紧凑
    nrows = math.ceil(len(columns) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5.5, nrows * 4.5), dpi=FIG_DPI)
    axes = np.array(axes).reshape(-1)

    for ax, column in zip(axes, columns):
        kind = infer_kind(column)
        
        if kind == "dist_violin":
            arr = pd.to_numeric(df[column], errors="coerce").dropna()
            sns.histplot(arr, kde=True, ax=ax, color="#4C9181", edgecolor="white", alpha=0.7)
            ax.axvline(arr.median(), color="#E67E22", linestyle="--", linewidth=1.5)
            ax.set_xlabel("")
        
        elif kind == "count_bar":
            counts = pd.to_numeric(df[column], errors="coerce").dropna().astype(int).value_counts().sort_index()
            sns.barplot(x=counts.index, y=counts.values, ax=ax, palette="crest")
            ax.set_xlabel("")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        
        elif kind == "binary_donut":
            counts = pd.to_numeric(df[column], errors="coerce").dropna().astype(int).value_counts().sort_index().reindex([0, 1], fill_value=0)
            ax.pie(counts.values, labels=["F", "P"], colors=["#E74C3C", "#2ECC71"], 
                   autopct='%1.0f%%', pctdistance=0.75, wedgeprops=dict(width=0.4, edgecolor='w'))
        
        else:
            counts = df[column].fillna("<NA>").astype(str).value_counts().head(8)
            sns.barplot(x=counts.values, y=counts.index, ax=ax, palette="mako")
            ax.set_ylabel("")
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)

        if kind != "binary_donut":
            ax.set_title(column, fontsize=13, fontweight="bold")
            apply_style(ax)
        else:
            ax.set_title(column, fontsize=13, fontweight="bold", y=0.95)

    # 隐藏多余的子图
    for ax in axes[len(columns):]:
        ax.axis("off")

    fig.suptitle("Molecular Properties Overview", fontsize=24, fontweight="bold", y=1.02)
    fig.tight_layout(pad=1.2)
    fig.savefig(outdir / "all_properties_overview.png", bbox_inches="tight", facecolor='white')
    plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input property CSV path")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Output directory")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Generating individual beautiful plots...")
    for column in df.columns:
        save_single_plot(df, column, outdir)

    print("Generating overview plot...")
    save_overview(df, outdir)
    print(f"Done! All highly aesthetic property plots saved to: {outdir}")

if __name__ == "__main__":
    main()