#!/usr/bin/env python
"""SHP2 (5XZR) benchmark: 按可视化作图规范生成完整分布图 + 对接分数图。

输出结构:
  evaluation/
  ├── DiffSBDD/   (单模型性质分布)
  ├── DiffGui/     (单模型性质分布)
  ├── model_comparison/  (对接+性质对比)
  └── benchmark_summary.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

# ── 全局美学 ──
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#d4d4d4'
plt.rcParams['axes.linewidth'] = 1.2

FIG_DPI = 300
SINGLE_FIGSIZE = (8, 6)
COMBINED_FIGSIZE = (18, 12)
COMPARISON_FIGSIZE = (8, 5)
THRESHOLD_COLOR = "#E74C3C"
THRESHOLD_LW = 2.0

# ── 路径 ──
BENCHMARK = Path("/home/md666/data/disk6/DiffB110/analysis/benchmark/5xzr_8j6_benchmark_20260611")
PROP_CSV = BENCHMARK / "evaluation" / "core" / "per_molecule_metrics.csv"
DOCK_BASE = BENCHMARK / "docking" / "unidock_fast_5xzr_8j6"
EVAL_DIR = BENCHMARK / "evaluation"

# ── 配置 ──
MODELS = ["DiffSBDD", "DiffGui"]
MODEL_COLORS = {"DiffSBDD": "#E74C3C", "DiffGui": "#3498DB", "Native8J6": "#2ECC71"}
MODEL_LABELS = {"DiffSBDD": "DiffSBDD", "DiffGui": "DiffGui", "Native8J6": "Native 8J6"}
NATIVE_DOCK = -12.737

PROPERTIES: list[dict] = [
    {"col": "qed",            "title": "QED",               "kind": "violin", "color": "#E74C3C", "ylim": (0, 1),     "crit_label": "> 0.5",  "crit_val": 0.5,  "crit_type": "gt"},
    {"col": "logp",           "title": "LogP",              "kind": "violin", "color": "#F39C12", "ylim": (-4, 10),   "crit_label": "0 - 3",  "crit_lo": 0.0, "crit_hi": 3.0, "crit_type": "range"},
    {"col": "tpsa",           "title": "TPSA",              "kind": "violin", "color": "#2ECC71", "ylim": (0, 320),   "crit_label": "< 120",  "crit_val": 120.0, "crit_type": "lt"},
    {"col": "ring_count",     "title": "Ring Count",        "kind": "violin", "color": "#9B59B6", "ylim": (0, 12),    "crit_label": "≤ 4",    "crit_val": 4.0,  "crit_type": "le"},
    {"col": "carbon_fraction","title": "Carbon Fraction",   "kind": "violin", "color": "#1ABC9C", "ylim": (0.3, 1.0), "crit_label": "< 0.78", "crit_val": 0.78, "crit_type": "lt"},
    {"col": "heavy_atoms",    "title": "Heavy Atoms",       "kind": "hist",   "color": "#4A90E2", "bins": 30,        "crit_label": "< 500",  "crit_val": 500.0, "crit_type": "lt"},
]

# ── 工具函数 ──

def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(labelsize=10)


def add_ref_line(ax, value, label, orientation="horizontal"):
    kw = dict(color=THRESHOLD_COLOR, linestyle="--", linewidth=THRESHOLD_LW, alpha=0.85, zorder=4)
    if orientation == "horizontal":
        ax.axhline(value, **kw)
        xlim = ax.get_xlim()
        ax.text(xlim[1]*0.97, value + (ax.get_ylim()[1]-ax.get_ylim()[0])*0.02,
                label, color=THRESHOLD_COLOR, fontweight='bold', fontsize=9, ha='right', va='bottom')
    else:
        ax.axvline(value, **kw)


def load_data():
    prop = pd.read_csv(PROP_CSV)
    prop = prop[prop["model"].isin(MODELS)].copy()

    docks = []
    for m in MODELS:
        p = DOCK_BASE / m / "docking_results.csv"
        if p.exists():
            df = pd.read_csv(p).rename(columns={"benchmark_index": "idx"})
            docks.append(df[["model", "idx", "success", "score"]])
    dock = pd.concat(docks, ignore_index=True)
    merged = prop.merge(dock, on=["model", "idx"], how="left")
    return merged


def plot_single_model(merged, model, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    sub = merged[(merged["model"] == model) & (merged["sanitize_success"] == True)].copy()

    # Single plots
    for prop in PROPERTIES:
        fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE, dpi=FIG_DPI)
        col = prop["col"]
        vals = pd.to_numeric(sub[col], errors="coerce").dropna()

        if prop["kind"] == "violin":
            sns.violinplot(y=vals, ax=ax, color=prop["color"], inner="quartile",
                           linewidth=1.5, alpha=0.6, width=0.5)
            sns.stripplot(y=vals, ax=ax, color="#444444", alpha=0.12, size=2.5, jitter=0.2, zorder=0)
            if "crit_val" in prop:
                add_ref_line(ax, prop["crit_val"], prop["crit_label"])
            elif "crit_lo" in prop:
                add_ref_line(ax, prop["crit_lo"], prop["crit_label"])
                add_ref_line(ax, prop["crit_hi"], prop["crit_label"])
            if "ylim" in prop:
                ax.set_ylim(prop["ylim"])
            ax.set_xticks([])
        else:
            sns.histplot(vals, bins=prop.get("bins", 30), kde=True, color=prop["color"],
                         ax=ax, edgecolor="white", alpha=0.7, linewidth=1.0)
            if "crit_val" in prop:
                add_ref_line(ax, prop["crit_val"], prop["crit_label"], orientation="vertical")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=8))

        ax.set_title(prop["title"], fontsize=14, fontweight="bold", pad=10)
        ax.set_ylabel(prop["title"] if prop["kind"] != "hist" else "Count / Density", fontsize=11)
        style_ax(ax)
        fig.tight_layout()
        fig.savefig(out_dir / f"{col}.png", bbox_inches="tight", facecolor='white')
        plt.close(fig)

    # Docking score (violin for this model)
    dock_vals = sub["score"].dropna()
    if len(dock_vals) > 0:
        fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE, dpi=FIG_DPI)
        sns.violinplot(y=dock_vals, ax=ax, color=MODEL_COLORS[model], inner="quartile",
                       linewidth=1.5, alpha=0.6, width=0.5)
        sns.stripplot(y=dock_vals, ax=ax, color="#444444", alpha=0.12, size=2.5, jitter=0.2, zorder=0)
        add_ref_line(ax, NATIVE_DOCK, f"Native 8J6: {NATIVE_DOCK:.2f}")
        ax.set_ylabel("UniDock Score (kcal/mol)", fontsize=11)
        ax.set_title(f"{model} Docking Score", fontsize=14, fontweight="bold", pad=10)
        ax.set_xticks([])
        style_ax(ax)
        fig.tight_layout()
        fig.savefig(out_dir / "docking_score.png", bbox_inches="tight", facecolor='white')
        plt.close(fig)

    # Combined — vertical stack of horizontal violins + mean column
    n_props = len(PROPERTIES)
    fig, axes = plt.subplots(n_props, 1, figsize=(10, 2.2 * n_props), dpi=FIG_DPI,
                              gridspec_kw={"hspace": 0.15})
    if n_props == 1:
        axes = [axes]
    for idx, (ax, prop) in enumerate(zip(axes, PROPERTIES)):
        col = prop["col"]
        vals = pd.to_numeric(sub[col], errors="coerce").dropna()
        val_mean = float(vals.mean()) if len(vals) > 0 else 0.0

        if prop["kind"] == "violin":
            parts = ax.violinplot(vals, positions=[0], vert=False, showmeans=True,
                                  showmedians=True, widths=1.4)
            for pc in parts["bodies"]:
                pc.set_facecolor(prop["color"])
                pc.set_alpha(0.6)
            if "crit_val" in prop:
                ax.axvline(prop["crit_val"], color=THRESHOLD_COLOR, linestyle="--",
                           linewidth=THRESHOLD_LW, alpha=0.85, zorder=2)
        else:
            sns.histplot(vals, bins=prop.get("bins", 30), kde=True, color=prop["color"],
                         ax=ax, edgecolor="white", alpha=0.7, linewidth=0.8)
            if "crit_val" in prop:
                ax.axvline(prop["crit_val"], color=THRESHOLD_COLOR, linestyle="--",
                           linewidth=THRESHOLD_LW, alpha=0.85, zorder=2)
        if "ylim" in prop:
            ax.set_xlim(prop["ylim"])
        ax.set_yticks([])
        ax.set_ylabel(prop["title"], fontsize=11, fontweight="bold", rotation=0,
                      labelpad=50, va="center")
        # mean annotation
        xr = ax.get_xlim()
        ax.text(xr[1] * 0.97, 0, f"μ={val_mean:.2f}", fontsize=7,
                color=prop["color"], ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor=prop["color"], alpha=0.7))
        style_ax(ax)
    axes[-1].set_xlabel("Value", fontsize=10)
    fig.suptitle(f"{model}  Core Molecular Properties (SHP2)", fontsize=14, fontweight="bold", y=1.005)
    fig.tight_layout()
    fig.savefig(out_dir / "core_properties_combined.png", bbox_inches="tight", facecolor='white')
    plt.close(fig)
    print(f"  [{model}] {len(sub)} molecules → {len(list(out_dir.glob('*.png')))} plots")


def plot_model_comparison(merged, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    valid = merged[(merged["sanitize_success"] == True) & (merged["success"] == True)].copy()

    # Docking score comparison
    fig, ax = plt.subplots(figsize=COMPARISON_FIGSIZE, dpi=FIG_DPI)
    dock_data = [valid[valid["model"]==m]["score"].dropna().values for m in MODELS]
    vp = ax.violinplot(dock_data, positions=[1, 2], showmeans=True, showmedians=True)
    for i, m in enumerate(MODELS):
        vp["bodies"][i].set_facecolor(MODEL_COLORS[m])
        vp["bodies"][i].set_alpha(0.65)
        # strip overlay
        y_vals = dock_data[i]
        x_jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(y_vals))
        ax.scatter(np.full(len(y_vals), i+1) + x_jitter, y_vals, color="#444444", alpha=0.12, s=6, zorder=0)
    add_ref_line(ax, NATIVE_DOCK, f"Native 8J6: {NATIVE_DOCK:.2f}")
    ax.set_xticks([1, 2])
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=11)
    ax.set_ylabel("UniDock Score (kcal/mol)", fontsize=11)
    ax.set_title("SHP2 (5XZR) Docking Score Distribution", fontsize=13, fontweight="bold")
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "docking_score_violin.png", bbox_inches="tight", facecolor='white')
    plt.close(fig)

    # Properties comparison — vertical stack of horizontal violins + mean column
    n_props = len(PROPERTIES)
    fig, axes = plt.subplots(n_props, 1, figsize=(10, 2.2 * n_props), dpi=FIG_DPI,
                              gridspec_kw={"hspace": 0.15})
    if n_props == 1:
        axes = [axes]

    for idx, (ax, prop) in enumerate(zip(axes, PROPERTIES)):
        col = prop["col"]
        plot_data = [valid[valid["model"]==m][col].dropna().values for m in MODELS]
        means = [np.mean(d) for d in plot_data]

        # Horizontal violins — DiffSBDD at y=2.5, DiffGui at y=0.8
        y_positions = [2.5, 0.8]
        for i, m in enumerate(MODELS):
            d = plot_data[i]
            if len(d) == 0:
                continue
            vp = ax.violinplot(d, positions=[y_positions[i]], vert=False, showmeans=True,
                               showmedians=True, widths=1.4)
            vp["bodies"][0].set_facecolor(MODEL_COLORS[m])
            vp["bodies"][0].set_alpha(0.65)

        # Threshold line
        if "crit_val" in prop:
            ax.axvline(prop["crit_val"], color=THRESHOLD_COLOR, linestyle="--",
                       linewidth=THRESHOLD_LW, alpha=0.85, zorder=2)

        # Y-axis: property name + model labels
        ax.set_yticks([2.5, 0.8])
        ax.set_yticklabels([f"{MODEL_LABELS[m]}" for m in MODELS], fontsize=8)
        ax.set_ylabel(prop["title"], fontsize=10, fontweight="bold", rotation=0,
                      labelpad=50, va="center")
        if "xlim" in prop:
            ax.set_xlim(prop["xlim"])
        elif "ylim" in prop:
            ax.set_xlim(prop["ylim"])

        # Mean annotations on the right
        x_right = ax.get_xlim()[1]
        for i, m in enumerate(MODELS):
            ax.text(x_right * 0.97, y_positions[i], f"μ={means[i]:.2f}",
                    fontsize=7, color=MODEL_COLORS[m], ha="left", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              edgecolor=MODEL_COLORS[m], alpha=0.7))

        style_ax(ax)
        if idx < n_props - 1:
            ax.set_xlabel("")

    axes[-1].set_xlabel("Value", fontsize=10)
    fig.suptitle("SHP2 (5XZR) Molecular Properties: DiffSBDD vs DiffGui",
                 fontsize=13, fontweight="bold", y=1.005)
    fig.tight_layout()
    fig.savefig(out_dir / "properties_comparison.png", bbox_inches="tight", facecolor='white')
    plt.close(fig)
    print(f"  [comparison] {len(valid)} valid+docked molecules → {len(list(out_dir.glob('*.png')))} plots")


def save_summary_csv(merged, out_dir):
    valid = merged[(merged["sanitize_success"] == True) & (merged["success"] == True)].copy()
    rows = []
    for model in MODELS:
        sub = valid[valid["model"] == model]
        n = len(sub)
        row = {"model": model, "n_valid_docked": n}
        # docking
        dock = sub["score"].dropna()
        row["dock_mean"] = round(float(dock.mean()), 3)
        row["dock_median"] = round(float(dock.median()), 3)
        row["dock_best"] = round(float(dock.min()), 3)
        row["dock_std"] = round(float(dock.std()), 3)
        # properties
        for prop in PROPERTIES:
            col = prop["col"]
            vals = pd.to_numeric(sub[col], errors="coerce").dropna()
            row[f"{col}_mean"] = round(float(vals.mean()), 3)
            row[f"{col}_median"] = round(float(vals.median()), 3)
            row[f"{col}_std"] = round(float(vals.std()), 3)
            # pass rate
            if prop.get("crit_type") == "gt":
                passed = int((vals > prop["crit_val"]).sum())
            elif prop.get("crit_type") == "lt":
                passed = int((vals < prop["crit_val"]).sum())
            elif prop.get("crit_type") == "le":
                passed = int((vals <= prop["crit_val"]).sum())
            elif prop.get("crit_type") == "range":
                passed = int(((vals >= prop["crit_lo"]) & (vals <= prop["crit_hi"])).sum())
            else:
                passed = len(vals)
            row[f"{col}_pass_rate"] = round(passed / max(len(vals), 1), 3)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "benchmark_summary.csv", index=False)
    print(f"  [summary] wrote {out_dir / 'benchmark_summary.csv'}")


def main():
    print("[SHP2 plots] Loading data ...")
    merged = load_data()

    print("[SHP2 plots] Single-model distributions ...")
    for model in MODELS:
        plot_single_model(merged, model, EVAL_DIR / model)

    print("[SHP2 plots] Model comparison ...")
    plot_model_comparison(merged, EVAL_DIR / "model_comparison")

    print("[SHP2 plots] Summary CSV ...")
    save_summary_csv(merged, EVAL_DIR)

    print("[SHP2 plots] Done.")


if __name__ == "__main__":
    main()
