#!/usr/bin/env python
"""Generate violin plots for SHP2 (5XZR) benchmark: DiffGui vs DiffSBDD + docking."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BENCHMARK = Path(
    "/home/md666/data/disk6/DiffB110/analysis/benchmark/5xzr_8j6_benchmark_20260611"
)
EVAL_CORE = BENCHMARK / "evaluation" / "core"
OUT_DIR = BENCHMARK / "evaluation" / "violin_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load per-molecule properties ──
prop_df = pd.read_csv(EVAL_CORE / "per_molecule_metrics.csv")
prop_df = prop_df[prop_df["model"].isin(["DiffGui", "DiffSBDD", "Native8J6"])].copy()

# ── Load per-molecule docking scores ──
dock_dfs = []
DOCK_BASE = BENCHMARK / "docking" / "unidock_fast_5xzr_8j6"
for model in ["DiffGui", "DiffSBDD"]:
    csv_path = DOCK_BASE / model / "docking_results.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = df.rename(columns={"benchmark_index": "idx"})
        dock_dfs.append(df[["model", "idx", "success", "score"]])
dock_df = pd.concat(dock_dfs, ignore_index=True)

# Merge
merged = prop_df.merge(dock_df, on=["model", "idx"], how="left")

# Only valid & docked molecules for property distributions
valid = merged[(merged["sanitize_success"] == True) & (merged["success"] == True)].copy()

# ── Plotting ──
plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

models = ["DiffSBDD", "DiffGui"]
colors = {"DiffSBDD": "#E74C3C", "DiffGui": "#3498DB", "Native8J6": "#2ECC71"}
labels = {"DiffSBDD": "DiffSBDD", "DiffGui": "DiffGui", "Native8J6": "Native 8J6"}
position_kw = {"positions": [1, 2]}

# 1. Docking score violin
fig, ax = plt.subplots(figsize=(6, 5))
data_dock = [valid[valid["model"] == m]["score"].dropna().values for m in models]
vp = ax.violinplot(data_dock, showmeans=True, showmedians=True, **position_kw)
for i, m in enumerate(models):
    vp["bodies"][i].set_facecolor(colors[m])
    vp["bodies"][i].set_alpha(0.7)
ax.set_xticks([1, 2])
ax.set_xticklabels([labels[m] for m in models])
ax.set_ylabel("UniDock Score (kcal/mol)")
ax.set_title("SHP2 (5XZR) Docking Score Distribution")
ax.axhline(y=-12.737, color=colors["Native8J6"], linestyle="--", alpha=0.6, label=f"Native 8J6: -12.74")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT_DIR / "docking_score_violin.png", dpi=200)
plt.close(fig)

# 2. Property violin plots (2×3 grid)
props = [
    ("qed", "QED"),
    ("logp", "LogP"),
    ("tpsa", "TPSA"),
    ("ring_count", "Ring Count"),
    ("carbon_fraction", "Carbon Fraction"),
    ("heavy_atoms", "Heavy Atoms"),
]

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
for ax, (col, title) in zip(axes.flat, props):
    data = [valid[valid["model"] == m][col].dropna().values for m in models]
    vp = ax.violinplot(data, showmeans=True, showmedians=True, **position_kw)
    for i, m in enumerate(models):
        vp["bodies"][i].set_facecolor(colors[m])
        vp["bodies"][i].set_alpha(0.7)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([labels[m] for m in models], fontsize=8)
    ax.set_title(title, fontsize=11)
fig.suptitle("SHP2 (5XZR) Molecular Properties: DiffSBDD vs DiffGui", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(OUT_DIR / "properties_violin_grid.png", dpi=200)
plt.close(fig)

# 3. Docking success + sanitize bar chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
summary = pd.read_csv(BENCHMARK / "summary" / "benchmark_summary.csv")
summary_sel = summary[summary["model"].isin(["DiffGui", "DiffSBDD"])]
bars1 = ax1.bar([labels.get(r["model"], r["model"]) for _, r in summary_sel.iterrows()],
                [r["dock_success_rate"] * 100 for _, r in summary_sel.iterrows()],
                color=[colors.get(r["model"], "#999") for _, r in summary_sel.iterrows()])
ax1.set_ylabel("Docking Success Rate (%)")
ax1.set_title("Docking Success Rate")
ax1.set_ylim(0, 105)
for bar, val in zip(bars1, [r["dock_success_rate"] * 100 for _, r in summary_sel.iterrows()]):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{val:.1f}%", ha="center", fontsize=10)

bars2 = ax2.bar([labels.get(r["model"], r["model"]) for _, r in summary_sel.iterrows()],
                [r["sanitize_rate_records"] * 100 for _, r in summary_sel.iterrows()],
                color=[colors.get(r["model"], "#999") for _, r in summary_sel.iterrows()])
ax2.set_ylabel("Sanitize Success Rate (%)")
ax2.set_title("Sanitize Success Rate")
ax2.set_ylim(0, 105)
for bar, val in zip(bars2, [r["sanitize_rate_records"] * 100 for _, r in summary_sel.iterrows()]):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{val:.1f}%", ha="center", fontsize=10)
fig.suptitle("SHP2 (5XZR) Benchmark: Dock & Sanitize", fontsize=12)
fig.tight_layout()
fig.savefig(OUT_DIR / "docking_sanitize_bars.png", dpi=200)
plt.close(fig)

# 4. Summary stats table CSV
stats = []
for m in models:
    sub = valid[valid["model"] == m]
    row = {"model": m, "n_valid_docked": len(sub)}
    for col, _ in props:
        vals = sub[col].dropna()
        row[f"{col}_mean"] = vals.mean()
        row[f"{col}_median"] = vals.median()
        row[f"{col}_std"] = vals.std()
    dock_vals = sub["score"].dropna()
    row["dock_score_mean"] = dock_vals.mean()
    row["dock_score_median"] = dock_vals.median()
    row["dock_score_best"] = dock_vals.min()
    stats.append(row)
stats_df = pd.DataFrame(stats)
stats_df.to_csv(OUT_DIR / "violin_summary_stats.csv", index=False)

print(f"[violin] Wrote {len(list(OUT_DIR.glob('*.png')))} plots + stats CSV to {OUT_DIR}")
for f in sorted(OUT_DIR.glob("*.png")):
    print(f"  {f.name}")
