"""Validation that 200 diffusion steps is equivalent to 500 for DiffGui docking.

Controlled comparison: the same base checkpoint, the same pocket, the same
seeds (1-7 / 8-14 / 15-21), the same 1000-attempt protocol; only num_timesteps
differs. Both batch were docked here with one identical UniDock configuration
(exhaustiveness=8, vina, fast, 20 A box).

Output: diffgui_200_vs_500_violin.png / .svg, diffgui_200_vs_500_summary.csv
"""

from pathlib import Path
import glob
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

H = Path(__file__).resolve().parent
LABELS = {200: "200 steps", 500: "500 steps"}
COLORS = {"200 steps": "#2e86c1", "500 steps": "#c0392b"}

frames, rows = [], []
for step in (200, 500):
    parts = []
    for p in sorted(glob.glob(str(H / f"docking/{step}steps_*/docking_results.csv"))):
        d = pd.read_csv(p)
        parts.append(d.loc[d["success"].astype(bool) & d["score"].notna(), "score"].astype(float))
    vals = pd.concat(parts)
    frames.append(pd.DataFrame({"steps": LABELS[step], "score": vals}))
    rows.append({"steps": LABELS[step], "n": len(vals), "mean": round(float(vals.mean()), 4),
                 "median": round(float(vals.median()), 4), "std": round(float(vals.std()), 4)})

data = pd.concat(frames, ignore_index=True)
summary = pd.DataFrame(rows)
summary.to_csv(H / "diffgui_200_vs_500_summary.csv", index=False)
print(summary.to_string(index=False))

a = data.loc[data["steps"] == LABELS[200], "score"].to_numpy()
b = data.loc[data["steps"] == LABELS[500], "score"].to_numpy()
pval = stats.mannwhitneyu(a, b).pvalue
cohen = (b.mean() - a.mean()) / np.sqrt((a.var() + b.var()) / 2)
print(f"mean diff = {abs(a.mean() - b.mean()):.4f} kcal/mol")
print(f"Mann-Whitney p = {pval:.3f}   Cohen d = {cohen:+.3f}")

sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
fig, ax = plt.subplots(figsize=(5.4, 6.0), dpi=300)
order = [LABELS[200], LABELS[500]]
sns.violinplot(data=data, x="steps", y="score", order=order, hue="steps", hue_order=order,
               palette=COLORS, inner=None, cut=0, linewidth=0.9, saturation=1.0,
               width=0.8, legend=False, ax=ax)

for i, lab in enumerate(order):
    vals = data.loc[data["steps"] == lab, "score"].to_numpy()
    mean = float(vals.mean())
    ax.hlines(mean, i - 0.30, i + 0.30, color="#2c3e50", linewidth=3.2, zorder=4)
    ax.scatter([i], [mean], s=54, color="white", edgecolor="#2c3e50", linewidth=1.8, zorder=5)
    ax.text(i, mean + 0.20, f"{mean:.2f}", ha="center", va="bottom", fontsize=13,
            fontweight="bold", color="#2c3e50", zorder=6,
            bbox=dict(facecolor="white", alpha=0.80, edgecolor="none", pad=1.4))

ax.set_xticks(range(len(order)))
ax.set_xticklabels(order, fontsize=17, fontweight="bold")
ax.set_xlabel("")
ax.set_ylabel("UniDock score (kcal/mol)", fontsize=15)
ax.tick_params(axis="y", labelsize=13)
lo = min(a.min(), b.min())
hi = max(a.max(), b.max())
ax.set_ylim(lo - 0.6, hi + 1.4)
ax.set_xlim(-0.60, 1.60)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.text(0.5, 0.975,
        f"Δ mean = {abs(a.mean() - b.mean()):.3f} kcal/mol\n"
        f"Mann-Whitney p = {pval:.2f}   |   Cohen d = {abs(cohen):.3f}",
        transform=ax.transAxes, ha="center", va="top", fontsize=10.5, color="#34495e",
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="#d4d4d4"))
fig.subplots_adjust(left=0.17, right=0.965, top=0.975, bottom=0.115)
fig.savefig(H / "diffgui_200_vs_500_violin.png", dpi=300)
fig.savefig(H / "diffgui_200_vs_500_violin.svg")
plt.close(fig)
print("Wrote diffgui_200_vs_500_violin.png")
