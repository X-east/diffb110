"""Plot a 2x3 panel of key DiffGui PB-fast RL training metrics per epoch.

Metrics: docking score (Vina), QED, SA, molecular diversity (ECFP4 mean
pairwise similarity), PB-valid rate (per attempts), LogP.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BASE = "/home/md666/data/disk6/DiffB110/analysis/diffgui_rl/pbfast_100ep_20260715_training_analysis"

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.edgecolor"] = "#d4d4d4"
plt.rcParams["axes.linewidth"] = 1.2

DIFFGUI_BLUE = "#3498DB"

df1 = pd.read_csv(os.path.join(BASE, "epoch_metrics_selected.csv"))
df2 = pd.read_csv(os.path.join(BASE, "molecule_epoch_summary.csv"))
df = df1.merge(df2[["epoch", "qed_raw_mean_calc", "sa_raw_mean_calc"]], on="epoch")

panels = [
    ("Docking Score (Vina, mean)", "docking_raw_mean", DIFFGUI_BLUE, "lower is better"),
    ("QED", "qed_raw_mean_calc", "#E74C3C", None),
    ("SA Score", "sa_raw_mean_calc", "#1ABC9C", None),
    ("Molecular Diversity (ECFP4 mean pairwise sim)", "ecfp4_mean_pairwise_similarity", "#F39C12", "lower = more diverse"),
    ("PB-valid Rate (per attempts)", "posebusters_pb_valid_rate_attempts", "#2ECC71", None),
    ("LogP", "logp_mean", "#9B59B6", None),
]

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 9))
for ax, (title, col, color, note) in zip(axes.flat, panels):
    ax.plot(df["epoch"], df[col], color=color, linewidth=2.0, marker="o", markersize=3)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(col)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    if note:
        ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=9, color="#888888")

fig.suptitle("DiffGui PB-fast RL (100ep): Docking / QED / SA / Diversity / PB-valid / LogP", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])

out_png = os.path.join(BASE, "group_overview_6metrics.png")
out_svg = os.path.join(BASE, "group_overview_6metrics.svg")
fig.savefig(out_png, dpi=300)
fig.savefig(out_svg, dpi=300)
plt.close(fig)
print("saved:", out_png)
print("saved:", out_svg)
print("epochs:", len(df))
