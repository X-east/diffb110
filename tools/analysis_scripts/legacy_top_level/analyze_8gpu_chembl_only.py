#!/usr/bin/env python3
"""Analyze 8gpu-chembl-only training run with comprehensive plots."""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path("/home/md666/data/disk6/DiffB110/OUT/8gpu-chembl-only_20260520_101340")
CSV_PATH = ROOT / "training_summary.csv"
EPOCH_DIR = ROOT / "epochs"

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
C_BLUE = "#4C72B0"
C_ORANGE = "#DD8452"
C_GREEN = "#55A868"
C_RED = "#C44E52"
C_PURPLE = "#8172B2"
C_BROWN = "#937860"
C_CYAN = "#64B5CD"

# Larger font sizes
FS = 14          # axis labels
FS_TITLE = 17    # subplot titles
FS_SUPTITLE = 20 # figure super titles
FS_LEGEND = 12   # legends
FS_TICK = 12     # tick labels
FS_TABLE = 14    # table text

# ---------------------------------------------------------------------------
# Load training summary
# ---------------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

# Parse failure_counts JSON string
def parse_failure_counts(s):
    try:
        return json.loads(s.replace("'", '"'))
    except Exception:
        return {}

fc_df = df["failure_counts"].apply(parse_failure_counts).apply(pd.Series).fillna(0).astype(int)
for col in fc_df.columns:
    df[col] = fc_df[col]

epochs = df["epoch"].values
N = len(epochs)

# ---------------------------------------------------------------------------
# Load latest epoch molecules
# ---------------------------------------------------------------------------
mol30 = pd.read_csv(EPOCH_DIR / "epoch_0030" / "molecules.csv")
# Filter valid molecules
valid30 = mol30[mol30["rdkit_valid"] == 1].copy()

# ---------------------------------------------------------------------------
# Figure 1: Training curves (6-panel)
# ---------------------------------------------------------------------------
fig1, axes = plt.subplots(2, 3, figsize=(20, 12))
fig1.suptitle("8GPU ChEMBL-Only (5/20) Training Curves  (epochs 1–30)", fontsize=FS_SUPTITLE, fontweight="bold", y=1.02)

# --- 1a: Avg Loss ---
ax = axes[0, 0]
ax.plot(epochs, df["avg_loss"], color=C_BLUE, linewidth=2.0, marker="o", markersize=5)
ax.set_xlabel("Epoch", fontsize=FS)
ax.set_ylabel("Avg Loss", fontsize=FS)
ax.set_title("(a) PPO Loss", fontsize=FS_TITLE, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.set_xlim(1, 30)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
ax.tick_params(labelsize=FS_TICK)

# --- 1b: RDKit Validity ---
ax = axes[0, 1]
ax.plot(epochs, df["rdkit_valid_ratio"] * 100, color=C_GREEN, linewidth=2.0, marker="s", markersize=5)
ax.set_xlabel("Epoch", fontsize=FS)
ax.set_ylabel("Validity (%)", fontsize=FS)
ax.set_title("(b) RDKit Validity Ratio", fontsize=FS_TITLE, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.set_xlim(1, 30)
ax.tick_params(labelsize=FS_TICK)

# --- 1c: Reward Mean ---
ax = axes[0, 2]
ax.plot(epochs, df["reward_mean"], color=C_ORANGE, linewidth=2.0, marker="^", markersize=5)
ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("Epoch", fontsize=FS)
ax.set_ylabel("Reward Mean", fontsize=FS)
ax.set_title("(c) Reward Mean", fontsize=FS_TITLE, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.set_xlim(1, 30)
ax.tick_params(labelsize=FS_TICK)

# --- 1d: ChEMBL Similarity ---
ax = axes[1, 0]
ax.plot(epochs, df["avg_chembl_similarity"], color=C_PURPLE, linewidth=2.0, marker="D", markersize=5)
ax.set_xlabel("Epoch", fontsize=FS)
ax.set_ylabel("Avg ChEMBL Similarity", fontsize=FS)
ax.set_title("(d) Avg ChEMBL Tanimoto Similarity", fontsize=FS_TITLE, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.set_xlim(1, 30)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
ax.tick_params(labelsize=FS_TICK)

# --- 1e: Failure Counts (stacked area) ---
ax = axes[1, 1]
ax.fill_between(epochs, 0, df["success"], label="Success", color=C_GREEN, alpha=0.7)
ax.fill_between(epochs, df["success"], df["success"] + df["fragmented_molecule"],
                label="Fragmented", color=C_ORANGE, alpha=0.7)
total = df["success"] + df["fragmented_molecule"] + df["sanitize_failed"]
ax.fill_between(epochs, total - df["sanitize_failed"], total,
                label="Sanitize Failed", color=C_RED, alpha=0.7)
ax.set_xlabel("Epoch", fontsize=FS)
ax.set_ylabel("Molecule Count", fontsize=FS)
ax.set_title("(e) Molecule Failure Breakdown", fontsize=FS_TITLE, fontweight="bold")
ax.legend(loc="upper right", fontsize=FS_LEGEND)
ax.set_xlim(1, 30)
ax.set_ylim(0, 520)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=FS_TICK)

# --- 1f: Overall Reward Trend with success count ---
ax = axes[1, 2]
color1, color2 = C_RED, C_GREEN
ax2 = ax.twinx()
line1 = ax.plot(epochs, df["reward_mean"], color=color1, linewidth=2.0, marker="o", markersize=5, label="Reward Mean")
line2 = ax2.plot(epochs, df["success"], color=color2, linewidth=2.0, marker="s", markersize=5, label="Success")
ax.set_xlabel("Epoch", fontsize=FS)
ax.set_ylabel("Reward Mean", fontsize=FS, color=color1)
ax2.set_ylabel("Success Count", fontsize=FS, color=color2)
ax.set_title("(f) Reward vs Success Count", fontsize=FS_TITLE, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.set_xlim(1, 30)
ax.tick_params(labelsize=FS_TICK)
ax2.tick_params(labelsize=FS_TICK)
lines = line1 + line2
ax.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=FS_LEGEND)

plt.tight_layout()
fig1.savefig(ROOT / "training_curves.png", dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"[OK] training_curves.png")

# ---------------------------------------------------------------------------
# Figure 2: Latest epoch molecule analysis (epoch 30)
# ---------------------------------------------------------------------------
fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
fig2.suptitle("Epoch 30 Molecule Distribution  (512 total, {0} valid)".format(len(valid30)),
              fontsize=FS_SUPTITLE, fontweight="bold", y=1.02)

# --- 2a: ChEMBL Prior Reward histogram ---
ax = axes[0, 0]
reward_bins = np.linspace(-2.0, 2.0, 25)
ax.hist(valid30["chembl_prior_reward"], bins=reward_bins, color=C_BLUE, alpha=0.8, edgecolor="white", linewidth=0.5)
ax.axvline(x=valid30["chembl_prior_reward"].mean(), color=C_RED, linestyle="--", linewidth=2.0,
           label=f'Mean = {valid30["chembl_prior_reward"].mean():.3f}')
ax.set_xlabel("ChEMBL Prior Reward", fontsize=FS)
ax.set_ylabel("Molecule Count", fontsize=FS)
ax.set_title("(a) ChEMBL Prior Reward Distribution", fontsize=FS_TITLE, fontweight="bold")
ax.legend(fontsize=FS_LEGEND)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=FS_TICK)

# --- 2b: ChEMBL Similarity histogram ---
ax = axes[0, 1]
sim_bins = np.linspace(0.15, 0.50, 25)
ax.hist(valid30["chembl_mean_similarity"], bins=sim_bins, color=C_ORANGE, alpha=0.8, edgecolor="white", linewidth=0.5)
ax.axvline(x=valid30["chembl_mean_similarity"].mean(), color=C_RED, linestyle="--", linewidth=2.0,
           label=f'Mean = {valid30["chembl_mean_similarity"].mean():.3f}')
ax.set_xlabel("ChEMBL Mean Tanimoto Similarity", fontsize=FS)
ax.set_ylabel("Molecule Count", fontsize=FS)
ax.set_title("(b) ChEMBL Top-10 Similarity Distribution", fontsize=FS_TITLE, fontweight="bold")
ax.legend(fontsize=FS_LEGEND)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=FS_TICK)

# --- 2c: Reward vs Similarity scatter ---
ax = axes[1, 0]
sc = ax.scatter(valid30["chembl_mean_similarity"], valid30["chembl_prior_reward"],
                c=valid30["chembl_mean_similarity"], cmap="viridis", alpha=0.6, s=20, edgecolors="none")
ax.set_xlabel("ChEMBL Mean Similarity", fontsize=FS)
ax.set_ylabel("ChEMBL Prior Reward", fontsize=FS)
ax.set_title("(c) Reward vs Similarity  (per molecule)", fontsize=FS_TITLE, fontweight="bold")
plt.colorbar(sc, ax=ax, label="Similarity")
ax.grid(True, alpha=0.3)
# Add reward function lines (config-based)
sim_thresh_low = 0.25
sim_thresh_high = 1.0
low_w = 8.0
high_w = 0.0
max_sim = 0.5
sim_range = np.linspace(0.15, 0.50, 50)
# Approximate reward function
def reward_func(sim):
    if sim < sim_thresh_low:
        return (sim - sim_thresh_low) * low_w
    elif sim < sim_thresh_high:
        return (sim - sim_thresh_low) * (1 - high_w) + high_w
    else:
        return high_w + (sim - sim_thresh_high) * high_w

reward_vals = np.array([reward_func(s) for s in sim_range])
# Clip to max_sim
over_max = sim_range > max_sim
reward_vals[over_max] = reward_func(max_sim) - (sim_range[over_max] - max_sim) * 5.0

ax.plot(sim_range, reward_vals, color=C_RED, linewidth=2, linestyle="--", label="Reward Function")
ax.legend(fontsize=FS_LEGEND)
ax.tick_params(labelsize=FS_TICK)

# --- 2d: Molecule size distribution (from SMILES length proxy) ---
ax = axes[1, 1]
smiles_lens = valid30["smiles"].dropna().str.len()
ax.hist(smiles_lens, bins=20, color=C_GREEN, alpha=0.8, edgecolor="white", linewidth=0.5)
ax.set_xlabel("SMILES Length (proxy for molecular size)", fontsize=FS)
ax.set_ylabel("Molecule Count", fontsize=FS)
ax.set_title("(d) Molecular Size Distribution", fontsize=FS_TITLE, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=FS_TICK)

plt.tight_layout()
fig2.savefig(ROOT / "epoch30_molecule_analysis.png", dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"[OK] epoch30_molecule_analysis.png")

# ---------------------------------------------------------------------------
# Figure 3: Epoch-to-epoch evolution of reward distribution (boxplot per epoch)
# ---------------------------------------------------------------------------
# Sample molecule-level data from selected epochs
sample_epochs = [1, 5, 10, 15, 20, 25, 30]
reward_data = []
sim_data = []
valid_epochs = []
for ep in sample_epochs:
    ep_dir = EPOCH_DIR / f"epoch_{ep:04d}"
    ep_csv = ep_dir / "molecules.csv"
    if ep_csv.exists():
        ep_df = pd.read_csv(ep_csv)
        ep_valid = ep_df[ep_df["rdkit_valid"] == 1]
        reward_data.append(ep_valid["chembl_prior_reward"].values)
        sim_data.append(ep_valid["chembl_mean_similarity"].values)
        valid_epochs.append(ep)

fig3, axes = plt.subplots(1, 2, figsize=(16, 8))
fig3.suptitle("Reward & Similarity Distribution Evolution", fontsize=FS_SUPTITLE, fontweight="bold", y=1.02)

# --- 3a: Reward boxplot ---
ax = axes[0]
bp1 = ax.boxplot(reward_data, labels=valid_epochs, patch_artist=True, widths=0.6,
                 medianprops={"color": "black", "linewidth": 1.5})
for patch, color in zip(bp1["boxes"], plt.cm.RdYlGn(np.linspace(0.1, 0.8, len(valid_epochs)))):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.axhline(y=0, color="gray", linestyle="--", alpha=0.4)
ax.set_xlabel("Epoch", fontsize=FS)
ax.set_ylabel("ChEMBL Prior Reward", fontsize=FS)
ax.set_title("(a) Reward Boxplot by Epoch", fontsize=FS_TITLE, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=FS_TICK)

# --- 3b: Similarity boxplot ---
ax = axes[1]
bp2 = ax.boxplot(sim_data, labels=valid_epochs, patch_artist=True, widths=0.6,
                 medianprops={"color": "black", "linewidth": 1.5})
for patch, color in zip(bp2["boxes"], plt.cm.Blues(np.linspace(0.2, 0.9, len(valid_epochs)))):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_xlabel("Epoch", fontsize=FS)
ax.set_ylabel("ChEMBL Mean Tanimoto Similarity", fontsize=FS)
ax.set_title("(b) Similarity Boxplot by Epoch", fontsize=FS_TITLE, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=FS_TICK)

plt.tight_layout()
fig3.savefig(ROOT / "distribution_evolution.png", dpi=150, bbox_inches="tight")
plt.close(fig3)
print(f"[OK] distribution_evolution.png")

# ---------------------------------------------------------------------------
# Figure 4: Key metric summary table (epoch 1 vs epoch 30)
# ---------------------------------------------------------------------------
fig4, ax = plt.subplots(figsize=(12, 5))
ax.axis("off")

summary_data = [
    ["Metric", "Epoch 1", "Epoch 30", "Change"],
    ["PPO Loss", f"{df['avg_loss'].iloc[0]:.4f}", f"{df['avg_loss'].iloc[-1]:.4f}",
     f"{df['avg_loss'].iloc[-1] - df['avg_loss'].iloc[0]:.4f}"],
    ["RDKit Validity", f"{df['rdkit_valid_ratio'].iloc[0]:.2%}", f"{df['rdkit_valid_ratio'].iloc[-1]:.2%}",
     f"+{(df['rdkit_valid_ratio'].iloc[-1] - df['rdkit_valid_ratio'].iloc[0])*100:.1f}pp"],
    ["Reward Mean", f"{df['reward_mean'].iloc[0]:.4f}", f"{df['reward_mean'].iloc[-1]:.4f}",
     f"{df['reward_mean'].iloc[-1] - df['reward_mean'].iloc[0]:.4f}"],
    ["ChEMBL Similarity", f"{df['avg_chembl_similarity'].iloc[0]:.4f}",
     f"{df['avg_chembl_similarity'].iloc[-1]:.4f}",
     f"{df['avg_chembl_similarity'].iloc[-1] - df['avg_chembl_similarity'].iloc[0]:+.4f}"],
    ["Valid Molecules", f"{int(df['success'].iloc[0])}/512",
     f"{int(df['success'].iloc[-1])}/512",
     f"+{int(df['success'].iloc[-1] - df['success'].iloc[0])}"],
    ["Fragmented", f"{int(df['fragmented_molecule'].iloc[0])}",
     f"{int(df['fragmented_molecule'].iloc[-1])}",
     f"{int(df['fragmented_molecule'].iloc[-1] - df['fragmented_molecule'].iloc[0]):+d}"],
    ["Sanitize Failed", f"{int(df['sanitize_failed'].iloc[0])}",
     f"{int(df['sanitize_failed'].iloc[-1])}",
     f"{int(df['sanitize_failed'].iloc[-1] - df['sanitize_failed'].iloc[0]):+d}"],
]

table = ax.table(cellText=summary_data, loc="center", cellLoc="center", colWidths=[0.2, 0.2, 0.2, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(FS_TABLE)
table.scale(1, 1.8)

# Style header
for i in range(len(summary_data[0])):
    cell = table[0, i]
    cell.set_facecolor("#4472C4")
    cell.set_text_props(color="white", fontweight="bold")

# Highlight change column
for i in range(1, len(summary_data)):
    cell = table[i, 3]
    val = summary_data[i][3]
    if val.startswith("+"):
        cell.set_facecolor("#C6EFCE")
    elif val.startswith("-"):
        cell.set_facecolor("#FFC7CE")
    elif val == "0":
        cell.set_facecolor("#E2EFDA")

ax.set_title("Training Progress Summary (5/20)  |  Epoch 1 → Epoch 30", fontsize=FS_SUPTITLE, fontweight="bold", pad=20)

plt.tight_layout()
fig4.savefig(ROOT / "summary_table.png", dpi=150, bbox_inches="tight")
plt.close(fig4)
print(f"[OK] summary_table.png")

# ---------------------------------------------------------------------------
# Figure 5: All 30 epochs reward/similarity violin plot
# ---------------------------------------------------------------------------
all_rewards = []
all_sims = []
all_epochs_labels = []
for ep in range(1, 31):
    ep_dir = EPOCH_DIR / f"epoch_{ep:04d}"
    ep_csv = ep_dir / "molecules.csv"
    if ep_csv.exists():
        ep_df = pd.read_csv(ep_csv)
        ep_valid = ep_df[ep_df["rdkit_valid"] == 1]
        all_rewards.append(ep_valid["chembl_prior_reward"].values)
        all_sims.append(ep_valid["chembl_mean_similarity"].values)
        all_epochs_labels.append(str(ep))

fig5, axes = plt.subplots(2, 1, figsize=(20, 14))
fig5.suptitle("Per-Epoch Molecule Distribution (violin + mean trend)", fontsize=FS_SUPTITLE, fontweight="bold", y=1.02)

# 5a: Reward violin
ax = axes[0]
parts = ax.violinplot(all_rewards, positions=range(1, 31), showmeans=True, showmedians=False, widths=0.7)
for pc in parts["bodies"]:
    pc.set_facecolor(C_BLUE)
    pc.set_alpha(0.4)
ax.plot(range(1, 31), [r.mean() for r in all_rewards], color=C_RED, linewidth=2, marker="o", markersize=4, label="Mean")
ax.set_xlabel("Epoch", fontsize=FS)
ax.set_ylabel("ChEMBL Prior Reward", fontsize=FS)
ax.set_title("(a) Reward Distribution per Epoch", fontsize=FS_TITLE, fontweight="bold")
ax.legend(fontsize=FS_LEGEND)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=FS_TICK)

# 5b: Similarity violin
ax = axes[1]
parts = ax.violinplot(all_sims, positions=range(1, 31), showmeans=True, showmedians=False, widths=0.7)
for pc in parts["bodies"]:
    pc.set_facecolor(C_GREEN)
    pc.set_alpha(0.4)
ax.plot(range(1, 31), [s.mean() for s in all_sims], color=C_RED, linewidth=2, marker="o", markersize=4, label="Mean")
ax.set_xlabel("Epoch", fontsize=FS)
ax.set_ylabel("ChEMBL Mean Tanimoto Similarity", fontsize=FS)
ax.set_title("(b) Similarity Distribution per Epoch", fontsize=FS_TITLE, fontweight="bold")
ax.legend(fontsize=FS_LEGEND)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=FS_TICK)

plt.tight_layout()
fig5.savefig(ROOT / "per_epoch_violin.png", dpi=150, bbox_inches="tight")
plt.close(fig5)
print(f"[OK] per_epoch_violin.png")

# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("  8GPU ChEMBL-Only (5/20) Training Summary")
print("=" * 70)
print(f"  Run ID      : 2026-05-20T10:06:22 (weight=15, top_k=3, lr=2e-5, time_step=200)")
print(f"  Status      : Epoch 30/100")
print(f"  Hardware    : 8 GPU DDP, batch_size=64/GPU")
print(f"  Reward      : chembl_prior=1.0 (QED=0, SA=0, Docking=0)")
print()
print(f"  {'Metric':<30s} {'Epoch 1':>12s} {'Epoch 30':>12s} {'Change':>12s}")
print("  " + "-" * 30 + " " + "-" * 12 + " " + "-" * 12 + " " + "-" * 12)
print(f"  {'PPO Loss':<30s} {df['avg_loss'].iloc[0]:>12.4f} {df['avg_loss'].iloc[-1]:>12.4f} {df['avg_loss'].iloc[-1] - df['avg_loss'].iloc[0]:>+12.4f}")
print(f"  {'RDKit Validity':<30s} {df['rdkit_valid_ratio'].iloc[0]:>11.2%} {df['rdkit_valid_ratio'].iloc[-1]:>11.2%} {df['rdkit_valid_ratio'].iloc[-1] - df['rdkit_valid_ratio'].iloc[0]:>+11.2%}")
print(f"  {'Reward Mean':<30s} {df['reward_mean'].iloc[0]:>12.4f} {df['reward_mean'].iloc[-1]:>12.4f} {df['reward_mean'].iloc[-1] - df['reward_mean'].iloc[0]:>+12.4f}")
print(f"  {'ChEMBL Similarity':<30s} {df['avg_chembl_similarity'].iloc[0]:>12.4f} {df['avg_chembl_similarity'].iloc[-1]:>12.4f} {df['avg_chembl_similarity'].iloc[-1] - df['avg_chembl_similarity'].iloc[0]:>+12.4f}")
print(f"  {'Valid Molecules':<30s} {int(df['success'].iloc[0]):>7d}/512 {int(df['success'].iloc[-1]):>7d}/512 {int(df['success'].iloc[-1] - df['success'].iloc[0]):>+8d}")
print(f"  {'Fragmented':<30s} {int(df['fragmented_molecule'].iloc[0]):>12d} {int(df['fragmented_molecule'].iloc[-1]):>12d} {int(df['fragmented_molecule'].iloc[-1] - df['fragmented_molecule'].iloc[0]):>+12d}")
print(f"  {'Sanitize Failed':<30s} {int(df['sanitize_failed'].iloc[0]):>12d} {int(df['sanitize_failed'].iloc[-1]):>12d} {int(df['sanitize_failed'].iloc[-1] - df['sanitize_failed'].iloc[0]):>+12d}")
print()
print(f"  Plots saved to: {ROOT}/")
print(f"    - training_curves.png")
print(f"    - epoch30_molecule_analysis.png")
print(f"    - distribution_evolution.png")
print(f"    - per_epoch_violin.png")
print(f"    - summary_table.png")
print("=" * 70)
