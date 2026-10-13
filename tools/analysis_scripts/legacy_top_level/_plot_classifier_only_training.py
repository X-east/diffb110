#!/usr/bin/env python3
"""
Comprehensive visualization of 7gpu-classifier-only_20260526_210715 training.
Generates ~20 plots across 5 categories, saved to runs/<run>/analysis_figures/
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ── Config ──────────────────────────────────────────────────
RUN_DIR = Path("/home/md666/data/disk6/DiffB110/runs/7gpu-classifier-only_20260526_210715")
OUT_DIR = RUN_DIR / "analysis_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.35,
})

# Phase boundaries (based on prior analysis)
PHASE_1_END = 24   # Healthy
PHASE_2_END = 30   # Collapse
PHASE_3_END = 37   # Dying
# Phase 4: 38-64 Dead

PHASE_COLORS = {
    "Phase 1: Normal":     "#2ecc71",
    "Phase 2: Collapse":   "#f39c12",
    "Phase 3: Dying":      "#e74c3c",
    "Phase 4: Dead":       "#95a5a6",
}

# ── Load data ───────────────────────────────────────────────
df = pd.read_csv(RUN_DIR / "training_summary.csv")
epochs = df["epoch"].values
n_plots = 0


def add_phase_bg(ax):
    """Shade phase backgrounds on any axis."""
    ymin, ymax = ax.get_ylim()
    if PHASE_1_END < len(epochs):
        ax.axvspan(1, PHASE_1_END, alpha=0.06, color=PHASE_COLORS["Phase 1: Normal"])
    if PHASE_1_END + 1 <= PHASE_2_END:
        ax.axvspan(PHASE_1_END + 1, PHASE_2_END, alpha=0.08, color=PHASE_COLORS["Phase 2: Collapse"])
    if PHASE_2_END + 1 <= PHASE_3_END:
        ax.axvspan(PHASE_2_END + 1, PHASE_3_END, alpha=0.10, color=PHASE_COLORS["Phase 3: Dying"])
    if PHASE_3_END + 1 <= len(epochs):
        ax.axvspan(PHASE_3_END + 1, len(epochs), alpha=0.08, color=PHASE_COLORS["Phase 4: Dead"])


def savefig(name):
    global n_plots
    n_plots += 1
    path = OUT_DIR / name
    plt.savefig(path)
    plt.close()
    return path


def parse_failure_counts(failure_str) -> dict:
    """Parse failure dict from CSV."""
    if pd.isna(failure_str) or failure_str == "":
        return {}
    s = str(failure_str)
    # Handle both single and double quoted JSON
    s = s.replace('""', '"')
    s = s.strip('"')
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return {}


# Parse failure counts for each epoch
failure_list = []
for _, row in df.iterrows():
    fc = parse_failure_counts(row["failure_counts"])
    failure_list.append(fc)

df["n_fragmented"] = [f.get("fragmented_molecule", 0) for f in failure_list]
df["n_sanitize_failed"] = [f.get("sanitize_failed", 0) for f in failure_list]
df["n_success"] = [f.get("success", 0) for f in failure_list]


# ============================================================
# Figure 1: Collapse overview — 4 panel
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Classifier-Only Training: Collapse Overview", fontsize=13, fontweight="bold")

# 1a: rdkit_valid (bar + line)
ax = axes[0, 0]
colors = ["#2ecc71" if e <= PHASE_1_END else "#f39c12" if e <= PHASE_2_END else "#e74c3c"
          for e in epochs]
ax.bar(epochs, df["rdkit_valid"], color=colors, width=0.8, alpha=0.85)
ax.set_ylabel("Valid molecules")
ax.set_title("rdkit_valid count")
ax.axhline(280, color="gray", ls="--", alpha=0.5, label="50% of batch")
add_phase_bg(ax)

# 1b: rdkit_valid_ratio + success ratio
ax = axes[0, 1]
ax.plot(epochs, df["rdkit_valid_ratio"], "o-", markersize=3, lw=1, color="#2c3e50", label="rdkit_valid_ratio")
ax.fill_between(epochs, 0, df["rdkit_valid_ratio"], alpha=0.1, color="#2c3e50")
ax.axhline(0.5, color="gray", ls="--", alpha=0.5)
ax.set_ylabel("Ratio")
ax.set_title("Valid ratio")
ax.set_ylim(-0.05, 1.05)
add_phase_bg(ax)

# 1c: PPO Loss
ax = axes[1, 0]
ax.plot(epochs, df["PPO_Loss"], "s-", markersize=3, lw=1.2, color="#8e44ad")
ax.set_ylabel("PPO Loss")
ax.set_title("PPO Loss (capped at 2.0)")
ax.axhline(2.0, color="red", ls="--", alpha=0.4, label="cap=2.0")
add_phase_bg(ax)

# 1d: reward_raw (absolute total reward)
ax = axes[1, 1]
ax.plot(epochs, df["avg_reward_raw"], "D-", markersize=3, lw=1.2, color="#e67e22")
ax.set_ylabel("avg_reward_raw")
ax.set_title("Average raw reward")
ax.axhline(0, color="gray", ls="--", alpha=0.4)
add_phase_bg(ax)

plt.tight_layout()
savefig("01_collapse_overview.png")
print(f"  [01] collapse_overview")


# ============================================================
# Figure 2: Failure decomposition (stacked bar)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(epochs, df["n_success"], label="Success", color="#2ecc71", width=0.8, alpha=0.85)
ax.bar(epochs, df["n_fragmented"], bottom=df["n_success"], label="Fragmented", color="#e74c3c", width=0.8, alpha=0.7)
ax.bar(epochs, df["n_sanitize_failed"],
       bottom=df["n_success"] + df["n_fragmented"],
       label="Sanitize Failed", color="#f39c12", width=0.8, alpha=0.7)
ax.set_ylabel("Molecule count")
ax.set_title("Failure Decomposition per Epoch (batch=560)")
ax.legend(loc="upper right", framealpha=0.9)
ax.set_xlim(0.5, len(epochs) + 0.5)
add_phase_bg(ax)
plt.tight_layout()
savefig("02_failure_decomposition.png")
print(f"  [02] failure_decomposition")


# ============================================================
# Figure 3: LogP evolution — 3 panel
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle("LogP Evolution", fontsize=12, fontweight="bold")

ax = axes[0]
ax.plot(epochs, df["logp_mean"], "o-", markersize=3, lw=1, color="#2980b9", label="Mean")
ax.set_ylabel("LogP mean")
ax.set_title("LogP Mean")
add_phase_bg(ax)

ax = axes[1]
ax.plot(epochs, df["logp_median"], "s-", markersize=3, lw=1, color="#27ae60", label="Median")
ax.set_ylabel("LogP median")
ax.set_title("LogP Median")
add_phase_bg(ax)

ax = axes[2]
ax.plot(epochs, df["logp_std"], "D-", markersize=3, lw=1, color="#e74c3c", label="Std")
ax.set_ylabel("LogP std")
ax.set_title("LogP Std")
add_phase_bg(ax)

plt.tight_layout()
savefig("03_logp_evolution.png")
print(f"  [03] logp_evolution")


# ============================================================
# Figure 4: TPSA evolution — 3 panel
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle("TPSA Evolution", fontsize=12, fontweight="bold")

ax = axes[0]
ax.plot(epochs, df["tpsa_mean"], "o-", markersize=3, lw=1, color="#2980b9")
ax.set_ylabel("TPSA mean")
add_phase_bg(ax)

ax = axes[1]
ax.plot(epochs, df["tpsa_median"], "s-", markersize=3, lw=1, color="#27ae60")
ax.set_ylabel("TPSA median")
add_phase_bg(ax)

ax = axes[2]
ax.plot(epochs, df["tpsa_std"], "D-", markersize=3, lw=1, color="#e74c3c")
ax.set_ylabel("TPSA std")
add_phase_bg(ax)

plt.tight_layout()
savefig("04_tpsa_evolution.png")
print(f"  [04] tpsa_evolution")


# ============================================================
# Figure 5: LogP + TPSA joint trajectory
# ============================================================
fig, ax = plt.subplots(figsize=(8, 7))
scatter = ax.scatter(
    df["logp_mean"][:PHASE_1_END],
    df["tpsa_mean"][:PHASE_1_END],
    c=range(1, PHASE_1_END + 1),
    cmap="viridis",
    s=50,
    edgecolors="black",
    linewidths=0.3,
    zorder=5,
    label=f"Epoch 1–{PHASE_1_END} (Normal)",
)
cbar = plt.colorbar(scatter, ax=ax, label="Epoch", shrink=0.85)
# Annotate a few key epochs
for i in [0, 4, 8, 13, 18, 23]:
    if i < PHASE_1_END:
        ax.annotate(
            str(i + 1),
            (df["logp_mean"].iloc[i], df["tpsa_mean"].iloc[i]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=7,
            alpha=0.8,
        )

# Epoch 25-30 dots for collapse phase
collapse_epochs_idx = list(range(PHASE_1_END, min(PHASE_2_END, len(epochs))))
ax.scatter(
    df["logp_mean"].iloc[collapse_epochs_idx],
    df["tpsa_mean"].iloc[collapse_epochs_idx],
    c="#e74c3c",
    marker="X",
    s=60,
    zorder=6,
    label=f"Epoch {PHASE_1_END+1}–{PHASE_2_END} (Collapse)",
)
for ci in collapse_epochs_idx:
    ax.annotate(
        str(ci + 1),
        (df["logp_mean"].iloc[ci], df["tpsa_mean"].iloc[ci]),
        textcoords="offset points",
        xytext=(7, -7),
        fontsize=7,
        color="#e74c3c",
        alpha=0.9,
    )

ax.set_xlabel("LogP Mean")
ax.set_ylabel("TPSA Mean")
ax.set_title("LogP vs TPSA Trajectory: Classifier Drives Hydrophobic Shift")
ax.legend(loc="upper left", framealpha=0.8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
savefig("05_logp_vs_tpsa_trajectory.png")
print(f"  [05] logp_vs_tpsa_trajectory")


# ============================================================
# Figure 6: Carbon fraction & Heteroatom (collapse detectors)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 7))
fig.suptitle("Collapse Detectors: Carbon Fraction & Heteroatom Counts", fontsize=12, fontweight="bold")

ax = axes[0, 0]
ax.plot(epochs, df["carbon_fraction"], "o-", markersize=3, lw=1.5, color="#8e44ad")
ax.axhline(0.85, color="#e74c3c", ls="--", alpha=0.6, label="warning=0.85")
ax.axhline(1.0, color="red", ls="--", alpha=0.4, label="all-carbon")
ax.set_ylabel("Carbon fraction")
ax.legend(fontsize=7)
add_phase_bg(ax)

ax = axes[0, 1]
ax.plot(epochs, df["hetero_atom_mean"], "o-", markersize=3, lw=1.5, color="#2c3e50")
ax.fill_between(
    epochs,
    df["hetero_atom_mean"] - df["hetero_atom_std"],
    df["hetero_atom_mean"] + df["hetero_atom_std"],
    alpha=0.15,
    color="#2c3e50",
)
ax.set_ylabel("Heteroatom count")
add_phase_bg(ax)

ax = axes[1, 0]
ax.plot(epochs, df["aromatic_ring_mean"], "o-", markersize=3, lw=1.5, color="#16a085")
ax.fill_between(
    epochs,
    df["aromatic_ring_mean"] - df["aromatic_ring_std"],
    df["aromatic_ring_mean"] + df["aromatic_ring_std"],
    alpha=0.15,
    color="#16a085",
)
ax.set_ylabel("Aromatic rings")
add_phase_bg(ax)

ax = axes[1, 1]
ax.plot(epochs, df["hba_mean"], "o-", markersize=3, lw=1, color="#2980b9", label="HBA")
ax.plot(epochs, df["hbd_mean"], "s-", markersize=3, lw=1, color="#e74c3c", label="HBD")
ax.set_ylabel("Count")
ax.set_title("H-Bond Acceptors / Donors")
ax.legend(fontsize=7)
add_phase_bg(ax)

plt.tight_layout()
savefig("06_collapse_detectors.png")
print(f"  [06] collapse_detectors")


# ============================================================
# Figure 7: Distribution alert signals
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 7))
fig.suptitle("Distribution Alert Signals", fontsize=12, fontweight="bold")

ax = axes[0, 0]
ax.plot(epochs, df["fraction_logp_high"], "o-", markersize=3, lw=1.2, color="#e67e22", label="logp_high")
ax.plot(epochs, df["fraction_tpsa_low"], "s-", markersize=3, lw=1.2, color="#2980b9", label="tpsa_low")
ax.plot(epochs, df["fraction_hba_zero_or_low"], "D-", markersize=3, lw=1.2, color="#8e44ad", label="hba_zero_or_low")
ax.set_ylabel("Fraction")
ax.set_title("Alert Fractions")
ax.legend(fontsize=7, loc="upper left")
add_phase_bg(ax)

ax = axes[0, 1]
ax.plot(epochs, df["descriptor_mmd_vs_baseline"], "o-", markersize=3, lw=1.5, color="#e74c3c")
ax.set_ylabel("MMD")
ax.set_title("Descriptor MMD vs Baseline")
add_phase_bg(ax)

ax = axes[1, 0]
ax.plot(epochs, df["ecfp4_mean_pairwise_similarity"], "o-", markersize=3, lw=1.2, color="#16a085", label="ECFP4")
ax.plot(epochs, df["fcfp4_mean_pairwise_similarity"], "s-", markersize=3, lw=1.2, color="#8e44ad", label="FCFP4")
ax.axhline(0.5, color="gray", ls="--", alpha=0.4, label="warning")
ax.set_ylabel("Mean pairwise similarity")
ax.set_title("Fingerprint Diversity (mode collapse detection)")
ax.legend(fontsize=7)
add_phase_bg(ax)

ax = axes[1, 1]
ax.plot(epochs, df["avg_loss"], "o-", markersize=3, lw=1, color="#2c3e50")
ax.set_ylabel("avg_loss")
ax.set_title("Average Loss")
add_phase_bg(ax)

plt.tight_layout()
savefig("07_distribution_alerts.png")
print(f"  [07] distribution_alerts")


# ============================================================
# Figure 8: Timing breakdown
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
fig.suptitle("Timing Breakdown per Epoch", fontsize=12, fontweight="bold")

ax = axes[0]
ax.plot(epochs, df["time_generate"], "o-", markersize=3, lw=1, color="#2980b9", label="Generate")
ax.plot(epochs, df["time_train"], "s-", markersize=3, lw=1, color="#e74c3c", label="Train")
ax.set_ylabel("Seconds")
ax.set_title("Generate vs Train time")
ax.legend(fontsize=7)
add_phase_bg(ax)

ax = axes[1]
ax.plot(epochs, df["time_annotate"], "o-", markersize=3, lw=0.8, color="#16a085", label="Annotate")
ax.plot(epochs, df["time_dataloader"], "s-", markersize=3, lw=0.8, color="#8e44ad", label="Dataloader")
ax.plot(epochs, df["time_filter"], "D-", markersize=3, lw=0.8, color="#f39c12", label="Filter")
ax.set_ylabel("Seconds")
ax.set_title("Small stages")
ax.legend(fontsize=6)
add_phase_bg(ax)

ax = axes[2]
total_time = (
    df["time_generate"] + df["time_train"] + df["time_annotate"]
    + df["time_dataloader"] + df["time_filter"] + df["time_reward"]
)
ax.plot(epochs, total_time / 60, "o-", markersize=3, lw=1.2, color="#2c3e50")
ax.set_ylabel("Minutes")
ax.set_title("Total epoch wall time")
ax.axhline(total_time[:PHASE_1_END].mean() / 60, color="gray", ls="--", alpha=0.5,
           label=f"avg={total_time[:PHASE_1_END].mean()/60:.1f} min")
ax.legend(fontsize=7)
add_phase_bg(ax)

plt.tight_layout()
savefig("08_timing_breakdown.png")
print(f"  [08] timing_breakdown")


# ============================================================
# Figure 9: Healthy phase zoom (epoch 1-24) — properties only
# ============================================================
healthy_mask = epochs <= PHASE_1_END
h_epochs = epochs[healthy_mask]
h_df = df[healthy_mask]

fig, axes = plt.subplots(4, 2, figsize=(14, 12))
fig.suptitle(f"Healthy Phase (Epoch 1–{PHASE_1_END}): Property Evolution", fontsize=13, fontweight="bold")

plot_specs = [
    (axes[0, 0], "rdkit_valid_ratio", "Valid ratio", "#2ecc71", None, (0, 1.05)),
    (axes[0, 1], "PPO_Loss", "PPO Loss", "#8e44ad", None, None),
    (axes[1, 0], "logp_mean", "LogP Mean", "#2980b9", None, None),
    (axes[1, 1], "tpsa_mean", "TPSA Mean", "#e74c3c", None, None),
    (axes[2, 0], "carbon_fraction", "Carbon Fraction", "#e67e22", (0.6, 0.95), None),
    (axes[2, 1], "hetero_atom_mean", "Heteroatoms", "#2c3e50", None, None),
    (axes[3, 0], "ecfp4_mean_pairwise_similarity", "ECFP4 intra-batch sim", "#16a085", None, None),
    (axes[3, 1], "descriptor_mmd_vs_baseline", "Descriptor MMD", "#c0392b", None, None),
]

for ax, col, title, color, ylim_override, ylim2_override in plot_specs:
    ax.plot(h_epochs, h_df[col], "o-", markersize=4, lw=1.5, color=color)
    ax.set_ylabel(title)
    if ylim_override:
        ax.set_ylim(*ylim_override)
    elif ylim2_override:
        ax.set_ylim(*ylim2_override)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
savefig("09_healthy_phase_detail.png")
print(f"  [09] healthy_phase_detail")


# ============================================================
# Figure 10: Phase 1 vs Phase 2 comparison — selected metrics
# ============================================================
fig, axes = plt.subplots(3, 3, figsize=(14, 11))
fig.suptitle("Phase 1 (Normal) vs Phase 2 (Collapse): Head-to-Head", fontsize=13, fontweight="bold")

# Build data for each phase
phase1 = df[df["epoch"] <= PHASE_1_END]
phase2 = df[(df["epoch"] > PHASE_1_END) & (df["epoch"] <= PHASE_2_END)]

comparisons = [
    ("rdkit_valid", "Valid count", axes[0, 0]),
    ("rdkit_valid_ratio", "Valid ratio", axes[0, 1]),
    ("PPO_Loss", "PPO Loss", axes[0, 2]),
    ("carbon_fraction", "Carbon fraction", axes[1, 0]),
    ("logp_mean", "LogP Mean", axes[1, 1]),
    ("tpsa_mean", "TPSA Mean", axes[1, 2]),
    ("hetero_atom_mean", "Heteroatoms", axes[2, 0]),
    ("ecfp4_mean_pairwise_similarity", "ECFP4 intra-sim", axes[2, 1]),
    ("descriptor_mmd_vs_baseline", "Descriptor MMD", axes[2, 2]),
]

for col, title, ax in comparisons:
    p1 = phase1[col].values
    p2 = phase2[col].values
    positions = [1, 2]
    parts = ax.violinplot([p1, p2], positions=positions, showmeans=True, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_alpha(0.5)
    parts["bodies"][0].set_facecolor("#2ecc71")
    parts["bodies"][1].set_facecolor("#f39c12")
    ax.set_xticks(positions)
    ax.set_xticklabels(["Phase 1\nNormal", "Phase 2\nCollapse"], fontsize=8)
    ax.set_ylabel(title, fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
savefig("10_phase1_vs_phase2.png")
print(f"  [10] phase1_vs_phase2")


# ============================================================
# Figure 11: Full-property dashboard (big overview)
# ============================================================
fig, axes = plt.subplots(5, 3, figsize=(16, 14))
fig.suptitle("Full Training Dashboard: 7-GPU Classifier-Only Experiment", fontsize=14, fontweight="bold")

dashboard_plots = [
    # Row 0
    (0, 0, "rdkit_valid", "Valid molecules", "#27ae60", None),
    (0, 1, "rdkit_valid_ratio", "Valid ratio", "#2ecc71", (0, 0.85)),
    (0, 2, "avg_reward_raw", "Avg raw reward", "#e67e22", None),
    # Row 1
    (1, 0, "PPO_Loss", "PPO Loss", "#8e44ad", None),
    (1, 1, "avg_loss", "Avg loss", "#2c3e50", None),
    (1, 2, "logp_mean", "LogP mean", "#2980b9", None),
    # Row 2
    (2, 0, "tpsa_mean", "TPSA mean", "#e74c3c", None),
    (2, 1, "carbon_fraction", "Carbon fraction", "#e67e22", (0.5, 1.05)),
    (2, 2, "hetero_atom_mean", "Heteroatoms", "#16a085", None),
    # Row 3
    (3, 0, "hba_mean", "HBA mean", "#2980b9", None),
    (3, 1, "hbd_mean", "HBD mean", "#8e44ad", None),
    (3, 2, "aromatic_ring_mean", "Aromatic rings", "#f39c12", None),
    # Row 4
    (4, 0, "ecfp4_mean_pairwise_similarity", "ECFP4 intra-sim", "#1abc9c", None),
    (4, 1, "fcfp4_mean_pairwise_similarity", "FCFP4 intra-sim", "#9b59b6", None),
    (4, 2, "descriptor_mmd_vs_baseline", "Descriptor MMD", "#e74c3c", None),
]

for row, col, column, title, color, ylim in dashboard_plots:
    ax = axes[row][col]
    ax.plot(epochs, df[column], "o-", markersize=2, lw=1, color=color)
    ax.set_title(title, fontsize=9)
    if ylim:
        ax.set_ylim(*ylim)
    add_phase_bg(ax)
    ax.tick_params(labelsize=7)

# phase legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=PHASE_COLORS["Phase 1: Normal"], alpha=0.3, label="Phase 1: Normal"),
    Patch(facecolor=PHASE_COLORS["Phase 2: Collapse"], alpha=0.3, label="Phase 2: Collapse"),
    Patch(facecolor=PHASE_COLORS["Phase 3: Dying"], alpha=0.3, label="Phase 3: Dying"),
    Patch(facecolor=PHASE_COLORS["Phase 4: Dead"], alpha=0.3, label="Phase 4: Dead"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=8, framealpha=0.9)

plt.tight_layout(rect=[0, 0.05, 1, 0.97])
savefig("11_full_dashboard.png")
print(f"  [11] full_dashboard")


# ============================================================
# Figure 12: HBA/HBD + aromatic rings with std bands
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle("Pharmacophore Properties with Std Bands", fontsize=12, fontweight="bold")

for ax, col_mean, col_std, title, color in [
    (axes[0], "hba_mean", "hba_std", "H-Bond Acceptors", "#2980b9"),
    (axes[1], "hbd_mean", "hbd_std", "H-Bond Donors", "#8e44ad"),
    (axes[2], "aromatic_ring_mean", "aromatic_ring_std", "Aromatic Rings", "#f39c12"),
]:
    ax.plot(epochs, df[col_mean], "o-", markersize=3, lw=1.5, color=color)
    ax.fill_between(
        epochs,
        df[col_mean] - df[col_std],
        df[col_mean] + df[col_std],
        alpha=0.15,
        color=color,
    )
    ax.set_ylabel(title)
    add_phase_bg(ax)

plt.tight_layout()
savefig("12_pharmacophore_bands.png")
print(f"  [12] pharmacophore_bands")


# ============================================================
# Figure 13: Reward + PPO Loss alignment (dual y-axis)
# ============================================================
fig, ax1 = plt.subplots(figsize=(12, 5))
ax2 = ax1.twinx()

ax1.plot(epochs, df["avg_reward_raw"], "D-", markersize=3, lw=1.2, color="#e67e22", label="Reward raw")
ax1.set_ylabel("Avg raw reward", color="#e67e22")
ax1.tick_params(axis="y", labelcolor="#e67e22")

ax2.plot(epochs, df["PPO_Loss"], "s-", markersize=3, lw=1.2, color="#8e44ad", label="PPO Loss")
ax2.set_ylabel("PPO Loss", color="#8e44ad")
ax2.tick_params(axis="y", labelcolor="#8e44ad")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

ax1.set_title("Reward vs PPO Loss: Reward Hacking Signal", fontsize=12, fontweight="bold")
add_phase_bg(ax1)

plt.tight_layout()
savefig("13_reward_vs_ppo_loss.png")
print(f"  [13] reward_vs_ppo_loss")


# ============================================================
# Figure 14: Fingerprint diversity trend (close-up)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(epochs, df["ecfp4_mean_pairwise_similarity"], "o-", markersize=4, lw=1.5, color="#1abc9c", label="ECFP4")
ax.plot(epochs, df["fcfp4_mean_pairwise_similarity"], "s-", markersize=4, lw=1.5, color="#9b59b6", label="FCFP4")
ax.axhline(0.3, color="gray", ls="--", alpha=0.4, label="diverse")
ax.axhline(0.5, color="#e74c3c", ls="--", alpha=0.5, label="warning")
ax.axhline(0.7, color="red", ls="--", alpha=0.5, label="danger")

# Healthy phase avg
h_ecfp4 = df["ecfp4_mean_pairwise_similarity"][:PHASE_1_END].mean()
h_fcfp4 = df["fcfp4_mean_pairwise_similarity"][:PHASE_1_END].mean()
ax.axhline(h_ecfp4, color="#1abc9c", ls=":", alpha=0.6, lw=0.8)
ax.axhline(h_fcfp4, color="#9b59b6", ls=":", alpha=0.6, lw=0.8)
ax.text(len(epochs) - 1, h_ecfp4 + 0.005, f"ECFP4 avg={h_ecfp4:.3f}", fontsize=7, color="#1abc9c", ha="right")
ax.text(len(epochs) - 1, h_fcfp4 - 0.012, f"FCFP4 avg={h_fcfp4:.3f}", fontsize=7, color="#9b59b6", ha="right")

ax.set_ylabel("Mean pairwise Tanimoto similarity")
ax.set_title("Fingerprint Diversity: Mode Collapse Indicator", fontsize=12, fontweight="bold")
ax.legend(fontsize=8)
add_phase_bg(ax)

plt.tight_layout()
savefig("14_fingerprint_diversity.png")
print(f"  [14] fingerprint_diversity")


# ============================================================
# Summary statistics
# ============================================================
print(f"\n{'='*60}")
print(f"Generated {n_plots} plots → {OUT_DIR}/")
print(f"{'='*60}")

# Key numbers for caption
p1 = df[df["epoch"] <= PHASE_1_END]
print(f"\nPhase 1 Summary (epoch 1-{PHASE_1_END}):")
print(f"  rdkit_valid:  {p1['rdkit_valid'].mean():.0f} ± {p1['rdkit_valid'].std():.0f}")
print(f"  valid_ratio:  {p1['rdkit_valid_ratio'].mean():.3f} ± {p1['rdkit_valid_ratio'].std():.3f}")
print(f"  LogP mean:    {p1['logp_mean'].mean():.2f} ± {p1['logp_mean'].std():.2f}")
print(f"  TPSA mean:    {p1['tpsa_mean'].mean():.1f} ± {p1['tpsa_mean'].std():.1f}")
print(f"  carbon frac:  {p1['carbon_fraction'].mean():.3f} ± {p1['carbon_fraction'].std():.3f}")
print(f"  heteroatoms:  {p1['hetero_atom_mean'].mean():.1f} ± {p1['hetero_atom_mean'].std():.1f}")
print(f"  ECFP4 sim:    {p1['ecfp4_mean_pairwise_similarity'].mean():.3f}")
print(f"  best epoch    14 (valid={p1['rdkit_valid'].max():.0f}, reward_raw={p1.loc[p1['rdkit_valid'].idxmax(), 'avg_reward_raw']:.3f})")

p4 = df[df["epoch"] > PHASE_3_END]
if len(p4) > 0:
    print(f"\nPhase 4 (epoch {PHASE_3_END+1}-{len(epochs)}): ALL DEAD")
    print(f"  Total epochs wasted: {len(p4)}")
    total_time = df["time_generate"].sum() + df["time_train"].sum()
    print(f"  Total wall time: ~{total_time/3600:.1f} hours")
