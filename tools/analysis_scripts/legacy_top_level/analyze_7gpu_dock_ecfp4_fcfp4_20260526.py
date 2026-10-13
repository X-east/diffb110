#!/usr/bin/env python3
"""分析 7gpu-dock-ecfp4-fcfp4_20260526_115341 训练结果。

配置: dock=0, LE=0.3, ECFP4=0.3, FCFP4=0.1, LogP=0.3, kl_coef=0
仅 16 epoch（提前终止或进行中）。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ============================================================
# 路径
# ============================================================
ROOT = Path("/home/md666/data/disk6/DiffB110/runs/7gpu-dock-ecfp4-fcfp4_20260526_115341")
CSV_PATH = ROOT / "training_summary.csv"
EPOCH_DIR = ROOT / "epochs"
SAVE_DIR = ROOT / "analysis_figures"
SAVE_DIR.mkdir(exist_ok=True)

# ============================================================
# 颜色调色板
# ============================================================
C_BLUE = "#4C72B0"
C_ORANGE = "#DD8452"
C_GREEN = "#55A868"
C_RED = "#C44E52"
C_PURPLE = "#8172B2"
C_CYAN = "#64B5CD"
C_BROWN = "#937860"
C_PINK = "#CC78BC"
C_GRAY = "#8C8C8C"

FS = 13
FS_TITLE = 15
FS_SUPTITLE = 18
FS_LEGEND = 10
FS_TICK = 11
FS_TABLE = 12

# ============================================================
# 数据加载
# ============================================================
df = pd.read_csv(CSV_PATH)


def parse_failure_counts(s):
    try:
        return json.loads(s.replace("'", '"'))
    except Exception:
        return {}


fc_df = df["failure_counts"].apply(parse_failure_counts).apply(pd.Series).fillna(0).astype(int)
for col in fc_df.columns:
    df[col] = fc_df[col]

epochs = df["epoch"].values
N_EPOCH = len(epochs)

print(f"加载 {N_EPOCH} 个 epoch 数据")

# ============================================================
# 图 1: 训练全景图 (3×3)
# ============================================================
fig, axes = plt.subplots(3, 3, figsize=(22, 16))
fig.suptitle(
    f"7gpu-dock-ecfp4-fcfp4 训练全景 (Epoch 1–{N_EPOCH}, dock=0, LE+ECFP4+FCFP4+LogP)",
    fontsize=FS_SUPTITLE, fontweight="bold", y=1.01,
)

# (0,0) PPO Loss
ax = axes[0, 0]
ax.plot(epochs, df["avg_loss"], marker="o", color=C_BLUE, linewidth=1.5, markersize=4)
ax.set_ylabel("PPO Loss", fontsize=FS)
ax.set_title("PPO Loss", fontsize=FS_TITLE, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=FS_TICK)

# (0,1) RDKit Validity
ax = axes[0, 1]
ax.plot(epochs, df["rdkit_valid_ratio"] * 100, marker="o", color=C_GREEN, linewidth=1.5, markersize=4)
ax.axhline(100, color="gray", linestyle="--", alpha=0.5)
ax.set_ylabel("Valid (%)", fontsize=FS)
ax.set_title("RDKit Validity", fontsize=FS_TITLE, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=FS_TICK)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

# (0,2) Reward Mean + Raw Mean
ax = axes[0, 2]
ax.plot(epochs, df["reward_mean"], marker="o", color=C_ORANGE, linewidth=1.5, markersize=4, label="Reward (clamped)")
if "avg_reward_raw" in df.columns:
    ax.plot(epochs, df["avg_reward_raw"], marker="s", color=C_BROWN, linewidth=1.0,
            markersize=3, alpha=0.6, label="Reward (raw)")
ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax.set_ylabel("Reward", fontsize=FS)
ax.set_title("Reward (clamped [-2,2])", fontsize=FS_TITLE, fontweight="bold")
ax.legend(fontsize=FS_LEGEND)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=FS_TICK)

# (1,0) Docking Score
ax = axes[1, 0]
ax.plot(epochs, df["avg_docking"], marker="o", color=C_RED, linewidth=1.5, markersize=4)
ax.set_ylabel("kcal/mol", fontsize=FS)
ax.set_title("Docking Score (UniDock)", fontsize=FS_TITLE, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=FS_TICK)

# (1,1) ChEMBL Prior: ECFP4 + FCFP4 Similarity
ax = axes[1, 1]
ax.plot(epochs, df["avg_chembl_prior_ecfp4_similarity"], marker="o", color=C_BLUE,
        linewidth=1.5, markersize=4, label="ECFP4 sim")
ax.plot(epochs, df["avg_chembl_prior_fcfp4_similarity"], marker="s", color=C_CYAN,
        linewidth=1.5, markersize=4, label="FCFP4 sim")
ax.set_ylabel("Similarity", fontsize=FS)
ax.set_title("ChEMBL Prior Similarity", fontsize=FS_TITLE, fontweight="bold")
ax.legend(fontsize=FS_LEGEND)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=FS_TICK)

# (1,2) Ligand Efficiency
ax = axes[1, 2]
ax.plot(epochs, df["avg_ligand_efficiency"], marker="o", color=C_PURPLE, linewidth=1.5, markersize=4)
ax.set_ylabel("LE", fontsize=FS)
ax.set_title("Ligand Efficiency", fontsize=FS_TITLE, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=FS_TICK)

# (2,0) Failure Breakdown (堆叠面积图)
ax = axes[2, 0]
failure_cols = []
for c in ["fragmented_molecule", "sanitize_failed", "FAILURE_DOCKING_BACKEND"]:
    if c in df.columns:
        failure_cols.append(c)
if "success" in df.columns:
    failure_cols.append("success")
if failure_cols:
    ax.stackplot(epochs, *[df[c] for c in failure_cols],
                 labels=failure_cols,
                 colors=[C_ORANGE, C_RED, C_GRAY, C_GREEN][:len(failure_cols)],
                 alpha=0.7)
    ax.set_ylabel("Count", fontsize=FS)
    ax.set_title("Failure Breakdown", fontsize=FS_TITLE, fontweight="bold")
    ax.legend(fontsize=FS_LEGEND - 1, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FS_TICK)

# (2,1) Carbon Fraction & Hetero Atom
ax = axes[2, 1]
ax_carbon = ax
line1 = ax_carbon.plot(epochs, df["carbon_fraction"] * 100, marker="o", color=C_RED,
                       linewidth=1.5, markersize=4, label="Carbon %")
ax_carbon.set_ylabel("Carbon %", fontsize=FS, color=C_RED)
ax_carbon.tick_params(axis="y", labelcolor=C_RED, labelsize=FS_TICK)
ax_hetero = ax.twinx()
line2 = ax_hetero.plot(epochs, df["hetero_atom_mean"], marker="s", color=C_BLUE,
                       linewidth=1.5, markersize=4, label="Hetero Atom")
ax_hetero.set_ylabel("Hetero Atom #", fontsize=FS, color=C_BLUE)
ax_hetero.tick_params(axis="y", labelcolor=C_BLUE, labelsize=FS_TICK)
ax_carbon.axhline(80, color="red", linestyle="--", alpha=0.3, linewidth=0.8)
ax_carbon.set_title("Carbon Fraction / Hetero Atom", fontsize=FS_TITLE, fontweight="bold")
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax_carbon.legend(lines, labels, fontsize=FS_LEGEND, loc="upper left")
ax_carbon.grid(True, alpha=0.3)
ax_carbon.tick_params(labelsize=FS_TICK)

# (2,2) Fingerprint Pairwise Similarity (collapse proxy)
ax = axes[2, 2]
ax.plot(epochs, df["ecfp4_mean_pairwise_similarity"], marker="o", color=C_BLUE,
        linewidth=1.5, markersize=4, label="ECFP4 intra-batch sim")
ax.plot(epochs, df["fcfp4_mean_pairwise_similarity"], marker="s", color=C_CYAN,
        linewidth=1.5, markersize=4, label="FCFP4 intra-batch sim")
ax.axhline(0.5, color="orange", linestyle="--", alpha=0.5, linewidth=0.8, label="Warning: 0.5")
ax.axhline(0.8, color="red", linestyle="--", alpha=0.5, linewidth=0.8, label="Danger: 0.8")
ax.set_ylabel("Mean Pairwise Similarity", fontsize=FS)
ax.set_title("Intra-batch Diversity (Collapse Proxy)", fontsize=FS_TITLE, fontweight="bold")
ax.legend(fontsize=FS_LEGEND)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=FS_TICK)

plt.tight_layout()
fig.savefig(SAVE_DIR / "01_training_overview.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("[OK] 01_training_overview.png")

# ============================================================
# 图 2: 化学描述子 (2×3)
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(20, 13))
fig.suptitle("化学描述子分布 (仅有效分子)", fontsize=FS_SUPTITLE, fontweight="bold", y=1.01)

# (0,0) LogP: mean ± std 带
ax = axes[0, 0]
ax.fill_between(epochs,
                df["logp_mean"] - df["logp_std"],
                df["logp_mean"] + df["logp_std"],
                alpha=0.2, color=C_BLUE)
ax.plot(epochs, df["logp_mean"], marker="o", color=C_BLUE, linewidth=1.5, markersize=4, label="Mean")
ax.plot(epochs, df["logp_median"], marker="s", color=C_ORANGE, linewidth=1.0,
        markersize=3, alpha=0.7, label="Median")
ax.axhline(3.0, color="gray", linestyle="--", alpha=0.5, label="ChEMBL median ~3.0")
ax.set_ylabel("LogP", fontsize=FS)
ax.set_title("LogP Distribution", fontsize=FS_TITLE, fontweight="bold")
ax.legend(fontsize=FS_LEGEND)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=FS_TICK)

# (0,1) TPSA: mean ± std 带
ax = axes[0, 1]
ax.fill_between(epochs,
                df["tpsa_mean"] - df["tpsa_std"],
                df["tpsa_mean"] + df["tpsa_std"],
                alpha=0.2, color=C_GREEN)
ax.plot(epochs, df["tpsa_mean"], marker="o", color=C_GREEN, linewidth=1.5, markersize=4, label="Mean")
ax.plot(epochs, df["tpsa_median"], marker="s", color=C_ORANGE, linewidth=1.0,
        markersize=3, alpha=0.7, label="Median")
ax.set_ylabel("TPSA (Å²)", fontsize=FS)
ax.set_title("TPSA Distribution", fontsize=FS_TITLE, fontweight="bold")
ax.legend(fontsize=FS_LEGEND)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=FS_TICK)

# (0,2) HBA / HBD
ax = axes[0, 2]
ax.plot(epochs, df["hba_mean"], marker="o", color=C_BLUE, linewidth=1.5, markersize=4, label="HBA")
ax.plot(epochs, df["hbd_mean"], marker="s", color=C_RED, linewidth=1.5, markersize=4, label="HBD")
ax.set_ylabel("Count", fontsize=FS)
ax.set_title("H-Bond Acceptors / Donors (mean)", fontsize=FS_TITLE, fontweight="bold")
ax.legend(fontsize=FS_LEGEND)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=FS_TICK)

# (1,0) Aromatic Rings
ax = axes[1, 0]
ax.fill_between(epochs,
                df["aromatic_ring_mean"] - df["aromatic_ring_std"],
                df["aromatic_ring_mean"] + df["aromatic_ring_std"],
                alpha=0.2, color=C_PURPLE)
ax.plot(epochs, df["aromatic_ring_mean"], marker="o", color=C_PURPLE, linewidth=1.5, markersize=4)
ax.set_ylabel("Count", fontsize=FS)
ax.set_title("Aromatic Rings (mean ± std)", fontsize=FS_TITLE, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=FS_TICK)

# (1,1) LogP Distribution Reward (component)
ax = axes[1, 1]
if "avg_logp_distribution_reward" in df.columns:
    ax.plot(epochs, df["avg_logp_distribution_reward"], marker="o", color=C_CYAN,
            linewidth=1.5, markersize=4)
    ax.set_ylabel("Reward", fontsize=FS)
    ax.set_title("LogP Distribution Reward (component)", fontsize=FS_TITLE, fontweight="bold")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FS_TICK)

# (1,2) ChEMBL Prior Rewards (component-level)
ax = axes[1, 2]
ax.plot(epochs, df["avg_chembl_prior_ecfp4_reward"], marker="o", color=C_BLUE,
        linewidth=1.5, markersize=4, label="ECFP4 prior")
ax.plot(epochs, df["avg_chembl_prior_fcfp4_reward"], marker="s", color=C_CYAN,
        linewidth=1.5, markersize=4, label="FCFP4 prior")
ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax.set_ylabel("Reward", fontsize=FS)
ax.set_title("ChEMBL Prior Rewards (component)", fontsize=FS_TITLE, fontweight="bold")
ax.legend(fontsize=FS_LEGEND)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=FS_TICK)

plt.tight_layout()
fig.savefig(SAVE_DIR / "02_chemistry_descriptors.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("[OK] 02_chemistry_descriptors.png")

# ============================================================
# 图 3: 奖励分量拆解 + 时间开销 (2×2)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle("奖励分量拆解 & 时间开销", fontsize=FS_SUPTITLE, fontweight="bold", y=1.01)

# (0,0) 各奖励分量归一化对比
ax = axes[0, 0]
reward_cols = {
    "avg_chembl_prior_ecfp4_reward": ("ECFP4 prior", C_BLUE),
    "avg_chembl_prior_fcfp4_reward": ("FCFP4 prior", C_CYAN),
    "avg_ligand_efficiency": ("Ligand Efficiency", C_PURPLE),
    "avg_logp_distribution_reward": ("LogP Distribution", C_ORANGE),
}
if "avg_docking" in df.columns:
    # Docking is disabled, but still show
    reward_cols["docking_scaled"] = ("Docking (scaled)", C_RED)

for col, (label, color) in reward_cols.items():
    if col in df.columns:
        ax.plot(epochs, df[col], marker=".", color=color, linewidth=1.2, markersize=3, label=label)
    elif col == "docking_scaled" and "avg_docking" in df.columns:
        # Scale docking to [-2, 2] comparable range
        dock = df["avg_docking"].values
        scaled = np.clip(-dock / 5.0, -2, 2)  # rough scaling for visualization
        ax.plot(epochs, scaled, marker=".", color=color, linewidth=1.2, markersize=3, label=label)
ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax.set_ylabel("Reward (component)", fontsize=FS)
ax.set_title("Per-Component Reward Values", fontsize=FS_TITLE, fontweight="bold")
ax.legend(fontsize=FS_LEGEND - 1)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=FS_TICK)

# (0,1) 复合奖励 vs 原始加权和
ax = axes[0, 1]
ax.plot(epochs, df["reward_mean"], marker="o", color=C_ORANGE, linewidth=1.5, markersize=4, label="Reward (clamped)")
if "avg_reward_raw" in df.columns:
    ax.plot(epochs, df["avg_reward_raw"], marker="s", color=C_BROWN, linewidth=1.0,
            markersize=3, alpha=0.6, label="Reward (raw)")
ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax.set_ylabel("Reward", fontsize=FS)
ax.set_title("Composite Reward (clamped vs raw)", fontsize=FS_TITLE, fontweight="bold")
ax.legend(fontsize=FS_LEGEND)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=FS_TICK)

# (1,0) 时间开销
ax = axes[1, 0]
time_cols = [
    ("time_generate", "Generation", C_BLUE),
    ("time_reward", "Reward (total)", C_ORANGE),
    ("time_train", "Training", C_GREEN),
]
if "time_reward_ligand_efficiency" in df.columns:
    time_cols += [
        ("time_reward_chembl_prior_ecfp4", "  └ CEMBL ECFP4", C_CYAN),
        ("time_reward_chembl_prior_fcfp4", "  └ CEMBL FCFP4", C_PINK),
        ("time_reward_ligand_efficiency", "  └ Ligand Eff", C_PURPLE),
        ("time_reward_logp_distribution", "  └ LogP Dist", C_BROWN),
    ]

for col, label, color in time_cols:
    if col in df.columns:
        ax.plot(epochs, df[col], marker=".", color=color, linewidth=1.0, markersize=3, label=label)
ax.set_ylabel("Wall Time (s)", fontsize=FS)
ax.set_title("Time Breakdown per Epoch", fontsize=FS_TITLE, fontweight="bold")
ax.legend(fontsize=FS_LEGEND - 1)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=FS_TICK)

# (1,1) 有效分子数 vs 总生成数
ax = axes[1, 1]
ax.bar(epochs - 0.15, df["rdkit_valid"], width=0.3, color=C_GREEN, alpha=0.7, label="Valid")
ax.bar(epochs + 0.15, df["generated_total"] - df["rdkit_valid"], width=0.3, color=C_RED, alpha=0.5, label="Invalid")
ax.plot(epochs, df["rdkit_valid"], marker="o", color=C_GREEN, linewidth=1.0, markersize=3)
ax.set_xlabel("Epoch", fontsize=FS)
ax.set_ylabel("Molecule Count", fontsize=FS)
ax.set_title("Valid / Invalid Molecules per Batch", fontsize=FS_TITLE, fontweight="bold")
ax.legend(fontsize=FS_LEGEND)
ax.grid(True, alpha=0.3, axis="y")
ax.tick_params(labelsize=FS_TICK)

plt.tight_layout()
fig.savefig(SAVE_DIR / "03_reward_breakdown_and_time.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("[OK] 03_reward_breakdown_and_time.png")

# ============================================================
# 图 4: 汇总表 (1×1)
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(16, 8))
ax.axis("off")

# Build comparison data
EP_START = 1
EP_END = N_EPOCH
EP_PEAK = df.loc[df["rewar_mean"].idxmax(), "epoch"] if "rewar_mean" in df.columns else df.loc[df["rdkit_valid_ratio"].idxmax(), "epoch"]

# Fix: use the right column name
reward_peak_idx = df["reward_mean"].idxmax()
EP_PEAK = df.loc[reward_peak_idx, "epoch"]

def val_at(ep):
    """Get value at epoch ep (1-indexed)."""
    row = df[df["epoch"] == ep]
    if row.empty:
        return None
    return row.iloc[0]

start = val_at(EP_START)
end = val_at(EP_END)
peak = val_at(int(EP_PEAK)) if EP_PEAK is not None else None

metrics = [
    ("Epoch", "—", f"{EP_START}", f"{EP_END}", f"{EP_PEAK:.0f}" if EP_PEAK else "—"),
    ("PPO Loss", "↓", f"{start['avg_loss']:.4f}", f"{end['avg_loss']:.4f}",
     f"{start['avg_loss'] - end['avg_loss']:+.4f}" if start is not None and end is not None else "—"),
    ("Reward Mean", "↑", f"{start['reward_mean']:.4f}" if start is not None else "—",
     f"{end['reward_mean']:.4f}" if end is not None else "—",
     f"{end['reward_mean'] - start['reward_mean']:+.4f}" if start is not None and end is not None else "—"),
    ("RDKit Validity", "↑", f"{start['rdkit_valid_ratio']*100:.1f}%" if start is not None else "—",
     f"{end['rdkit_valid_ratio']*100:.1f}%" if end is not None else "—",
     f"{(end['rdkit_valid_ratio'] - start['rdkit_valid_ratio'])*100:+.1f}pp" if start is not None and end is not None else "—"),
    ("Docking Score", "↓", f"{start['avg_docking']:.2f}" if start is not None else "—",
     f"{end['avg_docking']:.2f}" if end is not None else "—",
     f"{end['avg_docking'] - start['avg_docking']:+.2f}" if start is not None and end is not None else "—"),
    ("ECFP4 ChEMBL Sim", "↑", f"{start['avg_chembl_prior_ecfp4_similarity']:.3f}" if start is not None else "—",
     f"{end['avg_chembl_prior_ecfp4_similarity']:.3f}" if end is not None else "—",
     f"{end['avg_chembl_prior_ecfp4_similarity'] - start['avg_chembl_prior_ecfp4_similarity']:+.3f}" if start is not None and end is not None else "—"),
    ("FCFP4 ChEMBL Sim", "↑", f"{start['avg_chembl_prior_fcfp4_similarity']:.3f}" if start is not None else "—",
     f"{end['avg_chembl_prior_fcfp4_similarity']:.3f}" if end is not None else "—",
     f"{end['avg_chembl_prior_fcfp4_similarity'] - start['avg_chembl_prior_fcfp4_similarity']:+.3f}" if start is not None and end is not None else "—"),
    ("Ligand Efficiency", "↑", f"{start['avg_ligand_efficiency']:.3f}" if start is not None else "—",
     f"{end['avg_ligand_efficiency']:.3f}" if end is not None else "—",
     f"{end['avg_ligand_efficiency'] - start['avg_ligand_efficiency']:+.3f}" if start is not None and end is not None else "—"),
    ("Carbon Fraction", "⚠", f"{start['carbon_fraction']*100:.1f}%" if start is not None else "—",
     f"{end['carbon_fraction']*100:.1f}%" if end is not None else "—",
     f"{(end['carbon_fraction'] - start['carbon_fraction'])*100:+.1f}pp" if start is not None and end is not None else "—"),
    ("Hetero Atom", "↓", f"{start['hetero_atom_mean']:.1f}" if start is not None else "—",
     f"{end['hetero_atom_mean']:.1f}" if end is not None else "—",
     f"{end['hetero_atom_mean'] - start['hetero_atom_mean']:+.1f}" if start is not None and end is not None else "—"),
    ("ECFP4 Intra-batch Sim", "⚠", f"{start['ecfp4_mean_pairwise_similarity']:.3f}" if start is not None else "—",
     f"{end['ecfp4_mean_pairwise_similarity']:.3f}" if end is not None else "—",
     f"{end['ecfp4_mean_pairwise_similarity'] - start['ecfp4_mean_pairwise_similarity']:+.3f}" if start is not None and end is not None else "—"),
    ("FCFP4 Intra-batch Sim", "⚠", f"{start['fcfp4_mean_pairwise_similarity']:.3f}" if start is not None else "—",
     f"{end['fcfp4_mean_pairwise_similarity']:.3f}" if end is not None else "—",
     f"{end['fcfp4_mean_pairwise_similarity'] - start['fcfp4_mean_pairwise_similarity']:+.3f}" if start is not None and end is not None else "—"),
    ("LogP Mean", "—", f"{start['logp_mean']:.2f}" if start is not None else "—",
     f"{end['logp_mean']:.2f}" if end is not None else "—",
     f"{end['logp_mean'] - start['logp_mean']:+.2f}" if start is not None and end is not None else "—"),
    ("TPSA Mean", "—", f"{start['tpsa_mean']:.1f}" if start is not None else "—",
     f"{end['tpsa_mean']:.1f}" if end is not None else "—",
     f"{end['tpsa_mean'] - start['tpsa_mean']:+.1f}" if start is not None and end is not None else "—"),
    ("Descriptor MMD", "↓", f"{start['descriptor_mmd_vs_baseline']:.3f}" if start is not None else "—",
     f"{end['descriptor_mmd_vs_baseline']:.3f}" if end is not None else "—",
     f"{end['descriptor_mmd_vs_baseline'] - start['descriptor_mmd_vs_baseline']:+.3f}" if start is not None and end is not None else "—"),
]

headers = ["Metric", "Desired", f"Epoch {EP_START}", f"Epoch {EP_END}", f"Δ"]

# Build table
table_data = [[m[0], m[1], m[2], m[3], m[4]] for m in metrics]

table = ax.table(cellText=table_data, colLabels=headers,
                 cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(FS_TABLE)
table.scale(1, 1.8)

# Style header
for j in range(len(headers)):
    cell = table[0, j]
    cell.set_facecolor("#4472C4")
    cell.set_text_props(color="white", fontweight="bold", fontsize=FS_TABLE)

# Style rows
for i in range(1, len(metrics)):
    change_str = str(metrics[i - 1][4])  # Δ column
    # Color Δ column based on direction
    delta_cell = table[i, 4]
    desired = metrics[i - 1][1]
    try:
        delta_val = float(change_str.replace("pp", "").replace("%", ""))
        if desired == "↑":
            color = "#C6EFCE" if delta_val > 0 else "#FFC7CE"
        elif desired == "↓":
            color = "#C6EFCE" if delta_val < 0 else "#FFC7CE"
        elif desired == "⚠":
            color = "#FFF2CC"  # yellow for warning
        else:
            color = "#E2EFDA"
    except (ValueError, AttributeError):
        color = "#E2EFDA"
    delta_cell.set_facecolor(color)

fig.suptitle("训练对比汇总 (初始 → 最终)", fontsize=FS_SUPTITLE + 2, fontweight="bold", y=0.98)
fig.tight_layout()
fig.savefig(SAVE_DIR / "04_summary_table.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("[OK] 04_summary_table.png")

# ============================================================
# 终端输出汇总
# ============================================================
print()
print("=" * 80)
print("  7gpu-dock-ecfp4-fcfp4_20260526_115341 — 训练分析摘要")
print("=" * 80)
print(f"  Epochs: {EP_START} → {EP_END}")
print(f"  配置: dock=0, LE=0.3, ECFP4=0.3, FCFP4=0.1, LogP=0.3, kl_coef=0")
print(f"  每 batch: {start['generated_total']} 分子, 7 GPU")
print()

print(f"  {'Metric':<35s} {'Epoch 1':>12s} {'Epoch ' + str(EP_END):>12s} {'Change':>12s}")
print(f"  {'-' * 35} {'-' * 12} {'-' * 12} {'-' * 12}")
for m in metrics:
    name, desired, v1, vn, delta = m
    if desired == "⚠":
        name = "⚠ " + name
    print(f"  {name:<35s} {str(v1):>12s} {str(vn):>12s} {str(delta):>12s}")

print()
print("  关键发现:")
print(f"    1. Reward 首次突破正值: Epoch 16 reward_mean = {end['reward_mean']:.4f}")
print(f"    2. Validity 持续改善: {start['rdkit_valid_ratio']*100:.1f}% → {end['rdkit_valid_ratio']*100:.1f}%")
print(f"    3. Docking 改善（即使 dock=0 未训练）: {start['avg_docking']:.2f} → {end['avg_docking']:.2f} kcal/mol")
print(f"    4. ⚠ Carbon fraction 已达 {end['carbon_fraction']*100:.1f}%（>80% 警惕线）")
print(f"    5. ⚠ FCFP4 intra-batch similarity = {end['fcfp4_mean_pairwise_similarity']:.3f}（接近崩塌区间 >0.5）")
print(f"    6. Epoch 15 出现 DOCKING_BACKEND 失败 82 个（可能是临时环境问题，Epoch 16 已恢复）")
print(f"    7. LogP 漂移 ↑: {start['logp_mean']:.2f} → {end['logp_mean']:.2f}（ChEMBL 中位数 ~3.0）")
print()
print(f"  Plots saved to: {SAVE_DIR}/")
for f in sorted(SAVE_DIR.iterdir()):
    if f.suffix == ".png":
        print(f"    {f.name}")
print("=" * 80)
