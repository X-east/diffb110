#!/usr/bin/env python3
"""逐 epoch 拆解 LogP 分箱分布、间隙与预测 Reward。

分析:
    7gpu-dock-ecfp4-fcfp4_20260526_115341
    dock=0, LE=0.3, ECFP4=0.3, FCFP4=0.1, LogP=0.3, scale=6.0
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

# ============================================================
ROOT = Path("/home/md666/data/disk6/DiffB110/OUT/7gpu-dock-ecfp4-fcfp4_20260526_115341")
EPOCH_DIR = ROOT / "epochs"
SAVE_DIR = ROOT / "analysis_figures"
SAVE_DIR.mkdir(exist_ok=True)

# ============================================================
# 常量
# ============================================================
LOGP_BINS = [-float("inf"), 0.5, 1.5, 2.5, 3.0, 3.5, 4.5, 5.5, float("inf")]
N_BINS = len(LOGP_BINS) - 1
BIN_LABELS = [
    "(-∞,0.5)", "[0.5,1.5)", "[1.5,2.5)", "[2.5,3.0)",
    "[3.0,3.5)", "[3.5,4.5)", "[4.5,5.5)", "(5.5,+∞)",
]
# ChEMBL 1.55M reference
CHEMBL_PROPS = np.array([0.044, 0.093, 0.206, 0.140, 0.143, 0.227, 0.108, 0.039])

SCALE = 6.0
MIN_REWARD = -2.0
MAX_REWARD = 2.0

# 颜色
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
          "#8172B2", "#64B5CD", "#937860", "#CC78BC"]
C_BLUE = "#4C72B0"
C_ORANGE = "#DD8452"
C_GREEN = "#55A868"
C_RED = "#C44E52"
C_PURPLE = "#8172B2"
C_GRAY = "#8C8C8C"

FS = 12
FS_TITLE = 14
FS_SUPTITLE = 16
FS_TICK = 10
FS_LEGEND = 10

# ============================================================
# 从所有 epoch 提取 LogP + 分箱数据
# ============================================================
print("提取各 epoch 分子 LogP + 分箱...")
all_epochs = list(range(1, 17))
epoch_data: dict[int, dict] = {}

for ep in all_epochs:
    csv_path = EPOCH_DIR / f"epoch_{ep:04d}" / "molecules.csv"
    if not csv_path.exists():
        print(f"  [skip] epoch {ep}: {csv_path} not found")
        continue

    mdf = pd.read_csv(csv_path)
    valid = mdf[mdf["rdkit_valid"] == 1].copy()
    if len(valid) == 0:
        epoch_data[ep] = {
            "n_valid": 0, "n_total": len(mdf),
            "logps": [], "rewards": [], "avg_reward": 0.0, "avg_logp": 0.0,
        }
        continue

    logps = []
    for smi in valid["smiles"]:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                logps.append(np.nan)
            else:
                logps.append(Descriptors.MolLogP(mol))
        except Exception:
            logps.append(np.nan)

    valid["computed_logp"] = logps
    valid_logp = valid.dropna(subset=["computed_logp"])

    # Laplace-smoothed batch proportions
    counts = np.zeros(N_BINS)
    for lp in valid_logp["computed_logp"]:
        idx = int(np.digitize(lp, LOGP_BINS)) - 1
        if 0 <= idx < N_BINS:
            counts[idx] += 1
    laplace_props = (counts + 1.0) / (counts.sum() + N_BINS)
    gaps = CHEMBL_PROPS - laplace_props
    raw_rewards = SCALE * gaps
    clamped_rewards = np.clip(raw_rewards, MIN_REWARD, MAX_REWARD)

    # Assign per-molecule reward based on its bin
    mol_rewards = []
    for lp in valid_logp["computed_logp"]:
        idx = int(np.digitize(lp, LOGP_BINS)) - 1
        if 0 <= idx < N_BINS:
            mol_rewards.append(clamped_rewards[idx])
        else:
            mol_rewards.append(-1.0)  # failure

    epoch_data[ep] = {
        "n_valid": len(valid_logp),
        "n_total": len(mdf),
        "logps": valid_logp["computed_logp"].tolist(),
        "avg_logp": np.mean(valid_logp["computed_logp"]),
        "counts_raw": counts.tolist(),
        "laplace_props": laplace_props.tolist(),
        "gaps": gaps.tolist(),
        "clamped_rewards": clamped_rewards.tolist(),
        "avg_reward": np.mean(mol_rewards) if mol_rewards else 0.0,
    }

    print(f"  Epoch {ep:2d}: {len(valid_logp)}/{len(mdf)} valid, "
          f"avg LogP={epoch_data[ep]['avg_logp']:.2f}, "
          f"avg LogP-reward={epoch_data[ep]['avg_reward']:.3f}")

# ============================================================
# 图 1: LogP 分箱比例堆叠 + 参考线（每个 epoch）
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(20, 12), gridspec_kw={"height_ratios": [3, 1]})
fig.suptitle("LogP 分箱比例逐 Epoch 演变 vs ChEMBL 参考", fontsize=FS_SUPTITLE, fontweight="bold", y=1.01)

# (0) 堆叠面积图
ax = axes[0]
stack_data = np.zeros((len(all_epochs), N_BINS))
for i, ep in enumerate(all_epochs):
    if ep in epoch_data and epoch_data[ep]["n_valid"] > 0:
        bin_counts = np.array(epoch_data[ep]["counts_raw"])
        total = bin_counts.sum()
        if total > 0:
            stack_data[i] = bin_counts / total * 100

# Plot stacked bars
bottom = np.zeros(len(all_epochs))
for j in range(N_BINS):
    ax.bar(all_epochs, stack_data[:, j], bottom=bottom, width=0.7,
           color=COLORS[j], alpha=0.85, edgecolor="white", linewidth=0.3,
           label=BIN_LABELS[j] if j < N_BINS else None)
    bottom += stack_data[:, j]

# Reference lines
ref_cum = np.cumsum(CHEMBL_PROPS * 100)
for j in range(N_BINS - 1):
    ax.axhline(y=ref_cum[j], color=COLORS[j], linestyle="--", alpha=0.4, linewidth=0.8)

ax.set_ylabel("Molecule %", fontsize=FS)
ax.set_xlabel("Epoch", fontsize=FS)
ax.set_title("LogP Bin Proportions (raw counts)", fontsize=FS_TITLE, fontweight="bold")
ax.set_xlim(0.5, len(all_epochs) + 0.5)
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=FS_LEGEND - 1, ncol=1)
ax.grid(True, alpha=0.3, axis="y")
ax.tick_params(labelsize=FS_TICK)

# (1) 每 epoch 均值 LogP + 有效分子数
ax2 = axes[1]
avg_logps = [epoch_data[ep]["avg_logp"] for ep in all_epochs if ep in epoch_data]
n_valids = [epoch_data[ep]["n_valid"] for ep in all_epochs if ep in epoch_data]
ax2.bar(all_epochs, n_valids, width=0.6, color=C_GREEN, alpha=0.4, label="Valid count")
ax2_ax = ax2.twinx()
ax2_ax.plot(all_epochs, avg_logps, marker="o", color=C_RED, linewidth=2, markersize=6, label="Mean LogP")
ax2_ax.axhline(y=3.0, color="gray", linestyle="--", alpha=0.5, label="ChEMBL median ~3.0")
ax2_ax.axhline(y=4.13, color="red", linestyle=":", alpha=0.4, label=f"Epoch 16 = {avg_logps[-1]:.1f}")
ax2.set_ylabel("Valid Molecules", fontsize=FS, color=C_GREEN)
ax2_ax.set_ylabel("Mean LogP", fontsize=FS, color=C_RED)
ax2.set_xlabel("Epoch", fontsize=FS)
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_ax.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=FS_LEGEND, loc="upper left")
ax2.set_xlim(0.5, len(all_epochs) + 0.5)
ax2.grid(True, alpha=0.3)
ax2.tick_params(labelsize=FS_TICK)

plt.tight_layout()
fig.savefig(SAVE_DIR / "05_logp_bin_stacked.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("[OK] 05_logp_bin_stacked.png")

# ============================================================
# 图 2: 每 bin 间隙 (gap) 热力图
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(14, 8))
fig.suptitle("Per-Bin Gap: ChEMBL Reference − Batch Proportion", fontsize=FS_SUPTITLE, fontweight="bold", y=1.01)

gap_matrix = np.zeros((N_BINS, len(all_epochs)))
for i, ep in enumerate(all_epochs):
    if ep in epoch_data:
        gap_matrix[:, i] = epoch_data[ep]["gaps"]

im = ax.imshow(gap_matrix, aspect="auto", cmap="RdBu_r", vmin=-0.3, vmax=0.3, origin="upper")
ax.set_xticks(range(len(all_epochs)))
ax.set_xticklabels([str(ep) for ep in all_epochs], fontsize=FS_TICK)
ax.set_yticks(range(N_BINS))
ax.set_yticklabels(BIN_LABELS, fontsize=FS_TICK)
ax.set_xlabel("Epoch", fontsize=FS)
ax.set_title("Gap: Ref − Batch (Blue=under/encourage, Red=over/penalize)", fontsize=FS_TITLE, fontweight="bold")

# Colorbar
cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
cbar.set_label("Gap (Ref − Batch)", fontsize=FS)
cbar.ax.tick_params(labelsize=FS_TICK)

# Annotate cells with gap values
for i in range(N_BINS):
    for j in range(len(all_epochs)):
        val = gap_matrix[i, j]
        color = "white" if abs(val) > 0.15 else "black"
        ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8,
                color=color, fontweight="bold" if abs(val) > 0.1 else "normal")

plt.tight_layout()
fig.savefig(SAVE_DIR / "06_logp_gap_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("[OK] 06_logp_gap_heatmap.png")

# ============================================================
# 图 3: Per-Bin Reward (clamped) 逐 epoch 线图
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(22, 12))
fig.suptitle("Per-Bin LogP Distribution Reward (clamped [-2,2], scale=6)", fontsize=FS_SUPTITLE, fontweight="bold", y=1.01)

reward_matrix = np.zeros((N_BINS, len(all_epochs)))
for i, ep in enumerate(all_epochs):
    if ep in epoch_data:
        reward_matrix[:, i] = epoch_data[ep]["clamped_rewards"]

for bin_idx in range(N_BINS):
    row, col = bin_idx // 4, bin_idx % 4
    ax = axes[row, col]
    ax.plot(all_epochs, reward_matrix[bin_idx], marker="o", color=COLORS[bin_idx],
            linewidth=2, markersize=5)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    # Annotate ref proportion
    ref_pct = CHEMBL_PROPS[bin_idx] * 100
    ax.set_title(f"{BIN_LABELS[bin_idx]} (ChEMBL {ref_pct:.1f}%)", fontsize=FS_TITLE, fontweight="bold")
    ax.set_ylabel("Reward", fontsize=FS)
    ax.set_xlabel("Epoch", fontsize=FS)
    ax.set_xlim(0.5, len(all_epochs) + 0.5)
    ax.set_ylim(-2.5, 2.5)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FS_TICK)

plt.tight_layout()
fig.savefig(SAVE_DIR / "07_logp_per_bin_reward.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("[OK] 07_logp_per_bin_reward.png")

# ============================================================
# 图 4: 三大关键阶段对比 — 分箱直方图
# ============================================================
key_epochs = [1, 5, 10, 14, 16]
fig, axes = plt.subplots(1, len(key_epochs), figsize=(20, 6), sharey=True)
fig.suptitle("LogP 分箱分布逐阶段对比", fontsize=FS_SUPTITLE, fontweight="bold", y=1.05)

for j, ep in enumerate(key_epochs):
    ax = axes[j]
    if ep not in epoch_data or epoch_data[ep]["n_valid"] == 0:
        ax.text(0.5, 0.5, f"Epoch {ep}\nno data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"Epoch {ep}", fontsize=FS_TITLE, fontweight="bold")
        continue

    # Raw counts normalized to 100%
    counts = np.array(epoch_data[ep]["counts_raw"])
    pcts = counts / counts.sum() * 100
    ref_pcts = CHEMBL_PROPS * 100

    x = np.arange(N_BINS)
    width = 0.35
    bars1 = ax.bar(x - width / 2, pcts, width, color=C_BLUE, alpha=0.7, label="Batch")
    bars2 = ax.bar(x + width / 2, ref_pcts, width, color=C_ORANGE, alpha=0.5, label="ChEMBL")

    # Annotate rewards on top
    rewards = epoch_data[ep]["clamped_rewards"]
    for bin_i in range(N_BINS):
        r_val = rewards[bin_i]
        r_color = C_GREEN if r_val > 0 else C_RED
        ax.text(bin_i, max(pcts[bin_i], ref_pcts[bin_i]) + 2,
                f"{r_val:+.1f}", ha="center", fontsize=8, color=r_color,
                fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{lo:.0f}" if lo > -np.inf else "-∞"
                        for lo in [b for b in LOGP_BINS[:-1]]], fontsize=FS_TICK - 1)
    ax.set_title(f"Epoch {ep} (n={epoch_data[ep]['n_valid']})", fontsize=FS_TITLE, fontweight="bold")
    ax.set_ylabel("Molecules %", fontsize=FS)
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(labelsize=FS_TICK - 1)
    if j == len(key_epochs) - 1:
        ax.legend(fontsize=FS_LEGEND, loc="upper right")

plt.tight_layout()
fig.savefig(SAVE_DIR / "08_logp_bin_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("[OK] 08_logp_bin_comparison.png")

# ============================================================
# 图 5: 汇总 — Batch Proportions vs Reference + Reward 表
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(18, 10))
ax.axis("off")

# Build table: rows=bins, cols per epoch: proportion + gap + reward
selected_eps = [1, 5, 10, 14, 16]
headers = ["Bin", "ChEMBL\nRef (%)"]
for ep in selected_eps:
    headers += [f"E{ep}\nProp(%)", f"E{ep}\nGap", f"E{ep}\nReward"]

table_data = []
for bin_idx in range(N_BINS):
    row = [BIN_LABELS[bin_idx], f"{CHEMBL_PROPS[bin_idx]*100:.1f}"]
    for ep in selected_eps:
        if ep in epoch_data:
            ed = epoch_data[ep]
            props = ed["laplace_props"][bin_idx] * 100
            gaps = ed["gaps"][bin_idx]
            rewards = ed["clamped_rewards"][bin_idx]
            row += [
                f"{props:.1f}",
                f"{gaps:+.3f}",
                f"{rewards:+.2f}",
            ]
        else:
            row += ["—", "—", "—"]
    table_data.append(row)

# Average reward row
avg_row = ["Avg Reward", "—"]
for ep in selected_eps:
    if ep in epoch_data:
        avg_row += ["", "", f"{epoch_data[ep]['avg_reward']:+.3f}"]
    else:
        avg_row += ["", "", "—"]
table_data.append(avg_row)

table = ax.table(cellText=table_data, colLabels=headers,
                 cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.0)

# Style header
for j in range(len(headers)):
    cell = table[0, j]
    cell.set_facecolor("#4472C4")
    cell.set_text_props(color="white", fontweight="bold", fontsize=10)

# Style reward cells
for i in range(1, N_BINS + 2):  # +1 for avg row
    for ep_idx, ep in enumerate(selected_eps):
        reward_col = 2 + ep_idx * 3 + 2  # the "Reward" column for this epoch
        c = table[i, reward_col]
        try:
            val = float(c.get_text().get_text())
            if val > 0.05:
                c.set_facecolor("#C6EFCE")
            elif val < -0.05:
                c.set_facecolor("#FFC7CE")
            else:
                c.set_facecolor("#FFF2CC")
        except (ValueError, AttributeError):
            pass

fig.suptitle("LogP Distribution Reward 分箱拆解 (Laplace smoothed, scale=6, clip [-2,2])",
             fontsize=FS_SUPTITLE, fontweight="bold", y=0.98)
plt.tight_layout()
fig.savefig(SAVE_DIR / "09_logp_detail_table.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("[OK] 09_logp_detail_table.png")

# ============================================================
# 终端汇总
# ============================================================
print()
print("=" * 80)
print("  LogP 分箱拆解 — 逐 Epoch 关键数据")
print("=" * 80)
print(f"  Bins: {BIN_LABELS}")
print(f"  ChEMBL ref: {[f'{p*100:.1f}%' for p in CHEMBL_PROPS]}")
print()
print(f"  {'Epoch':>5s}  {'Valid':>5s}  {'LogP Avg':>8s}  {'LPR Avg':>8s}  "
      f"  {'Low(<0.5)':>10s}  {'Mid(2.5-4.5)':>14s}  {'High(>5.5)':>10s}")
print(f"  {'-'*5}  {'-'*5}  {'-'*8}  {'-'*8}  "
      f"  {'-'*10}  {'-'*14}  {'-'*10}")
for ep in all_epochs:
    if ep not in epoch_data or epoch_data[ep]["n_valid"] == 0:
        continue
    ed = epoch_data[ep]
    low_pct = sum(ed["counts_raw"][0:1]) / sum(ed["counts_raw"]) * 100
    mid_pct = sum(ed["counts_raw"][3:6]) / sum(ed["counts_raw"]) * 100  # bins 3,4,5 → 2.5-4.5
    high_pct = sum(ed["counts_raw"][6:7]) / sum(ed["counts_raw"]) * 100  # bin 6 → 4.5-5.5
    super_high_pct = sum(ed["counts_raw"][6:]) / sum(ed["counts_raw"]) * 100  # >4.5
    print(f"  {ep:5d}  {ed['n_valid']:5d}  {ed['avg_logp']:8.2f}  "
          f"{ed['avg_reward']:8.3f}  "
          f"  {low_pct:8.1f}%  {mid_pct:12.1f}%  {super_high_pct:8.1f}% >4.5")

print()
print(f"  ChEMBL ref:       3.00        —           {(CHEMBL_PROPS[0]+CHEMBL_PROPS[1])*100:8.1f}%  "
      f"{sum(CHEMBL_PROPS[3:6])*100:12.1f}%  {sum(CHEMBL_PROPS[6:])*100:8.1f}% >4.5")

print()
print("  关键发现:")
print(f"    1. LogP 均值从 {epoch_data[1]['avg_logp']:.2f} 升至 {epoch_data[16]['avg_logp']:.2f}")
print(f"       → ChEMBL 中位数 ~3.0，Epoch 16 已超越")
print(f"    2. LogP 分布奖励均值: {epoch_data[1]['avg_reward']:.3f} → {epoch_data[16]['avg_reward']:.3f}")
print(f"       → 在 0 附近振荡，强度不足")
print(f"    3. 高 LogP 区域 (>4.5) bin 的惩罚力度已达 {epoch_data[16]['clamped_rewards'][6]:.1f}~{epoch_data[16]['clamped_rewards'][7]:.1f}")
print(f"       → 惩罚已是满的，但无法逆转趋势")
print(f"    4. LogP 分布本身并未严重偏离 ChEMBL——模型找到了一种高 LogP+低 LogP 混合的平衡策略")
print()
print(f"  Plots saved to: {SAVE_DIR}/")
for f in sorted(SAVE_DIR.iterdir()):
    if f.suffix == ".png" and f.name.startswith(("05_", "06_", "07_", "08_", "09_")):
        print(f"    {f.name}")
print("=" * 80)
