#!/usr/bin/env python3
"""分析 6gpu-dock_20260513_110544 训练结果"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import json
import os

matplotlib.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'figure.figsize': (12, 8),
})

OUT_DIR = "/home/md666/data/disk6/DiffB110/OUT/6gpu-dock_20260513_110544"
SAVE_DIR = os.path.join(OUT_DIR, "analysis_figures")
os.makedirs(SAVE_DIR, exist_ok=True)

# ========== 1. 读取训练摘要 ==========
df = pd.read_csv(os.path.join(OUT_DIR, "training_summary.csv"))
df['epoch'] = df['epoch'].astype(int)

# ========== 2. 从每个 epoch 的 molecules.csv 收集详细分子数据 ==========
all_mols = []
for _, row in df.iterrows():
    ep = int(row['epoch'])
    csv_path = os.path.join(OUT_DIR, f"epochs/epoch_{ep:04d}/molecules.csv")
    if os.path.exists(csv_path):
        mdf = pd.read_csv(csv_path)
        mdf['epoch'] = ep
        all_mols.append(mdf)
mol_df = pd.concat(all_mols, ignore_index=True)

# 标记有效分子
valid_mol = mol_df[mol_df['rdkit_valid'] == 1].copy()

# ========== 2b. 用逐分子数据重新计算正确均值（仅有效分子）==========
# training_summary.csv 的 avg_sa/avg_qed/avg_docking 包含无效分子（0.0），需修正
epoch_stats_list = []
for ep in sorted(mol_df['epoch'].unique()):
    ep_valid = valid_mol[valid_mol['epoch'] == ep]
    ep_all = mol_df[mol_df['epoch'] == ep]
    n_total = len(ep_all)
    n_valid = len(ep_valid)
    row = {'epoch': ep, 'generated_count': n_total, 'valid_count': n_valid,
           'valid_ratio': n_valid / n_total if n_total > 0 else 0.0}
    if n_valid > 0:
        row['avg_qed'] = ep_valid['qed_raw'].mean()
        row['avg_sa'] = ep_valid['sa_raw'].mean()
        row['avg_docking'] = ep_valid['docking_raw'].mean()
        row['avg_qed_reward'] = ep_valid['qed_reward'].mean()
        row['avg_sa_reward'] = ep_valid['sa_reward'].mean()
        row['avg_docking_reward'] = ep_valid['docking_reward'].mean()
    else:
        row['avg_qed'] = row['avg_sa'] = row['avg_docking'] = 0.0
        row['avg_qed_reward'] = row['avg_sa_reward'] = row['avg_docking_reward'] = 0.0
    epoch_stats_list.append(row)
epoch_stats = pd.DataFrame(epoch_stats_list)

# 用修正值覆盖 df 中对应的列
df = df.merge(epoch_stats[['epoch', 'avg_qed', 'avg_sa', 'avg_docking']], on='epoch',
              suffixes=('_old', ''), how='left')
# 删除旧的 buggy 列
for col in ['avg_qed_old', 'avg_sa_old', 'avg_docking_old']:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

# ========== 3. 多面板趋势图 ==========
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
fig.suptitle("6GPU-Dock Training Analysis (8GJL, UniDock, valid-only stats)", fontsize=16, fontweight='bold')

# 3a. RDKit 有效性
ax = axes[0, 0]
ax.plot(epoch_stats['epoch'], epoch_stats['valid_ratio'] * 100, 'o-', color='#2E86AB', lw=2, markersize=6)
ax.axhline(y=80, color='gray', ls='--', alpha=0.5, label='80% threshold')
ax.set_xlabel('Epoch'); ax.set_ylabel('RDKit Valid (%)')
ax.set_title('Molecular Validity'); ax.legend(); ax.grid(True, alpha=0.3)

# 3b. 平均对接分数（仅有效分子）
ax = axes[0, 1]
ax.plot(df['epoch'], -df['avg_docking'], 'o-', color='#A23B72', lw=2, markersize=6)
ax.axhline(y=7.0, color='gray', ls='--', alpha=0.5, label='-7.0 kcal/mol')
ax.set_xlabel('Epoch'); ax.set_ylabel('-Docking Score (kcal/mol)')
ax.set_title('Avg Docking (valid only)'); ax.legend(); ax.grid(True, alpha=0.3)

# 3c. 各奖励维度归一化值（同样仅有效分子）
ax = axes[0, 2]
# 从 per-molecule 数据重算奖励均值（仅有效分子）
ax.plot(epoch_stats['epoch'], epoch_stats['avg_qed_reward'], 'o-', label='QED Reward', color='#F18F01')
ax.plot(epoch_stats['epoch'], epoch_stats['avg_sa_reward'], 's-', label='SA Reward', color='#C73E1D')
ax.plot(epoch_stats['epoch'], epoch_stats['avg_docking_reward'], '^-', label='Docking Reward', color='#2E86AB')
ax.set_xlabel('Epoch'); ax.set_ylabel('Avg Reward')
ax.set_title('Reward Components (valid only)')
ax.legend(); ax.grid(True, alpha=0.3)

# 3d. 训练损失 & PPO Loss
ax = axes[1, 0]
ax.plot(df['epoch'], df['avg_loss'], 'o-', color='#3B1F2B', lw=2, markersize=6)
ax.set_xlabel('Epoch'); ax.set_ylabel('Avg Loss')
ax.set_title('Training Loss'); ax.grid(True, alpha=0.3)

# 3e. 原始性质：QED & SA（仅有效分子）
ax = axes[1, 1]
ax2 = ax.twinx()
l1 = ax.plot(df['epoch'], df['avg_qed'], 'o-', color='#F18F01', lw=2, markersize=6, label='QED')
l2 = ax2.plot(df['epoch'], df['avg_sa'], 's-', color='#C73E1D', lw=2, markersize=6, label='SA Score')
ax.set_xlabel('Epoch'); ax.set_ylabel('QED'); ax2.set_ylabel('SA Score')
ax.set_title('QED & SA Score Trends (valid only)')
ax.grid(True, alpha=0.3)
ax.legend(l1 + l2, ['QED', 'SA Score'], loc='upper left')

# 3f. 失败类型分布
ax = axes[1, 2]
fail_types = ['fragmented_molecule', 'sanitize_failed', 'success']
fail_colors = ['#E71D36', '#FF9F1C', '#2EC4B6']
epochs = df['epoch'].values
bottom = np.zeros(len(epochs))
for ft, fc in zip(fail_types, fail_colors):
    vals = df['failure_counts'].apply(lambda x: json.loads(x.replace("'", '"')).get(ft, 0))
    ax.bar(epochs, vals, bottom=bottom, label=ft, color=fc, alpha=0.8, width=0.6)
    bottom += vals
ax.set_xlabel('Epoch'); ax.set_ylabel('Count (out of 144)')
ax.set_title('Molecule Generation Outcomes')
ax.legend(); ax.grid(True, alpha=0.3, axis='y')

# 3g. KL 散度
ax = axes[2, 0]
ax.plot(df['epoch'], df['avg_kl'], 'o-', color='#6A4C93', lw=2, markersize=6)
ax.set_xlabel('Epoch'); ax.set_ylabel('Avg KL')
ax.set_title('KL Divergence (prior preservation)')
ax.grid(True, alpha=0.3)

# 3h. 原始对接分数分布（每 epoch 箱线图）
ax = axes[2, 1]
epoch_boxes = []
epoch_labels = []
for ep in sorted(valid_mol['epoch'].unique()):
    sub = valid_mol[valid_mol['epoch'] == ep]['docking_raw']
    if len(sub) > 0:
        epoch_boxes.append(sub.values)
        epoch_labels.append(str(int(ep)))
bp = ax.boxplot(epoch_boxes, labels=epoch_labels, showfliers=True, patch_artist=True,
                widths=0.6)
for patch in bp['boxes']:
    patch.set_facecolor('#A23B72')
    patch.set_alpha(0.6)
ax.set_xlabel('Epoch'); ax.set_ylabel('Docking Score (kcal/mol)')
ax.set_title('Docking Distribution per Epoch')
ax.grid(True, alpha=0.3, axis='y')

# 3i. 分子复杂度 vs 对接分数散点
ax = axes[2, 2]
# 用最后 5 epoch 的有效分子
last_epochs = valid_mol[valid_mol['epoch'] >= 20]
sc = ax.scatter(last_epochs['sa_raw'], -last_epochs['docking_raw'],
                c=last_epochs['qed_raw'], cmap='viridis', s=20, alpha=0.7)
ax.set_xlabel('SA Score (higher = harder to synthesize)')
ax.set_ylabel('-Docking Score (kcal/mol)')
ax.set_title('Quality Tradeoff (epoch >= 20)')
plt.colorbar(sc, ax=ax, label='QED')
ax.grid(True, alpha=0.3)
# 标注最佳区域
ax.axhline(y=8.0, color='green', ls='--', alpha=0.3, label='dock < -8')
ax.axvline(x=5.0, color='orange', ls='--', alpha=0.3, label='SA < 5')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "training_trends.png"), bbox_inches='tight', dpi=200)
plt.close()
print(f"[OK] 训练趋势图 -> {SAVE_DIR}/training_trends.png")

# ========== 4. 最佳分子列表 ==========
# 对所有分子的 docking_raw 排序，取 Top 20
top_mols = valid_mol.sort_values('docking_raw', ascending=True).head(20).copy()
top_mols['rank'] = range(1, 21)

print("\n" + "="*80)
print("TOP 20 BEST DOCKING MOLECULES (across all epochs)")
print("="*80)
print(f"{'Rank':>5} {'Epoch':>6} {'QED':>8} {'SA':>8} {'Docking':>9} {'Total':>8} {'SMILES'}")
print("-"*80)
for _, r in top_mols.iterrows():
    print(f"{int(r['rank']):>5} {int(r['epoch']):>6} {r['qed_raw']:>8.3f} "
          f"{r['sa_raw']:>8.3f} {r['docking_raw']:>9.3f} {r['reward_total_raw']:>8.3f} "
          f"{r['smiles'][:60] if r['smiles'] != 'None' else 'INVALID'}")

# ========== 5. 最佳 epoch 优质分子统计 ==========
print("\n" + "="*80)
print("BEST EPOCHS SUMMARY")
print("="*80)
best_epochs = df.nsmallest(5, 'avg_docking')[['epoch', 'rdkit_valid_ratio', 'avg_qed', 'avg_sa', 'avg_docking']]
print(best_epochs.to_string(index=False))

# ========== 6. 每 epoch 优质分子比例分析 ==========
print("\n" + "="*80)
print("GOOD MOLECULE RATIO (docking < -7.0) per Epoch")
print("="*80)
for ep in sorted(valid_mol['epoch'].unique()):
    sub = valid_mol[valid_mol['epoch'] == ep]
    total = len(sub)
    good = (sub['docking_raw'] < -7.0).sum()
    very_good = (sub['docking_raw'] < -9.0).sum()
    print(f"Epoch {int(ep):>2}: valid={total:>3}, dock<-7={good:>3} ({good/total*100:>5.1f}%), "
          f"dock<-9={very_good:>3} ({very_good/total*100:>5.1f}%)")

# ========== 7. 高级分析图 ==========
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
fig2.suptitle("6GPU-Dock Advanced Analysis (epoch >= 18)", fontsize=14, fontweight='bold')

# 7a. 最佳分子 top1-5 的 SMILES 性质雷达图
# 选 epoch 23 (最佳 docking 均值) 的前 5 个分子
ep23_top = valid_mol[(valid_mol['epoch'] == 23)].nsmallest(5, 'docking_raw').reset_index(drop=True)
categories = ['QED', 'SA (inv)', 'Docking (inv)']
# 归一化到 [0,1]
qed_vals = ep23_top['qed_raw'].values
sa_inv = 1 - (ep23_top['sa_raw'].values - 1) / 9  # SA 1~10 映射到 1~0
dock_inv = (-ep23_top['docking_raw'].values - 5) / 10  # dock -5~-15 映射到 0~1

ax = axes2[0, 0]
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]
for i in range(len(ep23_top)):
    vals = [qed_vals[i], sa_inv[i], dock_inv[i]]
    vals += vals[:1]
    ax.plot(angles, vals, 'o-', label=f"#{i+1} dock={ep23_top.iloc[i]['docking_raw']:.2f}", lw=1.5)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.set_title('Top-5 Molecules Radar (Epoch 23)')
ax.set_ylim(0, 1.1)
ax.legend(fontsize=8, loc='upper right')
ax.grid(True)

# 7b. 奖励稳定性分析（滑动窗口）
ax = axes2[0, 1]
window = 3
df_roll = df['reward_mean'].rolling(window, center=True).mean()
ax.plot(df['epoch'], df['reward_mean'], 'o-', alpha=0.4, label='Per Epoch', color='gray')
ax.plot(df['epoch'], df_roll, 'o-', lw=2, color='#2E86AB', label=f'{window}-Epoch Avg')
ax.axhline(y=0, color='red', ls='--', alpha=0.3)
ax.set_xlabel('Epoch'); ax.set_ylabel('Mean Reward')
ax.set_title('Reward Stability'); ax.legend(); ax.grid(True, alpha=0.3)

# 7c. PPO Loss vs KL 散度
ax = axes2[1, 0]
sc = ax.scatter(df['avg_kl'], df['avg_ppo_loss'], c=df['epoch'], cmap='plasma',
                s=80, alpha=0.8, edgecolors='black', linewidth=0.5)
for _, r in df.iterrows():
    ax.annotate(str(int(r['epoch'])), (r['avg_kl'], r['avg_ppo_loss']),
                fontsize=7, ha='center', va='bottom')
ax.set_xlabel('Avg KL Divergence'); ax.set_ylabel('PPO Loss')
ax.set_title('PPO Loss vs KL (label = epoch)')
plt.colorbar(sc, ax=ax, label='Epoch')
ax.grid(True, alpha=0.3)

# 7d. 累积最佳分子
ax = axes2[1, 1]
all_sorted = valid_mol.sort_values('docking_raw', ascending=True)
cum_best = []
cum_min = float('inf')
for _, r in all_sorted.iterrows():
    if r['docking_raw'] < cum_min:
        cum_min = r['docking_raw']
    cum_best.append(cum_min)
ax.plot(range(len(all_sorted)), all_sorted['docking_raw'].values, '.', alpha=0.3, markersize=2, color='gray')
ax.plot(range(len(all_sorted)), cum_best, '-', color='#E71D36', lw=2, label='Cumulative Best')
ax.set_xlabel('Valid Molecule Index (sorted by docking)')
ax.set_ylabel('Docking Score')
ax.set_title('Cumulative Best Docking Score')
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "advanced_analysis.png"), bbox_inches='tight', dpi=200)
plt.close()
print(f"[OK] 高级分析图 -> {SAVE_DIR}/advanced_analysis.png")

# ========== 8. 最后 epoch 的 top 分子展示 ==========
print("\n" + "="*80)
print("EPOCH 25 TOP MOLECULES (rank0)")
print("="*80)
ep25 = valid_mol[valid_mol['epoch'] == 25].nsmallest(10, 'docking_raw')
print(f"{'Rank':>5} {'docking_raw':>10} {'QED':>8} {'SA':>8} {'SMILES'}")
print("-"*80)
for i, (_, r) in enumerate(ep25.iterrows()):
    print(f"{i+1:>5} {r['docking_raw']:>10.3f} {r['qed_raw']:>8.3f} {r['sa_raw']:>8.3f} {r['smiles'][:60]}")

print(f"\n{'='*80}")
print(f"Figure saved to: {SAVE_DIR}/")
print(f"Total valid molecules analyzed: {len(valid_mol)}")
print(f"Total epochs: {len(df)}")
print(f"Training completed: {df['timestamp'].iloc[0]} ~ {df['timestamp'].iloc[-1]}")
