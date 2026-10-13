#!/usr/bin/env python3
"""分析 6gpu-dock-chembl_20260519_180540 训练结果"""
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

OUT_DIR = "/home/md666/data/disk6/DiffB110/runs/6gpu-dock-chembl_20260519_180540"
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

# ========== 2b. 逐 epoch 统计（仅有效分子） ==========
epoch_stats_list = []
for ep in sorted(mol_df['epoch'].unique()):
    ep_valid = valid_mol[valid_mol['epoch'] == ep]
    ep_all = mol_df[mol_df['epoch'] == ep]
    n_total = len(ep_all)
    n_valid = len(ep_valid)
    row = {'epoch': ep, 'generated_count': n_total, 'valid_count': n_valid,
           'valid_ratio': n_valid / n_total if n_total > 0 else 0.0}
    if n_valid > 0:
        # 仅 docking_raw 有意义（qed/sa/similarity/diversity 均为 0）
        row['avg_docking'] = ep_valid['docking_raw'].mean()
        row['avg_docking_reward'] = ep_valid['docking_reward'].mean()
        row['avg_reward_total_raw'] = ep_valid['reward_total_raw'].mean()
        row['avg_reward_total_norm'] = ep_valid['reward_total_norm'].mean()
    else:
        row['avg_docking'] = 0.0
        row['avg_docking_reward'] = 0.0
        row['avg_reward_total_raw'] = 0.0
        row['avg_reward_total_norm'] = 0.0
    # 额外：dock 成功（docking_raw != 0）比例
    docked = ep_all[ep_all['docking_raw'] < 0]
    row['docking_success_ratio'] = len(docked) / n_total if n_total > 0 else 0.0
    epoch_stats_list.append(row)
epoch_stats = pd.DataFrame(epoch_stats_list)

# 用修正值覆盖 df
df = df.merge(epoch_stats[['epoch', 'avg_docking']], on='epoch',
              suffixes=('_old', ''), how='left')
if 'avg_docking_old' in df.columns:
    df.drop(columns=['avg_docking_old'], inplace=True)

# ========== 3. 多面板趋势图 ==========
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
fig.suptitle("6GPU-Dock-ChEMBL Training Analysis (docking+chembl_prior, valid-only stats)",
             fontsize=14, fontweight='bold')

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
ax.set_title('Avg Docking Score (valid only)'); ax.legend(); ax.grid(True, alpha=0.3)

# 3c. 奖励分量趋势
ax = axes[0, 2]
ax.plot(epoch_stats['epoch'], epoch_stats['avg_docking_reward'], 'o-',
        label='Docking Reward', color='#A23B72')
ax.plot(epoch_stats['epoch'], epoch_stats['avg_reward_total_raw'], 's-',
        label='Total Raw Reward', color='#2E86AB')
ax.set_xlabel('Epoch'); ax.set_ylabel('Avg Reward')
ax.set_title('Reward Components (valid only)')
ax.legend(); ax.grid(True, alpha=0.3)

# 3d. 训练损失
ax = axes[1, 0]
ax.plot(df['epoch'], df['avg_loss'], 'o-', color='#3B1F2B', lw=2, markersize=6)
ax.set_xlabel('Epoch'); ax.set_ylabel('Avg Loss')
ax.set_title('Training Loss'); ax.grid(True, alpha=0.3)

# 3e. 原始对接分数（epoch 均值 ± 标准差）
ax = axes[1, 1]
epoch_means = []
epoch_stds = []
epoch_nums = []
for ep in sorted(valid_mol['epoch'].unique()):
    sub = valid_mol[valid_mol['epoch'] == ep]['docking_raw']
    if len(sub) > 0:
        epoch_means.append(sub.mean())
        epoch_stds.append(sub.std())
        epoch_nums.append(ep)
epoch_means = np.array(epoch_means)
epoch_stds = np.array(epoch_stds)
epoch_nums = np.array(epoch_nums)
ax.plot(epoch_nums, -epoch_means, 'o-', color='#A23B72', lw=2, markersize=6, label='Mean Docking')
ax.fill_between(epoch_nums, -(epoch_means - epoch_stds), -(epoch_means + epoch_stds),
                alpha=0.2, color='#A23B72', label='±1 STD')
ax.set_xlabel('Epoch'); ax.set_ylabel('-Docking Score (kcal/mol)')
ax.set_title('Docking Score Trend (mean ± std)')
ax.legend(); ax.grid(True, alpha=0.3)

# 3f. 失败类型分布（含 DOCKING_BACKEND 失败）
ax = axes[1, 2]
fail_types = ['fragmented_molecule', 'sanitize_failed', 'FAILURE_DOCKING_BACKEND', 'success']
fail_colors = ['#E71D36', '#FF9F1C', '#FF6B35', '#2EC4B6']
epochs = df['epoch'].values
bottom = np.zeros(len(epochs))
for ft, fc in zip(fail_types, fail_colors):
    vals = df['failure_counts'].apply(
        lambda x: json.loads(x.replace("'", '"')).get(ft, 0))
    ax.bar(epochs, vals, bottom=bottom, label=ft.replace('FAILURE_', ''), color=fc, alpha=0.8, width=0.6)
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

# 3h. 原始对接分数分布（每 epoch 箱线图，含 DOCKING_FAILURE 分离）
ax = axes[2, 1]
epoch_boxes = []
epoch_labels = []
for ep in sorted(valid_mol['epoch'].unique()):
    sub = valid_mol[valid_mol['epoch'] == ep]['docking_raw']
    if len(sub) > 0:
        epoch_boxes.append(sub.values)
        epoch_labels.append(str(int(ep)))
bp = ax.boxplot(epoch_boxes, labels=epoch_labels, showfliers=True, patch_artist=True, widths=0.6)
for patch in bp['boxes']:
    patch.set_facecolor('#A23B72')
    patch.set_alpha(0.6)
ax.set_xlabel('Epoch'); ax.set_ylabel('Docking Score (kcal/mol)')
ax.set_title('Docking Distribution per Epoch')
ax.grid(True, alpha=0.3, axis='y')

# 3i. Dock 成功率 vs 有效分子数
ax = axes[2, 2]
ax2 = ax.twinx()
l1 = ax.plot(df['epoch'], df['rdkit_valid'] / 144 * 100, 'o-', color='#2E86AB',
             lw=2, markersize=6, label='RDKit Valid (%)')
l2 = ax2.plot(epoch_stats['epoch'], epoch_stats['docking_success_ratio'] * 100,
              's-', color='#F18F01', lw=2, markersize=6, label='Docking Success (%)')
ax.set_xlabel('Epoch'); ax.set_ylabel('RDKit Valid (%)'); ax2.set_ylabel('Docking Success (%)')
ax.set_title('Validity & Docking Success Rate')
ax.grid(True, alpha=0.3)
lines = l1 + l2
ax.legend(lines, [l.get_label() for l in lines], loc='lower left')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "training_trends.png"), bbox_inches='tight', dpi=200)
plt.close()
print(f"[OK] 训练趋势图 -> {SAVE_DIR}/training_trends.png")

# ========== 4. 最佳分子列表 ==========
top_mols = valid_mol.sort_values('docking_raw', ascending=True).head(20).copy()
top_mols['rank'] = range(1, 21)

print("\n" + "="*90)
print("TOP 20 BEST DOCKING MOLECULES (across all epochs)")
print("="*90)
print(f"{'Rank':>5} {'Epoch':>6} {'Dock(kcal)':>11} {'DockReward':>11} {'TotalRaw':>9} {'TotalNorm':>10} {'SMILES'}")
print("-"*90)
for _, r in top_mols.iterrows():
    print(f"{int(r['rank']):>5} {int(r['epoch']):>6} {r['docking_raw']:>11.3f} "
          f"{r['docking_reward']:>11.4f} {r['reward_total_raw']:>9.3f} {r['reward_total_norm']:>10.3f} "
          f"{str(r['smiles'])[:55] if str(r['smiles']) != 'None' else 'INVALID'}")

# ========== 5. 最佳 epoch 汇总 ==========
print("\n" + "="*80)
print("BEST EPOCHS SUMMARY (by avg docking)")
print("="*80)
best_by_dock = df.nsmallest(5, 'avg_docking')[['epoch', 'rdkit_valid_ratio', 'avg_docking', 'avg_loss']]
best_by_dock.columns = ['epoch', 'valid_ratio', 'avg_docking', 'avg_loss']
print(best_by_dock.to_string(index=False))

# ========== 6. 每 epoch 优质分子比例 ==========
print("\n" + "="*80)
print("GOOD MOLECULE RATIO per Epoch (valid molecules only)")
print("="*80)
for ep in sorted(valid_mol['epoch'].unique()):
    sub = valid_mol[valid_mol['epoch'] == ep]
    total = len(sub)
    good = (sub['docking_raw'] < -7.0).sum()
    very_good = (sub['docking_raw'] < -9.0).sum()
    excellent = (sub['docking_raw'] < -11.0).sum()
    print(f"Epoch {int(ep):>2}: valid={total:>3} | dock<-7={good:>3} ({good/total*100:>5.1f}%) | "
          f"dock<-9={very_good:>3} ({very_good/total*100:>5.1f}%) | "
          f"dock<-11={excellent:>3} ({excellent/total*100:>5.1f}%)")

# ========== 7. 高级分析图 ==========
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
fig2.suptitle("6GPU-Dock-ChEMBL Advanced Analysis", fontsize=14, fontweight='bold')

# 7a. 最后 5 epoch 的 Top-5 分子 docking 对比
ax = axes2[0, 0]
latest_epochs = sorted(valid_mol['epoch'].unique())[-5:]
for ep in latest_epochs:
    ep_top = valid_mol[valid_mol['epoch'] == ep].nsmallest(5, 'docking_raw')
    ax.plot(range(1, 6), -ep_top['docking_raw'].values, 'o-', label=f'Epoch {int(ep)}', lw=1.5)
ax.set_xlabel('Rank'); ax.set_ylabel('-Docking Score (kcal/mol)')
ax.set_title('Top-5 Docking per Epoch (last 5 epochs)')
ax.set_xticks(range(1, 6))
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# 7b. 奖励稳定性（滑动窗口）
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

# 7d. 累积最佳 docking 分数
ax = axes2[1, 1]
all_sorted = valid_mol.sort_values('docking_raw', ascending=True)
cum_best = []
cum_min = float('inf')
for _, r in all_sorted.iterrows():
    if r['docking_raw'] < cum_min:
        cum_min = r['docking_raw']
    cum_best.append(cum_min)
ax.plot(range(len(all_sorted)), all_sorted['docking_raw'].values, '.', alpha=0.3,
        markersize=2, color='gray', label='All valid molecules')
ax.plot(range(len(all_sorted)), cum_best, '-', color='#E71D36', lw=2,
        label='Cumulative Best')
ax.set_xlabel('Valid Molecule Index (sorted by docking)')
ax.set_ylabel('Docking Score (kcal/mol)')
ax.set_title('Cumulative Best Docking Score')
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "advanced_analysis.png"), bbox_inches='tight', dpi=200)
plt.close()
print(f"[OK] 高级分析图 -> {SAVE_DIR}/advanced_analysis.png")

# ========== 8. Dock 失败率趋势 ==========
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
fig3.suptitle("Docking Failure Analysis", fontsize=14, fontweight='bold')

# 8a. Dock 失败数趋势
ax = axes3[0]
dock_fails = df['failure_counts'].apply(
    lambda x: json.loads(x.replace("'", '"')).get('FAILURE_DOCKING_BACKEND', 0))
sani_fails = df['failure_counts'].apply(
    lambda x: json.loads(x.replace("'", '"')).get('sanitize_failed', 0))
frag_fails = df['failure_counts'].apply(
    lambda x: json.loads(x.replace("'", '"')).get('fragmented_molecule', 0))
ax.stackplot(df['epoch'], [frag_fails, sani_fails, dock_fails],
             labels=['Fragment', 'Sanitize', 'Dock Backend'],
             colors=['#E71D36', '#FF9F1C', '#FF6B35'], alpha=0.7)
ax.set_xlabel('Epoch'); ax.set_ylabel('Failure Count')
ax.set_title('Failure Types over Epochs')
ax.legend(loc='upper right'); ax.grid(True, alpha=0.3, axis='y')

# 8b. 随 epoch 变化的成功分子中各 docking 分段的占比
ax = axes3[1]
ep_ranges = []
for ep in sorted(valid_mol['epoch'].unique()):
    sub = valid_mol[valid_mol['epoch'] == ep]
    total = len(sub)
    if total > 0:
        ep_ranges.append({
            'epoch': ep,
            'pct_gt_7': (sub['docking_raw'] < -7.0).sum() / total * 100,
            'pct_gt_9': (sub['docking_raw'] < -9.0).sum() / total * 100,
            'pct_gt_11': (sub['docking_raw'] < -11.0).sum() / total * 100
        })
ep_range_df = pd.DataFrame(ep_ranges)
ax.fill_between(ep_range_df['epoch'], ep_range_df['pct_gt_11'], alpha=0.3, color='green', label='<-11')
ax.fill_between(ep_range_df['epoch'], ep_range_df['pct_gt_11'], ep_range_df['pct_gt_9'],
                alpha=0.3, color='orange', label='-9 ~ -11')
ax.fill_between(ep_range_df['epoch'], ep_range_df['pct_gt_9'], ep_range_df['pct_gt_7'],
                alpha=0.3, color='red', label='-7 ~ -9')
ax.plot(ep_range_df['epoch'], ep_range_df['pct_gt_7'], 'o-', color='#A23B72', lw=1, label='% < -7')
ax.set_xlabel('Epoch'); ax.set_ylabel('% of valid molecules')
ax.set_title('Docking Score Distribution over Epochs')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "docking_failure_analysis.png"), bbox_inches='tight', dpi=200)
plt.close()
print(f"[OK] Dock 失败分析图 -> {SAVE_DIR}/docking_failure_analysis.png")

# ========== 9. 最后 epoch 的 Top 分子 ==========
print("\n" + "="*90)
print("EPOCH 25 TOP MOLECULES (docking score)")
print("="*90)
ep25 = valid_mol[valid_mol['epoch'] == 25].nsmallest(10, 'docking_raw')
print(f"{'Rank':>5} {'docking_raw':>11} {'docking_reward':>14} {'total_raw':>10} {'total_norm':>11} {'SMILES'}")
print("-"*90)
for i, (_, r) in enumerate(ep25.iterrows()):
    print(f"{i+1:>5} {r['docking_raw']:>11.3f} {r['docking_reward']:>14.4f} "
          f"{r['reward_total_raw']:>10.3f} {r['reward_total_norm']:>11.3f} "
          f"{str(r['smiles'])[:55] if str(r['smiles']) != 'None' else 'INVALID'}")

print(f"\n{'='*80}")
print(f"Figures saved to: {SAVE_DIR}/")
print(f"Total valid molecules analyzed: {len(valid_mol)}")
print(f"Total epochs: {len(df)}")
print(f"Training: {df['timestamp'].iloc[0]} ~ {df['timestamp'].iloc[-1]}")
