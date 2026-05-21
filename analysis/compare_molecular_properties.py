#!/usr/bin/env python3
"""
对比 DDPO epoch 20 输出分子 与 原始生成分子的属性分布。
"""

import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
# 注册系统 CJK 字体
for _font_path in [
    '/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
]:
    if os.path.exists(_font_path):
        try:
            fm.fontManager.addfont(_font_path)
            _font_name = fm.FontProperties(fname=_font_path).get_name()
            plt.rcParams['font.sans-serif'] = [_font_name] + plt.rcParams.get('font.sans-serif', [])
            print(f"  [字体] 已加载: {_font_name}")
            break
        except Exception as e:
            print(f"  [字体] 加载 {_font_path} 失败: {e}")
plt.rcParams['axes.unicode_minus'] = False
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, Crippen, rdMolDescriptors

# ============================================================
# 配置
# ============================================================
EPOCH20_CSV = "/home/md666/data/disk6/DiffB110/OUT/6gpu-chembl-only_20260521_154957/epochs/epoch_0020/molecules.csv"
REF_CSV = "/home/md666/data/disk6/DiffB110/OUT/260407_generate_ligands_10000/guacamol_eval/260407generated_10000_properties.csv"
OUTPUT_DIR = "/home/md666/data/disk6/DiffB110/analysis/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 要计算的属性（统一命名以匹配 reference CSV）
PROP_NAMES = {
    "MolWt": "分子量 (MW)",
    "LogP": "LogP",
    "TPSA": "TPSA (Å²)",
    "HBA": "HBA",
    "HBD": "HBD",
    "RotBonds": "可旋转键数",
    "NumRings": "环数",
    "NumAromaticRings": "芳香环数",
    "QED": "QED",
    "SA_Score": "SA Score",
}


def compute_properties(smiles_list):
    """对 SMILES 列表计算分子属性"""
    results = {k: [] for k in PROP_NAMES}
    total = len(smiles_list)
    for i, smi in enumerate(smiles_list):
        if (i + 1) % 100 == 0:
            print(f"  Processing {i+1}/{total}...", flush=True)
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            for k in results:
                results[k].append(np.nan)
            continue
        results["MolWt"].append(Descriptors.ExactMolWt(mol))
        results["LogP"].append(Crippen.MolLogP(mol))
        results["TPSA"].append(Descriptors.TPSA(mol))
        results["HBA"].append(rdMolDescriptors.CalcNumHBA(mol))
        results["HBD"].append(rdMolDescriptors.CalcNumHBD(mol))
        results["RotBonds"].append(rdMolDescriptors.CalcNumRotatableBonds(mol))
        results["NumRings"].append(rdMolDescriptors.CalcNumRings(mol))
        results["NumAromaticRings"].append(rdMolDescriptors.CalcNumAromaticRings(mol))
        try:
            results["QED"].append(QED.qed(mol))
        except Exception:
            results["QED"].append(np.nan)
        try:
            from rdkit.Contrib.SA_Score import sascorer
            raw_score = sascorer.calculateScore(mol)
            # 归一化到 0-1 范围以匹配参考数据格式
            results["SA_Score"].append(raw_score / 10.0)
        except ImportError:
            results["SA_Score"].append(np.nan)
    return results


def load_epoch20_smiles(csv_path):
    """从 epoch CSV 读取所有有效 SMILES"""
    smiles = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            smi = row.get('smiles', '').strip()
            valid = row.get('rdkit_valid', '1').strip()
            if smi and valid == '1':
                smiles.append(smi)
    print(f"Epoch20: 读取到 {len(smiles)} 个有效 SMILES")
    return smiles


def load_ref_properties(csv_path):
    """加载参考 CSV 的属性"""
    ref = {k: [] for k in PROP_NAMES}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in PROP_NAMES:
                val = row.get(k, '').strip()
                ref[k].append(float(val) if val else np.nan)
    print(f"Reference: 读取到 {len(ref['QED'])} 个分子")
    return ref


def plot_comparison(epoch20_props, ref_props):
    """绘制对比分布图"""
    n_props = len(PROP_NAMES)
    n_cols = 3
    n_rows = int(np.ceil(n_props / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.5 * n_rows))
    axes = axes.flatten()

    for idx, (prop_key, prop_label) in enumerate(PROP_NAMES.items()):
        ax = axes[idx]
        d1 = np.array(epoch20_props[prop_key])
        d2 = np.array(ref_props[prop_key])
        d1 = d1[~np.isnan(d1)]
        d2 = d2[~np.isnan(d2)]

        bins = 50
        ax.hist(d2, bins=bins, alpha=0.5, density=True,
                color='#2196F3', label=f'Baseline (n={len(d2)})')
        ax.hist(d1, bins=bins, alpha=0.6, density=True,
                color='#FF5722', label=f'Epoch20 (n={len(d1)})')

        # 标注均值
        m1, m2 = np.mean(d1), np.mean(d2)
        ax.axvline(m1, color='#FF5722', ls='--', lw=1.5, alpha=0.8)
        ax.axvline(m2, color='#2196F3', ls='--', lw=1.5, alpha=0.8)
        ax.text(0.97, 0.95, f'Epoch20 μ={m1:.3f}', transform=ax.transAxes,
                va='top', ha='right', fontsize=9, color='#FF5722')
        ax.text(0.97, 0.87, f'Baseline μ={m2:.3f}', transform=ax.transAxes,
                va='top', ha='right', fontsize=9, color='#2196F3')

        ax.set_xlabel(prop_label)
        ax.set_ylabel('Density')
        ax.legend(fontsize=8, loc='upper left')
        ax.set_title(f'{prop_label}', fontsize=12)

    # 隐藏多余子图
    for idx in range(n_props, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'property_comparison.png')
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"对比图已保存: {fig_path}")


def plot_chemical_space_2d(epoch20_props, ref_props):
    """2D 化学空间映射: MW vs LogP, QED vs SA"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # (a) MW vs LogP
    ax = axes[0]
    ax.scatter(ref_props["LogP"], ref_props["MolWt"], c='#2196F3', s=8,
               alpha=0.3, label=f'Baseline (n={len(ref_props["LogP"])})', edgecolors='none')
    ax.scatter(epoch20_props["LogP"], epoch20_props["MolWt"], c='#FF5722', s=18,
               alpha=0.7, label=f'Epoch20 (n={len(epoch20_props["LogP"])})', edgecolors='k', linewidths=0.3)
    ax.set_xlabel('LogP')
    ax.set_ylabel('分子量 (MW)')
    ax.legend(markerscale=2, fontsize=10)
    ax.set_title('化学空间: MW vs LogP')

    # (b) QED vs SA
    ax = axes[1]
    ax.scatter(ref_props["SA_Score"], ref_props["QED"], c='#2196F3', s=8,
               alpha=0.3, label='Baseline', edgecolors='none')
    ax.scatter(epoch20_props["SA_Score"], epoch20_props["QED"], c='#FF5722', s=18,
               alpha=0.7, label='Epoch20', edgecolors='k', linewidths=0.3)
    ax.set_xlabel('SA Score')
    ax.set_ylabel('QED')
    ax.legend(markerscale=2, fontsize=10)
    ax.set_title('可合成性与类药性: SA vs QED')

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'chemical_space_2d.png')
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"2D 化学空间图已保存: {fig_path}")


def plot_table_comparison(epoch20_props, ref_props):
    """绘制统计对比表格图"""
    prop_keys = list(PROP_NAMES.keys())
    prop_labels = [PROP_NAMES[k] for k in prop_keys]
    epoch_means = []
    epoch_stds = []
    ref_means = []
    ref_stds = []
    for k in prop_keys:
        d1 = np.array(epoch20_props[k])
        d2 = np.array(ref_props[k])
        d1 = d1[~np.isnan(d1)]
        d2 = d2[~np.isnan(d2)]
        epoch_means.append(np.mean(d1))
        epoch_stds.append(np.std(d1))
        ref_means.append(np.mean(d2))
        ref_stds.append(np.std(d2))

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    cell_text = []
    for i, lbl in enumerate(prop_labels):
        cell_text.append([
            lbl,
            f"{ref_means[i]:.3f} ± {ref_stds[i]:.3f}",
            f"{epoch_means[i]:.3f} ± {epoch_stds[i]:.3f}",
            f"{(epoch_means[i] - ref_means[i]):+.3f}",
            f"{(epoch_means[i] - ref_means[i]) / ref_means[i] * 100:+.1f}%"
        ])
    col_labels = ['属性', 'Baseline (n={})'.format(len(ref_props['QED'])),
                  'Epoch20 (n={})'.format(len(epoch20_props['QED'])),
                  '差值', '相对变化%']
    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#37474F')
            cell.set_text_props(color='white', weight='bold')
        elif col in (3, 4) and row > 0:
            val = float(cell_text[row - 1][col].strip('+').strip('%'))
            if val > 0:
                cell.set_facecolor('#FFCDD2')
            elif val < 0:
                cell.set_facecolor('#C8E6C9')
            else:
                cell.set_facecolor('#ECEFF1')
        else:
            cell.set_facecolor('#ECEFF1' if row % 2 == 0 else '#FAFAFA')
    ax.set_title('分子属性统计对比', fontsize=16, fontweight='bold', pad=20)
    fig_path = os.path.join(OUTPUT_DIR, 'property_table.png')
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"统计表已保存: {fig_path}")


def plot_correlation_heatmap(epoch20_props, ref_props):
    """绘制相关性热力图对比"""
    keys = list(PROP_NAMES.keys())
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    for ax, data, title in [
        (axes[0], ref_props, 'Baseline'),
        (axes[1], epoch20_props, 'Epoch20')
    ]:
        arr = np.array([np.array(data[k]) for k in keys])
        corr = np.corrcoef(arr)
        im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
        ax.set_xticks(range(len(keys)))
        ax.set_yticks(range(len(keys)))
        ax.set_xticklabels([PROP_NAMES[k] for k in keys], rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels([PROP_NAMES[k] for k in keys], fontsize=8)
        ax.set_title(title, fontsize=14)
        for i in range(len(keys)):
            for j in range(len(keys)):
                val = corr[i, j]
                color = 'white' if abs(val) > 0.6 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=6, color=color)

    fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    fig.suptitle('属性相关性热力图对比', fontsize=16, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'correlation_heatmap.png')
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"相关性热力图已保存: {fig_path}")


def plot_atom_type_composition(epoch20_smiles):
    """分析原子组成"""
    from collections import Counter
    total_atoms = Counter()
    total_mols = 0
    for smi in epoch20_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        total_mols += 1
        for atom in mol.GetAtoms():
            sym = atom.GetSymbol()
            if sym != 'H':
                total_atoms[sym] += 1

    if not total_atoms:
        print("WARNING: 没有可用的分子做原子组成分析")
        return

    total = sum(total_atoms.values())
    labels = []
    sizes = []
    colors_list = []
    color_palette = {
        'C': '#555555', 'N': '#3050F8', 'O': '#FF0D0D',
        'S': '#FFFF30', 'P': '#FF8000', 'F': '#90E050',
        'Cl': '#1FF01F', 'Br': '#A62929', 'I': '#940094',
        'B': '#C0C000', 'Si': '#C0C0C0',
    }
    for sym, cnt in sorted(total_atoms.items(), key=lambda x: -x[1]):
        labels.append(f'{sym} ({cnt/total*100:.1f}%)')
        sizes.append(cnt)
        colors_list.append(color_palette.get(sym, '#999999'))

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors_list,
        autopct='', startangle=90,
        textprops={'fontsize': 10}
    )
    for i, (sym, cnt) in enumerate(sorted(total_atoms.items(), key=lambda x: -x[1])):
        angle = (wedges[i].theta2 + wedges[i].theta1) / 2
        x = 1.4 * np.cos(np.radians(angle))
        y = 1.4 * np.sin(np.radians(angle))
        ax.text(x, y, f'{cnt}', ha='center', va='center', fontsize=9,
                weight='bold')

    ax.set_title(f'Epoch20 分子原子组成 (总分子={total_mols})', fontsize=14, weight='bold')
    fig_path = os.path.join(OUTPUT_DIR, 'atom_composition.png')
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"原子组成图已保存: {fig_path}")


def print_stat_summary(epoch20_props, ref_props):
    """打印统计摘要"""
    print("\n" + "=" * 80)
    print(f"{'属性':<20} {'Baseline (Mean±Std)':<25} {'Epoch20 (Mean±Std)':<25} {'变化%':<10}")
    print("-" * 80)
    for key, label in PROP_NAMES.items():
        d1 = np.array(epoch20_props[key])
        d2 = np.array(ref_props[key])
        d1 = d1[~np.isnan(d1)]
        d2 = d2[~np.isnan(d2)]
        m1, s1 = np.mean(d1), np.std(d1)
        m2, s2 = np.mean(d2), np.std(d2)
        delta_pct = (m1 - m2) / m2 * 100 if m2 != 0 else 0
        arrow = "↑" if delta_pct > 0 else ("↓" if delta_pct < 0 else "→")
        print(f"{label:<20} {m2:>8.3f} ± {s2:<8.3f}    {m1:>8.3f} ± {s1:<8.3f}    {arrow} {delta_pct:>+6.1f}%")
    print("=" * 80)


def main():
    print("=" * 60)
    print("DDPO Epoch20 vs Baseline 分子属性分析")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/6] 加载 epoch20 SMILES...")
    epoch20_smiles = load_epoch20_smiles(EPOCH20_CSV)
    print(f"  有效分子: {len(epoch20_smiles)}")

    print("\n[2/6] 计算 epoch20 分子属性...")
    epoch20_props = compute_properties(epoch20_smiles)

    print("\n[3/6] 加载 baseline 属性...")
    ref_props = load_ref_properties(REF_CSV)

    # 2. 打印统计
    print("\n[4/6] 打印统计摘要...")
    print_stat_summary(epoch20_props, ref_props)

    # 3. 绘制图表
    print("\n[5/6] 绘制统计对比图...")
    plot_comparison(epoch20_props, ref_props)
    plot_chemical_space_2d(epoch20_props, ref_props)
    plot_table_comparison(epoch20_props, ref_props)
    plot_correlation_heatmap(epoch20_props, ref_props)
    plot_atom_type_composition(epoch20_smiles)

    print("\n[6/6] 保存详细数据...")
    import pandas as pd
    df_epoch = pd.DataFrame(epoch20_props)
    df_epoch['smiles'] = epoch20_smiles[:len(df_epoch)]
    out_csv = os.path.join(OUTPUT_DIR, 'epoch20_properties.csv')
    df_epoch.to_csv(out_csv, index=False)
    print(f"Epoch20 属性数据已保存: {out_csv}")

    print("\n" + "=" * 60)
    print("完成! 所有图片已保存到:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
