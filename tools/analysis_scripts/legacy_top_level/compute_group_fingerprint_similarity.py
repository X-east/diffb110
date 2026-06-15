"""
计算每个分子在同组内的平均 ECFP4 / FCFP4 Tanimoto 相似度。

基线（原始）：5827 个有效分子，全部视为一组
Epoch20：504 个分子，按 rank 分 7 组（每个 rank 约 72 个分子）

输出：每个分子的组内平均相似度 + 各组/全局统计摘要
"""
import sys
import os
import csv
import math
import numpy as np
from collections import defaultdict

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

def compute_fingerprint(smiles, radius=2, n_bits=2048, use_features=False):
    """计算单个分子的 ECFP4 (use_features=False) 或 FCFP4 (use_features=True) 指纹."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=radius, nBits=n_bits, useFeatures=use_features
    )
    return fp

def compute_intra_group_mean_similarities(smiles_list, group_label=None):
    """
    给定一组 SMILES，对每个分子计算它与组内其他所有分子（有效）的平均 Tanimoto 相似度。
    返回：列表 of dicts {smiles, ecfp4_mean_sim, fcfp4_mean_sim}, 以及组统计
    """
    # 先计算所有有效分子的 ECFP4 和 FCFP4 指纹
    ecfp4_fps = []
    fcfp4_fps = []
    valid_smiles = []
    for s in smiles_list:
        ecfp4 = compute_fingerprint(s, use_features=False)
        fcfp4 = compute_fingerprint(s, use_features=True)
        if ecfp4 is None or fcfp4 is None:
            continue
        ecfp4_fps.append(ecfp4)
        fcfp4_fps.append(fcfp4)
        valid_smiles.append(s)

    n = len(valid_smiles)
    if n < 2:
        results = []
        for s in valid_smiles:
            results.append({
                'smiles': s,
                'ecfp4_mean_sim': float('nan'),
                'fcfp4_mean_sim': float('nan'),
            })
        stats = {'group': group_label, 'n': n,
                 'ecfp4_mean_sim': float('nan'), 'ecfp4_std_sim': float('nan'),
                 'fcfp4_mean_sim': float('nan'), 'fcfp4_std_sim': float('nan'),
                 'n_pairs': 0}
        return results, stats

    # 计算：对每个分子，求它与所有其他分子的平均 Tanimoto
    results = []
    ecfp4_all_sims = []
    fcfp4_all_sims = []

    for i in range(n):
        # 其他分子的指纹
        others_ecfp4 = ecfp4_fps[:i] + ecfp4_fps[i+1:]
        others_fcfp4 = fcfp4_fps[:i] + fcfp4_fps[i+1:]

        # BulkTanimotoSimilarity 返回列表 of floats
        ecfp4_sims = BulkTanimotoSimilarity(ecfp4_fps[i], others_ecfp4)
        fcfp4_sims = BulkTanimotoSimilarity(fcfp4_fps[i], others_fcfp4)

        ecfp4_mean = float(np.mean(ecfp4_sims))
        fcfp4_mean = float(np.mean(fcfp4_sims))

        results.append({
            'smiles': valid_smiles[i],
            'ecfp4_mean_sim': ecfp4_mean,
            'fcfp4_mean_sim': fcfp4_mean,
        })
        ecfp4_all_sims.extend(ecfp4_sims)
        fcfp4_all_sims.extend(fcfp4_sims)

    stats = {
        'group': group_label,
        'n': n,
        'ecfp4_mean_sim': float(np.mean(ecfp4_all_sims)),
        'ecfp4_std_sim': float(np.std(ecfp4_all_sims)),
        'fcfp4_mean_sim': float(np.mean(fcfp4_all_sims)),
        'fcfp4_std_sim': float(np.std(fcfp4_all_sims)),
        'n_pairs': len(ecfp4_all_sims),
    }
    return results, stats


def main():
    print("=" * 80)
    print("计算组内平均 ECFP4 / FCFP4 Tanimoto 指纹相似度")
    print("=" * 80)

    # ========================
    # 1. 读取基线分子 (5827 个有效分子)
    # ========================
    print("\n[1/4] 读取基线分子...")
    baseline_csv = os.path.join(OUT_DIR, "..", "OUT", "260407_generate_ligands_10000",
                                "guacamol_eval", "260407generated_10000_properties.csv")
    baseline_smiles = []
    with open(baseline_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            baseline_smiles.append(row['smiles'])
    print(f"  基线分子数: {len(baseline_smiles)}")

    # ========================
    # 2. 读取 Epoch20 分子 (直接读聚合的 molecules.csv)
    # ========================
    print("\n[2/4] 读取 Epoch20 分子...")
    epoch20_dir = os.path.join(OUT_DIR, "..", "OUT", "6gpu-chembl-only_20260521_154957",
                               "epochs", "epoch_0020")
    epoch20_smiles = []
    epoch20_csv = os.path.join(epoch20_dir, "molecules.csv")
    with open(epoch20_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            smi = row['smiles'].strip()
            if smi and smi != 'None':
                epoch20_smiles.append(smi)
    print(f"  Epoch20 有效分子数: {len(epoch20_smiles)}")

    # ========================
    # 3. 计算基线组内指纹相似度
    # ========================
    print(f"\n[3/4] 计算基线组内指纹相似度 (n={len(baseline_smiles)})...")
    print("  这可能要花点时间（~17M 对分子）...")
    baseline_results, baseline_stats = compute_intra_group_mean_similarities(
        baseline_smiles, group_label="baseline"
    )
    print(f"  基线 ECFP4 组内平均 Tanimoto: {baseline_stats['ecfp4_mean_sim']:.4f} ± {baseline_stats['ecfp4_std_sim']:.4f}")
    print(f"  基线 FCFP4 组内平均 Tanimoto: {baseline_stats['fcfp4_mean_sim']:.4f} ± {baseline_stats['fcfp4_std_sim']:.4f}")

    # 保存基线预测结果
    baseline_out = os.path.join(OUT_DIR, "baseline_intra_group_similarity.csv")
    with open(baseline_out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['smiles', 'ecfp4_mean_sim', 'fcfp4_mean_sim'])
        writer.writeheader()
        writer.writerows(baseline_results)
    print(f"  基线预测结果保存至: {baseline_out}")

    # ========================
    # 4. 计算 Epoch20 整体组内指纹相似度
    # ========================
    print(f"\n[4/4] 计算 Epoch20 整体组内指纹相似度 (n={len(epoch20_smiles)})...")
    epoch20_results, epoch20_stats = compute_intra_group_mean_similarities(
        epoch20_smiles, group_label="epoch20"
    )
    print(f"  Epoch20 ECFP4 组内平均 Tanimoto: {epoch20_stats['ecfp4_mean_sim']:.4f} ± {epoch20_stats['ecfp4_std_sim']:.4f}")
    print(f"  Epoch20 FCFP4 组内平均 Tanimoto: {epoch20_stats['fcfp4_mean_sim']:.4f} ± {epoch20_stats['fcfp4_std_sim']:.4f}")

    # 保存 Epoch20 预测结果
    epoch20_out = os.path.join(OUT_DIR, "epoch20_intra_group_similarity.csv")
    with open(epoch20_out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['smiles', 'ecfp4_mean_sim', 'fcfp4_mean_sim'])
        writer.writeheader()
        writer.writerows(epoch20_results)
    print(f"  Epoch20 预测结果保存至: {epoch20_out}")

    # ========================
    # 5. 输出汇总
    # ========================
    print("\n" + "=" * 80)
    print("汇总")
    print("=" * 80)
    print(f"\n{'Group':<20} {'N':>6} {'ECFP4 Mean':>12} {'ECFP4 Std':>12} "
          f"{'FCFP4 Mean':>12} {'FCFP4 Std':>12} {'N Pairs':>10}")
    print("-" * 80)

    for label, stats in [("baseline", baseline_stats), ("epoch20", epoch20_stats)]:
        s = stats
        print(f"{label:<20} {s['n']:>6} {s['ecfp4_mean_sim']:>12.4f} {s['ecfp4_std_sim']:>12.4f} "
              f"{s['fcfp4_mean_sim']:>12.4f} {s['fcfp4_std_sim']:>12.4f} {s['n_pairs']:>10}")

    print("\n" + "=" * 80)
    print("完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
