#!/usr/bin/env python3
"""精确分析 ChEMBL 155万 LogP 分布，辅助分档决策。"""
import csv, gzip, math
import numpy as np

CHEMBL_CSV = "data/chembl/chembl_36_properties.csv.gz"

# Load
logp_vals = []
with gzip.open(CHEMBL_CSV, "rt") as f:
    for row in csv.DictReader(f):
        try:
            logp_vals.append(float(row["mol_log_p"]))
        except (ValueError, KeyError):
            pass

arr = np.array(logp_vals)
total = len(arr)
print(f"Total: {total:,} | Range: [{arr.min():.2f}, {arr.max():.2f}]")
print(f"Mean={arr.mean():.3f}, Std={arr.std():.4f}, Skew={((arr-arr.mean())**3).mean()/arr.std()**3:.3f}\n")

# ── Fine percentiles ──
pcts = [0.1, 0.5, 1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,
        65, 70, 75, 80, 85, 90, 95, 97, 98, 99, 99.5, 99.9]
print("Percentile Distribution:")
for i in range(0, len(pcts), 4):
    parts = []
    for j in range(i, min(i+4, len(pcts))):
        parts.append(f"P{pcts[j]:.1f}={np.percentile(arr, pcts[j]):.3f}")
    print("  " + "  ".join(parts))

# ── 方案1: 等比例分档 ──
print("\n" + "="*70)
print("方案1: 等比例分档 (每个 bin 分子数大致相等)")
for n_bins in [5, 6, 7, 8]:
    edges = []
    for i in range(1, n_bins):
        edges.append(np.percentile(arr, 100*i/n_bins))
    print(f"\n  {n_bins} bins: edges={[f'{e:.3f}' for e in edges]}")

# ── 方案2: 有解析意义的固定边界 ──
schemes = [
    ("6 bins (wide)",  [-float("inf"), 0.0, 1.5, 3.0, 4.5, 6.0, float("inf")]),
    ("6 bins (narrow)",[-float("inf"), 0.5, 2.0, 3.0, 4.0, 5.5, float("inf")]),
    ("7 bins",         [-float("inf"), 0.0, 1.5, 2.5, 3.5, 4.5, 6.0, float("inf")]),
    ("8 bins",         [-float("inf"), 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, float("inf")]),
    ("8 bins (fine)",  [-float("inf"), 0.0, 1.0, 1.5, 2.5, 3.5, 4.5, 5.5, float("inf")]),
]

print("\n" + "="*70)
print("方案2: 有解析意义的固定边界")
for label, edges in schemes:
    print(f"\n── {label} ──")
    for i in range(len(edges)-1):
        lo, hi = edges[i], edges[i+1]
        if lo == -float("inf"):
            mask = arr < hi
        elif hi == float("inf"):
            mask = arr >= lo
        else:
            mask = (arr >= lo) & (arr < hi)
        cnt = mask.sum()
        lo_s = "(-∞" if lo==-float("inf") else f"[{lo:.1f}"
        hi_s = "+∞)" if hi==float("inf") else f"{hi:.1f})"
        bar = "█" * max(1, int(cnt/total*100))
        print(f"  {lo_s}, {hi_s:<8} {cnt:>8,} ({cnt/total*100:5.1f}%)  {bar}")

# ── 方案3: 推荐方案 — 8 bins，边界兼顾等比例+解析意义 ──
print("\n" + "="*70)
print("★★★ 推荐方案: 8 bins — LogP分布匹配奖励 ★★★")
rec_edges = [-float("inf"), 0.5, 1.5, 2.5, 3.0, 3.5, 4.5, 5.5, float("inf")]
labels = [
    "极低 LogP (<0.5)",
    "低 (0.5-1.5)",
    "中等偏低 (1.5-2.5)",
    "ChEMBL峰值左 (2.5-3.0)",
    "ChEMBL峰值右 (3.0-3.5)",
    "中等偏高 (3.5-4.5)",
    "高 (4.5-5.5)",
    "极高 LogP (>5.5)",
]
total_check = 0
for i in range(len(rec_edges)-1):
    lo, hi = rec_edges[i], rec_edges[i+1]
    if lo == -float("inf"):
        mask = arr < hi
    elif hi == float("inf"):
        mask = arr >= lo
    else:
        mask = (arr >= lo) & (arr < hi)
    cnt = mask.sum()
    total_check += cnt
    lo_s = "(-∞" if lo==-float("inf") else f"[{lo:.1f}"
    hi_s = "+∞)" if hi==float("inf") else f"{hi:.1f})"
    ref_pct = cnt/total*100
    bar = "█" * max(1, int(ref_pct))
    print(f"  {labels[i]:<24s} {lo_s}, {hi_s:<8} {cnt:>8,} = {ref_pct:5.1f}%  {bar}")
print(f"  {'Total':>24s} {'':>16s} {total_check:>8,}")

print("\n" + "="*70)
print("奖励映射建议:")
print("  reward = scale × (ref_proportion - ema_batch_proportion[bin])")
print("  scale ≈ 10-15 (每个百分点的 gap 转换为 0.10-0.15 分)")
print("  clamp: [-2.0, +2.0]")
print()
print("示例:")
print("  若某 bin 参考占 15%, batch EMA 当前只占 5% → gap=+0.10")
print("  → reward = 12 × 0.10 = +1.2 (鼓励该 bin 的分子)")
print("  若某 bin 参考占 8%, batch EMA 当前占 20% → gap=-0.12")
print("  → reward = 12 × (-0.12) = -1.44 (惩罚该 bin 的分子)")
