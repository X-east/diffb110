#!/usr/bin/env python3
"""模拟 Laplace vs EMA 平滑在 LogP 分布奖励中的数值表现。

演示三种平滑方案在 batch 噪声下的表现：
  方案A: 无平滑（原始比例）
  方案B: 只有 Laplace 平滑
  方案C: 只有 EMA 平滑
  方案D: Laplace + EMA（推荐）
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True, linewidth=120)

# ── ChEMBL 参考比例 (8 bins) ──
REF = np.array([0.044, 0.093, 0.206, 0.140, 0.143, 0.227, 0.108, 0.039])
N_BINS = len(REF)
BATCH_SIZE = 400
SCALE = 12.0
EMA_ALPHA = 0.15

labels = [
    "(-∞,0.5)", "[0.5,1.5)", "[1.5,2.5)", "[2.5,3.0)",
    "[3.0,3.5)", "[3.5,4.5)", "[4.5,5.5)", "(5.5,+∞)",
]

print("=" * 85)
print("Laplace vs EMA 平滑效果演示")
print(f"  8 bins, batch={BATCH_SIZE}分子, ref={REF}")
print(f"  Laplace: +1 pseudocount per bin")
print(f"  EMA: alpha={EMA_ALPHA}, 等效窗口 ≈ {1/EMA_ALPHA:.0f} batches")
print("=" * 85)

# ── 场景1: 正常 batch（分布接近参考） ──
print("\n── 场景1: 正常 batch（分布接近ChEMBL参考）──")
rng = np.random.default_rng(42)
counts = rng.multinomial(BATCH_SIZE, REF)
raw_props = counts / BATCH_SIZE
# Laplace
lap_props = (counts + 1) / (BATCH_SIZE + N_BINS)
# EMA from previous (= REF)
ema_props = (1 - EMA_ALPHA) * REF + EMA_ALPHA * raw_props
# Laplace + EMA
lap_ema = (1 - EMA_ALPHA) * REF + EMA_ALPHA * lap_props

print(f"  {'Bin':<13} {'Count':>6} {'Raw%':>7} {'Ref%':>7} {'Laplace%':>9} {'EMA%':>9} {'L+EMA%':>9}")
for i in range(N_BINS):
    print(f"  {labels[i]:<13} {counts[i]:>6} {raw_props[i]*100:>6.1f}% "
          f"{REF[i]*100:>6.1f}% {lap_props[i]*100:>8.1f}% "
          f"{ema_props[i]*100:>8.1f}% {lap_ema[i]*100:>8.1f}%")

# ── 场景2: 异常 batch（某 bin 为 0） ──
print("\n── 场景2: 异常 batch（bin 0, 7 都是 0 分子）──")
bad_counts = counts.copy()
bad_counts[0] = 0      # (-∞,0.5) 正常应该有 ~18 个
bad_counts[7] = 0      # (5.5,+∞) 正常应该有 ~16 个
bad_counts[3] += 18    # 补偿到 bin 4
bad_counts[4] += 16
raw_props = bad_counts / BATCH_SIZE
lap_props = (bad_counts + 1) / (BATCH_SIZE + N_BINS)
ema_props = (1 - EMA_ALPHA) * REF + EMA_ALPHA * raw_props
lap_ema = (1 - EMA_ALPHA) * REF + EMA_ALPHA * lap_props

print(f"  {'Bin':<13} {'Count':>6} {'Raw%':>7} {'Ref%':>7} {'Laplace%':>9} {'EMA%':>9} {'L+EMA%':>9}")
for i in range(N_BINS):
    marker = " ← ZERO" if bad_counts[i] == 0 else ""
    print(f"  {labels[i]:<13} {bad_counts[i]:>6} {raw_props[i]*100:>6.1f}% "
          f"{REF[i]*100:>6.1f}% {lap_props[i]*100:>8.1f}% "
          f"{ema_props[i]*100:>8.1f}% {lap_ema[i]*100:>8.1f}%{marker}")

# Gap 对比
print(f"\n  gap = Ref - estimated (正=欠采样→奖励)")
print(f"  {'Bin':<13} {'RawGap':>10} {'LaplaceGap':>11} {'EMAGap':>10} {'L+EMAGap':>11}")
for i in range(N_BINS):
    rg = REF[i] - raw_props[i]
    lg = REF[i] - lap_props[i]
    eg = REF[i] - ema_props[i]
    le = REF[i] - lap_ema[i]
    print(f"  {labels[i]:<13} {rg:>+10.4f} {lg:>+11.4f} {eg:>+10.4f} {le:>+11.4f}")

# 对应 reward
print(f"\n  reward = clip({SCALE} × gap, -2, +2)")
print(f"  {'Bin':<13} {'RawRew':>10} {'LaplaceRew':>11} {'EMARew':>10} {'L+EMARew':>11}")
for i in range(N_BINS):
    rr = np.clip(SCALE * (REF[i] - raw_props[i]), -2, 2)
    lr = np.clip(SCALE * (REF[i] - lap_props[i]), -2, 2)
    er = np.clip(SCALE * (REF[i] - ema_props[i]), -2, 2)
    le = np.clip(SCALE * (REF[i] - lap_ema[i]), -2, 2)
    print(f"  {labels[i]:<13} {rr:>+10.3f} {lr:>+11.3f} {er:>+10.3f} {le:>+11.3f}")


# ── 场景3: 持续漂移的 EMA 追踪 ──
print("\n" + "=" * 85)
print("── 场景3: 连续 batch EMA 追踪 — 模型持续产出高 LogP 分子 ──")
print("  假设连续 10 批，每批分布都偏离 ChEMBL（高 LogP 过多）")
print()

# 模拟漂移分布
drift_ref = REF.copy()
drift_ref[6] = 0.20   # bin 7 (4.5-5.5) 从 10.8% → 20%
drift_ref[7] = 0.15   # bin 8 (>5.5) 从 3.9% → 15%
drift_ref[3] = 0.06   # bin 4 (2.5-3.0) 从 14% → 6%
drift_ref[4] = 0.07   # bin 5 (3.0-3.5) 从 14.3% → 7%

# 纯 EMA（无 Laplace）
ema_pure = REF.copy()
# Laplace+EMA
ema_lap = REF.copy()

print(f"{'Batch':>6} | {'Bin6(4.5-5.5)':>30} | {'Bin4(2.5-3.0)':>30}")
print(f"{'':>6} | {'Count':>5} {'Raw%':>6} {'EMA%':>6} {'L+EMA%':>7} {'gap':>7} | {'Count':>5} {'Raw%':>6} {'EMA%':>6} {'L+EMA%':>7} {'gap':>7}")
print("-" * 85)

for batch_num in range(1, 13):
    # 每批从漂移分布采样
    counts = rng.multinomial(BATCH_SIZE, drift_ref)
    raw = counts / BATCH_SIZE
    lap = (counts + 1) / (BATCH_SIZE + N_BINS)

    # 纯 EMA 更新
    ema_pure = (1 - EMA_ALPHA) * ema_pure + EMA_ALPHA * raw
    # Laplace + EMA 更新
    ema_lap = (1 - EMA_ALPHA) * ema_lap + EMA_ALPHA * lap

    gap6 = SCALE * (REF[6] - ema_pure[6])
    gap4 = SCALE * (REF[4] - ema_lap[4])

    print(f"  {batch_num:>4} | {counts[6]:>5} {raw[6]*100:>5.1f}% {ema_pure[6]*100:>5.1f}% "
          f"{ema_lap[6]*100:>6.1f}% {gap6:>+6.2f} | "
          f"{counts[3]:>5} {raw[3]*100:>5.1f}% {ema_pure[3]*100:>5.1f}% "
          f"{ema_lap[3]*100:>6.1f}% {gap4:>+6.2f}")


# ── 总结 ──
print("\n" + "=" * 85)
print("总结: 四种方案对比")
print("-" * 85)
print("  方案A 无平滑:")
print("    bin 为 0 → proportion=0% → gap 巨大 → 该 bin 所有分子被极度惩罚/奖励")
print("    对大噪声极其敏感，batch 级别的不稳定直接传导到 reward")
print()
print("  方案B 只用 Laplace:")
print("    bin 为 0 → proportion=1/408≈0.25% → gap 合理（非 0）")
print("    但单批噪声仍显著：60 还是 80 个分子在 bin 里差别很大")
print("    gap 随单批采样噪声大幅抖动，reward 不稳定")
print()
print("  方案C 只用 EMA:")
print("    跨 batch 平滑效果好，bin=0 时 4.4%→3.74%（不会被拉到 0）")
print("    但第一次看到 bin=0 时，proportion=0 会被 EMA 直接吸收 15%")
print("    需要 ~7 批才能完全反应真实变化")
print()
print("  方案D Laplace + EMA (推荐):")
print("    每批内：Laplace 防零（bin=0 → 0.25% 而非 0%）")
print("    跨批间：EMA 防抖动（α=0.15 → 等效窗口 ~7 批）")
print("    0 分子 bin 的 proportion 从 REF 平滑滑向 0.25%，而非跳到 0")
print("    → 回收一半的幅度，reward 稳定")
