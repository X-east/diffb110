#!/usr/bin/env python3
"""5827 RE-out 分子 vs ChEMBL 指纹的 top-k 相似度全面分析 + 出图出表。"""
from __future__ import annotations

import gzip
import json
import logging
import multiprocessing as mp
import pickle
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("chembl_sim_full")

# ── 路径 ──
CACHE_ECFP4 = Path("data/chembl/chembl_36_ecfp4_fingerprints.pkl.gz")
CACHE_FCFP4 = Path("data/chembl/chembl_36_fcfp4_fingerprints.pkl.gz")
QUERY_SMILES = Path("bishe/RE-out/260407generated_10000_valid_dedup2d.smiles")
OUT_DIR = Path("analysis/chembl_sim_full")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 参数 ──
SAMPLE_SIZE = 100_000
RANDOM_SEED = 42
TOP_KS = [1, 2, 3, 5, 10]
NUM_WORKERS = 18
BINS = np.arange(0.0, 1.025, 0.05)


def load_fingerprint_cache(path: Path):
    LOGGER.info("Loading %s ...", path)
    with gzip.open(path, "rb") as f:
        data = pickle.load(f)
    fps = data["fingerprints"]
    LOGGER.info("  → %d fingerprints", len(fps))
    return fps


def load_query_smiles(path: Path) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def compute_query_fps(smiles_list: list[str], radius: int, n_bits: int, use_features: bool) -> list:
    fps = []
    for smi in tqdm(smiles_list, desc=f"{'fcfp4' if use_features else 'ecfp4'} query"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(None)
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits, useFeatures=use_features))
    return fps


# ── 并行 worker ──
def _calc_topk_worker(args):
    """(idx, query_fp_pickle, ref_fps, ks, chunk_size) → (idx, {k: topk_mean})"""
    idx, qfp_ser, ref_fps, ks, chunk_size = args
    qfp = pickle.loads(qfp_ser) if isinstance(qfp_ser, bytes) else qfp_ser
    if qfp is None:
        return idx, {k: float("nan") for k in ks}

    all_sims: list[float] = []
    for start in range(0, len(ref_fps), chunk_size):
        sims = BulkTanimotoSimilarity(qfp, ref_fps[start:start + chunk_size])
        all_sims.extend(float(v) for v in sims)
    if not all_sims:
        return idx, {k: float("nan") for k in ks}
    all_sims.sort(reverse=True)
    result = {}
    for k in ks:
        vals = all_sims[:k]
        result[k] = float(np.mean(vals)) if vals else float("nan")
    return idx, result


def compute_all_topk(query_fps: list, ref_fps: list, ks: list[int],
                     chunk_size: int = 100_000, label: str = "",
                     pool: mp.Pool | None = None) -> pd.DataFrame:
    """返回 DataFrame: rows=queries, columns=top-1,2,3,5,10

    可传入已有 pool 复用，否则内部创建并自动销毁。
    """
    n = len(query_fps)
    LOGGER.info("Computing %s: %d queries × %d refs with %d workers ...",
                label, n, len(ref_fps), NUM_WORKERS)

    # 序列化 query 指纹以跨进程传输
    worker_args = []
    for i, qfp in enumerate(query_fps):
        worker_args.append((i, pickle.dumps(qfp) if qfp is not None else None,
                            ref_fps, ks, chunk_size))

    results = [None] * n
    own_pool = pool is None
    pool_to_use = pool if pool is not None else mp.Pool(NUM_WORKERS)
    try:
        for idx, topk_dict in tqdm(
            pool_to_use.imap_unordered(_calc_topk_worker, worker_args, chunksize=50),
            total=n, desc=label,
        ):
            results[idx] = topk_dict
    finally:
        if own_pool:
            pool_to_use.close()
            pool_to_use.join()

    rows = {k: [] for k in ks}
    for r in results:
        for k in ks:
            rows[k].append(r[k] if r else float("nan"))
    df = pd.DataFrame(rows)
    df.columns = [f"top{k}" for k in ks]
    return df


# ── 统计 & 输出 ──
def summary_stats(arr: pd.Series) -> dict:
    v = arr.dropna().values
    if len(v) == 0:
        return {}
    return {
        "count": int(len(v)),
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "min": float(np.min(v)),
        "p1": float(np.percentile(v, 1)),
        "p5": float(np.percentile(v, 5)),
        "p10": float(np.percentile(v, 10)),
        "p25": float(np.percentile(v, 25)),
        "median": float(np.median(v)),
        "p75": float(np.percentile(v, 75)),
        "p90": float(np.percentile(v, 90)),
        "p95": float(np.percentile(v, 95)),
        "p99": float(np.percentile(v, 99)),
        "max": float(np.max(v)),
    }


def bin_proportions(arr: pd.Series, bins: np.ndarray) -> dict:
    """每隔 0.05，落在各区间的分子占比"""
    v = arr.dropna().values
    if len(v) == 0:
        return {}
    counts, _ = np.histogram(v, bins=bins)
    props = counts / len(v)
    return {f"[{bins[i]:.2f},{bins[i+1]:.2f})": float(props[i])
            for i in range(len(bins) - 1)}


def percentile_cutoffs(arr: pd.Series) -> dict:
    """关键百分位对应的相似度截断值"""
    v = arr.dropna().values
    if len(v) == 0:
        return {}
    PCTS = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    vals = np.percentile(v, PCTS)
    return {str(p): float(va) for p, va in zip(PCTS, vals)}


def plot_boxplot(all_data: dict[str, pd.DataFrame], filename: str, title: str):
    """绘制 4 组 × 5 k 值的箱线图或小提琴图。"""
    n_groups = len(all_data)
    n_ks = len(TOP_KS)
    positions = []
    labels = []
    all_values = []

    for gi, (label, df) in enumerate(all_data.items()):
        base = gi * (n_ks + 1)  # 每组之间留 1 个空位
        for ki, k in enumerate(TOP_KS):
            col = f"top{k}"
            vals = df[col].dropna().values
            pos = base + ki
            positions.append(pos)
            labels.append(f"{label}\ntop-{k}")
            all_values.append(vals)

    fig, ax = plt.subplots(figsize=(4 * n_groups, 6))
    bp = ax.boxplot(all_values, positions=positions, widths=0.6,
                    patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="red", markersize=4))

    # 着色
    colors = plt.cm.Set2(np.linspace(0, 1, n_groups))
    for gi in range(n_groups):
        for ki in range(n_ks):
            idx = gi * n_ks + ki  # bp["boxes"] 是连续列表，索引 0..(n_groups*n_ks-1)
            bp["boxes"][idx].set_facecolor(colors[gi])
            bp["boxes"][idx].set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Top-k Mean Tanimoto Similarity")
    ax.set_title(title, fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=150)
    plt.close(fig)
    LOGGER.info("Saved %s", OUT_DIR / filename)


def plot_histograms(df: pd.DataFrame, filename: str, title: str, bins: np.ndarray):
    """每个 top-k 画一个直方图（0.05 间隔）"""
    n_ks = len(TOP_KS)
    fig, axes = plt.subplots(1, n_ks, figsize=(5 * n_ks, 4), sharey=True)
    if n_ks == 1:
        axes = [axes]

    for ax, k in zip(axes, TOP_KS):
        col = f"top{k}"
        vals = df[col].dropna().values
        ax.hist(vals, bins=bins, density=True, alpha=0.7, edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Similarity")
        ax.set_ylabel("Proportion")
        ax.set_title(f"top-{k} (n={len(vals)})")
        ax.set_xlim(0, 1)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=150)
    plt.close(fig)
    LOGGER.info("Saved %s", OUT_DIR / filename)


def run_analysis(query_fps: list, ref_fps: list, key: str, label: str,
                 pool: mp.Pool | None = None, chunk_size: int = 100_000) -> pd.DataFrame:
    csv_path = OUT_DIR / f"{key}_raw.csv"
    if csv_path.is_file():
        LOGGER.info("Loading cached %s from %s ...", key, csv_path)
        return pd.read_csv(csv_path)

    t0 = time.time()
    df = compute_all_topk(query_fps, ref_fps, TOP_KS, chunk_size=chunk_size,
                          label=label, pool=pool)
    LOGGER.info("%s done in %.1f s", label, time.time() - t0)

    # 立即写 CSV 作为中间缓存（防崩溃丢数据）
    df.to_csv(csv_path, index=False)
    LOGGER.info("Intermediate CSV saved to %s", csv_path)
    return df


def main():
    rng = random.Random(RANDOM_SEED)

    # 1. 加载参考指纹
    ref_ecfp4_full = load_fingerprint_cache(CACHE_ECFP4)
    ref_fcfp4_full = load_fingerprint_cache(CACHE_FCFP4)
    total_ref = len(ref_ecfp4_full)
    LOGGER.info("Total references: %d", total_ref)

    # 2. 随机采样 10 万
    sample_idx = sorted(rng.sample(range(total_ref), min(SAMPLE_SIZE, total_ref)))
    ref_ecfp4_100k = [ref_ecfp4_full[i] for i in sample_idx]
    ref_fcfp4_100k = [ref_fcfp4_full[i] for i in sample_idx]
    LOGGER.info("Sampled %d references (seed=%d)", len(ref_ecfp4_100k), RANDOM_SEED)

    # 3. 加载 query SMILES，计算指纹
    query_smiles = load_query_smiles(QUERY_SMILES)
    LOGGER.info("Loaded %d query SMILES", len(query_smiles))
    radius, n_bits = 2, 2048
    t0 = time.time()
    query_ecfp4 = compute_query_fps(query_smiles, radius, n_bits, use_features=False)
    query_fcfp4 = compute_query_fps(query_smiles, radius, n_bits, use_features=True)
    LOGGER.info("Query fingerprint computation done in %.1f s", time.time() - t0)

    # 4. 计算 4 组相似度（复用同一个 pool，每算完立即写 CSV）
    results = {}

    with mp.Pool(NUM_WORKERS) as pool:
        for key, qfps, rfps, label in [
            ("ECFP4_100K", query_ecfp4, ref_ecfp4_100k, "ECFP4 vs 100K"),
            ("FCFP4_100K", query_fcfp4, ref_fcfp4_100k, "FCFP4 vs 100K"),
            ("ECFP4_full", query_ecfp4, ref_ecfp4_full, "ECFP4 vs 1.55M"),
            ("FCFP4_full", query_fcfp4, ref_fcfp4_full, "FCFP4 vs 1.55M"),
        ]:
            df = run_analysis(qfps, rfps, key, label, pool=pool)
            results[key] = df

    # 5. 统计表 + 直方图 + 百分位
    full_report = {}
    for key, df in results.items():
        report = {"summary_stats": {}, "bin_proportions": {}, "percentile_cutoffs": {}}
        for k in TOP_KS:
            col = f"top{k}"
            report["summary_stats"][str(k)] = summary_stats(df[col])
            report["bin_proportions"][str(k)] = bin_proportions(df[col], BINS)
            report["percentile_cutoffs"][str(k)] = percentile_cutoffs(df[col])
        full_report[key] = report

        # 直方图
        plot_histograms(df, f"hist_{key}.png", key, BINS)

    # 6. 箱线图（4 组合一图）
    all_for_box = {}
    for key, df in results.items():
        all_for_box[key] = df
    plot_boxplot(all_for_box, "boxplot_all.png", "Top-k Tanimoto Similarity: RE-out vs ChEMBL")

    # 7. 写出报告
    report_path = OUT_DIR / "report.json"
    report_path.write_text(json.dumps(full_report, indent=2, ensure_ascii=False) + "\n")
    LOGGER.info("Report saved to %s", report_path)

    # 8. 打印摘要
    print(f"\n{'='*80}")
    print(f"  RE-out (5827) vs ChEMBL — Top-k Tanimoto Similarity Summary")
    print(f"{'='*80}")
    for key, df in results.items():
        print(f"\n  --- {key} ---")
        print(f"  {'top-k':>8} {'mean':>8} {'median':>8} {'p25':>8} {'p75':>8} {'min':>8} {'max':>8}")
        for k in TOP_KS:
            s = full_report[key]["summary_stats"][str(k)]
            print(f"  {k:>8d} {s['mean']:>8.4f} {s['median']:>8.4f} {s['p25']:>8.4f} {s['p75']:>8.4f} {s['min']:>8.4f} {s['max']:>8.4f}")

    print(f"\n  All results in {OUT_DIR}/")


if __name__ == "__main__":
    raise SystemExit(main())
