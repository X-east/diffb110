#!/usr/bin/env python3
"""从 155 万 ChEMBL 指纹中随机选 10 万，计算 5827 个 RE-out 分子的 top-k 相似度。"""
from __future__ import annotations

import gzip
import json
import logging
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("chembl_sim_analysis")

CACHE_ECFP4 = Path("data/chembl/chembl_36_ecfp4_fingerprints.pkl.gz")
CACHE_FCFP4 = Path("data/chembl/chembl_36_fcfp4_fingerprints.pkl.gz")
QUERY_SMILES = Path("bishe/RE-out/260407generated_10000_valid_dedup2d.smiles")
SAMPLE_SIZE = 100_000
RANDOM_SEED = 42
TOP_KS = [1, 2, 3, 5, 10]


def load_fingerprint_cache(path: Path):
    LOGGER.info("Loading %s ...", path)
    with gzip.open(path, "rb") as f:
        data = pickle.load(f)
    fps = data["fingerprints"]
    LOGGER.info("  → %d fingerprints, radius=%d, n_bits=%d",
                len(fps), data.get("radius", "?"), data.get("n_bits", "?"))
    return fps


def load_query_smiles(path: Path) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def compute_fingerprints(smiles_list: list[str], radius: int, n_bits: int, use_features: bool) -> list:
    fps = []
    failed = 0
    for smi in tqdm(smiles_list, desc=f"{'fcfp4' if use_features else 'ecfp4'}"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(None)
            failed += 1
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits, useFeatures=use_features)
        fps.append(fp)
    if failed:
        LOGGER.warning("  %d / %d SMILES parse failed", failed, len(smiles_list))
    return fps


def topk_analysis(query_fps: list, ref_fps: list, ks: list[int], chunk_size: int = 100_000) -> dict:
    """对每个 query 计算 top-k 相似度，返回各 k 的统计。"""
    # results[k] = [max_sim_for_query1, max_sim_for_query2, ...]
    results: dict[int, list[float]] = {k: [] for k in ks}

    valid_mask = [fp is not None for fp in query_fps]
    valid_fps = [fp for fp, ok in zip(query_fps, valid_mask) if ok]
    LOGGER.info("Computing similarity for %d valid molecules ...", len(valid_fps))

    for qfp in tqdm(valid_fps, desc="similarity"):
        # Batch Tanimoto in chunks
        all_sims: list[float] = []
        for start in range(0, len(ref_fps), chunk_size):
            sims = BulkTanimotoSimilarity(qfp, ref_fps[start:start + chunk_size])
            all_sims.extend(float(v) for v in sims)
        if not all_sims:
            for k in ks:
                results[k].append(0.0)
            continue
        all_sims.sort(reverse=True)
        for k in ks:
            topk_mean = float(np.mean(all_sims[:k])) if k <= len(all_sims) else float(np.mean(all_sims))
            results[k].append(topk_mean)

    # 对无效分子补充 NaN
    for k in ks:
        padded = []
        idx = 0
        for ok in valid_mask:
            if ok:
                padded.append(results[k][idx])
                idx += 1
            else:
                padded.append(float("nan"))
        results[k] = padded

    return results


def summarize(results: dict[int, list[float]], label: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"{'top-k':>8} {'mean':>10} {'median':>10} {'min':>10} {'max':>10} {'p25':>10} {'p75':>10}")
    print("-" * 70)
    for k in sorted(results.keys()):
        arr = np.array([v for v in results[k] if not np.isnan(v)])
        if len(arr) == 0:
            continue
        print(f"{k:>8d} {arr.mean():>10.4f} {np.median(arr):>10.4f} {arr.min():>10.4f} {arr.max():>10.4f} {np.percentile(arr, 25):>10.4f} {np.percentile(arr, 75):>10.4f}")


def main():
    rng = random.Random(RANDOM_SEED)

    # 1. 加载 ECFP4 参考指纹，随机采样 10 万
    ref_ecfp4 = load_fingerprint_cache(CACHE_ECFP4)
    total_ref = len(ref_ecfp4)
    sample_idx = sorted(rng.sample(range(total_ref), min(SAMPLE_SIZE, total_ref)))
    ref_sample = [ref_ecfp4[i] for i in sample_idx]
    LOGGER.info("Sampled %d / %d references (seed=%d)", len(ref_sample), total_ref, RANDOM_SEED)

    # 2. 加载 5827 个 query SMILES
    query_smiles = load_query_smiles(QUERY_SMILES)
    LOGGER.info("Loaded %d query SMILES", len(query_smiles))

    # 3. 计算 query 的 ECFP4 和 FCFP4
    radius = 2
    n_bits = 2048
    query_ecfp4 = compute_fingerprints(query_smiles, radius, n_bits, use_features=False)
    query_fcfp4 = compute_fingerprints(query_smiles, radius, n_bits, use_features=True)

    # 4. ECFP4 vs 10 万参考
    LOGGER.info("=== ECFP4 analysis ===")
    t0 = time.time()
    res_ecfp4 = topk_analysis(query_ecfp4, ref_sample, TOP_KS)
    LOGGER.info("ECFP4 done in %.1f s", time.time() - t0)
    summarize(res_ecfp4, "ECFP4 vs 100K ChEMBL")

    # 5. FCFP4 vs 10 万参考（FCFP4 参考指纹也要加载）
    ref_fcfp4 = load_fingerprint_cache(CACHE_FCFP4)
    ref_fcfp4_sample = [ref_fcfp4[i] for i in sample_idx]
    LOGGER.info("=== FCFP4 analysis ===")
    t0 = time.time()
    res_fcfp4 = topk_analysis(query_fcfp4, ref_fcfp4_sample, TOP_KS)
    LOGGER.info("FCFP4 done in %.1f s", time.time() - t0)
    summarize(res_fcfp4, "FCFP4 vs 100K ChEMBL")

    # 6. 汇总
    out = {
        "config": {
            "ref_total": total_ref,
            "sample_size": len(ref_sample),
            "sample_seed": RANDOM_SEED,
            "query_file": str(QUERY_SMILES),
            "n_queries": len(query_smiles),
            "radius": radius,
            "n_bits": n_bits,
            "top_ks": TOP_KS,
        },
        "ecfp4": {str(k): {"mean": float(np.mean([v for v in res_ecfp4[k] if not np.isnan(v)])),
                           "median": float(np.median([v for v in res_ecfp4[k] if not np.isnan(v)])),
                           "min": float(np.min([v for v in res_ecfp4[k] if not np.isnan(v)])),
                           "max": float(np.max([v for v in res_ecfp4[k] if not np.isnan(v)])),
                           "p25": float(np.percentile([v for v in res_ecfp4[k] if not np.isnan(v)], 25)),
                           "p75": float(np.percentile([v for v in res_ecfp4[k] if not np.isnan(v)], 75))}
                   for k in TOP_KS},
        "fcfp4": {str(k): {"mean": float(np.mean([v for v in res_fcfp4[k] if not np.isnan(v)])),
                           "median": float(np.median([v for v in res_fcfp4[k] if not np.isnan(v)])),
                           "min": float(np.min([v for v in res_fcfp4[k] if not np.isnan(v)])),
                           "max": float(np.max([v for v in res_fcfp4[k] if not np.isnan(v)])),
                           "p25": float(np.percentile([v for v in res_fcfp4[k] if not np.isnan(v)], 25)),
                           "p75": float(np.percentile([v for v in res_fcfp4[k] if not np.isnan(v)], 75))}
                   for k in TOP_KS},
    }
    out_path = Path("analysis/chembl_sim_results.json")
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    LOGGER.info("Results saved to %s", out_path)


if __name__ == "__main__":
    raise SystemExit(main())
