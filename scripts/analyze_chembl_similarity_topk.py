#!/usr/bin/env python3
"""Per-molecule top-1, top-5, top-10 ChEMBL Tanimoto similarity + plots."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from tqdm import tqdm

from ddpo.rewards.chembl_similarity import (
    compute_topk_tanimoto,
    load_chembl_fingerprint_cache,
)

RDLogger.DisableLog("rdApp.*")
LOGGER = logging.getLogger("analyze_topk")

SDF = Path("bishe/RE-out/260407generated_10000_valid_dedup2d.sdf")
CACHE = Path("data/chembl/chembl_36_morgan_fingerprints.pkl.gz")
OUT_DIR = Path("OUT/analysis/chembl_similarity")
TOP_KS = [1, 5, 10]
CHUNK_SIZE = 100000


def sanitize_mol(raw_mol):
    if raw_mol is None:
        return None
    try:
        mol = Chem.Mol(raw_mol)
        Chem.SanitizeMol(mol)
        return Chem.RemoveHs(mol)
    except Exception:
        return None


def compute_topk_all(query_fp, reference_fps, max_k=10, chunk_size=100000):
    """Return list of top-K similarities (sorted descending), length = max_k."""
    best = []
    for start in range(0, len(reference_fps), chunk_size):
        sims = AllChem.DataStructs.BulkTanimotoSimilarity(
            query_fp, reference_fps[start : start + chunk_size]
        )
        if not sims:
            continue
        if len(best) < max_k:
            best.extend(float(v) for v in sims)
            best.sort(reverse=True)
            best = best[:max_k]
            continue
        cutoff = best[-1]
        candidates = [float(v) for v in sims if float(v) > cutoff]
        if candidates:
            best.extend(candidates)
            best.sort(reverse=True)
            best = best[:max_k]
    return best


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sdf_path = SDF.expanduser().resolve()
    cache_path = CACHE.expanduser().resolve()
    out_dir = OUT_DIR.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    stem = sdf_path.stem

    LOGGER.info("Loading ChEMBL cache: %s", cache_path)
    cache = load_chembl_fingerprint_cache(str(cache_path))
    reference_fps = cache.fingerprints
    radius, n_bits = cache.radius, cache.n_bits
    LOGGER.info("Loaded %d reference fingerprints (radius=%d, nBits=%d)", len(reference_fps), radius, n_bits)

    supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False)
    all_values = {k: [] for k in TOP_KS}
    records = []

    LOGGER.info("Processing molecules...")
    for idx, raw_mol in enumerate(tqdm(supplier, desc="molecules")):
        mol = sanitize_mol(raw_mol)
        row = {
            "sample_idx": idx,
            "rdkit_valid": int(mol is not None),
            "canonical_smiles": "",
        }
        for k in TOP_KS:
            row[f"chembl_top{k}_mean_similarity"] = 0.0
        row["chembl_max_similarity"] = 0.0

        if mol is not None:
            row["canonical_smiles"] = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
            query_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
            sorted_top = compute_topk_all(query_fp, reference_fps, max_k=max(TOP_KS), chunk_size=CHUNK_SIZE)
            if sorted_top:
                row["chembl_max_similarity"] = round(float(sorted_top[0]), 6)
                for k in TOP_KS:
                    mean_val = float(np.mean(sorted_top[:k]))
                    row[f"chembl_top{k}_mean_similarity"] = round(mean_val, 6)
                    all_values[k].append(mean_val)
            else:
                for k in TOP_KS:
                    all_values[k].append(0.0)
        records.append(row)

    # ── summary ──
    LOGGER.info("Calculating summary statistics...")
    def percentile_summary(values):
        if not values:
            return {}
        arr = np.asarray(values, dtype=float)
        return {
            "count": int(arr.size),
            "mean": round(float(arr.mean()), 6),
            "std": round(float(arr.std()), 6),
            "min": round(float(arr.min()), 6),
            "p01": round(float(np.percentile(arr, 1)), 6),
            "p05": round(float(np.percentile(arr, 5)), 6),
            "p10": round(float(np.percentile(arr, 10)), 6),
            "p25": round(float(np.percentile(arr, 25)), 6),
            "p50": round(float(np.median(arr)), 6),
            "p75": round(float(np.percentile(arr, 75)), 6),
            "p90": round(float(np.percentile(arr, 90)), 6),
            "p95": round(float(np.percentile(arr, 95)), 6),
            "p99": round(float(np.percentile(arr, 99)), 6),
            "max": round(float(arr.max()), 6),
        }

    valid_count = sum(1 for r in records if r["rdkit_valid"])
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sdf_path": str(sdf_path),
        "fingerprint_cache": str(cache_path),
        "reference_fingerprint_count": len(reference_fps),
        "radius": radius,
        "n_bits": n_bits,
        "chunk_size": CHUNK_SIZE,
        "counts": {
            "total": len(records),
            "rdkit_valid": valid_count,
            "rdkit_invalid": len(records) - valid_count,
            "valid_rate": round(valid_count / len(records), 6) if records else 0.0,
        },
        "chembl_max_similarity": percentile_summary(all_values[1]),
        **{f"chembl_top{k}_mean_similarity": percentile_summary(all_values[k]) for k in TOP_KS},
    }

    # ── CSV ──
    csv_path = out_dir / f"{stem}_chembl_topk_{timestamp}.csv"
    import csv
    fieldnames = list(records[0].keys()) if records else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    LOGGER.info("CSV: %s", csv_path)

    # ── JSON summary ──
    summary_path = out_dir / f"{stem}_chembl_topk_summary_{timestamp}.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    LOGGER.info("Summary: %s", summary_path)

    # ── Plots ──
    LOGGER.info("Generating plots...")
    bins = 80

    # 1) Top-1 (max similarity)
    v = np.asarray(all_values[1], dtype=float)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(v, bins=bins, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(float(np.median(v)), color="red", ls="--", lw=1.5, label=f"median={np.median(v):.4f}")
    ax.axvline(float(np.mean(v)), color="darkorange", ls=":", lw=1.5, label=f"mean={np.mean(v):.4f}")
    ax.set_xlabel("Top-1 Tanimoto Similarity (Max)")
    ax.set_ylabel("Molecule Count")
    ax.set_title(f"ChEMBL Max Similarity Distribution (N={len(v)}, valid={valid_count})")
    ax.legend()
    fig.tight_layout()
    p1 = out_dir / f"{stem}_chembl_top1_hist_{timestamp}.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    LOGGER.info("Plot: %s", p1)

    # 2) Top-5 mean
    v = np.asarray(all_values[5], dtype=float)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(v, bins=bins, color="mediumseagreen", edgecolor="white", alpha=0.85)
    ax.axvline(float(np.median(v)), color="red", ls="--", lw=1.5, label=f"median={np.median(v):.4f}")
    ax.axvline(float(np.mean(v)), color="darkorange", ls=":", lw=1.5, label=f"mean={np.mean(v):.4f}")
    ax.set_xlabel("Top-5 Mean Tanimoto Similarity")
    ax.set_ylabel("Molecule Count")
    ax.set_title(f"ChEMBL Top-5 Mean Similarity Distribution (N={len(v)}, valid={valid_count})")
    ax.legend()
    fig.tight_layout()
    p2 = out_dir / f"{stem}_chembl_top5_mean_hist_{timestamp}.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    LOGGER.info("Plot: %s", p2)

    # 3) Top-10 mean
    v = np.asarray(all_values[10], dtype=float)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(v, bins=bins, color="coral", edgecolor="white", alpha=0.85)
    ax.axvline(float(np.median(v)), color="red", ls="--", lw=1.5, label=f"median={np.median(v):.4f}")
    ax.axvline(float(np.mean(v)), color="darkorange", ls=":", lw=1.5, label=f"mean={np.mean(v):.4f}")
    ax.set_xlabel("Top-10 Mean Tanimoto Similarity")
    ax.set_ylabel("Molecule Count")
    ax.set_title(f"ChEMBL Top-10 Mean Similarity Distribution (N={len(v)}, valid={valid_count})")
    ax.legend()
    fig.tight_layout()
    p3 = out_dir / f"{stem}_chembl_top10_mean_hist_{timestamp}.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    LOGGER.info("Plot: %s", p3)

    LOGGER.info("All done. CSV+JSON+3 plots in: %s", out_dir)


if __name__ == "__main__":
    main()
