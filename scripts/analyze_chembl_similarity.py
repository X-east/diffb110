#!/usr/bin/env python3
"""Analyze generated molecules against a ChEMBL Morgan fingerprint cache."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, QED, rdMolDescriptors
from tqdm import tqdm

from ddpo.rewards.chembl_similarity import compute_topk_tanimoto, load_chembl_fingerprint_cache

RDLogger.DisableLog("rdApp.*")
LOGGER = logging.getLogger("analyze_chembl_similarity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdf", required=True, help="Generated molecule SDF file.")
    parser.add_argument("--fingerprint-cache", required=True, help="ChEMBL fingerprint cache .pkl.gz from prepare_chembl_fingerprints.py.")
    parser.add_argument("--out-dir", default="runs/analysis/chembl_similarity")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=100000, help="Reference fingerprint chunk size for similarity calculation.")
    parser.add_argument("--max-molecules", type=int, default=0, help="Optional limit for generated molecules; 0 means all.")
    parser.add_argument("--low-threshold", type=float, default=0.25)
    parser.add_argument("--high-threshold", type=float, default=0.85)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def load_fingerprint_cache(path: Path):
    LOGGER.info("Loading ChEMBL fingerprint cache: %s", path)
    cache = load_chembl_fingerprint_cache(path)
    LOGGER.info(
        "Loaded %d reference fingerprints, radius=%d, n_bits=%d",
        cache.size,
        cache.radius,
        cache.n_bits,
    )
    return cache.fingerprints, cache.records, cache.radius, cache.n_bits


def sanitize_mol(mol):
    if mol is None:
        return None
    try:
        clean = Chem.Mol(mol)
        Chem.SanitizeMol(clean)
        return Chem.RemoveHs(clean)
    except Exception:
        return None


def mol_properties(mol) -> dict[str, Any]:
    return {
        "mol_wt": round(float(Descriptors.MolWt(mol)), 4),
        "heavy_atoms": int(mol.GetNumHeavyAtoms()),
        "formal_charge": int(Chem.GetFormalCharge(mol)),
        "num_rings": int(rdMolDescriptors.CalcNumRings(mol)),
        "qed": round(float(QED.qed(mol)), 6),
    }


def percentile_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "p05": None, "p10": None, "p25": None, "p50": None, "p75": None, "p90": None, "p95": None, "max": None}
    arr = np.asarray(values, dtype=float)
    return {
        "count": int(arr.size),
        "mean": round(float(arr.mean()), 6),
        "std": round(float(arr.std()), 6),
        "min": round(float(arr.min()), 6),
        "p05": round(float(np.percentile(arr, 5)), 6),
        "p10": round(float(np.percentile(arr, 10)), 6),
        "p25": round(float(np.percentile(arr, 25)), 6),
        "p50": round(float(np.percentile(arr, 50)), 6),
        "p75": round(float(np.percentile(arr, 75)), 6),
        "p90": round(float(np.percentile(arr, 90)), 6),
        "p95": round(float(np.percentile(arr, 95)), 6),
        "max": round(float(arr.max()), 6),
    }


def analyze(args: argparse.Namespace) -> tuple[Path, Path]:
    sdf_path = Path(args.sdf).expanduser().resolve()
    cache_path = Path(args.fingerprint_cache).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_fps, _, radius, n_bits = load_fingerprint_cache(cache_path)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    stem = sdf_path.stem
    csv_path = out_dir / f"{stem}_chembl_similarity_{timestamp}.csv"
    summary_path = out_dir / f"{stem}_chembl_similarity_summary_{timestamp}.json"

    supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False)
    records: list[dict[str, Any]] = []
    max_values: list[float] = []
    topk_values: list[float] = []
    low_penalties: list[float] = []
    high_penalties: list[float] = []
    total_seen = 0
    valid_count = 0

    max_molecules = int(args.max_molecules)
    for idx, raw_mol in enumerate(tqdm(supplier, desc="generated molecules")):
        if max_molecules > 0 and idx >= max_molecules:
            break
        total_seen += 1
        mol = sanitize_mol(raw_mol)
        row: dict[str, Any] = {
            "sample_idx": idx,
            "rdkit_valid": int(mol is not None),
            "canonical_smiles": "",
            "chembl_max_similarity": 0.0,
            f"chembl_top{args.top_k}_mean_similarity": 0.0,
            "chembl_low_similarity_penalty_raw": 0.0,
            "chembl_high_similarity_penalty_raw": 0.0,
        }
        if mol is not None:
            valid_count += 1
            row["canonical_smiles"] = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
            row.update(mol_properties(mol))
            query_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
            max_sim, topk_mean, topk = compute_topk_tanimoto(query_fp, reference_fps, args.top_k, args.chunk_size)
            low_penalty = max(0.0, float(args.low_threshold) - topk_mean)
            high_penalty = max(0.0, max_sim - float(args.high_threshold))
            row["chembl_max_similarity"] = round(max_sim, 6)
            row[f"chembl_top{args.top_k}_mean_similarity"] = round(topk_mean, 6)
            row["chembl_low_similarity_penalty_raw"] = round(low_penalty, 6)
            row["chembl_high_similarity_penalty_raw"] = round(high_penalty, 6)
            row["chembl_topk_values"] = ";".join(f"{value:.6f}" for value in topk)
            max_values.append(max_sim)
            topk_values.append(topk_mean)
            low_penalties.append(low_penalty)
            high_penalties.append(high_penalty)
        records.append(row)

    fieldnames = sorted({key for record in records for key in record.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sdf_path": str(sdf_path),
        "fingerprint_cache": str(cache_path),
        "reference_fingerprint_count": len(reference_fps),
        "radius": radius,
        "n_bits": n_bits,
        "top_k": int(args.top_k),
        "chunk_size": int(args.chunk_size),
        "low_threshold": float(args.low_threshold),
        "high_threshold": float(args.high_threshold),
        "counts": {
            "total_seen": int(total_seen),
            "rdkit_valid": int(valid_count),
            "rdkit_invalid": int(total_seen - valid_count),
            "valid_rate": round(float(valid_count / total_seen), 6) if total_seen else 0.0,
            "low_similarity_penalty_count": int(sum(value > 0 for value in low_penalties)),
            "high_similarity_penalty_count": int(sum(value > 0 for value in high_penalties)),
        },
        "chembl_max_similarity": percentile_summary(max_values),
        f"chembl_top{args.top_k}_mean_similarity": percentile_summary(topk_values),
        "low_similarity_penalty_raw": percentile_summary(low_penalties),
        "high_similarity_penalty_raw": percentile_summary(high_penalties),
        "outputs": {"csv": str(csv_path), "summary": str(summary_path)},
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    LOGGER.info("Wrote per-molecule CSV: %s", csv_path)
    LOGGER.info("Wrote summary JSON: %s", summary_path)
    return csv_path, summary_path


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    analyze(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
