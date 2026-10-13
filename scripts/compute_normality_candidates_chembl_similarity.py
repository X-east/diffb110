#!/usr/bin/env python3
"""Compute exact ChEMBL100k ECFP4 nearest-neighbor similarity for pre-screened molecules."""

from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import pickle
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")
REFERENCE_FPS: list[Any] = []
RADIUS = 2
N_BITS = 2048


def parse_args() -> argparse.Namespace:
    """Parse sidecar, reference-cache, and output options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screening-facts", type=Path, required=True)
    parser.add_argument("--chembl-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=18)
    return parser.parse_args()


def _init_worker(fingerprints: list[Any], radius: int, n_bits: int) -> None:
    """Install immutable ChEMBL fingerprints in each process."""
    global REFERENCE_FPS, RADIUS, N_BITS
    REFERENCE_FPS = fingerprints
    RADIUS = radius
    N_BITS = n_bits


def _similarity_row(task: tuple[str, str]) -> dict[str, float | str]:
    """Calculate exact top-1 and top-5 similarity for a single candidate."""
    trajectory_id, smiles = task
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"unreadable canonical SMILES for {trajectory_id}")
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, RADIUS, nBits=N_BITS)
    similarities = np.asarray(DataStructs.BulkTanimotoSimilarity(fingerprint, REFERENCE_FPS), dtype=np.float32)
    top5 = np.partition(similarities, -5)[-5:]
    return {
        "trajectory_id": trajectory_id,
        "chembl_ecfp4_top1_similarity": float(top5.max()),
        "chembl_ecfp4_top5_mean_similarity": float(top5.mean()),
    }


def plot_histogram(frame: pd.DataFrame, output_path: Path) -> None:
    """Show top-1 and top-5 ChEMBL similarity distributions."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=200)
    for axis, column, title in (
        (axes[0], "chembl_ecfp4_top1_similarity", "ECFP4 top-1 similarity"),
        (axes[1], "chembl_ecfp4_top5_mean_similarity", "ECFP4 top-5 mean similarity"),
    ):
        values = frame[column].to_numpy()
        axis.hist(values, bins=70, color="#287D8E", edgecolor="white")
        for threshold in (0.15, 0.20, 0.25, 0.85):
            axis.axvline(threshold, color="#B7483A" if threshold in (0.25, 0.85) else "#555555", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("Tanimoto similarity")
        axis.set_ylabel("Candidate count")
    figure.suptitle("ChEMBL100k support for normality-pre-screened DiffSBDD molecules", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def main() -> int:
    """Calculate exact ChEMBL similarities and write auditable sidecars."""
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    facts = pd.read_csv(args.screening_facts, low_memory=False)
    candidates = facts[facts["normality_pre_chembl"]].copy()
    if candidates.empty:
        raise ValueError("normality_pre_chembl has no candidates")
    with gzip.open(args.chembl_cache, "rb") as handle:
        cache = pickle.load(handle)
    fingerprints = cache["fingerprints"]
    radius = int(cache["radius"])
    n_bits = int(cache["n_bits"])
    if len(fingerprints) != 100_000:
        raise ValueError("expected exactly 100,000 ChEMBL fingerprints")
    tasks = list(zip(candidates["trajectory_id"].astype(str), candidates["canonical_smiles"].astype(str)))
    start = time.time()
    context = mp.get_context("fork")
    with context.Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(fingerprints, radius, n_bits),
    ) as pool:
        rows = pool.map(_similarity_row, tasks, chunksize=8)
    similarities = pd.DataFrame(rows)
    if len(similarities) != len(candidates) or similarities["trajectory_id"].duplicated().any():
        raise ValueError("similarity output lost or duplicated candidates")
    output = candidates.merge(similarities, on="trajectory_id", how="inner", validate="one_to_one")
    output["chembl_top1_ge_0_25"] = output["chembl_ecfp4_top1_similarity"].ge(0.25)
    output["chembl_top1_lt_0_85"] = output["chembl_ecfp4_top1_similarity"].lt(0.85)
    for threshold in (0.15, 0.20, 0.25):
        output[f"chembl_top5_mean_ge_{str(threshold).replace('.', '_')}"] = output["chembl_ecfp4_top5_mean_similarity"].ge(threshold)
    output.to_csv(args.output_dir / "normality_candidates_chembl_similarity.csv.gz", index=False, compression="gzip")
    threshold_counts = {
        "candidates_before_similarity": int(len(output)),
        "top1_ge_0_25": int(output["chembl_top1_ge_0_25"].sum()),
        "top1_ge_0_25_and_lt_0_85": int((output["chembl_top1_ge_0_25"] & output["chembl_top1_lt_0_85"]).sum()),
        "top5_mean_ge_0_15": int(output["chembl_top5_mean_ge_0_15"].sum()),
        "top5_mean_ge_0_20": int(output["chembl_top5_mean_ge_0_20"].sum()),
        "top5_mean_ge_0_25": int(output["chembl_top5_mean_ge_0_25"].sum()),
    }
    summary = {
        "status": "completed",
        "candidate_count": int(len(output)),
        "reference_count": len(fingerprints),
        "comparisons": int(len(output) * len(fingerprints)),
        "ecfp4": {"radius": radius, "n_bits": n_bits},
        "threshold_counts": threshold_counts,
        "top1_quantiles": {str(q): float(output["chembl_ecfp4_top1_similarity"].quantile(q)) for q in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)},
        "top5_mean_quantiles": {str(q): float(output["chembl_ecfp4_top5_mean_similarity"].quantile(q)) for q in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)},
        "elapsed_seconds": round(time.time() - start, 3),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    plot_histogram(output, args.output_dir / "chembl_similarity_histogram.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
