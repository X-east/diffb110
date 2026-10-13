#!/usr/bin/env python3
"""Test ChEMBL-based UMAP screens and benchmark sampled ECFP4 similarity."""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from scipy.spatial import ConvexHull, cKDTree

RDLogger.DisableLog("rdApp.*")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--chembl-fp", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--tanimoto-sample", type=int, default=500)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated = pd.read_csv(args.facts, low_memory=False)
    generated = generated[generated["rdkit_valid"].eq(1)].copy().reset_index(drop=True)
    chembl = pd.read_csv(args.reference, low_memory=False)
    chembl = chembl[chembl["model"].eq("ChEMBL36")].copy()
    cxy = chembl[["umap1", "umap2"]].to_numpy(float)
    gxy = generated[["umap1", "umap2"]].to_numpy(float)
    tree = cKDTree(cxy)
    nearest = tree.query(gxy, k=1)[0]
    neighbor_distances = tree.query(gxy, k=20)[0]
    local_density_distance = neighbor_distances[:, -1]
    radii = {r: tree.query_ball_point(gxy, r=r, return_length=True) for r in (0.2, 0.5, 1.0)}
    hull = ConvexHull(cxy)
    # Half-space test for the ChEMBL convex hull.
    inside = np.all(np.dot(gxy, hull.equations[:, :2].T) + hull.equations[:, 2] <= 1e-9, axis=1)
    mean = cxy.mean(axis=0)
    cov_inv = np.linalg.pinv(np.cov(cxy, rowvar=False))
    mahal = np.sqrt(np.einsum("ij,jk,ik->i", gxy - mean, cov_inv, gxy - mean))
    chembl_mahal = np.sqrt(np.einsum("ij,jk,ik->i", cxy - mean, cov_inv, cxy - mean))
    internal_neighbor_distances = tree.query(cxy, k=21)[0][:, -1]
    q = {
        "nearest_le_90pct_internal": float(np.quantile(tree.query(cxy, k=2)[0][:, 1], 0.90)),
        "density_k20_distance_le_chembl_p90": float(np.quantile(internal_neighbor_distances, 0.90)),
        "mahal_le_chembl_p95": float(np.quantile(chembl_mahal, 0.95)),
    }
    out = generated[["trajectory_id", "canonical_smiles", "umap1", "umap2"]].copy()
    out["umap_nearest_chembl_distance"] = nearest
    out["umap_chembl_count_r0_2"] = radii[0.2]
    out["umap_chembl_count_r0_5"] = radii[0.5]
    out["umap_chembl_count_r1_0"] = radii[1.0]
    out["umap_chembl_k20_distance"] = local_density_distance
    out["umap_inside_chembl_convex_hull"] = inside
    out["umap_chembl_mahalanobis"] = mahal
    out.to_csv(args.output_dir / "umap_screen_metrics.csv.gz", index=False, compression="gzip")
    rows = []
    for name, mask in {
        "umap_nearest_le_internal_p90": nearest <= q["nearest_le_90pct_internal"],
        "umap_density_r0_2_gt0": radii[0.2] > 0,
        "umap_density_r0_5_gt0": radii[0.5] > 0,
        "umap_density_r1_0_gt0": radii[1.0] > 0,
        "umap_density_k20_ge_chembl_p10": local_density_distance <= q["density_k20_distance_le_chembl_p90"],
        "umap_convex_hull_inside": inside,
        "umap_mahalanobis_le_chembl_p95": mahal <= q["mahal_le_chembl_p95"],
    }.items():
        rows.append({"scheme": name, "pass_count": int(mask.sum()), "rate_valid": float(mask.mean()), "rate_attempts": float(mask.sum() / 100000)})
    # Sampled exact ECFP4 benchmark against 100k ChEMBL fingerprints.
    with gzip.open(args.chembl_fp, "rb") as handle:
        fp_payload = pickle.load(handle)
    chembl_fps = fp_payload["fingerprints"]
    sample_n = min(args.tanimoto_sample, len(generated))
    sample = generated.iloc[np.linspace(0, len(generated) - 1, sample_n, dtype=int)]
    start = time.time()
    max_sim = []
    mean10 = []
    for smiles in sample["canonical_smiles"]:
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, chembl_fps), dtype=np.float32)
        max_sim.append(float(sims.max()))
        mean10.append(float(np.sort(sims)[-10:].mean()))
    elapsed = time.time() - start
    sim = np.asarray(max_sim)
    rows.extend([
        {"scheme": "ecfp4_tmax_ge_0_50_sample", "pass_count": int((sim >= .50).sum()), "rate_valid": float((sim >= .50).mean()), "rate_attempts": None},
        {"scheme": "ecfp4_tmax_ge_0_60_sample", "pass_count": int((sim >= .60).sum()), "rate_valid": float((sim >= .60).mean()), "rate_attempts": None},
        {"scheme": "ecfp4_tmax_ge_0_70_sample", "pass_count": int((sim >= .70).sum()), "rate_valid": float((sim >= .70).mean()), "rate_attempts": None},
        {"scheme": "ecfp4_mean_top10_ge_0_50_sample", "pass_count": int((np.asarray(mean10) >= .50).sum()), "rate_valid": float((np.asarray(mean10) >= .50).mean()), "rate_attempts": None},
    ])
    pd.DataFrame(rows).to_csv(args.output_dir / "screen_scheme_counts.csv", index=False)
    pd.DataFrame({"trajectory_id": sample["trajectory_id"].to_numpy(), "ecfp4_tmax": sim, "ecfp4_mean_top10": mean10}).to_csv(args.output_dir / "tanimoto_sample_metrics.csv", index=False)
    plot_rows = [row for row in rows if not row["scheme"].endswith("_sample")]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=180)
    axes[0].barh([row["scheme"] for row in plot_rows], [row["rate_valid"] * 100 for row in plot_rows], color="#16803C")
    axes[0].set(xlabel="Pass rate among 67,694 RDKit-valid molecules (%)", title="Existing-coordinate UMAP screens")
    axes[1].hist(sim, bins=30, color="#D95F59", alpha=0.85, label="max Tanimoto")
    axes[1].hist(mean10, bins=30, color="#3B78A4", alpha=0.65, label="mean top-10 Tanimoto")
    axes[1].set(xlabel="ECFP4 similarity to ChEMBL100k", ylabel="Sampled molecule count", title=f"Exact similarity test (n={sample_n})")
    axes[1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(args.output_dir / "screening_scheme_test_summary.png", dpi=220)
    plt.close(figure)
    summary = {"status": "complete", "valid_generated": len(generated), "chembl_reference": len(chembl), "tanimoto_sample": sample_n, "tanimoto_elapsed_seconds": elapsed, "tanimoto_queries_per_second": sample_n / elapsed, "thresholds": q, "note": "UMAP screens are descriptive; Tanimoto test is sampled against ChEMBL100k only."}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
