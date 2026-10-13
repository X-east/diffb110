#!/usr/bin/env python3
"""Fit UMAP jointly on full ChEMBL and five-model legal molecules."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import umap
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy import sparse

RDLogger.DisableLog("rdApp.error")


def query_matrix(frame: pd.DataFrame) -> sparse.csr_matrix:
    """Build ECFP4 CSR matrix for query SMILES."""
    rows = []
    indptr = np.zeros(len(frame) + 1, dtype=np.int64)
    for index, smiles in enumerate(frame["canonical_smiles"].astype(str)):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"invalid query SMILES at row {index}")
        bits = np.fromiter(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048).GetOnBits(), dtype=np.int32)
        rows.append(bits)
        indptr[index + 1] = indptr[index] + len(bits)
    indices = np.concatenate(rows) if rows else np.empty(0, dtype=np.int32)
    return sparse.csr_matrix((np.ones(len(indices), dtype=np.float32), indices, indptr), shape=(len(frame), 2048))


def main() -> None:
    """Fit and save one joint reducer and all coordinates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chembl-csr", type=Path, required=True)
    parser.add_argument("--chembl-csv", type=Path, required=True)
    parser.add_argument("--generated-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=28)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler(args.output_dir / "joint_fit.log", encoding="utf-8")], force=True)
    chembl = sparse.load_npz(args.chembl_csr).tocsr()
    chembl_ids = pd.read_csv(args.chembl_csv, usecols=["chembl_id", "canonical_smiles"])
    generated = pd.read_csv(args.generated_csv)
    query = generated[generated["sanitize_success"].eq(1) & generated["canonical_smiles"].notna()].copy().reset_index(drop=True)
    if len(query) != 10910:
        raise ValueError(f"expected 10910 valid generated molecules, found {len(query)}")
    qmatrix = query_matrix(query)
    logging.info("joint matrix: ChEMBL=%s generated=%s", chembl.shape, qmatrix.shape)
    matrix = sparse.vstack([chembl, qmatrix], format="csr").toarray().astype(np.float32, copy=False)
    started = time.time()
    reducer = umap.UMAP(n_components=2, metric="cosine", n_neighbors=20, min_dist=0.1, low_memory=True, n_jobs=args.workers, random_state=None, transform_seed=None, verbose=True)
    coordinates = reducer.fit_transform(matrix).astype(np.float32)
    if not np.isfinite(coordinates).all():
        raise ValueError("joint UMAP has non-finite coordinates")
    np.save(args.output_dir / "joint_coordinates.npy", coordinates)
    joblib.dump(reducer, args.output_dir / "joint_umap_model.joblib", compress=3)
    chembl_out = chembl_ids.copy()
    chembl_out.insert(0, "group", "ChEMBL36")
    chembl_out.insert(0, "joint_index", np.arange(len(chembl_out)))
    chembl_out["umap1"], chembl_out["umap2"] = coordinates[:len(chembl_out), 0], coordinates[:len(chembl_out), 1]
    generated_out = query.copy()
    generated_out.insert(0, "group", "Generated")
    generated_out.insert(0, "joint_index", np.arange(len(chembl_out), len(chembl_out) + len(generated_out)))
    generated_out["umap1"], generated_out["umap2"] = coordinates[len(chembl_out):, 0], coordinates[len(chembl_out):, 1]
    pd.concat([chembl_out, generated_out], ignore_index=True, sort=False).to_csv(args.output_dir / "joint_coordinates.csv.gz", index=False, compression="gzip")
    manifest = {"status": "complete", "fit_scope": "all standardized ChEMBL plus all valid five-model molecules", "chembl_count": len(chembl_out), "generated_count": len(generated_out), "total_count": len(coordinates), "workers": args.workers, "dense_float32": True, "metric": "cosine", "n_neighbors": 20, "min_dist": 0.1, "elapsed_seconds": time.time() - started}
    (args.output_dir / "joint_fit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logging.info("joint fit complete elapsed=%.1fs", manifest["elapsed_seconds"])


if __name__ == "__main__":
    main()
