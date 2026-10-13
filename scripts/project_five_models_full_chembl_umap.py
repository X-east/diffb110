#!/usr/bin/env python3
"""Project five-model valid molecules into a frozen full-ChEMBL UMAP map."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import multiprocessing as mp
import platform
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import AllChem
from scipy import sparse

RDLogger.DisableLog("rdApp.error")
LOGGER = logging.getLogger("project_five_models_full_chembl_umap")
MODELS = ["DiffGui", "DiffSBDD", "PMDM", "GCDM-SBDD", "TargetDiff"]
COLORS = {"DiffGui": "#3B82C4", "DiffSBDD": "#D95F59", "PMDM": "#D9A441", "GCDM-SBDD": "#8E78B7", "TargetDiff": "#3D9B72"}


def _task(item: tuple[int, str]) -> tuple[int, np.ndarray | None]:
    index, smiles = item
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return index, None
    return index, np.fromiter(AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048).GetOnBits(), dtype=np.int32)


def _csr(rows: list[np.ndarray | None]) -> sparse.csr_matrix:
    valid = [row if row is not None else np.empty(0, dtype=np.int32) for row in rows]
    indptr = np.zeros(len(valid) + 1, dtype=np.int64)
    for i, row in enumerate(valid):
        indptr[i + 1] = indptr[i] + len(row)
    indices = np.concatenate(valid) if indptr[-1] else np.empty(0, dtype=np.int32)
    return sparse.csr_matrix((np.ones(len(indices), dtype=np.float32), indices, indptr), shape=(len(valid), 2048))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--reducer", type=Path, required=True)
    parser.add_argument("--chembl-coordinates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=28)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler(args.output_dir / "projection.log", encoding="utf-8")], force=True)
    usecols = ["row_id", "repeat", "model", "source_sdf", "source_index", "sanitize_success", "canonical_smiles"]
    frame = pd.read_csv(args.input_csv, usecols=usecols)
    frame = frame[frame["model"].isin(MODELS) & frame["sanitize_success"].eq(1) & frame["canonical_smiles"].notna()].copy().reset_index(drop=True)
    LOGGER.info("valid input rows=%d", len(frame))
    tasks = list(enumerate(frame["canonical_smiles"].astype(str)))
    with mp.Pool(args.workers) as pool:
        results = pool.map(_task, tasks, chunksize=64)
    rows = [None] * len(results)
    for index, bits in results:
        rows[index] = bits
    matrix = _csr(rows)
    LOGGER.info("query CSR shape=%s nnz=%d", matrix.shape, matrix.nnz)
    reducer = joblib.load(args.reducer)
    projected = reducer.transform(matrix.toarray().astype(np.float32, copy=False)).astype(np.float32)
    if not np.isfinite(projected).all():
        raise ValueError("non-finite projected coordinates")
    frame["umap1"], frame["umap2"] = projected[:, 0], projected[:, 1]
    frame.to_csv(args.output_dir / "five_model_projected_coordinates.csv.gz", index=False, compression="gzip")
    chembl_xy = np.load(args.chembl_coordinates, mmap_mode="r")
    rng = np.random.default_rng(42)
    sample_n = min(200_000, len(chembl_xy))
    background = np.asarray(chembl_xy[rng.choice(len(chembl_xy), sample_n, replace=False)])
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.scatter(background[:, 0], background[:, 1], s=1, c="#B8B8B8", alpha=0.12, linewidths=0, rasterized=True, label=f"ChEMBL36 sample (n={sample_n:,})")
    for model in MODELS:
        group = frame[frame.model.eq(model)]
        ax.scatter(group.umap1, group.umap2, s=9, alpha=0.58, linewidths=0, color=COLORS[model], label=f"{model} (n={len(group):,})", rasterized=True)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2"); ax.set_title("Five-model molecules projected onto frozen full ChEMBL36 UMAP")
    ax.legend(markerscale=2, frameon=True, fontsize=9); fig.tight_layout()
    fig.savefig(args.output_dir / "five_model_full_chembl_umap_overview.png", dpi=350); fig.savefig(args.output_dir / "five_model_full_chembl_umap_overview.svg")
    plt.close(fig)
    summary = frame.groupby("model").agg(n=("row_id", "size"), unique_smiles=("canonical_smiles", "nunique"), centroid_umap1=("umap1", "mean"), centroid_umap2=("umap2", "mean")).reindex(MODELS).reset_index()
    summary.to_csv(args.output_dir / "five_model_projection_summary.csv", index=False)
    manifest = {"status": "complete", "input_csv": str(args.input_csv.resolve()), "input_csv_sha256": sha256(args.input_csv), "reducer": str(args.reducer.resolve()), "reducer_sha256": sha256(args.reducer), "chembl_coordinates": str(args.chembl_coordinates.resolve()), "counts": {"input_rows": int(len(pd.read_csv(args.input_csv, usecols=["row_id"]))), "valid_projected": int(len(frame)), "by_model": {m: int((frame.model == m).sum()) for m in MODELS}}, "fingerprint": {"type": "ECFP4", "radius": 2, "n_bits": 2048}, "workers": args.workers, "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "rdkit": rdBase.rdkitVersion, "umap_learn": umap.__version__}}
    (args.output_dir / "projection_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info("projection complete: %d molecules", len(frame))


if __name__ == "__main__":
    main()
