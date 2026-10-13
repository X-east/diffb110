#!/usr/bin/env python3
"""Fit and freeze one UMAP reducer on all standardized ChEMBL ECFP4 rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import umap
from scipy import sparse


LOGGER = logging.getLogger("fit_full_chembl_umap")


def sha256(path: Path) -> str:
    """Return a streaming SHA256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """Fit UMAP once and write a frozen reference package."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--chembl-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=28)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler(args.output_dir / "umap_fit.log", encoding="utf-8")], force=True)
    matrix = sparse.load_npz(args.matrix).tocsr()
    frame = pd.read_csv(args.chembl_csv, usecols=["chembl_id", "canonical_smiles"])
    if matrix.shape != (2_734_034, 2_048) or len(frame) != matrix.shape[0]:
        raise ValueError(f"input shape mismatch: matrix={matrix.shape}, csv={len(frame)}")
    LOGGER.info("converting CSR to float32 dense matrix for reliable cosine transform serialization")
    matrix = matrix.toarray().astype(np.float32, copy=False)
    LOGGER.info("fitting full ChEMBL UMAP shape=%s workers=%d", matrix.shape, args.workers)
    started = time.time()
    reducer = umap.UMAP(n_components=2, metric="cosine", n_neighbors=20, min_dist=0.1, low_memory=True, n_jobs=args.workers, random_state=None, transform_seed=None, verbose=True)
    coordinates = reducer.fit_transform(matrix).astype(np.float32)
    if not np.isfinite(coordinates).all():
        raise ValueError("UMAP produced non-finite coordinates")
    np.save(args.output_dir / "chembl_umap_coordinates.npy", coordinates)
    output = frame.copy()
    output.insert(0, "full_index", np.arange(len(output), dtype=np.int64))
    output["umap1"], output["umap2"] = coordinates[:, 0], coordinates[:, 1]
    output.to_csv(args.output_dir / "chembl_umap_coordinates.csv.gz", index=False, compression="gzip")
    # Coordinates are the primary full-reference artifact; validate them before
    # attempting to serialize PyNNDescent's optional search index.
    if not (args.output_dir / "chembl_umap_coordinates.npy").exists():
        raise RuntimeError("coordinate write did not produce the expected artifact")
    reducer_path = args.output_dir / "umap_model.joblib"
    try:
        temporary_reducer = reducer_path.with_suffix(".joblib.tmp")
        joblib.dump(reducer, temporary_reducer, compress=3)
        temporary_reducer.replace(reducer_path)
        reducer_save = "complete"
    except Exception as exc:
        (args.output_dir / "umap_model_save_error.txt").write_text(
            f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
        )
        reducer_save = "failed_after_coordinates"
        LOGGER.exception("reducer serialization failed after coordinates were saved")
    manifest = {"status": "complete" if reducer_save == "complete" else "coordinates_complete_reducer_save_failed", "reducer_save": reducer_save, "fit_scope": "all standardized unique ChEMBL parent molecules", "rows": int(len(frame)), "matrix_shape": list(matrix.shape), "input_nnz": int(sparse.load_npz(args.matrix).nnz), "metric": "cosine", "n_neighbors": 20, "min_dist": 0.1, "workers": args.workers, "random_state": None, "transform_seed": None, "dense_float32_fit": True, "elapsed_seconds": time.time() - started, "matrix_sha256": sha256(args.matrix), "chembl_csv_sha256": sha256(args.chembl_csv), "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "umap_learn": umap.__version__}}
    (args.output_dir / "umap_fit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info("frozen full ChEMBL UMAP complete elapsed=%.1fs", manifest["elapsed_seconds"])


if __name__ == "__main__":
    main()
