#!/usr/bin/env python3
"""Rebuild the historical 100k-ChEMBL/five-model UMAP without overwriting evidence.

The script snapshots provenance, preserves the old coordinate table, fits one
new reducer with the historical inputs and parameters, audits coordinate
agreement, reloads the saved reducer, and runs a transform smoke test.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import logging
import pickle
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/diffb110_umap_mpl")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/diffb110_umap_numba")

import joblib
import numpy as np
import pandas as pd
import umap
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy import sparse

RDLogger.DisableLog("rdApp.*")
LOGGER = logging.getLogger(__name__)
MODEL_ORDER = ["DiffGui", "DiffSBDD", "PMDM", "GCDM-SBDD", "TargetDiff"]


def sha256(path: Path) -> str:
    """Return a file hash without modifying the source file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bitvectors_to_csr(bit_vectors: list[object], n_bits: int = 2048) -> sparse.csr_matrix:
    """Convert RDKit bit vectors to the historical sparse matrix format."""
    indptr = np.zeros(len(bit_vectors) + 1, dtype=np.int64)
    for index, vector in enumerate(bit_vectors):
        indptr[index + 1] = indptr[index] + vector.GetNumOnBits()
    indices = np.empty(int(indptr[-1]), dtype=np.int32)
    for index, vector in enumerate(bit_vectors):
        start, stop = int(indptr[index]), int(indptr[index + 1])
        indices[start:stop] = np.fromiter(vector.GetOnBits(), dtype=np.int32, count=stop - start)
    return sparse.csr_matrix(
        (np.ones(len(indices), dtype=np.float32), indices, indptr),
        shape=(len(bit_vectors), n_bits),
    )


def smiles_fingerprints(smiles: list[str]) -> list[object]:
    """Build ECFP4 fingerprints with the historical RDKit settings."""
    result: list[object] = []
    for value in smiles:
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"cannot parse canonical SMILES: {value}")
        result.append(AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=2048))
    return result


def load_inputs(chembl_path: Path, generated_path: Path, coordinate_path: Path) -> tuple[sparse.csr_matrix, pd.DataFrame, pd.DataFrame]:
    """Load the exact historical row order and validate all counts."""
    with gzip.open(chembl_path, "rb") as handle:
        payload = pickle.load(handle)
    if payload["radius"] != 2 or payload["n_bits"] != 2048:
        raise ValueError(f"unexpected ChEMBL fingerprint settings: radius={payload['radius']} n_bits={payload['n_bits']}")
    chembl_records = pd.DataFrame(payload["records"])
    chembl_matrix = bitvectors_to_csr(payload["fingerprints"])
    if len(chembl_records) != 100_000 or chembl_matrix.shape != (100_000, 2048):
        raise ValueError("ChEMBL cache is not the expected 100k x 2048 input")
    generated = pd.read_csv(generated_path, low_memory=False)
    required = {"row_id", "repeat", "model", "source_sdf", "source_index", "sanitize_success", "canonical_smiles"}
    if not required.issubset(generated.columns):
        raise ValueError(f"generated table missing columns: {sorted(required.difference(generated.columns))}")
    generated = generated[
        generated["sanitize_success"].eq(1)
        & generated["model"].isin(MODEL_ORDER)
        & generated["canonical_smiles"].notna()
    ].copy()
    expected = {"DiffGui": 2641, "DiffSBDD": 1999, "PMDM": 1726, "GCDM-SBDD": 1570, "TargetDiff": 2974}
    observed = generated.groupby("model").size().to_dict()
    if observed != expected:
        raise ValueError(f"historical generated counts drifted: {observed}")
    generated_matrix = bitvectors_to_csr(smiles_fingerprints(generated["canonical_smiles"].astype(str).tolist()))
    coordinates = pd.read_csv(coordinate_path, compression="infer", low_memory=False)
    if len(coordinates) != 110_910:
        raise ValueError(f"expected 110,910 historical coordinates, found {len(coordinates):,}")
    matrix = sparse.vstack([chembl_matrix, generated_matrix], format="csr")
    return matrix, generated.reset_index(drop=True), coordinates


def write_row_manifest(output_dir: Path, generated: pd.DataFrame, coordinates: pd.DataFrame) -> None:
    """Write the immutable row-order manifest used for the fit and audit."""
    chembl = coordinates[coordinates["model"].eq("ChEMBL36")].copy()
    generated_coordinates = coordinates[coordinates["model"].isin(MODEL_ORDER)].copy()
    if len(chembl) != 100_000 or len(generated_coordinates) != 10_910:
        raise ValueError("coordinate table group counts do not match historical counts")
    rows = pd.DataFrame({"matrix_row": np.arange(len(coordinates), dtype=np.int64), "dataset": coordinates["dataset"].astype(str).to_numpy(), "model": coordinates["model"].astype(str).to_numpy(), "molecule_id": coordinates["molecule_id"].astype(str).to_numpy(), "canonical_smiles": coordinates["canonical_smiles"].astype(str).to_numpy()})
    rows.to_csv(output_dir / "fit_row_order_manifest.csv.gz", index=False, compression="gzip")
    generated.to_csv(output_dir / "historical_generated_valid_rows.csv.gz", index=False, compression="gzip")


def main() -> int:
    """Run a non-destructive historical UMAP rebuild and audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chembl-fingerprints", type=Path, required=True)
    parser.add_argument("--generated-table", type=Path, required=True)
    parser.add_argument("--old-coordinates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke-sdf", type=Path, default=None)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler(args.output_dir / "rebuild.log", encoding="utf-8")])
    started = time.time()
    sources = [args.chembl_fingerprints, args.generated_table, args.old_coordinates]
    provenance = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "python": sys.version, "platform": platform.platform(), "sources": {str(path): {"size": path.stat().st_size, "sha256": sha256(path)} for path in sources}, "parameters": {"n_components": 2, "metric": "cosine", "n_neighbors": 20, "min_dist": 0.1, "random_state": 42, "low_memory": True, "fingerprint": "ECFP4 radius=2 nBits=2048"}, "old_outputs_preserved": True}
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    matrix, generated, coordinates = load_inputs(args.chembl_fingerprints, args.generated_table, args.old_coordinates)
    write_row_manifest(args.output_dir, generated, coordinates)
    sparse.save_npz(args.output_dir / "fit_input_ecfp4_csr.npz", matrix)
    np.save(args.output_dir / "old_coordinates.npy", coordinates[["umap1", "umap2"]].to_numpy(dtype=np.float32))
    LOGGER.info("fitting historical UMAP rows=%d", matrix.shape[0])
    reducer = umap.UMAP(n_components=2, metric="cosine", n_neighbors=20, min_dist=0.1, random_state=42, low_memory=True, verbose=False)
    rebuilt = reducer.fit_transform(matrix)
    expected = coordinates[["umap1", "umap2"]].to_numpy(dtype=np.float32)
    delta = rebuilt - expected
    audit = {"n": int(len(expected)), "max_abs_difference": float(np.max(np.abs(delta))), "rmse": float(np.sqrt(np.mean(delta**2))), "median_euclidean_difference": float(np.median(np.linalg.norm(delta, axis=1))), "pass_threshold_1e-4": bool(np.max(np.abs(delta)) <= 1e-4)}
    pd.DataFrame({"matrix_row": np.arange(len(rebuilt)), "rebuilt_umap1": rebuilt[:, 0], "rebuilt_umap2": rebuilt[:, 1], "old_umap1": expected[:, 0], "old_umap2": expected[:, 1], "delta_umap1": delta[:, 0], "delta_umap2": delta[:, 1]}).to_csv(args.output_dir / "coordinate_reproduction_audit.csv.gz", index=False, compression="gzip")
    (args.output_dir / "coordinate_reproduction_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    np.save(args.output_dir / "rebuilt_coordinates.npy", rebuilt)
    rebuilt_table = coordinates.copy()
    rebuilt_table["umap1"] = rebuilt[:, 0]
    rebuilt_table["umap2"] = rebuilt[:, 1]
    rebuilt_table.to_csv(args.output_dir / "rebuilt_coordinates.csv.gz", index=False, compression="gzip")
    model_path = args.output_dir / "old100k_rebuilt_umap_model.joblib"
    joblib.dump(reducer, model_path, compress=3)
    reloaded = joblib.load(model_path)
    smoke_matrix = matrix[:20]
    smoke_coordinates = reloaded.transform(smoke_matrix)
    smoke = {"n": 20, "shape": list(smoke_coordinates.shape), "finite": bool(np.isfinite(smoke_coordinates).all()), "model_sha256": sha256(model_path), "model_size": model_path.stat().st_size}
    if args.smoke_sdf is not None:
        records: list[object] = []
        supplier = Chem.SDMolSupplier(str(args.smoke_sdf), sanitize=True)
        for molecule in supplier:
            if molecule is not None:
                records.append(AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=2048))
        if records:
            projected = reloaded.transform(bitvectors_to_csr(records))
            smoke["smoke_sdf"] = str(args.smoke_sdf)
            smoke["smoke_sdf_valid"] = len(records)
            smoke["smoke_sdf_finite"] = bool(np.isfinite(projected).all())
            np.save(args.output_dir / "smoke_sdf_projected_coordinates.npy", projected)
    (args.output_dir / "reload_transform_smoke.json").write_text(json.dumps(smoke, indent=2) + "\n", encoding="utf-8")
    provenance["elapsed_seconds"] = round(time.time() - started, 3)
    provenance["audit"] = audit
    provenance["reload_smoke"] = smoke
    (args.output_dir / "provenance_final.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("completed elapsed=%.1fs max_abs=%.6g rmse=%.6g", time.time() - started, audit["max_abs_difference"], audit["rmse"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
