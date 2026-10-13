#!/usr/bin/env python3
"""Project trajectory-linked DiffSBDD molecules into the five-model UMAP reference."""

from __future__ import annotations

import argparse
import gzip
import json
import logging
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
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy import sparse
import umap

RDLogger.DisableLog("rdApp.*")
LOGGER = logging.getLogger(__name__)
MODEL_ORDER = ["DiffGui", "DiffSBDD", "PMDM", "GCDM-SBDD", "TargetDiff"]
MODEL_COLORS = {
    "DiffGui": "#3B82C4",
    "DiffSBDD": "#D95F59",
    "PMDM": "#D9A441",
    "GCDM-SBDD": "#8E78B7",
    "TargetDiff": "#3D9B72",
}


def _fingerprint_one(task: tuple[int, str, int, int]) -> tuple[int, Any | None]:
    """Calculate one Morgan bit vector in a worker while preserving index."""
    index, smiles, radius, n_bits = task
    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return index, None
        return index, AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=n_bits)
    except Exception:  # noqa: BLE001
        return index, None


def _compute_fingerprints(
    smiles: list[str], radius: int, n_bits: int, workers: int
) -> list[Any]:
    """Calculate ordered ECFP4 fingerprints using the requested worker count."""
    tasks = [(index, value, radius, n_bits) for index, value in enumerate(smiles)]
    context = mp.get_context("spawn")
    with context.Pool(processes=workers) as pool:
        results = pool.map(_fingerprint_one, tasks, chunksize=128)
    failures = [index for index, fingerprint in results if fingerprint is None]
    if failures:
        raise ValueError(f"fingerprint failures: {len(failures)}")
    return [fingerprint for _, fingerprint in results]


def _bitvectors_to_csr(bit_vectors: list[Any], n_bits: int) -> sparse.csr_matrix:
    """Convert RDKit bit vectors to the sparse representation used by the old benchmark."""
    indptr = np.zeros(len(bit_vectors) + 1, dtype=np.int64)
    for index, bit_vector in enumerate(bit_vectors):
        indptr[index + 1] = indptr[index] + bit_vector.GetNumOnBits()
    indices = np.empty(int(indptr[-1]), dtype=np.int32)
    for index, bit_vector in enumerate(bit_vectors):
        start, stop = int(indptr[index]), int(indptr[index + 1])
        indices[start:stop] = np.fromiter(bit_vector.GetOnBits(), dtype=np.int32, count=stop - start)
    return sparse.csr_matrix(
        (np.ones(len(indices), dtype=np.float32), indices, indptr),
        shape=(len(bit_vectors), n_bits),
    )


def _load_chembl(path: Path) -> tuple[list[Any], pd.DataFrame, int, int]:
    """Load the exact fixed ChEMBL100k cache used in the old benchmark."""
    with gzip.open(path, "rb") as handle:
        payload = pickle.load(handle)
    records = pd.DataFrame(payload["records"])
    fingerprints = payload["fingerprints"]
    if len(records) != 100_000 or len(fingerprints) != 100_000:
        raise ValueError("expected exactly 100,000 ChEMBL records and fingerprints")
    return fingerprints, records, int(payload["radius"]), int(payload["n_bits"])


def _load_old_generated(path: Path) -> pd.DataFrame:
    """Load the exact sanitize-valid five-model rows used by the old benchmark."""
    columns = ["row_id", "repeat", "model", "source_sdf", "source_index", "sanitize_success", "canonical_smiles"]
    frame = pd.read_csv(path, usecols=columns)
    frame = frame[
        frame["sanitize_success"].eq(1)
        & frame["model"].isin(MODEL_ORDER)
        & frame["canonical_smiles"].notna()
    ].copy()
    observed = frame.groupby("model").size().to_dict()
    expected = {"DiffGui": 2641, "DiffSBDD": 1999, "PMDM": 1726, "GCDM-SBDD": 1570, "TargetDiff": 2974}
    if observed != expected:
        raise ValueError(f"old benchmark rows drifted: {observed}")
    return frame.reset_index(drop=True)


def _load_trajectory_manifest(path: Path) -> pd.DataFrame:
    """Load only valid trajectory molecules while retaining their source identifiers."""
    frame = pd.read_csv(path, low_memory=False)
    required = {"trajectory_id", "molecule_id", "rdkit_valid", "canonical_smiles", "rank", "shard_index", "mol_index", "sdf_path", "trajectory_shard"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"trajectory manifest missing columns: {sorted(missing)}")
    frame = frame[frame["rdkit_valid"].eq(1) & frame["canonical_smiles"].notna()].copy()
    if len(frame) != 67_694:
        raise ValueError(f"expected 67,694 valid trajectory rows, found {len(frame):,}")
    if frame["trajectory_id"].duplicated().any():
        raise ValueError("trajectory_id is not unique")
    return frame.reset_index(drop=True)


def _reference_table(chembl: pd.DataFrame, old_generated: pd.DataFrame, coordinates: np.ndarray) -> pd.DataFrame:
    """Attach reference coordinates to fixed ChEMBL and historical benchmark rows."""
    chembl_table = pd.DataFrame(
        {
            "dataset": "ChEMBL36 100k",
            "model": "ChEMBL36",
            "molecule_id": chembl["chembl_id"].astype(str),
            "canonical_smiles": chembl["canonical_smiles"].astype(str),
        }
    )
    generated_table = pd.DataFrame(
        {
            "dataset": "Historical benchmark",
            "model": old_generated["model"].astype(str),
            "molecule_id": old_generated["row_id"].astype(str),
            "canonical_smiles": old_generated["canonical_smiles"].astype(str),
        }
    )
    table = pd.concat([chembl_table, generated_table], ignore_index=True)
    table["umap1"] = coordinates[:, 0]
    table["umap2"] = coordinates[:, 1]
    return table


def _plot_reference(reference: pd.DataFrame, output_path: Path) -> None:
    """Plot the reconstructed five-model reference in the old visual style."""
    figure, axis = plt.subplots(figsize=(9.0, 7.2), dpi=220)
    chembl = reference[reference["model"] == "ChEMBL36"]
    axis.scatter(chembl["umap1"], chembl["umap2"], s=0.7, c="#A8ADB3", alpha=0.20, linewidths=0, label="ChEMBL36 (100k)")
    for model in MODEL_ORDER:
        group = reference[reference["model"] == model]
        axis.scatter(group["umap1"], group["umap2"], s=3.5, c=MODEL_COLORS[model], alpha=0.55, linewidths=0, label=f"{model} (n={len(group):,})")
    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2")
    axis.set_title("Five-model benchmark reference (ECFP4, joint UMAP)")
    axis.legend(frameon=False, markerscale=3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def _plot_trajectory_projection(reference: pd.DataFrame, trajectory: pd.DataFrame, output_path: Path) -> None:
    """Plot current trajectory molecules without refitting the reference coordinates."""
    figure, axis = plt.subplots(figsize=(9.0, 7.2), dpi=220)
    chembl = reference[reference["model"] == "ChEMBL36"]
    axis.scatter(chembl["umap1"], chembl["umap2"], s=0.7, c="#A8ADB3", alpha=0.20, linewidths=0, label="ChEMBL36 (100k)")
    axis.scatter(trajectory["umap1"], trajectory["umap2"], s=2.3, c="#D95F59", alpha=0.30, linewidths=0, label=f"DiffSBDD trajectory-valid (n={len(trajectory):,})")
    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2")
    axis.set_title("DiffSBDD trajectory projected into five-model UMAP reference")
    axis.legend(frameon=False, markerscale=4)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chembl-fingerprints", type=Path, required=True)
    parser.add_argument("--old-generated-csv", type=Path, required=True)
    parser.add_argument("--trajectory-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=28)
    parser.add_argument("--n-neighbors", type=int, default=20)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    """Rebuild the old reference and project valid DiffSBDD trajectories into it."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(args.output_dir / "run.log", mode="w", encoding="utf-8")],
    )
    start = time.time()
    chembl_fps, chembl, radius, n_bits = _load_chembl(args.chembl_fingerprints)
    old_generated = _load_old_generated(args.old_generated_csv)
    trajectory = _load_trajectory_manifest(args.trajectory_manifest)
    LOGGER.info("computing historical fingerprints rows=%d workers=%d", len(old_generated), args.workers)
    old_fps = _compute_fingerprints(old_generated["canonical_smiles"].tolist(), radius, n_bits, args.workers)
    reference_matrix = sparse.vstack([_bitvectors_to_csr(chembl_fps, n_bits), _bitvectors_to_csr(old_fps, n_bits)], format="csr")
    LOGGER.info("fitting historical reference UMAP rows=%d", reference_matrix.shape[0])
    reducer = umap.UMAP(n_components=2, metric="cosine", n_neighbors=args.n_neighbors, min_dist=args.min_dist, random_state=args.random_state, low_memory=True, verbose=True)
    reference_coordinates = reducer.fit_transform(reference_matrix)
    # The fitted UMAP object retains the full neighbor graph and raw matrix.
    # Serializing it can exceed the size of the source trajectory collection,
    # so retain the reference coordinates rather than writing a huge reducer.
    reference = _reference_table(chembl, old_generated, reference_coordinates)
    reference.to_csv(args.output_dir / "five_model_reference_coordinates.csv.gz", index=False, compression="gzip")
    _plot_reference(reference, args.output_dir / "five_model_reference_rebuilt.png")

    LOGGER.info("computing trajectory fingerprints rows=%d workers=%d", len(trajectory), args.workers)
    trajectory_fps = _compute_fingerprints(trajectory["canonical_smiles"].tolist(), radius, n_bits, args.workers)
    trajectory_matrix = _bitvectors_to_csr(trajectory_fps, n_bits)
    LOGGER.info("transforming trajectory rows into frozen reference UMAP")
    trajectory_coordinates = reducer.transform(trajectory_matrix)
    trajectory["umap1"] = trajectory_coordinates[:, 0]
    trajectory["umap2"] = trajectory_coordinates[:, 1]
    trajectory.to_csv(args.output_dir / "diffsbdd_trajectory_projected_coordinates.csv.gz", index=False, compression="gzip")
    _plot_trajectory_projection(reference, trajectory, args.output_dir / "diffsbdd_trajectory_projected_on_reference.png")

    summary = {
        "status": "completed",
        "reference": {"chembl": len(chembl), "historical_generated": len(old_generated), "reference_rows": len(reference), "historical_model_counts": old_generated.groupby("model").size().to_dict()},
        "trajectory": {"valid_rows": len(trajectory), "trajectory_id_duplicates": int(trajectory["trajectory_id"].duplicated().sum())},
        "ecfp4": {"radius": radius, "n_bits": n_bits},
        "umap": {"metric": "cosine", "n_neighbors": args.n_neighbors, "min_dist": args.min_dist, "random_state": args.random_state, "fit_scope": "old five-model benchmark only", "new_trajectory_operation": "transform"},
        "workers": args.workers,
        "elapsed_seconds": round(time.time() - start, 3),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("complete elapsed_seconds=%.1f", time.time() - start)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
