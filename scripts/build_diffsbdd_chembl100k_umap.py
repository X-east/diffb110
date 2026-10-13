#!/usr/bin/env python3
"""Build a trajectory-linked DiffSBDD/ChEMBL ECFP4 joint UMAP dataset."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")
LOGGER = logging.getLogger(__name__)


def _fingerprint_one(task: tuple[str, int, int]) -> tuple[int, np.ndarray | None]:
    """Compute one ECFP4 bit vector in a worker process."""
    index, smiles, radius, n_bits = task
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return index, None
        bit_vector = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        bits = np.fromiter((int(bit_vector.GetBit(i)) for i in range(n_bits)), dtype=np.uint8)
        return index, bits
    except Exception:  # noqa: BLE001
        return index, None


def _compute_fingerprints(
    smiles: list[str], radius: int, n_bits: int, workers: int
) -> tuple[np.ndarray, list[int]]:
    """Compute fingerprints with explicit worker count and preserve row order."""
    tasks = [(i, value, radius, n_bits) for i, value in enumerate(smiles)]
    matrix = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    failed: list[int] = []
    context = mp.get_context("spawn")
    with context.Pool(processes=workers) as pool:
        for index, bits in pool.imap_unordered(_fingerprint_one, tasks, chunksize=128):
            if bits is None:
                failed.append(index)
            else:
                matrix[index] = bits
    return matrix, sorted(failed)


def _load_chembl(path: Path) -> pd.DataFrame:
    """Load the exact 100k ChEMBL reference records from the fingerprint cache."""
    import gzip as _gzip
    import pickle

    with _gzip.open(path, "rb") as handle:
        payload = pickle.load(handle)
    records = pd.DataFrame(payload["records"])
    expected = {"chembl_id", "canonical_smiles"}
    missing = expected.difference(records.columns)
    if missing:
        raise ValueError(f"ChEMBL records missing columns: {sorted(missing)}")
    if len(records) != 100_000:
        raise ValueError(f"expected 100,000 ChEMBL records, found {len(records):,}")
    return records.reset_index(drop=True)


def _load_diffsbdd(run_dir: Path) -> pd.DataFrame:
    """Load all reconstruction rows and attach stable trajectory file paths."""
    molecule_root = run_dir / "molecules"
    rows: list[dict[str, Any]] = []
    csv_paths = sorted(molecule_root.glob("rank_*/shard_*/reconstruction.csv"))
    if len(csv_paths) != 10_000:
        raise ValueError(f"expected 10,000 reconstruction CSVs, found {len(csv_paths):,}")
    for csv_path in csv_paths:
        rank_name = csv_path.parent.parent.name
        shard_name = csv_path.parent.name
        rank = int(rank_name.split("_")[-1])
        shard_index = int(shard_name.split("_")[-1])
        sdf_path = csv_path.with_name(f"diffsbdd_collect_r{rank:02d}_s{shard_index:04d}.sdf")
        trajectory_path = run_dir / "shards" / rank_name / f"shard_{shard_index:04d}.pt"
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                sample_id = row["sample_id"]
                expected_id = f"r{rank:02d}_s{shard_index:04d}_m{int(row['mol_index']):04d}"
                if sample_id != expected_id:
                    raise ValueError(f"unstable sample id in {csv_path}: {sample_id} != {expected_id}")
                rows.append(
                    {
                        "dataset": "DiffSBDD",
                        "model": "DiffSBDD",
                        "molecule_id": sample_id,
                        "trajectory_id": sample_id,
                        "rank": rank,
                        "shard_index": shard_index,
                        "mol_index": int(row["mol_index"]),
                        "canonical_smiles": row["smiles"] if row["rdkit_valid"] == "1" else "",
                        "rdkit_valid": int(row["rdkit_valid"]),
                        "failure_type": row["failure_type"],
                        "reconstruction_csv": str(csv_path.resolve()),
                        "sdf_path": str(sdf_path.resolve()),
                        "trajectory_shard": str(trajectory_path.resolve()),
                    }
                )
    frame = pd.DataFrame(rows)
    if len(frame) != 100_000:
        raise ValueError(f"expected 100,000 DiffSBDD rows, found {len(frame):,}")
    if frame["trajectory_id"].duplicated().any():
        raise ValueError("trajectory_id is not unique")
    return frame.sort_values(["rank", "shard_index", "mol_index"], ignore_index=True)


def _write_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a compressed CSV without changing identifiers."""
    frame.to_csv(path, index=False, compression="gzip")


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--chembl-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=28)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-neighbors", type=int, default=20)
    parser.add_argument("--min-dist", type=float, default=0.1)
    return parser.parse_args()


def main() -> int:
    """Build fingerprints, joint UMAP coordinates, plots, and provenance."""
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.out_dir / "logs" / "fingerprint_umap.log", mode="w"),
        ],
    )
    start = time.time()
    run_dir = args.run_dir.resolve()
    chembl_cache = args.chembl_cache.resolve()
    LOGGER.info("loading trajectory-linked DiffSBDD rows")
    diffsbdd = _load_diffsbdd(run_dir)
    valid = diffsbdd[diffsbdd["rdkit_valid"] == 1].copy().reset_index(drop=True)
    LOGGER.info("DiffSBDD attempts=%d valid=%d", len(diffsbdd), len(valid))
    chembl = _load_chembl(chembl_cache)
    LOGGER.info("ChEMBL reference rows=%d", len(chembl))

    all_smiles = chembl["canonical_smiles"].tolist() + valid["canonical_smiles"].tolist()
    LOGGER.info("computing ECFP4 with workers=%d for rows=%d", args.workers, len(all_smiles))
    fingerprints, failed = _compute_fingerprints(all_smiles, args.radius, args.n_bits, args.workers)
    if failed:
        raise ValueError(f"fingerprint failures found: {len(failed)}")
    np.savez_compressed(args.out_dir / "joint_ecfp4_fingerprints.npz", fingerprints=fingerprints)

    LOGGER.info("fitting deterministic joint UMAP")
    reducer = umap.UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.random_state,
        low_memory=True,
    )
    coordinates = reducer.fit_transform(fingerprints.astype(np.float32, copy=False))
    chembl_coords = pd.DataFrame(
        {
            "dataset": "ChEMBL",
            "model": "ChEMBL36",
            "molecule_id": chembl["chembl_id"].astype(str),
            "trajectory_id": "",
            "rank": -1,
            "shard_index": -1,
            "mol_index": -1,
            "canonical_smiles": chembl["canonical_smiles"],
            "rdkit_valid": 1,
            "failure_type": "",
            "umap1": coordinates[: len(chembl), 0],
            "umap2": coordinates[: len(chembl), 1],
        }
    )
    diffsbdd_coords = valid.drop(
        columns=["reconstruction_csv", "sdf_path", "trajectory_shard"]
    ).copy()
    diffsbdd_coords["umap1"] = coordinates[len(chembl) :, 0]
    diffsbdd_coords["umap2"] = coordinates[len(chembl) :, 1]
    coordinate_table = pd.concat([chembl_coords, diffsbdd_coords], ignore_index=True)
    _write_gzip_csv(coordinate_table, args.out_dir / "joint_umap_coordinates.csv.gz")
    _write_gzip_csv(diffsbdd, args.out_dir / "diffsbdd_100k_manifest.csv.gz")

    fig, ax = plt.subplots(figsize=(8.5, 7.0), dpi=220)
    ax.scatter(chembl_coords["umap1"], chembl_coords["umap2"], s=2, c="#BDBDBD", alpha=0.12, linewidths=0, label="ChEMBL36 (100k)")
    ax.scatter(diffsbdd_coords["umap1"], diffsbdd_coords["umap2"], s=3, c="#D95F59", alpha=0.28, linewidths=0, label=f"DiffSBDD-valid (n={len(valid):,})")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title("DiffSBDD-valid vs ChEMBL36 (ECFP4, joint UMAP)")
    ax.legend(frameon=False, markerscale=3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "diffsbdd_valid_vs_chembl_umap.png", dpi=300)
    fig.savefig(args.out_dir / "diffsbdd_valid_vs_chembl_umap.svg")
    plt.close(fig)

    summary = {
        "status": "completed",
        "run_dir": str(run_dir),
        "chembl_cache": str(chembl_cache),
        "workers": args.workers,
        "fingerprint": {"type": "ECFP4", "radius": args.radius, "n_bits": args.n_bits},
        "umap": {"metric": "cosine", "n_neighbors": args.n_neighbors, "min_dist": args.min_dist, "random_state": args.random_state, "n_components": 2},
        "counts": {"diffsbdd_attempts": len(diffsbdd), "diffsbdd_valid": len(valid), "diffsbdd_invalid": int(len(diffsbdd) - len(valid)), "chembl": len(chembl), "joint_rows": len(coordinate_table)},
        "identifier": {"primary_key": "trajectory_id", "format": "rXX_sXXXX_mYYYY", "duplicate_count": int(diffsbdd["trajectory_id"].duplicated().sum()), "valid_rows_with_trajectory_shard": int(valid["trajectory_shard"].notna().sum())},
        "elapsed_seconds": round(time.time() - start, 3),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("completed in %.1f seconds; output=%s", time.time() - start, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
