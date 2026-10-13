#!/usr/bin/env python3
"""Build a restartable ECFP4-ChEMBL reward sidecar for frozen endpoints.

Only endpoints that already have a valid reconstructed molecule in the frozen
episode index are scored.  Invalid or technical-failure endpoints remain in
the output with finite sentinel values and explicit failure types, but are
never used as chemical preference examples by the pair builder.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import logging
import math
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import ExplicitBitVect


LOGGER = logging.getLogger("ecfp4_reward_sidecar")
REFERENCE_FPS: list[ExplicitBitVect] = []


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pickle_cache(path: Path) -> tuple[list[ExplicitBitVect], dict[str, Any]]:
    """Load the gzip pickle cache produced by the project ChEMBL tools."""
    import pickle

    with gzip.open(path, "rb") as handle:
        payload = pickle.load(handle)
    fingerprints = list(payload.get("fingerprints", []))
    if not fingerprints:
        raise ValueError(f"empty ChEMBL fingerprint cache: {path}")
    metadata = {
        "radius": int(payload.get("radius", 2)),
        "n_bits": int(payload.get("n_bits", 2048)),
        "fingerprint_type": str(payload.get("fingerprint_type", "ecfp4")),
        "records": len(payload.get("records", [])),
        "cache_size": len(fingerprints),
    }
    if metadata["radius"] != 2 or metadata["n_bits"] != 2048:
        raise ValueError(f"expected ECFP4 radius=2, n_bits=2048; got {metadata}")
    return fingerprints, metadata


def init_worker(reference_fps: list[ExplicitBitVect]) -> None:
    """Install fork-shared immutable reference fingerprints."""
    global REFERENCE_FPS
    REFERENCE_FPS = reference_fps


def fingerprint_from_smiles(smiles: str) -> ExplicitBitVect:
    """Parse one canonical SMILES and calculate ECFP4."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("SMILES parse failed")
    return AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=2048)


def score_query(item: tuple[int, str]) -> tuple[int, float, float, int]:
    """Return top-1, top-5 mean, and a finite success flag for one query."""
    query_index, smiles = item
    fingerprint = fingerprint_from_smiles(smiles)
    similarities = DataStructs.BulkTanimotoSimilarity(fingerprint, REFERENCE_FPS)
    if not similarities:
        raise ValueError("empty similarity result")
    top_k = sorted((float(value) for value in similarities), reverse=True)[:5]
    if not top_k or not all(math.isfinite(value) for value in top_k):
        raise ValueError("non-finite similarity result")
    return query_index, top_k[0], float(np.mean(top_k)), 1


def configure_logging(output_dir: Path) -> None:
    """Configure durable and terminal logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "ecfp4_reward.log", encoding="utf-8"),
        ],
        force=True,
    )


def load_episode_index(path: Path) -> pd.DataFrame:
    """Load the frozen endpoint index and retain its original row order."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            rows.append(row)
    frame = pd.DataFrame(rows)
    required = {"sample_id", "rdkit_valid", "failure_type", "num_nodes", "smiles", "shard_path", "shard_index"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"episode index missing columns: {missing}")
    if frame["sample_id"].duplicated().any():
        raise ValueError("episode index contains duplicate sample_id")
    return frame


def write_chunk(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one compressed, deterministic query chunk."""
    pd.DataFrame(rows).sort_values("query_index").to_csv(path, index=False, compression="gzip")


def score_valid_queries(
    queries: list[tuple[int, str]],
    reference_fps: list[ExplicitBitVect],
    chunks_dir: Path,
    workers: int,
    chunk_size: int,
) -> None:
    """Score valid queries with resumable chunk files."""
    context = mp.get_context("fork")
    started = time.time()
    with context.Pool(workers, initializer=init_worker, initargs=(reference_fps,)) as pool:
        for chunk_id, start in enumerate(range(0, len(queries), chunk_size)):
            stop = min(start + chunk_size, len(queries))
            output = chunks_dir / f"chunk_{chunk_id:05d}_{start}_{stop}.csv.gz"
            expected_indices = list(range(start, stop))
            if output.exists():
                existing = pd.read_csv(output)
                if existing["query_index"].tolist() == expected_indices:
                    LOGGER.info("resume chunk=%d [%d,%d)", chunk_id, start, stop)
                    continue
                raise ValueError(f"existing chunk has unexpected rows: {output}")
            chunk_rows: list[dict[str, Any]] = []
            failures = 0
            for query_index, top1, top5_mean, success in pool.imap(
                score_query,
                queries[start:stop],
                chunksize=1,
            ):
                chunk_rows.append(
                    {
                        "query_index": query_index,
                        "ecfp4_max_similarity": top1,
                        "ecfp4_top5_mean_similarity": top5_mean,
                        "ecfp4_success": success,
                    }
                )
            if len(chunk_rows) != stop - start or sorted(row["query_index"] for row in chunk_rows) != expected_indices:
                raise ValueError(f"incomplete query chunk: {chunk_id}")
            write_chunk(output, chunk_rows)
            LOGGER.info(
                "completed chunk=%d [%d,%d) elapsed=%.1fs failures=%d",
                chunk_id,
                start,
                stop,
                time.time() - started,
                failures,
            )


def plot_reward_distribution(frame: pd.DataFrame, output_dir: Path) -> None:
    """Save a compact reward distribution QA plot."""
    valid = frame.loc[frame["ecfp4_success"].eq(1), "ecfp4_top5_mean_similarity"].dropna()
    if valid.empty:
        return
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].hist(valid.to_numpy(), bins=60, color="#4472c4", alpha=0.9)
    axes[0].set_xlabel("ECFP4 top-5 mean Tanimoto")
    axes[0].set_ylabel("Valid endpoints")
    axes[0].set_title("Frozen endpoint reward distribution")
    axes[1].boxplot(valid.to_numpy(), vert=True, patch_artist=True, boxprops={"facecolor": "#70ad47"})
    axes[1].set_ylabel("ECFP4 top-5 mean Tanimoto")
    axes[1].set_title("Valid endpoint reward summary")
    figure.savefig(output_dir / "ecfp4_reward_distribution.png", dpi=220)
    figure.savefig(output_dir / "ecfp4_reward_distribution.svg")
    plt.close(figure)


def main() -> None:
    """Build the complete finite ECFP4 reward sidecar."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-index", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--chembl-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=28)
    parser.add_argument("--chunk-size", type=int, default=256)
    args = parser.parse_args()
    if args.workers <= 0 or args.chunk_size <= 0:
        raise ValueError("workers and chunk-size must be positive")
    for path in (args.episode_index, args.split_manifest, args.chembl_cache):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir = args.output_dir / "query_chunks"
    chunks_dir.mkdir(exist_ok=True)
    configure_logging(args.output_dir)

    index = load_episode_index(args.episode_index)
    split = pd.read_json(args.split_manifest, lines=True)[["sample_id", "split"]]
    if split["sample_id"].duplicated().any():
        raise ValueError("split manifest contains duplicate sample_id")
    index = index.merge(split, on="sample_id", how="left", validate="one_to_one")
    if index["split"].isna().any():
        raise ValueError("split manifest does not cover every endpoint")
    valid_mask = (
        index["rdkit_valid"].astype(bool)
        & index["failure_type"].eq("success")
        & index["smiles"].notna()
    )
    valid_indices = index.index[valid_mask].tolist()
    queries = [(query_index, str(index.at[row_index, "smiles"])) for query_index, row_index in enumerate(valid_indices)]
    reference_fps, cache_metadata = load_pickle_cache(args.chembl_cache)
    LOGGER.info(
        "start endpoints=%d valid_queries=%d reference=%d workers=%d chunk_size=%d",
        len(index),
        len(queries),
        len(reference_fps),
        args.workers,
        args.chunk_size,
    )
    score_valid_queries(queries, reference_fps, chunks_dir, args.workers, args.chunk_size)
    chunk_paths = sorted(chunks_dir.glob("chunk_*.csv.gz"))
    scored = pd.concat([pd.read_csv(path) for path in chunk_paths], ignore_index=True)
    if len(scored) != len(queries) or sorted(scored["query_index"].tolist()) != list(range(len(queries))):
        raise ValueError("query chunks are incomplete")
    scored = scored.sort_values("query_index").reset_index(drop=True)
    scored["source_index_row"] = [valid_indices[int(value)] for value in scored["query_index"]]
    index["ecfp4_max_similarity"] = np.nan
    index["ecfp4_top5_mean_similarity"] = np.nan
    index["ecfp4_success"] = 0
    index.loc[scored["source_index_row"], "ecfp4_max_similarity"] = scored["ecfp4_max_similarity"].to_numpy()
    index.loc[scored["source_index_row"], "ecfp4_top5_mean_similarity"] = scored["ecfp4_top5_mean_similarity"].to_numpy()
    index.loc[scored["source_index_row"], "ecfp4_success"] = scored["ecfp4_success"].to_numpy()
    index["ecfp4_reward"] = index["ecfp4_top5_mean_similarity"].fillna(-1.0)
    index["ecfp4_failure_type"] = np.where(index["ecfp4_success"].eq(1), "success", index["failure_type"].astype(str))
    index["ecfp4_reward_finite"] = np.isfinite(index["ecfp4_reward"])
    output_columns = [
        "sample_id", "split", "rdkit_valid", "failure_type", "ecfp4_failure_type",
        "num_nodes", "shard_path", "shard_index", "smiles", "ecfp4_success",
        "ecfp4_max_similarity", "ecfp4_top5_mean_similarity", "ecfp4_reward", "ecfp4_reward_finite",
    ]
    output = index[output_columns].copy()
    output.to_csv(args.output_dir / "ecfp4_reward_sidecar.csv", index=False)
    valid = output.loc[output["ecfp4_success"].eq(1), "ecfp4_reward"]
    summary = {
        "status": "completed",
        "episode_index": str(args.episode_index.resolve()),
        "episode_index_sha256": sha256_file(args.episode_index),
        "split_manifest": str(args.split_manifest.resolve()),
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "chembl_cache": str(args.chembl_cache.resolve()),
        "chembl_cache_sha256": sha256_file(args.chembl_cache),
        "endpoints": int(len(output)),
        "valid_queries": int(output["ecfp4_success"].sum()),
        "technical_failures": int((output["ecfp4_success"] == 0).sum()),
        "workers": int(args.workers),
        "chunk_size": int(args.chunk_size),
        "reward": {
            "name": "ecfp4_top5_mean_similarity",
            "min": float(valid.min()),
            "max": float(valid.max()),
            "mean": float(valid.mean()),
            "median": float(valid.median()),
            "quantiles": {str(q): float(valid.quantile(q)) for q in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)},
        },
        "cache": cache_metadata,
        "output": str((args.output_dir / "ecfp4_reward_sidecar.csv").resolve()),
    }
    (args.output_dir / "ecfp4_reward_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_reward_distribution(output, args.output_dir)
    LOGGER.info("completed sidecar=%s valid=%d failed=%d", summary["output"], summary["valid_queries"], summary["technical_failures"])


if __name__ == "__main__":
    main()
