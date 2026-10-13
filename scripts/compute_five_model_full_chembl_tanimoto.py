#!/usr/bin/env python3
"""Compute exact full-ChEMBL ECFP4 nearest neighbors for five-model molecules."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import DataStructs
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from scipy import sparse


REFERENCE_FPS: list[ExplicitBitVect] = []


def sha256(path: Path) -> str:
    """Return the SHA256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_logging(output_dir: Path) -> None:
    """Write progress both to terminal and durable task log."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output_dir / "tanimoto.log", encoding="utf-8")],
        force=True,
    )


def csr_to_bitvectors(matrix: sparse.csr_matrix) -> list[ExplicitBitVect]:
    """Reconstruct immutable RDKit bit vectors once for fork-shared scanning."""
    fingerprints: list[ExplicitBitVect] = []
    for row in range(matrix.shape[0]):
        bit_vector = ExplicitBitVect(matrix.shape[1])
        bit_vector.SetBitsFromList(matrix.indices[matrix.indptr[row]:matrix.indptr[row + 1]].tolist())
        fingerprints.append(bit_vector)
    return fingerprints


def init_worker(fingerprints: list[ExplicitBitVect]) -> None:
    """Install the fork-shared immutable full ChEMBL reference fingerprints."""
    global REFERENCE_FPS
    REFERENCE_FPS = fingerprints


def fingerprint_from_bits(on_bits: np.ndarray, n_bits: int = 2048) -> ExplicitBitVect:
    """Build one RDKit bit vector from sorted on-bit positions."""
    bit_vector = ExplicitBitVect(n_bits)
    bit_vector.SetBitsFromList(on_bits.tolist())
    return bit_vector


def nearest_neighbor(task: tuple[int, np.ndarray]) -> tuple[int, int, float]:
    """Scan all reference fingerprints for one exact top-1 Tanimoto neighbor."""
    query_index, on_bits = task
    similarities = DataStructs.BulkTanimotoSimilarity(fingerprint_from_bits(on_bits), REFERENCE_FPS)
    reference_index, similarity = max(enumerate(similarities), key=lambda pair: pair[1])
    return query_index, reference_index, float(similarity)


def query_csr(frame: pd.DataFrame) -> sparse.csr_matrix:
    """Rebuild ECFP4 query CSR from exact projected input identities."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    indptr = np.zeros(len(frame) + 1, dtype=np.int64)
    rows: list[np.ndarray] = []
    for index, smiles in enumerate(frame["canonical_smiles"].astype(str)):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"unreadable canonical SMILES at projected row {index}")
        on_bits = np.fromiter(AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048).GetOnBits(), dtype=np.int32)
        rows.append(on_bits)
        indptr[index + 1] = indptr[index] + len(on_bits)
    indices = np.concatenate(rows) if rows else np.empty(0, dtype=np.int32)
    return sparse.csr_matrix((np.ones(len(indices), dtype=np.float32), indices, indptr), shape=(len(frame), 2048))


def main() -> None:
    """Run restart-safe exact nearest-neighbor comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--chembl-csr", type=Path, required=True)
    parser.add_argument("--chembl-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=28)
    parser.add_argument("--query-chunk-size", type=int, default=100)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(args.output_dir)
    LOGGER = logging.getLogger("tanimoto")
    queries = pd.read_csv(args.queries)
    required = {"row_id", "model", "repeat", "canonical_smiles", "umap1", "umap2"}
    if missing := required.difference(queries):
        raise ValueError(f"missing query fields: {sorted(missing)}")
    reference_records = pd.read_csv(args.chembl_csv, usecols=["chembl_id", "canonical_smiles"])
    reference_matrix = sparse.load_npz(args.chembl_csr).tocsr()
    if reference_matrix.shape != (len(reference_records), 2048):
        raise ValueError(f"reference mismatch: {reference_matrix.shape} vs {len(reference_records)} records")
    LOGGER.info("building fork-shared RDKit reference fingerprints: n=%d", reference_matrix.shape[0])
    started = time.time()
    reference_fps = csr_to_bitvectors(reference_matrix)
    LOGGER.info("reference ready in %.1fs; queries=%d comparisons=%d", time.time() - started, len(queries), len(queries) * len(reference_fps))
    query_matrix = query_csr(queries)
    chunks_dir = args.output_dir / "query_chunks"
    chunks_dir.mkdir(exist_ok=True)
    context = mp.get_context("fork")
    with context.Pool(args.workers, initializer=init_worker, initargs=(reference_fps,)) as pool:
        for chunk_number, start in enumerate(range(0, len(queries), args.query_chunk_size)):
            stop = min(start + args.query_chunk_size, len(queries))
            output = chunks_dir / f"chunk_{chunk_number:04d}_{start}_{stop}.csv.gz"
            if output.exists():
                existing = pd.read_csv(output)
                if len(existing) == stop - start and existing["query_index"].tolist() == list(range(start, stop)):
                    LOGGER.info("resume chunk=%d [%d,%d)", chunk_number, start, stop)
                    continue
                raise ValueError(f"invalid completed chunk: {output}")
            tasks = [(index, query_matrix.indices[query_matrix.indptr[index]:query_matrix.indptr[index + 1]]) for index in range(start, stop)]
            rows = pool.map(nearest_neighbor, tasks, chunksize=1)
            result = pd.DataFrame(rows, columns=["query_index", "nearest_chembl_index", "chembl_ecfp4_top1_similarity"])
            result.to_csv(output, index=False, compression="gzip")
            LOGGER.info("complete chunk=%d [%d,%d) elapsed=%.1fs", chunk_number, start, stop, time.time() - started)
    chunk_paths = sorted(chunks_dir.glob("chunk_*.csv.gz"))
    result = pd.concat([pd.read_csv(path) for path in chunk_paths], ignore_index=True).sort_values("query_index")
    if len(result) != len(queries) or result["query_index"].tolist() != list(range(len(queries))):
        raise ValueError("query chunks do not form a complete ordered result")
    reference_match = reference_records.iloc[result["nearest_chembl_index"].to_numpy()].reset_index(drop=True)
    output = queries.reset_index(drop=True).copy()
    output["nearest_chembl_index"] = result["nearest_chembl_index"].to_numpy()
    output["chembl_ecfp4_top1_similarity"] = result["chembl_ecfp4_top1_similarity"].to_numpy()
    output["nearest_chembl_id"] = reference_match["chembl_id"].to_numpy()
    output["nearest_chembl_smiles"] = reference_match["canonical_smiles"].to_numpy()
    output.to_csv(args.output_dir / "five_model_full_chembl_exact_tanimoto.csv.gz", index=False, compression="gzip")
    summary = output.groupby("model")["chembl_ecfp4_top1_similarity"].agg(["count", "mean", "median", lambda x: x.quantile(.05), lambda x: x.quantile(.95)]).reset_index()
    summary.columns = ["model", "n", "mean", "median", "p05", "p95"]
    summary.to_csv(args.output_dir / "five_model_full_chembl_exact_tanimoto_summary.csv", index=False)
    manifest = {"status": "complete", "queries": str(args.queries.resolve()), "queries_sha256": sha256(args.queries), "chembl_csr": str(args.chembl_csr.resolve()), "chembl_csr_sha256": sha256(args.chembl_csr), "chembl_csv": str(args.chembl_csv.resolve()), "chembl_csv_sha256": sha256(args.chembl_csv), "queries": int(len(queries)), "reference": int(len(reference_records)), "comparisons": int(len(queries) * len(reference_records)), "workers": args.workers, "chunk_size": args.query_chunk_size, "fingerprint": {"type": "ECFP4", "radius": 2, "n_bits": 2048}, "elapsed_seconds": time.time() - started}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info("completed exact similarities in %.1fs", time.time() - started)


if __name__ == "__main__":
    main()
