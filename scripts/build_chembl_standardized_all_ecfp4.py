#!/usr/bin/env python3
"""Build a resumable sparse ECFP4 cache for all standardized ChEMBL parents."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import sparse
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import AllChem


RDLogger.DisableLog("rdApp.error")
LOGGER = logging.getLogger("chembl_standardized_all_ecfp4")


def parse_args() -> argparse.Namespace:
    """Parse cache-build arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=28)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--expected-count", type=int, default=2_734_034)
    parser.add_argument("--stage", choices=["chunks", "merge"], default="chunks")
    return parser.parse_args()


def sha256(path: Path) -> str:
    """Return a streaming SHA256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint_smiles(smiles: str) -> np.ndarray:
    """Return sorted ECFP4 on-bit indices for one canonical SMILES."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("RDKit could not parse standardized canonical SMILES")
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048)
    return np.fromiter(fingerprint.GetOnBits(), dtype=np.int32)


def _fingerprint_task(task: tuple[int, str]) -> tuple[int, np.ndarray]:
    """Fingerprint one indexed molecule in a worker process."""
    index, smiles = task
    return index, fingerprint_smiles(smiles)


def configure_logging(output_dir: Path) -> None:
    """Configure durable rank-zero-style logging."""
    output_dir.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    handler = logging.FileHandler(output_dir / "fingerprint_build.log", encoding="utf-8")
    handlers.append(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=handlers,
        force=True,
    )


def read_chunks(input_path: Path, chunk_size: int):
    """Yield fixed source-order chunks with exact global indices."""
    import gzip

    with gzip.open(input_path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        chunk: list[tuple[int, str]] = []
        for index, row in enumerate(reader):
            smiles = str(row["canonical_smiles"])
            chunk.append((index, smiles))
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


def write_chunk(path: Path, start: int, results: list[tuple[int, np.ndarray]]) -> None:
    """Atomically write one CSR-compatible chunk."""
    stop = start + len(results)
    if [index for index, _ in results] != list(range(start, stop)):
        raise ValueError(f"noncontiguous fingerprint result range [{start}, {stop})")
    indptr = np.zeros(len(results) + 1, dtype=np.int64)
    for row, (_, on_bits) in enumerate(results):
        indptr[row + 1] = indptr[row] + len(on_bits)
    indices = np.empty(int(indptr[-1]), dtype=np.int32)
    for row, (_, on_bits) in enumerate(results):
        indices[indptr[row]:indptr[row + 1]] = on_bits
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, start=np.int64(start), stop=np.int64(stop), indptr=indptr, indices=indices)
    temporary.replace(path)


def main() -> None:
    """Build all chunks, skipping independently validated completed chunks."""
    args = parse_args()
    configure_logging(args.output_dir)
    if args.workers < 1 or args.chunk_size < 1:
        raise ValueError("workers and chunk-size must be positive")
    chunk_dir = args.output_dir / "ecfp4_chunks"
    chunk_dir.mkdir(exist_ok=True)
    if args.stage == "merge":
        chunks = sorted(chunk_dir.glob("chunk_*.npz"))
        if not chunks:
            raise FileNotFoundError(f"no fingerprint chunks under {chunk_dir}")
        indptr_parts = [np.array([0], dtype=np.int64)]
        indices_parts: list[np.ndarray] = []
        expected_start = 0
        total_nnz = 0
        for path in chunks:
            with np.load(path) as chunk:
                start, stop = int(chunk["start"]), int(chunk["stop"])
                indptr, indices = chunk["indptr"], chunk["indices"]
            if start != expected_start or len(indptr) != stop - start + 1:
                raise ValueError(f"invalid or discontinuous chunk: {path}")
            indptr_parts.append(indptr[1:] + total_nnz)
            indices_parts.append(indices)
            expected_start, total_nnz = stop, total_nnz + len(indices)
        if expected_start != args.expected_count:
            raise ValueError(f"expected {args.expected_count:,} rows, found {expected_start:,}")
        matrix = sparse.csr_matrix(
            (np.ones(total_nnz, dtype=np.float32), np.concatenate(indices_parts), np.concatenate(indptr_parts)),
            shape=(args.expected_count, 2048),
        )
        cache_path = args.output_dir / "chembl_36_standardized_all_ecfp4_csr.npz"
        sparse.save_npz(cache_path, matrix, compressed=True)
        qc = {"status": "complete", "rows": int(matrix.shape[0]), "columns": int(matrix.shape[1]), "nnz": int(matrix.nnz), "chunk_count": len(chunks), "finite": bool(np.isfinite(matrix.data).all())}
        (args.output_dir / "ecfp4_cache_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
        LOGGER.info("merged CSR cache: %s shape=%s nnz=%d", cache_path, matrix.shape, matrix.nnz)
        return
    LOGGER.info("input=%s sha256=%s", args.input, sha256(args.input))
    LOGGER.info("ECFP4 radius=2 bits=2048 workers=%d chunk_size=%d", args.workers, args.chunk_size)
    completed = 0
    with Pool(processes=args.workers) as pool:
        for chunk_index, tasks in enumerate(read_chunks(args.input, args.chunk_size)):
            start, stop = tasks[0][0], tasks[-1][0] + 1
            output = chunk_dir / f"chunk_{chunk_index:05d}_{start}_{stop}.npz"
            if output.exists():
                with np.load(output) as cached:
                    valid = int(cached["start"]) == start and int(cached["stop"]) == stop and len(cached["indptr"]) == len(tasks) + 1
                if valid:
                    completed += len(tasks)
                    LOGGER.info("resume chunk=%d range=[%d,%d) completed=%d", chunk_index, start, stop, completed)
                    continue
                raise ValueError(f"invalid existing chunk: {output}")
            results = pool.map(_fingerprint_task, tasks, chunksize=128)
            write_chunk(output, start, results)
            completed += len(tasks)
            LOGGER.info("completed chunk=%d range=[%d,%d) molecules=%d", chunk_index, start, stop, completed)
    if completed != args.expected_count:
        raise ValueError(f"expected {args.expected_count:,} molecules, completed {completed:,}")
    manifest = {
        "status": "chunks_complete",
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "molecule_count": completed,
        "fingerprint": {"type": "ECFP4", "radius": 2, "n_bits": 2048},
        "workers": args.workers,
        "chunk_size": args.chunk_size,
        "rdkit": rdBase.rdkitVersion,
    }
    temporary = args.output_dir / "fingerprint_manifest.tmp.json"
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(args.output_dir / "fingerprint_manifest.json")
    LOGGER.info("fingerprint chunks complete: n=%d", completed)


if __name__ == "__main__":
    main()
