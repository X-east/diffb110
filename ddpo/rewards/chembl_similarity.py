"""Shared ChEMBL fingerprint similarity utilities.

This module is intentionally independent from PPO/DDPO runtime state so it can
be reused by both offline analysis scripts and reward components.
"""
from __future__ import annotations

import gzip
import pickle
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from rdkit import DataStructs


@dataclass(frozen=True)
class ChEMBLFingerprintCache:
    """Loaded ChEMBL Morgan fingerprint reference cache."""

    fingerprints: list[Any]
    records: list[dict[str, Any]]
    radius: int
    n_bits: int
    source_path: Path

    @property
    def size(self) -> int:
        return len(self.fingerprints)


def load_chembl_fingerprint_cache(path: str | Path) -> ChEMBLFingerprintCache:
    """Load a cache produced by ``scripts/prepare_chembl_fingerprints.py``."""
    cache_path = Path(path).expanduser().resolve()
    with gzip.open(cache_path, "rb") as handle:
        payload = pickle.load(handle)

    fingerprints = list(payload.get("fingerprints", []))
    if not fingerprints:
        raise ValueError(f"No fingerprints found in ChEMBL cache: {cache_path}")

    return ChEMBLFingerprintCache(
        fingerprints=fingerprints,
        records=list(payload.get("records", [])),
        radius=int(payload.get("radius", 2)),
        n_bits=int(payload.get("n_bits", 2048)),
        source_path=cache_path,
    )


def compute_topk_tanimoto(
    query_fp: Any,
    reference_fps: list[Any],
    top_k: int = 5,
    chunk_size: int = 100000,
) -> tuple[float, float, list[float]]:
    """Compute max and top-k mean Tanimoto similarity against references.

    Args:
        query_fp: RDKit ExplicitBitVect for the query molecule.
        reference_fps: Loaded RDKit fingerprint references.
        top_k: Number of nearest similarities to average.
        chunk_size: Number of reference fingerprints processed per bulk call.

    Returns:
        ``(max_similarity, topk_mean_similarity, sorted_topk_values)``.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not reference_fps:
        return 0.0, 0.0, []

    best: list[float] = []
    for start in range(0, len(reference_fps), chunk_size):
        sims = DataStructs.BulkTanimotoSimilarity(
            query_fp,
            reference_fps[start : start + chunk_size],
        )
        if not sims:
            continue

        if len(best) < top_k:
            best.extend(float(value) for value in sims)
            best.sort(reverse=True)
            best = best[:top_k]
            continue

        cutoff = best[-1]
        candidates = [float(value) for value in sims if float(value) > cutoff]
        if candidates:
            best.extend(candidates)
            best.sort(reverse=True)
            best = best[:top_k]

    if not best:
        return 0.0, 0.0, []
    return float(best[0]), float(mean(best[:top_k])), best[:top_k]
