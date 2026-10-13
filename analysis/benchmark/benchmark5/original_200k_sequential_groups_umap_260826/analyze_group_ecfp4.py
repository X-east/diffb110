#!/usr/bin/env python3
"""Quantify ECFP4 diversity and ChEMBL support for sequential groups."""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import DataStructs
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from scipy import sparse

LOGGER = logging.getLogger("group_ecfp4")
N_BITS = 2048
RADIUS = 2

_MATRIX: sparse.csr_matrix | None = None
_REFERENCE_FPS: list[ExplicitBitVect] = []


def parse_args() -> argparse.Namespace:
    """Parse input, reference, output, and worker options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--chembl-package",
        type=Path,
        default=Path("data/chembl/chembl_36_100k_ecfp4_fingerprints.pkl.gz"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=28)
    parser.add_argument("--chembl-samples-per-group", type=int, default=50)
    return parser.parse_args()


def configure_logging(output_dir: Path) -> None:
    """Configure terminal and durable analysis logging."""
    output_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "ecfp4_analysis.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(stream)
    root.addHandler(file_handler)


def load_matrix(path: Path) -> sparse.csr_matrix:
    """Load the traced selected ECFP4 CSR matrix."""
    with np.load(path) as loaded:
        required = {"data", "indices", "indptr", "shape"}
        missing = required.difference(loaded.files)
        if missing:
            raise ValueError(f"missing CSR fields in {path}: {sorted(missing)}")
        shape = tuple(int(value) for value in loaded["shape"])
        matrix = sparse.csr_matrix(
            (loaded["data"], loaded["indices"], loaded["indptr"]),
            shape=shape,
        )
    if matrix.shape[1] != N_BITS:
        raise ValueError(f"expected {N_BITS} fingerprint bits, got {matrix.shape}")
    return matrix


def fingerprint_key(matrix: sparse.csr_matrix, row: int) -> tuple[int, ...]:
    """Return a collision-free hashable representation of one sparse bit row."""
    start, stop = int(matrix.indptr[row]), int(matrix.indptr[row + 1])
    return tuple(int(value) for value in matrix.indices[start:stop])


def pairwise_metrics(matrix: sparse.csr_matrix, rows: np.ndarray) -> dict[str, float]:
    """Calculate exact all-pairs and nearest-neighbor Tanimoto metrics."""
    group_matrix = matrix[rows].tocsr()
    n_rows = group_matrix.shape[0]
    if n_rows < 2:
        return {
            "pairwise_n": 0.0,
            "mean_pairwise_tanimoto": np.nan,
            "median_pairwise_tanimoto": np.nan,
            "p95_pairwise_tanimoto": np.nan,
            "mean_nearest_neighbor_tanimoto": np.nan,
            "median_nearest_neighbor_tanimoto": np.nan,
            "p95_nearest_neighbor_tanimoto": np.nan,
        }
    intersections = (group_matrix @ group_matrix.T).toarray().astype(np.float32, copy=False)
    on_bits = np.asarray(group_matrix.sum(axis=1)).ravel().astype(np.float32)
    unions = on_bits[:, None] + on_bits[None, :] - intersections
    similarities = np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections),
        where=unions > 0,
    )
    np.fill_diagonal(similarities, np.nan)
    pairwise = similarities[np.triu_indices(n_rows, k=1)]
    nearest = np.nanmax(similarities, axis=1)
    return {
        "pairwise_n": float(pairwise.size),
        "mean_pairwise_tanimoto": float(np.nanmean(pairwise)),
        "median_pairwise_tanimoto": float(np.nanmedian(pairwise)),
        "p95_pairwise_tanimoto": float(np.nanquantile(pairwise, 0.95)),
        "mean_nearest_neighbor_tanimoto": float(np.nanmean(nearest)),
        "median_nearest_neighbor_tanimoto": float(np.nanmedian(nearest)),
        "p95_nearest_neighbor_tanimoto": float(np.nanquantile(nearest, 0.95)),
    }


def initialize_group_worker(matrix: sparse.csr_matrix) -> None:
    """Install the selected sparse matrix in a forked group worker."""
    global _MATRIX
    _MATRIX = matrix


def group_task(task: tuple[int, np.ndarray, list[str], int]) -> dict[str, Any]:
    """Calculate one group's exact fingerprint metrics."""
    group_index, rows, canonical_smiles, fixed_attempts = task
    if _MATRIX is None:
        raise RuntimeError("group worker matrix is not initialized")
    group_matrix = _MATRIX[rows].tocsr()
    on_bits = np.asarray(group_matrix.sum(axis=1)).ravel().astype(float)
    unique_fingerprints = len({fingerprint_key(group_matrix, row) for row in range(len(rows))})
    metrics = pairwise_metrics(_MATRIX, rows)
    unique_canonical = len(set(canonical_smiles))
    return {
        "group_index": group_index,
        "fixed_attempts": fixed_attempts,
        "valid_count": len(rows),
        "valid_rate": len(rows) / fixed_attempts,
        "unique_canonical_count": unique_canonical,
        "unique_canonical_fraction": unique_canonical / len(rows),
        "unique_ecfp4_count": unique_fingerprints,
        "unique_ecfp4_fraction": unique_fingerprints / len(rows),
        "mean_on_bits": float(np.mean(on_bits)),
        "median_on_bits": float(np.median(on_bits)),
        "std_on_bits": float(np.std(on_bits, ddof=1)),
        "p05_on_bits": float(np.quantile(on_bits, 0.05)),
        "p95_on_bits": float(np.quantile(on_bits, 0.95)),
        **metrics,
    }


def initialize_similarity_worker(reference_fps: list[ExplicitBitVect]) -> None:
    """Install immutable ChEMBL reference fingerprints in a forked worker."""
    global _REFERENCE_FPS
    _REFERENCE_FPS = reference_fps


def bit_vector_from_row(matrix: sparse.csr_matrix, row: int) -> ExplicitBitVect:
    """Convert one CSR fingerprint row to an RDKit explicit bit vector."""
    bit_vector = ExplicitBitVect(matrix.shape[1])
    start, stop = int(matrix.indptr[row]), int(matrix.indptr[row + 1])
    bit_vector.SetBitsFromList(matrix.indices[start:stop].tolist())
    return bit_vector


def similarity_task(task: tuple[int, int, int]) -> tuple[int, int, float, float]:
    """Scan ChEMBL100k for one sampled query molecule."""
    if _MATRIX is None or not _REFERENCE_FPS:
        raise RuntimeError("similarity worker state is not initialized")
    group_index, query_row, sample_order = task
    query = bit_vector_from_row(_MATRIX, query_row)
    similarities = np.asarray(
        DataStructs.BulkTanimotoSimilarity(query, _REFERENCE_FPS), dtype=np.float32
    )
    top5 = np.partition(similarities, -5)[-5:]
    return group_index, sample_order, float(similarities.max()), float(top5.mean())


def load_reference(path: Path) -> list[ExplicitBitVect]:
    """Load and validate the fixed ChEMBL100k ECFP4 package."""
    with gzip.open(path, "rb") as handle:
        payload = pickle.load(handle)
    if payload.get("radius") != RADIUS or payload.get("n_bits") != N_BITS:
        raise ValueError(f"unexpected ChEMBL fingerprint config in {path}")
    fingerprints = payload["fingerprints"]
    records = payload["records"]
    if len(fingerprints) != len(records) or len(fingerprints) != 100_000:
        raise ValueError(f"expected 100,000 ChEMBL fingerprints, got {len(fingerprints)}")
    return fingerprints


def reference_baseline(reference_fps: list[ExplicitBitVect], pair_count: int = 100_000) -> dict[str, float | int]:
    """Summarize ChEMBL100k ECFP4 density and a deterministic random-pair baseline."""
    on_bits = np.asarray([fingerprint.GetNumOnBits() for fingerprint in reference_fps], dtype=float)
    rng = np.random.default_rng(42)
    left = rng.integers(0, len(reference_fps), size=pair_count)
    right = rng.integers(0, len(reference_fps), size=pair_count)
    similarities = np.fromiter(
        (
            DataStructs.TanimotoSimilarity(reference_fps[index_a], reference_fps[index_b])
            for index_a, index_b in zip(left, right)
        ),
        dtype=float,
        count=pair_count,
    )
    return {
        "reference_n": len(reference_fps),
        "random_pair_n": pair_count,
        "random_seed": 42,
        "mean_on_bits": float(on_bits.mean()),
        "median_on_bits": float(np.median(on_bits)),
        "random_pair_mean_tanimoto": float(similarities.mean()),
        "random_pair_median_tanimoto": float(np.median(similarities)),
        "random_pair_p95_tanimoto": float(np.quantile(similarities, 0.95)),
    }


def summarize_sampled_chembl(
    frame: pd.DataFrame,
    matrix: sparse.csr_matrix,
    reference_path: Path,
    workers: int,
    samples_per_group: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate deterministic sampled top-1/top-5 ChEMBL similarities."""
    reference_fps = load_reference(reference_path)
    tasks: list[tuple[int, int, int]] = []
    for group_index, group in frame.groupby("group_index", sort=True):
        rows = group["selected_valid_row"].to_numpy(dtype=np.int64)
        sample_count = min(samples_per_group, len(rows))
        sample_positions = np.linspace(0, len(rows) - 1, sample_count, dtype=np.int64)
        tasks.extend(
            (int(group_index), int(rows[position]), sample_order)
            for sample_order, position in enumerate(sample_positions)
        )
    LOGGER.info(
        "ChEMBL100k sampled scan: queries=%d comparisons=%d workers=%d",
        len(tasks),
        len(tasks) * len(reference_fps),
        workers,
    )
    global _MATRIX, _REFERENCE_FPS
    _MATRIX = matrix
    _REFERENCE_FPS = reference_fps
    context = mp.get_context("fork")
    with context.Pool(
        processes=min(workers, len(tasks)),
        initializer=initialize_similarity_worker,
        initargs=(reference_fps,),
    ) as pool:
        result_rows = pool.map(similarity_task, tasks, chunksize=1)
    sampled = pd.DataFrame(
        result_rows,
        columns=[
            "group_index",
            "sample_order",
            "chembl100k_top1_tanimoto",
            "chembl100k_top5_mean_tanimoto",
        ],
    ).sort_values(["group_index", "sample_order"])
    summary = (
        sampled.groupby("group_index", sort=True)
        .agg(
            chembl100k_sample_n=("chembl100k_top1_tanimoto", "size"),
            chembl100k_top1_mean=("chembl100k_top1_tanimoto", "mean"),
            chembl100k_top1_median=("chembl100k_top1_tanimoto", "median"),
            chembl100k_top1_p95=("chembl100k_top1_tanimoto", lambda values: values.quantile(0.95)),
            chembl100k_top5_mean=("chembl100k_top5_mean_tanimoto", "mean"),
            chembl100k_top5_median=("chembl100k_top5_mean_tanimoto", "median"),
        )
        .reset_index()
    )
    return sampled, summary


def save_figure(fig: plt.Figure, stem: Path) -> None:
    """Save a Python-generated PNG and editable SVG."""
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_metrics(summary: pd.DataFrame, output_dir: Path) -> None:
    """Plot per-group ECFP4 composition, diversity, and ChEMBL support."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
            "font.size": 10,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    x = summary["group_index"]
    panels = [
        ("valid_rate", "Valid / fixed attempts (%)", 100, "#3B82C4"),
        ("unique_canonical_fraction", "Unique canonical SMILES (%)", 100, "#287D8E"),
        ("unique_ecfp4_fraction", "Unique ECFP4 fingerprints (%)", 100, "#4C9F70"),
        ("mean_on_bits", "Mean ECFP4 on-bits", 1, "#7A5AA6"),
        ("mean_pairwise_tanimoto", "Mean within-group Tanimoto", 1, "#D95F59"),
        ("chembl100k_top1_mean", "Sampled ChEMBL100k top-1 Tanimoto", 1, "#C07A24"),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.2), sharex=True)
    for axis, (column, ylabel, scale, color) in zip(axes.ravel(), panels):
        axis.plot(x, summary[column] * scale, color=color, marker="o", markersize=3, linewidth=1.5)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
        axis.set_xticks([1, 10, 20, 30, 40, 50])
        axis.set_xlabel("Sequential group")
    figure.suptitle("ECFP4 properties across 50 sequential Original groups", fontsize=17)
    figure.text(
        0.5,
        0.012,
        "Within-group Tanimoto is exact over all valid pairs; ChEMBL100k values use 50 evenly spaced valid queries per group.",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.95))
    save_figure(figure, output_dir / "ecfp4_group_metrics")

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for axis, mean_column, std_column, ylabel, color in (
        (axes[0], "mean_on_bits", "std_on_bits", "ECFP4 on-bits", "#7A5AA6"),
        (axes[1], "mean_pairwise_tanimoto", None, "Within-group mean Tanimoto", "#D95F59"),
        (axes[2], "chembl100k_top1_mean", None, "Sampled ChEMBL100k top-1 Tanimoto", "#C07A24"),
    ):
        axis.plot(x, summary[mean_column], color=color, marker="o", markersize=3, linewidth=1.5)
        if std_column is not None:
            low = summary[mean_column] - summary[std_column]
            high = summary[mean_column] + summary[std_column]
            axis.fill_between(x, low, high, color=color, alpha=0.16, linewidth=0)
        axis.set_xlabel("Sequential group")
        axis.set_ylabel(ylabel)
        axis.set_xticks([1, 10, 20, 30, 40, 50])
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    figure.suptitle("Group-level ECFP4 composition and similarity", fontsize=17)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(figure, output_dir / "ecfp4_group_similarity_summary")


def main() -> None:
    """Run exact group ECFP4 analysis and sampled ChEMBL comparison."""
    args = parse_args()
    if args.workers < 1 or args.chembl_samples_per_group < 1:
        raise ValueError("workers and chembl-samples-per-group must be positive")
    configure_logging(args.output_dir)
    started = time.time()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    frame = pd.read_csv(input_dir / "valid_molecules_embedding.csv.gz")
    required = {"selected_valid_row", "group_index", "canonical_smiles"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing input columns: {sorted(missing)}")
    frame = frame.sort_values("selected_valid_row").reset_index(drop=True)
    matrix = load_matrix(input_dir / "selected_ecfp4_matrix.npz")
    if matrix.shape[0] != len(frame) or frame["selected_valid_row"].tolist() != list(range(len(frame))):
        raise ValueError(f"matrix/frame row mismatch: matrix={matrix.shape} frame={frame.shape}")
    LOGGER.info("input valid molecules=%d matrix=%s workers=%d", len(frame), matrix.shape, args.workers)

    group_tasks = [
        (
            int(group_index),
            group["selected_valid_row"].to_numpy(dtype=np.int64),
            group["canonical_smiles"].astype(str).tolist(),
            3_000,
        )
        for group_index, group in frame.groupby("group_index", sort=True)
    ]
    global _MATRIX
    _MATRIX = matrix
    context = mp.get_context("fork")
    with context.Pool(
        processes=min(args.workers, len(group_tasks)),
        initializer=initialize_group_worker,
        initargs=(matrix,),
    ) as pool:
        group_rows = pool.map(group_task, group_tasks, chunksize=1)
    summary = pd.DataFrame(group_rows).sort_values("group_index").reset_index(drop=True)
    sampled, chembl_summary = summarize_sampled_chembl(
        frame,
        matrix,
        args.chembl_package.resolve(),
        args.workers,
        args.chembl_samples_per_group,
    )
    summary = summary.merge(chembl_summary, on="group_index", how="left", validate="one_to_one")
    summary["ecfp4_diversity_1_minus_mean_pairwise"] = 1.0 - summary["mean_pairwise_tanimoto"]
    summary.to_csv(output_dir / "group_ecfp4_stats.csv", index=False)
    sampled.to_csv(output_dir / "group_ecfp4_sampled_chembl100k.csv.gz", index=False, compression="gzip")
    reference_stats = reference_baseline(load_reference(args.chembl_package.resolve()))
    (output_dir / "ecfp4_reference_baseline.json").write_text(
        json.dumps(reference_stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    molecule_rows = frame[["selected_valid_row", "global_attempt", "group_index", "subgroup_index", "sample_id", "canonical_smiles"]].copy()
    molecule_rows["ecfp4_on_bits"] = np.asarray(matrix.sum(axis=1)).ravel().astype(np.int16)
    molecule_rows["ecfp4_fingerprint_key"] = [
        ",".join(str(value) for value in fingerprint_key(matrix, row))
        for row in range(matrix.shape[0])
    ]
    molecule_rows.to_csv(output_dir / "valid_molecule_ecfp4_metrics.csv.gz", index=False, compression="gzip")
    plot_metrics(summary, output_dir)

    payload = {
        "status": "complete",
        "input_dir": str(input_dir),
        "selected_valid_molecules": int(len(frame)),
        "groups": int(summary["group_index"].nunique()),
        "fixed_attempts_per_group": 3_000,
        "fingerprint": {"type": "ECFP4", "radius": RADIUS, "n_bits": N_BITS, "metric": "Tanimoto"},
        "within_group_similarity": "exact all valid-molecule pairs within each 3,000-attempt group",
        "chembl_reference": str(args.chembl_package.resolve()),
        "chembl_reference_count": 100_000,
        "chembl_query_sampling": {
            "queries_per_group": args.chembl_samples_per_group,
            "selection": "evenly spaced valid rows in each group",
            "comparisons": int(len(sampled) * 100_000),
        },
        "chembl_reference_baseline": reference_stats,
        "workers": args.workers,
        "elapsed_seconds": round(time.time() - started, 3),
        "interpretation": "ECFP4 Tanimoto is fingerprint similarity; it is separate from UMAP coordinate distance and does not establish activity or docking quality.",
    }
    (output_dir / "group_ecfp4_completion.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    LOGGER.info("completed group ECFP4 analysis in %.1fs", time.time() - started)


if __name__ == "__main__":
    main()
