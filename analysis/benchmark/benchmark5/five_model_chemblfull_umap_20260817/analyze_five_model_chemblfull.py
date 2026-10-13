#!/usr/bin/env python3
"""Project five formal generator groups into the full ChEMBL36 ECFP4 space."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import logging
import os
import pickle
import platform
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/diffb110_chemblfull_matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/diffb110_chemblfull_numba")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
import scipy
import sklearn
import umap
import yaml
from rdkit import Chem, DataStructs, RDLogger, rdBase
from rdkit.Chem import AllChem
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits


RDLogger.DisableLog("rdApp.error")
LOGGER = logging.getLogger("five_model_chemblfull_umap")

MODEL_ORDER = ["DiffGui", "DiffSBDD", "PMDM", "GCDM-SBDD", "TargetDiff"]
MODEL_COLORS = {
    "DiffGui": "#3B82C4",
    "DiffSBDD": "#D95F59",
    "PMDM": "#D9A441",
    "GCDM-SBDD": "#8E78B7",
    "TargetDiff": "#3D9B72",
}

_TRANSFORM_REDUCER: Any | None = None
_TRANSFORM_MATRIX: sparse.csr_matrix | None = None
_TRANSFORM_BASE_SEED = 42
_TRANSFORM_THREAD_LIMITER: Any | None = None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=["all", "embed", "summarize", "plot"], default="all"
    )
    return parser.parse_args()


def _load_yaml(path: Path, resolve_analysis_paths: bool = False) -> dict[str, Any]:
    """Load one YAML mapping and optionally resolve its analysis paths."""
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    if resolve_analysis_paths:
        base = path.resolve().parent
        for key in ["chembl_full", "chembl_landmarks", "generated_csv", "output_dir"]:
            value = config.get(key)
            if value is None:
                raise ValueError(f"missing required config key: {key}")
            candidate = Path(value)
            config[key] = candidate if candidate.is_absolute() else (base / candidate).resolve()
    return config


def _configure_logging(output_dir: Path) -> None:
    """Write chronological progress to both terminal and durable local log."""
    output_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "run.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(stream)
    root.addHandler(file_handler)
    logging.getLogger("fontTools").setLevel(logging.WARNING)


def _configure_plotting(style: dict[str, Any]) -> None:
    """Apply presentation-readable publication plotting defaults."""
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": style["font_size"],
            "axes.titlesize": style["title_size"],
            "axes.labelsize": style["axis_label_size"],
            "xtick.labelsize": style["tick_size"],
            "ytick.labelsize": style["tick_size"],
            "legend.fontsize": style["legend_size"],
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically so an interruption cannot leave a false sentinel."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _load_fingerprint_package(path: Path, expected_count: int | None) -> dict[str, Any]:
    """Load and validate a traced ChEMBL ECFP4 package."""
    LOGGER.info("loading fingerprint package: %s", path)
    with gzip.open(path, "rb") as handle:
        payload = pickle.load(handle)
    required = {"fingerprints", "records", "radius", "n_bits", "fingerprint_type", "chembl_version"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"fingerprint package is missing keys: {sorted(missing)}")
    if len(payload["fingerprints"]) != len(payload["records"]):
        raise ValueError("fingerprint and record counts differ")
    if expected_count is not None and len(payload["records"]) != expected_count:
        raise ValueError(
            f"expected {expected_count:,} records in {path.name}, found {len(payload['records']):,}"
        )
    if payload["radius"] != 2 or payload["n_bits"] != 2048:
        raise ValueError("formal analysis requires ECFP4 radius=2 and n_bits=2048")
    LOGGER.info("validated %s: n=%d", path.name, len(payload["records"]))
    return payload


def _fingerprint_task(task: tuple[int, str, int, int]) -> tuple[int, Any | None]:
    """Calculate one generated-molecule Morgan bit vector."""
    index, smiles, radius, n_bits = task
    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return index, None
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=n_bits)
        return index, fingerprint
    except Exception:
        return index, None


def _load_generated(
    path: Path, radius: int, n_bits: int, workers: int
) -> tuple[pd.DataFrame, list[Any]]:
    """Load formal sanitize-valid molecules and calculate ECFP4 in parallel."""
    usecols = [
        "row_id", "repeat", "model", "source_sdf", "source_index",
        "sanitize_success", "canonical_smiles",
    ]
    frame = pd.read_csv(path, usecols=usecols)
    frame = frame[
        frame["sanitize_success"].eq(1)
        & frame["model"].isin(MODEL_ORDER)
        & frame["canonical_smiles"].notna()
    ].copy()
    tasks = [
        (index, smiles, radius, n_bits)
        for index, smiles in enumerate(frame["canonical_smiles"].astype(str))
    ]
    LOGGER.info("calculating generated ECFP4: n=%d workers=%d", len(tasks), workers)
    with Pool(processes=workers) as pool:
        results = pool.map(_fingerprint_task, tasks, chunksize=64)
    valid = [(index, fingerprint) for index, fingerprint in results if fingerprint is not None]
    if len(valid) != len(frame):
        LOGGER.warning("dropped %d generated molecules without ECFP4", len(frame) - len(valid))
    frame = frame.iloc[[index for index, _ in valid]].reset_index(drop=True)
    fingerprints = [fingerprint for _, fingerprint in valid]
    counts = frame.groupby("model").size().reindex(MODEL_ORDER, fill_value=0)
    LOGGER.info("generated valid counts:\n%s", counts.to_string())
    return frame, fingerprints


def _bitvectors_to_csr(bit_vectors: list[Any], n_bits: int) -> sparse.csr_matrix:
    """Convert RDKit bit vectors to a compact sparse CSR matrix."""
    indptr = np.zeros(len(bit_vectors) + 1, dtype=np.int64)
    for index, bit_vector in enumerate(bit_vectors):
        indptr[index + 1] = indptr[index] + bit_vector.GetNumOnBits()
    indices = np.empty(int(indptr[-1]), dtype=np.int32)
    for index, bit_vector in enumerate(bit_vectors):
        start, stop = int(indptr[index]), int(indptr[index + 1])
        indices[start:stop] = np.fromiter(
            bit_vector.GetOnBits(), dtype=np.int32, count=stop - start
        )
    data = np.ones(len(indices), dtype=np.float32)
    matrix = sparse.csr_matrix((data, indices, indptr), shape=(len(bit_vectors), n_bits))
    LOGGER.info(
        "built CSR matrix: shape=%s nnz=%d memory=%.1f MiB",
        matrix.shape,
        matrix.nnz,
        (matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes) / 1024**2,
    )
    return matrix


def _match_landmarks(
    full_payload: dict[str, Any], landmark_payload: dict[str, Any]
) -> np.ndarray:
    """Map every fixed landmark to its exact row in the full ChEMBL package."""
    full_ids = [str(record["chembl_id"]) for record in full_payload["records"]]
    if len(set(full_ids)) != len(full_ids):
        raise ValueError("full ChEMBL package contains duplicate CHEMBL IDs")
    index_by_id = {chembl_id: index for index, chembl_id in enumerate(full_ids)}
    landmark_ids = [str(record["chembl_id"]) for record in landmark_payload["records"]]
    try:
        indices = np.asarray([index_by_id[chembl_id] for chembl_id in landmark_ids], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"landmark CHEMBL ID is absent from full package: {exc}") from exc
    mismatches = sum(
        DataStructs.TanimotoSimilarity(
            landmark_payload["fingerprints"][landmark_index],
            full_payload["fingerprints"][full_index],
        ) < 1.0
        for landmark_index, full_index in enumerate(indices)
    )
    if mismatches:
        raise ValueError(f"{mismatches} landmark fingerprints differ from the full package")
    LOGGER.info("matched and fingerprint-verified %d fixed landmarks", len(indices))
    return indices


def _save_generated_coordinates(
    generated: pd.DataFrame, coordinates: np.ndarray, output_path: Path
) -> None:
    """Save generated coordinates with exact source identity."""
    table = generated.copy()
    table["umap1"] = coordinates[:, 0]
    table["umap2"] = coordinates[:, 1]
    table.to_csv(output_path, index=False, compression="gzip")


def _fit_or_load_reducer(
    config: dict[str, Any],
    landmark_matrix: sparse.csr_matrix,
    generated_matrix: sparse.csr_matrix,
    generated: pd.DataFrame,
) -> Any:
    """Fit the fixed landmark map once or load its resumable model."""
    output_dir = config["output_dir"]
    reducer_path = output_dir / "umap_model.joblib"
    generated_path = output_dir / "generated_coordinates.csv.gz"
    landmark_path = output_dir / "landmark_coordinates.npy"
    if reducer_path.exists() and generated_path.exists() and landmark_path.exists():
        LOGGER.info("loading existing fitted UMAP model: %s", reducer_path)
        return joblib.load(reducer_path)

    fit_matrix = sparse.vstack([landmark_matrix, generated_matrix], format="csr")
    reducer = umap.UMAP(
        n_components=2,
        metric=config["umap"]["metric"],
        n_neighbors=int(config["umap"]["n_neighbors"]),
        min_dist=float(config["umap"]["min_dist"]),
        random_state=int(config["umap"]["random_state"]),
        transform_seed=int(config["umap"]["transform_seed"]),
        n_jobs=1,
        low_memory=True,
        verbose=True,
    )
    LOGGER.info("fitting deterministic landmark UMAP: shape=%s fit_threads=1", fit_matrix.shape)
    fit_coordinates = reducer.fit_transform(fit_matrix)
    landmark_count = landmark_matrix.shape[0]
    np.save(landmark_path, fit_coordinates[:landmark_count].astype(np.float32))
    _save_generated_coordinates(generated, fit_coordinates[landmark_count:], generated_path)
    joblib.dump(reducer, reducer_path, compress=3)
    LOGGER.info("saved fitted UMAP model and fit coordinates")
    return reducer


def _reset_chunk_search_state(reducer: Any, seed: int) -> None:
    """Make each transform chunk deterministic and independent for safe resume."""
    search_index = getattr(reducer, "_knn_search_index", None)
    if search_index is None or not hasattr(search_index, "search_rng_state"):
        raise RuntimeError("installed UMAP/PyNNDescent lacks resumable search_rng_state")
    rng = np.random.RandomState(seed)
    search_index.search_rng_state = rng.randint(-(2**31), 2**31 - 1, 3).astype(np.int64)


def _initialize_transform_worker() -> None:
    """Limit each forked transform worker to one CPU thread."""
    global _TRANSFORM_THREAD_LIMITER
    _TRANSFORM_THREAD_LIMITER = threadpool_limits(limits=1)
    numba.set_num_threads(1)
    if _TRANSFORM_REDUCER is not None:
        _TRANSFORM_REDUCER.verbose = False


def _transform_chunk_task(
    task: tuple[int, int, int, np.ndarray]
) -> tuple[int, int, int, np.ndarray, np.ndarray]:
    """Transform one deterministic ChEMBL chunk in a forked worker."""
    chunk_index, start, stop, query_indices = task
    if _TRANSFORM_REDUCER is None or _TRANSFORM_MATRIX is None:
        raise RuntimeError("transform worker was not initialized from the parent process")
    _reset_chunk_search_state(_TRANSFORM_REDUCER, _TRANSFORM_BASE_SEED + chunk_index)
    transformed = _TRANSFORM_REDUCER.transform(
        _TRANSFORM_MATRIX[query_indices]
    ).astype(np.float32)
    return chunk_index, start, stop, query_indices, transformed


def _transform_full_chembl(
    config: dict[str, Any],
    reducer: Any,
    full_matrix: sparse.csr_matrix,
    landmark_indices: np.ndarray,
) -> Path:
    """Transform all non-landmark ChEMBL rows in restart-safe fixed chunks."""
    output_dir = config["output_dir"]
    coordinate_path = output_dir / "chembl_coordinates.npy"
    landmark_coordinates = np.load(output_dir / "landmark_coordinates.npy")
    n_rows = full_matrix.shape[0]
    if not coordinate_path.exists():
        coordinates = np.lib.format.open_memmap(
            coordinate_path, mode="w+", dtype=np.float32, shape=(n_rows, 2)
        )
        coordinates[:] = np.nan
        coordinates[landmark_indices] = landmark_coordinates
        coordinates.flush()
    coordinates = np.load(coordinate_path, mmap_mode="r+")
    if coordinates.shape != (n_rows, 2):
        raise ValueError(f"unexpected coordinate array shape: {coordinates.shape}")
    coordinates[landmark_indices] = landmark_coordinates

    is_landmark = np.zeros(n_rows, dtype=bool)
    is_landmark[landmark_indices] = True
    workers = int(config["workers"])
    configured_chunk_size = int(config["transform_chunk_size"])
    # Ensure enough independent tasks to occupy all requested CPU cores even
    # if an older copied config still contains a coarse serial chunk size.
    parallel_chunk_size = int(np.ceil(n_rows / workers))
    chunk_size = min(configured_chunk_size, parallel_chunk_size)
    if chunk_size <= 10_000:
        raise ValueError("transform_chunk_size must exceed 10,000 to keep UMAP at 30 transform epochs")
    chunk_dir = output_dir / "transform_chunks"
    chunk_dir.mkdir(exist_ok=True)
    tasks: list[tuple[int, int, int, np.ndarray]] = []
    for chunk_index, start in enumerate(range(0, n_rows, chunk_size)):
        stop = min(start + chunk_size, n_rows)
        sentinel = chunk_dir / f"chunk_{chunk_index:04d}_{start}_{stop}.done.json"
        query_indices = np.arange(start, stop, dtype=np.int64)
        query_indices = query_indices[~is_landmark[start:stop]]
        if sentinel.exists():
            if np.isnan(coordinates[query_indices]).any():
                raise ValueError(f"completed chunk contains NaN coordinates: {sentinel}")
            LOGGER.info("resume skip chunk=%d range=[%d,%d) queries=%d", chunk_index, start, stop, len(query_indices))
            continue
        # Reuse only whole ranges written by an earlier chunking scheme. A
        # partial range is recomputed as one >10k batch so UMAP keeps 30 epochs.
        if np.isfinite(coordinates[query_indices]).all():
            _atomic_json(
                sentinel,
                {"chunk_index": chunk_index, "start": start, "stop": stop,
                 "query_count": int(len(query_indices)),
                 "reused_existing_coordinates": True, "finite": True},
            )
            continue
        tasks.append((chunk_index, start, stop, query_indices))

    if tasks:
        global _TRANSFORM_REDUCER, _TRANSFORM_MATRIX, _TRANSFORM_BASE_SEED
        _TRANSFORM_REDUCER = reducer
        _TRANSFORM_MATRIX = full_matrix
        _TRANSFORM_BASE_SEED = int(config["umap"]["transform_seed"])
        process_count = min(workers, len(tasks))
        LOGGER.info(
            "transforming full ChEMBL: rows=%d configured_chunk_size=%d effective_chunk_size=%d chunks=%d processes=%d threads_per_process=1",
            n_rows, configured_chunk_size, chunk_size, len(tasks), process_count,
        )
        with Pool(processes=process_count, initializer=_initialize_transform_worker) as pool:
            results = pool.imap_unordered(_transform_chunk_task, tasks, chunksize=1)
            for completed, result in enumerate(results, start=1):
                chunk_index, start, stop, query_indices, transformed = result
                coordinates[query_indices] = transformed
                coordinates.flush()
                sentinel = chunk_dir / f"chunk_{chunk_index:04d}_{start}_{stop}.done.json"
                _atomic_json(
                    sentinel,
                    {
                        "chunk_index": chunk_index,
                        "start": start,
                        "stop": stop,
                        "query_count": int(len(query_indices)),
                        "transform_seed": _TRANSFORM_BASE_SEED + chunk_index,
                        "finite": bool(np.isfinite(transformed).all()),
                        "process_parallel": True,
                    },
                )
                LOGGER.info(
                    "transform complete chunk=%d progress=%d/%d queries=%d",
                    chunk_index, completed, len(tasks), len(query_indices),
                )
    if not np.isfinite(coordinates).all():
        raise ValueError("full ChEMBL coordinate array contains non-finite values")
    return coordinate_path


def _write_chembl_coordinate_csv(
    records: list[dict[str, Any]], coordinate_path: Path, output_path: Path
) -> None:
    """Write full ChEMBL identities and coordinates as a compact gzip CSV."""
    if output_path.exists():
        LOGGER.info("full coordinate CSV already exists: %s", output_path)
        return
    coordinates = np.load(coordinate_path, mmap_mode="r")
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="", compresslevel=6) as handle:
        writer = csv.writer(handle)
        writer.writerow(["full_index", "chembl_id", "canonical_smiles", "umap1", "umap2"])
        for index, (record, xy) in enumerate(zip(records, coordinates)):
            writer.writerow(
                [index, record["chembl_id"], record["canonical_smiles"], float(xy[0]), float(xy[1])]
            )
    temporary.replace(output_path)
    LOGGER.info("saved full ChEMBL coordinate CSV: %s", output_path)


def _embedding_bounds(chembl_xy: np.ndarray, generated_xy: np.ndarray) -> tuple[float, float, float, float]:
    """Return robust shared plot bounds while retaining every generated point."""
    low = np.percentile(chembl_xy, 0.05, axis=0)
    high = np.percentile(chembl_xy, 99.95, axis=0)
    low = np.minimum(low, generated_xy.min(axis=0))
    high = np.maximum(high, generated_xy.max(axis=0))
    padding = np.maximum((high - low) * 0.03, 0.1)
    return float(low[0] - padding[0]), float(high[0] + padding[0]), float(low[1] - padding[1]), float(high[1] + padding[1])


def _density_grid(
    chembl_xy: np.ndarray, bounds: tuple[float, float, float, float], bins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a reusable 2D ChEMBL count grid."""
    xmin, xmax, ymin, ymax = bounds
    counts, x_edges, y_edges = np.histogram2d(
        chembl_xy[:, 0], chembl_xy[:, 1], bins=bins,
        range=[[xmin, xmax], [ymin, ymax]],
    )
    return counts.T, x_edges, y_edges


def _points_in_hdr(
    points: np.ndarray,
    density: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    mass: float,
) -> np.ndarray:
    """Classify points against a discrete highest-density ChEMBL mass region."""
    flat = density.ravel()
    ordered = np.sort(flat)[::-1]
    cumulative = np.cumsum(ordered) / ordered.sum()
    threshold = ordered[min(int(np.searchsorted(cumulative, mass)), len(ordered) - 1)]
    x_bin = np.searchsorted(x_edges, points[:, 0], side="right") - 1
    y_bin = np.searchsorted(y_edges, points[:, 1], side="right") - 1
    inside_bounds = (
        (x_bin >= 0) & (x_bin < density.shape[1])
        & (y_bin >= 0) & (y_bin < density.shape[0])
    )
    inside = np.zeros(len(points), dtype=bool)
    inside[inside_bounds] = density[y_bin[inside_bounds], x_bin[inside_bounds]] >= threshold
    return inside


def _summarize(
    config: dict[str, Any], full_payload: dict[str, Any], landmark_indices: np.ndarray
) -> None:
    """Calculate counts, 2D neighborhood distances, and ChEMBL HDR overlap."""
    output_dir = config["output_dir"]
    chembl_xy = np.load(output_dir / "chembl_coordinates.npy", mmap_mode="r")
    generated = pd.read_csv(output_dir / "generated_coordinates.csv.gz")
    generated_xy = generated[["umap1", "umap2"]].to_numpy(dtype=np.float32)
    bounds = _embedding_bounds(chembl_xy, generated_xy)
    density, x_edges, y_edges = _density_grid(
        chembl_xy, bounds, int(config["density_bins"])
    )
    np.savez_compressed(
        output_dir / "chembl_density_grid.npz",
        density=density, x_edges=x_edges, y_edges=y_edges, bounds=np.asarray(bounds),
    )

    attempts = int(config["attempts_per_model"])
    rows = []
    chembl_smiles = [str(record["canonical_smiles"]) for record in full_payload["records"]]
    chembl_centroid = np.mean(chembl_xy, axis=0)
    rows.append(
        {
            "group": "ChEMBL36", "fixed_attempts": np.nan, "n_embedded": len(chembl_xy),
            "embedded_per_attempt": np.nan, "unique_canonical_smiles": len(set(chembl_smiles)),
            "unique_fraction": len(set(chembl_smiles)) / len(chembl_smiles),
            "centroid_umap1": chembl_centroid[0], "centroid_umap2": chembl_centroid[1],
        }
    )
    for model in MODEL_ORDER:
        group = generated[generated["model"] == model]
        centroid = group[["umap1", "umap2"]].mean().to_numpy()
        rows.append(
            {
                "group": model, "fixed_attempts": attempts, "n_embedded": len(group),
                "embedded_per_attempt": len(group) / attempts,
                "unique_canonical_smiles": group["canonical_smiles"].nunique(),
                "unique_fraction": group["canonical_smiles"].nunique() / len(group),
                "centroid_umap1": centroid[0], "centroid_umap2": centroid[1],
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "population_counts.csv", index=False)

    for mass in config["hdr_masses"]:
        generated[f"inside_hdr_{int(float(mass) * 100)}"] = _points_in_hdr(
            generated_xy, density, x_edges, y_edges, float(mass)
        )
    neighbors = NearestNeighbors(n_neighbors=1, algorithm="kd_tree", n_jobs=int(config["workers"]))
    neighbors.fit(chembl_xy)
    generated["nearest_chembl_umap_distance"] = neighbors.kneighbors(
        generated_xy, return_distance=True
    )[0][:, 0]
    generated.to_csv(output_dir / "generated_embedding_metrics.csv.gz", index=False, compression="gzip")

    coverage_columns = [column for column in generated if column.startswith("inside_hdr_")]
    coverage_by_repeat = (
        generated.groupby(["model", "repeat"])[coverage_columns].mean().reset_index()
    )
    coverage_by_repeat.to_csv(output_dir / "coverage_by_repeat.csv", index=False)
    coverage_summary = coverage_by_repeat.groupby("model")[coverage_columns].agg(["mean", "std"])
    coverage_summary.columns = [f"{column}_{stat}" for column, stat in coverage_summary.columns]
    coverage_summary.reindex(MODEL_ORDER).reset_index().to_csv(
        output_dir / "coverage_summary.csv", index=False
    )

    distance_by_repeat = (
        generated.groupby(["model", "repeat"])["nearest_chembl_umap_distance"]
        .agg(n="size", mean="mean", median="median", p95=lambda values: values.quantile(0.95))
        .reset_index()
    )
    distance_by_repeat.to_csv(output_dir / "distance_to_chembl_by_repeat.csv", index=False)
    distance_summary = distance_by_repeat.groupby("model")[["mean", "median", "p95"]].agg(["mean", "std"])
    distance_summary.columns = [f"{metric}_{stat}" for metric, stat in distance_summary.columns]
    distance_summary.reindex(MODEL_ORDER).reset_index().to_csv(
        output_dir / "distance_to_chembl_summary.csv", index=False
    )

    landmark_xy = np.asarray(chembl_xy[landmark_indices])
    self_neighbors = NearestNeighbors(n_neighbors=2, algorithm="kd_tree", n_jobs=int(config["workers"]))
    self_neighbors.fit(chembl_xy)
    self_distance = self_neighbors.kneighbors(landmark_xy, return_distance=True)[0][:, 1]
    pd.DataFrame(
        {
            "reference": ["ChEMBL36 fixed 100k landmarks"],
            "n": [len(self_distance)],
            "median_second_neighbor_umap_distance": [np.median(self_distance)],
            "p95_second_neighbor_umap_distance": [np.quantile(self_distance, 0.95)],
        }
    ).to_csv(output_dir / "chembl_self_distance_reference.csv", index=False)


def _density_image(axis: plt.Axes, grid: dict[str, np.ndarray]) -> None:
    """Draw the rasterized full-ChEMBL density background."""
    density = grid["density"]
    bounds = grid["bounds"]
    positive = density[density > 0]
    axis.imshow(
        density,
        origin="lower",
        extent=bounds,
        cmap="Greys",
        norm=mcolors.LogNorm(vmin=max(1.0, float(np.quantile(positive, 0.05))), vmax=float(density.max())),
        interpolation="nearest",
        aspect="auto",
        alpha=0.88,
        rasterized=True,
    )


def _save_figure(fig: plt.Figure, stem: Path) -> None:
    """Export one figure to preview, vector, and print formats."""
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(
        stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def _plot(config: dict[str, Any], style: dict[str, Any]) -> None:
    """Render the overview, facets, and quantitative overlap summary."""
    output_dir = config["output_dir"]
    _configure_plotting(style)
    generated = pd.read_csv(output_dir / "generated_coordinates.csv.gz")
    with np.load(output_dir / "chembl_density_grid.npz") as loaded:
        grid = {key: loaded[key] for key in loaded.files}

    fig, axis = plt.subplots(figsize=tuple(style["overview_figsize"]))
    _density_image(axis, grid)
    rng = np.random.RandomState(int(config["umap"]["random_state"]))
    for model in MODEL_ORDER:
        group = generated[generated["model"] == model]
        group = group.iloc[rng.permutation(len(group))]
        axis.scatter(
            group["umap1"], group["umap2"], s=style["point_size"],
            color=MODEL_COLORS[model], alpha=style["point_alpha"], linewidths=0,
            rasterized=True, label=f"{model} (n={len(group):,})",
        )
    axis.set_title("Five-model chemical-space coverage against full ChEMBL36")
    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2")
    axis.legend(loc="upper left", ncol=2, markerscale=2.0)
    axis.set_xlim(grid["bounds"][:2])
    axis.set_ylim(grid["bounds"][2:])
    fig.text(
        0.5, 0.012,
        "Full ChEMBL36 density (n=1,550,090); five models use three fixed 1,000-attempt repeats. "
        "Only sanitize-valid generated molecules are embedded.",
        ha="center", fontsize=style["caption_size"],
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    _save_figure(fig, output_dir / "five_model_chemblfull_umap_overview_v001")

    fig, axes = plt.subplots(2, 3, figsize=tuple(style["facet_figsize"]), sharex=True, sharey=True)
    for axis, panel in zip(axes.ravel(), [*MODEL_ORDER, "ChEMBL36"]):
        _density_image(axis, grid)
        if panel != "ChEMBL36":
            group = generated[generated["model"] == panel]
            axis.scatter(
                group["umap1"], group["umap2"], s=style["facet_point_size"],
                color=MODEL_COLORS[panel], alpha=style["facet_point_alpha"],
                linewidths=0, rasterized=True,
            )
            axis.set_title(f"{panel} | n={len(group):,}")
        else:
            axis.set_title("ChEMBL36 | n=1,550,090")
        axis.set_xlabel("UMAP 1")
        axis.set_ylabel("UMAP 2")
        axis.set_xlim(grid["bounds"][:2])
        axis.set_ylim(grid["bounds"][2:])
    fig.suptitle("Model-specific coverage in one shared full-ChEMBL36 map")
    fig.text(
        0.5, 0.008,
        "UMAP is a descriptive neighborhood projection; 2D distance is not activity or molecular quality.",
        ha="center", fontsize=style["caption_size"],
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.96))
    _save_figure(fig, output_dir / "five_model_chemblfull_umap_facets_v001")

    coverage = pd.read_csv(output_dir / "coverage_summary.csv").set_index("model").reindex(MODEL_ORDER)
    distance = pd.read_csv(output_dir / "distance_to_chembl_summary.csv").set_index("model").reindex(MODEL_ORDER)
    counts = pd.read_csv(output_dir / "population_counts.csv").set_index("group").reindex(MODEL_ORDER)
    x = np.arange(len(MODEL_ORDER))
    colors = [MODEL_COLORS[model] for model in MODEL_ORDER]
    fig, axes = plt.subplots(1, 3, figsize=tuple(style["summary_figsize"]))
    axes[0].bar(x, counts["embedded_per_attempt"] * 100, color=colors)
    axes[0].set_ylabel("Embedded / fixed attempts (%)")
    axes[0].set_ylim(0, 105)
    axes[0].set_title("Valid embedding yield")
    axes[1].bar(
        x, coverage["inside_hdr_95_mean"] * 100,
        yerr=coverage["inside_hdr_95_std"] * 100, color=colors, capsize=3,
    )
    axes[1].set_ylabel("Inside ChEMBL 95% HDR (%)")
    axes[1].set_ylim(0, 105)
    axes[1].set_title("Full-ChEMBL overlap")
    axes[2].bar(
        x, distance["median_mean"], yerr=distance["median_std"],
        color=colors, capsize=3,
    )
    axes[2].set_ylabel("Nearest ChEMBL distance (2D UMAP)")
    axes[2].set_title("Local neighborhood distance")
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(MODEL_ORDER, rotation=28, ha="right")
    fig.text(
        0.5, 0.005, "Bars in panels B-C are repeat means; error bars are sample SD (n=3).",
        ha="center", fontsize=style["caption_size"],
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    _save_figure(fig, output_dir / "five_model_chemblfull_umap_summary_v001")

    history_path = output_dir / "style_history.csv"
    row = {
        "version": style["version"],
        "overview_figsize": "x".join(map(str, style["overview_figsize"])),
        "font_size": style["font_size"],
        "title_size": style["title_size"],
        "point_size": style["point_size"],
        "point_alpha": style["point_alpha"],
        "density_bins": config["density_bins"],
        "change_reason": style["change_reason"],
        "review_status": "rendered_pending_visual_qa",
    }
    existing = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
    if existing.empty or style["version"] not in existing["version"].astype(str).tolist():
        pd.concat([existing, pd.DataFrame([row])], ignore_index=True).to_csv(history_path, index=False)


def _write_manifest(
    config_path: Path,
    config: dict[str, Any],
    source_hashes: dict[str, str],
    full_count: int,
    landmark_count: int,
) -> None:
    """Write exact methods, lineage, versions, and interpretation boundaries."""
    output_dir = config["output_dir"]
    generated = pd.read_csv(output_dir / "generated_coordinates.csv.gz")
    effective_chunk_size = int(np.ceil(full_count / int(config["workers"])))
    manifest = {
        "analysis": "five formal generators versus full ChEMBL36 ECFP4 chemical space",
        "created_by": str(Path(__file__).resolve()),
        "config": str(config_path.resolve()),
        "sources": {
            "chembl_full": str(config["chembl_full"]),
            "chembl_landmarks": str(config["chembl_landmarks"]),
            "generated_csv": str(config["generated_csv"]),
            "sha256": source_hashes,
        },
        "counts": {
            "chembl_full": full_count,
            "chembl_fit_landmarks": landmark_count,
            "generated_embedded_total": int(len(generated)),
            "fixed_attempts_per_model": int(config["attempts_per_model"]),
            "generated_by_model": {
                model: int((generated["model"] == model).sum()) for model in MODEL_ORDER
            },
        },
        "fingerprint": {"type": "ECFP4", "radius": 2, "n_bits": 2048},
        "projection": {
            "algorithm": "UMAP",
            "fit_scope": "fixed ChEMBL36 100k landmarks plus all sanitize-valid five-model molecules",
            "transform_scope": "remaining ChEMBL36 molecules transformed into the fixed map",
            **config["umap"],
            "fit_threads": 1,
            "transform_processes": int(config["workers"]),
            "threads_per_transform_process": 1,
            "configured_transform_chunk_size": int(config["transform_chunk_size"]),
            "effective_transform_chunk_size": effective_chunk_size,
            "uniform_transform_chunking": True,
            "resume_strategy": "fixed full-index chunks with deterministic per-chunk search state",
        },
        "software": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__, "rdkit": rdBase.rdkitVersion,
            "umap_learn": umap.__version__, "matplotlib": matplotlib.__version__,
            "joblib": joblib.__version__, "platform": platform.platform(),
        },
        "interpretation_boundary": (
            "UMAP and all coordinate-space distances are descriptive only. They do not define labels "
            "and must not be interpreted as target activity, docking performance, or absolute quality."
        ),
    }
    _atomic_json(output_dir / "manifest.json", manifest)


def main() -> None:
    """Run the resumable embedding, summary, and Python-only figure workflow."""
    args = parse_args()
    config = _load_yaml(args.config, resolve_analysis_paths=True)
    output_dir = config["output_dir"]
    _configure_logging(output_dir)
    LOGGER.info("command: %s", " ".join(sys.argv))
    LOGGER.info("workers requested: %d", int(config["workers"]))
    shutil.copy2(args.config, output_dir / "analysis_config.snapshot.yaml")
    style_path = args.config.resolve().parent / config["style_config"]
    style = _load_yaml(style_path)
    shutil.copy2(style_path, output_dir / "style_config.snapshot.yaml")

    required_outputs = [
        output_dir / "chembl_coordinates.npy",
        output_dir / "generated_coordinates.csv.gz",
    ]
    full_payload: dict[str, Any] | None = None
    landmark_payload: dict[str, Any] | None = None
    landmark_indices_path = output_dir / "landmark_full_indices.npy"

    if args.stage in {"all", "embed"}:
        # Finish the 90-process fingerprint pool before loading 1.55 million
        # RDKit bit vectors, avoiding unnecessary copy-on-write memory pressure.
        generated, generated_fingerprints = _load_generated(
            config["generated_csv"], 2, 2048, int(config["workers"])
        )
        full_payload = _load_fingerprint_package(config["chembl_full"], 1_550_090)
        landmark_payload = _load_fingerprint_package(config["chembl_landmarks"], 100_000)
        landmark_indices = _match_landmarks(full_payload, landmark_payload)
        np.save(landmark_indices_path, landmark_indices)
        full_matrix = _bitvectors_to_csr(full_payload["fingerprints"], 2048)
        landmark_matrix = full_matrix[landmark_indices]
        generated_matrix = _bitvectors_to_csr(generated_fingerprints, 2048)
        reducer = _fit_or_load_reducer(
            config, landmark_matrix, generated_matrix, generated
        )
        coordinate_path = _transform_full_chembl(
            config, reducer, full_matrix, landmark_indices
        )
        _write_chembl_coordinate_csv(
            full_payload["records"], coordinate_path,
            output_dir / "chembl_coordinates.csv.gz",
        )

    if args.stage in {"all", "summarize"}:
        if not all(path.exists() for path in required_outputs) or not landmark_indices_path.exists():
            raise FileNotFoundError("embedding outputs are incomplete; run --stage embed first")
        if full_payload is None:
            full_payload = _load_fingerprint_package(config["chembl_full"], 1_550_090)
        landmark_indices = np.load(landmark_indices_path)
        _summarize(config, full_payload, landmark_indices)

    if args.stage in {"all", "plot"}:
        _plot(config, style)

    if full_payload is None:
        full_count = 1_550_090
    else:
        full_count = len(full_payload["records"])
    source_hashes = {
        "chembl_full": _sha256(config["chembl_full"]),
        "chembl_landmarks": _sha256(config["chembl_landmarks"]),
        "generated_csv": _sha256(config["generated_csv"]),
    }
    _write_manifest(
        args.config, config, source_hashes, full_count,
        int(len(np.load(landmark_indices_path))),
    )
    _atomic_json(output_dir / "completion.json", {"status": "complete", "stage": args.stage})
    LOGGER.info("completed successfully")


if __name__ == "__main__":
    main()
