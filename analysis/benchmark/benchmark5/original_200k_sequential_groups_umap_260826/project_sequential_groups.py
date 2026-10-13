#!/usr/bin/env python3
"""Split the first 150k attempts of the 200k Original run and project them.

The source run stores 10 attempts per rank-local shard.  The global order used
here is shard-major, then rank-major, then molecule-major:

    global_attempt = shard_index * 80 + rank * 10 + mol_index

Only the first 150,000 fixed attempts are selected, giving 50 consecutive
groups of 3,000 attempts and three consecutive 1,000-attempt subgroups per
group.  Invalid attempts remain in the attempt manifest and denominator, but
are not fingerprinted or embedded.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import logging
import multiprocessing as mp
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/diffb110_original_200k_umap_mpl")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/diffb110_original_200k_umap_numba")
os.environ.setdefault("NUMBA_DISABLE_CACHING", "1")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits


RDLogger.DisableLog("rdApp.error")
LOGGER = logging.getLogger("original_200k_sequential_groups")

RADIUS = 2
N_BITS = 2048
RANKS = 8
ATTEMPTS_PER_RANK_SHARD = 10
ATTEMPTS_PER_SHARD = RANKS * ATTEMPTS_PER_RANK_SHARD
TOTAL_ATTEMPTS = 200_000
SELECTED_ATTEMPTS = 150_000
GROUP_SIZE = 3_000
SUBGROUP_SIZE = 1_000
GROUP_COUNT = SELECTED_ATTEMPTS // GROUP_SIZE
SUBGROUPS_PER_GROUP = GROUP_SIZE // SUBGROUP_SIZE

_TRANSFORM_REDUCER: Any | None = None
_TRANSFORM_MATRIX: sparse.csr_matrix | None = None
_TRANSFORM_BASE_SEED = 42


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--frozen-umap-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=28)
    parser.add_argument(
        "--stage", choices=["all", "export", "project", "plot"], default="all"
    )
    return parser.parse_args()


def configure_logging(output_dir: Path) -> None:
    """Configure terminal and durable-file logging."""
    output_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(stream)
    root.addHandler(file_handler)


def sha256_file(path: Path) -> str:
    """Return a SHA256 digest for a regular file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def expected_sample_id(rank: int, shard_index: int, mol_index: int) -> str:
    """Return the collector's stable sample identifier."""
    return f"r{rank:02d}_s{shard_index:04d}_m{mol_index:04d}"


def load_reconstruction_rows(
    csv_path: Path, rank: int, shard_index: int
) -> list[dict[str, Any]]:
    """Load and validate the ten fixed-attempt reconstruction rows."""
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != ATTEMPTS_PER_RANK_SHARD:
        raise ValueError(f"{csv_path}: expected 10 rows, found {len(rows)}")
    output: list[dict[str, Any]] = []
    for row in rows:
        mol_index = int(row["mol_index"])
        if (
            int(row["rank"]) != rank
            or int(row["shard_index"]) != shard_index
            or mol_index not in range(ATTEMPTS_PER_RANK_SHARD)
            or row["sample_id"] != expected_sample_id(rank, shard_index, mol_index)
        ):
            raise ValueError(f"unexpected reconstruction identity in {csv_path}: {row}")
        global_attempt = shard_index * ATTEMPTS_PER_SHARD + rank * ATTEMPTS_PER_RANK_SHARD + mol_index
        if global_attempt >= SELECTED_ATTEMPTS:
            raise ValueError(f"out-of-range selected attempt: {global_attempt}")
        group_index = global_attempt // GROUP_SIZE + 1
        subgroup_index = (global_attempt % GROUP_SIZE) // SUBGROUP_SIZE + 1
        output.append(
            {
                "global_attempt": global_attempt,
                "group_index": group_index,
                "group_id": f"group_{group_index:03d}",
                "subgroup_index": subgroup_index,
                "subgroup_id": f"group_{group_index:03d}_subgroup_{subgroup_index:03d}",
                "rank": rank,
                "shard_index": shard_index,
                "mol_index": mol_index,
                "sample_id": row["sample_id"],
                "rdkit_valid": int(row["rdkit_valid"]),
                "failure_type": row["failure_type"],
                "smiles": row["smiles"],
                "source_reconstruction_csv": str(csv_path),
            }
        )
    return sorted(output, key=lambda item: int(item["mol_index"]))


def read_shard_molecules(sdf_path: Path) -> dict[str, Any]:
    """Read source molecules without re-sanitizing collector-valid records.

    The collector's reconstruction CSV is the fixed validity authority.  Some
    records marked valid there cannot be sanitized again by the current RDKit
    build, but their source 3D graph is still needed for provenance-preserving
    SDF export.
    """
    molecules: dict[str, Any] = {}
    if sdf_path.stat().st_size == 0:
        return molecules
    supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False)
    for molecule in supplier:
        if molecule is None or not molecule.HasProp("sample_id"):
            raise ValueError(f"missing sample_id in {sdf_path}")
        sample_id = molecule.GetProp("sample_id")
        molecules[sample_id] = molecule
    return molecules


def write_sdf_molecule(writer: Any, molecule: Any, row: dict[str, Any]) -> None:
    """Add selection provenance to one molecule before writing it."""
    molecule.SetProp("selection_global_attempt", str(row["global_attempt"]))
    molecule.SetProp("selection_group", str(row["group_id"]))
    molecule.SetProp("selection_subgroup", str(row["subgroup_id"]))
    writer.write(molecule)


def export_sequential_groups(run_root: Path, output_dir: Path) -> pd.DataFrame:
    """Export 50 consecutive 3,000-attempt groups and 150 subgroups."""
    groups_dir = output_dir / "groups"
    groups_dir.mkdir(parents=True, exist_ok=True)
    group_writers: dict[int, Any] = {}
    subgroup_writers: dict[tuple[int, int], Any] = {}
    group_paths: dict[int, Path] = {}
    subgroup_paths: dict[tuple[int, int], Path] = {}
    for group_index in range(1, GROUP_COUNT + 1):
        group_dir = groups_dir / f"group_{group_index:03d}"
        group_dir.mkdir(parents=True, exist_ok=True)
        group_path = group_dir / f"group_{group_index:03d}_3000_attempts.sdf"
        group_paths[group_index] = group_path
        group_writers[group_index] = Chem.SDWriter(str(group_path))
        group_writers[group_index].SetKekulize(False)
        for subgroup_index in range(1, SUBGROUPS_PER_GROUP + 1):
            subgroup_path = group_dir / (
                f"group_{group_index:03d}_subgroup_{subgroup_index:03d}_1000_attempts.sdf"
            )
            subgroup_paths[(group_index, subgroup_index)] = subgroup_path
            subgroup_writers[(group_index, subgroup_index)] = Chem.SDWriter(str(subgroup_path))
            subgroup_writers[(group_index, subgroup_index)].SetKekulize(False)

    attempt_rows: list[dict[str, Any]] = []
    valid_rows: list[dict[str, Any]] = []
    source_sdf_count = 0
    csv_smiles_fallback_count = 0
    try:
        for shard_index in range(SELECTED_ATTEMPTS // ATTEMPTS_PER_SHARD):
            for rank in range(RANKS):
                shard_dir = run_root / "molecules" / f"rank_{rank:02d}" / f"shard_{shard_index:04d}"
                reconstruction_csv = shard_dir / "reconstruction.csv"
                sdf_path = shard_dir / f"diffsbdd_collect_r{rank:02d}_s{shard_index:04d}.sdf"
                rows = load_reconstruction_rows(reconstruction_csv, rank, shard_index)
                valid_expected = {row["sample_id"] for row in rows if row["rdkit_valid"] == 1}
                try:
                    molecules = read_shard_molecules(sdf_path)
                except (OSError, ValueError) as exc:
                    LOGGER.warning("cannot read source SDF; using CSV SMILES fallback: %s (%s)", sdf_path, exc)
                    molecules = {}
                missing_valid_ids = valid_expected - set(molecules)
                unexpected_ids = set(molecules) - valid_expected
                if unexpected_ids:
                    LOGGER.warning(
                        "ignoring %d unexpected SDF records in %s",
                        len(unexpected_ids),
                        sdf_path,
                    )
                for row in rows:
                    row = dict(row)
                    row["source_sdf"] = str(sdf_path)
                    group_index = int(row["group_index"])
                    subgroup_index = int(row["subgroup_index"])
                    if row["rdkit_valid"] == 1:
                        molecule = molecules.get(row["sample_id"])
                        sdf_mode = "source_sdf"
                        if molecule is None:
                            if row["sample_id"] not in missing_valid_ids:
                                raise ValueError(
                                    f"unexpected missing source molecule: {row['sample_id']}"
                                )
                            molecule = Chem.MolFromSmiles(str(row["smiles"]))
                            if molecule is None:
                                raise ValueError(
                                    f"collector-valid row has unparsable SMILES: "
                                    f"{row['sample_id']} {row['smiles']}"
                                )
                            sdf_mode = "csv_smiles_fallback"
                            csv_smiles_fallback_count += 1
                        else:
                            source_sdf_count += 1
                        write_sdf_molecule(group_writers[group_index], molecule, row)
                        write_sdf_molecule(
                            subgroup_writers[(group_index, subgroup_index)], molecule, row
                        )
                        valid_row = dict(row)
                        valid_row["sdf_source_mode"] = sdf_mode
                        smiles_molecule = Chem.MolFromSmiles(str(row["smiles"]))
                        if smiles_molecule is None:
                            raise ValueError(
                                f"collector-valid row has unparsable SMILES: "
                                f"{row['sample_id']} {row['smiles']}"
                            )
                        valid_row["canonical_smiles"] = Chem.MolToSmiles(
                            smiles_molecule, canonical=True
                        )
                        valid_rows.append(valid_row)
                    attempt_rows.append(row)
            if (shard_index + 1) % 100 == 0:
                LOGGER.info("exported shard blocks: %d/%d", shard_index + 1, SELECTED_ATTEMPTS // ATTEMPTS_PER_SHARD)
    finally:
        for writer in group_writers.values():
            writer.close()
        for writer in subgroup_writers.values():
            writer.close()

    attempts = pd.DataFrame(attempt_rows).sort_values("global_attempt").reset_index(drop=True)
    valid = pd.DataFrame(valid_rows).sort_values("global_attempt").reset_index(drop=True)
    if len(attempts) != SELECTED_ATTEMPTS or attempts["global_attempt"].tolist() != list(range(SELECTED_ATTEMPTS)):
        raise ValueError("selected attempt manifest is not contiguous 0..149999")
    if len(valid) != int(attempts["rdkit_valid"].sum()):
        raise ValueError("valid export count disagrees with attempt manifest")
    attempts.to_csv(output_dir / "attempt_manifest.csv.gz", index=False, compression="gzip")
    valid.to_csv(output_dir / "valid_molecules.csv.gz", index=False, compression="gzip")

    group_manifest = (
        attempts.groupby(["group_index", "group_id"], sort=True)
        .agg(
            start_global_attempt=("global_attempt", "min"),
            end_global_attempt=("global_attempt", "max"),
            fixed_attempts=("global_attempt", "size"),
            rdkit_valid=("rdkit_valid", "sum"),
        )
        .reset_index()
    )
    group_manifest["valid_rate"] = group_manifest["rdkit_valid"] / group_manifest["fixed_attempts"]
    group_manifest["sdf_path"] = group_manifest["group_index"].map(
        lambda index: str(group_paths[int(index)])
    )
    group_manifest.to_csv(output_dir / "group_manifest.csv", index=False)

    subgroup_manifest = (
        attempts.groupby(["group_index", "group_id", "subgroup_index", "subgroup_id"], sort=True)
        .agg(
            start_global_attempt=("global_attempt", "min"),
            end_global_attempt=("global_attempt", "max"),
            fixed_attempts=("global_attempt", "size"),
            rdkit_valid=("rdkit_valid", "sum"),
        )
        .reset_index()
    )
    subgroup_manifest["valid_rate"] = subgroup_manifest["rdkit_valid"] / subgroup_manifest["fixed_attempts"]
    subgroup_manifest["sdf_path"] = [
        str(subgroup_paths[(int(group), int(subgroup))])
        for group, subgroup in zip(subgroup_manifest["group_index"], subgroup_manifest["subgroup_index"])
    ]
    subgroup_manifest.to_csv(output_dir / "subgroup_manifest.csv", index=False)
    atomic_json(
        output_dir / "export_completion.json",
        {
            "status": "complete",
            "source_total_attempts": TOTAL_ATTEMPTS,
            "selected_attempts": SELECTED_ATTEMPTS,
            "selected_range": [0, SELECTED_ATTEMPTS - 1],
            "groups": GROUP_COUNT,
            "attempts_per_group": GROUP_SIZE,
            "subgroups_per_group": SUBGROUPS_PER_GROUP,
            "attempts_per_subgroup": SUBGROUP_SIZE,
            "rdkit_valid_selected": int(len(valid)),
            "source_sdf_molecules": source_sdf_count,
            "csv_smiles_fallback_molecules": csv_smiles_fallback_count,
            "order_definition": "shard_index-major, rank-major, mol_index-major",
            "global_attempt_formula": "shard_index*80 + rank*10 + mol_index",
        },
    )
    LOGGER.info("export complete: attempts=%d valid=%d", len(attempts), len(valid))
    return valid


def fingerprint_task(task: tuple[int, str]) -> tuple[int, Any | None]:
    """Calculate one ECFP4 bit vector."""
    index, smiles = task
    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return index, None
        return index, AllChem.GetMorganFingerprintAsBitVect(molecule, RADIUS, nBits=N_BITS)
    except Exception:
        return index, None


def fingerprints_to_csr(bit_vectors: list[Any]) -> sparse.csr_matrix:
    """Convert RDKit bit vectors to compact CSR format."""
    indptr = np.zeros(len(bit_vectors) + 1, dtype=np.int64)
    for index, bit_vector in enumerate(bit_vectors):
        indptr[index + 1] = indptr[index] + bit_vector.GetNumOnBits()
    indices = np.empty(int(indptr[-1]), dtype=np.int32)
    for index, bit_vector in enumerate(bit_vectors):
        start, stop = int(indptr[index]), int(indptr[index + 1])
        indices[start:stop] = np.fromiter(bit_vector.GetOnBits(), dtype=np.int32, count=stop - start)
    return sparse.csr_matrix(
        (np.ones(len(indices), dtype=np.float32), indices, indptr),
        shape=(len(bit_vectors), N_BITS),
    )


def calculate_fingerprints(smiles: list[str], workers: int) -> sparse.csr_matrix:
    """Calculate ECFP4 fingerprints with bounded multiprocessing."""
    tasks = list(enumerate(smiles))
    with mp.Pool(processes=workers) as pool:
        results = pool.map(fingerprint_task, tasks, chunksize=128)
    results = sorted(results, key=lambda item: item[0])
    if any(bit_vector is None for _, bit_vector in results):
        raise ValueError("at least one exported valid molecule has no ECFP4 fingerprint")
    return fingerprints_to_csr([bit_vector for _, bit_vector in results])


def load_frozen_reducer(path: Path) -> Any:
    """Load the old frozen reducer without modifying the serialized file.

    The formal map was saved with an older Numba/PyNNDescent serialization
    format.  Serialized compiled callables are discarded and the current
    PyNNDescent sparse query closure is rebuilt from the saved search graph and
    random-projection tree.  The UMAP embedding, raw fit matrix, graph, and all
    scientific parameters remain those from the frozen file.
    """
    from numba.core.dispatcher import Dispatcher
    from numba.np.ufunc.dufunc import DUFunc
    from pynndescent.pynndescent_ import NNDescent
    from pynndescent.rp_trees import renumbaify_tree
    from pynndescent import sparse as pynn_sparse

    original_dispatcher_rebuild = Dispatcher._rebuild.__func__
    original_dufunc_rebuild = DUFunc._rebuild.__func__
    original_nndescent_setstate = NNDescent.__setstate__

    def discard_dispatcher(*_: Any, **__: Any) -> None:
        return None

    def discard_dufunc(*_: Any, **__: Any) -> None:
        return None

    def load_nndescent_state(self: Any, state: dict[str, Any]) -> None:
        self.__dict__ = state
        self._set_distance_func()
        self._search_forest = tuple(renumbaify_tree(tree) for tree in state["_search_forest"])

    Dispatcher._rebuild = classmethod(discard_dispatcher)
    DUFunc._rebuild = classmethod(discard_dufunc)
    NNDescent.__setstate__ = load_nndescent_state
    try:
        reducer = joblib.load(path)
    finally:
        Dispatcher._rebuild = classmethod(original_dispatcher_rebuild)
        DUFunc._rebuild = classmethod(original_dufunc_rebuild)
        NNDescent.__setstate__ = original_nndescent_setstate

    search_index = reducer._knn_search_index
    if not getattr(reducer, "_sparse_data", False):
        raise ValueError("formal frozen reducer is not sparse, unexpected for ECFP4")
    search_index._distance_func = pynn_sparse.sparse_named_distances["cosine"]
    search_index._init_sparse_search_function()
    reducer.n_epochs = 90  # current UMAP divides this by three -> formal 30 transform epochs
    LOGGER.info(
        "loaded frozen reducer: raw_shape=%s metric=%s neighbors=%s min_dist=%s",
        reducer._raw_data.shape,
        reducer.metric,
        reducer.n_neighbors,
        reducer.min_dist,
    )
    return reducer


def reset_transform_rng(reducer: Any, seed: int) -> None:
    """Set an independent deterministic PyNNDescent search state."""
    rng = np.random.RandomState(seed)
    reducer._knn_search_index.search_rng_state = rng.randint(
        -(2**31), 2**31 - 1, 3
    ).astype(np.int64)


def initialize_transform_worker() -> None:
    """Limit each transform worker to one native thread."""
    numba.set_num_threads(1)
    threadpool_limits(limits=1)


def transform_chunk_task(task: tuple[int, int, int]) -> tuple[int, int, np.ndarray]:
    """Transform one fixed chunk with the inherited frozen reducer."""
    chunk_index, start, stop = task
    if _TRANSFORM_REDUCER is None or _TRANSFORM_MATRIX is None:
        raise RuntimeError("transform worker was not initialized")
    reset_transform_rng(_TRANSFORM_REDUCER, _TRANSFORM_BASE_SEED + chunk_index)
    coordinates = _TRANSFORM_REDUCER.transform(_TRANSFORM_MATRIX[start:stop]).astype(np.float32)
    return start, stop, coordinates


def project_valid_molecules(
    valid: pd.DataFrame, frozen_root: Path, output_dir: Path, workers: int
) -> pd.DataFrame:
    """Fingerprint and project all selected valid molecules."""
    matrix = calculate_fingerprints(valid["canonical_smiles"].astype(str).tolist(), workers)
    np.savez_compressed(
        output_dir / "selected_ecfp4_matrix.npz",
        data=matrix.data,
        indices=matrix.indices,
        indptr=matrix.indptr,
        shape=np.asarray(matrix.shape, dtype=np.int64),
    )
    reducer = load_frozen_reducer(frozen_root / "umap_model.joblib")
    global _TRANSFORM_REDUCER, _TRANSFORM_MATRIX, _TRANSFORM_BASE_SEED
    _TRANSFORM_REDUCER = reducer
    _TRANSFORM_MATRIX = matrix
    _TRANSFORM_BASE_SEED = 42
    coordinates = np.empty((len(valid), 2), dtype=np.float32)
    # The compatibility reducer contains a rebuilt PyNNDescent search closure.
    # Keep transform in one process because sharing that mutable search state
    # across forked workers can terminate without a Python traceback.
    chunk_size = 4_000
    chunks = [
        (chunk_index, start, min(start + chunk_size, len(valid)))
        for chunk_index, start in enumerate(range(0, len(valid), chunk_size))
    ]
    LOGGER.info(
        "transforming %d molecules in %d serial chunks; workers=%d reserved for preprocessing and statistics",
        len(valid),
        len(chunks),
        workers,
    )
    for chunk_index, start, stop in chunks:
        reset_transform_rng(reducer, _TRANSFORM_BASE_SEED + chunk_index)
        coordinates[start:stop] = reducer.transform(matrix[start:stop]).astype(np.float32)
        LOGGER.info("transform complete: [%d,%d)", start, stop)
    if not np.isfinite(coordinates).all():
        raise ValueError("non-finite selected UMAP coordinates")
    projected = valid.copy()
    projected.insert(0, "selected_valid_row", np.arange(len(projected), dtype=np.int64))
    projected["umap1"] = coordinates[:, 0]
    projected["umap2"] = coordinates[:, 1]
    projected.to_csv(output_dir / "valid_molecules_embedding.csv.gz", index=False, compression="gzip")

    chembl_xy = np.load(frozen_root / "chembl_coordinates.npy", mmap_mode="r")
    neighbor = NearestNeighbors(n_neighbors=1, algorithm="kd_tree", n_jobs=workers)
    neighbor.fit(chembl_xy)
    projected["nearest_chembl_umap_distance"] = neighbor.kneighbors(
        projected[["umap1", "umap2"]].to_numpy(dtype=np.float32), return_distance=True
    )[0][:, 0]
    with np.load(frozen_root / "chembl_density_grid.npz") as loaded:
        density = loaded["density"]
        x_edges = loaded["x_edges"]
        y_edges = loaded["y_edges"]
    x_bin = np.searchsorted(x_edges, projected["umap1"].to_numpy(), side="right") - 1
    y_bin = np.searchsorted(y_edges, projected["umap2"].to_numpy(), side="right") - 1
    in_bounds = (
        (x_bin >= 0) & (x_bin < density.shape[1]) & (y_bin >= 0) & (y_bin < density.shape[0])
    )
    flat = np.sort(density.ravel())[::-1]
    cumulative = np.cumsum(flat) / flat.sum()
    for mass in (0.50, 0.90, 0.95, 0.99):
        threshold = flat[min(int(np.searchsorted(cumulative, mass)), len(flat) - 1)]
        inside = np.zeros(len(projected), dtype=bool)
        inside[in_bounds] = density[y_bin[in_bounds], x_bin[in_bounds]] >= threshold
        projected[f"inside_hdr_{int(mass * 100)}"] = inside
    projected.to_csv(output_dir / "valid_molecules_embedding.csv.gz", index=False, compression="gzip")

    group_stats = summarize_table(projected, "group_index", GROUP_SIZE)
    subgroup_stats = summarize_table(projected, ["group_index", "subgroup_index"], SUBGROUP_SIZE)
    group_stats.to_csv(output_dir / "group_stats.csv", index=False)
    subgroup_stats.to_csv(output_dir / "subgroup_stats.csv", index=False)
    atomic_json(
        output_dir / "projection_completion.json",
        {
            "status": "complete",
            "selected_attempts": SELECTED_ATTEMPTS,
            "embedded_valid_molecules": int(len(projected)),
            "workers": workers,
            "fingerprint": {"type": "ECFP4", "radius": RADIUS, "n_bits": N_BITS},
            "frozen_umap_model": str(frozen_root / "umap_model.joblib"),
            "umap_runtime": {"umap_learn": "0.5.12", "numba": numba.__version__, "transform_epochs": 30},
            "transform_compatibility": "saved UMAP embedding/raw data/search graph retained; current sparse query closure rebuilt in memory",
            "interpretation": "UMAP coordinates and distances are descriptive chemical-space measures, not activity or docking evidence.",
        },
    )
    LOGGER.info("projection complete: valid=%d", len(projected))
    return projected


def summarize_table(frame: pd.DataFrame, keys: str | list[str], fixed_attempts: int) -> pd.DataFrame:
    """Summarize valid-only projection metrics with fixed-attempt denominators."""
    if isinstance(keys, str):
        keys = [keys]
    summary = (
        frame.groupby(keys, sort=True)
        .agg(
            embedded_valid=("selected_valid_row", "size"),
            mean_nearest_chembl_umap_distance=("nearest_chembl_umap_distance", "mean"),
            median_nearest_chembl_umap_distance=("nearest_chembl_umap_distance", "median"),
            p95_nearest_chembl_umap_distance=("nearest_chembl_umap_distance", lambda values: values.quantile(0.95)),
            umap1_mean=("umap1", "mean"),
            umap2_mean=("umap2", "mean"),
            umap1_std=("umap1", "std"),
            umap2_std=("umap2", "std"),
            hdr50=("inside_hdr_50", "mean"),
            hdr90=("inside_hdr_90", "mean"),
            hdr95=("inside_hdr_95", "mean"),
            hdr99=("inside_hdr_99", "mean"),
        )
        .reset_index()
    )
    summary["fixed_attempts"] = fixed_attempts
    summary["valid_rate"] = summary["embedded_valid"] / fixed_attempts
    return summary


def density_image(axis: plt.Axes, grid: dict[str, np.ndarray]) -> None:
    """Draw the frozen full-ChEMBL density background."""
    density = grid["density"]
    positive = density[density > 0]
    axis.imshow(
        density,
        origin="lower",
        extent=grid["bounds"],
        cmap="Greys",
        norm=mcolors.LogNorm(vmin=max(1.0, float(np.quantile(positive, 0.05))), vmax=float(density.max())),
        interpolation="nearest",
        aspect="auto",
        alpha=0.88,
        rasterized=True,
    )


def common_bounds(frame: pd.DataFrame, grid: dict[str, np.ndarray]) -> tuple[float, float, float, float]:
    """Return shared bounds covering the frozen grid and all selected points."""
    base = np.asarray(grid["bounds"], dtype=float)
    points = frame[["umap1", "umap2"]].to_numpy(dtype=float)
    low = np.minimum(base[[0, 2]], points.min(axis=0))
    high = np.maximum(base[[1, 3]], points.max(axis=0))
    padding = 0.02 * np.maximum(high - low, 1.0)
    return float(low[0] - padding[0]), float(high[0] + padding[0]), float(low[1] - padding[1]), float(high[1] + padding[1])


def save_figure(fig: plt.Figure, stem: Path) -> None:
    """Save a Python-generated figure in quick-view and vector formats."""
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_outputs(frame: pd.DataFrame, frozen_root: Path, output_dir: Path) -> None:
    """Create one-group-per-panel UMAP facets and quantitative trends."""
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
            "font.size": 9,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    with np.load(frozen_root / "chembl_density_grid.npz") as loaded:
        grid = {key: loaded[key] for key in loaded.files}
    bounds = common_bounds(frame, grid)
    atomic_json(output_dir / "plot_bounds.json", {"bounds": list(bounds), "frozen_grid_bounds": grid["bounds"].tolist()})
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, GROUP_COUNT))

    fig, axes = plt.subplots(5, 10, figsize=(22, 11), sharex=True, sharey=True)
    for group_index, axis in enumerate(axes.ravel(), start=1):
        density_image(axis, grid)
        group = frame[frame["group_index"] == group_index]
        axis.scatter(group["umap1"], group["umap2"], s=4.0, color=colors[group_index - 1], alpha=0.68, linewidths=0, rasterized=True)
        axis.set_title(f"G{group_index:02d} | n={len(group):,}", fontsize=9)
        axis.set_xlim(bounds[:2])
        axis.set_ylim(bounds[2:])
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle("Sequential 3,000-attempt Original groups in the frozen ChEMBL36 UMAP", fontsize=17, y=0.995)
    fig.text(0.5, 0.006, "Each facet shows one consecutive group only; grey background is full ChEMBL36 density. Invalid attempts are excluded from points but retained in the 3,000-attempt denominator.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.025, 1, 0.97))
    save_figure(fig, output_dir / "sequential_group_umap_facets")

    fig, axis = plt.subplots(figsize=(12, 8))
    density_image(axis, grid)
    group_stats = pd.read_csv(output_dir / "group_stats.csv")
    axis.plot(group_stats["umap1_mean"], group_stats["umap2_mean"], color="#333333", linewidth=1.0, alpha=0.75, zorder=3)
    axis.scatter(group_stats["umap1_mean"], group_stats["umap2_mean"], c=np.arange(GROUP_COUNT), cmap="viridis", s=55, edgecolor="white", linewidth=0.5, zorder=4)
    for _, row in group_stats.iterrows():
        axis.text(row["umap1_mean"], row["umap2_mean"], f"G{int(row['group_index']):02d}", fontsize=7, ha="center", va="bottom")
    axis.set_title("Order of the 50 sequential groups: centroid trajectory", fontsize=17)
    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2")
    axis.set_xlim(bounds[:2])
    axis.set_ylim(bounds[2:])
    fig.text(0.5, 0.012, "Line connects group centroids in source order; it is a sampling-order diagnostic, not a trajectory in chemical space.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    save_figure(fig, output_dir / "sequential_group_centroid_trajectory")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    group_stats = pd.read_csv(output_dir / "group_stats.csv")
    x = group_stats["group_index"]
    axes[0].plot(x, group_stats["valid_rate"] * 100, color="#3B82C4", marker="o", markersize=3)
    axes[0].set_ylabel("RDKit-valid / fixed attempts (%)")
    axes[0].set_title("Validity by group")
    axes[1].plot(x, group_stats["mean_nearest_chembl_umap_distance"], color="#D95F59", marker="o", markersize=3)
    axes[1].set_ylabel("Mean nearest-ChEMBL UMAP distance")
    axes[1].set_title("Local distance by group")
    axes[2].plot(x, group_stats["hdr95"] * 100, color="#3D9B72", marker="o", markersize=3)
    axes[2].set_ylabel("Valid points inside ChEMBL 95% HDR (%)")
    axes[2].set_title("ChEMBL overlap by group")
    for axis in axes:
        axis.set_xlabel("Sequential group")
        axis.set_xticks([1, 10, 20, 30, 40, 50])
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    fig.suptitle("Distribution stability across 50 consecutive Original groups", fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, output_dir / "sequential_group_metric_trends")

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharex=True, sharey=True)
    subgroup_colors = ["#3B82C4", "#D95F59", "#3D9B72"]
    for subgroup_index, axis in enumerate(axes, start=1):
        density_image(axis, grid)
        subset = frame[frame["subgroup_index"] == subgroup_index]
        axis.scatter(subset["umap1"], subset["umap2"], s=4.0, color=subgroup_colors[subgroup_index - 1], alpha=0.35, linewidths=0, rasterized=True)
        axis.set_title(f"Subgroup {subgroup_index} | all 50 groups | n={len(subset):,}")
        axis.set_xlabel("UMAP 1")
        axis.set_ylabel("UMAP 2")
        axis.set_xlim(bounds[:2])
        axis.set_ylim(bounds[2:])
    fig.suptitle("The three consecutive 1,000-attempt positions within each 3,000-attempt group", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, output_dir / "sequential_subgroup_umap_overlay")


def write_manifest(args: argparse.Namespace, output_dir: Path, valid: pd.DataFrame | None) -> None:
    """Write the durable analysis manifest."""
    payload: dict[str, Any] = {
        "status": "complete",
        "analysis": "sequential 200k Original groups projected to the frozen full-ChEMBL36 UMAP",
        "source_run": str(args.run_root),
        "source_completion_report": str(args.run_root / "completion_report.json"),
        "frozen_umap_root": str(args.frozen_umap_root),
        "output_dir": str(output_dir),
        "selection": {
            "source_total_attempts": TOTAL_ATTEMPTS,
            "selected_range": [0, SELECTED_ATTEMPTS - 1],
            "selected_attempts": SELECTED_ATTEMPTS,
            "group_count": GROUP_COUNT,
            "group_size": GROUP_SIZE,
            "subgroups_per_group": SUBGROUPS_PER_GROUP,
            "subgroup_size": SUBGROUP_SIZE,
            "order": "shard_index-major, rank-major, mol_index-major",
            "global_attempt_formula": "shard_index*80 + rank*10 + mol_index",
        },
        "fingerprint": {"type": "ECFP4", "radius": RADIUS, "n_bits": N_BITS, "metric": "cosine"},
        "workers": args.workers,
        "software": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__, "umap_runtime": "0.5.12", "numba": numba.__version__, "rdkit": Chem.rdBase.rdkitVersion, "platform": platform.platform()},
        "interpretation_boundary": "UMAP and coordinate-space distances are descriptive only; they do not define activity, docking performance, or molecular quality.",
    }
    if valid is not None:
        payload["embedded_valid_molecules"] = int(len(valid))
    atomic_json(output_dir / "manifest.json", payload)


def main() -> None:
    """Run export, frozen projection, summaries, and Python-only plots."""
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if SELECTED_ATTEMPTS % GROUP_SIZE or GROUP_SIZE % SUBGROUP_SIZE:
        raise ValueError("selection sizes must divide exactly")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(args.output_dir)
    LOGGER.info("command: %s", " ".join(sys.argv))
    LOGGER.info("workers=%d run_root=%s frozen_umap=%s", args.workers, args.run_root, args.frozen_umap_root)

    valid: pd.DataFrame | None = None
    if args.stage in {"all", "export"}:
        valid = export_sequential_groups(args.run_root, args.output_dir)
    elif (args.output_dir / "valid_molecules.csv.gz").exists():
        valid = pd.read_csv(args.output_dir / "valid_molecules.csv.gz")

    if args.stage in {"all", "project"}:
        if valid is None:
            valid = pd.read_csv(args.output_dir / "valid_molecules.csv.gz")
        valid = project_valid_molecules(valid, args.frozen_umap_root, args.output_dir, args.workers)

    if args.stage in {"all", "plot"}:
        embedding_path = args.output_dir / "valid_molecules_embedding.csv.gz"
        if not embedding_path.exists():
            raise FileNotFoundError(embedding_path)
        valid = pd.read_csv(embedding_path)
        plot_outputs(valid, args.frozen_umap_root, args.output_dir)

    write_manifest(args, args.output_dir, valid)
    atomic_json(args.output_dir / "completion.json", {"status": "complete", "stage": args.stage})
    LOGGER.info("analysis completed: %s", args.output_dir)


if __name__ == "__main__":
    main()
