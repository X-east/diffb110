#!/usr/bin/env python3
"""Joint ECFP4-UMAP of five RE-CmeB generators and 100,000 ChEMBL molecules."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import pickle
import platform
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/diffb110_matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/diffb110_numba_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import AllChem
from scipy import sparse
import sklearn
import umap


RDLogger.DisableLog("rdApp.error")
LOGGER = logging.getLogger("five_model_chembl100k_umap")

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CHEMBL = ROOT / "data/chembl/chembl_36_100k_ecfp4_fingerprints.pkl.gz"
DEFAULT_GENERATED = (
    ROOT
    / "analysis/benchmark/benchmark5"
    / "8gjl_c1101_benchmark_20260706_1000steps_3rep_formal"
    / "raw_quality_vs_chembl_20260728/per_molecule_raw_quality.csv.gz"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent

MODEL_ORDER = ["DiffGui", "DiffSBDD", "PMDM", "GCDM-SBDD", "TargetDiff"]
MODEL_COLORS = {
    "DiffGui": "#3B82C4",
    "DiffSBDD": "#D95F59",
    "PMDM": "#D9A441",
    "GCDM-SBDD": "#8E78B7",
    "TargetDiff": "#3D9B72",
}
CHEMBL_COLOR = "#A8ADB3"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chembl-fingerprints", type=Path, default=DEFAULT_CHEMBL)
    parser.add_argument("--generated-csv", type=Path, default=DEFAULT_GENERATED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--n-neighbors", type=int, default=20)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _configure_logging(output_dir: Path) -> None:
    """Write chronological progress to the terminal and a durable local log."""
    output_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "run.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(stream)
    root_logger.addHandler(file_handler)


def _configure_plotting() -> None:
    """Apply publication-grade, slide-readable Python plotting defaults."""
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 13,
            "axes.titlesize": 17,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
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
    """Return the SHA-256 digest of one source file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_chembl(path: Path) -> tuple[list[Any], pd.DataFrame, dict[str, Any]]:
    """Load and validate the traced ChEMBL36 100k ECFP4 package."""
    LOGGER.info("loading ChEMBL fingerprints: %s", path)
    with gzip.open(path, "rb") as handle:
        payload = pickle.load(handle)
    required = {
        "fingerprints",
        "records",
        "radius",
        "n_bits",
        "fingerprint_type",
        "chembl_version",
        "sampled_from_total",
        "sampling_seed",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"ChEMBL fingerprint package is missing keys: {sorted(missing)}")
    fingerprints = payload["fingerprints"]
    records = pd.DataFrame(payload["records"])
    if len(fingerprints) != 100_000 or len(records) != 100_000:
        raise ValueError(
            f"expected 100,000 ChEMBL fingerprints and records, found "
            f"{len(fingerprints):,} and {len(records):,}"
        )
    if payload["radius"] != 2 or payload["n_bits"] != 2048:
        raise ValueError(
            f"expected ECFP4 radius=2 and n_bits=2048, found "
            f"radius={payload['radius']} and n_bits={payload['n_bits']}"
        )
    if records[["chembl_id", "canonical_smiles"]].isna().any().any():
        raise ValueError("ChEMBL records contain missing molecule identities")
    LOGGER.info(
        "validated ChEMBL package: n=%d radius=%d bits=%d seed=%s sampled_from=%s",
        len(records), payload["radius"], payload["n_bits"],
        payload["sampling_seed"], payload["sampled_from_total"],
    )
    return fingerprints, records, payload


def _fingerprint_task(task: tuple[int, str, int, int]) -> tuple[int, Any | None]:
    """Calculate one Morgan bit vector while preserving input order."""
    index, smiles, radius, n_bits = task
    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return index, None
        return index, AllChem.GetMorganFingerprintAsBitVect(
            molecule, radius, nBits=n_bits
        )
    except Exception:
        return index, None


def _load_generated(
    path: Path, radius: int, n_bits: int, workers: int
) -> tuple[pd.DataFrame, list[Any]]:
    """Load all sanitize-valid formal benchmark molecules and calculate ECFP4."""
    LOGGER.info("loading generated molecule table: %s", path)
    usecols = [
        "row_id",
        "repeat",
        "model",
        "source_sdf",
        "source_index",
        "sanitize_success",
        "canonical_smiles",
    ]
    frame = pd.read_csv(path, usecols=usecols)
    frame = frame[
        frame["sanitize_success"].eq(1)
        & frame["model"].isin(MODEL_ORDER)
        & frame["canonical_smiles"].notna()
    ].copy()
    frame["input_order"] = np.arange(len(frame), dtype=np.int64)
    LOGGER.info("calculating generated ECFP4: n=%d workers=%d", len(frame), workers)
    tasks = [
        (index, smiles, radius, n_bits)
        for index, smiles in enumerate(frame["canonical_smiles"].tolist())
    ]
    with Pool(processes=workers) as pool:
        results = pool.map(_fingerprint_task, tasks, chunksize=128)
    valid = [(index, fingerprint) for index, fingerprint in results if fingerprint is not None]
    if len(valid) != len(frame):
        dropped = sorted(set(range(len(frame))).difference(index for index, _ in valid))
        LOGGER.warning("dropped %d generated molecules without ECFP4", len(dropped))
        frame = frame.iloc[[index for index, _ in valid]].copy()
    fingerprints = [fingerprint for _, fingerprint in valid]
    frame = frame.reset_index(drop=True)
    counts = frame.groupby("model").size().reindex(MODEL_ORDER, fill_value=0)
    LOGGER.info("generated ECFP4 counts:\n%s", counts.to_string())
    return frame, fingerprints


def _bitvectors_to_csr(bit_vectors: list[Any], n_bits: int) -> sparse.csr_matrix:
    """Convert RDKit bit vectors directly to a compact sparse CSR matrix."""
    row_count = len(bit_vectors)
    indptr = np.zeros(row_count + 1, dtype=np.int64)
    for index, bit_vector in enumerate(bit_vectors):
        indptr[index + 1] = indptr[index] + bit_vector.GetNumOnBits()
    indices = np.empty(int(indptr[-1]), dtype=np.int32)
    for index, bit_vector in enumerate(bit_vectors):
        start, stop = int(indptr[index]), int(indptr[index + 1])
        indices[start:stop] = np.fromiter(
            bit_vector.GetOnBits(), dtype=np.int32, count=stop - start
        )
    data = np.ones(len(indices), dtype=np.float32)
    matrix = sparse.csr_matrix((data, indices, indptr), shape=(row_count, n_bits))
    LOGGER.info(
        "built sparse matrix: shape=%s nnz=%d memory=%.1f MiB",
        matrix.shape,
        matrix.nnz,
        (matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes) / 1024**2,
    )
    return matrix


def _build_coordinate_table(
    chembl_records: pd.DataFrame,
    generated: pd.DataFrame,
    coordinates: np.ndarray,
) -> pd.DataFrame:
    """Attach every joint-UMAP coordinate to its source molecule identity."""
    chembl = pd.DataFrame(
        {
            "dataset": "ChEMBL36 100k",
            "model": "ChEMBL36",
            "repeat": pd.NA,
            "molecule_id": chembl_records["chembl_id"].astype(str),
            "canonical_smiles": chembl_records["canonical_smiles"].astype(str),
            "source_sdf": pd.NA,
            "source_index": pd.NA,
        }
    )
    generated_table = pd.DataFrame(
        {
            "dataset": "Generated",
            "model": generated["model"].astype(str),
            "repeat": generated["repeat"].astype(str),
            "molecule_id": generated["row_id"].astype(str),
            "canonical_smiles": generated["canonical_smiles"].astype(str),
            "source_sdf": generated["source_sdf"].astype(str),
            "source_index": generated["source_index"],
        }
    )
    table = pd.concat([chembl, generated_table], ignore_index=True)
    if len(table) != len(coordinates):
        raise ValueError(f"identity rows ({len(table)}) != coordinate rows ({len(coordinates)})")
    table["umap1"] = coordinates[:, 0]
    table["umap2"] = coordinates[:, 1]
    return table


def _summarize_groups(coordinates: pd.DataFrame) -> pd.DataFrame:
    """Summarize counts and coordinate spread without treating UMAP as a quality score."""
    rows: list[dict[str, Any]] = []
    for model in ["ChEMBL36", *MODEL_ORDER]:
        group = coordinates[coordinates["model"] == model]
        xy = group[["umap1", "umap2"]].to_numpy()
        centroid = xy.mean(axis=0)
        rows.append(
            {
                "group": model,
                "n": int(len(group)),
                "unique_canonical_smiles": int(group["canonical_smiles"].nunique()),
                "unique_fraction": float(group["canonical_smiles"].nunique() / len(group)),
                "centroid_umap1": float(centroid[0]),
                "centroid_umap2": float(centroid[1]),
                "within_group_rms_radius": float(
                    np.sqrt(np.mean(np.sum((xy - centroid) ** 2, axis=1)))
                ),
            }
        )
    return pd.DataFrame(rows)


def _save_figure(fig: plt.Figure, stem: Path) -> None:
    """Export one Python-rendered figure in preview, vector and print formats."""
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(
        stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight",
        facecolor="white", pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def _plot_overview(coordinates: pd.DataFrame, output_dir: Path, seed: int) -> None:
    """Plot all five generators over one common ChEMBL100k background."""
    chembl = coordinates[coordinates["model"] == "ChEMBL36"]
    generated = coordinates[coordinates["dataset"] == "Generated"]
    fig, axis = plt.subplots(figsize=(11.8, 7.0))
    axis.scatter(
        chembl["umap1"], chembl["umap2"], s=0.7, color=CHEMBL_COLOR,
        alpha=0.10, linewidths=0, rasterized=True,
        label=f"ChEMBL36 reference (n={len(chembl):,})",
    )
    rng = np.random.default_rng(seed)
    for model in MODEL_ORDER:
        group = generated[generated["model"] == model]
        order = rng.permutation(len(group))
        group = group.iloc[order]
        axis.scatter(
            group["umap1"], group["umap2"], s=8.0, color=MODEL_COLORS[model],
            alpha=0.48, linewidths=0, rasterized=True,
            label=f"{model} (n={len(group):,})",
        )
    axis.set_title("Five-model chemical-space coverage against ChEMBL36")
    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2")
    axis.legend(loc="upper left", markerscale=2.2, ncol=2)
    fig.text(
        0.5, 0.012,
        "One joint embedding: ECFP4 (radius 2, 2048 bits), cosine UMAP; "
        "generated points include all sanitize-valid molecules from three formal repeats",
        ha="center", fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    _save_figure(fig, output_dir / "five_model_chembl100k_umap_overview")


def _plot_facets(coordinates: pd.DataFrame, output_dir: Path) -> None:
    """Plot one model-vs-ChEMBL panel per generator plus a ChEMBL-only reference panel."""
    chembl = coordinates[coordinates["model"] == "ChEMBL36"]
    generated = coordinates[coordinates["dataset"] == "Generated"]
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 8.2), sharex=True, sharey=True)
    panels = [*MODEL_ORDER, "ChEMBL36"]
    for axis, panel in zip(axes.ravel(), panels):
        if panel == "ChEMBL36":
            axis.scatter(
                chembl["umap1"], chembl["umap2"], s=0.6, color=CHEMBL_COLOR,
                alpha=0.14, linewidths=0, rasterized=True,
            )
            axis.set_title(f"ChEMBL36  |  n={len(chembl):,}", fontsize=18)
        else:
            group = generated[generated["model"] == panel]
            axis.scatter(
                chembl["umap1"], chembl["umap2"], s=0.5, color=CHEMBL_COLOR,
                alpha=0.09, linewidths=0, rasterized=True,
            )
            axis.scatter(
                group["umap1"], group["umap2"], s=7.5, color=MODEL_COLORS[panel],
                alpha=0.55, linewidths=0, rasterized=True,
            )
            axis.set_title(f"{panel}  |  n={len(group):,}", fontsize=18)
        axis.set_xlabel("UMAP 1", fontsize=16)
        axis.set_ylabel("UMAP 2", fontsize=16)
        axis.tick_params(labelsize=13)
    fig.suptitle(
        "Model-specific coverage and ChEMBL36 reference in one shared chemical-space map",
        fontsize=21,
    )
    fig.text(
        0.5, 0.008,
        "Coordinates are descriptive neighborhood projections; distance is not efficacy or molecular quality.",
        ha="center", fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.965))
    _save_figure(fig, output_dir / "five_model_chembl100k_umap_facets")


def _write_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    chembl_payload: dict[str, Any],
    coordinate_table: pd.DataFrame,
    source_hashes: dict[str, str],
) -> None:
    """Write exact data lineage, software versions and projection parameters."""
    manifest = {
        "analysis": "joint ECFP4-UMAP of five formal benchmark models and ChEMBL36 100k",
        "created_by": str(Path(__file__).resolve()),
        "sources": {
            "chembl_fingerprints": str(args.chembl_fingerprints.resolve()),
            "generated_molecules": str(args.generated_csv.resolve()),
            "sha256": source_hashes,
        },
        "chembl_package": {
            key: chembl_payload[key]
            for key in [
                "radius", "n_bits", "fingerprint_type", "chembl_version",
                "source", "sampled_from_total", "sampling_seed",
            ]
        },
        "projection": {
            "algorithm": "UMAP",
            "fit_scope": "one joint fit on ChEMBL100k plus all five model groups",
            "n_components": 2,
            "metric": args.metric,
            "n_neighbors": args.n_neighbors,
            "min_dist": args.min_dist,
            "random_state": args.random_state,
            "low_memory": True,
        },
        "counts": {
            "total": int(len(coordinate_table)),
            "chembl": int((coordinate_table["model"] == "ChEMBL36").sum()),
            "generated_total": int((coordinate_table["dataset"] == "Generated").sum()),
            "generated_by_model": {
                model: int((coordinate_table["model"] == model).sum()) for model in MODEL_ORDER
            },
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "rdkit": rdBase.rdkitVersion,
            "umap_learn": umap.__version__,
            "matplotlib": matplotlib.__version__,
            "platform": platform.platform(),
        },
        "interpretation_boundary": (
            "UMAP is a descriptive neighborhood visualization. Coordinate distance must not be "
            "interpreted as target efficacy, experimental activity, or absolute molecular quality."
        ),
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)


def main() -> None:
    """Run the complete joint embedding, export and provenance workflow."""
    args = parse_args()
    _configure_logging(args.output_dir)
    _configure_plotting()
    LOGGER.info("command: %s", " ".join(sys.argv))
    LOGGER.info("output directory: %s", args.output_dir.resolve())

    chembl_fingerprints, chembl_records, chembl_payload = _load_chembl(
        args.chembl_fingerprints
    )
    radius = int(chembl_payload["radius"])
    n_bits = int(chembl_payload["n_bits"])
    generated, generated_fingerprints = _load_generated(
        args.generated_csv, radius, n_bits, args.workers
    )

    chembl_matrix = _bitvectors_to_csr(chembl_fingerprints, n_bits)
    generated_matrix = _bitvectors_to_csr(generated_fingerprints, n_bits)
    joint_matrix = sparse.vstack([chembl_matrix, generated_matrix], format="csr")
    LOGGER.info("joint matrix: shape=%s nnz=%d", joint_matrix.shape, joint_matrix.nnz)

    reducer = umap.UMAP(
        n_components=2,
        metric=args.metric,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.random_state,
        low_memory=True,
        verbose=True,
    )
    LOGGER.info(
        "fitting one joint UMAP: n=%d metric=%s neighbors=%d min_dist=%.3f seed=%d",
        joint_matrix.shape[0], args.metric, args.n_neighbors, args.min_dist,
        args.random_state,
    )
    coordinates = reducer.fit_transform(joint_matrix)
    coordinate_table = _build_coordinate_table(chembl_records, generated, coordinates)
    coordinate_path = args.output_dir / "joint_umap_coordinates.csv.gz"
    coordinate_table.to_csv(coordinate_path, index=False, compression="gzip")
    LOGGER.info("saved coordinates: %s", coordinate_path)

    summary = _summarize_groups(coordinate_table)
    summary_path = args.output_dir / "group_summary.csv"
    summary.to_csv(summary_path, index=False)
    LOGGER.info("group summary:\n%s", summary.to_string(index=False))

    _plot_overview(coordinate_table, args.output_dir, args.random_state)
    _plot_facets(coordinate_table, args.output_dir)
    source_hashes = {
        "chembl_fingerprints": _sha256(args.chembl_fingerprints),
        "generated_molecules": _sha256(args.generated_csv),
    }
    _write_manifest(
        args.output_dir, args, chembl_payload, coordinate_table, source_hashes
    )
    LOGGER.info("completed successfully")


if __name__ == "__main__":
    main()
