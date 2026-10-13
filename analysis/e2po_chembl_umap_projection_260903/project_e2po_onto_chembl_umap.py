#!/usr/bin/env python3
"""Project E2PO training-checkpoint molecules onto a fixed ChEMBL36 100k UMAP map.

Purpose
-------
Answer one question for the E2PO mainline paper: **as training progresses, do the
generated molecules drift toward or away from real drug-like chemical space?**

Method and why it differs from the five-model figure
----------------------------------------------------
The existing ``five_model_chembl100k_umap_20260730`` figure used a **joint fit**
(ChEMBL100k + five generators embedded together) and did not persist the fitted
reducer, so its coordinates cannot be reused for new molecules.

This script instead:

1. Fits UMAP on **ChEMBL36 100k ECFP4 alone** -> a fixed, uncontaminated
   reference map. The ChEMBL background is therefore *identical for every
   generated group* and independent of which molecules we later add.
2. Projects every generated molecule into that fixed map with
   ``UMAP.transform()``.
3. Persists the fitted reducer so future molecules can be added to the *same*
   map without refitting.

Consequence to state in reports: the ChEMBL background here is **not** numerically
identical to the five-model joint-fit figure. It is the cleaner baseline for
"distance to ChEMBL" questions because the reference map is not distorted by the
generated molecules being scored.

Distance metrics
----------------
Two independent measures are reported per molecule:

* ``nearest_chembl_umap_distance``: Euclidean distance to the nearest ChEMBL
  point **in the 2D UMAP map**. Reported for direct comparability with the
  existing five-model ``nearest_chembl_distance_per_generated_molecule.csv.gz``.
* ``nearest_chembl_tanimoto_distance``: ``1 - max Tanimoto similarity`` over all
  100k ChEMBL fingerprints, computed **in ECFP4 space**. This is the
  defensible ground truth, because UMAP 2D distances are known to distort
  global geometry. Computed with chunked sparse matmuls to bound memory.

Inputs
------
* ChEMBL: ``data/chembl/chembl_36_100k_ecfp4_fingerprints.pkl.gz``
* Generated: the **relaxed largest-fragment** SDF of each checkpoint, matching
  the denominator used by ``evaluate_reasonable_molecules.py``.

Usage
-----
    python analysis/e2po_chembl_umap_projection_260903/project_e2po_onto_chembl_umap.py \
        --workers 18
"""

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
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/diffb110_matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/diffb110_numba_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from matplotlib.colors import Normalize
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import AllChem
from scipy import sparse
from scipy.spatial import cKDTree
import sklearn
import umap

RDLogger.DisableLog("rdApp.*")
LOGGER = logging.getLogger("e2po_chembl_umap_projection")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHEMBL = ROOT / "data/chembl/chembl_36_100k_ecfp4_fingerprints.pkl.gz"
DEFAULT_OUTPUT = Path(__file__).resolve().parent

GENERATION_ROOT = ROOT / "OUT/diffsbdd_e2po_mainline_intermediate_gen"
ORIGINAL_SDF = (
    ROOT
    / "analysis/diffsbdd_global_e2po_structure_matched_x_only_generation_1000_260901"
    / "molecules_relaxed_largest.sdf"
)

# (group label, source SDF, training update step used for ordering)
STEP_GROUPS: list[tuple[str, Path, int]] = [
    ("Original", ORIGINAL_SDF, -1),
    *[
        (
            f"step_{step:06d}",
            GENERATION_ROOT / f"step_{step:06d}" / "molecules_relaxed_largest.sdf",
            step,
        )
        for step in range(2000, 19000, 2000)
    ],
    ("final", GENERATION_ROOT / "step_final" / "molecules_relaxed_largest.sdf", 18116),
]

ORIGINAL_COLOR = "#3498DB"
STEP_4000_COLOR = "#E74C3C"
CHEMBL_COLOR = "#A8ADB3"
STEP_COLORMAP = "viridis"

# Panels shown in the facet figure, in training order.
FACET_GROUPS = [
    "Original",
    "step_002000",
    "step_004000",
    "step_006000",
    "step_010000",
    "step_014000",
    "step_018000",
    "final",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chembl-fingerprints", type=Path, default=DEFAULT_CHEMBL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--n-neighbors", type=int, default=20)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--reuse-reducer",
        action="store_true",
        help="Load the cached fitted reducer instead of refitting ChEMBL UMAP.",
    )
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
            "legend.fontsize": 12,
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
    return digest


def _load_chembl(path: Path) -> tuple[list[Any], pd.DataFrame, dict[str, Any]]:
    """Load and validate the ChEMBL36 100k ECFP4 fingerprint package."""
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
        return index, AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=n_bits)
    except Exception:
        return index, None


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


def _load_group_molecules(
    label: str, sdf_path: Path, step: int, radius: int, n_bits: int, workers: int
) -> pd.DataFrame | None:
    """Read one relaxed SDF and attach canonical SMILES plus ECFP4 bit vectors."""
    if not sdf_path.exists():
        LOGGER.error("missing SDF for group %s: %s", label, sdf_path)
        return None
    supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=True, removeHs=False)
    rows: list[dict[str, Any]] = []
    for source_index, molecule in enumerate(supplier):
        if molecule is None:
            continue
        try:
            smiles = Chem.MolToSmiles(Chem.RemoveHs(molecule))
        except Exception:
            continue
        if not smiles:
            continue
        rows.append(
            {
                "group": label,
                "step": step,
                "source_sdf": str(sdf_path),
                "source_index": source_index,
                "canonical_smiles": smiles,
            }
        )
    if not rows:
        LOGGER.error("no readable molecules for group %s", label)
        return None
    frame = pd.DataFrame(rows)
    LOGGER.info(
        "group %s: %d readable molecules, calculating ECFP4 with %d workers",
        label, len(frame), workers,
    )
    tasks = [
        (index, smiles, radius, n_bits)
        for index, smiles in enumerate(frame["canonical_smiles"].tolist())
    ]
    with Pool(processes=workers) as pool:
        results = pool.map(_fingerprint_task, tasks, chunksize=128)
    valid = {index: fp for index, fp in results if fp is not None}
    if len(valid) != len(frame):
        LOGGER.warning(
            "group %s: dropped %d molecules without ECFP4", label, len(frame) - len(valid)
        )
    frame = frame.iloc[[index for index in sorted(valid)]].reset_index(drop=True)
    frame["fingerprint"] = [valid[index] for index in sorted(valid)]
    LOGGER.info("group %s: %d molecules with ECFP4", label, len(frame))
    return frame


def _fit_or_load_reducer(
    chembl_matrix: sparse.csr_matrix, args: argparse.Namespace
) -> tuple[umap.UMAP, np.ndarray, bool]:
    """Fit UMAP on ChEMBL alone, or reuse a cached reducer."""
    cache_path = args.output_dir / f"chembl100k_umap_reducer_seed{args.random_state}.pkl.gz"
    coords_cache = args.output_dir / f"chembl100k_umap_coords_seed{args.random_state}.npy"
    if args.reuse_reducer and cache_path.exists() and coords_cache.exists():
        LOGGER.info("reusing cached reducer: %s", cache_path)
        with gzip.open(cache_path, "rb") as handle:
            reducer = pickle.load(handle)
        coords = np.load(coords_cache)
        return reducer, coords, True

    LOGGER.info(
        "fitting UMAP on ChEMBL100k only: n=%d metric=%s neighbors=%d min_dist=%.3f seed=%d",
        chembl_matrix.shape[0], args.metric, args.n_neighbors, args.min_dist,
        args.random_state,
    )
    reducer = umap.UMAP(
        n_components=2,
        metric=args.metric,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.random_state,
        low_memory=True,
        verbose=True,
    )
    coords = reducer.fit_transform(chembl_matrix)
    LOGGER.info("UMAP fit complete; persisting reducer for reuse")
    with gzip.open(cache_path, "wb") as handle:
        pickle.dump(reducer, handle)
    np.save(coords_cache, coords)
    return reducer, coords, False


def _nearest_tanimoto_distances(
    generated_matrix: sparse.csr_matrix,
    chembl_matrix: sparse.csr_matrix,
    *,
    gen_chunk: int = 500,
    chembl_chunk: int = 25_000,
) -> np.ndarray:
    """Return ``1 - max Tanimoto similarity`` of each row against all ChEMBL rows.

    Chunked so peak memory stays near ``gen_chunk * chembl_chunk * 4`` bytes.
    """
    n_generated = generated_matrix.shape[0]
    best = np.zeros(n_generated, dtype=np.float32)
    chembl_counts = np.asarray(chembl_matrix.sum(axis=1), dtype=np.float32).ravel()
    gen_counts = np.asarray(generated_matrix.sum(axis=1), dtype=np.float32).ravel()
    for gen_start in range(0, n_generated, gen_chunk):
        gen_stop = min(gen_start + gen_chunk, n_generated)
        gen_block = generated_matrix[gen_start:gen_stop]
        block_best = np.zeros(gen_stop - gen_start, dtype=np.float32)
        for chem_start in range(0, chembl_matrix.shape[0], chembl_chunk):
            chem_stop = min(chem_start + chembl_chunk, chembl_matrix.shape[0])
            chem_block = chembl_matrix[chem_start:chem_stop]
            intersections = (gen_block @ chem_block.T).toarray().astype(np.float32)
            union = (
                gen_counts[gen_start:gen_stop, None]
                + chembl_counts[chem_start:chem_stop][None, :]
                - intersections
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                similarity = np.where(union > 0, intersections / union, 0.0)
            block_best = np.maximum(block_best, similarity.max(axis=1))
        best[gen_start:gen_stop] = block_best
        LOGGER.info(
            "tanimoto nearest-neighbour: %d/%d rows", gen_stop, n_generated
        )
    return 1.0 - best


def _group_color(label: str, step: int, norm: Normalize, cmap: Any) -> Any:
    """Return the plotting colour for one group."""
    if label == "Original":
        return ORIGINAL_COLOR
    if label == "step_004000":
        return STEP_4000_COLOR
    return cmap(norm(step))


def _plot_overview(
    chembl_xy: np.ndarray,
    generated: pd.DataFrame,
    output_dir: Path,
    seed: int,
) -> None:
    """Plot every checkpoint over the fixed ChEMBL background in one panel."""
    norm = Normalize(
        vmin=min(generated["step"].min(), 0), vmax=generated["step"].max()
    )
    cmap = plt.get_cmap(STEP_COLORMAP)
    fig, axis = plt.subplots(figsize=(12.4, 7.6))
    axis.scatter(
        chembl_xy[:, 0], chembl_xy[:, 1], s=0.7, color=CHEMBL_COLOR,
        alpha=0.10, linewidths=0, rasterized=True,
        label=f"ChEMBL36 reference (n={len(chembl_xy):,})",
    )
    rng = np.random.default_rng(seed)
    order = sorted(generated["group"].unique(), key=lambda label: -generated.loc[
        generated["group"] == label, "step"
    ].iloc[0])
    for label in order:
        group = generated[generated["group"] == label]
        shuffled = group.iloc[rng.permutation(len(group))]
        step = int(group["step"].iloc[0])
        axis.scatter(
            shuffled["umap1"], shuffled["umap2"], s=9.0,
            color=_group_color(label, step, norm, cmap),
            alpha=0.55, linewidths=0, rasterized=True,
            label=f"{label} (n={len(group):,})",
        )
    axis.set_title("E2PO training checkpoints in fixed ChEMBL36 chemical space")
    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2")
    axis.legend(loc="upper left", markerscale=2.0, ncol=2)
    fig.text(
        0.5, 0.012,
        "ChEMBL36 100k reference fitted alone (ECFP4 radius 2 / 2048 bits, cosine UMAP, "
        "seed 42); generated molecules projected with UMAP.transform onto that fixed map.\n"
        "Molecules are relaxed largest fragments from 1000 attempts per checkpoint.",
        ha="center", fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    _save_figure(fig, output_dir / "e2po_chembl_umap_overview")


def _plot_facets(
    chembl_xy: np.ndarray, generated: pd.DataFrame, output_dir: Path
) -> None:
    """Plot one checkpoint-vs-ChEMBL panel per selected training stage."""
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 8.0), sharex=True, sharey=True)
    for axis, label in zip(axes.ravel(), FACET_GROUPS):
        axis.scatter(
            chembl_xy[:, 0], chembl_xy[:, 1], s=0.5, color=CHEMBL_COLOR,
            alpha=0.09, linewidths=0, rasterized=True,
        )
        group = generated[generated["group"] == label]
        colour = ORIGINAL_COLOR if label == "Original" else (
            STEP_4000_COLOR if label == "step_004000" else "#7B4EA8"
        )
        axis.scatter(
            group["umap1"], group["umap2"], s=8.0, color=colour,
            alpha=0.55, linewidths=0, rasterized=True,
        )
        axis.set_title(f"{label}  |  n={len(group):,}", fontsize=16)
        axis.set_xlabel("UMAP 1", fontsize=14)
        axis.set_ylabel("UMAP 2", fontsize=14)
        axis.tick_params(labelsize=12)
    fig.suptitle(
        "E2PO checkpoint coverage in one shared ChEMBL36 reference map", fontsize=20
    )
    fig.text(
        0.5, 0.008,
        "All panels share one fixed ChEMBL-fitted UMAP. Coordinate distance is descriptive "
        "neighbourhood structure, not efficacy or molecular quality.",
        ha="center", fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.955))
    _save_figure(fig, output_dir / "e2po_chembl_umap_facets")


def _plot_drift(summary: pd.DataFrame, output_dir: Path) -> None:
    """Plot distance-to-ChEMBL metrics as a function of training step."""
    steps = summary["step"].to_numpy(dtype=float)
    original = summary[summary["group"] == "Original"].iloc[0]
    train = summary[summary["step"] > 0].copy()
    train = train.sort_values("step")
    x_train = train["step"].to_numpy(dtype=float)

    panels = [
        ("nearest_chembl_umap_distance_median", "Nearest ChEMBL distance (UMAP map)"),
        ("nearest_chembl_tanimoto_distance_median", "Nearest ChEMBL distance (ECFP4 Tanimoto)"),
        ("fraction_within_chembl_umap_p95", "Fraction inside ChEMBL P95 radius"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.4))
    for axis, (column, title) in zip(axes, panels):
        axis.plot(
            x_train, train[column].to_numpy(), marker="o", color="#7B4EA8",
            linewidth=2.2, markersize=7, label="E2PO checkpoints",
        )
        baseline = float(original[column])
        axis.axhline(
            baseline, color=ORIGINAL_COLOR, linestyle="--", linewidth=2.0, alpha=0.9,
            label=f"Original: {baseline:.4f}",
        )
        best = train.loc[train["group"] == "step_004000", column]
        if len(best):
            axis.scatter(
                [4000], [float(best.iloc[0])], s=170, facecolors="none",
                edgecolors=STEP_4000_COLOR, linewidths=2.4, zorder=5,
                label="step_4000 (recommended)",
            )
        axis.set_title(title, fontsize=15)
        axis.set_xlabel("Training update step", fontsize=14)
        axis.set_ylabel(column.split("_")[-1] if "fraction" in column else "distance", fontsize=13)
        axis.legend(fontsize=11, loc="best")
        axis.tick_params(labelsize=12)
        axis.grid(alpha=0.3)
    fig.suptitle(
        "Distance to ChEMBL36 chemical space across E2PO training", fontsize=19
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save_figure(fig, output_dir / "e2po_chembl_drift_metrics")


def _save_figure(fig: plt.Figure, stem: Path) -> None:
    """Export one figure in preview, vector and print formats."""
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _write_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    chembl_payload: dict[str, Any],
    summary: pd.DataFrame,
    reused: bool,
) -> None:
    """Write data lineage, software versions and projection parameters."""
    manifest = {
        "analysis": "E2PO checkpoints projected onto a ChEMBL36-100k-only UMAP reference map",
        "created_by": str(Path(__file__).resolve()),
        "sources": {
            "chembl_fingerprints": str(args.chembl_fingerprints.resolve()),
            "sha256": {"chembl_fingerprints": _sha256(args.chembl_fingerprints)},
            "generated_groups": {
                label: str(path.resolve()) for label, path, _ in STEP_GROUPS
            },
        },
        "chembl_package": {
            key: chembl_payload[key]
            for key in [
                "radius", "n_bits", "fingerprint_type", "chembl_version",
                "sampled_from_total", "sampling_seed",
            ]
        },
        "projection": {
            "algorithm": "UMAP",
            "fit_scope": "ChEMBL36 100k only (generated molecules never influence the fit)",
            "generated_placement": "UMAP.transform into the fixed ChEMBL map",
            "reducer_reused_from_cache": reused,
            "n_components": 2,
            "metric": args.metric,
            "n_neighbors": args.n_neighbors,
            "min_dist": args.min_dist,
            "random_state": args.random_state,
            "low_memory": True,
        },
        "counts": {
            "chembl_reference": 100_000,
            "generated_by_group": {
                str(row["group"]): int(row["n"]) for _, row in summary.iterrows()
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
            "UMAP is a descriptive neighbourhood projection. Distances in the 2D map are "
            "not efficacy, activity, or absolute quality. The ECFP4 Tanimoto nearest-neighbour "
            "distance is the metric to cite for 'how far from ChEMBL' claims; the UMAP-space "
            "distance is included only for comparability with the earlier five-model figure."
        ),
        "difference_from_five_model_figure": (
            "The five-model figure used a joint UMAP fit (ChEMBL + five generators together) "
            "and did not persist its reducer. This map fits ChEMBL alone, so the background is "
            "not numerically identical to that figure but is reusable and uncontaminated."
        ),
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)


def main() -> None:
    """Run the projection, metric, plot and provenance workflow."""
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

    frames: list[pd.DataFrame] = []
    for label, sdf_path, step in STEP_GROUPS:
        frame = _load_group_molecules(label, sdf_path, step, radius, n_bits, args.workers)
        if frame is not None:
            frames.append(frame)
    if not frames:
        raise RuntimeError("no generated molecules could be loaded")
    generated = pd.concat(frames, ignore_index=True)
    LOGGER.info("generated molecules total: %d", len(generated))

    chembl_matrix = _bitvectors_to_csr(chembl_fingerprints, n_bits)
    generated_matrix = _bitvectors_to_csr(
        generated["fingerprint"].tolist(), n_bits
    )

    reducer, chembl_xy, reused = _fit_or_load_reducer(chembl_matrix, args)

    LOGGER.info(
        "projecting %d generated molecules into the fixed ChEMBL map", len(generated)
    )
    generated_xy = reducer.transform(generated_matrix)
    generated["umap1"] = generated_xy[:, 0]
    generated["umap2"] = generated_xy[:, 1]

    LOGGER.info("computing nearest-ChEMBL distances in the 2D map")
    tree = cKDTree(chembl_xy)
    generated["nearest_chembl_umap_distance"] = tree.query(
        generated_xy, k=1, workers=-1
    )[0]
    chembl_self = tree.query(chembl_xy, k=2, workers=-1)[0][:, 1]
    chembl_p95 = float(np.quantile(chembl_self, 0.95))

    LOGGER.info("computing nearest-ChEMBL Tanimoto distances in ECFP4 space")
    generated["nearest_chembl_tanimoto_distance"] = _nearest_tanimoto_distances(
        generated_matrix, chembl_matrix
    )

    chembl_centroid = chembl_xy.mean(axis=0)
    rows: list[dict[str, Any]] = [
        {
            "group": "ChEMBL36",
            "step": -2,
            "n": len(chembl_xy),
            "centroid_umap1": float(chembl_centroid[0]),
            "centroid_umap2": float(chembl_centroid[1]),
            "centroid_distance_to_chembl": 0.0,
            "nearest_chembl_umap_distance_mean": float(np.mean(chembl_self)),
            "nearest_chembl_umap_distance_median": float(np.median(chembl_self)),
            "nearest_chembl_tanimoto_distance_median": float("nan"),
            "fraction_within_chembl_umap_p95": float(np.mean(chembl_self <= chembl_p95)),
        }
    ]
    for label, _, step in STEP_GROUPS:
        group = generated[generated["group"] == label]
        if group.empty:
            continue
        xy = group[["umap1", "umap2"]].to_numpy()
        centroid = xy.mean(axis=0)
        umap_dist = group["nearest_chembl_umap_distance"].to_numpy()
        rows.append(
            {
                "group": label,
                "step": step,
                "n": int(len(group)),
                "centroid_umap1": float(centroid[0]),
                "centroid_umap2": float(centroid[1]),
                "centroid_distance_to_chembl": float(
                    np.linalg.norm(centroid - chembl_centroid)
                ),
                "nearest_chembl_umap_distance_mean": float(np.mean(umap_dist)),
                "nearest_chembl_umap_distance_median": float(np.median(umap_dist)),
                "nearest_chembl_tanimoto_distance_median": float(
                    np.median(group["nearest_chembl_tanimoto_distance"].to_numpy())
                ),
                "fraction_within_chembl_umap_p95": float(
                    np.mean(umap_dist <= chembl_p95)
                ),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "chembl_distance_summary.csv", index=False)
    LOGGER.info("distance summary:\n%s", summary.to_string(index=False))

    coordinates = generated.drop(columns=["fingerprint"]).copy()
    coordinates.to_csv(
        args.output_dir / "e2po_projected_coordinates.csv.gz",
        index=False, compression="gzip",
    )
    pd.DataFrame(
        {"umap1": chembl_xy[:, 0], "umap2": chembl_xy[:, 1]}
    ).to_csv(
        args.output_dir / "chembl_reference_coordinates.csv.gz",
        index=False, compression="gzip",
    )

    _plot_overview(chembl_xy, generated, args.output_dir, args.random_state)
    _plot_facets(chembl_xy, generated, args.output_dir)
    _plot_drift(summary, args.output_dir)
    _write_manifest(args.output_dir, args, chembl_payload, summary, reused)
    LOGGER.info("completed successfully")


if __name__ == "__main__":
    main()
