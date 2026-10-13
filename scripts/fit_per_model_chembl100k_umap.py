#!/usr/bin/env python3
"""Fit one independent ChEMBL100k-plus-model ECFP4 UMAP per benchmark model."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import multiprocessing as mp
import os
import pickle
import platform
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/diffb110_independent_umap_mplconfig")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/diffb110_independent_umap_numba_cache")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import umap
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import AllChem
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

RDLogger.DisableLog("rdApp.error")

LOGGER = logging.getLogger("fit_per_model_chembl100k_umap")
MODELS = ["DiffGui", "DiffSBDD", "PMDM", "GCDM-SBDD", "TargetDiff"]
COLORS = {
    "DiffGui": "#3B82C4",
    "DiffSBDD": "#D95F59",
    "PMDM": "#D9A441",
    "GCDM-SBDD": "#8E78B7",
    "TargetDiff": "#3D9B72",
}
CHEMBL_COLOR = "#A8ADB3"


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_plotting() -> None:
    """Configure readable Python/matplotlib output."""
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 12,
            "axes.titlesize": 17,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def bitvector_to_indices(bit_vector: Any) -> np.ndarray:
    """Convert one RDKit bit vector into sorted active-bit indices."""
    return np.fromiter(
        bit_vector.GetOnBits(), dtype=np.int32, count=bit_vector.GetNumOnBits()
    )


def bitvectors_to_csr(bit_vectors: list[Any], n_bits: int) -> sparse.csr_matrix:
    """Convert RDKit bit vectors into a compact binary CSR matrix."""
    indptr = np.zeros(len(bit_vectors) + 1, dtype=np.int64)
    active = [bitvector_to_indices(value) for value in bit_vectors]
    for index, row in enumerate(active):
        indptr[index + 1] = indptr[index] + len(row)
    indices = np.concatenate(active) if active else np.empty(0, dtype=np.int32)
    data = np.ones(len(indices), dtype=np.float32)
    return sparse.csr_matrix((data, indices, indptr), shape=(len(active), n_bits))


def fingerprint_task(task: tuple[int, str, int, int]) -> tuple[int, Any | None]:
    """Calculate one Morgan fingerprint while preserving input order."""
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


def generated_fingerprints(
    frame: pd.DataFrame, radius: int, n_bits: int, workers: int
) -> tuple[pd.DataFrame, list[Any]]:
    """Calculate fingerprints for all sanitize-valid generated molecules."""
    tasks = [
        (index, smiles, radius, n_bits)
        for index, smiles in enumerate(frame["canonical_smiles"].astype(str))
    ]
    with mp.Pool(workers) as pool:
        results = pool.map(fingerprint_task, tasks, chunksize=128)
    valid = [(index, fingerprint) for index, fingerprint in results if fingerprint is not None]
    if len(valid) != len(frame):
        valid_indices = [index for index, _ in valid]
        frame = frame.iloc[valid_indices].copy().reset_index(drop=True)
        LOGGER.warning("dropped %d generated molecules without ECFP4", len(tasks) - len(valid))
    return frame, [fingerprint for _, fingerprint in valid]


def load_inputs(
    chembl_path: Path, generated_path: Path, workers: int
) -> tuple[sparse.csr_matrix, pd.DataFrame, list[Any], dict[str, Any]]:
    """Load the traced ChEMBL package and all valid benchmark molecules."""
    with gzip.open(chembl_path, "rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("ChEMBL fingerprint package is not a mapping")
    required = {"fingerprints", "records", "radius", "n_bits", "fingerprint_type", "chembl_version"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"ChEMBL package missing keys: {sorted(missing)}")
    if len(payload["fingerprints"]) != 100_000 or len(payload["records"]) != 100_000:
        raise ValueError("expected exactly 100,000 ChEMBL molecules")
    if payload["radius"] != 2 or payload["n_bits"] != 2048:
        raise ValueError("expected ECFP4 radius=2 and 2048 bits")
    chembl_records = pd.DataFrame(payload["records"])
    generated = pd.read_csv(
        generated_path,
        usecols=["row_id", "repeat", "model", "source_sdf", "source_index", "sanitize_success", "canonical_smiles"],
    )
    generated = generated[
        generated["model"].isin(MODELS)
        & generated["sanitize_success"].eq(1)
        & generated["canonical_smiles"].notna()
    ].copy()
    # Fit chemical-space topology on unique structures; repeat frequency is
    # reported separately and must not alter the UMAP neighborhood graph.
    generated["canonical_smiles"] = generated["canonical_smiles"].astype(str)
    generated = generated.drop_duplicates(
        subset=["model", "canonical_smiles"], keep="first"
    ).reset_index(drop=True)
    generated, generated_fp = generated_fingerprints(
        generated, int(payload["radius"]), int(payload["n_bits"]), workers
    )
    chembl_matrix = bitvectors_to_csr(payload["fingerprints"], int(payload["n_bits"]))
    expected_unique = {
        "DiffGui": 2578,
        "DiffSBDD": 1999,
        "PMDM": 1726,
        "GCDM-SBDD": 1538,
        "TargetDiff": 2968,
    }
    observed_unique = generated.groupby("model")["canonical_smiles"].nunique().to_dict()
    if observed_unique != expected_unique:
        raise ValueError(
            f"unexpected per-model unique counts: observed={observed_unique}, "
            f"expected={expected_unique}"
        )
    LOGGER.info(
        "ChEMBL matrix=%s generated_unique=%d original_valid=%d",
        chembl_matrix.shape,
        len(generated),
        int(pd.read_csv(generated_path, usecols=["model", "sanitize_success", "canonical_smiles"])
            .query("model in @MODELS and sanitize_success == 1 and canonical_smiles == canonical_smiles")
            .shape[0]),
    )
    return chembl_matrix, generated, generated_fp, payload


def save_figure(figure: plt.Figure, stem: Path) -> None:
    """Save a Python-rendered figure in PNG, SVG and PDF formats."""
    figure.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def calculate_metrics(
    chembl_xy: np.ndarray, model_xy: np.ndarray, workers: int
) -> dict[str, float]:
    """Calculate descriptive 2D overlap metrics for one fitted map."""
    neighbors = NearestNeighbors(n_neighbors=1, n_jobs=workers).fit(chembl_xy)
    nearest = neighbors.kneighbors(model_xy, return_distance=True)[0][:, 0]
    density, x_edges, y_edges = np.histogram2d(chembl_xy[:, 0], chembl_xy[:, 1], bins=400)
    nonzero = density[density > 0]
    ordered = np.sort(density.ravel())[::-1]
    threshold_index = min(
        int(np.searchsorted(np.cumsum(ordered) / ordered.sum(), 0.95)),
        len(ordered) - 1,
    )
    threshold = ordered[threshold_index]
    x_bin = np.searchsorted(x_edges, model_xy[:, 0], side="right") - 1
    y_bin = np.searchsorted(y_edges, model_xy[:, 1], side="right") - 1
    valid = (
        (x_bin >= 0)
        & (x_bin < density.shape[0])
        & (y_bin >= 0)
        & (y_bin < density.shape[1])
    )
    inside_hdr95 = np.zeros(len(model_xy), dtype=bool)
    inside_hdr95[valid] = density[x_bin[valid], y_bin[valid]] >= threshold
    return {
        "median_nearest_chembl_umap_distance": float(np.median(nearest)),
        "p95_nearest_chembl_umap_distance": float(np.quantile(nearest, 0.95)),
        "inside_chembl_hdr95_fraction": float(inside_hdr95.mean()),
        "model_umap1_mean": float(model_xy[:, 0].mean()),
        "model_umap2_mean": float(model_xy[:, 1].mean()),
        "model_umap_rms_radius": float(
            np.sqrt(np.mean(np.sum((model_xy - model_xy.mean(axis=0)) ** 2, axis=1)))
        ),
        "chembl_nonzero_density_bins": float(len(nonzero)),
    }


def plot_model(
    model: str, chembl_xy: np.ndarray, model_xy: np.ndarray, output_dir: Path
) -> None:
    """Plot ChEMBL and one model in that model-specific coordinate system."""
    figure, axis = plt.subplots(figsize=(10.5, 7.4))
    axis.scatter(
        chembl_xy[:, 0], chembl_xy[:, 1], s=0.7, color=CHEMBL_COLOR,
        alpha=0.12, linewidths=0, rasterized=True,
    )
    axis.scatter(
        model_xy[:, 0], model_xy[:, 1], s=8.0, color=COLORS[model],
        alpha=0.58, linewidths=0, rasterized=True,
    )
    axis.set_title(f"{model}: independent fit of ChEMBL100k + {model}", fontsize=19)
    axis.set_xlabel("UMAP 1 (model-specific coordinate system)")
    axis.set_ylabel("UMAP 2 (model-specific coordinate system)")
    axis.grid(alpha=0.14)
    figure.text(
        0.5, 0.012,
        "Gray = ChEMBL100k; colored points = the fitted model. Absolute coordinates are not comparable across panels.",
        ha="center", fontsize=11,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 1))
    save_figure(figure, output_dir / f"{model}_chembl100k_independent_umap")


def plot_facets(
    coordinates: dict[str, tuple[np.ndarray, np.ndarray]], output_dir: Path
) -> None:
    """Create a contact sheet with independent axes for all five fitted maps."""
    figure, axes = plt.subplots(2, 3, figsize=(17, 10))
    for axis, model in zip(axes.ravel(), MODELS):
        chembl_xy, model_xy = coordinates[model]
        axis.scatter(chembl_xy[:, 0], chembl_xy[:, 1], s=0.45, color=CHEMBL_COLOR, alpha=0.10, linewidths=0, rasterized=True)
        axis.scatter(model_xy[:, 0], model_xy[:, 1], s=5.5, color=COLORS[model], alpha=0.56, linewidths=0, rasterized=True)
        axis.set_title(f"{model} | ChEMBL100k + {model}", fontsize=15)
        axis.set_xlabel("UMAP 1 (independent)")
        axis.set_ylabel("UMAP 2 (independent)")
        axis.grid(alpha=0.12)
    axes.ravel()[-1].axis("off")
    figure.suptitle("Auxiliary independent UMAP fits: each model merged with ChEMBL100k", fontsize=20)
    figure.text(
        0.5, 0.012,
        "Each panel is fitted independently; panel-to-panel coordinate positions and orientations must not be compared.",
        ha="center", fontsize=12,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.96))
    save_figure(figure, output_dir / "five_model_independent_umap_facets")


def main() -> None:
    """Fit, export and audit five independent ChEMBL100k UMAP maps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chembl-package", type=Path, required=True)
    parser.add_argument("--generated-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--n-neighbors", type=int, default=20)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.output_dir / "independent_fit.log", encoding="utf-8"),
        ],
        force=True,
    )
    configure_plotting()
    started = time.time()
    chembl_matrix, generated, generated_fp, chembl_payload = load_inputs(
        args.chembl_package, args.generated_csv, args.workers
    )
    generated_matrix = bitvectors_to_csr(generated_fp, int(chembl_payload["n_bits"]))
    all_coordinates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    summary_rows: list[dict[str, Any]] = []
    for model in MODELS:
        model_dir = args.output_dir / model
        model_dir.mkdir(parents=True, exist_ok=True)
        query = generated[generated["model"].eq(model)].reset_index(drop=True)
        query_indices = generated.index[generated["model"].eq(model)].to_numpy()
        query_matrix = generated_matrix[query_indices]
        joint_matrix = sparse.vstack([chembl_matrix, query_matrix], format="csr")
        LOGGER.info("fitting %s: ChEMBL=%d model=%d total=%d", model, chembl_matrix.shape[0], len(query), joint_matrix.shape[0])
        reducer = umap.UMAP(
            n_components=2,
            metric="cosine",
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            low_memory=True,
            random_state=args.random_state,
            verbose=True,
        )
        coordinates = reducer.fit_transform(joint_matrix).astype(np.float32)
        if not np.isfinite(coordinates).all():
            raise ValueError(f"non-finite coordinates for {model}")
        chembl_n = chembl_matrix.shape[0]
        chembl_xy = coordinates[:chembl_n]
        model_xy = coordinates[chembl_n:]
        all_coordinates[model] = (chembl_xy, model_xy)
        chembl_records = pd.DataFrame(chembl_payload["records"])
        coordinate_table = pd.concat(
            [
                pd.DataFrame({"group": "ChEMBL36", "molecule_id": chembl_records["chembl_id"].astype(str), "canonical_smiles": chembl_records["canonical_smiles"].astype(str), "umap1": chembl_xy[:, 0], "umap2": chembl_xy[:, 1]}),
                query.assign(group=model, umap1=model_xy[:, 0], umap2=model_xy[:, 1]),
            ],
            ignore_index=True,
        )
        coordinate_table.to_csv(model_dir / "coordinates.csv.gz", index=False, compression="gzip")
        np.save(model_dir / "coordinates.npy", coordinates)
        joblib.dump(reducer, model_dir / "umap_model.joblib", compress=3)
        metrics = calculate_metrics(chembl_xy, model_xy, args.workers)
        summary_rows.append({
            "model": model,
            "chembl_n": int(len(chembl_xy)),
            "model_n": int(len(model_xy)),
            "model_unique_smiles": int(query["canonical_smiles"].nunique()),
            "joint_n": int(len(coordinates)),
            **metrics,
        })
        pd.DataFrame([summary_rows[-1]]).to_csv(model_dir / "summary.csv", index=False)
        plot_model(model, chembl_xy, model_xy, model_dir)
        (model_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "model": model,
                    "fit_scope": f"ChEMBL100k plus unique {model} molecules",
                    "fit_is_independent_from_other_models": True,
                    "chembl_n": int(len(chembl_xy)),
                    "model_n": int(len(model_xy)),
                    "metric": "cosine",
                    "n_neighbors": args.n_neighbors,
                    "min_dist": args.min_dist,
                    "random_state": args.random_state,
                    "fingerprint": {"type": "ECFP4", "radius": 2, "n_bits": 2048},
                    "interpretation_boundary": "Coordinates are descriptive within this fitted map only; absolute coordinates are not comparable across independent model fits and are not activity or quality scores.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        del reducer, joint_matrix, coordinates
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "independent_umap_summary.csv", index=False)
    plot_facets(all_coordinates, args.output_dir)
    root_manifest = {
        "status": "complete",
        "fit_scope": "five independent fits, each containing the same ChEMBL100k plus one model",
        "models": MODELS,
        "chembl_package": str(args.chembl_package.resolve()),
        "chembl_package_sha256": sha256(args.chembl_package),
        "generated_csv": str(args.generated_csv.resolve()),
        "generated_csv_sha256": sha256(args.generated_csv),
        "parameters": {"metric": "cosine", "n_neighbors": args.n_neighbors, "min_dist": args.min_dist, "random_state": args.random_state, "workers": args.workers},
        "counts": {"chembl": 100_000, "generated_unique": int(len(generated)), "by_model": {model: int((generated["model"] == model).sum()) for model in MODELS}},
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "rdkit": rdBase.rdkitVersion, "umap_learn": umap.__version__, "matplotlib": matplotlib.__version__},
        "elapsed_seconds": time.time() - started,
        "interpretation_boundary": "This is an auxiliary local-topology diagnostic. Independent UMAP axes, orientations, and absolute coordinate distances must not be compared across models. Use original ECFP4-space metrics for cross-model ranking.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(root_manifest, indent=2), encoding="utf-8")
    LOGGER.info("completed five independent fits in %.1fs", root_manifest["elapsed_seconds"])


if __name__ == "__main__":
    main()
