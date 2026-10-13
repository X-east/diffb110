#!/usr/bin/env python3
"""Fit a ChEMBL100k-only UMAP and transform the completed 500-step scopes.

The ChEMBL reference is fitted once from the fixed 100,000-molecule ECFP4
package.  The generated molecules from first3, first6, and first9 are then
projected with ``transform``; they do not influence the fitted reference map.
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
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/diffb110_matplotlib_260831")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/diffb110_umap_numba_cache_260831")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from rdkit import RDLogger, rdBase
from scipy import sparse
import umap


RDLogger.DisableLog("rdApp.error")

MODEL_ORDER = ["DiffGui", "DiffSBDD", "PMDM", "GCDM-SBDD", "TargetDiff"]
MODEL_COLORS = {
    "DiffGui": "#3B82C4",
    "DiffSBDD": "#D95F59",
    "PMDM": "#D9A441",
    "GCDM-SBDD": "#8E78B7",
    "TargetDiff": "#3D9B72",
}
SCOPE_ORDER = ["first3", "first6", "first9"]
CHEMBL_COLOR = "#A8ADB3"
LOGGER = logging.getLogger("plot_chembl100k_reference")


def configure_logging(output_dir: Path) -> None:
    """Configure terminal and durable chronological logging."""
    output_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "run.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(stream)
    root.addHandler(file_handler)


def configure_plotting() -> None:
    """Set readable Python plotting defaults for the exported figures."""
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 13,
            "axes.titlesize": 17,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
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


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bitvectors_to_csr(bit_vectors: list[Any], n_bits: int) -> sparse.csr_matrix:
    """Convert RDKit bit vectors to a compact binary CSR matrix."""
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
    return sparse.csr_matrix((data, indices, indptr), shape=(len(bit_vectors), n_bits))


def load_chembl(path: Path) -> tuple[sparse.csr_matrix, pd.DataFrame, dict[str, Any]]:
    """Load and validate the fixed ChEMBL36 100k ECFP4 package."""
    with gzip.open(path, "rb") as handle:
        payload = pickle.load(handle)
    required = {
        "fingerprints", "records", "radius", "n_bits", "fingerprint_type",
        "chembl_version", "sampled_from_total", "sampling_seed",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"ChEMBL package missing keys: {sorted(missing)}")
    fingerprints = payload["fingerprints"]
    records = pd.DataFrame(payload["records"])
    if len(fingerprints) != 100_000 or len(records) != 100_000:
        raise ValueError(
            f"expected 100,000 fingerprints/records, found {len(fingerprints):,}/{len(records):,}"
        )
    if payload["radius"] != 2 or payload["n_bits"] != 2048:
        raise ValueError("expected ECFP4 radius=2 and n_bits=2048")
    matrix = bitvectors_to_csr(fingerprints, int(payload["n_bits"]))
    LOGGER.info(
        "loaded ChEMBL36 reference: n=%d shape=%s nnz=%d seed=%s sampled_from=%s",
        len(records), matrix.shape, matrix.nnz, payload["sampling_seed"],
        payload["sampled_from_total"],
    )
    return matrix, records, payload


def load_generated_scope(scope_dir: Path) -> tuple[pd.DataFrame, sparse.csr_matrix]:
    """Load one completed generated-only fit artifact for transformation."""
    table = pd.read_csv(scope_dir / "coordinates.csv.gz")
    matrix = sparse.load_npz(scope_dir / "fit_ecfp4_csr.npz").tocsr()
    if len(table) != matrix.shape[0]:
        raise ValueError(f"{scope_dir}: table/matrix mismatch {len(table)} vs {matrix.shape}")
    if not table["model"].isin(MODEL_ORDER).all():
        raise ValueError(f"{scope_dir}: unexpected model names")
    if not np.isfinite(table[["umap1", "umap2"]].to_numpy(dtype=float)).all():
        raise ValueError(f"{scope_dir}: input coordinates contain non-finite values")
    LOGGER.info("loaded generated scope=%s n=%d matrix=%s", scope_dir.name, len(table), matrix.shape)
    return table, matrix


def save_figure(figure: plt.Figure, stem: Path) -> None:
    """Save a figure as PNG, SVG and PDF."""
    figure.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_original_reference(chembl_xy: np.ndarray, output_dir: Path) -> None:
    """Plot the original ChEMBL-only 100k reference cloud."""
    figure, axis = plt.subplots(figsize=(11.8, 7.0))
    axis.scatter(
        chembl_xy[:, 0], chembl_xy[:, 1], s=0.7, color=CHEMBL_COLOR,
        alpha=0.13, linewidths=0, rasterized=True,
    )
    axis.set_title("Original ChEMBL36 100,000-molecule reference UMAP")
    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2")
    axis.text(
        0.99, 0.02, "fit: ChEMBL36 100k only\nECFP4 · cosine · k=20 · min_dist=0.1",
        transform=axis.transAxes, ha="right", va="bottom", fontsize=12,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
    )
    figure.tight_layout()
    save_figure(figure, output_dir / "chembl100k_original_reference")


def plot_scope_overview(
    chembl_xy: np.ndarray, transformed: dict[str, pd.DataFrame], output_dir: Path
) -> None:
    """Plot all three transformed cumulative scopes over one fixed reference."""
    figure, axes = plt.subplots(1, 3, figsize=(18.0, 6.0), sharex=True, sharey=True)
    rng = np.random.default_rng(42)
    for axis, scope in zip(axes, SCOPE_ORDER):
        axis.scatter(
            chembl_xy[:, 0], chembl_xy[:, 1], s=0.45, color=CHEMBL_COLOR,
            alpha=0.10, linewidths=0, rasterized=True,
            label="ChEMBL36 100k",
        )
        frame = transformed[scope]
        for model in MODEL_ORDER:
            group = frame[frame["model"] == model]
            order = rng.permutation(len(group))
            group = group.iloc[order]
            axis.scatter(
                group["umap1"], group["umap2"], s=6.8, color=MODEL_COLORS[model],
                alpha=0.48, linewidths=0, rasterized=True,
                label=model,
            )
        axis.set_title(f"{scope}  |  n={len(frame):,}")
        axis.set_xlabel("UMAP 1")
        axis.grid(alpha=0.12)
    axes[0].set_ylabel("UMAP 2")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.01))
    figure.suptitle(
        "500-step generated molecules transformed onto one ChEMBL36 100k reference map",
        fontsize=20, y=1.08,
    )
    figure.text(
        0.5, 0.01,
        "UMAP fit uses ChEMBL36 100k only; generated molecules are transform-only. "
        "The reference map can visually compress between-model differences.",
        ha="center", fontsize=11,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.93))
    save_figure(figure, output_dir / "chembl100k_reference_transform_overview")


def plot_reference_facets(
    chembl_xy: np.ndarray, transformed: dict[str, pd.DataFrame], output_dir: Path
) -> None:
    """Plot a ChEMBL-only panel and one overlay panel per cumulative scope."""
    figure, axes = plt.subplots(2, 2, figsize=(14.0, 11.0), sharex=True, sharey=True)
    panels = ["reference", *SCOPE_ORDER]
    for axis, panel in zip(axes.ravel(), panels):
        axis.scatter(
            chembl_xy[:, 0], chembl_xy[:, 1], s=0.5, color=CHEMBL_COLOR,
            alpha=0.10 if panel != "reference" else 0.14,
            linewidths=0, rasterized=True,
        )
        if panel == "reference":
            axis.set_title("ChEMBL36 original | n=100,000")
        else:
            frame = transformed[panel]
            for model in MODEL_ORDER:
                group = frame[frame["model"] == model]
                axis.scatter(
                    group["umap1"], group["umap2"], s=5.8, color=MODEL_COLORS[model],
                    alpha=0.48, linewidths=0, rasterized=True, label=model,
                )
            axis.set_title(f"{panel} transform | n={len(frame):,}")
        axis.set_xlabel("UMAP 1")
        axis.set_ylabel("UMAP 2")
        axis.grid(alpha=0.12)
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=CHEMBL_COLOR,
                   markeredgecolor="none", markersize=7, label="ChEMBL36 100k")
    ]
    handles.extend(
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=MODEL_COLORS[model],
                    markeredgecolor="none", markersize=7, label=model)
        for model in MODEL_ORDER
    )
    figure.legend(handles=handles, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 0.985))
    figure.suptitle(
        "Original ChEMBL36 100k reference and 500-step cumulative transforms",
        fontsize=21, y=1.02,
    )
    figure.text(
        0.5, 0.005,
        "One fixed nonlinear reference coordinate system; descriptive visualization only.",
        ha="center", fontsize=12,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 0.94))
    save_figure(figure, output_dir / "chembl100k_reference_transform_facets")


def write_manifest(
    output_dir: Path,
    chembl_path: Path,
    source_dir: Path,
    chembl_payload: dict[str, Any],
    transformed: dict[str, pd.DataFrame],
) -> None:
    """Write provenance and the transform contract."""
    manifest = {
        "status": "complete",
        "analysis": "500-step cumulative generated molecules on a ChEMBL36 100k-only UMAP",
        "fit_scope": "ChEMBL36 100k only",
        "transform_scopes": SCOPE_ORDER,
        "sources": {
            "chembl_fingerprints": str(chembl_path.resolve()),
            "chembl_fingerprints_sha256": sha256(chembl_path),
            "generated_fit_root": str(source_dir.resolve()),
        },
        "chembl_package": {
            key: chembl_payload[key]
            for key in [
                "radius", "n_bits", "fingerprint_type", "chembl_version",
                "source", "sampled_from_total", "sampling_seed",
            ]
        },
        "umap": {
            "metric": "cosine",
            "n_components": 2,
            "n_neighbors": 20,
            "min_dist": 0.1,
            "random_state": 42,
            "transform_seed": 42,
            "fit_threads": 1,
            "generated_points_in_fit": False,
        },
        "counts": {
            "chembl": 100_000,
            "generated_by_scope": {scope: int(len(transformed[scope])) for scope in SCOPE_ORDER},
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "rdkit": rdBase.rdkitVersion,
            "umap_learn": umap.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "interpretation_boundary": (
            "This is a fixed ChEMBL-reference transform for visual context. It can compress "
            "between-model differences; use the independent generated-only fits and original "
            "ECFP4 metrics for model-distribution separation. UMAP coordinates are not efficacy "
            "or molecular-quality scores."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chembl-fingerprints", type=Path,
        default=root / "data/chembl/chembl_36_100k_ecfp4_fingerprints.pkl.gz",
    )
    parser.add_argument(
        "--source-dir", type=Path,
        default=root / "analysis/benchmark/benchmark5/five_model_500_cumulative_generated_umap_260831",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=root / "analysis/benchmark/benchmark5/chembl100k_reference_500_cumulative_260831",
    )
    return parser.parse_args()


def main() -> int:
    """Fit the reference, transform all scopes, and export the figures."""
    args = parse_args()
    output_dir = args.output_dir.resolve()
    source_dir = args.source_dir.resolve()
    configure_logging(output_dir)
    configure_plotting()
    LOGGER.info("source_dir=%s", source_dir)
    LOGGER.info("output_dir=%s", output_dir)

    chembl_matrix, chembl_records, chembl_payload = load_chembl(args.chembl_fingerprints)
    reducer = umap.UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=20,
        min_dist=0.1,
        random_state=42,
        transform_seed=42,
        low_memory=True,
        n_jobs=1,
        verbose=True,
    )
    LOGGER.info("fitting ChEMBL-only UMAP on %s", chembl_matrix.shape)
    chembl_xy = reducer.fit_transform(chembl_matrix)
    if chembl_xy.shape != (100_000, 2) or not np.isfinite(chembl_xy).all():
        raise ValueError(f"invalid ChEMBL coordinates: shape={chembl_xy.shape}")
    np.save(output_dir / "chembl_coordinates.npy", chembl_xy)
    joblib.dump(reducer, output_dir / "chembl_umap_model.joblib")
    pd.DataFrame(
        {
            "chembl_id": chembl_records["chembl_id"].astype(str),
            "canonical_smiles": chembl_records["canonical_smiles"].astype(str),
            "umap1": chembl_xy[:, 0],
            "umap2": chembl_xy[:, 1],
        }
    ).to_csv(output_dir / "chembl_coordinates.csv.gz", index=False, compression="gzip")

    transformed: dict[str, pd.DataFrame] = {}
    for scope in SCOPE_ORDER:
        table, matrix = load_generated_scope(source_dir / scope)
        coordinates = reducer.transform(matrix)
        if coordinates.shape != (len(table), 2) or not np.isfinite(coordinates).all():
            raise ValueError(f"invalid transformed coordinates for {scope}: {coordinates.shape}")
        transformed_table = table.copy()
        transformed_table["umap1"] = coordinates[:, 0]
        transformed_table["umap2"] = coordinates[:, 1]
        transformed[scope] = transformed_table
        scope_dir = output_dir / scope
        scope_dir.mkdir(parents=True, exist_ok=True)
        np.save(scope_dir / "coordinates.npy", coordinates)
        transformed_table.to_csv(scope_dir / "coordinates.csv.gz", index=False, compression="gzip")
        LOGGER.info("transformed scope=%s n=%d", scope, len(transformed_table))

    plot_original_reference(chembl_xy, output_dir)
    plot_scope_overview(chembl_xy, transformed, output_dir)
    plot_reference_facets(chembl_xy, transformed, output_dir)
    write_manifest(output_dir, args.chembl_fingerprints, source_dir, chembl_payload, transformed)
    (output_dir / "completion.json").write_text(
        json.dumps({"status": "complete", "stage": "all"}, indent=2) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
