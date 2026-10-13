#!/usr/bin/env python3
"""Fit three ChEMBL100k + five-model joint UMAP maps for 500-step data."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import pickle
import platform
import time
from pathlib import Path
from typing import Any

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/diffb110_matplotlib_260831")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/diffb110_umap_numba_cache_260831")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
LOGGER = logging.getLogger("fit_joint_umap")


def configure_logging(output_dir: Path) -> None:
    """Configure terminal and durable run logging."""
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
    """Set readable plotting defaults for exported figures."""
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 13,
            "axes.titlesize": 17,
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
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
    """Return a file SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bitvectors_to_csr(bit_vectors: list[Any], n_bits: int) -> sparse.csr_matrix:
    """Convert RDKit bit vectors into a binary CSR matrix."""
    indptr = np.zeros(len(bit_vectors) + 1, dtype=np.int64)
    for index, bit_vector in enumerate(bit_vectors):
        indptr[index + 1] = indptr[index] + bit_vector.GetNumOnBits()
    indices = np.empty(int(indptr[-1]), dtype=np.int32)
    for index, bit_vector in enumerate(bit_vectors):
        start, stop = int(indptr[index]), int(indptr[index + 1])
        indices[start:stop] = np.fromiter(
            bit_vector.GetOnBits(), dtype=np.int32, count=stop - start
        )
    return sparse.csr_matrix(
        (np.ones(len(indices), dtype=np.float32), indices, indptr),
        shape=(len(bit_vectors), n_bits),
    )


def load_chembl(path: Path) -> tuple[sparse.csr_matrix, pd.DataFrame, dict[str, Any]]:
    """Load the fixed ChEMBL36 100k ECFP4 package."""
    with gzip.open(path, "rb") as handle:
        payload = pickle.load(handle)
    required = {
        "fingerprints", "records", "radius", "n_bits", "fingerprint_type",
        "chembl_version", "sampled_from_total", "sampling_seed",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"ChEMBL package missing keys: {sorted(missing)}")
    records = pd.DataFrame(payload["records"])
    fingerprints = payload["fingerprints"]
    if len(records) != 100_000 or len(fingerprints) != 100_000:
        raise ValueError(f"expected 100,000 ChEMBL rows, found {len(records):,}/{len(fingerprints):,}")
    if payload["radius"] != 2 or payload["n_bits"] != 2048:
        raise ValueError("expected ECFP4 radius=2 and n_bits=2048")
    if records[["chembl_id", "canonical_smiles"]].isna().any().any():
        raise ValueError("ChEMBL records contain missing identities")
    matrix = bitvectors_to_csr(fingerprints, 2048)
    LOGGER.info("ChEMBL reference loaded: rows=%d matrix=%s nnz=%d", len(records), matrix.shape, matrix.nnz)
    return matrix, records, payload


def load_scope(source_dir: Path, scope: str) -> tuple[pd.DataFrame, sparse.csr_matrix]:
    """Load one generated-only cumulative scope and its existing ECFP4 matrix."""
    scope_dir = source_dir / scope
    table = pd.read_csv(scope_dir / "coordinates.csv.gz")
    matrix = sparse.load_npz(scope_dir / "fit_ecfp4_csr.npz").tocsr()
    if len(table) != matrix.shape[0]:
        raise ValueError(f"{scope}: table/matrix mismatch {len(table)} vs {matrix.shape}")
    if not table["model"].isin(MODEL_ORDER).all():
        raise ValueError(f"{scope}: unexpected model name")
    if table["canonical_smiles"].isna().any():
        raise ValueError(f"{scope}: missing canonical SMILES")
    expected = {model: int((table["model"] == model).sum()) for model in MODEL_ORDER}
    LOGGER.info("generated scope=%s rows=%d by_model=%s", scope, len(table), expected)
    return table, matrix


def save_figure(figure: plt.Figure, stem: Path) -> None:
    """Save one figure in PNG, SVG and PDF formats."""
    figure.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def draw_chembl(axis: plt.Axes, coordinates: np.ndarray, alpha: float = 0.10) -> None:
    """Draw the 100k ChEMBL background cloud."""
    axis.scatter(
        coordinates[:, 0], coordinates[:, 1], s=0.48, color=CHEMBL_COLOR,
        alpha=alpha, linewidths=0, rasterized=True,
    )


def plot_scope_facets(
    scope: str, chembl_xy: np.ndarray, generated: pd.DataFrame, output_dir: Path
) -> None:
    """Plot ChEMBL and each model in one shared coordinate system for a scope."""
    figure, axes = plt.subplots(2, 3, figsize=(14.0, 10.0), sharex=True, sharey=True)
    for axis, panel in zip(axes.ravel(), [*MODEL_ORDER, "ChEMBL36"]):
        draw_chembl(axis, chembl_xy, alpha=0.14 if panel == "ChEMBL36" else 0.09)
        if panel == "ChEMBL36":
            axis.set_title("ChEMBL36 | n=100,000")
        else:
            group = generated[generated["model"] == panel]
            axis.scatter(
                group["umap1"], group["umap2"], s=6.4, color=MODEL_COLORS[panel],
                alpha=0.53, linewidths=0, rasterized=True,
            )
            axis.set_title(f"{panel} | n={len(group):,}")
        axis.set_xlabel("UMAP 1")
        axis.set_ylabel("UMAP 2")
        axis.grid(alpha=0.12)
    figure.suptitle(
        f"500-step {scope}: ChEMBL36 100k + five-model joint UMAP fit",
        fontsize=21, y=1.01,
    )
    figure.text(
        0.5, 0.005,
        "All ChEMBL and generated points participate in this scope-specific fit; coordinates are descriptive only.",
        ha="center", fontsize=11,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 0.95))
    save_figure(figure, output_dir / "joint_chembl100k_umap_facets")


def plot_scope_overview(
    chembl_by_scope: dict[str, np.ndarray],
    generated_by_scope: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """Plot the three scope-specific joint maps side by side."""
    figure, axes = plt.subplots(1, 3, figsize=(18.0, 6.2))
    for axis, scope in zip(axes, SCOPE_ORDER):
        chembl_xy = chembl_by_scope[scope]
        generated = generated_by_scope[scope]
        draw_chembl(axis, chembl_xy, alpha=0.10)
        for model in MODEL_ORDER:
            group = generated[generated["model"] == model]
            axis.scatter(
                group["umap1"], group["umap2"], s=5.8, color=MODEL_COLORS[model],
                alpha=0.48, linewidths=0, rasterized=True, label=model,
            )
        axis.set_title(f"{scope} | ChEMBL + generated n={len(generated) + 100_000:,}")
        axis.set_xlabel("UMAP 1")
        axis.grid(alpha=0.12)
    axes[0].set_ylabel("UMAP 2")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.01))
    figure.suptitle(
        "500-step cumulative ChEMBL100k + five-model joint UMAP fits",
        fontsize=20, y=1.08,
    )
    figure.text(
        0.5, 0.01,
        "Each panel is one independent single joint fit; absolute UMAP coordinates are not comparable across panels.",
        ha="center", fontsize=11,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.94))
    save_figure(figure, output_dir / "cumulative_joint_chembl100k_umap_overview")


def group_summary(scope: str, chembl_xy: np.ndarray, generated: pd.DataFrame) -> pd.DataFrame:
    """Summarize counts and within-group 2D spread for one joint map."""
    rows: list[dict[str, Any]] = []
    reference = pd.DataFrame({"model": "ChEMBL36", "umap1": chembl_xy[:, 0], "umap2": chembl_xy[:, 1]})
    combined = pd.concat([reference, generated[["model", "umap1", "umap2"]]], ignore_index=True)
    for model in ["ChEMBL36", *MODEL_ORDER]:
        group = combined[combined["model"] == model]
        xy = group[["umap1", "umap2"]].to_numpy(dtype=float)
        centroid = xy.mean(axis=0)
        rows.append(
            {
                "scope": scope,
                "group": model,
                "n": len(group),
                "centroid_umap1": centroid[0],
                "centroid_umap2": centroid[1],
                "within_group_rms_radius": float(np.sqrt(np.mean(np.sum((xy - centroid) ** 2, axis=1)))),
            }
        )
    return pd.DataFrame(rows)


def write_scope_artifacts(
    scope: str,
    output_dir: Path,
    chembl_records: pd.DataFrame,
    chembl_xy: np.ndarray,
    generated: pd.DataFrame,
) -> None:
    """Write coordinates, reducer and completion marker for one scope."""
    scope_dir = output_dir / scope
    scope_dir.mkdir(parents=True, exist_ok=True)
    np.save(scope_dir / "chembl_coordinates.npy", chembl_xy.astype(np.float32))
    chembl_out = pd.DataFrame(
        {
            "dataset": "ChEMBL36 100k",
            "model": "ChEMBL36",
            "repeat": pd.NA,
            "molecule_id": chembl_records["chembl_id"].astype(str),
            "canonical_smiles": chembl_records["canonical_smiles"].astype(str),
            "source_sdf": pd.NA,
            "source_index": pd.NA,
            "umap1": chembl_xy[:, 0],
            "umap2": chembl_xy[:, 1],
        }
    )
    generated_out = generated.copy()
    generated_out.insert(0, "dataset", "Generated")
    coordinates = pd.concat([chembl_out, generated_out], ignore_index=True, sort=False)
    coordinates.to_csv(scope_dir / "joint_umap_coordinates.csv.gz", index=False, compression="gzip")
    summary = group_summary(scope, chembl_xy, generated)
    summary.to_csv(scope_dir / "group_summary.csv", index=False)
    (scope_dir / "completion.json").write_text(
        json.dumps({"status": "complete", "stage": "fit"}, indent=2) + "\n", encoding="utf-8"
    )


def write_manifest(
    output_dir: Path,
    source_dir: Path,
    chembl_path: Path,
    chembl_payload: dict[str, Any],
    generated_by_scope: dict[str, pd.DataFrame],
    source_manifest_hash: str | None,
) -> None:
    """Write reproducibility metadata for all three fits."""
    manifest = {
        "status": "complete",
        "analysis": "500-step cumulative ChEMBL36 100k plus five-model joint UMAP",
        "fit_contract": "one fit_transform per scope on ChEMBL100k + all five valid generated groups",
        "scopes": SCOPE_ORDER,
        "source_dir": str(source_dir),
        "source_manifest_sha256": source_manifest_hash,
        "chembl": {
            "path": str(chembl_path),
            "sha256": sha256(chembl_path),
            "count": 100_000,
            "radius": chembl_payload["radius"],
            "n_bits": chembl_payload["n_bits"],
            "fingerprint_type": chembl_payload["fingerprint_type"],
            "chembl_version": chembl_payload["chembl_version"],
            "sampling_seed": chembl_payload["sampling_seed"],
            "sampled_from_total": chembl_payload["sampled_from_total"],
        },
        "fingerprint": {"type": "ECFP4", "radius": 2, "n_bits": 2048},
        "umap": {
            "metric": "cosine",
            "n_components": 2,
            "n_neighbors": 20,
            "min_dist": 0.1,
            "random_state": 42,
            "transform_seed": 42,
            "fit_threads": 1,
            "generated_in_fit": True,
            "generated_transform_only": False,
        },
        "counts": {
            scope: {
                "chembl": 100_000,
                "generated": int(len(generated_by_scope[scope])),
                "total": int(len(generated_by_scope[scope]) + 100_000),
                "by_model": {
                    model: int((generated_by_scope[scope]["model"] == model).sum())
                    for model in MODEL_ORDER
                },
            }
            for scope in SCOPE_ORDER
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
            "Each scope is an independent nonlinear joint map. ChEMBL dominates row counts and may "
            "compress between-model differences; coordinates are descriptive and not efficacy, docking, "
            "activity, or molecular-quality scores."
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
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--scope", choices=["all", *SCOPE_ORDER], default="all")
    return parser.parse_args()


def main() -> int:
    """Run the three scope-specific ChEMBL100k joint fits."""
    args = parse_args()
    output_dir = args.output_dir.resolve()
    source_dir = args.source_dir.resolve()
    configure_logging(output_dir)
    configure_plotting()
    LOGGER.info("source_dir=%s", source_dir)
    LOGGER.info("output_dir=%s", output_dir)
    chembl_matrix, chembl_records, chembl_payload = load_chembl(args.chembl_fingerprints.resolve())
    scopes = SCOPE_ORDER if args.scope == "all" else [args.scope]
    generated_by_scope: dict[str, pd.DataFrame] = {}
    chembl_by_scope: dict[str, np.ndarray] = {}
    summaries: list[pd.DataFrame] = []
    for scope in scopes:
        generated, generated_matrix = load_scope(source_dir, scope)
        joint_matrix = sparse.vstack([chembl_matrix, generated_matrix], format="csr")
        LOGGER.info("scope=%s joint matrix=%s nnz=%d", scope, joint_matrix.shape, joint_matrix.nnz)
        started = time.time()
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
        coordinates = reducer.fit_transform(joint_matrix).astype(np.float32)
        if coordinates.shape != (100_000 + len(generated), 2) or not np.isfinite(coordinates).all():
            raise ValueError(f"{scope}: invalid joint coordinates {coordinates.shape}")
        LOGGER.info("scope=%s fit_transform complete; saving coordinates", scope)
        chembl_xy = coordinates[:100_000]
        generated_out = generated.copy()
        generated_out["umap1"] = coordinates[100_000:, 0]
        generated_out["umap2"] = coordinates[100_000:, 1]
        write_scope_artifacts(scope, output_dir, chembl_records, chembl_xy, generated_out)
        plot_scope_facets(scope, chembl_xy, generated_out, output_dir / scope)
        generated_by_scope[scope] = generated_out
        chembl_by_scope[scope] = chembl_xy
        summaries.append(group_summary(scope, chembl_xy, generated_out))
        LOGGER.info("scope=%s complete elapsed=%.1fs", scope, time.time() - started)
    if args.scope == "all":
        plot_scope_overview(chembl_by_scope, generated_by_scope, output_dir)
        pd.concat(summaries, ignore_index=True).to_csv(output_dir / "joint_group_summary.csv", index=False)
        source_manifest = source_dir / "manifest.json"
        source_manifest_hash = sha256(source_manifest) if source_manifest.exists() else None
        write_manifest(
            output_dir, source_dir, args.chembl_fingerprints.resolve(), chembl_payload,
            generated_by_scope, source_manifest_hash,
        )
        (output_dir / "completion.json").write_text(
            json.dumps({"status": "complete", "stage": "all"}, indent=2) + "\n", encoding="utf-8"
        )
        LOGGER.info("all three joint fits complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
