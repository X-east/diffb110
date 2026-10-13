#!/usr/bin/env python3
"""Project additive DiffSBDD/Original repeats into the frozen ChEMBL UMAP."""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/diffb110_original_repeats_mplconfig")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/diffb110_original_repeats_numba")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

RDLogger.DisableLog("rdApp.error")
LOGGER = logging.getLogger("project_original_repeats")
DEFAULT_WORKERS = 28
RADIUS = 2
N_BITS = 2048


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--frozen-umap-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
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
    file_handler = logging.FileHandler(output_dir / "projection.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(stream)
    root.addHandler(file_handler)


def read_valid_molecules(path: Path, repeat: str) -> tuple[pd.DataFrame, list[Any]]:
    """Read one aggregate SDF and retain molecules passing RDKit sanitization."""
    supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
    rows: list[dict[str, Any]] = []
    molecules: list[Any] = []
    for source_index, molecule in enumerate(supplier):
        if molecule is None:
            continue
        try:
            Chem.SanitizeMol(molecule)
            smiles = Chem.MolToSmiles(molecule, canonical=True)
        except Exception:
            continue
        if not smiles:
            continue
        rows.append(
            {
                "repeat": repeat,
                "model": "DiffSBDD",
                "source_sdf": str(path),
                "source_index": source_index,
                "sanitize_success": 1,
                "canonical_smiles": smiles,
            }
        )
        molecules.append(molecule)
    return pd.DataFrame(rows), molecules


def fingerprint_task(task: tuple[int, str]) -> tuple[int, Any | None]:
    """Calculate one ECFP4 fingerprint from canonical SMILES."""
    index, smiles = task
    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return index, None
        return index, AllChem.GetMorganFingerprintAsBitVect(molecule, RADIUS, nBits=N_BITS)
    except Exception:
        return index, None


def fingerprints_to_csr(fingerprints: list[Any]) -> sparse.csr_matrix:
    """Convert RDKit bit vectors to a sparse CSR matrix."""
    indptr = np.zeros(len(fingerprints) + 1, dtype=np.int64)
    for index, fingerprint in enumerate(fingerprints):
        indptr[index + 1] = indptr[index] + fingerprint.GetNumOnBits()
    indices = np.empty(int(indptr[-1]), dtype=np.int32)
    for index, fingerprint in enumerate(fingerprints):
        start, stop = int(indptr[index]), int(indptr[index + 1])
        indices[start:stop] = np.fromiter(
            fingerprint.GetOnBits(), dtype=np.int32, count=stop - start
        )
    data = np.ones(len(indices), dtype=np.float32)
    return sparse.csr_matrix((data, indices, indptr), shape=(len(fingerprints), N_BITS))


def calculate_fingerprints(smiles: list[str], workers: int) -> sparse.csr_matrix:
    """Calculate ECFP4 fingerprints with bounded multiprocessing."""
    tasks = list(enumerate(smiles))
    with mp.Pool(processes=workers) as pool:
        results = pool.map(fingerprint_task, tasks, chunksize=64)
    valid = [fingerprint for _, fingerprint in results if fingerprint is not None]
    if len(valid) != len(smiles):
        raise ValueError(f"fingerprint calculation dropped {len(smiles) - len(valid)} molecules")
    return fingerprints_to_csr(valid)


def load_frozen_reducer(path: Path) -> Any:
    """Load the frozen UMAP across the numba serialization-version change."""
    # The formal map was serialized with an older numba Dispatcher state that
    # contains ``impl_kind``. Current numba ignores this legacy field but its
    # strict rebuild signature rejects it, so adapt the in-memory loader only.
    from numba.core.dispatcher import Dispatcher

    original_rebuild = Dispatcher._rebuild.__func__

    def rebuild_legacy(
        cls: type[Any],
        uuid: Any,
        py_func: Any,
        locals: Any,
        targetoptions: Any,
        can_compile: Any,
        sigs: Any,
        **_: Any,
    ) -> Any:
        return original_rebuild(
            cls, uuid, py_func, locals, targetoptions, can_compile, sigs
        )

    Dispatcher._rebuild = classmethod(rebuild_legacy)
    try:
        return joblib.load(path)
    finally:
        Dispatcher._rebuild = classmethod(original_rebuild)


def classify_hdr(points: np.ndarray, grid_path: Path) -> dict[str, np.ndarray]:
    """Apply the frozen ChEMBL HDR grid to newly projected coordinates."""
    payload = np.load(grid_path)
    density = payload["density"]
    x_edges = payload["x_edges"]
    y_edges = payload["y_edges"]
    flat = density.ravel()
    ordered = np.sort(flat)[::-1]
    cumulative = np.cumsum(ordered) / ordered.sum()
    x_bin = np.searchsorted(x_edges, points[:, 0], side="right") - 1
    y_bin = np.searchsorted(y_edges, points[:, 1], side="right") - 1
    in_bounds = (
        (x_bin >= 0)
        & (x_bin < density.shape[1])
        & (y_bin >= 0)
        & (y_bin < density.shape[0])
    )
    output: dict[str, np.ndarray] = {}
    for mass in (0.50, 0.90, 0.95, 0.99):
        threshold = ordered[min(int(np.searchsorted(cumulative, mass)), len(ordered) - 1)]
        inside = np.zeros(len(points), dtype=bool)
        inside[in_bounds] = density[y_bin[in_bounds], x_bin[in_bounds]] >= threshold
        output[f"inside_hdr_{int(mass * 100)}"] = inside
    return output


def load_new_repeats(run_root: Path, frozen_root: Path, workers: int) -> pd.DataFrame:
    """Read repeat4-6 aggregate SDFs and calculate frozen-map coordinates."""
    rows: list[pd.DataFrame] = []
    fingerprints: list[sparse.csr_matrix] = []
    for repeat in (4, 5, 6):
        repeat_dir = sorted(run_root.glob(f"repeat{repeat}_seed*"))
        if len(repeat_dir) != 1:
            raise FileNotFoundError(f"expected one directory for repeat{repeat}, found {repeat_dir}")
        sdf_path = repeat_dir[0] / "aggregated" / "diffsbdd.sdf"
        if not sdf_path.is_file():
            raise FileNotFoundError(sdf_path)
        frame, _ = read_valid_molecules(sdf_path, f"rep{repeat}")
        LOGGER.info("repeat=%d valid=%d/1000", repeat, len(frame))
        if frame.empty:
            continue
        rows.append(frame)
        fingerprints.append(calculate_fingerprints(frame["canonical_smiles"].tolist(), workers))
    if not rows:
        raise ValueError("no valid molecules found in additive repeats")
    generated = pd.concat(rows, ignore_index=True)
    generated.insert(0, "row_id", np.arange(len(generated), dtype=np.int64))
    matrix = sparse.vstack(fingerprints, format="csr")
    reducer = load_frozen_reducer(frozen_root / "umap_model.joblib")
    LOGGER.info("transforming %d additive molecules with frozen UMAP", len(generated))
    coordinates = reducer.transform(matrix).astype(np.float32)
    generated["umap1"] = coordinates[:, 0]
    generated["umap2"] = coordinates[:, 1]
    return generated


def summarize(
    generated: pd.DataFrame,
    old_metrics_path: Path,
    frozen_root: Path,
    output_dir: Path,
    workers: int,
) -> None:
    """Write per-repeat, pooled, and six-repeat Original summaries."""
    chembl_xy = np.load(frozen_root / "chembl_coordinates.npy", mmap_mode="r")
    neighbor = NearestNeighbors(n_neighbors=1, algorithm="kd_tree", n_jobs=workers)
    neighbor.fit(chembl_xy)
    generated["nearest_chembl_umap_distance"] = neighbor.kneighbors(
        generated[["umap1", "umap2"]].to_numpy(dtype=np.float32), return_distance=True
    )[0][:, 0]
    generated = generated.join(
        pd.DataFrame(classify_hdr(generated[["umap1", "umap2"]].to_numpy(), frozen_root / "chembl_density_grid.npz"), index=generated.index)
    )
    generated.to_csv(output_dir / "additional_original_embedding_metrics.csv.gz", index=False, compression="gzip")

    by_repeat = (
        generated.groupby("repeat")["nearest_chembl_umap_distance"]
        .agg(n="size", mean="mean", median="median", p95=lambda values: values.quantile(0.95))
        .reset_index()
    )
    by_repeat["fixed_attempts"] = 1000
    by_repeat["valid_rate"] = by_repeat["n"] / by_repeat["fixed_attempts"]
    by_repeat.to_csv(output_dir / "additional_original_distance_by_repeat.csv", index=False)

    old = pd.read_csv(old_metrics_path)
    old = old[(old["model"] == "DiffSBDD") & old["sanitize_success"].eq(1)].copy()
    old = old[["row_id", "repeat", "model", "source_sdf", "source_index", "sanitize_success", "canonical_smiles", "umap1", "umap2", "nearest_chembl_umap_distance"]]
    new = generated[old.columns].copy()
    combined = pd.concat([old, new], ignore_index=True)
    combined.to_csv(output_dir / "original_rep1_to_rep6_embedding_metrics.csv.gz", index=False, compression="gzip")
    pooled = (
        combined.groupby(lambda _: "rep1_rep6_pooled")["nearest_chembl_umap_distance"]
        .agg(n="size", mean="mean", median="median", p95=lambda values: values.quantile(0.95))
        .reset_index(drop=True)
    )
    pooled["fixed_attempts"] = 6000
    pooled["valid_rate"] = pooled["n"] / pooled["fixed_attempts"]
    pooled.to_csv(output_dir / "original_rep1_to_rep6_pooled_summary.csv", index=False)

    repeat_table = (
        combined.groupby("repeat")["nearest_chembl_umap_distance"]
        .agg(n="size", mean="mean", median="median", p95=lambda values: values.quantile(0.95))
        .reset_index()
    )
    repeat_table["fixed_attempts"] = 1000
    repeat_table["valid_rate"] = repeat_table["n"] / repeat_table["fixed_attempts"]
    repeat_table.to_csv(output_dir / "original_rep1_to_rep6_by_repeat.csv", index=False)

    comparison = pd.read_csv(frozen_root / "distance_to_chembl_summary.csv")
    original_row = comparison[comparison["model"] == "DiffSBDD"].copy()
    original_row["comparison"] = "formal_rep1_rep3"
    pooled_row = pooled.copy()
    pooled_row["comparison"] = "additive_rep1_rep6_pooled"
    pooled_row = pooled_row.rename(columns={"mean": "mean_mean", "median": "median_mean", "p95": "p95_mean"})
    comparison.to_csv(output_dir / "frozen_benchmark_distance_summary.csv", index=False)
    pd.DataFrame(
        [
            {"comparison": "formal_rep1_rep3", "n": int(old.shape[0]), "fixed_attempts": 3000, "mean": float(old["nearest_chembl_umap_distance"].mean()), "median": float(old["nearest_chembl_umap_distance"].median()), "p95": float(old["nearest_chembl_umap_distance"].quantile(0.95))},
            {"comparison": "additive_rep4_rep6", "n": int(new.shape[0]), "fixed_attempts": 3000, "mean": float(new["nearest_chembl_umap_distance"].mean()), "median": float(new["nearest_chembl_umap_distance"].median()), "p95": float(new["nearest_chembl_umap_distance"].quantile(0.95))},
            {"comparison": "pooled_rep1_rep6", "n": int(combined.shape[0]), "fixed_attempts": 6000, "mean": float(combined["nearest_chembl_umap_distance"].mean()), "median": float(combined["nearest_chembl_umap_distance"].median()), "p95": float(combined["nearest_chembl_umap_distance"].quantile(0.95))},
        ]
    ).to_csv(output_dir / "original_comparison_summary.csv", index=False)


def plot_overlay(generated: pd.DataFrame, old_metrics_path: Path, frozen_root: Path, output_dir: Path) -> None:
    """Create a compact ChEMBL background and additive-repeat overlay figure."""
    chembl_xy = np.load(frozen_root / "chembl_coordinates.npy", mmap_mode="r")
    old = pd.read_csv(old_metrics_path)
    old = old[(old["model"] == "DiffSBDD") & old["sanitize_success"].eq(1)]
    rng = np.random.default_rng(42)
    sample_n = min(200_000, len(chembl_xy))
    sample = rng.choice(len(chembl_xy), size=sample_n, replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    axes[0].scatter(chembl_xy[sample, 0], chembl_xy[sample, 1], s=1, c="#B8B8B8", alpha=0.08, linewidths=0, rasterized=True, label="ChEMBL36")
    axes[0].scatter(old["umap1"], old["umap2"], s=9, c="#777777", alpha=0.35, linewidths=0, label="Original rep1-3")
    colors = {"rep4": "#D95F59", "rep5": "#E69F00", "rep6": "#0072B2"}
    for repeat, group in generated.groupby("repeat"):
        axes[0].scatter(group["umap1"], group["umap2"], s=13, c=colors[repeat], alpha=0.7, linewidths=0, label=f"Original {repeat}")
    axes[0].set_title("Frozen ChEMBL UMAP overlay")
    axes[0].set_xlabel("UMAP1")
    axes[0].set_ylabel("UMAP2")
    axes[0].legend(frameon=False, markerscale=1.5)
    for repeat, group in generated.groupby("repeat"):
        axes[1].hist(group["nearest_chembl_umap_distance"], bins=45, histtype="step", linewidth=2, color=colors[repeat], label=f"Original {repeat}")
    axes[1].hist(old["nearest_chembl_umap_distance"], bins=45, histtype="step", linewidth=2, color="#777777", label="Original rep1-3")
    axes[1].set_title("Nearest-ChEMBL UMAP distance")
    axes[1].set_xlabel("Distance (lower is closer)")
    axes[1].set_ylabel("Molecules")
    axes[1].legend(frameon=False)
    fig.savefig(output_dir / "original_additional_frozen_umap_overlay.png", dpi=220)
    fig.savefig(output_dir / "original_additional_frozen_umap_overlay.svg")
    plt.close(fig)


def main() -> None:
    """Run additive projection and summary generation."""
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    output_dir = args.run_root / "projection"
    configure_logging(output_dir)
    LOGGER.info("command: %s", " ".join(__import__("sys").argv))
    LOGGER.info("workers=%d; frozen_umap=%s", args.workers, args.frozen_umap_root)
    generated = load_new_repeats(args.run_root, args.frozen_umap_root, args.workers)
    old_metrics = args.frozen_umap_root / "generated_embedding_metrics.csv.gz"
    summarize(generated, old_metrics, args.frozen_umap_root, output_dir, args.workers)
    generated = pd.read_csv(output_dir / "additional_original_embedding_metrics.csv.gz")
    plot_overlay(generated, old_metrics, args.frozen_umap_root, output_dir)
    manifest = {
        "status": "complete",
        "protocol": "fixed frozen UMAP transform",
        "repeats": [4, 5, 6],
        "fixed_attempts_per_repeat": 1000,
        "workers": args.workers,
        "fingerprint": {"type": "ECFP4", "radius": RADIUS, "n_bits": N_BITS},
        "frozen_umap_model": str(args.frozen_umap_root / "umap_model.joblib"),
        "interpretation": "UMAP distances are descriptive chemical-space neighborhood measures, not activity or docking evidence.",
    }
    (output_dir / "completion.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("projection completed: %s", output_dir)


if __name__ == "__main__":
    main()
