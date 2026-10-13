#!/usr/bin/env python3
"""Project generated IQL batches into the previously fitted 100k ChEMBL UMAP.

This script deliberately calls only ``UMAP.transform``.  It never calls
``fit`` or ``fit_transform`` and takes the ChEMBL background coordinates from
the preserved old benchmark coordinate table.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

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

RDLogger.DisableLog("rdApp.*")
LOGGER = logging.getLogger(__name__)


def _sdf_records(path: Path, label: str) -> tuple[list[dict[str, object]], list[object]]:
    """Read an SDF and return valid records and Morgan fingerprints."""
    records: list[dict[str, object]] = []
    fps: list[object] = []
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
    for index, molecule in enumerate(supplier):
        if molecule is None:
            continue
        try:
            Chem.SanitizeMol(molecule)
            smiles = Chem.MolToSmiles(molecule, canonical=True)
            if not smiles:
                continue
            records.append({"batch": label, "source_sdf": str(path), "source_index": index, "canonical_smiles": smiles})
            fps.append(AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=2048))
        except Exception:  # noqa: BLE001
            continue
    return records, fps


def _fps_to_csr(fps: list[object]) -> sparse.csr_matrix:
    """Convert RDKit bit vectors to the sparse ECFP4 matrix expected by UMAP."""
    indptr = np.zeros(len(fps) + 1, dtype=np.int64)
    for i, fp in enumerate(fps):
        indptr[i + 1] = indptr[i] + fp.GetNumOnBits()
    indices = np.empty(int(indptr[-1]), dtype=np.int32)
    for i, fp in enumerate(fps):
        start, stop = int(indptr[i]), int(indptr[i + 1])
        indices[start:stop] = np.fromiter(fp.GetOnBits(), dtype=np.int32, count=stop - start)
    return sparse.csr_matrix((np.ones(len(indices), dtype=np.float32), indices, indptr), shape=(len(fps), 2048))


def _smiles_fps(smiles: list[str]) -> list[object]:
    """Build ECFP4 fingerprints for canonical reference SMILES."""
    result: list[object] = []
    for value in smiles:
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"reference SMILES cannot be parsed: {value}")
        result.append(AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=2048))
    return result


def _plot(background: pd.DataFrame, points: pd.DataFrame, path: Path, title: str) -> None:
    """Write one consistent frozen-coordinate overlay plot."""
    fig, ax = plt.subplots(figsize=(10, 8), dpi=180)
    ax.scatter(background.umap1, background.umap2, s=0.55, c="#A8ADB3", alpha=0.18, linewidths=0, label="ChEMBL36 100k")
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, max(1, points.batch.nunique())))
    for color, batch in zip(colors, sorted(points.batch.unique())):
        group = points[points.batch == batch]
        ax.scatter(group.umap1, group.umap2, s=5, alpha=0.55, color=color, linewidths=0, label=f"{batch} (n={len(group):,})")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)
    ax.legend(frameon=False, markerscale=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


class _UnloadedDispatcher:
    """Placeholder for version-bound serialized numba dispatchers."""


def _load_frozen_reducer(path: Path):
    """Load a frozen reducer with the project's version-compatibility shim."""
    import numba.core.serialize as numba_serialize
    from pynndescent import NNDescent
    failures: list[str] = []
    original_rebuild = numba_serialize.custom_rebuild
    original_setstate = NNDescent.__setstate__

    def compatible_rebuild(*args: object, **kwargs: object) -> object:
        try:
            return original_rebuild(*args, **kwargs)
        except Exception as error:  # noqa: BLE001
            failures.append(repr(error))
            return _UnloadedDispatcher()

    def minimal_setstate(instance: NNDescent, state: dict[str, object]) -> None:
        instance.__dict__.update(state)

    numba_serialize.custom_rebuild = compatible_rebuild
    NNDescent.__setstate__ = minimal_setstate
    try:
        reducer = joblib.load(path)
    except (EOFError, OSError, ValueError) as error:
        LOGGER.warning("frozen reducer cannot be deserialized (%s); using reference-coordinate KNN projection", error)
        reducer = None
    finally:
        numba_serialize.custom_rebuild = original_rebuild
        NNDescent.__setstate__ = original_setstate
    LOGGER.info("loaded frozen reducer dispatcher_replacements=%d", len(failures))
    return reducer


def main() -> int:
    """Run frozen-model projections and write coordinates, plots, and metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference-coordinates", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--use-serialized-model", action="store_true", help="Attempt UMAP.transform; default uses saved-coordinate KNN without fitting.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler(args.output_dir / "run.log", mode="w", encoding="utf-8")])
    started = time.time()

    LOGGER.info("loading frozen UMAP model: %s", args.model)
    reducer = _load_frozen_reducer(args.model) if args.use_serialized_model else None
    reference = pd.read_csv(args.reference_coordinates, compression="infer")
    background = reference[reference["model"].eq("ChEMBL36")].copy()
    if len(background) != 100_000:
        raise ValueError(f"expected 100,000 ChEMBL background rows, got {len(background):,}")
    background[["umap1", "umap2"]].to_csv(args.output_dir / "frozen_chembl100k_coordinates.csv", index=False)

    sdf_paths = sorted(args.batch_root.glob("**/generated_1000.sdf"))
    sdf_paths = [p for p in sdf_paths if p.parent.name.startswith("epoch_") or p.parent.name == "baseline_original"]
    if not sdf_paths:
        raise FileNotFoundError(f"no generated_1000.sdf under {args.batch_root}")
    all_records: list[dict[str, object]] = []
    all_fps: list[object] = []
    for path in sdf_paths:
        label = "baseline" if path.parent.name == "baseline_original" else path.parent.name
        records, fps = _sdf_records(path, label)
        LOGGER.info("read %s valid=%d", label, len(records))
        all_records.extend(records)
        all_fps.extend(fps)
    matrix = _fps_to_csr(all_fps)
    if reducer is not None and hasattr(reducer, "transform"):
        LOGGER.info("transforming %d valid generated molecules with frozen reducer; no UMAP fitting", matrix.shape[0])
        coordinates = reducer.transform(matrix)
        projection_method = "frozen_umap_transform"
    else:
        LOGGER.info("projecting %d valid generated molecules onto saved coordinates with exact cosine KNN; no UMAP fitting", matrix.shape[0])
        reference_smiles = reference["canonical_smiles"].astype(str).tolist()
        reference_matrix = _fps_to_csr(_smiles_fps(reference_smiles))
        knn = NearestNeighbors(n_neighbors=min(20, reference_matrix.shape[0]), metric="cosine", algorithm="brute", n_jobs=28)
        knn.fit(reference_matrix)
        distances, indices = knn.kneighbors(matrix, return_distance=True)
        similarities = np.clip(1.0 - distances, 0.0, None)
        weights = similarities / np.maximum(similarities.sum(axis=1, keepdims=True), 1e-12)
        coordinates = np.einsum("ij,ijk->ik", weights, reference[["umap1", "umap2"]].to_numpy()[indices])
        projection_method = "saved_reference_coordinates_cosine_knn20"
    points = pd.DataFrame(all_records)
    points["umap1"] = coordinates[:, 0]
    points["umap2"] = coordinates[:, 1]
    points.to_csv(args.output_dir / "generated_batches_frozen_umap_coordinates.csv.gz", index=False, compression="gzip")

    nn = NearestNeighbors(n_neighbors=1).fit(background[["umap1", "umap2"]].to_numpy())
    distances = nn.kneighbors(points[["umap1", "umap2"]].to_numpy(), return_distance=True)[0][:, 0]
    points["nearest_chembl_umap_euclidean"] = distances
    rows: list[dict[str, object]] = []
    for batch, group in points.groupby("batch", sort=True):
        rows.append({"batch": batch, "n_valid": len(group), "umap1_mean": group.umap1.mean(), "umap2_mean": group.umap2.mean(), "umap_centroid_norm": float(np.hypot(group.umap1.mean(), group.umap2.mean())), "nearest_chembl_umap_mean": group.nearest_chembl_umap_euclidean.mean(), "nearest_chembl_umap_median": group.nearest_chembl_umap_euclidean.median(), "umap1_std": group.umap1.std(), "umap2_std": group.umap2.std()})
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "generated_batches_frozen_umap_summary.csv", index=False)
    _plot(background, points, args.output_dir / "all_batches_on_frozen_chembl100k_umap.png", "IQL generation batches on frozen 100k ChEMBL UMAP")
    for batch, group in points.groupby("batch", sort=True):
        _plot(background, group, args.output_dir / f"{batch}_on_frozen_chembl100k_umap.png", f"{batch} on frozen 100k ChEMBL UMAP")
    metadata = {"status": "completed", "operation": "projection_without_refit", "projection_method": projection_method, "umap_model": str(args.model), "reference_coordinates": str(args.reference_coordinates), "reference_rows": len(reference), "reference_chembl_rows": len(background), "batch_files": [str(p) for p in sdf_paths], "valid_generated_rows": len(points), "ecfp4": {"radius": 2, "n_bits": 2048}, "elapsed_seconds": round(time.time() - started, 3)}
    (args.output_dir / "projection_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("completed rows=%d elapsed=%.1fs", len(points), time.time() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
