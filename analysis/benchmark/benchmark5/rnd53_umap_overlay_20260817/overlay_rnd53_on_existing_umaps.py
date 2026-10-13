#!/usr/bin/env python3
"""Overlay the 53 RND-family EPI references on the existing UMAP maps."""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/diffb110_rnd53_matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/diffb110_rnd53_numba")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
import numba.core.serialize as numba_serialize
from pynndescent import NNDescent
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy import sparse


ROOT = Path(__file__).resolve().parents[4]
OLD_DIR = ROOT / "analysis/benchmark/benchmark5/five_model_chembl100k_umap_20260730"
FULL_DIR = ROOT / "analysis/benchmark/benchmark5/five_model_chemblfull_umap_20260817/results/analysis_v001"
RND_SDF = ROOT / "data/rnd-epi/epi_53_molecules_3d.sdf"
OUTPUT_DIR = Path(__file__).resolve().parent
RECORD_DIR = ROOT / "work_records/26-8/260817"

MODEL_ORDER = ["DiffGui", "DiffSBDD", "PMDM", "GCDM-SBDD", "TargetDiff"]
MODEL_COLORS = {
    "DiffGui": "#3B82C4",
    "DiffSBDD": "#D95F59",
    "PMDM": "#D9A441",
    "GCDM-SBDD": "#8E78B7",
    "TargetDiff": "#3D9B72",
}
CHEMBL_COLOR = "#A8ADB3"
RND_COLOR = "#111111"


class _UnloadedDispatcher:
    """Placeholder for incompatible serialized numba dispatchers."""


def sha256(path: Path) -> str:
    """Return a file SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bitvectors_to_csr(bit_vectors: list[object], n_bits: int = 2048) -> sparse.csr_matrix:
    """Convert RDKit bit vectors to a compact CSR matrix."""
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


def smiles_fingerprints(smiles: list[str], radius: int = 2, n_bits: int = 2048) -> list[object]:
    """Build ECFP4 bit vectors from canonical SMILES."""
    result: list[object] = []
    for value in smiles:
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"cannot parse SMILES: {value}")
        result.append(AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=n_bits))
    return result


def load_rnd() -> tuple[pd.DataFrame, list[object]]:
    """Load and validate the 53-molecule RND-family SDF."""
    rows: list[dict[str, object]] = []
    fingerprints: list[object] = []
    supplier = Chem.SDMolSupplier(str(RND_SDF), removeHs=True, sanitize=True)
    if len(supplier) != 53:
        raise ValueError(f"expected 53 RND molecules, found {len(supplier)}")
    for index, molecule in enumerate(supplier, start=1):
        if molecule is None:
            raise ValueError(f"RND molecule {index} failed RDKit parsing")
        smiles = Chem.MolToSmiles(molecule, canonical=True)
        source_smiles = molecule.GetProp("Canonical_SMILES") if molecule.HasProp("Canonical_SMILES") else ""
        rows.append(
            {
                "rnd_id": f"RND-{index:02d}",
                "sl_no": index,
                "preferred_name": molecule.GetProp("Preferred_Name"),
                "short_name": molecule.GetProp("Short_Name"),
                "target_system": molecule.GetProp("Target_System"),
                "canonical_smiles": smiles,
                "source_canonical_smiles": source_smiles,
                "formal_charge": Chem.GetFormalCharge(molecule),
                "heavy_atoms": molecule.GetNumHeavyAtoms(),
            }
        )
        fingerprints.append(AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048))
    frame = pd.DataFrame(rows)
    if frame["canonical_smiles"].nunique() != 53:
        raise ValueError("RND canonical SMILES are not unique")
    return frame, fingerprints


def load_old_inputs() -> tuple[sparse.csr_matrix, pd.DataFrame, pd.DataFrame]:
    """Load the exact 100k map inputs and its saved coordinates."""
    with gzip.open(ROOT / "data/chembl/chembl_36_100k_ecfp4_fingerprints.pkl.gz", "rb") as handle:
        chembl_payload = pickle.load(handle)
    chembl_matrix = bitvectors_to_csr(chembl_payload["fingerprints"])
    generated_path = ROOT / "analysis/benchmark/benchmark5/8gjl_c1101_benchmark_20260706_1000steps_3rep_formal/raw_quality_vs_chembl_20260728/per_molecule_raw_quality.csv.gz"
    generated = pd.read_csv(
        generated_path,
        usecols=["row_id", "repeat", "model", "source_sdf", "source_index", "sanitize_success", "canonical_smiles"],
    )
    generated = generated[
        generated["sanitize_success"].eq(1)
        & generated["model"].isin(MODEL_ORDER)
        & generated["canonical_smiles"].notna()
    ].copy()
    generated_matrix = bitvectors_to_csr(smiles_fingerprints(generated["canonical_smiles"].astype(str).tolist()))
    coordinate_table = pd.read_csv(OLD_DIR / "joint_umap_coordinates.csv.gz", low_memory=False)
    expected = chembl_matrix.shape[0] + generated_matrix.shape[0]
    if len(coordinate_table) != expected:
        raise ValueError(f"old coordinate row count {len(coordinate_table)} != {expected}")
    return sparse.vstack([chembl_matrix, generated_matrix], format="csr"), generated, coordinate_table


def fit_old_reducer(matrix: sparse.csr_matrix, old_coordinates: pd.DataFrame) -> umap.UMAP:
    """Re-fit the historical 100k UMAP and verify its coordinates before use."""
    cache_path = OUTPUT_DIR / "old100k_reproduced_umap_model.joblib"
    if cache_path.exists():
        return joblib.load(cache_path)
    reducer = umap.UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=20,
        min_dist=0.1,
        random_state=42,
        low_memory=True,
        verbose=False,
    )
    regenerated = reducer.fit_transform(matrix)
    expected = old_coordinates[["umap1", "umap2"]].to_numpy()
    if regenerated.shape != expected.shape:
        raise ValueError("historical UMAP shape mismatch")
    max_abs = float(np.max(np.abs(regenerated - expected)))
    rmse = float(np.sqrt(np.mean((regenerated - expected) ** 2)))
    (OUTPUT_DIR / "old100k_coordinate_reproduction_audit.json").write_text(
        json.dumps({"max_abs_difference": max_abs, "rmse": rmse, "n": len(expected)}, indent=2),
        encoding="utf-8",
    )
    if max_abs > 1e-4:
        raise RuntimeError(
            f"historical 100k UMAP could not be reproduced (max_abs={max_abs:.6g}, rmse={rmse:.6g})"
        )
    joblib.dump(reducer, cache_path, compress=3)
    return reducer


def load_full_reducer_compat() -> tuple[umap.UMAP, int]:
    """Load the old full-map embedding while replacing incompatible numba dispatchers.

    The saved model was produced with Python 3.8/UMAP 0.5.5, whereas this
    workspace uses Python 3.10/UMAP 0.5.12. Its embedding and raw input matrix
    remain serializable. For the 53 query points, transform is intentionally
    switched to exact sparse cosine neighbors, avoiding the incompatible saved
    approximate-search dispatcher.
    """
    failures: list[str] = []
    original_rebuild = numba_serialize.custom_rebuild
    original_setstate = NNDescent.__setstate__

    def compatible_rebuild(*args: object, **kwargs: object) -> object:
        try:
            return original_rebuild(*args, **kwargs)
        except Exception as error:  # Serialized dispatchers are version-bound.
            failures.append(repr(error))
            return _UnloadedDispatcher()

    def minimal_setstate(instance: NNDescent, state: dict[str, object]) -> None:
        instance.__dict__.update(state)

    numba_serialize.custom_rebuild = compatible_rebuild
    NNDescent.__setstate__ = minimal_setstate
    try:
        reducer = joblib.load(FULL_DIR / "umap_model.joblib")
    finally:
        numba_serialize.custom_rebuild = original_rebuild
        NNDescent.__setstate__ = original_setstate

    reducer._small_data = True
    reducer._knn_search_index = object()
    reducer.verbose = False
    return reducer, len(failures)


def validate_full_transfer(reducer: umap.UMAP) -> dict[str, float | int]:
    """Compare compatibility transforms against 20 held-out saved ChEMBL points."""
    landmark_indices = set(np.load(FULL_DIR / "landmark_full_indices.npy").tolist())
    selected: list[dict[str, object]] = []
    for chunk in pd.read_csv(FULL_DIR / "chembl_coordinates.csv.gz", chunksize=10_000):
        rows = chunk[~chunk["full_index"].isin(landmark_indices)]
        selected.extend(rows.head(20 - len(selected)).to_dict("records"))
        if len(selected) == 20:
            break
    if len(selected) != 20:
        raise RuntimeError("could not select 20 non-landmark ChEMBL validation points")
    frame = pd.DataFrame(selected)
    query = bitvectors_to_csr(smiles_fingerprints(frame["canonical_smiles"].astype(str).tolist()))
    predicted = reducer.transform(query)
    expected = frame[["umap1", "umap2"]].to_numpy()
    delta = predicted - expected
    return {
        "n": int(len(frame)),
        "rmse": float(np.sqrt(np.mean(delta ** 2))),
        "median_euclidean_error": float(np.median(np.linalg.norm(delta, axis=1))),
        "max_euclidean_error": float(np.max(np.linalg.norm(delta, axis=1))),
    }


def configure_plotting() -> None:
    """Use the same readable Python plotting style as the source figures."""
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 14,
            "axes.titlesize": 20,
            "axes.labelsize": 17,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: Path) -> None:
    """Export PNG, SVG, PDF and TIFF variants."""
    for suffix, kwargs in [
        ("png", {"dpi": 400}),
        ("svg", {}),
        ("pdf", {}),
        ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
    ]:
        fig.savefig(stem.with_suffix(f".{suffix}"), bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


def plot_old(old: pd.DataFrame, rnd: pd.DataFrame) -> None:
    """Render the 100k facet map with RND markers only in the ChEMBL36 panel."""
    chembl = old[old["model"].eq("ChEMBL36")]
    generated = old[old["dataset"].eq("Generated")]
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 8.2), sharex=True, sharey=True)
    for axis, panel in zip(axes.ravel(), [*MODEL_ORDER, "ChEMBL36"]):
        axis.scatter(chembl.umap1, chembl.umap2, s=0.6, color=CHEMBL_COLOR, alpha=0.14, linewidths=0, rasterized=True)
        if panel != "ChEMBL36":
            group = generated[generated.model.eq(panel)]
            axis.scatter(group.umap1, group.umap2, s=7.5, color=MODEL_COLORS[panel], alpha=0.55, linewidths=0, rasterized=True)
            axis.set_title(f"{panel}  |  n={len(group):,}", fontsize=18)
        else:
            axis.set_title(f"ChEMBL36  |  n={len(chembl):,}", fontsize=18)
        if panel == "ChEMBL36":
            axis.scatter(rnd.umap1, rnd.umap2, s=43, marker="D", facecolor=RND_COLOR, edgecolor="white", linewidth=0.5, alpha=0.95, zorder=8, label="RND-family EPI reference (n=53)")
        axis.set_xlabel("UMAP 1", fontsize=16)
        axis.set_ylabel("UMAP 2", fontsize=16)
        axis.tick_params(labelsize=13)
    fig.suptitle("100k ChEMBL36 chemical-space facets with RND references in the ChEMBL36 panel", fontsize=21)
    fig.text(0.5, 0.008, "Black diamonds are the 53 external RND-family EPI references shown only in the ChEMBL36 panel; UMAP distance is descriptive only.", ha="center", fontsize=13)
    fig.tight_layout(rect=(0, 0.025, 1, 0.965))
    save_figure(fig, OUTPUT_DIR / "five_model_chembl100k_umap_facets_rnd53")


def density_image(axis: plt.Axes, grid: dict[str, np.ndarray]) -> None:
    """Draw the full-ChEMBL density background."""
    density = grid["density"]
    bounds = grid["bounds"]
    positive = density[density > 0]
    axis.imshow(density, origin="lower", extent=bounds, cmap="Greys", norm=mcolors.LogNorm(vmin=max(1.0, float(np.quantile(positive, 0.05))), vmax=float(density.max())), interpolation="nearest", aspect="auto", alpha=0.88, rasterized=True)


def plot_full(generated: pd.DataFrame, rnd: pd.DataFrame) -> None:
    """Render the full-ChEMBL facet map with RND markers only in the ChEMBL36 panel."""
    with np.load(FULL_DIR / "chembl_density_grid.npz") as loaded:
        grid = {key: loaded[key] for key in loaded.files}
    rng = np.random.RandomState(42)
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.8), sharex=True, sharey=True)
    for axis, panel in zip(axes.ravel(), [*MODEL_ORDER, "ChEMBL36"]):
        density_image(axis, grid)
        if panel != "ChEMBL36":
            group = generated[generated.model.eq(panel)]
            axis.scatter(group.umap1, group.umap2, s=8.0, color=MODEL_COLORS[panel], alpha=0.58, linewidths=0, rasterized=True)
            axis.set_title(f"{panel} | n={len(group):,}")
        else:
            axis.set_title("ChEMBL36 | n=1,550,090")
        if panel == "ChEMBL36":
            axis.scatter(rnd.umap1, rnd.umap2, s=43, marker="D", facecolor=RND_COLOR, edgecolor="white", linewidth=0.5, alpha=0.95, zorder=8, label="RND-family EPI reference (n=53)")
        axis.set_xlabel("UMAP 1"); axis.set_ylabel("UMAP 2")
        axis.set_xlim(grid["bounds"][:2]); axis.set_ylim(grid["bounds"][2:])
    fig.suptitle("Full ChEMBL36 chemical-space facets with RND references in the ChEMBL36 panel")
    fig.text(0.5, 0.008, "Black diamonds are the 53 external RND-family EPI references shown only in the ChEMBL36 panel; UMAP distance is descriptive only.", ha="center", fontsize=13)
    fig.tight_layout(rect=(0, 0.025, 1, 0.96))
    save_figure(fig, OUTPUT_DIR / "five_model_chemblfull_umap_facets_rnd53")


def main() -> None:
    """Build coordinates, audit provenance, and render all overlays."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_plotting()
    rnd, rnd_fingerprints = load_rnd()
    old_matrix, generated_old, old_coordinates = load_old_inputs()
    rnd_matrix = bitvectors_to_csr(rnd_fingerprints)
    old_coordinate_path = OUTPUT_DIR / "rnd53_coordinates_chembl100k.csv"
    old_audit_path = OUTPUT_DIR / "old100k_coordinate_reproduction_audit.json"
    if old_coordinate_path.exists() and old_audit_path.exists():
        old_audit = json.loads(old_audit_path.read_text(encoding="utf-8"))
        rnd_old = pd.read_csv(old_coordinate_path)
        if len(rnd_old) != 53 or float(old_audit["max_abs_difference"]) > 1e-4:
            raise RuntimeError("cached 100k RND coordinates failed integrity validation")
    else:
        old_reducer = fit_old_reducer(old_matrix, old_coordinates)
        rnd_old_xy = old_reducer.transform(rnd_matrix)
        rnd_old = rnd.copy()
        rnd_old["umap1"], rnd_old["umap2"] = rnd_old_xy[:, 0], rnd_old_xy[:, 1]
        rnd_old.to_csv(old_coordinate_path, index=False)

    full_reducer, dispatcher_replacements = load_full_reducer_compat()
    full_transfer_validation = validate_full_transfer(full_reducer)
    if full_transfer_validation["median_euclidean_error"] > 0.25:
        raise RuntimeError(
            "full-map compatibility transform did not pass anchor validation: "
            f"median_error={full_transfer_validation['median_euclidean_error']:.4f}"
        )
    rnd_full_xy = full_reducer.transform(rnd_matrix)
    rnd_full = rnd.copy()
    rnd_full["umap1"], rnd_full["umap2"] = rnd_full_xy[:, 0], rnd_full_xy[:, 1]
    rnd_full.to_csv(OUTPUT_DIR / "rnd53_coordinates_chemblfull.csv", index=False)

    old_chembl_smiles = set(old_coordinates.loc[old_coordinates["model"].eq("ChEMBL36"), "canonical_smiles"].astype(str))
    full_chembl = pd.read_csv(FULL_DIR / "chembl_coordinates.csv.gz", usecols=["chembl_id", "canonical_smiles"], low_memory=False)
    full_smiles = set(full_chembl["canonical_smiles"].astype(str))
    generated_smiles = set(generated_old["canonical_smiles"].astype(str))
    audit = {
        "rnd_sdf": str(RND_SDF),
        "rnd_sdf_sha256": sha256(RND_SDF),
        "rnd_count": len(rnd),
        "rnd_unique_canonical_smiles": int(rnd["canonical_smiles"].nunique()),
        "overlap_exact_with_chembl100k": int(rnd["canonical_smiles"].isin(old_chembl_smiles).sum()),
        "overlap_exact_with_chemblfull": int(rnd["canonical_smiles"].isin(full_smiles).sum()),
        "overlap_exact_with_generated": int(rnd["canonical_smiles"].isin(generated_smiles).sum()),
        "full_umap_model": str(FULL_DIR / "umap_model.joblib"),
        "full_umap_model_sha256": sha256(FULL_DIR / "umap_model.joblib"),
        "full_model_numba_dispatcher_replacements": dispatcher_replacements,
        "full_transfer_validation": full_transfer_validation,
        "full_transform_method": "saved embedding plus exact sparse-cosine neighbor transform",
    }
    (OUTPUT_DIR / "rnd53_overlap_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = pd.DataFrame(
        [
            {"map": "chembl100k", "reference_n": 100000, "rnd_n": len(rnd_old), "rnd_umap1_min": rnd_old.umap1.min(), "rnd_umap1_max": rnd_old.umap1.max(), "rnd_umap2_min": rnd_old.umap2.min(), "rnd_umap2_max": rnd_old.umap2.max()},
            {"map": "chemblfull", "reference_n": 1550090, "rnd_n": len(rnd_full), "rnd_umap1_min": rnd_full.umap1.min(), "rnd_umap1_max": rnd_full.umap1.max(), "rnd_umap2_min": rnd_full.umap2.min(), "rnd_umap2_max": rnd_full.umap2.max()},
        ]
    )
    summary.to_csv(OUTPUT_DIR / "rnd53_coordinate_summary.csv", index=False)
    plot_old(old_coordinates, rnd_old)
    generated_full = pd.read_csv(FULL_DIR / "generated_coordinates.csv.gz", low_memory=False)
    plot_full(generated_full, rnd_full)
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    for stem in [
        "five_model_chembl100k_umap_facets_rnd53",
        "five_model_chemblfull_umap_facets_rnd53",
    ]:
        for suffix in ["png", "svg", "pdf", "tiff"]:
            source = OUTPUT_DIR / f"{stem}.{suffix}"
            if suffix == "png":
                (RECORD_DIR / source.name).write_bytes(source.read_bytes())
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
