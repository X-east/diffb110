#!/usr/bin/env python3
"""Unify completed 500/1000-step five-model benchmark evidence.

The script only reads source SDFs and existing evaluation artifacts.  It writes
new additive analysis artifacts under the requested output directory.
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
import pickle
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, QED, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


RDLogger.DisableLog("rdApp.*")
LOGGER = logging.getLogger("completed_comparison")

ROOT = Path("/data/disk6/DiffB110")
OUT = ROOT / "analysis/benchmark/benchmark5/five_model_completed_comparison_260827"
MODEL_ORDER = ["DiffGui", "DiffSBDD", "PMDM", "GCDM-SBDD", "TargetDiff"]
MODEL_COLORS = {
    "DiffGui": "#3B82C4",
    "DiffSBDD": "#D95F59",
    "PMDM": "#D9A441",
    "GCDM-SBDD": "#8E78B7",
    "TargetDiff": "#3D9B72",
}
MODEL_SDF_NAMES = {
    "DiffGui": "diffgui.sdf",
    "DiffSBDD": "diffsbdd.sdf",
    "PMDM": "pmdm.sdf",
    "GCDM-SBDD": "gcdm.sdf",
    "TargetDiff": "targetdiff.sdf",
}
STEPS = [500, 1000]
ATTEMPTS = 1000

REFERENCE_ECFP4 = ROOT / "data/chembl/chembl_36_100k_ecfp4_fingerprints.pkl.gz"
REFERENCE_FCFP4 = ROOT / "data/chembl/chembl_36_100k_fcfp4_fingerprints.pkl.gz"
FULL_UMAP_DIR = ROOT / "analysis/benchmark/benchmark5/five_model_chemblfull_umap_20260817/results/analysis_v001"
FULL_UMAP_COORDS = FULL_UMAP_DIR / "chembl_coordinates.npy"
FULL_UMAP_DENSITY = FULL_UMAP_DIR / "chembl_density_grid.npz"
LANDMARK_COORDS = FULL_UMAP_DIR / "landmark_coordinates.npy"

REFERENCE_FPS: list[Any] = []


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def setup_logging(output_dir: Path) -> None:
    """Configure terminal and durable logging."""
    output_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "run.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(stream)
    root.addHandler(file_handler)


def source_specs() -> list[dict[str, Any]]:
    """Return the frozen set of completed source SDFs."""
    specs: list[dict[str, Any]] = []
    five_roots = {
        "rep4": ROOT / "analysis/benchmark/benchmark5/five_model_repeats_500steps_260826/rep4_seed22_29",
        "rep5": ROOT / "analysis/benchmark/benchmark5/five_model_repeats_500steps_260826/rep5_seed30_37",
        "rep6": ROOT / "analysis/benchmark/benchmark5/five_model_repeats_500steps_260826/rep6_seed38_45",
        "rep7": ROOT / "analysis/benchmark/benchmark5/five_model_repeats_500steps_260826_cont1/rep7_seed46_53",
        "rep8": ROOT / "analysis/benchmark/benchmark5/five_model_repeats_500steps_260826_cont1/rep8_seed54_61",
    }
    for repeat, root in five_roots.items():
        for model in MODEL_ORDER:
            specs.append({"step": 500, "repeat": repeat, "model": model, "sdf": root / "aggregated" / MODEL_SDF_NAMES[model], "source_note": "500-step completed repeat"})

    formal_sources = ROOT / "analysis/benchmark/8gjl_c1101_benchmark_20260706_1000steps_3rep_formal/formal_3rep_sources.csv"
    with formal_sources.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            specs.append({"step": 1000, "repeat": row["repeat"], "model": row["model"], "sdf": Path(row["source_sdf"]), "source_note": row.get("source_note", "1000-step formal source")})
    if len(specs) != 40:
        raise ValueError(f"expected 40 completed source SDFs, found {len(specs)}")
    missing = [str(row["sdf"]) for row in specs if not Path(row["sdf"]).exists()]
    if missing:
        raise FileNotFoundError("missing source SDFs:\n" + "\n".join(missing))
    return specs


def safe_mol_properties(mol: Chem.Mol) -> dict[str, Any]:
    """Extract valid-molecule descriptors from the canonical molecule object."""
    clean = Chem.RemoveHs(mol)
    heavy = clean.GetNumHeavyAtoms()
    carbon = sum(atom.GetAtomicNum() == 6 for atom in clean.GetAtoms())
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=clean, includeChirality=False)
    except Exception:
        scaffold = ""
    return {
        "canonical_smiles": Chem.MolToSmiles(clean, canonical=True, isomericSmiles=False),
        "heavy_atoms": int(heavy),
        "hetero_atoms": int(heavy - carbon),
        "carbon_fraction": float(carbon / heavy) if heavy else 0.0,
        "ring_count": int(rdMolDescriptors.CalcNumRings(clean)),
        "aromatic_ring_count": int(rdMolDescriptors.CalcNumAromaticRings(clean)),
        "has_aromatic_ring": int(rdMolDescriptors.CalcNumAromaticRings(clean) > 0),
        "three_member_ring_count": int(sum(len(ring) == 3 for ring in Chem.GetSymmSSSR(clean))),
        "has_three_member_ring": int(any(len(ring) == 3 for ring in Chem.GetSymmSSSR(clean))),
        "rotatable_bonds": int(rdMolDescriptors.CalcNumRotatableBonds(clean)),
        "qed": float(QED.qed(clean)),
        "logp": float(Descriptors.MolLogP(clean)),
        "tpsa": float(Descriptors.TPSA(clean)),
        "hba": int(Descriptors.NumHAcceptors(clean)),
        "hbd": int(Descriptors.NumHDonors(clean)),
        "mw": float(Descriptors.MolWt(clean)),
        "num_fragments": int(len(Chem.GetMolFrags(clean))),
        "murcko_scaffold": scaffold,
    }


def read_source(spec: dict[str, Any], row_id_start: int) -> tuple[list[dict[str, Any]], int]:
    """Read one source SDF without changing the source file."""
    sdf_path = Path(spec["sdf"])
    rows: list[dict[str, Any]] = []
    supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False)
    for source_index, raw_mol in enumerate(supplier):
        row: dict[str, Any] = {
            "row_id": row_id_start + source_index,
            "step": int(spec["step"]),
            "repeat": spec["repeat"],
            "model": spec["model"],
            "source_sdf": str(sdf_path),
            "source_index": source_index,
            "rdkit_readable": int(raw_mol is not None),
            "sanitize_valid": 0,
            "canonical_smiles": "",
            "murcko_scaffold": "",
        }
        if raw_mol is not None:
            try:
                mol = Chem.Mol(raw_mol)
                Chem.SanitizeMol(mol)
                row.update(safe_mol_properties(mol))
                row["sanitize_valid"] = 1
            except Exception as exc:  # noqa: BLE001
                row["failure_reason"] = type(exc).__name__
        rows.append(row)
    return rows, row_id_start + len(rows)


def collect() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Collect per-record data, per-repeat summaries, and source manifest."""
    specs = source_specs()
    all_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    row_id = 0
    for spec in specs:
        rows, row_id = read_source(spec, row_id)
        all_rows.extend(rows)
        valid = [row for row in rows if row["sanitize_valid"]]
        smiles = [row["canonical_smiles"] for row in valid]
        scaffolds = [row["murcko_scaffold"] for row in valid if row["murcko_scaffold"]]
        source_rows.append({
            "step": spec["step"], "repeat": spec["repeat"], "model": spec["model"],
            "attempts": ATTEMPTS, "sdf_written": len(rows),
            "rdkit_readable": sum(row["rdkit_readable"] for row in rows),
            "sanitize_valid": len(valid), "unique_smiles": len(set(smiles)),
            "unique_scaffolds": len(set(scaffolds)),
            "valid_per_attempt": len(valid) / ATTEMPTS,
            "unique_per_attempt": len(set(smiles)) / ATTEMPTS,
            "unique_rate_valid": len(set(smiles)) / len(valid) if valid else 0.0,
            "source_sdf": str(spec["sdf"]), "source_note": spec["source_note"],
        })
        LOGGER.info("collected step=%d %s %s written=%d valid=%d", spec["step"], spec["repeat"], spec["model"], len(rows), len(valid))
    molecule_frame = pd.DataFrame(all_rows)
    source_frame = pd.DataFrame(source_rows)
    valid = molecule_frame[molecule_frame["sanitize_valid"].eq(1)].copy()
    descriptor_cols = ["heavy_atoms", "hetero_atoms", "carbon_fraction", "ring_count", "aromatic_ring_count", "three_member_ring_count", "rotatable_bonds", "qed", "logp", "tpsa", "hba", "hbd", "mw", "num_fragments"]
    descriptor_frame = valid.groupby(["step", "repeat", "model"], as_index=False)[descriptor_cols].mean()
    source_frame.to_csv(OUT / "per_repeat_summary.csv", index=False)
    molecule_frame.to_csv(OUT / "per_record_quality.csv.gz", index=False, compression="gzip")
    descriptor_frame.to_csv(OUT / "per_repeat_descriptor_means.csv", index=False)
    return molecule_frame, source_frame, descriptor_frame


def load_collected() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load collection artifacts."""
    return (
        pd.read_csv(OUT / "per_record_quality.csv.gz"),
        pd.read_csv(OUT / "per_repeat_summary.csv"),
        pd.read_csv(OUT / "per_repeat_descriptor_means.csv"),
    )


def bitvectors_to_csr(fingerprints: list[Any], n_bits: int = 2048) -> sparse.csr_matrix:
    """Convert RDKit bit vectors to a sparse binary matrix."""
    indptr = np.zeros(len(fingerprints) + 1, dtype=np.int64)
    for index, fp in enumerate(fingerprints):
        indptr[index + 1] = indptr[index] + fp.GetNumOnBits()
    indices = np.empty(int(indptr[-1]), dtype=np.int32)
    for index, fp in enumerate(fingerprints):
        start, stop = int(indptr[index]), int(indptr[index + 1])
        indices[start:stop] = np.fromiter(fp.GetOnBits(), dtype=np.int32, count=stop - start)
    return sparse.csr_matrix((np.ones(len(indices), dtype=np.float32), indices, indptr), shape=(len(fingerprints), n_bits))


def load_cache(path: Path) -> dict[str, Any]:
    """Load and validate one ChEMBL fingerprint package."""
    with gzip.open(path, "rb") as handle:
        payload = pickle.load(handle)
    if len(payload["fingerprints"]) != 100000 or payload["radius"] != 2 or payload["n_bits"] != 2048:
        raise ValueError(f"unexpected ChEMBL cache: {path}")
    return payload


def fp_for_smiles(smiles: str, use_features: bool) -> Any:
    """Build one ECFP4 or FCFP4 fingerprint."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"cannot parse query SMILES: {smiles}")
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, useFeatures=use_features)


def init_similarity_workers(fingerprints: list[Any]) -> None:
    """Install fork-shared reference fingerprints."""
    global REFERENCE_FPS
    REFERENCE_FPS = fingerprints


def similarity_task(task: tuple[int, list[int]]) -> tuple[int, float, float, float]:
    """Compute exact top-1, top-5 mean, and top-10 mean similarity."""
    query_index, on_bits = task
    query = DataStructs.ExplicitBitVect(2048)
    query.SetBitsFromList(on_bits)
    similarities = DataStructs.BulkTanimotoSimilarity(query, REFERENCE_FPS)
    ordered = np.sort(np.asarray(similarities, dtype=np.float32))[::-1]
    return query_index, float(ordered[0]), float(ordered[:5].mean()), float(ordered[:10].mean())


def compute_similarity_for_type(valid: pd.DataFrame, cache_path: Path, label: str, workers: int) -> pd.DataFrame:
    """Compute restart-safe exact similarities for unique query SMILES."""
    cache = load_cache(cache_path)
    unique = valid[["canonical_smiles"]].drop_duplicates().reset_index(drop=True)
    chunk_dir = OUT / f"similarity_chunks/{label.lower()}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_size = 100
    completed: list[pd.DataFrame] = []
    reference_fps = list(cache["fingerprints"])
    LOGGER.info("similarity label=%s reference=%d unique_queries=%d workers=%d", label, len(reference_fps), len(unique), workers)
    context = mp.get_context("fork")
    with context.Pool(workers, initializer=init_similarity_workers, initargs=(reference_fps,)) as pool:
        for chunk_index, start in enumerate(range(0, len(unique), chunk_size)):
            stop = min(start + chunk_size, len(unique))
            path = chunk_dir / f"chunk_{chunk_index:05d}_{start}_{stop}.csv.gz"
            if path.exists():
                part = pd.read_csv(path)
                if len(part) == stop - start:
                    completed.append(part)
                    continue
                path.unlink()
            tasks = []
            for query_index, smiles in enumerate(unique.loc[start:stop - 1, "canonical_smiles"], start=start):
                fp = fp_for_smiles(str(smiles), label == "FCFP4")
                tasks.append((query_index, [int(bit) for bit in fp.GetOnBits()]))
            values = pool.map(similarity_task, tasks, chunksize=1)
            part = pd.DataFrame(values, columns=["query_index", f"{label}_top1", f"{label}_top5_mean", f"{label}_top10_mean"])
            part["canonical_smiles"] = unique.loc[start:stop - 1, "canonical_smiles"].to_numpy()
            part.to_csv(path, index=False, compression="gzip")
            completed.append(part)
            LOGGER.info("similarity label=%s complete=%d/%d", label, stop, len(unique))
    result = pd.concat(completed, ignore_index=True).sort_values("query_index")
    result.to_csv(OUT / f"unique_similarity_{label.lower()}.csv.gz", index=False, compression="gzip")
    return result


def similarity(workers: int) -> None:
    """Calculate ECFP4 and FCFP4 ChEMBL similarities."""
    molecules, _, _ = load_collected()
    valid = molecules[molecules["sanitize_valid"].eq(1)].copy()
    ecfp = compute_similarity_for_type(valid, REFERENCE_ECFP4, "ECFP4", workers)
    fcfp = compute_similarity_for_type(valid, REFERENCE_FCFP4, "FCFP4", workers)
    joined = ecfp.merge(fcfp, on=["query_index", "canonical_smiles"], how="inner")
    valid = valid.merge(joined.drop(columns=["query_index"]), on="canonical_smiles", how="left")
    valid.to_csv(OUT / "per_valid_molecule_similarity.csv.gz", index=False, compression="gzip")
    summary = valid.groupby(["step", "repeat", "model"], as_index=False)[["ECFP4_top1", "ECFP4_top5_mean", "ECFP4_top10_mean", "FCFP4_top1", "FCFP4_top5_mean", "FCFP4_top10_mean"]].mean()
    summary.to_csv(OUT / "per_repeat_similarity_means.csv", index=False)
    pooled = valid.groupby(["step", "model"], as_index=False)[["ECFP4_top1", "ECFP4_top5_mean", "ECFP4_top10_mean", "FCFP4_top1", "FCFP4_top5_mean", "FCFP4_top10_mean"]].mean()
    pooled.to_csv(OUT / "pooled_similarity_means.csv", index=False)


def chemical_summary() -> None:
    """Summarize descriptors, Murcko scaffold support, and sampled diversity."""
    molecules, _, descriptors = load_collected()
    valid = molecules[molecules["sanitize_valid"].eq(1)].copy()
    rows: list[dict[str, Any]] = []
    for (step, repeat, model), group in valid.groupby(["step", "repeat", "model"], sort=True):
        smiles = group["canonical_smiles"].astype(str)
        scaffolds = group.loc[group["murcko_scaffold"].astype(str).ne(""), "murcko_scaffold"].astype(str)
        scaffold_counts = scaffolds.value_counts()
        sample = group.sample(n=min(500, len(group)), random_state=42).reset_index(drop=True)
        fps_ecfp = [fp_for_smiles(value, False) for value in sample["canonical_smiles"]]
        fps_fcfp = [fp_for_smiles(value, True) for value in sample["canonical_smiles"]]
        ecfp_bits = [set(int(bit) for bit in fp.GetOnBits()) for fp in fps_ecfp]
        fcfp_bits = [set(int(bit) for bit in fp.GetOnBits()) for fp in fps_fcfp]

        def mean_pairwise(bit_sets: list[set[int]]) -> float:
            values: list[float] = []
            for left in range(len(bit_sets)):
                for right in range(left):
                    union = len(bit_sets[left] | bit_sets[right])
                    values.append(len(bit_sets[left] & bit_sets[right]) / union if union else 0.0)
            return float(np.mean(values)) if values else 0.0

        rows.append({
            "step": int(step), "repeat": repeat, "model": model,
            "n_valid": len(group), "n_unique_smiles": smiles.nunique(),
            "unique_smiles_rate_valid": smiles.nunique() / len(group) if len(group) else 0.0,
            "n_nonempty_murcko_scaffolds": len(scaffold_counts),
            "unique_scaffold_per_valid": len(scaffold_counts) / len(group) if len(group) else 0.0,
            "scaffold_top1_fraction": float(scaffold_counts.iloc[0] / len(scaffolds)) if len(scaffold_counts) else 0.0,
            "ecfp4_sample_mean_pairwise_similarity": mean_pairwise(ecfp_bits),
            "fcfp4_sample_mean_pairwise_similarity": mean_pairwise(fcfp_bits),
            "diversity_sample_n": len(sample),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "per_repeat_chemical_summary.csv", index=False)
    pooled = summary.groupby(["step", "model"], as_index=False)[[
        "n_valid", "n_unique_smiles", "unique_smiles_rate_valid",
        "n_nonempty_murcko_scaffolds", "unique_scaffold_per_valid",
        "scaffold_top1_fraction", "ecfp4_sample_mean_pairwise_similarity",
        "fcfp4_sample_mean_pairwise_similarity",
    ]].mean()
    pooled.to_csv(OUT / "pooled_chemical_summary.csv", index=False)
    descriptors.to_csv(OUT / "per_repeat_descriptor_means.csv", index=False)


def load_optional_chemical_summary() -> pd.DataFrame:
    """Load pooled chemical summary, creating it if needed."""
    path = OUT / "pooled_chemical_summary.csv"
    if not path.exists():
        chemical_summary()
    return pd.read_csv(path)


def project_umap() -> None:
    """Project valid molecules using exact cosine KNN to fixed ChEMBL landmarks."""
    molecules, _, _ = load_collected()
    valid = molecules[molecules["sanitize_valid"].eq(1)].copy().reset_index(drop=True)
    cache = load_cache(REFERENCE_ECFP4)
    ref_matrix = bitvectors_to_csr(list(cache["fingerprints"]))
    query_fps = [fp_for_smiles(str(smiles), False) for smiles in valid["canonical_smiles"]]
    query_matrix = bitvectors_to_csr(query_fps)
    landmark_xy = np.load(LANDMARK_COORDS)
    if landmark_xy.shape != (100000, 2):
        raise ValueError(f"unexpected landmark coordinate shape: {landmark_xy.shape}")
    LOGGER.info("UMAP projection query=%d reference_landmarks=%d", len(valid), ref_matrix.shape[0])
    knn = NearestNeighbors(n_neighbors=20, metric="cosine", algorithm="brute", n_jobs=28)
    knn.fit(ref_matrix)
    distances, indices = knn.kneighbors(query_matrix, return_distance=True)
    similarities = np.clip(1.0 - distances, 0.0, None)
    weights = similarities / np.maximum(similarities.sum(axis=1, keepdims=True), 1e-12)
    xy = np.einsum("ij,ijk->ik", weights, landmark_xy[indices])
    full_xy = np.load(FULL_UMAP_COORDS, mmap_mode="r")
    nearest = NearestNeighbors(n_neighbors=1, algorithm="kd_tree", n_jobs=28)
    nearest.fit(full_xy)
    valid["umap1"] = xy[:, 0]
    valid["umap2"] = xy[:, 1]
    valid["nearest_chembl_umap_distance"] = nearest.kneighbors(xy, return_distance=True)[0][:, 0]
    density_payload = np.load(FULL_UMAP_DENSITY)
    density = density_payload["density"]
    x_edges = density_payload["x_edges"]
    y_edges = density_payload["y_edges"]
    for mass in [0.50, 0.90, 0.95, 0.99]:
        flat = density.ravel()
        ordered = np.sort(flat)[::-1]
        threshold = ordered[min(int(np.searchsorted(np.cumsum(ordered) / ordered.sum(), mass)), len(ordered) - 1)]
        xbin = np.searchsorted(x_edges, xy[:, 0], side="right") - 1
        ybin = np.searchsorted(y_edges, xy[:, 1], side="right") - 1
        inside = np.zeros(len(xy), dtype=bool)
        okay = (xbin >= 0) & (xbin < density.shape[1]) & (ybin >= 0) & (ybin < density.shape[0])
        inside[okay] = density[ybin[okay], xbin[okay]] >= threshold
        valid[f"inside_hdr_{int(mass * 100)}"] = inside
    valid.to_csv(OUT / "fixed_chembl_umap_projection.csv.gz", index=False, compression="gzip")
    summary = valid.groupby(["step", "repeat", "model"], as_index=False).agg(
        n=("row_id", "size"), nearest_chembl_umap_distance_mean=("nearest_chembl_umap_distance", "mean"),
        nearest_chembl_umap_distance_median=("nearest_chembl_umap_distance", "median"),
        inside_hdr_50=("inside_hdr_50", "mean"), inside_hdr_90=("inside_hdr_90", "mean"),
        inside_hdr_95=("inside_hdr_95", "mean"), inside_hdr_99=("inside_hdr_99", "mean"),
    )
    summary.to_csv(OUT / "fixed_chembl_umap_summary_by_repeat.csv", index=False)
    pooled = summary.groupby(["step", "model"], as_index=False)[["nearest_chembl_umap_distance_mean", "nearest_chembl_umap_distance_median", "inside_hdr_50", "inside_hdr_90", "inside_hdr_95", "inside_hdr_99"]].agg(["mean", "std"])
    pooled.columns = ["_".join(col).strip("_") for col in pooled.columns]
    pooled.reset_index().to_csv(OUT / "fixed_chembl_umap_summary.csv", index=False)


def docking_summary() -> pd.DataFrame:
    """Read existing 1000-step docking and optional additive 500-step docking."""
    rows: list[dict[str, Any]] = []
    formal_root = ROOT / "analysis/benchmark/8gjl_c1101_benchmark_20260706_1000steps_3rep_formal/docking"
    for repeat in ["rep1", "rep2", "rep3"]:
        for model in MODEL_ORDER:
            slug = model.lower().replace("-", "_")
            path = formal_root / repeat / slug / "docking_results.csv"
            rows.extend(read_docking_csv(path, 1000, repeat, model))
    new_root = OUT / "docking_500"
    for repeat in [f"rep{i}" for i in range(4, 9)]:
        for model in MODEL_ORDER:
            slug = model.lower().replace("-", "_")
            rows.extend(read_docking_csv(new_root / repeat / slug / "docking_results.csv", 500, repeat, model))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "docking_per_ligand_combined.csv.gz", index=False, compression="gzip")
    if frame.empty:
        return frame
    grouped = frame.groupby(["step", "repeat", "model"], as_index=False)
    summary = grouped.apply(summarize_docking, include_groups=False).reset_index(drop=True)
    summary.to_csv(OUT / "docking_summary_by_repeat.csv", index=False)
    return summary


def read_docking_csv(path: Path, step: int, repeat: str, model: str) -> list[dict[str, Any]]:
    """Read one docking result if available."""
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    if "score" not in frame or "success" not in frame:
        return []
    def as_bool(value: Any) -> bool:
        """Parse CSV boolean values without treating the string False as true."""
        return str(value).strip().lower() in {"1", "true", "yes", "y"}

    return [{"step": step, "repeat": repeat, "model": model, "score": row["score"], "success": as_bool(row["success"]), "source_docking_csv": str(path)} for _, row in frame.iterrows()]


def summarize_docking(group: pd.DataFrame) -> pd.Series:
    """Summarize lower-is-better docking scores."""
    scores = pd.to_numeric(group.loc[group["success"], "score"], errors="coerce").dropna().to_numpy()
    best_n = max(1, int(np.ceil(len(scores) * 0.05))) if len(scores) else 0
    return pd.Series({"docking_total_rows": len(group), "docking_success": len(scores), "docking_success_rate_valid_input": len(scores) / len(group) if len(group) else np.nan, "docking_score_mean": scores.mean() if len(scores) else np.nan, "docking_score_median": np.median(scores) if len(scores) else np.nan, "docking_score_p05": np.quantile(scores, 0.05) if len(scores) else np.nan, "docking_best5_mean": np.sort(scores)[:best_n].mean() if len(scores) else np.nan, "docking_rate_le_-10": np.mean(scores <= -10) if len(scores) else np.nan})


def configure_plotting() -> None:
    """Apply readable Python-only figure defaults."""
    matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 13, "axes.titlesize": 17, "axes.labelsize": 15, "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11, "legend.frameon": False, "axes.spines.right": False, "axes.spines.top": False, "svg.fonttype": "none", "pdf.fonttype": 42})


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Save PNG and editable SVG."""
    fig.savefig(path.with_suffix(".png"), dpi=350, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_all() -> None:
    """Generate comparison figures from saved tables."""
    configure_plotting()
    molecules, per_repeat, _ = load_collected()
    projection = pd.read_csv(OUT / "fixed_chembl_umap_projection.csv.gz")
    valid = projection[projection["sanitize_valid"].eq(1)]
    colors = [MODEL_COLORS[m] for m in MODEL_ORDER]
    # One model per facet keeps colored points semantically unambiguous.
    fig, axes = plt.subplots(2, 5, figsize=(22, 8), sharex=True, sharey=True)
    density_payload = np.load(FULL_UMAP_DENSITY)
    grid = density_payload["density"]
    bounds = density_payload["bounds"]
    positive = grid[grid > 0]
    for row_index, step in enumerate(STEPS):
        for col_index, model in enumerate(MODEL_ORDER):
            ax = axes[row_index, col_index]
            ax.imshow(grid, origin="lower", extent=bounds, cmap="Greys", norm=mcolors.LogNorm(vmin=max(1.0, float(np.quantile(positive, 0.05))), vmax=float(grid.max())), alpha=0.8, aspect="auto", interpolation="nearest", rasterized=True)
            group = valid[(valid["step"] == step) & valid["model"].eq(model)]
            ax.scatter(group["umap1"], group["umap2"], s=4, alpha=0.55, color=MODEL_COLORS[model], linewidths=0, rasterized=True)
            ax.set_title(f"{model}\n{step} steps | n={len(group):,}")
            ax.set_xlim(bounds[:2]); ax.set_ylim(bounds[2:])
            if col_index == 0: ax.set_ylabel("UMAP 2")
            if row_index == 1: ax.set_xlabel("UMAP 1")
    fig.suptitle("Completed five-model generations on the fixed full-ChEMBL UMAP", y=1.02, fontsize=20)
    fig.text(0.5, -0.01, "Gray: full ChEMBL36 density; colored points: sanitize-valid generated molecules. Projection uses cosine-KNN to fixed ChEMBL landmarks; UMAP is descriptive only.", ha="center", fontsize=11)
    save_figure(fig, OUT / "fixed_chembl_umap_completed_500_vs_1000_facets")

    summary = per_repeat.groupby(["step", "model"], as_index=False)[["valid_per_attempt", "unique_per_attempt"]].agg(["mean", "std"])
    summary.columns = ["_".join(col).strip("_") for col in summary.columns]
    summary = summary.reset_index()
    x = np.arange(len(MODEL_ORDER))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    width = 0.36
    for idx, step in enumerate(STEPS):
        group = summary[summary.step.eq(step)].set_index("model").reindex(MODEL_ORDER)
        axes[0].bar(x + (idx - 0.5) * width, group.valid_per_attempt_mean * 100, width, yerr=group.valid_per_attempt_std.fillna(0) * 100, capsize=3, color=colors, alpha=0.95 if idx == 0 else 0.55, label=f"{step} steps")
        axes[1].bar(x + (idx - 0.5) * width, group.unique_per_attempt_mean * 100, width, yerr=group.unique_per_attempt_std.fillna(0) * 100, capsize=3, color=colors, alpha=0.95 if idx == 0 else 0.55, label=f"{step} steps")
    axes[0].set_title("Sanitize-valid yield"); axes[0].set_ylabel("Per 1,000 attempts (%)")
    axes[1].set_title("Unique canonical SMILES")
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(MODEL_ORDER, rotation=28, ha="right"); ax.set_ylim(0, 105)
    axes[1].legend()
    fig.suptitle("Generation integrity across completed repeats", y=1.02, fontsize=20)
    save_figure(fig, OUT / "generation_integrity_500_vs_1000")

    sim = pd.read_csv(OUT / "pooled_similarity_means.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for idx, step in enumerate(STEPS):
        group = sim[sim.step.eq(step)].set_index("model").reindex(MODEL_ORDER)
        offset = (idx - 0.5) * width
        axes[0].bar(x + offset, group.ECFP4_top1, width, color=colors, alpha=0.95 if idx == 0 else 0.55, label=f"{step} steps")
        axes[1].bar(x + offset, group.FCFP4_top1, width, color=colors, alpha=0.95 if idx == 0 else 0.55, label=f"{step} steps")
    axes[0].set_title("ECFP4 ChEMBL top-1 similarity"); axes[1].set_title("FCFP4 ChEMBL top-1 similarity")
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(MODEL_ORDER, rotation=28, ha="right")
        ax.set_ylabel("Mean Tanimoto similarity")
    axes[1].legend()
    fig.suptitle("Fingerprint similarity to fixed ChEMBL100k", y=1.02, fontsize=20)
    save_figure(fig, OUT / "chembl_fingerprint_similarity_500_vs_1000")

    chemical = load_optional_chemical_summary()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    metrics = [("unique_scaffold_per_valid", "Unique Murcko scaffolds / valid"), ("scaffold_top1_fraction", "Top-1 scaffold fraction"), ("ecfp4_sample_mean_pairwise_similarity", "ECFP4 pairwise similarity")]
    for axis, (metric, title) in zip(axes, metrics):
        for idx, step in enumerate(STEPS):
            group = chemical[chemical.step.eq(step)].set_index("model").reindex(MODEL_ORDER)
            axis.bar(x + (idx - 0.5) * width, group[metric], width, color=colors, alpha=0.95 if idx == 0 else 0.55, label=f"{step} steps")
        axis.set_title(title); axis.set_xticks(x); axis.set_xticklabels(MODEL_ORDER, rotation=28, ha="right")
    axes[2].legend()
    fig.suptitle("Chemical diversity and scaffold concentration", y=1.02, fontsize=20)
    save_figure(fig, OUT / "chemical_diversity_500_vs_1000")


def write_report(similarity_available: bool) -> None:
    """Write a concise evidence-grounded Markdown report."""
    _, per_repeat, descriptors = load_collected()
    pooled = per_repeat.groupby(["step", "model"], as_index=False).agg(valid_per_attempt_mean=("valid_per_attempt", "mean"), valid_per_attempt_std=("valid_per_attempt", "std"), unique_per_attempt_mean=("unique_per_attempt", "mean"), unique_per_attempt_std=("unique_per_attempt", "std"), repeats=("repeat", "nunique"))
    pooled.to_csv(OUT / "pooled_generation_summary.csv", index=False)
    lines = [
        "# 已完成五模型横评统一分析（500/1000 步）",
        "",
        "## 分析范围",
        "",
        "本报告只纳入已完成且每模型固定 1,000 次尝试的批次：500 步 `rep4–rep8`（5 个重复）和 1000 步正式 `rep1–rep3`（3 个重复）。500 步 `rep9` 在运行中，未进入任何正式统计。",
        "",
        "## 生成完整性",
        "",
        "| diffusion steps | model | repeats | sanitize-valid / attempt | unique SMILES / attempt |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in pooled.itertuples(index=False):
        lines.append(f"| {int(row.step)} | {row.model} | {int(row.repeats)} | {row.valid_per_attempt_mean:.3f} +/- {row.valid_per_attempt_std:.3f} | {row.unique_per_attempt_mean:.3f} +/- {row.unique_per_attempt_std:.3f} |")
    lines += [
        "",
        "![Generation integrity](generation_integrity_500_vs_1000.png)",
        "",
        "## UMAP",
        "",
        "![Fixed ChEMBL UMAP facets](fixed_chembl_umap_completed_500_vs_1000_facets.png)",
        "",
        "UMAP 图中的灰色背景是既有 full-ChEMBL36 坐标密度，彩色点是各模型的 sanitize-valid 分子。新分子采用 ECFP4 cosine-KNN 对固定 ChEMBL landmark 坐标加权投影，没有重新拟合 UMAP；因此该图只支持化学空间邻近/覆盖的描述，不支持活性、docking 或绝对质量结论。",
        "",
        "## 指纹相似度",
        "",
    ]
    if similarity_available:
        sim = pd.read_csv(OUT / "pooled_similarity_means.csv")
        lines += ["相似度只在 sanitize-valid 分子上计算，ChEMBL 参考为固定 100k 缓存，ECFP4/FCFP4 均为 radius=2、2,048 bits；表中为分子对参考库 top-1、top-5 和 top-10 相似度均值。", "", "| steps | model | ECFP4 top1 | ECFP4 top5 | FCFP4 top1 | FCFP4 top5 |", "|---:|---|---:|---:|---:|---:|"]
        for row in sim.itertuples(index=False):
            lines.append(f"| {int(row.step)} | {row.model} | {row.ECFP4_top1:.4f} | {row.ECFP4_top5_mean:.4f} | {row.FCFP4_top1:.4f} | {row.FCFP4_top5_mean:.4f} |")
        lines += ["", "![Fingerprint similarity](chembl_fingerprint_similarity_500_vs_1000.png)"]
    else:
        lines.append("ECFP4/FCFP4 计算尚未完成；当前报告不填入估计值，完成后重新运行 `--stage report` 即可补入。")
    lines += [
        "",
        "## Docking",
        "",
    ]
    docking_path = OUT / "docking_summary_by_repeat.csv"
    docking = pd.read_csv(docking_path) if docking_path.exists() else pd.DataFrame()
    if docking.empty:
        lines.append("当前没有可用的 docking 汇总记录。")
    else:
        docking_pooled = docking.groupby("model", as_index=False).agg(
            repeats=("repeat", "nunique"),
            valid_input_mean=("docking_total_rows", "mean"),
            score_mean=("docking_score_mean", "mean"),
            score_median=("docking_score_median", "mean"),
            rate_le_minus10=("docking_rate_le_-10", "mean"),
        )
        docking_pooled["model_order"] = docking_pooled["model"].map({name: index for index, name in enumerate(MODEL_ORDER)})
        docking_pooled = docking_pooled.sort_values("model_order")
        steps_with_docking = sorted(docking["step"].astype(int).unique().tolist())
        missing_steps = [step for step in STEPS if step not in steps_with_docking]
        lines += [
            f"当前已接入 {', '.join(str(step) for step in steps_with_docking)} 步 UniDock 结果；按模型汇总如下。每个数值均来自 valid-input docking，低分更优。",
            "",
            "| model | repeats | valid docking inputs (mean) | score mean | score median | fraction <= -10 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in docking_pooled.itertuples(index=False):
            lines.append(f"| {row.model} | {int(row.repeats)} | {row.valid_input_mean:.1f} | {row.score_mean:.3f} | {row.score_median:.3f} | {row.rate_le_minus10:.3f} |")
        if missing_steps:
            lines += ["", f"尚未接入 {', '.join(str(step) for step in missing_steps)} 步 docking；这些步骤不使用生成成功率替代。"]
        lines += ["", "逐重复汇总：`docking_summary_by_repeat.csv`；逐配体结果：`docking_per_ligand_combined.csv.gz`。"]
    lines += [
        "",
        "![Chemical diversity](chemical_diversity_500_vs_1000.png)",
        "",
        "## 证据边界",
        "",
        "- 生成成功率分母固定为每模型每重复 1,000 attempts；SDF 实际写入数不是成功率分母。",
        "- 描述符、指纹和 UMAP 主要在 sanitize-valid 分子上统计；失败结构保留在逐记录表中。",
        "- 1000 步 `rep1` 的 DiffGui 使用显式 DiffGui 重跑，其他模型沿用正式 sources manifest 中记录的来源，不能把它们描述成同一批次同一随机轨迹。",
        "- UMAP 是辅助的表示空间图，不能替代 ECFP4/FCFP4 原空间相似度、结构合法性、骨架多样性、docking 或后续独立验证。",
        "",
        "## 文件",
        "",
        "- `per_record_quality.csv.gz`: 40 个 source SDF 的逐记录 RDKit/描述符结果。",
        "- `per_repeat_summary.csv`: 固定 attempts 分母下的重复级汇总。",
        "- `per_valid_molecule_similarity.csv.gz`: 指纹相似度逐分子结果（完成后）。",
        "- `fixed_chembl_umap_projection.csv.gz`: 固定 ChEMBL UMAP 投影逐分子结果。",
    ]
    (OUT / "completed_comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(workers: int, stage: str) -> None:
    """Write input lineage and execution settings."""
    payload = {
        "status": "in_progress",
        "stage": stage,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "completed_scope": {"500_steps": ["rep4", "rep5", "rep6", "rep7", "rep8"], "1000_steps": ["rep1", "rep2", "rep3"], "excluded": "500-step rep9 in progress"},
        "attempts_per_model_repeat": ATTEMPTS,
        "workers": workers,
        "fingerprints": {"ECFP4": {"cache": str(REFERENCE_ECFP4), "radius": 2, "n_bits": 2048}, "FCFP4": {"cache": str(REFERENCE_FCFP4), "radius": 2, "n_bits": 2048}},
        "umap": {"full_coordinates": str(FULL_UMAP_COORDS), "density_grid": str(FULL_UMAP_DENSITY), "projection": "exact cosine KNN20 weighted average of fixed 100k landmark coordinates", "interpretation": "descriptive only"},
        "source_manifest_sha256": sha256(ROOT / "analysis/benchmark/8gjl_c1101_benchmark_20260706_1000steps_3rep_formal/formal_3rep_sources.csv"),
    }
    (OUT / "manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    """Run the selected analysis stage."""
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["collect", "similarity", "chemistry", "umap", "plot", "report", "all"], default="collect")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=28)
    args = parser.parse_args()
    OUT = args.output_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    setup_logging(OUT)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    LOGGER.info("stage=%s workers=%d output=%s", args.stage, args.workers, OUT)
    write_manifest(args.workers, args.stage)
    if args.stage in {"collect", "all"}:
        collect()
    if args.stage in {"similarity", "all"}:
        similarity(args.workers)
    if args.stage in {"chemistry", "all", "plot", "report"}:
        chemical_summary()
    if args.stage in {"umap", "all"}:
        project_umap()
    if args.stage in {"plot", "all"}:
        plot_all()
    if args.stage in {"report", "all"}:
        has_similarity = (OUT / "pooled_similarity_means.csv").exists()
        docking_summary()
        write_report(has_similarity)
    if args.stage in {"all", "report"}:
        manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
        manifest["status"] = "complete"
        manifest["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (OUT / "completion.json").write_text(json.dumps({"status": "complete", "stage": args.stage}, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("completed stage=%s", args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
