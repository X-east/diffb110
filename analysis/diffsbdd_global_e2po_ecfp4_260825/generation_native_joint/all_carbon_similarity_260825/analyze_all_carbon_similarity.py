#!/usr/bin/env python3
"""Compare generated molecules with strict all-carbon ChEMBL molecules."""

from __future__ import annotations

import argparse
import csv
import gc
import gzip
import heapq
import json
import multiprocessing as mp
import pickle
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

REFERENCE_FPS: list[Any] = []


def load_cache(path: Path) -> dict[str, Any]:
    """Load a gzipped ChEMBL fingerprint cache."""
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def atom_tokens(smiles: str) -> list[str]:
    """Extract SMILES atom tokens for a fast carbon-only prefilter."""
    return re.findall(r"Cl|Br|[A-Z][a-z]?|[a-z]", smiles)


def strict_all_carbon(smiles: str) -> bool:
    """Return true only for one connected molecule whose heavy atoms are C."""
    if "." in smiles:
        return False
    tokens = atom_tokens(smiles)
    if any(token not in {"C", "c", "H"} for token in tokens):
        return False
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        return False
    heavy_atoms = [atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1]
    return bool(heavy_atoms) and all(atom.GetAtomicNum() == 6 for atom in heavy_atoms)


def load_all_carbon_indices(path: Path) -> tuple[list[int], list[str], dict[str, Any]]:
    """Find strict all-carbon rows while retaining cache row order."""
    cache = load_cache(path)
    records = cache["records"]
    indices: list[int] = []
    smiles: list[str] = []
    for index, record in enumerate(records):
        value = str(record["canonical_smiles"])
        if strict_all_carbon(value):
            indices.append(index)
            smiles.append(value)
    metadata = {
        "source": str(path.resolve()),
        "source_records": len(records),
        "all_carbon_records": len(indices),
        "radius": int(cache.get("radius", 2)),
        "n_bits": int(cache.get("n_bits", 2048)),
        "fingerprint_type": str(cache.get("fingerprint_type", "unknown")),
    }
    return indices, smiles, metadata


def selected_cached_fingerprints(path: Path, indices: list[int]) -> list[Any]:
    """Load only the all-carbon fingerprints from an aligned cache."""
    cache = load_cache(path)
    if len(cache["fingerprints"]) <= max(indices):
        raise ValueError(f"cache row count is smaller than requested indices: {path}")
    return [cache["fingerprints"][index] for index in indices]


def fingerprint(molecule: Chem.Mol, method: str) -> Any:
    """Calculate one of the supported 2D fingerprints."""
    if method == "ECFP4":
        return AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=2048)
    if method == "FCFP4":
        return AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=2048, useFeatures=True)
    if method == "ECFP6":
        return AllChem.GetMorganFingerprintAsBitVect(molecule, radius=3, nBits=2048)
    if method == "FCFP6":
        return AllChem.GetMorganFingerprintAsBitVect(molecule, radius=3, nBits=2048, useFeatures=True)
    if method == "MACCS":
        return MACCSkeys.GenMACCSKeys(molecule)
    if method == "AtomPair":
        return Pairs.GetHashedAtomPairFingerprintAsBitVect(molecule, nBits=2048)
    if method == "TopologicalTorsion":
        return rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(molecule, nBits=2048)
    raise ValueError(f"unknown fingerprint method: {method}")


def init_worker(reference_fps: list[Any]) -> None:
    """Install read-only reference fingerprints in a forked worker."""
    global REFERENCE_FPS
    REFERENCE_FPS = reference_fps


def score_query(item: tuple[int, Any]) -> tuple[int, list[float]]:
    """Return top-1/2/3/5 mean Tanimoto scores."""
    index, query_fp = item
    similarities = DataStructs.BulkTanimotoSimilarity(query_fp, REFERENCE_FPS)
    top = heapq.nlargest(5, (float(value) for value in similarities))
    if len(top) < 5:
        return index, [float("nan")] * 4
    return index, [top[0], top[1], top[2], float(np.mean(top))]


def valid_query_rows(path: Path) -> pd.DataFrame:
    """Load the already independently sanitized generated rows."""
    frame = pd.read_csv(path, low_memory=False)
    frame = frame[frame["sanitize_success"].eq(1)].copy()
    frame["query_index"] = np.arange(len(frame), dtype=np.int64)
    return frame.reset_index(drop=True)


def score_method(
    queries: pd.DataFrame,
    query_fps: list[Any],
    reference_fps: list[Any],
    method: str,
    workers: int,
) -> pd.DataFrame:
    """Score all queries against one all-carbon reference fingerprint set."""
    context = mp.get_context("fork")
    rows: list[list[float] | None] = [None] * len(query_fps)
    with context.Pool(workers, initializer=init_worker, initargs=(reference_fps,)) as pool:
        tasks = list(enumerate(query_fps))
        for index, scores in pool.imap(score_query, tasks, chunksize=1):
            rows[index] = scores
    result = queries[["method", "attempt", "canonical_smiles"]].copy()
    result["fingerprint"] = method
    result["all_carbon_reference"] = len(reference_fps)
    values = np.asarray(rows, dtype=float)
    result["top1"] = values[:, 0]
    result["top2"] = values[:, 1]
    result["top3"] = values[:, 2]
    result["top5_mean"] = values[:, 3]
    return result


def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    """Summarize top-k means and medians by method and generation batch."""
    output: list[dict[str, Any]] = []
    for (method, batch), group in scores.groupby(["fingerprint", "method"], sort=False):
        row: dict[str, Any] = {
            "fingerprint": method,
            "generation_method": batch,
            "n_valid_queries": len(group),
            "all_carbon_reference": int(group["all_carbon_reference"].iloc[0]),
        }
        for column in ("top1", "top2", "top3", "top5_mean"):
            values = group[column].dropna().to_numpy(float)
            row[f"{column}_mean"] = float(np.mean(values))
            row[f"{column}_median"] = float(np.median(values))
            row[f"{column}_p10"] = float(np.quantile(values, 0.10))
            row[f"{column}_p90"] = float(np.quantile(values, 0.90))
        output.append(row)
    return pd.DataFrame(output)


def plot_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    """Write top-k comparison figures."""
    methods = list(summary["fingerprint"].drop_duplicates())
    batches = list(summary["generation_method"].drop_duplicates())
    colors = {"Original": "#666666", "ECFP4-E2PO": "#4472A8"}
    x = np.arange(len(methods))
    width = 0.35
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for offset, batch in zip((-width / 2, width / 2), batches):
        subset = summary[summary["generation_method"].eq(batch)].set_index("fingerprint").reindex(methods)
        axes[0].bar(x + offset, subset["top1_mean"], width, label=batch, color=colors.get(batch, None))
        axes[1].bar(x + offset, subset["top5_mean_mean"], width, label=batch, color=colors.get(batch, None))
    axes[0].set_title("Top-1 similarity to strict all-carbon ChEMBL")
    axes[1].set_title("Top-5 mean similarity to strict all-carbon ChEMBL")
    axes[0].set_ylabel("Tanimoto similarity")
    axes[1].set_ylabel("Tanimoto similarity")
    for axis in axes:
        axis.set_xticks(x, methods, rotation=25, ha="right")
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.22)
        axis.legend(frameon=False)
    figure.savefig(output_dir / "all_carbon_topk_comparison.png", dpi=300, bbox_inches="tight")
    figure.savefig(output_dir / "all_carbon_topk_comparison.svg", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Run the all-carbon multi-fingerprint analysis."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-metrics", type=Path, required=True)
    parser.add_argument("--ecfp4-cache", type=Path, required=True)
    parser.add_argument("--fcfp4-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=28)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    query = valid_query_rows(args.query_metrics)
    if query["method"].value_counts().to_dict() != {"ECFP4-E2PO": 708, "Original": 653}:
        raise ValueError(f"unexpected query counts: {query['method'].value_counts().to_dict()}")
    ecfp4_indices, ecfp4_smiles, ecfp4_metadata = load_all_carbon_indices(args.ecfp4_cache)
    fcfp4_indices, fcfp4_smiles, fcfp4_metadata = load_all_carbon_indices(args.fcfp4_cache)
    if not ecfp4_indices or not fcfp4_indices:
        raise ValueError("no strict all-carbon ChEMBL reference molecules")
    if ecfp4_smiles != fcfp4_smiles:
        raise ValueError("ECFP4 and FCFP4 caches do not contain the same all-carbon row order")
    (args.output_dir / "reference_manifest.json").write_text(
        json.dumps(
            {
                "definition": "single-fragment molecule with all heavy atoms C",
                "ecfp4_reference": ecfp4_metadata,
                "fcfp4_reference": fcfp4_metadata,
                "n": len(ecfp4_indices),
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    reference_fps_by_method: dict[str, list[Any]] = {
        "ECFP4": selected_cached_fingerprints(args.ecfp4_cache, ecfp4_indices),
        "FCFP4": selected_cached_fingerprints(args.fcfp4_cache, fcfp4_indices),
    }
    molecules: list[Chem.Mol] = []
    for smiles in query["canonical_smiles"].astype(str):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"query SMILES failed: {smiles}")
        molecules.append(molecule)
    for method in ("ECFP6", "FCFP6", "MACCS", "AtomPair", "TopologicalTorsion"):
        reference_fps_by_method[method] = [fingerprint(Chem.MolFromSmiles(smiles), method) for smiles in ecfp4_smiles]
    query_fps_by_method = {
        method: [fingerprint(molecule, method) for molecule in molecules]
        for method in reference_fps_by_method
    }
    all_scores: list[pd.DataFrame] = []
    for method, reference_fps in reference_fps_by_method.items():
        scores = score_method(query, query_fps_by_method[method], reference_fps, method, args.workers)
        all_scores.append(scores)
        scores.to_csv(args.output_dir / f"{method.lower()}_all_carbon_scores.csv", index=False)
        del reference_fps
        gc.collect()
    score_frame = pd.concat(all_scores, ignore_index=True)
    summary = summarize(score_frame)
    summary.to_csv(args.output_dir / "all_carbon_similarity_summary.csv", index=False)
    plot_summary(summary, args.output_dir)
    lines = [
        "# Generated molecules versus strict all-carbon ChEMBL",
        "",
        "Definition: single-fragment ChEMBL36 molecules whose heavy atoms are all carbon; full ChEMBL36 aligned fingerprint caches are used.",
        "",
        "| fingerprint | generation method | valid queries | reference n | top-1 mean | top-2 mean | top-3 mean | top-5 mean |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.fingerprint} | {row.generation_method} | {row.n_valid_queries} | {row.all_carbon_reference} | "
            f"{row.top1_mean:.4f} | {row.top2_mean:.4f} | {row.top3_mean:.4f} | {row.top5_mean_mean:.4f} |"
        )
    lines += [
        "",
        "![All-carbon top-k comparison](all_carbon_topk_comparison.png)",
        "",
        "The values are nearest-neighbor similarities to the all-carbon reference set, not similarity to one fixed molecule. Lower values indicate weaker resemblance under that fingerprint, but different fingerprints are not numerically interchangeable.",
    ]
    (args.output_dir / "all_carbon_similarity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "status": "completed",
        "query_metrics": str(args.query_metrics.resolve()),
        "query_counts": query["method"].value_counts().to_dict(),
        "reference_definition": "ChEMBL36 single-fragment molecules whose heavy atoms are all carbon",
        "reference_count": len(ecfp4_indices),
        "fingerprints": list(reference_fps_by_method),
        "workers": args.workers,
    }
    (args.output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
