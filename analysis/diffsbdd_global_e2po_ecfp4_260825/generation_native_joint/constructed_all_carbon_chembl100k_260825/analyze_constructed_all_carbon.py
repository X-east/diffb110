#!/usr/bin/env python3
"""Measure several fingerprint similarities for constructed all-carbon molecules."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import multiprocessing as mp
import pickle
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors
from rdkit.Chem.AtomPairs import Pairs

RDLogger.DisableLog("rdApp.*")

METHODS = ("ECFP4", "FCFP4", "ECFP6", "FCFP6", "MACCS", "AtomPair", "TopologicalTorsion", "RDK")
REFERENCE_SMILES: list[str] = []

CONSTRUCTED = {
    "ethane": "CC",
    "n_hexane": "CCCCCC",
    "cyclohexane": "C1CCCCC1",
    "cyclohexene": "C1=CCCCC1",
    "benzene": "c1ccccc1",
    "toluene": "Cc1ccccc1",
    "ethylbenzene": "CCc1ccccc1",
    "styrene": "C=Cc1ccccc1",
    "naphthalene": "c1ccc2ccccc2c1",
    "biphenyl": "c1ccccc1-c2ccccc2",
    "fluorene": "c1ccc2c(c1)Cc1ccccc1-2",
    "anthracene": "c1ccc2cc3ccccc3cc2c1",
    "adamantane": "C1C2CC3CC1CC(C2)C3",
    "decalin": "C1CCC2CCCCC2C1",
}


def load_reference(path: Path) -> tuple[list[str], list[Any], dict[str, Any]]:
    """Load the exact 100,000 ChEMBL rows and cached ECFP4 fingerprints."""
    with gzip.open(path, "rb") as handle:
        payload = pickle.load(handle)
    records = payload.get("records", [])
    fingerprints = payload.get("fingerprints", [])
    if len(records) != 100_000 or len(fingerprints) != 100_000:
        raise ValueError(f"Expected exactly 100,000 reference rows; got {len(records)} records and {len(fingerprints)} fingerprints")
    smiles = [str(record["canonical_smiles"]) for record in records]
    metadata = {
        "source": str(path.resolve()),
        "records": len(records),
        "fingerprint_type_cached": payload.get("fingerprint_type"),
        "radius_cached": payload.get("radius"),
        "n_bits_cached": payload.get("n_bits"),
        "chembl_version": payload.get("chembl_version"),
        "sampling_seed": payload.get("sampling_seed"),
    }
    return smiles, list(fingerprints), metadata


def molecule_fingerprint(molecule: Chem.Mol, method: str) -> Any:
    """Calculate one configured fingerprint using fixed project settings."""
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
        return rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(molecule, nBits=2048)
    if method == "TopologicalTorsion":
        return rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(molecule, nBits=2048)
    if method == "RDK":
        return Chem.RDKFingerprint(molecule, fpSize=2048)
    raise ValueError(f"Unknown fingerprint method: {method}")


def fingerprint_smiles(item: tuple[str, str]) -> Any:
    """Build a fingerprint from one SMILES in a worker."""
    smiles, method = item
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"SMILES parse failed: {smiles}")
    return molecule_fingerprint(molecule, method)


def build_reference_fingerprints(smiles: list[str], method: str, workers: int) -> list[Any]:
    """Build one fingerprint list for the fixed ChEMBL100k rows."""
    if method == "ECFP4":
        raise ValueError("ECFP4 reference fingerprints must come from the cache")
    context = mp.get_context("fork")
    with context.Pool(workers) as pool:
        return list(pool.imap(fingerprint_smiles, ((value, method) for value in smiles), chunksize=256))


def validate_constructed() -> pd.DataFrame:
    """Validate and describe all constructed all-carbon molecules."""
    rows: list[dict[str, Any]] = []
    for name, smiles in CONSTRUCTED.items():
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"Constructed molecule failed to parse: {name}={smiles}")
        heavy_atoms = [atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1]
        if not heavy_atoms or any(atom.GetAtomicNum() != 6 for atom in heavy_atoms):
            raise ValueError(f"Constructed molecule is not all-carbon: {name}={smiles}")
        if len(Chem.GetMolFrags(molecule)) != 1:
            raise ValueError(f"Constructed molecule is disconnected: {name}={smiles}")
        rows.append(
            {
                "name": name,
                "smiles": smiles,
                "canonical_smiles": Chem.MolToSmiles(molecule, canonical=True),
                "heavy_atoms": molecule.GetNumHeavyAtoms(),
                "rings": rdMolDescriptors.CalcNumRings(molecule),
                "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(molecule),
            }
        )
    return pd.DataFrame(rows)


def score_query(query: Any, reference: list[Any]) -> dict[str, float]:
    """Return all-reference mean, median, top-1 and top-5 mean similarity."""
    similarities = np.asarray(DataStructs.BulkTanimotoSimilarity(query, reference), dtype=float)
    top = np.sort(similarities)[-5:][::-1]
    return {
        "mean_all_100k": float(np.mean(similarities)),
        "median_all_100k": float(np.median(similarities)),
        "p95_all_100k": float(np.quantile(similarities, 0.95)),
        "top1": float(top[0]),
        "top2": float(top[1]),
        "top3": float(top[2]),
        "top5_mean": float(np.mean(top)),
    }


def plot_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    """Save heatmaps for mean all-reference and nearest-neighbor similarities."""
    order = list(CONSTRUCTED)
    for value_column, stem, title in (
        ("mean_all_100k", "mean_similarity_heatmap", "Mean similarity across all ChEMBL100k molecules"),
        ("top5_mean", "top5_similarity_heatmap", "Top-5 mean similarity to ChEMBL100k"),
    ):
        pivot = summary.pivot(index="name", columns="fingerprint", values=value_column).reindex(index=order, columns=METHODS)
        figure, axis = plt.subplots(figsize=(13, 8), constrained_layout=True)
        image = axis.imshow(pivot.to_numpy(float), aspect="auto", cmap="viridis", vmin=0.0, vmax=max(0.5, float(np.nanmax(pivot.to_numpy(float)))))
        axis.set_xticks(np.arange(len(METHODS)), METHODS, rotation=35, ha="right")
        axis.set_yticks(np.arange(len(order)), order)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                axis.text(j, i, f"{pivot.iloc[i, j]:.3f}", ha="center", va="center", color="white" if pivot.iloc[i, j] < 0.25 else "black", fontsize=8)
        axis.set_title(title)
        axis.set_xlabel("Fingerprint")
        axis.set_ylabel("Constructed all-carbon molecule")
        figure.colorbar(image, ax=axis, label="Tanimoto similarity")
        figure.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
        figure.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
        plt.close(figure)


def write_report(summary: pd.DataFrame, molecules: pd.DataFrame, metadata: dict[str, Any], output_dir: Path) -> None:
    """Write a concise Chinese report with the distinction between mean and top-k."""
    mean_pivot = summary.pivot(index="name", columns="fingerprint", values="mean_all_100k").reindex(index=molecules["name"], columns=METHODS)
    lowest = mean_pivot.idxmin(axis=1)
    lines = [
        "# 构造全碳分子与 ChEMBL100k 指纹相似度",
        "",
        "## 定义",
        "",
        "构造了 14 个只含 C/H、单一连通片段的分子，覆盖短链、脂环、苯环、多环芳烃、稠合环和笼状结构。参考数据库是固定的 ChEMBL36 100,000 条记录。",
        "",
        "`mean_all_100k` 是一个构造分子与全部 100,000 个 ChEMBL 分子的 Tanimoto 相似度平均值；它不是最近邻相似度。`top1` 和 `top5_mean` 仅用于说明数据库中最接近的 ChEMBL 分子有多像。",
        "",
        "## 全部参考分子平均相似度",
        "",
        "| molecule | " + " | ".join(METHODS) + " | lowest mean metric |",
        "|---|" + "---:|" * len(METHODS) + "---|",
    ]
    for name in molecules["name"]:
        row = mean_pivot.loc[name]
        lines.append("| " + name + " | " + " | ".join(f"{float(row[method]):.4f}" for method in METHODS) + f" | {lowest.loc[name]} |")
    lines += [
        "",
        "![Mean similarity heatmap](mean_similarity_heatmap.png)",
        "",
        "![Top-5 similarity heatmap](top5_similarity_heatmap.png)",
        "",
        "## Interpretation",
        "",
        "平均相似度越低，说明该指纹下构造分子与整个 ChEMBL100k 分布的总体重合越弱；但不同指纹的数值不能直接横向当作同一量纲。一般应在同一指纹内部比较不同分子。",
        "",
        "本实验使用同一批 ChEMBL100k SMILES 为所有指纹的参考行；ECFP4 直接使用缓存，其他指纹从这些相同 SMILES 重新计算。",
        "",
        "## Evidence",
        "",
        "- per-query summary: `constructed_all_carbon_similarity_summary.csv`",
        "- all-reference raw scores: `constructed_all_carbon_similarity_long.csv.gz`",
        "- molecule definitions: `constructed_all_carbon_molecules.csv`",
        "- reference metadata: `analysis_manifest.json`",
    ]
    (output_dir / "constructed_all_carbon_similarity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the constructed all-carbon versus ChEMBL100k comparison."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--chembl-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=28)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    reference_smiles, ecfp4_reference, reference_metadata = load_reference(args.chembl_cache)
    molecules = validate_constructed()
    molecules.to_csv(args.output_dir / "constructed_all_carbon_molecules.csv", index=False)
    all_rows: list[dict[str, Any]] = []
    query_molecules = [Chem.MolFromSmiles(value) for value in molecules["canonical_smiles"]]
    for method in METHODS:
        reference_fps = ecfp4_reference if method == "ECFP4" else build_reference_fingerprints(reference_smiles, method, args.workers)
        query_fps = [molecule_fingerprint(molecule, method) for molecule in query_molecules]
        for name, query_fp in zip(molecules["name"], query_fps):
            row = {"name": name, "fingerprint": method}
            row.update(score_query(query_fp, reference_fps))
            all_rows.append(row)
        del reference_fps
    summary = pd.DataFrame(all_rows)
    summary.to_csv(args.output_dir / "constructed_all_carbon_similarity_summary.csv", index=False)
    summary.to_csv(args.output_dir / "constructed_all_carbon_similarity_long.csv.gz", index=False, compression="gzip")
    plot_summary(summary, args.output_dir)
    manifest = {
        "status": "completed",
        "reference": reference_metadata,
        "reference_used_for_all_methods": True,
        "methods": list(METHODS),
        "constructed_molecules": len(molecules),
        "workers": args.workers,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    (args.output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_report(summary, molecules, reference_metadata, args.output_dir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
