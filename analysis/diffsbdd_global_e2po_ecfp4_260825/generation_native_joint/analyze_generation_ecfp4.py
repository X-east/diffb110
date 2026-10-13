#!/usr/bin/env python3
"""Independently evaluate Original and ECFP4-E2PO fixed-attempt SDF files."""

from __future__ import annotations

import argparse
import csv
import gzip
import heapq
import json
import math
import multiprocessing as mp
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, FilterCatalog, Lipinski, QED, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "DiffSBDD" / "analysis" / "SA_Score"))
import sascorer

RDLogger.DisableLog("rdApp.*")

REFERENCE_FPS: list[Any] = []


def load_reference_fps(path: Path) -> list[Any]:
    """Load the project ECFP4 fingerprint cache."""
    with gzip.open(path, "rb") as handle:
        payload = pickle.load(handle)
    fingerprints = list(payload.get("fingerprints", []))
    metadata = {
        "radius": int(payload.get("radius", 2)),
        "n_bits": int(payload.get("n_bits", 2048)),
        "fingerprint_type": str(payload.get("fingerprint_type", "ecfp4")),
        "records": len(payload.get("records", [])),
    }
    if metadata["radius"] != 2 or metadata["n_bits"] != 2048:
        raise ValueError(f"Expected ECFP4 radius=2, n_bits=2048; got {metadata}")
    if not fingerprints:
        raise ValueError(f"Empty fingerprint cache: {path}")
    return fingerprints


def init_similarity_worker(reference_fps: list[Any]) -> None:
    """Install the read-only ChEMBL fingerprint cache in a forked worker."""
    global REFERENCE_FPS
    REFERENCE_FPS = reference_fps


def score_similarity(item: tuple[int, str]) -> tuple[int, float, float, float, float]:
    """Compute top-1/2/3 and top-5 mean ECFP4 Tanimoto for one molecule."""
    index, smiles = item
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return index, math.nan, math.nan, math.nan, math.nan
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=2048)
    similarities = DataStructs.BulkTanimotoSimilarity(fingerprint, REFERENCE_FPS)
    top = heapq.nlargest(5, (float(value) for value in similarities))
    if len(top) < 5:
        return index, math.nan, math.nan, math.nan, math.nan
    return index, top[0], top[1], top[2], float(np.mean(top))


def safe_float(value: float) -> float:
    """Return a JSON/CSV-safe finite float or NaN."""
    value = float(value)
    return value if math.isfinite(value) else math.nan


def bond_quality(molecule: Chem.Mol) -> tuple[int, int, float]:
    """Measure clearly implausible heavy-atom bond lengths from the conformer."""
    if molecule.GetNumConformers() == 0:
        return 0, 0, 0.0
    conformer = molecule.GetConformer()
    table = Chem.GetPeriodicTable()
    bad = 0
    checked = 0
    for bond in molecule.GetBonds():
        atom_a = bond.GetBeginAtom()
        atom_b = bond.GetEndAtom()
        if atom_a.GetAtomicNum() == 1 or atom_b.GetAtomicNum() == 1:
            continue
        ideal = table.GetRcovalent(atom_a.GetAtomicNum()) + table.GetRcovalent(atom_b.GetAtomicNum())
        if ideal <= 0:
            continue
        distance = conformer.GetAtomPosition(atom_a.GetIdx()).Distance(
            conformer.GetAtomPosition(atom_b.GetIdx())
        )
        checked += 1
        bad += int(distance / ideal < 0.70 or distance / ideal > 1.45)
    return bad, checked, bad / checked if checked else 0.0


def fused_system_size(molecule: Chem.Mol) -> int:
    """Return the largest atom count in a ring-overlap component."""
    rings = [set(ring) for ring in molecule.GetRingInfo().AtomRings()]
    if not rings:
        return 0
    remaining = list(rings)
    largest = 0
    while remaining:
        component = remaining.pop()
        changed = True
        while changed:
            changed = False
            kept: list[set[int]] = []
            for ring in remaining:
                if component.intersection(ring):
                    component.update(ring)
                    changed = True
                else:
                    kept.append(ring)
            remaining = kept
        largest = max(largest, len(component))
    return largest


def scaffold_smiles(molecule: Chem.Mol) -> str:
    """Return the canonical Murcko scaffold, or an empty string."""
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    return Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=True)


def build_catalog() -> FilterCatalog.FilterCatalog:
    """Build the standard RDKit Brenk structural-alert catalog."""
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    return FilterCatalog.FilterCatalog(params)


def metric_row(
    method: str,
    attempt: int,
    parsed: Chem.Mol | None,
    molecule: Chem.Mol | None,
    brenk: FilterCatalog.FilterCatalog,
) -> dict[str, Any]:
    """Extract one fixed-attempt row, retaining failed records explicitly."""
    row: dict[str, Any] = {
        "method": method,
        "attempt": attempt,
        "parsed": int(parsed is not None),
        "sanitize_success": int(molecule is not None),
        "failure_reason": "" if molecule is not None else ("parse_failed" if parsed is None else "sanitize_failed"),
        "canonical_smiles": "",
        "murcko_scaffold": "",
        "fragment_count": math.nan,
        "heavy_atoms": math.nan,
        "mw": math.nan,
        "logp": math.nan,
        "tpsa": math.nan,
        "hba": math.nan,
        "hbd": math.nan,
        "qed": math.nan,
        "sa_score": math.nan,
        "ring_count": math.nan,
        "aromatic_ring_count": math.nan,
        "three_member_ring_count": math.nan,
        "large_ring_count": math.nan,
        "fused_system_size": math.nan,
        "brenk_alert": math.nan,
        "carbon_fraction": math.nan,
        "hetero_atom_count": math.nan,
        "bad_bond_length_fraction": math.nan,
        "ecfp4_top1_similarity": math.nan,
        "ecfp4_top2_similarity": math.nan,
        "ecfp4_top3_similarity": math.nan,
        "ecfp4_top5_mean_similarity": math.nan,
    }
    if molecule is None:
        return row
    clean = Chem.RemoveHs(molecule)
    heavy_atoms = molecule.GetNumHeavyAtoms()
    carbon_atoms = sum(atom.GetAtomicNum() == 6 for atom in molecule.GetAtoms())
    bad, checked, bad_fraction = bond_quality(molecule)
    ring_sizes = [len(ring) for ring in molecule.GetRingInfo().AtomRings()]
    row.update(
        {
            "canonical_smiles": Chem.MolToSmiles(clean, canonical=True, isomericSmiles=True),
            "murcko_scaffold": scaffold_smiles(molecule),
            "fragment_count": len(Chem.GetMolFrags(molecule, asMols=False, sanitizeFrags=False)),
            "heavy_atoms": heavy_atoms,
            "mw": safe_float(Descriptors.MolWt(molecule)),
            "logp": safe_float(Crippen.MolLogP(molecule)),
            "tpsa": safe_float(Descriptors.TPSA(molecule)),
            "hba": Lipinski.NumHAcceptors(molecule),
            "hbd": Lipinski.NumHDonors(molecule),
            "qed": safe_float(QED.qed(molecule)),
            "sa_score": safe_float(sascorer.calculateScore(molecule)),
            "ring_count": rdMolDescriptors.CalcNumRings(molecule),
            "aromatic_ring_count": rdMolDescriptors.CalcNumAromaticRings(molecule),
            "three_member_ring_count": sum(size == 3 for size in ring_sizes),
            "large_ring_count": sum(size >= 9 for size in ring_sizes),
            "fused_system_size": fused_system_size(molecule),
            "brenk_alert": int(brenk.HasMatch(molecule)),
            "carbon_fraction": carbon_atoms / heavy_atoms if heavy_atoms else math.nan,
            "hetero_atom_count": heavy_atoms - carbon_atoms,
            "bad_bond_length_fraction": bad_fraction if checked else 0.0,
        }
    )
    return row


def read_sdf(path: Path, method: str, brenk: FilterCatalog.FilterCatalog) -> list[dict[str, Any]]:
    """Read one SDF without allowing RDKit's default sanitization to alter n."""
    supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
    rows: list[dict[str, Any]] = []
    for attempt, parsed in enumerate(supplier, start=1):
        molecule: Chem.Mol | None = None
        if parsed is not None:
            candidate = Chem.Mol(parsed)
            try:
                Chem.SanitizeMol(candidate)
                molecule = candidate
            except Exception:
                molecule = None
        rows.append(metric_row(method, attempt, parsed, molecule, brenk))
    return rows


def add_ecfp4_scores(rows: list[dict[str, Any]], reference_fps: list[Any], workers: int) -> None:
    """Score all valid rows against the exact training-time ChEMBL cache."""
    queries = [
        (index, str(row["canonical_smiles"]))
        for index, row in enumerate(rows)
        if row["sanitize_success"] == 1 and row["canonical_smiles"]
    ]
    context = mp.get_context("fork")
    with context.Pool(workers, initializer=init_similarity_worker, initargs=(reference_fps,)) as pool:
        for index, top1, top2, top3, top5 in pool.imap(score_similarity, queries, chunksize=1):
            rows[index]["ecfp4_top1_similarity"] = top1
            rows[index]["ecfp4_top2_similarity"] = top2
            rows[index]["ecfp4_top3_similarity"] = top3
            rows[index]["ecfp4_top5_mean_similarity"] = top5


def finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    """Return finite values for a valid-molecule metric."""
    return [float(row[key]) for row in rows if row["sanitize_success"] == 1 and math.isfinite(float(row[key]))]


def mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    """Compute a valid-molecule mean."""
    values = finite_values(rows, key)
    return float(np.mean(values)) if values else math.nan


def fraction_metric(rows: list[dict[str, Any]], key: str, threshold: float) -> float:
    """Compute the fraction of valid molecules meeting a threshold."""
    values = finite_values(rows, key)
    return float(np.mean(np.asarray(values) >= threshold)) if values else math.nan


def summary_row(method: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize fixed-attempt and valid-molecule denominators separately."""
    attempts = len(rows)
    valid = [row for row in rows if row["sanitize_success"] == 1]
    smiles = {str(row["canonical_smiles"]) for row in valid if row["canonical_smiles"]}
    scaffolds = {str(row["murcko_scaffold"]) for row in valid if row["murcko_scaffold"]}
    return {
        "method": method,
        "attempts": attempts,
        "sdf_records": attempts,
        "parsed_records": sum(row["parsed"] for row in rows),
        "sanitize_valid": len(valid),
        "valid_rate_attempts": len(valid) / attempts if attempts else math.nan,
        "unique_smiles": len(smiles),
        "unique_smiles_rate_attempts": len(smiles) / attempts if attempts else math.nan,
        "unique_smiles_rate_valid": len(smiles) / len(valid) if valid else math.nan,
        "unique_murcko_scaffolds": len(scaffolds),
        "unique_scaffolds_rate_attempts": len(scaffolds) / attempts if attempts else math.nan,
        "unique_scaffolds_rate_valid": len(scaffolds) / len(valid) if valid else math.nan,
        "fragment_rate_valid": fraction_metric(rows, "fragment_count", 2),
        "brenk_alert_rate_valid": fraction_metric(rows, "brenk_alert", 1),
        "three_member_ring_rate_valid": fraction_metric(rows, "three_member_ring_count", 1),
        "large_ring_rate_valid": fraction_metric(rows, "large_ring_count", 1),
        "bad_bond_rate_valid": fraction_metric(rows, "bad_bond_length_fraction", 1e-15),
        "heavy_atoms_mean_valid": mean_metric(rows, "heavy_atoms"),
        "mw_mean_valid": mean_metric(rows, "mw"),
        "logp_mean_valid": mean_metric(rows, "logp"),
        "tpsa_mean_valid": mean_metric(rows, "tpsa"),
        "qed_mean_valid": mean_metric(rows, "qed"),
        "sa_mean_valid": mean_metric(rows, "sa_score"),
        "ring_count_mean_valid": mean_metric(rows, "ring_count"),
        "aromatic_ring_mean_valid": mean_metric(rows, "aromatic_ring_count"),
        "carbon_fraction_mean_valid": mean_metric(rows, "carbon_fraction"),
        "ecfp4_top1_mean_valid": mean_metric(rows, "ecfp4_top1_similarity"),
        "ecfp4_top1_median_valid": float(np.median(finite_values(rows, "ecfp4_top1_similarity"))),
        "ecfp4_top2_mean_valid": mean_metric(rows, "ecfp4_top2_similarity"),
        "ecfp4_top2_median_valid": float(np.median(finite_values(rows, "ecfp4_top2_similarity"))),
        "ecfp4_top3_mean_valid": mean_metric(rows, "ecfp4_top3_similarity"),
        "ecfp4_top3_median_valid": float(np.median(finite_values(rows, "ecfp4_top3_similarity"))),
        "ecfp4_top5_mean_mean_valid": mean_metric(rows, "ecfp4_top5_mean_similarity"),
        "ecfp4_top5_mean_median_valid": float(np.median(finite_values(rows, "ecfp4_top5_mean_similarity"))),
        "ecfp4_top5_ge_030_rate_valid": fraction_metric(rows, "ecfp4_top5_mean_similarity", 0.30),
        "ecfp4_top5_ge_040_rate_valid": fraction_metric(rows, "ecfp4_top5_mean_similarity", 0.40),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a stable CSV using the union of row fields."""
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_outputs(summaries: list[dict[str, Any]], rows: list[dict[str, Any]], output_dir: Path) -> None:
    """Create a compact quantitative comparison figure set."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    labels = [str(row["method"]) for row in summaries]
    colors = ["#606060", "#4472A8"]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes[0, 0].bar(x, [row["valid_rate_attempts"] for row in summaries], color=colors)
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].set_ylabel("Valid / fixed attempts")
    axes[0, 0].set_title("Validity")
    axes[0, 1].bar(x - 0.18, [row["unique_smiles_rate_attempts"] for row in summaries], 0.36, label="unique SMILES", color="#8C6BB1")
    axes[0, 1].bar(x + 0.18, [row["unique_scaffolds_rate_attempts"] for row in summaries], 0.36, label="unique Murcko scaffolds", color="#D95F8D")
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].set_ylabel("Unique / fixed attempts")
    axes[0, 1].set_title("Diversity")
    axes[0, 1].legend(frameon=False, fontsize=9)
    axes[1, 0].bar(x - 0.18, [row["ecfp4_top1_mean_valid"] for row in summaries], 0.36, label="top-1", color="#2C7FB8")
    axes[1, 0].bar(x + 0.18, [row["ecfp4_top5_mean_mean_valid"] for row in summaries], 0.36, label="top-5 mean", color="#7FCDBB")
    axes[1, 0].set_ylim(0, 1.0)
    axes[1, 0].set_ylabel("Tanimoto / valid molecules")
    axes[1, 0].set_title("ChEMBL ECFP4 similarity")
    axes[1, 0].legend(frameon=False, fontsize=9)
    pathology_keys = ["fragment_rate_valid", "brenk_alert_rate_valid", "three_member_ring_rate_valid", "bad_bond_rate_valid"]
    pathology_labels = ["Fragments", "Brenk", "3-member rings", "Bad bonds"]
    width = 0.18
    for offset, key, title in zip(np.arange(4) - 1.5, pathology_keys, pathology_labels):
        axes[1, 1].bar(x + offset * width, [row[key] for row in summaries], width, label=title)
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].set_ylabel("Fraction / valid molecules")
    axes[1, 1].set_title("Structural quality flags")
    axes[1, 1].legend(frameon=False, fontsize=8, ncol=2)
    for axis in axes.flat:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.22)
    fig.suptitle("Original versus ECFP4-E2PO: fixed-attempt generation evaluation", fontsize=14)
    fig.savefig(output_dir / "generation_evaluation_overview.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "generation_evaluation_overview.svg", bbox_inches="tight")
    fig.savefig(output_dir / "generation_evaluation_overview.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for method, color in zip(labels, colors):
        method_rows = [row for row in rows if row["method"] == method]
        axes[0].hist(finite_values(method_rows, "ecfp4_top1_similarity"), bins=35, alpha=0.55, density=True, label=method, color=color)
        axes[1].hist(finite_values(method_rows, "ecfp4_top5_mean_similarity"), bins=35, alpha=0.55, density=True, label=method, color=color)
    axes[0].set_xlabel("Top-1 ECFP4 Tanimoto")
    axes[1].set_xlabel("Top-5 mean ECFP4 Tanimoto")
    for axis in axes:
        axis.set_ylabel("Density / valid molecules")
        axis.legend(frameon=False)
        axis.grid(axis="y", alpha=0.22)
    fig.suptitle("ChEMBL similarity distributions")
    fig.savefig(output_dir / "ecfp4_similarity_distributions.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "ecfp4_similarity_distributions.svg", bbox_inches="tight")
    plt.close(fig)


def write_report(
    summaries: list[dict[str, Any]],
    output_dir: Path,
    original_sdf: Path,
    e2po_sdf: Path,
    cache: Path,
) -> None:
    """Write the evidence-grounded Markdown report."""
    original, e2po = summaries
    rows = []
    for key, label in (
        ("attempts", "attempts"),
        ("sanitize_valid", "valid"),
        ("valid_rate_attempts", "valid / attempts"),
        ("unique_smiles", "unique SMILES"),
        ("unique_murcko_scaffolds", "unique scaffolds"),
        ("ecfp4_top1_mean_valid", "ECFP4 top-1 mean"),
        ("ecfp4_top2_mean_valid", "ECFP4 top-2 mean"),
        ("ecfp4_top3_mean_valid", "ECFP4 top-3 mean"),
        ("ecfp4_top5_mean_mean_valid", "ECFP4 top-5 mean"),
        ("ecfp4_top5_ge_030_rate_valid", "top-5 >= 0.30"),
        ("qed_mean_valid", "QED mean"),
        ("sa_mean_valid", "SA mean"),
        ("logp_mean_valid", "LogP mean"),
        ("tpsa_mean_valid", "TPSA mean"),
        ("fragment_rate_valid", "fragment rate"),
        ("brenk_alert_rate_valid", "Brenk rate"),
        ("bad_bond_rate_valid", "bad-bond rate"),
    ):
        def fmt(value: Any) -> str:
            return str(value) if key in {"attempts", "sanitize_valid", "unique_smiles", "unique_murcko_scaffolds"} else f"{float(value):.4f}"
        rows.append(f"| {label} | {fmt(original[key])} | {fmt(e2po[key])} |")
    report = [
        "# ECFP4-E2PO 固定生成独立评价",
        "",
        "## 结论",
        "",
        "在相同 1000 attempts、T=500、seed=1 和生成设置下，比较 Original 与 ECFP4-E2PO。核心判据是 ChEMBL ECFP4 相似度是否提高，同时检查有效性、重复度和结构病理是否发生明显退化。",
        "",
        "## 固定条件与分母",
        "",
        f"- Original SDF: `{original_sdf}`",
        f"- ECFP4-E2PO SDF: `{e2po_sdf}`",
        f"- ChEMBL reference: `{cache}`；ECFP4 radius=2、2048 bits、100,000 fingerprints",
        "- fixed-attempt 分母：每个方法 1000 条 SDF record；未对失败记录补齐或删除",
        "- valid-molecule 分母：仅用于 QED、LogP、TPSA、ECFP4 和结构旗标等分子级指标",
        "- `bad-bond rate` 定义为 valid 分子中至少一个重原子共价键长度超出共价半径和的 0.70--1.45 倍范围",
        "",
        "## Summary",
        "",
        "| metric | Original | ECFP4-E2PO |",
        "|---|---:|---:|",
        *rows,
        "",
        "## Figures",
        "",
        "![Generation evaluation overview](generation_evaluation_overview.png)",
        "",
        "![ECFP4 similarity distributions](ecfp4_similarity_distributions.png)",
        "",
        "## Interpretation boundary",
        "",
        "ECFP4 reward 只反映 2D 子结构接近 ChEMBL100k 的程度，不能单独证明靶点活性、3D pocket compatibility 或 docking 改善。若相似度上升但多样性、有效性或病理指标恶化，应解释为 reward alignment accompanied by a quality trade-off，而不是整体生成质量无条件改善。后续应结合 docking、ligand efficiency 和固定分母结构筛选决定是否进入更大规模训练。",
        "",
        "## Evidence files",
        "",
        "- per-attempt metrics: `per_attempt_metrics.csv`",
        "- summary: `generation_summary.csv`",
        "- ChEMBL similarity distribution: `ecfp4_similarity_distributions.png`",
    ]
    (output_dir / "generation_ecfp4_evaluation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    """Run the fixed-attempt evaluation and write all local evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-sdf", type=Path, required=True)
    parser.add_argument("--e2po-sdf", type=Path, required=True)
    parser.add_argument("--chembl-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=28)
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    for path in (args.original_sdf, args.e2po_sdf, args.chembl_cache):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    brenk = build_catalog()
    original_rows = read_sdf(args.original_sdf, "Original", brenk)
    e2po_rows = read_sdf(args.e2po_sdf, "ECFP4-E2PO", brenk)
    if len(original_rows) != 1000 or len(e2po_rows) != 1000:
        raise ValueError(f"Expected 1000 records per method; got {len(original_rows)} and {len(e2po_rows)}")
    reference_fps = load_reference_fps(args.chembl_cache)
    add_ecfp4_scores(original_rows, reference_fps, args.workers)
    add_ecfp4_scores(e2po_rows, reference_fps, args.workers)
    all_rows = original_rows + e2po_rows
    summaries = [summary_row("Original", original_rows), summary_row("ECFP4-E2PO", e2po_rows)]
    for summary in summaries:
        summary["delta_vs_original"] = 0.0 if summary["method"] == "Original" else summary["ecfp4_top5_mean_mean_valid"] - summaries[0]["ecfp4_top5_mean_mean_valid"]
    write_csv(args.output_dir / "per_attempt_metrics.csv", all_rows)
    write_csv(args.output_dir / "generation_summary.csv", summaries)
    plot_outputs(summaries, all_rows, args.output_dir)
    write_report(summaries, args.output_dir, args.original_sdf, args.e2po_sdf, args.chembl_cache)
    manifest = {
        "status": "completed",
        "methods": ["Original", "ECFP4-E2PO"],
        "attempts_per_method": 1000,
        "workers": args.workers,
        "chembl_cache": str(args.chembl_cache.resolve()),
        "chembl_records": len(reference_fps),
        "fingerprint": {"type": "ECFP4", "radius": 2, "n_bits": 2048},
        "original_sdf": str(args.original_sdf.resolve()),
        "e2po_sdf": str(args.e2po_sdf.resolve()),
        "output_dir": str(args.output_dir.resolve()),
    }
    (args.output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "summary": summaries}, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
