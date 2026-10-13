#!/usr/bin/env python3
"""Evaluate matched structure x-only generation against the fixed Original."""

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

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, FilterCatalog, Lipinski, QED, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "DiffSBDD" / "analysis" / "SA_Score"))

from DiffSBDD.analysis.SA_Score import sascorer  # noqa: E402
from ddpo.rewards.chemistry_guard_reward import ChemistryGuardReward  # noqa: E402
from ddpo.rewards.ring_complexity_reward import RingComplexityReward  # noqa: E402
from scripts.offline_rl.structure_pathology_rules import compute_structure_pathology  # noqa: E402

RDLogger.DisableLog("rdApp.*")

METHODS = ("Original", "Matched structure x-only")
ATTEMPTS = 1000
REFERENCE_FPS: list[Any] = []

CONTINUOUS_METRICS = (
    "heavy_atom_count",
    "mw",
    "logp",
    "tpsa",
    "hba",
    "hbd",
    "qed",
    "sa_score",
    "carbon_fraction",
    "hetero_atom_count",
    "ring_count",
    "aromatic_ring_count",
    "max_fused_ring_system_size",
    "spiro_atom_count",
    "bridgehead_atom_count",
    "bad_bond_length_fraction",
    "ecfp4_top1_similarity",
    "ecfp4_top5_mean_similarity",
)

PROPORTION_METRICS = {
    "parsed_rate_attempts": ("parsed", 1, "attempts"),
    "sanitize_rate_attempts": ("sanitize_success", 1, "attempts"),
    "unique_smiles_rate_valid": (None, None, "valid_unique_smiles"),
    "fragment_rate_valid": ("fragment_count", 2, "valid"),
    "brenk_alert_rate_valid": ("brenk_alert", 1, "valid"),
    "chemistry_guard_rate_valid": ("chemistry_guard_violation_count", 1, "valid"),
    "bad_bond_rate_valid": ("bad_bond_length_fraction", 1e-15, "valid"),
    "r1_large_ring_rate_valid": ("r1_large_ring_violation", 1, "valid"),
    "r2_ring_system_rate_valid": ("r2_ring_system_violation", 1, "valid"),
    "r3_four_ring_rate_valid": ("r3_four_ring_violation", 1, "valid"),
    "r4a_hetero_three_ring_rate_valid": ("r4a_three_ring_hetero_violation", 1, "valid"),
    "r4b_overattached_three_ring_rate_valid": ("r4b_three_ring_over_attached_violation", 1, "valid"),
    "r4c_fused_spiro_three_ring_rate_valid": ("r4c_three_ring_spiro_or_fused_violation", 1, "valid"),
    "r4_any_three_ring_rate_valid": ("r4_any_three_ring_violation", 1, "valid"),
    "r1_r4_pathology_rate_valid": ("is_structurally_pathological", 1, "valid"),
    "r5_carbon_guard_rate_valid": ("r5_carbon_fraction_violation", 1, "valid"),
    "r1_r5_pathology_rate_valid": ("is_pathological", 1, "valid"),
}


def _catalog() -> FilterCatalog.FilterCatalog:
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    return FilterCatalog.FilterCatalog(params)


def _finite(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _bond_quality(mol: Chem.Mol) -> tuple[int, int, float]:
    if mol.GetNumConformers() == 0:
        return 0, 0, 0.0
    conformer = mol.GetConformer()
    table = Chem.GetPeriodicTable()
    bad = 0
    checked = 0
    for bond in mol.GetBonds():
        atom_a = bond.GetBeginAtom()
        atom_b = bond.GetEndAtom()
        if atom_a.GetAtomicNum() == 1 or atom_b.GetAtomicNum() == 1:
            continue
        ideal = table.GetRcovalent(atom_a.GetAtomicNum()) + table.GetRcovalent(atom_b.GetAtomicNum())
        if ideal <= 0:
            continue
        distance = conformer.GetAtomPosition(atom_a.GetIdx()).Distance(conformer.GetAtomPosition(atom_b.GetIdx()))
        checked += 1
        bad += int(distance / ideal < 0.70 or distance / ideal > 1.45)
    return bad, checked, bad / checked if checked else 0.0


def _scaffold(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(
        MurckoScaffold.GetScaffoldForMol(mol), canonical=True, isomericSmiles=True
    )


def _empty_row(method: str, attempt: int, parsed: bool, reason: str) -> dict[str, Any]:
    return {
        "method": method,
        "attempt": attempt,
        "parsed": int(parsed),
        "sanitize_success": 0,
        "rdkit_valid": 0,
        "failure_reason": reason,
        "canonical_smiles": "",
        "murcko_scaffold": "",
    }


def _molecule_row(
    method: str,
    attempt: int,
    mol: Chem.Mol,
    brenk: FilterCatalog.FilterCatalog,
    guard: ChemistryGuardReward,
    rings: RingComplexityReward,
) -> dict[str, Any]:
    clean = Chem.RemoveHs(mol)
    guard_facts = guard._compute_facts(mol)
    ring_facts = rings._compute_facts(mol)
    pathology = compute_structure_pathology(mol)
    bad_bonds, checked_bonds, bad_fraction = _bond_quality(mol)
    return {
        "method": method,
        "attempt": attempt,
        "parsed": 1,
        "sanitize_success": 1,
        "rdkit_valid": 1,
        "failure_reason": "",
        "canonical_smiles": Chem.MolToSmiles(clean, canonical=True, isomericSmiles=True),
        "murcko_scaffold": _scaffold(mol),
        "fragment_count": guard_facts.fragment_count,
        "heavy_atom_count": mol.GetNumHeavyAtoms(),
        "atom_count": mol.GetNumAtoms(),
        "mw": _finite(Descriptors.MolWt(mol)),
        "logp": _finite(Crippen.MolLogP(mol)),
        "tpsa": _finite(Descriptors.TPSA(mol)),
        "hba": int(Lipinski.NumHAcceptors(mol)),
        "hbd": int(Lipinski.NumHDonors(mol)),
        "qed": _finite(QED.qed(mol)),
        "sa_score": _finite(sascorer.calculateScore(mol)),
        "brenk_alert": int(brenk.HasMatch(mol)),
        "chemistry_guard_violation_count": guard_facts.violation_count,
        "carbon_fraction": guard_facts.carbon_fraction,
        "hetero_atom_count": guard_facts.hetero_atom_count,
        "hetero_atom_fraction": guard_facts.hetero_atom_fraction,
        "ring_count": ring_facts.ring_count,
        "aromatic_ring_count": ring_facts.aromatic_ring_count,
        "three_member_ring_count": ring_facts.three_member_ring_count,
        "max_fused_ring_system_size": ring_facts.max_fused_ring_system_size,
        "spiro_atom_count": ring_facts.spiro_atom_count,
        "bridgehead_atom_count": ring_facts.bridgehead_atom_count,
        "bad_bond_count": bad_bonds,
        "checked_bond_count": checked_bonds,
        "bad_bond_length_fraction": bad_fraction,
        **pathology.as_row(),
        "ecfp4_top1_similarity": math.nan,
        "ecfp4_top2_similarity": math.nan,
        "ecfp4_top3_similarity": math.nan,
        "ecfp4_top5_mean_similarity": math.nan,
    }


def read_sdf(path: Path, method: str) -> tuple[list[dict[str, Any]], Counter[str]]:
    brenk = _catalog()
    guard = ChemistryGuardReward()
    rings = RingComplexityReward()
    rows: list[dict[str, Any]] = []
    elements: Counter[str] = Counter()
    supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
    for attempt, parsed in enumerate(supplier, start=1):
        if parsed is None:
            rows.append(_empty_row(method, attempt, False, "parse_failed"))
            continue
        mol = Chem.Mol(parsed)
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            rows.append(_empty_row(method, attempt, True, "sanitize_failed"))
            continue
        rows.append(_molecule_row(method, attempt, mol, brenk, guard, rings))
        elements.update(atom.GetSymbol() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)
    if len(rows) != ATTEMPTS:
        raise ValueError(f"Expected {ATTEMPTS} SDF records for {method}, got {len(rows)}: {path}")
    return rows, elements


def load_reference_fps(path: Path) -> list[Any]:
    with gzip.open(path, "rb") as handle:
        payload = pickle.load(handle)
    if int(payload.get("radius", -1)) != 2 or int(payload.get("n_bits", -1)) != 2048:
        raise ValueError("ChEMBL cache is not ECFP4 radius=2, 2048 bits")
    fingerprints = list(payload.get("fingerprints", []))
    if len(fingerprints) != 100_000:
        raise ValueError(f"Expected 100000 ChEMBL fingerprints, got {len(fingerprints)}")
    return fingerprints


def _init_similarity_worker(reference_fps: list[Any]) -> None:
    global REFERENCE_FPS
    REFERENCE_FPS = reference_fps


def _score_similarity(item: tuple[int, str]) -> tuple[int, float, float, float, float]:
    index, smiles = item
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return index, math.nan, math.nan, math.nan, math.nan
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    top = heapq.nlargest(5, DataStructs.BulkTanimotoSimilarity(fp, REFERENCE_FPS))
    return index, float(top[0]), float(top[1]), float(top[2]), float(np.mean(top))


def add_similarity(rows: list[dict[str, Any]], reference_fps: list[Any], workers: int) -> None:
    queries = [
        (index, str(row["canonical_smiles"]))
        for index, row in enumerate(rows)
        if row["sanitize_success"] == 1
    ]
    context = mp.get_context("fork")
    with context.Pool(workers, initializer=_init_similarity_worker, initargs=(reference_fps,)) as pool:
        for index, top1, top2, top3, top5 in pool.imap(_score_similarity, queries, chunksize=1):
            rows[index]["ecfp4_top1_similarity"] = top1
            rows[index]["ecfp4_top2_similarity"] = top2
            rows[index]["ecfp4_top3_similarity"] = top3
            rows[index]["ecfp4_top5_mean_similarity"] = top5


def _valid(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row["sanitize_success"] == 1]


def _values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray(
        [value for row in _valid(rows) if math.isfinite(value := _finite(row.get(key)))],
        dtype=float,
    )


def _internal_diversity(rows: list[dict[str, Any]]) -> float:
    unique = sorted({str(row["canonical_smiles"]) for row in _valid(rows)})
    fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, nBits=2048) for smiles in unique]
    if len(fps) < 2:
        return math.nan
    similarities: list[float] = []
    for index in range(1, len(fps)):
        similarities.extend(DataStructs.BulkTanimotoSimilarity(fps[index], fps[:index]))
    return 1.0 - float(np.mean(similarities))


def summary_row(method: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = _valid(rows)
    smiles = {str(row["canonical_smiles"]) for row in valid}
    scaffolds = {str(row["murcko_scaffold"]) for row in valid if row["murcko_scaffold"]}
    summary: dict[str, Any] = {
        "method": method,
        "attempts": ATTEMPTS,
        "sdf_records": len(rows),
        "parsed_records": sum(int(row["parsed"]) for row in rows),
        "sanitize_valid": len(valid),
        "unique_smiles": len(smiles),
        "unique_murcko_scaffolds_nonempty": len(scaffolds),
        "empty_murcko_scaffold_count": sum(not bool(row["murcko_scaffold"]) for row in valid),
        "ecfp4_internal_diversity_unique": _internal_diversity(rows),
    }
    for metric, (key, threshold, denominator) in PROPORTION_METRICS.items():
        if denominator == "attempts":
            summary[metric] = sum(float(row.get(key, 0)) >= float(threshold) for row in rows) / ATTEMPTS
        elif denominator == "valid_unique_smiles":
            summary[metric] = len(smiles) / len(valid) if valid else math.nan
        else:
            summary[metric] = (
                sum(float(row.get(key, 0)) >= float(threshold) for row in valid) / len(valid)
                if valid else math.nan
            )
    for key in CONTINUOUS_METRICS:
        values = _values(rows, key)
        summary[f"{key}_mean_valid"] = float(np.mean(values)) if values.size else math.nan
        summary[f"{key}_median_valid"] = float(np.median(values)) if values.size else math.nan
    return summary


def _wilson(count: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    p = count / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - half, center + half


def _proportion_count(rows: list[dict[str, Any]], metric: str) -> tuple[int, int]:
    key, threshold, denominator = PROPORTION_METRICS[metric]
    source = rows if denominator == "attempts" else _valid(rows)
    if denominator == "valid_unique_smiles":
        return len({str(row["canonical_smiles"]) for row in source}), len(source)
    return sum(float(row.get(key, 0)) >= float(threshold) for row in source), len(source)


def compare_proportions(original: list[dict[str, Any]], model: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for metric in PROPORTION_METRICS:
        count_o, n_o = _proportion_count(original, metric)
        count_m, n_m = _proportion_count(model, metric)
        rate_o = count_o / n_o if n_o else math.nan
        rate_m = count_m / n_m if n_m else math.nan
        low_o, high_o = _wilson(count_o, n_o)
        low_m, high_m = _wilson(count_m, n_m)
        output.append(
            {
                "metric": metric,
                "denominator_original": n_o,
                "count_original": count_o,
                "rate_original": rate_o,
                "denominator_model": n_m,
                "count_model": count_m,
                "rate_model": rate_m,
                "difference_model_minus_original": rate_m - rate_o,
                "difference_percentage_points": 100 * (rate_m - rate_o),
                "newcombe95_low": low_m - high_o,
                "newcombe95_high": high_m - low_o,
            }
        )
    return output


def compare_continuous(
    original: list[dict[str, Any]],
    model: list[dict[str, Any]],
    bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for metric in CONTINUOUS_METRICS:
        values_o = _values(original, metric)
        values_m = _values(model, metric)
        if not values_o.size or not values_m.size:
            continue
        mean_diff = float(np.mean(values_m) - np.mean(values_o))
        median_diff = float(np.median(values_m) - np.median(values_o))
        pooled_sd = math.sqrt(
            ((values_o.size - 1) * np.var(values_o, ddof=1) + (values_m.size - 1) * np.var(values_m, ddof=1))
            / (values_o.size + values_m.size - 2)
        )
        mean_boot = np.empty(bootstrap, dtype=float)
        median_boot = np.empty(bootstrap, dtype=float)
        for start in range(0, bootstrap, 250):
            stop = min(start + 250, bootstrap)
            size = stop - start
            sample_o = values_o[rng.integers(0, values_o.size, size=(size, values_o.size))]
            sample_m = values_m[rng.integers(0, values_m.size, size=(size, values_m.size))]
            mean_boot[start:stop] = sample_m.mean(axis=1) - sample_o.mean(axis=1)
            median_boot[start:stop] = np.median(sample_m, axis=1) - np.median(sample_o, axis=1)
        output.append(
            {
                "metric": metric,
                "n_original": values_o.size,
                "mean_original": float(np.mean(values_o)),
                "median_original": float(np.median(values_o)),
                "n_model": values_m.size,
                "mean_model": float(np.mean(values_m)),
                "median_model": float(np.median(values_m)),
                "mean_difference_model_minus_original": mean_diff,
                "mean_difference_bootstrap95_low": float(np.quantile(mean_boot, 0.025)),
                "mean_difference_bootstrap95_high": float(np.quantile(mean_boot, 0.975)),
                "median_difference_model_minus_original": median_diff,
                "median_difference_bootstrap95_low": float(np.quantile(median_boot, 0.025)),
                "median_difference_bootstrap95_high": float(np.quantile(median_boot, 0.975)),
                "cohens_d": mean_diff / pooled_sd if pooled_sd > 0 else math.nan,
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-sdf", type=Path, required=True)
    parser.add_argument("--model-sdf", type=Path, required=True)
    parser.add_argument("--chembl-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=260902)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    original, elements_o = read_sdf(args.original_sdf, METHODS[0])
    model, elements_m = read_sdf(args.model_sdf, METHODS[1])
    reference_fps = load_reference_fps(args.chembl_cache)
    add_similarity(original, reference_fps, args.workers)
    add_similarity(model, reference_fps, args.workers)

    summaries = [summary_row(METHODS[0], original), summary_row(METHODS[1], model)]
    proportion_comparisons = compare_proportions(original, model)
    continuous_comparisons = compare_continuous(original, model, args.bootstrap, args.seed)
    element_rows: list[dict[str, Any]] = []
    for method, elements in zip(METHODS, (elements_o, elements_m)):
        total = sum(elements.values())
        for element, count in sorted(elements.items()):
            element_rows.append(
                {"method": method, "element": element, "atom_count": count, "atom_fraction": count / total}
            )

    write_csv(args.output_dir / "per_attempt_metrics.csv", original + model)
    write_csv(args.output_dir / "generation_summary.csv", summaries)
    write_csv(args.output_dir / "proportion_comparisons.csv", proportion_comparisons)
    write_csv(args.output_dir / "continuous_comparisons.csv", continuous_comparisons)
    write_csv(args.output_dir / "element_distribution.csv", element_rows)
    manifest = {
        "status": "completed",
        "attempts_per_method": ATTEMPTS,
        "methods": list(METHODS),
        "original_sdf": str(args.original_sdf.resolve()),
        "model_sdf": str(args.model_sdf.resolve()),
        "chembl_cache": str(args.chembl_cache.resolve()),
        "chembl_fingerprints": len(reference_fps),
        "fingerprint": {"type": "ECFP4", "radius": 2, "n_bits": 2048},
        "workers": args.workers,
        "bootstrap_resamples": args.bootstrap,
        "bootstrap_seed": args.seed,
        "primary_endpoint": "r1_r4_pathology_rate_valid",
        "r5_reported_separately": True,
    }
    (args.output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "completed", "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
