#!/usr/bin/env python3
"""Recompute raw benchmark SDF quality metrics against the ChEMBL36 baseline."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import os
import sys
from multiprocessing import Pool
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/diffb110_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


LOGGER = logging.getLogger("raw_sdf_quality_vs_chembl")
ROOT = Path(__file__).resolve().parents[1]
REFERENCE_SCRIPT = (
    ROOT
    / "analysis"
    / "chembl_property_ring_distribution_20260624"
    / "analyze_chembl_property_ring_distribution.py"
)
DEFAULT_FORMAL_DIR = (
    ROOT
    / "analysis"
    / "benchmark"
    / "benchmark5"
    / "8gjl_c1101_benchmark_20260706_1000steps_3rep_formal"
)
DEFAULT_CHEMBL_DIR = ROOT / "analysis" / "chembl_standardized_all_quality_20260728"

REFERENCE_MODULE: ModuleType | None = None
RDLogger.DisableLog("rdApp.error")

RING_COLUMNS = [
    "total_ring_count",
    "aromatic_ring_count",
    "aliphatic_ring_count",
    "ring_3_count",
    "ring_4_count",
    "ring_5_count",
    "ring_6_count",
    "ring_7_count",
    "large_ring_count",
    "benzene_ring_count",
    "heterocycle_count",
    "aromatic_heterocycle_count",
    "fused_ring_system_count",
    "spiro_atom_count",
    "bridgehead_atom_count",
    "max_ring_size",
    "ring_system_count",
    "max_ring_system_size",
    "max_fused_ring_system_size",
]

EXTRA_NUMERIC_COLUMNS = [
    "ring_density",
    "aromatic_atom_fraction",
    "num_fragments",
    "largest_fragment_atom_fraction",
]

INDICATORS: dict[str, tuple[str, Callable[[pd.DataFrame], pd.Series]]] = {
    "three_member_ring": ("3-member ring >=1", lambda frame: frame["ring_3_count"] >= 1),
    "four_member_ring": ("4-member ring >=1", lambda frame: frame["ring_4_count"] >= 1),
    "large_ring": ("ring size >=8", lambda frame: frame["large_ring_count"] >= 1),
    "six_or_more_rings": ("total rings >=6", lambda frame: frame["total_ring_count"] >= 6),
    "large_fused_system": (
        "largest fused system >=3 rings",
        lambda frame: frame["max_fused_ring_system_size"] >= 3,
    ),
    "all_carbon": ("all-carbon heavy atoms", lambda frame: frame["all_carbon"] >= 1),
    "carbon_enriched_low_hetero": (
        "C >=90% and hetero <=1",
        lambda frame: frame["carbon_enriched_low_hetero"] >= 1,
    ),
    "sa_gt_6": ("SA score >6", lambda frame: frame["sa_score"] > 6),
    "pains_hit": ("PAINS hit", lambda frame: frame["pains_alert_count"] > 0),
    "nih_hit": ("NIH hit", lambda frame: frame["nih_alert_count"] > 0),
    "brenk_hit": ("Brenk hit", lambda frame: frame["brenk_alert_count"] > 0),
    "charged": ("formal charge !=0", lambda frame: frame["formal_charge"] != 0),
    "radical": ("radical electrons >0", lambda frame: frame["radical_electron_count"] > 0),
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources-csv",
        type=Path,
        default=DEFAULT_FORMAL_DIR / "formal_3rep_sources.csv",
        help="CSV with repeat, model, and source_sdf columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_FORMAL_DIR / "raw_quality_vs_chembl_20260728",
        help="Output directory for per-molecule tables, summaries, plots, and report.",
    )
    parser.add_argument(
        "--chembl-dir",
        type=Path,
        default=DEFAULT_CHEMBL_DIR,
        help="Completed ChEMBL quality baseline directory.",
    )
    parser.add_argument("--workers", type=int, default=18, help="RDKit worker count.")
    parser.add_argument(
        "--attempts-per-source",
        type=int,
        default=1000,
        help="Fixed generation-attempt denominator for every repeat/model source.",
    )
    parser.add_argument("--max-sources", type=int, default=None, help="Optional source cap for smoke tests.")
    parser.add_argument(
        "--max-records-per-source",
        type=int,
        default=None,
        help="Optional SDF record cap per source for smoke tests.",
    )
    return parser.parse_args()


def _load_reference_module() -> ModuleType:
    """Load the completed ChEMBL feature implementation under a stable module name."""
    module_name = "diffb110_chembl_quality_reference"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, REFERENCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load reference metric implementation: {REFERENCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _init_worker() -> None:
    """Initialize worker-local SA scorer and structural-alert catalogs."""
    global REFERENCE_MODULE
    REFERENCE_MODULE = _load_reference_module()
    REFERENCE_MODULE._init_worker(True)


def _full_features(smiles: str) -> dict[str, Any]:
    """Compute the ChEMBL-aligned descriptor panel for one canonical SMILES."""
    if REFERENCE_MODULE is None:
        _init_worker()
    assert REFERENCE_MODULE is not None
    features = REFERENCE_MODULE._compute_rdkit_features(smiles)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return features
    heavy_atoms = int(mol.GetNumHeavyAtoms())
    carbon_atoms = sum(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms())
    formal_charge = int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))
    mol_wt = float(Descriptors.MolWt(mol))
    mol_log_p = float(Crippen.MolLogP(mol))
    hba = int(Lipinski.NumHAcceptors(mol))
    hbd = int(Lipinski.NumHDonors(mol))
    fragments = Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
    largest_fragment = max((len(fragment) for fragment in fragments), default=0)
    aromatic_atoms = sum(atom.GetIsAromatic() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)
    total_rings = int(features.get("total_ring_count", 0))
    features.update(
        {
            "heavy_atoms": heavy_atoms,
            "formal_charge": formal_charge,
            "num_rings": int(rdMolDescriptors.CalcNumRings(mol)),
            "mol_wt": mol_wt,
            "mol_log_p": mol_log_p,
            "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
            "num_h_acceptors": hba,
            "num_h_donors": hbd,
            "num_rotatable_bonds": int(Lipinski.NumRotatableBonds(mol)),
            "num_heteroatoms": int(rdMolDescriptors.CalcNumHeteroatoms(mol)),
            "fraction_csp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
            "num_carbon_atoms": int(carbon_atoms),
            "carbon_fraction": carbon_atoms / heavy_atoms if heavy_atoms else math.nan,
            "lipinski_violations": int(mol_wt > 500)
            + int(mol_log_p > 5)
            + int(hba > 10)
            + int(hbd > 5),
            "ring_density": total_rings / heavy_atoms if heavy_atoms else math.nan,
            "aromatic_atom_fraction": aromatic_atoms / heavy_atoms if heavy_atoms else math.nan,
            "num_fragments": len(fragments),
            "largest_fragment_atom_fraction": largest_fragment / heavy_atoms if heavy_atoms else math.nan,
        }
    )
    return features


def _resolve_source_path(raw_path: str) -> Path:
    """Resolve equivalent project mount prefixes without changing source provenance."""
    path = Path(raw_path)
    if path.exists():
        return path
    alternate = Path(str(path).replace("/home/md666/data/disk6/DiffB110", "/data/disk6/DiffB110"))
    if alternate.exists():
        return alternate
    raise FileNotFoundError(f"missing source SDF: {path}")


def _count_sdf_records(path: Path) -> int:
    """Count explicit SDF record delimiters independently of RDKit parsing."""
    count = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip() == "$$$$":
                count += 1
    return count


def _read_raw_sources(
    sources: pd.DataFrame,
    max_records_per_source: int | None,
) -> tuple[pd.DataFrame, list[tuple[int, str]], list[dict[str, Any]]]:
    """Read raw SDF records unsanitized and preserve all parser/sanitize denominators."""
    raw_rows: list[dict[str, Any]] = []
    feature_jobs: list[tuple[int, str]] = []
    source_facts: list[dict[str, Any]] = []
    row_id = 0
    for source in sources.itertuples(index=False):
        path = _resolve_source_path(str(source.source_sdf))
        explicit_records = _count_sdf_records(path)
        supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False, strictParsing=False)
        parsed_records = 0
        for source_index, mol in enumerate(supplier):
            if max_records_per_source is not None and source_index >= max_records_per_source:
                break
            parsed_records += 1
            row: dict[str, Any] = {
                "row_id": row_id,
                "repeat": str(source.repeat),
                "model": str(source.model),
                "source_sdf": str(path),
                "source_index": int(source_index),
                "rdkit_readable": int(mol is not None),
                "sanitize_success": 0,
                "sanitize_error": "",
                "canonical_smiles": "",
            }
            if mol is not None:
                clean = Chem.Mol(mol)
                try:
                    Chem.SanitizeMol(clean)
                    canonical = Chem.MolToSmiles(Chem.RemoveHs(clean), isomericSmiles=True, canonical=True)
                    row["sanitize_success"] = 1
                    row["canonical_smiles"] = canonical
                    feature_jobs.append((row_id, canonical))
                except Exception as exc:
                    row["sanitize_error"] = f"{type(exc).__name__}: {exc}"
            raw_rows.append(row)
            row_id += 1
        source_facts.append(
            {
                "repeat": str(source.repeat),
                "model": str(source.model),
                "source_sdf": str(path),
                "explicit_sdf_records": explicit_records,
                "records_read_for_analysis": parsed_records,
            }
        )
        LOGGER.info(
            "read %s/%s records=%s explicit=%s valid_jobs=%s",
            source.repeat,
            source.model,
            parsed_records,
            explicit_records,
            sum(job[0] >= row_id - parsed_records for job in feature_jobs),
        )
    return pd.DataFrame(raw_rows), feature_jobs, source_facts


def _compute_features(
    rows: pd.DataFrame,
    feature_jobs: list[tuple[int, str]],
    workers: int,
) -> pd.DataFrame:
    """Compute all valid-molecule features and merge them onto preserved raw records."""
    smiles = [job[1] for job in feature_jobs]
    with Pool(processes=workers, initializer=_init_worker) as pool:
        computed = list(pool.imap(_full_features, smiles, chunksize=64))
    feature_frame = pd.DataFrame(computed)
    feature_frame.insert(0, "row_id", [job[0] for job in feature_jobs])
    merged = rows.merge(feature_frame, on="row_id", how="left", validate="one_to_one")
    merged["exact_duplicate"] = 0
    merged["murcko_scaffold_duplicate"] = 0
    for _key, group in merged.groupby(["repeat", "model"], sort=False):
        valid = group[group["sanitize_success"] == 1]
        merged.loc[valid.index, "exact_duplicate"] = (
            valid["canonical_smiles"].duplicated(keep="first").astype(int)
        )
        nonempty_scaffolds = valid[valid["scaffold_smiles"].fillna("").astype(str) != ""]
        merged.loc[nonempty_scaffolds.index, "murcko_scaffold_duplicate"] = (
            nonempty_scaffolds["scaffold_smiles"].duplicated(keep="first").astype(int)
        )
    return merged


def _numeric_columns(reference: ModuleType, frame: pd.DataFrame) -> list[str]:
    """Return the complete numeric metric panel present in the generated table."""
    ordered = list(reference.PROPERTY_COLUMNS) + RING_COLUMNS + EXTRA_NUMERIC_COLUMNS
    return [column for column in dict.fromkeys(ordered) if column in frame.columns]


def _source_summary(
    frame: pd.DataFrame,
    source_facts: list[dict[str, Any]],
    attempts_per_source: int,
) -> pd.DataFrame:
    """Summarize generation integrity with fixed-attempt denominators."""
    facts = {(row["repeat"], row["model"]): row for row in source_facts}
    rows: list[dict[str, Any]] = []
    for (repeat, model), group in frame.groupby(["repeat", "model"], sort=False):
        fact = facts[(repeat, model)]
        valid = group[group["sanitize_success"] == 1]
        record_count = int(fact["explicit_sdf_records"])
        unique_smiles = int(valid["canonical_smiles"].nunique())
        nonempty_scaffolds = valid["scaffold_smiles"].fillna("").astype(str)
        nonempty_scaffolds = nonempty_scaffolds[nonempty_scaffolds != ""]
        rows.append(
            {
                "repeat": repeat,
                "model": model,
                "attempts": attempts_per_source,
                "sdf_records": record_count,
                "not_written": max(attempts_per_source - record_count, 0),
                "rdkit_readable": int(group["rdkit_readable"].sum()),
                "sanitize_valid": int(group["sanitize_success"].sum()),
                "sanitize_failed": int((group["rdkit_readable"] - group["sanitize_success"]).sum()),
                "exact_unique_smiles": unique_smiles,
                "exact_duplicates": int(valid["exact_duplicate"].sum()),
                "unique_nonempty_murcko_scaffolds": int(nonempty_scaffolds.nunique()),
                "murcko_scaffold_duplicates": int(valid["murcko_scaffold_duplicate"].sum()),
                "written_per_attempt": record_count / attempts_per_source,
                "sanitize_valid_per_attempt": len(valid) / attempts_per_source,
                "exact_unique_per_attempt": unique_smiles / attempts_per_source,
                "exact_duplicate_rate_valid": int(valid["exact_duplicate"].sum()) / max(len(valid), 1),
                "murcko_scaffold_duplicate_rate_nonempty": int(
                    valid["murcko_scaffold_duplicate"].sum()
                )
                / max(len(nonempty_scaffolds), 1),
                "source_sdf": str(fact["source_sdf"]),
            }
        )
    return pd.DataFrame(rows)


def _metric_summary(frame: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    """Write replicate-level and pooled model distributions for every numeric metric."""
    rows: list[dict[str, Any]] = []

    def add(scope: str, model: str, repeat: str, group: pd.DataFrame) -> None:
        valid = group[group["sanitize_success"] == 1]
        for metric in numeric_columns:
            values = pd.to_numeric(valid[metric], errors="coerce").dropna()
            if values.empty:
                continue
            quantiles = values.quantile([0.01, 0.05, 0.5, 0.95, 0.99])
            rows.append(
                {
                    "scope": scope,
                    "model": model,
                    "repeat": repeat,
                    "metric": metric,
                    "count": int(len(values)),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                    "p01": float(quantiles.loc[0.01]),
                    "p05": float(quantiles.loc[0.05]),
                    "median": float(quantiles.loc[0.5]),
                    "p95": float(quantiles.loc[0.95]),
                    "p99": float(quantiles.loc[0.99]),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )

    for (repeat, model), group in frame.groupby(["repeat", "model"], sort=False):
        add("repeat", model, repeat, group)
    for model, group in frame.groupby("model", sort=False):
        add("model_pooled", model, "pooled", group)
    return pd.DataFrame(rows)


def _model_integrity_summary(source_summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fixed-attempt integrity counts and repeat variability by model."""
    count_columns = [
        "attempts",
        "sdf_records",
        "not_written",
        "rdkit_readable",
        "sanitize_valid",
        "sanitize_failed",
        "exact_unique_smiles",
        "exact_duplicates",
        "unique_nonempty_murcko_scaffolds",
        "murcko_scaffold_duplicates",
    ]
    rows: list[dict[str, Any]] = []
    for model, group in source_summary.groupby("model", sort=False):
        row: dict[str, Any] = {"model": model, "repeats": int(len(group))}
        for column in count_columns:
            row[column] = int(group[column].sum())
        attempts = max(row["attempts"], 1)
        valid = max(row["sanitize_valid"], 1)
        row["written_per_attempt"] = row["sdf_records"] / attempts
        row["sanitize_valid_per_attempt"] = row["sanitize_valid"] / attempts
        row["exact_unique_per_attempt"] = row["exact_unique_smiles"] / attempts
        row["exact_duplicate_rate_valid"] = row["exact_duplicates"] / valid
        for column in ["written_per_attempt", "sanitize_valid_per_attempt", "exact_unique_per_attempt"]:
            row[f"{column}_repeat_std"] = float(group[column].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def _metric_repeat_summary(metric_summary: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-repeat property means and medians as mean plus sample SD."""
    repeat_rows = metric_summary[metric_summary["scope"] == "repeat"]
    rows: list[dict[str, Any]] = []
    for (model, metric), group in repeat_rows.groupby(["model", "metric"], sort=False):
        rows.append(
            {
                "model": model,
                "metric": metric,
                "n_repeats": int(group["repeat"].nunique()),
                "repeat_mean": float(group["mean"].mean()),
                "repeat_sd": float(group["mean"].std(ddof=1)),
                "repeat_min": float(group["mean"].min()),
                "repeat_max": float(group["mean"].max()),
                "repeat_median_mean": float(group["median"].mean()),
                "repeat_median_sd": float(group["median"].std(ddof=1)),
            }
        )
    return pd.DataFrame(rows)


def _chembl_tail_rates(
    frame: pd.DataFrame,
    chembl_property_summary: Path,
) -> pd.DataFrame:
    """Calculate generated-molecule rates outside uncropped ChEMBL P1-P99 ranges."""
    reference = pd.read_csv(chembl_property_summary).set_index("property")
    rows: list[dict[str, Any]] = []

    def add(scope: str, model: str, repeat: str, group: pd.DataFrame) -> None:
        valid = group[group["sanitize_success"] == 1]
        for metric in reference.index:
            if metric not in valid.columns:
                continue
            values = pd.to_numeric(valid[metric], errors="coerce").dropna()
            if values.empty:
                continue
            lower = float(reference.loc[metric, "p01"])
            upper = float(reference.loc[metric, "p99"])
            below = float((values < lower).mean())
            above = float((values > upper).mean())
            rows.append(
                {
                    "scope": scope,
                    "model": model,
                    "repeat": repeat,
                    "metric": metric,
                    "count": int(len(values)),
                    "chembl_p01": lower,
                    "chembl_p99": upper,
                    "below_chembl_p01_rate": below,
                    "above_chembl_p99_rate": above,
                    "outside_chembl_p01_p99_rate": below + above,
                }
            )

    for (repeat, model), group in frame.groupby(["repeat", "model"], sort=False):
        add("repeat", model, repeat, group)
    for model, group in frame.groupby("model", sort=False):
        add("model_pooled", model, "pooled", group)
    return pd.DataFrame(rows)


def _tail_repeat_summary(tails: pd.DataFrame) -> pd.DataFrame:
    """Summarize ChEMBL-tail rates across the three repeats with sample SD."""
    repeat_rows = tails[tails["scope"] == "repeat"]
    rows: list[dict[str, Any]] = []
    rate_columns = [
        "below_chembl_p01_rate",
        "above_chembl_p99_rate",
        "outside_chembl_p01_p99_rate",
    ]
    for (model, metric), group in repeat_rows.groupby(["model", "metric"], sort=False):
        row: dict[str, Any] = {
            "model": model,
            "metric": metric,
            "n_repeats": int(group["repeat"].nunique()),
            "chembl_p01": float(group["chembl_p01"].iloc[0]),
            "chembl_p99": float(group["chembl_p99"].iloc[0]),
        }
        for column in rate_columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_sd"] = float(group[column].std(ddof=1))
            row[f"{column}_min"] = float(group[column].min())
            row[f"{column}_max"] = float(group[column].max())
        rows.append(row)
    return pd.DataFrame(rows)


def _chembl_indicator_rates(chembl_features: Path, halogen_p99: float) -> dict[str, float]:
    """Stream exact ChEMBL rates for structural-pathology indicators."""
    required = sorted(
        {
            column
            for column in [
                "ring_3_count",
                "ring_4_count",
                "large_ring_count",
                "total_ring_count",
                "max_fused_ring_system_size",
                "all_carbon",
                "carbon_enriched_low_hetero",
                "sa_score",
                "pains_alert_count",
                "nih_alert_count",
                "brenk_alert_count",
                "formal_charge",
                "radical_electron_count",
                "halogen_fraction",
                "rdkit_valid",
            ]
        }
    )
    sums = {name: 0 for name in INDICATORS}
    sums["halogen_above_chembl_p99"] = 0
    total = 0
    for chunk in pd.read_csv(chembl_features, usecols=required, chunksize=200000):
        chunk = chunk[chunk["rdkit_valid"] == 1]
        total += len(chunk)
        for name, (_label, predicate) in INDICATORS.items():
            sums[name] += int(predicate(chunk).fillna(False).sum())
        sums["halogen_above_chembl_p99"] += int((chunk["halogen_fraction"] > halogen_p99).sum())
    return {name: count / max(total, 1) for name, count in sums.items()}


def _indicator_table(
    frame: pd.DataFrame,
    chembl_features: Path,
    chembl_property_summary: Path,
) -> pd.DataFrame:
    """Compare pooled model pathology rates with the full ChEMBL reference."""
    property_summary = pd.read_csv(chembl_property_summary).set_index("property")
    halogen_p99 = float(property_summary.loc["halogen_fraction", "p99"])
    chembl_rates = _chembl_indicator_rates(chembl_features, halogen_p99)
    definitions = dict(INDICATORS)
    definitions["halogen_above_chembl_p99"] = (
        f"halogen fraction > ChEMBL P99 ({halogen_p99:.4g})",
        lambda data: data["halogen_fraction"] > halogen_p99,
    )
    rows: list[dict[str, Any]] = []
    for name, (label, _predicate) in definitions.items():
        rows.append(
            {
                "scope": "reference",
                "dataset": "ChEMBL36-standardized-all",
                "model": "ChEMBL36",
                "repeat": "reference",
                "indicator": name,
                "label": label,
                "count_valid": 2734034,
                "rate": chembl_rates[name],
            }
        )
    for (repeat, model), group in frame.groupby(["repeat", "model"], sort=False):
        valid = group[group["sanitize_success"] == 1]
        for name, (label, predicate) in definitions.items():
            rows.append(
                {
                    "scope": "repeat",
                    "dataset": "raw_generated_sdf",
                    "model": model,
                    "repeat": repeat,
                    "indicator": name,
                    "label": label,
                    "count_valid": int(len(valid)),
                    "rate": float(predicate(valid).fillna(False).mean()),
                }
            )
    for model, group in frame.groupby("model", sort=False):
        valid = group[group["sanitize_success"] == 1]
        for name, (label, predicate) in definitions.items():
            rows.append(
                {
                    "scope": "model_pooled",
                    "dataset": "raw_generated_sdf",
                    "model": model,
                    "repeat": "pooled",
                    "indicator": name,
                    "label": label,
                    "count_valid": int(len(valid)),
                    "rate": float(predicate(valid).fillna(False).mean()),
                }
            )
    return pd.DataFrame(rows)


def _indicator_repeat_summary(indicators: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-repeat pathology rates as mean plus sample SD."""
    repeat_rows = indicators[indicators["scope"] == "repeat"]
    rows: list[dict[str, Any]] = []
    for (model, indicator, label), group in repeat_rows.groupby(
        ["model", "indicator", "label"], sort=False
    ):
        rows.append(
            {
                "model": model,
                "indicator": indicator,
                "label": label,
                "n_repeats": int(group["repeat"].nunique()),
                "rate_mean": float(group["rate"].mean()),
                "rate_sd": float(group["rate"].std(ddof=1)),
                "rate_min": float(group["rate"].min()),
                "rate_max": float(group["rate"].max()),
            }
        )
    return pd.DataFrame(rows)


def _plot_indicator_heatmap(indicators: pd.DataFrame, output_path: Path) -> None:
    """Plot pooled indicator percentages by dataset/model."""
    plotted = indicators[indicators["scope"].isin(["reference", "model_pooled"])]
    matrix = plotted.pivot(index="model", columns="label", values="rate") * 100.0
    order = [name for name in ["ChEMBL36", "DiffGui", "DiffSBDD", "PMDM", "GCDM-SBDD", "TargetDiff"] if name in matrix.index]
    matrix = matrix.reindex(order)
    fig, ax = plt.subplots(figsize=(13.0, 4.8))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="magma", vmin=0)
    ax.set_xticks(np.arange(len(matrix.columns)), labels=matrix.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)), labels=matrix.index)
    ax.set_title("Structural-pathology prevalence: raw generated SDFs vs ChEMBL36")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("% sanitize-valid molecules")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_tail_heatmap(tails: pd.DataFrame, output_path: Path) -> None:
    """Plot model rates outside the uncropped ChEMBL P1-P99 range."""
    pooled = tails[tails["scope"] == "model_pooled"]
    matrix = pooled.pivot(index="model", columns="metric", values="outside_chembl_p01_p99_rate") * 100.0
    order = [name for name in ["DiffGui", "DiffSBDD", "PMDM", "GCDM-SBDD", "TargetDiff"] if name in matrix.index]
    matrix = matrix.reindex(order)
    fig, ax = plt.subplots(figsize=(13.0, 4.2))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="viridis", vmin=0)
    ax.set_xticks(np.arange(len(matrix.columns)), labels=matrix.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)), labels=matrix.index)
    ax.set_title("Raw generated molecules outside ChEMBL36 P1-P99")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("% sanitize-valid molecules")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _write_report(
    output_dir: Path,
    source_summary: pd.DataFrame,
    model_integrity: pd.DataFrame,
    indicators: pd.DataFrame,
    indicator_repeat_summary: pd.DataFrame,
) -> None:
    """Write a concise Markdown report with embedded tables and figures."""
    selected = indicator_repeat_summary[
        indicator_repeat_summary["indicator"].isin(
            ["all_carbon", "carbon_enriched_low_hetero", "six_or_more_rings", "sa_gt_6", "brenk_hit"]
        )
    ]
    selected = selected.copy()
    selected["mean_sd_percent"] = selected.apply(
        lambda row: f"{100.0 * row['rate_mean']:.2f} +/- {100.0 * row['rate_sd']:.2f}", axis=1
    )
    selected_table = selected.pivot(index="model", columns="indicator", values="mean_sd_percent")
    lines = [
        "# Raw Generated SDF Quality Versus ChEMBL36",
        "",
        "## Scope",
        "",
        "This analysis uses the 15 original generated SDF files (five models x three repeats). "
        "No sanitize filtering, docking pose, or refill sampling is used before denominator accounting.",
        "",
        "## Generation Integrity",
        "",
        _markdown_table(model_integrity),
        "",
        "### Repeat-level details",
        "",
        _markdown_table(source_summary),
        "",
        "## Structural Pathology",
        "",
        "![Structural pathology prevalence](indicator_rate_heatmap.png)",
        "",
        _markdown_table(selected_table.reset_index()),
        "",
        "## ChEMBL Tail Rates",
        "",
        "![Rates outside ChEMBL P1-P99](chembl_tail_rate_heatmap.png)",
        "",
        "## Evidence Files",
        "",
        "- [Per-molecule metrics](per_molecule_raw_quality.csv.gz)",
        "- [Per-source integrity](source_integrity_summary.csv)",
        "- [Model integrity](model_integrity_summary.csv)",
        "- [All metric distributions](metric_summary_long.csv)",
        "- [Property mean and SD across repeats](metric_repeat_summary.csv)",
        "- [ChEMBL tail rates](chembl_tail_rates.csv)",
        "- [ChEMBL tail-rate mean and SD](chembl_tail_repeat_summary.csv)",
        "- [Indicator comparison](indicator_rates_vs_chembl.csv)",
        "- [Indicator-rate mean and SD](indicator_repeat_summary.csv)",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without the optional tabulate package."""
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.4f}"
            )
    headers = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        values = [str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    """Run the raw-SDF quality recomputation and ChEMBL comparison."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_format = "%(asctime)s %(levelname)s %(name)s %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.output_dir / "raw_quality_vs_chembl.log", mode="w", encoding="utf-8"),
        ],
    )
    LOGGER.info(
        "start sources_csv=%s chembl_dir=%s workers=%s attempts_per_source=%s output=%s",
        args.sources_csv,
        args.chembl_dir,
        args.workers,
        args.attempts_per_source,
        args.output_dir,
    )
    sources = pd.read_csv(args.sources_csv)
    required_columns = {"repeat", "model", "source_sdf"}
    if not required_columns.issubset(sources.columns):
        raise ValueError(f"sources CSV missing columns: {sorted(required_columns - set(sources.columns))}")
    if args.max_sources is not None:
        sources = sources.head(args.max_sources)
    reference = _load_reference_module()
    raw_rows, feature_jobs, source_facts = _read_raw_sources(sources, args.max_records_per_source)
    LOGGER.info("feature jobs=%s raw rows=%s workers=%s", len(feature_jobs), len(raw_rows), args.workers)
    full = _compute_features(raw_rows, feature_jobs, args.workers)
    full.to_csv(args.output_dir / "per_molecule_raw_quality.csv.gz", index=False, compression="gzip")

    source_summary = _source_summary(full, source_facts, args.attempts_per_source)
    source_summary.to_csv(args.output_dir / "source_integrity_summary.csv", index=False)
    model_integrity = _model_integrity_summary(source_summary)
    model_integrity.to_csv(args.output_dir / "model_integrity_summary.csv", index=False)
    numeric_columns = _numeric_columns(reference, full)
    metric_summary = _metric_summary(full, numeric_columns)
    metric_summary.to_csv(args.output_dir / "metric_summary_long.csv", index=False)
    metric_repeat_summary = _metric_repeat_summary(metric_summary)
    metric_repeat_summary.to_csv(args.output_dir / "metric_repeat_summary.csv", index=False)

    chembl_property_summary = args.chembl_dir / "property_summary.csv"
    chembl_features = args.chembl_dir / "chembl_36_per_molecule_ring_features.csv.gz"
    tails = _chembl_tail_rates(full, chembl_property_summary)
    tails.to_csv(args.output_dir / "chembl_tail_rates.csv", index=False)
    tail_repeat_summary = _tail_repeat_summary(tails)
    tail_repeat_summary.to_csv(args.output_dir / "chembl_tail_repeat_summary.csv", index=False)
    indicators = _indicator_table(full, chembl_features, chembl_property_summary)
    indicators.to_csv(args.output_dir / "indicator_rates_vs_chembl.csv", index=False)
    indicator_repeat_summary = _indicator_repeat_summary(indicators)
    indicator_repeat_summary.to_csv(args.output_dir / "indicator_repeat_summary.csv", index=False)

    _plot_indicator_heatmap(indicators, args.output_dir / "indicator_rate_heatmap.png")
    _plot_indicator_heatmap(indicators, args.output_dir / "indicator_rate_heatmap.svg")
    _plot_tail_heatmap(tails, args.output_dir / "chembl_tail_rate_heatmap.png")
    _plot_tail_heatmap(tails, args.output_dir / "chembl_tail_rate_heatmap.svg")
    _write_report(
        args.output_dir,
        source_summary,
        model_integrity,
        indicators,
        indicator_repeat_summary,
    )

    summary = {
        "sources": int(len(sources)),
        "attempts_per_source": int(args.attempts_per_source),
        "raw_sdf_records_analyzed": int(len(full)),
        "sanitize_valid": int(full["sanitize_success"].sum()),
        "workers": int(args.workers),
        "reference_metric_script": str(REFERENCE_SCRIPT),
        "chembl_reference": str(args.chembl_dir),
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("completed output=%s", args.output_dir)


if __name__ == "__main__":
    main()
