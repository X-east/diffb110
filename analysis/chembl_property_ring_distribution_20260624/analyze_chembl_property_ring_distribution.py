#!/usr/bin/env python3
"""Summarize ChEMBL property, ring-count, scaffold, and diversity distributions."""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import os
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/diffb110_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, FilterCatalog, QED, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


LOGGER = logging.getLogger("chembl_property_ring_distribution")

ROOT = Path(__file__).resolve().parents[2]
SA_SCORE_DIR = ROOT / "DiffSBDD" / "analysis" / "SA_Score"

COUNT_BINS = ("0", "1", "2", "3", "4", "5", ">=6")
RING_COUNT_FEATURES = (
    ("total_ring_count", "Total rings"),
    ("aromatic_ring_count", "Aromatic rings"),
    ("aliphatic_ring_count", "Aliphatic rings"),
    ("ring_3_count", "3-member rings"),
    ("ring_4_count", "4-member rings"),
    ("ring_5_count", "5-member rings"),
    ("ring_6_count", "6-member rings"),
    ("ring_7_count", "7-member rings"),
    ("large_ring_count", "Large rings (size >=8)"),
    ("benzene_ring_count", "Benzene rings"),
    ("heterocycle_count", "Heterocycles"),
    ("aromatic_heterocycle_count", "Aromatic heterocycles"),
    ("fused_ring_system_count", "Fused ring systems"),
    ("spiro_atom_count", "Spiro atoms"),
    ("bridgehead_atom_count", "Bridgehead atoms"),
    ("ring_system_count", "Ring systems"),
    ("max_ring_system_size", "Largest ring system (rings)"),
    ("max_fused_ring_system_size", "Largest fused system (rings)"),
)
PROPERTY_COLUMNS = (
    "mol_wt",
    "mol_log_p",
    "tpsa",
    "num_h_acceptors",
    "num_h_donors",
    "num_rotatable_bonds",
    "heavy_atoms",
    "formal_charge",
    "num_heteroatoms",
    "carbon_fraction",
    "fraction_csp3",
    "lipinski_violations",
    "qed",
    "sa_score",
    "bertz_complexity",
    "element_diversity",
    "radical_electron_count",
    "halogen_fraction",
    "all_carbon",
    "carbon_enriched_low_hetero",
    "pains_alert_count",
    "nih_alert_count",
    "brenk_alert_count",
)

_SASCORER: Any | None = None
_FILTER_CATALOGS: dict[str, Any] = {}


@dataclass
class Paths:
    """Output paths produced by the analysis."""

    output_dir: Path
    per_molecule: Path
    ring_distribution: Path
    property_summary: Path
    scaffold_summary: Path
    scaffold_top: Path
    diversity_summary: Path
    summary_json: Path
    report_md: Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "chembl" / "chembl_36_properties.csv.gz",
        help="Input ChEMBL properties CSV containing canonical_smiles.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "analysis" / "chembl_property_ring_distribution_20260624",
        help="Directory for CSV, plots, and Markdown report.",
    )
    parser.add_argument("--workers", type=int, default=18, help="Parallel RDKit worker count.")
    parser.add_argument("--chunksize", type=int, default=20000, help="Rows per pandas chunk.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row cap for smoke tests.")
    parser.add_argument("--force", action="store_true", help="Recompute per-molecule features if present.")
    parser.add_argument("--no-sa", action="store_true", help="Skip SA score calculation.")
    parser.add_argument("--similarity-sample", type=int, default=50000, help="Sample size for ECFP4 diversity.")
    parser.add_argument("--similarity-pairs", type=int, default=200000, help="Random fingerprint pairs.")
    return parser.parse_args()


def main() -> None:
    """Run the complete analysis."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    paths = _paths(args.output_dir)
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    if args.force or not paths.per_molecule.exists():
        _write_per_molecule_features(args, paths.per_molecule)
    else:
        LOGGER.info("reusing existing per-molecule features: %s", paths.per_molecule)

    features = pd.read_csv(paths.per_molecule)
    valid = features[features["rdkit_valid"] == 1].copy()
    LOGGER.info("loaded features rows=%s valid=%s", len(features), len(valid))

    ring_distribution = _ring_distribution(valid)
    ring_distribution.to_csv(paths.ring_distribution, index=False)

    property_summary = _property_summary(valid)
    property_summary.to_csv(paths.property_summary, index=False)

    scaffold_summary, scaffold_top = _scaffold_tables(valid)
    pd.DataFrame([scaffold_summary]).to_csv(paths.scaffold_summary, index=False)
    scaffold_top.to_csv(paths.scaffold_top, index=False)

    diversity_summary = _fingerprint_diversity(valid, args.similarity_sample, args.similarity_pairs)
    pd.DataFrame([diversity_summary]).to_csv(paths.diversity_summary, index=False)

    _plot_ring_heatmap(ring_distribution, paths.output_dir / "ring_count_distribution_heatmap.png")
    _plot_ring_heatmap(ring_distribution, paths.output_dir / "ring_count_distribution_heatmap.svg")
    _plot_ring_prevalence(ring_distribution, paths.output_dir / "ring_feature_prevalence.png")
    _plot_ring_prevalence(ring_distribution, paths.output_dir / "ring_feature_prevalence.svg")
    _plot_property_histograms(valid, paths.output_dir / "core_property_distributions.png")
    _plot_property_histograms(valid, paths.output_dir / "core_property_distributions.svg")
    _plot_scaffold_top(scaffold_top, paths.output_dir / "top_scaffolds.png")
    _plot_scaffold_top(scaffold_top, paths.output_dir / "top_scaffolds.svg")

    summary = {
        "input": str(args.input),
        "rows": int(len(features)),
        "rdkit_valid": int(len(valid)),
        "rdkit_valid_ratio": float(len(valid) / max(len(features), 1)),
        "ring_distribution_csv": str(paths.ring_distribution),
        "property_summary_csv": str(paths.property_summary),
        "scaffold_summary_csv": str(paths.scaffold_summary),
        "scaffold_top_csv": str(paths.scaffold_top),
        "diversity_summary_csv": str(paths.diversity_summary),
        "plots": [
            "ring_count_distribution_heatmap.png",
            "ring_feature_prevalence.png",
            "core_property_distributions.png",
            "top_scaffolds.png",
        ],
        "scaffold_summary": scaffold_summary,
        "diversity_summary": diversity_summary,
    }
    paths.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_report(paths, summary, property_summary, ring_distribution, scaffold_top)
    LOGGER.info("done: %s", paths.report_md)


def _paths(output_dir: Path) -> Paths:
    return Paths(
        output_dir=output_dir,
        per_molecule=output_dir / "chembl_36_per_molecule_ring_features.csv.gz",
        ring_distribution=output_dir / "ring_count_distribution.csv",
        property_summary=output_dir / "property_summary.csv",
        scaffold_summary=output_dir / "scaffold_summary.csv",
        scaffold_top=output_dir / "top_scaffolds.csv",
        diversity_summary=output_dir / "fingerprint_diversity_summary.csv",
        summary_json=output_dir / "summary.json",
        report_md=output_dir / "chembl_property_ring_distribution_report.md",
    )


def _write_per_molecule_features(args: argparse.Namespace, output_path: Path) -> None:
    """Compute RDKit-derived per-molecule features and stream them to CSV."""
    from multiprocessing import Pool

    selected_columns = [
        "chembl_id",
        "canonical_smiles",
        "heavy_atoms",
        "formal_charge",
        "num_rings",
        "mol_wt",
        "mol_log_p",
        "tpsa",
        "num_h_acceptors",
        "num_h_donors",
        "num_rotatable_bonds",
        "num_heteroatoms",
        "fraction_csp3",
        "num_carbon_atoms",
        "carbon_fraction",
        "lipinski_violations",
    ]
    row_limit = args.max_rows
    rows_written = 0
    first = True
    with gzip.open(output_path, "wt", encoding="utf-8", newline="") as handle:
        with Pool(processes=args.workers, initializer=_init_worker, initargs=(not args.no_sa,)) as pool:
            for chunk_index, chunk in enumerate(
                pd.read_csv(args.input, usecols=selected_columns, chunksize=args.chunksize)
            ):
                if row_limit is not None:
                    remaining = int(row_limit) - rows_written
                    if remaining <= 0:
                        break
                    chunk = chunk.head(remaining)
                smiles = chunk["canonical_smiles"].fillna("").astype(str).tolist()
                computed = list(pool.imap(_compute_rdkit_features, smiles, chunksize=256))
                computed_df = pd.DataFrame(computed)
                out = pd.concat([chunk.reset_index(drop=True), computed_df], axis=1)
                out.to_csv(handle, index=False, header=first)
                first = False
                rows_written += len(out)
                LOGGER.info("processed chunk=%s rows_written=%s", chunk_index, rows_written)
                if row_limit is not None and rows_written >= row_limit:
                    break


def _init_worker(compute_sa: bool) -> None:
    """Initialize optional worker-local SA scorer."""
    global _SASCORER
    if not compute_sa:
        _SASCORER = None
    else:
        if str(SA_SCORE_DIR) not in sys.path:
            sys.path.insert(0, str(SA_SCORE_DIR))
        import sascorer  # type: ignore

        _SASCORER = sascorer
    _init_filter_catalogs()


def _init_filter_catalogs() -> None:
    """Construct worker-local RDKit structural-alert catalogs."""
    global _FILTER_CATALOGS
    catalog_types = {
        "pains": FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS,
        "nih": FilterCatalog.FilterCatalogParams.FilterCatalogs.NIH,
        "brenk": FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK,
    }
    _FILTER_CATALOGS = {}
    for name, catalog_type in catalog_types.items():
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(catalog_type)
        _FILTER_CATALOGS[name] = FilterCatalog.FilterCatalog(params)


def _compute_rdkit_features(smiles: str) -> dict[str, Any]:
    """Compute ring and scaffold features for one SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return _invalid_features()
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return _invalid_features()

    atom_rings = [tuple(ring) for ring in mol.GetRingInfo().AtomRings()]
    aromatic_flags = [_ring_is_aromatic(mol, ring) for ring in atom_rings]
    hetero_flags = [_ring_has_hetero(mol, ring) for ring in atom_rings]

    ring_sizes = Counter(len(ring) for ring in atom_rings)
    ring_system_sizes = _ring_component_sizes(atom_rings, min_shared_atoms=1)
    fused_system_sizes = _ring_component_sizes(atom_rings, min_shared_atoms=2)
    aromatic_ring_count = sum(1 for value in aromatic_flags if value)
    heterocycle_count = sum(1 for value in hetero_flags if value)
    aromatic_heterocycle_count = sum(1 for arom, het in zip(aromatic_flags, hetero_flags) if arom and het)
    benzene_ring_count = sum(1 for ring in atom_rings if _ring_is_benzene(mol, ring))

    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        scaffold = ""

    qed = _safe_float(QED.qed(mol))
    sa_score = math.nan
    if _SASCORER is not None:
        try:
            sa_score = _safe_float(_SASCORER.calculateScore(mol))
        except Exception:
            sa_score = math.nan

    heavy_atoms = mol.GetNumHeavyAtoms()
    carbon_atoms = sum(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms())
    hetero_atoms = sum(
        atom.GetAtomicNum() not in (1, 6) for atom in mol.GetAtoms()
    )
    halogen_atoms = sum(atom.GetAtomicNum() in (9, 17, 35, 53, 85) for atom in mol.GetAtoms())
    alert_matches = {
        name: _alert_matches(catalog, mol) for name, catalog in _FILTER_CATALOGS.items()
    }

    return {
        "rdkit_valid": 1,
        "qed": qed,
        "sa_score": sa_score,
        "scaffold_smiles": scaffold,
        "total_ring_count": len(atom_rings),
        "aromatic_ring_count": aromatic_ring_count,
        "aliphatic_ring_count": len(atom_rings) - aromatic_ring_count,
        "ring_3_count": ring_sizes.get(3, 0),
        "ring_4_count": ring_sizes.get(4, 0),
        "ring_5_count": ring_sizes.get(5, 0),
        "ring_6_count": ring_sizes.get(6, 0),
        "ring_7_count": ring_sizes.get(7, 0),
        "large_ring_count": sum(count for size, count in ring_sizes.items() if size >= 8),
        "benzene_ring_count": benzene_ring_count,
        "heterocycle_count": heterocycle_count,
        "aromatic_heterocycle_count": aromatic_heterocycle_count,
        "fused_ring_system_count": _fused_ring_system_count(atom_rings),
        "spiro_atom_count": int(rdMolDescriptors.CalcNumSpiroAtoms(mol)),
        "bridgehead_atom_count": int(rdMolDescriptors.CalcNumBridgeheadAtoms(mol)),
        "max_ring_size": max(ring_sizes, default=0),
        "ring_system_count": len(ring_system_sizes),
        "max_ring_system_size": max(ring_system_sizes, default=0),
        "max_fused_ring_system_size": max(fused_system_sizes, default=0),
        "bertz_complexity": _safe_float(Descriptors.BertzCT(mol)),
        "element_diversity": len({atom.GetAtomicNum() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1}),
        "radical_electron_count": int(Descriptors.NumRadicalElectrons(mol)),
        "halogen_fraction": halogen_atoms / heavy_atoms if heavy_atoms else math.nan,
        "all_carbon": int(heavy_atoms > 0 and carbon_atoms == heavy_atoms),
        "carbon_enriched_low_hetero": int(
            heavy_atoms > 0 and carbon_atoms / heavy_atoms >= 0.9 and hetero_atoms <= 1
        ),
        "pains_alert_count": len(alert_matches["pains"]),
        "nih_alert_count": len(alert_matches["nih"]),
        "brenk_alert_count": len(alert_matches["brenk"]),
        "pains_alert_names": "|".join(alert_matches["pains"]),
        "nih_alert_names": "|".join(alert_matches["nih"]),
        "brenk_alert_names": "|".join(alert_matches["brenk"]),
    }


def _invalid_features() -> dict[str, Any]:
    """Return null-ish values for invalid molecules."""
    row = {
        "rdkit_valid": 0,
        "qed": math.nan,
        "sa_score": math.nan,
        "scaffold_smiles": "",
    }
    for key, _label in RING_COUNT_FEATURES:
        row[key] = math.nan
    return row


def _alert_matches(catalog: Any, mol: Chem.Mol) -> list[str]:
    """Return de-duplicated RDKit alert descriptions for one catalog."""
    try:
        return sorted({match.GetDescription() for match in catalog.GetMatches(mol)})
    except Exception:
        return []


def _ring_component_sizes(atom_rings: list[tuple[int, ...]], min_shared_atoms: int) -> list[int]:
    """Return ring-component sizes using a declared atom-sharing criterion."""
    ring_sets = [set(ring) for ring in atom_rings]
    remaining = set(range(len(ring_sets)))
    sizes: list[int] = []
    while remaining:
        start = remaining.pop()
        component = {start}
        stack = [start]
        while stack:
            current = stack.pop()
            neighbours = {
                index
                for index in remaining
                if len(ring_sets[current].intersection(ring_sets[index])) >= min_shared_atoms
            }
            remaining.difference_update(neighbours)
            component.update(neighbours)
            stack.extend(neighbours)
        sizes.append(len(component))
    return sizes


def _ring_is_aromatic(mol: Chem.Mol, ring: Iterable[int]) -> bool:
    return all(mol.GetAtomWithIdx(int(idx)).GetIsAromatic() for idx in ring)


def _ring_has_hetero(mol: Chem.Mol, ring: Iterable[int]) -> bool:
    return any(mol.GetAtomWithIdx(int(idx)).GetAtomicNum() != 6 for idx in ring)


def _ring_is_benzene(mol: Chem.Mol, ring: Iterable[int]) -> bool:
    atoms = [mol.GetAtomWithIdx(int(idx)) for idx in ring]
    return len(atoms) == 6 and all(atom.GetAtomicNum() == 6 and atom.GetIsAromatic() for atom in atoms)


def _fused_ring_system_count(atom_rings: list[tuple[int, ...]]) -> int:
    """Count connected ring systems where rings share at least two atoms."""
    n_rings = len(atom_rings)
    if n_rings < 2:
        return 0
    ring_sets = [set(ring) for ring in atom_rings]
    adjacency: list[set[int]] = [set() for _ in range(n_rings)]
    for i in range(n_rings):
        for j in range(i + 1, n_rings):
            if len(ring_sets[i].intersection(ring_sets[j])) >= 2:
                adjacency[i].add(j)
                adjacency[j].add(i)
    seen: set[int] = set()
    fused_systems = 0
    for start in range(n_rings):
        if start in seen:
            continue
        stack = [start]
        component: set[int] = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(adjacency[node] - component)
        seen.update(component)
        if len(component) >= 2:
            fused_systems += 1
    return fused_systems


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def _ring_distribution(valid: pd.DataFrame) -> pd.DataFrame:
    """Create ring-feature count-bin distribution table."""
    rows: list[dict[str, Any]] = []
    denominator = int(len(valid))
    for feature, label in RING_COUNT_FEATURES:
        values = valid[feature].fillna(0).astype(int)
        for bin_label in COUNT_BINS:
            if bin_label == ">=6":
                count = int((values >= 6).sum())
            else:
                count = int((values == int(bin_label)).sum())
            rows.append(
                {
                    "feature": feature,
                    "label": label,
                    "count_bin": bin_label,
                    "molecule_count": count,
                    "fraction_valid": float(count / max(denominator, 1)),
                    "percent_valid": float(100.0 * count / max(denominator, 1)),
                }
            )
    return pd.DataFrame(rows)


def _property_summary(valid: pd.DataFrame) -> pd.DataFrame:
    """Summarize numeric property columns."""
    rows: list[dict[str, Any]] = []
    for column in PROPERTY_COLUMNS:
        if column not in valid.columns:
            continue
        values = pd.to_numeric(valid[column], errors="coerce").dropna()
        if values.empty:
            continue
        quantiles = values.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        rows.append(
            {
                "property": column,
                "count": int(values.shape[0]),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "min": float(values.min()),
                "p01": float(quantiles.loc[0.01]),
                "p05": float(quantiles.loc[0.05]),
                "p25": float(quantiles.loc[0.25]),
                "median": float(quantiles.loc[0.5]),
                "p75": float(quantiles.loc[0.75]),
                "p95": float(quantiles.loc[0.95]),
                "p99": float(quantiles.loc[0.99]),
                "max": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def _scaffold_tables(valid: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    """Summarize Murcko scaffold diversity."""
    scaffolds = valid["scaffold_smiles"].fillna("").astype(str)
    counter = Counter(scaffolds)
    total = int(len(scaffolds))
    nonempty_counter = Counter({key: value for key, value in counter.items() if key})
    top_rows = []
    for rank, (scaffold, count) in enumerate(nonempty_counter.most_common(30), start=1):
        top_rows.append(
            {
                "rank": rank,
                "scaffold_smiles": scaffold,
                "molecule_count": int(count),
                "fraction_valid": float(count / max(total, 1)),
                "percent_valid": float(100.0 * count / max(total, 1)),
            }
        )
    top = pd.DataFrame(top_rows)
    summary = {
        "valid_molecules": total,
        "unique_canonical_smiles": int(valid["canonical_smiles"].nunique()),
        "unique_ratio": float(valid["canonical_smiles"].nunique() / max(total, 1)),
        "unique_nonempty_scaffolds": int(len(nonempty_counter)),
        "scaffold_diversity_ratio": float(len(nonempty_counter) / max(total, 1)),
        "acyclic_or_empty_scaffold_count": int(counter.get("", 0)),
        "acyclic_or_empty_scaffold_fraction": float(counter.get("", 0) / max(total, 1)),
        "top1_scaffold_fraction": float(top.iloc[0]["fraction_valid"]) if not top.empty else 0.0,
        "top10_scaffold_fraction": float(top.head(10)["fraction_valid"].sum()) if not top.empty else 0.0,
        "top30_scaffold_fraction": float(top.head(30)["fraction_valid"].sum()) if not top.empty else 0.0,
    }
    return summary, top


def _fingerprint_diversity(valid: pd.DataFrame, sample_size: int, pair_count: int) -> dict[str, Any]:
    """Estimate ECFP4 bit density and random-pair Tanimoto similarity."""
    smiles = valid["canonical_smiles"].dropna().astype(str).tolist()
    rng = random.Random(260624)
    if len(smiles) > sample_size:
        smiles = rng.sample(smiles, sample_size)
    fps = []
    bit_densities = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        fps.append(fp)
        bit_densities.append(float(fp.GetNumOnBits() / 2048.0))
    if len(fps) < 2:
        return {
            "sample_size": int(len(fps)),
            "pair_count": 0,
            "ecfp4_bit_density_mean": math.nan,
            "ecfp4_random_pair_tanimoto_mean": math.nan,
            "ecfp4_random_pair_tanimoto_median": math.nan,
            "ecfp4_random_pair_tanimoto_p95": math.nan,
        }
    similarities = []
    for _ in range(pair_count):
        i = rng.randrange(len(fps))
        j = rng.randrange(len(fps) - 1)
        if j >= i:
            j += 1
        similarities.append(float(DataStructs.TanimotoSimilarity(fps[i], fps[j])))
    sim = np.array(similarities, dtype=float)
    return {
        "sample_size": int(len(fps)),
        "pair_count": int(len(similarities)),
        "ecfp4_bit_density_mean": float(np.mean(bit_densities)),
        "ecfp4_bit_density_std": float(np.std(bit_densities)),
        "ecfp4_random_pair_tanimoto_mean": float(np.mean(sim)),
        "ecfp4_random_pair_tanimoto_median": float(np.median(sim)),
        "ecfp4_random_pair_tanimoto_p95": float(np.quantile(sim, 0.95)),
        "ecfp4_random_pair_tanimoto_p99": float(np.quantile(sim, 0.99)),
    }


def _plot_ring_heatmap(ring_distribution: pd.DataFrame, output_path: Path) -> None:
    """Plot feature-by-count-bin ring distribution heatmap."""
    matrix = ring_distribution.pivot(index="label", columns="count_bin", values="percent_valid").reindex(
        index=[label for _feature, label in RING_COUNT_FEATURES],
        columns=list(COUNT_BINS),
    )
    fig, ax = plt.subplots(figsize=(8.8, 6.5))
    im = ax.imshow(matrix.values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(matrix.columns)), labels=matrix.columns)
    ax.set_yticks(np.arange(len(matrix.index)), labels=matrix.index)
    ax.set_xlabel("Count per molecule")
    ax.set_title("ChEMBL ring-related count distributions (% valid molecules)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("% valid molecules")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_ring_prevalence(ring_distribution: pd.DataFrame, output_path: Path) -> None:
    """Plot fraction of molecules with at least one feature occurrence."""
    rows = []
    for feature, label in RING_COUNT_FEATURES:
        subset = ring_distribution[ring_distribution["feature"] == feature]
        zero = float(subset.loc[subset["count_bin"] == "0", "percent_valid"].iloc[0])
        rows.append({"label": label, "percent_with_feature": 100.0 - zero})
    df = pd.DataFrame(rows).sort_values("percent_with_feature")
    fig, ax = plt.subplots(figsize=(8.0, 6.2))
    ax.barh(df["label"], df["percent_with_feature"], color="#377eb8")
    ax.set_xlabel("% valid molecules with count >= 1")
    ax.set_title("Ring-feature prevalence in ChEMBL")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_property_histograms(valid: pd.DataFrame, output_path: Path) -> None:
    """Plot compact histograms of core drug-like properties."""
    columns = [
        ("mol_wt", "MW"),
        ("mol_log_p", "LogP"),
        ("tpsa", "TPSA"),
        ("num_h_acceptors", "HBA"),
        ("num_h_donors", "HBD"),
        ("num_rotatable_bonds", "Rotatable bonds"),
        ("qed", "QED"),
        ("sa_score", "SA score"),
    ]
    sample = valid.sample(n=min(len(valid), 200000), random_state=260624) if len(valid) > 200000 else valid
    fig, axes = plt.subplots(2, 4, figsize=(13.0, 6.0))
    for ax, (column, label) in zip(axes.ravel(), columns):
        values = pd.to_numeric(sample[column], errors="coerce").dropna()
        ax.hist(values, bins=60, color="#4daf4a", alpha=0.88)
        ax.set_title(label)
        ax.set_ylabel("Molecules")
    fig.suptitle("Core ChEMBL property distributions")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_scaffold_top(scaffold_top: pd.DataFrame, output_path: Path) -> None:
    """Plot top Murcko scaffold frequencies."""
    if scaffold_top.empty:
        return
    plot_df = scaffold_top.head(20).copy()
    plot_df["short_scaffold"] = [f"#{rank}" for rank in plot_df["rank"]]
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.barh(plot_df["short_scaffold"][::-1], plot_df["percent_valid"][::-1], color="#984ea3")
    ax.set_xlabel("% valid molecules")
    ax.set_ylabel("Murcko scaffold rank")
    ax.set_title("Top Murcko scaffold frequencies")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _write_report(
    paths: Paths,
    summary: dict[str, Any],
    property_summary: pd.DataFrame,
    ring_distribution: pd.DataFrame,
    scaffold_top: pd.DataFrame,
) -> None:
    """Write Markdown report with embedded plots and compact tables."""
    core_props = property_summary[property_summary["property"].isin(PROPERTY_COLUMNS)].copy()
    core_props = core_props[
        ["property", "count", "mean", "std", "p05", "median", "p95", "min", "max"]
    ].round(4)
    ring_prevalence_rows = []
    for feature, label in RING_COUNT_FEATURES:
        subset = ring_distribution[ring_distribution["feature"] == feature]
        zero = float(subset.loc[subset["count_bin"] == "0", "percent_valid"].iloc[0])
        ge6 = float(subset.loc[subset["count_bin"] == ">=6", "percent_valid"].iloc[0])
        ring_prevalence_rows.append(
            {
                "feature": label,
                "% count>=1": round(100.0 - zero, 4),
                "% count>=6": round(ge6, 4),
            }
        )
    ring_prevalence = pd.DataFrame(ring_prevalence_rows)
    scaffold_preview = scaffold_top.head(10)[["rank", "molecule_count", "percent_valid", "scaffold_smiles"]].copy()
    if not scaffold_preview.empty:
        scaffold_preview["percent_valid"] = scaffold_preview["percent_valid"].round(4)

    text = f"""# ChEMBL 36 Property And Ring Distribution Report

## Dataset

- Input: `{summary['input']}`
- Rows: `{summary['rows']:,}`
- RDKit valid: `{summary['rdkit_valid']:,}` ({summary['rdkit_valid_ratio']:.4%})
- Count bins for ring-related features: `0 / 1 / 2 / 3 / 4 / 5 / >=6`
- Large ring definition: ring size `>=8`
- Fingerprint diversity is estimated from a sampled ECFP4 set, not all pairwise combinations.

## Core Property Distributions

![Core property distributions](core_property_distributions.png)

{_markdown_table(core_props)}

## Ring Count Distributions

![Ring count distribution heatmap](ring_count_distribution_heatmap.png)

![Ring feature prevalence](ring_feature_prevalence.png)

{_markdown_table(ring_prevalence)}

Full ring count-bin table: [ring_count_distribution.csv](ring_count_distribution.csv)

## Scaffold And Diversity

![Top scaffolds](top_scaffolds.png)

Scaffold summary: [scaffold_summary.csv](scaffold_summary.csv)

Top scaffold preview:

{_markdown_table(scaffold_preview)}

Fingerprint diversity summary: [fingerprint_diversity_summary.csv](fingerprint_diversity_summary.csv)

## Evidence Files

- Per-molecule features: [chembl_36_per_molecule_ring_features.csv.gz](chembl_36_per_molecule_ring_features.csv.gz)
- Property summary: [property_summary.csv](property_summary.csv)
- Ring distribution: [ring_count_distribution.csv](ring_count_distribution.csv)
- Scaffold summary: [scaffold_summary.csv](scaffold_summary.csv)
- Top scaffolds: [top_scaffolds.csv](top_scaffolds.csv)
- JSON summary: [summary.json](summary.json)
"""
    paths.report_md.write_text(text, encoding="utf-8")


def _markdown_table(df: pd.DataFrame) -> str:
    """Render a small DataFrame as GitHub-flavored Markdown without tabulate."""
    if df.empty:
        return "_No rows._"
    rendered = df.copy()
    for column in rendered.columns:
        rendered[column] = rendered[column].map(_format_markdown_cell)
    header = "| " + " | ".join(str(column) for column in rendered.columns) + " |"
    separator = "| " + " | ".join("---" for _ in rendered.columns) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in rendered.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def _format_markdown_cell(value: Any) -> str:
    """Format a scalar for Markdown table cells."""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.4g}"
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()
