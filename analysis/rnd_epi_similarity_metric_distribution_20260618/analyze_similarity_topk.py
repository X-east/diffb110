#!/usr/bin/env python3
"""Analyze top-k similarity distributions from ChEMBL to RND-EPI seeds."""

from __future__ import annotations

import gzip
import json
import logging
import multiprocessing as mp
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen, MACCSkeys, rdMolDescriptors


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "rnd_epi_similarity_metric_distribution_20260618"
SEED_SDF = PROJECT_ROOT / "data" / "rnd-epi" / "epi_53_molecules_3d.sdf"
CHEMBL_PROPERTIES = PROJECT_ROOT / "data" / "chembl" / "chembl_36_properties.csv.gz"
N_JOBS = 18
CHUNK_SIZE = 8_000
TOPK = (1, 3, 5)
DESCRIPTOR_COLUMNS = [
    "heavy_atoms",
    "mol_wt",
    "mol_log_p",
    "tpsa",
    "num_h_acceptors",
    "num_h_donors",
    "num_rotatable_bonds",
    "num_rings",
    "num_aromatic_rings",
    "num_heteroatoms",
    "carbon_fraction",
]
FINGERPRINT_METRICS = [
    "ecfp4",
    "ecfp6",
    "fcfp4",
    "maccs",
    "atom_pair",
    "topological_torsion",
]
ALL_METRICS = FINGERPRINT_METRICS + ["descriptor_similarity"]

LOGGER = logging.getLogger("rnd_epi_similarity_topk")
_WORKER_SEEDS: list[dict[str, object]] = []
_WORKER_SEED_FPS: dict[str, list[DataStructs.ExplicitBitVect]] = {}
_WORKER_SEED_DESC_Z: np.ndarray | None = None
_WORKER_DESC_MEAN: np.ndarray | None = None
_WORKER_DESC_STD: np.ndarray | None = None


def _canonicalize_mol(mol: Chem.Mol) -> Chem.Mol | None:
    """Return a sanitized hydrogen-stripped molecule or None."""
    if mol is None:
        return None
    mol = Chem.RemoveHs(mol)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol


def _fingerprints_for_mol(mol: Chem.Mol) -> dict[str, DataStructs.ExplicitBitVect]:
    """Build all bit-vector fingerprints for one molecule."""
    return {
        "ecfp4": AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048),
        "ecfp6": AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048),
        "fcfp4": AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=2, nBits=2048, useFeatures=True
        ),
        "maccs": MACCSkeys.GenMACCSKeys(mol),
        "atom_pair": rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(
            mol, nBits=2048
        ),
        "topological_torsion": rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(
            mol, nBits=2048
        ),
    }


def _load_seed_molecules() -> list[dict[str, object]]:
    """Load RND-EPI seeds and compute fingerprints/descriptors."""
    seeds: list[dict[str, object]] = []
    supplier = Chem.SDMolSupplier(str(SEED_SDF), removeHs=False)
    for index, raw_mol in enumerate(supplier, start=1):
        mol = _canonicalize_mol(raw_mol)
        if mol is None:
            LOGGER.warning("Skipping unreadable seed index=%d", index)
            continue
        desc = _descriptor_vector_from_mol_and_props(mol, None)
        seeds.append(
            {
                "seed_index": index,
                "seed_name": raw_mol.GetProp("Preferred_Name")
                if raw_mol.HasProp("Preferred_Name")
                else raw_mol.GetProp("_Name"),
                "seed_smiles": Chem.MolToSmiles(mol, canonical=True),
                "fingerprints": _fingerprints_for_mol(mol),
                "descriptor_vector": desc,
            }
        )
    if not seeds:
        raise RuntimeError(f"No valid seeds loaded from {SEED_SDF}")
    return seeds


def _descriptor_vector_from_mol_and_props(
    mol: Chem.Mol,
    props: dict[str, object] | None,
) -> np.ndarray:
    """Build the 2D descriptor vector used for descriptor-space distance."""
    if props is None:
        heavy_atoms = mol.GetNumHeavyAtoms()
        num_carbon = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
        num_hetero = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in (1, 6))
        values = {
            "heavy_atoms": heavy_atoms,
            "mol_wt": rdMolDescriptors.CalcExactMolWt(mol),
            "mol_log_p": Crippen.MolLogP(mol),
            "tpsa": rdMolDescriptors.CalcTPSA(mol),
            "num_h_acceptors": rdMolDescriptors.CalcNumHBA(mol),
            "num_h_donors": rdMolDescriptors.CalcNumHBD(mol),
            "num_rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "num_rings": rdMolDescriptors.CalcNumRings(mol),
            "num_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "num_heteroatoms": num_hetero,
            "carbon_fraction": num_carbon / heavy_atoms if heavy_atoms else 0.0,
        }
    else:
        values = props
    return np.array([float(values[column]) for column in DESCRIPTOR_COLUMNS], dtype=np.float32)


def _normal_allowed(mol: Chem.Mol, props: dict[str, object]) -> tuple[bool, str]:
    """Exclude only obviously abnormal molecules."""
    allowed_atomic_nums = {1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53}
    atomic_nums = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    heavy_atoms = int(props["heavy_atoms"])
    mol_wt = float(props["mol_wt"])
    logp = float(props["mol_log_p"])
    tpsa = float(props["tpsa"])
    hba = int(props["num_h_acceptors"])
    hbd = int(props["num_h_donors"])
    rotb = int(props["num_rotatable_bonds"])
    carbon_fraction = float(props["carbon_fraction"])
    num_heteroatoms = int(props["num_heteroatoms"])
    formal_charge = int(props["formal_charge"])

    if "." in str(props["canonical_smiles"]):
        return False, "multi_fragment"
    if any(num not in allowed_atomic_nums for num in atomic_nums):
        return False, "unusual_element"
    if not any(num == 6 for num in atomic_nums):
        return False, "no_carbon"
    if heavy_atoms < 8:
        return False, "too_few_heavy_atoms"
    if heavy_atoms > 75:
        return False, "too_many_heavy_atoms"
    if mol_wt < 120:
        return False, "mw_too_low"
    if mol_wt > 900:
        return False, "mw_too_high"
    if tpsa > 260:
        return False, "tpsa_too_high"
    if logp > 11:
        return False, "logp_too_high"
    if logp < -5:
        return False, "logp_too_low"
    if rotb > 25:
        return False, "rotb_too_high"
    if hba > 20:
        return False, "hba_too_high"
    if hbd > 12:
        return False, "hbd_too_high"
    if carbon_fraction > 0.97 and num_heteroatoms == 0:
        return False, "all_carbon_pah_like"
    if abs(formal_charge) > 3:
        return False, "formal_charge_extreme"
    return True, "normal"


def _topk_mean(values: list[float], k: int) -> float:
    """Return the mean of the top-k values."""
    if not values:
        return 0.0
    top = sorted(values, reverse=True)[:k]
    return float(np.mean(top))


def _topk_fingerprint_scores(
    fp: DataStructs.ExplicitBitVect,
    seed_fps: list[DataStructs.ExplicitBitVect],
) -> dict[str, float]:
    """Compute top1/top3/top5 Tanimoto similarities to seeds."""
    similarities = DataStructs.BulkTanimotoSimilarity(fp, seed_fps)
    return {f"top{k}": _topk_mean(similarities, k) for k in TOPK}


def _descriptor_similarity_scores(desc: np.ndarray) -> dict[str, float]:
    """Compute top1/top3/top5 descriptor-space similarities to seeds."""
    if (
        _WORKER_SEED_DESC_Z is None
        or _WORKER_DESC_MEAN is None
        or _WORKER_DESC_STD is None
    ):
        raise RuntimeError("Descriptor worker cache has not been initialized")
    desc_z = (desc - _WORKER_DESC_MEAN) / _WORKER_DESC_STD
    distances = np.linalg.norm(_WORKER_SEED_DESC_Z - desc_z, axis=1)
    similarities = (1.0 / (1.0 + distances)).tolist()
    return {f"top{k}": _topk_mean(similarities, k) for k in TOPK}


def _init_worker(
    seeds: list[dict[str, object]],
    desc_mean: np.ndarray,
    desc_std: np.ndarray,
) -> None:
    """Initialize worker-local seed caches."""
    global _WORKER_SEEDS, _WORKER_SEED_FPS, _WORKER_SEED_DESC_Z
    global _WORKER_DESC_MEAN, _WORKER_DESC_STD
    _WORKER_SEEDS = seeds
    _WORKER_SEED_FPS = {
        metric: [seed["fingerprints"][metric] for seed in seeds]
        for metric in FINGERPRINT_METRICS
    }
    _WORKER_DESC_MEAN = desc_mean
    _WORKER_DESC_STD = desc_std
    seed_desc = np.stack([seed["descriptor_vector"] for seed in seeds], axis=0)
    _WORKER_SEED_DESC_Z = (seed_desc - desc_mean) / desc_std


def _screen_chunk(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Analyze one ChEMBL chunk in a worker."""
    output_rows: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    for props in rows:
        counts["total_records"] += 1
        smiles = str(props["canonical_smiles"])
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            counts["rdkit_parse_failed"] += 1
            continue
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            counts["sanitize_failed"] += 1
            continue
        allowed, reason = _normal_allowed(mol, props)
        if not allowed:
            counts[reason] += 1
            continue
        counts["normal_records"] += 1

        fps = _fingerprints_for_mol(mol)
        out: dict[str, object] = {
            "chembl_id": props["chembl_id"],
            "canonical_smiles": smiles,
            "mol_wt": float(props["mol_wt"]),
            "mol_log_p": float(props["mol_log_p"]),
            "tpsa": float(props["tpsa"]),
            "heavy_atoms": int(props["heavy_atoms"]),
            "num_rings": int(props["num_rings"]),
            "num_aromatic_rings": int(props["num_aromatic_rings"]),
            "num_heteroatoms": int(props["num_heteroatoms"]),
            "carbon_fraction": float(props["carbon_fraction"]),
        }
        for metric in FINGERPRINT_METRICS:
            scores = _topk_fingerprint_scores(fps[metric], _WORKER_SEED_FPS[metric])
            for top_name, value in scores.items():
                out[f"{metric}_{top_name}"] = value

        desc = _descriptor_vector_from_mol_and_props(mol, props)
        desc_scores = _descriptor_similarity_scores(desc)
        for top_name, value in desc_scores.items():
            out[f"descriptor_similarity_{top_name}"] = value

        output_rows.append(out)

    counts["analyzed_records"] = len(output_rows)
    return output_rows, dict(counts)


def _load_descriptor_stats(seeds: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    """Use ChEMBL distribution stats when available, with seed-aware fallback."""
    stats_path = PROJECT_ROOT / "data" / "chembl" / "chembl_36_properties_stats.json"
    if stats_path.exists():
        with stats_path.open() as handle:
            stats = json.load(handle)
        statistics = stats.get("statistics", {})
        means = []
        stds = []
        for column in DESCRIPTOR_COLUMNS:
            column_stats = statistics.get(column, {})
            means.append(float(column_stats.get("mean", 0.0)))
            std = float(column_stats.get("std", 1.0))
            stds.append(std if std > 1e-6 else 1.0)
        return np.array(means, dtype=np.float32), np.array(stds, dtype=np.float32)

    seed_desc = np.stack([seed["descriptor_vector"] for seed in seeds], axis=0)
    mean = seed_desc.mean(axis=0)
    std = seed_desc.std(axis=0)
    std[std <= 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def _iter_chunks() -> object:
    """Yield ChEMBL property-table chunks as lists of records."""
    usecols = [
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
        "num_aromatic_rings",
        "num_heteroatoms",
        "num_carbon_atoms",
        "carbon_fraction",
    ]
    for chunk in pd.read_csv(CHEMBL_PROPERTIES, usecols=usecols, chunksize=CHUNK_SIZE):
        yield chunk.to_dict("records")


def _collect_similarity_table() -> tuple[pd.DataFrame, dict[str, int]]:
    """Run the full ChEMBL top-k analysis."""
    seeds = _load_seed_molecules()
    desc_mean, desc_std = _load_descriptor_stats(seeds)
    rows: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    processed = 0
    LOGGER.info(
        "Starting top-k analysis with %d seeds, %d workers, chunk_size=%d",
        len(seeds),
        N_JOBS,
        CHUNK_SIZE,
    )
    with mp.Pool(
        processes=N_JOBS,
        initializer=_init_worker,
        initargs=(seeds, desc_mean, desc_std),
    ) as pool:
        for chunk_rows, chunk_counts in pool.imap_unordered(_screen_chunk, _iter_chunks(), chunksize=1):
            rows.extend(chunk_rows)
            counts.update(chunk_counts)
            processed += int(chunk_counts.get("total_records", 0))
            if processed and processed % 96000 == 0:
                LOGGER.info(
                    "Processed %d rows, analyzed=%d",
                    processed,
                    counts.get("analyzed_records", 0),
                )
    counts["seed_count"] = len(seeds)
    return pd.DataFrame.from_records(rows), dict(counts)


def _summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize top-k score distributions by metric."""
    rows: list[dict[str, object]] = []
    quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    for metric in ALL_METRICS:
        for k in TOPK:
            col = f"{metric}_top{k}"
            values = df[col]
            row = {
                "metric": metric,
                "topk": f"top{k}",
                "mean": values.mean(),
                "std": values.std(),
                "min": values.min(),
                "max": values.max(),
            }
            for quantile in quantiles:
                row[f"q{int(quantile * 100):02d}"] = values.quantile(quantile)
            rows.append(row)
    return pd.DataFrame(rows)


def _write_metric_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Write top1 metric correlation matrix."""
    columns = [f"{metric}_top1" for metric in ALL_METRICS]
    corr = df[columns].corr()
    corr.to_csv(OUTPUT_DIR / "top1_metric_correlation.csv")
    return corr


def _plot_histograms(df: pd.DataFrame) -> None:
    """Plot top-k histogram panels for each metric."""
    fig, axes = plt.subplots(4, 2, figsize=(13, 16))
    axes = axes.flatten()
    for ax, metric in zip(axes, ALL_METRICS):
        for k in TOPK:
            col = f"{metric}_top{k}"
            ax.hist(df[col], bins=70, alpha=0.45, label=f"top{k}", density=True)
        ax.set_title(metric)
        ax.set_xlabel("Similarity")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
    for ax in axes[len(ALL_METRICS):]:
        ax.axis("off")
    fig.suptitle("ChEMBL to RND-EPI seed similarity distributions", y=0.995)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "topk_similarity_histograms.png", dpi=180)
    plt.close(fig)


def _plot_boxplots(df: pd.DataFrame) -> None:
    """Plot top1/top3/top5 score distributions as boxplots."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    for ax, k in zip(axes, TOPK):
        columns = [f"{metric}_top{k}" for metric in ALL_METRICS]
        ax.boxplot([df[col].to_numpy() for col in columns], labels=ALL_METRICS, showfliers=False)
        ax.set_title(f"top{k}")
        ax.set_ylabel("Similarity")
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "topk_similarity_boxplots.png", dpi=180)
    plt.close(fig)


def _plot_correlation(corr: pd.DataFrame) -> None:
    """Plot top1 metric correlation heatmap."""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), labels=corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)), labels=corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Top1 similarity metric correlation")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "top1_metric_correlation.png", dpi=180)
    plt.close(fig)


def _format_markdown_table(df: pd.DataFrame, index: bool = False) -> str:
    """Format a DataFrame as a Markdown table without optional dependencies."""
    table = df.reset_index() if index else df.copy()
    table = table.fillna("")
    headers = [str(column) for column in table.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in table.to_numpy().tolist():
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _write_report(summary: pd.DataFrame, counts: dict[str, int], corr: pd.DataFrame) -> None:
    """Write a Markdown report."""
    report_path = OUTPUT_DIR / "rnd_epi_similarity_metric_distribution_report.md"
    focused_cols = ["metric", "topk", "mean", "q05", "q10", "q25", "q50", "q75", "q90", "q95", "q99", "max"]
    top1 = summary[summary["topk"].eq("top1")][focused_cols].round(4)
    top3 = summary[summary["topk"].eq("top3")][focused_cols].round(4)
    top5 = summary[summary["topk"].eq("top5")][focused_cols].round(4)
    counts_df = pd.DataFrame(
        [{"item": key, "count": value} for key, value in sorted(counts.items())]
    )
    with report_path.open("w") as handle:
        handle.write("# RND-EPI Similarity Metric Distribution\n\n")
        handle.write("## Setup\n\n")
        handle.write("- Seed set: `data/rnd-epi/epi_53_molecules_3d.sdf` (53 RND-EPI molecules).\n")
        handle.write("- ChEMBL source: `data/chembl/chembl_36_properties.csv.gz`.\n")
        handle.write("- Metrics: ECFP4, ECFP6, FCFP4, MACCS, AtomPair, TopologicalTorsion, RDKit 2D descriptor similarity.\n")
        handle.write("- For each ChEMBL molecule and metric, top1/top3/top5 are the mean of the best 1/3/5 similarities to the 53 seeds.\n")
        handle.write("- Filters: only obvious abnormal molecules were excluded.\n\n")
        handle.write("## Counts\n\n")
        handle.write(_format_markdown_table(counts_df, index=False))
        handle.write("\n\n## Top1 Summary\n\n")
        handle.write(_format_markdown_table(top1, index=False))
        handle.write("\n\n## Top3 Summary\n\n")
        handle.write(_format_markdown_table(top3, index=False))
        handle.write("\n\n## Top5 Summary\n\n")
        handle.write(_format_markdown_table(top5, index=False))
        handle.write("\n\n## Figures\n\n")
        handle.write("![Top-k histograms](topk_similarity_histograms.png)\n\n")
        handle.write("![Top-k boxplots](topk_similarity_boxplots.png)\n\n")
        handle.write("![Top1 metric correlation](top1_metric_correlation.png)\n\n")
        handle.write("## Top1 Correlation Matrix\n\n")
        handle.write(_format_markdown_table(corr.round(3), index=True))
        handle.write("\n\n## Output Files\n\n")
        handle.write("- `similarity_topk_scores.csv.gz`: per-molecule top1/top3/top5 scores.\n")
        handle.write("- `similarity_topk_summary.csv`: metric distribution summary.\n")
        handle.write("- `top1_metric_correlation.csv`: top1 metric correlation matrix.\n")
        handle.write("- `abnormal_filter_counts.json`: abnormal-filter counts.\n")


def main() -> None:
    """Run the top-k similarity distribution analysis."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=OUTPUT_DIR / "analyze_similarity_topk.log",
        filemode="w",
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=logging.INFO,
    )
    LOGGER.addHandler(logging.StreamHandler())
    df, counts = _collect_similarity_table()
    LOGGER.info("Writing outputs for %d analyzed records", len(df))
    with gzip.open(OUTPUT_DIR / "similarity_topk_scores.csv.gz", "wt", newline="") as handle:
        df.to_csv(handle, index=False)
    summary = _summary_table(df)
    summary.to_csv(OUTPUT_DIR / "similarity_topk_summary.csv", index=False)
    with (OUTPUT_DIR / "abnormal_filter_counts.json").open("w") as handle:
        json.dump(counts, handle, indent=2, ensure_ascii=False)
    corr = _write_metric_correlation(df)
    _plot_histograms(df)
    _plot_boxplots(df)
    _plot_correlation(corr)
    _write_report(summary, counts, corr)
    LOGGER.info("Done. Outputs written to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
