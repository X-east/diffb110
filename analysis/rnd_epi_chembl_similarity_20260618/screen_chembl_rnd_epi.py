#!/usr/bin/env python3
"""Screen local ChEMBL molecules against known RND-EPI seed molecules."""

from __future__ import annotations

import csv
import gzip
import json
import logging
import multiprocessing as mp
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem.Scaffolds import MurckoScaffold


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "rnd_epi_chembl_similarity_20260618"
SEED_SDF = PROJECT_ROOT / "data" / "rnd-epi" / "epi_53_molecules_3d.sdf"
CHEMBL_PROPERTIES = PROJECT_ROOT / "data" / "chembl" / "chembl_36_properties.csv.gz"
N_JOBS = 18
CHUNK_SIZE = 10_000

LOGGER = logging.getLogger("rnd_epi_screen")
_WORKER_SEEDS: list[dict[str, object]] = []
_WORKER_SEED_SMILES: set[str] = set()
_WORKER_SEED_FPS: dict[str, list[DataStructs.ExplicitBitVect]] = {}


@dataclass(frozen=True)
class ScreenGroup:
    """One candidate-pool selection rule."""

    group_id: str
    name: str
    purpose: str

    def match(self, row: dict[str, object]) -> bool:
        """Return whether one molecule matches this group."""
        ecfp4 = float(row["ecfp4_max"])
        ecfp6 = float(row["ecfp6_max"])
        fcfp4 = float(row["fcfp4_max"])
        maccs = float(row["maccs_max"])

        if self.group_id == "G01":
            return bool(row["is_seed_exact"])
        if self.group_id == "G02":
            return ecfp4 >= 0.70 or ecfp6 >= 0.65
        if self.group_id == "G03":
            return fcfp4 >= 0.70
        if self.group_id == "G04":
            return ecfp4 >= 0.55 or ecfp6 >= 0.50
        if self.group_id == "G05":
            return fcfp4 >= 0.55
        if self.group_id == "G06":
            return maccs >= 0.80
        if self.group_id == "G07":
            return max(ecfp4, ecfp6, fcfp4) >= 0.40
        if self.group_id == "G08":
            return fcfp4 >= 0.45 and ecfp4 < 0.40
        if self.group_id == "G09":
            return maccs >= 0.70 and fcfp4 >= 0.30
        if self.group_id == "G10":
            return max(ecfp4, ecfp6, fcfp4) >= 0.30 or maccs >= 0.65
        raise ValueError(f"Unknown group: {self.group_id}")


GROUPS: list[ScreenGroup] = [
    ScreenGroup("G01", "exact_seed", "ChEMBL entry exactly matches one of 53 RND-EPI seeds"),
    ScreenGroup("G02", "strict_structural", "ECFP4 >= 0.70 or ECFP6 >= 0.65 to any seed"),
    ScreenGroup("G03", "strict_functional", "FCFP4 >= 0.70 to any seed"),
    ScreenGroup("G04", "medium_high_structural", "ECFP4 >= 0.55 or ECFP6 >= 0.50 to any seed"),
    ScreenGroup("G05", "medium_high_functional", "FCFP4 >= 0.55 to any seed"),
    ScreenGroup("G06", "maccs_broad", "MACCS >= 0.80 to any seed"),
    ScreenGroup("G07", "broad_fingerprint_union", "max(ECFP4, ECFP6, FCFP4) >= 0.40 to any seed"),
    ScreenGroup("G08", "scaffold_hopping", "FCFP4 >= 0.45 and ECFP4 < 0.40 to any seed"),
    ScreenGroup("G09", "broad_pharmacophore", "MACCS >= 0.70 and FCFP4 >= 0.30 to any seed"),
    ScreenGroup("G10", "very_broad_prepool", "max(ECFP4, ECFP6, FCFP4) >= 0.30 or MACCS >= 0.65"),
]


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


def _load_seed_molecules() -> list[dict[str, object]]:
    """Load RND-EPI seed molecules and their fingerprints."""
    seeds: list[dict[str, object]] = []
    supplier = Chem.SDMolSupplier(str(SEED_SDF), removeHs=False)
    for index, raw_mol in enumerate(supplier, start=1):
        mol = _canonicalize_mol(raw_mol)
        if mol is None:
            LOGGER.warning("Skipping unreadable seed at index %d", index)
            continue
        smiles = Chem.MolToSmiles(mol, canonical=True)
        seeds.append(
            {
                "seed_index": index,
                "seed_name": raw_mol.GetProp("Preferred_Name")
                if raw_mol.HasProp("Preferred_Name")
                else raw_mol.GetProp("_Name"),
                "seed_short_name": raw_mol.GetProp("Short_Name")
                if raw_mol.HasProp("Short_Name")
                else "",
                "seed_smiles": smiles,
                "ecfp4": AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048),
                "ecfp6": AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048),
                "fcfp4": AllChem.GetMorganFingerprintAsBitVect(
                    mol, radius=2, nBits=2048, useFeatures=True
                ),
                "maccs": MACCSkeys.GenMACCSKeys(mol),
            }
        )
    if not seeds:
        raise RuntimeError(f"No valid seed molecules loaded from {SEED_SDF}")
    return seeds


def _init_worker(seeds: list[dict[str, object]]) -> None:
    """Initialize worker-local seed fingerprint caches."""
    global _WORKER_SEEDS, _WORKER_SEED_SMILES, _WORKER_SEED_FPS
    _WORKER_SEEDS = seeds
    _WORKER_SEED_SMILES = {str(seed["seed_smiles"]) for seed in seeds}
    _WORKER_SEED_FPS = {
        "ecfp4": [seed["ecfp4"] for seed in seeds],
        "ecfp6": [seed["ecfp6"] for seed in seeds],
        "fcfp4": [seed["fcfp4"] for seed in seeds],
        "maccs": [seed["maccs"] for seed in seeds],
    }


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


def _max_similarity(
    fp: DataStructs.ExplicitBitVect,
    seed_fps: list[DataStructs.ExplicitBitVect],
) -> tuple[float, int]:
    """Return max Tanimoto similarity and best seed offset."""
    similarities = DataStructs.BulkTanimotoSimilarity(fp, seed_fps)
    best_index = max(range(len(similarities)), key=similarities.__getitem__)
    return float(similarities[best_index]), int(best_index)


def _scaffold_smiles(mol: Chem.Mol) -> str:
    """Return Bemis-Murcko scaffold SMILES, falling back to empty string."""
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return ""
        return Chem.MolToSmiles(scaffold, canonical=True)
    except Exception:
        return ""


def _screen_chunk(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Screen one ChEMBL property-table chunk in a worker process."""
    matched_records: list[dict[str, object]] = []
    abnormal_counts: Counter[str] = Counter()
    normal_count = 0

    for props in rows:
        abnormal_counts["total_records"] += 1
        chembl_id = props["chembl_id"]
        smiles = str(props["canonical_smiles"])
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            abnormal_counts["rdkit_parse_failed"] += 1
            continue
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            abnormal_counts["sanitize_failed"] += 1
            continue

        allowed, reason = _normal_allowed(mol, props)
        if not allowed:
            abnormal_counts[reason] += 1
            continue
        normal_count += 1

        ecfp4_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        ecfp6_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048)
        fcfp4_fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=2, nBits=2048, useFeatures=True
        )
        maccs_fp = MACCSkeys.GenMACCSKeys(mol)
        ecfp4_max, ecfp4_best = _max_similarity(ecfp4_fp, _WORKER_SEED_FPS["ecfp4"])
        ecfp6_max, ecfp6_best = _max_similarity(ecfp6_fp, _WORKER_SEED_FPS["ecfp6"])
        fcfp4_max, fcfp4_best = _max_similarity(fcfp4_fp, _WORKER_SEED_FPS["fcfp4"])
        maccs_max, maccs_best = _max_similarity(maccs_fp, _WORKER_SEED_FPS["maccs"])

        best_metric_values = {
            "ecfp4": ecfp4_max,
            "ecfp6": ecfp6_max,
            "fcfp4": fcfp4_max,
            "maccs": maccs_max,
        }
        best_metric = max(best_metric_values, key=best_metric_values.get)
        best_seed_index = {
            "ecfp4": ecfp4_best,
            "ecfp6": ecfp6_best,
            "fcfp4": fcfp4_best,
            "maccs": maccs_best,
        }[best_metric]
        best_seed = _WORKER_SEEDS[best_seed_index]

        row = {
            "chembl_id": chembl_id,
            "canonical_smiles": smiles,
            "is_seed_exact": smiles in _WORKER_SEED_SMILES,
            "ecfp4_max": ecfp4_max,
            "ecfp6_max": ecfp6_max,
            "fcfp4_max": fcfp4_max,
            "maccs_max": maccs_max,
            "best_metric": best_metric,
            "best_metric_similarity": best_metric_values[best_metric],
            "best_seed_index": best_seed["seed_index"],
            "best_seed_name": best_seed["seed_name"],
            "best_seed_smiles": best_seed["seed_smiles"],
            "heavy_atoms": int(props["heavy_atoms"]),
            "formal_charge": int(props["formal_charge"]),
            "num_rings": int(props["num_rings"]),
            "mol_wt": float(props["mol_wt"]),
            "mol_log_p": float(props["mol_log_p"]),
            "tpsa": float(props["tpsa"]),
            "num_h_acceptors": int(props["num_h_acceptors"]),
            "num_h_donors": int(props["num_h_donors"]),
            "num_rotatable_bonds": int(props["num_rotatable_bonds"]),
            "num_aromatic_rings": int(props["num_aromatic_rings"]),
            "num_heteroatoms": int(props["num_heteroatoms"]),
            "num_carbon_atoms": int(props["num_carbon_atoms"]),
            "carbon_fraction": float(props["carbon_fraction"]),
        }
        groups = [group.group_id for group in GROUPS if group.match(row)]
        if groups:
            row["matched_groups"] = ";".join(groups)
            row["broadest_group"] = groups[-1]
            row["strictest_group"] = groups[0]
            row["scaffold_smiles"] = _scaffold_smiles(mol)
            matched_records.append(row)

    abnormal_counts["normal_records"] = normal_count
    abnormal_counts["matched_records"] = len(matched_records)
    return matched_records, dict(abnormal_counts)


def _iter_screened_records() -> tuple[list[dict[str, object]], dict[str, int]]:
    """Return matched ChEMBL records and abnormal-filter counts."""
    seeds = _load_seed_molecules()
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

    matched_records: list[dict[str, object]] = []
    abnormal_counts: Counter[str] = Counter()
    processed = 0
    LOGGER.info("Using %d worker processes, chunk_size=%d", N_JOBS, CHUNK_SIZE)
    with mp.Pool(processes=N_JOBS, initializer=_init_worker, initargs=(seeds,)) as pool:
        chunk_iter = pd.read_csv(CHEMBL_PROPERTIES, usecols=usecols, chunksize=CHUNK_SIZE)
        row_iter = (chunk.to_dict("records") for chunk in chunk_iter)
        for chunk_matches, chunk_counts in pool.imap_unordered(_screen_chunk, row_iter, chunksize=1):
            matched_records.extend(chunk_matches)
            abnormal_counts.update(chunk_counts)
            processed += int(chunk_counts.get("total_records", 0))
            if processed and processed % 100000 == 0:
                LOGGER.info(
                    "Processed %d ChEMBL rows, normal=%d, matched=%d",
                    processed,
                    abnormal_counts.get("normal_records", 0),
                    len(matched_records),
                )

    abnormal_counts["matched_records"] = len(matched_records)
    abnormal_counts["seed_count"] = len(seeds)
    LOGGER.info(
        "Screened total=%d, normal=%d, matched=%d",
        abnormal_counts.get("total_records", 0),
        abnormal_counts.get("normal_records", 0),
        len(matched_records),
    )
    return matched_records, dict(abnormal_counts)


def _write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    """Write rows to CSV."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _assign_groups(records: list[dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assign group memberships and build summary tables."""
    df = pd.DataFrame.from_records(records)
    matched_df = df.copy()

    cumulative_ids: set[str] = set()
    summary_rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    for group in GROUPS:
        mask = df["matched_groups"].str.contains(group.group_id, regex=False)
        group_df = df[mask].copy()
        group_ids = set(group_df["chembl_id"].astype(str))
        exclusive_ids = group_ids - cumulative_ids
        exclusive_df = group_df[group_df["chembl_id"].astype(str).isin(exclusive_ids)]
        cumulative_ids |= group_ids

        summary_rows.append(
            {
                "group_id": group.group_id,
                "name": group.name,
                "purpose": group.purpose,
                "cumulative_hits": len(group_df),
                "exclusive_hits": len(exclusive_df),
                "unique_smiles": group_df["canonical_smiles"].nunique(),
                "scaffold_count": group_df["scaffold_smiles"].replace("", pd.NA).nunique(),
                "seed_coverage": group_df["best_seed_index"].nunique() if len(group_df) else 0,
                "ecfp4_median": group_df["ecfp4_max"].median() if len(group_df) else 0.0,
                "ecfp6_median": group_df["ecfp6_max"].median() if len(group_df) else 0.0,
                "fcfp4_median": group_df["fcfp4_max"].median() if len(group_df) else 0.0,
                "maccs_median": group_df["maccs_max"].median() if len(group_df) else 0.0,
                "mw_median": group_df["mol_wt"].median() if len(group_df) else 0.0,
                "logp_median": group_df["mol_log_p"].median() if len(group_df) else 0.0,
                "tpsa_median": group_df["tpsa"].median() if len(group_df) else 0.0,
                "carbon_fraction_median": group_df["carbon_fraction"].median() if len(group_df) else 0.0,
            }
        )

        if len(group_df):
            per_seed = group_df.groupby(["best_seed_index", "best_seed_name"]).size().reset_index(name="hits")
            for _, seed_row in per_seed.iterrows():
                seed_rows.append(
                    {
                        "group_id": group.group_id,
                        "best_seed_index": int(seed_row["best_seed_index"]),
                        "best_seed_name": seed_row["best_seed_name"],
                        "hits": int(seed_row["hits"]),
                    }
                )

    return matched_df, pd.DataFrame(summary_rows), pd.DataFrame(seed_rows)


def _plot_summary(summary: pd.DataFrame, matched_df: pd.DataFrame) -> None:
    """Generate summary plots."""
    plt.figure(figsize=(11, 5))
    x = summary["group_id"]
    plt.bar(x, summary["cumulative_hits"], label="single-rule hits")
    plt.bar(x, summary["exclusive_hits"], label="new hits after stricter groups")
    plt.yscale("log")
    plt.ylabel("Molecules (log scale)")
    plt.xlabel("Screen group")
    plt.title("RND-EPI ChEMBL candidate counts by fingerprint rule")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "group_hit_counts.png", dpi=180)
    plt.close()

    if len(matched_df):
        plt.figure(figsize=(8, 5))
        plt.hist(matched_df["ecfp4_max"], bins=40, alpha=0.55, label="ECFP4")
        plt.hist(matched_df["ecfp6_max"], bins=40, alpha=0.55, label="ECFP6")
        plt.hist(matched_df["fcfp4_max"], bins=40, alpha=0.55, label="FCFP4")
        plt.hist(matched_df["maccs_max"], bins=40, alpha=0.55, label="MACCS")
        plt.xlabel("Max Tanimoto similarity to any of 53 seeds")
        plt.ylabel("Matched candidates")
        plt.title("Similarity distribution among broad matched candidates")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "matched_similarity_distribution.png", dpi=180)
        plt.close()

        plt.figure(figsize=(9, 5))
        plt.scatter(
            matched_df["mol_wt"],
            matched_df["mol_log_p"],
            c=matched_df["best_metric_similarity"],
            s=8,
            cmap="viridis",
            alpha=0.7,
        )
        plt.xlabel("MW")
        plt.ylabel("LogP")
        plt.title("RND-EPI matched candidates: MW vs LogP")
        cbar = plt.colorbar()
        cbar.set_label("Best similarity")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "matched_mw_logp.png", dpi=180)
        plt.close()


def _format_markdown_table(df: pd.DataFrame, index: bool = False) -> str:
    """Format a DataFrame as a simple Markdown table without extra dependencies."""
    table = df.reset_index() if index else df.copy()
    table = table.fillna("")
    headers = [str(column) for column in table.columns]
    rows = [[str(value) for value in row] for row in table.to_numpy().tolist()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _write_report(
    summary: pd.DataFrame,
    abnormal_counts: dict[str, int],
    matched_df: pd.DataFrame,
    seed_summary: pd.DataFrame,
) -> None:
    """Write the Markdown report."""
    report_path = OUTPUT_DIR / "rnd_epi_chembl_similarity_report.md"
    abnormal_df = (
        pd.DataFrame(
            [{"reason": key, "count": value} for key, value in sorted(abnormal_counts.items())]
        )
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    top_seed = seed_summary.sort_values("hits", ascending=False).head(15) if len(seed_summary) else seed_summary
    prop_cols = [
        "mol_wt",
        "mol_log_p",
        "tpsa",
        "heavy_atoms",
        "num_rings",
        "num_aromatic_rings",
        "carbon_fraction",
    ]
    prop_summary = matched_df[prop_cols].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).T
    prop_summary = prop_summary[["min", "10%", "25%", "50%", "75%", "90%", "max"]]

    with report_path.open("w") as handle:
        handle.write("# RND-EPI ChEMBL Similarity Screen\n\n")
        handle.write("## Setup\n\n")
        handle.write("- Seed set: `data/rnd-epi/epi_53_molecules_3d.sdf`\n")
        handle.write("- ChEMBL source: `data/chembl/chembl_36_properties.csv.gz`\n")
        handle.write("- Rule: a ChEMBL molecule matches a group if it reaches the threshold against **any one** of the 53 RND-EPI seeds.\n")
        handle.write("- Filters: only obvious abnormal molecules were removed; property ranges were not used as drug-likeness gates.\n\n")
        handle.write("## Group Counts\n\n")
        handle.write(_format_markdown_table(summary, index=False))
        handle.write("\n\n![Group hit counts](group_hit_counts.png)\n\n")
        handle.write("## Abnormal Filter Counts\n\n")
        handle.write(_format_markdown_table(abnormal_df, index=False))
        handle.write("\n\n## Matched Candidate Property Distribution\n\n")
        handle.write(_format_markdown_table(prop_summary.round(3), index=True))
        handle.write("\n\n![Similarity distribution](matched_similarity_distribution.png)\n\n")
        handle.write("![MW vs LogP](matched_mw_logp.png)\n\n")
        handle.write("## Top Seed Coverage\n\n")
        handle.write(_format_markdown_table(top_seed, index=False))
        handle.write("\n\n## Output Files\n\n")
        handle.write("- `group_summary.csv`: per-group hit counts and medians.\n")
        handle.write("- `matched_candidates.csv.gz`: all molecules matched by at least one group.\n")
        handle.write("- `seed_coverage_by_group.csv`: per-group best-seed hit counts.\n")
        handle.write("- `abnormal_filter_counts.json`: reasons for obvious-abnormal exclusions.\n")


def main() -> None:
    """Run the ChEMBL screen."""
    logging.basicConfig(
        filename=OUTPUT_DIR / "screen_chembl_rnd_epi.log",
        filemode="w",
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=logging.INFO,
    )
    LOGGER.addHandler(logging.StreamHandler())
    LOGGER.info("Starting RND-EPI ChEMBL similarity screen")

    records, abnormal_counts = _iter_screened_records()
    matched_df, summary, seed_summary = _assign_groups(records)
    LOGGER.info("Matched candidates: %d / normal records: %d", len(matched_df), len(records))

    summary.to_csv(OUTPUT_DIR / "group_summary.csv", index=False)
    seed_summary.to_csv(OUTPUT_DIR / "seed_coverage_by_group.csv", index=False)
    with gzip.open(OUTPUT_DIR / "matched_candidates.csv.gz", "wt", newline="") as handle:
        matched_df.to_csv(handle, index=False)
    with (OUTPUT_DIR / "abnormal_filter_counts.json").open("w") as handle:
        json.dump(abnormal_counts, handle, indent=2, ensure_ascii=False)

    _plot_summary(summary, matched_df)
    _write_report(summary, abnormal_counts, matched_df, seed_summary)
    LOGGER.info("Done. Outputs written to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
