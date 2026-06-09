from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "OUT"
GEN_DIR = OUT_DIR / "260407_generate_ligands_10000"
ANALYSIS_DIR = ROOT / "analysis" / "rl_data_sufficiency_20260608"
CLASSIFIER_CSV = GEN_DIR / "classifier_selected_v0_1_base_distribution" / "base_diffsbdd_classifier_scores.csv"


def canonical_smiles(mol: Chem.Mol | None) -> str | None:
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def scaffold_smiles(smiles: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold, canonical=True) if scaffold is not None else ""
    except Exception:
        return ""


def summarize_sdfs() -> tuple[list[dict], set[str]]:
    rows: list[dict] = []
    unique_smiles: set[str] = set()
    sdf_paths = sorted(GEN_DIR.glob("*.sdf"))
    for path in sdf_paths:
        supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
        record_count = 0
        valid_count = 0
        file_unique: set[str] = set()
        for mol in supplier:
            record_count += 1
            smi = canonical_smiles(mol)
            if smi is None:
                continue
            valid_count += 1
            file_unique.add(smi)
            unique_smiles.add(smi)
        rows.append(
            {
                "source": str(path.relative_to(ROOT)),
                "kind": "sdf",
                "records": record_count,
                "rdkit_valid": valid_count,
                "valid_ratio": valid_count / record_count if record_count else 0.0,
                "unique_smiles": len(file_unique),
            }
        )
    return rows, unique_smiles


def summarize_classifier_csv() -> dict:
    if not CLASSIFIER_CSV.exists():
        return {}
    df = pd.read_csv(CLASSIFIER_CSV)
    smiles = df["smiles"].dropna().astype(str)
    unique_smiles = set(smiles)
    scaffolds = [scaffold_smiles(smi) for smi in unique_smiles]
    scaffold_counts = Counter(s for s in scaffolds if s)
    n = len(df)
    return {
        "rows": int(n),
        "unique_smiles": int(len(unique_smiles)),
        "unique_scaffolds": int(len(scaffold_counts)),
        "top_scaffold_fraction": float(scaffold_counts.most_common(1)[0][1] / len(unique_smiles)) if unique_smiles and scaffold_counts else 0.0,
        "classifier_prob_ge_0_5_rate": float((df["classifier_prob"] >= 0.5).mean()) if "classifier_prob" in df else None,
        "classifier_prob_ge_0_8_rate": float((df["classifier_prob"] >= 0.8).mean()) if "classifier_prob" in df else None,
        "carbon_gt_0_8_rate": float((df["carbon_fraction"] > 0.8).mean()) if "carbon_fraction" in df else None,
        "logp_gt_4_rate": float((df["logp"] > 4.0).mean()) if "logp" in df else None,
        "tpsa_lt_40_rate": float((df["tpsa"] < 40.0).mean()) if "tpsa" in df else None,
        "mean_logp": float(df["logp"].mean()) if "logp" in df else None,
        "mean_tpsa": float(df["tpsa"].mean()) if "tpsa" in df else None,
        "mean_carbon_fraction": float(df["carbon_fraction"].mean()) if "carbon_fraction" in df else None,
    }


def summarize_training_csvs(max_unique_collect: int = 300000) -> tuple[dict, list[dict]]:
    paths = sorted(OUT_DIR.glob("*/epochs/epoch_*/molecules.csv"))
    total_rows = 0
    valid_rows = 0
    unique_smiles: set[str] = set()
    run_rows: defaultdict[str, int] = defaultdict(int)
    run_valid: defaultdict[str, int] = defaultdict(int)
    run_files: defaultdict[str, int] = defaultdict(int)
    columns_seen: set[str] = set()

    for path in paths:
        run_name = path.parts[path.parts.index("OUT") + 1]
        try:
            with open(path, "r", encoding="utf-8", errors="replace", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames:
                    columns_seen.update(reader.fieldnames)
                count = 0
                valid = 0
                for row in reader:
                    count += 1
                    smi = row.get("smiles") or row.get("SMILES") or ""
                    if smi and len(unique_smiles) < max_unique_collect:
                        unique_smiles.add(str(smi))
                    rdkit_valid = row.get("rdkit_valid")
                    if rdkit_valid is not None:
                        try:
                            if int(float(rdkit_valid)) == 1:
                                valid += 1
                        except ValueError:
                            pass
                total_rows += count
                valid_rows += valid
                run_rows[run_name] += count
                run_valid[run_name] += valid
                run_files[run_name] += 1
        except Exception:
            continue

    top_runs = []
    for run_name, count in sorted(run_rows.items(), key=lambda item: item[1], reverse=True)[:30]:
        top_runs.append(
            {
                "run": run_name,
                "molecule_rows": int(count),
                "valid_rows": int(run_valid[run_name]),
                "valid_ratio": float(run_valid[run_name] / count) if count else 0.0,
                "molecules_csv_files": int(run_files[run_name]),
            }
        )
    summary = {
        "molecules_csv_files": int(len(paths)),
        "molecule_rows": int(total_rows),
        "valid_rows_with_rdkit_valid_1": int(valid_rows),
        "valid_ratio_from_available_rdkit_valid": float(valid_rows / total_rows) if total_rows else 0.0,
        "unique_smiles_collected_cap": int(len(unique_smiles)),
        "has_reward_total": "reward_total" in columns_seen,
        "has_reward_total_clamped": "reward_total_clamped" in columns_seen,
        "has_docking_raw": "docking_raw" in columns_seen,
        "has_classifier_reward": "chembl_classifier_reward" in columns_seen,
        "has_carbon_fraction": "carbon_fraction" in columns_seen,
        "has_bad_bond_length_fraction": "bad_bond_length_fraction" in columns_seen,
    }
    return summary, top_runs


def plot_classifier_descriptors() -> None:
    if not CLASSIFIER_CSV.exists():
        return
    df = pd.read_csv(CLASSIFIER_CSV)
    cols = [c for c in ["carbon_fraction", "hetero_fraction", "logp", "tpsa", "classifier_prob"] if c in df]
    if not cols:
        return
    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 3.2), dpi=160)
    if len(cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        ax.hist(df[col].dropna(), bins=40, color="#4C78A8", alpha=0.8)
        ax.set_title(col)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "base_diffsbdd_descriptor_histograms.png")
    fig.savefig(ANALYSIS_DIR / "base_diffsbdd_descriptor_histograms.svg")


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    sdf_rows, sdf_unique = summarize_sdfs()
    classifier_summary = summarize_classifier_csv()
    training_summary, top_runs = summarize_training_csvs()
    plot_classifier_descriptors()

    source_rows = list(sdf_rows)
    with open(ANALYSIS_DIR / "sdf_source_summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "kind", "records", "rdkit_valid", "valid_ratio", "unique_smiles"])
        writer.writeheader()
        writer.writerows(source_rows)

    with open(ANALYSIS_DIR / "top_training_runs_by_molecules.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", "molecule_rows", "valid_rows", "valid_ratio", "molecules_csv_files"])
        writer.writeheader()
        writer.writerows(top_runs)

    summary = {
        "sdf_total_records": int(sum(row["records"] for row in sdf_rows)),
        "sdf_total_rdkit_valid": int(sum(row["rdkit_valid"] for row in sdf_rows)),
        "sdf_union_unique_smiles": int(len(sdf_unique)),
        "classifier_csv": classifier_summary,
        "training_csv": training_summary,
    }
    with open(ANALYSIS_DIR / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
