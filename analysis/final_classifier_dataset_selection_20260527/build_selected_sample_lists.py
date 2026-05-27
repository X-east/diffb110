#!/usr/bin/env python3
"""Build selected positive and negative sample lists for classifier v0.1."""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-diffb110")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold


RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = Path(__file__).resolve().parent
DRYRUN_DIR = ROOT / "analysis/dataset_selection_dryrun_20260527"

SEED = 42
N_WORKERS = 18
TARGET_POSITIVES = 100_000
TARGET_NEGATIVES = 70_000
POSITIVE_OVERSAMPLE = 135_000
MAX_POSITIVES_PER_SCAFFOLD = 10

NEGATIVE_QUOTAS = {
    "high_reward_bad": 9_000,
    "co_only_mostly_carbon": 10_000,
    "aromatic_deficient_bad_topology": 10_000,
    "high_carbon_low_hetero": 9_000,
    "non_oxygen_hetero_loss": 9_000,
    "hydrophobic_low_polarity": 9_000,
    "ring_complex": 10_000,
    "three_member_ring": 10_000,
}

LABEL_PRIORITY = {
    "high_reward_bad": 10,
    "co_only_mostly_carbon": 9,
    "aromatic_deficient_bad_topology": 8,
    "high_carbon_low_hetero": 7,
    "non_oxygen_hetero_loss": 6,
    "hydrophobic_low_polarity": 5,
    "ring_complex": 4,
    "three_member_ring": 3,
}


def scaffold_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return ""
    return Chem.MolToSmiles(scaffold, canonical=True)


def add_aromatic_class(pool: pd.DataFrame) -> pd.DataFrame:
    pool = pool.copy()
    pool["aromatic_carbocycle_approx"] = (
        pool["num_aromatic_rings"] - pool["num_aromatic_heterocycles"]
    ).clip(lower=0)
    has_carb = pool["aromatic_carbocycle_approx"].gt(0)
    has_hetero = pool["num_aromatic_heterocycles"].gt(0)
    pool["aromatic_class"] = np.select(
        [
            has_carb & has_hetero,
            has_carb,
            has_hetero,
        ],
        ["mixed_aromatic", "benzene_like_only", "heteroaromatic_only"],
        default="no_aromatic",
    )
    pool["final_positive_stratum"] = pool["positive_stratum"].astype(str) + "|" + pool["aromatic_class"].astype(str)
    return pool


def proportional_stratified_sample(
    df: pd.DataFrame,
    stratum_col: str,
    n: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    counts = df[stratum_col].value_counts()
    raw = counts / counts.sum() * n
    alloc = np.floor(raw).astype(int)
    alloc[alloc == 0] = 1
    alloc = np.minimum(alloc, counts)

    diff = n - int(alloc.sum())
    if diff > 0:
        remainders = (raw - np.floor(raw)).sort_values(ascending=False)
        for key in remainders.index:
            if diff <= 0:
                break
            if alloc[key] < counts[key]:
                alloc[key] += 1
                diff -= 1
    elif diff < 0:
        for key in alloc.sort_values(ascending=False).index:
            if diff >= 0:
                break
            if alloc[key] > 1:
                alloc[key] -= 1
                diff += 1

    parts = []
    for key, take in alloc.items():
        group = df[df[stratum_col].eq(key)]
        parts.append(group.sample(n=int(take), random_state=int(rng.integers(0, 2**31 - 1))))
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def select_positives() -> pd.DataFrame:
    pool = pd.read_csv(DRYRUN_DIR / "positive_candidate_pool.csv.gz", compression="gzip")
    pool = add_aromatic_class(pool)
    oversampled = proportional_stratified_sample(pool, "final_positive_stratum", POSITIVE_OVERSAMPLE, SEED)

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        scaffolds = list(executor.map(scaffold_smiles, oversampled["canonical_smiles"].tolist(), chunksize=1000))
    oversampled["murcko_scaffold"] = scaffolds

    selected_rows = []
    scaffold_counts: dict[str, int] = {}
    for _, row in oversampled.iterrows():
        scaffold = row["murcko_scaffold"] or f"empty:{row['chembl_id']}"
        if scaffold_counts.get(scaffold, 0) >= MAX_POSITIVES_PER_SCAFFOLD:
            continue
        selected_rows.append(row)
        scaffold_counts[scaffold] = scaffold_counts.get(scaffold, 0) + 1
        if len(selected_rows) >= TARGET_POSITIVES:
            break

    if len(selected_rows) < TARGET_POSITIVES:
        selected_ids = {row["chembl_id"] for row in selected_rows}
        remainder = oversampled[~oversampled["chembl_id"].isin(selected_ids)]
        need = TARGET_POSITIVES - len(selected_rows)
        selected_rows.extend(remainder.head(need).to_dict("records"))

    selected = pd.DataFrame(selected_rows).head(TARGET_POSITIVES).copy()
    selected["label"] = 1
    return selected


def label_score(labels: str) -> int:
    return sum(LABEL_PRIORITY.get(label, 0) for label in str(labels).split("|"))


def select_negatives() -> pd.DataFrame:
    neg = pd.read_csv(DRYRUN_DIR / "negative_candidates_labeled.csv.gz", compression="gzip")
    neg["label_score"] = neg["labels"].map(label_score)
    neg = neg.sort_values(
        ["top_reward_decile_any", "label_score", "max_reward_total_raw"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    selected_indices: set[int] = set()
    for label, quota in NEGATIVE_QUOTAS.items():
        subset = neg[neg["labels"].str.contains(label, regex=False, na=False)]
        chosen = 0
        for idx in subset.index:
            if idx in selected_indices:
                continue
            selected_indices.add(idx)
            chosen += 1
            if chosen >= quota:
                break

    selected = neg.loc[sorted(selected_indices)].copy()
    if len(selected) < TARGET_NEGATIVES:
        remaining = neg.drop(index=selected.index)
        selected = pd.concat([selected, remaining.head(TARGET_NEGATIVES - len(selected))], ignore_index=True)
    elif len(selected) > TARGET_NEGATIVES:
        selected = selected.sort_values(
            ["top_reward_decile_any", "label_score", "max_reward_total_raw"],
            ascending=[False, False, False],
        ).head(TARGET_NEGATIVES)

    selected["label"] = 0
    return selected.sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def explode_label_counts(neg: pd.DataFrame) -> pd.DataFrame:
    counts: dict[str, int] = {}
    for labels in neg["labels"].fillna(""):
        for label in labels.split("|"):
            if label:
                counts[label] = counts.get(label, 0) + 1
    return pd.DataFrame(
        [{"label": key, "selected_count": value} for key, value in sorted(counts.items(), key=lambda x: -x[1])]
    )


def write_smi_files(pos: pd.DataFrame, neg: pd.DataFrame) -> None:
    pos_smi = pd.DataFrame({"smiles": pos["canonical_smiles"], "label": 1})
    pos_smi.to_csv(ANALYSIS_DIR / "selected_positives_100k.smi", sep="\t", index=False)

    neg_smi = pd.DataFrame(
        {
            "smiles": neg["canonical_smiles"],
            "label": neg["primary_label"],
            "all_labels": neg["labels"],
        }
    )
    neg_smi.to_csv(ANALYSIS_DIR / "selected_negatives_70k_labeled.smi", sep="\t", index=False)

    combined = pd.concat(
        [
            pd.DataFrame({"smiles": pos["canonical_smiles"], "label": 1, "source": "chembl_strict"}),
            pd.DataFrame({"smiles": neg["canonical_smiles"], "label": 0, "source": neg["primary_label"]}),
        ],
        ignore_index=True,
    ).sample(frac=1.0, random_state=SEED)
    combined.to_csv(ANALYSIS_DIR / "selected_classifier_dataset_170k.tsv", sep="\t", index=False)


def plot_outputs(pos: pd.DataFrame, neg_counts: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    pos["aromatic_class"].value_counts().plot(kind="bar", ax=axes[0], color="#2b6cb0")
    axes[0].set_title("Selected positives by aromatic class")
    axes[0].tick_params(axis="x", rotation=25)
    pos["heterocycle_class"].value_counts().plot(kind="bar", ax=axes[1], color="#2f855a")
    axes[1].set_title("Selected positives by heterocycle class")
    axes[1].tick_params(axis="x", rotation=25)
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "selected_positive_distribution.png", dpi=220)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    view = neg_counts.sort_values("selected_count", ascending=True)
    ax.barh(view["label"], view["selected_count"], color="#c2410c")
    ax.set_xlabel("Selected negatives")
    ax.set_title("Selected negative category coverage")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "selected_negative_category_coverage.png", dpi=220)
    plt.close()


def write_report(pos: pd.DataFrame, neg: pd.DataFrame, neg_counts: pd.DataFrame) -> None:
    pos_aromatic = pos["aromatic_class"].value_counts(normalize=True).mul(100)
    count_lines = "\n".join(
        f"- `{row.label}`: {int(row.selected_count):,}"
        for row in neg_counts.itertuples(index=False)
    )
    aromatic_lines = "\n".join(
        f"- `{key}`: {value:.2f}%"
        for key, value in pos_aromatic.items()
    )

    report = f"""# 分类器训练样本选择 v0.1

生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 一、产物性质

这是训练样本清单的 v0.1 候选版本，仍然放在 analysis 目录中，没有覆盖
`chembl_classifier/data`。

## 二、样本规模

- selected positives：{len(pos):,}
- selected negatives：{len(neg):,}
- combined dataset：{len(pos) + len(neg):,}
- positive scaffold cap：每个 sampled Murcko scaffold 最多 {MAX_POSITIVES_PER_SCAFFOLD} 条

## 三、正样本分布

正样本从 strict ChEMBL property pool 中按性质分层、芳香层分层抽样，并对 sampled
Murcko scaffold 做限额。

芳香类别：

{aromatic_lines}

![Selected positive distribution](selected_positive_distribution.png)

## 四、负样本覆盖

负样本来自 generated hard negatives，不包含全部 OUT 分子。

{count_lines}

![Selected negative category coverage](selected_negative_category_coverage.png)

## 五、下一步

这版可以作为分类器 v0.1 的候选训练集。真正替换当前训练流程前，还需要让
`chembl_classifier/data/build_dataset.py` 支持从这些显式样本清单读取正负样本，
并在训练后输出 per-category recall、high_reward_bad false-negative rate 和
scaffold split 结果。
"""
    (ANALYSIS_DIR / "selected_dataset_report.md").write_text(report)


def main() -> None:
    print("Selecting positives...")
    pos = select_positives()
    print("Selecting negatives...")
    neg = select_negatives()
    neg_counts = explode_label_counts(neg)

    pos.to_csv(ANALYSIS_DIR / "selected_positives_100k.csv.gz", index=False, compression="gzip")
    neg.to_csv(ANALYSIS_DIR / "selected_negatives_70k.csv.gz", index=False, compression="gzip")
    neg_counts.to_csv(ANALYSIS_DIR / "selected_negative_category_counts.csv", index=False)
    write_smi_files(pos, neg)
    plot_outputs(pos, neg_counts)
    write_report(pos, neg, neg_counts)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "target_positives": TARGET_POSITIVES,
        "target_negatives": TARGET_NEGATIVES,
        "selected_positives": int(len(pos)),
        "selected_negatives": int(len(neg)),
        "positive_aromatic_class_counts": pos["aromatic_class"].value_counts().to_dict(),
        "negative_label_counts": dict(zip(neg_counts["label"], neg_counts["selected_count"])),
        "max_positives_per_scaffold": MAX_POSITIVES_PER_SCAFFOLD,
    }
    (ANALYSIS_DIR / "selected_dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    manifest = {
        "created_at": summary["created_at"],
        "root": str(ROOT),
        "analysis_dir": str(ANALYSIS_DIR),
        "outputs": [
            "analysis_manifest.json",
            "build_selected_sample_lists.py",
            "selected_classifier_dataset_170k.tsv",
            "selected_dataset_report.md",
            "selected_dataset_summary.json",
            "selected_negative_category_counts.csv",
            "selected_negative_category_coverage.png",
            "selected_negatives_70k.csv.gz",
            "selected_negatives_70k_labeled.smi",
            "selected_positive_distribution.png",
            "selected_positives_100k.csv.gz",
            "selected_positives_100k.smi",
        ],
    }
    (ANALYSIS_DIR / "analysis_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("Done.")


if __name__ == "__main__":
    main()
