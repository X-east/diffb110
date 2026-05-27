#!/usr/bin/env python3
"""Dry-run positive/negative candidate-pool selection for the ChEMBL classifier."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "18")
os.environ.setdefault("MKL_NUM_THREADS", "18")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "18")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "18")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-diffb110")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = Path(__file__).resolve().parent
REWARD_ANALYSIS_DIR = ROOT / "analysis/reward_hacking_diagnosis_20260527"
HETERO_ANALYSIS_DIR = ROOT / "analysis/hetero_composition_diagnosis_20260527"
AROMATIC_ANALYSIS_DIR = ROOT / "analysis/aromatic_benzene_diagnosis_20260527"


def strict_property_pool_mask(props: pd.DataFrame) -> pd.Series:
    """Return current strict ChEMBL property pool mask."""
    property_core = (
        props["heavy_atoms"].between(10, 30)
        & props["mol_wt"].between(150, 500)
        & props["mol_log_p"].between(-1, 6)
        & props["tpsa"].between(20, 140)
        & props["num_h_acceptors"].between(1, 10)
        & props["num_h_donors"].between(0, 5)
        & props["num_rotatable_bonds"].le(10)
        & props["num_heteroatoms"].ge(1)
        & props["carbon_fraction"].between(0.35, 0.90)
    )
    return (
        property_core
        & props["lipinski_violations"].eq(0)
        & props["carbon_fraction"].le(0.88)
        & props["num_heteroatoms"].ge(2)
        & props["num_aromatic_rings"].le(4)
        & props["num_rings"].le(6)
        & props["mol_log_p"].between(0, 5.5)
        & props["tpsa"].between(20, 120)
    )


def bin_series(series: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(series, bins=bins, labels=labels, include_lowest=True).astype(str)


def build_positive_pool() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build current-version strict ChEMBL positive candidate pool."""
    cols = [
        "chembl_id",
        "canonical_smiles",
        "heavy_atoms",
        "num_rings",
        "mol_wt",
        "mol_log_p",
        "tpsa",
        "num_h_acceptors",
        "num_h_donors",
        "num_rotatable_bonds",
        "num_aromatic_rings",
        "num_aromatic_heterocycles",
        "num_saturated_heterocycles",
        "num_heteroatoms",
        "num_carbon_atoms",
        "carbon_fraction",
        "lipinski_violations",
    ]
    props = pd.read_csv(
        ROOT / "data/chembl/chembl_36_properties.csv.gz",
        usecols=cols,
        compression="gzip",
    )
    mask = strict_property_pool_mask(props)
    pool = props.loc[mask].copy()

    pool["logp_bin"] = bin_series(
        pool["mol_log_p"], [0, 1, 2, 3, 4, 5.5], ["0-1", "1-2", "2-3", "3-4", "4-5.5"]
    )
    pool["tpsa_bin"] = bin_series(
        pool["tpsa"], [20, 40, 60, 80, 100, 120], ["20-40", "40-60", "60-80", "80-100", "100-120"]
    )
    pool["carbon_bin"] = bin_series(
        pool["carbon_fraction"],
        [0.35, 0.60, 0.70, 0.80, 0.88],
        ["0.35-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.88"],
    )
    pool["hetero_bin"] = bin_series(
        pool["num_heteroatoms"], [2, 4, 6, 8, 12, 40], ["2-4", "4-6", "6-8", "8-12", "12+"]
    )
    pool["ring_bin"] = bin_series(
        pool["num_rings"], [0, 1, 2, 3, 4, 6], ["0-1", "1-2", "2-3", "3-4", "4-6"]
    )
    pool["heterocycle_class"] = np.select(
        [
            pool["num_aromatic_heterocycles"].gt(0) & pool["num_saturated_heterocycles"].gt(0),
            pool["num_aromatic_heterocycles"].gt(0),
            pool["num_saturated_heterocycles"].gt(0),
        ],
        ["mixed_heterocycle", "aromatic_heterocycle", "saturated_heterocycle"],
        default="no_heterocycle",
    )
    pool["positive_stratum"] = (
        pool["logp_bin"].astype(str)
        + "|"
        + pool["tpsa_bin"].astype(str)
        + "|"
        + pool["carbon_bin"].astype(str)
        + "|"
        + pool["hetero_bin"].astype(str)
        + "|"
        + pool["heterocycle_class"].astype(str)
    )

    summary = {
        "chembl_rows": int(len(props)),
        "strict_positive_candidates": int(len(pool)),
        "strict_positive_fraction": float(len(pool) / len(props)),
        "strata": int(pool["positive_stratum"].nunique()),
        "quantiles": json.loads(
            pool[
                [
                    "heavy_atoms",
                    "mol_wt",
                    "mol_log_p",
                    "tpsa",
                    "num_heteroatoms",
                    "carbon_fraction",
                    "num_rings",
                    "num_aromatic_rings",
                ]
            ]
            .quantile([0.01, 0.05, 0.5, 0.95, 0.99])
            .to_json()
        ),
    }
    return pool, summary


def canonical_smiles(smiles: object) -> str | None:
    if not isinstance(smiles, str) or not smiles or smiles == "None":
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def reward_col(df: pd.DataFrame) -> str | None:
    for col in ["reward_total_raw", "reward_total", "reward_total_norm"]:
        if col in df.columns:
            return col
    return None


def add_top_reward_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Flag top reward decile within each run when a reward column exists."""
    df = df.copy()
    df["top_reward_decile"] = 0
    for run, idx in df.groupby("run").groups.items():
        group = df.loc[idx]
        rcol = reward_col(group)
        if not rcol:
            continue
        valid = group[group["valid"].astype(bool) & group[rcol].notna()]
        if len(valid) < 50:
            continue
        cutoff = valid[rcol].quantile(0.90)
        df.loc[valid.index[valid[rcol] >= cutoff], "top_reward_decile"] = 1
    return df


def load_generated_flags() -> pd.DataFrame:
    """Load and combine reward-hacking and hetero-composition generated flags."""
    base = pd.read_csv(REWARD_ANALYSIS_DIR / "molecule_flags_compact.csv.gz", compression="gzip")
    comp = pd.read_csv(HETERO_ANALYSIS_DIR / "generated_composition_flags.csv.gz", compression="gzip")
    if len(base) != len(comp):
        raise ValueError("Generated flag files have different row counts.")
    for col in comp.columns:
        if col not in base.columns:
            base[col] = comp[col].values
    aromatic_path = AROMATIC_ANALYSIS_DIR / "generated_aromatic_flags.csv.gz"
    if aromatic_path.exists():
        aromatic = pd.read_csv(aromatic_path, compression="gzip")
        if len(base) != len(aromatic):
            raise ValueError("Generated aromatic flag file has a different row count.")
        for col in aromatic.columns:
            if col not in base.columns:
                base[col] = aromatic[col].values
    return add_top_reward_flags(base)


def assign_negative_labels(row: pd.Series) -> list[str]:
    """Assign hard-negative categories. Ambiguous generated molecules stay unlabeled."""
    if not bool(row.get("valid", False)):
        return []

    labels: list[str] = []
    high_reward = int(row.get("top_reward_decile", 0)) == 1
    strict_fail = int(row.get("strict_core_pass", 0)) == 0
    high_carbon = int(row.get("high_carbon_90", 0)) == 1
    very_high_carbon = int(row.get("high_carbon_95", 0)) == 1
    low_hetero = int(row.get("low_hetero", 0)) == 1
    no_hetero = int(row.get("no_hetero", 0)) == 1
    co_only = int(row.get("co_only_mostly_c_o1_3", 0)) == 1
    co_only_hard = int(row.get("co_only_high_c_low_o", 0)) == 1
    no_non_o = int(row.get("no_non_o_hetero", 0)) == 1
    tri = int(row.get("has_3ring", 0)) == 1
    ring_complex = int(row.get("ring_complex", 0)) == 1
    hydrophobic_low_polar = int(row.get("logp_high", 0)) == 1 or int(row.get("tpsa_low", 0)) == 1
    aromatic_deficient = int(row.get("has_aromatic_ring_precise", 1)) == 0

    if very_high_carbon or (high_carbon and low_hetero) or no_hetero:
        labels.append("high_carbon_low_hetero")
    if co_only_hard or (co_only and (high_reward or high_carbon or strict_fail)):
        labels.append("co_only_mostly_carbon")
    if no_non_o and high_carbon:
        labels.append("non_oxygen_hetero_loss")
    if tri and (high_reward or strict_fail):
        labels.append("three_member_ring")
    if ring_complex and (high_reward or strict_fail):
        labels.append("ring_complex")
    if hydrophobic_low_polar and (high_carbon or co_only or low_hetero):
        labels.append("hydrophobic_low_polarity")
    if aromatic_deficient and (high_reward or tri or ring_complex) and (high_carbon or co_only or low_hetero):
        labels.append("aromatic_deficient_bad_topology")

    if high_reward and strict_fail and labels:
        labels.append("high_reward_bad")

    return sorted(set(labels))


def build_negative_candidates() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build deduplicated generated hard-negative candidates."""
    gen = load_generated_flags()
    gen["negative_labels_list"] = gen.apply(assign_negative_labels, axis=1)
    candidates = gen[gen["negative_labels_list"].map(bool)].copy()
    candidates["canonical_smiles"] = [canonical_smiles(s) for s in candidates["smiles"]]
    candidates = candidates[candidates["canonical_smiles"].notna()].copy()

    grouped: dict[str, dict[str, Any]] = {}
    for _, row in candidates.iterrows():
        smi = row["canonical_smiles"]
        rec = grouped.setdefault(
            smi,
            {
                "canonical_smiles": smi,
                "example_smiles": row["smiles"],
                "labels": set(),
                "runs": set(),
                "epochs": set(),
                "max_reward_total_raw": np.nan,
                "max_chembl_classifier_reward": np.nan,
                "strict_core_pass_any": 0,
                "top_reward_decile_any": 0,
                "carbon_fraction_max": np.nan,
                "oxygen_count_max": np.nan,
                "nitrogen_count_max": np.nan,
                "halogen_count_max": np.nan,
            },
        )
        rec["labels"].update(row["negative_labels_list"])
        rec["runs"].add(str(row.get("run")))
        rec["epochs"].add(str(row.get("epoch")))
        rec["strict_core_pass_any"] = max(rec["strict_core_pass_any"], int(row.get("strict_core_pass", 0)))
        rec["top_reward_decile_any"] = max(rec["top_reward_decile_any"], int(row.get("top_reward_decile", 0)))
        for src, dst in [
            ("reward_total_raw", "max_reward_total_raw"),
            ("chembl_classifier_reward", "max_chembl_classifier_reward"),
            ("carbon_fraction_calc", "carbon_fraction_max"),
            ("oxygen_count", "oxygen_count_max"),
            ("nitrogen_count", "nitrogen_count_max"),
            ("halogen_count", "halogen_count_max"),
        ]:
            value = row.get(src, np.nan)
            if pd.notna(value):
                current = rec[dst]
                rec[dst] = float(value) if pd.isna(current) else max(float(current), float(value))

    records = []
    label_counter: Counter[str] = Counter()
    for rec in grouped.values():
        labels = sorted(rec["labels"])
        for label in labels:
            label_counter[label] += 1
        records.append(
            {
                "canonical_smiles": rec["canonical_smiles"],
                "example_smiles": rec["example_smiles"],
                "labels": "|".join(labels),
                "primary_label": labels[0] if labels else "",
                "runs": "|".join(sorted(rec["runs"])),
                "epochs": "|".join(sorted(rec["epochs"], key=lambda x: int(float(x)) if x not in {"nan", "None"} else -1)),
                "max_reward_total_raw": rec["max_reward_total_raw"],
                "max_chembl_classifier_reward": rec["max_chembl_classifier_reward"],
                "strict_core_pass_any": rec["strict_core_pass_any"],
                "top_reward_decile_any": rec["top_reward_decile_any"],
                "carbon_fraction_max": rec["carbon_fraction_max"],
                "oxygen_count_max": rec["oxygen_count_max"],
                "nitrogen_count_max": rec["nitrogen_count_max"],
                "halogen_count_max": rec["halogen_count_max"],
            }
        )
    negatives = pd.DataFrame(records).sort_values(
        ["top_reward_decile_any", "max_reward_total_raw"], ascending=[False, False]
    )
    label_counts = pd.DataFrame(
        [{"label": label, "candidate_count": count} for label, count in label_counter.most_common()]
    )
    return negatives, label_counts


def plot_outputs(pos: pd.DataFrame, neg_counts: pd.DataFrame) -> None:
    """Create simple dry-run plots."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    pos["carbon_bin"].value_counts().sort_index().plot(kind="bar", ax=axes[0], color="#2b6cb0")
    axes[0].set_title("Strict ChEMBL positive pool: carbon fraction bins")
    axes[0].set_ylabel("Molecules")
    axes[0].tick_params(axis="x", rotation=30)

    pos["heterocycle_class"].value_counts().plot(kind="bar", ax=axes[1], color="#2f855a")
    axes[1].set_title("Strict ChEMBL positive pool: heterocycle classes")
    axes[1].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "positive_pool_strata.png", dpi=220)
    plt.close()

    if not neg_counts.empty:
        top = neg_counts.sort_values("candidate_count", ascending=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top["label"], top["candidate_count"], color="#c2410c")
        ax.set_xlabel("Deduplicated candidate molecules")
        ax.set_title("Generated hard-negative candidate categories")
        ax.grid(axis="x", alpha=0.25)
        plt.tight_layout()
        plt.savefig(ANALYSIS_DIR / "negative_candidate_categories.png", dpi=220)
        plt.close()


def write_report(pos_summary: dict[str, Any], neg: pd.DataFrame, neg_counts: pd.DataFrame) -> None:
    """Write dry-run selection report."""
    count_lines = "\n".join(
        f"- `{row.label}`: {int(row.candidate_count):,}"
        for row in neg_counts.itertuples(index=False)
    )
    report = f"""# 正负样本候选池 Dry-run

生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 一、原则

本 dry-run 不覆盖现有 `chembl_classifier/data`，只在 analysis 目录中生成候选池和统计。

正样本来自当前版本 cleaned ChEMBL36 的 strict property pool。负样本只来自明确命中
reward loophole 的 OUT generated hard negatives；OUT 生成分子默认不是负样本。

## 二、正样本候选池

- ChEMBL 总量：{pos_summary['chembl_rows']:,}
- strict positive candidates：{pos_summary['strict_positive_candidates']:,}
- strict positive fraction：{pos_summary['strict_positive_fraction']:.2%}
- positive strata：{pos_summary['strata']:,}

![Positive pool strata](positive_pool_strata.png)

## 三、负样本候选池

- deduplicated generated hard negatives：{len(neg):,}

类别计数：

{count_lines}

![Negative candidate categories](negative_candidate_categories.png)

## 四、建议

下一步不要直接训练，而是先确定目标训练集规模和采样比例。

建议第一版：

- 正样本：从 strict positive pool 中分层抽样 100k。
- 负样本：从 generated hard negatives 中按类别均衡抽样；每个 confirmed loophole 至少几千条。
- 保留 generated-good/gray 分子作为未标注池和外部检查集，不进入负样本。
- 训练后必须报告 per-category recall、high_reward_bad false-negative rate、scaffold split AUC。
"""
    (ANALYSIS_DIR / "selection_dryrun_report.md").write_text(report)


def main() -> None:
    print("Building strict ChEMBL positive candidate pool...")
    pos, pos_summary = build_positive_pool()
    pos.to_csv(ANALYSIS_DIR / "positive_candidate_pool.csv.gz", index=False, compression="gzip")
    (ANALYSIS_DIR / "positive_candidate_summary.json").write_text(
        json.dumps(pos_summary, ensure_ascii=False, indent=2)
    )
    pos["positive_stratum"].value_counts().rename_axis("positive_stratum").reset_index(name="count").to_csv(
        ANALYSIS_DIR / "positive_stratum_counts.csv", index=False
    )

    print("Building generated hard-negative candidate pool...")
    neg, neg_counts = build_negative_candidates()
    neg.to_csv(ANALYSIS_DIR / "negative_candidates_labeled.csv.gz", index=False, compression="gzip")
    neg_counts.to_csv(ANALYSIS_DIR / "negative_category_counts.csv", index=False)

    print("Writing plots and report...")
    plot_outputs(pos, neg_counts)
    write_report(pos_summary, neg, neg_counts)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(ROOT),
        "analysis_dir": str(ANALYSIS_DIR),
        "outputs": [
            "analysis_manifest.json",
            "negative_candidate_categories.png",
            "negative_candidates_labeled.csv.gz",
            "negative_category_counts.csv",
            "positive_candidate_pool.csv.gz",
            "positive_candidate_summary.json",
            "positive_pool_strata.png",
            "positive_stratum_counts.csv",
            "select_candidate_pools.py",
            "selection_dryrun_report.md",
        ],
    }
    (ANALYSIS_DIR / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    print("Done.")


if __name__ == "__main__":
    main()
