#!/usr/bin/env python3
"""Diagnose aromatic and benzene-like ring distribution in generated molecules."""

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


RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = Path(__file__).resolve().parent
REWARD_ANALYSIS_DIR = ROOT / "analysis/reward_hacking_diagnosis_20260527"
N_WORKERS = 18

SELECTED_RUNS = [
    "7gpu-classifier-only_20260526_210715",
    "7gpu-dock-ecfp4-fcfp4_20260525_212140",
    "7gpu-dock-ecfp4-fcfp4_20260526_115341",
    "7gpu-dock-ecfp4-fcfp4_20260521_212927",
    "6gpu-chembl-only_20260521_154957",
    "8gpu-chembl-only_20260520_101340",
    "qed-sa-docking-8gjl-8gpu_20260508_212050",
]


def strict_property_pool_mask(props: pd.DataFrame) -> pd.Series:
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


def generated_aromatic_features(smiles: object) -> dict[str, Any]:
    if (
        not isinstance(smiles, str)
        or not smiles
        or smiles == "None"
        or smiles.startswith("NonTensorData")
    ):
        return {"aromatic_parse_ok": 0}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"aromatic_parse_ok": 0}

    aromatic_rings = 0
    aromatic_carbocycles = 0
    benzene_like_rings = 0
    aromatic_heterocycles = 0

    for ring in mol.GetRingInfo().AtomRings():
        atoms = [mol.GetAtomWithIdx(idx) for idx in ring]
        if not all(atom.GetIsAromatic() for atom in atoms):
            continue
        aromatic_rings += 1
        all_carbon = all(atom.GetAtomicNum() == 6 for atom in atoms)
        if all_carbon:
            aromatic_carbocycles += 1
            if len(ring) == 6:
                benzene_like_rings += 1
        else:
            aromatic_heterocycles += 1

    return {
        "aromatic_parse_ok": 1,
        "aromatic_ring_count_precise": aromatic_rings,
        "aromatic_carbocycle_count": aromatic_carbocycles,
        "benzene_like_ring_count": benzene_like_rings,
        "aromatic_heterocycle_count_precise": aromatic_heterocycles,
        "has_aromatic_ring_precise": int(aromatic_rings > 0),
        "has_aromatic_carbocycle": int(aromatic_carbocycles > 0),
        "has_benzene_like_ring": int(benzene_like_rings > 0),
        "has_aromatic_heterocycle_precise": int(aromatic_heterocycles > 0),
        "aromatic_deficient": int(aromatic_rings == 0),
    }


def add_generated_aromatic_features(df: pd.DataFrame) -> pd.DataFrame:
    unique = pd.Series(df["smiles"].drop_duplicates().tolist())
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        feats = list(executor.map(generated_aromatic_features, unique.tolist(), chunksize=1000))
    fmap = dict(zip(unique.tolist(), feats))
    feat_df = pd.DataFrame([fmap.get(s, {"aromatic_parse_ok": 0}) for s in df["smiles"]])
    return pd.concat([df.reset_index(drop=True), feat_df], axis=1)


def summarize(df: pd.DataFrame, label: str, generated: bool = False) -> dict[str, Any]:
    if generated:
        data = df[df["valid"].astype(bool) & df["aromatic_parse_ok"].eq(1)].copy()
        cols = {
            "aromatic_ring_count": "aromatic_ring_count_precise",
            "aromatic_carbocycle_count": "aromatic_carbocycle_count",
            "benzene_like_ring_count": "benzene_like_ring_count",
            "aromatic_heterocycle_count": "aromatic_heterocycle_count_precise",
            "has_aromatic_ring": "has_aromatic_ring_precise",
            "has_aromatic_carbocycle": "has_aromatic_carbocycle",
            "has_benzene_like_ring": "has_benzene_like_ring",
            "has_aromatic_heterocycle": "has_aromatic_heterocycle_precise",
        }
    else:
        data = df.copy()
        data["aromatic_carbocycle_approx"] = (
            data["num_aromatic_rings"] - data["num_aromatic_heterocycles"]
        ).clip(lower=0)
        cols = {
            "aromatic_ring_count": "num_aromatic_rings",
            "aromatic_carbocycle_count": "aromatic_carbocycle_approx",
            "benzene_like_ring_count": "aromatic_carbocycle_approx",
            "aromatic_heterocycle_count": "num_aromatic_heterocycles",
            "has_aromatic_ring": "num_aromatic_rings",
            "has_aromatic_carbocycle": "aromatic_carbocycle_approx",
            "has_benzene_like_ring": "aromatic_carbocycle_approx",
            "has_aromatic_heterocycle": "num_aromatic_heterocycles",
        }

    out: dict[str, Any] = {"label": label, "n": int(len(data))}
    if data.empty:
        return out
    for public, col in cols.items():
        if public.startswith("has_"):
            out[f"{public}_rate"] = float((data[col] > 0).mean())
        else:
            out[f"{public}_mean"] = float(data[col].mean())
            out[f"{public}_p50"] = float(data[col].quantile(0.50))
            out[f"{public}_p95"] = float(data[col].quantile(0.95))
    return out


def reward_col(df: pd.DataFrame) -> str | None:
    for col in ["reward_total_raw", "reward_total", "reward_total_norm"]:
        if col in df.columns:
            return col
    return None


def top_reward_summaries(gen: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run, group in gen.groupby("run", sort=True):
        rcol = reward_col(group)
        if not rcol:
            continue
        valid = group[group["valid"].astype(bool) & group["aromatic_parse_ok"].eq(1) & group[rcol].notna()]
        if len(valid) < 50:
            continue
        top = valid[valid[rcol] >= valid[rcol].quantile(0.90)]
        row = summarize(top, f"{run}:top_reward_decile", generated=True)
        row["run"] = run
        row["top_n"] = int(len(top))
        row["reward_col"] = rcol
        rows.append(row)
    return pd.DataFrame(rows)


def selected_epoch_summaries(gen: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run in SELECTED_RUNS:
        subset = gen[gen["run"].eq(run)]
        if subset.empty:
            continue
        for epoch, group in subset.groupby("epoch", sort=True):
            row = summarize(group, f"{run}:epoch_{epoch}", generated=True)
            row["run"] = run
            row["epoch"] = int(epoch)
            rows.append(row)
    return pd.DataFrame(rows)


def plot_outputs(global_df: pd.DataFrame, top_df: pd.DataFrame) -> None:
    labels = {
        "chembl36_cleaned": "ChEMBL cleaned",
        "chembl36_strict_property_pool": "ChEMBL strict",
        "out_valid_generated": "OUT generated",
    }
    metrics = [
        ("has_aromatic_ring_rate", "Has aromatic ring"),
        ("has_benzene_like_ring_rate", "Has benzene-like ring"),
        ("has_aromatic_heterocycle_rate", "Has aromatic heterocycle"),
    ]
    data = global_df.set_index("label")[[m[0] for m in metrics]].rename(index=labels)
    data.columns = [m[1] for m in metrics]
    ax = (data * 100).T.plot(kind="bar", figsize=(10, 5), width=0.75)
    ax.set_ylabel("Molecules (%)")
    ax.set_title("Aromatic and benzene-like ring distribution")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "global_aromatic_benzene_comparison.png", dpi=220)
    plt.close()

    if not top_df.empty:
        view = top_df[top_df["top_n"].ge(100)].sort_values("has_benzene_like_ring_rate").head(15)
        fig, ax = plt.subplots(figsize=(11, 7))
        y = range(len(view))
        ax.barh(y, view["has_benzene_like_ring_rate"] * 100, color="#2b6cb0", label="Benzene-like")
        ax.scatter(view["has_aromatic_ring_rate"] * 100, y, color="#c2410c", label="Any aromatic", zorder=3)
        ax.set_yticks(list(y))
        ax.set_yticklabels(view["run"])
        ax.invert_yaxis()
        ax.set_xlabel("Top reward decile molecules (%)")
        ax.set_title("Runs with low benzene-like ring rate in top reward region")
        ax.grid(axis="x", alpha=0.25)
        ax.legend(frameon=False, loc="lower right")
        plt.tight_layout()
        plt.savefig(ANALYSIS_DIR / "top_reward_low_benzene_runs.png", dpi=220)
        plt.close()


def write_report(global_rows: list[dict[str, Any]], top_df: pd.DataFrame) -> None:
    g = {row["label"]: row for row in global_rows}

    def pct(label: str, key: str) -> str:
        value = g[label].get(key, np.nan)
        return "NA" if pd.isna(value) else f"{100 * float(value):.2f}%"

    top_lines = []
    if not top_df.empty:
        view = top_df[top_df["top_n"].ge(100)].sort_values("has_benzene_like_ring_rate").head(8)
        for _, row in view.iterrows():
            top_lines.append(
                "- {run}: top reward benzene-like={benz:.2%}, any aromatic={aro:.2%}, aromatic heterocycle={het:.2%}".format(
                    run=row["run"],
                    benz=row.get("has_benzene_like_ring_rate", np.nan),
                    aro=row.get("has_aromatic_ring_rate", np.nan),
                    het=row.get("has_aromatic_heterocycle_rate", np.nan),
                )
            )
    top_text = "\n".join(top_lines) if top_lines else "- No top reward summaries."

    report = f"""# 苯环与芳香环分布专项诊断

生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 一、问题

本分析检查生成分子是否缺少苯环/芳香碳环。这里区分：

- `has_aromatic_ring`：任意芳香环。
- `has_benzene_like_ring`：六元芳香碳环，近似苯环/苯并环。
- `has_aromatic_heterocycle`：芳香杂环。

ChEMBL 的 benzene-like 指标使用 `num_aromatic_rings - num_aromatic_heterocycles`
近似；OUT 生成分子用 RDKit ring atoms 精确判断六元芳香碳环。

## 二、全局对比

| 指标 | ChEMBL36 cleaned | strict ChEMBL property pool | OUT valid generated |
|---|---:|---:|---:|
| has aromatic ring | {pct('chembl36_cleaned', 'has_aromatic_ring_rate')} | {pct('chembl36_strict_property_pool', 'has_aromatic_ring_rate')} | {pct('out_valid_generated', 'has_aromatic_ring_rate')} |
| has benzene-like ring | {pct('chembl36_cleaned', 'has_benzene_like_ring_rate')} | {pct('chembl36_strict_property_pool', 'has_benzene_like_ring_rate')} | {pct('out_valid_generated', 'has_benzene_like_ring_rate')} |
| has aromatic heterocycle | {pct('chembl36_cleaned', 'has_aromatic_heterocycle_rate')} | {pct('chembl36_strict_property_pool', 'has_aromatic_heterocycle_rate')} | {pct('out_valid_generated', 'has_aromatic_heterocycle_rate')} |

![Global aromatic benzene comparison](global_aromatic_benzene_comparison.png)

## 三、Top reward 区域中苯环更少的 run

{top_text}

![Top reward low benzene runs](top_reward_low_benzene_runs.png)

## 四、解释

如果 OUT 的 benzene-like rate 明显低于 strict ChEMBL pool，说明模型偏向脂肪环、
复杂饱和环、三元环或非芳香稠环，而不是 ChEMBL 常见的芳香碳环/芳香杂环分布。

这不应该作为“无苯环就是负样本”的硬规则，因为很多药物可以无苯环。更合理的做法是：

1. 正样本抽样加入 aromatic/benzene/heteroaromatic strata。
2. RL 监控每轮生成分子的 aromatic-ring distribution。
3. 对“无芳香环 + 高碳 + 复杂饱和环/三元环”的组合模式作为 hard negative 候选。
"""
    (ANALYSIS_DIR / "aromatic_benzene_report.md").write_text(report)


def main() -> None:
    print("Loading ChEMBL properties...")
    chembl_cols = [
        "chembl_id",
        "canonical_smiles",
        "heavy_atoms",
        "mol_wt",
        "mol_log_p",
        "tpsa",
        "num_h_acceptors",
        "num_h_donors",
        "num_rotatable_bonds",
        "num_heteroatoms",
        "carbon_fraction",
        "num_rings",
        "num_aromatic_rings",
        "num_aromatic_heterocycles",
        "lipinski_violations",
    ]
    chembl = pd.read_csv(ROOT / "data/chembl/chembl_36_properties.csv.gz", usecols=chembl_cols, compression="gzip")
    strict = chembl.loc[strict_property_pool_mask(chembl)].copy()

    print("Loading and parsing generated molecules...")
    gen = pd.read_csv(REWARD_ANALYSIS_DIR / "molecule_flags_compact.csv.gz", compression="gzip")
    gen = add_generated_aromatic_features(gen)

    global_rows = [
        summarize(chembl, "chembl36_cleaned", generated=False),
        summarize(strict, "chembl36_strict_property_pool", generated=False),
        summarize(gen, "out_valid_generated", generated=True),
    ]
    global_df = pd.DataFrame(global_rows)
    global_df.to_csv(ANALYSIS_DIR / "global_aromatic_benzene_summary.csv", index=False)
    (ANALYSIS_DIR / "global_aromatic_benzene_summary.json").write_text(
        json.dumps({row["label"]: row for row in global_rows}, ensure_ascii=False, indent=2)
    )

    selected_epoch_summaries(gen).to_csv(ANALYSIS_DIR / "selected_epoch_aromatic_summary.csv", index=False)
    top_df = top_reward_summaries(gen)
    top_df.to_csv(ANALYSIS_DIR / "top_reward_aromatic_summary.csv", index=False)

    compact_cols = [
        "run",
        "epoch",
        "smiles",
        "valid",
        "strict_core_pass",
        "reward_total_raw",
        "aromatic_ring_count_precise",
        "aromatic_carbocycle_count",
        "benzene_like_ring_count",
        "aromatic_heterocycle_count_precise",
        "has_aromatic_ring_precise",
        "has_benzene_like_ring",
        "has_aromatic_heterocycle_precise",
        "has_3ring",
        "ring_complex",
        "carbon_fraction_calc",
        "hetero",
        "logp_calc",
        "tpsa_calc",
    ]
    gen[[col for col in compact_cols if col in gen.columns]].to_csv(
        ANALYSIS_DIR / "generated_aromatic_flags.csv.gz", index=False, compression="gzip"
    )

    plot_outputs(global_df, top_df)
    write_report(global_rows, top_df)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(ROOT),
        "analysis_dir": str(ANALYSIS_DIR),
        "workers": N_WORKERS,
        "outputs": [
            "analysis_manifest.json",
            "analyze_aromatic_benzene.py",
            "aromatic_benzene_report.md",
            "generated_aromatic_flags.csv.gz",
            "global_aromatic_benzene_comparison.png",
            "global_aromatic_benzene_summary.csv",
            "global_aromatic_benzene_summary.json",
            "selected_epoch_aromatic_summary.csv",
            "top_reward_aromatic_summary.csv",
            "top_reward_low_benzene_runs.png",
        ],
    }
    (ANALYSIS_DIR / "analysis_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("Done.")


if __name__ == "__main__":
    main()
