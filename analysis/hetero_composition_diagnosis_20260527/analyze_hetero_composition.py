#!/usr/bin/env python3
"""Diagnose C/O-only and non-oxygen heteroatom collapse.

This analysis checks a specific reward-hacking risk: molecules becoming mostly
carbon with only a small number of oxygen atoms, instead of preserving the
heteroatom diversity seen in ChEMBL-like molecules.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = Path(__file__).resolve().parent
PREV_ANALYSIS_DIR = ROOT / "analysis/reward_hacking_diagnosis_20260527"

SELECTED_RUNS = [
    "7gpu-classifier-only_20260526_210715",
    "7gpu-dock-ecfp4-fcfp4_20260525_212140",
    "7gpu-dock-ecfp4-fcfp4_20260526_115341",
    "7gpu-dock-ecfp4-fcfp4_20260521_212927",
    "6gpu-chembl-only_20260521_154957",
    "8gpu-chembl-only_20260520_101340",
    "qed-sa-docking-8gjl-8gpu_20260508_212050",
]

HALOGENS = {"F", "Cl", "Br", "I"}
TRACKED_ELEMENTS = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Si"]


def element_features(smiles: object) -> dict[str, Any]:
    """Return heavy-atom element counts and C/O-only collapse flags."""
    if (
        not isinstance(smiles, str)
        or not smiles
        or smiles == "None"
        or smiles.startswith("NonTensorData")
    ):
        return {"element_parse_ok": 0}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"element_parse_ok": 0}

    counts: Counter[str] = Counter()
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() > 1:
            counts[atom.GetSymbol()] += 1

    heavy = sum(counts.values())
    c = counts.get("C", 0)
    o = counts.get("O", 0)
    n = counts.get("N", 0)
    s = counts.get("S", 0)
    p = counts.get("P", 0)
    halogen_count = sum(counts.get(x, 0) for x in HALOGENS)
    hetero = heavy - c
    non_o_hetero = hetero - o
    carbon_fraction = c / heavy if heavy else 0.0
    oxygen_fraction = o / heavy if heavy else 0.0
    hetero_symbols = sorted(sym for sym, value in counts.items() if sym != "C" and value > 0)
    non_o_symbols = sorted(sym for sym in hetero_symbols if sym != "O")
    element_set = ",".join(sorted(counts)) if counts else "none"
    hetero_signature = "+".join(hetero_symbols) if hetero_symbols else "none"

    row: dict[str, Any] = {
        "element_parse_ok": 1,
        "element_set": element_set,
        "hetero_signature": hetero_signature,
        "heavy_elem": heavy,
        "carbon_count": c,
        "oxygen_count": o,
        "nitrogen_count": n,
        "sulfur_count": s,
        "phosphorus_count": p,
        "halogen_count": halogen_count,
        "hetero_count_elem": hetero,
        "non_o_hetero_count": non_o_hetero,
        "hetero_type_count": len(hetero_symbols),
        "non_o_hetero_type_count": len(non_o_symbols),
        "carbon_fraction_elem": carbon_fraction,
        "oxygen_fraction_elem": oxygen_fraction,
        "oxygen_to_hetero_fraction": o / hetero if hetero else np.nan,
        "has_nitrogen": int(n > 0),
        "has_oxygen": int(o > 0),
        "has_sulfur": int(s > 0),
        "has_phosphorus": int(p > 0),
        "has_halogen": int(halogen_count > 0),
        "has_non_o_hetero": int(non_o_hetero > 0),
        "no_non_o_hetero": int(non_o_hetero == 0),
        "oxygen_only_hetero": int(o > 0 and non_o_hetero == 0),
        "co_only": int(c > 0 and o > 0 and non_o_hetero == 0),
        "only_carbon": int(c > 0 and hetero == 0),
        "co_only_high_c": int(c > 0 and o > 0 and non_o_hetero == 0 and carbon_fraction >= 0.85),
        "co_only_high_c_low_o": int(
            c > 0 and 1 <= o <= 2 and non_o_hetero == 0 and carbon_fraction >= 0.85
        ),
        "co_only_mostly_c_o1_3": int(
            c > 0 and 1 <= o <= 3 and non_o_hetero == 0 and carbon_fraction >= 0.80
        ),
        "carbon_rich_no_non_o_hetero": int(carbon_fraction >= 0.85 and non_o_hetero == 0),
        "carbon_rich_no_nitrogen": int(carbon_fraction >= 0.85 and n == 0),
    }
    for element in TRACKED_ELEMENTS:
        row[f"{element.lower()}_count"] = counts.get(element, 0)
    return row


def add_element_features(df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
    """Append element features with a cache for repeated generated SMILES."""
    cache: dict[object, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for idx, smiles in enumerate(df[smiles_col].tolist(), start=1):
        if smiles not in cache:
            cache[smiles] = element_features(smiles)
        rows.append(cache[smiles])
        if idx % 200000 == 0:
            print(f"  parsed {idx:,} molecules")
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def property_pool_mask(props: pd.DataFrame) -> pd.Series:
    """Same property-only strict pool definition as the prior diagnosis."""
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


def summarize(df: pd.DataFrame, label: str) -> dict[str, Any]:
    """Summarize C/O-only and hetero-diversity flags."""
    if "valid" in df.columns:
        data = df[df["valid"].astype(bool) & df["element_parse_ok"].eq(1)].copy()
    else:
        data = df[df["element_parse_ok"].eq(1)].copy()

    out: dict[str, Any] = {
        "label": label,
        "n_total": int(len(df)),
        "n_parse_ok_valid_basis": int(len(data)),
    }
    if data.empty:
        return out

    rate_cols = [
        "only_carbon",
        "co_only",
        "oxygen_only_hetero",
        "co_only_high_c",
        "co_only_high_c_low_o",
        "co_only_mostly_c_o1_3",
        "carbon_rich_no_non_o_hetero",
        "carbon_rich_no_nitrogen",
        "no_non_o_hetero",
        "has_non_o_hetero",
        "has_nitrogen",
        "has_oxygen",
        "has_sulfur",
        "has_phosphorus",
        "has_halogen",
    ]
    for col in rate_cols:
        out[f"{col}_rate"] = float(data[col].mean())

    quantile_cols = [
        "carbon_fraction_elem",
        "oxygen_fraction_elem",
        "oxygen_count",
        "nitrogen_count",
        "sulfur_count",
        "halogen_count",
        "hetero_count_elem",
        "non_o_hetero_count",
        "hetero_type_count",
        "non_o_hetero_type_count",
    ]
    for col in quantile_cols:
        out[f"{col}_mean"] = float(data[col].mean())
        out[f"{col}_p05"] = float(data[col].quantile(0.05))
        out[f"{col}_p50"] = float(data[col].quantile(0.50))
        out[f"{col}_p95"] = float(data[col].quantile(0.95))
        out[f"{col}_p99"] = float(data[col].quantile(0.99))
    return out


def count_signatures(df: pd.DataFrame, label: str, n: int = 40) -> pd.DataFrame:
    """Return top hetero signatures and element sets."""
    if "valid" in df.columns:
        data = df[df["valid"].astype(bool) & df["element_parse_ok"].eq(1)].copy()
    else:
        data = df[df["element_parse_ok"].eq(1)].copy()
    total = len(data)
    rows: list[dict[str, Any]] = []
    for col in ["hetero_signature", "element_set"]:
        counts = data[col].value_counts().head(n)
        for key, value in counts.items():
            rows.append(
                {
                    "dataset": label,
                    "field": col,
                    "value": key,
                    "count": int(value),
                    "fraction": float(value / total) if total else np.nan,
                }
            )
    return pd.DataFrame(rows)


def reward_column(df: pd.DataFrame) -> str | None:
    for col in ["reward_total_raw", "reward_total", "reward_total_norm"]:
        if col in df.columns:
            return col
    return None


def top_reward_summaries(generated: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize and sample top reward C/O-only collapse examples."""
    summary_rows: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []
    for run, group in generated.groupby("run", sort=True):
        rcol = reward_column(group)
        if not rcol:
            continue
        valid = group[group["valid"].astype(bool) & group["element_parse_ok"].eq(1)]
        valid = valid[valid[rcol].notna()]
        if len(valid) < 50:
            continue
        top_n = max(20, int(np.ceil(len(valid) * 0.10)))
        top = valid.sort_values(rcol, ascending=False).head(top_n)
        row = summarize(top, f"{run}:top_reward_decile")
        row["run"] = run
        row["top_n"] = int(len(top))
        row["reward_col"] = rcol
        summary_rows.append(row)

        candidates = top[
            top["co_only_mostly_c_o1_3"].eq(1)
            | top["co_only_high_c"].eq(1)
            | top["carbon_rich_no_non_o_hetero"].eq(1)
        ].sort_values(rcol, ascending=False)
        for _, item in candidates.head(25).iterrows():
            example_rows.append(
                {
                    "run": run,
                    "epoch": item.get("epoch"),
                    "reward_col": rcol,
                    "reward_value": item.get(rcol),
                    "smiles": item.get("smiles"),
                    "carbon_fraction": item.get("carbon_fraction_elem"),
                    "oxygen_count": item.get("oxygen_count"),
                    "nitrogen_count": item.get("nitrogen_count"),
                    "sulfur_count": item.get("sulfur_count"),
                    "halogen_count": item.get("halogen_count"),
                    "hetero_signature": item.get("hetero_signature"),
                    "co_only_high_c": item.get("co_only_high_c"),
                    "co_only_high_c_low_o": item.get("co_only_high_c_low_o"),
                    "co_only_mostly_c_o1_3": item.get("co_only_mostly_c_o1_3"),
                    "logp": item.get("logp_calc"),
                    "tpsa": item.get("tpsa_calc"),
                    "strict_core_pass": item.get("strict_core_pass"),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(example_rows)


def selected_epoch_summaries(generated: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in SELECTED_RUNS:
        subset = generated[generated["run"].eq(run)]
        if subset.empty:
            continue
        for epoch, group in subset.groupby("epoch", sort=True):
            row = summarize(group, f"{run}:epoch_{epoch}")
            row["run"] = run
            row["epoch"] = int(epoch)
            rows.append(row)
    return pd.DataFrame(rows)


def write_report(
    chembl_summary: dict[str, Any],
    strict_summary: dict[str, Any],
    generated_summary: dict[str, Any],
    top_reward_summary: pd.DataFrame,
) -> None:
    """Write a compact Chinese report."""
    def pct(d: dict[str, Any], key: str) -> str:
        value = d.get(key)
        if value is None or pd.isna(value):
            return "NA"
        return f"{100 * float(value):.2f}%"

    top_lines = []
    if not top_reward_summary.empty:
        ranked = top_reward_summary.sort_values("co_only_mostly_c_o1_3_rate", ascending=False)
        for _, row in ranked.head(8).iterrows():
            top_lines.append(
                "- {run}: top reward decile C/O-only mostly-carbon O1-3 = {rate:.2%}, "
                "no non-O hetero = {no_non_o:.2%}, has N = {has_n:.2%}".format(
                    run=row["run"],
                    rate=row.get("co_only_mostly_c_o1_3_rate", np.nan),
                    no_non_o=row.get("no_non_o_hetero_rate", np.nan),
                    has_n=row.get("has_nitrogen_rate", np.nan),
                )
            )
    top_text = "\n".join(top_lines) if top_lines else "- No top reward summaries available."

    report = f"""# C/O-only 与非氧杂原子分布专项诊断

生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 一、问题定义

本分析针对的不是“全氧崩坏”，而是更符合当前担忧的：

> 生成分子主要由大量碳和少量氧组成，缺少 N、S、卤素、P 等非氧杂原子多样性。

这类模式会让分子看起来满足了“有杂原子/极性不要太低”的简单规则，但本质上仍然是
composition collapse。

## 二、核心指标

- `co_only`：重原子元素只包含 C 和 O，且至少有一个 O。
- `oxygen_only_hetero`：杂原子全是 O，没有 N/S/P/卤素等非氧杂原子。
- `co_only_high_c`：C/O-only 且 carbon fraction >= 0.85。
- `co_only_high_c_low_o`：C/O-only、carbon fraction >= 0.85、O 原子数为 1-2。
- `co_only_mostly_c_o1_3`：C/O-only、carbon fraction >= 0.80、O 原子数为 1-3。
- `no_non_o_hetero`：没有任何非氧杂原子，包括全碳和 C/O-only。

## 三、ChEMBL36 vs OUT 总体对比

| 指标 | ChEMBL36 cleaned | strict ChEMBL property pool | OUT valid generated |
|---|---:|---:|---:|
| co_only | {pct(chembl_summary, 'co_only_rate')} | {pct(strict_summary, 'co_only_rate')} | {pct(generated_summary, 'co_only_rate')} |
| oxygen_only_hetero | {pct(chembl_summary, 'oxygen_only_hetero_rate')} | {pct(strict_summary, 'oxygen_only_hetero_rate')} | {pct(generated_summary, 'oxygen_only_hetero_rate')} |
| co_only_high_c | {pct(chembl_summary, 'co_only_high_c_rate')} | {pct(strict_summary, 'co_only_high_c_rate')} | {pct(generated_summary, 'co_only_high_c_rate')} |
| co_only_high_c_low_o | {pct(chembl_summary, 'co_only_high_c_low_o_rate')} | {pct(strict_summary, 'co_only_high_c_low_o_rate')} | {pct(generated_summary, 'co_only_high_c_low_o_rate')} |
| co_only_mostly_c_o1_3 | {pct(chembl_summary, 'co_only_mostly_c_o1_3_rate')} | {pct(strict_summary, 'co_only_mostly_c_o1_3_rate')} | {pct(generated_summary, 'co_only_mostly_c_o1_3_rate')} |
| no_non_o_hetero | {pct(chembl_summary, 'no_non_o_hetero_rate')} | {pct(strict_summary, 'no_non_o_hetero_rate')} | {pct(generated_summary, 'no_non_o_hetero_rate')} |
| has_nitrogen | {pct(chembl_summary, 'has_nitrogen_rate')} | {pct(strict_summary, 'has_nitrogen_rate')} | {pct(generated_summary, 'has_nitrogen_rate')} |
| has_sulfur | {pct(chembl_summary, 'has_sulfur_rate')} | {pct(strict_summary, 'has_sulfur_rate')} | {pct(generated_summary, 'has_sulfur_rate')} |
| has_halogen | {pct(chembl_summary, 'has_halogen_rate')} | {pct(strict_summary, 'has_halogen_rate')} | {pct(generated_summary, 'has_halogen_rate')} |
| has_non_o_hetero | {pct(chembl_summary, 'has_non_o_hetero_rate')} | {pct(strict_summary, 'has_non_o_hetero_rate')} | {pct(generated_summary, 'has_non_o_hetero_rate')} |

## 四、Top reward 区域

{top_text}

## 五、结论用法

如果 OUT 或 top reward decile 中 `co_only_high_c_low_o`、`co_only_mostly_c_o1_3`、
`no_non_o_hetero` 显著高于 ChEMBL baseline，就说明模型不是简单“缺氧”，也不是
“氧过多”，而是走向 **C/O-only mostly-carbon collapse**。

对应的数据集策略应是：

1. 不把所有生成分子当负样本。
2. 把 C/O-only 且 high-carbon、low-O、high-reward 的样本作为 hard negatives。
3. 正样本抽样必须按 hetero signature 分层，保证 N、S、卤素、N+O、N+O+halogen 等
   ChEMBL 常见组合不被 C/O-only 分子淹没。
4. logP/TPSA 只能做区间约束，不能把“更低 logP”当作持续正奖励。
"""
    (ANALYSIS_DIR / "co_only_composition_report.md").write_text(report)


def main() -> None:
    print("Loading ChEMBL36 properties...")
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
        "num_aromatic_rings",
        "num_rings",
        "carbon_fraction",
        "lipinski_violations",
    ]
    chembl = pd.read_csv(
        ROOT / "data/chembl/chembl_36_properties.csv.gz",
        usecols=chembl_cols,
        compression="gzip",
    )
    print(f"ChEMBL rows: {len(chembl):,}")
    chembl = add_element_features(chembl, "canonical_smiles")
    strict_mask = property_pool_mask(chembl)
    chembl_strict = chembl.loc[strict_mask].copy()

    print("Loading generated molecule flags...")
    generated = pd.read_csv(
        PREV_ANALYSIS_DIR / "molecule_flags_compact.csv.gz",
        compression="gzip",
    )
    generated = add_element_features(generated, "smiles")

    print("Writing summaries...")
    chembl_summary = summarize(chembl, "chembl36_cleaned")
    strict_summary = summarize(chembl_strict, "chembl36_strict_property_pool")
    generated_summary = summarize(generated, "out_valid_generated")

    summaries = pd.DataFrame([chembl_summary, strict_summary, generated_summary])
    summaries.to_csv(ANALYSIS_DIR / "global_composition_summary.csv", index=False)
    (ANALYSIS_DIR / "global_composition_summary.json").write_text(
        json.dumps(
            {
                "chembl36_cleaned": chembl_summary,
                "chembl36_strict_property_pool": strict_summary,
                "out_valid_generated": generated_summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    signature_counts = pd.concat(
        [
            count_signatures(chembl, "chembl36_cleaned"),
            count_signatures(chembl_strict, "chembl36_strict_property_pool"),
            count_signatures(generated, "out_valid_generated"),
        ],
        ignore_index=True,
    )
    signature_counts.to_csv(ANALYSIS_DIR / "hetero_signature_counts.csv", index=False)

    run_rows = []
    for run, group in generated.groupby("run", sort=True):
        row = summarize(group, run)
        row["run"] = run
        run_rows.append(row)
    pd.DataFrame(run_rows).to_csv(ANALYSIS_DIR / "run_composition_summary.csv", index=False)

    selected_epoch_summaries(generated).to_csv(
        ANALYSIS_DIR / "selected_epoch_composition_summary.csv", index=False
    )

    top_summary, top_examples = top_reward_summaries(generated)
    top_summary.to_csv(ANALYSIS_DIR / "top_reward_composition_summary.csv", index=False)
    top_examples.to_csv(ANALYSIS_DIR / "top_reward_co_only_examples.csv", index=False)

    compact_cols = [
        "run",
        "epoch",
        "smiles",
        "valid",
        "strict_core_pass",
        "reward_total_raw",
        "chembl_classifier_reward",
        "carbon_fraction_elem",
        "oxygen_count",
        "nitrogen_count",
        "sulfur_count",
        "phosphorus_count",
        "halogen_count",
        "hetero_signature",
        "co_only",
        "oxygen_only_hetero",
        "co_only_high_c",
        "co_only_high_c_low_o",
        "co_only_mostly_c_o1_3",
        "no_non_o_hetero",
        "has_non_o_hetero",
        "has_nitrogen",
        "has_sulfur",
        "has_halogen",
        "logp_calc",
        "tpsa_calc",
    ]
    existing = [col for col in compact_cols if col in generated.columns]
    generated[existing].to_csv(
        ANALYSIS_DIR / "generated_composition_flags.csv.gz",
        index=False,
        compression="gzip",
    )

    write_report(chembl_summary, strict_summary, generated_summary, top_summary)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(ROOT),
        "analysis_dir": str(ANALYSIS_DIR),
        "chembl_rows": int(len(chembl)),
        "generated_rows": int(len(generated)),
        "outputs": [
            "analysis_manifest.json",
            "analyze_hetero_composition.py",
            "co_only_composition_report.md",
            "generated_composition_flags.csv.gz",
            "global_composition_summary.csv",
            "global_composition_summary.json",
            "hetero_signature_counts.csv",
            "run_composition_summary.csv",
            "selected_epoch_composition_summary.csv",
            "top_reward_co_only_examples.csv",
            "top_reward_composition_summary.csv",
        ],
    }
    (ANALYSIS_DIR / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    print("Done.")


if __name__ == "__main__":
    main()
