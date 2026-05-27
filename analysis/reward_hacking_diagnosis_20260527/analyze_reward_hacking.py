#!/usr/bin/env python3
"""Diagnose reward hacking and molecule-distribution drift in DiffB110 OUT runs.

This script is intentionally read-only with respect to training data and model
files. It writes analysis artifacts only into its own directory.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

try:
    import yaml
except Exception:  # pragma: no cover - optional in non-project envs
    yaml = None


RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "OUT"
ANALYSIS_DIR = Path(__file__).resolve().parent

SELECTED_RUNS = [
    "7gpu-classifier-only_20260526_210715",
    "7gpu-dock-ecfp4-fcfp4_20260525_212140",
    "7gpu-dock-ecfp4-fcfp4_20260526_115341",
    "7gpu-dock-ecfp4-fcfp4_20260521_212927",
    "6gpu-chembl-only_20260521_154957",
    "8gpu-chembl-only_20260520_101340",
    "qed-sa-docking-8gjl-8gpu_20260508_212050",
]


def _largest_component_size(edges: dict[int, set[int]], n_nodes: int) -> int:
    """Return largest connected component size in a small ring graph."""
    if n_nodes <= 0:
        return 0
    seen: set[int] = set()
    best = 1
    for start in range(n_nodes):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for nxt in edges.get(node, set()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        best = max(best, size)
    return best


def molecule_features(smiles: object) -> dict[str, float | int]:
    """Compute structural and property flags for one SMILES-like value."""
    if (
        not isinstance(smiles, str)
        or not smiles
        or smiles == "None"
        or smiles.startswith("NonTensorData")
    ):
        return {"parse_ok": 0}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"parse_ok": 0}

    atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    heavy = len(atoms)
    carbon = sum(1 for atom in atoms if atom.GetAtomicNum() == 6)
    hetero = heavy - carbon
    element_types = len({atom.GetSymbol() for atom in atoms})
    carbon_fraction = carbon / heavy if heavy else 0.0

    rings = [tuple(ring) for ring in mol.GetRingInfo().AtomRings()]
    ring_sets = [set(ring) for ring in rings]
    ring_sizes = [len(ring) for ring in rings]
    aromatic_ring_ids = [
        idx
        for idx, ring in enumerate(rings)
        if all(mol.GetAtomWithIdx(atom_idx).GetIsAromatic() for atom_idx in ring)
    ]
    aromatic_ring_id_set = set(aromatic_ring_ids)

    fused_edges: dict[int, set[int]] = defaultdict(set)
    aromatic_fused_edges: dict[int, set[int]] = defaultdict(set)
    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            if len(ring_sets[i].intersection(ring_sets[j])) >= 2:
                fused_edges[i].add(j)
                fused_edges[j].add(i)
                if i in aromatic_ring_id_set and j in aromatic_ring_id_set:
                    aromatic_fused_edges[i].add(j)
                    aromatic_fused_edges[j].add(i)

    max_fused = _largest_component_size(fused_edges, len(rings))
    max_aromatic_fused = _largest_component_size(
        aromatic_fused_edges, len(aromatic_ring_ids)
    )

    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hba = Lipinski.NumHAcceptors(mol)
    hbd = Lipinski.NumHDonors(mol)
    rotb = Lipinski.NumRotatableBonds(mol)
    bridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)

    strict_core = (
        10 <= heavy <= 30
        and 150 <= mw <= 500
        and -1 <= logp <= 6
        and 20 <= tpsa <= 140
        and 1 <= hba <= 10
        and hbd <= 5
        and rotb <= 10
        and hetero >= 1
        and 0.35 <= carbon_fraction <= 0.90
        and not any(size == 3 for size in ring_sizes)
        and len(rings) <= 6
        and max_aromatic_fused <= 3
    )

    ring_complex = (
        len(rings) >= 5 or max_fused >= 4 or bridgehead >= 2 or spiro >= 2
    )

    return {
        "parse_ok": 1,
        "heavy": heavy,
        "hetero": hetero,
        "element_types": element_types,
        "carbon_fraction_calc": carbon_fraction,
        "mw_calc": mw,
        "logp_calc": logp,
        "tpsa_calc": tpsa,
        "hba_calc": hba,
        "hbd_calc": hbd,
        "rotb_calc": rotb,
        "ring_count": len(rings),
        "aromatic_rings": len(aromatic_ring_ids),
        "max_fused_ring_cluster": max_fused,
        "max_aromatic_fused_cluster": max_aromatic_fused,
        "bridgehead": bridgehead,
        "spiro": spiro,
        "strict_core_pass": int(strict_core),
        "high_carbon_90": int(carbon_fraction >= 0.90),
        "high_carbon_95": int(carbon_fraction >= 0.95),
        "no_hetero": int(hetero == 0),
        "low_hetero": int(hetero <= 1),
        "few_element_types": int(element_types <= 2),
        "has_3ring": int(any(size == 3 for size in ring_sizes)),
        "has_4ring": int(any(size == 4 for size in ring_sizes)),
        "large_ring": int(any(size >= 8 for size in ring_sizes)),
        "ring_complex": int(ring_complex),
        "aromatic_fused_ge4": int(max_aromatic_fused >= 4),
        "pah_like": int(hetero == 0 and len(aromatic_ring_ids) >= 2),
        "logp_high": int(logp > 5.5),
        "tpsa_low": int(tpsa < 20),
        "heavy_high": int(heavy > 30),
    }


def reward_col(df: pd.DataFrame) -> str | None:
    """Choose the best available molecule-level reward column."""
    for col in ["reward_total_raw", "reward_total", "reward_total_norm"]:
        if col in df.columns:
            return col
    return None


def summarize(df: pd.DataFrame) -> dict[str, Any]:
    """Summarize structural flags for a molecule dataframe."""
    out: dict[str, Any] = {"n": int(len(df))}
    out["valid_rate"] = float(df["valid"].mean()) if len(df) else np.nan
    valid = df[df["valid"]]
    out["valid_n"] = int(len(valid))
    if valid.empty:
        return out

    for col in [
        "strict_core_pass",
        "high_carbon_90",
        "high_carbon_95",
        "no_hetero",
        "low_hetero",
        "few_element_types",
        "has_3ring",
        "ring_complex",
        "aromatic_fused_ge4",
        "pah_like",
        "logp_high",
        "tpsa_low",
        "heavy_high",
    ]:
        out[f"{col}_rate"] = float(valid[col].mean())

    for col in [
        "carbon_fraction_calc",
        "hetero",
        "element_types",
        "ring_count",
        "aromatic_rings",
        "max_fused_ring_cluster",
        "max_aromatic_fused_cluster",
        "logp_calc",
        "tpsa_calc",
        "heavy",
    ]:
        out[f"{col}_mean"] = float(valid[col].mean())
        out[f"{col}_p95"] = float(valid[col].quantile(0.95))

    return out


def load_all_molecules() -> pd.DataFrame:
    """Load combined molecule CSV files from all OUT runs."""
    frames: list[pd.DataFrame] = []
    for path in sorted(OUT_DIR.glob("*/epochs/epoch_*/molecules.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "smiles" not in df.columns:
            continue
        df = df.copy()
        df["run"] = path.parts[-4]
        df["epoch"] = int(path.parent.name.split("_")[-1])
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def add_molecule_flags(raw: pd.DataFrame) -> pd.DataFrame:
    """Compute and append molecular flags, caching repeated SMILES."""
    cache: dict[object, dict[str, float | int]] = {}
    rows: list[dict[str, float | int]] = []
    for smiles in raw["smiles"].tolist():
        if smiles not in cache:
            cache[smiles] = molecule_features(smiles)
        rows.append(cache[smiles])

    flags = pd.DataFrame(rows)
    df = pd.concat([raw.reset_index(drop=True), flags], axis=1)
    df["valid"] = (
        df.get("rdkit_valid", 0).fillna(0).astype(int).eq(1)
        & df["parse_ok"].fillna(0).astype(int).eq(1)
    )
    df.attrs["unique_smiles_or_values"] = len(cache)
    return df


def build_run_summaries(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build per-run and selected per-epoch summaries."""
    run_rows = []
    for run, group in df.groupby("run"):
        row = summarize(group)
        row["run"] = run
        row["epochs"] = int(group["epoch"].nunique())
        row["last_epoch"] = int(group["epoch"].max())
        run_rows.append(row)
    run_summary = pd.DataFrame(run_rows).sort_values("n", ascending=False)

    epoch_rows = []
    for run in SELECTED_RUNS:
        group = df[df["run"].eq(run)]
        if group.empty:
            continue
        rcol = reward_col(group)
        for epoch, epoch_group in group.groupby("epoch"):
            row = summarize(epoch_group)
            row["run"] = run
            row["epoch"] = int(epoch)
            row["reward_col"] = rcol
            if rcol is not None:
                row["reward_mean"] = float(
                    pd.to_numeric(epoch_group[rcol], errors="coerce").mean()
                )
            epoch_rows.append(row)
    epoch_summary = pd.DataFrame(epoch_rows).sort_values(["run", "epoch"])
    return run_summary, epoch_summary


def build_top_reward_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build top reward decile summaries and bad molecule examples."""
    decile_rows = []
    example_rows = []
    for run, group in df.groupby("run"):
        rcol = reward_col(group)
        if rcol is None:
            continue
        valid = group[group["valid"]].copy()
        valid[rcol] = pd.to_numeric(valid[rcol], errors="coerce")
        valid = valid.dropna(subset=[rcol])
        if len(valid) < 100:
            continue
        top_n = max(50, int(len(valid) * 0.10))
        top = valid.nlargest(top_n, rcol)
        row = summarize(top)
        row["run"] = run
        row["top_n"] = int(len(top))
        row["reward_col"] = rcol
        decile_rows.append(row)

        if run not in SELECTED_RUNS:
            continue
        picked = 0
        for _, mol_row in valid.sort_values(rcol, ascending=False).iterrows():
            bad_tags = []
            if int(mol_row.get("high_carbon_90", 0)):
                bad_tags.append("high_carbon_90")
            if int(mol_row.get("low_hetero", 0)):
                bad_tags.append("low_hetero")
            if int(mol_row.get("has_3ring", 0)):
                bad_tags.append("three_member_ring")
            if int(mol_row.get("ring_complex", 0)):
                bad_tags.append("ring_complex")
            if int(mol_row.get("logp_high", 0)):
                bad_tags.append("logp_high")
            if int(mol_row.get("tpsa_low", 0)):
                bad_tags.append("tpsa_low")
            if int(mol_row.get("few_element_types", 0)):
                bad_tags.append("few_element_types")
            if not bad_tags:
                continue
            example_rows.append(
                {
                    "run": run,
                    "epoch": int(mol_row["epoch"]),
                    "reward_col": rcol,
                    "reward_value": float(mol_row[rcol]),
                    "bad_tags": "|".join(bad_tags),
                    "smiles": mol_row.get("smiles"),
                    "carbon_fraction": mol_row.get("carbon_fraction_calc"),
                    "hetero": mol_row.get("hetero"),
                    "ring_count": mol_row.get("ring_count"),
                    "max_fused_ring_cluster": mol_row.get("max_fused_ring_cluster"),
                    "logp": mol_row.get("logp_calc"),
                    "tpsa": mol_row.get("tpsa_calc"),
                    "heavy": mol_row.get("heavy"),
                }
            )
            picked += 1
            if picked >= 30:
                break

    return (
        pd.DataFrame(decile_rows).sort_values("top_n", ascending=False),
        pd.DataFrame(example_rows).sort_values(["run", "reward_value"], ascending=[True, False]),
    )


def chembl_baselines() -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute property-only ChEMBL baseline and positive-pool estimates."""
    props = pd.read_csv(ROOT / "data/chembl/chembl_36_properties.csv.gz", compression="gzip")
    baseline = {
        "n": int(len(props)),
        "carbon_fraction_mean": float(props["carbon_fraction"].mean()),
        "carbon_fraction_p95": float(props["carbon_fraction"].quantile(0.95)),
        "carbon_fraction_p99": float(props["carbon_fraction"].quantile(0.99)),
        "carbon_ge_0_90": float((props["carbon_fraction"] >= 0.90).mean()),
        "carbon_ge_0_95": float((props["carbon_fraction"] >= 0.95).mean()),
        "no_hetero": float((props["num_heteroatoms"] == 0).mean()),
        "low_hetero_le_1": float((props["num_heteroatoms"] <= 1).mean()),
        "hetero_le_2": float((props["num_heteroatoms"] <= 2).mean()),
        "aromatic_rings_ge_5": float((props["num_aromatic_rings"] >= 5).mean()),
        "rings_ge_6": float((props["num_rings"] >= 6).mean()),
        "logp_gt_5_5": float((props["mol_log_p"] > 5.5).mean()),
        "tpsa_lt_20": float((props["tpsa"] < 20).mean()),
        "lipinski0": float((props["lipinski_violations"] == 0).mean()),
    }

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
    strict_property_pool = (
        property_core
        & props["lipinski_violations"].eq(0)
        & props["carbon_fraction"].le(0.88)
        & props["num_heteroatoms"].ge(2)
        & props["num_aromatic_rings"].le(4)
        & props["num_rings"].le(6)
        & props["mol_log_p"].between(0, 5.5)
        & props["tpsa"].between(20, 120)
    )
    pool = {
        "base_all": int(len(props)),
        "property_core_loose": int(property_core.sum()),
        "property_core_lipinski0": int((property_core & props["lipinski_violations"].eq(0)).sum()),
        "recommended_strict_property_pool": int(strict_property_pool.sum()),
        "recommended_strict_property_pool_fraction": float(strict_property_pool.mean()),
    }
    quantiles = props.loc[
        strict_property_pool,
        [
            "heavy_atoms",
            "mol_wt",
            "mol_log_p",
            "tpsa",
            "num_heteroatoms",
            "carbon_fraction",
            "num_rings",
            "num_aromatic_rings",
            "num_rotatable_bonds",
        ],
    ].quantile([0.01, 0.05, 0.5, 0.95, 0.99])
    pool["recommended_pool_quantiles"] = json.loads(quantiles.to_json())
    return baseline, pool


def negative_label_counts() -> dict[str, int]:
    """Count existing negative labels."""
    path = ROOT / "chembl_classifier/data/negatives/all_negatives_labeled.smi"
    if not path.exists():
        return {}
    labels = Counter()
    for line in path.read_text().splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 2:
            labels[parts[1]] += 1
    return dict(labels.most_common())


def key_config_summary() -> dict[str, Any]:
    """Extract compact reward/train config for selected runs."""
    result: dict[str, Any] = {}
    if yaml is None:
        return result
    for run in SELECTED_RUNS:
        path = OUT_DIR / run / "config.yaml"
        if not path.exists():
            continue
        cfg = yaml.safe_load(path.read_text())
        reward = cfg.get("reward", {})
        train = cfg.get("train", {})
        result[run] = {
            "weights": reward.get("weights"),
            "normalization_enabled": reward.get("normalization", {}).get("enabled"),
            "per_component_normalization_enabled": reward.get(
                "per_component_normalization", {}
            ).get("enabled"),
            "learning_rate": train.get("learning_rate"),
            "kl_coef": train.get("kl_coef"),
            "ppo_epochs": train.get("ppo_epochs"),
            "epochs_configured": train.get("epochs"),
        }
    return result


def write_report(
    df: pd.DataFrame,
    run_summary: pd.DataFrame,
    epoch_summary: pd.DataFrame,
    top_decile: pd.DataFrame,
    chembl_baseline: dict[str, Any],
    positive_pool: dict[str, Any],
    neg_counts: dict[str, int],
) -> None:
    """Write a concise Chinese diagnosis report."""
    global_stats = summarize(df)
    major = run_summary[
        (run_summary["n"] >= 3000)
        & ~run_summary["run"].str.contains("smoke", case=False, na=False)
    ].copy()
    report = f"""# DiffB110 Reward Hacking 与正负样本策略诊断

生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 一、分析范围

- 分子级记录：{len(df):,} rows
- 覆盖 run 数：{df['run'].nunique()}
- 唯一 SMILES/异常值：{df.attrs.get('unique_smiles_or_values', 'NA'):,}
- 输入目录：`OUT/*/epochs/epoch_*/molecules.csv`
- 本分析只读训练输出、ChEMBL property 表和当前分类器负样本文件；没有生成训练集。

## 二、核心结论

1. 当前问题不是单个坏结构，而是 RL 后生成分布整体偏移。
2. 全局有效分子里，严格药物合理核心通过率只有 {global_stats.get('strict_core_pass_rate', np.nan):.3f}。
3. 三元环比例极高：全局 valid 分子中 {global_stats.get('has_3ring_rate', np.nan):.3f} 含三元环。
4. 高碳问题显著：全局 valid 分子中 C>=0.90 比例为 {global_stats.get('high_carbon_90_rate', np.nan):.3f}，而 ChEMBL36 baseline 只有 {chembl_baseline['carbon_ge_0_90']:.3f}。
5. 低杂原子问题显著：全局 valid 分子中 hetero<=1 比例为 {global_stats.get('low_hetero_rate', np.nan):.3f}，而 ChEMBL36 baseline 只有 {chembl_baseline['low_hetero_le_1']:.3f}。
6. 高 reward 区域仍富集坏结构，说明 reward 可以被利用，不能只靠当前二分类器兜底。

## 三、最关键的崩坏证据

### 7gpu-dock-ecfp4-fcfp4_20260525_212140

- epoch 1：C>=0.90 为 0.012，strict core 为 0.276。
- epoch 34 reward 最好：C>=0.90 升到 0.357，三元环 0.578，复杂环 0.399。
- epoch 62：C>=0.90 为 0.435，hetero<=1 为 0.272，三元环 0.682，复杂环 0.609，strict core 降到 0.172。
- 解释：docking + ChEMBL prior + logP distribution 能维持有效率，但会推动高碳、多环、少杂原子结构。

### 6gpu-chembl-only_20260521_154957

- epoch 16 reward 最好：C>=0.90 为 0.643，hetero<=1 为 0.464，no hetero 为 0.200。
- epoch 20：C>=0.90 为 0.720，hetero<=1 为 0.548，strict core 只有 0.152。
- 解释：仅 ChEMBL prior/指纹相似度不足以约束元素组成，模型会找到高碳相似片段捷径。

### 7gpu-classifier-only_20260526_210715

- epoch 23 reward 最好，但三元环为 0.544，复杂环为 0.340。
- epoch 26 开始有效率低于 0.5，epoch 64 有效率为 0。
- 解释：当前分类器奖励单独使用时会被策略推到不可采样/不可解析区域，不能作为唯一 reward。

## 四、ChEMBL 正样本池

- ChEMBL36 cleaned 总量：{positive_pool['base_all']:,}
- property_core_loose：{positive_pool['property_core_loose']:,}
- property_core_lipinski0：{positive_pool['property_core_lipinski0']:,}
- recommended_strict_property_pool：{positive_pool['recommended_strict_property_pool']:,}

结论：正样本不缺，问题是不能随机抽 100k。应从 strict property pool 中 scaffold-stratified + descriptor-stratified 抽样。

## 五、当前负样本覆盖缺口

当前负样本标签分布：

{json.dumps(neg_counts, ensure_ascii=False, indent=2)}

主要缺口：

- 三元环没有作为独立负样本标签。
- 复杂桥环/稠合非芳香环没有作为独立负样本标签。
- 高 reward 但坏结构的样本没有系统性 hard-negative mining。
- 负样本以 all-carbon 和若干 pain point 为主，对“看起来单分子合理但分布错误”的模式覆盖不足。

## 六、建议的数据集方向

正样本：

- 从 recommended strict ChEMBL pool 抽样。
- 按 Bemis-Murcko scaffold 限额，避免大系列支配。
- 按 heavy atoms、carbon fraction、hetero count、ring count、logP、TPSA 分箱抽样。
- 不要把 ChEMBL 中高碳、低杂原子、极端 logP/TPSA、复杂稠环样本打成强正样本。

负样本：

- 从 OUT 中按 failure category 分层收集。
- 必须新增：three_member_ring、ring_complex、high_carbon_90、low_hetero、few_element_types、high_reward_bad、invalid_after_rl。
- hard negative 应优先来自 reward 高但 strict_core_pass=0 的分子。
- 不要把低 ChEMBL similarity 单独当负样本；低相似度只有叠加坏结构/坏性质时才是负样本。

奖励上线：

- 分类器只能作为 soft distribution reward。
- 三元环、高碳、无/低杂原子、复杂环应作为显式 hard penalty 或 hard filter。
- 每轮 RL 后做 hard-negative mining：高 classifier/reward 且规则失败的分子加入下一版负样本。

## 七、产物说明

- `molecule_flags_compact.csv.gz`：每个生成分子的结构标签和关键描述子。
- `run_summary.csv`：每个 run 的整体统计。
- `major_run_summary.csv`：主要非 smoke run 的统计。
- `selected_epoch_summary.csv`：关键 run 的逐 epoch 统计。
- `top_reward_decile_summary.csv`：每个 run top reward 10% 的坏结构比例。
- `top_reward_bad_examples.csv`：高 reward 坏分子代表样例。
- `chembl_baseline.json`：ChEMBL36 baseline。
- `chembl_positive_pool.json`：推荐正样本池规模估计。
- `negative_label_counts.json`：当前负样本标签统计。
- `key_config_summary.json`：关键 run 的 reward/train 配置摘要。
"""
    (ANALYSIS_DIR / "diagnosis_report.md").write_text(report)

    major.to_csv(ANALYSIS_DIR / "major_run_summary.csv", index=False)


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_all_molecules()
    df = add_molecule_flags(raw)

    compact_cols = [
        "run",
        "epoch",
        "batch_idx",
        "rank",
        "molecule_id",
        "smiles",
        "rdkit_valid",
        "valid",
        "reward_total_raw",
        "reward_total",
        "reward_total_norm",
        "chembl_classifier_reward",
        "chembl_prior_ecfp4_similarity",
        "chembl_prior_fcfp4_similarity",
        "docking_raw",
        "ligand_efficiency_raw",
        "parse_ok",
        "strict_core_pass",
        "high_carbon_90",
        "high_carbon_95",
        "no_hetero",
        "low_hetero",
        "few_element_types",
        "has_3ring",
        "ring_complex",
        "aromatic_fused_ge4",
        "pah_like",
        "logp_high",
        "tpsa_low",
        "heavy_high",
        "carbon_fraction_calc",
        "hetero",
        "element_types",
        "ring_count",
        "aromatic_rings",
        "max_fused_ring_cluster",
        "max_aromatic_fused_cluster",
        "logp_calc",
        "tpsa_calc",
        "heavy",
    ]
    existing_cols = [col for col in compact_cols if col in df.columns]
    df[existing_cols].to_csv(
        ANALYSIS_DIR / "molecule_flags_compact.csv.gz",
        index=False,
        compression="gzip",
    )

    run_summary, epoch_summary = build_run_summaries(df)
    run_summary.to_csv(ANALYSIS_DIR / "run_summary.csv", index=False)
    epoch_summary.to_csv(ANALYSIS_DIR / "selected_epoch_summary.csv", index=False)

    top_decile, top_examples = build_top_reward_tables(df)
    top_decile.to_csv(ANALYSIS_DIR / "top_reward_decile_summary.csv", index=False)
    top_examples.to_csv(ANALYSIS_DIR / "top_reward_bad_examples.csv", index=False)

    chembl_base, positive_pool = chembl_baselines()
    neg_counts = negative_label_counts()
    configs = key_config_summary()

    (ANALYSIS_DIR / "chembl_baseline.json").write_text(
        json.dumps(chembl_base, ensure_ascii=False, indent=2)
    )
    (ANALYSIS_DIR / "chembl_positive_pool.json").write_text(
        json.dumps(positive_pool, ensure_ascii=False, indent=2)
    )
    (ANALYSIS_DIR / "negative_label_counts.json").write_text(
        json.dumps(neg_counts, ensure_ascii=False, indent=2)
    )
    (ANALYSIS_DIR / "key_config_summary.json").write_text(
        json.dumps(configs, ensure_ascii=False, indent=2)
    )

    write_report(
        df,
        run_summary,
        epoch_summary,
        top_decile,
        chembl_base,
        positive_pool,
        neg_counts,
    )

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(ROOT),
        "analysis_dir": str(ANALYSIS_DIR),
        "molecule_rows": int(len(df)),
        "runs": int(df["run"].nunique()),
        "unique_smiles_or_values": int(df.attrs.get("unique_smiles_or_values", 0)),
        "outputs": sorted(path.name for path in ANALYSIS_DIR.iterdir() if path.is_file()),
    }
    (ANALYSIS_DIR / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
