#!/usr/bin/env python
"""Analyze DiffSBDD single-reward matrix runs and write figures/reports."""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "analysis" / "diffsbdd_single_reward_matrix_20260613"
OFFLINE_DOCKING_DIR = ANALYSIS_DIR / "offline_docking_all_epochs_gpu"
WORK_RECORD = ROOT / "work_records" / "26-6" / "260613" / "260613-02 DiffSBDD单奖励矩阵九组训练深度分析.md"


@dataclass(frozen=True)
class RunSpec:
    label: str
    reward: str
    path: Path


RUNS = [
    RunSpec("01_docking", "docking", ROOT / "runs/diffsbdd-single-reward-20260612-01-docking_20260612_210459"),
    RunSpec("02_qed", "qed", ROOT / "runs/diffsbdd-single-reward-20260612-02-qed_20260612_213118"),
    RunSpec("03_sa", "sa", ROOT / "runs/diffsbdd-single-reward-20260612-03-sa_20260612_215558"),
    RunSpec("04_tpsa", "tpsa_distribution", ROOT / "runs/diffsbdd-single-reward-20260612-04-tpsa_distribution_20260612_222038"),
    RunSpec("05_ecfp4", "ecfp4_prior", ROOT / "runs/diffsbdd-single-reward-20260612-05-ecfp4_prior_20260612_224533"),
    RunSpec("06_fcfp4", "fcfp4_prior", ROOT / "runs/diffsbdd-single-reward-20260612-06-fcfp4_prior_20260612_231147"),
    RunSpec("07_logp", "logp_distribution", ROOT / "runs/diffsbdd-single-reward-20260612-07-logp_distribution_20260612_233802"),
    RunSpec("08_le", "ligand_efficiency", ROOT / "runs/diffsbdd-single-reward-20260612-08-ligand_efficiency_20260613_000317"),
    RunSpec("09_classifier", "chembl_classifier", ROOT / "runs/diffsbdd-single-reward-20260612-09-chembl_classifier_20260613_002952"),
]


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    WORK_RECORD.parent.mkdir(parents=True, exist_ok=True)

    summary = _load_training_summary()
    mol_epoch = _load_molecule_epoch_metrics()
    merged = summary.merge(mol_epoch, on=["run_label", "reward", "epoch"], how="left")
    merged = _merge_offline_docking_metrics(merged)
    final = _build_final_table(merged)
    deltas = _build_delta_table(merged)

    summary.to_csv(ANALYSIS_DIR / "training_summary_long.csv", index=False)
    mol_epoch.to_csv(ANALYSIS_DIR / "molecule_epoch_metrics.csv", index=False)
    merged.to_csv(ANALYSIS_DIR / "merged_epoch_metrics.csv", index=False)
    final.to_csv(ANALYSIS_DIR / "final_epoch_comparison.csv", index=False)
    deltas.to_csv(ANALYSIS_DIR / "epoch10_minus_epoch1_delta.csv", index=False)

    _write_plots(merged, final, deltas)
    report = _build_report(merged, final, deltas)
    (ANALYSIS_DIR / "report.md").write_text(report, encoding="utf-8")
    WORK_RECORD.write_text(report, encoding="utf-8")
    print(ANALYSIS_DIR)
    print(WORK_RECORD)


def _load_training_summary() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for spec in RUNS:
        path = spec.path / "training_summary.csv"
        df = pd.read_csv(path)
        df.insert(0, "run_label", spec.label)
        df.insert(1, "reward", spec.reward)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["epoch"] = df["epoch"].astype(int)
    for col in df.columns:
        if col in {"run_label", "reward", "timestamp", "failure_counts", "checkpoint_path", "reward_enabled"}:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["fragmented_count"] = df["failure_counts"].map(lambda x: _failure_count(x, "fragmented_molecule"))
    df["sanitize_failed_count"] = df["failure_counts"].map(lambda x: _failure_count(x, "sanitize_failed"))
    df["success_count"] = df["failure_counts"].map(lambda x: _failure_count(x, "success"))
    df["fragmented_ratio"] = df["fragmented_count"] / df["generated_total"].replace(0, np.nan)
    df["sanitize_failed_ratio"] = df["sanitize_failed_count"] / df["generated_total"].replace(0, np.nan)
    return df


def _failure_count(raw: Any, key: str) -> float:
    if not isinstance(raw, str) or not raw.strip():
        return 0.0
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return 0.0
    return float(payload.get(key, 0.0))


def _load_molecule_epoch_metrics() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in RUNS:
        for epoch_dir in sorted((spec.path / "epochs").glob("epoch_*")):
            try:
                epoch = int(epoch_dir.name.split("_")[-1])
            except ValueError:
                continue
            mol_csv = epoch_dir / "molecules.csv"
            if not mol_csv.exists():
                continue
            rows.append(
                {
                    "run_label": spec.label,
                    "reward": spec.reward,
                    "epoch": epoch,
                    **_molecule_metrics_from_csv(mol_csv),
                }
            )
    return pd.DataFrame(rows)


def _merge_offline_docking_metrics(merged: pd.DataFrame) -> pd.DataFrame:
    """Overlay unified offline UniDock metrics for every reward/epoch when available."""
    path = OFFLINE_DOCKING_DIR / "offline_docking_by_epoch.csv"
    if not path.exists():
        return merged
    offline = pd.read_csv(path)
    run_to_label = {spec.path.name: spec.label for spec in RUNS}
    offline["run_label"] = offline["run"].map(run_to_label)
    keep = [
        "run_label",
        "reward",
        "epoch",
        "docking_success_count",
        "docking_success_rate_all",
        "docking_success_rate_valid",
        "avg_docking",
        "best_docking",
        "avg_ligand_efficiency",
    ]
    offline = offline[[col for col in keep if col in offline.columns]].copy()
    offline = offline.rename(
        columns={
            "avg_docking": "offline_avg_docking",
            "best_docking": "offline_best_docking",
            "avg_ligand_efficiency": "offline_avg_ligand_efficiency",
        }
    )
    out = merged.merge(offline, on=["run_label", "reward", "epoch"], how="left")
    if "offline_avg_docking" in out.columns:
        out["avg_docking_training"] = out.get("avg_docking")
        out["avg_docking"] = out["offline_avg_docking"].combine_first(out.get("avg_docking"))
    if "offline_avg_ligand_efficiency" in out.columns:
        out["avg_ligand_efficiency_training"] = out.get("avg_ligand_efficiency")
        out["avg_ligand_efficiency"] = out["offline_avg_ligand_efficiency"].combine_first(
            out.get("avg_ligand_efficiency")
        )
    return out


def _molecule_metrics_from_csv(path: Path) -> dict[str, Any]:
    mols: list[Chem.Mol] = []
    total = 0
    invalid = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            total += 1
            if str(row.get("rdkit_valid", "0")) not in {"1", "1.0"}:
                invalid += 1
                continue
            smiles = str(row.get("smiles") or "").strip()
            if not smiles or smiles == "None":
                invalid += 1
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                invalid += 1
                continue
            mols.append(mol)
    return _aggregate_mols(mols, total=total, invalid=invalid)


def _aggregate_mols(mols: list[Chem.Mol], *, total: int, invalid: int) -> dict[str, Any]:
    if not mols:
        return {
            "mol_total": total,
            "mol_valid": 0,
            "mol_invalid_or_missing": invalid,
        }
    per = [_per_mol_features(mol) for mol in mols]
    out: dict[str, Any] = {
        "mol_total": total,
        "mol_valid": len(mols),
        "mol_invalid_or_missing": invalid,
    }
    numeric_keys = sorted({key for row in per for key, value in row.items() if isinstance(value, (int, float))})
    for key in numeric_keys:
        vals = np.asarray([float(row.get(key, 0.0)) for row in per], dtype=float)
        out[f"{key}_mean"] = float(np.mean(vals))
        out[f"{key}_median"] = float(np.median(vals))
        out[f"{key}_max"] = float(np.max(vals))
        out[f"{key}_frac_gt0"] = float(np.mean(vals > 0))
    return out


def _per_mol_features(mol: Chem.Mol) -> dict[str, float]:
    atom_rings = [tuple(r) for r in mol.GetRingInfo().AtomRings()]
    ring_sets = [set(r) for r in atom_rings]
    ring_count = len(atom_rings)
    ring_sizes = [len(r) for r in atom_rings]
    large_ring_count = sum(1 for size in ring_sizes if size >= 7)
    three_ring_indices = [idx for idx, ring in enumerate(atom_rings) if len(ring) == 3]
    three_fused = 0
    for idx in three_ring_indices:
        tri = ring_sets[idx]
        if any(j != idx and tri & ring_sets[j] for j in range(len(atom_rings))):
            three_fused += 1
    large_fused_six = 0
    for idx, ring in enumerate(atom_rings):
        if len(ring) < 7:
            continue
        s = ring_sets[idx]
        if any(j != idx and len(atom_rings[j]) == 6 and len(s & ring_sets[j]) >= 2 for j in range(len(atom_rings))):
            large_fused_six += 1
    fused_max = _max_fused_ring_system_size(atom_rings)
    benzene_count = _benzene_ring_count(mol, atom_rings)
    aromatic_hetero_count = _aromatic_heterocycle_count(mol, atom_rings)
    halogen_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() in {9, 17, 35, 53})
    heavy = mol.GetNumHeavyAtoms()
    carbon = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
    hetero = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in {1, 6})
    return {
        "heavy_atom_count": float(heavy),
        "carbon_fraction_rdkit": float(carbon / heavy) if heavy else 0.0,
        "hetero_atom_count_rdkit": float(hetero),
        "qed_rdkit": float(QED.qed(mol)),
        "logp_rdkit": float(Crippen.MolLogP(mol)),
        "tpsa_rdkit": float(rdMolDescriptors.CalcTPSA(mol)),
        "hba_rdkit": float(Lipinski.NumHAcceptors(mol)),
        "hbd_rdkit": float(Lipinski.NumHDonors(mol)),
        "ring_count_rdkit": float(ring_count),
        "aromatic_ring_count_rdkit": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "aliphatic_ring_count_rdkit": float(rdMolDescriptors.CalcNumAliphaticRings(mol)),
        "benzene_ring_count": float(benzene_count),
        "aromatic_heterocycle_count": float(aromatic_hetero_count),
        "large_ring_count": float(large_ring_count),
        "large_ring_fused_six_count": float(large_fused_six),
        "three_member_ring_count_rdkit": float(len(three_ring_indices)),
        "three_member_fused_or_spiro_count": float(three_fused),
        "max_fused_ring_system_size": float(fused_max),
        "halogen_count": float(halogen_count),
        "over_halogenated_small": float(halogen_count >= 3 and heavy <= 24),
        "ring_pathology_flag": float(
            ring_count >= 4
            or large_ring_count > 0
            or large_fused_six > 0
            or len(three_ring_indices) > 1
            or three_fused > 0
        ),
        "high_carbon_flag": float(carbon / heavy >= 0.90) if heavy else 0.0,
    }


def _max_fused_ring_system_size(atom_rings: list[tuple[int, ...]]) -> int:
    if not atom_rings:
        return 0
    ring_sets = [set(ring) for ring in atom_rings]
    adjacency = [set() for _ in ring_sets]
    for i, left in enumerate(ring_sets):
        for j in range(i + 1, len(ring_sets)):
            if len(left & ring_sets[j]) >= 2:
                adjacency[i].add(j)
                adjacency[j].add(i)
    seen: set[int] = set()
    largest = 1
    for start in range(len(ring_sets)):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for nxt in adjacency[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        largest = max(largest, size)
    return largest


def _benzene_ring_count(mol: Chem.Mol, atom_rings: list[tuple[int, ...]]) -> int:
    count = 0
    for ring in atom_rings:
        atoms = [mol.GetAtomWithIdx(i) for i in ring]
        if len(ring) == 6 and all(atom.GetAtomicNum() == 6 and atom.GetIsAromatic() for atom in atoms):
            count += 1
    return count


def _aromatic_heterocycle_count(mol: Chem.Mol, atom_rings: list[tuple[int, ...]]) -> int:
    count = 0
    for ring in atom_rings:
        atoms = [mol.GetAtomWithIdx(i) for i in ring]
        if all(atom.GetIsAromatic() for atom in atoms) and any(atom.GetAtomicNum() not in {1, 6} for atom in atoms):
            count += 1
    return count


def _build_final_table(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for reward, df in merged.groupby("reward", sort=False):
        row = df.sort_values("epoch").iloc[-1].to_dict()
        rows.append(row)
    final = pd.DataFrame(rows)
    columns = [
        "run_label",
        "reward",
        "epoch",
        "reward_mean",
        "avg_reward_raw",
        "rdkit_valid_ratio",
        "fragmented_ratio",
        "sanitize_failed_ratio",
        "avg_docking",
        "avg_ligand_efficiency",
        "avg_qed",
        "avg_sa",
        "avg_chembl_classifier_reward",
        "avg_chembl_prior_ecfp4_reward",
        "avg_chembl_prior_fcfp4_reward",
        "logp_mean",
        "tpsa_mean",
        "carbon_fraction",
        "hetero_atom_mean",
        "aromatic_ring_mean",
        "descriptor_mmd_vs_baseline",
        "heavy_atom_count_mean",
        "ring_count_rdkit_mean",
        "benzene_ring_count_mean",
        "max_fused_ring_system_size_mean",
        "large_ring_count_mean",
        "three_member_ring_count_rdkit_mean",
        "ring_pathology_flag_mean",
        "halogen_count_mean",
        "over_halogenated_small_mean",
        "KL_Penalty_Weighted",
        "PPO_ApproxKL_K3",
    ]
    return final[[c for c in columns if c in final.columns]]


def _build_delta_table(merged: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "reward_mean",
        "rdkit_valid_ratio",
        "fragmented_ratio",
        "avg_docking",
        "avg_ligand_efficiency",
        "logp_mean",
        "tpsa_mean",
        "carbon_fraction",
        "hetero_atom_mean",
        "aromatic_ring_mean",
        "descriptor_mmd_vs_baseline",
        "heavy_atom_count_mean",
        "ring_count_rdkit_mean",
        "benzene_ring_count_mean",
        "max_fused_ring_system_size_mean",
        "large_ring_count_mean",
        "three_member_ring_count_rdkit_mean",
        "ring_pathology_flag_mean",
        "halogen_count_mean",
        "KL_Penalty_Weighted",
    ]
    rows = []
    for reward, df in merged.groupby("reward", sort=False):
        ordered = df.sort_values("epoch")
        first = ordered.iloc[0]
        last = ordered.iloc[-1]
        row = {"reward": reward, "run_label": str(last["run_label"])}
        for metric in metrics:
            if metric in ordered.columns:
                row[f"{metric}_e1"] = _safe_float(first.get(metric))
                row[f"{metric}_e10"] = _safe_float(last.get(metric))
                row[f"{metric}_delta"] = row[f"{metric}_e10"] - row[f"{metric}_e1"]
        rows.append(row)
    return pd.DataFrame(rows)


def _write_plots(merged: pd.DataFrame, final: pd.DataFrame, deltas: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 130,
        }
    )
    _plot_training_grid(merged)
    _plot_property_grid(merged)
    _plot_ring_grid(merged)
    _plot_final_bar_grid(final)
    _plot_delta_heatmap(deltas)


def _plot_training_grid(df: pd.DataFrame) -> None:
    metrics = [
        ("reward_mean", "Reward mean"),
        ("rdkit_valid_ratio", "RDKit valid ratio"),
        ("fragmented_ratio", "Fragmented ratio"),
        ("avg_docking", "Docking score"),
        ("KL_Penalty_Weighted", "Weighted KL"),
        ("PPO_ApproxKL_K3", "PPO K3 KL"),
    ]
    _line_grid(df, metrics, ANALYSIS_DIR / "training_curves_overview.png")


def _plot_property_grid(df: pd.DataFrame) -> None:
    metrics = [
        ("carbon_fraction", "Carbon fraction"),
        ("hetero_atom_mean", "Hetero atoms"),
        ("logp_mean", "LogP"),
        ("tpsa_mean", "TPSA"),
        ("heavy_atom_count_mean", "Heavy atoms"),
        ("descriptor_mmd_vs_baseline", "Descriptor MMD"),
    ]
    _line_grid(df, metrics, ANALYSIS_DIR / "molecular_property_trends.png")


def _plot_ring_grid(df: pd.DataFrame) -> None:
    metrics = [
        ("ring_count_rdkit_mean", "Ring count"),
        ("aromatic_ring_mean", "Aromatic rings"),
        ("benzene_ring_count_mean", "Benzene rings"),
        ("max_fused_ring_system_size_mean", "Max fused system"),
        ("large_ring_count_mean", "Large rings >=7"),
        ("ring_pathology_flag_mean", "Pathology fraction"),
    ]
    _line_grid(df, metrics, ANALYSIS_DIR / "ring_and_scaffold_trends.png")


def _line_grid(df: pd.DataFrame, metrics: list[tuple[str, str]], path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    axes = axes.flatten()
    for ax, (metric, title) in zip(axes, metrics):
        if metric not in df.columns:
            ax.axis("off")
            continue
        for reward, sub in df.groupby("reward", sort=False):
            sub = sub.sort_values("epoch")
            ax.plot(sub["epoch"], sub[metric], marker="o", linewidth=1.5, markersize=3, label=reward)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_final_bar_grid(final: pd.DataFrame) -> None:
    metrics = [
        ("rdkit_valid_ratio", "Final valid ratio"),
        ("avg_docking", "Final docking"),
        ("carbon_fraction", "Final carbon fraction"),
        ("tpsa_mean", "Final TPSA"),
        ("ring_pathology_flag_mean", "Final pathology fraction"),
        ("descriptor_mmd_vs_baseline", "Final MMD"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    axes = axes.flatten()
    plot_df = final.copy()
    for ax, (metric, title) in zip(axes, metrics):
        if metric not in plot_df.columns:
            ax.axis("off")
            continue
        ax.bar(plot_df["reward"], plot_df[metric], color="#4c78a8")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(ANALYSIS_DIR / "final_epoch_bar_comparison.png", bbox_inches="tight")
    plt.close(fig)


def _plot_delta_heatmap(deltas: pd.DataFrame) -> None:
    metric_roots = [
        "rdkit_valid_ratio",
        "avg_docking",
        "logp_mean",
        "tpsa_mean",
        "carbon_fraction",
        "hetero_atom_mean",
        "heavy_atom_count_mean",
        "ring_pathology_flag_mean",
        "descriptor_mmd_vs_baseline",
    ]
    values = []
    row_labels = []
    for _, row in deltas.iterrows():
        row_labels.append(row["reward"])
        values.append([_safe_float(row.get(f"{metric}_delta")) for metric in metric_roots])
    arr = np.asarray(values, dtype=float)
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    im = ax.imshow(arr, aspect="auto", cmap="coolwarm")
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.set_xticks(range(len(metric_roots)), metric_roots, rotation=45, ha="right")
    ax.set_title("Epoch 10 - Epoch 1 change by single reward")
    fig.colorbar(im, ax=ax, shrink=0.8)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.savefig(ANALYSIS_DIR / "epoch_delta_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def _build_report(merged: pd.DataFrame, final: pd.DataFrame, deltas: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.extend(
        [
            "# DiffSBDD 单奖励矩阵九组训练深度分析",
            "",
            "## 1. 实验目的",
            "",
            "本次九组训练的核心目的不是直接寻找最终最优 reward，而是做单因素诊断：固定 DiffSBDD 强化学习参数，每次只开启一个奖励函数，观察它在 10 个 epoch 内会把分子性质、有效率、docking、分子大小、环结构和分布漂移推向什么方向。",
            "",
            "统一训练设置：200 步扩散，8 GPU，每卡 18 分子，全局 batch 144，PPO 复用 2 次，10 epoch，学习率 2e-5，KL 系数 0.05，`kl_mode=analytic_current_ref`，GoodMoleculePool 与 ChemistryGuard 关闭。",
            "",
            "## 2. 输出文件",
            "",
            f"- 长表：`{_rel(ANALYSIS_DIR / 'merged_epoch_metrics.csv')}`",
            f"- 终点比较：`{_rel(ANALYSIS_DIR / 'final_epoch_comparison.csv')}`",
            f"- 变化量比较：`{_rel(ANALYSIS_DIR / 'epoch10_minus_epoch1_delta.csv')}`",
            f"- 训练曲线总览：`{_rel(ANALYSIS_DIR / 'training_curves_overview.png')}`",
            f"- 分子性质趋势：`{_rel(ANALYSIS_DIR / 'molecular_property_trends.png')}`",
            f"- 环与 scaffold 趋势：`{_rel(ANALYSIS_DIR / 'ring_and_scaffold_trends.png')}`",
            f"- 终点柱状对比：`{_rel(ANALYSIS_DIR / 'final_epoch_bar_comparison.png')}`",
            f"- epoch10-epoch1 热图：`{_rel(ANALYSIS_DIR / 'epoch_delta_heatmap.png')}`",
            f"- 统一离线 docking 每分子结果：`{_rel(OFFLINE_DOCKING_DIR / 'offline_docking_per_molecule.csv')}`",
            f"- 统一离线 docking 每 epoch 汇总：`{_rel(OFFLINE_DOCKING_DIR / 'offline_docking_by_epoch.csv')}`",
            f"- 统一离线 docking 趋势图：`{_rel(OFFLINE_DOCKING_DIR / 'offline_docking_trends.png')}`",
            "",
            "![training curves](" + _rel_from_work_record(ANALYSIS_DIR / "training_curves_overview.png") + ")",
            "",
            "![property trends](" + _rel_from_work_record(ANALYSIS_DIR / "molecular_property_trends.png") + ")",
            "",
            "![ring trends](" + _rel_from_work_record(ANALYSIS_DIR / "ring_and_scaffold_trends.png") + ")",
            "",
            "![final comparison](" + _rel_from_work_record(ANALYSIS_DIR / "final_epoch_bar_comparison.png") + ")",
            "",
            "![delta heatmap](" + _rel_from_work_record(ANALYSIS_DIR / "epoch_delta_heatmap.png") + ")",
            "",
            "## 3. 终点总表",
            "",
            _markdown_table(_format_final_table(final)),
            "",
            "## 4. epoch 10 相对 epoch 1 的变化",
            "",
            _markdown_table(_format_delta_table(deltas)),
            "",
            "## 5. 关键结论",
            "",
        ]
    )
    lines.extend(_key_findings(final, deltas))
    lines.extend(["", "## 6. 分组分析", ""])
    for spec in RUNS:
        lines.extend(_run_analysis(spec.reward, merged, final, deltas))
    lines.extend(
        [
            "",
            "## 7. 对下一步 reward 设计的启发",
            "",
            "1. 不建议直接把 docking、classifier、分布约束、guard 全部混在一起训练。九组结果显示，每个单奖励都会带来不同副作用；混合后很难判断是哪个变量造成漂移。",
            "2. DiffSBDD 的核心风险仍然是底座合法性和重建链条：多数组的 RDKit valid ratio 只有约 0.5-0.7，fragmented molecule 比例不低。奖励函数可以改变分布，但不能完全修复点云到键重建的问题。",
            "3. 对最终组合 reward，建议每轮只增加一个模块：先确定 docking 或 ligand efficiency 作为性能驱动，再单独加一个分布约束或结构约束，观察是否牺牲有效率和 docking。",
            "4. 如果目标是防止全碳/疏水漂移，优先考虑 `logp_distribution` 或 `tpsa_distribution` 这类分布约束，而不是只靠 classifier。classifier 更适合作为后期人工经验过滤或弱辅助，不适合作为唯一主 reward。",
            "5. 如果目标是保持药物样和可合成性，QED/SA 的方向相对温和，但它们对靶点 docking 的直接贡献弱，需要和性能项组合验证。",
            "6. KL=0.05 在本轮未表现为主要训练瓶颈：`KL_Penalty_Weighted` 数值整体较小，说明当前更大的问题是 reward 对分子分布的选择压力和底座重建质量，而不是 KL 过强压住训练。",
            "",
            "## 8. 建议的下一轮最小组合测试",
            "",
            "| 组合 | 目的 | 预期观察 |",
            "|---|---|---|",
            "| docking + logp_distribution | 测试能否保 docking 同时压疏水漂移 | docking 是否下降、logP/carbon 是否回落 |",
            "| docking + tpsa_distribution | 测试能否保 docking 同时维持极性 | TPSA/HBA/HBD 是否稳定，有效率是否下降 |",
            "| ligand_efficiency + logp_distribution | 测试是否避免大分子和高疏水 shortcut | heavy atoms、logP、docking/LE 三者关系 |",
            "| docking + chembl_prior_ecfp4 | 测试结构先验是否改善可用骨架 | ChEMBL 相似度是否提升、fragment 是否改善 |",
            "| docking + classifier | 只作为人工经验方向验证 | 是否出现 classifier reward hacking 或误杀正常分子 |",
            "",
            "下一轮仍建议每次只改变一个变量，例如先固定 docking 权重，再只加入一个约束模块，避免训练后无法归因。",
        ]
    )
    return "\n".join(lines) + "\n"


def _format_final_table(final: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "reward": "reward",
        "rdkit_valid_ratio": "valid",
        "fragmented_ratio": "fragment",
        "avg_docking": "dock",
        "avg_ligand_efficiency": "LE",
        "logp_mean": "logP",
        "tpsa_mean": "TPSA",
        "carbon_fraction": "C_frac",
        "hetero_atom_mean": "hetero",
        "heavy_atom_count_mean": "heavy",
        "ring_count_rdkit_mean": "rings",
        "benzene_ring_count_mean": "benzene",
        "ring_pathology_flag_mean": "pathology",
        "descriptor_mmd_vs_baseline": "MMD",
    }
    out = final[[c for c in cols if c in final.columns]].rename(columns=cols)
    return _round_df(out)


def _format_delta_table(deltas: pd.DataFrame) -> pd.DataFrame:
    selected = {
        "reward": "reward",
        "rdkit_valid_ratio_delta": "valid_d",
        "avg_docking_delta": "dock_d",
        "logp_mean_delta": "logP_d",
        "tpsa_mean_delta": "TPSA_d",
        "carbon_fraction_delta": "C_frac_d",
        "heavy_atom_count_mean_delta": "heavy_d",
        "ring_pathology_flag_mean_delta": "pathology_d",
        "descriptor_mmd_vs_baseline_delta": "MMD_d",
    }
    out = deltas[[c for c in selected if c in deltas.columns]].rename(columns=selected)
    return _round_df(out)


def _key_findings(final: pd.DataFrame, deltas: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    best_valid = final.sort_values("rdkit_valid_ratio", ascending=False).iloc[0]
    worst_valid = final.sort_values("rdkit_valid_ratio", ascending=True).iloc[0]
    best_dock = final.sort_values("avg_docking", ascending=True).iloc[0]
    low_path = final.sort_values("ring_pathology_flag_mean", ascending=True).iloc[0]
    high_path = final.sort_values("ring_pathology_flag_mean", ascending=False).iloc[0]
    low_mmd = final.sort_values("descriptor_mmd_vs_baseline", ascending=True).iloc[0]
    lines.append(
        f"- 有效率最高的是 `{best_valid['reward']}`，epoch10 RDKit valid ratio={best_valid['rdkit_valid_ratio']:.3f}；最低的是 `{worst_valid['reward']}`，valid ratio={worst_valid['rdkit_valid_ratio']:.3f}。"
    )
    lines.append(
        "- 训练时只有 `docking` 和 `ligand_efficiency` 两组启用了 docking provider；本报告已额外对 90 个 epoch SDF 做统一离线 UniDock，因此终点表里的 docking/LE 对九组都可横向比较。"
    )
    lines.append(
        f"- 统一离线 docking 终点最强的是 `{best_dock['reward']}`，avg_docking={best_dock['avg_docking']:.3f}；但需要同时看 fragment、carbon、ring pathology，避免只优化对接分数。"
    )
    lines.append(
        f"- 环病态比例最低的是 `{low_path['reward']}`，pathology fraction={low_path['ring_pathology_flag_mean']:.3f}；最高的是 `{high_path['reward']}`，pathology fraction={high_path['ring_pathology_flag_mean']:.3f}。"
    )
    lines.append(
        f"- 与 ChEMBL 6D 描述符分布最接近的是 `{low_mmd['reward']}`，descriptor MMD={low_mmd['descriptor_mmd_vs_baseline']:.3f}。这类指标比单一 reward mean 更能反映分子性质是否整体跑偏。"
    )
    carbon_max = deltas.sort_values("carbon_fraction_delta", ascending=False).iloc[0]
    carbon_min = deltas.sort_values("carbon_fraction_delta", ascending=True).iloc[0]
    lines.append(
        f"- 本轮没有观察到高碳漂移：9 组 carbon_fraction 从 epoch1 到 epoch10 均未上升；变化最小的是 `{carbon_max['reward']}` ({carbon_max['carbon_fraction_delta']:+.3f})，下降最多的是 `{carbon_min['reward']}` ({carbon_min['carbon_fraction_delta']:+.3f})。"
    )
    tpsa_min = deltas.sort_values("tpsa_mean_delta", ascending=True).iloc[0]
    tpsa_max = deltas.sort_values("tpsa_mean_delta", ascending=False).iloc[0]
    lines.append(
        f"- TPSA 整体没有坍缩：除 `{tpsa_min['reward']}` 基本持平 ({tpsa_min['tpsa_mean_delta']:+.2f}) 外，多数组上升；上升最多的是 `{tpsa_max['reward']}` ({tpsa_max['tpsa_mean_delta']:+.2f})。"
    )
    return lines


def _run_analysis(reward: str, merged: pd.DataFrame, final: pd.DataFrame, deltas: pd.DataFrame) -> list[str]:
    sub = merged[merged["reward"] == reward].sort_values("epoch")
    fin = final[final["reward"] == reward].iloc[0]
    delta = deltas[deltas["reward"] == reward].iloc[0]
    lines = [f"### {reward}", ""]
    dock_text = _fmt(fin.get("avg_docking"))
    le_text = _fmt(fin.get("avg_ligand_efficiency"))
    lines.append(
        f"- 终点：valid={_fmt(fin.get('rdkit_valid_ratio'))}，fragment={_fmt(fin.get('fragmented_ratio'))}，dock={dock_text}，LE={le_text}。"
    )
    lines.append(
        f"- 分子性质终点：logP={_fmt(fin.get('logp_mean'))}，TPSA={_fmt(fin.get('tpsa_mean'))}，carbon={_fmt(fin.get('carbon_fraction'))}，hetero={_fmt(fin.get('hetero_atom_mean'))}，heavy={_fmt(fin.get('heavy_atom_count_mean'))}。"
    )
    lines.append(
        f"- 环结构终点：rings={_fmt(fin.get('ring_count_rdkit_mean'))}，benzene={_fmt(fin.get('benzene_ring_count_mean'))}，max fused={_fmt(fin.get('max_fused_ring_system_size_mean'))}，pathology={_fmt(fin.get('ring_pathology_flag_mean'))}。"
    )
    lines.append(
        f"- 训练变化：valid { _signed(delta.get('rdkit_valid_ratio_delta')) }，logP { _signed(delta.get('logp_mean_delta')) }，TPSA { _signed(delta.get('tpsa_mean_delta')) }，carbon { _signed(delta.get('carbon_fraction_delta')) }，pathology { _signed(delta.get('ring_pathology_flag_mean_delta')) }。"
    )
    lines.extend(_interpret_reward(reward, fin, delta, sub))
    lines.append("")
    return lines


def _interpret_reward(reward: str, fin: pd.Series, delta: pd.Series, sub: pd.DataFrame) -> list[str]:
    text: dict[str, str] = {
        "docking": "docking 单独训练最能回答性能上限，但它只关心对接分数，容易忽视分子合法性、极性和结构复杂度。若 docking 改善同时 carbon/pathology 上升，就说明它在利用疏水或复杂结构 shortcut。",
        "qed": "QED 是通用 drug-like 压力，通常会倾向中等大小、较温和的极性和疏水性。它适合作为稳定化项，但不直接保证靶点适配。",
        "sa": "SA 主要约束合成可及性，理论上会压过复杂结构；如果 valid 或 docking 明显下降，说明它可能把生成方向推得过于保守或简单。",
        "tpsa_distribution": "TPSA distribution 是极性分布约束，不是 TPSA 越高越好。它的价值在于防止极性坍缩或低 TPSA 漂移，但单独使用时可能缺少靶点选择压力。",
        "ecfp4_prior": "ECFP4 prior 更偏结构骨架相似性，能把分子拉回 ChEMBL 常见结构空间；但如果只用它，可能优化相似度而不是靶点性能。",
        "fcfp4_prior": "FCFP4 prior 更偏功能团/药效团环境，相比 ECFP4 更关注原子类型和局部化学特征；它是否保护杂原子和 TPSA 是重点。",
        "logp_distribution": "LogP distribution 直接约束疏水性，是防全碳/高疏水漂移的重要候选。单独使用时要看是否牺牲 docking 和有效率。",
        "ligand_efficiency": "Ligand efficiency 用 docking/重原子数约束大分子 shortcut。它比 docking 更贴近实际候选优化，但单独使用可能偏向小分子或对失败样本过敏。",
        "chembl_classifier": "classifier 是人工经验约束的代理。它能否作为 reward，取决于是否减少坏分子而不误杀正常结构；单独训练尤其要警惕 classifier reward hacking。",
    }
    lines = [f"- 解读：{text.get(reward, '')}"]
    if _safe_float(delta.get("carbon_fraction_delta")) > 0.03:
        lines.append("- 风险：碳比例明显上升，提示可能朝高碳/疏水方向漂移。")
    if _safe_float(delta.get("tpsa_mean_delta")) < -10:
        lines.append("- 风险：TPSA 明显下降，提示极性损失。")
    if _safe_float(fin.get("ring_pathology_flag_mean")) > 0.50:
        lines.append("- 风险：终点仍有较高环病态比例，单独作为结构质量控制不足。")
    if _safe_float(fin.get("rdkit_valid_ratio")) < 0.60:
        lines.append("- 风险：valid ratio 偏低，说明该 reward 不能单独解决重建合法性。")
    return lines


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    columns = [str(col) for col in df.columns]
    rows = []
    for _, row in df.iterrows():
        rows.append([_md_cell(row[col]) for col in df.columns])
    widths = [
        max(len(columns[idx]), *(len(row[idx]) for row in rows))
        for idx in range(len(columns))
    ]
    header = "| " + " | ".join(columns[idx].ljust(widths[idx]) for idx in range(len(columns))) + " |"
    sep = "| " + " | ".join("-" * widths[idx] for idx in range(len(columns))) + " |"
    body = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(columns))) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def _md_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def _round_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else round(float(x), 3))
    return out


def _fmt(value: Any) -> str:
    value = _safe_float(value)
    return "NA" if math.isnan(value) else f"{value:.3f}"


def _signed(value: Any) -> str:
    value = _safe_float(value)
    return "NA" if math.isnan(value) else f"{value:+.3f}"


def _safe_float(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _rel_from_work_record(path: Path) -> str:
    return os.path.relpath(path, WORK_RECORD.parent)


if __name__ == "__main__":
    main()
