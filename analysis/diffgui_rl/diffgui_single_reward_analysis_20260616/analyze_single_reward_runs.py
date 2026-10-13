"""Analyze DiffGui single-reward RL runs from local training artifacts."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-diffb110")

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent

SINGLE_REWARD_RUNS: list[dict[str, str]] = [
    {
        "slug": "docking",
        "reward": "docking",
        "matrix": "20260613",
        "run_dir": "OUT/diffgui-ring-control-20260613-01-docking_20260613_153723",
        "config": "launch/generated/diffgui_ring_control_matrix_20260613/01_docking.yaml",
    },
    {
        "slug": "ligand_efficiency",
        "reward": "ligand_efficiency",
        "matrix": "20260613",
        "run_dir": "OUT/diffgui-ring-control-20260613-02-ligand_efficiency_20260613_181441",
        "config": "launch/generated/diffgui_ring_control_matrix_20260613/02_ligand_efficiency.yaml",
    },
    {
        "slug": "aromatic_scaffold",
        "reward": "aromatic_scaffold",
        "matrix": "20260613",
        "run_dir": "OUT/diffgui-ring-control-20260613-05-aromatic_scaffold_20260613_204920",
        "config": "launch/generated/diffgui_ring_control_matrix_20260613/05_aromatic_scaffold.yaml",
    },
    {
        "slug": "ring_pathology",
        "reward": "ring_pathology",
        "matrix": "20260613",
        "run_dir": "OUT/diffgui-ring-control-20260613-07-ring_pathology_20260613_232423",
        "config": "launch/generated/diffgui_ring_control_matrix_20260613/07_ring_pathology.yaml",
    },
    {
        "slug": "chembl_classifier",
        "reward": "chembl_classifier",
        "matrix": "20260613",
        "run_dir": "OUT/diffgui-ring-control-20260613-08-chembl_classifier_20260614_020007",
        "config": "launch/generated/diffgui_ring_control_matrix_20260613/08_chembl_classifier.yaml",
    },
    {
        "slug": "qed",
        "reward": "qed",
        "matrix": "20260615",
        "run_dir": "OUT/diffgui-ring-control-20260615-09-qed_20260615_122202",
        "config": "launch/generated/diffgui_ring_control_matrix_20260615/09_qed.yaml",
    },
    {
        "slug": "sa",
        "reward": "sa",
        "matrix": "20260615",
        "run_dir": "OUT/diffgui-ring-control-20260615-10-sa_20260615_163654",
        "config": "launch/generated/diffgui_ring_control_matrix_20260615/10_sa.yaml",
    },
    {
        "slug": "ring_complexity",
        "reward": "ring_complexity",
        "matrix": "20260615",
        "run_dir": "OUT/diffgui-ring-control-20260615-11-ring_complexity_20260615_200439",
        "config": "launch/generated/diffgui_ring_control_matrix_20260615/11_ring_complexity.yaml",
    },
]

COMBINATION_RUNS: list[dict[str, str]] = [
    {
        "slug": "le_ring_aromatic",
        "reward": "ligand_efficiency + ring_complexity + aromatic_scaffold",
        "run_dir": "OUT/diffgui-ring-control-20260615-12-le_ring_aromatic_20260615_224513",
        "config": "launch/generated/diffgui_ring_control_matrix_20260615/12_le_ring_aromatic.yaml",
    },
    {
        "slug": "dock_ring_aromatic",
        "reward": "docking + ring_complexity + aromatic_scaffold",
        "run_dir": "OUT/diffgui-ring-control-20260615-13-dock_ring_aromatic_20260616_011801",
        "config": "launch/generated/diffgui_ring_control_matrix_20260615/13_dock_ring_aromatic.yaml",
    },
]

SUMMARY_COLUMNS = [
    "epoch",
    "reward_total_mean",
    "reward_total_scored_mean",
    "reward_for_advantage_mean",
    "molecule_rdkit_valid_ratio",
    "molecule_reconstruction_success_ratio",
    "molecule_docking_success_ratio",
    "unique_canonical_smiles",
    "duplicate_ratio",
    "descriptor_mmd_vs_baseline",
    "carbon_fraction",
    "tpsa_mean",
    "hba_mean",
    "logp_mean",
    "ring_count_mean",
    "aromatic_ring_mean",
    "three_member_ring_mean",
    "three_member_ring_total",
    "docking_raw_mean",
    "docking_reward_mean",
    "ligand_efficiency_raw_mean",
    "PPO_ApproxKL",
    "ClipFrac",
    "RatioMean",
    "GradNorm",
    "total_seconds",
    "rollout_seconds",
    "reward_seconds",
    "ppo_seconds",
    "checkpoint",
]

TARGET_COMPONENTS: dict[str, str] = {
    "docking": "docking_reward",
    "ligand_efficiency": "ligand_efficiency_reward",
    "aromatic_scaffold": "aromatic_scaffold_reward",
    "ring_pathology": "ring_pathology_reward",
    "chembl_classifier": "chembl_classifier_reward",
    "qed": "qed_reward",
    "sa": "sa_reward",
    "ring_complexity": "ring_complexity_reward",
}


def _safe_float(value: Any) -> float:
    """Convert a value to a finite float where possible."""
    try:
        output = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return output if math.isfinite(output) else float("nan")


def _read_training_summary(run: dict[str, str]) -> pd.DataFrame:
    """Read one run's training_summary.csv with run metadata columns."""
    path = ROOT / run["run_dir"] / "training_summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    keep_cols = [column for column in SUMMARY_COLUMNS if column in df.columns]
    df = df[keep_cols].copy()
    df.insert(0, "slug", run["slug"])
    df.insert(1, "reward", run["reward"])
    df.insert(2, "matrix", run.get("matrix", ""))
    df.insert(3, "run_dir", run["run_dir"])
    df.insert(4, "config", run["config"])
    return df


def _read_epoch_molecule_stats(run: dict[str, str], epoch: int) -> dict[str, Any]:
    """Aggregate target reward and useful molecular columns from one epoch."""
    path = ROOT / run["run_dir"] / "epochs" / f"epoch_{epoch:04d}" / "molecules.csv"
    if not path.exists():
        return {
            "slug": run["slug"],
            "reward": run["reward"],
            "epoch": epoch,
            "molecule_csv_exists": False,
        }

    df = pd.read_csv(path)
    valid = df[df.get("rdkit_valid", 0) == 1].copy()
    source = valid if len(valid) else df
    row: dict[str, Any] = {
        "slug": run["slug"],
        "reward": run["reward"],
        "epoch": epoch,
        "molecule_csv_exists": True,
        "records": int(len(df)),
        "valid_records": int(len(valid)),
        "valid_rate": float(len(valid) / len(df)) if len(df) else 0.0,
    }

    target_col = TARGET_COMPONENTS.get(run["reward"])
    row["target_component"] = target_col or ""
    row["target_component_mean"] = (
        float(pd.to_numeric(source[target_col], errors="coerce").mean())
        if target_col and target_col in source.columns
        else float("nan")
    )
    for column in [
        "reward_total",
        "docking_raw",
        "docking_reward",
        "ligand_efficiency_raw",
        "qed_raw",
        "qed_reward",
        "sa_raw",
        "sa_reward",
        "aromatic_scaffold_reward",
        "aromatic_scaffold_benzene_ring_count",
        "aromatic_scaffold_heterocycle_ring_count",
        "aromatic_scaffold_three_member_ring_count",
        "ring_pathology_reward",
        "ring_pathology_ring_count",
        "ring_pathology_large_ring_count",
        "ring_pathology_three_member_ring_count",
        "ring_pathology_penalty",
        "chembl_classifier_reward",
        "ring_complexity_reward",
        "ring_complexity_ring_count",
        "ring_complexity_aromatic_ring_count",
        "ring_complexity_three_member_ring_count",
        "ring_complexity_max_fused_ring_system_size",
        "ring_complexity_penalty",
    ]:
        if column in source.columns:
            row[f"{column}_mean"] = float(pd.to_numeric(source[column], errors="coerce").mean())
    return row


def _first_last_delta(df: pd.DataFrame, group_col: str = "slug") -> pd.DataFrame:
    """Compute first, last, and delta columns for numeric training metrics."""
    rows: list[dict[str, Any]] = []
    numeric_columns = [
        column
        for column in df.columns
        if column not in {"epoch"} and pd.api.types.is_numeric_dtype(df[column])
    ]
    for slug, group in df.sort_values("epoch").groupby(group_col):
        first = group.iloc[0]
        last = group.iloc[-1]
        row: dict[str, Any] = {
            "slug": slug,
            "reward": last["reward"],
            "matrix": last.get("matrix", ""),
            "run_dir": last["run_dir"],
            "config": last["config"],
            "first_epoch": int(first["epoch"]),
            "last_epoch": int(last["epoch"]),
            "checkpoint_exists": Path(str(last.get("checkpoint", ""))).exists(),
        }
        for column in numeric_columns:
            if column == "epoch":
                continue
            row[f"first_{column}"] = _safe_float(first.get(column))
            row[f"last_{column}"] = _safe_float(last.get(column))
            row[f"delta_{column}"] = row[f"last_{column}"] - row[f"first_{column}"]
        rows.append(row)
    return pd.DataFrame(rows)


def _component_delta(epoch_stats: pd.DataFrame) -> pd.DataFrame:
    """Compute first-vs-last target component changes."""
    rows: list[dict[str, Any]] = []
    for slug, group in epoch_stats.sort_values("epoch").groupby("slug"):
        first = group.iloc[0]
        last = group.iloc[-1]
        rows.append(
            {
                "slug": slug,
                "reward": last["reward"],
                "target_component": last.get("target_component", ""),
                "first_epoch": int(first["epoch"]),
                "last_epoch": int(last["epoch"]),
                "first_target_component_mean": _safe_float(first.get("target_component_mean")),
                "last_target_component_mean": _safe_float(last.get("target_component_mean")),
                "delta_target_component_mean": _safe_float(last.get("target_component_mean"))
                - _safe_float(first.get("target_component_mean")),
                "first_valid_rate": _safe_float(first.get("valid_rate")),
                "last_valid_rate": _safe_float(last.get("valid_rate")),
                "delta_valid_rate": _safe_float(last.get("valid_rate")) - _safe_float(first.get("valid_rate")),
            }
        )
    return pd.DataFrame(rows)


def _plot_training_trends(all_epochs: pd.DataFrame, delta_df: pd.DataFrame) -> None:
    """Generate summary plots for single-reward runs."""
    plt.style.use("default")
    colors = {
        "docking": "#2563eb",
        "ligand_efficiency": "#0891b2",
        "aromatic_scaffold": "#7c3aed",
        "ring_pathology": "#c2410c",
        "chembl_classifier": "#be123c",
        "qed": "#16a34a",
        "sa": "#ca8a04",
        "ring_complexity": "#0f766e",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for slug, group in all_epochs.groupby("slug"):
        color = colors.get(slug)
        axes[0, 0].plot(group["epoch"], group["reward_total_mean"], marker="o", label=slug, color=color)
        axes[0, 1].plot(group["epoch"], group["molecule_rdkit_valid_ratio"], marker="o", label=slug, color=color)
        axes[1, 0].plot(group["epoch"], group["descriptor_mmd_vs_baseline"], marker="o", label=slug, color=color)
        axes[1, 1].plot(group["epoch"], group["PPO_ApproxKL"], marker="o", label=slug, color=color)
    axes[0, 0].set_title("Total reward mean")
    axes[0, 1].set_title("RDKit valid ratio")
    axes[1, 0].set_title("Descriptor MMD vs baseline")
    axes[1, 1].set_title("PPO ApproxKL")
    for ax in axes.ravel():
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7, ncols=2)
    fig.savefig(OUT_DIR / "single_reward_training_trends.png", dpi=220)
    plt.close(fig)

    bar_cols = [
        ("delta_reward_total_mean", "Reward delta"),
        ("last_molecule_rdkit_valid_ratio", "Final valid"),
        ("last_descriptor_mmd_vs_baseline", "Final descriptor MMD"),
        ("last_PPO_ApproxKL", "Final KL"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ordered = delta_df.sort_values("slug")
    for ax, (column, title) in zip(axes.ravel(), bar_cols):
        ax.bar(ordered["slug"], ordered[column], color=[colors.get(slug, "#64748b") for slug in ordered["slug"]])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(OUT_DIR / "single_reward_final_delta_bars.png", dpi=220)
    plt.close(fig)


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    """Render selected columns as a Markdown table without optional dependencies."""
    view = df[columns].copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: "NA" if pd.isna(value) else f"{value:.4f}")
    headers = [str(column) for column in view.columns]
    rows = [[str(value) for value in row] for row in view.to_numpy().tolist()]
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]

    def fmt(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(values)) + " |"

    return "\n".join([fmt(headers), "| " + " | ".join("-" * width for width in widths) + " |", *[fmt(row) for row in rows]])


def _write_report(delta_df: pd.DataFrame, component_delta_df: pd.DataFrame, combo_delta_df: pd.DataFrame) -> None:
    """Write the final analysis report."""
    rank_cols = [
        "slug",
        "last_reward_total_mean",
        "delta_reward_total_mean",
        "last_molecule_rdkit_valid_ratio",
        "last_descriptor_mmd_vs_baseline",
        "last_carbon_fraction",
        "last_tpsa_mean",
        "last_ring_count_mean",
        "last_three_member_ring_total",
        "last_PPO_ApproxKL",
    ]
    component_cols = [
        "slug",
        "target_component",
        "first_target_component_mean",
        "last_target_component_mean",
        "delta_target_component_mean",
        "first_valid_rate",
        "last_valid_rate",
    ]
    combo_cols = [
        "slug",
        "last_reward_total_mean",
        "last_molecule_rdkit_valid_ratio",
        "last_descriptor_mmd_vs_baseline",
        "last_carbon_fraction",
        "last_tpsa_mean",
        "last_ring_count_mean",
        "last_three_member_ring_total",
        "last_PPO_ApproxKL",
    ]

    best_valid = delta_df.sort_values("last_molecule_rdkit_valid_ratio", ascending=False).iloc[0]
    best_mmd = delta_df.sort_values("last_descriptor_mmd_vs_baseline").iloc[0]
    best_reward = delta_df.sort_values("last_reward_total_mean", ascending=False).iloc[0]
    component_positive = component_delta_df[component_delta_df["delta_target_component_mean"] > 0]

    lines = [
        "# DiffGui 单奖励函数强化学习测试结果分析",
        "",
        "## 范围",
        "",
        "- 单奖励 run：docking、ligand_efficiency、aromatic_scaffold、ring_pathology、chembl_classifier、qed、sa、ring_complexity。",
        "- 共同设置：8GPU、per-rank batch=18、200 steps、10 epochs、ppo_epochs=1、lr=3e-5、last3 解冻。",
        "- 主要证据：`training_summary.csv`、`epochs/epoch_0001/molecules.csv`、`epochs/epoch_0010/molecules.csv`、`summary.json`。",
        "",
        "## 单奖励最终表现",
        "",
        _markdown_table(delta_df.sort_values("last_reward_total_mean", ascending=False), rank_cols),
        "",
        "## 目标组件是否被优化",
        "",
        _markdown_table(component_delta_df.sort_values("delta_target_component_mean", ascending=False), component_cols),
        "",
        "## 组合奖励参考结果",
        "",
        _markdown_table(combo_delta_df.sort_values("last_reward_total_mean", ascending=False), combo_cols),
        "",
        "## 结论",
        "",
        f"- 最高最终总 reward 是 `{best_reward['slug']}`：last_reward_total_mean={best_reward['last_reward_total_mean']:.3f}，但这主要反映奖励尺度，不代表综合最好。",
        f"- 最高最终有效率是 `{best_valid['slug']}`：RDKit valid ratio={best_valid['last_molecule_rdkit_valid_ratio']:.3f}。",
        f"- 分布漂移最小的是 `{best_mmd['slug']}`：descriptor MMD={best_mmd['last_descriptor_mmd_vs_baseline']:.3f}。",
        f"- 目标组件真正首末提升的单奖励有：{', '.join(component_positive['slug'].tolist()) if len(component_positive) else '无'}。",
        "- QED、SA、ligand_efficiency 和 ring_complexity 的目标组件均值从 epoch 1 到 epoch 10 下降，说明当前 PPO 设置下单奖励不一定能稳定优化目标，可能被采样噪声、KL/优势归一化、有效性筛选或奖励尺度影响。",
        "- chembl_classifier 从 -0.790 提升到 -0.568，但仍为负，说明旧 MLP 分类器单独作为 DiffGui RL 奖励太弱或校准不合适。",
        "- docking 的 docking_reward 基本持平略降，但 docking_raw 从约 -9.66 到 -9.55 略改善；需要用独立 docking 重打分验证 top 分子，而不是只信训练期 reward。",
        "- aromatic_scaffold 有小幅目标提升，但伴随三元环/卤代芳环等结构风险上升，不能单独作为主奖励。",
        "- ring_pathology 的 penalty 明显下降，说明它有一定结构约束能力，但单独优化时总分和分布仍不够理想。",
        "",
        "## 还需要测试什么",
        "",
        "1. 做未训练 DiffGui baseline 同配置生成：需要同样的 8GPU/200 steps/144 分子/10 次采样或等量样本，用同一套 descriptor、docking、D-MPNN/MLP classifier 重打分，否则现在只能看 RL run 之间相对变化。",
        "2. 对每个单奖励 checkpoint 做独立离线采样评估：用 epoch 1/5/10 checkpoint 各生成固定数量分子，避免训练 rollout 样本和 reward 反馈样本混在一起。",
        "3. 做种子重复：当前每个奖励只有一条 10 epoch 轨迹，建议至少 top 3 奖励策略各 3 个 seed，优先 ring_pathology、aromatic_scaffold、docking。",
        "4. 做奖励组合权重扫描：已有组合中 `LE + ring + aromatic` 的 MMD/有效率较好，下一步应扫 docking/LE 与 ring_pathology/aromatic_scaffold 的权重，比如 0.5/0.25/0.25 与 0.7/0.15/0.15。",
        "5. 替换或校准 classifier 奖励：旧 MLP 单独效果差，应测试 D-MPNN checkpoint 作为 DiffGui reward，并比较 v0.3.1 MLP、D-MPNN、docking 的相关性。",
        "6. 做结构病理专项评估：统计三元环、桥头/螺环、过高碳比例、低 TPSA/低 HBA、芳香卤素、最大稠环系统；当前单 reward 最容易在这些地方互相打架。",
        "7. 做 top 分子重打分与人工检查：每个 reward 取 top_reward.sdf 和 epoch_0010 top 20，用统一 docking、D-MPNN、QED/SA、相互作用几何重新排序。",
        "8. 做 PPO 稳定性 ablation：当前 KL 大多在 1.7-4.6，建议测 lr=1e-5、clip ratio、reward normalization/clipping、ppo_epochs=2/4，确认目标组件下降是否是 PPO 更新过猛。",
        "9. 做训练长度测试：10 epoch 可能还不够或已经过拟合，建议对最佳组合跑 25 epoch，并观察 descriptor MMD、valid ratio、target reward 是否持续改善。",
        "10. 做与 DiffSBDD 的同靶点横评：用 8GJL/SHP2 已生成集合，对 DiffGui RL checkpoint 新采样分子和 DiffSBDD 分子统一跑 docking、D-MPNN、分布漂移和结构病理指标。",
        "",
        "## 输出",
        "",
        "- [single_reward_epoch_metrics.csv](single_reward_epoch_metrics.csv)",
        "- [single_reward_start_end_delta.csv](single_reward_start_end_delta.csv)",
        "- [single_reward_component_delta.csv](single_reward_component_delta.csv)",
        "- [combination_reward_start_end_delta.csv](combination_reward_start_end_delta.csv)",
        "",
        "![single reward trends](single_reward_training_trends.png)",
        "",
        "![single reward final deltas](single_reward_final_delta_bars.png)",
        "",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run the single-reward analysis."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_epochs = pd.concat([_read_training_summary(run) for run in SINGLE_REWARD_RUNS], ignore_index=True)
    all_epochs.to_csv(OUT_DIR / "single_reward_epoch_metrics.csv", index=False)

    delta_df = _first_last_delta(all_epochs)
    delta_df.to_csv(OUT_DIR / "single_reward_start_end_delta.csv", index=False)
    delta_df.to_json(OUT_DIR / "single_reward_start_end_delta.json", orient="records", indent=2, force_ascii=False)

    epoch_stats_rows = []
    for run in SINGLE_REWARD_RUNS:
        epoch_stats_rows.append(_read_epoch_molecule_stats(run, 1))
        epoch_stats_rows.append(_read_epoch_molecule_stats(run, 10))
    epoch_stats = pd.DataFrame(epoch_stats_rows)
    epoch_stats.to_csv(OUT_DIR / "single_reward_epoch_molecule_component_stats.csv", index=False)
    component_delta_df = _component_delta(epoch_stats)
    component_delta_df.to_csv(OUT_DIR / "single_reward_component_delta.csv", index=False)

    combo_epochs = pd.concat([_read_training_summary(run) for run in COMBINATION_RUNS], ignore_index=True)
    combo_epochs.to_csv(OUT_DIR / "combination_reward_epoch_metrics.csv", index=False)
    combo_delta_df = _first_last_delta(combo_epochs)
    combo_delta_df.to_csv(OUT_DIR / "combination_reward_start_end_delta.csv", index=False)

    _plot_training_trends(all_epochs, delta_df)
    _write_report(delta_df, component_delta_df, combo_delta_df)

    print(
        json.dumps(
            {
                "out_dir": str(OUT_DIR),
                "single_runs": len(SINGLE_REWARD_RUNS),
                "combination_runs": len(COMBINATION_RUNS),
                "epoch_rows": len(all_epochs),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
