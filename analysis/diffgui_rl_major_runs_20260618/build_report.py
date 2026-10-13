"""Summarize major DiffGui RL training runs and plot metric trajectories."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/diffb110_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = PROJECT_ROOT / "OUT"
ANALYSIS_DIR = Path(__file__).resolve().parent
WORK_RECORD_PATH = PROJECT_ROOT / "work_records" / "26-6" / "260618" / "260618-02 DiffGui强化学习主要训练汇总.md"
WORK_RECORD_DIR = WORK_RECORD_PATH.parent

GROUP_ALIAS_PREFIX = {
    "核心 next6 / 环病理组合矩阵": "DG-NEXT",
    "早期正式大训练 / 背景短训": "DG-MAIN",
    "Reward 归因实验 / 单奖励矩阵": "DG-ATTR",
    "旧组合奖励 / 环控制矩阵": "DG-COMB",
}

GROUP_FIGURE_FILES = {
    "核心 next6 / 环病理组合矩阵": {
        "training": "group_next6_training_curves.png",
        "properties": "group_next6_molecule_properties.png",
        "heatmap": "group_next6_final_heatmap.png",
    },
    "早期正式大训练 / 背景短训": {
        "training": "group_early_training_curves.png",
        "properties": "group_early_molecule_properties.png",
        "heatmap": "group_early_final_heatmap.png",
    },
    "Reward 归因实验 / 单奖励矩阵": {
        "training": "group_reward_attribution_training_curves.png",
        "properties": "group_reward_attribution_molecule_properties.png",
        "heatmap": "group_reward_attribution_final_heatmap.png",
    },
    "旧组合奖励 / 环控制矩阵": {
        "training": "group_combination_training_curves.png",
        "properties": "group_combination_molecule_properties.png",
        "heatmap": "group_combination_final_heatmap.png",
    },
}


@dataclass(frozen=True)
class RunSpec:
    """One DiffGui run selected for the major-run review."""

    label: str
    slug: str
    group: str
    run_dir: str
    target_component: str | None = None
    note: str = ""


RUN_SPECS = [
    RunSpec(
        label="dock + aromatic + pathology",
        slug="next6_dock_aromatic_pathology",
        group="核心 next6 / 环病理组合矩阵",
        run_dir="diffgui-next6-20260618-01-dock_aromatic_pathology_20260618_211501",
        target_component=None,
        note="next6 核心矩阵：docking + aromatic_scaffold + ring_pathology，20 epoch。",
    ),
    RunSpec(
        label="LE + aromatic + pathology",
        slug="next6_le_aromatic_pathology",
        group="核心 next6 / 环病理组合矩阵",
        run_dir="diffgui-next6-20260618-02-le_aromatic_pathology_20260619_020218",
        target_component=None,
        note="next6 核心矩阵：ligand_efficiency + aromatic_scaffold + ring_pathology，20 epoch。",
    ),
    RunSpec(
        label="dock + LE + aromatic + pathology",
        slug="next6_dock_le_aromatic_pathology",
        group="核心 next6 / 环病理组合矩阵",
        run_dir="diffgui-next6-20260618-03-dock_le_aromatic_pathology_20260619_064544",
        target_component=None,
        note="next6 核心矩阵：docking + ligand_efficiency + aromatic_scaffold + ring_pathology，20 epoch。",
    ),
    RunSpec(
        label="ring last3 + pathology + guard",
        slug="next6_ring_last3_pathology_guard",
        group="核心 next6 / 环病理组合矩阵",
        run_dir="diffgui-next6-20260618-04-ring_last3_pathology_guard_20260619_112410",
        target_component=None,
        note="next6 核心矩阵：旧 ring last3 组合加 ring_pathology 与 chemistry_guard，20 epoch。",
    ),
    RunSpec(
        label="chembl + aromatic + pathology",
        slug="next6_chembl_aromatic_pathology",
        group="核心 next6 / 环病理组合矩阵",
        run_dir="diffgui-next6-20260618-05-chembl_aromatic_pathology_20260619_160546",
        target_component=None,
        note="next6 核心矩阵：chembl_classifier + aromatic_scaffold + ring_pathology，20 epoch。",
    ),
    RunSpec(
        label="dock + chembl + aromatic + pathology",
        slug="next6_dock_chembl_aromatic_pathology",
        group="核心 next6 / 环病理组合矩阵",
        run_dir="diffgui-next6-20260618-06-dock_chembl_aromatic_pathology_20260619_204502",
        target_component=None,
        note="next6 核心矩阵：docking + chembl_classifier + aromatic_scaffold + ring_pathology，20 epoch。",
    ),
    RunSpec(
        label="Dock+classifier v0.2.1",
        slug="dock_classifier_v021",
        group="早期正式大训练 / 背景短训",
        run_dir="diffgui-8gpu-chembl21-dock_20260609_224313",
        target_component="chembl_classifier_reward",
        note="初版 docking + MLP classifier 正式训练，用户手动停止于 21 个完整 summary epoch。",
    ),
    RunSpec(
        label="Pure docking short",
        slug="pure_docking_short",
        group="早期正式大训练 / 背景短训",
        run_dir="diffgui-8gpu-docking-only-lr1e5-last2_20260610_094719",
        target_component="docking_reward",
        note="关键短训：虽只有 5 个完整 epoch，但用于判断单独 docking 是否能推出趋势。",
    ),
    RunSpec(
        label="Ring reward last3",
        slug="ring_last3_lr3e5",
        group="早期正式大训练 / 背景短训",
        run_dir="diffgui-8gpu-ring-last3-lr3e5_20260610_114650",
        target_component="ring_complexity_reward",
        note="重点结构 reward 训练；epoch 24 只有 rank-level 文件，summary 以 epoch 1-23 为准。",
    ),
    RunSpec(
        label="Classifier v0.3.1 short",
        slug="classifier_v031_short",
        group="早期正式大训练 / 背景短训",
        run_dir="diffgui-8gpu-b36-classifier-v0-3-1-ppo4_20260610_221410",
        target_component="chembl_classifier_reward",
        note="关键短训：虽只有 8 个完整 epoch，但代表坏分子审阅后 classifier v0.3.1 测试。",
    ),
    RunSpec(
        label="docking",
        slug="docking",
        group="Reward 归因实验 / 单奖励矩阵",
        run_dir="diffgui-ring-control-20260613-01-docking_20260613_153723",
        target_component="docking_reward",
        note="单奖励归因：docking=1.0。",
    ),
    RunSpec(
        label="ligand_efficiency",
        slug="ligand_efficiency",
        group="Reward 归因实验 / 单奖励矩阵",
        run_dir="diffgui-ring-control-20260613-02-ligand_efficiency_20260613_181441",
        target_component="ligand_efficiency_reward",
        note="单奖励归因：ligand_efficiency=1.0。",
    ),
    RunSpec(
        label="aromatic_scaffold",
        slug="aromatic_scaffold",
        group="Reward 归因实验 / 单奖励矩阵",
        run_dir="diffgui-ring-control-20260613-05-aromatic_scaffold_20260613_204920",
        target_component="aromatic_scaffold_reward",
        note="单奖励归因：aromatic_scaffold=1.0。",
    ),
    RunSpec(
        label="ring_pathology",
        slug="ring_pathology",
        group="Reward 归因实验 / 单奖励矩阵",
        run_dir="diffgui-ring-control-20260613-07-ring_pathology_20260613_232423",
        target_component="ring_pathology_reward",
        note="单奖励归因：ring_pathology=1.0。",
    ),
    RunSpec(
        label="chembl_classifier",
        slug="chembl_classifier",
        group="Reward 归因实验 / 单奖励矩阵",
        run_dir="diffgui-ring-control-20260613-08-chembl_classifier_20260614_020007",
        target_component="chembl_classifier_reward",
        note="单奖励归因：chembl_classifier=1.0。",
    ),
    RunSpec(
        label="qed",
        slug="qed",
        group="Reward 归因实验 / 单奖励矩阵",
        run_dir="diffgui-ring-control-20260615-09-qed_20260615_122202",
        target_component="qed_reward",
        note="单奖励归因：qed=1.0；122001 是启动失败/无 summary 的同名早期目录。",
    ),
    RunSpec(
        label="sa",
        slug="sa",
        group="Reward 归因实验 / 单奖励矩阵",
        run_dir="diffgui-ring-control-20260615-10-sa_20260615_163654",
        target_component="sa_reward",
        note="单奖励归因：sa=1.0。",
    ),
    RunSpec(
        label="ring_complexity",
        slug="ring_complexity",
        group="Reward 归因实验 / 单奖励矩阵",
        run_dir="diffgui-ring-control-20260615-11-ring_complexity_20260615_200439",
        target_component="ring_complexity_reward",
        note="单奖励归因：ring_complexity=1.0。",
    ),
    RunSpec(
        label="LE + ring + aromatic",
        slug="le_ring_aromatic",
        group="旧组合奖励 / 环控制矩阵",
        run_dir="diffgui-ring-control-20260615-12-le_ring_aromatic_20260615_224513",
        target_component=None,
        note="组合奖励：ligand_efficiency + ring_complexity + aromatic_scaffold。",
    ),
    RunSpec(
        label="docking + ring + aromatic",
        slug="dock_ring_aromatic",
        group="旧组合奖励 / 环控制矩阵",
        run_dir="diffgui-ring-control-20260615-13-dock_ring_aromatic_20260616_011801",
        target_component=None,
        note="组合奖励：docking + ring_complexity + aromatic_scaffold。",
    ),
]


EXCLUDED_PATTERNS = (
    "smoke",
    "replay-gate",
    "docking-stage1",
    "docking-train-smoke",
)

METRIC_COLUMNS = [
    "reward_total_mean",
    "reward_total_scored_mean",
    "molecule_rdkit_valid_ratio",
    "reconstruction_success_ratio",
    "docking_success_ratio_total",
    "docking_raw_mean",
    "docking_reward_mean",
    "ligand_efficiency_raw_mean",
    "descriptor_mmd_vs_baseline",
    "carbon_fraction",
    "tpsa_mean",
    "ring_count_mean",
    "aromatic_ring_mean",
    "three_member_ring_mean",
    "three_member_ring_total",
    "duplicate_ratio",
    "scaffold_top1_fraction",
    "PPO_ApproxKL",
    "ClipFrac",
    "RatioMean",
    "total_seconds",
    "ppo_seconds",
    "rollout_seconds",
    "reward_seconds",
]


def main() -> None:
    """Build all analysis outputs."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    selected = [build_run_record(spec) for spec in RUN_SPECS]
    selected_df = pd.DataFrame(selected)
    selected_df = assign_aliases(selected_df)
    selected_df.to_csv(ANALYSIS_DIR / "major_runs_summary.csv", index=False)
    selected_df.to_json(ANALYSIS_DIR / "major_runs_summary.json", orient="records", indent=2, force_ascii=False)

    epoch_df = build_epoch_metrics()
    epoch_df = epoch_df.merge(selected_df[["slug", "run_alias", "short_label"]], on="slug", how="left")
    epoch_df.to_csv(ANALYSIS_DIR / "major_runs_epoch_metrics.csv", index=False)

    component_df = build_component_metrics()
    component_df = component_df.merge(selected_df[["slug", "run_alias", "short_label"]], on="slug", how="left")
    component_df.to_csv(ANALYSIS_DIR / "reward_component_first_last.csv", index=False)

    excluded_df = build_excluded_runs()
    excluded_df.to_csv(ANALYSIS_DIR / "excluded_or_nonmajor_diffgui_runs.csv", index=False)

    write_plots(epoch_df, selected_df)
    copy_work_record_figures()
    write_work_record(selected_df, component_df, excluded_df)


def assign_aliases(selected_df: pd.DataFrame) -> pd.DataFrame:
    """Add stable short aliases for tables and plot legends."""
    out = selected_df.copy()
    aliases: list[str] = []
    counters: dict[str, int] = {}
    for _, row in out.iterrows():
        group = str(row["group"])
        counters[group] = counters.get(group, 0) + 1
        aliases.append(f"{GROUP_ALIAS_PREFIX.get(group, 'DG-RUN')}-{counters[group]:02d}")
    out.insert(0, "run_alias", aliases)
    out.insert(2, "short_label", out["run_alias"] + " " + out["label"].astype(str))
    out.insert(3, "run_link", [f"[{alias}](../../../{run_dir})" for alias, run_dir in zip(out["run_alias"], out["run_dir"])])
    return out


def build_run_record(spec: RunSpec) -> dict[str, Any]:
    """Summarize one selected run from config, summary CSV, and molecules CSV."""
    run_path = OUT_ROOT / spec.run_dir
    cfg = read_yaml(run_path / "config" / "resolved_config.yaml")
    summary = read_csv_rows(run_path / "training_summary.csv")
    if not summary:
        raise RuntimeError(f"Selected run has no training_summary rows: {run_path}")
    first = summary[0]
    last = summary[-1]
    weights = nonzero_weights(cfg.get("reward", {}).get("weights", {}))
    train_cfg = cfg.get("train", {})
    diffgui_training = cfg.get("diffgui_training", {})
    sweep_cfg = cfg.get("diffgui_docking_batch_sweep", {})
    first_epoch = int_float(first.get("epoch"))
    last_epoch = int_float(last.get("epoch"))
    target_first, target_last = target_component_values(run_path, first_epoch, last_epoch, spec.target_component)
    avg_total_seconds = mean_numeric(row.get("total_seconds") for row in summary)
    avg_ppo_seconds = mean_numeric(row.get("ppo_seconds") for row in summary)
    ppo_fraction = avg_ppo_seconds / avg_total_seconds if avg_total_seconds and not math.isnan(avg_total_seconds) else np.nan
    return {
        "group": spec.group,
        "label": spec.label,
        "slug": spec.slug,
        "run_dir": str(run_path.relative_to(PROJECT_ROOT)),
        "config_name": cfg.get("name", ""),
        "epochs_config": int_or_blank(diffgui_training.get("epochs")),
        "epochs_with_summary": len(summary),
        "summary_epoch_first": first_epoch,
        "summary_epoch_last": last_epoch,
        "world_size": int_or_blank(last.get("world_size")),
        "per_rank_batch_size": int_or_blank(last.get("per_rank_batch_size") or diffgui_training.get("batch_size")),
        "global_batch_size": int_or_blank(last.get("global_batch_size")),
        "num_timesteps": int_or_blank(last.get("num_timesteps") or sweep_cfg.get("num_timesteps")),
        "ligand_size_mode": sweep_cfg.get("ligand_size_mode", ""),
        "ligand_atoms": sweep_cfg.get("ligand_atoms", ""),
        "gui_strength": sweep_cfg.get("gui_strength", ""),
        "lr": train_cfg.get("learning_rate", ""),
        "ppo_epochs": train_cfg.get("ppo_epochs", ""),
        "clip_range": train_cfg.get("clip_range", ""),
        "precision": train_cfg.get("precision", ""),
        "freeze_mode": (train_cfg.get("freeze") or {}).get("mode", ""),
        "train_last_n_blocks": (train_cfg.get("freeze") or {}).get("train_last_n_blocks", ""),
        "reward_weights": format_weights(weights),
        "target_component": spec.target_component or "",
        "target_component_first_mean": target_first,
        "target_component_last_mean": target_last,
        "delta_target_component_mean": delta(target_first, target_last),
        "first_reward_total_mean": number(first.get("reward_total_mean")),
        "last_reward_total_mean": number(last.get("reward_total_mean")),
        "delta_reward_total_mean": delta(first.get("reward_total_mean"), last.get("reward_total_mean")),
        "first_reward_total_scored_mean": number(first.get("reward_total_scored_mean")),
        "last_reward_total_scored_mean": number(last.get("reward_total_scored_mean")),
        "delta_reward_total_scored_mean": delta(first.get("reward_total_scored_mean"), last.get("reward_total_scored_mean")),
        "first_valid_ratio": number(first.get("molecule_rdkit_valid_ratio")),
        "last_valid_ratio": number(last.get("molecule_rdkit_valid_ratio")),
        "delta_valid_ratio": delta(first.get("molecule_rdkit_valid_ratio"), last.get("molecule_rdkit_valid_ratio")),
        "first_reconstruction_success_ratio": number(first.get("reconstruction_success_ratio")),
        "last_reconstruction_success_ratio": number(last.get("reconstruction_success_ratio")),
        "first_docking_success_ratio_total": number(first.get("docking_success_ratio_total")),
        "last_docking_success_ratio_total": number(last.get("docking_success_ratio_total")),
        "delta_docking_success_ratio_total": delta(first.get("docking_success_ratio_total"), last.get("docking_success_ratio_total")),
        "first_docking_raw_mean": number(first.get("docking_raw_mean")),
        "last_docking_raw_mean": number(last.get("docking_raw_mean")),
        "delta_docking_raw_mean": delta(first.get("docking_raw_mean"), last.get("docking_raw_mean")),
        "first_docking_reward_mean": number(first.get("docking_reward_mean")),
        "last_docking_reward_mean": number(last.get("docking_reward_mean")),
        "delta_docking_reward_mean": delta(first.get("docking_reward_mean"), last.get("docking_reward_mean")),
        "last_descriptor_mmd_vs_baseline": number(last.get("descriptor_mmd_vs_baseline")),
        "first_carbon_fraction": number(first.get("carbon_fraction")),
        "last_carbon_fraction": number(last.get("carbon_fraction")),
        "delta_carbon_fraction": delta(first.get("carbon_fraction"), last.get("carbon_fraction")),
        "first_tpsa_mean": number(first.get("tpsa_mean")),
        "last_tpsa_mean": number(last.get("tpsa_mean")),
        "delta_tpsa_mean": delta(first.get("tpsa_mean"), last.get("tpsa_mean")),
        "first_ring_count_mean": number(first.get("ring_count_mean")),
        "last_ring_count_mean": number(last.get("ring_count_mean")),
        "delta_ring_count_mean": delta(first.get("ring_count_mean"), last.get("ring_count_mean")),
        "first_aromatic_ring_mean": number(first.get("aromatic_ring_mean")),
        "last_aromatic_ring_mean": number(last.get("aromatic_ring_mean")),
        "delta_aromatic_ring_mean": delta(first.get("aromatic_ring_mean"), last.get("aromatic_ring_mean")),
        "first_three_member_ring_mean": number(first.get("three_member_ring_mean")),
        "last_three_member_ring_mean": number(last.get("three_member_ring_mean")),
        "delta_three_member_ring_mean": delta(first.get("three_member_ring_mean"), last.get("three_member_ring_mean")),
        "last_three_member_ring_total": number(last.get("three_member_ring_total")),
        "last_duplicate_ratio": number(last.get("duplicate_ratio")),
        "last_scaffold_top1_fraction": number(last.get("scaffold_top1_fraction")),
        "last_PPO_ApproxKL": number(last.get("PPO_ApproxKL")),
        "last_ClipFrac": number(last.get("ClipFrac")),
        "avg_total_seconds": avg_total_seconds,
        "avg_ppo_seconds": avg_ppo_seconds,
        "ppo_time_fraction": ppo_fraction,
        "note": spec.note,
    }


def build_epoch_metrics() -> pd.DataFrame:
    """Return long-form epoch metrics for all selected runs."""
    rows: list[dict[str, Any]] = []
    spec_by_dir = {spec.run_dir: spec for spec in RUN_SPECS}
    for run_dir, spec in spec_by_dir.items():
        run_path = OUT_ROOT / run_dir
        for row in read_csv_rows(run_path / "training_summary.csv"):
            out: dict[str, Any] = {
                "group": spec.group,
                "label": spec.label,
                "slug": spec.slug,
                "run_dir": str(run_path.relative_to(PROJECT_ROOT)),
                "epoch": int_float(row.get("epoch")),
            }
            for metric in METRIC_COLUMNS:
                out[metric] = number(row.get(metric))
            rows.append(out)
    return pd.DataFrame(rows)


def build_component_metrics() -> pd.DataFrame:
    """Compute first/last molecule-level reward components for selected runs."""
    rows: list[dict[str, Any]] = []
    component_names = [
        "docking_reward",
        "ligand_efficiency_reward",
        "qed_reward",
        "sa_reward",
        "aromatic_scaffold_reward",
        "ring_complexity_reward",
        "ring_pathology_reward",
        "chembl_classifier_reward",
        "chemistry_guard_reward",
    ]
    for spec in RUN_SPECS:
        run_path = OUT_ROOT / spec.run_dir
        summary = read_csv_rows(run_path / "training_summary.csv")
        if not summary:
            continue
        first_epoch = int_float(summary[0].get("epoch"))
        last_epoch = int_float(summary[-1].get("epoch"))
        first_df = read_molecules(run_path, first_epoch)
        last_df = read_molecules(run_path, last_epoch)
        for component in component_names:
            if component not in first_df.columns and component not in last_df.columns:
                continue
            first_value = safe_series_mean(first_df.get(component))
            last_value = safe_series_mean(last_df.get(component))
            rows.append(
                {
                    "group": spec.group,
                    "label": spec.label,
                    "slug": spec.slug,
                    "component": component,
                    "first_epoch": first_epoch,
                    "last_epoch": last_epoch,
                    "first_mean": first_value,
                    "last_mean": last_value,
                    "delta_mean": numeric_delta(first_value, last_value),
                    "is_target_component": int(component == spec.target_component),
                }
            )
    return pd.DataFrame(rows)


def build_excluded_runs() -> pd.DataFrame:
    """List DiffGui run directories that were not selected as major runs."""
    selected = {spec.run_dir for spec in RUN_SPECS}
    rows: list[dict[str, Any]] = []
    for run_path in sorted(OUT_ROOT.glob("diffgui*")):
        if not run_path.is_dir() or run_path.name in selected:
            continue
        summary_path = run_path / "training_summary.csv"
        cfg_path = run_path / "config" / "resolved_config.yaml"
        rows_count = len(read_csv_rows(summary_path)) if summary_path.exists() else 0
        reason = "not selected"
        lowered = run_path.name.lower()
        if any(pattern in lowered for pattern in EXCLUDED_PATTERNS):
            reason = "smoke/debug/replay-gate/preflight run"
        elif rows_count == 0:
            reason = "no training_summary.csv or no complete summary epoch"
        elif rows_count < 5:
            reason = "too short and not a designated key short-run"
        cfg_name = ""
        if cfg_path.exists():
            cfg_name = str(read_yaml(cfg_path).get("name", ""))
        rows.append(
            {
                "run_dir": str(run_path.relative_to(PROJECT_ROOT)),
                "config_name": cfg_name,
                "epochs_with_summary": rows_count,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def write_plots(epoch_df: pd.DataFrame, selected_df: pd.DataFrame) -> None:
    """Write analysis figures."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.dpi": 140,
        }
    )
    plot_key_curve_panels(epoch_df)
    plot_final_metric_heatmap(selected_df, ANALYSIS_DIR / "major_runs_final_metric_heatmap.png")
    plot_next6_tradeoff_summary(selected_df)
    for group in selected_df["group"].drop_duplicates():
        group_df = epoch_df[epoch_df["group"] == group]
        figure_files = GROUP_FIGURE_FILES[group]
        plot_curve_grid(
            group_df,
            metrics=[
                ("reward_total_mean", "Reward total mean"),
                ("docking_raw_mean", "Docking raw mean"),
                ("docking_reward_mean", "Docking reward mean"),
                ("molecule_rdkit_valid_ratio", "RDKit valid ratio"),
                ("docking_success_ratio_total", "Docking success ratio"),
                ("PPO_ApproxKL", "PPO ApproxKL"),
            ],
            path=ANALYSIS_DIR / figure_files["training"],
            title=f"{group_plot_title(group)}: training overview",
        )
        plot_curve_grid(
            group_df,
            metrics=[
                ("carbon_fraction", "Carbon fraction"),
                ("tpsa_mean", "TPSA mean"),
                ("aromatic_ring_mean", "Aromatic ring mean"),
                ("three_member_ring_mean", "Three-member ring mean"),
            ],
            path=ANALYSIS_DIR / figure_files["properties"],
            title=f"{group_plot_title(group)}: generated molecule properties",
        )
        group_selected = selected_df[selected_df["group"] == group]
        plot_final_metric_heatmap(group_selected, ANALYSIS_DIR / figure_files["heatmap"], title=f"{group_plot_title(group)}: final metrics")


def plot_curve_grid(
    df: pd.DataFrame,
    *,
    metrics: list[tuple[str, str]],
    path: Path,
    title: str,
    layout: tuple[int, int] = (3, 2),
    legend_cols: int = 3,
) -> None:
    """Plot several metrics as trajectories by group."""
    slugs = list(dict.fromkeys(df["slug"].tolist()))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(slugs), 1)))
    color_by_slug = {slug: colors[idx] for idx, slug in enumerate(slugs)}
    nrows, ncols = layout
    if len(metrics) <= 4:
        nrows, ncols = 2, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.5, 4.0 * nrows), constrained_layout=True)
    axes_flat = axes.flatten()
    for ax, (metric, ylabel) in zip(axes_flat, metrics):
        endpoints: list[dict[str, Any]] = []
        for slug in slugs:
            sub = df[df["slug"] == slug].sort_values("epoch")
            if metric not in sub.columns or sub[metric].dropna().empty:
                continue
            label = str(sub["run_alias"].iloc[0])
            color = color_by_slug[slug]
            ax.plot(
                sub["epoch"],
                sub[metric],
                label=label,
                linewidth=1.8,
                alpha=0.9,
                color=color,
            )
            valid_points = sub[["epoch", metric]].dropna()
            if not valid_points.empty:
                last_point = valid_points.iloc[-1]
                endpoints.append(
                    {
                        "x": float(last_point["epoch"]),
                        "y": float(last_point[metric]),
                        "label": label,
                        "color": color,
                    }
                )
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        annotate_line_ends(ax, endpoints)
    for ax in axes_flat[len(metrics):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=15, y=1.03)
    fig.savefig(path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def annotate_line_ends(
    ax: plt.Axes,
    endpoints: list[dict[str, Any]],
    *,
    min_gap_px: float = 13.0,
    font_size: int = 7,
) -> None:
    """Attach non-overlapping labels to the right end of plotted lines."""
    clean = [
        point
        for point in endpoints
        if np.isfinite(point.get("x", np.nan)) and np.isfinite(point.get("y", np.nan))
    ]
    if not clean:
        return

    x_low, x_high = ax.get_xlim()
    x_span = max(abs(x_high - x_low), 1.0)
    max_x = max(float(point["x"]) for point in clean)
    label_x = max_x + x_span * 0.025
    ax.set_xlim(x_low, max(x_high, max_x + x_span * 0.22))
    ax.figure.canvas.draw()

    y_low, y_high = ax.get_ylim()
    if abs(y_high - y_low) < 1e-12:
        y_low -= 0.5
        y_high += 0.5
        ax.set_ylim(y_low, y_high)
        ax.figure.canvas.draw()

    bbox = ax.get_window_extent()
    available = max(bbox.height - 8.0, 1.0)
    gap = min_gap_px
    if len(clean) > 1:
        gap = min(min_gap_px, max(6.0, available / max(len(clean), 1)))

    items = []
    for point in clean:
        _, y_display = ax.transData.transform((float(point["x"]), float(point["y"])))
        items.append((y_display, point))
    items.sort(key=lambda item: item[0])

    y_min = bbox.y0 + 4.0
    y_max = bbox.y1 - 4.0
    positions = [float(item[0]) for item in items]
    for idx in range(1, len(positions)):
        positions[idx] = max(positions[idx], positions[idx - 1] + gap)
    if positions and positions[-1] > y_max:
        overflow = positions[-1] - y_max
        positions = [pos - overflow for pos in positions]
    for idx in range(len(positions) - 2, -1, -1):
        positions[idx] = min(positions[idx], positions[idx + 1] - gap)
    if positions and positions[0] < y_min:
        underflow = y_min - positions[0]
        positions = [pos + underflow for pos in positions]

    inverse = ax.transData.inverted()
    for position, (_, point) in zip(positions, items):
        x_display = ax.transData.transform((label_x, float(point["y"])))[0]
        y_label = inverse.transform((x_display, position))[1]
        ax.annotate(
            str(point["label"]),
            xy=(float(point["x"]), float(point["y"])),
            xytext=(label_x, y_label),
            textcoords="data",
            color=point["color"],
            fontsize=font_size,
            va="center",
            ha="left",
            arrowprops={
                "arrowstyle": "-",
                "lw": 0.7,
                "color": point["color"],
                "alpha": 0.65,
                "shrinkA": 0,
                "shrinkB": 2,
            },
            bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.75},
            clip_on=False,
        )


def plot_key_curve_panels(epoch_df: pd.DataFrame) -> None:
    """Plot readable key curves for representative DiffGui RL runs."""
    selected_aliases = {
        "DG-NEXT-01",
        "DG-NEXT-02",
        "DG-NEXT-03",
        "DG-NEXT-04",
        "DG-NEXT-05",
        "DG-NEXT-06",
        "DG-MAIN-03",
        "DG-MAIN-04",
        "DG-ATTR-03",
        "DG-ATTR-05",
        "DG-COMB-01",
        "DG-COMB-02",
    }
    df = epoch_df[epoch_df["run_alias"].isin(selected_aliases)]
    plot_curve_grid(
        df,
        metrics=[
            ("reward_total_mean", "Reward total mean"),
            ("docking_raw_mean", "Docking raw mean"),
            ("docking_reward_mean", "Docking reward mean"),
            ("molecule_rdkit_valid_ratio", "RDKit valid ratio"),
            ("three_member_ring_mean", "Three-member ring mean"),
            ("aromatic_ring_mean", "Aromatic ring mean"),
        ],
        path=ANALYSIS_DIR / "major_runs_key_curve_panels.png",
        title="Readable key curves for representative DiffGui RL runs",
        legend_cols=4,
    )


def plot_next6_tradeoff_summary(selected_df: pd.DataFrame) -> None:
    """Plot the current next6 trade-off summary."""
    df = selected_df[selected_df["group"] == "核心 next6 / 环病理组合矩阵"].copy()
    if df.empty:
        return
    colors = plt.cm.tab10(np.linspace(0, 1, len(df)))
    color_by_alias = {str(row["run_alias"]): color for (_, row), color in zip(df.iterrows(), colors)}
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.0, 4.2),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.05, 1.05, 0.9]},
    )

    axes[0].scatter(
        df["last_descriptor_mmd_vs_baseline"],
        df["last_valid_ratio"],
        s=70,
        c=colors,
        edgecolors="#333333",
        linewidths=0.5,
    )
    axes[0].set_xlabel("Descriptor MMD vs baseline (lower better)")
    axes[0].set_ylabel("RDKit valid ratio")
    axes[0].set_title("Distribution vs validity")

    axes[1].scatter(
        df["last_three_member_ring_mean"],
        df["last_aromatic_ring_mean"],
        s=70,
        c=colors,
        edgecolors="#333333",
        linewidths=0.5,
    )
    axes[1].set_xlabel("Three-member ring mean (lower better)")
    axes[1].set_ylabel("Aromatic ring mean")
    axes[1].set_title("Ring pathology vs aromatic scaffold")

    bar_df = df.sort_values("last_reward_total_mean", ascending=True)
    axes[2].barh(
        bar_df["run_alias"],
        bar_df["last_reward_total_mean"],
        color=[color_by_alias[str(alias)] for alias in bar_df["run_alias"]],
        alpha=0.88,
    )
    axes[2].axvline(0.0, color="#555555", linewidth=0.8, alpha=0.8)
    axes[2].set_xlabel("Final reward_total_mean")
    axes[2].set_title("Online reward endpoint")
    axes[2].tick_params(axis="y", labelsize=8)

    for (_, row), color in zip(df.iterrows(), colors):
        axes[0].annotate(
            row["run_alias"],
            xy=(row["last_descriptor_mmd_vs_baseline"], row["last_valid_ratio"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
            color=color,
        )
        axes[1].annotate(
            row["run_alias"],
            xy=(row["last_three_member_ring_mean"], row["last_aromatic_ring_mean"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
            color=color,
        )
    fig.suptitle("DiffGui next6 core matrix: endpoint trade-offs", fontsize=14)
    for suffix in ("png", "svg"):
        fig.savefig(ANALYSIS_DIR / f"next6_tradeoff_summary.{suffix}", bbox_inches="tight", dpi=180)
    plt.close(fig)


def plot_final_metric_heatmap(selected_df: pd.DataFrame, path: Path, title: str = "DiffGui major runs: final metrics") -> None:
    """Plot final endpoint metrics with short aliases."""
    metrics = [
        ("epochs_with_summary", "epochs"),
        ("last_reward_total_mean", "reward"),
        ("last_valid_ratio", "valid"),
        ("last_descriptor_mmd_vs_baseline", "MMD"),
        ("last_ring_count_mean", "rings"),
        ("last_aromatic_ring_mean", "arom"),
        ("last_three_member_ring_mean", "3-ring"),
        ("last_PPO_ApproxKL", "KL"),
    ]
    df = selected_df.copy()
    values = df[[metric for metric, _ in metrics]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    normalized = normalize_columns(values)
    labels = [f"{row.run_alias} {row.label}" for row in df.itertuples()]
    fig, ax = plt.subplots(figsize=(11.0, max(3.0, len(df) * 0.45)))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#eeeeee")
    image = ax.imshow(normalized, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([label for _, label in metrics], rotation=25, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            if np.isnan(value):
                text = "NA"
            elif metrics[col_idx][0] == "epochs_with_summary":
                text = str(int(value))
            else:
                text = f"{value:.2f}"
            ax.text(col_idx, row_idx, text, ha="center", va="center", fontsize=6)
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="column-normalized value")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def normalize_columns(values: np.ndarray) -> np.ndarray:
    """Normalize each numeric column to 0-1, preserving NaN."""
    out = np.full(values.shape, np.nan, dtype=float)
    for col_idx in range(values.shape[1]):
        col = values[:, col_idx]
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            continue
        low = np.nanmin(finite)
        high = np.nanmax(finite)
        if abs(high - low) < 1e-12:
            out[:, col_idx] = np.where(np.isfinite(col), 0.5, np.nan)
        else:
            out[:, col_idx] = (col - low) / (high - low)
    return out


def build_work_record_sections(
    selected_df: pd.DataFrame,
    component_df: pd.DataFrame,
    excluded_df: pd.DataFrame,
) -> list[str]:
    """Build the detailed Markdown sections for the daily work record."""
    next6_count = int((selected_df["group"] == "核心 next6 / 环病理组合矩阵").sum())
    attr_count = int((selected_df["group"] == "Reward 归因实验 / 单奖励矩阵").sum())
    old_combo_count = int((selected_df["group"] == "旧组合奖励 / 环控制矩阵").sum())
    early_count = int((selected_df["group"] == "早期正式大训练 / 背景短训").sum())
    sections = [
        "# DiffGui 强化学习历次主要训练汇总",
        "",
        "本文件是当日工作记录主产物；分析目录只保存脚本、CSV/JSON 和图件证据。",
        "",
        (
            f"纳入主要训练 {len(selected_df)} 个，其中当前核心是 {next6_count} 个 `diffgui-next6` 20 epoch 组合矩阵；"
            f"另保留 {attr_count} 个单奖励归因 run、{old_combo_count} 个旧组合 run、{early_count} 个早期正式/短训背景 run。"
        ),
        "",
        "## 范围与筛选",
        "",
        (
            "本记录全面扫描 `OUT/diffgui*` 中带 `training_summary.csv` 的 DiffGui RL 运行，"
            "以 10 个完整 summary epoch 作为主要训练筛选线；2026-06-21 更新后将 `diffgui-next6-20260618-*` "
            "六个 20 epoch run 作为当前核心矩阵。同时保留 `pure docking` 5 轮和 `classifier v0.3.1` 8 轮等关键短训作为背景。"
            "smoke、preflight、replay-gate smoke、无完整 summary 的启动失败目录不进入主要训练表。"
        ),
        "",
        "证据来源：每个 run 的 `config/resolved_config.yaml`、`training_summary.csv`、首末 `epochs/epoch_*/molecules.csv`。",
        "",
        "## 可读性整理原则",
        "",
        f"DiffGui 训练目录名同样过长，直接放入图例会让图表变成路径清单；{len(selected_df)} 个 run 的多项指标全部叠在一起也会形成不可读曲线团。本记录改为和 DiffSBDD RL 总结一致的三层结构：",
        "",
        "- 图里只使用稳定短别名，例如 `DG-NEXT-02`、`DG-MAIN-03`、`DG-ATTR-05`、`DG-COMB-02`。",
        "- 正文表格用短别名作为 run 链接，完整目录放在别名映射表和 CSV 中。",
        "- 主图只放核心 next6、代表性背景和最终指标热图；每条研究线拆成训练曲线、分子性质曲线、最终指标热图。",
        "",
        "## Reward 组件含义速览",
        "",
        "后文的 `reward` 列都是这些组件的加权和；无效分子或组件计算失败通常会进入有限负奖励/惩罚。",
        "",
        reward_component_glossary_markdown(),
        "",
        "## 总结先行",
        "",
        build_conclusion(selected_df),
        "",
        "## 总体图",
        "",
        "![代表性关键曲线](major_runs_key_curve_panels.png)",
        "",
        "![最终指标热图](major_runs_final_metric_heatmap.png)",
        "",
        "![next6 核心矩阵 tradeoff](next6_tradeoff_summary.png)",
        "",
        "## 分组分析",
        "",
    ]
    for group in selected_df["group"].drop_duplicates():
        group_df = selected_df[selected_df["group"] == group]
        sections.extend(group_section(group, group_df, component_df))

    sections.extend(
        [
            "## Reward 归因组件首末变化",
            "",
            df_to_markdown(
                component_df[component_df["is_target_component"] == 1][
                    [
                        "group",
                        "run_alias",
                        "label",
                        "component",
                        "first_mean",
                        "last_mean",
                        "delta_mean",
                    ]
                ],
                float_digits=3,
            ),
            "",
            "## 未纳入主要表的 DiffGui run",
            "",
            df_to_markdown(excluded_df, float_digits=3),
            "",
            "## 附表：主要训练总览",
            "",
            df_to_markdown(
                selected_df[
                    [
                        "group",
                        "run_link",
                        "label",
                        "epochs_with_summary",
                        "per_rank_batch_size",
                        "global_batch_size",
                        "num_timesteps",
                        "lr",
                        "ppo_epochs",
                        "freeze_mode",
                        "train_last_n_blocks",
                        "last_reward_total_mean",
                        "delta_reward_total_mean",
                        "last_valid_ratio",
                        "last_docking_success_ratio_total",
                        "last_descriptor_mmd_vs_baseline",
                        "last_ring_count_mean",
                        "last_aromatic_ring_mean",
                        "last_three_member_ring_mean",
                    ]
                ],
                float_digits=3,
            ),
            "",
            "## 附表：奖励函数总表",
            "",
            "下表不按研究线分组，单独列出每个主要 run 的 reward 组成和归因目标，便于横向检查配置差异。",
            "",
            reward_table_markdown(selected_df, include_group=True),
            "",
            "## 附表：别名映射表",
            "",
            df_to_markdown(
                selected_df[
                    [
                        "run_alias",
                        "label",
                        "group",
                        "run_dir",
                    ]
                ],
                float_digits=3,
            ),
            "",
            "## 输出文件",
            "",
            "- [major_runs_summary.csv](../../../analysis/diffgui_rl_major_runs_20260618/major_runs_summary.csv)",
            "- [major_runs_epoch_metrics.csv](../../../analysis/diffgui_rl_major_runs_20260618/major_runs_epoch_metrics.csv)",
            "- [reward_component_first_last.csv](../../../analysis/diffgui_rl_major_runs_20260618/reward_component_first_last.csv)",
            "- [excluded_or_nonmajor_diffgui_runs.csv](../../../analysis/diffgui_rl_major_runs_20260618/excluded_or_nonmajor_diffgui_runs.csv)",
            "- 正文引用图件均已复制到本工作记录同级目录，Markdown 图片链接使用这些本地副本。",
            "- `major_runs_key_curve_panels.png`：代表性关键曲线。",
            "- `major_runs_final_metric_heatmap.png`：全部主要 run 的最终指标热图。",
            "- `next6_tradeoff_summary.png/svg`：next6 六组核心矩阵的 endpoint tradeoff 图。",
            "- `group_*_training_curves.png`：每条研究线的训练曲线。",
            "- `group_*_molecule_properties.png`：每条研究线的生成分子性质曲线。",
            "- `group_*_final_heatmap.png`：每条研究线的最终指标热图。",
            "",
            "## 验证",
            "",
            "- `conda run -n dif python -m py_compile analysis/diffgui_rl_major_runs_20260618/build_report.py`：通过。",
            "- `conda run -n dif python analysis/diffgui_rl_major_runs_20260618/build_report.py`：通过，已重建 CSV/JSON、图件和本工作记录。",
            "",
            "## 解释限制",
            "",
            (
                "这些指标主要来自训练期 rollout 样本和在线 reward。它们适合比较训练轨迹和 reward 响应，"
                "但不能替代 checkpoint 离线等量采样、独立 docking/D-MPNN 重打分和人工结构审计。"
                "`ring_last3_lr3e5` 的 epoch 24 只有 rank-level 文件，本记录按完整 `training_summary.csv` 的 23 轮解释。"
            ),
            "",
        ]
    )
    return sections


def reward_component_glossary_markdown() -> str:
    """Return a compact explanation table for reward components."""
    rows = [
        {
            "reward 组件": "docking",
            "内容": "对接打分；原始 docking score 越负越好，训练里归一化成越高越好。",
            "主要作用": "直接推结合强度，也提供 LE 的原始输入。",
        },
        {
            "reward 组件": "ligand_efficiency",
            "内容": "LE = -docking_score / heavy_atom_count。",
            "主要作用": "偏向单位原子结合效率更高、不要只靠变大的配体。",
        },
        {
            "reward 组件": "aromatic_scaffold",
            "内容": "奖励单苯环；惩罚无苯环、过多苯环/杂环、融合环、三元环和卤素过多。",
            "主要作用": "保留简单芳香骨架，同时压异常环形态。",
        },
        {
            "reward 组件": "ring_pathology",
            "内容": "从基础分扣除环病理：环过多、大环、大环并六元环、三元环过多或 fused/spiro 三元环。",
            "主要作用": "专门压 DiffGui 里出现的结构病理环。",
        },
        {
            "reward 组件": "chemistry_guard",
            "内容": "检查碳比例、杂原子比例、三元环、碎片数、异常键长和异常卤素位置。",
            "主要作用": "基础化学护栏，防止明显坏结构被继续放大。",
        },
        {
            "reward 组件": "chembl_classifier",
            "内容": "Morgan count + RDKit 2D 特征输入 MLP，输出 ChEMBL-like 概率并映射为奖励/惩罚。",
            "主要作用": "把分子拉回更像 ChEMBL 的 drug-like 化学空间。",
        },
        {
            "reward 组件": "ring_complexity",
            "内容": "软惩罚超过目标的总环数、芳香环数、融合环、三元环、螺原子和桥头原子。",
            "主要作用": "旧版环复杂度控制；比 ring_pathology 更宽泛。",
        },
        {
            "reward 组件": "qed",
            "内容": "RDKit QED，范围 0-1，越高越 drug-like。",
            "主要作用": "常规成药性倾向。",
        },
        {
            "reward 组件": "sa",
            "内容": "原始 SA 为 1-10，越低越易合成；reward=(10-SA)/9，越高越易合成。",
            "主要作用": "常规合成可及性倾向。",
        },
    ]
    return df_to_markdown(pd.DataFrame(rows), float_digits=3)


def build_conclusion(df: pd.DataFrame) -> str:
    """Return concise overall conclusions from selected metrics."""
    next6 = df[df["group"] == "核心 next6 / 环病理组合矩阵"]
    best_valid = next6.loc[next6["last_valid_ratio"].astype(float).idxmax()]
    best_mmd = next6.loc[next6["last_descriptor_mmd_vs_baseline"].astype(float).idxmin()]
    best_aromatic = next6.loc[next6["last_aromatic_ring_mean"].astype(float).idxmax()]
    best_three_ring = next6.loc[next6["last_three_member_ring_mean"].astype(float).idxmin()]
    old_ring = df[df["slug"] == "ring_last3_lr3e5"].iloc[0]
    if best_valid["slug"] == best_mmd["slug"]:
        distribution_line = (
            f"- next6 中 valid 与 MMD 最优均为 `{display_run(best_valid)}`："
            f"valid={fmt(best_valid['last_valid_ratio'])}，MMD={fmt(best_mmd['last_descriptor_mmd_vs_baseline'])}，"
            "这是当前最稳的分布/合法性方案。"
        )
    else:
        distribution_line = (
            f"- next6 中 valid 最优为 `{display_run(best_valid)}`，"
            f"valid={fmt(best_valid['last_valid_ratio'])}；MMD 最优为 `{display_run(best_mmd)}`，"
            f"MMD={fmt(best_mmd['last_descriptor_mmd_vs_baseline'])}。"
        )
    return "\n".join(
        [
            "- DiffGui PPO 工程链路已经能支撑 8GPU、200 step、20 epoch 的 next6 串行矩阵；时间瓶颈仍主要在 PPO 更新。",
            distribution_line,
            (
                f"- aromatic 保留最好的是 `{display_run(best_aromatic)}`，"
                f"aromatic_ring_mean={fmt(best_aromatic['last_aromatic_ring_mean'])}；"
                "这是旧 ring-last3 思路加 pathology/guard 后的主要价值。"
            ),
            (
                f"- 三元环最低的是 `{display_run(best_three_ring)}`，"
                f"three_member_ring_mean={fmt(best_three_ring['last_three_member_ring_mean'])}，"
                "说明 chembl + aromatic + pathology 更像结构病理抑制对照。"
            ),
            (
                "- 加 docking 不是单调变好：`dock + aromatic + pathology` docking success 高，"
                "但 MMD 不如 LE 方案；`dock + chembl + aromatic + pathology` 的 aromatic 明显偏低。"
            ),
            (
                f"- 早期 `Ring reward last3` 仍作为背景基线保留：valid={fmt(old_ring['last_valid_ratio'])}，"
                f"MMD={fmt(old_ring['last_descriptor_mmd_vs_baseline'])}，但 aromatic_ring_mean 下降，"
                "所以现在应以 next6 的 aromatic/pathology 拆分矩阵作为核心证据。"
            ),
            "- Reward 归因实验继续定位为诊断工具：`aromatic_scaffold`、`chembl_classifier`、`ring_pathology` 的目标组件有正向信号；`QED/SA/LE/ring_complexity` 单独优化不稳定。",
        ]
    )


def group_section(group: str, group_df: pd.DataFrame, component_df: pd.DataFrame) -> list[str]:
    """Return a Markdown section for one group."""
    figure_files = GROUP_FIGURE_FILES[group]
    lines = [
        f"### {group}",
        "",
        f"![{group} 训练曲线]({figure_files['training']})",
        "",
        f"![{group} 生成分子性质]({figure_files['properties']})",
        "",
        f"![{group} 最终指标热图]({figure_files['heatmap']})",
        "",
    ]
    reward_cols = [
        "run_link",
        "label",
        "epochs_with_summary",
        "reward_weights",
        "target_component",
    ]
    lines.extend(
        [
            "奖励函数表：",
            "",
            reward_table_markdown(group_df, include_group=False),
            "",
        ]
    )
    table_cols = [
        "run_link",
        "label",
        "epochs_with_summary",
        "last_reward_total_mean",
        "delta_reward_total_mean",
        "target_component_first_mean",
        "target_component_last_mean",
        "delta_target_component_mean",
        "last_valid_ratio",
        "last_docking_success_ratio_total",
        "last_descriptor_mmd_vs_baseline",
        "last_aromatic_ring_mean",
        "last_three_member_ring_mean",
        "last_PPO_ApproxKL",
    ]
    lines.append(df_to_markdown(group_df[table_cols], float_digits=3))
    lines.append("")
    lines.extend(group_interpretation(group, group_df))
    lines.append("")
    if group.startswith("Reward"):
        target = component_df[
            (component_df["group"] == group)
            & (component_df["is_target_component"] == 1)
        ][["run_alias", "label", "component", "first_mean", "last_mean", "delta_mean"]]
        lines.append("目标组件首末变化：")
        lines.append("")
        lines.append(df_to_markdown(target, float_digits=3))
        lines.append("")
    return lines


def reward_table_markdown(df: pd.DataFrame, *, include_group: bool) -> str:
    """Return a compact reward-configuration table for the report."""
    cols = [
        "run_link",
        "label",
        "epochs_with_summary",
        "reward_weights",
        "target_component",
    ]
    if include_group:
        cols.insert(0, "group")
    display_df = df[cols].rename(
        columns={
            "run_link": "run",
            "epochs_with_summary": "epochs",
            "reward_weights": "reward",
            "target_component": "target",
        }
    )
    return df_to_markdown(display_df, float_digits=3)


def group_interpretation(group: str, group_df: pd.DataFrame) -> list[str]:
    """Return group-specific interpretation bullets."""
    if group.startswith("核心 next6"):
        best_valid = group_df.loc[group_df["last_valid_ratio"].astype(float).idxmax()]
        best_mmd = group_df.loc[group_df["last_descriptor_mmd_vs_baseline"].astype(float).idxmin()]
        best_aromatic = group_df.loc[group_df["last_aromatic_ring_mean"].astype(float).idxmax()]
        best_three_ring = group_df.loc[group_df["last_three_member_ring_mean"].astype(float).idxmin()]
        if best_valid["slug"] == best_mmd["slug"]:
            distribution_line = (
                f"- `{display_run(best_valid)}` 同时给出最高 valid={fmt(best_valid['last_valid_ratio'])} "
                f"和最低 MMD={fmt(best_mmd['last_descriptor_mmd_vs_baseline'])}，适合作为当前最稳的分布/合法性基线。"
            )
        else:
            distribution_line = (
                f"- `{display_run(best_valid)}` 的 valid 最高={fmt(best_valid['last_valid_ratio'])}；"
                f"`{display_run(best_mmd)}` 的 MMD 最低={fmt(best_mmd['last_descriptor_mmd_vs_baseline'])}，"
                "二者需要按合法性和分布相似性分开取舍。"
            )
        return [
            "- 这 6 个 run 是当前 DiffGui 总结的核心证据：同为 8GPU、20 epoch、batch/rank=18 的组合矩阵，重点比较 aromatic_scaffold、ring_pathology、docking、LE、chembl_classifier 和 chemistry_guard 的取舍。",
            distribution_line,
            (
                f"- `{display_run(best_aromatic)}` 的 aromatic_ring_mean={fmt(best_aromatic['last_aromatic_ring_mean'])} 最高，"
                "说明 ring-last3 思路加 pathology/guard 后更能保住芳香骨架。"
            ),
            (
                f"- `{display_run(best_three_ring)}` 的 three_member_ring_mean={fmt(best_three_ring['last_three_member_ring_mean'])} 最低，"
                "更适合作为压结构病理的对照，但 reward_total_mean 为负，不能按总 reward 选它。"
            ),
            "- 加 docking 后 docking success 维持高位，但不会自动改善 MMD、芳香或三元环；后续组合需要按结构安全和分布指标一起筛。",
        ]
    if group.startswith("早期"):
        ring = group_df[group_df["slug"] == "ring_last3_lr3e5"]
        classifier = group_df[group_df["slug"] == "classifier_v031_short"]
        pure = group_df[group_df["slug"] == "pure_docking_short"]
        lines = [
            "- `Dock+classifier v0.2.1` 是最早完整正式训练，21 个 summary epoch 后 reward_total_mean 首末下降，说明直接迁移 docking + 老 MLP classifier 不够。",
            "- `Pure docking short` 虽只有 5 轮，但作为关键短训显示单独 docking 没有形成明确改善趋势，不能当主策略。",
        ]
        if not ring.empty:
            row = ring.iloc[0]
            lines.append(
                "- `Ring reward last3` 是该组最明确的正向训练："
                f"reward_total_mean {fmt(row['first_reward_total_mean'])}->{fmt(row['last_reward_total_mean'])}，"
                f"valid {fmt(row['first_valid_ratio'])}->{fmt(row['last_valid_ratio'])}；"
                "但芳香环下降、大环/复杂拓扑仍需新 reward 拆分。"
            )
        if not classifier.empty:
            row = classifier.iloc[0]
            lines.append(
                "- `Classifier v0.3.1 short` 单 epoch 成本高，8 轮内 reward_total_mean 仍低；"
                f"最终 PPO_ApproxKL={fmt(row['last_PPO_ApproxKL'])}，需要先做 classifier 校准和更小 PPO 设置。"
            )
        return lines
    if group.startswith("Reward"):
        positives = group_df[group_df["delta_target_component_mean"] > 0]
        negative_labels = group_df[group_df["delta_target_component_mean"] <= 0]["label"].tolist()
        lines = [
            "- 该组用于判断单个 reward 是否真的推动自身目标组件，而不是只看总分。",
            "- 目标组件首末上升的单奖励包括：" + ", ".join(positives["label"].tolist()) + "。",
        ]
        if negative_labels:
            lines.append(
                "- 目标组件没有上升或下降的单奖励包括："
                + ", ".join(negative_labels)
                + "；这些 reward 单独使用时不稳定，可能受奖励尺度、优势归一化、采样噪声和 PPO 更新强度影响。"
            )
        lines.append("- `docking` 最终总分高，但 docking_reward 目标组件略降，必须用独立重打分验证 top 分子。")
        return lines
    return [
        "- 两个组合奖励都是 10 轮矩阵测试，目标是把 docking/LE 与 ring/aromatic 约束组合起来。",
        "- `LE + ring + aromatic` 的 MMD 和三元环更好，适合作为保守分布控制对照。",
        "- `docking + ring + aromatic` 有效率最高，但三元环和 ring_count 仍偏高，说明 docking 加入后会把结构病理压力重新拉回来。",
    ]


def display_run(row: pd.Series) -> str:
    """Return alias plus human-readable label for prose."""
    return f"{row['run_alias']} {row['label']}"


def group_slug(group: str) -> str:
    """Return a stable ASCII slug for a report group."""
    if group.startswith("核心 next6"):
        return "group_next6_core_matrix"
    if group.startswith("早期"):
        return "group_early_major_and_key_short_runs"
    if group.startswith("Reward"):
        return "group_reward_attribution_single_reward_matrix"
    if group.startswith("旧组合"):
        return "group_combination_ring_control_matrix"
    return "group_misc"


def group_plot_title(group: str) -> str:
    """Return an ASCII plot title for a report group."""
    if group.startswith("核心 next6"):
        return "Core next6 ring-pathology combination matrix"
    if group.startswith("早期"):
        return "Early background and key short runs"
    if group.startswith("Reward"):
        return "Reward attribution single-reward matrix"
    if group.startswith("旧组合"):
        return "Legacy combination ring-control matrix"
    return "Misc group"


def write_work_record(
    selected_df: pd.DataFrame,
    component_df: pd.DataFrame,
    excluded_df: pd.DataFrame,
) -> None:
    """Write a durable work record for this analysis task."""
    record_dir = WORK_RECORD_DIR
    record_dir.mkdir(parents=True, exist_ok=True)
    record_path = WORK_RECORD_PATH
    record_path.write_text(
        "\n".join(build_work_record_sections(selected_df, component_df, excluded_df)),
        encoding="utf-8",
    )


def copy_work_record_figures() -> None:
    """Copy report figures next to the work record for self-contained Markdown."""
    WORK_RECORD_DIR.mkdir(parents=True, exist_ok=True)
    for filename in work_record_figure_files():
        source = ANALYSIS_DIR / filename
        if source.exists():
            shutil.copy2(source, WORK_RECORD_DIR / filename)


def work_record_figure_files() -> list[str]:
    """Return analysis figures referenced or documented by the work record."""
    figure_files = [
        "major_runs_key_curve_panels.png",
        "major_runs_final_metric_heatmap.png",
        "next6_tradeoff_summary.png",
        "next6_tradeoff_summary.svg",
    ]
    for group_files in GROUP_FIGURE_FILES.values():
        figure_files.extend(group_files.values())
    return figure_files


def read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file."""
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read CSV rows as dictionaries."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_molecules(run_path: Path, epoch: int) -> pd.DataFrame:
    """Read merged epoch molecule CSV for one run."""
    path = run_path / "epochs" / f"epoch_{int(epoch):04d}" / "molecules.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def target_component_values(run_path: Path, first_epoch: int, last_epoch: int, component: str | None) -> tuple[float, float]:
    """Return first/last mean for a target molecule-level reward component."""
    if not component:
        return np.nan, np.nan
    first_df = read_molecules(run_path, first_epoch)
    last_df = read_molecules(run_path, last_epoch)
    return safe_series_mean(first_df.get(component)), safe_series_mean(last_df.get(component))


def safe_series_mean(series: pd.Series | None) -> float:
    """Mean numeric values from a possibly missing pandas Series."""
    if series is None:
        return np.nan
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else np.nan


def nonzero_weights(weights: dict[str, Any]) -> dict[str, float]:
    """Return non-zero reward weights preserving config order."""
    out: dict[str, float] = {}
    for key, value in weights.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if abs(numeric) > 1e-12:
            out[str(key)] = numeric
    return out


def format_weights(weights: dict[str, float]) -> str:
    """Format reward weights compactly."""
    if not weights:
        return ""
    return " + ".join(f"{key} {value:g}" for key, value in weights.items())


def number(value: Any) -> float:
    """Convert a value to float, returning NaN when impossible."""
    if value is None or value == "":
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def int_float(value: Any) -> int:
    """Convert numeric CSV values like '1.0' to int."""
    numeric = number(value)
    if math.isnan(numeric):
        return 0
    return int(numeric)


def int_or_blank(value: Any) -> int | str:
    """Convert a value to int or blank string."""
    if value is None or value == "":
        return ""
    return int_float(value)


def delta(first: Any, last: Any) -> float:
    """Return last minus first with NaN handling."""
    return numeric_delta(number(first), number(last))


def numeric_delta(first: float, last: float) -> float:
    """Return last minus first for floats."""
    if math.isnan(first) or math.isnan(last):
        return np.nan
    return float(last - first)


def mean_numeric(values: Any) -> float:
    """Mean of numeric iterable values."""
    nums = [number(value) for value in values]
    nums = [value for value in nums if not math.isnan(value)]
    return float(np.mean(nums)) if nums else np.nan


def df_to_markdown(df: pd.DataFrame, *, float_digits: int = 3) -> str:
    """Convert a DataFrame to a compact GitHub Markdown table."""
    if df.empty:
        return "_无数据_"
    display_df = df.copy()
    for col in display_df.columns:
        if pd.api.types.is_numeric_dtype(display_df[col]):
            display_df[col] = display_df[col].map(lambda value: fmt_for_column(col, value, digits=float_digits))
    display_df = display_df.fillna("").astype(str)
    headers = list(display_df.columns)
    rows = display_df.values.tolist()
    lines = [
        "| " + " | ".join(escape_md_cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(escape_md_cell(value) for value in row) + " |")
    return "\n".join(lines)


def escape_md_cell(value: str) -> str:
    """Escape Markdown table cell separators."""
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def fmt(value: Any, digits: int = 3) -> str:
    """Format a scalar for report tables."""
    numeric = number(value)
    if math.isnan(numeric):
        return ""
    if abs(numeric) >= 100:
        return f"{numeric:.1f}"
    return f"{numeric:.{digits}f}"


def fmt_for_column(column: str, value: Any, digits: int = 3) -> str:
    """Format values with column-aware precision."""
    numeric = number(value)
    if math.isnan(numeric):
        return ""
    if column in {
        "epochs_with_summary",
        "summary_epoch_first",
        "summary_epoch_last",
        "world_size",
        "per_rank_batch_size",
        "global_batch_size",
        "num_timesteps",
        "ppo_epochs",
        "train_last_n_blocks",
        "first_epoch",
        "last_epoch",
        "epochs_config",
        "epochs",
    }:
        return str(int(round(numeric)))
    if column in {"lr", "clip_range"}:
        return f"{numeric:g}"
    return fmt(numeric, digits=digits)


if __name__ == "__main__":
    main()
