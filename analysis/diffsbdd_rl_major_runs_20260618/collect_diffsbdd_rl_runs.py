from __future__ import annotations

"""Collect a repository-local inventory of historical DiffSBDD RL runs."""

import csv
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "OUT"
ANALYSIS_DIR = ROOT / "analysis" / "diffsbdd_rl_major_runs_20260618"
WORK_RECORD_PATH = (
    ROOT
    / "work_records"
    / "26-6"
    / "260618"
    / "260618-10 DiffSBDD强化学习主要训练复筛.md"
)
RUN_NAME_RE = re.compile(r"^(?P<name>.+)_(?P<stamp>\d{8}_\d{6})$")
STRICT_GROUP_ORDER = {
    "01_early_baseline": 1,
    "02_qed_sa_docking_baseline": 2,
    "03_kl_constraint": 3,
    "04_chembl_prior": 4,
    "05_classifier_reward": 5,
    "06_chemistry_guard": 6,
    "07_reward_attribution": 7,
}
STRICT_GROUP_LABELS = {
    "01_early_baseline": "early DDPO baseline",
    "02_qed_sa_docking_baseline": "QED/SA/docking baseline",
    "03_kl_constraint": "KL constraint sweep",
    "04_chembl_prior": "ChEMBL prior route",
    "05_classifier_reward": "classifier route",
    "06_chemistry_guard": "chemistry guard route",
    "07_reward_attribution": "reward attribution matrix",
}
STRICT_GROUP_LABELS_ZH = {
    "01_early_baseline": "早期 DDPO / QED 基线",
    "02_qed_sa_docking_baseline": "QED+SA+docking 主线",
    "03_kl_constraint": "KL 约束扫描",
    "04_chembl_prior": "ChEMBL prior / 分布先验",
    "05_classifier_reward": "分类器奖励路线",
    "06_chemistry_guard": "化学护栏路线",
    "07_reward_attribution": "reward 归因实验",
}
GROUP_COLORS = {
    "01_early_baseline": "#5b6770",
    "02_qed_sa_docking_baseline": "#2f6f9f",
    "03_kl_constraint": "#7b5ba7",
    "04_chembl_prior": "#3b8f5a",
    "05_classifier_reward": "#b05c46",
    "06_chemistry_guard": "#c28f2c",
    "07_reward_attribution": "#4d8c8a",
}
RUN_ALIAS_PREFIXES = {
    "01_early_baseline": "BASE",
    "02_qed_sa_docking_baseline": "QSD",
    "03_kl_constraint": "KL",
    "04_chembl_prior": "PRIOR",
    "05_classifier_reward": "CLS",
    "06_chemistry_guard": "GUARD",
    "07_reward_attribution": "ATTR",
}
GROUP_FIGURE_FILES = {
    "01_early_baseline": {
        "training": "group_01_early_baseline_training_curves.png",
        "properties": "group_01_early_baseline_molecule_properties.png",
        "heatmap": "group_01_early_baseline_final_heatmap.png",
    },
    "02_qed_sa_docking_baseline": {
        "training": "group_02_qed_sa_docking_training_curves.png",
        "properties": "group_02_qed_sa_docking_molecule_properties.png",
        "heatmap": "group_02_qed_sa_docking_final_heatmap.png",
    },
    "03_kl_constraint": {
        "training": "group_03_kl_constraint_training_curves.png",
        "properties": "group_03_kl_constraint_molecule_properties.png",
        "heatmap": "group_03_kl_constraint_final_heatmap.png",
    },
    "04_chembl_prior": {
        "training": "group_04_chembl_prior_training_curves.png",
        "properties": "group_04_chembl_prior_molecule_properties.png",
        "heatmap": "group_04_chembl_prior_final_heatmap.png",
    },
    "05_classifier_reward": {
        "training": "group_05_classifier_training_curves.png",
        "properties": "group_05_classifier_molecule_properties.png",
        "heatmap": "group_05_classifier_final_heatmap.png",
    },
    "06_chemistry_guard": {
        "training": "group_06_chemistry_guard_training_curves.png",
        "properties": "group_06_chemistry_guard_molecule_properties.png",
        "heatmap": "group_06_chemistry_guard_final_heatmap.png",
    },
    "07_reward_attribution": {
        "training": "group_07_reward_attribution_training_curves.png",
        "properties": "group_07_reward_attribution_molecule_properties.png",
        "heatmap": "group_07_reward_attribution_final_heatmap.png",
    },
}


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _int_from_float(row: dict[str, str], key: str) -> int | None:
    value = _float(row, key)
    return None if value is None else int(round(value))


def _positive_weights(config: dict[str, Any]) -> dict[str, float]:
    reward_cfg = config.get("reward", {}) if isinstance(config.get("reward", {}), dict) else {}
    weights = reward_cfg.get("weights", {})
    if isinstance(weights, dict) and weights:
        result: dict[str, float] = {}
        for key, value in weights.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if numeric > 0:
                result[str(key)] = numeric
        return result
    enabled = reward_cfg.get("enabled", [])
    if isinstance(enabled, list):
        fallback_weight = 1.0 / max(len(enabled), 1)
        return {str(name): fallback_weight for name in enabled}
    return {}


def _reward_string(config: dict[str, Any]) -> str:
    weights = _positive_weights(config)
    if not weights:
        return ""
    return " + ".join(f"{name}:{value:g}" for name, value in weights.items())


def _classify_run(run_dir: Path, config: dict[str, Any], last_epoch: int) -> tuple[str, bool]:
    name = run_dir.name
    lower = name.lower()
    if lower.startswith("diffgui") or "diffgui" in lower:
        return "exclude_diffgui", False
    if lower.startswith("smoke") or "smoke" in lower:
        return "smoke_debug", False
    if "good_pool_audit" in lower:
        return "good_pool_audit", False
    if "single-reward" in lower:
        return "single_reward_matrix", True
    if "anticarbon" in lower or "guard" in lower:
        return "chemistry_guard", True
    if "logprob-mode" in lower:
        return "logprob_diagnostic", True
    if "kl-" in lower or "kl0" in lower or "kl-diagnostic" in lower:
        return "kl_diagnostic", True
    if "classifier" in lower:
        return "classifier_reward", True
    if "ecfp4" in lower or "fcfp4" in lower or "chembl" in lower:
        return "chembl_prior", True
    if "molskill" in lower:
        return "molskill_test", True
    if "qed-sa-docking" in lower or lower.startswith("6gpu-dock") or lower.startswith("8gpu-dock"):
        return "qed_sa_docking_or_dock", True
    if lower.startswith("diffsbdd-ddpo"):
        return ("early_ddpo_smoke" if last_epoch <= 3 else "early_ddpo_training"), last_epoch > 3
    return "other", last_epoch >= 5


def _is_principal(row: dict[str, Any]) -> bool:
    """Return whether a major run is important enough for the curated list."""
    if int(row.get("include_major", 0)) != 1:
        return False
    category = str(row.get("category", ""))
    last_epoch = int(row.get("last_epoch") or 0)
    status = str(row.get("status") or "")
    if category == "single_reward_matrix":
        return True
    if category in {"kl_diagnostic", "logprob_diagnostic"}:
        return last_epoch >= 5
    if category == "chemistry_guard":
        return last_epoch >= 4
    if category in {"classifier_reward", "chembl_prior", "qed_sa_docking_or_dock"}:
        return last_epoch >= 10 or (status == "completed" and last_epoch >= 5)
    if category == "early_ddpo_training":
        return last_epoch >= 10
    return False


def _is_strict_major(row: dict[str, Any]) -> bool:
    """Return whether a run is a main run with at least ten recorded epochs."""
    if int(row.get("include_principal", 0)) != 1:
        return False
    if int(row.get("last_epoch") or 0) < 10:
        return False
    if str(row.get("category", "")) == "logprob_diagnostic":
        return False
    return True


def _strict_group(row: dict[str, Any]) -> str:
    category = str(row.get("category", ""))
    if category == "single_reward_matrix":
        return "07_reward_attribution"
    if category == "chemistry_guard":
        return "06_chemistry_guard"
    if category == "classifier_reward":
        return "05_classifier_reward"
    if category == "chembl_prior":
        return "04_chembl_prior"
    if category == "kl_diagnostic":
        return "03_kl_constraint"
    if category == "qed_sa_docking_or_dock":
        return "02_qed_sa_docking_baseline"
    if category == "early_ddpo_training":
        return "01_early_baseline"
    return "01_early_baseline"


def _short_title(row: dict[str, Any]) -> str:
    """Return a readable one-line description for reports and plot legends."""
    run_dir = str(row.get("run_dir", ""))
    reward_setup = str(row.get("reward_setup", ""))
    kl_coef = row.get("kl_coef")
    category = str(row.get("category", ""))

    if category == "single_reward_matrix":
        reward_name = reward_setup.split(":")[0] if ":" in reward_setup else reward_setup
        return f"单奖励 {reward_name}"
    if category == "kl_diagnostic":
        return f"QSD + KL={kl_coef}"
    if "diffsbdd-ddpo-qed-sa-1gpu" in run_dir:
        return "QED 单奖励 1GPU"
    if category == "qed_sa_docking_or_dock":
        return f"QED/SA/docking KL={kl_coef}" if kl_coef not in ("", None) else "QED/SA/docking early"
    if "7gpu-classifier-only" in run_dir:
        return "classifier only"
    if "7gpu-dock-classifier-v0-1_20260529" in run_dir:
        return "dock+classifier formal"
    if "7gpu-dock-classifier-v0-1_20260528" in run_dir:
        return "dock+classifier early"
    if "diffsbdd-8gpu-b64-anticarbon-guard" in run_dir:
        return "anti-carbon guard b64"
    if "7gpu-dock-ecfp4-fcfp4_20260525" in run_dir:
        return "dock+ECFP/FCFP+logP"
    if "7gpu-dock-ecfp4-fcfp4_20260526" in run_dir:
        return "LE+ECFP/FCFP+logP"
    if "7gpu-dock-ecfp4-fcfp4_20260521" in run_dir:
        return "dock+prior+diversity"
    if "chembl-only" in run_dir:
        return "ChEMBL prior only"
    if "6gpu-dock-chembl" in run_dir:
        return "dock+ChEMBL prior"
    return str(row.get("run_name") or run_dir.replace("OUT/", ""))


def _compact_config(row: dict[str, Any]) -> str:
    parts = []
    if row.get("num_gpus") not in ("", None):
        parts.append(f"{row['num_gpus']}GPU")
    if row.get("batch_size_per_rank") not in ("", None):
        parts.append(f"batch/rank={row['batch_size_per_rank']}")
    if row.get("global_generated_final") not in ("", None):
        parts.append(f"gen/epoch={row['global_generated_final']}")
    if row.get("time_step") not in ("", None):
        parts.append(f"T={row['time_step']}")
    if row.get("learning_rate") not in ("", None):
        parts.append(f"lr={row['learning_rate']}")
    if row.get("kl_coef") not in ("", None):
        kl_text = f"KL={row['kl_coef']}"
        if row.get("kl_mode") not in ("", None):
            kl_text += f"({row['kl_mode']})"
        parts.append(kl_text)
    if row.get("ppo_epochs") not in ("", None):
        parts.append(f"PPO={row['ppo_epochs']}")
    if row.get("logprob_mode") not in ("", None):
        parts.append(f"logprob={row['logprob_mode']}")
    return "; ".join(parts)


def _compact_final(row: dict[str, Any]) -> str:
    parts = []
    metric_specs = [
        ("valid", "rdkit_valid_end", 3),
        ("dock", "avg_docking_end", 3),
        ("C", "carbon_fraction_end", 3),
        ("TPSA", "tpsa_mean_end", 1),
        ("LogP", "logp_mean_end", 2),
        ("hetero", "hetero_atom_mean_end", 2),
    ]
    for label, key, digits in metric_specs:
        value = row.get(key)
        if value in ("", None):
            continue
        if label == "dock" and isinstance(value, float) and abs(value) < 1e-12:
            continue
        if isinstance(value, float):
            parts.append(f"{label}={value:.{digits}f}")
        else:
            parts.append(f"{label}={value}")
    return "; ".join(parts)


def _diagnosis(row: dict[str, Any]) -> str:
    run_dir = str(row.get("run_dir", ""))
    reward_setup = str(row.get("reward_setup", ""))
    category = str(row.get("category", ""))
    last_epoch = int(row.get("last_epoch") or 0)
    valid_end = row.get("rdkit_valid_end")
    docking_end = row.get("avg_docking_end")
    carbon_end = row.get("carbon_fraction_end")
    tpsa_end = row.get("tpsa_mean_end")
    logp_end = row.get("logp_mean_end")

    if "7gpu-dock-classifier-v0-1_20260529_133609" in run_dir:
        return "docking 和 valid 提升，但出现高碳、低 TPSA、低杂原子漂移，是后续重置 reward 的核心证据。"
    if "7gpu-dock-classifier-v0-1_20260528_000841" in run_dir:
        return "同一 classifier 组合的较早正式试验，方向类似但漂移和 docking 改善弱于 0529 run。"
    if "7gpu-classifier-only" in run_dir:
        return "classifier 单独导航导致有效率坍缩，说明不能把分类器作为唯一 RL reward。"
    if "7gpu-dock-ecfp4-fcfp4_20260525_212140" in run_dir:
        return "docking/valid 明显变好，但 carbon fraction 极高、TPSA 降低，prior+logP 没有守住药化分布。"
    if "7gpu-dock-ecfp4-fcfp4_20260526_115341" in run_dir:
        return "LE+prior+logP 即使弱化 docking，也会走向高 LogP、高碳、低 TPSA。"
    if "diffsbdd-8gpu-b64-anticarbon-guard" in run_dir:
        return "化学护栏能压住高碳，但 docking 明显变差，属于过修正，需要更软的 guard 或权重扫描。"
    if category == "single_reward_matrix":
        reward_name = reward_setup.split(":")[0] if ":" in reward_setup else reward_setup
        return f"{reward_name} 单奖励 10 epoch 归因：用于看单一 reward 的方向性，不能单独代表最终方案。"
    if category == "kl_diagnostic":
        return "KL 扫描用于约束强度归因；旧 run 缺 descriptor 指标，主要看 valid/docking 和稳定性。"
    if category == "qed_sa_docking_or_dock":
        if isinstance(docking_end, float) and docking_end > 0:
            return "早期 docking 字段口径与后续不同，适合作为流程验证，不宜直接比较 docking 数值。"
        return "QED/SA/docking 基线能提供 docking 方向，但缺少分布护栏，后续才补 prior/classifier/guard。"
    if category == "chembl_prior":
        if isinstance(carbon_end, float) and carbon_end > 0.8:
            return "prior/LogP 组合带来 docking 或 valid 提升，但伴随明显高碳化。"
        return "ChEMBL prior 路线用于替代稀疏 classifier，整体仍需要 descriptor 级护栏。"
    if category == "early_ddpo_training":
        return "早期 DDPO 流程和 QED 单奖励验证，证明训练链路可跑，不作为当前最优 reward。"
    if isinstance(valid_end, float) and valid_end < 0.2:
        return "最终有效率过低，主要作为失败模式证据。"
    if isinstance(carbon_end, float) and isinstance(tpsa_end, float) and carbon_end > 0.78 and tpsa_end < 70:
        return "末期呈高碳低极性漂移，需要 guard 或 reward 重构。"
    if isinstance(logp_end, float) and logp_end > 3.0:
        return "末期 LogP 偏高，说明疏水/体积方向仍偏强。"
    return "作为对应路线的主要训练证据，需结合分子级审计判断可用性。"


def _run_stamp(run_dir: Path) -> str:
    match = RUN_NAME_RE.match(run_dir.name)
    return match.group("stamp") if match else ""


def collect_runs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(OUT_ROOT.glob("*/training_summary.csv")):
        run_dir = summary_path.parent
        summary_rows = _read_csv_rows(summary_path)
        if not summary_rows:
            continue
        first = summary_rows[0]
        last = summary_rows[-1]
        config = _read_yaml(run_dir / "config.yaml")
        manifest = _read_json(run_dir / "run_manifest.json")
        ckpt_rows = _read_csv_rows(run_dir / "checkpoint_manifest.csv")
        last_epoch = _int_from_float(last, "epoch") or len(summary_rows)
        category, include_major = _classify_run(run_dir, config, last_epoch)
        train_cfg = config.get("train", {}) if isinstance(config.get("train", {}), dict) else {}
        model_cfg = config.get("model", {}) if isinstance(config.get("model", {}), dict) else {}
        dataloader_cfg = config.get("dataloader", {}) if isinstance(config.get("dataloader", {}), dict) else {}
        ddp_cfg = config.get("ddp", {}) if isinstance(config.get("ddp", {}), dict) else {}
        row = {
            "run_dir": str(run_dir.relative_to(ROOT)),
            "run_name": str(config.get("name") or manifest.get("run_name") or run_dir.name),
            "stamp": _run_stamp(run_dir),
            "category": category,
            "include_major": int(include_major),
            "status": str(manifest.get("status", "")),
            "epochs_recorded": len(summary_rows),
            "last_epoch": last_epoch,
            "configured_epochs": train_cfg.get("epochs", ""),
            "num_gpus": ddp_cfg.get("num_gpus", ""),
            "batch_size_per_rank": dataloader_cfg.get("batch_size", ""),
            "global_generated_final": _int_from_float(last, "generated_total"),
            "time_step": model_cfg.get("time_step", ""),
            "learning_rate": train_cfg.get("learning_rate", ""),
            "kl_coef": train_cfg.get("kl_coef", ""),
            "kl_mode": train_cfg.get("kl_mode", ""),
            "ppo_epochs": train_cfg.get("ppo_epochs", ""),
            "logprob_mode": model_cfg.get("logprob_mode", ""),
            "reward_setup": _reward_string(config),
            "rdkit_valid_start": _float(first, "rdkit_valid_ratio"),
            "rdkit_valid_end": _float(last, "rdkit_valid_ratio"),
            "avg_docking_start": _float(first, "avg_docking"),
            "avg_docking_end": _float(last, "avg_docking"),
            "reward_mean_start": _float(first, "reward_mean"),
            "reward_mean_end": _float(last, "reward_mean"),
            "avg_reward_raw_end": _float(last, "avg_reward_raw"),
            "carbon_fraction_start": _float(first, "carbon_fraction"),
            "carbon_fraction_end": _float(last, "carbon_fraction"),
            "hetero_atom_mean_start": _float(first, "hetero_atom_mean"),
            "hetero_atom_mean_end": _float(last, "hetero_atom_mean"),
            "tpsa_mean_start": _float(first, "tpsa_mean"),
            "tpsa_mean_end": _float(last, "tpsa_mean"),
            "logp_mean_start": _float(first, "logp_mean"),
            "logp_mean_end": _float(last, "logp_mean"),
            "aromatic_ring_mean_end": _float(last, "aromatic_ring_mean"),
            "descriptor_mmd_vs_baseline_end": _float(last, "descriptor_mmd_vs_baseline"),
            "checkpoint_count": len(ckpt_rows),
            "last_checkpoint": ckpt_rows[-1].get("path", "") if ckpt_rows else "",
        }
        row["include_principal"] = int(_is_principal(row))
        row["include_strict_major"] = int(_is_strict_major(row))
        strict_group = _strict_group(row)
        row["strict_group"] = strict_group
        row["strict_group_label"] = STRICT_GROUP_LABELS[strict_group]
        row["strict_group_label_zh"] = STRICT_GROUP_LABELS_ZH[strict_group]
        row["run_alias"] = ""
        row["plot_label"] = ""
        row["short_title"] = _short_title(row)
        row["config_summary"] = _compact_config(row)
        row["final_metric_summary"] = _compact_final(row)
        row["diagnosis"] = _diagnosis(row)
        rows.append(row)
    return rows


def assign_run_aliases(rows: list[dict[str, Any]]) -> None:
    """Assign stable short aliases for strict-major runs."""
    strict_rows = [row for row in rows if int(row.get("include_strict_major", 0)) == 1]
    strict_rows.sort(
        key=lambda row: (
            STRICT_GROUP_ORDER.get(str(row.get("strict_group", "")), 99),
            str(row.get("stamp") or row.get("run_dir")),
        )
    )
    counters: dict[str, int] = {}
    for row in strict_rows:
        group = str(row["strict_group"])
        counters[group] = counters.get(group, 0) + 1
        alias = f"{RUN_ALIAS_PREFIXES.get(group, 'RUN')}-{counters[group]:02d}"
        row["run_alias"] = alias
        row["plot_label"] = alias

    for row in rows:
        if int(row.get("include_strict_major", 0)) != 1:
            row["run_alias"] = ""
            row["plot_label"] = ""


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt(value: Any, digits: int = 3) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown_table(rows: list[dict[str, Any]], cols: list[tuple[str, str]], max_rows: int | None = None) -> str:
    selected = rows if max_rows is None else rows[:max_rows]
    header = "| " + " | ".join(title for title, _ in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = []
    for row in selected:
        body.append("| " + " | ".join(_fmt(row.get(key)) for _, key in cols) + " |")
    return "\n".join([header, sep, *body])


def write_plots(rows: list[dict[str, Any]]) -> None:
    """Write simple inventory plots for the report."""
    principal = [row for row in rows if int(row.get("include_principal", 0)) == 1]
    counts: dict[str, int] = {}
    for row in principal:
        counts[str(row["category"])] = counts.get(str(row["category"]), 0) + 1

    categories = sorted(counts)
    values = [counts[category] for category in categories]
    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    ax.bar(categories, values, color="#3f6f8f")
    ax.set_title("Principal DiffSBDD RL runs by category")
    ax.set_ylabel("Run count")
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "principal_runs_by_category.png", dpi=180)
    plt.close(fig)

    timeline_rows = [
        row for row in principal
        if str(row.get("stamp", "")) and int(row.get("last_epoch") or 0) > 0
    ]
    timeline_rows.sort(key=lambda row: str(row["stamp"]))
    labels = [
        str(row["run_dir"]).replace("OUT/", "").replace("_2026", "\n2026").replace("_2025", "\n2025").replace("_2024", "\n2024")
        for row in timeline_rows
    ]
    epochs = [int(row.get("last_epoch") or 0) for row in timeline_rows]
    colors = ["#9270a8" if "single_reward" in str(row["category"]) else "#4b8f3a" for row in timeline_rows]
    fig_width = max(12.0, len(labels) * 0.35)
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    ax.bar(range(len(labels)), epochs, color=colors, alpha=0.85)
    ax.set_title("Principal DiffSBDD RL run lengths")
    ax.set_ylabel("Recorded epochs")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "principal_run_epoch_timeline.png", dpi=180)
    plt.close(fig)


def write_strict_plots(rows: list[dict[str, Any]]) -> None:
    """Write grouped plots for strict major runs."""
    strict = [row for row in rows if int(row.get("include_strict_major", 0)) == 1]
    strict.sort(
        key=lambda row: (
            STRICT_GROUP_ORDER.get(str(row.get("strict_group", "")), 99),
            str(row.get("stamp") or row.get("run_dir")),
        )
    )
    counts: dict[str, int] = {}
    for row in strict:
        group = str(row["strict_group"])
        counts[group] = counts.get(group, 0) + 1

    group_keys = sorted(counts, key=lambda key: STRICT_GROUP_ORDER.get(key, 99))
    fig, ax = plt.subplots(figsize=(10.0, 4.2))
    x_positions = list(range(len(group_keys)))
    ax.bar(
        x_positions,
        [counts[key] for key in group_keys],
        color=[GROUP_COLORS[key] for key in group_keys],
    )
    ax.set_title("Strict major DiffSBDD RL runs by research line")
    ax.set_ylabel("Run count")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([STRICT_GROUP_LABELS[key] for key in group_keys], rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "strict_major_runs_by_group.png", dpi=180)
    plt.close(fig)

    labels = [f"{row['run_alias']}\n{int(row.get('last_epoch') or 0)} ep" for row in strict]
    epochs = [int(row.get("last_epoch") or 0) for row in strict]
    colors = [GROUP_COLORS[str(row["strict_group"])] for row in strict]
    fig_width = max(12.0, len(labels) * 0.36)
    fig, ax = plt.subplots(figsize=(fig_width, 5.0))
    ax.bar(range(len(labels)), epochs, color=colors, alpha=0.9)
    ax.axhline(10, color="#444444", linewidth=1.0, linestyle="--")
    ax.set_title("Strict major run lengths (>=10 epochs)")
    ax.set_ylabel("Recorded epochs")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "strict_major_epoch_timeline.png", dpi=180)
    plt.close(fig)

    _write_strict_summary_heatmap(strict)
    _write_strict_key_curve_panels(strict)

    _write_group_overview_plots(strict)


def _line_label(row: dict[str, Any]) -> str:
    return str(row.get("plot_label") or row.get("run_alias") or row.get("run_name"))


def _metric_series(summary_rows: list[dict[str, str]], metric_keys: tuple[str, ...]) -> dict[str, tuple[list[int], list[float]]]:
    series: dict[str, tuple[list[int], list[float]]] = {}
    for metric_key in metric_keys:
        xs: list[int] = []
        ys: list[float] = []
        for summary_row in summary_rows:
            x_value = _int_from_float(summary_row, "epoch")
            y_value = _float(summary_row, metric_key)
            if x_value is None or y_value is None:
                continue
            xs.append(x_value)
            ys.append(y_value)
        if xs and ys and not all(abs(value) < 1e-12 for value in ys):
            series[metric_key] = (xs, ys)
    return series


def _value(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if abs(numeric) < 1e-12 and key in {"avg_docking_end", "carbon_fraction_end", "tpsa_mean_end", "logp_mean_end", "hetero_atom_mean_end"}:
        return None
    return numeric


def _normalize_columns(values: np.ndarray) -> np.ndarray:
    normalized = np.full(values.shape, np.nan, dtype=float)
    for col_idx in range(values.shape[1]):
        col = values[:, col_idx]
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            continue
        low = np.nanmin(finite)
        high = np.nanmax(finite)
        if abs(high - low) < 1e-12:
            normalized[:, col_idx] = np.where(np.isfinite(col), 0.5, np.nan)
        else:
            normalized[:, col_idx] = (col - low) / (high - low)
    return normalized


def _write_strict_summary_heatmap(strict_rows: list[dict[str, Any]]) -> None:
    rows = sorted(strict_rows, key=lambda row: (STRICT_GROUP_ORDER.get(str(row["strict_group"]), 99), str(row["run_alias"])))
    metric_specs = [
        ("last_epoch", "epochs"),
        ("rdkit_valid_end", "valid"),
        ("avg_docking_end", "dock"),
        ("carbon_fraction_end", "C frac"),
        ("tpsa_mean_end", "TPSA"),
        ("logp_mean_end", "LogP"),
        ("hetero_atom_mean_end", "hetero"),
    ]
    values = np.array(
        [[_value(row, key) if _value(row, key) is not None else np.nan for key, _ in metric_specs] for row in rows],
        dtype=float,
    )
    normalized = _normalize_columns(values)
    labels = [str(row["run_alias"]) for row in rows]
    fig_height = max(8.0, len(rows) * 0.32)
    fig, ax = plt.subplots(figsize=(11.0, fig_height))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#eeeeee")
    image = ax.imshow(normalized, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_title("Strict major runs: normalized final metrics (read by row alias)")
    ax.set_xticks(range(len(metric_specs)))
    ax.set_xticklabels([label for _, label in metric_specs], rotation=25, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            if not np.isfinite(value):
                text = "NA"
            elif metric_specs[col_idx][0] == "last_epoch":
                text = f"{int(value)}"
            else:
                text = f"{value:.2f}"
            ax.text(col_idx, row_idx, text, ha="center", va="center", fontsize=6, color="#111111")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="column-normalized value")
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "strict_major_final_metric_heatmap.png", dpi=180)
    plt.close(fig)


def _write_strict_key_curve_panels(strict_rows: list[dict[str, Any]]) -> None:
    selected = [
        row for row in strict_rows
        if row.get("strict_group") in {"02_qed_sa_docking_baseline", "04_chembl_prior", "05_classifier_reward", "06_chemistry_guard"}
    ]
    selected = [
        row for row in selected
        if row.get("run_alias") in {"QSD-03", "QSD-04", "PRIOR-07", "PRIOR-08", "CLS-01", "CLS-03", "GUARD-01"}
    ]
    metric_specs = [
        ("rdkit_valid_ratio", "Valid ratio"),
        ("avg_docking", "Docking"),
        ("carbon_fraction", "Carbon fraction"),
        ("tpsa_mean", "TPSA"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    for ax, (metric_key, title) in zip(axes.flat, metric_specs):
        endpoints: list[dict[str, Any]] = []
        for row in selected:
            summary_rows = _read_csv_rows(ROOT / str(row["run_dir"]) / "training_summary.csv")
            series = _metric_series(summary_rows, (metric_key,))
            if metric_key not in series:
                continue
            xs, ys = series[metric_key]
            color = GROUP_COLORS[str(row["strict_group"])]
            label = str(row["plot_label"])
            ax.plot(
                xs,
                ys,
                linewidth=1.8,
                alpha=0.88,
                color=color,
                label=label,
            )
            endpoints.append({"x": float(xs[-1]), "y": float(ys[-1]), "label": label, "color": color})
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.25)
        _annotate_line_ends(ax, endpoints)
    fig.suptitle("Readable key curves for representative major RL runs", y=0.995)
    fig.tight_layout(rect=(0, 0.03, 1, 0.965))
    fig.savefig(ANALYSIS_DIR / "strict_major_key_curve_panels.png", dpi=180)
    plt.close(fig)


def _write_group_overview_plots(strict_rows: list[dict[str, Any]]) -> None:
    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in strict_rows:
        by_group.setdefault(str(row["strict_group"]), []).append(row)

    for group, items in sorted(by_group.items(), key=lambda pair: STRICT_GROUP_ORDER.get(pair[0], 99)):
        items.sort(key=lambda row: str(row.get("stamp") or row.get("run_dir")))
        _write_group_curve_grid(
            group,
            items,
            [
                ("rdkit_valid_ratio", "Valid ratio"),
                ("reward_mean", "Reward mean"),
                ("avg_docking", "Docking"),
                ("KL_Penalty_Weighted", "KL weighted"),
            ],
            GROUP_FIGURE_FILES[group]["training"],
            "training curves",
        )
        _write_group_curve_grid(
            group,
            items,
            [
                ("carbon_fraction", "Carbon fraction"),
                ("tpsa_mean", "TPSA"),
                ("logp_mean", "LogP"),
                ("hetero_atom_mean", "Hetero atoms"),
            ],
            GROUP_FIGURE_FILES[group]["properties"],
            "generated-molecule properties",
        )
        _write_group_final_heatmap(group, items, GROUP_FIGURE_FILES[group]["heatmap"])


def _write_group_curve_grid(
    group: str,
    items: list[dict[str, Any]],
    metric_specs: list[tuple[str, str]],
    filename: str,
    title_suffix: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2))
    cmap = plt.get_cmap("tab10")
    for ax, (metric_key, title) in zip(axes.flat, metric_specs):
        plotted = False
        endpoints: list[dict[str, Any]] = []
        for run_idx, row in enumerate(items):
            summary_rows = _read_csv_rows(ROOT / str(row["run_dir"]) / "training_summary.csv")
            series = _metric_series(summary_rows, (metric_key,))
            if metric_key not in series:
                continue
            xs, ys = series[metric_key]
            color = cmap(run_idx % 10)
            label = _line_label(row)
            ax.plot(
                xs,
                ys,
                linewidth=1.55,
                alpha=0.82,
                color=color,
                label=label,
            )
            endpoints.append({"x": float(xs[-1]), "y": float(ys[-1]), "label": label, "color": color})
            plotted = True
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.25)
        _annotate_line_ends(ax, endpoints, font_size=6)
        if not plotted:
            ax.text(0.5, 0.5, "not recorded", transform=ax.transAxes, ha="center", va="center", color="#666666")
    fig.suptitle(f"{STRICT_GROUP_LABELS[group]}: {title_suffix}", y=0.995)
    fig.tight_layout(rect=(0, 0.03, 1, 0.965))
    fig.savefig(ANALYSIS_DIR / filename, dpi=180)
    plt.close(fig)


def _annotate_line_ends(
    ax: plt.Axes,
    endpoints: list[dict[str, Any]],
    *,
    min_gap_px: float = 13.0,
    font_size: int = 7,
) -> None:
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
        ax.set_ylim(y_low - 0.5, y_high + 0.5)
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


def _write_group_final_heatmap(group: str, items: list[dict[str, Any]], filename: str) -> None:
    metric_specs = [
        ("last_epoch", "epochs"),
        ("rdkit_valid_end", "valid"),
        ("avg_docking_end", "dock"),
        ("carbon_fraction_end", "C frac"),
        ("tpsa_mean_end", "TPSA"),
        ("logp_mean_end", "LogP"),
        ("hetero_atom_mean_end", "hetero"),
    ]
    values = np.array(
        [[_value(row, key) if _value(row, key) is not None else np.nan for key, _ in metric_specs] for row in items],
        dtype=float,
    )
    normalized = _normalize_columns(values)
    labels = [str(row["run_alias"]) for row in items]
    fig, ax = plt.subplots(figsize=(10.0, max(2.6, len(items) * 0.48)))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#eeeeee")
    image = ax.imshow(normalized, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_title(f"{STRICT_GROUP_LABELS[group]}: final metrics")
    ax.set_xticks(range(len(metric_specs)))
    ax.set_xticklabels([label for _, label in metric_specs], rotation=25, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            text = "NA" if not np.isfinite(value) else (f"{int(value)}" if metric_specs[col_idx][0] == "last_epoch" else f"{value:.2f}")
            ax.text(col_idx, row_idx, text, ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / filename, dpi=180)
    plt.close(fig)


def _markdown_table_raw(rows: list[dict[str, Any]], cols: list[tuple[str, str]], max_rows: int | None = None) -> str:
    selected = rows if max_rows is None else rows[:max_rows]
    header = "| " + " | ".join(title for title, _ in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = []
    for row in selected:
        values = []
        for _, key in cols:
            value = row.get(key, "")
            values.append("" if value is None else str(value).replace("\n", " "))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, sep, *body])


def _analysis_relpath(filename: str) -> str:
    return f"../../../analysis/diffsbdd_rl_major_runs_20260618/{filename}"


def _group_note(group: str) -> str:
    notes = {
        "01_early_baseline": "早期 DDPO/QED 基线主要证明训练链路可用；没有完整分子性质监控，不能作为当前 reward 优劣判断。",
        "02_qed_sa_docking_baseline": "QED+SA+docking 能提供 docking 方向，但缺少碳比例、TPSA、杂原子和环复杂度护栏。",
        "03_kl_constraint": "KL 扫描说明约束强度会影响 valid/docking 稳定性，但 KL 本身不改变 reward 的化学偏好。",
        "04_chembl_prior": "ChEMBL prior 和 logP/LE 组合能提高部分指标，但典型后果是高碳、低 TPSA、低杂原子漂移。",
        "05_classifier_reward": "分类器奖励需要和 docking/化学护栏配合；classifier-only 的有效率坍缩说明它不能单独当导航目标。",
        "06_chemistry_guard": "化学护栏能压住高碳，但当前强度会牺牲 docking，需要软约束和权重扫描。",
        "07_reward_attribution": "reward 归因实验只看单个 reward 的方向性，价值是定位偏差来源，不是最终方案。",
    }
    return notes.get(group, "")


def write_work_record(rows: list[dict[str, Any]]) -> None:
    strict = [row for row in rows if int(row.get("include_strict_major", 0)) == 1]
    strict.sort(
        key=lambda row: (
            STRICT_GROUP_ORDER.get(str(row.get("strict_group", "")), 99),
            str(row.get("stamp") or row.get("run_dir")),
        )
    )
    for row in strict:
        run_dir = str(row["run_dir"])
        row["run_md"] = f"[{row['run_alias']}](../../../{run_dir})"
        row["path_md"] = f"`{run_dir}`"
    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in strict:
        by_group.setdefault(str(row["strict_group"]), []).append(row)
    count_rows = [
        {
            "group": STRICT_GROUP_LABELS_ZH[group],
            "runs": len(items),
            "epoch_range": f"{min(int(item['last_epoch']) for item in items)}-{max(int(item['last_epoch']) for item in items)}",
        }
        for group, items in sorted(by_group.items(), key=lambda pair: STRICT_GROUP_ORDER.get(pair[0], 99))
    ]

    table_cols = [
        ("run", "run_md"),
        ("short name", "short_title"),
        ("epochs", "last_epoch"),
        ("config", "config_summary"),
        ("final metrics", "final_metric_summary"),
        ("analysis", "diagnosis"),
    ]
    reward_cols = [
        ("run", "run_md"),
        ("short name", "short_title"),
        ("group", "strict_group_label_zh"),
        ("epochs", "last_epoch"),
        ("reward", "reward_setup"),
    ]
    alias_cols = [
        ("alias", "run_alias"),
        ("short name", "short_title"),
        ("group", "strict_group_label_zh"),
        ("full path", "path_md"),
    ]
    report = [
        "# DiffSBDD强化学习主要训练复筛",
        "",
        "日期：2026-06-18",
        "",
        "## 背景",
        "",
        "上一版 `DiffSBDD强化学习历次主要训练清单` 覆盖面偏宽，包含 5 epoch logprob 诊断、短启动和较多追溯项。本记录按用户要求只保留“主要的、10轮训练以上”的 DiffSBDD RL 训练，并在每条研究线下补充对应的训练过程和生成分子性质总图。",
        "",
        "本文件就是本任务的主报告和工作记录；`analysis/diffsbdd_rl_major_runs_20260618/` 只保留 CSV、脚本和图片证据，不再另写一份重复详细报告。",
        "",
        "## 可读性整理原则",
        "",
        "原始 run 目录名同时包含模型、reward、GPU、日期和时间，直接放进图例会导致图变成路径清单；同时把 32 个 run 的多项指标全部叠在一张图里，会形成不可读的曲线团。本次整理采用三层结构：",
        "",
        "- 图里只使用稳定短别名，例如 `QSD-03`、`PRIOR-07`、`CLS-03`。",
        "- 正文表格用短别名作为链接，完整目录放在别名映射表和 CSV 中。",
        "- 主图只画关键指标；每条研究线拆成训练曲线、分子性质曲线、最终指标热图三类图。",
        "",
        "## 筛选规则",
        "",
        "- 数据来源：`OUT/*/training_summary.csv`，并读取同目录 `config.yaml`、`run_manifest.json`、`checkpoint_manifest.csv`。",
        "- 只保留 DiffSBDD RL 主线训练，排除 DiffGui、smoke/debug、good-pool audit、5 epoch logprob 诊断和少于 10 epoch 的短启动。",
        "- `10轮以上` 在这里按 `last_epoch >= 10` 处理；reward 归因矩阵正好是 10 epoch，因此作为单独研究线保留。",
        "- 部分早期 run 的 manifest 仍写 `running`，这里以已落盘 `training_summary.csv` 最后一行为准。",
        "",
        "## 分组总览",
        "",
        _markdown_table_raw(count_rows, [("研究线", "group"), ("run 数", "runs"), ("epoch 范围", "epoch_range")]),
        "",
        f"- 纳入 10 轮以上主要训练：{len(strict)} 个 run。",
        f"- reward 归因实验单独分组：{len(by_group.get('07_reward_attribution', []))} 个 10 epoch 单奖励 run。",
        "",
        f"![主要训练分组]({_analysis_relpath('strict_major_runs_by_group.png')})",
        "",
        f"![训练轮数时间线]({_analysis_relpath('strict_major_epoch_timeline.png')})",
        "",
        "## 总的训练指标演变图",
        "",
        "下图把纳入清单的所有主要 run 叠加在一起。不同研究线用不同颜色；没有实时计算的指标不会强行补零，例如未启用 docking 的 run 不参与 docking 曲线解释。",
        "",
        f"![总的训练指标演变]({_analysis_relpath('strict_major_key_curve_panels.png')})",
        "",
        "下图按训练开始时间排列每个 run 的最终指标，用来观察从早期 QED/docking、KL、prior、classifier 到 guard/reward 归因的路线变化。",
        "",
        f"![终点指标演变]({_analysis_relpath('strict_major_final_metric_heatmap.png')})",
        "",
        "## 别名映射表",
        "",
        _markdown_table_raw(strict, alias_cols),
        "",
        "## 奖励函数总表",
        "",
        "下表不按研究线分组，单独列出每个主要 run 的 reward 组成，便于直接核对配置差异。",
        "",
        _markdown_table_raw(strict, reward_cols),
        "",
        "## 按研究线分组",
    ]
    for group, items in sorted(by_group.items(), key=lambda pair: STRICT_GROUP_ORDER.get(pair[0], 99)):
        figure_files = GROUP_FIGURE_FILES[group]
        report.extend(
            [
                "",
                f"### {STRICT_GROUP_LABELS_ZH[group]}",
                "",
                _group_note(group),
                "",
                f"![{STRICT_GROUP_LABELS_ZH[group]}训练曲线]({_analysis_relpath(figure_files['training'])})",
                "",
                f"![{STRICT_GROUP_LABELS_ZH[group]}生成分子性质]({_analysis_relpath(figure_files['properties'])})",
                "",
                f"![{STRICT_GROUP_LABELS_ZH[group]}最终指标热图]({_analysis_relpath(figure_files['heatmap'])})",
                "",
                "奖励函数表：",
                "",
                _markdown_table_raw(items, reward_cols),
                "",
                _markdown_table_raw(items, table_cols),
            ]
        )
    report.extend(
        [
            "",
            "## 总体判断",
            "",
            "1. 早期 QED/QED+SA+docking 证明 DDPO 链路能跑，但当时缺少 descriptor 监控，不能只按 reward 或 docking 字段判断分子质量。",
            "2. KL 扫描说明约束强度需要保留，但单靠 KL 不能解决 reward 方向偏差；后续更关键的是 reward 结构和分布护栏。",
            "3. ChEMBL prior、classifier、LE/logP 组合能改善某些目标指标，但多次训练都暴露高碳、低 TPSA、低杂原子漂移。",
            "4. classifier-only 是明确失败模式：有效率坍缩，说明分类器不能单独作为导航器。",
            "5. chemistry guard 能压住高碳但会牺牲 docking，说明 guard 应该做软约束/分段惩罚/权重扫描，而不是粗暴强压。",
            "6. reward 归因矩阵应继续作为诊断工具，而不是正式最优模型；它的价值是拆出每个 reward 的单独倾向。",
            "",
            "## 输出文件",
            "",
            "- `diffsbdd_rl_strict_major_training_runs.csv`：本报告纳入的 10 轮以上主要训练。",
            "- `strict_major_runs_by_group.png`：主线分组计数图。",
            "- `strict_major_epoch_timeline.png`：训练轮数时间线。",
            "- `strict_major_key_curve_panels.png`：代表性主线 run 的关键趋势图。",
            "- `strict_major_final_metric_heatmap.png`：所有主要 run 的最终指标热图。",
            "- `group_*_training_curves.png`：每条研究线的训练曲线。",
            "- `group_*_molecule_properties.png`：每条研究线的生成分子性质曲线。",
            "- `group_*_final_heatmap.png`：每条研究线的最终指标热图。",
            "- `diffsbdd_rl_principal_training_runs.csv`：更宽松的 curated 清单，保留短诊断 run 供追溯。",
            "",
            "## 验证",
            "",
            "```bash",
            "conda run -n dif python -m py_compile analysis/diffsbdd_rl_major_runs_20260618/collect_diffsbdd_rl_runs.py",
            "conda run -n dif python analysis/diffsbdd_rl_major_runs_20260618/collect_diffsbdd_rl_runs.py",
            "```",
            "",
            "结果：脚本通过语法检查并成功生成 CSV 和 PNG 图。运行时只有 matplotlib 本机缓存目录不可写导致的临时缓存提示，不影响输出。",
        ]
    )
    WORK_RECORD_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    rows = collect_runs()
    assign_run_aliases(rows)
    rows.sort(key=lambda row: str(row.get("stamp") or row.get("run_dir")))
    major = [row for row in rows if int(row["include_major"]) == 1]
    principal = [row for row in rows if int(row["include_principal"]) == 1]
    strict_major = [row for row in rows if int(row["include_strict_major"]) == 1]
    excluded = [row for row in rows if int(row["include_major"]) == 0]
    write_csv(ANALYSIS_DIR / "diffsbdd_rl_all_training_runs.csv", rows)
    write_csv(ANALYSIS_DIR / "diffsbdd_rl_major_training_runs.csv", major)
    write_csv(ANALYSIS_DIR / "diffsbdd_rl_principal_training_runs.csv", principal)
    write_csv(ANALYSIS_DIR / "diffsbdd_rl_strict_major_training_runs.csv", strict_major)
    write_csv(ANALYSIS_DIR / "diffsbdd_rl_excluded_runs.csv", excluded)
    write_plots(rows)
    write_strict_plots(rows)
    write_work_record(rows)


if __name__ == "__main__":
    main()
