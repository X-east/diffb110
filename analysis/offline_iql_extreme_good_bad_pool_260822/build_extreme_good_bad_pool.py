#!/usr/bin/env python3
"""Build a margin-separated good versus hard-bad DiffSBDD trajectory pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


LOGGER = logging.getLogger("extreme_good_bad_pool")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOOD_DISTANCE_MAX = 0.01541512126968382
BAD_DISTANCE_MIN = 0.03
REWARD_VERSION = "rank40_good_vs_structural_or_chembl_far_bad_margin_v1"


def _parse_args() -> argparse.Namespace:
    """Parse immutable source and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary-manifest",
        type=Path,
        default=PROJECT_ROOT
        / "analysis/offline_iql_binary_rank40_diagnosis_260821/binary_rank40_manifest.csv.gz",
    )
    parser.add_argument(
        "--descriptor-facts",
        type=Path,
        default=PROJECT_ROOT
        / "analysis/offline_iql_100k_reasonableness_schemes_260819/expanded_reasonableness_facts.csv.gz",
    )
    parser.add_argument(
        "--source-sidecar",
        type=Path,
        default=PROJECT_ROOT
        / "analysis/offline_iql_filtered_bc_rank40_260822/inputs/binary_reward_sidecar.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "analysis/offline_iql_extreme_good_bad_pool_260822/inputs",
    )
    parser.add_argument(
        "--work-record",
        type=Path,
        default=PROJECT_ROOT
        / "work_records/26-8/260822/260822-04 极端好坏轨迹二分类数据池.md",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    """Return the SHA256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON artifact atomically."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_ids(path: Path, values: list[str]) -> None:
    """Write a stable sorted sample-ID list."""
    path.write_text(json.dumps(sorted(values), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_sources(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and one-to-one join frozen labels with endpoint descriptors."""
    labels = pd.read_csv(args.binary_manifest, compression="gzip")
    facts = pd.read_csv(args.descriptor_facts, compression="gzip", low_memory=False)
    if labels["trajectory_id"].duplicated().any() or facts["trajectory_id"].duplicated().any():
        raise ValueError("source trajectory IDs must be unique")
    if len(labels) != 67_694:
        raise ValueError(f"unexpected valid label count: {len(labels)}")
    selected_fact_columns = [
        "trajectory_id",
        "qed",
        "sa_raw",
        "logp",
        "tpsa",
        "mol_wt",
        "heavy_atom_count",
        "carbon_fraction",
        "three_member_ring_count",
        "fused_ring_count",
        "max_fused_ring_system_size",
        "complex_ring_count",
        "total_ring_count",
        "aromatic_ring_count",
        "hetero_atom_count",
        "brenk_count",
        "pains_count",
        "nih_count",
        "umap_chembl_k20_distance",
        "umap_chembl_mahalanobis",
    ]
    merged = labels.merge(
        facts[selected_fact_columns],
        on="trajectory_id",
        how="left",
        validate="one_to_one",
    )
    if merged[selected_fact_columns[1:]].isna().all(axis=1).any():
        raise ValueError("descriptor join left unmatched valid trajectories")
    return labels, merged


def _label_extremes(frame: pd.DataFrame) -> pd.DataFrame:
    """Assign good, hard-bad, or excluded-middle labels with a fixed margin."""
    good = (
        frame["binary_label"].eq("preferred")
        & ~frame["structural_hard_bad"].astype(bool)
        & frame["umap_nearest_chembl_distance"].le(GOOD_DISTANCE_MAX)
    )
    structural_bad = frame["structural_hard_bad"].astype(bool)
    chembl_far_bad = frame["umap_nearest_chembl_distance"].gt(BAD_DISTANCE_MIN)
    hard_bad = frame["binary_label"].eq("non_preferred") & (structural_bad | chembl_far_bad)
    if bool((good & hard_bad).any()):
        raise RuntimeError("good and hard-bad labels overlap")
    labeled = frame.copy()
    labeled["extreme_label"] = "excluded_middle"
    labeled["extreme_target"] = 0.0
    labeled.loc[good, ["extreme_label", "extreme_target"]] = ["good", 1.0]
    labeled.loc[hard_bad, ["extreme_label", "extreme_target"]] = ["hard_bad", -1.0]
    labeled["structural_bad_component"] = structural_bad
    labeled["chembl_far_bad_component"] = chembl_far_bad
    labeled["hard_bad_category"] = ""
    labeled.loc[hard_bad & structural_bad & ~chembl_far_bad, "hard_bad_category"] = "structural_only"
    labeled.loc[hard_bad & ~structural_bad & chembl_far_bad, "hard_bad_category"] = "chembl_far_only"
    labeled.loc[hard_bad & structural_bad & chembl_far_bad, "hard_bad_category"] = "structural_and_chembl_far"
    return labeled


def _build_sidecar(
    source_path: Path,
    output_path: Path,
    good_ids: set[str],
    bad_ids: set[str],
) -> None:
    """Create a full-index +1/-1/0 sidecar without changing schema semantics."""
    sidecar = pd.read_csv(source_path)
    if sidecar["sample_id"].duplicated().any() or len(sidecar) != 100_000:
        raise ValueError("source sidecar must contain 100,000 unique episodes")
    known = set(sidecar["sample_id"].astype(str))
    missing = sorted((good_ids | bad_ids) - known)
    if missing:
        raise ValueError(f"selected IDs absent from source sidecar: {missing[:5]}")
    sidecar["reward_version"] = REWARD_VERSION
    sidecar["reward_total_train"] = 0.0
    sidecar["reward_total_raw"] = 0.0
    sidecar["failure_reason"] = "excluded_middle_or_invalid"
    good_mask = sidecar["sample_id"].isin(good_ids)
    bad_mask = sidecar["sample_id"].isin(bad_ids)
    sidecar.loc[good_mask, ["reward_total_train", "reward_total_raw", "failure_reason"]] = [
        1.0,
        1.0,
        "extreme_good",
    ]
    sidecar.loc[bad_mask, ["reward_total_train", "reward_total_raw", "failure_reason"]] = [
        -1.0,
        -1.0,
        "extreme_hard_bad",
    ]
    if int(good_mask.sum()) != len(good_ids) or int(bad_mask.sum()) != len(bad_ids):
        raise RuntimeError("sidecar label count mismatch")
    sidecar.to_csv(output_path, index=False)


def _write_plots(frame: pd.DataFrame, output_dir: Path) -> list[str]:
    """Plot class composition and the explicit ChEMBL margin."""
    palette = {"good": "#2E7D32", "hard_bad": "#C62828"}
    plotted = frame[frame["extreme_label"].isin(["good", "hard_bad"])]
    split_counts = pd.crosstab(plotted["split"], plotted["extreme_label"]).reindex(
        index=["train", "validation", "test"], columns=["good", "hard_bad"], fill_value=0
    )
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    split_counts.plot.bar(ax=ax, color=[palette["good"], palette["hard_bad"]], rot=0)
    ax.set_title("Margin-separated trajectory labels", fontsize=20)
    ax.set_xlabel("Frozen scaffold split", fontsize=15)
    ax.set_ylabel("Valid complete trajectories", fontsize=15)
    ax.tick_params(labelsize=12)
    ax.legend(title="Extreme label", fontsize=11, title_fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "extreme_class_counts.png", dpi=200)
    fig.savefig(output_dir / "extreme_class_counts.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for label in ("good", "hard_bad"):
        values = plotted.loc[plotted["extreme_label"].eq(label), "umap_nearest_chembl_distance"]
        ax.hist(values, bins=70, density=True, alpha=0.48, color=palette[label], label=f"{label} (n={len(values):,})")
    ax.axvline(GOOD_DISTANCE_MAX, color="#2E7D32", linestyle="--", linewidth=2, label=f"good max={GOOD_DISTANCE_MAX:.4f}")
    ax.axvline(BAD_DISTANCE_MIN, color="#C62828", linestyle="--", linewidth=2, label=f"far-bad min={BAD_DISTANCE_MIN:.2f}")
    ax.set_title("Nearest-ChEMBL UMAP distance with an excluded margin", fontsize=20)
    ax.set_xlabel("Nearest ChEMBL UMAP distance", fontsize=15)
    ax.set_ylabel("Density", fontsize=15)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_dir / "extreme_distance_margin.png", dpi=200)
    fig.savefig(output_dir / "extreme_distance_margin.svg")
    plt.close(fig)

    bad_counts = (
        plotted.loc[plotted["extreme_label"].eq("hard_bad"), "hard_bad_category"]
        .value_counts()
        .reindex(["structural_only", "chembl_far_only", "structural_and_chembl_far"])
    )
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    bad_counts.plot.bar(ax=ax, color=["#A93226", "#D35400", "#7D3C98"], rot=0)
    ax.set_title("Hard-bad trajectory composition", fontsize=20)
    ax.set_xlabel("")
    ax.set_ylabel("Trajectories", fontsize=15)
    ax.tick_params(labelsize=12)
    ax.grid(axis="y", alpha=0.25)
    for patch in ax.patches:
        ax.annotate(f"{int(patch.get_height()):,}", (patch.get_x() + patch.get_width() / 2, patch.get_height()), ha="center", va="bottom", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / "hard_bad_composition.png", dpi=200)
    fig.savefig(output_dir / "hard_bad_composition.svg")
    plt.close(fig)
    return ["extreme_class_counts.png", "extreme_distance_margin.png", "hard_bad_composition.png"]


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table."""
    lines = [
        "| " + " | ".join(map(str, frame.columns)) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _write_record(
    path: Path,
    output_dir: Path,
    summary_table: pd.DataFrame,
    images: list[str],
    summary: dict[str, Any],
) -> None:
    """Write the self-contained work record and copy local images."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for image in images:
        shutil.copy2(output_dir / image, path.parent / image)
    lines = [
        "# 极端好/坏轨迹二分类数据池",
        "",
        "## 目的",
        "",
        "此前仅用 25,211 条好轨迹做恒权 BC，actor 只是在模仿原策略，100 轮生成几乎未改善。本数据池只保留标签明确的好坏两端，排除中间模糊样本，为 critic/分类器或带正负方向的 actor 目标提供对比。",
        "",
        "## 固定规则",
        "",
        f"- 好：原 Rank40 preferred，最近 ChEMBL UMAP 距离 ≤ {GOOD_DISTANCE_MAX:.6f}，且 `structural_hard_bad=False`。",
        f"- 坏：原 non-preferred，且 `structural_hard_bad=True` 或最近 ChEMBL UMAP 距离 > {BAD_DISTANCE_MIN:.2f}。",
        "- 排除：其余有效中间样本及所有无效终局轨迹。",
        "- 结构硬坏沿用既有审计规则：复杂环 >1、稠合环 >4、高碳或异常元素；未引入新的直觉式分子规则。",
        "",
        _markdown_table(summary_table),
        "",
        "只有 good 和 hard_bad 进入极端二分类池；excluded_middle 不进入模型投喂。",
        "",
        "![类别与冻结split计数](extreme_class_counts.png)",
        "",
        "距离阈值之间保留显式空白区，防止近边界样本把监督信号冲淡。结构硬坏无论距离多近都归入 hard_bad。",
        "",
        "![ChEMBL距离margin](extreme_distance_margin.png)",
        "",
        "坏轨迹由结构异常、ChEMBL 远端及二者重叠三部分组成。",
        "",
        "![坏轨迹构成](hard_bad_composition.png)",
        "",
        "## 训练使用限制",
        "",
        "- 普通 BC 不能同时模仿 good 和 hard_bad。",
        "- critic/二分类器可直接使用 `extreme_target=+1/-1`。",
        "- actor 必须使用能区分正负方向的目标，或由 critic/分类器产生权重；不能把 hard_bad 当作 NLL 正样本。",
        "- 当前 sidecar 对 good/hard_bad/未使用样本分别写入 +1/-1/0，但是否进入采样仍应由两个 ID 列表显式控制。",
        "",
        "## 证据文件",
        "",
        f"- [完整极端池 manifest]({output_dir / 'extreme_good_bad_manifest.csv.gz'})",
        f"- [仅投喂的 good/bad manifest]({output_dir / 'selected_extremes_manifest.csv.gz'})",
        f"- [Good IDs]({output_dir / 'good_sample_ids.json'})",
        f"- [Hard-bad IDs]({output_dir / 'hard_bad_sample_ids.json'})",
        f"- [+1/-1/0 reward sidecar]({output_dir / 'extreme_reward_sidecar.csv'})",
        f"- [机器可读摘要]({output_dir / 'summary.json'})",
        "",
        f"数据来源与哈希记录在 `summary.json`；最终选择 good={summary['good_rows']:,}、hard_bad={summary['hard_bad_rows']:,}。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    """Build, validate, visualize, and record the extreme trajectory pool."""
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output_dir / "build.log", encoding="utf-8")],
    )
    LOGGER.info("load frozen manifests")
    _, merged = _load_sources(args)
    labeled = _label_extremes(merged)
    selected = labeled[labeled["extreme_label"].isin(["good", "hard_bad"])].copy()
    good_ids = set(selected.loc[selected["extreme_label"].eq("good"), "trajectory_id"].astype(str))
    bad_ids = set(selected.loc[selected["extreme_label"].eq("hard_bad"), "trajectory_id"].astype(str))
    if good_ids & bad_ids:
        raise RuntimeError("good and hard-bad ID sets overlap")
    labeled.to_csv(output_dir / "extreme_good_bad_manifest.csv.gz", index=False, compression="gzip")
    selected.to_csv(output_dir / "selected_extremes_manifest.csv.gz", index=False, compression="gzip")
    _write_ids(output_dir / "good_sample_ids.json", list(good_ids))
    _write_ids(output_dir / "hard_bad_sample_ids.json", list(bad_ids))
    _build_sidecar(args.source_sidecar, output_dir / "extreme_reward_sidecar.csv", good_ids, bad_ids)
    images = _write_plots(labeled, output_dir)

    summary_table = (
        pd.crosstab(labeled["split"], labeled["extreme_label"])
        .reindex(index=["train", "validation", "test"], columns=["good", "hard_bad", "excluded_middle"], fill_value=0)
        .reset_index()
    )
    summary_table.loc[len(summary_table)] = [
        "total",
        int(labeled["extreme_label"].eq("good").sum()),
        int(labeled["extreme_label"].eq("hard_bad").sum()),
        int(labeled["extreme_label"].eq("excluded_middle").sum()),
    ]
    summary_table.to_csv(output_dir / "class_split_counts.csv", index=False)
    category_counts = selected.loc[selected["extreme_label"].eq("hard_bad"), "hard_bad_category"].value_counts()
    summary = {
        "status": "completed",
        "rule_version": REWARD_VERSION,
        "good_distance_max": GOOD_DISTANCE_MAX,
        "bad_distance_min_strict": BAD_DISTANCE_MIN,
        "explicit_distance_margin": [GOOD_DISTANCE_MAX, BAD_DISTANCE_MIN],
        "valid_source_rows": int(len(labeled)),
        "good_rows": len(good_ids),
        "hard_bad_rows": len(bad_ids),
        "excluded_middle_valid_rows": int(labeled["extreme_label"].eq("excluded_middle").sum()),
        "selected_rows": int(len(selected)),
        "class_ratio_bad_to_good": len(bad_ids) / len(good_ids),
        "good_bad_overlap": 0,
        "hard_bad_categories": {str(key): int(value) for key, value in category_counts.items()},
        "source_references": {
            "binary_manifest": str(args.binary_manifest.resolve()),
            "binary_manifest_sha256": _sha256(args.binary_manifest),
            "descriptor_facts": str(args.descriptor_facts.resolve()),
            "descriptor_facts_sha256": _sha256(args.descriptor_facts),
            "source_sidecar": str(args.source_sidecar.resolve()),
            "source_sidecar_sha256": _sha256(args.source_sidecar),
            "borrowed_rules": "existing rank40 train-frozen threshold and existing structural_hard_bad audit",
            "project_difference": "adds a >0.03 far-negative threshold and excludes all valid middle samples",
            "expected_benefit": "stronger endpoint label margin for good/bad state separability",
            "risk": "terminal labels may still be weakly identifiable from intermediate geometric states",
            "acceptance_test": "held-out timestep-wise good/bad AUC and fresh-generation comparison",
        },
        "outputs": {},
    }
    for name in [
        "extreme_good_bad_manifest.csv.gz",
        "selected_extremes_manifest.csv.gz",
        "good_sample_ids.json",
        "hard_bad_sample_ids.json",
        "extreme_reward_sidecar.csv",
        "class_split_counts.csv",
    ]:
        path = output_dir / name
        summary["outputs"][name] = {"path": str(path), "sha256": _sha256(path)}
    _atomic_json(output_dir / "summary.json", summary)
    _write_record(args.work_record.resolve(), output_dir, summary_table, images, summary)
    LOGGER.info(
        "completed good=%d hard_bad=%d excluded_middle=%d",
        len(good_ids),
        len(bad_ids),
        summary["excluded_middle_valid_rows"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
