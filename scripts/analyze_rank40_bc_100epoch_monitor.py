#!/usr/bin/env python3
"""Analyze the Rank40 BC checkpoint panel and write the durable technical report."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOGGER = logging.getLogger("rank40_bc_100epoch_analysis")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONDA = Path("/home/md666/miniforge3/bin/conda")
UMAP_MODEL = (
    PROJECT_ROOT
    / "analysis"
    / "benchmark"
    / "benchmark5"
    / "rnd53_umap_overlay_20260817"
    / "rebuild_compat_20260820_run9"
    / "old100k_portable_exact_umap_model.joblib"
)
CHEMBL_100K = PROJECT_ROOT / "data" / "chembl" / "chembl_36_100k_ecfp4_fingerprints.pkl.gz"
CHEMBL_QUALITY = PROJECT_ROOT / "analysis" / "chembl_standardized_all_quality_20260728"
COMPLETED_EPOCHS = tuple(range(10, 101, 10))


def _parse_args() -> argparse.Namespace:
    """Parse generation, training, evaluation, and report paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--poll-seconds", type=int, default=30)
    return parser.parse_args()


def _utc_now() -> str:
    """Return current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _wait_for_generation(generation_dir: Path, poll_seconds: int) -> None:
    """Wait for all ten fixed-attempt checkpoint batches."""
    status_path = generation_dir / "pipeline_status.json"
    last_marker: tuple[Any, Any] | None = None
    while True:
        status: dict[str, Any] = {}
        if status_path.is_file():
            status = json.loads(status_path.read_text(encoding="utf-8"))
        marker = (status.get("status"), status.get("current_completed_epoch"))
        if marker != last_marker:
            LOGGER.info("generation wait status=%s checkpoint=%s", *marker)
            last_marker = marker
        if status.get("status") == "completed" and int(status.get("checkpoint_count", 0)) == 10:
            return
        if status.get("status") == "failed":
            raise RuntimeError(f"generation pipeline failed: {status}")
        time.sleep(max(5, poll_seconds))


def _load_quality_module() -> ModuleType:
    """Load the existing ChEMBL-aligned raw-SDF quality implementation."""
    path = PROJECT_ROOT / "scripts" / "analyze_raw_sdf_quality_vs_chembl.py"
    spec = importlib.util.spec_from_file_location("rank40_quality_reference", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _source_rows(generation_dir: Path) -> pd.DataFrame:
    """Build the auditable source table for baseline and checkpoint batches."""
    rows: list[dict[str, Any]] = [
        {
            "repeat": "fixed_seeds_1_to_8",
            "model": "Original",
            "completed_epoch": 0,
            "source_sdf": str((generation_dir / "baseline_original" / "generated_1000_attempts.sdf").resolve()),
        }
    ]
    for epoch in COMPLETED_EPOCHS:
        source = generation_dir / f"epoch_{epoch:03d}" / "generated_1000_attempts.sdf"
        if not source.is_file():
            raise FileNotFoundError(source)
        rows.append(
            {
                "repeat": "fixed_seeds_1_to_8",
                "model": f"Epoch {epoch:03d}",
                "completed_epoch": epoch,
                "source_sdf": str(source.resolve()),
            }
        )
    return pd.DataFrame(rows)


def _run_quality_analysis(sources: pd.DataFrame, output_dir: Path, workers: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute fixed-denominator integrity and ChEMBL-aligned molecular descriptors."""
    module = _load_quality_module()
    raw_rows, jobs, facts = module._read_raw_sources(sources, None)
    LOGGER.info("quality feature jobs=%d raw_records=%d workers=%d", len(jobs), len(raw_rows), workers)
    full = module._compute_features(raw_rows, jobs, workers)
    full = full.merge(sources[["model", "completed_epoch"]], on="model", how="left", validate="many_to_one")
    full.to_csv(output_dir / "per_molecule_quality.csv.gz", index=False, compression="gzip")
    integrity = module._source_summary(full, facts, 1000)
    integrity = integrity.merge(sources[["model", "completed_epoch"]], on="model", how="left", validate="one_to_one")
    integrity = integrity.sort_values("completed_epoch")
    integrity.to_csv(output_dir / "checkpoint_integrity.csv", index=False)
    reference = module._load_reference_module()
    metric_summary = module._metric_summary(full, module._numeric_columns(reference, full))
    metric_summary = metric_summary[metric_summary["scope"] == "model_pooled"].copy()
    metric_summary = metric_summary.merge(sources[["model", "completed_epoch"]], on="model", how="left")
    metric_summary = metric_summary.sort_values(["metric", "completed_epoch"])
    metric_summary.to_csv(output_dir / "checkpoint_descriptor_summary_long.csv", index=False)
    tails = module._chembl_tail_rates(full, CHEMBL_QUALITY / "property_summary.csv")
    tails = tails[tails["scope"] == "model_pooled"].merge(
        sources[["model", "completed_epoch"]], on="model", how="left"
    )
    tails.to_csv(output_dir / "checkpoint_chembl_tail_rates.csv", index=False)
    return full, integrity, metric_summary


def _run_umap(sources: pd.DataFrame, output_dir: Path) -> Path:
    """Project every batch into one frozen 100k-ChEMBL ECFP4 UMAP space."""
    umap_dir = output_dir / "portable_umap"
    manifest = umap_dir / "analysis_manifest.json"
    if manifest.is_file() and json.loads(manifest.read_text(encoding="utf-8")).get("status") == "completed":
        LOGGER.info("reuse completed UMAP analysis=%s", umap_dir)
        return umap_dir
    if umap_dir.exists():
        raise FileExistsError(f"partial UMAP output requires inspection: {umap_dir}")
    command = [
        str(CONDA), "run", "-n", "dif", "python",
        str(PROJECT_ROOT / "scripts" / "analyze_iql_generation_portable_umap.py"),
        "--model", str(UMAP_MODEL),
        "--chembl", str(CHEMBL_100K),
        "--batch-root", str(output_dir),
        "--output", str(umap_dir),
    ]
    for row in sources.itertuples(index=False):
        label = "baseline_original" if row.completed_epoch == 0 else f"epoch_{int(row.completed_epoch):03d}"
        command.extend(["--batch", f"{label}={row.source_sdf}"])
    with (output_dir / "portable_umap.log").open("a", encoding="utf-8") as handle:
        subprocess.run(command, cwd=PROJECT_ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
    return umap_dir


def _style() -> None:
    """Apply presentation-readable static figure typography."""
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
        }
    )


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    """Save one plot as PNG and SVG."""
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def _training_plot(training_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Aggregate per-step local metrics into one training-epoch trend."""
    metrics = pd.read_csv(training_dir / "metrics" / "metrics.csv")
    grouped = metrics.groupby("epoch", as_index=False).agg(
        actor_loss_median=("actor_loss", "median"),
        actor_loss_mean=("actor_loss", "mean"),
        q_loss_mean=("q_loss", "mean"),
        value_loss_mean=("value_loss", "mean"),
        elapsed_seconds=("elapsed_seconds", "max"),
        global_step=("global_step", "max"),
    )
    grouped["completed_epoch"] = grouped["epoch"] + 1
    grouped.to_csv(output_dir / "training_epoch_summary.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    axes[0].plot(grouped["completed_epoch"], grouped["actor_loss_median"], color="#1565C0", linewidth=2)
    axes[0].set(title="BC actor loss by completed epoch", xlabel="Completed epoch", ylabel="Median step loss")
    axes[0].set_yscale("symlog", linthresh=1e-3)
    axes[0].grid(alpha=0.25)
    axes[1].plot(grouped["completed_epoch"], grouped["q_loss_mean"], label="Q compatibility loss", color="#EF6C00")
    axes[1].plot(grouped["completed_epoch"], grouped["value_loss_mean"], label="V compatibility loss", color="#2E7D32")
    axes[1].set(title="Shared-runtime critic diagnostics", xlabel="Completed epoch", ylabel="Mean loss")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    _save(fig, output_dir, "training_curve_summary")
    return grouped


def _integrity_plot(integrity: pd.DataFrame, output_dir: Path) -> None:
    """Plot fixed-attempt validity, uniqueness, and scaffold support."""
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.plot(integrity["completed_epoch"], 100 * integrity["sanitize_valid_per_attempt"], marker="o", label="Sanitize-valid / 1000")
    ax.plot(integrity["completed_epoch"], 100 * integrity["exact_unique_per_attempt"], marker="s", label="Unique valid / 1000")
    ax.plot(
        integrity["completed_epoch"],
        100 * integrity["unique_nonempty_murcko_scaffolds"] / integrity["attempts"],
        marker="^",
        label="Unique non-empty Murcko scaffolds / 1000",
    )
    ax.set(title="Fresh fixed-attempt generation integrity", xlabel="Completed epoch (0 = original)", ylabel="Percent of 1,000 attempts")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend()
    _save(fig, output_dir, "generation_integrity_trend")


def _descriptor_plots(metric_summary: pd.DataFrame, output_dir: Path) -> None:
    """Plot descriptor and structural-pathology drift against ChEMBL means."""
    chembl = pd.read_csv(CHEMBL_QUALITY / "property_summary.csv").set_index("property")
    panels = [
        ("carbon_fraction", "Carbon fraction"),
        ("num_heteroatoms", "Hetero atom count"),
        ("element_diversity", "Element diversity"),
        ("mol_log_p", "logP"),
        ("tpsa", "TPSA"),
        ("total_ring_count", "Ring count"),
        ("aromatic_ring_count", "Aromatic rings"),
        ("ring_3_count", "Three-member rings"),
        ("max_fused_ring_system_size", "Largest fused-ring system"),
        ("qed", "QED"),
        ("sa_score", "SA score (lower is easier)"),
        ("brenk_alert_count", "Brenk alert count"),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    for ax, (metric, title) in zip(axes.flat, panels):
        rows = metric_summary[metric_summary["metric"] == metric]
        ax.plot(rows["completed_epoch"], rows["mean"], marker="o", color="#1565C0")
        if metric in chembl.index:
            ax.axhline(float(chembl.loc[metric, "mean"]), color="#555555", linestyle="--", linewidth=1.3, label="ChEMBL mean")
        ax.set_title(title)
        ax.set_xlabel("Completed epoch")
        ax.grid(alpha=0.2)
    axes.flat[0].legend()
    _save(fig, output_dir, "descriptor_and_pathology_trends")


def _umap_plots(umap_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Create readable small multiples and ChEMBL-proximity trends."""
    rows = pd.read_csv(umap_dir / "per_molecule_portable_umap_metrics.csv.gz")
    summary = pd.read_csv(umap_dir / "batch_summary.csv")
    summary["completed_epoch"] = summary["batch"].map(
        lambda value: 0 if value == "baseline_original" else int(str(value).split("_")[-1])
    )
    summary = summary.sort_values("completed_epoch")
    summary.to_csv(output_dir / "checkpoint_umap_summary.csv", index=False)
    reference = np.load(umap_dir / "chembl_reference_umap_coordinates.npy")
    labels = [("baseline_original", "Original")] + [(f"epoch_{epoch:03d}", f"Epoch {epoch}") for epoch in COMPLETED_EPOCHS]
    fig, axes = plt.subplots(3, 4, figsize=(18, 13.2), sharex=True, sharey=True)
    for ax, (batch, title) in zip(axes.flat, labels):
        current = rows[rows["batch"] == batch]
        ax.scatter(reference[:, 0], reference[:, 1], s=0.6, alpha=0.06, color="#888888", rasterized=True)
        ax.scatter(current["umap1"], current["umap2"], s=8, alpha=0.55, color="#D84315", rasterized=True)
        ax.set_title(f"{title} (valid n={len(current)})")
    axes.flat[-1].axis("off")
    fig.supxlabel("Frozen UMAP 1")
    fig.supylabel("Frozen UMAP 2")
    fig.suptitle("Generated molecules in one frozen 100k-ChEMBL ECFP4 space", fontsize=18, y=1.01)
    _save(fig, output_dir, "chembl_umap_checkpoint_facets")
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.plot(summary["completed_epoch"], summary["nearest_chembl_tanimoto_mean"], marker="o", label="Nearest ChEMBL Tanimoto")
    ax.plot(summary["completed_epoch"], summary["chembl_tanimoto_20th_mean"], marker="s", label="20th-neighbor Tanimoto")
    ax.set(title="Chemical-space proximity to ChEMBL", xlabel="Completed epoch (0 = original)", ylabel="Mean ECFP4 Tanimoto")
    ax.grid(alpha=0.25)
    ax.legend()
    _save(fig, output_dir, "chembl_proximity_trend")
    return summary


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a small exact-value Markdown table without optional dependencies."""
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    lines = [
        "| " + " | ".join(map(str, display.columns)) + " |",
        "| " + " | ".join("---" for _ in display.columns) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def _write_report(
    report_path: Path,
    output_dir: Path,
    integrity: pd.DataFrame,
    metric_summary: pd.DataFrame,
    umap_summary: pd.DataFrame,
) -> None:
    """Write the final answer-first technical report and self-contained figures."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    figure_stems = [
        "training_curve_summary",
        "generation_integrity_trend",
        "descriptor_and_pathology_trends",
        "chembl_umap_checkpoint_facets",
        "chembl_proximity_trend",
    ]
    for stem in figure_stems:
        shutil.copy2(output_dir / f"{stem}.png", report_path.parent / f"{stem}.png")
    selected_metrics = ["carbon_fraction", "num_heteroatoms", "mol_log_p", "tpsa", "qed", "sa_score", "ring_3_count", "brenk_alert_count"]
    metric_wide = metric_summary[metric_summary["metric"].isin(selected_metrics)].pivot(
        index="completed_epoch", columns="metric", values="mean"
    ).reset_index()
    table = integrity[["completed_epoch", "sanitize_valid_per_attempt", "exact_unique_per_attempt"]].merge(metric_wide, on="completed_epoch")
    table = table.merge(
        umap_summary[["completed_epoch", "nearest_chembl_tanimoto_mean", "nearest_umap_reference_distance_mean"]],
        on="completed_epoch",
    )
    table["sanitize_valid_per_attempt"] *= 100
    table["exact_unique_per_attempt"] *= 100
    table = table.rename(
        columns={
            "completed_epoch": "完成轮次",
            "sanitize_valid_per_attempt": "有效率%",
            "exact_unique_per_attempt": "独特率%",
            "carbon_fraction": "碳占比",
            "num_heteroatoms": "杂原子数",
            "mol_log_p": "logP",
            "tpsa": "TPSA",
            "qed": "QED",
            "sa_score": "SA",
            "ring_3_count": "三元环",
            "brenk_alert_count": "Brenk",
            "nearest_chembl_tanimoto_mean": "最近ChEMBL相似度",
            "nearest_umap_reference_distance_mean": "UMAP最近距离",
        }
    )
    best_valid = integrity.loc[integrity["sanitize_valid_per_attempt"].idxmax()]
    trained_umap = umap_summary[umap_summary["completed_epoch"] > 0]
    closest = trained_umap.loc[trained_umap["nearest_chembl_tanimoto_mean"].idxmax()]
    original = integrity[integrity["completed_epoch"] == 0].iloc[0]
    lines = [
        "# Rank40 全好轨迹 BC：100 轮训练与每 10 轮生成评估",
        "",
        "## 技术摘要",
        "",
        f"本实验把终局排名前 40% 且通过不良分子剔除的 25,211 条轨迹全部视为好轨迹，只训练 DiffSBDD actor 的最后两个 EGNN block。训练共 100 轮；每 10 轮使用同一组 seed 1–8 做 1,000 次全新生成，并与未强化 Original 和冻结的 100k ChEMBL ECFP4/UMAP 空间比较。该实验是有意允许训练集泄漏和过拟合的 filtered BC 探索，不是通过 critic 优势加权的正式 IQL actor。",
        "",
        f"固定尝试口径下，最高 sanitize-valid 率出现在完成第 {int(best_valid['completed_epoch'])} 轮（{100 * float(best_valid['sanitize_valid_per_attempt']):.2f}%）；Original 为 {100 * float(original['sanitize_valid_per_attempt']):.2f}%。训练检查点中，平均最近 ChEMBL ECFP4 相似度最高的是第 {int(closest['completed_epoch'])} 轮（{float(closest['nearest_chembl_tanimoto_mean']):.4f}）。这两个最优点不一定重合，因此不能只凭训练 loss 或单一 UMAP 图选择最终 checkpoint。",
        "",
        "## 新鲜生成表现随轮次变化",
        "",
        "下图以 1,000 次生成尝试为统一分母，区分写入、sanitize-valid、精确唯一分子和 Murcko scaffold 支持。它是判断继续训练是否真正改善生成，而非仅拟合轨迹的首要证据。",
        "",
        "![固定尝试生成完整性](generation_integrity_trend.png)",
        "",
        _markdown_table(table),
        "",
        "## 训练本身是否稳定",
        "",
        "actor loss 跨 diffusion 时间点尺度差异很大，因此主图使用每轮 step loss 中位数并采用对称对数轴；Q/V 曲线仅是共享离线运行时的兼容性诊断，不参与 BC actor 的权重计算。",
        "",
        "![训练曲线](training_curve_summary.png)",
        "",
        "## 分子组成与结构病理是否漂移",
        "",
        "描述符只在 RDKit sanitize-valid 分子上计算。虚线为 ChEMBL36 全量标准化参考均值；重点监测全碳/高碳低杂原子倾向、杂原子数、元素多样性、logP、TPSA、三元环、稠合环、SA 与 Brenk 警报。均值接近 ChEMBL 不代表口袋适配或活性，只用于发现分布漂移和 reward-hacking 类失败。",
        "",
        "![描述符与结构病理](descriptor_and_pathology_trends.png)",
        "",
        "## 与 ChEMBL 的共同化学空间",
        "",
        "所有批次都通过同一个冻结 UMAP transform 投影到同一 100k ChEMBL ECFP4 空间；没有为各轮重新拟合坐标轴。小图中的灰色为 ChEMBL，橙色为该批次有效分子。UMAP 仅保留局部二维结构，结论必须与 ECFP4 Tanimoto 和描述符共同解释。",
        "",
        "![ChEMBL UMAP分面](chembl_umap_checkpoint_facets.png)",
        "",
        "最近邻和第 20 邻居 Tanimoto 分别反映是否贴近某个已知 ChEMBL 分子、以及是否进入较稠密的 ChEMBL 邻域。",
        "",
        "![ChEMBL邻近度](chembl_proximity_trend.png)",
        "",
        "## 数据、定义与方法",
        "",
        "- 训练数据：100,000 条完整 DiffSBDD 去噪轨迹中，67,694 条可做 Rank40 二分类；25,211 条 preferred 全部进入训练，42,483 条 non-preferred 被排除。",
        "- 模型：以原始 DiffSBDD checkpoint 为底座，只更新 dynamics EGNN block 3–4；BC 权重恒为 1，学习率 1e-6，8 GPU DDP，每轮 101 个固定 diffusion decision index。",
        "- 检查点：代码 epoch index 9/19/…/99 对应完成第 10/20/…/100 轮；生成统一导出 rank-0 同步 actor。",
        "- 生成：Original 和每个训练检查点都重新固定生成 1,000 次（8 GPU × 125），seed 为 1–8，timesteps=1000，resamplings=10，OpenBabel 重建。",
        "- 有效率：sanitize-valid 分子数 / 1,000 固定尝试；独特率：sanitize-valid canonical SMILES 去重数 / 1,000。描述符分母只含 sanitize-valid 分子。",
        "- ChEMBL：结构病理与描述符对照使用 ChEMBL36 标准化全量汇总；降维与近邻相似度使用冻结 100k ChEMBL ECFP4（radius=2, 2048 bits）参考。",
        "",
        "## 限制、不确定性与稳健性",
        "",
        "- 这是刻意允许泄漏/过拟合的探索性 filtered BC，不满足正式 IQL 的无泄漏 held-out critic gate。",
        "- 每个检查点只有一套固定 seed 1–8；跨轮可配对比较，但不能估计多重复随机方差。",
        "- 本报告尚未使用独立 docking、目标相互作用或实验活性作为选择依据；ChEMBL 接近度不是口袋条件性能。",
        "- UMAP 是二维可视化，距离会受 reducer 和训练参考影响；定量判断同时保留 ECFP4 Tanimoto、原始描述符和固定尝试有效率。",
        "- actor loss 不能直接排序生成质量；最终 checkpoint 必须由新鲜生成指标决定。",
        "",
        "## 建议的下一步",
        "",
        "1. 不直接默认第 100 轮最好；从有效率、唯一率、结构病理和 ChEMBL 邻近度的 Pareto 候选中保留 2–3 个 checkpoint。",
        "2. 对候选 checkpoint 与 Original 做同预算、独立 seed 的新 1,000 次生成，并补充 docking/相互作用筛选。",
        "3. 若某个中期 checkpoint 已达到峰值而后续恶化，则以该点作为 filtered BC 结果，并把继续训练导致的漂移作为负结果记录。",
        "",
        "## 仍需回答的问题",
        "",
        "- preferred 轨迹的终局改善能否在独立 seed 和独立 docking 上复现？",
        "- 最优有效率、最优结构安全性和最优 ChEMBL 邻近度是否指向同一 checkpoint？",
        "- filtered BC 相对 Original 的变化来自 pocket-conditioned 几何学习，还是主要来自终局分子组成偏置？",
        "",
        "## 支撑证据",
        "",
        f"- [检查点生成完整性]({output_dir / 'checkpoint_integrity.csv'})",
        f"- [逐分子质量指标]({output_dir / 'per_molecule_quality.csv.gz'})",
        f"- [描述符汇总]({output_dir / 'checkpoint_descriptor_summary_long.csv'})",
        f"- [ChEMBL尾部分布]({output_dir / 'checkpoint_chembl_tail_rates.csv'})",
        f"- [UMAP与相似度汇总]({output_dir / 'checkpoint_umap_summary.csv'})",
        f"- [训练逐轮汇总]({output_dir / 'training_epoch_summary.csv'})",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    """Wait, analyze, visualize, and write the self-contained work record."""
    args = _parse_args()
    training_dir = args.training_dir.resolve()
    generation_dir = args.generation_dir.resolve()
    output_dir = args.output_dir.resolve()
    report_path = args.report_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output_dir / "analysis.log", mode="a", encoding="utf-8")],
    )
    status_path = output_dir / "analysis_status.json"
    _atomic_json(status_path, {"status": "waiting_for_generation", "started_at": _utc_now()})
    try:
        _wait_for_generation(generation_dir, args.poll_seconds)
        _atomic_json(status_path, {"status": "analyzing_quality", "updated_at": _utc_now()})
        sources = _source_rows(generation_dir)
        sources.to_csv(output_dir / "sources.csv", index=False)
        _style()
        _, integrity, metric_summary = _run_quality_analysis(sources, output_dir, args.workers)
        _training_plot(training_dir, output_dir)
        _integrity_plot(integrity, output_dir)
        _descriptor_plots(metric_summary, output_dir)
        _atomic_json(status_path, {"status": "analyzing_umap", "updated_at": _utc_now()})
        umap_dir = _run_umap(sources, output_dir)
        umap_summary = _umap_plots(umap_dir, output_dir)
        chart_map = {
            "training_curve_summary": "BC actor and compatibility critic losses across 100 epochs",
            "generation_integrity_trend": "fixed-attempt validity, uniqueness, and scaffold support",
            "descriptor_and_pathology_trends": "molecular descriptor and structural-pathology drift versus ChEMBL means",
            "chembl_umap_checkpoint_facets": "common frozen ChEMBL UMAP small multiples",
            "chembl_proximity_trend": "nearest and twentieth-neighbor ECFP4 Tanimoto trends",
        }
        _atomic_json(output_dir / "chart_map.json", chart_map)
        _write_report(report_path, output_dir, integrity, metric_summary, umap_summary)
        summary = {
            "status": "completed",
            "checkpoint_batches": 10,
            "generated_attempts": 11000,
            "baseline_attempts": 1000,
            "report_path": str(report_path),
            "output_dir": str(output_dir),
            "completed_at": _utc_now(),
        }
        _atomic_json(status_path, summary)
        LOGGER.info("analysis complete report=%s", report_path)
        return 0
    except Exception as exc:
        LOGGER.exception("analysis monitor failed")
        _atomic_json(status_path, {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "updated_at": _utc_now()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
