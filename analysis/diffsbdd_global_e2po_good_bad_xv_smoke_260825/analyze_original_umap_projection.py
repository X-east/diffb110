#!/usr/bin/env python3
"""Analyze four generation batches projected by one frozen historical UMAP."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


ORDER = ["baseline_original", "x_only", "v_only", "native_joint"]
DISPLAY = {
    "baseline_original": "Original",
    "x_only": "x_only",
    "v_only": "v_only",
    "native_joint": "native_joint",
}
COLORS = {
    "baseline_original": "#6B7280",
    "x_only": "#2563EB",
    "v_only": "#D97706",
    "native_joint": "#B91C1C",
}


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nearest_neighbor_stats(coords: np.ndarray) -> dict[str, float]:
    """Summarize within-batch 2D nearest-neighbor spacing."""
    if len(coords) < 3:
        return {"nn1_p05": float("nan"), "nn1_median": float("nan"), "nn5_median": float("nan")}
    k = min(6, len(coords))
    distances = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=1).fit(coords).kneighbors(coords, return_distance=True)[0]
    return {
        "nn1_p05": float(np.quantile(distances[:, 1], 0.05)),
        "nn1_median": float(np.median(distances[:, 1])),
        "nn5_median": float(np.median(distances[:, min(5, k - 1)])),
    }


def occupied_grid(coords: np.ndarray, reference: np.ndarray, bins: int = 50) -> tuple[int, int, float, float]:
    """Measure reference-grid occupancy and the fraction inside its bounding box."""
    low = reference.min(axis=0)
    high = reference.max(axis=0)
    scale = np.where(high > low, high - low, 1.0)
    ref_index = np.floor((reference - low) / scale * bins).astype(int)
    query_index = np.floor((coords - low) / scale * bins).astype(int)
    ref_index = np.clip(ref_index, 0, bins - 1)
    inside = np.all((coords >= low) & (coords <= high), axis=1)
    query_inside = query_index[inside]
    ref_cells = set(map(tuple, ref_index.tolist()))
    query_cells = set(map(tuple, query_inside.tolist()))
    return len(ref_cells), len(query_cells), float(len(query_cells) / len(ref_cells)), float(inside.mean())


def summarize(molecules: pd.DataFrame, reference: np.ndarray) -> pd.DataFrame:
    """Compute batch-level UMAP displacement, spread, coverage, and density facts."""
    baseline = molecules[molecules["batch"].eq("baseline_original")][["umap1", "umap2"]].to_numpy(float)
    baseline_centroid = baseline.mean(axis=0)
    ref_cells, _, _, _ = occupied_grid(reference, reference)
    rows: list[dict[str, object]] = []
    for batch in ORDER:
        current_frame = molecules[molecules["batch"].eq(batch)].copy()
        coords = current_frame[["umap1", "umap2"]].to_numpy(float)
        centroid = coords.mean(axis=0)
        covariance = np.cov(coords, rowvar=False) if len(coords) > 1 else np.zeros((2, 2))
        eigenvalues = np.linalg.eigvalsh(covariance)
        eigenvalues = np.clip(eigenvalues, 0.0, None)
        ref_count, cells, cell_fraction, bbox_fraction = occupied_grid(coords, reference)
        row: dict[str, object] = {
            "batch": batch,
            "method": DISPLAY[batch],
            "valid": int(len(coords)),
            "centroid_umap1": float(centroid[0]),
            "centroid_umap2": float(centroid[1]),
            "centroid_shift_vs_original": float(np.linalg.norm(centroid - baseline_centroid)),
            "umap1_std": float(np.std(coords[:, 0])),
            "umap2_std": float(np.std(coords[:, 1])),
            "covariance_ellipse_area_1sigma": float(np.pi * np.sqrt(np.prod(eigenvalues))),
            "reference_grid_cells": int(ref_count),
            "occupied_reference_grid_cells": int(cells),
            "reference_grid_coverage": float(cell_fraction),
            "inside_reference_bbox_rate": float(bbox_fraction),
            "nearest_reference_distance_median": float(np.median(current_frame["nearest_umap_reference_distance"])),
            "nearest_reference_distance_p95": float(np.quantile(current_frame["nearest_umap_reference_distance"], 0.95)),
        }
        row.update(nearest_neighbor_stats(coords))
        rows.append(row)
    summary = pd.DataFrame(rows)
    original_nn = summary.loc[summary["batch"].eq("baseline_original"), "nn1_median"].iloc[0]
    original_area = summary.loc[summary["batch"].eq("baseline_original"), "covariance_ellipse_area_1sigma"].iloc[0]
    summary["nn1_median_ratio_vs_original"] = summary["nn1_median"] / original_nn
    summary["spread_area_ratio_vs_original"] = summary["covariance_ellipse_area_1sigma"] / original_area
    return summary


def save_plot(fig: plt.Figure, output: Path, stem: str) -> None:
    """Save a presentation-readable PNG and editable SVG."""
    fig.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_overview(molecules: pd.DataFrame, reference: np.ndarray, summary: pd.DataFrame, output: Path) -> None:
    """Draw one shared-coordinate overview with method centroids."""
    fig, ax = plt.subplots(figsize=(10.5, 8.0))
    ax.scatter(reference[:, 0], reference[:, 1], s=2, c="#D1D5DB", alpha=0.16, linewidths=0, label="ChEMBL 100k reference")
    for batch in ORDER:
        current = molecules[molecules["batch"].eq(batch)]
        ax.scatter(current["umap1"], current["umap2"], s=13, alpha=0.62, c=COLORS[batch], linewidths=0, label=DISPLAY[batch])
        row = summary[summary["batch"].eq(batch)].iloc[0]
        ax.scatter(row["centroid_umap1"], row["centroid_umap2"], s=95, c=COLORS[batch], edgecolor="black", linewidth=0.8, marker="X")
    ax.set_title("Four DiffSBDD preference variants on the original frozen UMAP", fontsize=16, pad=12)
    ax.set_xlabel("UMAP1", fontsize=13)
    ax.set_ylabel("UMAP2", fontsize=13)
    ax.legend(frameon=False, fontsize=10, ncol=2, loc="best")
    ax.grid(alpha=0.15)
    save_plot(fig, output, "umap_projection_overview_original_model")


def plot_facets(molecules: pd.DataFrame, reference: np.ndarray, summary: pd.DataFrame, output: Path) -> None:
    """Draw method-specific panels while retaining identical limits."""
    all_coords = np.vstack([reference, molecules[["umap1", "umap2"]].to_numpy(float)])
    low = np.quantile(all_coords, 0.002, axis=0)
    high = np.quantile(all_coords, 0.998, axis=0)
    margin = (high - low) * 0.04
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=True)
    for ax, batch in zip(axes.ravel(), ORDER):
        current = molecules[molecules["batch"].eq(batch)]
        row = summary[summary["batch"].eq(batch)].iloc[0]
        ax.scatter(reference[:, 0], reference[:, 1], s=1.8, c="#D1D5DB", alpha=0.14, linewidths=0)
        ax.scatter(current["umap1"], current["umap2"], s=14, c=COLORS[batch], alpha=0.65, linewidths=0)
        ax.scatter(row["centroid_umap1"], row["centroid_umap2"], s=100, c=COLORS[batch], edgecolor="black", linewidth=0.8, marker="X")
        ax.set_title(f"{DISPLAY[batch]} (n={int(row['valid'])})", fontsize=15)
        ax.grid(alpha=0.15)
        ax.set_xlim(low[0] - margin[0], high[0] + margin[0])
        ax.set_ylim(low[1] - margin[1], high[1] + margin[1])
    axes[1, 0].set_xlabel("UMAP1", fontsize=13)
    axes[1, 1].set_xlabel("UMAP1", fontsize=13)
    axes[0, 0].set_ylabel("UMAP2", fontsize=13)
    axes[1, 0].set_ylabel("UMAP2", fontsize=13)
    fig.suptitle("Same original UMAP coordinate system; gray = ChEMBL reference", fontsize=17, y=0.995)
    fig.tight_layout()
    save_plot(fig, output, "umap_projection_facets_original_model")


def plot_metrics(summary: pd.DataFrame, output: Path) -> None:
    """Draw displacement, spread, and local-density diagnostics."""
    labels = [DISPLAY[x] for x in ORDER]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    panels = [
        ("centroid_shift_vs_original", "Centroid shift vs Original", "UMAP distance"),
        ("spread_area_ratio_vs_original", "1-sigma spread ratio", "Ratio"),
        ("nn1_median_ratio_vs_original", "Nearest-neighbor spacing ratio", "Ratio"),
    ]
    for ax, (column, title, ylabel) in zip(axes, panels):
        values = summary[column].to_numpy(float)
        ax.bar(x, values, color=[COLORS[b] for b in ORDER], alpha=0.82)
        ax.axhline(1.0 if "ratio" in column else 0.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(title, fontsize=13)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save_plot(fig, output, "umap_projection_metrics_original_model")


def write_report(summary: pd.DataFrame, output: Path, model: Path, chembl: Path, molecules_path: Path) -> None:
    """Write a concise evidence-bounded Chinese report."""
    table = summary[[
        "method", "valid", "centroid_shift_vs_original", "reference_grid_coverage",
        "inside_reference_bbox_rate", "nearest_reference_distance_median",
        "covariance_ellipse_area_1sigma", "nn1_median", "nn1_median_ratio_vs_original",
    ]].copy()
    table.columns = ["方法", "有效数", "质心位移", "参考网格覆盖", "参考包围盒内比例", "最近参考距离中位数", "1sigma面积", "最近邻中位数", "最近邻/Original"]
    table["质心位移"] = table["质心位移"].map(lambda x: f"{x:.4f}")
    table["参考网格覆盖"] = table["参考网格覆盖"].map(lambda x: f"{x:.1%}")
    table["参考包围盒内比例"] = table["参考包围盒内比例"].map(lambda x: f"{x:.1%}")
    table["最近参考距离中位数"] = table["最近参考距离中位数"].map(lambda x: f"{x:.5f}")
    table["1sigma面积"] = table["1sigma面积"].map(lambda x: f"{x:.3f}")
    table["最近邻中位数"] = table["最近邻中位数"].map(lambda x: f"{x:.5f}")
    table["最近邻/Original"] = table["最近邻/Original"].map(lambda x: f"{x:.3f}")
    table_header = "| " + " | ".join(table.columns) + " |"
    table_rule = "| " + " | ".join(["---"] * len(table.columns)) + " |"
    table_rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in table.itertuples(index=False, name=None)
    ]
    markdown_table = "\n".join([table_header, table_rule, *table_rows])
    lines = [
        "# 原始 UMAP 模型统一投影分析",
        "",
        "## 结论",
        "",
        "四组分子已经在同一个历史 frozen UMAP 坐标系中完成投影。相对 Original，`x_only` 的整体位置和离散度最接近；`v_only` 与 `native_joint` 明显向另一个化学空间区域偏移，但这两个变体的局部 UMAP 间距没有显示出强烈的整体性塌缩证据。结合前面的高碳/低 TPSA 与 guard violation 漂移，这个偏移更应解释为组成/表示捷径导致的化学空间迁移，而不是质量改善。",
        "",
        "## 统一投影规范",
        "",
        f"- frozen 模型：`{model}`",
        f"- 模型 SHA-256：`{sha256(model)}`",
        f"- reference：`{chembl}`；ECFP4，Morgan radius=2，2048 bits",
        "- 本次只调用模型的 `transform`，没有 `fit` 或重新拟合；四组均以 1000 attempts 为固定分母，只有 RDKit 有效分子进入 UMAP 统计。",
        "- 该 portable 模型的 embedding 中包含历史 fit 样本；本分析沿用原项目脚本，只将其前 100,000 行作为 ChEMBL reference overlay，保证与此前生成监控同坐标系。",
        "",
        "## 结果",
        "",
        markdown_table,
        "",
        "![统一坐标系总览](umap_projection_overview_original_model.png)",
        "",
        "![四组分面图](umap_projection_facets_original_model.png)",
        "",
        "![位移、离散度和局部间距](umap_projection_metrics_original_model.png)",
        "",
        "## 解读",
        "",
        "- `x_only`：质心位移和参考空间统计最接近 Original；目前没有看到明显的 UMAP 空间漂移。",
        "- `v_only`：质心明显迁移，TPSA 已从约 61.94 降到约 39.24，碳比例从约 0.747 升到约 0.833；UMAP 迁移与组成漂移方向一致，不能把 docking 改善单独归因于有效偏好学习。",
        "- `native_joint`：迁移幅度与 `v_only` 同类，且最近参考距离中位数更高；它仍可作为候选，但必须先通过 representation-balanced、组成约束和独立质量门控。",
        "- 局部塌缩判据仅使用同批次二维最近邻间距和 1-sigma 协方差面积。UMAP 是非线性可视化，局部间距变化只能作为预警，不能替代 ECFP4 多样性、scaffold、有效性、结构病理和 docking 的独立检验。",
        "",
        "## 原始证据",
        "",
        f"- 分子级投影：`{molecules_path}`",
        "- 批次汇总：`batch_summary.csv`",
        "- 位移/覆盖/密度汇总：`umap_projection_summary.csv`",
        "- UMAP reference 坐标：`chembl_reference_umap_coordinates.npy`",
    ]
    (output / "original_umap_projection_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """Create statistics, figures, and report for one frozen UMAP projection."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--chembl", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    molecules_path = args.input_dir / "per_molecule_portable_umap_metrics.csv.gz"
    molecules = pd.read_csv(molecules_path)
    reference = np.load(args.input_dir / "chembl_reference_umap_coordinates.npy")
    summary = summarize(molecules, reference)
    summary.to_csv(args.output_dir / "umap_projection_summary.csv", index=False)
    plot_overview(molecules, reference, summary, args.output_dir)
    plot_facets(molecules, reference, summary, args.output_dir)
    plot_metrics(summary, args.output_dir)
    manifest = {
        "status": "complete",
        "model": str(args.model.resolve()),
        "model_sha256": sha256(args.model),
        "chembl": str(args.chembl.resolve()),
        "input_projection_manifest": str((args.input_dir / "analysis_manifest.json").resolve()),
        "valid_rows": int(len(molecules)),
        "reference_rows": int(len(reference)),
        "methods": [DISPLAY[x] for x in ORDER],
        "fit_performed": False,
        "interpretation_boundary": "UMAP coordinate, coverage, distance, and local-spacing metrics are descriptive only.",
    }
    (args.output_dir / "projection_analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_report(summary, args.output_dir, args.model, args.chembl, molecules_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
