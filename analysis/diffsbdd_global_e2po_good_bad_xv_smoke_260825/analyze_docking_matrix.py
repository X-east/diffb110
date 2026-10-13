"""Merge UniDock results with fixed-attempt generation metrics."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
GEN_ROOT = ROOT / "generation_comparison"
VARIANTS = ("original", "x_only", "v_only", "native_joint")


def _float(value: object) -> float:
    """Convert a CSV value to a finite float when possible."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _read_metrics() -> dict[tuple[str, int], dict[str, str]]:
    """Read per-attempt generation metrics keyed by method and attempt."""
    rows: dict[tuple[str, int], dict[str, str]] = {}
    with (ROOT / "generation_molecule_metrics.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[(row["variant"], int(row["attempt"]))] = row
    return rows


def _read_docking(variant: str) -> dict[int, dict[str, str]]:
    """Read one UniDock result CSV keyed by zero-based benchmark index."""
    path = GEN_ROOT / variant / "docking" / "docking_results.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return {int(row["benchmark_index"]): row for row in csv.DictReader(handle)}


def merge_rows() -> list[dict[str, object]]:
    """Merge generation and docking fields without changing fixed denominators."""
    generation = _read_metrics()
    merged: list[dict[str, object]] = []
    for variant in VARIANTS:
        docking = _read_docking(variant)
        for attempt in range(1, 1001):
            generation_row = generation[(variant, attempt)]
            dock_row = docking[attempt - 1]
            score = _float(dock_row.get("score")) if dock_row.get("success") == "True" else float("nan")
            heavy = _float(generation_row.get("heavy_atom_count"))
            ligand_efficiency = (-score / heavy) if math.isfinite(score) and heavy > 0 else float("nan")
            merged.append(
                {
                    **generation_row,
                    "docking_success": int(dock_row.get("success") == "True"),
                    "docking_score": score,
                    "docking_ligand_efficiency": ligand_efficiency,
                    "docking_failure_reason": dock_row.get("failure_reason", ""),
                    "docking_fragments_original": dock_row.get("fragments_original", ""),
                }
            )
    return merged


def _valid_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Select RDKit-valid rows with a finite docking score."""
    return [
        row for row in rows
        if int(row["rdkit_valid"]) == 1 and int(row["docking_success"]) == 1 and math.isfinite(float(row["docking_score"]))
    ]


def _quantile(values: list[float], q: float) -> float:
    """Return a NaN-safe percentile."""
    return float(np.percentile(values, q)) if values else float("nan")


def summarize(merged: list[dict[str, object]]) -> list[dict[str, object]]:
    """Summarize all-attempt and RDKit-valid docking funnels."""
    summaries: list[dict[str, object]] = []
    for variant in VARIANTS:
        rows = [row for row in merged if row["variant"] == variant]
        valid = _valid_rows(rows)
        scores_all = [float(row["docking_score"]) for row in rows if math.isfinite(float(row["docking_score"]))]
        scores = [float(row["docking_score"]) for row in valid]
        les = [float(row["docking_ligand_efficiency"]) for row in valid if math.isfinite(float(row["docking_ligand_efficiency"]))]
        summaries.append(
            {
                "variant": variant,
                "attempts": 1000,
                "docking_backend_success": len(scores_all),
                "docking_backend_success_rate_attempts": len(scores_all) / 1000,
                "rdkit_valid": sum(int(row["rdkit_valid"]) for row in rows),
                "valid_docking_count": len(valid),
                "valid_docking_rate_attempts": len(valid) / 1000,
                "valid_docking_rate_valid": len(valid) / sum(int(row["rdkit_valid"]) for row in rows),
                "score_mean_all_backend": float(np.mean(scores_all)) if scores_all else float("nan"),
                "score_median_all_backend": float(np.median(scores_all)) if scores_all else float("nan"),
                "score_mean_rdkit_valid": float(np.mean(scores)) if scores else float("nan"),
                "score_median_rdkit_valid": float(np.median(scores)) if scores else float("nan"),
                "score_p10_rdkit_valid": _quantile(scores, 10),
                "score_p90_rdkit_valid": _quantile(scores, 90),
                "score_le_minus10_rate_valid": sum(score <= -10 for score in scores) / len(scores) if scores else float("nan"),
                "score_le_minus12_rate_valid": sum(score <= -12 for score in scores) / len(scores) if scores else float("nan"),
                "score_le_minus13_rate_valid": sum(score <= -13 for score in scores) / len(scores) if scores else float("nan"),
                "ligand_efficiency_mean_valid": float(np.mean(les)) if les else float("nan"),
                "ligand_efficiency_median_valid": float(np.median(les)) if les else float("nan"),
                "ligand_efficiency_p90_valid": _quantile(les, 90),
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a stable CSV from dictionaries."""
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_plots(summaries: list[dict[str, object]]) -> None:
    """Plot robust docking and ligand-efficiency comparisons."""
    labels = [str(row["variant"]) for row in summaries]
    colors = ["#4d4d4d", "#d73027", "#1a9850", "#4575b4"]
    positions = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    axes[0].bar(positions - 0.2, [float(row["score_median_rdkit_valid"]) for row in summaries], 0.4, label="median", color=colors)
    axes[0].bar(positions + 0.2, [float(row["score_p10_rdkit_valid"]) for row in summaries], 0.4, label="best decile threshold (p10)", color="#9ecae1")
    axes[0].axhline(-10, linestyle="--", color="#555555", linewidth=1)
    axes[0].set_ylabel("UniDock Vina score (lower is better)")
    axes[0].set_title("RDKit-valid docking scores")
    axes[0].set_xticks(positions, labels)
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(positions - 0.2, [float(row["ligand_efficiency_median_valid"]) for row in summaries], 0.4, label="median", color=colors)
    axes[1].bar(positions + 0.2, [float(row["ligand_efficiency_p90_valid"]) for row in summaries], 0.4, label="p90", color="#fdae6b")
    axes[1].set_ylabel("-Vina score / heavy atoms")
    axes[1].set_title("Docking ligand efficiency")
    axes[1].set_xticks(positions, labels)
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("DiffSBDD Global E2PO docking (T=500, fixed 1000 attempts)")
    fig.tight_layout()
    fig.savefig(ROOT / "generation_docking_comparison.png", dpi=200)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(11, 5.5))
    width = 0.18
    for index, key in enumerate(("score_le_minus10_rate_valid", "score_le_minus12_rate_valid", "score_le_minus13_rate_valid")):
        axis.bar(positions + (index - 1) * width, [float(row[key]) for row in summaries], width, label=key.replace("score_", "").replace("_rate_valid", "").replace("_", " "))
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Fraction among RDKit-valid molecules")
    axis.set_title("Docking threshold rates")
    axis.set_xticks(positions, labels)
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(ROOT / "generation_docking_thresholds.png", dpi=200)
    plt.close(fig)


def write_final_report(summaries: list[dict[str, object]]) -> None:
    """Write the final combined analysis and method diagnosis."""
    training = {}
    with (ROOT / "variant_summary.csv").open(newline="", encoding="utf-8") as handle:
        training = {row["variant"]: row for row in csv.DictReader(handle)}
    generation = {}
    with (ROOT / "generation_summary.csv").open(newline="", encoding="utf-8") as handle:
        generation = {row["variant"]: row for row in csv.DictReader(handle)}
    lines = [
        "# DiffSBDD Global E2PO T=500 一条龙结果分析",
        "",
        "## 实验边界",
        "",
        "四组统一生成：Original、x_only、v_only、native_joint；每组固定 1000 attempts，T=500、seed=1、batch=25、resamplings=10、jump_length=1、OpenBabel、all_frags。生成、RDKit 分析和 UniDock 均没有补样。",
        "",
        "## 关键结果",
        "",
        "| 方法 | test pair acc | valid/1000 | QED | C fraction | TPSA | guard violation | valid docking median | LE median | valid <= -10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        variant = str(row["variant"])
        train = training.get(variant)
        gen = generation[variant]
        pair_accuracy = float(train["test_pair_accuracy"]) if train is not None else float("nan")
        pair_text = f"{pair_accuracy:.4f}" if math.isfinite(pair_accuracy) else "baseline"
        lines.append(
            f"| {variant} | {pair_text} | {int(gen['sanitize_valid'])}/1000 | {float(gen['qed_mean']):.3f} | {float(gen['carbon_fraction_mean']):.3f} | {float(gen['tpsa_mean']):.2f} | {float(gen['guard_violation_rate_valid']):.3f} | {float(row['score_median_rdkit_valid']):.3f} | {float(row['ligand_efficiency_median_valid']):.3f} | {float(row['score_le_minus10_rate_valid']):.3f} |"
        )
    lines += [
        "",
        "## Figures",
        "",
        "![Generation validity, diversity and descriptors](generation_validity_diversity.png)",
        "",
        "![Valid-molecule descriptors](generation_descriptor_comparison.png)",
        "",
        "![Structural pathology](generation_pathology_comparison.png)",
        "",
        "![Docking and ligand efficiency](generation_docking_comparison.png)",
        "",
        "![Docking thresholds](generation_docking_thresholds.png)",
        "",
        "## 问题诊断",
        "",
        "1. `v_only` 的 pair fitting 明显最强（test 0.9732），但它不是可靠的生成质量指标。它的 RDKit-valid 率为 71.1%，同时有效分子 carbon fraction=0.833、TPSA=39.24、chemistry-guard violation=63.7%；相对 Original（0.747、61.94、40.4%）出现明显组成和极性漂移。",
        "2. `native_joint` 比 `v_only` 稍稳，但仍保留相同方向的 shortcut：carbon fraction=0.823、TPSA=40.52、guard violation=60.7%。它的 RDKit-valid docking median 最好（-10.333），但尚不足以抵消结构质量漂移。",
        "3. `x_only` 的 pair accuracy 接近随机（test 0.5354），但生成化学分布最接近 Original，valid docking median=-9.729，说明低 pair accuracy 不能单独证明生成支线失败；它更像是没有学到有效 preference signal，而不是发生严重的 v-only 组成偏移。",
        "4. UniDock backend 对四组均为 1000/1000，但这只是后端成功，不等于分子有效；最终比较使用 RDKit-valid 子集。`v_only` 的 score mean 被少数极差正分数拉高，因此 median、p10 和 threshold rate 比 mean 更可信。",
        "",
        "## 适配结论",
        "",
        "当前实现不应继续沿用单独 v-MSE 作为 DiffSBDD Global E2PO 的主目标。最合理的保留对象是 `native_joint` 的 native x/v 语义，但需要加入生成质量门控和 representation-balancing：至少报告并约束 carbon fraction、TPSA、hetero atoms、Brenk/pathology、RDKit validity，并以 RDKit-valid fixed-attempt docking/ligand-efficiency 作为验收。",
        "",
        "下一轮优先级应是： (a) 做 component-balanced/gradient-balanced 的 x/v 权重消融；(b) 保留 independent-noise 与 native parameterization；(c) 对连续 reward gap 做真实非恒定 pair 训练；(d) 每个 checkpoint 都做同样的 1000-attempt valid-only funnel。不要把 v-only 的高 pair accuracy 当作成功，也不要把 endpoint E2PO 宣称为完整 trajectory DPO 或逐步 credit assignment。",
        "",
        "## Evidence files",
        "",
        "- `variant_summary.csv` and `timestep_accuracy.csv`: training fit",
        "- `generation_summary.csv`, `generation_molecule_metrics.csv`, `generation_pathology_summary.csv`: fixed-attempt generation",
        "- `generation_docking_summary.csv`, `generation_docking_metrics.csv`: UniDock and valid-only docking funnel",
        "- `generation_docking_comparison.png`, `generation_docking_thresholds.png`: docking figures",
    ]
    (ROOT / "generation_final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Merge, summarize, plot, and report the completed docking matrix."""
    merged = merge_rows()
    summaries = summarize(merged)
    write_csv(ROOT / "generation_docking_metrics.csv", merged)
    write_csv(ROOT / "generation_docking_summary.csv", summaries)
    write_plots(summaries)
    write_final_report(summaries)
    print(json.dumps({"merged_rows": len(merged), "summary": str(ROOT / "generation_docking_summary.csv")}, indent=2))


if __name__ == "__main__":
    main()
