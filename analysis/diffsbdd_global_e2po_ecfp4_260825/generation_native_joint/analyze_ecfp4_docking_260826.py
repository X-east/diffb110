#!/usr/bin/env python3
"""Summarize current ECFP4-E2PO and Original UniDock results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = [
    ("Original", "original", "#606060"),
    ("ECFP4-E2PO", "ecfp4_e2po", "#2c7fb8"),
]
THRESHOLDS = [-10.0, -12.0, -13.0]
EXTREME_ANOMALY_CUTOFF = 1000.0


def read_method(method: str, directory: str, root: Path) -> pd.DataFrame:
    """Merge docking scores with fixed-attempt generation validity."""
    metrics = pd.read_csv(root / "evaluation" / "per_attempt_metrics.csv")
    metrics = metrics.loc[metrics["method"].eq(method)].copy()
    docking = pd.read_csv(root / "docking_260826" / directory / "docking_results.csv")
    docking = docking.rename(columns={"benchmark_index": "attempt_index"})
    docking["attempt"] = docking["attempt_index"] + 1
    docking["score"] = pd.to_numeric(docking["score"], errors="coerce")
    merged = metrics.merge(
        docking[["attempt", "success", "score", "fragments_original"]],
        on="attempt",
        how="left",
        validate="one_to_one",
    )
    if len(merged) != 1000 or merged["score"].isna().any():
        raise ValueError(f"Unexpected merged docking rows for {method}: {len(merged)}")
    merged["method"] = method
    # Keep the parsed UniDock result untouched and add an analysis-only clean
    # overlay.  Only the extreme +2000-class UniDock output is excluded;
    # smaller positive scores remain in the statistics by project decision.
    merged["docking_score_raw"] = merged["score"]
    merged["docking_score_clean"] = merged["docking_score_raw"].where(
        merged["docking_score_raw"] < EXTREME_ANOMALY_CUTOFF
    )
    merged["ligand_efficiency_raw"] = -merged["docking_score_raw"] / merged["heavy_atoms"]
    merged["ligand_efficiency_clean"] = (
        -merged["docking_score_clean"] / merged["heavy_atoms"]
    )
    merged["ligand_efficiency"] = merged["ligand_efficiency_raw"]

    def classify(row: pd.Series) -> str:
        if not bool(row["success"]):
            return "backend_not_success"
        if int(row["sanitize_success"]) != 1:
            return "sanitize_invalid"
        if float(row["docking_score_raw"]) >= EXTREME_ANOMALY_CUTOFF:
            return "extreme_positive_docking_anomaly"
        if float(row["docking_score_raw"]) >= 0:
            return "positive_score_retained"
        return "clean_score"

    merged["docking_score_status"] = merged.apply(classify, axis=1)
    return merged


def summarize(merged: pd.DataFrame) -> pd.DataFrame:
    """Compute fixed-attempt, valid-only, robust score, and LE metrics."""
    rows: list[dict[str, object]] = []
    for method, _, _ in METHODS:
        data = merged.loc[merged["method"].eq(method)].copy()
        valid = data.loc[data["sanitize_success"].eq(1)].copy()
        clean = valid.loc[valid["docking_score_status"].ne("extreme_positive_docking_anomaly")].copy()
        row: dict[str, object] = {
            "method": method,
            "attempts": len(data),
            "backend_success": int(data["success"].eq(True).sum()),
            "backend_success_rate_attempts": float(data["success"].eq(True).mean()),
            "sanitize_valid": len(valid),
            "valid_rate_attempts": float(len(valid) / len(data)),
            "clean_docking_n": len(clean),
            "clean_docking_rate_attempts": float(len(clean) / len(data)),
            "score_mean_all_attempts_raw": float(data["docking_score_raw"].mean()),
            "score_mean_valid_raw": float(valid["docking_score_raw"].mean()),
            "score_median_valid_raw": float(valid["docking_score_raw"].median()),
            "score_p10_valid_raw": float(valid["docking_score_raw"].quantile(0.10)),
            "score_p90_valid_raw": float(valid["docking_score_raw"].quantile(0.90)),
            "score_mean_clean": float(clean["docking_score_clean"].mean()),
            "score_median_clean": float(clean["docking_score_clean"].median()),
            "score_p10_clean": float(clean["docking_score_clean"].quantile(0.10)),
            "score_p90_clean": float(clean["docking_score_clean"].quantile(0.90)),
            "nonnegative_score_n_valid": int((valid["score"] >= 0).sum()),
            "nonnegative_score_n_all": int((data["score"] >= 0).sum()),
            "extreme_anomaly_n_valid": int(
                (valid["docking_score_status"] == "extreme_positive_docking_anomaly").sum()
            ),
            "extreme_anomaly_n_all": int(
                (data["docking_score_status"] == "extreme_positive_docking_anomaly").sum()
            ),
            "ligand_efficiency_mean_clean": float(clean["ligand_efficiency_clean"].mean()),
            "ligand_efficiency_median_clean": float(clean["ligand_efficiency_clean"].median()),
            "ligand_efficiency_p90_clean": float(clean["ligand_efficiency_clean"].quantile(0.90)),
        }
        for threshold in THRESHOLDS:
            suffix = int(abs(threshold))
            row[f"score_le_{suffix}_rate_clean"] = float(
                (clean["docking_score_clean"] <= threshold).mean()
            )
            row[f"score_le_{suffix}_rate_attempts"] = float(
                (data["docking_score_raw"] <= threshold).mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_outputs(summary: pd.DataFrame, merged: pd.DataFrame, output_dir: Path) -> None:
    """Plot robust docking and ligand-efficiency comparisons."""
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    labels = summary["method"].tolist()
    colors = [dict((name, color) for name, _, color in METHODS)[label] for label in labels]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    width = 0.25
    axes[0].bar(x - width, summary["score_median_clean"], width, label="median", color=colors)
    axes[0].bar(x, summary["score_p10_clean"], width, label="p10", color="#9ecae1")
    axes[0].bar(x + width, summary["score_mean_clean"], width, label="mean, score < +1000", color="#fdae6b")
    axes[0].axhline(-10, color="#555555", linestyle="--", linewidth=1)
    axes[0].set_ylabel("UniDock Vina score (lower is better)")
    axes[0].set_title("Docking scores after +2000-class anomaly exclusion")
    axes[0].set_xticks(x, labels)
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x - width, summary["ligand_efficiency_median_clean"], width, label="median", color=colors)
    axes[1].bar(x, summary["ligand_efficiency_p90_clean"], width, label="p90", color="#74c476")
    axes[1].set_ylabel("-Vina score / heavy atoms")
    axes[1].set_title("Docking ligand efficiency")
    axes[1].set_xticks(x, labels)
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Current ECFP4-E2PO docking evaluation: fixed 1000 attempts")
    fig.savefig(output_dir / "docking_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "docking_comparison.svg", bbox_inches="tight")
    plt.close(fig)

    threshold_keys = [f"score_le_{int(abs(threshold))}_rate_clean" for threshold in THRESHOLDS]
    fig, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    width = 0.30
    for index, (threshold, key) in enumerate(zip(THRESHOLDS, threshold_keys)):
        axis.bar(x + (index - 1) * width, summary[key], width, label=f"score <= {threshold:g}")
    axis.set_ylim(0, 0.7)
    axis.set_ylabel("Fraction among clean valid molecules")
    axis.set_title("Docking threshold rates after anomaly exclusion")
    axis.set_xticks(x, labels)
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(output_dir / "docking_thresholds.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "docking_thresholds.svg", bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    for method, _, color in METHODS:
        values = merged.loc[
            merged["method"].eq(method)
            & merged["sanitize_success"].eq(1)
            & merged["docking_score_status"].ne("extreme_positive_docking_anomaly"),
            "docking_score_clean",
        ]
        axis.hist(values, bins=35, alpha=0.55, density=True, label=method, color=color)
    axis.set_xlabel("UniDock Vina score; +1000 and above excluded")
    axis.set_ylabel("Density / valid molecules")
    axis.set_title("Docking-score distributions after +2000-class anomaly exclusion")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(output_dir / "docking_score_distributions.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "docking_score_distributions.svg", bbox_inches="tight")
    plt.close(fig)


def write_report(summary: pd.DataFrame, output_dir: Path, root: Path) -> None:
    """Write an evidence-bounded docking report."""
    original = summary.loc[summary["method"].eq("Original")].iloc[0]
    e2po = summary.loc[summary["method"].eq("ECFP4-E2PO")].iloc[0]
    project_root = root.parent.parent.parent
    original_input = project_root / "analysis" / "diffsbdd_global_e2po_good_bad_xv_smoke_260825" / "generation_comparison" / "original" / "molecules.sdf"
    anomalies = pd.read_csv(output_dir / "docking_anomalies.csv")
    anomaly_text = []
    for _, row in anomalies.iterrows():
        anomaly_text.append(
            f"- {row['method']} attempt `{int(row['attempt'])}`: raw score `{float(row['docking_score_raw']):+.3f}`, status `{row['docking_score_status']}`。"
        )
    lines = [
        "# ECFP4-E2PO 当前生成对接评价",
        "",
        "## 结论",
        "",
        f"在同一受体、同一盒子和 UniDock fast/Vina 设置下，仅排除 `score >= {EXTREME_ANOMALY_CUTOFF:g}` 的 +2000 类极端异常后，ECFP4-E2PO 的 clean docking median 为 `{e2po['score_median_clean']:.3f}`，Original 为 `{original['score_median_clean']:.3f}`；变化为 `{e2po['score_median_clean'] - original['score_median_clean']:+.3f}`，负值代表 ECFP4-E2PO 更好。ECFP4-E2PO 的 clean p10 也由 `{original['score_p10_clean']:.3f}` 变为 `{e2po['score_p10_clean']:.3f}`。+2.164 和 +9.440 等较小正分数按当前约定保留。",
        f"clean 阈值比例为：score ≤ -10 `{100 * original['score_le_10_rate_clean']:.2f}% → {100 * e2po['score_le_10_rate_clean']:.2f}%`，score ≤ -12 `{100 * original['score_le_12_rate_clean']:.2f}% → {100 * e2po['score_le_12_rate_clean']:.2f}%`。clean 分子 ligand-efficiency median 为 `{original['ligand_efficiency_median_clean']:.4f} → {e2po['ligand_efficiency_median_clean']:.4f}`。",
        "因此，当前这批 ECFP4-E2PO 分子在 docking 的稳健中心趋势和中高质量阈值比例上有改善，但这不是训练时直接优化 docking 的结果，仍需结合结构病态、PoseBusters 和重复实验确认。",
        "",
        "## 固定条件与分母",
        "",
        f"- Original SDF：`{original_input}`；逐次 generation metrics：`{root / 'evaluation' / 'per_attempt_metrics.csv'}` 中的 Original 记录。",
        f"- ECFP4-E2PO SDF：`{root / 'generation' / 'molecules.sdf'}`",
        f"- receptor：`{root.parent.parent.parent / 'data' / '8gjl.pdbqt'}`",
        "- docking box center=(207.293, 212.430, 172.771)，size=(20,20,20)。",
        "- UniDock Vina，fast search，num_modes=1，batch_size=128；两组均为 1000/1000 backend-success。",
        f"- sanitize-valid 分子分母为 Original `{int(original['sanitize_valid'])}`、ECFP4-E2PO `{int(e2po['sanitize_valid'])}`；仅排除极端异常后的 clean docking 分母为 `{int(original['clean_docking_n'])}` 与 `{int(e2po['clean_docking_n'])}`。fixed-attempt 有效率仍为 `{int(original['sanitize_valid'])}/1000` 与 `{int(e2po['sanitize_valid'])}/1000`，没有因清洗而改写生成有效率。",
        "",
        "## Summary",
        "",
        "| metric | Original | ECFP4-E2PO |",
        "|---|---:|---:|",
        f"| backend success | {int(original['backend_success'])}/1000 | {int(e2po['backend_success'])}/1000 |",
        f"| sanitize valid | {int(original['sanitize_valid'])}/1000 | {int(e2po['sanitize_valid'])}/1000 |",
        f"| docking clean n | {int(original['clean_docking_n'])} | {int(e2po['clean_docking_n'])} |",
        f"| docking median, clean | {original['score_median_clean']:.4f} | {e2po['score_median_clean']:.4f} |",
        f"| docking p10, clean | {original['score_p10_clean']:.4f} | {e2po['score_p10_clean']:.4f} |",
        f"| mean, clean score < +1000 | {original['score_mean_clean']:.4f} | {e2po['score_mean_clean']:.4f} |",
        f"| score <= -10, clean | {100 * original['score_le_10_rate_clean']:.2f}% | {100 * e2po['score_le_10_rate_clean']:.2f}% |",
        f"| score <= -12, clean | {100 * original['score_le_12_rate_clean']:.2f}% | {100 * e2po['score_le_12_rate_clean']:.2f}% |",
        f"| score <= -13, clean | {100 * original['score_le_13_rate_clean']:.2f}% | {100 * e2po['score_le_13_rate_clean']:.2f}% |",
        f"| ligand efficiency median, clean | {original['ligand_efficiency_median_clean']:.4f} | {e2po['ligand_efficiency_median_clean']:.4f} |",
        f"| nonnegative parsed score, valid | {int(original['nonnegative_score_n_valid'])} | {int(e2po['nonnegative_score_n_valid'])} |",
        "",
        "![Docking score comparison](docking_comparison.png)",
        "",
        "![Docking threshold rates](docking_thresholds.png)",
        "",
        "![Docking score distributions](docking_score_distributions.png)",
        "",
        "## 异常与解释边界",
        "",
        f"- raw valid-score mean 为 `{original['score_mean_valid_raw']:.4f}` 与 `{e2po['score_mean_valid_raw']:.4f}`；其中仅有 `{int(original['extreme_anomaly_n_valid'])}` 和 `{int(e2po['extreme_anomaly_n_valid'])}` 个 `score >= {EXTREME_ANOMALY_CUTOFF:g}` 的极端异常被排除。+2.164、+9.440 等较小正分数不属于本次排除范围。",
        "- 本次不删除原始 UniDock CSV、batch 输出 SDF 或输入 SDF；只在 `docking_per_attempt_metrics.csv` 中增加 clean 列，并将极端异常输出到单独的清单。fixed-attempt 的 1000 分母和 sanitize-valid 分母保持可追溯。",
        "- 异常清单：",
        *anomaly_text,
        "- docking improvement 只说明当前 docking proxy 变好，不等于真实结合亲和力或生物活性改善；还需要 ligand efficiency、PoseBusters/结构质量、重复 seed 和人工相互作用检查。",
        "",
        "## Evidence files",
        "",
        "- `docking_summary.csv`：汇总指标。",
        "- `docking_per_attempt_metrics.csv`：逐次 generation+docking 合并表。",
        f"- `docking_anomalies.csv`：`score >= {EXTREME_ANOMALY_CUTOFF:g}` 的极端 docking 分数清单，原始分数和 attempt 映射均保留。",
        "- `original/docking_results.csv` 与 `ecfp4_e2po/docking_results.csv`：原始 UniDock 结果。",
        "- `run_ecfp4_docking_260826.sh`：对接启动脚本。",
    ]
    (output_dir / "docking_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the current ECFP4-E2PO docking analysis."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.root / "docking_260826"
    merged = pd.concat([read_method(method, directory, args.root) for method, directory, _ in METHODS], ignore_index=True)
    summary = summarize(merged)
    merged.to_csv(output_dir / "docking_per_attempt_metrics.csv", index=False)
    anomaly_columns = [
        "method", "attempt", "docking_score_raw",
        "docking_score_status", "sanitize_success", "success", "heavy_atoms",
        "fragments_original", "canonical_smiles",
    ]
    merged.loc[merged["docking_score_status"].eq("extreme_positive_docking_anomaly"), anomaly_columns].to_csv(
        output_dir / "docking_anomalies.csv", index=False
    )
    summary.to_csv(output_dir / "docking_summary.csv", index=False)
    plot_outputs(summary, merged, output_dir)
    write_report(summary, output_dir, args.root)
    manifest = {
        "status": "completed",
        "methods": [method for method, _, _ in METHODS],
        "fixed_attempts_per_method": 1000,
        "valid_only_metrics": True,
        "raw_score_column": "docking_score_raw",
        "clean_score_column": "docking_score_clean",
        "clean_score_rule": f"sanitize_valid and docking_score_raw < {EXTREME_ANOMALY_CUTOFF:g}",
        "extreme_anomaly_cutoff": EXTREME_ANOMALY_CUTOFF,
        "smaller_positive_scores": "retained",
        "raw_docking_outputs_preserved": True,
        "receptor": str((args.root.parent.parent.parent / "data" / "8gjl.pdbqt").resolve()),
        "box": {"center": [207.293, 212.430, 172.771], "size": [20.0, 20.0, 20.0]},
        "backend": "UniDock Vina fast",
        "output_dir": str(output_dir.resolve()),
    }
    (output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
