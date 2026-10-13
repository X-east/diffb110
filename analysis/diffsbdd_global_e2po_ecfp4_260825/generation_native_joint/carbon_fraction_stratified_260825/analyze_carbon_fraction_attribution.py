#!/usr/bin/env python3
"""Test whether ECFP4 similarity gains are explained by carbon-fraction drift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = ["Original", "ECFP4-E2PO"]
METRICS = [
    "ecfp4_top1_similarity",
    "ecfp4_top2_similarity",
    "ecfp4_top3_similarity",
    "ecfp4_top5_mean_similarity",
]
METRIC_LABELS = {
    "ecfp4_top1_similarity": "ECFP4 top-1",
    "ecfp4_top2_similarity": "ECFP4 top-2",
    "ecfp4_top3_similarity": "ECFP4 top-3",
    "ecfp4_top5_mean_similarity": "ECFP4 top-5 mean",
}
BIN_EDGES = [-np.inf, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, np.inf]
BIN_LABELS = [
    "<=0.60",
    "0.60-0.70",
    "0.70-0.75",
    "0.75-0.80",
    "0.80-0.85",
    "0.85-0.90",
    "0.90-0.95",
    ">0.95",
]


def load_valid_rows(path: Path) -> pd.DataFrame:
    """Load valid rows and assign the predefined carbon-fraction bins."""
    data = pd.read_csv(path)
    required = {"method", "sanitize_success", "carbon_fraction", *METRICS}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    valid = data.loc[data["sanitize_success"].eq(1)].copy()
    valid["carbon_bin"] = pd.cut(
        valid["carbon_fraction"],
        bins=BIN_EDGES,
        labels=BIN_LABELS,
        include_lowest=True,
        right=True,
    )
    return valid


def summarize_bins(valid: pd.DataFrame) -> pd.DataFrame:
    """Summarize fixed-attempt counts and valid-molecule metrics by bin."""
    rows: list[dict[str, object]] = []
    all_data = pd.read_csv(valid.attrs["source_path"])
    for method in METHODS:
        method_all = all_data.loc[all_data["method"].eq(method)]
        method_valid = valid.loc[valid["method"].eq(method)]
        for label in BIN_LABELS:
            group = method_valid.loc[method_valid["carbon_bin"].eq(label)]
            row: dict[str, object] = {
                "method": method,
                "carbon_bin": label,
                "attempts_with_carbon_bin": int(
                    method_all["carbon_fraction"].between(
                        BIN_EDGES[BIN_LABELS.index(label)],
                        BIN_EDGES[BIN_LABELS.index(label) + 1],
                        inclusive="right",
                    ).sum()
                ),
                "valid_n": int(len(group)),
                "valid_fraction_of_all_attempts": float(len(group) / len(method_all)) if len(method_all) else np.nan,
                "carbon_fraction_mean": float(group["carbon_fraction"].mean()) if len(group) else np.nan,
            }
            for metric in METRICS:
                row[f"{metric}_mean"] = float(group[metric].mean()) if len(group) else np.nan
                row[f"{metric}_median"] = float(group[metric].median()) if len(group) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def common_support_summary(bin_summary: pd.DataFrame, min_n: int) -> pd.DataFrame:
    """Compare bins with at least min_n valid molecules in both methods."""
    counts = bin_summary.pivot(index="carbon_bin", columns="method", values="valid_n")
    common_bins = [
        label
        for label in BIN_LABELS
        if all(int(counts.loc[label, method]) >= min_n for method in METHODS)
    ]
    rows: list[dict[str, object]] = []
    for label in common_bins:
        original = bin_summary.query("method == 'Original' and carbon_bin == @label").iloc[0]
        e2po = bin_summary.query("method == 'ECFP4-E2PO' and carbon_bin == @label").iloc[0]
        row: dict[str, object] = {
            "carbon_bin": label,
            "original_n": int(original["valid_n"]),
            "e2po_n": int(e2po["valid_n"]),
            "original_carbon_fraction_mean": float(original["carbon_fraction_mean"]),
            "e2po_carbon_fraction_mean": float(e2po["carbon_fraction_mean"]),
        }
        for metric in METRICS:
            original_value = float(original[f"{metric}_mean"])
            e2po_value = float(e2po[f"{metric}_mean"])
            row[f"original_{metric}_mean"] = original_value
            row[f"e2po_{metric}_mean"] = e2po_value
            row[f"delta_{metric}"] = e2po_value - original_value
        rows.append(row)
    return pd.DataFrame(rows)


def reweighted_summary(bin_summary: pd.DataFrame) -> pd.DataFrame:
    """Reweight ECFP4-E2PO to the Original valid carbon-bin distribution."""
    counts = bin_summary.pivot(index="carbon_bin", columns="method", values="valid_n").reindex(BIN_LABELS)
    original_weights = counts["Original"] / counts["Original"].sum()
    rows: list[dict[str, object]] = []
    for metric in METRICS:
        original_values = bin_summary.query("method == 'Original'").set_index("carbon_bin")[f"{metric}_mean"].reindex(BIN_LABELS)
        e2po_values = bin_summary.query("method == 'ECFP4-E2PO'").set_index("carbon_bin")[f"{metric}_mean"].reindex(BIN_LABELS)
        original_mean = float(np.sum(original_weights.to_numpy() * original_values.to_numpy()))
        e2po_matched_mean = float(np.sum(original_weights.to_numpy() * e2po_values.to_numpy()))
        e2po_raw_mean = float(
            bin_summary.query("method == 'ECFP4-E2PO'")[f"{metric}_mean"]
            .mul(bin_summary.query("method == 'ECFP4-E2PO'")["valid_n"])
            .sum()
            / counts["ECFP4-E2PO"].sum()
        )
        rows.append(
            {
                "metric": metric,
                "original_valid_mean": original_mean,
                "e2po_raw_valid_mean": e2po_raw_mean,
                "e2po_carbon_reweighted_mean": e2po_matched_mean,
                "raw_delta_e2po_minus_original": e2po_raw_mean - original_mean,
                "reweighted_delta_e2po_minus_original": e2po_matched_mean - original_mean,
                "raw_gain_explained_by_carbon_fraction": 1.0
                - (e2po_matched_mean - original_mean) / (e2po_raw_mean - original_mean)
                if e2po_raw_mean != original_mean
                else np.nan,
            }
        )
    weight_rows = pd.DataFrame(
        {
            "carbon_bin": BIN_LABELS,
            "original_valid_n": counts["Original"].astype(int).to_numpy(),
            "e2po_valid_n": counts["ECFP4-E2PO"].astype(int).to_numpy(),
            "original_target_weight": original_weights.to_numpy(),
        }
    )
    return pd.DataFrame(rows), weight_rows


def nearest_match_summary(valid: pd.DataFrame, caliper: float) -> pd.DataFrame:
    """Match each Original molecule to the nearest E2PO carbon fraction."""
    original = valid.loc[valid["method"].eq("Original")].sort_values("carbon_fraction").reset_index(drop=True)
    e2po = valid.loc[valid["method"].eq("ECFP4-E2PO")].sort_values("carbon_fraction").reset_index(drop=True)
    e2_carbon = e2po["carbon_fraction"].to_numpy()
    matches: list[dict[str, object]] = []
    for _, original_row in original.iterrows():
        carbon = float(original_row["carbon_fraction"])
        index = int(np.searchsorted(e2_carbon, carbon))
        candidates = [candidate for candidate in (index - 1, index) if 0 <= candidate < len(e2po)]
        nearest = min(candidates, key=lambda candidate: abs(float(e2_carbon[candidate]) - carbon))
        difference = float(e2_carbon[nearest] - carbon)
        row: dict[str, object] = {
            "original_carbon_fraction": carbon,
            "e2po_matched_carbon_fraction": float(e2_carbon[nearest]),
            "carbon_fraction_difference": difference,
            "within_caliper": int(abs(difference) <= caliper),
        }
        for metric in METRICS:
            row[f"original_{metric}"] = float(original_row[metric])
            row[f"e2po_matched_{metric}"] = float(e2po.iloc[nearest][metric])
            row[f"delta_{metric}"] = row[f"e2po_matched_{metric}"] - row[f"original_{metric}"]
        matches.append(row)
    matched = pd.DataFrame(matches)
    rows: list[dict[str, object]] = []
    for metric in METRICS:
        in_caliper = matched.loc[matched["within_caliper"].eq(1)]
        rows.append(
            {
                "metric": metric,
                "caliper": caliper,
                "matched_original_n": int(len(matched)),
                "within_caliper_n": int(len(in_caliper)),
                "mean_abs_carbon_fraction_difference": float(matched["carbon_fraction_difference"].abs().mean()),
                "within_caliper_original_mean": float(in_caliper[f"original_{metric}"].mean()),
                "within_caliper_e2po_matched_mean": float(in_caliper[f"e2po_matched_{metric}"].mean()),
                "within_caliper_delta": float(in_caliper[f"delta_{metric}"].mean()),
            }
        )
    return pd.DataFrame(rows), matched


def plot_outputs(bin_summary: pd.DataFrame, reweighted: pd.DataFrame, weights: pd.DataFrame, matched: pd.DataFrame, output_dir: Path) -> None:
    """Create figures for stratified and distribution-adjusted comparisons."""
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    colors = {"Original": "#5f6368", "ECFP4-E2PO": "#2c7fb8"}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for axis, metric in zip(axes.flat, METRICS):
        for method in METHODS:
            subset = bin_summary.query("method == @method")
            axis.plot(BIN_LABELS, subset[f"{metric}_mean"], marker="o", linewidth=2, label=method, color=colors[method])
        axis.set_title(METRIC_LABELS[metric])
        axis.set_xlabel("Carbon fraction bin")
        axis.set_ylabel("Mean Tanimoto among valid molecules")
        axis.tick_params(axis="x", rotation=40)
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("ECFP4 similarity within carbon-fraction strata", fontsize=16)
    fig.savefig(output_dir / "carbon_fraction_stratified.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "carbon_fraction_stratified.svg", bbox_inches="tight")
    plt.close(fig)

    x = np.arange(len(METRICS))
    fig, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    raw = reweighted["e2po_raw_valid_mean"].to_numpy()
    original = reweighted["original_valid_mean"].to_numpy()
    adjusted = reweighted["e2po_carbon_reweighted_mean"].to_numpy()
    width = 0.25
    axis.bar(x - width, original, width, label="Original", color=colors["Original"])
    axis.bar(x, raw, width, label="ECFP4-E2PO raw", color="#e78ac3")
    axis.bar(x + width, adjusted, width, label="ECFP4-E2PO carbon-reweighted", color=colors["ECFP4-E2PO"])
    axis.set_xticks(x, [METRIC_LABELS[metric] for metric in METRICS], rotation=20)
    axis.set_ylabel("Mean Tanimoto among valid molecules")
    axis.set_title("Overall similarity before and after matching Original carbon distribution")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(output_dir / "carbon_fraction_reweighted.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "carbon_fraction_reweighted.svg", bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    axis.hist(matched["carbon_fraction_difference"], bins=30, color="#6baed6", edgecolor="white")
    axis.axvline(0, color="#333333", linewidth=1)
    axis.set_xlabel("Matched ECFP4-E2PO carbon fraction - Original carbon fraction")
    axis.set_ylabel("Number of matched Original molecules")
    axis.set_title("Nearest-neighbor carbon-fraction matching diagnostic")
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(output_dir / "carbon_fraction_matching_diagnostic.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "carbon_fraction_matching_diagnostic.svg", bbox_inches="tight")
    plt.close(fig)


def write_report(
    bin_summary: pd.DataFrame,
    common: pd.DataFrame,
    reweighted: pd.DataFrame,
    weights: pd.DataFrame,
    matching_summary: pd.DataFrame,
    output_dir: Path,
    input_path: Path,
) -> None:
    """Write an evidence-bounded interpretation report."""
    top5 = reweighted.loc[reweighted["metric"].eq("ecfp4_top5_mean_similarity")].iloc[0]
    match_top5 = matching_summary.loc[matching_summary["metric"].eq("ecfp4_top5_mean_similarity")].iloc[0]
    common_delta = float(common["delta_ecfp4_top5_mean_similarity"].mean()) if len(common) else np.nan
    report = [
        "# ECFP4-E2PO 碳比例归因测试",
        "",
        "## 结论",
        "",
        f"ECFP4 top-5 mean 的原始 valid-molecule 均值从 `{top5['original_valid_mean']:.4f}` 变为 `{top5['e2po_raw_valid_mean']:.4f}`，原始差值为 `{top5['raw_delta_e2po_minus_original']:+.4f}`。将 ECFP4-E2PO 按 Original 的 carbon fraction 分布重加权后，均值为 `{top5['e2po_carbon_reweighted_mean']:.4f}`，差值为 `{top5['reweighted_delta_e2po_minus_original']:+.4f}`。分布校正后的差值没有缩小（反而增加 `{top5['reweighted_delta_e2po_minus_original'] - top5['raw_delta_e2po_minus_original']:+.4f}`），所以不能把 top-5 mean 的提升归因于 carbon fraction 分布漂移。",
        f"在共同支持区间（两组每个 carbon bin 至少 20 个 valid 分子）内，top-5 mean 的分箱差值平均为 `{common_delta:+.4f}`；这个值是等权的分箱平均，不是按样本数加权的总体均值。最近邻匹配到 Original 碳比例后，top-5 mean 差值为 `{match_top5['within_caliper_delta']:+.4f}`（caliper={match_top5['caliper']:.2f}，有效匹配 `{int(match_top5['within_caliper_n'])}/{int(match_top5['matched_original_n'])}`）。",
        "因此，不能把总体 ECFP4 相似度上升直接解释为纯粹的碳比例效应：虽然 ECFP4-E2PO 的 carbon fraction 分布明显右移，但分布重加权后 top-5 mean 差异没有缩小，精确碳比例匹配后的差异接近 0。应将其表述为总体提升与碳比例漂移有关联，但当前数据不足以证明碳比例是主要解释变量。",
        "",
        "## 数据和分母",
        "",
        f"- 输入逐次指标：`{input_path}`",
        "- Original 与 ECFP4-E2PO 均为 1000 fixed attempts；无效记录保留在固定尝试分母中。",
        "- Original：653/1000 valid；ECFP4-E2PO：708/1000 valid。",
        "- 分层与 ECFP4 均值只在 sanitize-valid 分子上计算；因此它们回答的是“有效分子条件下”的归因问题，不能替代固定尝试有效率。",
        "- carbon fraction 分箱：`<=0.60`、`0.60-0.70`、`0.70-0.75`、`0.75-0.80`、`0.80-0.85`、`0.85-0.90`、`0.90-0.95`、`>0.95`。",
        "- 重加权目标是 Original 的 valid-molecule carbon-bin 分布；没有对 invalid attempts 进行插补。",
        "",
        "## Raw and carbon-reweighted summary",
        "",
        "| metric | Original valid mean | ECFP4-E2PO raw | ECFP4-E2PO carbon-reweighted | raw delta | reweighted delta | fraction explained by carbon |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in reweighted.iterrows():
        report.append(
            f"| {METRIC_LABELS[row['metric']]} | {row['original_valid_mean']:.4f} | {row['e2po_raw_valid_mean']:.4f} | {row['e2po_carbon_reweighted_mean']:.4f} | {row['raw_delta_e2po_minus_original']:+.4f} | {row['reweighted_delta_e2po_minus_original']:+.4f} | {100 * row['raw_gain_explained_by_carbon_fraction']:.1f}% |"
        )
    report.extend(
        [
            "",
            "![Carbon-fraction stratified comparison](carbon_fraction_stratified.png)",
            "",
            "![Raw and carbon-reweighted comparison](carbon_fraction_reweighted.png)",
            "",
            "![Matching diagnostic](carbon_fraction_matching_diagnostic.png)",
            "",
            "## Interpretation",
            "",
            "- 分箱曲线用于检查同一 carbon fraction 区间内的条件差异；由于每个区间内仍有重原子数、环系和具体子结构差异，不能把它当作完全因果调整。",
            "- 重加权结果是对 carbon fraction 的描述性标准化，不是随机实验或因果效应估计。",
            "- ECFP4 top-1 可能被少数高相似邻居抬高；top-5 mean 更适合作为主要稳健指标，但仍然只代表 ChEMBL100k 的 2D 子结构接近度。",
            "- 后续若要判断 E2PO 是否值得保留，应把 carbon fraction 校正后的 ECFP4 结果与有效率、scaffold 多样性、病态结构、docking 和 ligand efficiency 一起看。",
            "",
            "## Evidence files",
            "",
            "- `carbon_fraction_bin_summary.csv`：所有预设 carbon bins 的计数和分层均值。",
            "- `carbon_fraction_common_support_summary.csv`：两组每个 bin 至少 20 个 valid 分子的共同支持比较。",
            "- `carbon_fraction_reweighted_summary.csv`：按 Original valid carbon-bin 分布重加权。",
            "- `carbon_fraction_reweighting_weights.csv`：重加权目标权重。",
            "- `carbon_fraction_nearest_match_summary.csv`：最近邻匹配的汇总。",
            "- `carbon_fraction_nearest_matches.csv`：逐个 Original 分子的匹配明细。",
        ]
    )
    (output_dir / "carbon_fraction_analysis_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    """Run the carbon-fraction attribution analysis."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-common-bin-n", type=int, default=20)
    parser.add_argument("--matching-caliper", type=float, default=0.02)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    valid = load_valid_rows(args.input)
    valid.attrs["source_path"] = str(args.input)
    bin_summary = summarize_bins(valid)
    common = common_support_summary(bin_summary, args.min_common_bin_n)
    reweighted, weights = reweighted_summary(bin_summary)
    matching_summary, matches = nearest_match_summary(valid, args.matching_caliper)
    bin_summary.to_csv(args.output_dir / "carbon_fraction_bin_summary.csv", index=False)
    common.to_csv(args.output_dir / "carbon_fraction_common_support_summary.csv", index=False)
    reweighted.to_csv(args.output_dir / "carbon_fraction_reweighted_summary.csv", index=False)
    weights.to_csv(args.output_dir / "carbon_fraction_reweighting_weights.csv", index=False)
    matching_summary.to_csv(args.output_dir / "carbon_fraction_nearest_match_summary.csv", index=False)
    matches.to_csv(args.output_dir / "carbon_fraction_nearest_matches.csv", index=False)
    plot_outputs(bin_summary, reweighted, weights, matches, args.output_dir)
    write_report(bin_summary, common, reweighted, weights, matching_summary, args.output_dir, args.input)
    manifest = {
        "status": "completed",
        "input": str(args.input.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "methods": METHODS,
        "fixed_attempts_per_method": 1000,
        "valid_only_for_conditional_metrics": True,
        "carbon_bin_edges": BIN_EDGES,
        "carbon_bin_labels": BIN_LABELS,
        "common_support_min_n_per_method": args.min_common_bin_n,
        "matching_target": "Original valid carbon-fraction distribution",
        "matching_caliper": args.matching_caliper,
        "metrics": METRICS,
    }
    (args.output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "common_bins": common["carbon_bin"].tolist(), "reweighted": reweighted.to_dict(orient="records")}, indent=2))


if __name__ == "__main__":
    main()
