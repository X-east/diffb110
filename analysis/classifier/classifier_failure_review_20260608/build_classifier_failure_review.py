from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "analysis" / "classifier_failure_review_20260608"


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def main() -> None:
    eval_json = ROOT / "chembl_classifier/models/evaluation_selected_v0_1/chembl_classifier_selected_v0_1_evaluation.json"
    base_summary_json = ROOT / "analysis/classifier_test_260407_sdf_20260527/summary.json"
    subgroup_json = ROOT / "analysis/classifier_test_260407_sdf_20260527/subgroup_risk_summary.json"
    run_delta_csv = ROOT / "analysis/rl_high_carbon_sparse_reward_20260604/run_start_end_delta.csv"
    reward_sparse_csv = ROOT / "analysis/rl_high_carbon_sparse_reward_20260604/reward_sparsity_by_epoch.csv"

    eval_data = json.loads(eval_json.read_text())
    base_summary = json.loads(base_summary_json.read_text())
    subgroup = json.loads(subgroup_json.read_text())
    run_delta = pd.read_csv(run_delta_csv)
    sparse = pd.read_csv(reward_sparse_csv)

    latest = run_delta.loc[run_delta["run_label"] == "dock_classifier_0529"].iloc[0]
    sparse_latest_epoch25 = sparse.loc[
        (sparse["run_label"] == "dock_classifier_0529") & (sparse["epoch"] == 25)
    ].iloc[0]

    rows = [
        {
            "issue": "静态评估过于乐观",
            "metric": "heldout_test_f1",
            "value": f"{eval_data['test']['f1']:.6f}",
            "source": str(eval_json.relative_to(ROOT)),
            "interpretation": "测试集区分 selected positives/negatives 很好，但不能证明能约束 RL 生成的 OOD 分布。",
        },
        {
            "issue": "基础生成分布上 reward 很稀疏",
            "metric": "base_prob_lt_0_2_rate",
            "value": pct(base_summary["threshold_rates"]["prob_lt_0_2"]),
            "source": str(base_summary_json.relative_to(ROOT)),
            "interpretation": "按当前 cap_low=0.2 映射，大量基础样本直接进入 -1 地板。",
        },
        {
            "issue": "高分组仍不干净",
            "metric": "prob_ge_0_8_ring_complex_rate",
            "value": pct(subgroup["prob_ge_0_8"]["ring_complex"]),
            "source": str(subgroup_json.relative_to(ROOT)),
            "interpretation": "分类器能压极端高碳，但没有充分压住复杂环/局部拓扑问题。",
        },
        {
            "issue": "高分组仍不干净",
            "metric": "prob_ge_0_8_three_member_ring_rate",
            "value": pct(subgroup["prob_ge_0_8"]["three_member_ring"]),
            "source": str(subgroup_json.relative_to(ROOT)),
            "interpretation": "三元环在 classifier 高概率组仍有明显残留。",
        },
        {
            "issue": "正式 RL 出现高碳漂移",
            "metric": "carbon_fraction_epoch1_to_25",
            "value": f"{latest['carbon_fraction_start']:.6f} -> {latest['carbon_fraction_end']:.6f}",
            "source": str(run_delta_csv.relative_to(ROOT)),
            "interpretation": "dock+classifier 后分布仍向高碳移动，说明 classifier 没有守住群体分布。",
        },
        {
            "issue": "正式 RL 出现低极性/低杂原子漂移",
            "metric": "tpsa_epoch1_to_25",
            "value": f"{latest['tpsa_mean_start']:.3f} -> {latest['tpsa_mean_end']:.3f}",
            "source": str(run_delta_csv.relative_to(ROOT)),
            "interpretation": "TPSA 明显下降，与希望保留药化极性/杂原子分布的目标冲突。",
        },
        {
            "issue": "PPO 可用正向组合样本太少",
            "metric": "dock8_and_classifier_positive_valid_rate_epoch25",
            "value": pct(sparse_latest_epoch25["dock8_and_classifier_positive_valid_rate"]),
            "source": str(reward_sparse_csv.relative_to(ROOT)),
            "interpretation": "docking 好且 classifier 为正的有效分子太少，更新主要被单轴 docking 或惩罚主导。",
        },
    ]
    evidence = pd.DataFrame(rows)
    evidence.to_csv(OUT_DIR / "issue_evidence_summary.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Three-layer MLP classifier failure review", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    metrics = {
        "accuracy": eval_data["test"]["accuracy"],
        "ROC-AUC": eval_data["test"]["roc_auc"],
        "F1": eval_data["test"]["f1"],
    }
    ax.bar(metrics.keys(), metrics.values(), color=["#4C78A8", "#72B7B2", "#59A14F"])
    ax.set_ylim(0.98, 1.001)
    ax.set_ylabel("score")
    ax.set_title("Held-out split looks excellent")
    for i, v in enumerate(metrics.values()):
        ax.text(i, v + 0.0005, f"{v:.4f}", ha="center", fontsize=9)

    ax = axes[0, 1]
    thresh = base_summary["threshold_rates"]
    threshold_plot = {
        "P<0.2": thresh["prob_lt_0_2"],
        "P>=0.5": thresh["prob_ge_0_5"],
        "P>=0.8": thresh["prob_ge_0_8"],
    }
    ax.bar(threshold_plot.keys(), [v * 100 for v in threshold_plot.values()], color=["#E15759", "#F28E2B", "#76B7B2"])
    ax.set_ylabel("molecules (%)")
    ax.set_title("Base generated molecules are near-binary")
    for i, v in enumerate(threshold_plot.values()):
        ax.text(i, v * 100 + 1, f"{v * 100:.1f}%", ha="center", fontsize=9)

    ax = axes[1, 0]
    bad_high = {
        "3-ring": subgroup["prob_ge_0_8"]["three_member_ring"],
        "ring complex": subgroup["prob_ge_0_8"]["ring_complex"],
        "C>=0.90": subgroup["prob_ge_0_8"]["c_ge_090"],
    }
    ax.bar(bad_high.keys(), [v * 100 for v in bad_high.values()], color=["#B07AA1", "#EDC948", "#9C755F"])
    ax.set_ylabel("rate in P>=0.8 group (%)")
    ax.set_title("High classifier score is not a clean molecule filter")
    for i, v in enumerate(bad_high.values()):
        ax.text(i, v * 100 + 1, f"{v * 100:.1f}%", ha="center", fontsize=9)

    ax = axes[1, 1]
    drift = {
        "carbon frac": latest["carbon_fraction_end"] / latest["carbon_fraction_start"],
        "hetero atoms": latest["hetero_atom_mean_end"] / latest["hetero_atom_mean_start"],
        "TPSA": latest["tpsa_mean_end"] / latest["tpsa_mean_start"],
        "LogP": latest["logp_mean_end"] / latest["logp_mean_start"],
    }
    ax.axhline(1.0, color="#333333", linewidth=1)
    ax.bar(drift.keys(), drift.values(), color=["#E15759", "#4E79A7", "#59A14F", "#F28E2B"])
    ax.set_ylabel("epoch25 / epoch1")
    ax.set_title("RL still drifts under dock + classifier")
    ax.tick_params(axis="x", rotation=20)
    for i, v in enumerate(drift.values()):
        ax.text(i, v + 0.05, f"{v:.2f}x", ha="center", fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_DIR / "classifier_failure_review.png", dpi=180)
    fig.savefig(OUT_DIR / "classifier_failure_review.svg")


if __name__ == "__main__":
    main()
