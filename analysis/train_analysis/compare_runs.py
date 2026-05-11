"""
对比分析两个 DDPO 训练输出目录。

用法:
    python analysis/train_analysis/compare_runs.py /path/to/run_A /path/to/run_B [--label-a A] [--label-b B]

输出:
    - 在第一个运行目录下生成 compare_vs_<B_label>/ 子文件夹
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning)

# ── 全局样式 ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "figure.figsize": (12, 6),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "lines.linewidth": 1.5,
})

# 两套配色: 主色 + 辅色
COLOR_A = "#1f77b4"  # 蓝色 (KL>0)
COLOR_B = "#d62728"  # 红色 (KL=0)
COLOR_A_LIGHT = "#73b4dc"
COLOR_B_LIGHT = "#e88a8a"
DIFF_COLOR = "#2ca02c"  # 绿色 (差值)

# ══════════════════════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════════════════════


def _safe_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _parse_failure_counts(failure_str: str) -> dict[str, int]:
    if isinstance(failure_str, dict):
        return failure_str
    try:
        return json.loads(failure_str.replace("'", '"'))
    except (json.JSONDecodeError, AttributeError):
        return {}


def _load_training_summary(run_dir: str) -> pd.DataFrame | None:
    path = os.path.join(run_dir, "training_summary.csv")
    if not os.path.isfile(path):
        print(f"[WARN] 未找到 {path}")
        return None
    df = pd.read_csv(path)
    df["failure_counts"] = df["failure_counts"].apply(_parse_failure_counts)
    return df


def _load_epoch_molecules(run_dir: str, epoch: int) -> pd.DataFrame | None:
    path = os.path.join(run_dir, "epochs", f"epoch_{epoch:04d}", "molecules.csv")
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path)


def _get_available_epochs(run_dir: str) -> list[int]:
    epochs_dir = os.path.join(run_dir, "epochs")
    if not os.path.isdir(epochs_dir):
        return []
    epochs = []
    for d in os.listdir(epochs_dir):
        if m := re.match(r"epoch_(\d+)", d):
            epochs.append(int(m.group(1)))
    return sorted(epochs)


def _make_comparison_df(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    """合并两个 summary，计算差值 (A - B) 列。"""
    merged = df_a[["epoch"]].copy()
    merged = merged.rename(columns={"epoch": "epoch"})

    col_map = {
        "reward_mean": "Reward Mean",
        "reward_max": "Reward Max",
        "reward_min": "Reward Min",
        "avg_qed": "QED",
        "avg_sa": "SA",
        "avg_docking": "Docking",
        "rdkit_valid_ratio": "Valid Ratio",
        "avg_kl": "KL",
        "avg_ppo_loss": "PPO Loss",
        "avg_qed_norm": "QED Norm",
        "avg_sa_norm": "SA Norm",
        "avg_docking_norm": "Docking Norm",
    }

    for col, label in col_map.items():
        if col in df_a.columns and col in df_b.columns:
            merged[f"{col}_a"] = df_a[col].values
            merged[f"{col}_b"] = df_b[col].values
            merged[f"{col}_diff"] = df_a[col].values - df_b[col].values
    return merged


# ══════════════════════════════════════════════════════════════════════════
#  Part 1: 训练曲线叠加对比
# ══════════════════════════════════════════════════════════════════════════


def plot_overlaid_curves(
    summary_a: pd.DataFrame, summary_b: pd.DataFrame,
    label_a: str, label_b: str, save_dir: str,
) -> str:
    """叠加对比: Reward, Sub-Rewards, KL, Loss, 有效性。"""
    epochs_a = summary_a["epoch"]
    epochs_b = summary_b["epoch"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # ── 奖励曲线 ──
    ax = axes[0, 0]
    ax.plot(epochs_a, summary_a["reward_mean"], label=f"{label_a} mean", color=COLOR_A)
    ax.plot(epochs_b, summary_b["reward_mean"], label=f"{label_b} mean", color=COLOR_B)
    ax.fill_between(epochs_a, summary_a["reward_min"], summary_a["reward_max"],
                     alpha=0.15, color=COLOR_A, label=f"{label_a} range")
    ax.fill_between(epochs_b, summary_b["reward_min"], summary_b["reward_max"],
                     alpha=0.15, color=COLOR_B, label=f"{label_b} range")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reward (z-score)")
    ax.set_title("Reward Comparison")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── 子奖励原始值 ──
    ax = axes[0, 1]
    ax.plot(epochs_a, summary_a["avg_qed"], label=f"{label_a} QED", color=COLOR_A)
    ax.plot(epochs_b, summary_b["avg_qed"], label=f"{label_b} QED", color=COLOR_B, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("QED")
    ax.set_title("QED (raw)")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── SA & Docking ──
    ax = axes[0, 2]
    ax.plot(epochs_a, summary_a["avg_sa"], label=f"{label_a} SA", color=COLOR_A, alpha=0.7)
    ax.plot(epochs_b, summary_b["avg_sa"], label=f"{label_b} SA", color=COLOR_B, alpha=0.7)
    ax2 = ax.twinx()
    ax2.plot(epochs_a, summary_a["avg_docking"], label=f"{label_a} Docking", color=COLOR_A, linestyle=":")
    ax2.plot(epochs_b, summary_b["avg_docking"], label=f"{label_b} Docking", color=COLOR_B, linestyle=":")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("SA")
    ax2.set_ylabel("Docking Score")
    ax.set_title("SA & Docking (raw)")
    ax.grid(alpha=0.3)
    ax2.invert_yaxis()

    # ── PPO Loss ──
    ax = axes[1, 0]
    ax.plot(epochs_a, summary_a["avg_ppo_loss"], label=f"{label_a} Loss", color=COLOR_A)
    ax.plot(epochs_b, summary_b["avg_ppo_loss"], label=f"{label_b} Loss", color=COLOR_B)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("PPO Loss")
    ax.set_title("PPO Loss Comparison")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── KL ──
    ax = axes[1, 1]
    if "avg_kl" in summary_a.columns and "avg_kl" in summary_b.columns:
        ax.plot(epochs_a, summary_a["avg_kl"], label=f"{label_a} KL", color=COLOR_A)
        ax.plot(epochs_b, summary_b["avg_kl"], label=f"{label_b} KL", color=COLOR_B)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("KL Divergence")
    ax.set_title("KL Divergence (log scale)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── 有效性 ──
    ax = axes[1, 2]
    ax.plot(epochs_a, summary_a["rdkit_valid_ratio"] * 100, label=f"{label_a}", color=COLOR_A)
    ax.plot(epochs_b, summary_b["rdkit_valid_ratio"] * 100, label=f"{label_b}", color=COLOR_B)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Valid Ratio (%)")
    ax.set_title("Molecule Validity")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "01_overlaid_curves.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════════════════
#  Part 2: 差值曲线
# ══════════════════════════════════════════════════════════════════════════


def plot_difference_curves(comp: pd.DataFrame, save_dir: str) -> str:
    """A - B 差值曲线 (A 有 KL, B 无 KL)。"""
    epochs = comp["epoch"]

    diff_cols = [
        ("reward_mean_diff", "Reward Mean Diff (A - B)"),
        ("avg_qed_diff", "QED Diff (A - B)"),
        ("avg_docking_diff", "Docking Diff (A - B)"),
        ("rdkit_valid_ratio_diff", "Valid Ratio Diff (A - B)"),
        ("avg_ppo_loss_diff", "PPO Loss Diff (A - B)"),
    ]
    n = len(diff_cols)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)

    for idx, (col, title) in enumerate(diff_cols):
        ax = axes[idx]
        ax.plot(epochs, comp[col], color=DIFF_COLOR)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.6)
        ax.fill_between(epochs, 0, comp[col], alpha=0.2, color=DIFF_COLOR,
                         where=(comp[col] >= 0), label="A > B")
        ax.fill_between(epochs, 0, comp[col], alpha=0.2, color=COLOR_B,
                         where=(comp[col] < 0), label="B > A")
        ax.set_ylabel(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Epoch")
    plt.tight_layout()
    path = os.path.join(save_dir, "02_difference_curves.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════════════════
#  Part 3: 失败模式对比
# ══════════════════════════════════════════════════════════════════════════


def plot_failure_comparison(
    summary_a: pd.DataFrame, summary_b: pd.DataFrame,
    label_a: str, label_b: str, save_dir: str,
) -> str:
    """两个运行的失败模式堆叠柱状图对比。"""
    def get_failure_types(summary):
        ft_set: set[str] = set()
        for fc in summary["failure_counts"]:
            ft_set.update(fc.keys())
        ft_set.discard("success")
        return sorted(ft_set)

    ft_a = get_failure_types(summary_a)
    ft_b = get_failure_types(summary_b)
    all_types = sorted(set(ft_a) | set(ft_b))

    if not all_types:
        return ""

    def total_failures_by_type(summary, failure_types):
        totals = Counter()
        for fc in summary["failure_counts"]:
            for k, v in fc.items():
                if k != "success":
                    totals[k] += v
        return [totals.get(ft, 0) for ft in failure_types]

    totals_a = total_failures_by_type(summary_a, all_types)
    totals_b = total_failures_by_type(summary_b, all_types)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── 堆叠面积图对比 ──
    ax = axes[0]
    epochs = summary_a["epoch"]
    data_a = {ft: [] for ft in all_types}
    data_b = {ft: [] for ft in all_types}
    for fc_a, fc_b in zip(summary_a["failure_counts"], summary_b["failure_counts"]):
        total_a = sum(v for k, v in fc_a.items() if k != "success") or 1
        total_b = sum(v for k, v in fc_b.items() if k != "success") or 1
        for ft in all_types:
            data_a[ft].append(fc_a.get(ft, 0) / total_a * 100)
            data_b[ft].append(fc_b.get(ft, 0) / total_b * 100)

    ax.stackplot(epochs, data_a.values(), labels=[f"{label_a}: {ft}" for ft in all_types],
                  alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Proportion (%)")
    ax.set_title(f"Failure Type — {label_a}")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.stackplot(epochs, data_b.values(), labels=[f"{label_b}: {ft}" for ft in all_types],
                  alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Proportion (%)")
    ax.set_title(f"Failure Type — {label_b}")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "03_failure_comparison.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

    # ── 额外: 失败类型总量对比条形图 ──
    fig, ax = plt.subplots(figsize=(10, max(4, len(all_types) * 0.6)))
    y = np.arange(len(all_types))
    h = 0.35
    ax.barh(y - h / 2, totals_a, h, label=label_a, color=COLOR_A, alpha=0.8)
    ax.barh(y + h / 2, totals_b, h, label=label_b, color=COLOR_B, alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(all_types)
    ax.set_xlabel("Total Failure Count")
    ax.set_title("Total Failure Count by Type")
    ax.legend()
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    path2 = os.path.join(save_dir, "03b_failure_total_bars.png")
    fig.savefig(path2, bbox_inches="tight")
    plt.close(fig)
    return f"{path}, {path2}"


# ══════════════════════════════════════════════════════════════════════════
#  Part 4: 最佳分子对比
# ══════════════════════════════════════════════════════════════════════════


def analyze_best_molecules(
    run_dir: str, label: str, epochs: list[int], top_n: int = 10,
) -> pd.DataFrame:
    """提取该运行的全运行最佳分子榜单。"""
    all_best: list[dict] = []
    for ep in epochs:
        df = _load_epoch_molecules(run_dir, ep)
        if df is None or df.empty:
            continue
        valid = df[df["rdkit_valid"] == 1].copy()
        if valid.empty:
            continue
        top = valid.nlargest(top_n, "reward_total")
        for _, row in top.iterrows():
            d = row.to_dict()
            d["run_label"] = label
            all_best.append(d)

    df_all = pd.DataFrame(all_best) if all_best else pd.DataFrame()
    if not df_all.empty:
        df_all = df_all.nlargest(top_n * 5, "reward_total").drop_duplicates(subset="smiles")
    return df_all


def plot_best_molecule_comparison(
    all_best_a: pd.DataFrame, all_best_b: pd.DataFrame,
    label_a: str, label_b: str, save_dir: str,
) -> str:
    """两个运行的最佳分子性质对比散点/分布图。"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # ── QED vs Docking 散点 ──
    ax = axes[0, 0]
    if not all_best_a.empty:
        ax.scatter(all_best_a["qed_raw"], all_best_a["docking_raw"],
                    s=20, alpha=0.6, color=COLOR_A, label=label_a, edgecolors="none")
    if not all_best_b.empty:
        ax.scatter(all_best_b["qed_raw"], all_best_b["docking_raw"],
                    s=20, alpha=0.6, color=COLOR_B, label=label_b, marker="s", edgecolors="none")
    ax.set_xlabel("QED")
    ax.set_ylabel("Docking Score")
    ax.set_title("Top Molecules: QED vs Docking")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.invert_yaxis()

    # ── Reward 分布直方图 ──
    ax = axes[0, 1]
    if not all_best_a.empty:
        ax.hist(all_best_a["reward_total"], bins=20, alpha=0.6, color=COLOR_A, label=label_a)
    if not all_best_b.empty:
        ax.hist(all_best_b["reward_total"], bins=20, alpha=0.6, color=COLOR_B, label=label_b)
    ax.set_xlabel("Reward (z-score)")
    ax.set_ylabel("Count")
    ax.set_title("Top Molecules: Reward Distribution")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── Docking 分布 ──
    ax = axes[1, 0]
    if not all_best_a.empty:
        ax.hist(all_best_a["docking_raw"], bins=20, alpha=0.6, color=COLOR_A, label=label_a)
    if not all_best_b.empty:
        ax.hist(all_best_b["docking_raw"], bins=20, alpha=0.6, color=COLOR_B, label=label_b)
    ax.set_xlabel("Docking Score")
    ax.set_ylabel("Count")
    ax.set_title("Top Molecules: Docking Distribution")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── Best per epoch trend ──
    ax = axes[1, 1]

    def best_per_epoch_trend(run_dir, epochs):
        result = []
        for ep in epochs:
            df = _load_epoch_molecules(run_dir, ep)
            if df is None or df.empty:
                continue
            valid = df[df["rdkit_valid"] == 1]
            if valid.empty:
                continue
            best = valid.loc[valid["reward_total"].idxmax()]
            result.append({"epoch": ep, "docking_raw": best["docking_raw"],
                           "qed_raw": best["qed_raw"], "reward_total": best["reward_total"]})
        return pd.DataFrame(result)

    run_a_epochs = _get_available_epochs(
        os.path.dirname(all_best_a.iloc[0]["epoch_path"]) if "epoch_path" in all_best_a.columns else ""
    )
    # fallback: use molecules dir
    run_a_dir = save_dir.replace("/compare_vs_", "/../")  # hacky, let's just use the original

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Best Docking Score")
    ax.set_title("Best Molecule Docking per Epoch")
    ax.invert_yaxis()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "04_best_molecule_comparison.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════════════════
#  Part 5: 统计指标表格对比
# ══════════════════════════════════════════════════════════════════════════


def compute_summary_stats(summary: pd.DataFrame) -> dict[str, str]:
    """计算关键指标的 起始/均值/最佳/最终。"""
    stats = {}
    metrics = [
        ("reward_mean", ".4f", True),
        ("avg_qed", ".4f", True),
        ("avg_sa", ".2f", False),
        ("avg_docking", ".2f", False),
        ("rdkit_valid_ratio", ".1%", True),
        ("avg_kl", ".6f", None),
        ("avg_ppo_loss", ".4f", False),
    ]
    for metric, fmt, higher_better in metrics:
        if metric not in summary.columns:
            continue
        first = summary[metric].iloc[0]
        last = summary[metric].iloc[-1]
        mean_val = summary[metric].mean()
        if higher_better is True:
            best = summary[metric].max()
        elif higher_better is False:
            best = summary[metric].min()
        else:
            best = mean_val
        stats[metric] = f"{first:{fmt}} → {mean_val:{fmt}} → {last:{fmt}} (best: {best:{fmt}})"
    return stats


# ══════════════════════════════════════════════════════════════════════════
#  Part 6: 报告生成
# ══════════════════════════════════════════════════════════════════════════


def generate_report(
    label_a: str, label_b: str,
    summary_a: pd.DataFrame, summary_b: pd.DataFrame,
    plot_paths: list[str],
    save_dir: str,
) -> str:
    lines: list[str] = []

    lines.append("=" * 72)
    lines.append(f"  对比分析报告 — {label_a} vs {label_b}")
    lines.append("=" * 72)
    lines.append("")

    n_epochs = min(len(summary_a), len(summary_b))
    lines.append(f"  Epochs: {n_epochs}")
    lines.append(f"  {label_a}: {len(summary_a)} epochs")
    lines.append(f"  {label_b}: {len(summary_b)} epochs")
    lines.append("")

    # ── 1. 关键指标对比表格 ──
    lines.append("-" * 72)
    lines.append("  1. 关键指标对比 (start → mean → end, best)")
    lines.append("-" * 72)
    lines.append("")

    stats_a = compute_summary_stats(summary_a)
    stats_b = compute_summary_stats(summary_b)

    metric_labels = {
        "reward_mean": "Reward Mean",
        "avg_qed": "QED",
        "avg_sa": "SA",
        "avg_docking": "Docking",
        "rdkit_valid_ratio": "Valid Ratio",
        "avg_kl": "KL",
        "avg_ppo_loss": "PPO Loss",
    }

    lines.append(f"  {'指标':<25} {label_a:<45} {label_b:<45}")
    lines.append("  " + "-" * 115)
    for metric, label in metric_labels.items():
        sa = stats_a.get(metric, "N/A")
        sb = stats_b.get(metric, "N/A")
        lines.append(f"  {label:<25} {sa:<45} {sb:<45}")

    lines.append("")

    # ── 2. 差异摘要 ──
    lines.append("-" * 72)
    lines.append("  2. 差异摘要 (A - B)")
    lines.append("-" * 72)
    lines.append("")

    comp = _make_comparison_df(summary_a, summary_b)
    diff_cols = {
        "reward_mean_diff": "Reward Mean Diff",
        "avg_qed_diff": "QED Diff",
        "avg_docking_diff": "Docking Diff",
        "rdkit_valid_ratio_diff": "Valid Ratio Diff",
        "avg_kl_diff": "KL Diff",
    }
    for col, label in diff_cols.items():
        if col not in comp.columns:
            continue
        mean_diff = comp[col].mean()
        min_diff = comp[col].min()
        max_diff = comp[col].max()
        direction = "A > B" if mean_diff > 0 else "B > A"
        lines.append(f"  {label:<25}  mean={mean_diff:+.4f}  [{min_diff:+.4f}, {max_diff:+.4f}]  ← {direction}")

    lines.append("")

    # ── 3. 有效性对比 ──
    lines.append("-" * 72)
    lines.append("  3. 有效性对比")
    lines.append("-" * 72)
    lines.append("")
    for label, summary in [(label_a, summary_a), (label_b, summary_b)]:
        avg_v = summary["rdkit_valid_ratio"].mean() * 100
        min_v = summary["rdkit_valid_ratio"].min() * 100
        max_v = summary["rdkit_valid_ratio"].max() * 100
        lines.append(f"  {label:<20}  Valid: {avg_v:.1f}%  [{min_v:.1f}% ~ {max_v:.1f}%]")

    # ── 4. 失败模式对比 ──
    lines.append("")
    lines.append("-" * 72)
    lines.append("  4. 失败模式 (全 epoch 汇总)")
    lines.append("-" * 72)
    lines.append("")

    for label, summary in [(label_a, summary_a), (label_b, summary_b)]:
        all_failures: Counter = Counter()
        for fc in summary["failure_counts"]:
            for k, v in fc.items():
                if k != "success":
                    all_failures[k] += v
        total_fails = sum(all_failures.values())
        total_gen = summary["generated_total"].sum()
        total_valid = summary["rdkit_valid"].sum()
        lines.append(f"  ── {label} ──")
        lines.append(f"  总生成: {total_gen}, 有效: {total_valid}, 失败: {total_fails}")
        for ft, cnt in sorted(all_failures.items(), key=lambda x: -x[1]):
            lines.append(f"    {ft:<30} {cnt:>6} ({cnt / total_fails * 100:.1f}%)" if total_fails > 0 else f"    {ft}: 0")

    lines.append("")

    # ── 5. 图表 ──
    lines.append("-" * 72)
    lines.append("  5. 生成图表")
    lines.append("-" * 72)
    lines.append("")
    for p in plot_paths:
        if p:
            lines.append(f"  📊 {os.path.basename(p)}")
    lines.append("")

    lines.append("=" * 72)

    report = "\n".join(lines)
    report_path = os.path.join(save_dir, "report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    return report_path


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="对比分析两个 DDPO 训练输出目录",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""用法示例:
  python analysis/train_analysis/compare_runs.py \\
      OUT/kl-0.01_20260511_162333 \\
      OUT/kl-0_20260511_153246 \\
      --label-a "KL=0.01" --label-b "KL=0"
""",
    )
    parser.add_argument("run_a", type=str, help="第一个训练目录 (如 KL>0)")
    parser.add_argument("run_b", type=str, help="第二个训练目录 (如 KL=0)")
    parser.add_argument("--label-a", type=str, default=None, help="运行 A 标签 (默认使用目录名)")
    parser.add_argument("--label-b", type=str, default=None, help="运行 B 标签 (默认使用目录名)")
    parser.add_argument("--top-n", type=int, default=10, help="每个 epoch 保留的最佳分子数")
    args = parser.parse_args()

    run_a = os.path.abspath(args.run_a)
    run_b = os.path.abspath(args.run_b)

    for d, name in [(run_a, "run_a"), (run_b, "run_b")]:
        if not os.path.isdir(d):
            print(f"[ERROR] 目录不存在: {d}")
            sys.exit(1)

    label_a = args.label_a or os.path.basename(run_a.rstrip("/"))
    label_b = args.label_b or os.path.basename(run_b.rstrip("/"))

    # 输出到 run_a 目录下
    save_dir = _safe_dir(os.path.join(run_a, f"compare_vs_{label_b.replace('/', '_')}"))
    print(f"\n📁 对比输出: {save_dir}\n")

    # ── 加载数据 ──
    summary_a = _load_training_summary(run_a)
    summary_b = _load_training_summary(run_b)

    if summary_a is None or summary_b is None:
        print("[ERROR] 缺少 training_summary.csv")
        sys.exit(1)

    # 截取到共有的 epoch 数
    min_epochs = min(len(summary_a), len(summary_b))
    summary_a = summary_a.iloc[:min_epochs].reset_index(drop=True)
    summary_b = summary_b.iloc[:min_epochs].reset_index(drop=True)

    print(f"✅ {label_a}: {len(summary_a)} epochs")
    print(f"✅ {label_b}: {len(summary_b)} epochs")

    plot_paths: list[str] = []

    # ── 1. 叠加曲线 ──
    print("  绘图: 训练曲线叠加对比...")
    plot_paths.append(plot_overlaid_curves(summary_a, summary_b, label_a, label_b, save_dir))

    # ── 2. 差值曲线 ──
    print("  绘图: 差值曲线...")
    comp = _make_comparison_df(summary_a, summary_b)
    plot_paths.append(plot_difference_curves(comp, save_dir))

    # ── 3. 失败模式对比 ──
    print("  绘图: 失败模式对比...")
    plot_paths.append(plot_failure_comparison(summary_a, summary_b, label_a, label_b, save_dir))

    # ── 4. 报告 ──
    print("  生成: 对比分析报告...")
    report_path = generate_report(
        label_a, label_b, summary_a, summary_b, plot_paths, save_dir,
    )

    print(f"\n✅ 对比分析完成!\n")
    print(f"  报告: {report_path}")
    for p in plot_paths:
        if p:
            for sub_p in p.split(", "):
                if sub_p.strip():
                    print(f"  图表: {sub_p.strip()}")

    print()
    try:
        with open(report_path) as f:
            print(f.read())
    except Exception:
        pass


if __name__ == "__main__":
    main()
