"""
分析一次 DDPO 训练的完整输出目录。

用法:
    python analysis/train_analysis/analyze_run.py /path/to/OUT/run_name_YYYYMMDD_HHMMSS

输出:
    - 在输出目录下生成 analysis/ 子文件夹，包含所有图表和报告
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
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

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

COLORS = sns.color_palette("husl", 8)

# ══════════════════════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════════════════════


def _safe_dir(path: str) -> str:
    """确保路径存在，返回原路径。"""
    os.makedirs(path, exist_ok=True)
    return path


def _parse_failure_counts(failure_str: str) -> dict[str, int]:
    """解析 failure_counts 字段，支持字符串 dict 或已解析的 dict。"""
    if isinstance(failure_str, dict):
        return failure_str
    try:
        return json.loads(failure_str.replace("'", '"'))
    except (json.JSONDecodeError, AttributeError):
        return {}


def _load_training_summary(run_dir: str) -> pd.DataFrame | None:
    """加载 training_summary.csv。"""
    path = os.path.join(run_dir, "training_summary.csv")
    if not os.path.isfile(path):
        print(f"[WARN] 未找到 {path}")
        return None
    df = pd.read_csv(path)
    df["failure_counts"] = df["failure_counts"].apply(_parse_failure_counts)
    return df


def _load_epoch_molecules(run_dir: str, epoch: int) -> pd.DataFrame | None:
    """加载指定 epoch 的 molecules.csv。"""
    path = os.path.join(run_dir, "epochs", f"epoch_{epoch:04d}", "molecules.csv")
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path)


def _load_epoch_metrics(run_dir: str, epoch: int) -> dict[str, Any] | None:
    """加载指定 epoch 的 metrics.json。"""
    path = os.path.join(run_dir, "epochs", f"epoch_{epoch:04d}", "metrics.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _get_available_epochs(run_dir: str) -> list[int]:
    """获取所有存在的 epoch 编号。"""
    epochs_dir = os.path.join(run_dir, "epochs")
    if not os.path.isdir(epochs_dir):
        return []
    epochs = []
    for d in os.listdir(epochs_dir):
        if m := re.match(r"epoch_(\d+)", d):
            epochs.append(int(m.group(1)))
    return sorted(epochs)


# ══════════════════════════════════════════════════════════════════════════
#  第一部分: 训练曲线
# ══════════════════════════════════════════════════════════════════════════


def plot_training_curves(summary: pd.DataFrame, save_dir: str) -> str:
    """核心训练指标曲线: reward, QED, SA, Docking, KL, Loss, 有效性。"""
    epochs = summary["epoch"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # ── 奖励曲线 ──
    ax = axes[0, 0]
    ax.plot(epochs, summary["reward_mean"], label="reward_mean", color=COLORS[0])
    ax.plot(epochs, summary["reward_max"], label="reward_max", color=COLORS[1], alpha=0.7)
    ax.plot(epochs, summary["reward_min"], label="reward_min", color=COLORS[2], alpha=0.7)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reward (z-score)")
    ax.set_title("Train Reward")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── 子奖励 ──
    ax = axes[0, 1]
    if "avg_qed_norm" in summary.columns:
        ax.plot(epochs, summary["avg_qed_norm"], label="QED (norm)", color=COLORS[0])
        ax.plot(epochs, summary["avg_sa_norm"], label="SA (norm)", color=COLORS[1])
        ax.plot(epochs, summary["avg_docking_norm"], label="Docking (norm)", color=COLORS[2])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalized Reward")
    ax.set_title("Sub-Rewards (z-score)")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── 子奖励原始值 ──
    ax = axes[0, 2]
    ax.plot(epochs, summary["avg_qed"], label="QED raw", color=COLORS[0])
    ax.plot(epochs, summary["avg_sa"], label="SA raw", color=COLORS[3])
    ax2 = ax.twinx()
    ax2.plot(epochs, summary["avg_docking"], label="Docking raw", color=COLORS[2], alpha=0.8)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("QED / SA")
    ax2.set_ylabel("Docking Score")
    ax.set_title("Raw Sub-Rewards")
    ax.grid(alpha=0.3)

    # ── 损失 ──
    ax = axes[1, 0]
    ax.plot(epochs, summary["avg_ppo_loss"], label="PPO Loss", color=COLORS[4])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("PPO Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── KL ──
    ax = axes[1, 1]
    ax.plot(epochs, summary["avg_kl"], label="KL Divergence", color=COLORS[5])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("KL")
    ax.set_title("KL Divergence (log scale)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── 有效性 ──
    ax = axes[1, 2]
    ax.plot(epochs, summary["rdkit_valid_ratio"] * 100, label="RDKit Valid %", color=COLORS[6])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Valid Ratio (%)")
    ax.set_title("Molecule Validity")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "01_training_curves.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════════════════
#  第二部分: 失败模式分析
# ══════════════════════════════════════════════════════════════════════════


def plot_failure_analysis(summary: pd.DataFrame, save_dir: str) -> str:
    """失败类型分布随时间的变化。"""
    if "failure_counts" not in summary.columns:
        return ""

    failure_types: set[str] = set()
    for fc in summary["failure_counts"]:
        failure_types.update(fc.keys())
    failure_types.discard("success")
    failure_types = sorted(failure_types)

    if not failure_types:
        return ""

    epochs = summary["epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── 堆叠面积图 ──
    ax = axes[0]
    data = {ft: [] for ft in failure_types}
    for fc in summary["failure_counts"]:
        total = sum(v for k, v in fc.items() if k != "success")
        for ft in failure_types:
            data[ft].append(fc.get(ft, 0) / max(total, 1) * 100 if total > 0 else 0)

    ax.stackplot(epochs, data.values(), labels=failure_types, alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Proportion (%)")
    ax.set_title("Failure Type Composition")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    # ── 失败总数 ──
    ax = axes[1]
    success = summary["rdkit_valid"]
    total = summary["generated_total"]
    fails = total - success
    ax.fill_between(epochs, 0, fails, label="Failed", alpha=0.4, color=COLORS[1])
    ax.fill_between(epochs, fails, total, label="Valid", alpha=0.4, color=COLORS[0])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Molecule Count")
    ax.set_title("Valid vs Failed Molecules")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "02_failure_analysis.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════════════════
#  第三部分: 逐 epoch 分子级分析
# ══════════════════════════════════════════════════════════════════════════


def analyze_molecule_properties(
    run_dir: str, epochs: list[int], top_n: int = 10
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    从所有 epoch 逐分子数据中提取:
      1. 每个 epoch 的最佳分子 (前 top_n)
      2. 全运行最佳分子榜单
    """
    best_per_epoch: list[dict] = []
    all_best: list[dict] = []

    for ep in epochs:
        df = _load_epoch_molecules(run_dir, ep)
        if df is None or df.empty:
            continue
        valid = df[df["rdkit_valid"] == 1].copy()
        if valid.empty:
            continue

        # 用 reward_total (z-score 后) 作为排序标准
        top = valid.nlargest(top_n, "reward_total")
        for _, row in top.iterrows():
            all_best.append(row.to_dict())

        best = valid.loc[valid["reward_total"].idxmax()]
        best_per_epoch.append(
            {
                "epoch": ep,
                "reward_total": best["reward_total"],
                "reward_total_raw": best["reward_total_raw"],
                "qed_raw": best["qed_raw"],
                "sa_raw": best["sa_raw"],
                "docking_raw": best["docking_raw"],
                "smiles": best["smiles"],
            }
        )

    df_best_per_epoch = pd.DataFrame(best_per_epoch) if best_per_epoch else pd.DataFrame()
    df_all_best = pd.DataFrame(all_best) if all_best else pd.DataFrame()
    if not df_all_best.empty:
        df_all_best = df_all_best.nlargest(top_n * 5, "reward_total").drop_duplicates(
            subset="smiles"
        )
    return df_best_per_epoch, df_all_best


def plot_best_molecules_trend(best_per_epoch: pd.DataFrame, save_dir: str) -> str:
    """每个 epoch 的最佳分子评分趋势。"""
    if best_per_epoch.empty:
        return ""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 最佳分子 reward_total 趋势
    ax = axes[0]
    ax.plot(
        best_per_epoch["epoch"],
        best_per_epoch["reward_total"],
        marker="o", markersize=3, color=COLORS[0], label="Best reward_total (z-score)",
    )
    ax.plot(
        best_per_epoch["epoch"],
        best_per_epoch["reward_total_raw"],
        marker="s", markersize=3, color=COLORS[1], alpha=0.7, label="Best reward_total_raw",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reward")
    ax.set_title("Best Molecule Reward per Epoch")
    ax.legend()
    ax.grid(alpha=0.3)

    # 最佳分子 docking 趋势
    ax = axes[1]
    ax.plot(
        best_per_epoch["epoch"],
        best_per_epoch["docking_raw"],
        marker="o", markersize=3, color=COLORS[2],
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Docking Score")
    ax.set_title("Best Molecule Docking per Epoch")
    ax.axhline(y=-10, color="green", linestyle="--", alpha=0.5, label="Strong binder (-10)")
    ax.axhline(y=-7, color="orange", linestyle="--", alpha=0.5, label="Good binder (-7)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.invert_yaxis()  # docking 越负越好

    plt.tight_layout()
    path = os.path.join(save_dir, "03_best_molecule_trend.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════════════════
#  第四部分: 分布可视化
# ══════════════════════════════════════════════════════════════════════════


def plot_property_distributions(
    run_dir: str, sample_epochs: list[int], save_dir: str
) -> str:
    """选定 epoch 的分子性质分布对比。"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    epoch_data: dict[int, pd.DataFrame] = {}
    for ep in sample_epochs:
        df = _load_epoch_molecules(run_dir, ep)
        if df is not None and not df.empty:
            epoch_data[ep] = df[df["rdkit_valid"] == 1].copy()

    if not epoch_data:
        plt.close(fig)
        return ""

    props = [
        ("qed_raw", "QED", 0, 1),
        ("sa_raw", "SA", 1, 10),
        ("docking_raw", "Docking Score", None, None),
        ("reward_total", "Reward (z-score)", None, None),
        ("qed_reward", "QED Reward", -1, 1),
        ("docking_reward", "Docking Reward", -1, 1.5),
    ]

    for idx, (col, label, xmin, xmax) in enumerate(props):
        ax = axes[idx // 3, idx % 3]
        for i, (ep, df) in enumerate(epoch_data.items()):
            if col in df.columns:
                data = df[col].dropna()
                if len(data) > 0:
                    sns.kdeplot(
                        data, ax=ax, label=f"Epoch {ep}", color=COLORS[i % len(COLORS)],
                        fill=True, alpha=0.3,
                    )
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.set_title(f"Distribution of {label}")
        if xmin is not None:
            ax.set_xlim(xmin, xmax)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "04_property_distributions.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════════════════
#  第五部分: 奖励相关性矩阵 + 散点图
# ══════════════════════════════════════════════════════════════════════════


def plot_reward_correlation(run_dir: str, epoch: int, save_dir: str) -> str:
    """指定 epoch 的分子级奖励相关性散点图矩阵。"""
    df = _load_epoch_molecules(run_dir, epoch)
    if df is None or df.empty:
        return ""

    valid = df[df["rdkit_valid"] == 1].copy()
    if valid.empty:
        return ""

    cols = ["qed_raw", "sa_raw", "docking_raw", "reward_total_raw", "reward_total"]
    available = [c for c in cols if c in valid.columns]

    fig, axes = plt.subplots(len(available), len(available), figsize=(14, 14))

    for i, col_i in enumerate(available):
        for j, col_j in enumerate(available):
            ax = axes[i, j]
            if i == j:
                ax.hist(valid[col_i].dropna(), bins=25, color=COLORS[0], alpha=0.7)
                ax.set_title(col_i, fontsize=9)
            else:
                ax.scatter(
                    valid[col_j], valid[col_i], s=15, alpha=0.5, color=COLORS[0], edgecolors="none",
                )
                # 计算相关系数
                mask = valid[col_i].notna() & valid[col_j].notna()
                if mask.sum() > 2:
                    corr = valid.loc[mask, col_i].corr(valid.loc[mask, col_j])
                    ax.text(
                        0.95, 0.95, f"r={corr:.2f}", transform=ax.transAxes,
                        ha="right", va="top", fontsize=9,
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                    )
            ax.tick_params(labelsize=7)
            if j > 0:
                ax.set_ylabel("")
            if i < len(available) - 1:
                ax.set_xlabel("")

    plt.tight_layout()
    path = os.path.join(save_dir, f"05_correlation_epoch{epoch:04d}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════════════════
#  第六部分: 子奖励趋势对比 (raw vs norm)
# ══════════════════════════════════════════════════════════════════════════


def plot_raw_vs_norm_rewards(summary: pd.DataFrame, save_dir: str) -> str:
    """raw vs normalized 子奖励的对比，说明 z-score 的影响。"""
    if "avg_qed_norm" not in summary.columns:
        return ""

    epochs = summary["epoch"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, (raw_col, norm_col, label) in enumerate(
        [
            ("avg_qed", "avg_qed_norm", "QED"),
            ("avg_sa", "avg_sa_norm", "SA"),
            ("avg_docking", "avg_docking_norm", "Docking"),
        ]
    ):
        ax = axes[idx]
        ax2 = ax.twinx()

        l1 = ax.plot(epochs, summary[raw_col], label=f"{label} raw", color=COLORS[0])
        l2 = ax2.plot(epochs, summary[norm_col], label=f"{label} norm", color=COLORS[1], alpha=0.8)

        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"{label} Raw", color=COLORS[0])
        ax2.set_ylabel(f"{label} Norm (z-score)", color=COLORS[1])
        ax.set_title(f"{label}: Raw vs Normalized")

        ax.tick_params(axis="y", labelcolor=COLORS[0])
        ax2.tick_params(axis="y", labelcolor=COLORS[1])
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "06_raw_vs_norm.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════════════════
#  第七部分: 报告生成
# ══════════════════════════════════════════════════════════════════════════


def generate_report(
    run_dir: str,
    summary: pd.DataFrame | None,
    best_per_epoch: pd.DataFrame,
    all_best: pd.DataFrame,
    run_config: dict | None,
    plot_paths: list[str],
    save_dir: str,
) -> str:
    """生成结构化的文本分析报告。"""
    lines: list[str] = []

    # ── 总体信息 ──
    run_name = os.path.basename(run_dir.rstrip("/"))

    lines.append("=" * 72)
    lines.append(f"  训练分析报告 — {run_name}")
    lines.append("=" * 72)
    lines.append("")

    if run_config:
        lines.append(f"  靶标: {run_config.get('pocket', {}).get('pdb_path', 'N/A')}")
        lines.append(f"  GPU: {run_config.get('ddp', {}).get('num_gpus', 'N/A')}")
        reward_weights = run_config.get("reward", {}).get("weights", {})
        lines.append(f"  奖励权重: {reward_weights}")
        norm_cfg = run_config.get("reward", {}).get("normalization", {})
        norm_status = "开启" if norm_cfg.get("enabled", True) else "关闭"
        lines.append(f"  z-score 归一化: {norm_status}")
        lines.append("")

    if summary is not None and not summary.empty:
        n_epochs = len(summary)
        last = summary.iloc[-1]

        lines.append(f"  总 epochs: {n_epochs}")
        lines.append(f"  分子/epoch: {summary['generated_total'].iloc[0]}")
        lines.append("")

        # ── 关键指标总结 ──
        lines.append("-" * 72)
        lines.append("  1. 关键指标摘要")
        lines.append("-" * 72)
        lines.append("")

        # 最佳 epoch (根据 reward_max 或 avg_docking)
        best_docking_epoch = summary.loc[summary["avg_docking"].idxmin()]
        best_reward_epoch = summary.loc[summary["reward_mean"].idxmax()]

        lines.append(f"  最佳平均 Docking:     epoch {int(best_docking_epoch['epoch']):>4d}"
                      f"  docking={best_docking_epoch['avg_docking']:.2f}"
                      f"  reward={best_docking_epoch['reward_mean']:.3f}")
        lines.append(f"  最佳平均 Reward:      epoch {int(best_reward_epoch['epoch']):>4d}"
                      f"  reward={best_reward_epoch['reward_mean']:.3f}"
                      f"  docking={best_reward_epoch['avg_docking']:.2f}")
        lines.append("")

        lines.append(f"  {'指标':<20} {'起始':>10} {'均值':>10} {'最佳':>10} {'最终':>10}")
        lines.append("  " + "-" * 60)
        for metric, label, fmt, higher_better in [
            ("reward_mean", "Reward Mean", ".3f", True),
            ("avg_qed", "QED", ".3f", True),
            ("avg_sa", "SA", ".2f", False),
            ("avg_docking", "Docking", ".2f", False),
            ("rdkit_valid_ratio", "Valid Ratio", ".1%", True),
        ]:
            if metric not in summary.columns:
                continue
            first = summary[metric].iloc[0]
            mean_val = summary[metric].mean()
            if higher_better:
                best_val = summary[metric].max()
            else:
                best_val = summary[metric].min()
            final_val = summary[metric].iloc[-1]
            arrow = "↑" if (higher_better and final_val >= first) or (not higher_better and final_val <= first) else "↓"
            lines.append(f"  {label:<20} {first:>10{fmt}} {mean_val:>10{fmt}} "
                          f"{best_val:>10{fmt}} {final_val:>10{fmt}} {arrow}")

        lines.append("")

        # ── 有效性和失败 ──
        lines.append("-" * 72)
        lines.append("  2. 有效性与失败")
        lines.append("-" * 72)
        lines.append("")

        avg_valid = summary["rdkit_valid_ratio"].mean() * 100
        min_valid = summary["rdkit_valid_ratio"].min() * 100
        max_valid = summary["rdkit_valid_ratio"].max() * 100
        lines.append(f"  有效性: {avg_valid:.1f}% (min={min_valid:.1f}%, max={max_valid:.1f}%)")

        # 失败类型总计
        all_failures: Counter = Counter()
        for fc in summary["failure_counts"]:
            for k, v in fc.items():
                if k != "success":
                    all_failures[k] += v
        total_fails = sum(all_failures.values())
        if total_fails > 0:
            lines.append(f"  总失败分子: {total_fails} ({summary['generated_total'].sum() - summary['rdkit_valid'].sum()})")
            lines.append("  失败类型分布:")
            for ft, cnt in sorted(all_failures.items(), key=lambda x: -x[1]):
                lines.append(f"    {ft:<30} {cnt:>6} ({cnt/total_fails*100:.1f}%)")

        lines.append("")

        # ── KL 与 PPO Loss ──
        lines.append("-" * 72)
        lines.append("  3. 训练稳定性")
        lines.append("-" * 72)
        lines.append("")
        if "avg_kl" in summary.columns:
            lines.append(f"  KL 均值: {summary['avg_kl'].mean():.6f}  (min={summary['avg_kl'].min():.2e}, max={summary['avg_kl'].max():.2e})")
        if "avg_ppo_loss" in summary.columns:
            lines.append(f"  PPO Loss 均值: {summary['avg_ppo_loss'].mean():.4f}  (min={summary['avg_ppo_loss'].min():.4f}, max={summary['avg_ppo_loss'].max():.4f})")
        lines.append("")

        # ── 分子排名 ──
        lines.append("-" * 72)
        lines.append("  4. 全训练最佳分子 (Top 10)")
        lines.append("-" * 72)
        lines.append("")
        if not all_best.empty:
            top10 = all_best.head(10)
            for i, (_, row) in enumerate(top10.iterrows(), 1):
                ep = int(row["epoch"])
                dock = row["docking_raw"]
                qed = row["qed_raw"]
                sa = row["sa_raw"]
                rew = row["reward_total"]
                smi = str(row["smiles"])[:60]
                lines.append(f"  #{i:<2} ep={ep:<4} dock={dock:<7.2f} qed={qed:.3f} sa={sa:.2f} rew={rew:+.3f}  {smi}")

        lines.append("")

        # ── 最佳 epoch 的分子 ──
        if not best_per_epoch.empty:
            lines.append("-" * 72)
            lines.append("  5. 每 epoch 最佳分子均值趋势")
            lines.append("-" * 72)
            lines.append("")
            lines.append(f"  最佳 docking 均值: {best_per_epoch['docking_raw'].mean():.2f}")
            lines.append(f"  最佳 docking 范围: [{best_per_epoch['docking_raw'].min():.2f}, {best_per_epoch['docking_raw'].max():.2f}]")
            lines.append(f"  最佳 reward 均值: {best_per_epoch['reward_total'].mean():.3f}")
            lines.append(f"  最佳 reward 范围: [{best_per_epoch['reward_total'].min():.3f}, {best_per_epoch['reward_total'].max():.3f}]")

        lines.append("")

    # ── 图表列表 ──
    lines.append("-" * 72)
    lines.append("  6. 生成图表")
    lines.append("-" * 72)
    lines.append("")
    valid_plots = [p for p in plot_paths if p]
    for p in valid_plots:
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
        description="分析 DDPO 训练输出目录",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""用法示例:
  python analysis/train_analysis/analyze_run.py OUT/qed-sa-docking-8gjl-8gpu_20260508_212050

  # 只生成曲线和报告，跳过耗时的逐 epoch 分子分析
  python analysis/train_analysis/analyze_run.py OUT/xxx --quick

  # 指定相关性分析的具体 epoch
  python analysis/train_analysis/analyze_run.py OUT/xxx --corr-epochs 50 200 350
""",
    )
    parser.add_argument("run_dir", type=str, help="训练输出目录路径 (如 OUT/run_name_YYYYMMDD_HHMMSS)")
    parser.add_argument("--quick", action="store_true", help="快速模式: 跳过逐 epoch 分子加载和分布分析")
    parser.add_argument("--corr-epochs", type=int, nargs="+", default=None, help="相关性散点图的 epoch (默认: 最后3个)")
    parser.add_argument("--top-n", type=int, default=10, help="每个 epoch 保留的最佳分子数 (默认: 10)")
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        print(f"[ERROR] 目录不存在: {run_dir}")
        sys.exit(1)

    save_dir = _safe_dir(os.path.join(run_dir, "analysis"))
    print(f"\n📁 输出目录: {run_dir}")
    print(f"📁 分析结果: {save_dir}\n")

    # ── 加载配置 ──
    config_path = os.path.join(run_dir, "config.yaml")
    run_config: dict | None = None
    if os.path.isfile(config_path):
        try:
            import yaml
            with open(config_path) as f:
                run_config = yaml.safe_load(f)
        except Exception:
            pass

    # ── 加载训练摘要 ──
    summary = _load_training_summary(run_dir)
    if summary is None:
        print("[ERROR] 缺少 training_summary.csv，无法分析。")
        sys.exit(1)

    epochs = _get_available_epochs(run_dir)
    print(f"✅ 总 epochs: {len(summary)}  (分子数据可用 epoch 数: {len(epochs)})")

    # ── 图表: 训练曲线 ──
    print("  绘图: 训练曲线...")
    plot_paths: list[str] = []
    plot_paths.append(plot_training_curves(summary, save_dir))

    # ── 图表: 失败分析 ──
    print("  绘图: 失败模式分析...")
    plot_paths.append(plot_failure_analysis(summary, save_dir))

    # ── 图表: raw vs norm ──
    print("  绘图: Raw vs Norm 对比...")
    plot_paths.append(plot_raw_vs_norm_rewards(summary, save_dir))

    # ── 分子级分析 ──
    best_per_epoch: pd.DataFrame = pd.DataFrame()
    all_best: pd.DataFrame = pd.DataFrame()

    if not args.quick and epochs:
        print("  分析: 逐 epoch 最佳分子...")
        best_per_epoch, all_best = analyze_molecule_properties(run_dir, epochs, top_n=args.top_n)
        if not best_per_epoch.empty:
            plot_paths.append(plot_best_molecules_trend(best_per_epoch, save_dir))

        # 分布对比: 选取代表性 epoch（均匀采样 4 个，或指定）
        n_total = len(epochs)
        if n_total <= 4:
            sample_epochs = epochs
        else:
            sample_positions = [0, n_total // 3, n_total * 2 // 3, -1]
            sample_epochs = sorted(set(epochs[max(0, p)] for p in sample_positions))
        print(f"  绘图: 分子性质分布 (epochs={sample_epochs})...")
        plot_paths.append(plot_property_distributions(run_dir, sample_epochs, save_dir))

        # 相关性散点图
        corr_epochs = args.corr_epochs
        if corr_epochs is None:
            corr_epochs = epochs[-3:] if len(epochs) >= 3 else epochs
        for ep in corr_epochs:
            plot_paths.append(plot_reward_correlation(run_dir, ep, save_dir))
            print(f"  绘图: 奖励相关性矩阵 (epoch {ep})...")

    # ── 报告 ──
    print("  生成: 文本分析报告...")
    report_path = generate_report(
        run_dir, summary, best_per_epoch, all_best, run_config, plot_paths, save_dir,
    )

    print(f"\n✅ 分析完成!\n")
    print(f"  报告: {report_path}")
    for p in plot_paths:
        if p:
            print(f"  图表: {p}")

    # 打印报告摘要到终端
    print("\n" + "=" * 72)
    try:
        with open(report_path) as f:
            print(f.read())
    except Exception:
        pass


if __name__ == "__main__":
    main()
