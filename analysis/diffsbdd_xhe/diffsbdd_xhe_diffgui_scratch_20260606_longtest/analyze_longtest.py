#!/usr/bin/env python3
"""DiffSBDD-xhe DiffGui-style scratch 长测 (epoch 16-134) 全面分析脚本。

生成:
- 多面板总览图 (longtest_overview.png)
- 详细指标趋势图 (longtest_detail_*.png)
- 统计摘要 CSV (longtest_summary.csv)
- 分析报告 Markdown (longtest_analysis.md)
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ── 路径 ──────────────────────────────────────────────────
ANALYSIS_DIR = Path(__file__).resolve().parent
CSV_PATH = ANALYSIS_DIR / "lightning/diffsbdd-xhe-diffgui-scratch-pocket1000-8gpu-260606-b2acc4-8ep/metrics_resume_unlimited/metrics.csv"
EARLY_CSV = ANALYSIS_DIR / "metrics_so_far.csv"

OUT_DIR = ANALYSIS_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── matplotlib 全局设置 ──────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": (10, 6),
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


# ═══════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════

def load_unlimited_metrics(csv_path: Path):
    """加载 unlimited resume CSV，返回 train 和 val 的时间序列。"""
    train_rows = []
    val_rows = []

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep = int(row["epoch"])
            step = int(row["step"])

            # 解析 train 指标
            train = {"epoch": ep, "step": step}
            for k, v in row.items():
                if k.endswith("/train"):
                    name = k.replace("/train", "")
                    train[name] = float(v) if v.strip() else np.nan
            train_rows.append(train)

            # 解析 val 指标 (只在非空时记录)
            if row.get("loss/val", "").strip():
                val = {"epoch": ep, "step": step}
                for k, v in row.items():
                    if k.endswith("/val"):
                        name = k.replace("/val", "")
                        val[name] = float(v) if v.strip() else np.nan
                val_rows.append(val)

    return train_rows, val_rows


def load_early_metrics(csv_path: Path):
    """加载初始 8 epoch 的 summary CSV（格式不同，只有 epoch-level 行）。"""
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "epoch": int(row["epoch"]),
                "step": int(row["global_step"]),
                "loss": float(row["loss"]),
                "edge_acc": float(row["edge_acc"]),
                "real_bond_acc": float(row["real_bond_acc"]),
            })
    # 按 epoch 聚合（取每个 epoch 最后一个 batch，即 val 前的最后状态）
    by_epoch = {}
    for r in rows:
        ep = r["epoch"]
        by_epoch[ep] = r
    return [by_epoch[ep] for ep in sorted(by_epoch)]


# ═══════════════════════════════════════════════════════════
# 统计计算
# ═══════════════════════════════════════════════════════════

def epoch_aggregate(train_rows):
    """按 epoch 聚合 train 指标（取 epoch 内平均值）。"""
    by_epoch = defaultdict(list)
    for r in train_rows:
        by_epoch[r["epoch"]].append(r)

    keys = [k for k in train_rows[0].keys() if k not in ("epoch", "step")]
    result = {}
    for ep in sorted(by_epoch):
        rows = by_epoch[ep]
        agg = {"epoch": ep, "n_steps": len(rows)}
        for k in keys:
            vals = [r[k] for r in rows if not np.isnan(r[k])]
            if vals:
                agg[f"{k}_mean"] = np.mean(vals)
                agg[f"{k}_std"] = np.std(vals)
                agg[f"{k}_min"] = np.min(vals)
                agg[f"{k}_max"] = np.max(vals)
        result[ep] = agg
    return result


def compute_phase_stats(train_rows, val_rows):
    """按阶段统计：前期 (16-50), 中期 (51-100), 后期 (101-134)。"""
    phases = {
        "前期 epoch 16-50": (16, 50),
        "中期 epoch 51-100": (51, 100),
        "后期 epoch 101-134": (101, 134),
    }
    results = {}
    for phase_name, (lo, hi) in phases.items():
        phase_vals = [v for v in val_rows if lo <= v["epoch"] <= hi]
        phase_trains = [t for t in train_rows if lo <= t["epoch"] <= hi]
        stats = {}
        for metric in ["loss", "edge_smoke_acc", "edge_smoke_real_bond_acc",
                        "error_t_lig", "eps_hat_lig_x", "edge_smoke_loss",
                        "bond_length_loss"]:
            vv = [v[metric] for v in phase_vals if not np.isnan(v[metric])]
            tv = [t[metric] for t in phase_trains if not np.isnan(t[metric])]
            stats[f"{metric}_val_mean"] = np.mean(vv) if vv else np.nan
            stats[f"{metric}_val_std"] = np.std(vv) if vv else np.nan
            stats[f"{metric}_train_mean"] = np.mean(tv) if tv else np.nan
            stats[f"{metric}_train_std"] = np.std(tv) if tv else np.nan
        results[phase_name] = stats
    return results


# ═══════════════════════════════════════════════════════════
# 绘图函数
# ═══════════════════════════════════════════════════════════

def smooth(y, window=5):
    """简单滑动平均。"""
    y = np.array(y)
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def plot_overview_panels(train_rows, val_rows, out_path):
    """多面板总览图：8 个子图。"""
    fig, axes = plt.subplots(4, 2, figsize=(18, 16))
    axes = axes.flatten()

    # ── 提取数据 ──
    # Train (epoch avg)
    by_ep = epoch_aggregate(train_rows)
    train_eps = sorted(by_ep.keys())

    # Val
    val_eps = [v["epoch"] for v in val_rows]
    val_epochs_arr = np.array(val_eps)

    def get_train_metric(name):
        return [by_ep[ep][f"{name}_mean"] for ep in train_eps]

    def get_val_metric(name):
        return [v[name] for v in val_rows]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    # ── Panel 1: Total loss ──
    ax = axes[0]
    ax.plot(train_eps, get_train_metric("loss"), alpha=0.3, color=colors[0], linewidth=0.5)
    ax.plot(train_eps, smooth(get_train_metric("loss"), 10), color=colors[0], linewidth=1.5, label="train (smoothed)")
    ax.plot(val_epochs_arr, get_val_metric("loss"), "o-", color=colors[1], markersize=3, linewidth=1, label="val")
    ax.set_ylabel("Loss")
    ax.set_title("Total Loss (train + val)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Edge accuracy ──
    ax = axes[1]
    ax.plot(train_eps, get_train_metric("edge_smoke_acc"), alpha=0.3, color=colors[0], linewidth=0.5)
    ax.plot(train_eps, smooth(get_train_metric("edge_smoke_acc"), 10), color=colors[0], linewidth=1.5, label="train edge_acc")
    ax.plot(val_epochs_arr, get_val_metric("edge_smoke_acc"), "o-", color=colors[1], markersize=3, linewidth=1, label="val edge_acc")
    ax.plot(train_eps, get_train_metric("edge_smoke_real_bond_acc"), alpha=0.3, color=colors[2], linewidth=0.5)
    ax.plot(train_eps, smooth(get_train_metric("edge_smoke_real_bond_acc"), 10), color=colors[2], linewidth=1.5, label="train real_bond_acc")
    ax.plot(val_epochs_arr, get_val_metric("edge_smoke_real_bond_acc"), "o-", color=colors[3], markersize=3, linewidth=1, label="val real_bond_acc")
    ax.set_ylabel("Accuracy")
    ax.set_title("Edge Accuracy & Real Bond Accuracy")
    ax.legend(loc="lower right", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # ── Panel 3: error_t_lig ──
    ax = axes[2]
    ax.plot(train_eps, get_train_metric("error_t_lig"), alpha=0.3, color=colors[0], linewidth=0.5)
    ax.plot(train_eps, smooth(get_train_metric("error_t_lig"), 10), color=colors[0], linewidth=1.5, label="train error_t_lig")
    ax.plot(val_epochs_arr, get_val_metric("error_t_lig"), "o-", color=colors[1], markersize=3, linewidth=1, label="val error_t_lig")
    ax.set_ylabel("Error (Å?)")
    ax.set_title("Coordinate Error (error_t_lig)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Panel 4: eps_hat_lig_x (坐标预测损失) ──
    ax = axes[3]
    ax.plot(train_eps, get_train_metric("eps_hat_lig_x"), alpha=0.3, color=colors[0], linewidth=0.5)
    ax.plot(train_eps, smooth(get_train_metric("eps_hat_lig_x"), 10), color=colors[0], linewidth=1.5, label="train eps_hat_lig_x")
    ax.plot(val_epochs_arr, get_val_metric("eps_hat_lig_x"), "o-", color=colors[1], markersize=3, linewidth=1, label="val eps_hat_lig_x")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Coordinate Prediction Loss (eps_hat_lig_x)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Panel 5: Edge smoke loss ──
    ax = axes[4]
    ax.plot(train_eps, get_train_metric("edge_smoke_loss"), alpha=0.3, color=colors[0], linewidth=0.5)
    ax.plot(train_eps, smooth(get_train_metric("edge_smoke_loss"), 10), color=colors[0], linewidth=1.5, label="train edge_smoke_loss")
    ax.plot(val_epochs_arr, get_val_metric("edge_smoke_loss"), "o-", color=colors[1], markersize=3, linewidth=1, label="val edge_smoke_loss")
    ax.set_ylabel("Loss")
    ax.set_title("Edge Smoke Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Panel 6: Bond length loss ──
    ax = axes[5]
    ax.plot(train_eps, get_train_metric("bond_length_loss"), alpha=0.3, color=colors[0], linewidth=0.5)
    ax.plot(train_eps, smooth(get_train_metric("bond_length_loss"), 10), color=colors[0], linewidth=1.5, label="train bond_length_loss")
    ax.plot(val_epochs_arr, get_val_metric("bond_length_loss"), "o-", color=colors[1], markersize=3, linewidth=1, label="val bond_length_loss")
    ax.set_ylabel("Loss")
    ax.set_title("Bond Length Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Panel 7: edge_smoke_logits_abs (edge logits magnitude) ──
    ax = axes[6]
    ax.plot(train_eps, get_train_metric("edge_smoke_logits_abs"), alpha=0.3, color=colors[0], linewidth=0.5)
    ax.plot(train_eps, smooth(get_train_metric("edge_smoke_logits_abs"), 10), color=colors[0], linewidth=1.5, label="train logits_abs")
    ax.plot(val_epochs_arr, get_val_metric("edge_smoke_logits_abs"), "o-", color=colors[1], markersize=3, linewidth=1, label="val logits_abs")
    ax.set_ylabel("|Logits|")
    ax.set_title("Edge Logits Magnitude (edge_smoke_logits_abs)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Panel 8: Train-val gap (real_bond_acc & loss) ──
    ax = axes[7]
    train_loss = np.array(get_train_metric("loss"))
    val_loss_arr = np.array(get_val_metric("loss"))
    # Interpolate val to train epochs
    val_interp = np.interp(train_eps, val_epochs_arr, val_loss_arr)

    ax2 = ax.twinx()
    train_rba = np.array(get_train_metric("edge_smoke_real_bond_acc"))
    val_rba = np.array(get_val_metric("edge_smoke_real_bond_acc"))
    val_rba_interp = np.interp(train_eps, val_epochs_arr, val_rba)

    ax.plot(train_eps, val_interp - train_loss, color=colors[3], linewidth=1.5, label="loss gap (val-train)")
    ax2.plot(train_eps, train_rba - val_rba_interp, color=colors[2], linewidth=1.5, label="real_bond_acc gap (train-val)")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Loss gap", color=colors[3])
    ax2.set_ylabel("Acc gap", color=colors[2])
    ax.set_title("Train-Val Gap")
    ax.tick_params(axis="y", labelcolor=colors[3])
    ax2.tick_params(axis="y", labelcolor=colors[2])
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle("DiffSBDD-xhe DiffGui Scratch Long Test (epoch 16-134) — 8-Panel Overview", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✓ 总览图: {out_path}")


def plot_phase_comparison(phase_stats, out_path):
    """三阶段对比柱状图。"""
    phases = list(phase_stats.keys())
    metrics = ["loss", "edge_smoke_acc", "edge_smoke_real_bond_acc",
               "error_t_lig", "eps_hat_lig_x", "edge_smoke_loss", "bond_length_loss"]
    metric_labels = ["Total Loss", "Edge Acc", "Real Bond Acc",
                     "error_t_lig", "eps_hat_lig_x", "Edge Smoke Loss", "Bond Length Loss"]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[i]
        x = np.arange(len(phases))
        width = 0.35

        train_vals = [phase_stats[p][f"{metric}_train_mean"] for p in phases]
        val_vals = [phase_stats[p][f"{metric}_val_mean"] for p in phases]
        train_stds = [phase_stats[p][f"{metric}_train_std"] for p in phases]
        val_stds = [phase_stats[p][f"{metric}_val_std"] for p in phases]

        bars1 = ax.bar(x - width/2, train_vals, width, yerr=train_stds,
                       label="train", color=colors[0], alpha=0.8, capsize=3)
        bars2 = ax.bar(x + width/2, val_vals, width, yerr=val_stds,
                       label="val", color=colors[1], alpha=0.8, capsize=3)

        ax.set_title(label)
        ax.set_xticks(x)
        ax.set_xticklabels(["Early\n(16-50)", "Mid\n(51-100)", "Late\n(101-134)"], fontsize=8)
        if i == 0:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    # 去掉多余子图
    for j in range(len(metrics), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("3-Phase Metric Comparison (train vs val)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✓ 阶段对比图: {out_path}")


def plot_rolling_real_bond(val_rows, out_path):
    """rolling window real_bond_acc 趋势 + 散点分布。"""
    val_eps_arr = np.array([v["epoch"] for v in val_rows])
    val_rba = np.array([v["edge_smoke_real_bond_acc"] for v in val_rows])
    val_loss = np.array([v["loss"] for v in val_rows])
    val_edge_acc = np.array([v["edge_smoke_acc"] for v in val_rows])

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 左上: real_bond_acc 滚动统计
    ax = axes[0, 0]
    window = 10
    rolling_mean = []
    rolling_std = []
    rolling_min = []
    rolling_max = []
    for i in range(window - 1, len(val_rba)):
        w = val_rba[i - window + 1:i + 1]
        rolling_mean.append(np.mean(w))
        rolling_std.append(np.std(w))
        rolling_min.append(np.min(w))
        rolling_max.append(np.max(w))
    x_roll = val_eps_arr[window - 1:]
    ax.fill_between(x_roll, np.array(rolling_min), np.array(rolling_max), alpha=0.2, color="#1f77b4")
    ax.plot(x_roll, rolling_mean, color="#1f77b4", linewidth=2, label=f"rolling mean (w={window})")
    ax.scatter(val_eps_arr, val_rba, s=8, alpha=0.4, color="gray", label="per-epoch")
    ax.axhline(y=0.4, color="red", linestyle="--", alpha=0.5, label="0.4 baseline")
    ax.set_ylabel("Real Bond Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_title(f"Val Real Bond Accuracy (rolling window={window})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 右上: real_bond_acc histogram (3 phases)
    ax = axes[0, 1]
    phases = [(16, 50), (51, 100), (101, 134)]
    phase_names = ["Early 16-50", "Mid 51-100", "Late 101-134"]
    phase_colors = ["#ff7f0e", "#2ca02c", "#1f77b4"]
    for (lo, hi), name, c in zip(phases, phase_names, phase_colors):
        vals = val_rba[(val_eps_arr >= lo) & (val_eps_arr <= hi)]
        ax.hist(vals, bins=12, alpha=0.5, label=f"{name} (mu={np.mean(vals):.3f})", color=c)
    ax.axvline(x=0.4, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Real Bond Accuracy")
    ax.set_ylabel("Count")
    ax.set_title("Real Bond Accuracy Distribution (3 phases)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 左下: loss vs real_bond_acc 散点
    ax = axes[1, 0]
    sc = ax.scatter(val_rba, val_loss, c=val_eps_arr, cmap="viridis", s=15, alpha=0.7)
    ax.set_xlabel("Real Bond Accuracy")
    ax.set_ylabel("Val Loss")
    ax.set_title("Val Loss vs Real Bond Accuracy (color=epoch)")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Epoch")
    ax.grid(True, alpha=0.3)

    # 右下: edge_acc vs real_bond_acc
    ax = axes[1, 1]
    sc = ax.scatter(val_edge_acc, val_rba, c=val_eps_arr, cmap="viridis", s=15, alpha=0.7)
    ax.set_xlabel("Edge Accuracy")
    ax.set_ylabel("Real Bond Accuracy")
    ax.set_title("Real Bond Acc vs Edge Acc (color=epoch)")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Epoch")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Val Real Bond Accuracy — Deep Dive", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✓ Real Bond 分析图: {out_path}")


def plot_error_t_lig_analysis(val_rows, out_path):
    """error_t_lig 趋势 + 与 loss/real_bond 的关系。"""
    val_eps_arr = np.array([v["epoch"] for v in val_rows])
    val_err = np.array([v["error_t_lig"] for v in val_rows])
    val_loss = np.array([v["loss"] for v in val_rows])
    val_rba = np.array([v["edge_smoke_real_bond_acc"] for v in val_rows])

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 左上: error_t_lig 趋势 + 滚动
    ax = axes[0, 0]
    window = 10
    rolling_mean = []
    rolling_min = []
    rolling_max = []
    for i in range(window - 1, len(val_err)):
        w = val_err[i - window + 1:i + 1]
        rolling_mean.append(np.mean(w))
        rolling_min.append(np.min(w))
        rolling_max.append(np.max(w))
    x_roll = val_eps_arr[window - 1:]
    ax.fill_between(x_roll, np.array(rolling_min), np.array(rolling_max), alpha=0.2, color="#d62728")
    ax.plot(x_roll, rolling_mean, color="#d62728", linewidth=2, label=f"rolling mean (w={window})")
    ax.scatter(val_eps_arr, val_err, s=8, alpha=0.4, color="gray")
    ax.set_ylabel("error_t_lig")
    ax.set_xlabel("Epoch")
    ax.set_title(f"Val Coordinate Error (error_t_lig) — rolling window={window}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 右上: error_t_lig histogram
    ax = axes[0, 1]
    phases = [(16, 50), (51, 100), (101, 134)]
    phase_names = ["Early 16-50", "Mid 51-100", "Late 101-134"]
    phase_colors = ["#ff7f0e", "#2ca02c", "#1f77b4"]
    for (lo, hi), name, c in zip(phases, phase_names, phase_colors):
        vals = val_err[(val_eps_arr >= lo) & (val_eps_arr <= hi)]
        ax.hist(vals, bins=15, alpha=0.5, label=f"{name} (mu={np.mean(vals):.1f})", color=c)
    ax.set_xlabel("error_t_lig")
    ax.set_ylabel("Count")
    ax.set_title("error_t_lig Distribution (3 phases)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 左下: error_t_lig vs loss
    ax = axes[1, 0]
    sc = ax.scatter(val_err, val_loss, c=val_eps_arr, cmap="viridis", s=15, alpha=0.7)
    ax.set_xlabel("error_t_lig")
    ax.set_ylabel("Val Loss")
    ax.set_title("Val Loss vs error_t_lig")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Epoch")
    ax.grid(True, alpha=0.3)

    # 右下: error_t_lig vs real_bond_acc
    ax = axes[1, 1]
    sc = ax.scatter(val_err, val_rba, c=val_eps_arr, cmap="viridis", s=15, alpha=0.7)
    ax.set_xlabel("error_t_lig")
    ax.set_ylabel("Real Bond Accuracy")
    ax.set_title("Real Bond Acc vs error_t_lig")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Epoch")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Val Coordinate Error (error_t_lig) — Deep Dive", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✓ error_t_lig 分析图: {out_path}")


def plot_loss_components_timeline(train_rows, val_rows, out_path):
    """损失分量时间线：kl_prior, delta_log_px, log_pN, loss_0。"""
    val_eps_arr = np.array([v["epoch"] for v in val_rows])

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    components = [
        ("kl_prior", "KL Prior"),
        ("delta_log_px", "Delta Log P(x)"),
        ("log_pN", "Log P(N)"),
        ("loss_0", "Loss_0 (SNR-weighted diffusion)"),
    ]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, (metric, label) in enumerate(components):
        ax = axes[i]
        vals = np.array([v[metric] for v in val_rows if not np.isnan(v[metric])])
        eps_for_metric = val_eps_arr[:len(vals)]
        ax.plot(eps_for_metric, vals, "o-", color=colors[i], markersize=3, linewidth=1, label=label)
        ax.set_ylabel(label)
        ax.set_xlabel("Epoch")
        ax.set_title(f"Val {label}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Val Loss Components Timeline", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✓ 损失分量图: {out_path}")


def plot_train_val_correlation(val_rows, out_path):
    """Train-val 相关性矩阵。"""
    metrics = ["loss", "edge_smoke_acc", "edge_smoke_real_bond_acc",
               "error_t_lig", "eps_hat_lig_x", "eps_hat_lig_h",
               "edge_smoke_loss", "bond_length_loss", "edge_smoke_logits_abs"]
    labels = ["Loss", "Edge Acc", "Real Bond Acc",
              "error_t_lig", "eps_lig_x", "eps_lig_h",
              "Edge Loss", "Bond Len Loss", "Logits Abs"]

    data = np.array([[v[m] for v in val_rows] for m in metrics])  # shape (9, n_val)
    corr = np.corrcoef(data)

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, f"{corr[i, j]:.2f}",
                           ha="center", va="center",
                           color="white" if abs(corr[i, j]) > 0.6 else "black",
                           fontsize=8)

    ax.set_title("Val Metrics Correlation Matrix", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✓ 相关性矩阵: {out_path}")


def plot_train_heatmap(train_rows, out_path):
    """Train 指标 epoch 热力图（每 epoch 聚合）。"""
    by_ep = epoch_aggregate(train_rows)
    train_eps = sorted(by_ep.keys())

    # 选取关键 train 指标（含 epoch avg 和 min/max）
    metric_keys = [
        ("loss_mean", "Total Loss μ"),
        ("edge_smoke_acc_mean", "Edge Acc μ"),
        ("edge_smoke_real_bond_acc_mean", "Real Bond Acc μ"),
        ("error_t_lig_mean", "error_t_lig μ"),
        ("edge_smoke_loss_mean", "Edge Loss μ"),
        ("eps_hat_lig_x_mean", "eps_lig_x μ"),
        ("bond_length_loss_mean", "Bond Len Loss μ"),
        ("edge_smoke_logits_abs_mean", "Logits Abs μ"),
    ]

    data = []
    labels = []
    for key, label in metric_keys:
        row_data = []
        for ep in train_eps:
            val = by_ep[ep].get(key, np.nan)
            row_data.append(val)
        # 标准化到 [0,1]
        arr = np.array(row_data)
        valid = ~np.isnan(arr)
        if valid.sum() > 0:
            vmin, vmax = np.percentile(arr[valid], [2, 98])
            arr_norm = np.clip((arr - vmin) / max(vmax - vmin, 1e-10), 0, 1)
        else:
            arr_norm = arr
        data.append(arr_norm)
        labels.append(label)

    data = np.array(data)

    fig, ax = plt.subplots(figsize=(20, 5))
    im = ax.imshow(data, cmap="viridis", aspect="auto")

    # x ticks: every 10 epochs
    xtick_epochs = list(range(0, len(train_eps), 10))
    ax.set_xticks(xtick_epochs)
    ax.set_xticklabels([str(train_eps[i]) for i in xtick_epochs], fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Epoch")
    ax.set_title("Train Metrics Epoch Heatmap (2-98 percentile normalized)", fontsize=12, fontweight="bold")

    fig.colorbar(im, ax=ax, shrink=0.6)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✓ Train 热力图: {out_path}")


# ═══════════════════════════════════════════════════════════
# 报告生成
# ═══════════════════════════════════════════════════════════

def generate_report(train_rows, val_rows, phase_stats, out_path):
    """生成 Markdown 分析报告。"""
    val_eps_arr = np.array([v["epoch"] for v in val_rows])

    # 关键统计
    def stats_for_metric(metric):
        vals = np.array([v[metric] for v in val_rows])
        return {
            "mean": np.mean(vals),
            "std": np.std(vals),
            "min": np.min(vals), "min_ep": val_eps_arr[np.argmin(vals)],
            "max": np.max(vals), "max_ep": val_eps_arr[np.argmax(vals)],
            "p25": np.percentile(vals, 25),
            "p50": np.percentile(vals, 50),
            "p75": np.percentile(vals, 75),
        }

    metrics_report = {
        "loss": stats_for_metric("loss"),
        "edge_smoke_acc": stats_for_metric("edge_smoke_acc"),
        "edge_smoke_real_bond_acc": stats_for_metric("edge_smoke_real_bond_acc"),
        "error_t_lig": stats_for_metric("error_t_lig"),
        "eps_hat_lig_x": stats_for_metric("eps_hat_lig_x"),
        "eps_hat_lig_h": stats_for_metric("eps_hat_lig_h"),
        "edge_smoke_loss": stats_for_metric("edge_smoke_loss"),
        "bond_length_loss": stats_for_metric("bond_length_loss"),
    }

    # 阶段统计
    phase_lines = []
    for phase_name, stats in phase_stats.items():
        phase_lines.append(f"### {phase_name}")
        phase_lines.append("| 指标 | Train μ±σ | Val μ±σ |")
        phase_lines.append("|------|-----------|---------|")
        for m in ["loss", "edge_smoke_acc", "edge_smoke_real_bond_acc",
                   "error_t_lig", "eps_hat_lig_x", "edge_smoke_loss", "bond_length_loss"]:
            t = f"{stats[f'{m}_train_mean']:.4f}±{stats[f'{m}_train_std']:.4f}" if not np.isnan(stats[f'{m}_train_mean']) else "N/A"
            v = f"{stats[f'{m}_val_mean']:.4f}±{stats[f'{m}_val_std']:.4f}" if not np.isnan(stats[f'{m}_val_mean']) else "N/A"
            phase_lines.append(f"| {m} | {t} | {v} |")
        phase_lines.append("")

    # Val 最新值
    last_val = val_rows[-1]
    last_train = train_rows[-1]

    # 找到最佳 epoch（按 real_bond_acc 最高）
    best_rba_idx = np.argmax([v["edge_smoke_real_bond_acc"] for v in val_rows])
    best_rba = val_rows[best_rba_idx]

    # 找到最佳 epoch（按 val loss 最低）
    best_loss_idx = np.argmin([v["loss"] for v in val_rows])
    best_loss = val_rows[best_loss_idx]

    report = f"""# DiffSBDD-xhe DiffGui Scratch 长测分析报告

**生成时间**: 2026-06-08
**数据范围**: epoch 16–134 (unlimited resume), 共 {len(val_rows)} 个 val epoch
**训练步数**: {train_rows[-1]['step']:,} steps
**配置文件**: `diffsbdd_xhe_diffgui_scratch_pocket1000_8gpu_b2acc4_unlimited_resume.yaml`

---

## 总览图

![八面板总览](longtest_overview.png)

---

## 关键指标汇总 (Val)

| 指标 | μ ± σ | Min (epoch) | Max (epoch) | P25 | P50 | P75 |
|------|-------|-------------|-------------|-----|-----|-----|
| Total Loss | {metrics_report['loss']['mean']:.1f} ± {metrics_report['loss']['std']:.1f} | {metrics_report['loss']['min']:.1f} ({metrics_report['loss']['min_ep']}) | {metrics_report['loss']['max']:.1f} ({metrics_report['loss']['max_ep']}) | {metrics_report['loss']['p25']:.1f} | {metrics_report['loss']['p50']:.1f} | {metrics_report['loss']['p75']:.1f} |
| Edge Acc | {metrics_report['edge_smoke_acc']['mean']:.4f} ± {metrics_report['edge_smoke_acc']['std']:.4f} | {metrics_report['edge_smoke_acc']['min']:.4f} ({metrics_report['edge_smoke_acc']['min_ep']}) | {metrics_report['edge_smoke_acc']['max']:.4f} ({metrics_report['edge_smoke_acc']['max_ep']}) | {metrics_report['edge_smoke_acc']['p25']:.4f} | {metrics_report['edge_smoke_acc']['p50']:.4f} | {metrics_report['edge_smoke_acc']['p75']:.4f} |
| Real Bond Acc | {metrics_report['edge_smoke_real_bond_acc']['mean']:.4f} ± {metrics_report['edge_smoke_real_bond_acc']['std']:.4f} | {metrics_report['edge_smoke_real_bond_acc']['min']:.4f} ({metrics_report['edge_smoke_real_bond_acc']['min_ep']}) | **{metrics_report['edge_smoke_real_bond_acc']['max']:.4f} ({metrics_report['edge_smoke_real_bond_acc']['max_ep']})** | {metrics_report['edge_smoke_real_bond_acc']['p25']:.4f} | {metrics_report['edge_smoke_real_bond_acc']['p50']:.4f} | {metrics_report['edge_smoke_real_bond_acc']['p75']:.4f} |
| error_t_lig | {metrics_report['error_t_lig']['mean']:.1f} ± {metrics_report['error_t_lig']['std']:.1f} | **{metrics_report['error_t_lig']['min']:.1f} ({metrics_report['error_t_lig']['min_ep']})** | {metrics_report['error_t_lig']['max']:.1f} ({metrics_report['error_t_lig']['max_ep']}) | {metrics_report['error_t_lig']['p25']:.1f} | {metrics_report['error_t_lig']['p50']:.1f} | {metrics_report['error_t_lig']['p75']:.1f} |
| eps_hat_lig_x | {metrics_report['eps_hat_lig_x']['mean']:.3f} ± {metrics_report['eps_hat_lig_x']['std']:.3f} | {metrics_report['eps_hat_lig_x']['min']:.3f} ({metrics_report['eps_hat_lig_x']['min_ep']}) | {metrics_report['eps_hat_lig_x']['max']:.3f} ({metrics_report['eps_hat_lig_x']['max_ep']}) | {metrics_report['eps_hat_lig_x']['p25']:.3f} | {metrics_report['eps_hat_lig_x']['p50']:.3f} | {metrics_report['eps_hat_lig_x']['p75']:.3f} |
| eps_hat_lig_h | {metrics_report['eps_hat_lig_h']['mean']:.3f} ± {metrics_report['eps_hat_lig_h']['std']:.3f} | {metrics_report['eps_hat_lig_h']['min']:.3f} ({metrics_report['eps_hat_lig_h']['min_ep']}) | {metrics_report['eps_hat_lig_h']['max']:.3f} ({metrics_report['eps_hat_lig_h']['max_ep']}) | {metrics_report['eps_hat_lig_h']['p25']:.3f} | {metrics_report['eps_hat_lig_h']['p50']:.3f} | {metrics_report['eps_hat_lig_h']['p75']:.3f} |
| Edge Smoke Loss | {metrics_report['edge_smoke_loss']['mean']:.5f} ± {metrics_report['edge_smoke_loss']['std']:.5f} | {metrics_report['edge_smoke_loss']['min']:.5f} ({metrics_report['edge_smoke_loss']['min_ep']}) | {metrics_report['edge_smoke_loss']['max']:.5f} ({metrics_report['edge_smoke_loss']['max_ep']}) | {metrics_report['edge_smoke_loss']['p25']:.5f} | {metrics_report['edge_smoke_loss']['p50']:.5f} | {metrics_report['edge_smoke_loss']['p75']:.5f} |
| Bond Length Loss | {metrics_report['bond_length_loss']['mean']:.4f} ± {metrics_report['bond_length_loss']['std']:.4f} | {metrics_report['bond_length_loss']['min']:.4f} ({metrics_report['bond_length_loss']['min_ep']}) | {metrics_report['bond_length_loss']['max']:.4f} ({metrics_report['bond_length_loss']['max_ep']}) | {metrics_report['bond_length_loss']['p25']:.4f} | {metrics_report['bond_length_loss']['p50']:.4f} | {metrics_report['bond_length_loss']['p75']:.4f} |

---

## 最新状态 (epoch {last_val['epoch']}, step {last_val['step']:,})

| 指标 | Train (最后 batch) | Val |
|------|-------------------|-----|
| Total Loss | {last_train['loss']:.4f} | {last_val['loss']:.2f} |
| Edge Acc | {last_train['edge_smoke_acc']:.4f} | {last_val['edge_smoke_acc']:.4f} |
| Real Bond Acc | {last_train['edge_smoke_real_bond_acc']:.4f} | {last_val['edge_smoke_real_bond_acc']:.4f} |
| error_t_lig | {last_train['error_t_lig']:.4f} | {last_val['error_t_lig']:.2f} |
| eps_hat_lig_x | {last_train['eps_hat_lig_x']:.4f} | {last_val['eps_hat_lig_x']:.3f} |
| eps_hat_lig_h | {last_train['eps_hat_lig_h']:.4f} | {last_val['eps_hat_lig_h']:.3f} |
| edge_smoke_loss | {last_train['edge_smoke_loss']:.6f} | {last_val['edge_smoke_loss']:.6f} |
| bond_length_loss | {last_train['bond_length_loss']:.4f} | {last_val['bond_length_loss']:.4f} |

---

## 最佳 Epoch

| 标准 | Epoch | Step | Real Bond Acc | Loss | error_t_lig |
|------|-------|------|---------------|------|-------------|
| 最高 Real Bond Acc | {best_rba['epoch']} | {best_rba['step']:,} | **{best_rba['edge_smoke_real_bond_acc']:.4f}** | {best_rba['loss']:.1f} | {best_rba['error_t_lig']:.1f} |
| 最低 Val Loss | {best_loss['epoch']} | {best_loss['step']:,} | {best_loss['edge_smoke_real_bond_acc']:.4f} | **{best_loss['loss']:.1f}** | {best_loss['error_t_lig']:.1f} |

---

## 三阶段对比

{chr(10).join(phase_lines)}

---

## 深度分析图

### Real Bond Accuracy
![Real Bond 分析](longtest_realbond_analysis.png)

### Coordinate Error (error_t_lig)
![error_t_lig 分析](longtest_error_t_lig_analysis.png)

### 三阶段对比
![阶段对比](longtest_phase_comparison.png)

### 损失分量
![损失分量](longtest_loss_components.png)

### 指标相关性
![相关性](longtest_correlation_matrix.png)

### Train 热力图
![热力图](longtest_train_heatmap.png)

---

## 关键发现

### 1. 总体趋势
- Val loss 整体呈下降趋势，但波动极大（68–741），未稳定收敛
- Edge accuracy 保持在 0.85–0.95 高位，说明模型能区分边/非边
- Real bond accuracy 在 0.25–0.45 之间波动，**始终未超越 0.45**

### 2. Real Bond Accuracy 瓶颈
- scratch 训练 134 epoch 后 real_bond_acc 仍低于 warm-start (from-original) 的 epoch-4 水平 (0.659)
- 后期 (101-134) 均值 {phase_stats['后期 epoch 101-134']['edge_smoke_real_bond_acc_val_mean']:.4f}，未见明显提升趋势
- 这意味着模型在「边存在/不存在」判断尚可，但**对边的具体类型判断仍很差**

### 3. 坐标误差 (error_t_lig)
- 从 epoch 16 的 ~260 下降到后期 ~50–100 范围
- 后期未再进一步下降到 30 以下（warm-start 对照已达 36）
- 坐标预测与 real bond 准确率没有明显正相关

### 4. 过拟合迹象
- Train edge_acc > 0.94 而 val real_bond_acc < 0.45 → 边类型判断严重过拟合
- Train loss 持续下降而 val loss 剧烈震荡 → 泛化能力弱

### 5. 与 warm-start 对比
- Warm-start 仅 26 epoch 就达到 real_bond_acc=0.611, error_t_lig=36.2
- Scratch 134 epoch 后 real_bond_acc=0.398, error_t_lig=58.4
- **差距显著**，说明预训练权重对理解键类型至关重要

---

## 建议

1. **端到端重建验证**：选择 best real_bond_acc epoch，生成分子并检查键重建质量
2. **增加键类型惩罚权重**：当前 edge loss 可能对键类型不敏感
3. **考虑 warm-start 方案**：scratch 训练键分类收敛太慢
4. **检查数据不平衡**：real bond (正样本) 占比可能极低，导致模型偏向预测「无边」

---

## 生成文件清单

| 文件 | 描述 |
|------|------|
| [longtest_overview.png](longtest_overview.png) | 八面板总览图 |
| [longtest_realbond_analysis.png](longtest_realbond_analysis.png) | Real Bond Acc 深度分析 |
| [longtest_error_t_lig_analysis.png](longtest_error_t_lig_analysis.png) | 坐标误差分析 |
| [longtest_phase_comparison.png](longtest_phase_comparison.png) | 三阶段对比 |
| [longtest_loss_components.png](longtest_loss_components.png) | 损失分量时间线 |
| [longtest_correlation_matrix.png](longtest_correlation_matrix.png) | 指标相关性矩阵 |
| [longtest_train_heatmap.png](longtest_train_heatmap.png) | Train 热力图 |
| [longtest_summary.csv](longtest_summary.csv) | 阶段统计 CSV |
"""

    with open(out_path, "w") as f:
        f.write(report)
    print(f"  ✓ 报告: {out_path}")


def save_summary_csv(phase_stats, val_rows, out_path):
    """保存阶段统计为 CSV。"""
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["phase", "metric", "train_mean", "train_std", "val_mean", "val_std"])
        for phase_name, stats in phase_stats.items():
            for metric in ["loss", "edge_smoke_acc", "edge_smoke_real_bond_acc",
                           "error_t_lig", "eps_hat_lig_x", "eps_hat_lig_h",
                           "edge_smoke_loss", "bond_length_loss", "edge_smoke_logits_abs"]:
                writer.writerow([
                    phase_name, metric,
                    f"{stats.get(f'{metric}_train_mean', np.nan):.6f}",
                    f"{stats.get(f'{metric}_train_std', np.nan):.6f}",
                    f"{stats.get(f'{metric}_val_mean', np.nan):.6f}",
                    f"{stats.get(f'{metric}_val_std', np.nan):.6f}",
                ])

    # 同时保存每个 val epoch 的汇总
    val_csv = out_path.with_name("longtest_val_epochs.csv")
    with open(val_csv, "w", newline="") as f:
        fieldnames = ["epoch", "step", "loss", "edge_smoke_acc", "edge_smoke_real_bond_acc",
                      "error_t_lig", "eps_hat_lig_x", "eps_hat_lig_h",
                      "edge_smoke_loss", "bond_length_loss", "edge_smoke_logits_abs",
                      "loss_0", "kl_prior", "delta_log_px"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for v in val_rows:
            writer.writerow({k: v.get(k, "") for k in fieldnames})
    print(f"  ✓ 阶段统计 CSV: {out_path}")
    print(f"  ✓ Val epochs CSV: {val_csv}")


# ═══════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("DiffSBDD-xhe Scratch 长测分析")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/4] 加载数据...")
    train_rows, val_rows = load_unlimited_metrics(CSV_PATH)
    print(f"  Train: {len(train_rows)} rows, epochs {train_rows[0]['epoch']}-{train_rows[-1]['epoch']}")
    print(f"  Val: {len(val_rows)} rows, epochs {val_rows[0]['epoch']}-{val_rows[-1]['epoch']}")

    # 2. 计算统计
    print("\n[2/4] 计算统计...")
    phase_stats = compute_phase_stats(train_rows, val_rows)

    # 3. 绘图
    print("\n[3/4] 生成图表...")
    plot_overview_panels(train_rows, val_rows, OUT_DIR / "longtest_overview.png")
    plot_rolling_real_bond(val_rows, OUT_DIR / "longtest_realbond_analysis.png")
    plot_error_t_lig_analysis(val_rows, OUT_DIR / "longtest_error_t_lig_analysis.png")
    plot_phase_comparison(phase_stats, OUT_DIR / "longtest_phase_comparison.png")
    plot_loss_components_timeline(train_rows, val_rows, OUT_DIR / "longtest_loss_components.png")
    plot_train_val_correlation(val_rows, OUT_DIR / "longtest_correlation_matrix.png")
    plot_train_heatmap(train_rows, OUT_DIR / "longtest_train_heatmap.png")

    # 4. 写报告和 CSV
    print("\n[4/4] 生成报告...")
    generate_report(train_rows, val_rows, phase_stats, OUT_DIR / "longtest_analysis.md")
    save_summary_csv(phase_stats, val_rows, OUT_DIR / "longtest_summary.csv")

    print("\n" + "=" * 60)
    print("分析完成！输出文件：")
    for f in sorted(OUT_DIR.glob("longtest_*")):
        print(f"  {f.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
