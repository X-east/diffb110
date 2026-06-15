"""Summarize the 2026-06-04 KL diagnostic run."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


RUN_DIR = Path("OUT/kl-diagnostic-8gpu-kl0p02-25ep_20260604_110012")
ANALYSIS_DIR = Path("analysis/kl_diagnostic_20260604_kl0p02_25ep")
SPIKE_THRESHOLD = 0.005


def _to_float(row: dict[str, str], key: str) -> float:
    """Read a float metric from a CSV row."""
    value = row.get(key, "")
    return float(value) if value not in ("", None) else 0.0


def _load_rows() -> list[dict[str, str]]:
    """Load the training summary rows."""
    summary_path = RUN_DIR / "training_summary.csv"
    with summary_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    """Write selected rows and fields to CSV."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _plot_metric(rows: list[dict[str, str]], metric: str, path: Path, ylabel: str) -> None:
    """Plot one metric by epoch."""
    epochs = [int(row["epoch"]) for row in rows]
    values = [_to_float(row, metric) for row in rows]
    plt.figure(figsize=(9, 4.8))
    plt.plot(epochs, values, marker="o", linewidth=1.8, markersize=4)
    if metric == "avg_kl":
        plt.axhline(SPIKE_THRESHOLD, color="tab:red", linestyle="--", linewidth=1.2, label="spike threshold 0.005")
        plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    """Create CSV summaries, figures, and a Markdown report."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    rows = _load_rows()
    spikes = [row for row in rows if _to_float(row, "avg_kl") > SPIKE_THRESHOLD]
    fields = [
        "epoch",
        "avg_kl",
        "KL_Penalty_Weighted",
        "PPO_ApproxKL",
        "PPO_ApproxKL_K3",
        "PPO_LogRatio_AbsP95",
        "PPO_Ratio_Mean",
        "ClipFrac_HighT",
        "ClipFrac_MidT",
        "ClipFrac_LowT",
        "avg_loss",
        "avg_ppo_loss",
        "avg_docking",
        "rdkit_valid_ratio",
    ]
    _write_csv(ANALYSIS_DIR / "kl_diagnostic_selected_metrics.csv", rows, fields)
    _write_csv(ANALYSIS_DIR / "kl_diagnostic_spike_epochs.csv", spikes, fields)

    _plot_metric(rows, "avg_kl", ANALYSIS_DIR / "avg_kl_by_epoch.png", "avg_kl")
    _plot_metric(rows, "PPO_ApproxKL", ANALYSIS_DIR / "ppo_approxkl_by_epoch.png", "PPO_ApproxKL")
    _plot_metric(rows, "KL_Penalty_Weighted", ANALYSIS_DIR / "weighted_kl_by_epoch.png", "KL_Penalty_Weighted")

    latest = rows[-1]
    spike_lines = "\n".join(
        f"| {row['epoch']} | {row['avg_kl']} | {row['KL_Penalty_Weighted']} | "
        f"{row['PPO_ApproxKL']} | {row['PPO_LogRatio_AbsP95']} | "
        f"{row['ClipFrac_HighT']} | {row['ClipFrac_MidT']} | {row['ClipFrac_LowT']} | "
        f"{row['avg_loss']} | {row['avg_docking']} | {row['rdkit_valid_ratio']} |"
        for row in spikes
    )
    report = f"""# KL 0.02 诊断短跑报告

## 目的

复现旧 `kl0p02` 训练中 `avg_kl > {SPIKE_THRESHOLD}` 的两次 KL 跳跃，并利用新增 PPO/KL 诊断列判断跳跃是否真的足以压住 PPO 更新。

## 运行设置

| 项目 | 设置 |
|---|---|
| 配置 | `launch/generated/kl_diagnostics/8gpu_dock_kl0p02_25epoch_diagnostics.yaml` |
| 输出目录 | `{RUN_DIR}` |
| GPU | 8 |
| 全局 batch | 96 |
| seed | 42 |
| KL 系数 | 0.02 |
| epoch | 25 |
| KL 跳跃阈值 | `avg_kl > {SPIKE_THRESHOLD}` |

## 结果

| 指标 | 数值 |
|---|---:|
| 完成 epoch | {len(rows)} |
| KL 跳跃次数 | {len(spikes)} |
| 最终 `avg_kl` | {latest['avg_kl']} |
| 最终 `KL_Penalty_Weighted` | {latest['KL_Penalty_Weighted']} |
| 最终 `PPO_ApproxKL` | {latest['PPO_ApproxKL']} |
| 最终 `avg_loss` | {latest['avg_loss']} |
| 最终 `avg_ppo_loss` | {latest['avg_ppo_loss']} |
| 最终 `avg_docking` | {latest['avg_docking']} |
| 最终 `rdkit_valid_ratio` | {latest['rdkit_valid_ratio']} |

## KL 跳跃 epoch

| epoch | avg_kl | KL_Penalty_Weighted | PPO_ApproxKL | PPO_LogRatio_AbsP95 | ClipFrac_HighT | ClipFrac_MidT | ClipFrac_LowT | avg_loss | avg_docking | rdkit_valid_ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{spike_lines}

## 图

![avg_kl_by_epoch](avg_kl_by_epoch.png)

![ppo_approxkl_by_epoch](ppo_approxkl_by_epoch.png)

![weighted_kl_by_epoch](weighted_kl_by_epoch.png)

## 初步结论

1. 两次 KL 跳跃已经复现，位置为 epoch 12 和 epoch 20，与旧 `kl0p02` 记录一致。
2. epoch 12 的 `avg_kl=0.030007`，但乘以 `kl_coef=0.02` 后只有 `0.000600`，远小于同 epoch 的 `avg_ppo_loss=0.360759`。
3. epoch 20 的 `avg_kl=0.007593`，加权后只有 `0.000152`，同样远小于 `avg_ppo_loss=0.547394`。
4. 两个跳跃 epoch 的 `ClipFrac_HighT/MidT/LowT` 都没有同步异常，`ClipFrac_LowT` 维持在 0.028571，说明 PPO clip 不是这两次 KL 跳跃的主要限制。
5. `PPO_ApproxKL` 在 epoch 12 同步升高到 0.036748，但 epoch 20 只到 0.019502，说明 frozen-reference KL 跳跃和 old-policy approximate KL 相关但不完全等价。

## 后续判断

这次结果支持前一份 KL 对比报告的判断：日志里的 frozen analytic KL 会跳，但加权后对 loss 的直接贡献很小。下一步应继续检查 reward/advantage 稀疏、docking outlier、logprob 尺度和 KL 归一化，而不是单纯认为 `kl_coef=0.02` 直接把模型训练压死。
"""
    (ANALYSIS_DIR / "report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
