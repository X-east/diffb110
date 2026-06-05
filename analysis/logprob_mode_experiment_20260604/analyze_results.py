"""Analyze short DDPO runs for DiffSBDD logprob aggregation modes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ANALYSIS_DIR = Path("analysis/logprob_mode_experiment_20260604")


@dataclass(frozen=True)
class RunSpec:
    """A completed run to compare."""

    mode: str
    out_dir: Path


RUNS = [
    RunSpec(
        mode="mean_per_element",
        out_dir=Path("OUT/logprob-mode-mean_per_element-kl0p02-5ep_20260604_151726"),
    ),
    RunSpec(
        mode="mean_per_atom",
        out_dir=Path("OUT/logprob-mode-mean_per_atom-kl0p02-5ep_20260604_152126"),
    ),
    RunSpec(
        mode="full_mean_per_atom",
        out_dir=Path("OUT/logprob-mode-full_mean_per_atom-kl0p02-5ep_20260604_152630"),
    ),
]

KEY_COLUMNS = [
    "epoch",
    "generated_total",
    "rdkit_valid",
    "rdkit_valid_ratio",
    "reward_mean",
    "avg_reward_raw",
    "avg_loss",
    "avg_ppo_loss",
    "avg_kl",
    "KL_Penalty_Weighted",
    "PPO_LogRatio_AbsMean",
    "PPO_LogRatio_AbsP95",
    "PPO_LogRatio_AbsMax",
    "PPO_LogRatio_Mean",
    "PPO_LogRatio_Std",
    "PPO_ApproxKL",
    "PPO_ApproxKL_K3",
    "PPO_Ratio_Mean",
    "PPO_Ratio_Std",
    "PPO_Ratio_P95",
    "ClipFrac_HighT",
    "ClipFrac_MidT",
    "ClipFrac_LowT",
    "avg_docking",
    "avg_qed",
    "avg_sa",
    "carbon_fraction",
    "hetero_atom_mean",
    "aromatic_ring_mean",
    "descriptor_mmd_vs_baseline",
]


def _read_runs() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for spec in RUNS:
        csv_path = spec.out_dir / "training_summary.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing training summary: {csv_path}")
        frame = pd.read_csv(csv_path)
        frame.insert(0, "mode", spec.mode)
        frame.insert(1, "out_dir", str(spec.out_dir))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _final_summary(all_rows: pd.DataFrame) -> pd.DataFrame:
    finals = all_rows.sort_values(["mode", "epoch"]).groupby("mode", as_index=False).tail(1)
    keep = ["mode", "out_dir", *[column for column in KEY_COLUMNS if column in finals.columns]]
    return finals[keep].sort_values("mode")


def _format_float(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    rows = frame[columns].copy()
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for _, row in rows.iterrows():
        body.append("| " + " | ".join(_format_float(row[column]) for column in columns) + " |")
    return "\n".join([header, divider, *body])


def _plot_lines(
    all_rows: pd.DataFrame,
    metrics: list[str],
    filename: str,
    title: str,
    ylabel: str,
) -> None:
    available = [metric for metric in metrics if metric in all_rows.columns]
    if not available:
        return
    fig, axes = plt.subplots(len(available), 1, figsize=(8, 3.0 * len(available)), sharex=True)
    if len(available) == 1:
        axes = [axes]
    for axis, metric in zip(axes, available, strict=True):
        for mode, frame in all_rows.groupby("mode"):
            axis.plot(frame["epoch"], frame[metric], marker="o", label=mode)
        axis.set_ylabel(metric)
        axis.grid(True, alpha=0.3)
        axis.legend()
    axes[-1].set_xlabel("epoch")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / filename, dpi=180)
    plt.close(fig)


def _write_report(all_rows: pd.DataFrame, finals: pd.DataFrame) -> None:
    final_cols = [
        "mode",
        "rdkit_valid_ratio",
        "reward_mean",
        "avg_reward_raw",
        "avg_docking",
        "avg_loss",
        "PPO_LogRatio_AbsMean",
        "PPO_LogRatio_AbsP95",
        "PPO_ApproxKL",
        "PPO_ApproxKL_K3",
        "KL_Penalty_Weighted",
        "carbon_fraction",
        "hetero_atom_mean",
        "aromatic_ring_mean",
    ]
    final_cols = [column for column in final_cols if column in finals.columns]

    p95 = finals.set_index("mode")["PPO_LogRatio_AbsP95"].to_dict()
    approx_kl = finals.set_index("mode")["PPO_ApproxKL"].to_dict()
    weighted_kl = finals.set_index("mode")["KL_Penalty_Weighted"].to_dict()
    validity = finals.set_index("mode")["rdkit_valid_ratio"].to_dict()

    lines = [
        "# DiffSBDD logprob 模式短程对照实验",
        "",
        "## 实验设置",
        "",
        "- 基础配置：`OUT/qed-sa-docking-8gjl-8gpu-kl-0p02_20260511_171944/config.yaml`。",
        "- 三组均为 8 GPU、每卡 batch 12、global batch 96、5 epoch、100 步采样、seed 42。",
        "- 只改 `model.logprob_mode` 和必要的 clamp；reward、KL 系数、采样目标保持一致。",
        "- 输出目录见 `final_summary.csv` 的 `out_dir` 列。",
        "",
        "## 最后一轮核心指标",
        "",
        _markdown_table(finals, final_cols),
        "",
        "## 关键结论",
        "",
        (
            "- `mean_per_element` 保留原始尺度，最后一轮 "
            f"`PPO_LogRatio_AbsP95={p95.get('mean_per_element', float('nan')):.4g}`，"
            f"`PPO_ApproxKL={approx_kl.get('mean_per_element', float('nan')):.4g}`，"
            "更新幅度最小。"
        ),
        (
            "- `mean_per_atom` 把特征维度先求和再按原子平均，最后一轮 "
            f"`PPO_LogRatio_AbsP95={p95.get('mean_per_atom', float('nan')):.4g}`，"
            f"`PPO_ApproxKL={approx_kl.get('mean_per_atom', float('nan')):.4g}`，"
            "明显放大 PPO 比率变化，但没有出现 clip frac 大面积爆炸。"
        ),
        (
            "- `full_mean_per_atom` 加入完整 Normal 常数和 `-log(sigma)` 后，"
            f"最后一轮 `PPO_LogRatio_AbsP95={p95.get('full_mean_per_atom', float('nan')):.4g}`，"
            f"`PPO_ApproxKL={approx_kl.get('full_mean_per_atom', float('nan')):.4g}`；"
            "在这个短跑里没有比 `mean_per_atom` 更稳定或更有效。"
        ),
        (
            "- 三组最终有效率分别为 "
            f"`mean_per_element={validity.get('mean_per_element', float('nan')):.2%}`、"
            f"`mean_per_atom={validity.get('mean_per_atom', float('nan')):.2%}`、"
            f"`full_mean_per_atom={validity.get('full_mean_per_atom', float('nan')):.2%}`。"
            "5 epoch 太短，质量趋势只能作为烟测，不应作为最终分子质量判断。"
        ),
        (
            "- `avg_kl` / `KL_Penalty_Weighted` 仍然很小，最后一轮加权 KL 约为 "
            f"`mean_per_element={weighted_kl.get('mean_per_element', float('nan')):.4g}`、"
            f"`mean_per_atom={weighted_kl.get('mean_per_atom', float('nan')):.4g}`、"
            f"`full_mean_per_atom={weighted_kl.get('full_mean_per_atom', float('nan')):.4g}`。"
            "这支持之前判断：真正限制更新的更可能是 logprob 尺度、稀疏奖励和 PPO 比率，而不是当前 KL 惩罚过强。"
        ),
        "",
        "## 图",
        "",
        "![PPO logratio p95](ppo_logratio_abs_p95_by_epoch.png)",
        "",
        "![PPO approximate KL](ppo_approxkl_by_epoch.png)",
        "",
        "![Clip fraction](clipfrac_by_epoch.png)",
        "",
        "![Loss reward validity](loss_reward_validity_by_mode.png)",
        "",
        "![Molecular descriptors](descriptor_by_mode.png)",
        "",
        "## 文件",
        "",
        "- `summary_by_epoch.csv`：三组逐 epoch 指标长表。",
        "- `final_summary.csv`：三组最后一轮摘要。",
        "- `analyze_results.py`：本报告和图的生成脚本。",
        "",
        "## 下一步建议",
        "",
        "1. 保留默认 `mean_per_element` 兼容旧实验，但后续 RL 调参优先尝试 `mean_per_atom`。",
        "2. 先跑 20-30 epoch 的 `mean_per_atom`，确认 `PPO_LogRatio_AbsP95` 是否持续增长、KL 是否出现用户说的周期性跳跃。",
        "3. 同时把 reward 稀疏问题拆出来：增加化学护栏和好分子池重排/重训，否则 logprob 尺度修好后仍可能只是在高碳方向更快偏移。",
        "",
    ]
    (ANALYSIS_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Load completed run CSV files and write tables, plots, and a Markdown report."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = _read_runs()
    key_cols = ["mode", "out_dir", *[column for column in KEY_COLUMNS if column in all_rows.columns]]
    all_rows[key_cols].to_csv(ANALYSIS_DIR / "summary_by_epoch.csv", index=False)
    finals = _final_summary(all_rows)
    finals.to_csv(ANALYSIS_DIR / "final_summary.csv", index=False)

    _plot_lines(
        all_rows,
        ["PPO_LogRatio_AbsP95", "PPO_LogRatio_AbsMean", "PPO_LogRatio_AbsMax"],
        "ppo_logratio_abs_p95_by_epoch.png",
        "PPO logratio magnitude by epoch",
        "logratio",
    )
    _plot_lines(
        all_rows,
        ["PPO_ApproxKL", "PPO_ApproxKL_K3", "KL_Penalty_Weighted"],
        "ppo_approxkl_by_epoch.png",
        "PPO KL diagnostics by epoch",
        "KL",
    )
    _plot_lines(
        all_rows,
        ["ClipFrac_HighT", "ClipFrac_MidT", "ClipFrac_LowT"],
        "clipfrac_by_epoch.png",
        "Clip fraction by timestep bucket",
        "clip fraction",
    )
    _plot_lines(
        all_rows,
        ["avg_loss", "reward_mean", "rdkit_valid_ratio", "avg_docking"],
        "loss_reward_validity_by_mode.png",
        "Training and molecule summary by epoch",
        "value",
    )
    _plot_lines(
        all_rows,
        ["carbon_fraction", "hetero_atom_mean", "aromatic_ring_mean", "descriptor_mmd_vs_baseline"],
        "descriptor_by_mode.png",
        "Descriptor drift by epoch",
        "value",
    )
    _write_report(all_rows, finals)


if __name__ == "__main__":
    main()
