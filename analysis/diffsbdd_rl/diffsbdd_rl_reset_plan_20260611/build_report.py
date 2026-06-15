from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_DIR = Path("analysis/diffsbdd_rl_reset_plan_20260611")


RUNS = [
    {
        "label": "dock+classifier v0.1",
        "short": "dock_cls_0529",
        "path": Path("OUT/7gpu-dock-classifier-v0-1_20260529_133609/training_summary.csv"),
        "type": "control_lost",
        "reward": "docking 0.5 + classifier 0.5; KL=0",
        "diagnosis": "docking 变好但高碳/低 TPSA/低杂原子漂移明显",
    },
    {
        "label": "classifier only",
        "short": "cls_only_0526",
        "path": Path("OUT/7gpu-classifier-only_20260526_210715/training_summary.csv"),
        "type": "collapse",
        "reward": "classifier 1.0; KL=0",
        "diagnosis": "classifier 单独做导航会把有效率推崩，不能作为唯一 reward",
    },
    {
        "label": "dock+prior+logP",
        "short": "dock_prior_logp_0525",
        "path": Path("OUT/7gpu-dock-ecfp4-fcfp4_20260525_212140/training_summary.csv"),
        "type": "control_lost",
        "reward": "docking + ECFP/FCFP prior + logP distribution; KL=0",
        "diagnosis": "docking/valid 变好，但高碳和少杂原子更严重",
    },
    {
        "label": "LE+prior+logP",
        "short": "le_prior_logp_0526",
        "path": Path("OUT/7gpu-dock-ecfp4-fcfp4_20260526_115341/training_summary.csv"),
        "type": "control_lost",
        "reward": "LE + ECFP/FCFP prior + logP distribution; KL=0",
        "diagnosis": "没有 docking 也会向高 logP、高碳、低 TPSA 漂移",
    },
    {
        "label": "anti-carbon guard b64",
        "short": "guard_b64_0611",
        "path": Path("OUT/diffsbdd-8gpu-b64-anticarbon-guard-20260611_20260611_155548/training_summary.csv"),
        "type": "over_control",
        "reward": "docking 0.2 + prior/logP + chemistry_guard 0.3 + KL=0.001",
        "diagnosis": "高碳被压住，但芳香环几乎消失，docking 变差，属于过修",
    },
]


METRICS = [
    "rdkit_valid_ratio",
    "reward_mean",
    "avg_docking",
    "logp_mean",
    "tpsa_mean",
    "carbon_fraction",
    "hetero_atom_mean",
    "aromatic_ring_mean",
    "fraction_logp_high",
    "fraction_tpsa_low",
]


def _load_selected_runs() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for run in RUNS:
        if not run["path"].exists():
            continue
        df = pd.read_csv(run["path"])
        if df.empty:
            continue
        df["run_label"] = run["label"]
        df["run_short"] = run["short"]
        frames.append(df)

        first = df.iloc[0]
        last = df.iloc[-1]
        row: dict[str, object] = {
            "run_label": run["label"],
            "run_short": run["short"],
            "epochs": int(last["epoch"]),
            "generated_per_epoch": int(last["generated_total"]) if "generated_total" in df.columns else "",
            "reward_setup": run["reward"],
            "diagnosis": run["diagnosis"],
            "type": run["type"],
        }
        for metric in METRICS:
            if metric in df.columns:
                start = float(first[metric])
                end = float(last[metric])
                row[f"{metric}_start"] = start
                row[f"{metric}_end"] = end
                row[f"{metric}_delta"] = end - start
        summary_rows.append(row)

    all_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    return all_df, summary


def _write_tables(all_df: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    all_df.to_csv(OUT_DIR / "selected_run_epoch_metrics.csv", index=False)
    summary.to_csv(OUT_DIR / "selected_run_start_end_delta.csv", index=False)

    readable_cols = [
        "run_label",
        "epochs",
        "generated_per_epoch",
        "reward_setup",
        "rdkit_valid_ratio_start",
        "rdkit_valid_ratio_end",
        "avg_docking_start",
        "avg_docking_end",
        "carbon_fraction_start",
        "carbon_fraction_end",
        "hetero_atom_mean_start",
        "hetero_atom_mean_end",
        "tpsa_mean_start",
        "tpsa_mean_end",
        "logp_mean_start",
        "logp_mean_end",
        "aromatic_ring_mean_start",
        "aromatic_ring_mean_end",
        "diagnosis",
    ]
    readable = summary[[col for col in readable_cols if col in summary.columns]]
    readable.to_csv(OUT_DIR / "selected_run_readable_summary.csv", index=False)
    return readable


def _write_plots(all_df: pd.DataFrame, summary: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size": 10, "axes.unicode_minus": False})

    label_map = {
        "dock_cls_0529": "dock_classifier_0529",
        "dock_prior_logp_0525": "dock_prior_logp_0525",
        "cls_only_0526": "classifier_only_0526",
        "le_prior_logp_0526": "le_prior_logp_0526",
        "guard_b64_0611": "guard_b64_0611",
    }

    line_styles = {
        "dock_cls_0529": {"color": "#4C78A8", "linestyle": "-"},
        "dock_prior_logp_0525": {"color": "#F58518", "linestyle": "-"},
        "cls_only_0526": {"color": "#54A24B", "linestyle": "-"},
        "le_prior_logp_0526": {"color": "#B279A2", "linestyle": "-"},
        "guard_b64_0611": {"color": "#E45756", "linestyle": "--"},
    }

    core_metrics = [
        ("carbon_fraction", "Carbon fraction", None),
        ("hetero_atom_mean", "Hetero atoms", None),
        ("tpsa_mean", "TPSA", None),
        ("avg_docking", "Avg docking score", None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8), constrained_layout=True)
    for ax, (metric, title, _) in zip(axes.flat, core_metrics):
        for run in RUNS:
            sub = all_df[all_df["run_short"] == run["short"]]
            if sub.empty or metric not in sub.columns:
                continue
            style = line_styles.get(run["short"], {})
            ax.plot(
                sub["epoch"],
                sub[metric],
                marker="o",
                markersize=2.6,
                linewidth=1.7,
                label=label_map.get(run["short"], run["short"]),
                **style,
            )
        if metric == "carbon_fraction":
            ax.axhline(0.80, color="#111111", linewidth=1.1, linestyle=":")
            ax.text(
                0.98,
                0.83,
                "C=0.80 guard line",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                color="#333333",
            )
        if metric == "avg_docking":
            ax.invert_yaxis()
            ax.text(
                0.02,
                0.04,
                "inverted axis; lower is better",
                transform=ax.transAxes,
                fontsize=8,
                color="#555555",
            )
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("DiffSBDD RL selected run descriptor drift", fontsize=14, fontweight="bold")
    fig.savefig(OUT_DIR / "selected_run_descriptor_drift.png", dpi=220)
    fig.savefig(OUT_DIR / "selected_run_descriptor_drift.svg")
    plt.close(fig)

    plot_metrics = [
        ("carbon_fraction", "Carbon fraction", (0.55, 0.82)),
        ("hetero_atom_mean", "Hetero atom mean", None),
        ("tpsa_mean", "TPSA mean", (60, 120)),
        ("logp_mean", "logP mean", (0.5, 3.5)),
        ("aromatic_ring_mean", "Aromatic ring mean", None),
        ("avg_docking", "Docking score", None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, (metric, title, band) in zip(axes.flat, plot_metrics):
        for run in RUNS:
            sub = all_df[all_df["run_short"] == run["short"]]
            if sub.empty or metric not in sub.columns:
                continue
            style = line_styles.get(run["short"], {})
            ax.plot(
                sub["epoch"],
                sub[metric],
                marker="o",
                markersize=2.5,
                linewidth=1.5,
                label=label_map.get(run["short"], run["short"]),
                **style,
            )
        if band is not None:
            ax.axhspan(band[0], band[1], color="tab:green", alpha=0.08)
        if metric == "avg_docking":
            ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("DiffSBDD RL selected run drift comparison, extended metrics", fontsize=14)
    fig.savefig(OUT_DIR / "selected_run_descriptor_drift_6panel.png", dpi=180)
    fig.savefig(OUT_DIR / "selected_run_descriptor_drift_6panel.svg")
    plt.close(fig)

    bar_metrics = [
        ("carbon_fraction_delta", "Delta carbon fraction"),
        ("hetero_atom_mean_delta", "Delta hetero atom mean"),
        ("tpsa_mean_delta", "Delta TPSA mean"),
        ("logp_mean_delta", "Delta logP mean"),
        ("avg_docking_delta", "Delta docking score"),
        ("aromatic_ring_mean_delta", "Delta aromatic ring mean"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, (metric, title) in zip(axes.flat, bar_metrics):
        ax.bar(summary["run_label"], summary[metric])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Start-to-end metric deltas", fontsize=14)
    fig.savefig(OUT_DIR / "selected_run_start_end_delta.png", dpi=180)
    plt.close(fig)


def _next_plan() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "step": "T1",
                "purpose": "确认 anti-carbon 是否只是过强",
                "change_only": "只改 chemistry_guard weight: 0.30 -> 0.10",
                "keep_fixed": "violation_penalty、其他 reward 权重、lr=1e-5、ppo_epochs=2、KL=0.001 不变",
                "small_run": "8GPU, batch 16-32/card, 3-5 epochs",
                "pass_signal": "carbon_fraction 0.65-0.78; hetero_atom_mean 5.5-7.5; aromatic_ring_mean 不低于 0.04; docking 不劣于 -7.5",
                "stop_signal": "aromatic_ring_mean 继续接近 0 或 avg_docking > -6.5",
            },
            {
                "step": "T2",
                "purpose": "确认单分子违规惩罚是否过强",
                "change_only": "只改 violation_penalty: 0.35 -> 0.15",
                "keep_fixed": "使用 T1 选定的 chemistry_guard weight；其他 reward 权重、lr、ppo_epochs、KL 不变",
                "small_run": "8GPU, batch 16-32/card, 3-5 epochs",
                "pass_signal": "docking/LE 有回升，carbon/TPSA/hetero 仍不过界",
                "stop_signal": "高碳漂移重新出现，或 valid/sanitize 明显变差",
            },
            {
                "step": "T3",
                "purpose": "恢复苯环/芳香骨架，不奖励多稠环",
                "change_only": "只新增/调整 aromatic scaffold reward weight: 0.03-0.05",
                "keep_fixed": "使用 T1/T2 最稳 guard；不动 docking/先验/logP；不动学习率",
                "small_run": "8GPU, batch 16-32/card, 3-5 epochs",
                "pass_signal": "aromatic_ring_mean 回到 0.06-0.15；复杂环不过度上升；carbon 不超过 0.80",
                "stop_signal": "多环/稠环比例肉眼明显增加，或 carbon_fraction 再上 0.82",
            },
            {
                "step": "T4",
                "purpose": "判断 KL/DePPA-style 是否能稳住漂移",
                "change_only": "只切换 kl_mode: analytic_current_ref vs pretrained_logprob_k3；kl_coef 不变",
                "keep_fixed": "用 T1/T2/T3 最稳 reward，不动任何 reward 权重",
                "small_run": "每组 3 epochs",
                "pass_signal": "reward 更新时 PPO_LogRatio/KL 稳，描述符漂移变慢",
                "stop_signal": "KL 很小但漂移照旧，说明 reward 问题优先于 KL",
            },
            {
                "step": "T5",
                "purpose": "再看 docking 是否能回升",
                "change_only": "只改 docking reward weight: 0.20 -> 0.25 或 0.30",
                "keep_fixed": "结构 guard 和 KL 固定为前面最稳版本；不动其他 reward 权重",
                "small_run": "8GPU, batch 16-32/card, 5 epochs",
                "pass_signal": "avg_docking 变更负，同时 carbon/TPSA/hetero 不越界",
                "stop_signal": "docking 变好但 carbon_fraction 快速上升、TPSA 快速下降",
            },
        ]
    )


def _write_report(readable: pd.DataFrame, plan: pd.DataFrame) -> None:
    def to_markdown_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "_无数据_"
        display = df.copy()
        for col in display.columns:
            if pd.api.types.is_float_dtype(display[col]):
                display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
            else:
                display[col] = display[col].map(lambda x: "" if pd.isna(x) else str(x))
        headers = list(display.columns)
        rows = display.values.tolist()
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in rows:
            lines.append("| " + " | ".join(str(value).replace("\n", " ") for value in row) + " |")
        return "\n".join(lines)

    md = [
        "# DiffSBDD RL 旧训练漂移与下一轮小步测试计划\n",
        "生成日期：2026-06-11\n",
        "## 核心判断\n",
        "- 旧的 docking / classifier / ChEMBL prior / logP 组合不是没有学到东西，而是学到的方向经常是“高碳、低 TPSA、少杂原子、强疏水”。",
        "- 最新 anti-carbon guard b64 证明显式描述符护栏能压住高碳，但 `chemistry_guard=0.30` 叠加 strict pool/过滤后过强：芳香环几乎消失，docking 从约 -7.8 走弱到约 -5.7。",
        "- 下一步不要大改 reward，也不要一次上复杂分类器/复杂 guard；奖励函数部分必须执行单变量原则：每次实验只允许改动一个 reward 相关变量，其余 reward 权重和阈值固定，否则训练结束后无法判断是哪一项带来了变化。",
        "- DePPA 的启发不是照搬整套，而是三点：rank-based dense reward、invalid 相对惩罚、pretrained KL 防偏离。\n",
        "## 关键图\n",
        "![selected run descriptor drift](selected_run_descriptor_drift.png)\n",
        "![selected run start end delta](selected_run_start_end_delta.png)\n",
        "## 代表性 run 起止对比\n",
        to_markdown_table(readable),
        "\n## DePPA 做了什么，和当前 DiffB110 的关系\n",
        "- DePPA 也是在 DiffSBDD/DDPM 反扩散轨迹上做 PPO 式更新：rollout 时保存每个反扩散步的 observation/action/old logprob，reward 后重新算 new logprob，用 PPO clip loss 更新 denoiser。",
        "- 它的 reward 默认是 QED、SA、Vina score、diversity/distance，可选 strain；每个 reward 先在本批 valid 分子里做 rank/percentile，再映射到正态分布，所以信号更稠密。",
        "- invalid 分子不是简单固定 -1，而是设为 valid 分子 combined reward 均值减 3 个标准差；这比硬阈值更平滑。",
        "- 它可选 frozen pretrained model，使用 sampled K3 近似 KL(current || pretrained) 防止策略偏离预训练模型。当前项目已经有 `pretrained_logprob_k3` 和 `analytic_current_ref` 两个可测 KL 模式。",
        "- 但 DePPA 本身没有解决你现在发现的药化问题：高碳、稠环、七元环、大环并六元环、苯环需要但多稠环不需要。这些仍要靠你自己的 descriptor/structure guard。\n",
        "## 后续实验纪律\n",
        "- 奖励函数部分只允许单变量改动：一次实验只改一个 reward 权重、阈值或开关。",
        "- 如果要测试 KL mode、学习率、batch、PPO epochs，这些应作为单独实验，不和 reward 变量同时改。",
        "- 每次实验记录必须写清楚：唯一改动项、固定项、预期方向、通过/停止标准。否则结果不能用于总结 reward 设计。\n",
        "## 建议下一轮小步测试\n",
        to_markdown_table(plan),
        "\n## 文件\n",
        "- `selected_run_epoch_metrics.csv`：所选 run 的逐 epoch 指标。",
        "- `selected_run_start_end_delta.csv`：起止变化。",
        "- `selected_run_readable_summary.csv`：更容易读的摘要表。",
        "- `incremental_test_plan.csv`：下一轮小步测试表。",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_df, summary = _load_selected_runs()
    readable = _write_tables(all_df, summary)
    _write_plots(all_df, summary)
    plan = _next_plan()
    plan.to_csv(OUT_DIR / "incremental_test_plan.csv", index=False)
    _write_report(readable, plan)
    print(OUT_DIR)


if __name__ == "__main__":
    main()
