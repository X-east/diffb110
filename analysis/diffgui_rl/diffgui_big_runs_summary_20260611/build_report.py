from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OUT_DIR = Path("analysis/diffgui_big_runs_summary_20260611")
OUT_DIR.mkdir(parents=True, exist_ok=True)


RUNS = [
    {
        "run_id": "chembl21_dock_last1_lr1e6",
        "label": "Dock+classifier v0.2.1",
        "run_dir": Path("OUT/diffgui-8gpu-chembl21-dock_20260609_224313"),
        "reward": "docking 0.5 + classifier v0.2.1 0.5",
        "setting": "8GPU, b24/card, 200 steps, last1, lr=1e-6, PPO=2",
        "note": "初版正式训练，用户手动停止于 22 epoch 前后。",
    },
    {
        "run_id": "docking_only_last2_lr1e5",
        "label": "Pure docking",
        "run_dir": Path("OUT/diffgui-8gpu-docking-only-lr1e5-last2_20260610_094719"),
        "reward": "docking 1.0",
        "setting": "8GPU, b24/card, 200 steps, last2, lr=1e-5, PPO=1",
        "note": "用于判断单独 docking 是否能推出改善趋势，5 个完整 epoch 后停止。",
    },
    {
        "run_id": "ring_last3_lr3e5",
        "label": "Ring reward",
        "run_dir": Path("OUT/diffgui-8gpu-ring-last3-lr3e5_20260610_114650"),
        "reward": "docking 0.35 + ring_complexity 0.30 + chemistry_guard 0.10",
        "setting": "8GPU, b24/card, 200 steps, last3, lr=3e-5, PPO=1",
        "note": "重点测试能否改善复杂环；23 个完整 summary epoch，epoch24 只有 rank-level 文件。",
    },
    {
        "run_id": "classifier_v031_b36_ppo4",
        "label": "Classifier v0.3.1",
        "run_dir": Path("OUT/diffgui-8gpu-b36-classifier-v0-3-1-ppo4_20260610_221410"),
        "reward": "docking 0.5 + classifier v0.3.1 0.5",
        "setting": "8GPU, b36/card, 200 steps, last1, lr=1e-6, PPO=4",
        "note": "用坏分子审阅后重训分类器测试；8 个完整 epoch，单 epoch PPO 成本很高。",
    },
]

ABORTED_RUNS = [
    {
        "run_id": "aromatic_last3_lr3e5",
        "label": "Aromatic scaffold",
        "run_dir": Path("OUT/diffgui-8gpu-aromatic-last3-lr3e5_20260610_195621"),
        "reward": "docking 0.30 + aromatic_scaffold 0.35 + ring_complexity 0.10 + chemistry_guard 0.10",
        "setting": "8GPU, b24/card, 200 steps, last3, lr=3e-5, PPO=1",
        "note": "launch log 显示 2026-06-10 19:58 SIGTERM；无 training_summary.csv，不能作为有效训练趋势。",
    }
]


METRICS = [
    "reward_total_mean",
    "reward_total_scored_mean",
    "docking_raw_mean",
    "docking_reward_mean",
    "reconstruction_success_ratio",
    "docking_success_ratio_total",
    "molecule_rdkit_valid_ratio",
    "carbon_fraction",
    "hetero_atom_mean",
    "aromatic_ring_mean",
    "ring_count_mean",
    "three_member_ring_mean",
    "PPO_ApproxKL",
    "ClipFrac",
    "total_seconds",
    "rollout_seconds",
    "reward_seconds",
    "ppo_seconds",
]


def _read_manifest(run_dir: Path) -> dict:
    path = run_dir / "run_manifest.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_runs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    epoch_frames = []
    overview_rows = []
    delta_rows = []

    for meta in RUNS:
        run_dir = meta["run_dir"]
        summary_path = run_dir / "training_summary.csv"
        manifest = _read_manifest(run_dir)
        df = pd.read_csv(summary_path)
        df["run_id"] = meta["run_id"]
        df["run_label"] = meta["label"]
        df["reward_setting"] = meta["reward"]
        df["setting"] = meta["setting"]
        epoch_frames.append(df)

        first = df.iloc[0]
        last = df.iloc[-1]
        row = {
            "run_id": meta["run_id"],
            "label": meta["label"],
            "reward": meta["reward"],
            "setting": meta["setting"],
            "note": meta["note"],
            "manifest_status": manifest.get("status", "NA"),
            "epochs_with_summary": int(len(df)),
            "first_epoch": int(first["epoch"]),
            "last_epoch": int(last["epoch"]),
            "global_batch_size": int(last.get("global_batch_size", 0)),
            "per_rank_batch_size": int(last.get("per_rank_batch_size", 0)),
            "num_timesteps": int(last.get("num_timesteps", 0)),
            "avg_total_seconds": float(df["total_seconds"].mean()),
            "avg_rollout_seconds": float(df["rollout_seconds"].mean()),
            "avg_reward_seconds": float(df["reward_seconds"].mean()),
            "avg_ppo_seconds": float(df["ppo_seconds"].mean()),
            "ppo_time_fraction": float(df["ppo_seconds"].sum() / df["total_seconds"].sum()),
        }
        for metric in METRICS:
            if metric in df.columns:
                row[f"first_{metric}"] = float(first[metric])
                row[f"last_{metric}"] = float(last[metric])
                row[f"delta_{metric}"] = float(last[metric] - first[metric])
        overview_rows.append(row)

        for metric in METRICS:
            if metric in df.columns:
                delta_rows.append(
                    {
                        "run_id": meta["run_id"],
                        "label": meta["label"],
                        "metric": metric,
                        "first": float(first[metric]),
                        "last": float(last[metric]),
                        "delta": float(last[metric] - first[metric]),
                    }
                )

    aborted_rows = []
    for meta in ABORTED_RUNS:
        manifest = _read_manifest(meta["run_dir"])
        aborted_rows.append(
            {
                "run_id": meta["run_id"],
                "label": meta["label"],
                "reward": meta["reward"],
                "setting": meta["setting"],
                "note": meta["note"],
                "manifest_status": manifest.get("status", "NA"),
                "epochs_with_summary": 0,
                "first_epoch": None,
                "last_epoch": None,
            }
        )

    epoch_df = pd.concat(epoch_frames, ignore_index=True)
    overview_df = pd.DataFrame(overview_rows + aborted_rows)
    delta_df = pd.DataFrame(delta_rows)
    return epoch_df, overview_df, delta_df


def _plot_lines(epoch_df: pd.DataFrame, filename: str, title: str, metrics: list[tuple[str, str]]) -> None:
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 3.2 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]
    for ax, (metric, ylabel) in zip(axes, metrics):
        for label, df in epoch_df.groupby("run_label", sort=False):
            if metric not in df.columns or df[metric].isna().all():
                continue
            ax.plot(df["epoch"], df[metric], marker="o", linewidth=1.7, markersize=3.5, label=label)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    axes[0].set_title(title)
    axes[-1].set_xlabel("epoch")
    axes[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=180)
    fig.savefig(OUT_DIR / filename.replace(".png", ".svg"))
    plt.close(fig)


def _plot_delta(delta_df: pd.DataFrame) -> None:
    metrics = [
        "reward_total_mean",
        "reconstruction_success_ratio",
        "docking_success_ratio_total",
        "carbon_fraction",
        "aromatic_ring_mean",
        "ring_count_mean",
        "three_member_ring_mean",
    ]
    sub = delta_df[delta_df["metric"].isin(metrics)].copy()
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.ravel()
    for ax, metric in zip(axes, metrics):
        part = sub[sub["metric"] == metric]
        ax.bar(part["label"], part["delta"], color="#4C78A8")
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.grid(axis="y", alpha=0.25)
    for ax in axes[len(metrics) :]:
        ax.axis("off")
    fig.suptitle("DiffGui run first-to-last deltas", y=1.0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_start_end_delta_bars.png", dpi=180)
    fig.savefig(OUT_DIR / "04_start_end_delta_bars.svg")
    plt.close(fig)


def _write_markdown(overview_df: pd.DataFrame) -> None:
    cols = [
        "label",
        "reward",
        "setting",
        "epochs_with_summary",
        "last_reward_total_mean",
        "delta_reward_total_mean",
        "last_reconstruction_success_ratio",
        "last_docking_success_ratio_total",
        "last_carbon_fraction",
        "last_aromatic_ring_mean",
        "last_ring_count_mean",
        "last_three_member_ring_mean",
        "avg_total_seconds",
        "ppo_time_fraction",
        "note",
    ]
    present = [c for c in cols if c in overview_df.columns]
    table = overview_df[present].copy()
    for c in table.columns:
        if pd.api.types.is_float_dtype(table[c]):
            table[c] = table[c].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    table = table.fillna("")
    header = "| " + " | ".join(table.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(table.columns)) + " |"
    body = "\n".join(
        "| " + " | ".join(str(value).replace("\n", " ") for value in row) + " |"
        for row in table.itertuples(index=False, name=None)
    )
    markdown_table = "\n".join([header, sep, body])

    report = f"""# DiffGui 几次大训练复盘

## 结论先行

1. DiffGui RL 工程已经跑通，200 step / 8 GPU 的 rollout 和 reward 评分稳定，真正的时间瓶颈主要在 PPO 更新。
2. 初版 `docking + classifier v0.2.1` 没有看到 reward 或 docking 的明确上升，说明简单把 DiffSBDD 的 reward 迁移到 DiffGui 不够。
3. `pure docking / lr=1e-5 / last2` 5 个 epoch 内 docking reward 反而略降，不能证明单独 docking 能把方向推好。
4. `ring-last3 / lr=3e-5` 能明显提高总 reward、重建成功率和 docking 成功率，也降低三元环，但同时把芳香环/苯环压下去，并让大环/复杂环问题暴露出来。
5. `classifier v0.3.1 / b36 / PPO=4` 单 epoch 成本过高，前 8 epoch reward 在低位波动，没有看到足够强的改善趋势。
6. `aromatic_scaffold` 那次只启动后被 SIGTERM，没有完整 summary，不能作为训练结论。

## 训练总览

{markdown_table}

## 图件

![reward and success](01_reward_success_trends.png)

![descriptor trends](02_descriptor_trends.png)

![docking and runtime](03_docking_runtime_trends.png)

![start end delta](04_start_end_delta_bars.png)

## 解释

DiffGui 的底座比 DiffSBDD 更稳，因为它原生预测 atom/bond/halfedge/position，成功写入 SDF 的重建合法性更高。但这几次 RL 说明：底座更强不等于 reward 可以粗糙。只奖励 docking 或笼统惩罚 ring，会把模型推向新的漏洞：苯环减少、复杂脂环/大环仍然存在、卤素和小分子异常需要单独处理。

当前最有价值的结论不是“某个 reward 已经成功”，而是已经定位了下一版 reward 的边界：

- docking 可以保留，但不应单独主导；
- classifier 可以做软约束，但必须用更贴近人工审阅规则的负样本；
- ring reward 不能只惩罚环数量或复杂度，需要明确奖励合理单苯环/合理芳香骨架；
- 强负样本应包括多稠杂环、桥头/螺原子复杂拓扑、七元及以上大环，特别是大环并六元环；
- 正常的两个环稠合、合理三元环、一两个卤素不应被过度惩罚。

## 建议

下一轮 DiffGui 不建议再开 100 epoch 大跑。先用每卡 12-24 分子、3-5 epoch 的小测试，每次只改 1-2 个变量：

1. 固定 `last3, lr=1e-5 或 2e-5, PPO=1`，先把训练速度和 PPO 稳定性拉回来。
2. reward 用 `docking 0.25-0.35 + classifier 0.25-0.35 + rule reward 0.20-0.30`。
3. rule reward 里分开处理：单苯环/合理芳香骨架弱奖励，多稠杂环/大环并六元环/复杂桥螺拓扑强扣分，过小高卤素分子强扣分。
4. 每轮必须输出分子级 reason flags，并抽样进 Maestro 看，不只看 epoch 均值。
"""
    (OUT_DIR / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    epoch_df, overview_df, delta_df = _load_runs()
    epoch_df.to_csv(OUT_DIR / "diffgui_epoch_metrics_long.csv", index=False)
    overview_df.to_csv(OUT_DIR / "diffgui_big_runs_overview.csv", index=False)
    delta_df.to_csv(OUT_DIR / "diffgui_start_end_delta.csv", index=False)

    _plot_lines(
        epoch_df,
        "01_reward_success_trends.png",
        "DiffGui reward and success trends",
        [
            ("reward_total_mean", "reward mean"),
            ("reconstruction_success_ratio", "reconstruction success"),
            ("docking_success_ratio_total", "docking success total"),
        ],
    )
    _plot_lines(
        epoch_df,
        "02_descriptor_trends.png",
        "DiffGui descriptor trends",
        [
            ("carbon_fraction", "carbon fraction"),
            ("hetero_atom_mean", "hetero atoms"),
            ("aromatic_ring_mean", "aromatic rings"),
            ("three_member_ring_mean", "3-member rings"),
        ],
    )
    _plot_lines(
        epoch_df,
        "03_docking_runtime_trends.png",
        "DiffGui docking and runtime trends",
        [
            ("docking_raw_mean", "docking raw, lower better"),
            ("docking_reward_mean", "docking reward"),
            ("ppo_seconds", "PPO seconds"),
            ("rollout_seconds", "rollout seconds"),
        ],
    )
    _plot_delta(delta_df)
    _write_markdown(overview_df)


if __name__ == "__main__":
    main()
