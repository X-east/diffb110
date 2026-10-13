"""绘制 DiffGui Next6-04 两阶段 RL 漂移图（合法提升 → reward hacking）。

用法：
    conda run -n dif python analysis/diffgui_rl_r1r5_drift/plot_rl_drift_two_stage.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from pathlib import Path  # noqa: E402

OUT = Path(__file__).resolve().parent

sns.set_theme(style="whitegrid", context="paper", font_scale=1.25)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.edgecolor"] = "#d4d4d4"
plt.rcParams["axes.linewidth"] = 1.2

BASE = OUT / (
    "diffgui-next6-20260618-04-ring_last3_pathology_guard"
    "_20260619_112410_r1r5_drift.csv"
)
CONT = OUT / (
    "diffgui-next6-20260618-04-ring_last3_pathology_guard"
    "-continue-to50_20260621_225705_r1r5_drift.csv"
)

PANELS = [
    ("reward_scored_mean", "Reward (scored molecules)", "#2E86C1"),
    ("reward_for_advantage_mean", "Reward (with failure penalty)", "#1B4F72"),
    ("tpsa_mean", "TPSA (A^2)", "#2ECC71"),
    ("n_hetero_mean", "Hetero atom count", "#F39C12"),
    ("frac_carbon_ge_0p90", "Fraction with C-ratio >= 0.90", "#C0392B"),
    ("r5_violation", "R5 carbon-fraction violation rate", "#8E44AD"),
    ("reasonable_ratio_r1r5", "Reasonable rate (R1-R5 all pass)", "#16A085"),
]


def load() -> pd.DataFrame:
    frames = []
    for p in (BASE, CONT):
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        raise SystemExit("no drift csv found; run compute script first")
    df = pd.concat(frames, ignore_index=True).sort_values("epoch")
    return df.reset_index(drop=True)


def main() -> None:
    df = load()
    stage2_start = int(df.loc[df["epoch"] >= 21, "epoch"].min())

    fig, axes = plt.subplots(4, 2, figsize=(15, 14))
    axes = axes.flatten()

    for ax, (col, label, color) in zip(axes, PANELS):
        if col not in df.columns:
            ax.axis("off")
            continue
        ax.plot(
            df["epoch"], df[col], marker="o", linewidth=2.2,
            markersize=6, color=color,
        )
        ax.axvline(
            stage2_start - 0.5, color="#E74C3C", linestyle="--",
            linewidth=2.0, alpha=0.85, zorder=4,
        )
        ax.text(
            stage2_start + 0.6, ax.get_ylim()[1],
            "continued training (hacking)",
            color="#E74C3C", fontweight="bold", fontsize=10, va="top",
        )
        sns.despine(ax=ax)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(label, fontsize=12, fontweight="bold")

    # 最后一个子图：reward 与 TPSA 的双轴对照（核心论点）
    ax = axes[7]
    ax2 = ax.twinx()
    l1 = ax.plot(
        df["epoch"], df["reward_scored_mean"], marker="o",
        linewidth=2.2, color="#2E86C1", label="Reward (scored)",
    )
    l2 = ax2.plot(
        df["epoch"], df["tpsa_mean"], marker="s",
        linewidth=2.2, color="#C0392B", label="TPSA",
    )
    ax.axvline(
        stage2_start - 0.5, color="#E74C3C", linestyle="--",
        linewidth=2.0, alpha=0.85,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reward (scored)", color="#2E86C1")
    ax2.set_ylabel("TPSA (A^2)", color="#C0392B")
    ax.set_title(
        "Reward rises while TPSA collapses", fontsize=12, fontweight="bold"
    )
    ax.legend(l1 + l2, [l.get_label() for l in l1 + l2], loc="upper left")
    sns.despine(ax=ax, right=False)

    fig.suptitle(
        "DiffGui Next6-04 RL: legitimate improvement (ep 1-20) "
        "then reward hacking (ep 21-50)",
        fontsize=15, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))

    for ext in ("png", "svg"):
        fig.savefig(OUT / f"diffgui_rl_two_stage_drift.{ext}", dpi=300)
    plt.close(fig)
    print("saved: diffgui_rl_two_stage_drift.png / .svg")


if __name__ == "__main__":
    main()
