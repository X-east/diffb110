"""Plot the ESRPO step-wise yield-vs-quality trade-off curve (paper main figure).

Reads the per-step metrics CSV produced by ``evaluate_reasonable_molecules.py``
and plots the three decisive quantities against the optimizer update step:

  * reasonable-molecule yield per 1000 attempts (主收益, 越高越好)
  * docking score median of reasonable molecules (越低越好)
  * QED median of reasonable molecules (越高越好)

The best停留点 is the step that maximises the yield while keeping quality
close to the baseline; it is highlighted on the figure.

Usage:
  python tools/analysis_scripts/plot_esrpo_step_tradeoff.py \
      --metrics analysis/e2po_mainline_reasonable_eval/reasonable_molecules_metrics.csv \
      --baseline-yield 30.5 --baseline-dock -8.92 --baseline-qed 0.647 \
      --out-dir analysis/e2po_mainline_reasonable_eval
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper", font_scale=1.25)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.edgecolor"] = "#d4d4d4"
plt.rcParams["axes.linewidth"] = 1.2

YIELD_COLOR = "#E74C3C"
DOCK_COLOR = "#3498DB"
QED_COLOR = "#2ECC71"
BASELINE_GRAY = "#888888"


def step_order(label: str, baseline_label: str = "Original") -> int:
    """Order intermediate steps numerically; baseline first, 'final' last.

    The baseline (the un-trained/starting model) is placed at the very left of
    the curve so the trajectory starts from the beginning.
    """
    text = str(label).replace("step_", "")
    if text == baseline_label:
        return -1
    if text == "final":
        return 10**9
    try:
        return int(text)
    except ValueError:
        return 10**9


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--baseline-yield", type=float, default=None)
    parser.add_argument("--baseline-dock", type=float, default=None)
    parser.add_argument("--baseline-qed", type=float, default=None)
    parser.add_argument("--baseline-label", default="Original")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.metrics)
    df["step"] = df["label"].map(lambda v: step_order(v, args.baseline_label))
    df = df.sort_values("step")

    steps = df["step"].tolist()
    tick_labels = []
    for s in steps:
        if s < 0:
            tick_labels.append(args.baseline_label)
        elif s >= 10**9:
            tick_labels.append("final")
        else:
            tick_labels.append(f"{int(s)//1000}k")
    yield_pct = df["reasonable_yield_per_attempt_pct"].tolist()
    dock = df["dock_median_reasonable"].tolist()
    qed = df["qed_median_reasonable"].tolist()

    # Best step selection follows the project success criterion:
    #   "raise the reasonable-molecule yield while keeping quality comparable".
    # Quality gate first (QED must stay >= 85% of the baseline), then pick the
    # highest yield among the steps that pass the gate.
    yield_arr = np.asarray(yield_pct, dtype=float)
    qed_arr = np.asarray(qed, dtype=float)
    step_arr = np.asarray(steps, dtype=float)
    # The baseline itself is not a candidate stopping point.
    is_train_step = step_arr >= 0
    if args.baseline_qed is not None:
        gate = args.baseline_qed * 0.85
        eligible = (qed_arr >= gate) & is_train_step
    else:
        eligible = (yield_arr >= 0.90 * np.nanmax(yield_arr)) & is_train_step
    if eligible.any():
        best_idx = int(np.nanargmax(np.where(eligible, yield_arr, np.nan)))
        best_step = steps[best_idx]
        best_label = tick_labels[best_idx]
    else:
        best_step = None
        best_label = None

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    base_mask = [s < 0 for s in steps]

    def _mark_baseline(ax, values, color):
        """Highlight the baseline point (start of training) with a distinct marker."""
        if not any(base_mask):
            return
        bx = [s for s, m in zip(steps, base_mask) if m]
        by = [v for v, m in zip(values, base_mask) if m]
        ax.scatter(bx, by, marker="s", s=150, color=BASELINE_GRAY, zorder=6,
                   label=f"{args.baseline_label} (start)")

    # --- yield ---
    ax = axes[0]
    ax.plot(steps, yield_pct, "o-", color=YIELD_COLOR, linewidth=2.4, markersize=9)
    _mark_baseline(ax, yield_pct, YIELD_COLOR)
    if args.baseline_yield is not None:
        ax.axhline(
            args.baseline_yield,
            color=BASELINE_GRAY,
            linestyle="--",
            linewidth=1.8,
            alpha=0.9,
            label=f"{args.baseline_label} {args.baseline_yield:.1f}%",
        )
    if best_step is not None:
        ax.scatter([best_step], [yield_pct[best_idx]], s=260, facecolors="none",
                   edgecolors="black", linewidths=2.0, zorder=5)
        ax.annotate(f"best step {best_label}", (best_step, yield_pct[best_idx]),
                    textcoords="offset points", xytext=(0, 16), ha="center", fontweight="bold")
    ax.set_title("Reasonable-molecule yield (R1-R5 clean)")
    ax.set_ylabel("% of 1000 attempts")
    ax.set_xlabel("DPO update step")
    ax.set_xticks(steps)
    ax.set_xticklabels(tick_labels)
    ax.legend(loc="lower right", frameon=False)

    # --- docking ---
    ax = axes[1]
    ax.plot(steps, dock, "s-", color=DOCK_COLOR, linewidth=2.2, markersize=8)
    _mark_baseline(ax, dock, DOCK_COLOR)
    if args.baseline_dock is not None:
        ax.axhline(
            args.baseline_dock,
            color=BASELINE_GRAY,
            linestyle="--",
            linewidth=1.8,
            alpha=0.9,
            label=f"{args.baseline_label} {args.baseline_dock:.2f}",
        )
    if best_step is not None:
        ax.scatter([best_step], [dock[best_idx]], s=260, facecolors="none",
                   edgecolors="black", linewidths=2.0, zorder=5)
    ax.set_title("Docking median of reasonable molecules\n(lower = better)")
    ax.set_ylabel("kcal/mol")
    ax.set_xlabel("DPO update step")
    ax.set_xticks(steps)
    ax.set_xticklabels(tick_labels)
    ax.legend(loc="upper right", frameon=False)

    # --- QED ---
    ax = axes[2]
    ax.plot(steps, qed, "^-", color=QED_COLOR, linewidth=2.2, markersize=8)
    _mark_baseline(ax, qed, QED_COLOR)
    if args.baseline_qed is not None:
        ax.axhline(
            args.baseline_qed,
            color=BASELINE_GRAY,
            linestyle="--",
            linewidth=1.8,
            alpha=0.9,
            label=f"{args.baseline_label} {args.baseline_qed:.3f}",
        )
    if best_step is not None:
        ax.scatter([best_step], [qed[best_idx]], s=260, facecolors="none",
                   edgecolors="black", linewidths=2.0, zorder=5)
    ax.set_title("QED median of reasonable molecules")
    ax.set_ylabel("QED")
    ax.set_xlabel("DPO update step")
    ax.set_xticks(steps)
    ax.set_xticklabels(tick_labels)
    ax.legend(loc="upper right", frameon=False)

    fig.suptitle(
        "ESRPO training-step trade-off: yield rises, quality declines after the sweet spot",
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout()
    out_path = out_dir / "step_tradeoff_curve.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if best_step is not None:
        print(
            f"best step = {best_label} (step={best_step}) "
            f"yield={yield_pct[best_idx]:.1f}% dock={dock[best_idx]:.2f} qed={qed[best_idx]:.3f}"
        )
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
