"""Generate paper figures for the model-selection / step-sensitivity narrative.

Outputs (saved into the same directory as this script, then copied to the
work_records folder by the caller):
- Fig4_r1r5_grouped_bar.png   : R1-R5 violation rates per model (1000-step, 5-rep)
- Fig5_selection_funnel.png   : 5 -> 2 -> 1 two-stage selection funnel
- Fig11_steps_500_vs_1000.png : step-sensitivity comparison (500 vs 1000 steps)

Data sources (absolute paths to the self-contained work_records CSVs):
- model_selection_summary.csv         (Fig 4, per-valid R1-R5, 5-rep)
- steps_500_vs_1000_comparison.csv    (Fig 11, per-attempt + docking, 500/1000)

Pure analysis/plotting; no model/training code changes.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.edgecolor"] = "#d4d4d4"
plt.rcParams["axes.linewidth"] = 1.2

HERE = Path(__file__).resolve().parent
WR = Path("/home/md666/data/disk6/DiffB110/work_records/26-9/260903")

# Model color palette (consistent with project spec where defined)
MODEL_COLORS = {
    "DiffSBDD": "#E74C3C",   # red
    "DiffGui": "#3498DB",    # blue
    "TargetDiff": "#2ECC71", # green
    "PMDM": "#F39C12",       # orange
    "GCDM-SBDD": "#9B59B6",  # purple
}

MODEL_ORDER = ["GCDM-SBDD", "DiffSBDD", "TargetDiff", "DiffGui", "PMDM"]


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")


def fig4_r1r5(path_summary: Path):
    df = pd.read_csv(path_summary)
    df = df.set_index("model").loc[MODEL_ORDER]

    rules = [
        ("R1 large ring", "r1_large_ring_violation"),
        ("R2 ring system", "r2_ring_system_violation"),
        ("R3 four ring", "r3_four_ring_violation"),
        ("R4 any (3-ring)", "r1r4_any"),
        ("R5 carbon frac", "r5_carbon_fraction_violation"),
        ("R1-R5 any", "r1r5_any"),
    ]
    n = len(rules)
    x = np.arange(len(MODEL_ORDER))
    width = 0.13
    palette = sns.color_palette("Blues", n)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    for i, (label, col) in enumerate(rules):
        offset = (i - (n - 1) / 2) * width
        vals = df[col].values * 100
        bars = ax.bar(x + offset, vals, width, label=label, color=palette[i],
                      edgecolor="white", linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=7, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, fontsize=12)
    ax.set_ylabel("Violation rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("R1-R5 structural pathology violation rates (1000-step generation, 5 repeats)")
    ax.legend(title="Rule", ncol=3, fontsize=9, loc="upper right")
    style_axes(ax)
    fig.tight_layout()
    out = HERE / "Fig4_r1r5_grouped_bar.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Wrote {out}")
    return out


def fig5_funnel(path_summary: Path):
    """Two-stage selection funnel: 5 -> 2 -> 1."""
    df = pd.read_csv(path_summary).set_index("model")

    def fmt(model):
        r1r5 = df.loc[model, "r1r5_any"] * 100
        r2 = df.loc[model, "r2_ring_system_violation"] * 100
        r5 = df.loc[model, "r5_carbon_fraction_violation"] * 100
        return f"{model}\n(R1-R5 {r1r5:.0f}%, R2 {r2:.0f}%, R5 {r5:.0f}%)"

    stage0 = [fmt(m) for m in MODEL_ORDER]
    stage1 = [fmt(m) for m in ["DiffSBDD", "TargetDiff"]]
    stage2 = [fmt("DiffSBDD")]

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    ax.axis("off")

    # funnel rows: y from top to bottom
    rows = [
        (0.78, 5, stage0, "#BDC3C7", "Stage 0: five candidate generators"),
        (0.52, 2, stage1, "#7FB3D5", "Stage 1 filter: remove DiffGui & PMDM\n(highest R1-R5 pathology 79-94%; unstable structure)"),
        (0.26, 1, stage2, "#E74C3C", "Stage 2 filter: remove TargetDiff\n(pathology explodes 20%->64% at 1000 steps; step-unstable)"),
    ]
    for y, count, labels, color, title in rows:
        # centered band
        left = 0.5 - count * 0.06
        width = count * 0.12
        ax.add_patch(plt.Rectangle((left, y - 0.07), width, 0.14,
                                   facecolor=color, alpha=0.85, edgecolor="white"))
        # place model chips as text inside the band
        for j, lab in enumerate(labels):
            cx = 0.5 + (j - (count - 1) / 2) * 0.12
            ax.text(cx, y, lab, ha="center", va="center", fontsize=10,
                    color="white", fontweight="bold")
        ax.text(0.02, y, title, ha="left", va="center", fontsize=10.5,
                fontweight="bold", color="#2C3E50")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Model selection funnel: 5 -> 2 -> 1", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = HERE / "Fig5_selection_funnel.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Wrote {out}")
    return out


def fig11_step_sensitivity(path_cmp: Path):
    raw = pd.read_csv(path_cmp)
    # long format: one row per (model, step); pivot to wide per metric
    metrics = [
        ("Valid rate (%)", "valid%", 0, 100),
        ("R1-R4 pathology (%)", "r1r4_any_%", 0, 100),
        ("Docking mean (kcal/mol)", "dock", -12, -6),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)
    x = np.arange(len(MODEL_ORDER))
    width = 0.38
    for ax, (title, col, lo, hi) in zip(axes, metrics):
        v500 = []
        v1000 = []
        for m in MODEL_ORDER:
            sub = raw[(raw["model"] == m)]
            v500.append(float(sub[sub["step"] == 500][col].iloc[0]))
            v1000.append(float(sub[sub["step"] == 1000][col].iloc[0]))
        v500 = np.array(v500)
        v1000 = np.array(v1000)
        b1 = ax.bar(x - width / 2, v500, width, label="500 steps", color="#95A5A6")
        b2 = ax.bar(x + width / 2, v1000, width, label="1000 steps", color="#2C3E50")
        for bars in (b1, b2):
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + (hi - lo) * 0.01,
                        f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_ORDER, rotation=20, ha="right", fontsize=10)
        ax.set_title(title)
        ax.set_ylim(lo, hi)
        style_axes(ax)
        ax.legend(fontsize=9)
    fig.suptitle("Step-count sensitivity: 500 vs 1000 diffusion steps (per-attempt metrics)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = HERE / "Fig11_steps_500_vs_1000.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Wrote {out}")
    return out


def main():
    p_summary = WR / "model_selection_summary.csv"
    p_cmp = WR / "steps_500_vs_1000_comparison.csv"
    fig4_r1r5(p_summary)
    fig5_funnel(p_summary)
    fig11_step_sensitivity(p_cmp)


if __name__ == "__main__":
    main()
