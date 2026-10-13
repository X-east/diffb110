"""Re-plot Figure 3.2: R1-R5 violation rates, 500-step benchmark, 3 repeats.

Reason: the thesis table 3.4 was recomputed on the 3-repeat protocol
(rep4/rep5/rep6, matching the 1000-step formal benchmark) instead of the
5-repeat robustness subset. This script regenerates the figure from the
matching CSV so that figure and table share one data source.

Input : model_r1r5_rates_500steps_3rep.csv   (from compute_r1r5_500steps.py)
Output: docs/images/fig3_3_model_pathology.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

HERE = Path(__file__).resolve().parent
OUT = Path("/home/md666/data/disk6/DiffB110/docs/images/fig3_3_model_pathology.png")

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.edgecolor"] = "#d4d4d4"
plt.rcParams["axes.linewidth"] = 1.2

MODEL_ORDER = ["GCDM-SBDD", "TargetDiff", "DiffSBDD", "DiffGui", "PMDM"]

RULES: tuple[tuple[str, str], ...] = (
    ("R1 large ring", "r1_large_ring_violation"),
    ("R2 ring system", "r2_ring_system_violation"),
    ("R3 four ring", "r3_four_ring_violation"),
    ("R4 any (3-ring)", "r1r4_any"),
    ("R5 carbon frac", "r5_carbon_fraction_violation"),
    ("R1-R5 any", "r1r5_any"),
)


def style_axes(ax: plt.Axes) -> None:
    """Hide top/right spines and soften the remaining ones."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")


def main() -> None:
    """Render the grouped bar chart for the 3-repeat 500-step benchmark."""
    df = pd.read_csv(HERE / "model_r1r5_rates_500steps_3rep.csv")
    df = df.set_index("model").loc[MODEL_ORDER]

    n = len(RULES)
    x = np.arange(len(MODEL_ORDER))
    width = 0.13
    palette = sns.color_palette("Blues", n)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    for i, (label, column) in enumerate(RULES):
        offset = (i - (n - 1) / 2) * width
        vals = df[column].values * 100
        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=label,
            color=palette[i],
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, value in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.6,
                f"{value:.0f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, fontsize=12)
    ax.set_ylabel("Violation rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title(
        "R1-R5 structural pathology violation rates (500-step generation, 3 repeats)"
    )
    ax.legend(title="Rule", ncol=3, fontsize=9, loc="upper right")
    style_axes(ax)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300)
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
