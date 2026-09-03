#!/usr/bin/env python
"""Plot R1-R5 and Brenk violation rates against the ChEMBL reference.

The purpose is to show WHERE the E2PO mainline lands relative to a real
drug-like reference, instead of only comparing against the Original baseline.
ChEMBL rates come from two in-project measurements that share the same rule
implementation and the same BRENK FilterCatalog definition used by
``evaluate_reasonable_molecules.py``:

- R1-R5: ``analysis/diffsbdd_structure_pathology_260831/
  chembl_structure_pathology_summary.json`` (ChEMBL36, n = 1,550,090)
- Brenk: ``analysis/diffsbdd_global_dpo_t1000_v1/formal_5epoch_native_joint_1000gen/
  evaluation/chembl_quality/indicator_rates_vs_chembl.csv``
  (ChEMBL36-standardized-all, n = 2,734,034, ``brenk_alert_count > 0``)

Model rates are taken from ``reasonable_molecules_metrics.csv`` in this
directory (denominator = relaxed molecules, largest fragment).
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.edgecolor"] = "#d4d4d4"
plt.rcParams["axes.linewidth"] = 1.2

HERE = Path(__file__).resolve().parent
METRICS_CSV = HERE / "reasonable_molecules_metrics.csv"

# (column in CSV, display label, ChEMBL reference rate in %)
INDICATORS = [
    ("r1_large_ring_violation_rate_of_relaxed_pct", "R1 large ring", 4.11),
    ("r2_ring_system_violation_rate_of_relaxed_pct", "R2 ring system", 3.32),
    ("r3_four_ring_violation_rate_of_relaxed_pct", "R3 four ring", 2.47),
    ("r4a_three_ring_hetero_violation_rate_of_relaxed_pct", "R4a 3-ring hetero", 0.33),
    ("r4b_three_ring_over_attached_violation_rate_of_relaxed_pct", "R4b 3-ring over-att", 1.11),
    ("r4c_three_ring_spiro_or_fused_violation_rate_of_relaxed_pct", "R4c 3-ring spiro/fused", 0.65),
    ("r5_carbon_fraction_violation_rate_of_relaxed_pct", "R5 carbon fraction", 6.26),
    ("brenk_alert_rate_of_relaxed_pct", "Brenk alert", 39.32),
]

SERIES = [
    ("Original", "#3498DB"),
    ("step_004000", "#E74C3C"),
    ("step_final", "#F39C12"),
]
SERIES_LABEL = {
    "Original": "Original (baseline)",
    "step_004000": "E2PO step_4000",
    "step_final": "E2PO final",
}
CHEMBL_COLOR = "#2ECC71"
CHEMBL_CLEAN_RATE = 84.72


def load_rates() -> dict[str, dict[str, float]]:
    """Read per-label indicator rates from the mainline metrics CSV."""
    with METRICS_CSV.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_label = {row["label"]: row for row in rows}
    rates: dict[str, dict[str, float]] = {}
    for label in ("Original", "step_004000", "step_final"):
        row = by_label[label]
        rates[label] = {
            column: float(row[column]) for column, _, _ in INDICATORS
        }
    return rates


def main() -> None:
    """Render the grouped bar chart and a clean-rate companion panel."""
    rates = load_rates()

    fig, (ax_rules, ax_clean) = plt.subplots(1, 2, figsize=(16, 6.2))

    labels = [label for _, label, _ in INDICATORS]
    x = np.arange(len(labels), dtype=float)
    width = 0.20

    ax_rules.bar(
        x - 1.5 * width,
        [chembl for _, _, chembl in INDICATORS],
        width,
        label="ChEMBL36 reference",
        color=CHEMBL_COLOR,
        edgecolor="white",
        linewidth=0.8,
    )
    for offset, (label, color) in zip((-0.5, 0.5, 1.5), SERIES):
        values = [rates[label][column] for column, _, _ in INDICATORS]
        ax_rules.bar(
            x + offset * width,
            values,
            width,
            label=SERIES_LABEL[label],
            color=color,
            edgecolor="white",
            linewidth=0.8,
        )

    ax_rules.set_xticks(x)
    ax_rules.set_xticklabels(labels, rotation=32, ha="right", fontsize=11)
    ax_rules.set_ylabel("Violation rate (%)", fontsize=13)
    ax_rules.set_title(
        "Per-rule violation rates vs ChEMBL reference", fontsize=15, fontweight="bold"
    )
    ax_rules.legend(fontsize=11, loc="upper right", frameon=True)
    ax_rules.tick_params(axis="y", labelsize=11)
    ax_rules.grid(axis="x", visible=False)

    # Right panel: reasonable-molecule rate against the ChEMBL clean rate.
    with METRICS_CSV.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    order = [row["label"] for row in rows]
    clean_rates = [
        float(row["reasonable_pct_of_relaxed"])
        for row in rows
        if row["label"] != "Original"
    ]
    order_no_base = [label for label in order if label != "Original"]
    full_labels = ["ChEMBL36", "Original"] + [
        label.replace("step_", "step ").replace("_", "") for label in order_no_base
    ]
    full_values = (
        [CHEMBL_CLEAN_RATE]
        + [
            float(row["reasonable_pct_of_relaxed"])
            for row in rows
            if row["label"] == "Original"
        ]
        + clean_rates
    )
    colors = [CHEMBL_COLOR, "#3498DB"] + [
        "#E74C3C" if label == "step_004000" else "#95A5A6" for label in order_no_base
    ]
    bars = ax_clean.bar(
        np.arange(len(full_values)), full_values, color=colors, edgecolor="white", linewidth=0.8
    )
    ax_clean.axhline(
        CHEMBL_CLEAN_RATE,
        color="#2ECC71",
        linestyle="--",
        linewidth=2.0,
        alpha=0.85,
        zorder=4,
    )
    ax_clean.text(
        0.1,
        CHEMBL_CLEAN_RATE + 1.5,
        f"ChEMBL clean: {CHEMBL_CLEAN_RATE:.2f}%",
        color="#2E8B57",
        fontweight="bold",
        fontsize=11,
    )
    ax_clean.set_xticks(np.arange(len(full_values)))
    ax_clean.set_xticklabels(full_labels, rotation=45, ha="right", fontsize=10)
    ax_clean.set_ylabel("Reasonable molecules (% of relaxed)", fontsize=13)
    ax_clean.set_title(
        "Reasonable-molecule rate vs ChEMBL clean rate", fontsize=15, fontweight="bold"
    )
    ax_clean.tick_params(axis="y", labelsize=11)
    ax_clean.grid(axis="x", visible=False)
    for bar, value in zip(bars, full_values):
        ax_clean.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.8,
            f"{value:.1f}",
            ha="center",
            fontsize=9,
        )

    fig.tight_layout()
    out_path = HERE / "rules_vs_chembl_reference.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
