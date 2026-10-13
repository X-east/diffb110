#!/usr/bin/env python
"""Compare R1-R5 pathology rates between the old and new ChEMBL references.

Old reference : data/chembl/chembl_36_molecules.csv.gz        (n = 1,550,090)
New reference : data/chembl_smallmol_reference/reference_smallmol.csv.gz (n = 1,790,085)

Reads the two ``*_r1r5.json`` sidecars produced by ``compute_reference_r1r5.py``
and renders one two-panel figure:
  (A) per-rule violation rate, old vs new
  (B) R1-R4 / R1-R5 clean rate, old vs new
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

HERE = Path(__file__).resolve().parent

OLD_JSON = HERE / "old_reference_crosscheck_r1r5.json"
NEW_JSON = HERE / "new_smallmol_reference_r1r5.json"
OUT_PNG = HERE / "reference_r1r5_compare.png"
OUT_SVG = HERE / "reference_r1r5_compare.svg"

RULE_ORDER = [
    ("r1_large_ring_violation", "R1 large ring\n(>=7)"),
    ("r2_ring_system_violation", "R2 ring system\n(>3)"),
    ("r3_four_ring_violation", "R3 four-\nmembered"),
    ("r4a_three_ring_hetero_violation", "R4a 3-ring\nhetero"),
    ("r4b_three_ring_over_attached_violation", "R4b 3-ring\nover-attached"),
    ("r4c_three_ring_spiro_or_fused_violation", "R4c 3-ring\nspiro/fused"),
    ("r5_carbon_fraction_violation", "R5 carbon\nfraction >0.85"),
]

COLOR_OLD = "#95A5A6"
COLOR_NEW = "#E74C3C"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    """Render and save the old-vs-new R1-R5 comparison figure."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.35)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.edgecolor"] = "#d4d4d4"
    plt.rcParams["axes.linewidth"] = 1.2

    old = _load(OLD_JSON)
    new = _load(NEW_JSON)

    labels = [label for _, label in RULE_ORDER]
    old_rates = [old["rule_rates"][key] * 100 for key, _ in RULE_ORDER]
    new_rates = [new["rule_rates"][key] * 100 for key, _ in RULE_ORDER]

    x = np.arange(len(RULE_ORDER))
    width = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.4),
                             gridspec_kw={"width_ratios": [2.0, 1.0]})

    # ---- panel A: per-rule violation rate ---------------------------------
    ax = axes[0]
    b1 = ax.bar(x - width / 2, old_rates, width, label="Old reference (n=1,550,090)",
                color=COLOR_OLD, edgecolor="white", linewidth=1.0)
    b2 = ax.bar(x + width / 2, new_rates, width,
                label="New small-mol reference (n=1,790,085)",
                color=COLOR_NEW, edgecolor="white", linewidth=1.0)
    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.annotate(f"{h:.2f}%", (rect.get_x() + rect.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=11, fontweight="bold",
                        xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Violation rate (%)", fontsize=14, fontweight="bold")
    ax.set_title("(A) Per-rule R1-R5 violation rate", fontsize=16, fontweight="bold")
    ax.legend(fontsize=12, frameon=True)
    ax.set_ylim(0, max(max(old_rates), max(new_rates)) * 1.22)
    sns.despine(ax=ax, top=True, right=True)

    # ---- panel B: clean rate ----------------------------------------------
    ax2 = axes[1]
    agg_keys = [("r1_r4_clean", "R1-R4 clean\n(theoretical ceiling)"),
                ("r1_r5_clean", "R1-R5 clean\n(all rules)")]
    old_clean = [old["aggregates"][k]["rate"] * 100 for k, _ in agg_keys]
    new_clean = [new["aggregates"][k]["rate"] * 100 for k, _ in agg_keys]
    x2 = np.arange(len(agg_keys))
    b3 = ax2.bar(x2 - width / 2, old_clean, width, color=COLOR_OLD,
                 edgecolor="white", linewidth=1.0)
    b4 = ax2.bar(x2 + width / 2, new_clean, width, color=COLOR_NEW,
                 edgecolor="white", linewidth=1.0)
    for bars in (b3, b4):
        for rect in bars:
            h = rect.get_height()
            ax2.annotate(f"{h:.2f}%", (rect.get_x() + rect.get_width() / 2, h),
                         ha="center", va="top", fontsize=12, fontweight="bold",
                         color="white", xytext=(0, -6), textcoords="offset points")
    ax2.axhline(84.7235, color="#2ECC71", linestyle="--", linewidth=2.0, alpha=0.85,
                zorder=4)
    ax2.text(1.47, 85.0, "old headline\n84.72%", color="#2ECC71",
             fontweight="bold", fontsize=11, va="bottom", ha="right")
    ax2.set_xticks(x2)
    ax2.set_xticklabels([l for _, l in agg_keys], fontsize=12)
    ax2.set_ylabel("Clean rate (%)", fontsize=14, fontweight="bold")
    ax2.set_title("(B) Structural-compliance rate", fontsize=16, fontweight="bold")
    ax2.set_ylim(0, 100)
    sns.despine(ax=ax2, top=True, right=True)

    fig.suptitle("ChEMBL R1-R5 reference: old (size-windowed) vs new (small-molecule, no size window)",
                 fontsize=17, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_SVG, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_SVG}")

    print("\nold R1-R5 clean: %.4f%%  new R1-R5 clean: %.4f%%"
          % (old["aggregates"]["r1_r5_clean"]["rate"] * 100,
             new["aggregates"]["r1_r5_clean"]["rate"] * 100))


if __name__ == "__main__":
    main()
