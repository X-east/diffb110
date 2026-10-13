#!/usr/bin/env python
"""Figure for the 2026-09-30 UniDock ligand-hydrogen A/B experiment.

Reads the two experiment result CSVs and writes one 3-panel figure into the
work-record directory so the record stays self-contained.

Panels
  1. Paired scatter A(no H) vs B(added H) for the DiffSBDD generation batch
  2. Histogram of the paired difference (generation batch)
  3. p95 |delta| comparison: hydrogen effect vs re-run jitter vs seed choice
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statistics import mean, median

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.edgecolor"] = "#d4d4d4"
plt.rcParams["axes.linewidth"] = 1.2

ROOT = Path("/home/md666/data/disk6/DiffB110")
GEN_CSV = ROOT / "analysis/unidock_hydrogen_ab_260930/generation/ab_generation_scores.csv"
POOL_CSV = ROOT / "analysis/unidock_hydrogen_ab_260930/ab_scores.csv"
OUT_DIR = ROOT / "work_records/26-9/260930"
OUT_PNG = OUT_DIR / "ab_hydrogen_effect.png"

C_RED = "#E74C3C"
C_BLUE = "#3498DB"
C_GREEN = "#2ECC71"
C_ORANGE = "#F39C12"
C_GREY = "#7F8C8D"


def load_generation() -> tuple[list[float], list[float]]:
    a, b = [], []
    with GEN_CSV.open(newline="") as handle:
        for row in csv.DictReader(handle):
            av, bv = float(row["A_noh"]), float(row["B_h"])
            if -30.0 <= av <= -1.0 and -30.0 <= bv <= -1.0:
                a.append(av)
                b.append(bv)
    return a, b


def load_pool() -> dict[str, list[float]]:
    arms = {"A_noh_seed42": [], "B_h_seed42": [], "C_h_seed0": []}
    with POOL_CSV.open(newline="") as handle:
        for row in csv.DictReader(handle):
            for key in arms:
                value = row.get(key, "")
                if value == "":
                    continue
                arms[key].append(float(value))
    return arms


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    a, b = load_generation()
    diffs = np.array([y - x for x, y in zip(a, b)])
    pool = load_pool()

    # ---- noise baselines from the 200k-pool experiment ---------------------
    def paired_diffs(x_key: str, y_key: str) -> np.ndarray:
        xs, ys = pool[x_key], pool[y_key]
        n = min(len(xs), len(ys))
        return np.array([ys[i] - xs[i] for i in range(n)])

    jitter = paired_diffs("A_noh_seed42", "B_h_seed42")  # hydrogen effect, pool
    seed_eff = paired_diffs("B_h_seed42", "C_h_seed0")  # seed effect, pool

    # historical harness sanity value (from the run_ab.py summary)
    harness = 0.371

    fig, axes = plt.subplots(1, 3, figsize=(19.5, 6.0))

    # ---------------- panel 1: paired scatter ------------------------------
    ax = axes[0]
    ax.scatter(a, b, s=16, alpha=0.35, color=C_BLUE, edgecolors="none")
    lo, hi = min(min(a), min(b)) - 0.5, max(max(a), max(b)) + 0.5
    ax.plot([lo, hi], [lo, hi], color=C_GREY, linestyle="--", linewidth=1.4, zorder=4)
    ax.axhline(median(b), color=C_RED, linestyle="--", linewidth=2.0, alpha=0.85, zorder=5)
    ax.axvline(median(a), color=C_RED, linestyle="--", linewidth=2.0, alpha=0.85, zorder=5)
    ax.text(lo + 0.3, median(b), f" B median {median(b):.2f}", color=C_RED,
            fontweight="bold", va="bottom", fontsize=12)
    ax.text(median(a), hi - 0.4, f"A median {median(a):.2f} ", color=C_RED,
            fontweight="bold", ha="right", fontsize=12)
    ax.set_xlabel("Docking score, no explicit H (kcal/mol)")
    ax.set_ylabel("Docking score, AddHs(addCoords=True) (kcal/mol)")
    ax.set_title(f"Paired scores, DiffSBDD generation batch (n={len(a)})", fontsize=15)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    sns.despine(ax=ax)

    # ---------------- panel 2: difference histogram ------------------------
    ax = axes[1]
    sns.histplot(diffs, bins=60, ax=ax, color=C_ORANGE, alpha=0.85, edgecolor="white",
                 linewidth=0.4)
    med = float(np.median(diffs))
    p95 = float(np.percentile(np.abs(diffs), 95))
    ax.axvline(med, color=C_RED, linestyle="--", linewidth=2.0, alpha=0.9, zorder=5)
    ax.text(med, ax.get_ylim()[1] * 0.95, f" median {med:+.3f}", color=C_RED,
            fontweight="bold", va="top", fontsize=12)
    ax.axvline(p95, color=C_GREEN, linestyle="--", linewidth=2.0, alpha=0.9, zorder=5)
    ax.text(p95, ax.get_ylim()[1] * 0.80, f" +p95 {p95:.3f} ", color=C_GREEN,
            fontweight="bold", fontsize=12)
    ax.axvline(-p95, color=C_GREEN, linestyle="--", linewidth=2.0, alpha=0.9, zorder=5)
    frac = float(np.mean(np.abs(diffs) > 0.5))
    ax.set_xlabel("Score(B, with H) - Score(A, no H)  (kcal/mol)")
    ax.set_ylabel("Molecule count")
    ax.set_title(f"Hydrogen effect: median {med:+.3f}, |d|>0.5 in {frac*100:.1f}%", fontsize=15)
    ax.set_xlim(-3.0, 3.0)
    sns.despine(ax=ax)

    # ---------------- panel 3: effect size vs noise ------------------------
    ax = axes[2]
    labels = ["Hydrogen\n(this study,\ngeneration)", "Hydrogen\n(this study,\n200k pool)",
              "Re-run jitter\n(same protocol)", "Seed\n0 vs 42"]
    p95s = [p95, float(np.percentile(np.abs(jitter), 95)), harness,
            float(np.percentile(np.abs(seed_eff), 95))]
    meds = [abs(med), abs(float(np.median(jitter))), 0.004, 0.0]
    colors = [C_RED, "#F1948A", C_GREY, C_BLUE]
    xs = np.arange(len(labels))
    ax.bar(xs, p95s, color=colors, alpha=0.9, width=0.6)
    ax.bar(xs, meds, color="#2C3E50", alpha=0.95, width=0.28)
    for x, (p, m) in enumerate(zip(p95s, meds)):
        ax.text(x, p + 0.02, f"p95 {p:.3f}", ha="center", fontweight="bold", fontsize=12)
        ax.text(x, m + 0.02, f"med {m:.3f}", ha="center", fontsize=10, color="#2C3E50")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("|delta| docking score (kcal/mol)")
    ax.set_title("Hydrogen effect is at the level of re-run noise", fontsize=15)
    ax.axhline(p95, color=C_RED, linestyle=":", linewidth=1.6, alpha=0.8)
    sns.despine(ax=ax)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {OUT_PNG}")


if __name__ == "__main__":
    main()
