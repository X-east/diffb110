#!/usr/bin/env python
"""SYNTHETIC vs real R1-R4 trajectory comparison plot.

Both curves are recomputed HERE from source with ONE consistent definition
(all parseable rows of each epoch's molecules.csv -- no docking_success filter),
so the comparison cannot be distorted by a denominator mismatch.

The output carries an explicit SYNTHETIC banner; see README.md.

Follows .codebuddy/rules/可视化作图规范.mdc.
"""
from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path("/home/md666/data/disk6/DiffB110")
sys.path.insert(0, str(ROOT))
from scripts.offline_rl.structure_pathology_rules import (  # noqa: E402
    compute_structure_pathology_from_smiles as PATHOLOGY,
)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.edgecolor"] = "#d4d4d4"
plt.rcParams["axes.linewidth"] = 1.2

HERE = Path(__file__).resolve().parent
REAL_RUN = ROOT / "runs/diffgui-deppa-full-lr1e5-wd1e4-b8x8-t200-100ep_20260626_165809"

C_REAL = "#3498DB"
C_SUB = "#E67E22"
C_R4C = "#2ECC71"
C_REF = "#E74C3C"


def profile(smiles_iter) -> tuple[int, int, int, int]:
    """Return (n, n_r14, n_r4b, n_r4c) -- one definition for both sides."""
    n = a = b = c = 0
    for s in smiles_iter:
        f = PATHOLOGY(s)
        if f is None:
            continue
        n += 1
        a += int(f.structural_violation_count) > 0
        b += int(f.r4b_three_ring_over_attached_violation)
        c += int(f.r4c_three_ring_spiro_or_fused_violation)
    return n, a, b, c


def curve(directory: Path, pattern: str) -> pd.DataFrame:
    rows = []
    for p in sorted(glob.glob(str(directory / pattern))):
        ep = int(re.search(r"epoch_(\d+)", p).group(1))
        d = pd.read_csv(p, low_memory=False)
        smis = d["canonical_smiles"].fillna("").astype(str)
        n, a, b, c = profile(smis)
        if n == 0:
            continue
        rec = {"ep": ep, "n": n, "r14": a / n * 100, "r4b": b, "r4c": c}
        for col in ("docking_raw", "qed_raw"):
            v = pd.to_numeric(d.get(col), errors="coerce").dropna()
            rec[col] = float(v.mean()) if len(v) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    real = curve(REAL_RUN / "epochs", "epoch_*/molecules.csv")
    sub = curve(HERE / "epochs", "epoch_*_molecules.csv")
    print(f"real epochs={len(real)}  sub epochs={len(sub)}")
    print(f"real  R1-R4 mean={real.r14.mean():.2f}% SD={real.r14.std():.2f}pp")
    print(f"sub R1-R4 mean={sub.r14.mean():.2f}% SD={sub.r14.std():.2f}pp")
    print(f"R4b max |diff| = {np.abs(real.r4b.values - sub.r4b.values).max()}")
    print(f"R4c max |diff| = {np.abs(real.r4c.values - sub.r4c.values).max()}")

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.4))
    (a1, a2), (a3, a4) = axes

    # ---- panel 1: aggregate R1-R4 -----------------------------------------
    ax = a1
    ax.plot(real.ep, real.r14, color=C_REAL, lw=1.9, marker="o", ms=2.6,
            label="Real DiffGui RL (measured)")
    ax.plot(sub.ep, sub.r14, color=C_SUB, lw=1.5, ls="--", marker="s",
            ms=2.6, alpha=0.95, label="Substituted counterfactual")
    tgt = sub.r14.mean()
    ax.axhline(tgt, color=C_REF, ls="--", lw=2.0, alpha=0.85, zorder=4)
    ax.text(real.ep.min(), tgt + 1.0, f"target mean: {tgt:.1f}%",
            color=C_REF, fontweight="bold", fontsize=9)
    ax.set_xlabel("epoch")
    ax.set_ylabel("R1-R4 violation rate (%)")
    ax.set_title("Aggregate R1-R4: trend removed, fluctuation kept")
    ax.set_ylim(min(real.r14.min(), sub.r14.min()) - 4,
                max(real.r14.max(), sub.r14.max()) + 4)
    ax.legend(frameon=True, fontsize=9, loc="lower left")
    sns.despine(ax=ax, top=True, right=True)

    # ---- panel 2: R4b / R4c counts (identical by construction) -------------
    ax = a2
    ax.plot(real.ep, real.r4c, color=C_R4C, lw=3.4, alpha=0.45,
            label="R4c real (thick)")
    ax.plot(sub.ep, sub.r4c, color=C_R4C, lw=1.2, ls=(0, (1, 1.6)),
            label="R4c substituted (dotted)")
    ax.plot(real.ep, real.r4b, color=C_REAL, lw=3.4, alpha=0.45,
            label="R4b real (thick)")
    ax.plot(sub.ep, sub.r4b, color=C_SUB, lw=1.2, ls=(0, (1, 1.6)),
            label="R4b substituted (dotted)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("violating molecule count")
    ax.set_title("R4b / R4c counts: dotted lies exactly on thick "
                 "(max |diff| = %d)" % int(np.abs(real.r4b.values - sub.r4b.values).max()))
    ax.legend(frameon=True, fontsize=8.5, ncol=2, loc="upper left")
    sns.despine(ax=ax, top=True, right=True)

    # ---- panel 3: docking --------------------------------------------------
    ax = a3
    ax.plot(real.ep, real.docking_raw, color=C_REAL, lw=1.7, marker="o", ms=2.4,
            label=f"Real (mean {real.docking_raw.mean():.3f})")
    ax.plot(sub.ep, sub.docking_raw, color=C_SUB, lw=1.3, ls="--",
            marker="s", ms=2.4, alpha=0.95,
            label=f"Substituted (mean {sub.docking_raw.mean():.3f})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean docking_raw (kcal/mol)")
    ax.set_title("Docking: unchanged (mean |shift| = %.4f kcal/mol)"
                 % abs(sub.docking_raw.mean() - real.docking_raw.mean()))
    ax.legend(frameon=True, fontsize=9, loc="lower left")
    sns.despine(ax=ax, top=True, right=True)

    # ---- panel 4: QED ------------------------------------------------------
    ax = a4
    ax.plot(real.ep, real.qed_raw, color=C_REAL, lw=1.7, marker="o", ms=2.4,
            label=f"Real (mean {real.qed_raw.mean():.4f})")
    ax.plot(sub.ep, sub.qed_raw, color=C_SUB, lw=1.3, ls="--",
            marker="s", ms=2.4, alpha=0.95,
            label=f"Substituted (mean {sub.qed_raw.mean():.4f})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean qed_raw")
    ax.set_title("QED: unchanged (mean |shift| = %.4f)"
                 % abs(sub.qed_raw.mean() - real.qed_raw.mean()))
    ax.legend(frameon=True, fontsize=9, loc="lower left")
    sns.despine(ax=ax, top=True, right=True)

    fig.suptitle("SYNTHETIC COUNTERFACTUAL - NOT A MODEL RESULT",
                 color=C_REF, fontweight="bold", fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    out = HERE / "compare_real_vs_substituted.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {out}")


if __name__ == "__main__":
    main()
