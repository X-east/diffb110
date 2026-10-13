"""Plot bond-length / bond-angle geometry failure distribution: epoch 1 vs epoch 89.

The 200-step DiffGui DePPA run shows PB-valid collapsing from 91% (epoch 1) to
28% (epoch 89), driven by bond_lengths / bond_angles violations. This script
runs PoseBusters with `full_report=True` on those two epochs and plots the
per-molecule violation magnitudes (longest_bond_relative_length,
most_extreme_relative_angle) so the *severity* of geometric drift is visible,
not just pass/fail counts.

Outputs (next to this script):
    bond_geom_epoch1_vs_89.csv
    bond_geom_failure_dist.png
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from rdkit import Chem

import posebusters.posebusters as _pbmod
from posebusters import PoseBusters

# RDKit 2025 changed GetProp signature; monkeypatch PoseBusters' name reader.
def _get_name(self, mol_pred: Chem.Mol) -> str:  # type: ignore[no-untyped-def]
    try:
        return str(mol_pred.GetProp("_Name"))
    except Exception:
        return "mol"


_pbmod.PoseBusters._get_name = _get_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RUN_DIR = Path(
    "/home/md666/data/disk6/DiffB110/runs/diffgui-deppa-full-lr1e5-wd1e4-b8x8-t200-100ep_20260626_165809"
)
RECEPTOR = Path(
    "/home/md666/data/disk6/DiffB110/analysis/benchmark/8gjl_c1101_benchmark_20260603_500mol_500steps/inputs/8GJL_C1101_pocket8A.pdb"
)
OUT_DIR = Path(__file__).resolve().parent

EPOCHS = [1, 89]
COL_BL = "longest_bond_relative_length"      # >1 => a bond exceeds upper bound
COL_BA = "most_extreme_relative_angle"       # >1 => an angle violates bounds
COL_NOB = "number_outlier_bonds"
COL_NOA = "number_outlier_angles"


def _run_epoch(epoch: int) -> pd.DataFrame:
    ep_dir = RUN_DIR / f"epoch_{epoch:04d}"
    sdf = ep_dir / "molecules.sdf"
    sup = Chem.SDMolSupplier(str(sdf), removeHs=False)
    mols = [m for m in sup if m is not None]

    tmp = tempfile.NamedTemporaryFile(suffix=".sdf", delete=False)
    p = tmp.name
    tmp.close()
    w = Chem.SDWriter(p)
    names = []
    for i, m in enumerate(mols):
        mid = m.GetProp("mol_id") if m.HasProp("mol_id") else f"mol{i}"
        m.SetProp("_Name", mid)
        names.append(mid)
        w.write(m)
    w.close()

    buster = PoseBusters(config="dock")
    df = buster.bust(mol_pred=p, mol_cond=str(RECEPTOR), full_report=True)
    Path(p).unlink(missing_ok=True)

    df = df.reset_index(drop=True)
    df.insert(0, "epoch", epoch)
    df.insert(1, "mol_id", names[: len(df)])
    return df


def main() -> None:
    frames = []
    for ep in EPOCHS:
        logger.info("Running PoseBusters on epoch %d", ep)
        frames.append(_run_epoch(ep))
    data = pd.concat(frames, ignore_index=True)

    keep = ["epoch", "mol_id", "bond_lengths", "bond_angles", COL_BL, COL_BA,
            COL_NOB, COL_NOA, "mol_pred_loaded"]
    keep = [c for c in keep if c in data.columns]
    out = data[keep].copy()
    out.to_csv(OUT_DIR / "bond_geom_epoch1_vs_89.csv", index=False)
    logger.info("Wrote %s", OUT_DIR / "bond_geom_epoch1_vs_89.csv")

    # ---- plotting ----
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 1.2

    color_map = {1: "#3498DB", 89: "#E74C3C"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    metrics = [
        (COL_BL, "Bond length violation ratio\n(max distance / upper bound)", axes[0]),
        (COL_BA, "Bond angle violation ratio\n(most extreme relative angle)", axes[1]),
    ]
    for col, title, ax in metrics:
        order = EPOCHS
        sns.violinplot(
            data=out, x="epoch", y=col, order=order,
            palette=[color_map[e] for e in order], inner="quartile",
            ax=ax, linewidth=1.2,
        )
        sns.stripplot(
            data=out, x="epoch", y=col, order=order,
            color="black", alpha=0.25, size=3.5, ax=ax, jitter=True,
        )
        ax.axhline(1.0, color="#E74C3C", linestyle="--", linewidth=2.0, alpha=0.85, zorder=4)
        ax.text(0.02, 1.02, "valid threshold = 1.0", color="#E74C3C",
                fontsize=9, transform=ax.get_yaxis_transform(), fontweight="bold")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("ratio (1.0 = boundary)")
        # fail counts
        for e in order:
            sub = out[out["epoch"] == e]
            n = len(sub)
            nfail = int((sub[col] > 1.0).sum()) if col in sub else 0
            ax.text(order.index(e), ax.get_ylim()[1], f"{nfail}/{n} fail",
                    ha="center", va="bottom", fontsize=10, fontweight="bold",
                    color=color_map[e])

    # hide top/right spines
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Geometry drift: Epoch 1 (blue) vs Epoch 89 (red) — 200-step DiffGui DePPA",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_DIR / "bond_geom_failure_dist.png", dpi=300)
    plt.close(fig)
    logger.info("Wrote %s", OUT_DIR / "bond_geom_failure_dist.png")

    # fail-rate summary
    for col, label in [(COL_BL, "bond_lengths"), (COL_BA, "bond_angles")]:
        print(f"\n{label}:")
        for e in EPOCHS:
            sub = out[out["epoch"] == e]
            n = len(sub)
            nfail = int((sub[col] > 1.0).sum())
            print(f"  epoch {e}: {nfail}/{n} = {nfail / n * 100:.2f}% violate")


if __name__ == "__main__":
    main()
