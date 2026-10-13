#!/usr/bin/env python
"""DiffGui PRETRAINED baseline at 200 / 500 diffusion steps, reasonable subset.

Completes the DiffGui comparison: the RL main training runs at t=200, while the
five-model benchmark runs at t=500.  To compare "before vs after RL" without the
step-count confound we need the pretrained model evaluated at the SAME t=200.

Sources (read-only):
    generation : analysis/benchmark/diffgui_step_sweep_500_200_1000attempts_20260707_run2/
                 <N>steps_rep<k>_seed*/filtered_sanitize_valid/diffgui.sanitize_valid.sdf
    docking    : analysis/benchmark/diffgui_step_sweep_100_200_500_1000_stats_20260710/
                 docking_100_200_500/<N>steps_rep<k>_seed*/diffgui/docking_results.csv
                 (join key = benchmark_index == row index in the sanitize_valid SDF)

Writes one CSV next to this script.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import QED

RDLogger.DisableLog("rdApp.*")

ROOT = Path("/home/md666/data/disk6/DiffB110")
sys.path.insert(0, str(ROOT))
from scripts.offline_rl.structure_pathology_rules import (  # noqa: E402
    compute_structure_pathology_from_smiles,
)

GEN = ROOT / "analysis/benchmark/diffgui_step_sweep_500_200_1000attempts_20260707_run2"
DOCK = ROOT / "analysis/benchmark/diffgui_step_sweep_100_200_500_1000_stats_20260710/docking_100_200_500"
OUT = Path(__file__).resolve().parent / "baseline_step200_500_reasonable.csv"
REPS = {1: "seed1_7", 2: "seed8_14", 3: "seed15_21"}


def run_one(steps: int, rep: int) -> dict | None:
    tag = f"{steps}steps_rep{rep}_{REPS[rep]}"
    sdf_p = GEN / tag / "filtered_sanitize_valid/diffgui.sanitize_valid.sdf"
    dk_p = DOCK / tag / "diffgui/docking_results.csv"
    if not sdf_p.exists() or not dk_p.exists():
        return None

    mols = [m for m in Chem.SDMolSupplier(str(sdf_p), sanitize=True, removeHs=False)]
    dock = pd.read_csv(dk_p)
    ok = dock[dock["success"].astype(str).eq("True")]

    rec = []
    for _, row in ok.iterrows():
        i = int(row["benchmark_index"])
        if i >= len(mols) or mols[i] is None:
            continue
        smi = Chem.MolToSmiles(mols[i])
        f = compute_structure_pathology_from_smiles(smi)
        if f is None:
            continue
        rec.append((float(row["score"]), int(f.violation_count),
                    int(f.structural_violation_count), float(QED.qed(mols[i])), smi))
    if not rec:
        return None

    d = np.array([r[0] for r in rec])
    vc = np.array([r[1] for r in rec])
    svc = np.array([r[2] for r in rec])
    q = np.array([r[3] for r in rec])
    smis = [r[4] for r in rec]
    r = vc == 0
    return {
        "steps": steps, "rep": rep, "n_attempts": len(mols), "n_docked": len(rec),
        "n_unique_smiles": len(set(smis)),
        "all_mean": float(d.mean()), "all_median": float(np.median(d)),
        "n_reasonable": int(r.sum()), "reasonable_yield_pct": float(r.mean() * 100),
        "reasonable_mean": float(d[r].mean()) if r.any() else float("nan"),
        "reasonable_median": float(np.median(d[r])) if r.any() else float("nan"),
        "reasonable_qed": float(q[r].mean()) if r.any() else float("nan"),
        "r1r4_violation_pct": float((svc > 0).mean() * 100),
        "r1r5_violation_pct": float((vc > 0).mean() * 100),
    }


def main() -> None:
    rows = []
    for steps in (200, 500):
        for rep in (1, 2, 3):
            st = run_one(steps, rep)
            if st is None:
                print(f"[skip] steps={steps} rep={rep}")
                continue
            rows.append(st)
            print(f"[{steps:4d}步 rep{rep}] n={st['n_docked']:4d} | 全口径 {st['all_mean']:7.3f} | "
                  f"合理 n={st['n_reasonable']:4d} ({st['reasonable_yield_pct']:5.1f}%) "
                  f"均值={st['reasonable_mean']:7.3f} 中位={st['reasonable_median']:7.3f} | "
                  f"R1R4={(st['r1r4_violation_pct']):5.1f}%", flush=True)

    cols = ["steps", "rep", "n_attempts", "n_docked", "n_unique_smiles", "all_mean",
            "all_median", "n_reasonable", "reasonable_yield_pct", "reasonable_mean",
            "reasonable_median", "reasonable_qed", "r1r4_violation_pct", "r1r5_violation_pct"]
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r_ in rows:
            w.writerow(r_)

    print("\n" + "=" * 76)
    print("DiffGui 预训练基线（3 rep 均值）")
    print("=" * 76)
    for steps in (200, 500):
        g = [r_ for r_ in rows if r_["steps"] == steps]
        if not g:
            continue
        print(f"{steps:4d} 步: 全口径 {np.mean([x['all_mean'] for x in g]):7.3f} | "
              f"合理分子产率 {np.mean([x['reasonable_yield_pct'] for x in g]):5.1f}% | "
              f"合理分子对接 均值 {np.mean([x['reasonable_mean'] for x in g]):7.3f} "
              f"中位 {np.mean([x['reasonable_median'] for x in g]):7.3f} | "
              f"合理分子 QED {np.mean([x['reasonable_qed'] for x in g]):.3f} | "
              f"R1R4 {np.mean([x['r1r4_violation_pct'] for x in g]):5.1f}%")
    print(f"[wrote] {OUT}")


if __name__ == "__main__":
    main()
