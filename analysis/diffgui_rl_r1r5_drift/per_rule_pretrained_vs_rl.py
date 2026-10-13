#!/usr/bin/env python
"""DiffGui pretrained (t=200) vs RL-trained: R1-R5 per-rule comparison.

Motivation: the aggregate R1-R4 violation rate of the DiffGui RL run drops
noticeably over training, which on its own would read as "structural quality
improved".  This script checks that claim two ways:

  1. per-rule breakdown  -- aggregate can fall while a specific rule explodes
  2. size-stratified     -- RL molecules are smaller, and smaller molecules
                            have fewer rings, so any aggregate drop must be
                            re-checked at matched heavy-atom counts

Read-only; writes one CSV next to this script.
"""
from __future__ import annotations

import csv
import glob
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path("/home/md666/data/disk6/DiffB110")
sys.path.insert(0, str(ROOT))
from scripts.offline_rl.structure_pathology_rules import (  # noqa: E402
    compute_structure_pathology_from_smiles as FN,
)

GEN = ROOT / "analysis/benchmark/diffgui_step_sweep_500_200_1000attempts_20260707_run2"
DOCK = ROOT / "analysis/benchmark/diffgui_step_sweep_100_200_500_1000_stats_20260710/docking_100_200_500"
RL = ROOT / "runs/diffgui-deppa-full-lr1e5-wd1e4-b8x8-t200-100ep_20260626_165809"
OUT = Path(__file__).resolve().parent / "per_rule_pretrained_vs_rl.csv"
REPS = {1: "seed1_7", 2: "seed8_14", 3: "seed15_21"}
KEYS = [
    ("r1_large_ring_violation", "R1 大环"),
    ("r2_ring_system_violation", "R2 环系规模"),
    ("r3_four_ring_violation", "R3 四元环"),
    ("r4a_three_ring_hetero_violation", "R4a 三元环杂原子"),
    ("r4b_three_ring_over_attached_violation", "R4b 三元环过连接"),
    ("r4c_three_ring_spiro_or_fused_violation", "R4c 三元环螺/稠"),
    ("r5_carbon_fraction_violation", "R5 碳比例"),
]


def collect_pretrained(steps: int = 200) -> pd.DataFrame:
    recs = []
    for rep, seed in REPS.items():
        tag = f"{steps}steps_rep{rep}_{seed}"
        sdf = GEN / tag / "filtered_sanitize_valid/diffgui.sanitize_valid.sdf"
        dk = DOCK / tag / "diffgui/docking_results.csv"
        if not sdf.exists() or not dk.exists():
            continue
        mols = list(Chem.SDMolSupplier(str(sdf), sanitize=True, removeHs=False))
        d = pd.read_csv(dk)
        for _, r in d[d["success"].astype(str).eq("True")].iterrows():
            i = int(r["benchmark_index"])
            if i >= len(mols) or mols[i] is None:
                continue
            f = FN(Chem.MolToSmiles(mols[i]))
            if f is None:
                continue
            recs.append({"source": "pretrained_t200", "hac": mols[i].GetNumHeavyAtoms(),
                         "score": float(r["score"]),
                         **{k: int(getattr(f, k)) for k, _ in KEYS},
                         "r14": int(f.structural_violation_count),
                         "r15": int(f.violation_count)})
    return pd.DataFrame(recs)


def collect_rl(lo: int = 1, hi: int = 89) -> pd.DataFrame:
    recs = []
    for p in sorted(glob.glob(str(RL / "epochs/epoch_*/molecules.csv"))):
        ep = int(re.search(r"epoch_(\d+)", p).group(1))
        if not (lo <= ep <= hi):
            continue
        d = pd.read_csv(p, low_memory=False)
        ok = d[pd.to_numeric(d["docking_raw"], errors="coerce").notna()
               & d["docking_success"].fillna(0).eq(1)]
        for s, n, sc in zip(ok["canonical_smiles"].astype(str),
                            ok["num_atoms"].astype(int), ok["docking_raw"].astype(float)):
            f = FN(s)
            if f is None:
                continue
            recs.append({"source": "rl_t200", "hac": int(n), "score": float(sc),
                         **{k: int(getattr(f, k)) for k, _ in KEYS},
                         "r14": int(f.structural_violation_count),
                         "r15": int(f.violation_count)})
    return pd.DataFrame(recs)


def main() -> None:
    a = collect_pretrained(200)
    b = collect_rl()
    df = pd.concat([a, b], ignore_index=True)
    print(f"pretrained n={len(a)}  RL n={len(b)}")

    rows = []
    print("\n" + "=" * 72)
    print(f"{'规则':<20s}{'预训练(200步)':>14s}{'RL(全期)':>12s}{'变化':>12s}")
    print("-" * 72)
    for k, lab in KEYS:
        pa = a[k].mean() * 100
        pb = b[k].mean() * 100
        rows.append({"rule": lab, "pretrained_pct": pa, "rl_pct": pb, "delta_pp": pb - pa})
        print(f"{lab:<20s}{pa:>13.1f}%{pb:>11.1f}%{pb - pa:>+11.1f}pp")
    for k, lab in (("r14", "R1-R4 任一"), ("r15", "R1-R5 任一")):
        pa, pb = a[k].gt(0).mean() * 100, b[k].gt(0).mean() * 100
        rows.append({"rule": lab, "pretrained_pct": pa, "rl_pct": pb, "delta_pp": pb - pa})
        print(f"{lab:<20s}{pa:>13.1f}%{pb:>11.1f}%{pb - pa:>+11.1f}pp")
    print("-" * 72)
    print(f"{'重原子数均值':<20s}{a.hac.mean():>13.2f}{b.hac.mean():>12.2f}"
          f"{b.hac.mean() - a.hac.mean():>+12.2f}")
    print(f"{'全口径对接均值':<20s}{a.score.mean():>13.3f}{b.score.mean():>12.3f}"
          f"{b.score.mean() - a.score.mean():>+12.3f}")

    # ---- size-stratified R1-R4 violation rate -----------------------------
    print("\n" + "=" * 72)
    print("按重原子数分箱：R1-R4 违规率（消除尺寸混杂）")
    print("=" * 72)
    edges = [0, 14, 16, 18, 20, 22, 24, 100]
    lab = [f"{edges[i]}-{edges[i+1]-1}" for i in range(len(edges) - 1)]
    a["bin"] = pd.cut(a.hac, bins=edges, labels=lab, right=False)
    b["bin"] = pd.cut(b.hac, bins=edges, labels=lab, right=False)
    print(f"{'重原子数':<12s}{'预训练 n':>10s}{'违规率':>9s}{'RL n':>8s}{'违规率':>9s}{'变化':>10s}")
    str_rows = []
    for L in lab:
        ga, gb = a[a.bin == L], b[b.bin == L]
        if len(ga) < 20 or len(gb) < 20:
            continue
        pa = ga.r14.gt(0).mean() * 100
        pb = gb.r14.gt(0).mean() * 100
        str_rows.append({"hac_bin": L, "n_pre": len(ga), "pre_pct": pa,
                         "n_rl": len(gb), "rl_pct": pb, "delta_pp": pb - pa})
        print(f"{L:<12s}{len(ga):>10d}{pa:>8.1f}%{len(gb):>8d}{pb:>8.1f}%{pb - pa:>+9.1f}pp")

    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["rule", "pretrained_pct", "rl_pct", "delta_pp"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[wrote] {OUT}")


if __name__ == "__main__":
    main()
