#!/usr/bin/env python
"""Mean docking score of REASONABLE molecules in the DiffGui RL main training.

Answers a single question: after DiffGui reinforcement learning, among the
generated molecules that pass the R1-R5 structural-pathology口径 (i.e. the
"reasonable" subset), what is the mean docking score?

Run under study:
    runs/diffgui-deppa-full-lr1e5-wd1e4-b8x8-t200-100ep_20260626_165809
    (epoch 1 docking -9.781 -> best -11.283 @ epoch 79, 89 completed epochs;
     matches the numbers reported in the thesis section 2.3.5)

Two denominators are reported for every epoch:
    all_docked     : every molecule with docking_success == 1
    reasonable     : the above, restricted to violation_count == 0 (R1-R5 clean)

Read-only w.r.t. the run; writes one CSV next to this script.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path("/home/md666/data/disk6/DiffB110")
sys.path.insert(0, str(ROOT))
from scripts.offline_rl.structure_pathology_rules import (  # noqa: E402
    compute_structure_pathology_from_smiles,
)

RUN = ROOT / "runs/diffgui-deppa-full-lr1e5-wd1e4-b8x8-t200-100ep_20260626_165809"
OUT = Path(__file__).resolve().parent / "docking_of_reasonable_molecules.csv"
EPOCHS = [1, 2, 3, 5, 7, 9, 11, 13, 15, 20, 25, 30, 35, 40, 45, 50, 55, 59,
          60, 65, 70, 75, 79, 80, 85, 89]


def score_epoch(csv_path: Path) -> dict | None:
    """Return docking statistics for one epoch, split by structural reasonableness."""
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception:
        return None
    if "docking_raw" not in df.columns or "canonical_smiles" not in df.columns:
        return None

    ok = df[df["docking_success"].fillna(0).astype(float).eq(1)].copy()
    ok = ok[np.isfinite(pd.to_numeric(ok["docking_raw"], errors="coerce"))]
    if ok.empty:
        return None

    parsed = []
    for smi, d in zip(ok["canonical_smiles"].astype(str), ok["docking_raw"].astype(float)):
        f = compute_structure_pathology_from_smiles(smi)
        if f is not None:
            parsed.append((d, int(f.violation_count), int(f.structural_violation_count)))
    if not parsed:
        return None

    dock = np.asarray([p[0] for p in parsed])
    vc = np.asarray([p[1] for p in parsed])
    svc = np.asarray([p[2] for p in parsed])

    r = vc == 0                     # R1-R5 clean  == thesis "reasonable"
    s = svc == 0                    # R1-R4 clean  == structural only
    return {
        "n_docked": dock.size,
        "all_mean": float(dock.mean()),
        "all_median": float(np.median(dock)),
        "n_reasonable": int(r.sum()),
        "reasonable_share": float(r.mean()),
        "reasonable_mean": float(dock[r].mean()) if r.any() else float("nan"),
        "reasonable_median": float(np.median(dock[r])) if r.any() else float("nan"),
        "n_struct_clean": int(s.sum()),
        "struct_clean_mean": float(dock[s].mean()) if s.any() else float("nan"),
        "n_unreasonable": int((~r).sum()),
        "unreasonable_mean": float(dock[~r].mean()) if (~r).any() else float("nan"),
    }


def main() -> None:
    if not RUN.exists():
        raise SystemExit(f"run dir not found: {RUN}")
    rows = []
    for ep in EPOCHS:
        p = RUN / f"epochs/epoch_{ep:04d}/molecules.csv"
        if not p.exists():
            continue
        st = score_epoch(p)
        if st is None:
            continue
        st["epoch"] = ep
        rows.append(st)
        print(f"[ep {ep:3d}] all {st['all_mean']:7.3f} | "
              f"reasonable n={st['n_reasonable']:4d} ({st['reasonable_share']:.3f}) "
              f"mean={st['reasonable_mean']:7.3f} med={st['reasonable_median']:7.3f} | "
              f"unreasonable mean={st['unreasonable_mean']:7.3f}", flush=True)

    if not rows:
        raise SystemExit("no epochs scored")
    cols = ["epoch", "n_docked", "all_mean", "all_median", "n_reasonable",
            "reasonable_share", "reasonable_mean", "reasonable_median",
            "n_struct_clean", "struct_clean_mean", "n_unreasonable", "unreasonable_mean"]
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in sorted(rows, key=lambda x: x["epoch"]):
            w.writerow({k: r.get(k) for k in cols})

    first, best = rows[0], min(rows, key=lambda r: r["all_mean"])
    print("\n" + "=" * 78)
    print(f"epoch {first['epoch']} : 全口径 {first['all_mean']:.3f} | "
          f"合理分子 n={first['n_reasonable']} 均值 {first['reasonable_mean']:.3f} "
          f"中位 {first['reasonable_median']:.3f}")
    print(f"epoch {best['epoch']} : 全口径 {best['all_mean']:.3f} | "
          f"合理分子 n={best['n_reasonable']} 均值 {best['reasonable_mean']:.3f} "
          f"中位 {best['reasonable_median']:.3f}")
    print(f"[wrote] {OUT}")


if __name__ == "__main__":
    main()
