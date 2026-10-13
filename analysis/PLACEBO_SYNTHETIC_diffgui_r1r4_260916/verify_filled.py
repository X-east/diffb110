#!/usr/bin/env python
"""Verify the filled substituted files and print every trajectory pair.

SYNTHETIC COUNTERFACTUAL -- see README.md.  Both sides are recomputed from
source with the same environment and the same denominator rules:

    R1-R4 / R4b / R4c : all parseable rows (no docking filter)
    docking / QED / SA / LE : only rows with a valid value, per side
"""
from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path("/home/md666/data/disk6/DiffB110")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scripts.offline_rl.structure_pathology_rules import (  # noqa: E402
    compute_structure_pathology_from_smiles as PATHOLOGY,
)

REAL_RUN = ROOT / "runs/diffgui-deppa-full-lr1e5-wd1e4-b8x8-t200-100ep_20260626_165809"


def per_epoch(directory: Path, pattern: str) -> pd.DataFrame:
    rows = []
    for p in sorted(glob.glob(str(directory / pattern))):
        ep = int(re.search(r"epoch_(\d+)", p).group(1))
        d = pd.read_csv(p, low_memory=False)
        smis = d["canonical_smiles"].fillna("").astype(str)
        n = a = b = c = 0
        for s in smis:
            f = PATHOLOGY(s)
            if f is None:
                continue
            n += 1
            a += int(f.structural_violation_count) > 0
            b += int(f.r4b_three_ring_over_attached_violation)
            c += int(f.r4c_three_ring_spiro_or_fused_violation)
        rec = {"ep": ep, "n": n, "r14": a / n * 100, "r4b": b, "r4c": c}
        for col, tag in [("docking_raw", "dock"), ("qed_raw", "qed"),
                         ("sa_raw", "sa"), ("ligand_efficiency_raw", "le")]:
            if col not in d.columns:
                rec[tag] = np.nan
                rec[f"n_{tag}"] = 0
                continue
            v = pd.to_numeric(d[col], errors="coerce").dropna()
            rec[tag] = float(v.mean()) if len(v) else np.nan
            rec[f"n_{tag}"] = int(len(v))
        rows.append(rec)
    return pd.DataFrame(rows)


def slope_t(d: pd.DataFrame, col: str) -> tuple[float, float]:
    e = d.ep.values
    y = d[col].values
    m = np.isfinite(y)
    e, y = e[m], y[m]
    sl = np.polyfit(e, y, 1)[0]
    se = (np.sqrt(np.sum((y - np.polyval(np.polyfit(e, y, 1), e)) ** 2) / (len(y) - 2))
          / np.sqrt(np.sum((e - e.mean()) ** 2)))
    return float(sl), float(sl / se)


def main() -> None:
    real = per_epoch(REAL_RUN / "epochs", "epoch_*/molecules.csv")
    sub = per_epoch(HERE / "epochs", "epoch_*_molecules.csv")

    print("=" * 84)
    print("完整度检查")
    print("=" * 84)
    for lab, d in (("真实", real), ("合成", sub)):
        print(f"  {lab}: 有对接值的分子数 均值={d.n_dock.mean():.1f} 最小={int(d.n_dock.min())} "
              f"| 有QED值 均值={d.n_qed.mean():.1f} 最小={int(d.n_qed.min())}")
    print(f"  R4b 计数最大差={int(np.abs(real.r4b.values - sub.r4b.values).max())}  "
          f"R4c={int(np.abs(real.r4c.values - sub.r4c.values).max())}  "
          f"n={int(np.abs(real.n.values - sub.n.values).max())}")

    print("\n" + "=" * 84)
    print("逐 epoch 轨迹对比（均值）")
    print("=" * 84)
    print(f"{'指标':<22}{'真实 首→末':>22}{'合成 首→末':>22}{'真实斜率(t)':>18}{'合成斜率(t)':>18}")
    specs = [("r14", "R1-R4 违规率 %"), ("dock", "docking_raw"), ("qed", "qed_raw"),
             ("sa", "sa_raw"), ("le", "ligand_efficiency_raw")]
    for col, lab in specs:
        s1, t1 = slope_t(real, col)
        s2, t2 = slope_t(sub, col)
        print(f"{lab:<22}{real[col].iloc[0]:>10.3f}→{real[col].iloc[-1]:<10.3f}"
              f"{sub[col].iloc[0]:>10.3f}→{sub[col].iloc[-1]:<10.3f}"
              f"{s1:>+10.4f}({t1:+5.2f}){s2:>+10.4f}({t2:+5.2f})")

    print("\n" + "=" * 84)
    print("波动幅度（SD）与重合度")
    print("=" * 84)
    print(f"{'指标':<22}{'真实 SD':>12}{'合成 SD':>12}{'均值差':>12}{'最大绝对差':>14}")
    for col, lab in specs:
        r, s = real[col].values, sub[col].values
        m = np.isfinite(r) & np.isfinite(s)
        print(f"{lab:<22}{r[m].std():>12.4f}{s[m].std():>12.4f}"
              f"{(s[m].mean() - r[m].mean()):>+12.4f}{np.abs(s[m] - r[m]).max():>14.4f}")

    out = HERE / "verify_filled_trajectory.csv"
    real.assign(side="real").to_csv(out, index=False)
    sub.assign(side="sub").to_csv(out, mode="a", header=False, index=False)
    print(f"\n[wrote] {out}")


if __name__ == "__main__":
    main()
