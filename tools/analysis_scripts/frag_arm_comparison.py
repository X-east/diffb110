#!/usr/bin/env python
"""Compare ESRPO arms on the two axes that matter: docking and fragment realism.

The existing ``evaluate_reasonable_molecules.py`` covers the R1-R5 structural
pathology口径 but does NOT compute fragment realism, so this script adds:

    docking   : median / mean / p10, plus the count of non-physical outliers
                (docking_raw > 0 -- these exist in this pool, max was +331)
    soft_novel: BRICS fragments vs the ChEMBL reference inventory, i.e. the
                same definition used to build the B/C/D preference pairs

soft_novel definition (identical to profile_candidate_200k.py):
    soft_novel(freq) = 1.0                           if freq <= 0
                     = 1 - log1p(freq)/log1p(1000)   otherwise
    molecule value   = mean over its BRICS fragments

Reference inventory: frag_inventory_500k.csv.gz  (|R| = 500,000 ChEMBL molecules)
Real-real null p95 for soft_novel at |R|=500k: 0.66667  (candidate_200k_summary.json)

Usage
-----
    python frag_arm_comparison.py \
        --input "B_step_002000:.../molecules_relaxed_largest.sdf:.../docking_results.csv" \
        [--input ...] \
        --out-dir analysis/e2po_frag_bcd_compare
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS

RDLogger.DisableLog("rdApp.*")

ROOT = Path("/home/md666/data/disk6/DiffB110")
DEFAULT_INV = ROOT / "analysis/reference_support_experiment/frag_inventory_500k.csv.gz"
SOFT_T = 1000
NULL_P95 = 0.66667          # real ChEMBL holdout p95 at |R| = 500k


def soft_novel(freq: int, t: int = SOFT_T) -> float:
    if freq <= 0:
        return 1.0
    return max(0.0, 1.0 - math.log1p(freq) / math.log1p(t))


def load_inventory(path: Path) -> Counter:
    inv: Counter = Counter()
    with gzip.open(path, "rt", newline="") as fh:
        for row in csv.DictReader(fh):
            inv[row["fragment"]] = int(row["count"])
    return inv


def mol_soft_novel(mol, inv: Counter) -> float | None:
    if mol is None:
        return None
    try:
        frags = list(BRICS.BRICSDecompose(mol))
    except Exception:
        return None
    if not frags:
        return None
    return float(sum(soft_novel(inv.get(f, 0)) for f in frags) / len(frags))


def read_docking(path: Path) -> tuple[list[float], int]:
    """Return (scores of successes, n_success)."""
    scores: list[float] = []
    n_ok = 0
    if not path.is_file():
        return scores, n_ok
    with path.open("r", newline="") as fh:
        for row in csv.DictReader(fh):
            if str(row.get("success", "True")) != "True":
                continue
            n_ok += 1
            try:
                scores.append(float(row["score"]))
            except (TypeError, ValueError):
                continue
    return scores, n_ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", action="append", required=True,
                    help="label:sdf_path:docking_csv  (repeatable)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--inventory", type=Path, default=DEFAULT_INV)
    args = ap.parse_args()

    inv = load_inventory(args.inventory)
    print(f"[inv] loaded {len(inv):,} fragments from {args.inventory.name}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for item in args.input:
        parts = item.split(":")
        if len(parts) != 3:
            print(f"[skip] malformed --input: {item}")
            continue
        label, sdf_path, dock_path = parts
        sdf_p, dock_p = Path(sdf_path), Path(dock_path)

        arm = label.split("_")[0]
        step = label.split("_", 1)[1] if "_" in label else label

        vals: list[float] = []
        n_mol = 0
        if sdf_p.is_file():
            for mol in Chem.SDMolSupplier(str(sdf_p), removeHs=False, sanitize=True):
                n_mol += 1
                v = mol_soft_novel(mol, inv)
                if v is not None:
                    vals.append(v)
        scores, n_ok = read_docking(dock_p)

        outliers = [s for s in scores if s > 0.0]
        row = {
            "label": label, "arm": arm, "step": step,
            "n_molecules": n_mol,
            "n_soft_novel": len(vals),
            "n_docked": len(scores),
            "soft_novel_mean": round(statistics.fmean(vals), 4) if vals else None,
            "soft_novel_median": round(statistics.median(vals), 4) if vals else None,
            "soft_novel_share_above_null_p95": round(
                sum(1 for v in vals if v > NULL_P95) / len(vals), 4) if vals else None,
            "docking_median": round(statistics.median(scores), 4) if scores else None,
            "docking_mean": round(statistics.fmean(scores), 4) if scores else None,
            "docking_p10": round(sorted(scores)[max(0, int(0.10 * len(scores)) - 1)], 4) if scores else None,
            "docking_min_best": round(min(scores), 4) if scores else None,
            "n_docking_outliers_gt0": len(outliers),
        }
        rows.append(row)
        print(f"[scored] {label:<22s} n={n_mol:<5d} docked={len(scores):<5d} "
              f"dock_med={row['docking_median']} soft={row['soft_novel_mean']}")

    out_csv = out_dir / "frag_arm_comparison.csv"
    if rows:
        with out_csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    (out_dir / "frag_arm_comparison.json").write_text(
        json.dumps({"null_p95": NULL_P95, "soft_T": SOFT_T, "rows": rows}, indent=2),
        encoding="utf-8")

    # ---- 汇总表 ----
    print("\n" + "=" * 108)
    print("按臂汇总（docking 用中位数；均值受非物理离群值污染，仅供参考）")
    print("=" * 108)
    print(f"{'arm':<6}{'点数':>6}{'对接中位(均值)':>20}{'soft_novel均值':>16}"
          f"{'超真实p95占比':>15}{'离群值':>8}")
    for arm in sorted({r["arm"] for r in rows}):
        sub = [r for r in rows if r["arm"] == arm]
        dm = [r["docking_median"] for r in sub if r["docking_median"] is not None]
        sm = [r["soft_novel_mean"] for r in sub if r["soft_novel_mean"] is not None]
        sp = [r["soft_novel_share_above_null_p95"] for r in sub
              if r["soft_novel_share_above_null_p95"] is not None]
        o = sum(r["n_docking_outliers_gt0"] for r in sub)
        print(f"{arm:<6}{len(sub):>6}"
              f"{(f'{statistics.fmean(dm):.3f}' if dm else '-'):>20}"
              f"{(f'{statistics.fmean(sm):.4f}' if sm else '-'):>16}"
              f"{(f'{statistics.fmean(sp):.4f}' if sp else '-'):>15}{o:>8}")
    print(f"\n[wrote] {out_csv}")


if __name__ == "__main__":
    main()
