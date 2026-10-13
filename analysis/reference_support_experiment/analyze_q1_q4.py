#!/usr/bin/env python
"""Formal analysis of the four validation questions (Q1, Q2, Q4).

Q1 discrimination : fraction of generated molecules inside the real-real normal range
Q2 sensitivity    : monotone improvement along the E2PO trajectory
Q4 decoupling     : are there molecules with low SNN but high fragment support?
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path("/home/md666/data/disk6/DiffB110/analysis/reference_support_experiment")
SCORES = HERE / "metric_scores_by_set.csv"
NULL_JSON = HERE / "chembl_null_summary.json"

# "normal range" = at or above the 5th percentile of the real-real null
IN_RANGE_P = 0.05
# Q4 quadrant cut-offs
LOW_SNN = 0.10
HIGH_FRAG = 0.30


def load() -> dict[str, dict[str, np.ndarray]]:
    by_set: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    with SCORES.open("r", newline="") as fh:
        for row in csv.DictReader(fh):
            key = f"{row['group']}|{row['set_label']}"
            for col in (
                "snn",
                "frag_cov_thr1",
                "frag_cov_thr10",
                "frag_cov_thr100",
                "scaff_log",
                "sup_snn",
                "sup_frag_cov_thr1",
                "sup_frag_cov_thr10",
                "sup_frag_cov_thr100",
                "sup_scaff_log",
                "scaff_in_ref",
            ):
                try:
                    by_set[key][col].append(float(row[col]))
                except (KeyError, ValueError):
                    pass
    return {k: {c: np.asarray(v) for c, v in d.items()} for k, d in by_set.items()}


def main() -> None:
    null = json.loads(NULL_JSON.read_text())
    data = load()

    print("=" * 104)
    print("Null (real-real) reference summary")
    print("=" * 104)
    for k, v in null.items():
        print(
            f"  {k:<18s} p05={v['p05']:.4f}  p25={v['p25']:.4f}  "
            f"median={v['median']:.4f}  p95={v['p95']:.4f}"
        )

    print()
    print("=" * 104)
    print("Q1 discrimination — share of molecules inside the real-real normal range "
          f"(support >= {IN_RANGE_P})")
    print("=" * 104)
    print(f"{'set':<32s}{'n':>7s}{'snn_raw':>10s}{'inRange_SNN':>13s}"
          f"{'frag_raw':>10s}{'inRange_Frag':>14s}{'scaff_sup':>11s}")
    for key in sorted(data):
        d = data[key]
        n = len(d["snn"])
        in_snn = float((d["sup_snn"] >= IN_RANGE_P).mean())
        in_frag = float((d["sup_frag_cov_thr10"] >= IN_RANGE_P).mean())
        print(
            f"{key:<32s}{n:>7d}{d['snn'].mean():>10.3f}{in_snn:>13.3f}"
            f"{d['frag_cov_thr10'].mean():>10.3f}{in_frag:>14.3f}"
            f"{d['sup_scaff_log'].mean():>11.3f}"
        )

    print()
    print("=" * 104)
    print("Q4 decoupling — SNN vs fragment support")
    print("=" * 104)
    print(f"{'set':<32s}{'r(sup_snn,sup_frag)':>22s}{'lowSNN&highFrag':>17s}"
          f"{'highSNN&lowFrag':>17s}")
    for key in sorted(data):
        d = data[key]
        a, b = d["sup_snn"], d["sup_frag_cov_thr10"]
        if len(a) < 10:
            continue
        r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
        q_lh = float(((a < LOW_SNN) & (b > HIGH_FRAG)).mean())
        q_hl = float(((a > HIGH_FRAG) & (b < LOW_SNN)).mean())
        print(f"{key:<32s}{r:>22.3f}{q_lh:>17.3f}{q_hl:>17.3f}")

    print()
    print("=" * 104)
    print("Q2 sensitivity — E2PO trajectory (mean support)")
    print("=" * 104)
    print(f"{'step':<16s}{'sup_snn':>10s}{'sup_frag10':>12s}{'sup_scaff':>11s}")
    for key in sorted(k for k in data if k.startswith("e2po_mainline")):
        d = data[key]
        print(
            f"{key.split('|')[1]:<16s}{d['sup_snn'].mean():>10.4f}"
            f"{d['sup_frag_cov_thr10'].mean():>12.4f}{d['sup_scaff_log'].mean():>11.4f}"
        )


if __name__ == "__main__":
    main()
