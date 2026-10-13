#!/usr/bin/env python
"""Estimate preference-pair yield BEFORE spending GPU hours on docking.

Three of the four confounder constraints (element composition, heavy-atom
count, TPSA) are computable without docking, so the matching yield can be
measured for free.  The fourth constraint (docking gap <= 1.5) is assumed to
keep roughly 40-60% of the remaining pairs; the reported numbers are therefore
an upper bound and can be multiplied accordingly.

Reports, for several delta thresholds, how many (winner, loser) pairs survive:
    exact element composition  AND  equal heavy-atom count  AND  |dTPSA| <= 30
    AND  soft_novel(loser) - soft_novel(winner) >= delta
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

HERE = Path("/home/md666/data/disk6/DiffB110/analysis/reference_support_experiment")
PROFILES = HERE / "candidate_200k_profiles.csv"

DELTAS = (0.15, 0.25, 0.35)
TPSA_TOL = 30.0
MAX_LOSERS_PER_WINNER = 5      # cap so one winner cannot dominate the pair set


def main() -> None:
    groups: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
    with PROFILES.open("r", newline="") as fh:
        for row in csv.DictReader(fh):
            key = (row["element_composition"], int(row["heavy_atom_count"]))
            groups[key].append((float(row["soft_novel"]), float(row["tpsa"])))
    print(f"[pool] groups (composition, hac): {len(groups):,}")

    counts = {d: 0 for d in DELTAS}
    winners = {d: set() for d in DELTAS}

    for key, items in groups.items():
        if len(items) < 2:
            continue
        items.sort()                       # ascending soft_novel (best first)
        novels = [x[0] for x in items]
        tpsas = [x[1] for x in items]
        n = len(items)
        for i in range(n):
            nw = novels[i]
            tw = tpsas[i]
            kept = 0
            for j in range(n - 1, -1, -1):   # worst first
                if novels[j] - nw < DELTAS[0]:
                    break
                if abs(tpsas[j] - tw) > TPSA_TOL:
                    continue
                gap = novels[j] - nw
                for d in DELTAS:
                    if gap >= d:
                        counts[d] += 1
                        winners[d].add((key, i))
                kept += 1
                if kept >= MAX_LOSERS_PER_WINNER:
                    break

    print()
    print(f"{'delta':>8s}{'pairs':>12s}{'unique winners':>16s}"
          f"{'~after docking(50%)':>22s}")
    for d in DELTAS:
        print(f"{d:>8.2f}{counts[d]:>12,}{len(winners[d]):>16,}{int(counts[d] * 0.5):>22,}")

    print()
    print("NOTE: docking gap <= 1.5 is not applied here; halve the pair count")
    print("      as a rough planning figure.")


if __name__ == "__main__":
    main()
