"""Aggregate the 5-model 1000-step 3-rep benchmark into a two-caliber docking
comparison: full-caliber (all docked molecules) vs reasonable-caliber (R1-R5
clean only). Mirrors the E2PO presentation so the two can be compared directly.
"""
from __future__ import annotations

import csv
import statistics
from pathlib import Path

BASE = Path(__file__).resolve().parent
REASONABLE_CSV = BASE / "reasonable_molecules_metrics.csv"
FULL_CSV = (
    BASE.parent
    / "8gjl_c1101_benchmark_20260706_1000steps_3rep_formal"
    / "formal_3rep_docking_model_summary.csv"
)
OUT_CSV = BASE / "five_model_docking_full_vs_reasonable.csv"

# reasonable label -> formal model name (capitalized, GCDM-SBDD mapping)
MODEL_ORDER = ["DiffSBDD", "DiffGui", "TargetDiff", "PMDM", "GCDM-SBDD"]
REASON_LABEL = {
    "DiffSBDD": "diffsbdd",
    "DiffGui": "diffgui",
    "TargetDiff": "targetdiff",
    "PMDM": "pmdm",
    "GCDM-SBDD": "gcdm",
}


def main() -> None:
    # ---- full-caliber (all docked molecules) ----
    full: dict[str, dict] = {}
    with FULL_CSV.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            full[row["model"]] = {
                "full_dock_mean": float(row["score_mean_mean"]),
                "full_dock_median": float(row["score_median_mean"]),
            }

    # ---- reasonable-caliber (R1-R5 clean) grouped by base model ----
    groups: dict[str, dict[str, list[float]]] = {m: {} for m in MODEL_ORDER}
    with REASONABLE_CSV.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            label = row["label"]
            base = label.rsplit("-rep", 1)[0]  # "diffsbdd-rep1" -> "diffsbdd"
            # map back to canonical model name
            canon = next((k for k, v in REASON_LABEL.items() if v == base), None)
            if canon is None:
                continue
            g = groups[canon]
            for col in (
                "dock_median_reasonable",
                "dock_mean_reasonable",
                "reasonable_yield_per_attempt_pct",
                "reasonable_pct_of_relaxed",
                "qed_median_reasonable",
                "reasonable_count",
            ):
                g.setdefault(col, []).append(float(row[col]))

    rows = []
    for m in MODEL_ORDER:
        g = groups[m]
        f = full.get(m, {})
        rows.append(
            {
                "model": m,
                "full_dock_mean": round(f.get("full_dock_mean", float("nan")), 3),
                "full_dock_median": round(f.get("full_dock_median", float("nan")), 3),
                "reasonable_dock_mean": round(statistics.mean(g["dock_mean_reasonable"]), 3),
                "reasonable_dock_median": round(
                    statistics.mean(g["dock_median_reasonable"]), 3
                ),
                "reasonable_yield_per_attempt_pct": round(
                    statistics.mean(g["reasonable_yield_per_attempt_pct"]), 2
                ),
                "reasonable_pct_of_relaxed": round(
                    statistics.mean(g["reasonable_pct_of_relaxed"]), 2
                ),
                "qed_median_reasonable": round(
                    statistics.mean(g["qed_median_reasonable"]), 3
                ),
                "n_reasonable_total": int(sum(g["reasonable_count"])),
            }
        )

    fields = [
        "model",
        "full_dock_mean",
        "full_dock_median",
        "reasonable_dock_mean",
        "reasonable_dock_median",
        "reasonable_yield_per_attempt_pct",
        "reasonable_pct_of_relaxed",
        "qed_median_reasonable",
        "n_reasonable_total",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"[ok] wrote {OUT_CSV}\n")
    hdr = f"{'model':12s} {'full_med':>9s} {'reason_med':>11s} {'yield%':>7s} {'relaxed%':>9s}"
    print(hdr)
    for r in rows:
        print(
            f"{r['model']:12s} {r['full_dock_median']:9.2f} "
            f"{r['reasonable_dock_median']:11.2f} {r['reasonable_yield_per_attempt_pct']:7.2f} "
            f"{r['reasonable_pct_of_relaxed']:9.2f}"
        )


if __name__ == "__main__":
    main()
