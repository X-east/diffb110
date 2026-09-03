"""Aggregate E2PO checkpoint full-molecule (all relaxed) docking scores.

This is an aggregation-only task: every E2PO checkpoint's generation pipeline
already produced `docking_relaxed/docking_results.csv` covering ALL relaxed
molecules (not just the reasonable subset). We recompute docking statistics on
the FULL set so it is directly comparable to the benchmark's full-molecule
docking mean (DiffSBDD benchmark = -9.46).

Outputs:
- e2po_full_molecule_docking.csv  (per-checkpoint full-set docking stats)
- prints a comparison table against the benchmark reference.

No scientific/model changes here; pure evidence aggregation.
"""

import csv
import json
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# checkpoint label -> docking_results.csv path
CHECKPOINTS = {
    "Original": REPO_ROOT
    / "analysis"
    / "diffsbdd_global_e2po_structure_matched_x_only_generation_1000_260901"
    / "docking_relaxed"
    / "docking_results.csv",
    "step_002000": REPO_ROOT
    / "OUT"
    / "diffsbdd_e2po_mainline_intermediate_gen"
    / "step_002000"
    / "docking_relaxed"
    / "docking_results.csv",
    "step_004000": REPO_ROOT
    / "OUT"
    / "diffsbdd_e2po_mainline_intermediate_gen"
    / "step_004000"
    / "docking_relaxed"
    / "docking_results.csv",
    "step_006000": REPO_ROOT
    / "OUT"
    / "diffsbdd_e2po_mainline_intermediate_gen"
    / "step_006000"
    / "docking_relaxed"
    / "docking_results.csv",
    "step_008000": REPO_ROOT
    / "OUT"
    / "diffsbdd_e2po_mainline_intermediate_gen"
    / "step_008000"
    / "docking_relaxed"
    / "docking_results.csv",
    "step_010000": REPO_ROOT
    / "OUT"
    / "diffsbdd_e2po_mainline_intermediate_gen"
    / "step_010000"
    / "docking_relaxed"
    / "docking_results.csv",
    "step_012000": REPO_ROOT
    / "OUT"
    / "diffsbdd_e2po_mainline_intermediate_gen"
    / "step_012000"
    / "docking_relaxed"
    / "docking_results.csv",
    "step_014000": REPO_ROOT
    / "OUT"
    / "diffsbdd_e2po_mainline_intermediate_gen"
    / "step_014000"
    / "docking_relaxed"
    / "docking_results.csv",
    "step_016000": REPO_ROOT
    / "OUT"
    / "diffsbdd_e2po_mainline_intermediate_gen"
    / "step_016000"
    / "docking_relaxed"
    / "docking_results.csv",
    "step_018000": REPO_ROOT
    / "OUT"
    / "diffsbdd_e2po_mainline_intermediate_gen"
    / "step_018000"
    / "docking_relaxed"
    / "docking_results.csv",
    "step_final": REPO_ROOT
    / "OUT"
    / "diffsbdd_e2po_mainline_intermediate_gen"
    / "step_final"
    / "docking_relaxed"
    / "docking_results.csv",
}

# Benchmark reference: DiffSBDD 1000-step full-molecule docking mean (documented).
BENCHMARK_DIFFSBDD_FULL_DOCKING = -9.46


def parse_float(s) -> float:
    if s is None:
        return None
    s = str(s).strip()
    if s == "" or s.lower() in ("nan", "none", "null"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def aggregate(path: Path):
    if not path.exists():
        return None
    scores = []
    total = 0
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            success = str(row.get("success", "")).strip().lower() == "true"
            if not success:
                continue
            sc = parse_float(row.get("score"))
            if sc is None:
                continue
            scores.append(sc)
    if not scores:
        return {
            "total": total,
            "docked": 0,
            "mean": None,
            "std": None,
            "min": None,
            "median": None,
        }
    return {
        "total": total,
        "docked": len(scores),
        "mean": statistics.fmean(scores),
        "std": statistics.pstdev(scores),
        "min": min(scores),
        "median": statistics.median(scores),
    }


def main():
    out_rows = []
    print(f"{'checkpoint':<14}{'total':>8}{'docked':>8}{'mean':>10}{'std':>9}{'min':>9}{'median':>9}")
    print("-" * 67)
    for label, path in CHECKPOINTS.items():
        agg = aggregate(path)
        if agg is None:
            print(f"{label:<14}  (MISSING: {path})")
            continue
        mean = agg["mean"]
        mean_s = f"{mean:.3f}" if mean is not None else "NA"
        std_s = f"{agg['std']:.3f}" if agg["std"] is not None else "NA"
        min_s = f"{agg['min']:.3f}" if agg["min"] is not None else "NA"
        med_s = f"{agg['median']:.3f}" if agg["median"] is not None else "NA"
        print(f"{label:<14}{agg['total']:>8}{agg['docked']:>8}{mean_s:>10}{std_s:>9}{min_s:>9}{med_s:>9}")
        out_rows.append({
            "checkpoint": label,
            "total_relaxed": agg["total"],
            "docked": agg["docked"],
            "full_docking_mean": mean,
            "full_docking_std": agg["std"],
            "full_docking_min": agg["min"],
            "full_docking_median": agg["median"],
            "delta_vs_benchmark": (mean - BENCHMARK_DIFFSBDD_FULL_DOCKING) if mean is not None else None,
        })

    print("-" * 67)
    print(f"Benchmark DiffSBDD full-molecule docking mean: {BENCHMARK_DIFFSBDD_FULL_DOCKING:.3f}")

    out_path = Path(__file__).resolve().parent / "e2po_full_molecule_docking.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\nWrote: {out_path}")

    # JSON for easy downstream figure use
    json_path = Path(__file__).resolve().parent / "e2po_full_molecule_docking.json"
    with json_path.open("w") as f:
        json.dump(out_rows, f, indent=2)
    print(f"Wrote: {json_path}")


if __name__ == "__main__":
    main()
