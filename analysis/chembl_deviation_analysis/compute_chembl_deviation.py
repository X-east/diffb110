#!/usr/bin/env python
"""Quantify how far each generative model deviates from the ChEMBL drug-like reference.

Rationale
---------
The structural-pathology criterion (R1-R5) is re-interpreted here as
"the ways a generated molecule differs from a normal drug-like molecule".
The reference is ChEMBL36 scored with the *identical* rule implementation
(``scripts/offline_rl/structure_pathology_rules.py``), so every per-rule
threshold cancels out in the ratio and the absolute threshold value no longer
needs to be defended.

Sources
-------
ChEMBL reference : analysis/diffsbdd_structure_pathology_260831/
                   chembl_structure_pathology_summary.json   (n = 1,550,090)
E2PO mainline    : analysis/e2po_mainline_reasonable_eval/
                   reasonable_molecules_metrics.csv           (denominator = relaxed molecules)
Five-model bench : analysis/benchmark/benchmark5/five_model_500steps_full_eval_260903/
                   model_r1r5_rates_500steps_3rep.csv         (denominator = n_valid relaxed)

Outputs
-------
rule_deviation_summary.csv : one row per (group, label, rule) with
                             gen_pct, chembl_pct, excess_ratio, excess_pp
overall_clean_rate.csv     : structural compliance vs the ChEMBL clean rate
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("/home/md666/data/disk6/DiffB110")
CHEMBL_JSON = (
    ROOT
    / "analysis/diffsbdd_structure_pathology_260831"
    / "chembl_structure_pathology_summary.json"
)
E2PO_CSV = (
    ROOT / "analysis/e2po_mainline_reasonable_eval" / "reasonable_molecules_metrics.csv"
)
FIVE_CSV = (
    ROOT
    / "analysis/benchmark/benchmark5/five_model_500steps_full_eval_260903"
    / "model_r1r5_rates_500steps_3rep.csv"
)
OUT_DIR = ROOT / "analysis/chembl_deviation_analysis"

# (key, display, chembl-json key, e2po-csv column, five-model-csv column)
RULES = [
    ("R1", "large ring (>=7)", "r1_large_ring_violation",
     "r1_large_ring_violation_rate_of_relaxed_pct", "r1_large_ring_violation"),
    ("R2", "ring system > 3", "r2_ring_system_violation",
     "r2_ring_system_violation_rate_of_relaxed_pct", "r2_ring_system_violation"),
    ("R3", "four-membered ring", "r3_four_ring_violation",
     "r3_four_ring_violation_rate_of_relaxed_pct", "r3_four_ring_violation"),
    ("R4a", "3-ring + hetero", "r4a_three_ring_hetero_violation",
     "r4a_three_ring_hetero_violation_rate_of_relaxed_pct", "r4a_three_ring_hetero_violation"),
    ("R4b", "3-ring over-attached", "r4b_three_ring_over_attached_violation",
     "r4b_three_ring_over_attached_violation_rate_of_relaxed_pct",
     "r4b_three_ring_over_attached_violation"),
    ("R4c", "3-ring spiro/fused", "r4c_three_ring_spiro_or_fused_violation",
     "r4c_three_ring_spiro_or_fused_violation_rate_of_relaxed_pct",
     "r4c_three_ring_spiro_or_fused_violation"),
    ("R5", "carbon fraction > 0.85", "r5_carbon_fraction_violation",
     "r5_carbon_fraction_violation_rate_of_relaxed_pct", "r5_carbon_fraction_violation"),
]


def load_chembl() -> dict[str, float]:
    """Return ChEMBL per-rule violation rates in percent."""
    payload = json.loads(CHEMBL_JSON.read_text(encoding="utf-8"))
    return {
        key: payload["rule_rates"][json_key] * 100.0
        for key, _, json_key, _, _ in RULES
    }


def load_e2po() -> dict[str, dict[str, float]]:
    """Return {label: {rule_key: percent}} for the E2PO mainline steps."""
    with E2PO_CSV.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        label = row["label"]
        out[label] = {key: float(row[col]) for key, _, _, col, _ in RULES}
        out[label]["_clean"] = float(row["reasonable_pct_of_relaxed"])
    return out


def load_five_model() -> dict[str, dict[str, float]]:
    """Return {model: {rule_key: percent}} for the 5-model 500-step benchmark."""
    with FIVE_CSV.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        model = row["model"]
        out[model] = {key: float(row[col]) * 100.0 for key, _, _, _, col in RULES}
        out[model]["_clean"] = (1.0 - float(row["r1r5_any"])) * 100.0
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chembl = load_chembl()
    e2po = load_e2po()
    five = load_five_model()
    chembl_clean = 84.72346766961918

    print("=" * 108)
    print("ChEMBL36 reference (n = 1,550,090), identical rule implementation")
    print("=" * 108)
    for key, display, _, _, _ in RULES:
        print(f"  {key:<4s} {display:<24s} {chembl[key]:8.4f}%")
    print(f"  {'ALL':<4s} {'R1-R5 union':<24s} {100 - chembl_clean:8.4f}%   "
          f"(clean = {chembl_clean:.4f}%)")

    groups: list[tuple[str, dict[str, dict[str, float]]]] = [
        ("E2PO-mainline", e2po),
        ("five-model-500steps", five),
    ]

    summary_rows: list[dict[str, object]] = []
    for group, table in groups:
        for label, values in table.items():
            for key, display, _, _, _ in RULES:
                gen = values[key]
                ref = chembl[key]
                ratio = (gen / ref) if ref > 0 else float("nan")
                summary_rows.append(
                    {
                        "group": group,
                        "label": label,
                        "rule": key,
                        "rule_display": display,
                        "gen_pct": round(gen, 4),
                        "chembl_pct": round(ref, 4),
                        "excess_ratio": round(ratio, 3),
                        "excess_pp": round(gen - ref, 4),
                    }
                )

    # markdown-style console tables
    for group, table in groups:
        print()
        print("=" * 108)
        print(f"{group}  -- violation rate / ChEMBL excess ratio")
        print("=" * 108)
        header = f"{'label':<18s}" + "".join(f"{k:>9s}" for k, _, _, _, _ in RULES)
        print(header)
        print(f"{'ChEMBL':<18s}" + "".join(f"{chembl[k]:9.3f}" for k, _, _, _, _ in RULES))
        print("-" * len(header))
        for label, values in table.items():
            cells = ""
            for key, _, _, _, _ in RULES:
                ref = chembl[key]
                ratio = values[key] / ref if ref > 0 else float("nan")
                cells += f"{ratio:9.2f}"
            print(f"{label:<18s}{cells}")

    # overall compliance
    print()
    print("=" * 108)
    print("Overall structural compliance (R1-R5 clean) vs ChEMBL clean rate")
    print("=" * 108)
    print(f"{'label':<22s}{'clean%':>10s}{'vs ChEMBL':>12s}")
    print(f"{'ChEMBL36':<22s}{chembl_clean:10.2f}{1.0:12.3f}")
    overall = []
    for group, table in groups:
        for label, values in table.items():
            clean = values["_clean"]
            ratio = clean / chembl_clean
            print(f"{label:<22s}{clean:10.2f}{ratio:12.3f}")
            overall.append(
                {
                    "group": group,
                    "label": label,
                    "clean_pct": round(clean, 4),
                    "chembl_clean_pct": round(chembl_clean, 4),
                    "ratio_vs_chembl": round(ratio, 4),
                }
            )

    with (OUT_DIR / "rule_deviation_summary.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    with (OUT_DIR / "overall_clean_rate.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(overall[0].keys()))
        writer.writeheader()
        writer.writerows(overall)

    print()
    print(f"wrote {OUT_DIR / 'rule_deviation_summary.csv'}")
    print(f"wrote {OUT_DIR / 'overall_clean_rate.csv'}")


if __name__ == "__main__":
    main()
