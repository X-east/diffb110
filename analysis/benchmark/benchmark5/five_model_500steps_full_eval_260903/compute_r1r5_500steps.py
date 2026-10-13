"""Compute E2PO-native R1-R5 pathology rates for the 500-step benchmark.

Reuses ``structure_pathology_rules`` (the same module that built the E2PO
preference pairs) so the model-selection criterion and the optimisation
objective share one identical rule set.

The repeat subset is selectable so the 500-step evaluation can be aligned to
the same 3-repeat protocol used by the 1000-step formal benchmark.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "offline_rl"))

from structure_pathology_rules import (  # noqa: E402
    compute_structure_pathology_from_smiles,
)

QUALITY_CSV = (
    REPO_ROOT / "analysis/benchmark/benchmark5/five_model_completed_comparison_260827"
    "/per_record_quality.csv.gz"
)
DEFAULT_REPEATS = ("rep4", "rep5", "rep6")

RULE_COLS: tuple[str, ...] = (
    "r1_large_ring_violation",
    "r2_ring_system_violation",
    "r3_four_ring_violation",
    "r4a_three_ring_hetero_violation",
    "r4b_three_ring_over_attached_violation",
    "r4c_three_ring_spiro_or_fused_violation",
    "r5_carbon_fraction_violation",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=int, default=500)
    parser.add_argument(
        "--repeats",
        nargs="+",
        default=list(DEFAULT_REPEATS),
        help="Repeat ids to include (default: rep4 rep5 rep6, matching 3-repeat protocol).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "model_r1r5_rates_500steps.csv",
    )
    return parser.parse_args()


def main() -> None:
    """Compute and persist per-model R1-R5 rates for the selected repeats."""
    args = parse_args()
    warnings.filterwarnings("ignore")

    frame = pd.read_csv(QUALITY_CSV)
    frame = frame[frame["step"] == args.step]
    frame = frame[frame["repeat"].isin(args.repeats)]
    frame = frame[frame["canonical_smiles"].notna()].copy()

    if frame.empty:
        raise SystemExit(f"no records for step={args.step} repeats={args.repeats}")

    rows: list[dict[str, bool]] = []
    for smiles in frame["canonical_smiles"]:
        facts = compute_structure_pathology_from_smiles(smiles)
        if facts is None:
            rows.append({column: False for column in RULE_COLS})
        else:
            rows.append({column: bool(getattr(facts, column)) for column in RULE_COLS})

    rules = pd.DataFrame(rows, index=frame.index)
    rules["model"] = frame["model"].to_numpy()
    rules["r1r4_any"] = rules[list(RULE_COLS[:6])].any(axis=1)
    rules["r1r5_any"] = rules[list(RULE_COLS)].any(axis=1)

    summary = (
        rules.groupby("model")
        .agg(
            n_valid=("r1r4_any", "size"),
            **{column: (column, "mean") for column in RULE_COLS},
            r1r4_any=("r1r4_any", "mean"),
            r1r5_any=("r1r5_any", "mean"),
        )
        .reset_index()
        .sort_values("r1r4_any")
    )
    summary.to_csv(args.out, index=False)

    print(f"step={args.step} repeats={args.repeats} n={len(frame)}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
