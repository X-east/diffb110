"""Compute E2PO-native R1-R5 pathology rates for the five benchmark models.

This provides the structural-pathology evidence used to select a backbone in
the benchmark stage. It deliberately reuses ``structure_pathology_rules`` (the
same module that built the E2PO preference pairs) so the selection criterion
and the optimisation objective are defined by one identical rule set.

Key difference from the existing ``raw_quality_vs_chembl`` indicator table:
that table uses ``ring size >= 8`` for the large-ring indicator, while the E2PO
R1 rule uses ``ring size >= 7``. The two are not interchangeable.
"""
from __future__ import annotations

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

RULE_COLS: tuple[str, ...] = (
    "r1_large_ring_violation",
    "r2_ring_system_violation",
    "r3_four_ring_violation",
    "r4a_three_ring_hetero_violation",
    "r4b_three_ring_over_attached_violation",
    "r4c_three_ring_spiro_or_fused_violation",
    "r5_carbon_fraction_violation",
)


def main() -> None:
    """Compute and persist per-model R1-R5 violation rates at 1000 steps."""
    warnings.filterwarnings("ignore")
    frame = pd.read_csv(QUALITY_CSV)
    frame = frame[frame["step"] == 1000].copy()
    frame = frame[frame["canonical_smiles"].notna()].copy()

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

    grouped = rules.groupby("model")
    summary = grouped.agg(
        n_valid=("r1r4_any", "size"),
        **{column: (column, "mean") for column in RULE_COLS},
        r1r4_any=("r1r4_any", "mean"),
        r1r5_any=("r1r5_any", "mean"),
    )
    summary = summary.reset_index().sort_values("r1r4_any")
    summary.to_csv(Path(__file__).parent / "model_r1r5_rates_1000steps.csv", index=False)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
