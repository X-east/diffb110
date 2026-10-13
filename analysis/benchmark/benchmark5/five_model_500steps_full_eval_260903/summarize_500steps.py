"""Summarize the 500-step five-model benchmark into 1000-step-comparable tables.

Produces:
- docking_summary_by_repeat.csv  : per repeat x model docking metrics
- docking_model_summary.csv      : per model mean/std over the 5 repeats
- model_selection_summary.csv    : legality + R1-R5 + docking joined
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parent
DOCK_DIR = EXP / "docking"
DEFAULT_REPEATS = ("rep4", "rep5", "rep6")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repeats",
        nargs="+",
        default=list(DEFAULT_REPEATS),
        help="Repeat ids to include (default: rep4 rep5 rep6, matching 3-repeat protocol).",
    )
    return parser.parse_args()


def _load_docking() -> pd.DataFrame:
    """Load per-repeat docking metrics from each run_summary.json."""
    rows: list[dict[str, object]] = []
    for summary_path in sorted(DOCK_DIR.glob("*/*/run_summary.json")):
        payload = json.loads(summary_path.read_text())
        scores = pd.read_csv(summary_path.parent / "docking_results.csv")
        column = "score" if "score" in scores.columns else scores.columns[-1]
        values = pd.to_numeric(scores[column], errors="coerce").dropna().to_numpy()
        rows.append(
            {
                "repeat": summary_path.parents[1].name,
                "model": payload["model"],
                "total_ligands": payload.get("total_ligands", len(values)),
                "success_count": payload.get("success_count", len(values)),
                "success_rate": payload.get("success_rate", np.nan),
                "score_mean": float(np.mean(values)) if len(values) else np.nan,
                "score_median": float(np.median(values)) if len(values) else np.nan,
                "score_best_min": float(np.min(values)) if len(values) else np.nan,
                "score_p05": float(np.quantile(values, 0.05)) if len(values) else np.nan,
                "best_1pct_mean": (
                    float(np.mean(np.sort(values)[: max(1, len(values) // 100)]))
                    if len(values)
                    else np.nan
                ),
                "rate_le_-10": float(np.mean(values <= -10.0)) if len(values) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Build and persist the 500-step benchmark summary tables."""
    args = parse_args()
    per_repeat = _load_docking()
    per_repeat = per_repeat[per_repeat["repeat"].isin(args.repeats)].copy()
    per_repeat.to_csv(EXP / "docking_summary_by_repeat.csv", index=False)
    print(f"repeats={args.repeats} rows={len(per_repeat)}")

    grouped = per_repeat.groupby("model")
    model_summary = grouped.agg(
        repeats=("score_mean", "size"),
        total_ligands_mean=("total_ligands", "mean"),
        success_rate_mean=("success_rate", "mean"),
        score_mean_mean=("score_mean", "mean"),
        score_mean_std=("score_mean", "std"),
        score_median_mean=("score_median", "mean"),
        score_best_min_mean=("score_best_min", "mean"),
        score_p05_mean=("score_p05", "mean"),
        best_1pct_mean_mean=("best_1pct_mean", "mean"),
        **{"rate_le_-10_mean": ("rate_le_-10", "mean")},
    ).reset_index()

    # Legality from the existing completed-comparison ledger.
    ledger = pd.read_csv(
        EXP.parent / "five_model_completed_comparison_260827" / "per_repeat_summary.csv"
    )
    legality = (
        ledger[
            (ledger["step"] == 500) & (ledger["repeat"].isin(args.repeats))
        ]
        .groupby("model")
        .agg(
            valid_per_attempt_mean=("valid_per_attempt", "mean"),
            valid_per_attempt_std=("valid_per_attempt", "std"),
        )
        .reset_index()
    )
    rules = pd.read_csv(EXP / "model_r1r5_rates_500steps.csv")

    joined = model_summary.merge(legality, on="model", how="left").merge(
        rules, on="model", how="left"
    )
    joined = joined.sort_values("score_mean_mean")
    joined.to_csv(EXP / "model_selection_summary.csv", index=False)

    print("=== 500-step docking model summary ===")
    print(model_summary.sort_values("score_mean_mean").to_string(index=False))
    print()
    print("=== joined selection table (sorted by docking mean) ===")
    show = [
        "model",
        "valid_per_attempt_mean",
        "r1_large_ring_violation",
        "r2_ring_system_violation",
        "r5_carbon_fraction_violation",
        "r1r4_any",
        "score_mean_mean",
    ]
    print(joined[show].to_string(index=False))


if __name__ == "__main__":
    main()
