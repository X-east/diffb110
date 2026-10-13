#!/usr/bin/env python3
"""Summarize the five-model generation funnel across three fixed-attempt repeats."""

from __future__ import annotations

import logging
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/diffb110_funnel_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOGGER = logging.getLogger("generation_funnel_repeat_summary")
ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
FORMAL_DIR = (
    ROOT
    / "analysis/benchmark/benchmark5"
    / "8gjl_c1101_benchmark_20260706_1000steps_3rep_formal"
)
RAW_COUNTS_PATH = FORMAL_DIR / "formal_3rep_per_repeat_counts.csv"
PB_SUMMARY_PATH = FORMAL_DIR / "posebusters_queue/posebusters_summary.csv"
QVINA_ANNOTATED_PATH = (
    FORMAL_DIR
    / "qvina_score_only_20260721/results/per_ligand_scores_annotated.csv"
)
DOCKING_REPEAT_PATH = FORMAL_DIR / "formal_3rep_docking_per_repeat.csv"

MODEL_ORDER = ["DiffGui", "DiffSBDD", "PMDM", "GCDM-SBDD", "TargetDiff"]
REPEAT_ORDER = ["rep1", "rep2", "rep3"]
MODEL_COLORS = {
    "DiffGui": "#3B82C4",
    "DiffSBDD": "#D95F59",
    "PMDM": "#D9A441",
    "GCDM-SBDD": "#8E78B7",
    "TargetDiff": "#3D9B72",
}
STAGES = [
    ("sdf_written", "SDF written"),
    ("sanitize_valid", "Sanitize-valid"),
    ("unique_smiles", "Unique SMILES"),
    ("pb_valid", "PB-valid"),
    ("qvina_raw_pose_success", "QVina raw-pose success"),
    ("full_redocking_success", "UniDock full-redocking success"),
]
EXPECTED_TOTALS = {
    "DiffGui": [2641, 2641, 2610, 2522, 2641, 2641],
    "DiffSBDD": [3000, 1999, 1999, 1309, 1805, 1999],
    "PMDM": [1726, 1726, 1726, 1359, 1726, 1726],
    "GCDM-SBDD": [3000, 1570, 1557, 895, 1570, 1570],
    "TargetDiff": [2974, 2974, 2972, 2182, 2931, 2974],
}


def _configure_logging() -> None:
    """Configure terminal and durable file logging."""
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    for handler in [
        logging.StreamHandler(),
        logging.FileHandler(
            OUTPUT_DIR / "generation_funnel_repeat_summary.log",
            mode="w",
            encoding="utf-8",
        ),
    ]:
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)


def _configure_plotting() -> None:
    """Apply presentation-readable plotting defaults."""
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 12,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _load_repeat_counts() -> pd.DataFrame:
    """Merge the six repeat-level funnel outcomes from their authoritative files."""
    raw = pd.read_csv(
        RAW_COUNTS_PATH,
        usecols=[
            "repeat",
            "model",
            "attempts",
            "sdf_written",
            "sanitize_valid",
            "unique_smiles",
        ],
    )
    pb = pd.read_csv(
        PB_SUMMARY_PATH,
        usecols=["repeat", "model", "pb_valid_count"],
    ).rename(columns={"pb_valid_count": "pb_valid"})

    qvina = pd.read_csv(
        QVINA_ANNOTATED_PATH,
        usecols=[
            "repeat",
            "model",
            "is_reference",
            "fragment_reduced",
            "success",
        ],
    )
    qvina = qvina[
        ~qvina["is_reference"].eq(True)
        & qvina["success"].eq(True)
        & ~qvina["fragment_reduced"].eq(True)
    ]
    qvina_counts = (
        qvina.groupby(["repeat", "model"], sort=False)
        .size()
        .rename("qvina_raw_pose_success")
        .reset_index()
    )

    docking = pd.read_csv(
        DOCKING_REPEAT_PATH,
        usecols=["repeat", "model", "success_count"],
    ).rename(columns={"success_count": "full_redocking_success"})

    merged = raw.merge(pb, on=["repeat", "model"], validate="one_to_one")
    merged = merged.merge(
        qvina_counts,
        on=["repeat", "model"],
        validate="one_to_one",
    )
    merged = merged.merge(
        docking,
        on=["repeat", "model"],
        validate="one_to_one",
    )
    merged = merged[
        merged["model"].isin(MODEL_ORDER) & merged["repeat"].isin(REPEAT_ORDER)
    ].copy()
    merged["model"] = pd.Categorical(
        merged["model"],
        categories=MODEL_ORDER,
        ordered=True,
    )
    merged["repeat"] = pd.Categorical(
        merged["repeat"],
        categories=REPEAT_ORDER,
        ordered=True,
    )
    merged = merged.sort_values(["model", "repeat"]).reset_index(drop=True)

    if len(merged) != 15:
        raise ValueError(f"expected 15 model-repeat rows, found {len(merged)}")
    if not merged["attempts"].eq(1000).all():
        raise ValueError("every repeat must use exactly 1,000 fixed attempts")
    for model, expected in EXPECTED_TOTALS.items():
        observed = [
            int(merged.loc[merged["model"].eq(model), stage].sum())
            for stage, _ in STAGES
        ]
        if observed != expected:
            raise ValueError(
                f"{model} funnel totals differ from the audited table: "
                f"observed={observed}, expected={expected}"
            )
    return merged


def _build_long_tables(
    repeat_counts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build repeat-level and mean-plus-sample-SD long-form tables."""
    repeat_rows: list[dict[str, object]] = []
    for row in repeat_counts.itertuples(index=False):
        for stage, stage_label in STAGES:
            count = int(getattr(row, stage))
            repeat_rows.append(
                {
                    "model": str(row.model),
                    "repeat": str(row.repeat),
                    "attempts": int(row.attempts),
                    "stage": stage,
                    "stage_label": stage_label,
                    "success_count": count,
                    "success_rate": count / int(row.attempts),
                }
            )
    by_repeat = pd.DataFrame(repeat_rows)
    summary = (
        by_repeat.groupby(
            ["model", "stage", "stage_label"],
            sort=False,
            observed=True,
        )
        .agg(
            repeats=("repeat", "nunique"),
            attempts_total=("attempts", "sum"),
            success_count_total=("success_count", "sum"),
            success_rate_mean=("success_rate", "mean"),
            success_rate_std=("success_rate", "std"),
        )
        .reset_index()
    )
    return by_repeat, summary


def _save_figure(fig: plt.Figure, stem: Path) -> None:
    """Save one static figure in preview, vector, and print formats."""
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def _plot_summary(summary: pd.DataFrame) -> None:
    """Plot six repeat-level success-rate comparisons with sample-SD error bars."""
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.4), sharey=True)
    positions = np.arange(len(MODEL_ORDER))
    colors = [MODEL_COLORS[model] for model in MODEL_ORDER]

    for axis, (stage, stage_label) in zip(axes.flat, STAGES):
        frame = (
            summary[summary["stage"].eq(stage)]
            .set_index("model")
            .reindex(MODEL_ORDER)
        )
        means = frame["success_rate_mean"].to_numpy(dtype=float) * 100.0
        errors = frame["success_rate_std"].to_numpy(dtype=float) * 100.0
        bars = axis.bar(
            positions,
            means,
            yerr=errors,
            color=colors,
            width=0.72,
            capsize=4,
            error_kw={"elinewidth": 1.2, "capthick": 1.2},
        )
        for bar, mean, error in zip(bars, means, errors):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                mean + error + 1.3,
                f"{mean:.1f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
        axis.set_title(stage_label, loc="left")
        axis.set_xticks(positions, MODEL_ORDER, rotation=24, ha="right")
        axis.set_ylim(0, 108)
        axis.grid(axis="y", color="#D5D8DC", linewidth=0.7, alpha=0.7)
        axis.set_axisbelow(True)

    axes[0, 0].set_ylabel("Success rate per 1,000 attempts (%)")
    axes[1, 0].set_ylabel("Success rate per 1,000 attempts (%)")
    fig.text(
        0.5,
        0.005,
        "Mean ± sample SD across three independent fixed-attempt batches (n=3). "
        "QVina excludes records requiring largest-fragment reduction.",
        ha="center",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    _save_figure(fig, OUTPUT_DIR / "generation_legality_funnel_repeat_rates")


def main() -> None:
    """Build repeat-level funnel tables and the corresponding uncertainty plot."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _configure_logging()
    _configure_plotting()
    repeat_counts = _load_repeat_counts()
    by_repeat, summary = _build_long_tables(repeat_counts)
    repeat_counts.to_csv(
        OUTPUT_DIR / "generation_funnel_counts_by_repeat.csv",
        index=False,
    )
    by_repeat.to_csv(
        OUTPUT_DIR / "generation_funnel_rates_by_repeat.csv",
        index=False,
    )
    summary.to_csv(
        OUTPUT_DIR / "generation_funnel_repeat_summary.csv",
        index=False,
    )
    _plot_summary(summary)
    LOGGER.info("validated 5 models x 3 repeats x 6 outcomes")
    LOGGER.info("repeat summary:\n%s", summary.to_string(index=False))
    LOGGER.info("completed successfully")


if __name__ == "__main__":
    main()
