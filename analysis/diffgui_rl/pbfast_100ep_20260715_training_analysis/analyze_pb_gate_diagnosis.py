#!/usr/bin/env python3
"""Diagnose PB-fast failure modes and compare PB-gated and ungated endpoints."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
PB_RUN = ROOT / "runs/diffgui-deppa-pbfast-cpu4-lr1e5-wd1e4-b8x7-t200-100ep_20260715_2018"
NO_PB_RUN = ROOT / "runs/diffgui-deppa-full-lr1e5-wd1e4-b8x8-t200-100ep_20260626_165809"
NO_PB_ENDPOINT_DIR = ROOT / "analysis/diffgui_deppa_200_vs_1000_training_posebusters_20260710/posebusters_per_epoch"
WORK_RECORD_DIR = ROOT / "work_records/26-7/260716"

LOGGER = logging.getLogger("pb_gate_diagnosis")
WINDOWS = {
    "early_01_10": (1, 10),
    "middle_46_55": (46, 55),
    "late_91_100": (91, 100),
}
DISPLAY_CHECKS = [
    "pb_check_all_atoms_connected",
    "pb_check_bond_lengths",
    "pb_check_bond_angles",
    "pb_check_internal_steric_clash",
    "pb_check_minimum_distance_to_protein",
]
CHECK_LABELS = {
    "pb_check_all_atoms_connected": "All atoms connected",
    "pb_check_bond_lengths": "Bond lengths",
    "pb_check_bond_angles": "Bond angles",
    "pb_check_internal_steric_clash": "Internal steric clash",
    "pb_check_minimum_distance_to_protein": "Protein minimum distance",
}


def _setup_logging() -> None:
    """Configure file and terminal logging."""
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    file_handler = logging.FileHandler(OUT_DIR / "pb_gate_diagnosis.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)


def _load_pb_rows() -> pd.DataFrame:
    """Load all rank-local PoseBusters rows and attach their epoch."""
    frames: list[pd.DataFrame] = []
    for path in sorted((PB_RUN / "epochs").glob("epoch_*/rank_*/posebusters.csv")):
        frame = pd.read_csv(path)
        frame["epoch"] = int(path.parents[1].name.removeprefix("epoch_"))
        frames.append(frame)
    rows = pd.concat(frames, ignore_index=True)
    # Reconstruction failures are absent from rank-local PoseBusters CSVs;
    # the authoritative all-attempt denominator remains training_summary.csv.
    if len(rows) < 5_000:
        raise RuntimeError(f"unexpectedly few PB rows, found {len(rows)}")
    LOGGER.info("loaded online PoseBusters rows=%d", len(rows))
    return rows


def _failure_by_window(rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate individual PB check failures in the three analysis windows."""
    output: list[dict[str, object]] = []
    check_columns = [column for column in rows.columns if column.startswith("pb_check_")]
    for window, (start, end) in WINDOWS.items():
        part = rows[rows["epoch"].between(start, end) & rows["pb_evaluated"].eq(1)]
        for check in check_columns:
            known = part[part[check].isin([0, 1])]
            failures = int(known[check].eq(0).sum())
            output.append(
                {
                    "window": window,
                    "start_epoch": start,
                    "end_epoch": end,
                    "check": check.removeprefix("pb_check_"),
                    "evaluated": int(len(known)),
                    "failures": failures,
                    "failure_rate": failures / len(known) if len(known) else float("nan"),
                }
            )
    result = pd.DataFrame(output)
    result.to_csv(OUT_DIR / "pb_failure_by_window.csv", index=False)
    return result


def _old_endpoint_fast_pass(epoch: int) -> tuple[int, int, dict[str, int]]:
    """Re-evaluate an old full-dock table with the current 21 fast checks."""
    old_path = NO_PB_ENDPOINT_DIR / f"diffgui_t200_epoch_{epoch:04d}_posebusters.csv"
    old_rows = pd.read_csv(old_path)
    current_path = next((PB_RUN / "epochs/epoch_0001").glob("rank_*/posebusters.csv"))
    current_columns = pd.read_csv(current_path, nrows=1).columns
    required = [column.removeprefix("pb_check_") for column in current_columns if column.startswith("pb_check_")]
    normalized_old = {column.replace("-", "_"): column for column in old_rows.columns}
    mapped = [normalized_old[check] for check in required]
    passed = old_rows[mapped].fillna(False).astype(bool).all(axis=1)
    failures = {
        check: int((~old_rows[normalized_old[check]].fillna(False).astype(bool)).sum())
        for check in required
    }
    return int(len(old_rows)), int(passed.sum()), failures


def _endpoint_comparison(rows: pd.DataFrame) -> pd.DataFrame:
    """Compare epoch 1 and late endpoints under the same fast PB checks."""
    pb_summary = pd.read_csv(PB_RUN / "training_summary.csv").set_index("epoch")
    no_pb_summary = pd.read_csv(NO_PB_RUN / "training_summary.csv").set_index("epoch")
    result: list[dict[str, object]] = []
    for label, run_type, epoch, attempts in [
        ("PB epoch 1", "pb_gated", 1, 56),
        ("PB epoch 89", "pb_gated", 89, 56),
    ]:
        summary = pb_summary.loc[epoch]
        evaluated = int(summary["posebusters_pb_evaluated"])
        passed = int(summary["posebusters_pb_valid"])
        result.append(
            {
                "label": label,
                "run_type": run_type,
                "epoch": epoch,
                "attempts": attempts,
                "pb_evaluated": evaluated,
                "pb_valid": passed,
                "pb_valid_rate_evaluated": passed / evaluated,
                "pb_valid_rate_attempts": passed / attempts,
                "docking_raw_mean": float(summary["docking_raw_mean"]),
                "ppo_approx_kl": float(summary["PPO_ApproxKL"]),
            }
        )
    for label, epoch in [("No PB epoch 1", 1), ("No PB epoch 89", 89)]:
        evaluated, passed, _ = _old_endpoint_fast_pass(epoch)
        summary = no_pb_summary.loc[epoch]
        result.append(
            {
                "label": label,
                "run_type": "no_pb_gate",
                "epoch": epoch,
                "attempts": 64,
                "pb_evaluated": evaluated,
                "pb_valid": passed,
                "pb_valid_rate_evaluated": passed / evaluated,
                "pb_valid_rate_attempts": passed / 64,
                "docking_raw_mean": float(summary["docking_raw_mean"]),
                "ppo_approx_kl": float(summary["PPO_ApproxKL"]),
            }
        )
    comparison = pd.DataFrame(result)
    comparison.to_csv(OUT_DIR / "pb_vs_nopb_endpoint_comparison.csv", index=False)
    return comparison


def _plot(failures: pd.DataFrame, comparison: pd.DataFrame) -> None:
    """Plot failure-mode drift and same-check endpoint PB-valid yields."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    colors = ["#4C78A8", "#F2CF5B", "#E45756"]
    x = range(len(DISPLAY_CHECKS))
    width = 0.24
    for index, (window, _) in enumerate(WINDOWS.items()):
        part = failures[failures["window"].eq(window)].set_index("check")
        rates = [100 * float(part.loc[check.removeprefix("pb_check_"), "failure_rate"]) for check in DISPLAY_CHECKS]
        axes[0].bar([value + (index - 1) * width for value in x], rates, width, label=window, color=colors[index])
    axes[0].set_xticks(list(x), [CHECK_LABELS[check] for check in DISPLAY_CHECKS], rotation=25, ha="right")
    axes[0].set_ylabel("Failure rate among PB-evaluated molecules (%)")
    axes[0].set_title("PB-fast geometry failures increase during PPO")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    ordered = comparison.set_index("label").loc[["PB epoch 1", "No PB epoch 1", "PB epoch 89", "No PB epoch 89"]]
    rates = 100 * ordered["pb_valid_rate_attempts"]
    bars = axes[1].bar(range(len(ordered)), rates, color=["#4C78A8", "#9D755D", "#4C78A8", "#9D755D"])
    axes[1].set_xticks(range(len(ordered)), ordered.index, rotation=20, ha="right")
    axes[1].set_ylabel("PB-valid / all generation attempts (%)")
    axes[1].set_title("Same 21 fast checks: PB gate preserves late legality")
    axes[1].set_ylim(0, 100)
    axes[1].grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, rates):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.1f}%", ha="center", fontsize=9)

    fig.tight_layout()
    png_path = OUT_DIR / "10_pb_gate_geometry_diagnosis.png"
    svg_path = OUT_DIR / "10_pb_gate_geometry_diagnosis.svg"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    shutil.copy2(png_path, WORK_RECORD_DIR / png_path.name)
    LOGGER.info("wrote figure=%s", png_path)


def main() -> None:
    """Run the complete PB-gate diagnosis."""
    _setup_logging()
    rows = _load_pb_rows()
    failures = _failure_by_window(rows)
    comparison = _endpoint_comparison(rows)
    _plot(failures, comparison)
    LOGGER.info("analysis complete failure_rows=%d comparison_rows=%d", len(failures), len(comparison))


if __name__ == "__main__":
    main()
