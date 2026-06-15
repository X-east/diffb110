"""Combine DiffSBDD-xhe edge metric CSV files into one comparison table."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
INPUTS = [
    ("1epoch_b4", ROOT / "eval_edge_metrics" / "edge_metrics_summary.csv"),
    ("3epoch_b2_last", ROOT / "eval_edge_metrics_b2_3ep" / "edge_metrics_summary.csv"),
    ("3epoch_b2_best", ROOT / "eval_edge_metrics_b2_3ep_best" / "edge_metrics_summary.csv"),
    (
        "3epoch_b4_single_forward",
        ROOT / "eval_edge_metrics_b4_3ep_single_forward" / "edge_metrics_summary.csv",
    ),
]
OUTPUT_CSV = ROOT / "combined_edge_metrics.csv"
OUTPUT_PNG = ROOT / "combined_edge_metrics_val.png"


def _read_rows(label: str, path: Path) -> list[dict[str, str]]:
    """Read one metric CSV and annotate rows with run label."""
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["run"] = label
    return rows


def main() -> None:
    """Write combined CSV and a validation-metric plot."""
    rows: list[dict[str, str]] = []
    seen_random_splits: set[str] = set()
    for label, path in INPUTS:
        for row in _read_rows(label, path):
            if row["model"] == "warm_start_random_edge":
                if row["split"] in seen_random_splits:
                    continue
                row["run"] = "baseline"
                seen_random_splits.add(row["split"])
            rows.append(row)

    fieldnames = ["run", *[field for field in rows[0] if field != "run"]]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    val_rows = [row for row in rows if row["split"] == "val"]
    labels = [row["run"] for row in val_rows]
    metrics = ["edge_acc", "real_bond_acc", "binary_bond_f1"]
    x_positions = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for offset, metric in enumerate(metrics):
        values = [float(row[metric]) for row in val_rows]
        ax.bar(x_positions + (offset - 1) * width, values, width, label=metric)
    ax.set_ylim(0, 1)
    ax.set_ylabel("validation score")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=180)
    plt.close(fig)

    print(f"wrote {OUTPUT_CSV}")
    print(f"wrote {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
