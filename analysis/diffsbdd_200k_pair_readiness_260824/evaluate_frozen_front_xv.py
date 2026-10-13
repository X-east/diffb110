#!/usr/bin/env python3
"""Summarize fresh-generation quality for the frozen-front x/v binary matrix."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MATRIX = ROOT / "analysis" / "diffsbdd_global_dpo_t1000_frozen_front_xv"


def _load_base() -> Any:
    """Load the sibling SDF summary helper."""
    spec = importlib.util.spec_from_file_location("pair_readiness", HERE / "build_pair_readiness_artifact.py")
    if spec is None or spec.loader is None:
        raise ImportError("cannot load pair-readiness helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    """Write matrix quality CSV and a compact validity/scaffold plot."""
    helper = _load_base()
    rows: list[dict[str, Any]] = []
    for variant in ("x_only", "x_dominant", "balanced", "v_dominant", "v_only"):
        for epoch in range(1, 6):
            sdf = MATRIX / variant / "generation" / f"epoch_{epoch:04d}" / "molecules.sdf"
            row = helper._sdf_summary(f"{variant}_epoch_{epoch}", sdf)
            row["variant"] = variant
            row["epoch"] = epoch
            rows.append(row)
    out = HERE / "frozen_front_xv_quality.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for variant in ("x_only", "x_dominant", "balanced", "v_dominant", "v_only"):
        subset = [row for row in rows if row["variant"] == variant]
        epochs = [row["epoch"] for row in subset]
        axes[0].plot(epochs, [row["valid_rate"] for row in subset], marker="o", label=variant)
        axes[1].plot(epochs, [row["murcko_scaffolds"] for row in subset], marker="o", label=variant)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("RDKit-valid rate")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("Murcko scaffolds / 1000 attempts")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle("Frozen-front x/v matrix: binary-label fresh generation", fontsize=14)
    fig.savefig(HERE / "frozen_front_xv_quality.png", dpi=180)
    fig.savefig(HERE / "frozen_front_xv_quality.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
