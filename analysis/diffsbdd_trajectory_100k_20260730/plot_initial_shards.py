#!/usr/bin/env python3
"""Summarize and plot the first completed shard from every collection rank."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    """Parse input and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Read rank manifests and emit a CSV plus a two-panel PNG."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int]] = []
    for manifest in sorted((args.run_dir / "manifests").glob("rank_*.jsonl")):
        first_line = manifest.read_text(encoding="utf-8").splitlines()[0]
        record = json.loads(first_line)
        attempts = int(record["attempts"])
        rows.append(
            {
                "rank": int(record["rank"]),
                "seed": int(record["seed"]),
                "fixed_attempts": attempts,
                "obabel_reconstructed": int(record["obabel_reconstructed"]),
                "rdkit_valid": int(record["rdkit_valid"]),
                "rdkit_valid_rate": float(record["rdkit_valid"]) / attempts,
                "elapsed_seconds": float(record["elapsed_seconds"]),
                "shard_bytes": int(record["shard_bytes"]),
                "peak_allocated_mb": float(record["torch_peak_allocated_mb"]),
            }
        )
    if len(rows) != 8:
        raise RuntimeError(f"Expected first-shard data for 8 ranks, found {len(rows)}")

    csv_path = args.output_dir / "initial_shard_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    ranks = [int(row["rank"]) for row in rows]
    valid_rates = [100.0 * float(row["rdkit_valid_rate"]) for row in rows]
    elapsed = [float(row["elapsed_seconds"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), constrained_layout=True)
    axes[0].bar(ranks, valid_rates, color="#2A9D8F", width=0.72)
    axes[0].axhline(sum(valid_rates) / len(valid_rates), color="#264653", linestyle="--")
    axes[0].set(title="RDKit-valid rate", xlabel="Rank (seed = rank + 1)", ylabel="Valid (%)")
    axes[0].set_xticks(ranks)
    axes[0].set_ylim(0, 100)

    axes[1].bar(ranks, elapsed, color="#E76F51", width=0.72)
    axes[1].axhline(sum(elapsed) / len(elapsed), color="#264653", linestyle="--")
    axes[1].set(title="First-shard generation time", xlabel="Rank", ylabel="Seconds / 10 attempts")
    axes[1].set_xticks(ranks)
    axes[1].set_ylim(0, max(elapsed) * 1.2)
    fig.suptitle("DiffSBDD T=1000, nodes 10-30: initial 8-rank evidence", fontsize=12)
    fig.savefig(args.output_dir / "initial_shard_metrics.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
