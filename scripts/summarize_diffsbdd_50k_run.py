"""Summarize a completed multi-rank DiffSBDD generation run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-ranks", type=int, required=True)
    parser.add_argument("--expected-shards-per-rank", type=int, required=True)
    parser.add_argument("--attempts-per-shard", type=int, required=True)
    parser.add_argument("--timesteps", type=int, required=True)
    args = parser.parse_args()

    rank_summaries = []
    for rank in range(args.expected_ranks):
        path = args.run_dir / "ranks" / f"rank_{rank:02d}" / "rank_summary.json"
        if path.exists():
            rank_summaries.append(json.loads(path.read_text(encoding="utf-8")))
    expected_attempts = (
        args.expected_ranks * args.expected_shards_per_rank * args.attempts_per_shard
    )
    fixed_attempts = sum(int(row["fixed_attempts"]) for row in rank_summaries)
    reconstructed = sum(int(row["reconstructed"]) for row in rank_summaries)
    complete = (
        len(rank_summaries) == args.expected_ranks
        and fixed_attempts == expected_attempts
        and all(
            int(row["completed_shards"]) == args.expected_shards_per_rank
            and int(row["timesteps"]) == args.timesteps
            and int(row["seed"]) == int(row["rank"]) + 1
            for row in rank_summaries
        )
    )
    aggregate_path = args.run_dir / "aggregated" / "diffsbdd_50k.summary.json"
    aggregate = (
        json.loads(aggregate_path.read_text(encoding="utf-8"))
        if aggregate_path.exists()
        else None
    )
    summary = {
        "status": "complete" if complete else "incomplete",
        "expected_fixed_attempts": expected_attempts,
        "fixed_attempts": fixed_attempts,
        "reconstructed": reconstructed,
        "reconstruction_failures": fixed_attempts - reconstructed,
        "timesteps": args.timesteps,
        "determinism_scope": (
            "rank seeds and per-shard RNG continuation are fixed; CUDA kernels may not "
            "be bitwise deterministic across processes"
        ),
        "rank_seeds": [int(row["seed"]) for row in rank_summaries],
        "completed_ranks": len(rank_summaries),
        "expected_ranks": args.expected_ranks,
        "rank_summaries": rank_summaries,
        "aggregate": aggregate,
    }
    output = args.run_dir / "summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
