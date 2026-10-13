#!/usr/bin/env python3
"""Validate fixed-attempt completion of a sharded DiffSBDD trajectory run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """Parse validation arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-ranks", type=int, required=True)
    parser.add_argument("--expected-shards-per-rank", type=int, required=True)
    parser.add_argument("--attempts-per-shard", type=int, required=True)
    parser.add_argument("--timesteps", type=int, required=True)
    parser.add_argument("--max-n-nodes", type=int, required=True)
    parser.add_argument("--latent-dim", type=int, default=13)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """Check every manifest row and write a machine-readable completion report."""
    args = parse_args()
    run_dir = args.run_dir.resolve()
    expected_total_shards = args.expected_ranks * args.expected_shards_per_rank
    expected_attempts = expected_total_shards * args.attempts_per_shard
    rank_rows: list[dict[str, Any]] = []
    errors: list[str] = []

    for rank in range(args.expected_ranks):
        manifest_path = run_dir / "manifests" / f"rank_{rank:02d}.jsonl"
        records = _read_jsonl(manifest_path)
        rank_attempts = 0
        rank_bytes = 0
        rank_obabel = 0
        rank_valid = 0
        for expected_index, record in enumerate(records):
            if int(record.get("shard_index", -1)) != expected_index:
                errors.append(f"rank {rank}: non-contiguous shard at row {expected_index}")
            if int(record.get("rank", -1)) != rank or int(record.get("seed", -1)) != rank + 1:
                errors.append(f"rank {rank}: rank/seed mismatch at shard {expected_index}")
            if int(record.get("attempts", -1)) != args.attempts_per_shard:
                errors.append(f"rank {rank}: attempts mismatch at shard {expected_index}")
            if int(record.get("num_timesteps", -1)) != args.timesteps:
                errors.append(f"rank {rank}: timestep mismatch at shard {expected_index}")
            expected_latent = [
                args.attempts_per_shard,
                args.timesteps + 2,
                args.max_n_nodes,
                args.latent_dim,
            ]
            expected_logps = [args.attempts_per_shard, args.timesteps + 1]
            if record.get("latent_shape") != expected_latent:
                errors.append(f"rank {rank}: latent shape mismatch at shard {expected_index}")
            if record.get("logps_shape") != expected_logps:
                errors.append(f"rank {rank}: logps shape mismatch at shard {expected_index}")
            shard_path = run_dir / str(record.get("shard_path", ""))
            if not shard_path.exists():
                errors.append(f"rank {rank}: missing shard {shard_path}")
            elif shard_path.stat().st_size != int(record.get("shard_bytes", -1)):
                errors.append(f"rank {rank}: shard size mismatch {shard_path}")
            rank_attempts += int(record.get("attempts", 0))
            rank_bytes += int(record.get("shard_bytes", 0))
            rank_obabel += int(record.get("obabel_reconstructed", 0))
            rank_valid += int(record.get("rdkit_valid", 0))

        boundary_hashes_verified = 0
        for record in ([records[0], records[-1]] if records else []):
            shard_path = run_dir / str(record["shard_path"])
            if shard_path.exists() and _sha256(shard_path) == str(record.get("shard_sha256")):
                boundary_hashes_verified += 1
            else:
                errors.append(f"rank {rank}: boundary shard hash mismatch {shard_path}")
        rank_rows.append(
            {
                "rank": rank,
                "seed": rank + 1,
                "completed_shards": len(records),
                "fixed_attempts": rank_attempts,
                "obabel_reconstructed": rank_obabel,
                "rdkit_valid": rank_valid,
                "shard_bytes": rank_bytes,
                "boundary_hashes_verified": boundary_hashes_verified,
            }
        )

    completed_shards = sum(row["completed_shards"] for row in rank_rows)
    fixed_attempts = sum(row["fixed_attempts"] for row in rank_rows)
    manifest_complete = (
        completed_shards == expected_total_shards
        and fixed_attempts == expected_attempts
        and all(
            row["completed_shards"] == args.expected_shards_per_rank for row in rank_rows
        )
    )
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
    summary_complete = bool(
        summary
        and summary.get("status") == "completed"
        and int(summary.get("expected_fixed_attempts", -1)) == expected_attempts
        and int(summary.get("attempts_total", -1)) == expected_attempts
    )
    complete = manifest_complete and summary_complete and not errors
    report = {
        "status": "complete" if complete else "incomplete",
        "run_dir": str(run_dir),
        "expected_fixed_attempts": expected_attempts,
        "fixed_attempts": fixed_attempts,
        "expected_shards": expected_total_shards,
        "completed_shards": completed_shards,
        "obabel_reconstructed": sum(row["obabel_reconstructed"] for row in rank_rows),
        "rdkit_valid": sum(row["rdkit_valid"] for row in rank_rows),
        "trajectory_bytes": sum(row["shard_bytes"] for row in rank_rows),
        "summary_present": summary is not None,
        "summary_complete": summary_complete,
        "errors": errors,
        "ranks": rank_rows,
    }
    output = run_dir / "completion_report.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not complete and not args.allow_incomplete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
