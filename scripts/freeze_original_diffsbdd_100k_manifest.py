#!/usr/bin/env python3
"""Freeze the completed original 100k collection manifests after an append run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest_paths(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\n")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shards-per-rank", type=int, default=1250)
    parser.add_argument("--world-size", type=int, default=8)
    args = parser.parse_args()
    if args.shards_per_rank <= 0 or args.world_size <= 0:
        raise ValueError("shards-per-rank and world-size must be positive")
    manifest_dir = args.run_dir / "manifests"
    out_dir = args.output_dir / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    frozen_paths: list[Path] = []
    summary_rows = []
    for rank in range(args.world_size):
        source = manifest_dir / f"rank_{rank:02d}.jsonl"
        if not source.is_file():
            raise FileNotFoundError(source)
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        selected = [row for row in rows if row.get("status") == "completed" and int(row["shard_index"]) < args.shards_per_rank]
        selected.sort(key=lambda row: int(row["shard_index"]))
        indices = [int(row["shard_index"]) for row in selected]
        expected = list(range(args.shards_per_rank))
        if indices != expected:
            raise ValueError(f"rank {rank} original shard indices are not exactly 0..{args.shards_per_rank - 1}")
        output = out_dir / source.name
        output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in selected), encoding="utf-8")
        frozen_paths.append(output)
        summary_rows.append({"rank": rank, "source_rows": len(rows), "frozen_original_rows": len(selected), "first_shard": indices[0], "last_shard": indices[-1]})
    manifest_sha256 = digest_paths(frozen_paths)
    summary = {
        "status": "complete",
        "source_run_dir": str(args.run_dir.resolve()),
        "world_size": args.world_size,
        "shards_per_rank": args.shards_per_rank,
        "fixed_attempts": args.world_size * args.shards_per_rank * 10,
        "manifest_snapshot_sha256": manifest_sha256,
        "rank_rows": summary_rows,
        "boundary": "Only completed original shards 0..1249 per rank are included; interrupted continuation shards are excluded.",
    }
    (args.output_dir / "frozen_original_100k_manifest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
