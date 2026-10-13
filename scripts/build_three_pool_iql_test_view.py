#!/usr/bin/env python3
"""Build reproducible High/Medium/Low test pools and a minimal shard view."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def read_manifests(run_dir: Path) -> dict[tuple[int, int], dict[str, object]]:
    result: dict[tuple[int, int], dict[str, object]] = {}
    for path in sorted((run_dir / "manifests").glob("rank_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("status") == "completed" and int(row["shard_index"]) < 1250:
                result[(int(row["rank"]), int(row["shard_index"]))] = row
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--high", type=Path, required=True)
    ap.add_argument("--rnd53-dir", type=Path, required=True)
    ap.add_argument("--all-valid-facts", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--low-count", type=int, default=419)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    high = pd.read_csv(args.high, low_memory=False).copy()
    high["quality_pool"] = "high"; high["test_reward"] = 1.0
    all_pairs = pd.concat([pd.read_csv(path, low_memory=False) for path in sorted(args.rnd53_dir.glob("RND-*_nearest100.csv.gz"))], ignore_index=True)
    near = all_pairs.sort_values(["trajectory_id", "ecfp4_distance", "epi_id"], kind="mergesort").drop_duplicates("trajectory_id").copy()
    high_ids = set(high.trajectory_id.astype(str))
    medium = near[~near.trajectory_id.astype(str).isin(high_ids)].copy()
    medium["quality_pool"] = "medium"; medium["test_reward"] = 0.0
    valid = pd.read_csv(args.all_valid_facts, low_memory=False)
    excluded = set(near.trajectory_id.astype(str))
    low_candidates = valid[~valid.trajectory_id.astype(str).isin(excluded)].copy()
    low_candidates = low_candidates.sort_values("trajectory_id", kind="mergesort").head(args.low_count).copy()
    low_candidates["quality_pool"] = "low"; low_candidates["test_reward"] = -1.0
    keep = ["trajectory_id", "quality_pool", "test_reward", "rank", "shard_index", "mol_index", "trajectory_shard", "canonical_smiles", "rdkit_valid", "qed", "sa_raw", "logp", "tpsa", "brenk_count", "ecfp4_distance"]
    for frame in (high, medium, low_candidates):
        for column in keep:
            if column not in frame:
                frame[column] = None
    pools = pd.concat([high[keep], medium[keep], low_candidates[keep]], ignore_index=True)
    if pools.trajectory_id.duplicated().any():
        raise ValueError("pool trajectory IDs must be disjoint")
    expected = {"high": 419, "medium": 1660, "low": args.low_count}
    actual = pools.quality_pool.value_counts().to_dict()
    if actual != expected:
        raise ValueError(f"unexpected pool counts {actual}, expected {expected}")
    manifests = read_manifests(args.run_dir)
    selected: dict[int, list[dict[str, object]]] = {rank: [] for rank in range(8)}
    for rank, shard_index in sorted(set(zip(pools["rank"].astype(int), pools["shard_index"].astype(int)))):
        row = manifests.get((rank, shard_index))
        if row is None:
            raise ValueError(f"missing original manifest row rank={rank} shard={shard_index}")
        selected[rank].append(row)
    view_manifest_dir = args.output_dir / "run_view" / "manifests"
    view_manifest_dir.mkdir(parents=True, exist_ok=True)
    for rank, rows in selected.items():
        rows.sort(key=lambda row: int(row["shard_index"]))
        (view_manifest_dir / f"rank_{rank:02d}.jsonl").write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    shards_link = args.output_dir / "run_view" / "shards"
    if not shards_link.exists():
        shards_link.symlink_to((args.run_dir / "shards").resolve(), target_is_directory=True)
    pools.sort_values(["quality_pool", "trajectory_id"], kind="mergesort").to_csv(args.output_dir / "three_pool_candidates.csv.gz", index=False, compression="gzip")
    (args.output_dir / "pool_ids.json").write_text(json.dumps({name: sorted(pools.loc[pools.quality_pool.eq(name), "trajectory_id"].astype(str)) for name in ("high", "medium", "low")}, indent=2) + "\n", encoding="utf-8")
    summary = {"status": "complete", "selection_order": "deterministic trajectory_id order; no random sampling", "pool_counts": actual, "unique_episodes": int(len(pools)), "selected_shards": int(sum(len(rows) for rows in selected.values())), "boundary": "shards are selected only from original rank/shard indices 0..1249"}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
