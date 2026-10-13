#!/usr/bin/env python3
"""Write a complete test sidecar and a train split limited to three pools."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--pools", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    from ddpo.offline_rl.dataset import load_trajectory_shard
    from ddpo.offline_rl.index import SplitAssignment, SplitManifest, load_episode_index, write_split_manifest
    from ddpo.offline_rl.reward_sidecar import (
        EpisodeRewardFacts,
        REWARD_SIDECAR_SCHEMA_VERSION,
        write_reward_sidecar,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    index = load_episode_index(args.index)
    pools = pd.read_csv(args.pools, low_memory=False)
    reward_by_id = dict(zip(pools.trajectory_id.astype(str), pools.test_reward.astype(float)))
    pool_by_id = dict(zip(pools.trajectory_id.astype(str), pools.quality_pool.astype(str)))
    reward_rows = []
    seen = set()
    for row in index.rows:
        if row.shard_path not in seen:
            seen.add(row.shard_path)
    shard_cache = {}
    for row in index.rows:
        key = (row.shard_path, row.source_shard_sha256)
        shard = shard_cache.get(key)
        if shard is None:
            shard = load_trajectory_shard(args.run_dir / row.shard_path, expected_sha256=row.source_shard_sha256)
            shard_cache[key] = shard
        facts = shard.all_episode_facts()[row.episode_index]
        reward = float(reward_by_id.get(row.sample_id, -1.0))
        reward_rows.append(EpisodeRewardFacts(
            sample_id=row.sample_id, source_trajectory_format="diffsbdd_frozen_policy_trajectory_v1",
            source_checkpoint_sha256=row.source_checkpoint_sha256, source_shard_sha256=row.source_shard_sha256,
            reward_schema_version=REWARD_SIDECAR_SCHEMA_VERSION, reward_version="three_pool_test_plus1_zero_minus1_v1",
            reconstruction_success=bool(row.rdkit_valid), sanitize_success=bool(row.rdkit_valid), rdkit_valid=bool(row.rdkit_valid),
            pb_evaluated=False, pb_valid=False, docking_input_success=False, docking_success=False, actual_good=False,
            reward_total_train=reward, failure_reason=row.failure_type,
        ))
    sidecar_path = write_reward_sidecar(args.output_dir / "reward_sidecar.csv", reward_rows)
    split_rows = [SplitAssignment(
        split_schema_version="diffsbdd_episode_split_v1", split_version="three_pool_balanced_test_v1",
        episode_index_sha256=index.sha256, sample_id=row.sample_id,
        split="train" if row.sample_id in pool_by_id else "validation",
        stratum=pool_by_id.get(row.sample_id, "non_pool"),
    ) for row in index.rows]
    split_path = write_split_manifest(args.output_dir / "split_manifest.jsonl", SplitManifest(split_rows))
    summary = {"status":"complete", "indexed_episodes":len(index.rows), "train_pool_counts":pools.quality_pool.value_counts().to_dict(), "train_episodes":int(sum(row.sample_id in pool_by_id for row in index.rows)), "validation_nonpool_episodes":int(sum(row.sample_id not in pool_by_id for row in index.rows)), "sidecar":str(sidecar_path), "split":str(split_path)}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
