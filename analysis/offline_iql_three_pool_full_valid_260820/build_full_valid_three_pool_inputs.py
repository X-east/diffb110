#!/usr/bin/env python3
"""Build the deterministic full-valid three-pool exploratory IQL inputs."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from ddpo.offline_rl.index import (
    SPLIT_MANIFEST_SCHEMA_VERSION,
    SplitAssignment,
    SplitManifest,
    load_episode_index,
    write_split_manifest,
)
from ddpo.offline_rl.reward_sidecar import (
    REWARD_SIDECAR_SCHEMA_VERSION,
    EpisodeRewardFacts,
    write_reward_sidecar,
)


ROOT = Path("/data/disk6/DiffB110")
INDEX_PATH = ROOT / "analysis/offline_iql_frozen_original_100k_260819/index/episode_index.jsonl"
OLD_POOL_PATH = ROOT / "analysis/offline_iql_three_pool_test_260819/pool_ids.json"
OUT = ROOT / "analysis/offline_iql_three_pool_full_valid_260820"
RUN_DIR = ROOT / "analysis/offline_iql_frozen_original_100k_260819/run_view"
TRAJECTORY_FORMAT = "diffsbdd_frozen_policy_trajectory_v1"
REWARD_VERSION = "three_pool_full_valid_plus1_zero_minus1_v1"


def sha256_file(path: Path) -> str:
    """Return a file SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """Build mutually exclusive pools, complete sidecar, and valid-only train split."""
    rows = [json.loads(line) for line in INDEX_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {str(row["sample_id"]): row for row in rows}
    old_pools = json.loads(OLD_POOL_PATH.read_text(encoding="utf-8"))
    high = {str(value) for value in old_pools["high"]}
    medium = {str(value) for value in old_pools["medium"]}
    valid = {str(row["sample_id"]) for row in rows if bool(row["rdkit_valid"])}
    all_ids = set(by_id)
    if not high or not medium or not high.isdisjoint(medium):
        raise ValueError("High and Medium must be non-empty and disjoint")
    if not (high | medium) <= valid:
        raise ValueError("High/Medium contain non-valid or unknown episodes")
    # Low contains every remaining fixed-attempt episode, including invalid
    # reconstructions, so invalid trajectories receive the same finite -1 label
    # and are actually exposed to the training sampler.
    low = all_ids - high - medium
    pools = {name: sorted(values) for name, values in (("high", high), ("medium", medium), ("low", low))}
    if len(set().union(*(set(values) for values in pools.values()))) != len(all_ids):
        raise ValueError("three pools do not cover all episodes exactly")

    output_training = OUT / "training_inputs"
    output_training.mkdir(parents=True, exist_ok=True)
    (OUT / "pool_ids.json").write_text(
        json.dumps(pools, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    reward_rows: list[EpisodeRewardFacts] = []
    for sample_id in sorted(by_id):
        row = by_id[sample_id]
        is_valid = bool(row["rdkit_valid"])
        reward = 1.0 if sample_id in high else 0.0 if sample_id in medium else -1.0
        if not is_valid:
            reward = -1.0
        reward_rows.append(
            EpisodeRewardFacts(
                sample_id=sample_id,
                source_trajectory_format=TRAJECTORY_FORMAT,
                source_checkpoint_sha256=str(row["source_checkpoint_sha256"]),
                source_shard_sha256=str(row["source_shard_sha256"]),
                reward_schema_version=REWARD_SIDECAR_SCHEMA_VERSION,
                reward_version=REWARD_VERSION,
                reconstruction_success=is_valid,
                sanitize_success=is_valid,
                rdkit_valid=is_valid,
                pb_evaluated=False,
                pb_valid=False,
                docking_input_success=False,
                docking_success=False,
                actual_good=False,
                reward_total_train=reward,
                failure_reason=str(row["failure_type"]),
            )
        )
    sidecar_path = write_reward_sidecar(output_training / "reward_sidecar.csv", reward_rows)

    # Split provenance uses the canonical EpisodeIndex digest, not the raw
    # JSONL file digest (which also includes formatting/newline choices).
    index_sha = load_episode_index(INDEX_PATH).sha256
    split_rows = [
        SplitAssignment(
            split_schema_version=SPLIT_MANIFEST_SCHEMA_VERSION,
            split_version="full_valid_train_invalid_test_exploratory_v1",
            episode_index_sha256=index_sha,
            sample_id=sample_id,
                split="train",
            stratum=f"pool={'high' if sample_id in high else 'medium' if sample_id in medium else 'low' if sample_id in low else 'invalid'}|valid={int(bool(by_id[sample_id]['rdkit_valid']))}",
        )
        for sample_id in sorted(by_id)
    ]
    split_manifest = SplitManifest(split_rows)
    write_split_manifest(output_training / "split_manifest.jsonl", split_manifest)

    pool_payload = json.dumps(pools, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    summary = {
        "status": "complete",
        "scope": "exploratory_full_valid_three_pool_training_inputs",
        "source_index": str(INDEX_PATH),
        "source_run_dir": str(RUN_DIR),
        "source_episode_index_sha256": index_sha,
        "source_episode_count": len(rows),
        "source_rdkit_valid_count": len(valid),
        "pool_counts": {name: len(values) for name, values in pools.items()},
        "pool_intersections": {"high_medium": len(high & medium), "high_low": len(high & low), "medium_low": len(medium & low)},
        "train_count": len(rows),
        "test_count_invalid": 0,
        "reward_version": REWARD_VERSION,
        "reward_counts": {"plus_one_high": len(high), "zero_medium": len(medium), "minus_one_low_total": len(low), "minus_one_low_valid_residual": len(low & valid), "minus_one_invalid": len(rows) - len(valid)},
        "selection_order": "sorted sample_id within each pool; sampler sequential=true",
        "pool_ids_sha256": hashlib.sha256(pool_payload.encode("utf-8")).hexdigest(),
        "reward_sidecar_sha256": sha256_file(sidecar_path),
        "split_manifest_sha256": sha256_file(output_training / "split_manifest.jsonl"),
        "formal_training": False,
        "warning": "Exploratory label test only; no docking/PB labels and no formal model-selection split.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
