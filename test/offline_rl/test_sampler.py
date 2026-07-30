"""Tests for split-aware shard-local offline transition sampling."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from ddpo.offline_rl.index import (
    SPLIT_MANIFEST_SCHEMA_VERSION,
    EpisodeIndex,
    SplitAssignment,
    SplitManifest,
    build_episode_index,
)
from ddpo.offline_rl.reward_sidecar import (
    REWARD_SIDECAR_SCHEMA_VERSION,
    EpisodeRewardFacts,
    RewardSidecar,
)
from ddpo.offline_rl.sampler import DecisionTimeBin, OfflineTransitionSampler


CHECKPOINT_SHA256 = "c" * 64


def _make_payload(rank: int, shard_index: int) -> dict[str, object]:
    """Build one valid two-episode trajectory payload."""
    batch_size = 2
    diffusion_steps = 3
    num_actions = diffusion_steps + 1
    max_nodes = 4
    sample_ids = [
        f"r{rank:02d}_s{shard_index:04d}_m{index:04d}"
        for index in range(batch_size)
    ]
    latents = torch.randn(batch_size, num_actions + 1, max_nodes, 5)
    nodesxsample = torch.tensor([2, 4], dtype=torch.long)
    latents[0, :, 2:] = 0
    x = torch.randn(batch_size, max_nodes, 3)
    categorical = torch.ones(batch_size, max_nodes, 2)
    x[0, 2:] = 0
    categorical[0, 2:] = 0
    return {
        "format": "diffsbdd_frozen_policy_trajectory_v1",
        "latents": latents,
        "logps": torch.zeros(batch_size, num_actions),
        "timesteps": torch.tensor([2, 1, 0, 0]).repeat(batch_size, 1),
        "nodesxsample": nodesxsample,
        "x": x,
        "categorical": categorical,
        "sample_ids": sample_ids,
        "metadata": {
            "rank": rank,
            "shard_index": shard_index,
            "seed": rank + 1,
            "num_timesteps": diffusion_steps,
            "max_n_nodes": max_nodes,
            "checkpoint_sha256": CHECKPOINT_SHA256,
        },
        "reconstruction_rows": [
            {
                "sample_id": sample_ids[0],
                "rdkit_valid": 1,
                "failure_type": "success",
                "smiles": "CC",
            },
            {
                "sample_id": sample_ids[1],
                "rdkit_valid": 0,
                "failure_type": "sanitize_failed",
                "smiles": "",
            },
        ],
    }


def _make_run(tmp_path: Path, *, shards: int = 6) -> Path:
    """Build a collector-like run with enough shards for cache/DDP tests."""
    run_dir = tmp_path / "collector_run"
    manifest_rows: dict[int, list[dict[str, object]]] = {0: [], 1: []}
    for global_shard_index in range(shards):
        rank = global_shard_index % 2
        shard_index = global_shard_index // 2
        relative_path = f"shards/rank_{rank:02d}/shard_{shard_index:04d}.pt"
        shard_path = run_dir / relative_path
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(_make_payload(rank, shard_index), shard_path)
        digest = hashlib.sha256(shard_path.read_bytes()).hexdigest()
        manifest_rows[rank].append(
            {
                "status": "completed",
                "rank": rank,
                "shard_index": shard_index,
                "shard_path": relative_path,
                "shard_sha256": digest,
            }
        )
    manifest_dir = run_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for rank, rows in manifest_rows.items():
        (manifest_dir / f"rank_{rank:02d}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
    return run_dir


def _all_train_manifest(index: EpisodeIndex) -> SplitManifest:
    """Assign every fixture episode to train."""
    return SplitManifest(
        SplitAssignment(
            split_schema_version=SPLIT_MANIFEST_SCHEMA_VERSION,
            split_version="all_train_fixture_v1",
            episode_index_sha256=index.sha256,
            sample_id=row.sample_id,
            split="train",
            stratum="fixture",
        )
        for row in index.rows
    )


def _reward_sidecar(index: EpisodeIndex) -> RewardSidecar:
    """Build complete finite rewards for every indexed fixture episode."""
    return RewardSidecar(
        EpisodeRewardFacts(
            sample_id=row.sample_id,
            source_trajectory_format="diffsbdd_frozen_policy_trajectory_v1",
            source_checkpoint_sha256=row.source_checkpoint_sha256,
            source_shard_sha256=row.source_shard_sha256,
            reward_schema_version=REWARD_SIDECAR_SCHEMA_VERSION,
            reward_version="fixture_reward_v1",
            reconstruction_success=row.rdkit_valid,
            sanitize_success=row.rdkit_valid,
            rdkit_valid=row.rdkit_valid,
            pb_valid=False,
            docking_input_success=False,
            docking_success=False,
            actual_good=False,
            reward_total_train=float(row.shard_index + row.episode_index + 1),
            failure_reason=row.failure_type,
        )
        for row in index.rows
    )


def _sampler(
    run_dir: Path,
    index: EpisodeIndex,
    manifest: SplitManifest,
    *,
    bins: tuple[DecisionTimeBin, ...],
    sidecar: RewardSidecar | None = None,
    seed: int = 1,
    rank: int = 0,
    world_size: int = 1,
    same_decision: bool = True,
) -> OfflineTransitionSampler:
    """Construct one fixture sampler."""
    return OfflineTransitionSampler(
        run_dir=run_dir,
        episode_index=index,
        split_manifest=manifest,
        split="train",
        time_bins=bins,
        reward_sidecar=sidecar,
        cache_size=8,
        seed=seed,
        rank=rank,
        world_size=world_size,
        same_decision_index_per_batch=same_decision,
    )


def test_sampler_is_shard_local_same_time_and_cache_aware(tmp_path: Path) -> None:
    """Every batch should reuse one shard and one decision when configured."""
    run_dir = _make_run(tmp_path)
    index = build_episode_index(run_dir)
    manifest = _all_train_manifest(index)
    sampler = _sampler(
        run_dir,
        index,
        manifest,
        bins=(DecisionTimeBin(0, 2),),
    )

    for _ in range(30):
        batch = sampler.sample_batch(8)
        shard_prefixes = {
            sample_id.rsplit("_m", maxsplit=1)[0]
            for sample_id in batch.sample_ids
        }
        assert len(shard_prefixes) == 1
        assert len(set(batch.decision_index.tolist())) == 1
        assert 0 <= batch.decision_index[0].item() <= 2
        assert batch.reward is None

    stats = sampler.stats()
    assert stats.batches_sampled == 30
    assert stats.transitions_sampled == 240
    assert stats.cache_hits > 0
    assert stats.cache_misses <= sampler.num_eligible_shards
    assert 0.0 < stats.cache_hit_rate <= 1.0


def test_sampler_applies_sidecar_reward_only_to_final_decode(
    tmp_path: Path,
) -> None:
    """Joined episode rewards must remain sparse terminal rewards."""
    run_dir = _make_run(tmp_path)
    index = build_episode_index(run_dir)
    manifest = _all_train_manifest(index)
    sidecar = _reward_sidecar(index)
    ordinary_sampler = _sampler(
        run_dir,
        index,
        manifest,
        bins=(DecisionTimeBin(0, 0),),
        sidecar=sidecar,
    )
    final_sampler = _sampler(
        run_dir,
        index,
        manifest,
        bins=(DecisionTimeBin(3, 3),),
        sidecar=sidecar,
    )

    ordinary = ordinary_sampler.sample_batch(6)
    terminal = final_sampler.sample_batch(6)

    assert ordinary.reward is not None
    assert torch.equal(ordinary.reward, torch.zeros_like(ordinary.reward))
    assert not bool(ordinary.done.any())
    assert terminal.reward is not None
    assert bool(torch.all(terminal.reward > 0))
    assert bool(terminal.done.all())


def test_sampler_checkpoint_restores_exact_next_batch(tmp_path: Path) -> None:
    """Restoring RNG state should reproduce sample IDs, times, and tensors."""
    run_dir = _make_run(tmp_path)
    index = build_episode_index(run_dir)
    manifest = _all_train_manifest(index)
    bins = (DecisionTimeBin(0, 1), DecisionTimeBin(2, 3, weight=2.0))
    original = _sampler(
        run_dir,
        index,
        manifest,
        bins=bins,
        seed=7,
        same_decision=False,
    )
    for _ in range(3):
        original.sample_batch(5)
    state = original.state_dict()
    expected = original.sample_batch(7)

    restored = _sampler(
        run_dir,
        index,
        manifest,
        bins=bins,
        seed=7,
        same_decision=False,
    )
    restored.load_state_dict(state)
    actual = restored.sample_batch(7)

    assert actual.sample_ids == expected.sample_ids
    torch.testing.assert_close(actual.state_latent, expected.state_latent)
    torch.testing.assert_close(actual.action_latent, expected.action_latent)
    assert torch.equal(actual.decision_index, expected.decision_index)
    assert restored.stats().batches_sampled == original.stats().batches_sampled


def test_sampler_rank_shards_are_disjoint_and_cover_all(tmp_path: Path) -> None:
    """Stable shard ownership should avoid DDP rank overlap."""
    run_dir = _make_run(tmp_path, shards=12)
    index = build_episode_index(run_dir)
    manifest = _all_train_manifest(index)
    rank_zero = _sampler(
        run_dir,
        index,
        manifest,
        bins=(DecisionTimeBin(0, 0),),
        rank=0,
        world_size=2,
    )
    rank_one = _sampler(
        run_dir,
        index,
        manifest,
        bins=(DecisionTimeBin(0, 0),),
        rank=1,
        world_size=2,
    )

    assert set(rank_zero.shard_paths).isdisjoint(rank_one.shard_paths)
    assert set(rank_zero.shard_paths) | set(rank_one.shard_paths) == {
        row.shard_path for row in index.rows
    }


def test_sampler_rejects_identity_drift_and_invalid_time_bins(
    tmp_path: Path,
) -> None:
    """Resume and time contracts must fail closed on incompatible inputs."""
    run_dir = _make_run(tmp_path)
    index = build_episode_index(run_dir)
    manifest = _all_train_manifest(index)
    sampler = _sampler(
        run_dir,
        index,
        manifest,
        bins=(DecisionTimeBin(0, 3),),
    )
    state = sampler.state_dict()
    state["split"] = "validation"
    with pytest.raises(ValueError, match="identity mismatch"):
        sampler.load_state_dict(state)

    different_bins = _sampler(
        run_dir,
        index,
        manifest,
        bins=(DecisionTimeBin(0, 0),),
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        different_bins.load_state_dict(sampler.state_dict())

    with pytest.raises(ValueError, match="exceeds final decision"):
        _sampler(
            run_dir,
            index,
            manifest,
            bins=(DecisionTimeBin(4, 4),),
        )
