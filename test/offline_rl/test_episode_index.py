"""Tests for global episode indexes and deterministic split manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from ddpo.offline_rl.index import (
    EPISODE_INDEX_SCHEMA_VERSION,
    SPLIT_MANIFEST_SCHEMA_VERSION,
    EpisodeIndex,
    EpisodeIndexError,
    EpisodeIndexRow,
    SplitAssignment,
    SplitManifest,
    build_episode_index,
    build_split_manifest,
    load_episode_index,
    load_split_manifest,
    write_episode_index,
    write_split_manifest,
)


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


def _make_run(tmp_path: Path) -> tuple[Path, dict[str, bytes]]:
    """Build a collector-like run with two ranks and two shards each."""
    run_dir = tmp_path / "collector_run"
    source_bytes: dict[str, bytes] = {}
    for rank in range(2):
        manifest_path = run_dir / "manifests" / f"rank_{rank:02d}.jsonl"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_rows = []
        for shard_index in range(2):
            relative_path = (
                f"shards/rank_{rank:02d}/shard_{shard_index:04d}.pt"
            )
            shard_path = run_dir / relative_path
            shard_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(_make_payload(rank, shard_index), shard_path)
            content = shard_path.read_bytes()
            source_bytes[relative_path] = content
            manifest_rows.append(
                {
                    "status": "completed",
                    "rank": rank,
                    "shard_index": shard_index,
                    "shard_path": relative_path,
                    "shard_sha256": hashlib.sha256(content).hexdigest(),
                }
            )
        manifest_path.write_text(
            "".join(
                json.dumps(row, sort_keys=True) + "\n"
                for row in manifest_rows
            ),
            encoding="utf-8",
        )
    return run_dir, source_bytes


def _index_row(index: int, *, valid: bool, nodes: int) -> EpisodeIndexRow:
    """Build one standalone index row for split tests."""
    return EpisodeIndexRow(
        index_schema_version=EPISODE_INDEX_SCHEMA_VERSION,
        source_run_id="run",
        manifest_snapshot_sha256="a" * 64,
        sample_id=f"sample-{index:04d}",
        shard_path=f"shards/shard_{index // 10:04d}.pt",
        source_shard_sha256=f"{index // 10:064x}",
        episode_index=index % 10,
        rank=0,
        shard_index=index // 10,
        seed=1,
        num_nodes=nodes,
        num_diffusion_steps=1000,
        max_n_nodes=30,
        rdkit_valid=valid,
        failure_type="success" if valid else "sanitize_failed",
        smiles="CC" if valid else None,
        source_checkpoint_sha256="c" * 64,
    )


def test_build_episode_index_validates_manifests_and_preserves_shards(
    tmp_path: Path,
) -> None:
    """Indexing should verify hashes, expose facts, and leave shards untouched."""
    run_dir, source_bytes = _make_run(tmp_path)

    episode_index = build_episode_index(run_dir)

    assert len(episode_index.rows) == 8
    assert len(episode_index.sha256) == 64
    assert episode_index.source_run_id == "collector_run"
    first = episode_index.get("r00_s0000_m0000")
    assert first.rank == 0
    assert first.shard_index == 0
    assert first.episode_index == 0
    assert first.num_nodes == 2
    assert first.rdkit_valid is True
    for relative_path, content in source_bytes.items():
        assert (run_dir / relative_path).read_bytes() == content


def test_build_episode_index_rejects_tampered_manifest_digest(
    tmp_path: Path,
) -> None:
    """A manifest/shard digest mismatch must fail before indexing facts."""
    run_dir, _ = _make_run(tmp_path)
    manifest_path = run_dir / "manifests" / "rank_00.jsonl"
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["shard_sha256"] = "0" * 64
    manifest_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(EpisodeIndexError, match="failed to validate indexed shard"):
        build_episode_index(run_dir)


def test_max_shards_per_rank_is_balanced_and_snapshot_bound(
    tmp_path: Path,
) -> None:
    """Smoke limits should select the same prefix from every rank."""
    run_dir, _ = _make_run(tmp_path)

    episode_index = build_episode_index(run_dir, max_shards_per_rank=1)

    assert len(episode_index.rows) == 4
    assert {(row.rank, row.shard_index) for row in episode_index.rows} == {
        (0, 0),
        (1, 0),
    }
    assert len({row.manifest_snapshot_sha256 for row in episode_index.rows}) == 1


def test_split_is_deterministic_order_independent_and_stratified() -> None:
    """Stable hashing should produce exact per-stratum 80/10/10 counts."""
    rows = tuple(
        _index_row(
            index,
            valid=index < 20,
            nodes=20 if index < 20 else 25,
        )
        for index in range(40)
    )
    forward = EpisodeIndex(rows)
    reverse = EpisodeIndex(reversed(rows))
    reward_strata = {
        row.sample_id: "good" if row.rdkit_valid else "invalid"
        for row in rows
    }

    first = build_split_manifest(
        forward,
        split_version="split_v1",
        salt="frozen-salt",
        reward_strata=reward_strata,
    )
    second = build_split_manifest(
        reverse,
        split_version="split_v1",
        salt="frozen-salt",
        reward_strata=reward_strata,
    )

    assert first.rows == second.rows
    counts: dict[tuple[str, str], int] = {}
    for row in first.rows:
        counts[(row.stratum, row.split)] = (
            counts.get((row.stratum, row.split), 0) + 1
        )
    for stratum in {row.stratum for row in first.rows}:
        assert counts[(stratum, "train")] == 16
        assert counts[(stratum, "validation")] == 2
        assert counts[(stratum, "test")] == 2
    first.validate_against(forward)


def test_split_requires_exact_reward_strata_and_index_provenance() -> None:
    """Partial strata and a changed index must not reuse a frozen split."""
    index = EpisodeIndex(
        (
            _index_row(0, valid=True, nodes=20),
            _index_row(1, valid=False, nodes=25),
        )
    )
    with pytest.raises(EpisodeIndexError, match="cover"):
        build_split_manifest(
            index,
            split_version="split_v1",
            salt="salt",
            reward_strata={"sample-0000": "good"},
        )

    manifest = build_split_manifest(
        index,
        split_version="split_v1",
        salt="salt",
    )
    changed_index = EpisodeIndex(
        (
            _index_row(0, valid=True, nodes=20),
            _index_row(2, valid=False, nodes=25),
        )
    )
    with pytest.raises(EpisodeIndexError, match="coverage mismatch"):
        manifest.validate_against(changed_index)


def test_split_enforces_global_ratio_with_many_tiny_strata() -> None:
    """Small validity/node strata must not collapse validation and test sets."""
    rows = tuple(
        _index_row(
            index,
            valid=index % 2 == 0,
            nodes=10 + index % 20,
        )
        for index in range(80)
    )
    index = EpisodeIndex(rows)

    manifest = build_split_manifest(
        index,
        split_version="split_v1",
        salt="salt",
    )

    counts = {
        split: sum(row.split == split for row in manifest.rows)
        for split in ("train", "validation", "test")
    }
    assert counts == {"train": 64, "validation": 8, "test": 8}


def test_index_and_split_jsonl_round_trip(tmp_path: Path) -> None:
    """Canonical JSONL artifacts should round-trip with exact index binding."""
    index = EpisodeIndex(
        tuple(
            _index_row(i, valid=i % 2 == 0, nodes=20 + i % 2)
            for i in range(20)
        )
    )
    manifest = build_split_manifest(
        index,
        split_version="split_v1",
        salt="salt",
    )
    index_path = write_episode_index(tmp_path / "episode_index.jsonl", index)
    split_path = write_split_manifest(
        tmp_path / "split_manifest.jsonl",
        manifest,
    )

    loaded_index = load_episode_index(index_path)
    loaded_split = load_split_manifest(split_path)

    assert loaded_index.rows == index.rows
    assert loaded_index.sha256 == index.sha256
    assert loaded_split.rows == manifest.rows
    loaded_split.validate_against(loaded_index)


def test_index_and_split_reject_duplicate_rows_and_schema_drift() -> None:
    """Duplicate IDs, duplicate locators, and wrong schemas must fail."""
    first = _index_row(0, valid=True, nodes=20)
    with pytest.raises(EpisodeIndexError, match="duplicate sample_id"):
        EpisodeIndex((first, first))
    with pytest.raises(EpisodeIndexError, match="duplicate shard locator"):
        EpisodeIndex(
            (
                first,
                replace(first, sample_id="different"),
            )
        )
    with pytest.raises(EpisodeIndexError, match="unsupported index schema"):
        replace(first, index_schema_version="other")
    with pytest.raises(EpisodeIndexError, match="run-relative"):
        replace(first, shard_path="/absolute/shard.pt")

    assignment = SplitAssignment(
        split_schema_version=SPLIT_MANIFEST_SCHEMA_VERSION,
        split_version="split_v1",
        episode_index_sha256="a" * 64,
        sample_id="sample",
        split="train",
        stratum="unscored",
    )
    with pytest.raises(EpisodeIndexError, match="duplicate sample_id"):
        SplitManifest((assignment, assignment))
    with pytest.raises(EpisodeIndexError, match="hex digest"):
        replace(assignment, episode_index_sha256="z" * 64)
