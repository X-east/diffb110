"""Tests for disk-backed good-molecule and trajectory replay utilities."""

from __future__ import annotations

import json

import pytest
import torch

from ddpo.runtime.protocol import DataProto, TensorDict
from ddpo.runtime.replay import (
    DiskTrajectoryReplayBuffer,
    GoodMoleculePool,
    GoodMoleculeSelector,
    ReplayCoordinator,
)


def _make_rollout_batch(scores: list[float] | None = None) -> DataProto:
    """Build a tiny CPU DataProto with the tensors needed by PPO replay."""
    scores = scores or [0.1, 0.9, 0.8, -0.2]
    batch_size = len(scores)
    timesteps = 3
    nodes = 2
    dims = 4
    batch = TensorDict(
        {
            "latents": torch.arange(batch_size * (timesteps + 1) * nodes * dims, dtype=torch.float32).reshape(
                batch_size,
                timesteps + 1,
                nodes,
                dims,
            ),
            "logps": torch.zeros(batch_size, timesteps),
            "nodesxsample": torch.full((batch_size,), nodes, dtype=torch.long),
            "timesteps": torch.arange(timesteps, dtype=torch.long).repeat(batch_size, 1),
            "advantages": torch.tensor(scores, dtype=torch.float32),
            "reward_total_clamped": torch.tensor(scores, dtype=torch.float32),
            "reward_total_raw": torch.tensor(scores, dtype=torch.float32),
            "x": torch.zeros(batch_size, nodes, 3),
            "categorical": torch.zeros(batch_size, nodes, dims),
        },
        batch_size=batch_size,
    )
    return DataProto(
        batch=batch,
        non_tensor_batch={
            "smiles": ["A", "B", "B", "D"],
            "rdkit_valid": [1, 1, 1, 0],
            "failure_type": ["success", "success", "success", "sanitize_failed"],
        },
        meta_info={"max_n_nodes": nodes, "condition": False},
    )


def test_good_molecule_selector_filters_top_valid_unique() -> None:
    """Selector should keep valid high-score molecules and deduplicate SMILES."""
    selector = GoodMoleculeSelector(
        {
            "top_k_per_epoch": 2,
            "require_valid": True,
            "deduplicate_smiles": True,
        }
    )
    selection = selector.select(_make_rollout_batch(), epoch=1, rank=0)

    assert selection.indices == [1, 0]
    assert [record["smiles"] for record in selection.records] == ["B", "A"]
    assert [record["score"] for record in selection.records] == pytest.approx([0.9, 0.1])


def test_good_molecule_pool_writes_jsonl_manifest(tmp_path) -> None:
    """GoodMoleculePool should persist compact molecule records without tensors."""
    selector = GoodMoleculeSelector({"top_k_per_epoch": 2})
    pool = GoodMoleculePool(tmp_path, selector=selector, rank=2, enabled=True)

    selection = pool.add_batch(_make_rollout_batch(), epoch=3)

    assert len(selection) == 2
    manifest_lines = pool.manifest_path.read_text(encoding="utf-8").strip().splitlines()
    records = [json.loads(line) for line in manifest_lines]
    assert [record["rank"] for record in records] == [2, 2]
    assert [record["epoch"] for record in records] == [3, 3]
    assert [record["smiles"] for record in records] == ["B", "A"]


def test_disk_trajectory_replay_roundtrip_keeps_tensors_on_cpu(tmp_path) -> None:
    """Stored and sampled PPO trajectories should remain CPU tensors."""
    data = _make_rollout_batch()
    selector = GoodMoleculeSelector({"top_k_per_epoch": 2})
    selection = selector.select(data, epoch=1, rank=0)
    replay = DiskTrajectoryReplayBuffer(
        tmp_path,
        rank=0,
        enabled=True,
        replay_ratio=0.5,
        max_age_epochs=3,
        seed=123,
    )

    store_metrics = replay.add_batch(data, selection=selection, epoch=1)
    sampled, sample_metrics = replay.sample(epoch=2, current_batch_size=len(data))

    assert store_metrics["replay_store_count"] == 2.0
    assert sampled is not None
    assert len(sampled) == 2
    assert sample_metrics["replay_sample_count"] == 2.0
    assert set(sampled.batch.keys()) == {"latents", "logps", "nodesxsample", "timesteps", "advantages"}
    assert all(tensor.device.type == "cpu" for tensor in sampled.batch.values())
    assert sampled.non_tensor_batch == {}


def test_replay_coordinator_samples_history_then_stores_current(tmp_path) -> None:
    """Coordinator should mimic TopHistory ordering: sample old, then put current."""
    selector = GoodMoleculeSelector({"top_k_per_epoch": 2})
    pool = GoodMoleculePool(tmp_path / "good", selector=selector, enabled=True)
    replay = DiskTrajectoryReplayBuffer(
        tmp_path / "traj",
        enabled=True,
        replay_ratio=0.5,
        max_age_epochs=3,
        seed=7,
    )
    coordinator = ReplayCoordinator(good_molecule_pool=pool, trajectory_replay=replay)

    first = coordinator.stage_for_training(_make_rollout_batch(), epoch=1)
    second = coordinator.stage_for_training(_make_rollout_batch([0.2, 1.0, 0.6, -0.1]), epoch=2)

    assert len(first.samples) == 4
    assert first.metrics["replay_sample_count"] == 0.0
    assert len(second.samples) == 6
    assert second.metrics["replay_sample_count"] == 2.0
    assert second.metrics["good_pool_selected_count"] == 2.0
    assert all(tensor.device.type == "cpu" for tensor in second.samples.batch.values())
