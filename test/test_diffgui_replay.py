"""Tests for decoupled DiffGui filtering, good-pool, and replay helpers."""

from __future__ import annotations

from pathlib import Path

import torch

from ddpo.diffgui.runtime.actor import DiffGuiPolicyBatch
from ddpo.diffgui.runtime.good_pool import DiffGuiGoodMoleculePool
from ddpo.diffgui.runtime.replay import (
    DiffGuiTrajectoryReplayBuffer,
    build_train_batch_with_replay,
    concat_diffgui_policy_batches,
)
from ddpo.diffgui.runtime.training_filter import DiffGuiTrainingFilter, annotate_molecule_rows


def _make_batch(graph_count: int = 3, *, offset: float = 0.0) -> DiffGuiPolicyBatch:
    """Build a tiny valid policy batch with two ligand nodes and one halfedge per graph."""
    ligand_batch = torch.arange(graph_count, dtype=torch.long).repeat_interleave(2)
    protein_batch = torch.arange(graph_count, dtype=torch.long)
    halfedge_batch = torch.arange(graph_count, dtype=torch.long)
    edges = []
    for graph_index in range(graph_count):
        start = graph_index * 2
        edges.append([start, start + 1])
    halfedge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    steps = 2
    node_count = int(ligand_batch.shape[0])
    edge_count = int(halfedge_batch.shape[0])
    batch = DiffGuiPolicyBatch(
        advantages=torch.arange(graph_count, dtype=torch.float32) + 1.0 + offset,
        protein_node=torch.zeros(graph_count, 4) + offset,
        protein_pos=torch.zeros(graph_count, 3) + offset,
        protein_batch=protein_batch,
        ligand_batch=ligand_batch,
        halfedge_index=halfedge_index,
        halfedge_batch=halfedge_batch,
        node_traj=torch.zeros(steps + 1, node_count, dtype=torch.long),
        pos_centered_traj=torch.zeros(steps + 1, node_count, 3) + offset,
        halfedge_traj=torch.zeros(steps + 1, edge_count, dtype=torch.long),
        node_log_state_traj=torch.zeros(steps + 1, node_count, 5) + offset,
        halfedge_log_state_traj=torch.zeros(steps + 1, edge_count, 6) + offset,
        old_logprob=torch.zeros(steps, graph_count) + offset,
        timesteps=torch.ones(steps, graph_count, dtype=torch.long),
    )
    batch.validate()
    return batch


def _rows() -> list[dict[str, object]]:
    """Return molecule rows with valid, invalid, duplicate, and ring-complex examples."""
    return [
        {
            "local_mol_index": 0,
            "global_mol_id": "e0001_r00_m0000",
            "reconstruction_success": 1,
            "rdkit_valid": 1,
            "reward_found": 1,
            "reward_total": 3.0,
            "reward_for_advantage": 3.0,
            "canonical_smiles": "CCO",
            "ring_count": 0,
            "three_member_ring_count": 0,
        },
        {
            "local_mol_index": 1,
            "global_mol_id": "e0001_r00_m0001",
            "reconstruction_success": 1,
            "rdkit_valid": 1,
            "reward_found": 1,
            "reward_total": -10.0,
            "reward_for_advantage": -10.0,
            "canonical_smiles": "C1CC1",
            "ring_count": 1,
            "three_member_ring_count": 1,
        },
        {
            "local_mol_index": 2,
            "global_mol_id": "e0001_r00_m0002",
            "reconstruction_success": 1,
            "rdkit_valid": 0,
            "reward_found": 1,
            "reward_total": 4.0,
            "reward_for_advantage": 4.0,
            "canonical_smiles": "CCN",
            "ring_count": 0,
            "three_member_ring_count": 0,
        },
    ]


def test_training_filter_discards_invalid_and_ring_complex_rows() -> None:
    """Training filter should keep records but exclude bad graphs from PPO."""
    training_filter = DiffGuiTrainingFilter(
        {
            "enabled": True,
            "require_rdkit_valid": True,
            "quality_filters": {"three_member_ring_count": {"max": 0}},
        }
    )

    selection = training_filter.select(_rows(), batch_size=3)
    annotated = annotate_molecule_rows(_rows(), selection, good_indices={0})

    assert selection.train_indices == [0]
    assert selection.metrics["train_selected_count"] == 1.0
    assert annotated[0]["training_keep"] == 1
    assert annotated[0]["good_pool_selected"] == 1
    assert annotated[1]["training_filter_reason"] == "quality_max:three_member_ring_count"
    assert annotated[2]["training_filter_reason"] == "rdkit_invalid"


def test_good_pool_selects_top_training_kept_molecules(tmp_path: Path) -> None:
    """Good pool should select only training-kept valid molecules."""
    training_filter = DiffGuiTrainingFilter({"enabled": True, "quality_filters": {"three_member_ring_count": {"max": 0}}})
    training_selection = training_filter.select(_rows(), batch_size=3)
    pool = DiffGuiGoodMoleculePool(
        tmp_path,
        {"enabled": True, "top_k_per_rank_per_epoch": 2, "score_key": "reward_total"},
        rank=0,
    )

    selection = pool.add_epoch(molecule_rows=_rows(), training_selection=training_selection, epoch=1)

    assert selection.indices == [0]
    assert selection.records[0]["canonical_smiles"] == "CCO"
    assert pool.manifest_path.exists()
    assert selection.metrics["good_pool_retained_count"] == 1.0


def test_concat_diffgui_policy_batches_rebases_graphs_and_halfedges() -> None:
    """Concatenating policy batches should rebase graph and ligand node ids."""
    first = _make_batch(2, offset=0.0)
    second = _make_batch(1, offset=2.0)

    combined = concat_diffgui_policy_batches([first, second])

    assert combined.advantages.tolist() == [1.0, 2.0, 3.0]
    assert combined.ligand_batch.tolist() == [0, 0, 1, 1, 2, 2]
    assert combined.halfedge_batch.tolist() == [0, 1, 2]
    assert combined.halfedge_index.tolist() == [[0, 2, 4], [1, 3, 5]]
    assert combined.old_logprob.shape == (2, 3)
    combined.validate()


def test_trajectory_replay_store_sample_and_build_train_batch(tmp_path: Path) -> None:
    """Replay buffer should store good trajectories and sample only previous epochs."""
    batch = _make_batch(3)
    training_filter = DiffGuiTrainingFilter({"enabled": True})
    training_selection = training_filter.select(_rows(), batch_size=3)
    pool = DiffGuiGoodMoleculePool(
        tmp_path / "pool",
        {"enabled": False, "top_k_per_rank_per_epoch": 1, "score_key": "reward_total"},
        rank=0,
    )
    good_selection = pool.select(molecule_rows=_rows(), training_selection=training_selection, epoch=1)
    buffer = DiffGuiTrajectoryReplayBuffer(
        tmp_path / "replay",
        {
            "enabled": True,
            "store_enabled": True,
            "replay_ratio": 0.5,
            "max_age_epochs": 3,
            "sample_strategy": "recent",
            "seed": 7,
        },
        rank=0,
    )

    store_metrics = buffer.store(policy_batch=batch, selection=good_selection, epoch=1)
    train_batch, replay_metrics = build_train_batch_with_replay(
        current_batch=batch.select_graphs([0, 1]),
        replay_buffer=buffer,
        epoch=2,
    )

    assert store_metrics["trajectory_replay_store_count"] == 1.0
    assert replay_metrics["trajectory_replay_sample_count"] == 1.0
    assert train_batch.advantages.shape[0] == 3
    train_batch.validate()


def test_replay_gate_decays_ratio_on_kl_or_clip_violation(tmp_path: Path) -> None:
    """Replay gate should lower the effective replay ratio after stale-policy signals."""
    buffer = DiffGuiTrajectoryReplayBuffer(
        tmp_path / "replay_gate",
        {
            "enabled": True,
            "store_enabled": False,
            "replay_ratio": 0.4,
            "max_approx_kl": 0.03,
            "max_clipfrac": 0.30,
            "ratio_decay": 0.5,
        },
        rank=0,
    )

    metrics = buffer.observe_update_metrics(
        {"trajectory_replay_sample_count": 2.0, "PPO_ApproxKL": 0.05, "ClipFrac": 0.10, "RatioMean": 1.0},
        epoch=3,
    )

    assert metrics["trajectory_replay_gate_violation"] == 1.0
    assert metrics["trajectory_replay_gate_ratio_before"] == 0.4
    assert metrics["trajectory_replay_gate_ratio_after"] == 0.2
    assert buffer.current_replay_ratio == 0.2
    assert buffer.gate_state_path.exists()


def test_replay_gate_ignores_current_only_updates(tmp_path: Path) -> None:
    """Replay gate should not decay ratio when no replay samples were used."""
    buffer = DiffGuiTrajectoryReplayBuffer(
        tmp_path / "replay_gate_no_sample",
        {"enabled": True, "replay_ratio": 0.4, "max_approx_kl": 0.03},
        rank=0,
    )

    metrics = buffer.observe_update_metrics(
        {"trajectory_replay_sample_count": 0.0, "PPO_ApproxKL": 1.0, "ClipFrac": 1.0, "RatioMean": 2.0},
        epoch=1,
    )

    assert metrics["trajectory_replay_gate_observed"] == 0.0
    assert metrics["trajectory_replay_gate_violation"] == 0.0
    assert buffer.current_replay_ratio == 0.4
