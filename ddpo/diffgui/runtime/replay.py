"""Short-window trajectory replay utilities for DiffGui PPO."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from ddpo.diffgui.runtime.actor import DiffGuiPolicyBatch
from ddpo.diffgui.runtime.good_pool import GoodPoolSelection
from ddpo.diffgui.runtime.training_filter import ddp_min_int


@dataclass
class ReplaySample:
    """Sampled replay batch plus metrics."""

    batch: DiffGuiPolicyBatch | None = None
    metrics: dict[str, float] = field(default_factory=dict)


class DiffGuiTrajectoryReplayBuffer:
    """Disk-backed short-window replay buffer for DiffGui policy batches."""

    def __init__(
        self,
        root_dir: str | Path,
        config: dict[str, Any] | None = None,
        *,
        rank: int = 0,
    ) -> None:
        """Initialize from ``diffgui_trajectory_replay`` YAML config."""
        cfg = config or {}
        self.root_dir = Path(root_dir)
        self.rank = int(rank)
        self.enabled = bool(cfg.get("enabled", False))
        self.store_enabled = bool(cfg.get("store_enabled", self.enabled))
        self.warmup_epochs = max(0, int(cfg.get("warmup_epochs", 0)))
        self.replay_ratio = max(0.0, float(cfg.get("replay_ratio", 0.0)))
        self.sample_size = None if cfg.get("sample_size") is None else max(0, int(cfg.get("sample_size")))
        self.max_replay_graphs_per_rank = cfg.get("max_replay_graphs_per_rank")
        if self.max_replay_graphs_per_rank is not None:
            self.max_replay_graphs_per_rank = max(0, int(self.max_replay_graphs_per_rank))
        self.max_age_epochs = max(1, int(cfg.get("max_age_epochs", 3)))
        self.sample_strategy = str(cfg.get("sample_strategy", "top_score_recent"))
        self.seed = int(cfg.get("seed", 42))
        self.rng = random.Random(self.seed + self.rank)
        self.shards_dir = self.root_dir / f"rank_{self.rank:02d}" / "shards"
        self.manifest_path = self.root_dir / f"trajectory_manifest_rank{self.rank:02d}.jsonl"
        self._manifest = self._load_manifest()
        if self.enabled or self.store_enabled:
            self.shards_dir.mkdir(parents=True, exist_ok=True)
            self.root_dir.mkdir(parents=True, exist_ok=True)

    def store(
        self,
        *,
        policy_batch: DiffGuiPolicyBatch,
        selection: GoodPoolSelection,
        epoch: int,
    ) -> dict[str, float]:
        """Store selected good trajectories as a CPU shard."""
        if not self.store_enabled or len(selection) == 0:
            return {"trajectory_replay_store_count": 0.0}
        selected = policy_batch.select_graphs(selection.indices).to("cpu")
        shard_path = self.shards_dir / f"epoch_{int(epoch):04d}_rank_{self.rank:02d}.pt"
        torch.save({"policy_batch": selected, "records": selection.records}, shard_path)
        scores = [float(record["score"]) for record in selection.records if _float_or_none(record.get("score")) is not None]
        manifest_record = {
            "epoch": int(epoch),
            "rank": int(self.rank),
            "path": str(shard_path.relative_to(self.root_dir)),
            "count": int(len(selection)),
            "score_mean": float(sum(scores) / len(scores)) if scores else 0.0,
            "score_max": float(max(scores)) if scores else 0.0,
            "smiles": [record.get("canonical_smiles") or record.get("smiles") or "" for record in selection.records],
        }
        self._append_manifest(manifest_record)
        return {
            "trajectory_replay_store_count": float(len(selection)),
            "trajectory_replay_store_score_mean": manifest_record["score_mean"],
            "trajectory_replay_store_score_max": manifest_record["score_max"],
        }

    def sample(self, *, epoch: int, current_batch_size: int) -> ReplaySample:
        """Sample replay trajectories from previous epochs."""
        if not self.enabled or int(epoch) <= self.warmup_epochs:
            return ReplaySample(
                metrics={
                    "trajectory_replay_enabled": float(self.enabled),
                    "trajectory_replay_sample_count": 0.0,
                    "trajectory_replay_eligible_shards": 0.0,
                }
            )
        target_count = self._target_sample_count(current_batch_size)
        if target_count <= 0:
            eligible_count = len(self._eligible_shards(epoch))
            return ReplaySample(
                metrics={
                    "trajectory_replay_enabled": 1.0,
                    "trajectory_replay_sample_count": 0.0,
                    "trajectory_replay_eligible_shards": float(eligible_count),
                }
            )
        eligible = self._eligible_shards(epoch)
        if not eligible:
            return ReplaySample(
                metrics={
                    "trajectory_replay_enabled": 1.0,
                    "trajectory_replay_sample_count": 0.0,
                    "trajectory_replay_eligible_shards": 0.0,
                }
            )
        eligible = self._ordered_shards(eligible)
        loaded: list[DiffGuiPolicyBatch] = []
        loaded_ages: list[int] = []
        loaded_count = 0
        for record in eligible:
            shard_path = self.root_dir / str(record["path"])
            if not shard_path.exists():
                continue
            payload = torch.load(shard_path, map_location="cpu")
            shard_batch = payload["policy_batch"]
            if not isinstance(shard_batch, DiffGuiPolicyBatch):
                continue
            loaded.append(shard_batch.to("cpu"))
            loaded_count += int(shard_batch.advantages.shape[0])
            loaded_ages.extend([int(epoch) - int(record.get("epoch", epoch))] * int(shard_batch.advantages.shape[0]))
            if loaded_count >= target_count:
                break
        if not loaded:
            return ReplaySample(
                metrics={
                    "trajectory_replay_enabled": 1.0,
                    "trajectory_replay_sample_count": 0.0,
                    "trajectory_replay_eligible_shards": float(len(eligible)),
                }
            )
        batch = concat_diffgui_policy_batches(loaded)
        if int(batch.advantages.shape[0]) > target_count:
            indices = self.rng.sample(range(int(batch.advantages.shape[0])), target_count)
            batch = batch.select_graphs(sorted(indices))
            loaded_ages = [loaded_ages[index] for index in indices]
        age_mean = float(sum(loaded_ages) / len(loaded_ages)) if loaded_ages else 0.0
        age_max = float(max(loaded_ages)) if loaded_ages else 0.0
        return ReplaySample(
            batch=batch,
            metrics={
                "trajectory_replay_enabled": 1.0,
                "trajectory_replay_sample_count": float(batch.advantages.shape[0]),
                "trajectory_replay_eligible_shards": float(len(eligible)),
                "trajectory_replay_age_mean": age_mean,
                "trajectory_replay_age_max": age_max,
            },
        )

    def _target_sample_count(self, current_batch_size: int) -> int:
        """Resolve target replay graph count."""
        if self.sample_size is not None:
            target = int(self.sample_size)
        else:
            target = int(math.floor(max(0, int(current_batch_size)) * self.replay_ratio))
        if self.max_replay_graphs_per_rank is not None:
            target = min(target, int(self.max_replay_graphs_per_rank))
        return max(0, target)

    def _eligible_shards(self, epoch: int) -> list[dict[str, Any]]:
        """Return previous-epoch shards inside the age window."""
        min_epoch = int(epoch) - self.max_age_epochs
        return [
            record
            for record in self._manifest
            if min_epoch <= int(record.get("epoch", -1)) < int(epoch)
            and int(record.get("count", 0)) > 0
        ]

    def _ordered_shards(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Order eligible shards by configured strategy."""
        ordered = list(records)
        if self.sample_strategy == "random":
            self.rng.shuffle(ordered)
        elif self.sample_strategy == "recent":
            ordered.sort(key=lambda record: int(record.get("epoch", -1)), reverse=True)
        elif self.sample_strategy == "top_score":
            ordered.sort(key=lambda record: float(record.get("score_max", 0.0)), reverse=True)
        else:
            ordered.sort(
                key=lambda record: (
                    int(record.get("epoch", -1)),
                    float(record.get("score_max", 0.0)),
                ),
                reverse=True,
            )
        return ordered

    def _load_manifest(self) -> list[dict[str, Any]]:
        """Load replay manifest."""
        if not self.manifest_path.exists():
            return []
        records: list[dict[str, Any]] = []
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def _append_manifest(self, record: dict[str, Any]) -> None:
        """Append one replay manifest record."""
        with self.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self._manifest.append(record)


def concat_diffgui_policy_batches(batches: list[DiffGuiPolicyBatch]) -> DiffGuiPolicyBatch:
    """Concatenate graph-disjoint DiffGui policy batches with re-based ids."""
    if not batches:
        raise ValueError("batches must not be empty")
    if len(batches) == 1:
        batches[0].validate()
        return batches[0]
    first = batches[0]
    for batch in batches:
        batch.validate()
        if int(batch.old_logprob.shape[0]) != int(first.old_logprob.shape[0]):
            raise ValueError("All DiffGuiPolicyBatch objects must have the same number of timesteps.")
        if batch.gui_strength != first.gui_strength:
            raise ValueError("Cannot concatenate DiffGuiPolicyBatch objects with different gui_strength.")

    device = first.advantages.device
    advantages: list[torch.Tensor] = []
    protein_node: list[torch.Tensor] = []
    protein_pos: list[torch.Tensor] = []
    protein_batch: list[torch.Tensor] = []
    ligand_batch: list[torch.Tensor] = []
    halfedge_index: list[torch.Tensor] = []
    halfedge_batch: list[torch.Tensor] = []
    node_traj: list[torch.Tensor] = []
    pos_traj: list[torch.Tensor] = []
    halfedge_traj: list[torch.Tensor] = []
    node_log_state_traj: list[torch.Tensor] = []
    halfedge_log_state_traj: list[torch.Tensor] = []
    old_logprob: list[torch.Tensor] = []
    timesteps: list[torch.Tensor] = []
    pred_node: list[torch.Tensor] = []
    pred_pos: list[torch.Tensor] = []
    pred_halfedge: list[torch.Tensor] = []
    batch_lab: list[torch.Tensor] = []
    graph_offset = 0
    node_offset = 0
    has_pred_node = all(batch.pred_node is not None for batch in batches)
    has_pred_pos = all(batch.pred_pos is not None for batch in batches)
    has_pred_halfedge = all(batch.pred_halfedge is not None for batch in batches)
    has_batch_lab = all(batch.batch_lab is not None for batch in batches)

    for batch in batches:
        batch = batch.to(device)
        graph_count = int(batch.advantages.shape[0])
        advantages.append(batch.advantages)
        protein_node.append(batch.protein_node)
        protein_pos.append(batch.protein_pos)
        protein_batch.append(batch.protein_batch + graph_offset)
        ligand_batch.append(batch.ligand_batch + graph_offset)
        halfedge_index.append(batch.halfedge_index + node_offset)
        halfedge_batch.append(batch.halfedge_batch + graph_offset)
        node_traj.append(batch.node_traj)
        pos_traj.append(batch.pos_centered_traj)
        halfedge_traj.append(batch.halfedge_traj)
        node_log_state_traj.append(batch.node_log_state_traj)
        halfedge_log_state_traj.append(batch.halfedge_log_state_traj)
        old_logprob.append(batch.old_logprob)
        timesteps.append(batch.timesteps)
        if has_pred_node and batch.pred_node is not None:
            pred_node.append(batch.pred_node)
        if has_pred_pos and batch.pred_pos is not None:
            pred_pos.append(batch.pred_pos)
        if has_pred_halfedge and batch.pred_halfedge is not None:
            pred_halfedge.append(batch.pred_halfedge)
        if has_batch_lab and batch.batch_lab is not None:
            batch_lab.append(batch.batch_lab)
        graph_offset += graph_count
        node_offset += int(batch.ligand_batch.shape[0])

    combined = DiffGuiPolicyBatch(
        advantages=torch.cat(advantages, dim=0),
        protein_node=torch.cat(protein_node, dim=0),
        protein_pos=torch.cat(protein_pos, dim=0),
        protein_batch=torch.cat(protein_batch, dim=0),
        ligand_batch=torch.cat(ligand_batch, dim=0),
        halfedge_index=torch.cat(halfedge_index, dim=1),
        halfedge_batch=torch.cat(halfedge_batch, dim=0),
        node_traj=torch.cat(node_traj, dim=1),
        pos_centered_traj=torch.cat(pos_traj, dim=1),
        halfedge_traj=torch.cat(halfedge_traj, dim=1),
        node_log_state_traj=torch.cat(node_log_state_traj, dim=1),
        halfedge_log_state_traj=torch.cat(halfedge_log_state_traj, dim=1),
        old_logprob=torch.cat(old_logprob, dim=1),
        timesteps=torch.cat(timesteps, dim=1),
        pred_node=torch.cat(pred_node, dim=0) if has_pred_node else None,
        pred_pos=torch.cat(pred_pos, dim=0) if has_pred_pos else None,
        pred_halfedge=torch.cat(pred_halfedge, dim=0) if has_pred_halfedge else None,
        batch_lab=torch.cat(batch_lab, dim=0) if has_batch_lab else None,
        gui_strength=first.gui_strength,
    )
    combined.validate()
    return combined


def build_train_batch_with_replay(
    *,
    current_batch: DiffGuiPolicyBatch,
    replay_buffer: DiffGuiTrajectoryReplayBuffer | None,
    epoch: int,
    ddp_device: torch.device | str | None = None,
) -> tuple[DiffGuiPolicyBatch, dict[str, float]]:
    """Return current-only or current-plus-replay training batch."""
    metrics = {
        "trajectory_replay_enabled": float(replay_buffer.enabled if replay_buffer is not None else False),
        "trajectory_replay_current_count": float(current_batch.advantages.shape[0]),
        "trajectory_replay_sample_count": 0.0,
    }
    if replay_buffer is None:
        return current_batch, metrics
    replay_sample = replay_buffer.sample(epoch=epoch, current_batch_size=int(current_batch.advantages.shape[0]))
    metrics.update(replay_sample.metrics)
    if replay_sample.batch is None or int(replay_sample.batch.advantages.shape[0]) == 0:
        metrics["trajectory_replay_train_total_count"] = float(current_batch.advantages.shape[0])
        return current_batch, metrics
    local_replay_count = int(replay_sample.batch.advantages.shape[0])
    replay_count = local_replay_count
    if ddp_device is not None:
        replay_count = ddp_min_int(local_replay_count, device=ddp_device)
    metrics["trajectory_replay_ddp_min_count"] = float(replay_count)
    if replay_count <= 0:
        metrics["trajectory_replay_sample_count"] = 0.0
        metrics["trajectory_replay_train_total_count"] = float(current_batch.advantages.shape[0])
        return current_batch, metrics
    if replay_count < local_replay_count:
        replay_sample.batch = replay_sample.batch.select_graphs(list(range(replay_count)))
        metrics["trajectory_replay_sample_count"] = float(replay_count)
    train_batch = concat_diffgui_policy_batches([current_batch, replay_sample.batch.to(current_batch.advantages.device)])
    metrics["trajectory_replay_train_total_count"] = float(train_batch.advantages.shape[0])
    return train_batch, metrics


def _float_or_none(value: Any) -> float | None:
    """Convert scalar-like values to finite float."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None
