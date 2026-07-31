"""Tests for offline-IQL DDP scheduling, aggregation, and recovery contracts."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import ddpo.offline_rl.ddp as ddp_module
from ddpo.offline_rl.ddp import OfflineIQLDDPCoordinator
from ddpo.offline_rl.recording import LocalIQLRecorder
from ddpo.offline_rl.runner import IQLRunnerConfig
from ddpo.offline_rl.sampler import (
    DecisionTimeBin,
    SamplerStats,
    SharedDecisionIndexScheduler,
)
from ddpo.offline_rl.trainer import (
    ActorStepMetrics,
    IQLStepMetrics,
    _optimizer_step,
)
from ddpo.runtime import DistributedContext


def _context(*, rank: int = 0, world_size: int = 1) -> DistributedContext:
    """Return a non-initialized distributed context for collective doubles."""
    return DistributedContext(
        enabled=False,
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        device=torch.device("cpu"),
    )


def _runner_config(*, epochs: int = 2) -> IQLRunnerConfig:
    """Return a bounded CPU runner configuration."""
    return IQLRunnerConfig(
        epochs=epochs,
        steps_per_epoch=2,
        batch_size=2,
        device="cpu",
        log_every_steps=1,
    )


def _actor_metrics(
    *,
    loss: float,
    weight_mean: float,
    effective_sample_size: float,
) -> ActorStepMetrics:
    """Build one finite local actor metric object."""
    return ActorStepMetrics(
        loss=loss,
        log_prob_mean=-1.0,
        weight_mean=weight_mean,
        weight_max=3.0,
        effective_sample_size=effective_sample_size,
        clipped_fraction=0.25,
        grad_norm=0.5,
    )


def _step_metrics(
    *,
    global_step: int = 3,
    offset: float = 0.0,
    actor: ActorStepMetrics | None = None,
    gradient_loss_scale: float = 1.0,
) -> IQLStepMetrics:
    """Build one complete finite local IQL metric object."""
    return IQLStepMetrics(
        global_step=global_step,
        value_loss=1.0 + offset,
        q_loss=2.0 + offset,
        q1_loss=2.1 + offset,
        q2_loss=2.2 + offset,
        value_grad_norm=0.3 + offset,
        q_grad_norm=0.4 + offset,
        q1_mean=0.5 + offset,
        q2_mean=0.6 + offset,
        value_mean=0.7 + offset,
        td_target_mean=1.0 + offset,
        td_target_std=2.0,
        td_target_min=-2.0 + offset,
        td_target_max=4.0 + offset,
        advantage_mean=0.5 + offset,
        advantage_std=1.0,
        q_disagreement_mean=0.1 + offset,
        actor_updated=actor is not None,
        actor=actor,
        gradient_loss_scale=gradient_loss_scale,
    )


def _metric_fact(
    *,
    rank: int,
    batch_size: int,
    metrics: IQLStepMetrics,
) -> dict[str, Any]:
    """Wrap local metrics in the coordinator's gathered fact schema."""
    return {
        "rank": rank,
        "batch_size": batch_size,
        "metrics": asdict(metrics),
        "sampler_stats": asdict(
            SamplerStats(
                batches_sampled=rank + 1,
                transitions_sampled=batch_size,
                cache_hits=rank,
                cache_misses=1,
                shard_load_seconds=0.1 + rank,
                sample_seconds=0.2 + rank,
            )
        ),
    }


class _FakeTrainer:
    """Minimal trainer boundary required by coordinator unit tests."""

    def __init__(self) -> None:
        """Initialize a zero global step."""
        self.global_step = 0


class _FakeSampler:
    """Minimal sampler identity required by coordinator unit tests."""

    def __init__(
        self,
        *,
        rank: int = 0,
        world_size: int = 1,
        episode_counts: tuple[int, ...] = (2,),
    ) -> None:
        """Create disjoint synthetic ownership and all-train assignments."""
        self.rank = rank
        self.world_size = world_size
        self.time_bins = (DecisionTimeBin(0, 3),)
        self.same_decision_index_per_batch = True
        self.num_eligible_episodes = episode_counts[rank]
        self.num_eligible_shards = 1
        self.shard_paths = (f"shards/rank_{rank:02d}/shard_0000.pt",)
        rows = [
            SimpleNamespace(sample_id=f"sample_{index}")
            for index in range(sum(episode_counts))
        ]
        self.episode_index = SimpleNamespace(rows=rows)
        self.split = "train"
        self.split_manifest = SimpleNamespace(
            split_for=lambda sample_id: (
                "train" if str(sample_id).startswith("sample_") else "test"
            )
        )


def _coordinator(
    tmp_path: Path,
    *,
    rank: int = 0,
    world_size: int = 1,
    episode_counts: tuple[int, ...] = (2,),
    epochs: int = 2,
) -> OfflineIQLDDPCoordinator:
    """Build a coordinator with lightweight typed boundary doubles."""
    sampler = _FakeSampler(
        rank=rank,
        world_size=world_size,
        episode_counts=episode_counts,
    )
    return OfflineIQLDDPCoordinator(
        trainer=_FakeTrainer(),  # type: ignore[arg-type]
        sampler=sampler,  # type: ignore[arg-type]
        runner_config=_runner_config(epochs=epochs),
        dist_ctx=_context(rank=rank, world_size=world_size),
        output_dir=tmp_path,
        run_identity={"dataset": "fixture", "version": 1},
        shared_decision_scheduler=SharedDecisionIndexScheduler(
            sampler.time_bins,
            seed=1,
        ),
    )


def test_shared_scheduler_resume_reproduces_next_decisions() -> None:
    """A restored scheduler must continue at the exact next draw."""
    bins = (
        DecisionTimeBin(0, 1, 1.0),
        DecisionTimeBin(2, 3, 2.0),
    )
    uninterrupted = SharedDecisionIndexScheduler(bins, seed=1)
    prefix = [uninterrupted.next_decision_index() for _ in range(5)]
    state = uninterrupted.state_dict()
    expected_suffix = [
        uninterrupted.next_decision_index() for _ in range(6)
    ]

    restored = SharedDecisionIndexScheduler(bins, seed=1)
    restored.load_state_dict(state)
    observed_suffix = [restored.next_decision_index() for _ in range(6)]

    assert len(prefix) == 5
    assert restored.draw_count == 11
    assert observed_suffix == expected_suffix


def test_shared_scheduler_rejects_identity_drift() -> None:
    """Seed or time-bin drift must make scheduler restoration fail."""
    source = SharedDecisionIndexScheduler((DecisionTimeBin(0, 3),), seed=1)
    state = source.state_dict()
    changed = SharedDecisionIndexScheduler((DecisionTimeBin(0, 2),), seed=1)

    with pytest.raises(ValueError, match="identity mismatch"):
        changed.load_state_dict(state)


def test_shared_scheduler_survives_manifest_json_round_trip() -> None:
    """JSON list conversion must not change scheduler identity on resume."""
    scheduler = SharedDecisionIndexScheduler(
        (DecisionTimeBin(0, 1), DecisionTimeBin(2, 3, 2.0)),
        seed=1,
    )
    scheduler.next_decision_index()
    safe_state = ddp_module._json_safe_scheduler_state(
        scheduler.state_dict()
    )
    serialized = json.loads(json.dumps(safe_state))
    restored_state = ddp_module._scheduler_state_from_json(serialized)
    restored = SharedDecisionIndexScheduler(scheduler.time_bins, seed=1)

    restored.load_state_dict(restored_state)

    assert restored.draw_count == 1
    assert restored.next_decision_index() == scheduler.next_decision_index()


def test_global_step_aggregation_recovers_exact_actor_ess() -> None:
    """Rank-local weight moments should combine into global actor ESS."""
    first = _metric_fact(
        rank=0,
        batch_size=2,
        metrics=_step_metrics(
            actor=_actor_metrics(
                loss=1.0,
                weight_mean=2.0,
                effective_sample_size=1.6,
            ),
            gradient_loss_scale=0.5,
        ),
    )
    second = _metric_fact(
        rank=1,
        batch_size=2,
        metrics=_step_metrics(
            offset=2.0,
            actor=_actor_metrics(
                loss=3.0,
                weight_mean=2.0,
                effective_sample_size=2.0,
            ),
            gradient_loss_scale=1.5,
        ),
    )

    aggregated = ddp_module._aggregate_step_metrics([first, second])

    assert aggregated.value_loss == pytest.approx(2.0)
    assert aggregated.q_loss == pytest.approx(3.0)
    assert aggregated.td_target_mean == pytest.approx(2.0)
    assert aggregated.td_target_std == pytest.approx(2.2360679775)
    assert aggregated.advantage_mean == pytest.approx(1.5)
    assert aggregated.advantage_std == pytest.approx(1.4142135624)
    assert aggregated.gradient_loss_scale == pytest.approx(1.0)
    assert aggregated.actor is not None
    assert aggregated.actor.loss == pytest.approx(2.0)
    assert aggregated.actor.weight_mean == pytest.approx(2.0)
    assert aggregated.actor.effective_sample_size == pytest.approx(64.0 / 18.0)


def test_global_step_aggregation_rejects_rank_schedule_drift() -> None:
    """Ranks cannot disagree about global step or actor update cadence."""
    with pytest.raises(RuntimeError, match="global_step"):
        ddp_module._aggregate_step_metrics(
            [
                _metric_fact(
                    rank=0,
                    batch_size=2,
                    metrics=_step_metrics(global_step=1),
                ),
                _metric_fact(
                    rank=1,
                    batch_size=2,
                    metrics=_step_metrics(global_step=2),
                ),
            ]
        )
    with pytest.raises(RuntimeError, match="actor update schedule"):
        ddp_module._aggregate_step_metrics(
            [
                _metric_fact(
                    rank=0,
                    batch_size=2,
                    metrics=_step_metrics(actor=None),
                ),
                _metric_fact(
                    rank=1,
                    batch_size=2,
                    metrics=_step_metrics(
                        actor=_actor_metrics(
                            loss=1.0,
                            weight_mean=1.0,
                            effective_sample_size=2.0,
                        )
                    ),
                ),
            ]
        )


def test_global_sampler_stats_are_summed_across_ranks() -> None:
    """Global sampler diagnostics should represent all rank-local work."""
    facts = [
        _metric_fact(rank=0, batch_size=2, metrics=_step_metrics()),
        _metric_fact(rank=1, batch_size=3, metrics=_step_metrics()),
    ]

    stats = ddp_module._aggregate_sampler_stats(facts)

    assert stats.batches_sampled == 3
    assert stats.transitions_sampled == 5
    assert stats.cache_hits == 1
    assert stats.cache_misses == 2
    assert stats.shard_load_seconds == pytest.approx(1.2)
    assert stats.sample_seconds == pytest.approx(1.4)


def test_gradient_loss_scale_changes_only_backward_objective() -> None:
    """DDP ownership scaling should scale gradients without changing loss."""
    module = nn.Linear(1, 1, bias=False)
    module.weight.data.fill_(1.0)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    loss = module.weight.square().sum()

    grad_norm = _optimizer_step(
        loss=loss,
        module=module,
        optimizer=optimizer,
        max_grad_norm=100.0,
        component="fixture",
        gradient_loss_scale=2.0,
    )

    assert float(loss.detach()) == pytest.approx(1.0)
    assert grad_norm == pytest.approx(4.0)
    assert float(module.weight.detach()) == pytest.approx(0.6)


def test_ownership_is_disjoint_complete_and_episode_weighted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ownership must cover the split and derive episode-uniform scales."""
    coordinator = _coordinator(
        tmp_path,
        world_size=2,
        episode_counts=(3, 7),
    )
    gathered = [
        {
            "rank": 0,
            "num_eligible_episodes": 3,
            "num_eligible_shards": 1,
            "shard_paths": ("shards/rank_00/shard_0000.pt",),
        },
        {
            "rank": 1,
            "num_eligible_episodes": 7,
            "num_eligible_shards": 1,
            "shard_paths": ("shards/rank_01/shard_0000.pt",),
        },
    ]
    monkeypatch.setattr(
        ddp_module,
        "all_gather_objects",
        lambda value, context: gathered,
    )
    monkeypatch.setattr(
        ddp_module,
        "broadcast_object",
        lambda value, context: value,
    )

    ownership = coordinator._collect_ownership()

    assert ownership[0]["loss_scale"] == pytest.approx(0.6)
    assert ownership[1]["loss_scale"] == pytest.approx(1.4)
    assert sum(
        item["num_eligible_episodes"] for item in ownership
    ) == 10


def test_ownership_rejects_overlapping_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two ranks cannot claim the same immutable trajectory shard."""
    coordinator = _coordinator(
        tmp_path,
        world_size=2,
        episode_counts=(1, 1),
    )
    gathered = [
        {
            "rank": rank,
            "num_eligible_episodes": 1,
            "num_eligible_shards": 1,
            "shard_paths": ("shards/shared.pt",),
        }
        for rank in range(2)
    ]
    monkeypatch.setattr(
        ddp_module,
        "all_gather_objects",
        lambda value, context: gathered,
    )
    monkeypatch.setattr(
        ddp_module,
        "broadcast_object",
        lambda value, context: value,
    )

    with pytest.raises(RuntimeError, match="overlaps"):
        coordinator._collect_ownership()


def test_recorder_preflight_uses_public_fresh_run_api(tmp_path: Path) -> None:
    """DDP fresh preflight should call the existing recorder contract."""
    coordinator = _coordinator(tmp_path / "run")
    local = LocalIQLRecorder(
        tmp_path / "run" / "ranks" / "rank_0000",
        rank=0,
        stream_to_terminal=False,
    )
    coordinator._global_recorder = LocalIQLRecorder(
        tmp_path / "run",
        rank=0,
        record_scope="global_aggregated",
        stream_to_terminal=False,
    )
    try:
        coordinator._preflight_recorders(local, None, -1, 0)
    finally:
        local.close()
        coordinator._global_recorder.close()


def test_resume_manifest_validates_rank_checkpoint_integrity(
    tmp_path: Path,
) -> None:
    """Resume should bind one rank checkpoint to size and SHA256."""
    coordinator = _coordinator(tmp_path)
    coordinator._ownership = (
        {
            "rank": 0,
            "num_eligible_episodes": 2,
            "num_eligible_shards": 1,
            "shard_paths_sha256": "a" * 64,
            "loss_scale": 1.0,
        },
    )
    checkpoint_path = tmp_path / "ranks/rank_0000/checkpoints/epoch_0000.pt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"checkpoint")
    scheduler_state = ddp_module._json_safe_scheduler_state(
        coordinator.scheduler.state_dict()
    )
    manifest = {
        "format": ddp_module.DDP_MANIFEST_FORMAT,
        "epoch": 0,
        "global_step": 2,
        "world_size": 1,
        "run_identity": coordinator.run_identity,
        "run_identity_sha256": coordinator.run_identity_sha256,
        "runner_config": asdict(coordinator.runner_config),
        "ownership": list(coordinator._ownership),
        "rank_checkpoints": [
            {
                "rank": 0,
                "path": checkpoint_path.relative_to(tmp_path).as_posix(),
                "size_bytes": checkpoint_path.stat().st_size,
                "sha256": ddp_module._sha256_file(checkpoint_path),
                "global_step": 2,
            }
        ],
        "shared_decision_scheduler": scheduler_state,
    }
    manifest_path = tmp_path / "checkpoints/epoch_0000/manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    restored_path, epoch, global_step = (
        coordinator._validate_resume_manifest(manifest_path)
    )

    assert restored_path == checkpoint_path
    assert epoch == 0
    assert global_step == 2

    checkpoint_path.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="integrity failure"):
        coordinator._validate_resume_manifest(manifest_path)


def test_rank_failure_does_not_publish_a_new_collective_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unsynchronized rank exception must not enter another collective."""
    coordinator = _coordinator(tmp_path)
    publish_calls: list[str] = []
    monkeypatch.setattr(coordinator, "_validate_components", lambda: None)
    monkeypatch.setattr(
        coordinator,
        "_collect_ownership",
        lambda: (
            {
                "rank": 0,
                "num_eligible_episodes": 2,
                "num_eligible_shards": 1,
                "shard_paths_sha256": "a" * 64,
                "loss_scale": 1.0,
            },
        ),
    )
    monkeypatch.setattr(coordinator, "_preflight_output", lambda resume: None)
    monkeypatch.setattr(
        coordinator,
        "_preflight_recorders",
        lambda *args: None,
    )
    monkeypatch.setattr(
        coordinator,
        "_publish_root_status",
        lambda status, **kwargs: publish_calls.append(status),
    )

    class _FailingRunner:
        """Runner double that injects a rank-local training failure."""

        def __init__(self, **kwargs: Any) -> None:
            """Accept the coordinator construction boundary."""
            del kwargs

        def run(self, **kwargs: Any) -> tuple[()]:
            """Raise before any synchronized step callback."""
            del kwargs
            raise RuntimeError("rank-local failure")

    monkeypatch.setattr(ddp_module, "OfflineIQLRunner", _FailingRunner)

    with pytest.raises(RuntimeError, match="rank-local failure"):
        coordinator.run()

    assert publish_calls == ["running"]
    status = json.loads(
        (tmp_path / "status.json").read_text(encoding="utf-8")
    )
    assert status["status"] == "failed"
