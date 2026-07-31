"""Two-process Gloo integration test for offline-IQL DDP fresh/resume."""

from __future__ import annotations

import csv
import json
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from ddpo.offline_rl.ddp import OfflineIQLDDPCoordinator
from ddpo.offline_rl.runner import IQLRunnerConfig
from ddpo.offline_rl.sampler import (
    DecisionTimeBin,
    SamplerStats,
    SharedDecisionIndexScheduler,
)
from ddpo.offline_rl.trainer import ActorStepMetrics, IQLStepMetrics
from ddpo.runtime import DistributedContext


class _DistributedFixtureSampler:
    """Rank-local deterministic sampler with uneven episode ownership."""

    def __init__(self, rank: int, world_size: int) -> None:
        """Initialize ownership, transition progress, and decision history."""
        self.rank = rank
        self.world_size = world_size
        self.time_bins = (DecisionTimeBin(0, 3),)
        self.same_decision_index_per_batch = True
        self.num_eligible_episodes = (3, 7)[rank]
        self.num_eligible_shards = 1
        self.shard_paths = (f"shards/rank_{rank:02d}/shard_0000.pt",)
        self.episode_index = SimpleNamespace(
            rows=[
                SimpleNamespace(sample_id=f"sample_{index}")
                for index in range(10)
            ]
        )
        self.split = "train"
        self.split_manifest = SimpleNamespace(split_for=lambda _: "train")
        self.samples = 0
        self.decisions: list[int] = []

    def sample_batch(
        self,
        batch_size: int,
        *,
        device: str,
        non_blocking: bool,
        decision_index: int | None = None,
    ) -> torch.Tensor:
        """Return a rank-distinct tensor and retain the shared decision."""
        del non_blocking
        if decision_index is None:
            raise ValueError("distributed fixture requires an explicit decision")
        self.samples += batch_size
        self.decisions.append(decision_index)
        return torch.full(
            (batch_size, 1),
            float(self.rank + 1),
            device=torch.device(device),
        )

    def stats(self) -> SamplerStats:
        """Return cumulative local fixture counters."""
        return SamplerStats(
            batches_sampled=len(self.decisions),
            transitions_sampled=self.samples,
            cache_hits=max(0, len(self.decisions) - 1),
            cache_misses=min(1, len(self.decisions)),
            shard_load_seconds=0.01,
            sample_seconds=0.02,
        )

    def state_dict(self) -> dict[str, Any]:
        """Serialize exact rank-local sample and decision progress."""
        return {
            "format": "distributed_fixture_sampler_v1",
            "rank": self.rank,
            "world_size": self.world_size,
            "samples": self.samples,
            "decisions": list(self.decisions),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore progress only for the same rank/world identity."""
        expected = {
            "format": "distributed_fixture_sampler_v1",
            "rank": self.rank,
            "world_size": self.world_size,
        }
        if any(state.get(key) != value for key, value in expected.items()):
            raise ValueError("distributed fixture sampler identity mismatch")
        self.samples = int(state["samples"])
        self.decisions = [int(value) for value in state["decisions"]]


class _DistributedFixtureTrainer:
    """Small real-DDP trainer implementing the runner checkpoint boundary."""

    def __init__(self) -> None:
        """Create three independently synchronized scalar modules."""
        self.online_q = DDP(nn.Linear(1, 1, bias=False))
        self.value = DDP(nn.Linear(1, 1, bias=False))
        policy = DDP(nn.Linear(1, 1, bias=False))
        self.actor_updater = SimpleNamespace(policy=policy)
        self._modules = (self.online_q, self.value, policy)
        self._optimizers = tuple(
            torch.optim.SGD(module.parameters(), lr=0.05)
            for module in self._modules
        )
        self.global_step = 0

    def step(
        self,
        transitions: torch.Tensor,
        *,
        gradient_loss_scale: float = 1.0,
    ) -> IQLStepMetrics:
        """Update all DDP modules in a fixed collective order."""
        losses: list[float] = []
        grad_norms: list[float] = []
        for target, module, optimizer in zip(
            (0.25, 0.5, 0.75),
            self._modules,
            self._optimizers,
        ):
            optimizer.zero_grad(set_to_none=True)
            prediction = module(transitions)
            loss = (prediction - target).square().mean()
            (loss * gradient_loss_scale).backward()
            gradient = next(module.parameters()).grad
            if gradient is None:
                raise RuntimeError("fixture DDP gradient is missing")
            grad_norms.append(float(gradient.norm().detach()))
            optimizer.step()
            losses.append(float(loss.detach()))
        step = self.global_step
        self.global_step += 1
        actor = ActorStepMetrics(
            loss=losses[2],
            log_prob_mean=-losses[2],
            weight_mean=1.0,
            weight_max=1.0,
            effective_sample_size=float(transitions.shape[0]),
            clipped_fraction=0.0,
            grad_norm=grad_norms[2],
        )
        return IQLStepMetrics(
            global_step=step,
            value_loss=losses[1],
            q_loss=losses[0],
            q1_loss=losses[0],
            q2_loss=losses[0],
            value_grad_norm=grad_norms[1],
            q_grad_norm=grad_norms[0],
            q1_mean=losses[0],
            q2_mean=losses[0],
            value_mean=losses[1],
            td_target_mean=0.0,
            td_target_std=0.0,
            td_target_min=0.0,
            td_target_max=0.0,
            advantage_mean=0.0,
            advantage_std=0.0,
            q_disagreement_mean=0.0,
            actor_updated=True,
            actor=actor,
            gradient_loss_scale=gradient_loss_scale,
        )

    def state_dict(self) -> dict[str, Any]:
        """Serialize synchronized modules, optimizers, and progress."""
        return {
            "format": "distributed_fixture_trainer_v1",
            "global_step": self.global_step,
            "modules": [
                module.module.state_dict() for module in self._modules
            ],
            "optimizers": [
                optimizer.state_dict() for optimizer in self._optimizers
            ],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a complete fixture trainer checkpoint."""
        if state.get("format") != "distributed_fixture_trainer_v1":
            raise ValueError("bad distributed fixture trainer state")
        for module, module_state in zip(
            self._modules,
            state["modules"],
        ):
            module.module.load_state_dict(module_state, strict=True)
        for optimizer, optimizer_state in zip(
            self._optimizers,
            state["optimizers"],
        ):
            optimizer.load_state_dict(optimizer_state)
        self.global_step = int(state["global_step"])

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        epoch: int,
        sampler_state: dict[str, Any],
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """Write the same outer checkpoint contract as OfflineIQLTrainer."""
        output_path = Path(path)
        torch.save(
            {
                "trainer": self.state_dict(),
                "sampler": sampler_state,
                "epoch": epoch,
                "extra": {} if extra is None else dict(extra),
            },
            output_path,
        )
        return output_path


def _module_values(trainer: _DistributedFixtureTrainer) -> tuple[float, ...]:
    """Return synchronized scalar parameters for cross-rank comparison."""
    return tuple(
        float(next(module.module.parameters()).detach().reshape(-1)[0])
        for module in trainer._modules
    )


def _build_coordinator(
    *,
    rank: int,
    world_size: int,
    output_dir: Path,
    epochs: int,
) -> tuple[
    OfflineIQLDDPCoordinator,
    _DistributedFixtureTrainer,
    _DistributedFixtureSampler,
]:
    """Build fresh DDP-wrapped fixture runtime components."""
    trainer = _DistributedFixtureTrainer()
    sampler = _DistributedFixtureSampler(rank, world_size)
    config = IQLRunnerConfig(
        epochs=epochs,
        steps_per_epoch=2,
        batch_size=2,
        device="cpu",
        log_every_steps=1,
    )
    context = DistributedContext(
        enabled=True,
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        device=torch.device("cpu"),
    )
    coordinator = OfflineIQLDDPCoordinator(
        trainer=trainer,  # type: ignore[arg-type]
        sampler=sampler,  # type: ignore[arg-type]
        runner_config=config,
        dist_ctx=context,
        output_dir=output_dir,
        run_identity={
            "dataset": "two_rank_fixture",
            "episode_index_sha256": "a" * 64,
            "reward_version": "fixture_reward_v1",
        },
        shared_decision_scheduler=SharedDecisionIndexScheduler(
            sampler.time_bins,
            seed=1,
        ),
    )
    return coordinator, trainer, sampler


def _gloo_worker(
    rank: int,
    world_size: int,
    init_path: str,
    output_path: str,
    result_path: str,
) -> None:
    """Run fresh epoch zero, rebuild runtime, and resume epoch one."""
    previous_gloo_interface = os.environ.get("GLOO_SOCKET_IFNAME")
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{init_path}",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=40),
        )
        output_dir = Path(output_path)
        first, first_trainer, first_sampler = _build_coordinator(
            rank=rank,
            world_size=world_size,
            output_dir=output_dir,
            epochs=1,
        )
        first.run()
        first_values = _module_values(first_trainer)
        gathered_first: list[object] = [None] * world_size
        dist.all_gather_object(gathered_first, first_values)
        if len(set(gathered_first)) != 1:
            raise RuntimeError(
                f"fresh DDP parameters diverged: {gathered_first}"
            )

        resumed, resumed_trainer, resumed_sampler = _build_coordinator(
            rank=rank,
            world_size=world_size,
            output_dir=output_dir,
            epochs=2,
        )
        resumed.run(
            resume_manifest=output_dir / "checkpoints" / "latest.json"
        )
        final_values = _module_values(resumed_trainer)
        gathered_values: list[object] = [None] * world_size
        gathered_decisions: list[object] = [None] * world_size
        dist.all_gather_object(gathered_values, final_values)
        dist.all_gather_object(
            gathered_decisions,
            tuple(resumed_sampler.decisions),
        )
        if len(set(gathered_values)) != 1:
            raise RuntimeError(
                f"resumed DDP parameters diverged: {gathered_values}"
            )
        if len(set(gathered_decisions)) != 1:
            raise RuntimeError(
                f"shared decisions diverged: {gathered_decisions}"
            )
        if resumed_trainer.global_step != 4:
            raise RuntimeError("resumed global step is not 4")
        if len(resumed_sampler.decisions) != 4:
            raise RuntimeError("resumed sampler did not retain four decisions")

        if rank == 0:
            latest = json.loads(
                (output_dir / "checkpoints" / "latest.json").read_text(
                    encoding="utf-8"
                )
            )
            manifest = json.loads(
                (output_dir / latest["manifest_path"]).read_text(
                    encoding="utf-8"
                )
            )
            Path(result_path).write_text(
                json.dumps(
                    {
                        "global_step": resumed_trainer.global_step,
                        "decisions": resumed_sampler.decisions,
                        "rank_checkpoints": len(
                            manifest["rank_checkpoints"]
                        ),
                        "scheduler_draw_count": manifest[
                            "shared_decision_scheduler"
                        ]["draw_count"],
                    }
                ),
                encoding="utf-8",
            )
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        if previous_gloo_interface is None:
            os.environ.pop("GLOO_SOCKET_IFNAME", None)
        else:
            os.environ["GLOO_SOCKET_IFNAME"] = previous_gloo_interface


@pytest.mark.skipif(
    not dist.is_available()
    or os.environ.get("DIFFSBDD_RUN_DDP_MULTIPROCESS_TEST") != "1",
    reason=(
        "requires torch.distributed and "
        "DIFFSBDD_RUN_DDP_MULTIPROCESS_TEST=1 outside the sandbox"
    ),
)
def test_two_rank_gloo_fresh_and_resume(tmp_path: Path) -> None:
    """Two real processes should synchronize, checkpoint, rebuild, and resume."""
    world_size = 2
    init_path = tmp_path / "gloo_init"
    output_dir = tmp_path / "run"
    result_path = tmp_path / "result.json"

    mp.spawn(
        _gloo_worker,
        args=(
            world_size,
            str(init_path),
            str(output_dir),
            str(result_path),
        ),
        nprocs=world_size,
        join=True,
    )

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["global_step"] == 4
    assert len(result["decisions"]) == 4
    assert result["rank_checkpoints"] == 2
    assert result["scheduler_draw_count"] == 4

    with (output_dir / "metrics" / "metrics.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        global_steps = list(csv.DictReader(handle))
    with (output_dir / "metrics" / "epochs.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        global_epochs = list(csv.DictReader(handle))
    assert [int(row["global_step"]) for row in global_steps] == [0, 1, 2, 3]
    assert [int(row["epoch"]) for row in global_epochs] == [0, 1]
    assert all(
        row["record_scope"] == "global_aggregated"
        for row in global_steps
    )
