"""Native DDP coordination for resumable DiffSBDD offline IQL runs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from ddpo.runtime import (
    DistributedContext,
    all_gather_objects,
    broadcast_object,
    is_main_process,
)

from .recording import EpochMetrics, LocalIQLRecorder
from .runner import IQLRunnerConfig, OfflineIQLRunner
from .sampler import (
    OfflineTransitionSampler,
    SamplerStats,
    SharedDecisionIndexScheduler,
)
from .trainer import ActorStepMetrics, IQLStepMetrics, OfflineIQLTrainer


DDP_MANIFEST_FORMAT = "diffsbdd_offline_iql_ddp_manifest_v1"
DDP_LATEST_FORMAT = "diffsbdd_offline_iql_ddp_latest_v1"


def _canonical_json(payload: object) -> str:
    """Serialize a small manifest object with deterministic key ordering."""
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _digest_json(payload: object) -> str:
    """Return a SHA256 digest for a JSON-serializable object."""
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _digest_paths(paths: tuple[str, ...]) -> str:
    """Return a stable digest for one rank's owned shard paths."""
    return hashlib.sha256("\n".join(paths).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    """Hash one durable file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON object with flush, fsync, and atomic replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _utc_now() -> str:
    """Return a UTC timestamp for manifest provenance."""
    return datetime.now(timezone.utc).isoformat()


def _json_safe_scheduler_state(state: dict[str, Any]) -> dict[str, Any]:
    """Convert the scheduler byte tensor into a manifest-safe JSON object."""
    safe = dict(state)
    generator_state = safe.get("generator_state")
    if not isinstance(generator_state, torch.Tensor):
        raise TypeError("scheduler state generator_state must be a tensor")
    safe["generator_state"] = generator_state.detach().cpu().tolist()
    return safe


def _scheduler_state_from_json(payload: object) -> dict[str, Any]:
    """Restore the scheduler byte tensor from a JSON manifest payload."""
    if not isinstance(payload, dict):
        raise TypeError("manifest shared_decision_scheduler must be an object")
    state = dict(payload)
    generator_state = state.get("generator_state")
    if not isinstance(generator_state, list):
        raise TypeError("manifest scheduler generator_state must be a list")
    time_bins = state.get("time_bins")
    if not isinstance(time_bins, list) or any(
        not isinstance(time_bin, list) or len(time_bin) != 3
        for time_bin in time_bins
    ):
        raise TypeError(
            "manifest scheduler time_bins must be a list of three-item lists"
        )
    state["generator_state"] = torch.tensor(generator_state, dtype=torch.uint8)
    state["time_bins"] = tuple(tuple(time_bin) for time_bin in time_bins)
    return state


def _combine_mean_std(
    means_and_stds: list[tuple[float, float, int]],
) -> tuple[float, float]:
    """Combine population means/stds without treating ranks as samples."""
    total = sum(count for _, _, count in means_and_stds)
    if total <= 0:
        raise ValueError("cannot combine statistics with no samples")
    mean = sum(mean * count for mean, _, count in means_and_stds) / total
    variance = sum(
        count * (std * std + (local_mean - mean) ** 2)
        for local_mean, std, count in means_and_stds
    ) / total
    return mean, math.sqrt(max(variance, 0.0))


def _weighted_optional_mean(values: list[tuple[float | None, int]]) -> float | None:
    """Average optional values by their update counts, preserving all-skipped."""
    present = [
        (float(value), int(weight))
        for value, weight in values
        if value is not None
    ]
    if not present:
        return None
    total_weight = sum(weight for _, weight in present)
    if total_weight <= 0:
        raise ValueError("actor loss values are present but have no updates")
    result = sum(value * weight for value, weight in present) / total_weight
    if not math.isfinite(result):
        raise FloatingPointError("global actor loss mean is not finite")
    return result


def _aggregate_step_metrics(
    facts: list[dict[str, Any]],
) -> IQLStepMetrics:
    """Aggregate one synchronized step using transition-count weighting."""
    if not facts:
        raise ValueError("cannot aggregate an empty DDP metric batch")
    first_metrics = facts[0]["metrics"]
    global_step = int(first_metrics["global_step"])
    if any(int(fact["metrics"]["global_step"]) != global_step for fact in facts):
        raise RuntimeError("DDP ranks disagree on global_step")
    counts = [int(fact["batch_size"]) for fact in facts]
    if any(count <= 0 for count in counts):
        raise ValueError("DDP batch sizes must be positive")
    total = sum(counts)

    def weighted(field: str) -> float:
        return sum(
            float(fact["metrics"][field]) * count
            for fact, count in zip(facts, counts)
        ) / total

    td_target_mean, td_target_std = _combine_mean_std(
        [
            (
                float(fact["metrics"]["td_target_mean"]),
                float(fact["metrics"]["td_target_std"]),
                count,
            )
            for fact, count in zip(facts, counts)
        ]
    )
    advantage_mean, advantage_std = _combine_mean_std(
        [
            (
                float(fact["metrics"]["advantage_mean"]),
                float(fact["metrics"]["advantage_std"]),
                count,
            )
            for fact, count in zip(facts, counts)
        ]
    )
    actor_values = [fact["metrics"]["actor"] for fact in facts]
    if any((value is None) != (actor_values[0] is None) for value in actor_values):
        raise RuntimeError("DDP ranks disagree on actor update schedule")
    actor: ActorStepMetrics | None = None
    if actor_values[0] is not None:
        actor_facts = [value for value in actor_values if value is not None]
        actor_count = sum(counts)
        weight_sum = sum(
            float(value["weight_mean"]) * count
            for value, count in zip(actor_facts, counts)
        )
        weight_sq_sum = sum(
            (
                float(value["weight_mean"]) * count
            )
            ** 2
            / max(float(value["effective_sample_size"]), 1e-12)
            for value, count in zip(actor_facts, counts)
        )
        actor = ActorStepMetrics(
            loss=sum(
                float(value["loss"]) * count
                for value, count in zip(actor_facts, counts)
            ) / actor_count,
            log_prob_mean=sum(
                float(value["log_prob_mean"]) * count
                for value, count in zip(actor_facts, counts)
            ) / actor_count,
            weight_mean=weight_sum / actor_count,
            weight_max=max(float(value["weight_max"]) for value in actor_facts),
            effective_sample_size=(weight_sum * weight_sum) / max(weight_sq_sum, 1e-12),
            clipped_fraction=sum(
                float(value["clipped_fraction"]) * count
                for value, count in zip(actor_facts, counts)
            ) / actor_count,
            grad_norm=sum(
                float(value["grad_norm"]) * count
                for value, count in zip(actor_facts, counts)
            ) / actor_count,
        )

    return IQLStepMetrics(
        global_step=global_step,
        value_loss=weighted("value_loss"),
        q_loss=weighted("q_loss"),
        q1_loss=weighted("q1_loss"),
        q2_loss=weighted("q2_loss"),
        value_grad_norm=weighted("value_grad_norm"),
        q_grad_norm=weighted("q_grad_norm"),
        q1_mean=weighted("q1_mean"),
        q2_mean=weighted("q2_mean"),
        value_mean=weighted("value_mean"),
        td_target_mean=td_target_mean,
        td_target_std=td_target_std,
        td_target_min=min(float(fact["metrics"]["td_target_min"]) for fact in facts),
        td_target_max=max(float(fact["metrics"]["td_target_max"]) for fact in facts),
        advantage_mean=advantage_mean,
        advantage_std=advantage_std,
        q_disagreement_mean=weighted("q_disagreement_mean"),
        actor_updated=actor is not None,
        actor=actor,
        gradient_loss_scale=weighted("gradient_loss_scale"),
    )


def _aggregate_sampler_stats(facts: list[dict[str, Any]]) -> SamplerStats:
    """Sum rank-local cumulative sampler counters and timings."""
    fields = (
        "batches_sampled",
        "transitions_sampled",
        "cache_hits",
        "cache_misses",
        "shard_load_seconds",
        "sample_seconds",
    )
    values = {
        field: sum(float(fact["sampler_stats"][field]) for fact in facts)
        for field in fields
    }
    return SamplerStats(
        batches_sampled=int(round(values["batches_sampled"])),
        transitions_sampled=int(round(values["transitions_sampled"])),
        cache_hits=int(round(values["cache_hits"])),
        cache_misses=int(round(values["cache_misses"])),
        shard_load_seconds=values["shard_load_seconds"],
        sample_seconds=values["sample_seconds"],
    )


class OfflineIQLDDPCoordinator:
    """Coordinate rank-local IQL runners and publish global epoch manifests."""

    def __init__(
        self,
        *,
        trainer: OfflineIQLTrainer,
        sampler: OfflineTransitionSampler,
        runner_config: IQLRunnerConfig,
        dist_ctx: DistributedContext,
        output_dir: str | Path,
        run_identity: dict[str, Any],
        shared_decision_scheduler: SharedDecisionIndexScheduler,
    ) -> None:
        """Store components and validate immutable DDP identity inputs."""
        if dist_ctx.world_size <= 0 or dist_ctx.rank < 0:
            raise ValueError("invalid distributed context")
        if dist_ctx.rank >= dist_ctx.world_size:
            raise ValueError("rank must be smaller than world_size")
        if not run_identity:
            raise ValueError("run_identity cannot be empty")
        _canonical_json(run_identity)
        if tuple(sampler.time_bins) != tuple(shared_decision_scheduler.time_bins):
            raise ValueError("sampler and shared scheduler time bins differ")
        self.trainer = trainer
        self.sampler = sampler
        self.runner_config = runner_config
        self.dist_ctx = dist_ctx
        self.output_dir = Path(output_dir).resolve()
        self.run_identity = dict(run_identity)
        self.run_identity_sha256 = _digest_json(self.run_identity)
        self.scheduler = shared_decision_scheduler
        self.rank = int(dist_ctx.rank)
        self.world_size = int(dist_ctx.world_size)
        self.rank_loss_scale = 1.0
        self._ownership: tuple[dict[str, Any], ...] = ()
        self._last_scheduler_state: dict[str, Any] | None = None
        self._global_recorder: LocalIQLRecorder | None = None
        self._run_started = 0.0
        self._current_decision_index: int | None = None
        self._resolved_resume_manifest: Path | None = None

    def run(
        self,
        *,
        resume_manifest: str | Path | None = None,
    ) -> tuple[EpochMetrics, ...]:
        """Run all configured epochs and return this rank's local summaries."""
        self._validate_components()
        self._ownership = self._collect_ownership()
        self.rank_loss_scale = float(self._ownership[self.rank]["loss_scale"])
        self._preflight_output(resume_manifest)

        resume_checkpoint: Path | None = None
        resume_epoch = -1
        resume_global_step = 0
        if resume_manifest is not None:
            (
                resume_checkpoint,
                resume_epoch,
                resume_global_step,
            ) = self._validate_resume_manifest(Path(resume_manifest))

        local_output = self.output_dir / "ranks" / f"rank_{self.rank:04d}"
        local_recorder = LocalIQLRecorder(
            local_output,
            rank=self.rank,
            record_scope="rank_local",
            stream_to_terminal=False,
        )
        if is_main_process(self.dist_ctx):
            self._global_recorder = LocalIQLRecorder(
                self.output_dir,
                rank=0,
                record_scope="global_aggregated",
                stream_to_terminal=True,
            )

        try:
            self._preflight_recorders(
                local_recorder,
                resume_manifest,
                resume_epoch,
                resume_global_step,
            )
            if resume_manifest is not None:
                self._restore_scheduler_from_manifest(Path(resume_manifest))
            self._publish_root_status(
                "running",
                epoch=resume_epoch + 1 if resume_manifest is not None else 0,
                global_step=resume_global_step,
            )
            self._run_started = time.perf_counter()
            runner = OfflineIQLRunner(
                trainer=self.trainer,
                sampler=self.sampler,
                recorder=local_recorder,
                config=self.runner_config,
                rank=self.rank,
            )
            summaries = runner.run(
                resume_checkpoint=resume_checkpoint,
                decision_index_provider=self._decision_index_provider,
                step_callback=self._step_callback,
                epoch_callback=self._epoch_callback,
                checkpoint_extra_factory=self._checkpoint_extra_factory,
                gradient_loss_scale=self.rank_loss_scale,
            )
            self._publish_root_status(
                "completed",
                epoch=self.runner_config.epochs - 1,
                global_step=self.trainer.global_step,
            )
            return summaries
        except Exception as exc:
            local_recorder.logger.exception(
                "DDP rank failed global_step=%d",
                self.trainer.global_step,
            )
            try:
                local_recorder.record_status(
                    "failed",
                    epoch=None,
                    global_step=self.trainer.global_step,
                    error=f"{type(exc).__name__}: {exc}",
                )
            except Exception:
                local_recorder.logger.exception(
                    "failed to write rank-local failure status"
                )
            if self._global_recorder is not None:
                try:
                    self._global_recorder.record_status(
                        "failed",
                        epoch=None,
                        global_step=self.trainer.global_step,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                except Exception:
                    self._global_recorder.logger.exception(
                        "failed to write root failure status"
                    )
            raise
        finally:
            local_recorder.close()
            if self._global_recorder is not None:
                self._global_recorder.close()

    def _validate_components(self) -> None:
        """Fail before training if rank ownership or DDP wrappers are unsafe."""
        local_error: str | None = None
        try:
            if (
                self.sampler.rank != self.rank
                or self.sampler.world_size != self.world_size
            ):
                raise ValueError(
                    "sampler rank/world_size does not match distributed context"
                )
            if self.world_size > 1:
                modules = (
                    self.trainer.online_q,
                    self.trainer.value,
                    self.trainer.actor_updater.policy,
                )
                if not all(isinstance(module, DDP) for module in modules):
                    raise TypeError(
                        "world_size > 1 requires DDP-wrapped online_q, value, "
                        "and policy"
                    )
                if not self.sampler.same_decision_index_per_batch:
                    raise ValueError(
                        "formal DDP requires same_decision_index_per_batch=True"
                    )
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        gathered = all_gather_objects(local_error, self.dist_ctx)
        first_error = next((item for item in gathered if item), None)
        first_error = broadcast_object(first_error, self.dist_ctx)
        if first_error:
            raise RuntimeError(f"offline IQL DDP preflight failed: {first_error}")

    def _collect_ownership(self) -> tuple[dict[str, Any], ...]:
        """Gather shard ownership and derive episode-uniform gradient scales."""
        local = {
            "rank": self.rank,
            "num_eligible_episodes": self.sampler.num_eligible_episodes,
            "num_eligible_shards": self.sampler.num_eligible_shards,
            "shard_paths": tuple(self.sampler.shard_paths),
        }
        gathered = all_gather_objects(local, self.dist_ctx)
        validation_error: str | None = None
        ownership: list[dict[str, Any]] = []
        if is_main_process(self.dist_ctx):
            try:
                by_rank = {int(item["rank"]): item for item in gathered}
                if set(by_rank) != set(range(self.world_size)):
                    raise ValueError("DDP ownership ranks are incomplete")
                seen: set[str] = set()
                total_episodes = sum(
                    int(item["num_eligible_episodes"]) for item in gathered
                )
                expected_episodes = sum(
                    1
                    for row in self.sampler.episode_index.rows
                    if self.sampler.split_manifest.split_for(row.sample_id)
                    == self.sampler.split
                )
                if total_episodes != expected_episodes:
                    raise ValueError(
                        "rank ownership does not cover the configured split: "
                        f"observed={total_episodes} expected={expected_episodes}"
                    )
                if total_episodes <= 0:
                    raise ValueError("configured split has no eligible episodes")
                for item in sorted(gathered, key=lambda value: int(value["rank"])):
                    paths = tuple(item["shard_paths"])
                    overlap = seen.intersection(paths)
                    if overlap:
                        raise ValueError(
                            f"rank shard ownership overlaps: {sorted(overlap)[:3]}"
                        )
                    seen.update(paths)
                    count = int(item["num_eligible_episodes"])
                    ownership.append(
                        {
                            "rank": int(item["rank"]),
                            "num_eligible_episodes": count,
                            "num_eligible_shards": int(item["num_eligible_shards"]),
                            "shard_paths_sha256": _digest_paths(paths),
                            "loss_scale": self.world_size * count / total_episodes,
                        }
                    )
            except Exception as exc:
                validation_error = f"{type(exc).__name__}: {exc}"
        result = broadcast_object(
            {
                "error": validation_error,
                "ownership": ownership,
            },
            self.dist_ctx,
        )
        if not isinstance(result, dict) or result.get("error"):
            raise RuntimeError(
                "offline IQL ownership preflight failed: "
                f"{None if not isinstance(result, dict) else result.get('error')}"
            )
        received = result.get("ownership")
        if not isinstance(received, list) or len(received) != self.world_size:
            raise RuntimeError("DDP ownership payload is incomplete")
        return tuple(received)

    def _preflight_output(self, resume_manifest: str | Path | None) -> None:
        """Reject accidental fresh-run reuse before any rank writes evidence."""
        local_error: str | None = None
        try:
            if resume_manifest is None:
                allowed_bootstrap_names = {
                    "config",
                    "logs",
                    "launch.log",
                    "run_manifest.json",
                }
                unexpected = (
                    []
                    if not self.output_dir.exists()
                    else sorted(
                        path.name
                        for path in self.output_dir.iterdir()
                        if path.name not in allowed_bootstrap_names
                    )
                )
                if unexpected:
                    raise FileExistsError(
                        "fresh DDP output directory contains unexpected entries: "
                        f"{unexpected}"
                    )
            elif not self.output_dir.exists():
                raise FileNotFoundError(
                    f"resume output directory does not exist: {self.output_dir}"
                )
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        errors = all_gather_objects(local_error, self.dist_ctx)
        first_error = next((item for item in errors if item), None)
        first_error = broadcast_object(first_error, self.dist_ctx)
        if first_error:
            raise RuntimeError(f"DDP output preflight failed: {first_error}")

    def _preflight_recorders(
        self,
        local_recorder: LocalIQLRecorder,
        resume_manifest: str | Path | None,
        checkpoint_epoch: int,
        global_step: int,
    ) -> None:
        """Check rank-local and root-global evidence without a partial start."""
        local_error: str | None = None
        try:
            if resume_manifest is None:
                local_recorder.prepare_fresh_run()
                if self._global_recorder is not None:
                    self._global_recorder.prepare_fresh_run()
            else:
                local_recorder.prepare_resume(
                    checkpoint_epoch=checkpoint_epoch,
                    trainer_global_step=global_step,
                )
                if self._global_recorder is not None:
                    self._global_recorder.prepare_resume(
                        checkpoint_epoch=checkpoint_epoch,
                        trainer_global_step=global_step,
                    )
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        errors = all_gather_objects(local_error, self.dist_ctx)
        first_error = next((item for item in errors if item), None)
        first_error = broadcast_object(first_error, self.dist_ctx)
        if first_error:
            raise RuntimeError(f"DDP recorder preflight failed: {first_error}")

    def _publish_root_status(
        self,
        status: str,
        *,
        epoch: int | None,
        global_step: int,
        error: str | None = None,
    ) -> None:
        """Publish root status and synchronize a root write failure."""
        root_error: str | None = None
        if self._global_recorder is not None:
            try:
                self._global_recorder.record_status(
                    status,
                    epoch=epoch,
                    global_step=global_step,
                    error=error,
                )
            except Exception as exc:
                root_error = f"{type(exc).__name__}: {exc}"
        root_error = broadcast_object(root_error, self.dist_ctx)
        if root_error:
            raise RuntimeError(f"root status write failed: {root_error}")

    def _decision_index_provider(self, epoch: int, epoch_step: int) -> int:
        """Draw on rank zero and broadcast one decision to every rank."""
        del epoch, epoch_step
        decision_index: int | None = None
        if is_main_process(self.dist_ctx):
            decision_index = self.scheduler.next_decision_index()
        received = broadcast_object(decision_index, self.dist_ctx)
        if isinstance(received, bool) or not isinstance(received, int):
            raise RuntimeError("DDP shared decision index is not an integer")
        self._current_decision_index = received
        return received

    def _checkpoint_extra_factory(self, epoch: int) -> dict[str, Any]:
        """Attach DDP identity and the rank-zero scheduler state to each rank file."""
        del epoch
        scheduler_state = self.scheduler.state_dict()
        if is_main_process(self.dist_ctx):
            self._last_scheduler_state = scheduler_state
        return {
            "format": "diffsbdd_offline_iql_ddp_rank_extra_v1",
            "rank": self.rank,
            "world_size": self.world_size,
            "run_identity_sha256": self.run_identity_sha256,
            "rank_loss_scale": self.rank_loss_scale,
            "shared_decision_scheduler": (
                _json_safe_scheduler_state(scheduler_state)
                if is_main_process(self.dist_ctx)
                else None
            ),
        }

    def _step_callback(
        self,
        epoch: int,
        epoch_step: int,
        metrics: IQLStepMetrics,
        sampler_stats: SamplerStats,
    ) -> None:
        """Aggregate and durably record one synchronized global step."""
        fact = {
            "rank": self.rank,
            "epoch": epoch,
            "epoch_step": epoch_step,
            "batch_size": self.runner_config.batch_size,
            "decision_index": self._current_decision_index,
            "metrics": asdict(metrics),
            "sampler_stats": asdict(sampler_stats),
        }
        gathered = all_gather_objects(fact, self.dist_ctx)
        root_error: str | None = None
        if self._global_recorder is not None:
            try:
                expected_decision = gathered[0]["decision_index"]
                if any(
                    item["decision_index"] != expected_decision
                    for item in gathered
                ):
                    raise RuntimeError("DDP ranks disagree on decision_index")
                global_metrics = _aggregate_step_metrics(gathered)
                global_stats = _aggregate_sampler_stats(gathered)
                self._global_recorder.record_step(
                    epoch=epoch,
                    epoch_step=epoch_step,
                    elapsed_seconds=time.perf_counter() - self._run_started,
                    metrics=global_metrics,
                    sampler_stats=global_stats,
                )
                self._global_recorder.logger.info(
                    "global progress epoch=%d epoch_step=%d/%d global_step=%d "
                    "value_loss=%.6f q_loss=%.6f actor_loss=%s decision_index=%d",
                    epoch,
                    epoch_step + 1,
                    self.runner_config.steps_per_epoch,
                    global_metrics.global_step,
                    global_metrics.value_loss,
                    global_metrics.q_loss,
                    (
                        "skipped"
                        if global_metrics.actor is None
                        else f"{global_metrics.actor.loss:.6f}"
                    ),
                    expected_decision,
                )
            except Exception as exc:
                root_error = f"{type(exc).__name__}: {exc}"
        root_error = broadcast_object(root_error, self.dist_ctx)
        if root_error:
            raise RuntimeError(f"global step recording failed: {root_error}")

    def _epoch_callback(self, summary: EpochMetrics) -> None:
        """Publish an all-rank checkpoint manifest before local epoch evidence."""
        checkpoint_path = Path(summary.checkpoint_path).resolve()
        try:
            relative_path = checkpoint_path.relative_to(self.output_dir).as_posix()
            checkpoint_size = checkpoint_path.stat().st_size
            checkpoint_sha256 = _sha256_file(checkpoint_path)
            local_error: str | None = None
        except Exception as exc:
            relative_path = ""
            checkpoint_size = -1
            checkpoint_sha256 = ""
            local_error = f"{type(exc).__name__}: {exc}"
        fact = {
            "rank": self.rank,
            "summary": asdict(summary),
            "relative_path": relative_path,
            "size_bytes": checkpoint_size,
            "sha256": checkpoint_sha256,
            "error": local_error,
        }
        gathered = all_gather_objects(fact, self.dist_ctx)
        root_error: str | None = next(
            (item["error"] for item in gathered if item.get("error")),
            None,
        )
        if self._global_recorder is not None and root_error is None:
            try:
                epochs = {int(item["summary"]["epoch"]) for item in gathered}
                global_steps = {
                    int(item["summary"]["global_step"]) for item in gathered
                }
                if epochs != {summary.epoch} or len(global_steps) != 1:
                    raise RuntimeError(
                        "DDP ranks disagree on epoch checkpoint identity"
                    )
                rank_checkpoints = [
                    {
                        "rank": int(item["rank"]),
                        "path": item["relative_path"],
                        "size_bytes": int(item["size_bytes"]),
                        "sha256": item["sha256"],
                        "global_step": int(item["summary"]["global_step"]),
                    }
                    for item in sorted(gathered, key=lambda value: int(value["rank"]))
                ]
                if len(rank_checkpoints) != self.world_size:
                    raise RuntimeError(
                        "DDP checkpoint manifest rank count is incomplete"
                    )
                if self._last_scheduler_state is None:
                    raise RuntimeError("rank-zero scheduler state is missing")
                manifest_dir = (
                    self.output_dir
                    / "checkpoints"
                    / f"epoch_{summary.epoch:04d}"
                )
                manifest_path = manifest_dir / "manifest.json"
                manifest = {
                    "format": DDP_MANIFEST_FORMAT,
                    "epoch": summary.epoch,
                    "global_step": summary.global_step,
                    "world_size": self.world_size,
                    "run_identity": self.run_identity,
                    "run_identity_sha256": self.run_identity_sha256,
                    "runner_config": asdict(self.runner_config),
                    "ownership": list(self._ownership),
                    "rank_checkpoints": rank_checkpoints,
                    "shared_decision_scheduler": _json_safe_scheduler_state(
                        self._last_scheduler_state
                    ),
                    "created_at": _utc_now(),
                }
                _atomic_write_json(manifest_path, manifest)
                manifest_sha256 = _sha256_file(manifest_path)
                latest_path = self.output_dir / "checkpoints" / "latest.json"
                _atomic_write_json(
                    latest_path,
                    {
                        "format": DDP_LATEST_FORMAT,
                        "epoch": summary.epoch,
                        "global_step": summary.global_step,
                        "manifest_path": manifest_path.relative_to(
                            self.output_dir
                        ).as_posix(),
                        "manifest_sha256": manifest_sha256,
                    },
                )
                global_summary = EpochMetrics(
                    epoch=summary.epoch,
                    rank=0,
                    global_step=summary.global_step,
                    steps=summary.steps,
                    elapsed_seconds=max(
                        float(item["summary"]["elapsed_seconds"])
                        for item in gathered
                    ),
                    value_loss_mean=sum(
                        float(item["summary"]["value_loss_mean"])
                        for item in gathered
                    ) / self.world_size,
                    q_loss_mean=sum(
                        float(item["summary"]["q_loss_mean"])
                        for item in gathered
                    ) / self.world_size,
                    actor_updates=sum(
                        int(item["summary"]["actor_updates"])
                        for item in gathered
                    ),
                    actor_loss_mean=_weighted_optional_mean(
                        [
                            (
                                item["summary"]["actor_loss_mean"],
                                int(item["summary"]["actor_updates"]),
                            )
                            for item in gathered
                        ]
                    ),
                    checkpoint_path=str(manifest_path),
                )
                self._global_recorder.record_epoch(global_summary)
                self._global_recorder.record_status(
                    "running",
                    epoch=summary.epoch,
                    global_step=summary.global_step,
                )
            except Exception as exc:
                root_error = f"{type(exc).__name__}: {exc}"
        root_error = broadcast_object(root_error, self.dist_ctx)
        if root_error:
            raise RuntimeError(f"global checkpoint manifest failed: {root_error}")

    def _validate_resume_manifest(
        self,
        manifest_input: Path,
    ) -> tuple[Path, int, int]:
        """Validate one complete manifest and return this rank's checkpoint."""
        manifest_path = self._resolve_manifest_path(manifest_input)
        self._resolved_resume_manifest = manifest_path
        manifest: dict[str, Any] | None = None
        root_error: str | None = None
        if is_main_process(self.dist_ctx):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("format") != DDP_MANIFEST_FORMAT:
                    raise ValueError("unsupported DDP manifest format")
                if int(manifest["world_size"]) != self.world_size:
                    raise ValueError("DDP manifest world_size mismatch")
                if manifest.get("run_identity_sha256") != self.run_identity_sha256:
                    raise ValueError("DDP manifest run identity mismatch")
                if manifest.get("run_identity") != self.run_identity:
                    raise ValueError("DDP manifest run identity payload mismatch")
                if manifest.get("ownership") != list(self._ownership):
                    raise ValueError("DDP manifest shard ownership mismatch")
                saved_runner_config = manifest.get("runner_config")
                current_runner_config = asdict(self.runner_config)
                for field in (
                    "steps_per_epoch",
                    "batch_size",
                    "checkpoint_every_epochs",
                    "non_blocking_transfer",
                ):
                    if saved_runner_config.get(field) != current_runner_config[field]:
                        raise ValueError(
                            f"DDP manifest runner config mismatch for {field}"
                        )
                if int(manifest["epoch"]) < 0 or int(manifest["global_step"]) < 0:
                    raise ValueError(
                        "DDP manifest epoch/global_step must be non-negative"
                    )
            except Exception as exc:
                root_error = f"{type(exc).__name__}: {exc}"
        root_error = broadcast_object(root_error, self.dist_ctx)
        if root_error:
            raise RuntimeError(f"DDP resume manifest validation failed: {root_error}")
        manifest = broadcast_object(manifest, self.dist_ctx)
        if not isinstance(manifest, dict):
            raise RuntimeError("DDP resume manifest payload is missing")
        records = manifest.get("rank_checkpoints")
        if not isinstance(records, list):
            raise RuntimeError("DDP resume manifest rank_checkpoints is missing")
        record = next(
            (item for item in records if int(item["rank"]) == self.rank),
            None,
        )
        if not isinstance(record, dict):
            raise RuntimeError(
                f"DDP resume manifest has no rank {self.rank} checkpoint"
            )
        checkpoint_path = (self.output_dir / str(record["path"])).resolve()
        try:
            checkpoint_path.relative_to(self.output_dir)
            observed_size = checkpoint_path.stat().st_size
            observed_sha256 = _sha256_file(checkpoint_path)
            local_fact = {
                "rank": self.rank,
                "ok": (
                    observed_size == int(record["size_bytes"])
                    and observed_sha256 == str(record["sha256"])
                ),
                "size_bytes": observed_size,
                "sha256": observed_sha256,
                "error": None,
            }
        except Exception as exc:
            local_fact = {
                "rank": self.rank,
                "ok": False,
                "size_bytes": -1,
                "sha256": "",
                "error": f"{type(exc).__name__}: {exc}",
            }
        facts = all_gather_objects(local_fact, self.dist_ctx)
        integrity_error: str | None = None
        if is_main_process(self.dist_ctx):
            bad = [item for item in facts if not item.get("ok")]
            if bad:
                integrity_error = f"rank checkpoint integrity failure: {bad}"
        integrity_error = broadcast_object(integrity_error, self.dist_ctx)
        if integrity_error:
            raise RuntimeError(integrity_error)
        return checkpoint_path, int(manifest["epoch"]), int(manifest["global_step"])

    def _resolve_manifest_path(self, input_path: Path) -> Path:
        """Resolve a manifest or latest pointer through one collective decision."""
        resolved: str | None = None
        error: str | None = None
        if is_main_process(self.dist_ctx):
            try:
                candidate = input_path.resolve()
                payload = json.loads(candidate.read_text(encoding="utf-8"))
                if payload.get("format") == DDP_LATEST_FORMAT:
                    manifest_path = (
                        self.output_dir / str(payload["manifest_path"])
                    ).resolve()
                    manifest_path.relative_to(self.output_dir)
                    if _sha256_file(manifest_path) != payload.get("manifest_sha256"):
                        raise ValueError("latest pointer manifest SHA256 mismatch")
                    resolved = str(manifest_path)
                else:
                    candidate.relative_to(self.output_dir)
                    resolved = str(candidate)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        result = broadcast_object({"path": resolved, "error": error}, self.dist_ctx)
        if not isinstance(result, dict) or result.get("error"):
            raise RuntimeError(
                "DDP manifest path resolution failed: "
                f"{None if not isinstance(result, dict) else result.get('error')}"
            )
        path = result.get("path")
        if not isinstance(path, str) or not path:
            raise RuntimeError("DDP manifest path resolution returned no path")
        return Path(path)

    def _restore_scheduler_from_manifest(self, manifest_input: Path) -> None:
        """Broadcast and restore the rank-zero scheduler state on every rank."""
        manifest_path = self._resolved_resume_manifest or self._resolve_manifest_path(
            manifest_input
        )
        payload: object = None
        if is_main_process(self.dist_ctx):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = manifest.get("shared_decision_scheduler")
        payload = broadcast_object(payload, self.dist_ctx)
        self.scheduler.load_state_dict(_scheduler_state_from_json(payload))
