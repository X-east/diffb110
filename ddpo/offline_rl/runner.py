"""Epoch-level orchestration and exact resume for DiffSBDD offline IQL."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from .recording import EpochMetrics, LocalIQLRecorder
from .sampler import OfflineTransitionSampler, SamplerStats
from .trainer import IQLStepMetrics, OfflineIQLTrainer


@dataclass(frozen=True)
class IQLRunnerConfig:
    """Runtime controls for a finite epoch-based offline IQL run."""

    epochs: int
    steps_per_epoch: int
    batch_size: int
    device: str = "cpu"
    log_every_steps: int = 10
    checkpoint_every_epochs: int = 1
    non_blocking_transfer: bool = False

    def __post_init__(self) -> None:
        """Validate formal-run cadence and positive work sizes."""
        for name, value in (
            ("epochs", self.epochs),
            ("steps_per_epoch", self.steps_per_epoch),
            ("batch_size", self.batch_size),
            ("log_every_steps", self.log_every_steps),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.checkpoint_every_epochs != 1:
            raise ValueError(
                "formal offline IQL requires checkpoint_every_epochs == 1"
            )
        torch.device(self.device)


DecisionIndexProvider = Callable[[int, int], int | None]
StepCallback = Callable[[int, int, IQLStepMetrics, SamplerStats], None]
EpochCallback = Callable[[EpochMetrics], None]
CheckpointExtraFactory = Callable[[int], dict[str, Any]]


class OfflineIQLRunner:
    """Connect a sampler, IQL trainer, recorder, and epoch checkpoints."""

    def __init__(
        self,
        *,
        trainer: OfflineIQLTrainer,
        sampler: OfflineTransitionSampler,
        recorder: LocalIQLRecorder,
        config: IQLRunnerConfig,
        rank: int = 0,
    ) -> None:
        """Store decoupled runtime components and validate rank identity."""
        if rank < 0:
            raise ValueError("rank must be non-negative")
        if recorder.rank != rank:
            raise ValueError("runner and recorder rank must match")
        self.trainer = trainer
        self.sampler = sampler
        self.recorder = recorder
        self.config = config
        self.rank = rank

    def run(
        self,
        *,
        resume_checkpoint: str | Path | None = None,
        decision_index_provider: DecisionIndexProvider | None = None,
        step_callback: StepCallback | None = None,
        epoch_callback: EpochCallback | None = None,
        checkpoint_extra_factory: CheckpointExtraFactory | None = None,
        gradient_loss_scale: float = 1.0,
    ) -> tuple[EpochMetrics, ...]:
        """Train through configured epochs and return completed summaries."""
        if not math.isfinite(gradient_loss_scale) or gradient_loss_scale <= 0:
            raise ValueError("gradient_loss_scale must be finite and positive")
        started = time.perf_counter()
        current_epoch: int | None = None
        summaries: list[EpochMetrics] = []
        evidence_prepared = False
        try:
            if resume_checkpoint is None:
                self.recorder.prepare_fresh_run()
                start_epoch = 0
            else:
                start_epoch = self._restore_checkpoint(resume_checkpoint)
            evidence_prepared = True
            self.recorder.record_status(
                "running",
                epoch=start_epoch,
                global_step=self.trainer.global_step,
            )
            self.recorder.logger.info(
                "run started start_epoch=%d epochs=%d steps_per_epoch=%d "
                "batch_size=%d device=%s resume=%s",
                start_epoch,
                self.config.epochs,
                self.config.steps_per_epoch,
                self.config.batch_size,
                self.config.device,
                resume_checkpoint,
            )

            for epoch in range(start_epoch, self.config.epochs):
                current_epoch = epoch
                summary = self._run_epoch(
                    epoch,
                    run_started=started,
                    decision_index_provider=decision_index_provider,
                    step_callback=step_callback,
                    epoch_callback=epoch_callback,
                    checkpoint_extra_factory=checkpoint_extra_factory,
                    gradient_loss_scale=gradient_loss_scale,
                )
                summaries.append(summary)

            final_epoch = (
                self.config.epochs - 1
                if self.config.epochs > 0
                else current_epoch
            )
            self.recorder.record_status(
                "completed",
                epoch=final_epoch,
                global_step=self.trainer.global_step,
            )
            self.recorder.logger.info(
                "run completed final_epoch=%s global_step=%d elapsed=%.3fs",
                final_epoch,
                self.trainer.global_step,
                time.perf_counter() - started,
            )
            return tuple(summaries)
        except Exception as exc:
            if evidence_prepared:
                self.recorder.logger.exception(
                    "run failed epoch=%s global_step=%d",
                    current_epoch,
                    self.trainer.global_step,
                )
                self.recorder.record_status(
                    "failed",
                    epoch=current_epoch,
                    global_step=self.trainer.global_step,
                    error=f"{type(exc).__name__}: {exc}",
                )
            raise

    def _run_epoch(
        self,
        epoch: int,
        *,
        run_started: float,
        decision_index_provider: DecisionIndexProvider | None,
        step_callback: StepCallback | None,
        epoch_callback: EpochCallback | None,
        checkpoint_extra_factory: CheckpointExtraFactory | None,
        gradient_loss_scale: float,
    ) -> EpochMetrics:
        """Run one epoch, checkpoint it, then publish its summary row."""
        epoch_started = time.perf_counter()
        step_metrics: list[IQLStepMetrics] = []
        for epoch_step in range(self.config.steps_per_epoch):
            decision_index = (
                None
                if decision_index_provider is None
                else decision_index_provider(epoch, epoch_step)
            )
            sample_kwargs: dict[str, Any] = {
                "device": self.config.device,
                "non_blocking": self.config.non_blocking_transfer,
            }
            if decision_index is not None:
                sample_kwargs["decision_index"] = decision_index
            transitions = self.sampler.sample_batch(
                self.config.batch_size,
                **sample_kwargs,
            )
            if gradient_loss_scale == 1.0:
                metrics = self.trainer.step(transitions)
            else:
                metrics = self.trainer.step(
                    transitions,
                    gradient_loss_scale=gradient_loss_scale,
                )
            step_metrics.append(metrics)
            elapsed = time.perf_counter() - run_started
            self.recorder.record_step(
                epoch=epoch,
                epoch_step=epoch_step,
                elapsed_seconds=elapsed,
                metrics=metrics,
                sampler_stats=self.sampler.stats(),
            )
            if step_callback is not None:
                step_callback(
                    epoch,
                    epoch_step,
                    metrics,
                    self.sampler.stats(),
                )
            if (
                epoch_step == 0
                or (epoch_step + 1) % self.config.log_every_steps == 0
                or epoch_step + 1 == self.config.steps_per_epoch
            ):
                actor_loss = (
                    "skipped"
                    if metrics.actor is None
                    else f"{metrics.actor.loss:.6f}"
                )
                self.recorder.logger.info(
                    "progress epoch=%d epoch_step=%d/%d global_step=%d "
                    "elapsed=%.3fs value_loss=%.6f q_loss=%.6f "
                    "actor_loss=%s",
                    epoch,
                    epoch_step + 1,
                    self.config.steps_per_epoch,
                    metrics.global_step,
                    elapsed,
                    metrics.value_loss,
                    metrics.q_loss,
                    actor_loss,
                )

        actor_losses = [
            metrics.actor.loss
            for metrics in step_metrics
            if metrics.actor is not None
        ]
        epoch_elapsed = time.perf_counter() - epoch_started
        checkpoint_path = (
            self.recorder.checkpoint_dir
            / f"epoch_{epoch:04d}_rank_{self.rank:04d}.pt"
        )
        preliminary = {
            "epoch": epoch,
            "rank": self.rank,
            "global_step": self.trainer.global_step,
            "steps": len(step_metrics),
            "elapsed_seconds": epoch_elapsed,
            "value_loss_mean": _mean(
                [metrics.value_loss for metrics in step_metrics]
            ),
            "q_loss_mean": _mean(
                [metrics.q_loss for metrics in step_metrics]
            ),
            "actor_updates": len(actor_losses),
            "actor_loss_mean": (
                None if not actor_losses else _mean(actor_losses)
            ),
            "checkpoint_path": str(checkpoint_path),
        }
        checkpoint_extra = {
            "format": "diffsbdd_offline_iql_runner_epoch_v1",
            "runner_config": asdict(self.config),
            "epoch_metrics": preliminary,
        }
        if checkpoint_extra_factory is not None:
            additional = checkpoint_extra_factory(epoch)
            if not isinstance(additional, dict):
                raise TypeError("checkpoint extra factory must return a dict")
            checkpoint_extra["coordinator_extra"] = additional
        saved_path = self.trainer.save_checkpoint(
            checkpoint_path,
            epoch=epoch,
            sampler_state=self.sampler.state_dict(),
            extra=checkpoint_extra,
        )
        summary = EpochMetrics(
            **{
                **preliminary,
                "checkpoint_path": str(saved_path),
            }
        )
        if epoch_callback is not None:
            epoch_callback(summary)
        self.recorder.record_epoch(summary)
        self.recorder.record_status(
            "running",
            epoch=epoch,
            global_step=self.trainer.global_step,
        )
        self.recorder.logger.info(
            "epoch completed epoch=%d global_step=%d elapsed=%.3fs "
            "value_loss_mean=%.6f q_loss_mean=%.6f actor_updates=%d "
            "actor_loss_mean=%s checkpoint=%s",
            epoch,
            summary.global_step,
            epoch_elapsed,
            summary.value_loss_mean,
            summary.q_loss_mean,
            summary.actor_updates,
            (
                "none"
                if summary.actor_loss_mean is None
                else f"{summary.actor_loss_mean:.6f}"
            ),
            saved_path,
        )
        return summary

    def _restore_checkpoint(self, path: str | Path) -> int:
        """Restore trainer/sampler state and align durable metric tails."""
        checkpoint_path = Path(path)
        try:
            payload = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            payload = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise TypeError("offline IQL checkpoint payload must be a dict")
        required = {"trainer", "sampler", "epoch"}
        missing = required - set(payload)
        if missing:
            raise ValueError(
                f"offline IQL checkpoint missing fields: {sorted(missing)}"
            )
        checkpoint_epoch = int(payload["epoch"])
        if checkpoint_epoch < 0:
            raise ValueError("checkpoint epoch must be non-negative")
        self._validate_resume_config(payload.get("extra"))
        self.trainer.load_state_dict(payload["trainer"])
        self.sampler.load_state_dict(payload["sampler"])
        self.recorder.prepare_resume(
            checkpoint_epoch=checkpoint_epoch,
            trainer_global_step=self.trainer.global_step,
        )
        self.recorder.logger.info(
            "checkpoint restored path=%s epoch=%d global_step=%d",
            checkpoint_path,
            checkpoint_epoch,
            self.trainer.global_step,
        )
        return checkpoint_epoch + 1

    def _validate_resume_config(self, extra: object) -> None:
        """Reject work-shape changes that invalidate epoch-exact resume."""
        if not isinstance(extra, dict):
            return
        saved_config = extra.get("runner_config")
        if not isinstance(saved_config, dict):
            return
        for field in (
            "steps_per_epoch",
            "batch_size",
            "checkpoint_every_epochs",
        ):
            current_value = getattr(self.config, field)
            saved_value = saved_config.get(field)
            if saved_value != current_value:
                raise ValueError(
                    f"runner checkpoint config mismatch for {field}: "
                    f"saved={saved_value!r} current={current_value!r}"
                )


def _mean(values: list[float]) -> float:
    """Return an accurate finite mean for a non-empty float list."""
    if not values:
        raise ValueError("cannot compute an epoch mean from no values")
    result = math.fsum(values) / len(values)
    if not math.isfinite(result):
        raise FloatingPointError("epoch mean is not finite")
    return result
