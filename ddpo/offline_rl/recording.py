"""Durable local CSV, JSON, and log recording for offline IQL training."""

from __future__ import annotations

import csv
import json
import logging
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .sampler import SamplerStats
from .trainer import ActorStepMetrics, IQLStepMetrics


STEP_METRIC_FIELDS = (
    "epoch",
    "epoch_step",
    "global_step",
    "rank",
    "record_scope",
    "elapsed_seconds",
    "value_loss",
    "q_loss",
    "q1_loss",
    "q2_loss",
    "value_grad_norm",
    "q_grad_norm",
    "q1_mean",
    "q2_mean",
    "value_mean",
    "td_target_mean",
    "td_target_std",
    "td_target_min",
    "td_target_max",
    "advantage_mean",
    "advantage_std",
    "q_disagreement_mean",
    "gradient_loss_scale",
    "actor_updated",
    "actor_loss",
    "actor_log_prob_mean",
    "actor_weight_mean",
    "actor_weight_max",
    "actor_effective_sample_size",
    "actor_clipped_fraction",
    "actor_grad_norm",
    "sampler_batches",
    "sampler_transitions",
    "sampler_cache_hits",
    "sampler_cache_misses",
    "sampler_cache_hit_rate",
    "sampler_shard_load_seconds",
    "sampler_sample_seconds",
)

EPOCH_METRIC_FIELDS = (
    "epoch",
    "rank",
    "record_scope",
    "global_step",
    "steps",
    "elapsed_seconds",
    "value_loss_mean",
    "q_loss_mean",
    "actor_updates",
    "actor_loss_mean",
    "checkpoint_path",
)


def _utc_now() -> str:
    """Return a stable UTC timestamp suitable for JSON records."""
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON object atomically and durably."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _append_csv_row(
    path: Path,
    fieldnames: tuple[str, ...],
    row: dict[str, Any],
) -> None:
    """Append one schema-checked CSV row and fsync it."""
    unknown = set(row) - set(fieldnames)
    if unknown:
        raise ValueError(f"unknown CSV fields for {path}: {sorted(unknown)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fieldnames})
        handle.flush()
        os.fsync(handle.fileno())


def _read_csv_rows(
    path: Path,
    expected_fields: tuple[str, ...],
) -> list[dict[str, str]]:
    """Read a CSV file while enforcing its exact durable schema."""
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != expected_fields:
            raise ValueError(
                f"CSV schema mismatch for {path}: "
                f"expected {expected_fields}, got {reader.fieldnames}"
            )
        return list(reader)


def _atomic_rewrite_csv(
    path: Path,
    fieldnames: tuple[str, ...],
    rows: Iterable[dict[str, Any]],
) -> None:
    """Atomically replace a CSV with the supplied schema and rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {field: row.get(field, "") for field in fieldnames}
                )
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class EpochMetrics:
    """Aggregated diagnostics and checkpoint identity for one epoch."""

    epoch: int
    rank: int
    global_step: int
    steps: int
    elapsed_seconds: float
    value_loss_mean: float
    q_loss_mean: float
    actor_updates: int
    actor_loss_mean: float | None
    checkpoint_path: str

    def __post_init__(self) -> None:
        """Validate counts and finite aggregate metrics."""
        if self.epoch < 0 or self.rank < 0 or self.global_step < 0:
            raise ValueError("epoch, rank, and global_step must be non-negative")
        if self.steps <= 0 or self.actor_updates < 0:
            raise ValueError("steps must be positive and actor_updates non-negative")
        finite_values = (
            self.elapsed_seconds,
            self.value_loss_mean,
            self.q_loss_mean,
        )
        if not all(math.isfinite(value) for value in finite_values):
            raise ValueError("epoch metrics must be finite")
        if self.actor_loss_mean is not None and not math.isfinite(
            self.actor_loss_mean
        ):
            raise ValueError("actor_loss_mean must be finite when present")


class LocalIQLRecorder:
    """Record one rank's offline-IQL progress without remote trackers."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        rank: int = 0,
        record_scope: str = "rank_local",
        stream_to_terminal: bool = True,
    ) -> None:
        """Create output paths and an isolated file/terminal logger."""
        if rank < 0:
            raise ValueError("rank must be non-negative")
        if record_scope not in {"rank_local", "global_aggregated"}:
            raise ValueError(
                "record_scope must be rank_local or global_aggregated"
            )
        self.output_dir = Path(output_dir)
        self.metrics_dir = self.output_dir / "metrics"
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.step_metrics_path = self.metrics_dir / "metrics.csv"
        self.epoch_metrics_path = self.metrics_dir / "epochs.csv"
        self.status_path = self.output_dir / "status.json"
        self.rank = rank
        self.record_scope = record_scope
        self.stream_to_terminal = bool(stream_to_terminal)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._build_logger()
        self._started_at = _utc_now()

    def _build_logger(self) -> logging.Logger:
        """Build a recorder-owned logger without mutating root logging."""
        logger = logging.getLogger(
            f"ddpo.offline_rl.rank_{self.rank}.{id(self)}"
        )
        logger.setLevel(logging.INFO)
        logger.propagate = False
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | rank=%(rank)s | "
            "%(name)s | %(message)s"
        )
        log_name = "train.log" if self.rank == 0 else f"train_rank_{self.rank:04d}.log"
        file_handler = logging.FileHandler(
            self.output_dir / log_name,
            encoding="utf-8",
            delay=True,
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(_RankFilter(self.rank))
        logger.addHandler(file_handler)
        if self.stream_to_terminal:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            stream_handler.addFilter(_RankFilter(self.rank))
            logger.addHandler(stream_handler)
        return logger

    def prepare_fresh_run(self) -> None:
        """Reject accidental reuse of an output directory with metric rows."""
        step_rows = _read_csv_rows(
            self.step_metrics_path,
            STEP_METRIC_FIELDS,
        )
        epoch_rows = _read_csv_rows(
            self.epoch_metrics_path,
            EPOCH_METRIC_FIELDS,
        )
        checkpoints = tuple(self.checkpoint_dir.glob("*.pt"))
        if step_rows or epoch_rows or checkpoints:
            raise FileExistsError(
                f"output directory already contains training evidence: "
                f"{self.output_dir}"
            )

    def prepare_resume(
        self,
        *,
        checkpoint_epoch: int,
        trainer_global_step: int,
    ) -> None:
        """Drop only uncheckpointed CSV tails before exact epoch resume."""
        if checkpoint_epoch < 0 or trainer_global_step < 0:
            raise ValueError("resume epoch and global step must be non-negative")
        step_rows = _read_csv_rows(
            self.step_metrics_path,
            STEP_METRIC_FIELDS,
        )
        epoch_rows = _read_csv_rows(
            self.epoch_metrics_path,
            EPOCH_METRIC_FIELDS,
        )
        kept_steps = [
            row
            for row in step_rows
            if int(row["global_step"]) < trainer_global_step
            and int(row["epoch"]) <= checkpoint_epoch
        ]
        kept_epochs = [
            row
            for row in epoch_rows
            if int(row["epoch"]) <= checkpoint_epoch
        ]
        if step_rows:
            kept_global_steps = [
                int(row["global_step"]) for row in kept_steps
            ]
            if kept_global_steps != list(range(trainer_global_step)):
                raise ValueError(
                    "existing step metrics do not form the checkpointed "
                    f"global-step prefix [0,{trainer_global_step})"
                )
        if epoch_rows:
            kept_epoch_ids = [int(row["epoch"]) for row in kept_epochs]
            if kept_epoch_ids != list(range(checkpoint_epoch + 1)):
                raise ValueError(
                    "existing epoch metrics do not form the checkpointed "
                    f"epoch prefix [0,{checkpoint_epoch}]"
                )
        _atomic_rewrite_csv(
            self.step_metrics_path,
            STEP_METRIC_FIELDS,
            kept_steps,
        )
        _atomic_rewrite_csv(
            self.epoch_metrics_path,
            EPOCH_METRIC_FIELDS,
            kept_epochs,
        )
        self.logger.info(
            "resume evidence aligned checkpoint_epoch=%d global_step=%d "
            "kept_steps=%d kept_epochs=%d",
            checkpoint_epoch,
            trainer_global_step,
            len(kept_steps),
            len(kept_epochs),
        )

    def record_step(
        self,
        *,
        epoch: int,
        epoch_step: int,
        elapsed_seconds: float,
        metrics: IQLStepMetrics,
        sampler_stats: SamplerStats,
    ) -> None:
        """Append one stable step row, including blank actor fields if skipped."""
        actor_fields = _actor_metric_fields(metrics.actor)
        row: dict[str, Any] = {
            "epoch": epoch,
            "epoch_step": epoch_step,
            "global_step": metrics.global_step,
            "rank": self.rank,
            "record_scope": self.record_scope,
            "elapsed_seconds": elapsed_seconds,
            "value_loss": metrics.value_loss,
            "q_loss": metrics.q_loss,
            "q1_loss": metrics.q1_loss,
            "q2_loss": metrics.q2_loss,
            "value_grad_norm": metrics.value_grad_norm,
            "q_grad_norm": metrics.q_grad_norm,
            "q1_mean": metrics.q1_mean,
            "q2_mean": metrics.q2_mean,
            "value_mean": metrics.value_mean,
            "td_target_mean": metrics.td_target_mean,
            "td_target_std": metrics.td_target_std,
            "td_target_min": metrics.td_target_min,
            "td_target_max": metrics.td_target_max,
            "advantage_mean": metrics.advantage_mean,
            "advantage_std": metrics.advantage_std,
            "q_disagreement_mean": metrics.q_disagreement_mean,
            "gradient_loss_scale": metrics.gradient_loss_scale,
            "actor_updated": int(metrics.actor_updated),
            "sampler_batches": sampler_stats.batches_sampled,
            "sampler_transitions": sampler_stats.transitions_sampled,
            "sampler_cache_hits": sampler_stats.cache_hits,
            "sampler_cache_misses": sampler_stats.cache_misses,
            "sampler_cache_hit_rate": sampler_stats.cache_hit_rate,
            "sampler_shard_load_seconds": sampler_stats.shard_load_seconds,
            "sampler_sample_seconds": sampler_stats.sample_seconds,
            **actor_fields,
        }
        _append_csv_row(self.step_metrics_path, STEP_METRIC_FIELDS, row)

    def record_epoch(self, metrics: EpochMetrics) -> None:
        """Append one epoch row only after its checkpoint is durable."""
        if metrics.rank != self.rank:
            raise ValueError("epoch metric rank does not match recorder rank")
        row = asdict(metrics)
        row["record_scope"] = self.record_scope
        _append_csv_row(
            self.epoch_metrics_path,
            EPOCH_METRIC_FIELDS,
            row,
        )

    def record_status(
        self,
        status: str,
        *,
        epoch: int | None = None,
        global_step: int | None = None,
        error: str | None = None,
    ) -> None:
        """Atomically record running, completed, or failed state."""
        if status not in {"running", "completed", "failed"}:
            raise ValueError(f"unsupported run status {status!r}")
        payload: dict[str, Any] = {
            "format": "diffsbdd_offline_iql_status_v1",
            "status": status,
            "rank": self.rank,
            "record_scope": self.record_scope,
            "started_at": self._started_at,
            "updated_at": _utc_now(),
            "epoch": epoch,
            "global_step": global_step,
        }
        if error is not None:
            payload["error"] = error
        _atomic_write_json(self.status_path, payload)

    def close(self) -> None:
        """Flush, close, and detach recorder-owned logging handlers."""
        for handler in tuple(self.logger.handlers):
            handler.flush()
            handler.close()
            self.logger.removeHandler(handler)


class _RankFilter(logging.Filter):
    """Inject a fixed rank into every local recorder log record."""

    def __init__(self, rank: int) -> None:
        """Store the non-negative rank."""
        super().__init__()
        self.rank = rank

    def filter(self, record: logging.LogRecord) -> bool:
        """Attach rank and retain the record."""
        record.rank = self.rank
        return True


def _actor_metric_fields(
    actor: ActorStepMetrics | None,
) -> dict[str, float | str]:
    """Flatten optional actor metrics into the stable step CSV schema."""
    if actor is None:
        return {
            "actor_loss": "",
            "actor_log_prob_mean": "",
            "actor_weight_mean": "",
            "actor_weight_max": "",
            "actor_effective_sample_size": "",
            "actor_clipped_fraction": "",
            "actor_grad_norm": "",
        }
    return {
        "actor_loss": actor.loss,
        "actor_log_prob_mean": actor.log_prob_mean,
        "actor_weight_mean": actor.weight_mean,
        "actor_weight_max": actor.weight_max,
        "actor_effective_sample_size": actor.effective_sample_size,
        "actor_clipped_fraction": actor.clipped_fraction,
        "actor_grad_norm": actor.grad_norm,
    }
