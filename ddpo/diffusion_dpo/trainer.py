"""Configuration-driven DiffSBDD Gaussian-DPO training and evaluation."""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch

from .dpo_loss import compute_dpo_loss
from .aggregation import aggregate_transition_scores
from .pairs import PreferencePair
from .scorer import DiffSBDDGaussianTransitionScorer
from .trajectory import TrajectoryStore, collate_transitions


LOGGER = logging.getLogger(__name__)
TrainingMode = Literal["single_transition", "sampled_multistep", "full_valid_trajectory"]
Aggregation = Literal["mean", "sum"]


@dataclass(frozen=True)
class TrainerConfig:
    """Immutable configuration for one auditable Gaussian-DPO run."""

    output_dir: Path
    epochs: int = 1
    batch_pairs: int = 1
    learning_rate: float = 1.0e-6
    beta: float = 0.1
    max_grad_norm: float = 1.0
    num_timesteps: int = 1000
    reduction: str = "sum"
    include_normal_constant: bool = True
    include_sigma_term: bool = True
    x_weight: float = 1.0
    v_weight: float = 1.0
    mode: TrainingMode = "sampled_multistep"
    train_timesteps_per_batch: int = 8
    evaluation_mode: TrainingMode = "sampled_multistep"
    evaluation_timesteps_per_batch: int = 16
    aggregation: Aggregation = "mean"
    seed: int = 1
    resume_checkpoint: Path | None = None


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for a provenance file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class GaussianDPOTrainer:
    """Train a DiffSBDD current model against a frozen reference model."""

    def __init__(
        self,
        *,
        current: DiffSBDDGaussianTransitionScorer,
        reference: DiffSBDDGaussianTransitionScorer,
        train_pairs: list[PreferencePair],
        validation_pairs: list[PreferencePair],
        config: TrainerConfig,
        checkpoint_source: str,
    ) -> None:
        """Initialize models, pair data, optimizer, and durable run files."""
        if not train_pairs or not validation_pairs:
            raise ValueError("train and validation pairs must be non-empty")
        self._validate_config(config)
        self.current = current
        self.reference = reference
        self.train_pairs = train_pairs
        self.validation_pairs = validation_pairs
        self.config = config
        self.store = TrajectoryStore(expected_timesteps=config.num_timesteps)
        self.valid_timesteps = current.valid_transition_timesteps()
        reference_timesteps = reference.valid_transition_timesteps()
        if self.valid_timesteps != reference_timesteps:
            raise ValueError("current and reference native valid timestep sets differ")
        if not self.valid_timesteps:
            raise ValueError("native schedule has no non-degenerate transitions")
        self.optimizer = torch.optim.AdamW(
            self.current.model.model.parameters(),
            lr=config.learning_rate,
        )
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "metrics").mkdir(exist_ok=True)
        self.metrics_path = self.output_dir / "metrics" / "metrics.csv"
        self.checkpoint_source = checkpoint_source
        self.start_epoch = 1
        for parameter in self.reference.model.model.parameters():
            parameter.requires_grad_(False)
        self.reference.model.model.eval()
        if config.resume_checkpoint is not None:
            self._load_checkpoint(config.resume_checkpoint)

    @staticmethod
    def _validate_config(config: TrainerConfig) -> None:
        """Reject ambiguous or unsafe training configurations early."""
        if config.num_timesteps != 1000:
            raise ValueError("this branch is intentionally fixed to T=1000")
        if config.epochs <= 0 or config.batch_pairs <= 0:
            raise ValueError("epochs and batch_pairs must be positive")
        if config.learning_rate <= 0 or config.beta <= 0:
            raise ValueError("learning_rate and beta must be positive")
        if config.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if config.x_weight < 0 or config.v_weight < 0 or config.x_weight + config.v_weight <= 0:
            raise ValueError("x_weight and v_weight must be non-negative and not both zero")
        valid_modes = {"single_transition", "sampled_multistep", "full_valid_trajectory"}
        if config.mode not in valid_modes or config.evaluation_mode not in valid_modes:
            raise ValueError(f"unsupported DPO mode: {config.mode}/{config.evaluation_mode}")
        if config.train_timesteps_per_batch <= 0 or config.evaluation_timesteps_per_batch <= 0:
            raise ValueError("timestep counts must be positive")

    def _load_checkpoint(self, path: Path) -> None:
        """Restore current model, optimizer, epoch, and random states."""
        if not path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {path}")
        payload = torch.load(path, map_location=self.current.model.device, weights_only=False)
        if payload.get("checkpoint_version") != 2:
            raise ValueError(f"unsupported Gaussian-DPO checkpoint version: {payload.get('checkpoint_version')}")
        self.current.model.model.load_state_dict(payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        last_epoch = int(payload["epoch"])
        if last_epoch < 0 or last_epoch >= self.config.epochs:
            raise ValueError(f"resume epoch {last_epoch} is incompatible with target epochs {self.config.epochs}")
        self.start_epoch = last_epoch + 1
        if "python_random_state" in payload:
            random.setstate(payload["python_random_state"])
        if "numpy_random_state" in payload:
            np.random.set_state(payload["numpy_random_state"])
        if "torch_random_state" in payload:
            torch.set_rng_state(payload["torch_random_state"].detach().cpu())
        if torch.cuda.is_available() and payload.get("torch_cuda_random_state") is not None:
            torch.cuda.set_rng_state_all(
                [state.detach().cpu() for state in payload["torch_cuda_random_state"]]
            )
        LOGGER.info("resumed checkpoint=%s last_epoch=%d next_epoch=%d", path, last_epoch, self.start_epoch)

    def _pair_batch(self, pairs: list[PreferencePair], timestep: int) -> dict[str, torch.Tensor]:
        """Load winner/loser transitions and collate them at one common t."""
        transitions = []
        for pair in pairs:
            winner = self.store.sample(pair.winner_shard, pair.winner_index)
            loser = self.store.sample(pair.loser_shard, pair.loser_index)
            transitions.extend([
                winner.transition_for_timestep(timestep),
                loser.transition_for_timestep(timestep),
            ])
        return collate_transitions(transitions, device=self.current.model.device)

    def _select_timesteps(self, *, epoch: int, batch_index: int, training: bool) -> list[int]:
        """Select deterministic valid timesteps for one batch."""
        mode = self.config.mode if training else self.config.evaluation_mode
        if mode == "full_valid_trajectory":
            return list(self.valid_timesteps)
        count = 1 if mode == "single_transition" else (
            self.config.train_timesteps_per_batch if training else self.config.evaluation_timesteps_per_batch
        )
        count = min(count, len(self.valid_timesteps))
        rng = random.Random(self.config.seed + epoch * 1_000_003 + batch_index * 9_176 + int(training))
        if count == len(self.valid_timesteps):
            return list(self.valid_timesteps)
        return sorted(rng.sample(self.valid_timesteps, count))

    def _score_pair_batch(
        self,
        pairs: list[PreferencePair],
        timesteps: list[int],
        *,
        train_current: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return aggregated current/reference scores with shape ``[B, 2]``."""
        current_scores: list[torch.Tensor] = []
        reference_scores: list[torch.Tensor] = []
        current_context = torch.enable_grad() if train_current else torch.no_grad()
        with current_context:
            for timestep in timesteps:
                batch = self._pair_batch(pairs, timestep)
                current_score = self.current.score(
                    batch["z_t"], batch["z_s"], batch["t"], batch["node_mask"],
                    reduction=self.config.reduction,  # type: ignore[arg-type]
                    include_normal_constant=self.config.include_normal_constant,
                    include_sigma_term=self.config.include_sigma_term,
                )
                current_scores.append(
                    (self.config.x_weight * current_score.coordinate
                     + self.config.v_weight * current_score.atom_feature).reshape(-1, 2)
                )
        with torch.no_grad():
            for timestep in timesteps:
                batch = self._pair_batch(pairs, timestep)
                reference_score = self.reference.score(
                    batch["z_t"], batch["z_s"], batch["t"], batch["node_mask"],
                    reduction=self.config.reduction,  # type: ignore[arg-type]
                    include_normal_constant=self.config.include_normal_constant,
                    include_sigma_term=self.config.include_sigma_term,
                )
                reference_scores.append(
                    (self.config.x_weight * reference_score.coordinate
                     + self.config.v_weight * reference_score.atom_feature).reshape(-1, 2)
                )
        return (
            aggregate_transition_scores(current_scores, self.config.aggregation),
            aggregate_transition_scores(reference_scores, self.config.aggregation),
        )

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one deterministic shuffled epoch and return scalar metrics."""
        self.current.model.model.train()
        rng = random.Random(self.config.seed + epoch)
        pairs = list(self.train_pairs)
        rng.shuffle(pairs)
        losses: list[float] = []
        margins: list[float] = []
        grad_norms: list[float] = []
        start = time.monotonic()
        for batch_index, batch_start in enumerate(range(0, len(pairs), self.config.batch_pairs)):
            batch_pairs = pairs[batch_start : batch_start + self.config.batch_pairs]
            timesteps = self._select_timesteps(epoch=epoch, batch_index=batch_index, training=True)
            current, reference = self._score_pair_batch(batch_pairs, timesteps, train_current=True)
            facts = compute_dpo_loss(
                current[:, 0], current[:, 1], reference[:, 0], reference[:, 1], beta=self.config.beta
            )
            if not torch.isfinite(facts.loss):
                raise FloatingPointError(f"non-finite DPO loss at epoch={epoch}, batch={batch_start}")
            self.optimizer.zero_grad(set_to_none=True)
            facts.loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.current.model.model.parameters(), self.config.max_grad_norm
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite gradient norm at epoch={epoch}, batch={batch_start}")
            self.optimizer.step()
            losses.append(float(facts.loss.detach().cpu()))
            margins.extend(facts.relative_margin.detach().cpu().tolist())
            grad_norms.append(float(grad_norm.detach().cpu()))
        return {
            "epoch": float(epoch),
            "train_dpo_loss": float(np.mean(losses)),
            "train_relative_margin": float(np.mean(margins)),
            "train_grad_norm_mean": float(np.mean(grad_norms)),
            "train_updates": float(len(losses)),
            "train_timesteps_per_batch": float(len(self._select_timesteps(epoch=epoch, batch_index=0, training=True))),
            "effective_valid_steps": float(len(self.valid_timesteps)),
            "elapsed_seconds": float(time.monotonic() - start),
        }

    @torch.no_grad()
    def evaluate(self, pairs: list[PreferencePair], epoch: int) -> dict[str, float]:
        """Evaluate aggregated pair ordering with deterministic timestep choices."""
        self.current.model.model.eval()
        current_margins: list[float] = []
        reference_margins: list[float] = []
        relative_margins: list[float] = []
        timestep_counts: list[int] = []
        for batch_index, index in enumerate(range(0, len(pairs), self.config.batch_pairs)):
            batch_pairs = pairs[index : index + self.config.batch_pairs]
            timesteps = self._select_timesteps(epoch=epoch, batch_index=batch_index, training=False)
            current, reference = self._score_pair_batch(batch_pairs, timesteps, train_current=False)
            current_margin = current[:, 0] - current[:, 1]
            reference_margin = reference[:, 0] - reference[:, 1]
            current_margins.extend(current_margin.cpu().tolist())
            reference_margins.extend(reference_margin.cpu().tolist())
            relative_margins.extend((current_margin - reference_margin).cpu().tolist())
            timestep_counts.append(len(timesteps))
        values = np.asarray(current_margins, dtype=float)
        return {
            "validation_pair_accuracy": float(np.mean(values > 0.0)),
            "validation_mean_current_margin": float(np.mean(values)),
            "validation_mean_reference_margin": float(np.mean(reference_margins)),
            "validation_mean_relative_margin": float(np.mean(relative_margins)),
            "validation_pairs": float(len(values)),
            "validation_timesteps_per_batch": float(np.mean(timestep_counts)),
        }

    def save_checkpoint(self, epoch: int, metrics: dict[str, float]) -> Path:
        """Save a resumable epoch checkpoint and update its pointer."""
        path = self.output_dir / "checkpoints" / f"diffsbdd-gaussian-dpo-epoch={epoch:04d}.pt"
        payload = {
            "checkpoint_version": 2,
            "epoch": epoch,
            "model_state_dict": self.current.model.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "resolved_config": {
                **asdict(self.config),
                "output_dir": str(self.config.output_dir),
                "resume_checkpoint": None if self.config.resume_checkpoint is None else str(self.config.resume_checkpoint),
            },
            "checkpoint_source": self.checkpoint_source,
            "checkpoint_source_sha256": sha256_file(self.checkpoint_source),
            "valid_timesteps": self.valid_timesteps,
            "metrics": metrics,
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "torch_cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        torch.save(payload, path)
        (self.output_dir / "checkpoint_latest.json").write_text(
            json.dumps({"epoch": epoch, "path": str(path), "checkpoint_version": 2}, indent=2), encoding="utf-8"
        )
        return path

    def train(self) -> pd.DataFrame:
        """Run remaining epochs and append local metrics without duplicate epochs."""
        existing = pd.read_csv(self.metrics_path) if self.metrics_path.exists() else pd.DataFrame()
        rows = existing.to_dict("records")
        if self.start_epoch > self.config.epochs:
            return existing
        try:
            for epoch in range(self.start_epoch, self.config.epochs + 1):
                train_metrics = self.train_epoch(epoch)
                train_eval = self.evaluate(self.train_pairs, epoch)
                validation = self.evaluate(self.validation_pairs, epoch)
                row = {
                    **train_metrics,
                    "train_pair_accuracy": train_eval["validation_pair_accuracy"],
                    "train_mean_current_margin": train_eval["validation_mean_current_margin"],
                    "train_mean_relative_margin_eval": train_eval["validation_mean_relative_margin"],
                    **validation,
                }
                checkpoint = self.save_checkpoint(epoch, row)
                row["checkpoint_path"] = str(checkpoint)
                rows = [item for item in rows if int(item.get("epoch", -1)) != epoch]
                rows.append(row)
                pd.DataFrame(rows).sort_values("epoch").to_csv(self.metrics_path, index=False)
                LOGGER.info(
                    "epoch=%d/%d updates=%d loss=%.6f val_accuracy=%.6f val_relative_margin=%.6f checkpoint=%s",
                    epoch, self.config.epochs, int(row["train_updates"]), row["train_dpo_loss"],
                    row["validation_pair_accuracy"], row["validation_mean_relative_margin"], checkpoint,
                )
            (self.output_dir / "completion.json").write_text(
                json.dumps({"status": "completed", "epochs": self.config.epochs, "effective_valid_steps": len(self.valid_timesteps)}, indent=2), encoding="utf-8"
            )
            self._update_run_status("completed", completed_epochs=self.config.epochs)
        except BaseException as exc:
            self._update_run_status("failed", error=f"{type(exc).__name__}: {exc}")
            raise
        return pd.DataFrame(rows).sort_values("epoch")

    def _update_run_status(self, status: str, **extra: object) -> None:
        """Write a small durable status artifact for monitoring and recovery."""
        payload = {"status": status, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **extra}
        (self.output_dir / "run_status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
