"""Resumable endpoint-based DiffSBDD Global DPO trainer."""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
import torch

from .endpoint_data import EndpointStore, GlobalPreferencePair
from .global_score import compute_global_score
from .loss import GlobalDPOFacts, compute_global_dpo_loss, compute_global_esrpo_loss
from .paired_noise import build_paired_noisy_batch


LOGGER = logging.getLogger(__name__)
GlobalDPOObjective = Literal["binary_dpo", "esrpo"]

# The objective was renamed from "e2po" to "esrpo" (Endpoint Structural-Rule
# Preference Optimization) on 2026-09-18. The legacy token stays accepted so
# that historical configs and run manifests remain replayable; it is mapped onto
# the current name before validation. See
# work_records/26-9/260918/260918-02 E2PO更名为ESRPO全面更新.md
LEGACY_OBJECTIVE_ALIASES: dict[str, str] = {"e2po": "esrpo"}


def normalize_objective(value: str) -> str:
    """Map a legacy objective token onto its current name."""
    normalized = LEGACY_OBJECTIVE_ALIASES.get(str(value), str(value))
    if normalized != value:
        LOGGER.warning(
            "objective %r is a legacy alias of %r; update the config to the current name",
            value,
            normalized,
        )
    return normalized


@dataclass(frozen=True)
class GlobalDPOTrainerConfig:
    """Immutable configuration for one Global DPO run."""

    output_dir: Path
    epochs: int = 1
    batch_pairs: int = 1
    learning_rate: float = 1.0e-6
    beta: float = 1.0
    reward_temperature: float = 1.0
    max_grad_norm: float = 1.0
    timestep_grid: int = 500
    train_timesteps_per_batch: int = 1
    validation_timesteps: tuple[int, ...] = (500,)
    train_eval_pair_count: int = 512
    x_weight: float = 0.5
    v_weight: float = 0.5
    noise_coupling: str = "common_random"
    freeze_encoder_decoder: bool = False
    freeze_egnn_first_n_blocks: int = 0
    objective: GlobalDPOObjective = "binary_dpo"
    variant: str = "unknown"
    seed: int = 1
    resume_checkpoint: Path | None = None
    save_every_updates: int = 0  # 0 = 仅 epoch 末存盘（默认行为，向后兼容）


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aggregate_scores(scores: list[torch.Tensor]) -> torch.Tensor:
    """Average scores from multiple independently noised timesteps."""
    if not scores:
        raise ValueError("at least one timestep score is required")
    return torch.stack(scores, dim=0).mean(dim=0)


class GlobalDPOTrainer:
    """Train a trainable endpoint denoiser against a frozen reference."""

    def __init__(
        self,
        *,
        current: object,
        reference: object,
        train_pairs: list[GlobalPreferencePair],
        validation_pairs: list[GlobalPreferencePair],
        test_pairs: list[GlobalPreferencePair] | None,
        config: GlobalDPOTrainerConfig,
        checkpoint_source: str,
        epoch_callback: Callable[[int, Path], None] | None = None,
    ) -> None:
        """Initialize models, endpoint cache, optimizer, and output artifacts."""
        self._validate_config(config)
        if not train_pairs or not validation_pairs:
            raise ValueError("train and validation pairs must be non-empty")
        if test_pairs is not None and not test_pairs:
            raise ValueError("test_pairs must be non-empty when provided")
        self.current = current
        self.reference = reference
        self.train_pairs = train_pairs
        self.validation_pairs = validation_pairs
        self.test_pairs = test_pairs
        self.config = replace(config, objective=normalize_objective(config.objective))
        self.checkpoint_source = checkpoint_source
        self.epoch_callback = epoch_callback
        self.store = EndpointStore()
        self.device = torch.device(current.device)
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "metrics").mkdir(exist_ok=True)
        self.metrics_path = self.output_dir / "metrics" / "metrics.csv"
        self.start_epoch = 1
        for parameter in self.reference.model.parameters():
            parameter.requires_grad_(False)
        self.reference.model.eval()
        self.freeze_report = self._apply_freeze_policy()
        trainable_parameters = [
            parameter for parameter in self.current.model.parameters() if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise RuntimeError("Global DPO freeze policy left no trainable parameters")
        self.optimizer = torch.optim.AdamW(trainable_parameters, lr=config.learning_rate)
        LOGGER.info("global_dpo_freeze_report=%s", self.freeze_report)
        if config.resume_checkpoint is not None:
            self._load_checkpoint(config.resume_checkpoint)

    @staticmethod
    def _resolve_module(root: torch.nn.Module, path: str) -> torch.nn.Module:
        """Resolve a dotted module path and fail loudly when it is absent."""
        module: object = root
        for part in path.split("."):
            module = getattr(module, part, None)
            if module is None:
                raise AttributeError(f"model module path not found: {path}")
        if not isinstance(module, torch.nn.Module):
            raise TypeError(f"model path is not a torch module: {path}")
        return module

    def _apply_freeze_policy(self) -> dict[str, Any]:
        """Freeze requested DiffSBDD front/back modules before optimizer setup."""
        # DiffSBDDModel.model is the loaded LigandPocketDDPM itself.  The
        # adapter exposes it as ``current.model``; there is no extra wrapper
        # module between the adapter and ``ddpm.dynamics``.
        core_model = self.current.model
        if not isinstance(core_model, torch.nn.Module):
            raise TypeError("current.model must be the DiffSBDD torch module")
        for parameter in core_model.parameters():
            parameter.requires_grad_(True)

        frozen_paths: list[str] = []
        if self.config.freeze_encoder_decoder:
            frozen_paths.extend([
                "ddpm.dynamics.atom_encoder",
                "ddpm.dynamics.residue_encoder",
                "ddpm.dynamics.atom_decoder",
                "ddpm.dynamics.residue_decoder",
            ])

        block_count = int(self.config.freeze_egnn_first_n_blocks)
        if block_count < 0:
            raise ValueError("freeze_egnn_first_n_blocks must be non-negative")
        if block_count:
            denoiser = self._resolve_module(core_model, "ddpm.dynamics.egnn")
            block_names = sorted(
                name for name, module in denoiser.named_children() if name.startswith("e_block_")
            )
            if block_count > len(block_names):
                raise ValueError(
                    f"requested {block_count} EGNN blocks but found {len(block_names)}: {block_names}"
                )
            frozen_paths.extend(f"ddpm.dynamics.egnn.{name}" for name in block_names[:block_count])

        frozen_parameter_count = 0
        for path in frozen_paths:
            module = self._resolve_module(core_model, path)
            for parameter in module.parameters():
                if parameter.requires_grad:
                    frozen_parameter_count += parameter.numel()
                parameter.requires_grad_(False)

        total_parameter_count = sum(parameter.numel() for parameter in core_model.parameters())
        trainable_parameter_count = sum(
            parameter.numel() for parameter in core_model.parameters() if parameter.requires_grad
        )
        return {
            "freeze_encoder_decoder": bool(self.config.freeze_encoder_decoder),
            "freeze_egnn_first_n_blocks": block_count,
            "frozen_module_paths": frozen_paths,
            "total_parameters": int(total_parameter_count),
            "trainable_parameters": int(trainable_parameter_count),
            "frozen_parameters": int(total_parameter_count - trainable_parameter_count),
            "frozen_parameter_count_from_paths": int(frozen_parameter_count),
            "trainable_ratio": float(trainable_parameter_count / total_parameter_count),
        }

    @staticmethod
    def _validate_config(config: GlobalDPOTrainerConfig) -> None:
        """Reject unsafe or ambiguous Global DPO settings."""
        if config.epochs <= 0 or config.batch_pairs <= 0:
            raise ValueError("epochs and batch_pairs must be positive")
        if config.learning_rate <= 0 or config.beta <= 0 or config.max_grad_norm <= 0:
            raise ValueError("learning_rate, beta, and max_grad_norm must be positive")
        if config.reward_temperature <= 0:
            raise ValueError("reward_temperature must be positive")
        if config.objective not in {"binary_dpo", "esrpo"}:
            raise ValueError(f"unsupported Global DPO objective: {config.objective}")
        if config.timestep_grid <= 0:
            raise ValueError("timestep_grid must be positive")
        if config.train_timesteps_per_batch <= 0:
            raise ValueError("train_timesteps_per_batch must be positive")
        if not config.validation_timesteps:
            raise ValueError("validation_timesteps must be non-empty")
        if any(not 1 <= timestep <= config.timestep_grid for timestep in config.validation_timesteps):
            raise ValueError("validation timestep is outside the configured grid")
        if config.train_eval_pair_count <= 0:
            raise ValueError("train_eval_pair_count must be positive")
        if config.x_weight < 0 or config.v_weight < 0 or config.x_weight + config.v_weight <= 0:
            raise ValueError("x_weight and v_weight must be non-negative and not both zero")
        if config.freeze_egnn_first_n_blocks < 0:
            raise ValueError("freeze_egnn_first_n_blocks must be non-negative")
        if config.save_every_updates < 0:
            raise ValueError("save_every_updates must be non-negative")
        if config.noise_coupling not in {"common_random", "independent"}:
            raise ValueError(
                "noise_coupling must be 'common_random' or 'independent'"
            )

    def _load_checkpoint(self, path: Path) -> None:
        """Restore current model, optimizer, and random states."""
        if not path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("checkpoint_version") not in {1, 2}:
            raise ValueError(f"unsupported Global DPO checkpoint version: {payload.get('checkpoint_version')}")
        self.current.model.load_state_dict(payload["model_state_dict"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        last_epoch = int(payload["epoch"])
        if not 0 <= last_epoch < self.config.epochs:
            raise ValueError(f"resume epoch {last_epoch} incompatible with target epochs {self.config.epochs}")
        self.start_epoch = last_epoch + 1
        if "python_random_state" in payload:
            random.setstate(payload["python_random_state"])
        if "numpy_random_state" in payload:
            np.random.set_state(payload["numpy_random_state"])
        if "torch_random_state" in payload:
            torch.set_rng_state(payload["torch_random_state"].detach().cpu())
        if torch.cuda.is_available() and payload.get("torch_cuda_random_state") is not None:
            torch.cuda.set_rng_state_all([state.detach().cpu() for state in payload["torch_cuda_random_state"]])
        LOGGER.info("resumed checkpoint=%s last_epoch=%d next_epoch=%d", path, last_epoch, self.start_epoch)

    def _items_for_pairs(self, pairs: list[GlobalPreferencePair]) -> list[object]:
        """Load endpoints in the required winner-then-loser order."""
        items: list[object] = []
        for pair in pairs:
            winner = self.store.sample(
                pair.winner_shard,
                pair.winner_index,
                expected_sample_id=pair.winner_id,
            )
            loser = self.store.sample(
                pair.loser_shard,
                pair.loser_index,
                expected_sample_id=pair.loser_id,
            )
            winner_n_nodes = pair.winner_n_nodes or pair.n_nodes
            loser_n_nodes = pair.loser_n_nodes or pair.n_nodes
            if winner.n_nodes != winner_n_nodes or loser.n_nodes != loser_n_nodes:
                raise ValueError(
                    f"pair node count mismatch for {pair.winner_id}/{pair.loser_id}: "
                    f"manifest={winner_n_nodes}/{loser_n_nodes}, "
                    f"actual={winner.n_nodes}/{loser.n_nodes}"
                )
            items.extend([winner, loser])
        return items

    def _select_train_timesteps(self, epoch: int, batch_index: int) -> list[int]:
        """Select deterministic random timestep values for one train batch."""
        rng = random.Random(self.config.seed + epoch * 1_000_003 + batch_index * 9_176)
        return [rng.randint(1, self.config.timestep_grid) for _ in range(self.config.train_timesteps_per_batch)]

    def _pair_rewards(self, pairs: list[GlobalPreferencePair]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return finite winner/loser rewards for an ESRPO batch."""
        if self.config.objective != "esrpo":
            raise RuntimeError("pair rewards requested for a non-ESRPO objective")
        missing = [
            pair.winner_id
            for pair in pairs
            if pair.reward_winner is None or pair.reward_loser is None
        ]
        if missing:
            raise ValueError(
                "ESRPO requires finite winner/loser rewards; missing rewards for "
                f"{missing[:3]}"
            )
        winner = torch.tensor(
            [float(pair.reward_winner) for pair in pairs],
            dtype=torch.float32,
            device=self.device,
        )
        loser = torch.tensor(
            [float(pair.reward_loser) for pair in pairs],
            dtype=torch.float32,
            device=self.device,
        )
        if not torch.isfinite(winner).all() or not torch.isfinite(loser).all():
            raise FloatingPointError("ESRPO pair rewards contain non-finite values")
        return winner, loser

    def _compute_preference_loss(
        self,
        pairs: list[GlobalPreferencePair],
        current_scores: torch.Tensor,
        reference_scores: torch.Tensor,
    ) -> GlobalDPOFacts:
        """Compute the configured binary-DPO or reward-weighted ESRPO loss."""
        if self.config.objective == "binary_dpo":
            return compute_global_dpo_loss(
                current_scores,
                reference_scores,
                beta=self.config.beta,
            )
        winner_rewards, loser_rewards = self._pair_rewards(pairs)
        return compute_global_esrpo_loss(
            current_scores,
            reference_scores,
            winner_rewards,
            loser_rewards,
            beta=self.config.beta,
            reward_temperature=self.config.reward_temperature,
        )

    def _score_batch(
        self,
        pairs: list[GlobalPreferencePair],
        timesteps: list[int],
        *,
        noise_seed_base: int,
        train_current: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute mean current/reference scores for a pair batch."""
        items = self._items_for_pairs(pairs)
        current_scores: list[torch.Tensor] = []
        reference_scores: list[torch.Tensor] = []
        context = torch.enable_grad() if train_current else torch.no_grad()
        with context:
            for index, timestep in enumerate(timesteps):
                noisy = build_paired_noisy_batch(
                    self.current,
                    items,
                    timestep,
                    timestep_grid=self.config.timestep_grid,
                    noise_seed=noise_seed_base + index,
                    noise_coupling=self.config.noise_coupling,
                )
                current_scores.append(
                    compute_global_score(
                        self.current,
                        noisy,
                        x_weight=self.config.x_weight,
                        v_weight=self.config.v_weight,
                    ).score.reshape(-1, 2)
                )
        with torch.no_grad():
            for index, timestep in enumerate(timesteps):
                noisy = build_paired_noisy_batch(
                    self.reference,
                    items,
                    timestep,
                    timestep_grid=self.config.timestep_grid,
                    noise_seed=noise_seed_base + index,
                    noise_coupling=self.config.noise_coupling,
                )
                reference_scores.append(
                    compute_global_score(
                        self.reference,
                        noisy,
                        x_weight=self.config.x_weight,
                        v_weight=self.config.v_weight,
                    ).score.reshape(-1, 2)
                )
        return _aggregate_scores(current_scores), _aggregate_scores(reference_scores)

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Train one epoch with deterministic pair order and timestep seeds."""
        self.current.model.train()
        shuffled = list(self.train_pairs)
        random.Random(self.config.seed + epoch).shuffle(shuffled)
        losses: list[float] = []
        margins: list[float] = []
        gradient_norms: list[float] = []
        winner_weights: list[float] = []
        loser_weights: list[float] = []
        reward_gaps: list[float] = []
        self._update_count = 0
        start = time.monotonic()
        for batch_index, batch_start in enumerate(range(0, len(shuffled), self.config.batch_pairs)):
            batch_pairs = shuffled[batch_start : batch_start + self.config.batch_pairs]
            timesteps = self._select_train_timesteps(epoch, batch_index)
            current, reference = self._score_batch(
                batch_pairs,
                timesteps,
                noise_seed_base=self.config.seed + epoch * 1_000_003 + batch_index * 13_337,
                train_current=True,
            )
            facts = self._compute_preference_loss(batch_pairs, current, reference)
            if not torch.isfinite(facts.loss):
                raise FloatingPointError(f"non-finite Global DPO loss at epoch={epoch}, batch={batch_start}")
            self.optimizer.zero_grad(set_to_none=True)
            facts.loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(self.current.model.parameters(), self.config.max_grad_norm)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"non-finite Global DPO gradient at epoch={epoch}, batch={batch_start}")
            self.optimizer.step()
            self._update_count += 1
            losses.append(float(facts.loss.detach().cpu()))
            margins.extend(facts.relative_margin.detach().cpu().tolist())
            gradient_norms.append(float(gradient_norm.detach().cpu()))
            winner_weights.extend(facts.winner_weight.detach().cpu().tolist())
            loser_weights.extend(facts.loser_weight.detach().cpu().tolist())
            if self.config.objective == "esrpo":
                reward_gaps.extend(
                    [
                        float(pair.reward_winner) - float(pair.reward_loser)
                        for pair in batch_pairs
                    ]
                )
            if self.config.save_every_updates > 0 and self._update_count % self.config.save_every_updates == 0:
                window_metrics = {
                    "epoch": epoch,
                    "update_step": self._update_count,
                    "train_dpo_loss": float(np.mean(losses)),
                    "train_relative_margin": float(np.mean(margins)),
                    "train_gradient_norm_mean": float(np.mean(gradient_norms)),
                }
                self.save_intermediate_checkpoint(self._update_count, window_metrics)
        return {
            "epoch": float(epoch),
            "train_dpo_loss": float(np.mean(losses)),
            "train_relative_margin": float(np.mean(margins)),
            "train_gradient_norm_mean": float(np.mean(gradient_norms)),
            "train_updates": float(len(losses)),
            "update_step": float(self._update_count),
            "train_timesteps_per_batch": float(self.config.train_timesteps_per_batch),
            "train_mean_winner_weight": float(np.mean(winner_weights)),
            "train_mean_loser_weight": float(np.mean(loser_weights)),
            "train_mean_reward_gap": float(np.mean(reward_gaps)) if reward_gaps else 0.0,
            "elapsed_seconds": float(time.monotonic() - start),
        }

    @torch.no_grad()
    def evaluate_pairs(
        self,
        pairs: list[GlobalPreferencePair],
        *,
        epoch: int,
        label: str,
    ) -> dict[str, float]:
        """Evaluate fixed timestep buckets with fixed pair denominator."""
        self.current.model.eval()
        bucket_accuracies: list[float] = []
        all_margins: list[float] = []
        for bucket_index, timestep in enumerate(self.config.validation_timesteps):
            margins: list[float] = []
            for batch_index, batch_start in enumerate(range(0, len(pairs), self.config.batch_pairs)):
                batch_pairs = pairs[batch_start : batch_start + self.config.batch_pairs]
                current, _reference = self._score_batch(
                    batch_pairs,
                    [timestep],
                    noise_seed_base=self.config.seed + epoch * 1_000_003 + bucket_index * 100_003 + batch_index,
                    train_current=False,
                )
                margins.extend((current[:, 0] - current[:, 1]).cpu().tolist())
            values = np.asarray(margins, dtype=float)
            accuracy = float(np.mean(values > 0.0))
            bucket_accuracies.append(accuracy)
            all_margins.extend(values.tolist())
        all_values = np.asarray(all_margins, dtype=float)
        result: dict[str, float] = {
            f"{label}_pair_accuracy": float(np.mean(bucket_accuracies)),
            f"{label}_mean_margin": float(np.mean(all_values)),
            f"{label}_pairs_per_bucket": float(len(pairs)),
            f"{label}_bucket_count": float(len(self.config.validation_timesteps)),
        }
        for timestep, accuracy in zip(self.config.validation_timesteps, bucket_accuracies):
            result[f"{label}_pair_accuracy_t{timestep:04d}"] = accuracy
        return result

    def save_checkpoint(self, epoch: int, metrics: dict[str, float]) -> Path:
        """Save model, optimizer, RNG states, and provenance for one epoch."""
        path = self.output_dir / "checkpoints" / f"diffsbdd-global-dpo-epoch={epoch:04d}.pt"
        payload: dict[str, Any] = {
            "checkpoint_version": 2,
            "epoch": epoch,
            "model_state_dict": self.current.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "resolved_config": {
                **asdict(self.config),
                "output_dir": str(self.config.output_dir),
                "resume_checkpoint": None if self.config.resume_checkpoint is None else str(self.config.resume_checkpoint),
            },
            "checkpoint_source": self.checkpoint_source,
            "checkpoint_source_sha256": sha256_file(self.checkpoint_source),
            "freeze_report": self.freeze_report,
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

    def save_intermediate_checkpoint(self, update_step: int, metrics: dict[str, float]) -> Path:
        """Save a step-level snapshot inside an epoch for intermediate evaluation.

        Unlike ``save_checkpoint``, this does NOT update ``checkpoint_latest.json``
        so that the resume entry point stays at the last completed epoch boundary.
        """
        epoch = int(metrics.get("epoch", self.start_epoch))
        path = (
            self.output_dir
            / "checkpoints"
            / f"diffsbdd-global-dpo-epoch={epoch:04d}_step={update_step:06d}.pt"
        )
        payload: dict[str, Any] = {
            "checkpoint_version": 2,
            "epoch": epoch,
            "update_step": update_step,
            "is_intermediate": True,
            "model_state_dict": self.current.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "resolved_config": {
                **asdict(self.config),
                "output_dir": str(self.config.output_dir),
                "resume_checkpoint": None if self.config.resume_checkpoint is None else str(self.config.resume_checkpoint),
            },
            "checkpoint_source": self.checkpoint_source,
            "checkpoint_source_sha256": sha256_file(self.checkpoint_source),
            "freeze_report": self.freeze_report,
            "metrics": metrics,
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "torch_cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        torch.save(payload, path)
        LOGGER.info(
            "intermediate_checkpoint=%s epoch=%d step=%d loss=%.6f",
            path,
            epoch,
            update_step,
            float(metrics.get("train_dpo_loss", float("nan"))),
        )
        return path

    def train(self) -> pd.DataFrame:
        """Run remaining epochs, write metrics, and mark completion."""
        existing = pd.read_csv(self.metrics_path) if self.metrics_path.exists() else pd.DataFrame()
        rows = existing.to_dict("records")
        if self.start_epoch > self.config.epochs:
            return existing
        train_eval_pairs = self.train_pairs[: self.config.train_eval_pair_count]
        try:
            for epoch in range(self.start_epoch, self.config.epochs + 1):
                train_metrics = self.train_epoch(epoch)
                train_eval = self.evaluate_pairs(train_eval_pairs, epoch=epoch, label="train_eval")
                validation = self.evaluate_pairs(self.validation_pairs, epoch=epoch, label="validation")
                test_metrics: dict[str, float] = {}
                if self.test_pairs is not None and epoch == self.config.epochs:
                    test_metrics = self.evaluate_pairs(self.test_pairs, epoch=epoch, label="test")
                row = {**train_metrics, **train_eval, **validation, **test_metrics}
                checkpoint = self.save_checkpoint(epoch, row)
                if self.epoch_callback is not None:
                    self.epoch_callback(epoch, checkpoint)
                row["checkpoint_path"] = str(checkpoint)
                rows = [item for item in rows if int(item.get("epoch", -1)) != epoch]
                rows.append(row)
                pd.DataFrame(rows).sort_values("epoch").to_csv(self.metrics_path, index=False)
                LOGGER.info(
                    "epoch=%d/%d updates=%d loss=%.6f validation_accuracy=%.6f checkpoint=%s",
                    epoch,
                    self.config.epochs,
                    int(row["train_updates"]),
                    row["train_dpo_loss"],
                    row["validation_pair_accuracy"],
                    checkpoint,
                )
            (self.output_dir / "completion.json").write_text(
                json.dumps({"status": "completed", "epochs": self.config.epochs}, indent=2), encoding="utf-8"
            )
            self._update_status("completed", completed_epochs=self.config.epochs)
        except BaseException as exc:
            self._update_status("failed", error=f"{type(exc).__name__}: {exc}")
            raise
        return pd.DataFrame(rows).sort_values("epoch")

    def _update_status(self, status: str, **extra: object) -> None:
        """Write a durable status artifact."""
        payload = {"status": status, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **extra}
        (self.output_dir / "run_status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
