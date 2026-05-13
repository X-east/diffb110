"""
Single-GPU active training entrypoint for DiffSBDD + RLPF-verl_diffusion RL.

This entrypoint is intentionally organized in fully separated layers:
- CLI / config loading
- runtime assembly
- sample preparation
- policy update
- training execution

The goal is to keep training ownership explicit without reintroducing
Lightning as the active workflow owner.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import ddpo.utils  # noqa: F401
import torch

from constants import dataset_params
from ddpo.integration import DiffSBDDModel
from ddpo.rewards.factory import build_rewarder
from ddpo.runtime import DiffSBDDDataLoader, DiffSBDDRollout, set_seed, setup_logging
from ddpo.runtime.build import (
    apply_cli_overrides,
    build_actor,
    build_dataloader,
    build_filter,
    build_model,
    build_reward_normalizer,
    load_config,
    resolve_device,
    save_config_snapshot,
)
from ddpo.runtime.cli import parse_args
from ddpo.runtime.recording import CheckpointFacts, RunRecorder, RunStatus, create_run_context
from ddpo.runtime.train_helpers import (
    BatchStats,
    EpochStepResult,
    PreparedSamples,
    build_batch_facts,
    build_epoch_facts,
    compute_batch_stats,
    finish_run,
    prepare_training_samples,
    update_policy,
)
from ddpo.utils import DiffSBDDFilter
from ddpo.utils.suppress_warnings import setup_chemistry_logging

# Disable output buffering for immediate console output
os.environ["PYTHONUNBUFFERED"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "RLPF-verl_diffusion"))
sys.path.insert(0, str(PROJECT_ROOT / "DiffSBDD"))


@dataclass
class RuntimeBundle:
    args: argparse.Namespace
    config: dict[str, Any]
    seed: int
    device: str
    dataset_name: str
    dataset_info: dict[str, Any]
    run_context: Any
    logger: Any
    model: DiffSBDDModel
    dataloader: DiffSBDDDataLoader
    rollout: DiffSBDDRollout
    rewarder: Any
    reward_normalizer: Any
    filters: DiffSBDDFilter
    actor: Any
    run_recorder: RunRecorder
    max_epochs: int
    checkpoint_interval: int


def build_runtime(args: argparse.Namespace, config: dict[str, Any]) -> RuntimeBundle:
    """Assemble the complete single-GPU runtime bundle."""
    config = apply_cli_overrides(config, args)

    seed = int(config.get("train", {}).get("seed", 42))
    set_seed(seed)
    setup_chemistry_logging()

    run_context = create_run_context(config, save_path=args.save_path)
    logger = setup_logging(
        log_dir=run_context.logs_dir,
        log_level=args.log_level,
        console_enabled=True,
    )
    save_config_snapshot(config, run_context.config_snapshot_path)

    device = resolve_device(config)
    dataset_name = config.get("dataset", "crossdock")
    dataset_info = dataset_params[dataset_name]

    logger.info("Initializing single-GPU DiffSBDD RL runtime")
    logger.info("Config path: %s", args.config_path)
    logger.info("Output dir: %s", run_context.output_dir)
    logger.info("Seed: %d", seed)
    logger.info("Device: %s", device)

    model = build_model(config, device, dataset_name)
    dataloader = build_dataloader(config, dataset_info, device)
    rollout = DiffSBDDRollout(model=model, config=config)
    rewarder = build_rewarder(config=config, dataset_info=dataset_info, logger=logger)
    reward_normalizer = build_reward_normalizer(config=config, logger=logger)
    filters = build_filter(config, dataset_info)
    actor = build_actor(model, config, dataset_info)
    run_recorder = RunRecorder(run_context=run_context, logger=logger, world_size=1)

    logger.info("Initialized DiffSBDDModel")
    logger.info("Initialized DiffSBDDDataLoader")
    logger.info("Initialized DiffSBDDRollout")
    logger.info("Initialized rewarder: %s", type(rewarder).__name__)
    logger.info("Initialized reward normalizer: %s", type(reward_normalizer).__name__)
    logger.info("Initialized filter: %s", filters)
    logger.info("Initialized actor: %s", type(actor).__name__)
    logger.info("Initialized RunRecorder")

    max_epochs = int(
        config.get("train", {}).get(
            "epochs",
            config.get("dataloader", {}).get("epochs", 1),
        )
    )
    checkpoint_interval = int(config.get("train", {}).get("checkpoint_interval", 1))

    return RuntimeBundle(
        args=args,
        config=config,
        seed=seed,
        device=device,
        dataset_name=dataset_name,
        dataset_info=dataset_info,
        run_context=run_context,
        logger=logger,
        model=model,
        dataloader=dataloader,
        rollout=rollout,
        rewarder=rewarder,
        reward_normalizer=reward_normalizer,
        filters=filters,
        actor=actor,
        run_recorder=run_recorder,
        max_epochs=max_epochs,
        checkpoint_interval=checkpoint_interval,
    )


def checkpoint_path_for_epoch(runtime: RuntimeBundle, epoch: int) -> str:
    """Return the canonical checkpoint path for one epoch."""
    checkpoint_name = f"diffsbdd-ddpo-epoch={epoch:04d}.pt"
    return str(Path(runtime.run_context.checkpoints_dir) / checkpoint_name)



def build_checkpoint_payload(runtime: RuntimeBundle, epoch: int, metrics: dict[str, Any]) -> dict[str, Any]:
    """Build the serialized single-GPU checkpoint payload."""
    return {
        "epoch": epoch,
        "config": runtime.config,
        "seed": runtime.seed,
        "model_state_dict": runtime.model.model.state_dict(),
        "optimizer_state_dict": runtime.actor.optimizer.state_dict(),
        "metrics": metrics,
    }



def save_checkpoint(runtime: RuntimeBundle, epoch: int, metrics: dict[str, Any]) -> CheckpointFacts | None:
    """Save one checkpoint when due and report it through the recorder."""
    if (epoch % runtime.checkpoint_interval) != 0:
        return None

    checkpoint_path = checkpoint_path_for_epoch(runtime, epoch)
    payload = build_checkpoint_payload(runtime, epoch, metrics)
    torch.save(payload, checkpoint_path)

    checkpoint_file = Path(checkpoint_path)
    exists = checkpoint_file.exists()
    size_bytes = checkpoint_file.stat().st_size if exists else 0
    avg_loss = float(metrics.get("Loss", metrics.get("loss", metrics.get("ppo_loss", 0.0))))

    facts = CheckpointFacts(
        epoch=epoch,
        path=checkpoint_path,
        kind="interval",
        monitor="avg_loss",
        monitor_value=avg_loss,
        exists=exists,
        size_bytes=size_bytes,
        confirmed=exists and size_bytes > 0,
        saved_at=datetime.now().isoformat(timespec="seconds"),
    )
    runtime.run_recorder.on_checkpoint_saved(facts)
    return facts



def summarize_epoch(
    runtime: RuntimeBundle,
    epoch: int,
    prepared: PreparedSamples,
    metrics: dict[str, Any],
    batch_stats: BatchStats,
) -> EpochStepResult:
    """Compute local epoch summary stats from the prepared samples."""
    return EpochStepResult(
        epoch=epoch,
        metrics=metrics,
        reward_mean=batch_stats.reward_mean,
        reward_min=batch_stats.reward_min,
        reward_max=batch_stats.reward_max,
        avg_qed=batch_stats.avg_qed,
        avg_sa=batch_stats.avg_sa,
        avg_docking=batch_stats.avg_docking,
        avg_similarity=batch_stats.avg_similarity,
        generated_count=batch_stats.generated_count,
        valid_count=batch_stats.valid_count,
        valid_ratio=batch_stats.valid_ratio,
        filtering_ratio=prepared.filtering_ratio,
        novelty_penalty_ratio=prepared.novelty_penalty_ratio,
        failure_counts=batch_stats.failure_counts,
    )



def log_epoch_result(runtime: RuntimeBundle, result: EpochStepResult) -> None:
    """Log the compact end-of-epoch summary."""
    runtime.logger.info(
        "[Epoch %d/%d] core loop complete | generated=%d valid=%d filtering_ratio=%.4f novelty_penalty_ratio=%.4f reward_mean=%.4f metrics=%s",
        result.epoch,
        runtime.max_epochs,
        result.generated_count,
        result.valid_count,
        result.filtering_ratio,
        result.novelty_penalty_ratio,
        result.reward_mean,
        result.metrics,
    )



def run_epoch(runtime: RuntimeBundle, epoch: int) -> EpochStepResult:
    """Run one fully explicit single-GPU epoch."""
    prepared = prepare_training_samples(runtime, epoch)
    metrics = update_policy(runtime, prepared, epoch)
    batch_stats = compute_batch_stats(prepared)
    batch_facts = build_batch_facts(
        prepared=prepared,
        epoch=epoch,
        batch_idx=0,
        batch_stats=batch_stats,
    )
    runtime.run_recorder.on_batch_end(batch_facts)
    epoch_facts = build_epoch_facts(
        runtime=runtime,
        epoch=epoch,
        metrics=metrics,
        batch_stats=batch_stats,
        prepared=prepared,
    )
    runtime.run_recorder.on_epoch_end(epoch_facts)
    save_checkpoint(runtime, epoch, metrics)
    result = summarize_epoch(runtime, epoch, prepared, metrics, batch_stats)
    log_epoch_result(runtime, result)
    return result



def run_training_loop(runtime: RuntimeBundle) -> None:
    """Run the full single-GPU training loop."""
    runtime.run_recorder.on_run_started()
    runtime.logger.info(
        "Starting single-GPU core training loop for %d epochs", runtime.max_epochs
    )
    try:
        for epoch in range(1, runtime.max_epochs + 1):
            run_epoch(runtime, epoch)
    except KeyboardInterrupt:
        finish_run(runtime, RunStatus.INTERRUPTED, message="KeyboardInterrupt")
        raise
    except Exception as exc:
        finish_run(runtime, RunStatus.FAILED, message=f"{type(exc).__name__}: {exc}")
        raise
    else:
        runtime.logger.info("Single-GPU core training loop finished")
        finish_run(runtime, RunStatus.COMPLETED)



def main() -> None:
    """Program entrypoint: parse, assemble runtime, execute training."""
    args = parse_args()
    config = load_config(args.config_path)
    runtime = build_runtime(args, config)
    run_training_loop(runtime)


if __name__ == "__main__":
    main()
