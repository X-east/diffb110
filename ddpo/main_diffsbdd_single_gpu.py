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
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

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

# Activate wandb stub before importing DiffSBDD-related modules
import ddpo.utils  # noqa: F401

from constants import dataset_params
from ddpo.integration import DiffSBDDModel
from ddpo.rewards.factory import build_rewarder
from ddpo.runtime import DiffSBDDDataLoader, DiffSBDDRollout, set_seed, setup_logging
from ddpo.runtime.recording import (
    BatchFacts,
    CheckpointFacts,
    EpochFacts,
    RunLifecycleFacts,
    RunRecorder,
    RunStatus,
    create_run_context,
)
from ddpo.utils import DiffSBDDFilter, annotate_batch_molecule_semantics
from ddpo.utils.suppress_warnings import setup_chemistry_logging
from verl_diffusion.worker.actor.edm_actor import EDMActor


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
    filters: DiffSBDDFilter
    actor: Any
    run_recorder: RunRecorder
    max_epochs: int
    checkpoint_interval: int


@dataclass
class PreparedSamples:
    samples: Any
    filtering_ratio: float
    novelty_penalty_ratio: float


@dataclass
class EpochStepResult:
    epoch: int
    metrics: dict[str, Any]
    reward_mean: float
    reward_min: float
    reward_max: float
    avg_qed: float
    avg_sa: float
    generated_count: int
    valid_count: int
    valid_ratio: float
    filtering_ratio: float
    novelty_penalty_ratio: float
    failure_counts: dict[str, int]


@dataclass
class BatchStats:
    reward_mean: float
    reward_min: float
    reward_max: float
    avg_qed: float
    avg_sa: float
    generated_count: int
    valid_count: int
    valid_ratio: float
    failure_counts: dict[str, int]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the single-GPU entrypoint."""
    parser = argparse.ArgumentParser(
        description="Single-GPU DiffSBDD + RLPF RL training entrypoint"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Optional output directory override.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed override.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reserved for future checkpoint resume support.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Reserved for future checkpoint resume support.",
    )
    return parser.parse_args()



def load_config(config_path: str) -> dict[str, Any]:
    """Load YAML configuration from disk."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)



def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Apply CLI overrides while preserving config as the source of truth."""
    if args.seed is not None:
        config.setdefault("train", {})["seed"] = args.seed
    return config



def resolve_device(config: dict[str, Any]) -> str:
    """Resolve runtime device from config with CUDA-aware fallback."""
    return config.get(
        "model", {}
    ).get(
        "device",
        "cuda" if torch.cuda.is_available() else "cpu",
    )



def save_config_snapshot(config: dict[str, Any], snapshot_path: str) -> None:
    """Persist the resolved config snapshot for reproducibility."""
    with open(snapshot_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)



def build_model(config: dict[str, Any], device: str, dataset_name: str) -> DiffSBDDModel:
    """Construct the DiffSBDD model wrapper."""
    return DiffSBDDModel(
        checkpoint_path=config["model"]["checkpoint"],
        pocket_pdb_path=config["pocket"]["pdb_path"],
        ref_ligand_path=config["pocket"]["ref_ligand"],
        device=device,
        dataset_name=dataset_name,
    )



def build_dataloader(
    config: dict[str, Any],
    dataset_info: dict[str, Any],
    device: str,
) -> DiffSBDDDataLoader:
    """Construct the single-GPU prompt dataloader."""
    return DiffSBDDDataLoader(
        config=config,
        dataset_info=dataset_info,
        device=device,
        num_batches=config.get("dataloader", {}).get("epochs"),
    )



def build_filter(config: dict[str, Any], dataset_info: dict[str, Any]) -> DiffSBDDFilter:
    """Construct the canonical molecule filter."""
    return DiffSBDDFilter(
        dataset_info=dataset_info,
        enable_filtering=bool(config.get("filter", {}).get("enable_filtering", False)),
        enable_penalty=bool(config.get("filter", {}).get("enable_penalty", False)),
        novelty_penalty_coef=float(
            config.get("filter", {}).get("novelty_penalty_coef", 0.1)
        ),
    )



def build_actor(model: DiffSBDDModel, config: dict[str, Any], dataset_info: dict[str, Any]):
    """Construct the PPO actor and inject explicit max-node semantics."""
    actor = EDMActor(model, config)
    actor.max_n_nodes = config.get("dataloader", {}).get(
        "max_n_nodes",
        config.get("dataloader", {}).get("max_nodes", dataset_info.get("max_n_nodes", 50)),
    )
    return actor



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
    filters = build_filter(config, dataset_info)
    actor = build_actor(model, config, dataset_info)
    run_recorder = RunRecorder(run_context=run_context, logger=logger, world_size=1)

    logger.info("Initialized DiffSBDDModel")
    logger.info("Initialized DiffSBDDDataLoader")
    logger.info("Initialized DiffSBDDRollout")
    logger.info("Initialized rewarder: %s", type(rewarder).__name__)
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
        filters=filters,
        actor=actor,
        run_recorder=run_recorder,
        max_epochs=max_epochs,
        checkpoint_interval=checkpoint_interval,
    )



def prepare_training_samples(runtime: RuntimeBundle, epoch: int) -> PreparedSamples:
    """Prepare one training sample batch with explicit stage boundaries."""
    logger = runtime.logger

    logger.info("[Epoch %d/%d] generate_batch", epoch, runtime.max_epochs)
    prompts = runtime.dataloader.generate_batch()

    logger.info("[Epoch %d/%d] generate_minibatch", epoch, runtime.max_epochs)
    samples = runtime.rollout.generate_minibatch(prompts)

    logger.info("[Epoch %d/%d] annotate_batch_molecule_semantics", epoch, runtime.max_epochs)
    samples = annotate_batch_molecule_semantics(samples, dataset_info=runtime.dataset_info)

    logger.info("[Epoch %d/%d] filters.filter", epoch, runtime.max_epochs)
    samples, filtering_ratio, novelty_penalty_ratio = runtime.filters.filter(samples)

    logger.info("[Epoch %d/%d] rewarder.calculate_rewards", epoch, runtime.max_epochs)
    samples = runtime.rewarder.calculate_rewards(samples)

    if "advantages" not in samples.batch:
        logger.info("[Epoch %d/%d] advantages = rewards", epoch, runtime.max_epochs)
        samples.batch["advantages"] = samples.batch["rewards"]

    return PreparedSamples(
        samples=samples,
        filtering_ratio=filtering_ratio,
        novelty_penalty_ratio=novelty_penalty_ratio,
    )



def update_policy(runtime: RuntimeBundle, prepared: PreparedSamples, epoch: int) -> dict[str, Any]:
    """Execute the PPO policy update on prepared samples."""
    runtime.logger.info("[Epoch %d/%d] actor.update_policy", epoch, runtime.max_epochs)
    metrics = runtime.actor.update_policy(prepared.samples)
    if metrics is None:
        return {}
    return dict(metrics)



def _to_float_list(values: Any) -> list[float]:
    if values is None:
        return []
    if isinstance(values, torch.Tensor):
        return [float(v) for v in values.detach().cpu().reshape(-1).tolist()]
    return [float(v) for v in list(values)]



def _to_int_list(values: Any) -> list[int]:
    if values is None:
        return []
    if isinstance(values, torch.Tensor):
        return [int(v) for v in values.detach().cpu().reshape(-1).tolist()]
    return [int(v) for v in list(values)]



def _to_str_list(values: Any) -> list[str]:
    if values is None:
        return []
    return [str(v) for v in list(values)]



def compute_batch_stats(prepared: PreparedSamples) -> BatchStats:
    """Compute batch-local statistics used for recording and console summaries."""
    samples = prepared.samples

    rewards = _to_float_list(samples.batch.get("rewards"))
    qed_scores = _to_float_list(samples.batch.get("qed_scores", samples.batch.get("qed")))
    sa_scores = _to_float_list(samples.batch.get("sa_scores", samples.batch.get("sa")))
    valid_flags = _to_int_list(samples.batch.get("rdkit_valid"))
    failure_types = _to_str_list(samples.batch.get("failure_type"))

    reward_mean = float(sum(rewards) / len(rewards)) if rewards else 0.0
    reward_min = float(min(rewards)) if rewards else 0.0
    reward_max = float(max(rewards)) if rewards else 0.0
    avg_qed = float(sum(qed_scores) / len(qed_scores)) if qed_scores else 0.0
    avg_sa = float(sum(sa_scores) / len(sa_scores)) if sa_scores else 0.0
    generated_count = len(valid_flags) if valid_flags else int(samples.batch.batch_size[0])
    valid_count = int(sum(valid_flags)) if valid_flags else 0
    valid_ratio = float(valid_count / generated_count) if generated_count > 0 else 0.0
    failure_counts = dict(Counter(failure_types)) if failure_types else {}

    return BatchStats(
        reward_mean=reward_mean,
        reward_min=reward_min,
        reward_max=reward_max,
        avg_qed=avg_qed,
        avg_sa=avg_sa,
        generated_count=generated_count,
        valid_count=valid_count,
        valid_ratio=valid_ratio,
        failure_counts=failure_counts,
    )



def build_batch_facts(
    prepared: PreparedSamples,
    epoch: int,
    batch_idx: int,
    batch_stats: BatchStats,
) -> BatchFacts:
    """Build recorder-ready batch facts from the prepared sample batch."""
    samples = prepared.samples
    smiles_values = list(samples.batch.get("smiles", []))
    valid_flags = _to_int_list(samples.batch.get("rdkit_valid"))
    qed_scores = _to_float_list(samples.batch.get("qed_scores", samples.batch.get("qed")))
    sa_scores = _to_float_list(samples.batch.get("sa_scores", samples.batch.get("sa")))
    rewards = _to_float_list(samples.batch.get("rewards"))
    drug_rewards = _to_float_list(samples.batch.get("druglikeness_reward"))
    docking_rewards = _to_float_list(samples.batch.get("docking_reward"))

    molecule_rows = []
    for molecule_id in range(batch_stats.generated_count):
        molecule_rows.append(
            {
                "epoch": epoch,
                "batch_idx": batch_idx,
                "rank": 0,
                "molecule_id": molecule_id,
                "smiles": smiles_values[molecule_id] if molecule_id < len(smiles_values) else "",
                "rdkit_valid": valid_flags[molecule_id] if molecule_id < len(valid_flags) else 0,
                "qed": qed_scores[molecule_id] if molecule_id < len(qed_scores) else 0.0,
                "sa_raw": sa_scores[molecule_id] if molecule_id < len(sa_scores) else 0.0,
                "reward_total": rewards[molecule_id] if molecule_id < len(rewards) else 0.0,
                "reward_druglikeness": drug_rewards[molecule_id] if molecule_id < len(drug_rewards) else 0.0,
                "reward_docking": docking_rewards[molecule_id] if molecule_id < len(docking_rewards) else 0.0,
            }
        )

    return BatchFacts(
        epoch=epoch,
        batch_idx=batch_idx,
        rank=0,
        generated_count=batch_stats.generated_count,
        valid_count=batch_stats.valid_count,
        reward_mean=batch_stats.reward_mean,
        reward_min=batch_stats.reward_min,
        reward_max=batch_stats.reward_max,
        avg_qed=batch_stats.avg_qed,
        avg_sa=batch_stats.avg_sa,
        filtering_ratio=prepared.filtering_ratio,
        novelty_penalty_ratio=prepared.novelty_penalty_ratio,
        failure_counts=batch_stats.failure_counts,
        molecule_rows=molecule_rows,
    )



def build_epoch_facts(
    runtime: RuntimeBundle,
    epoch: int,
    metrics: dict[str, Any],
    batch_stats: BatchStats,
    prepared: PreparedSamples,
) -> EpochFacts:
    """Build recorder-ready epoch facts from batch-local summaries."""
    avg_loss = float(metrics.get("Loss", metrics.get("loss", metrics.get("ppo_loss", 0.0))))
    checkpoint_due = (epoch % runtime.checkpoint_interval) == 0

    return EpochFacts(
        epoch=epoch,
        timestamp=datetime.now().isoformat(timespec="seconds"),
        avg_loss=avg_loss,
        reward_mean=batch_stats.reward_mean,
        reward_min=batch_stats.reward_min,
        reward_max=batch_stats.reward_max,
        avg_qed=batch_stats.avg_qed,
        avg_sa=batch_stats.avg_sa,
        generated_total=batch_stats.generated_count,
        rdkit_valid=batch_stats.valid_count,
        rdkit_valid_ratio=batch_stats.valid_ratio,
        filtering_ratio_mean=prepared.filtering_ratio,
        novelty_penalty_ratio_mean=prepared.novelty_penalty_ratio,
        failure_counts=batch_stats.failure_counts,
        extra_metrics={"max_epochs": runtime.max_epochs},
        checkpoint_due=checkpoint_due,
    )



def record_epoch(
    runtime: RuntimeBundle,
    epoch: int,
    prepared: PreparedSamples,
    metrics: dict[str, Any],
    batch_stats: BatchStats,
) -> None:
    """Persist batch-level and epoch-level recorder outputs for one epoch."""
    batch_facts = build_batch_facts(prepared=prepared, epoch=epoch, batch_idx=0, batch_stats=batch_stats)
    runtime.run_recorder.on_batch_end(batch_facts)

    epoch_facts = build_epoch_facts(
        runtime=runtime,
        epoch=epoch,
        metrics=metrics,
        batch_stats=batch_stats,
        prepared=prepared,
    )
    runtime.run_recorder.on_epoch_end(epoch_facts)



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



def build_run_lifecycle(runtime: RuntimeBundle, status: RunStatus, message: str | None = None) -> RunLifecycleFacts:
    """Build the final run lifecycle payload."""
    return RunLifecycleFacts(
        run_id=runtime.run_context.run_id,
        run_name=runtime.run_context.run_name,
        output_dir=runtime.run_context.output_dir,
        status=status,
        started_at=runtime.run_context.started_at,
        ended_at=datetime.now().isoformat(timespec="seconds"),
        message=message,
    )



def finish_run(runtime: RuntimeBundle, status: RunStatus, message: str | None = None) -> None:
    """Record terminal run lifecycle status through the recorder facade."""
    lifecycle = build_run_lifecycle(runtime, status=status, message=message)
    extra_lines = [f"Output Dir: {runtime.run_context.output_dir}"]
    if message:
        extra_lines.append(f"Message   : {message}")
    runtime.run_recorder.on_run_finished(lifecycle, extra_lines=extra_lines)



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
    record_epoch(runtime, epoch, prepared, metrics, batch_stats)
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
