from __future__ import annotations

"""Native PyTorch DDP training entrypoint for DiffSBDD + DDPO."""

import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "RLPF-verl_diffusion"))
sys.path.insert(0, str(PROJECT_ROOT / "DiffSBDD"))

from constants import dataset_params
from ddpo.main_diffsbdd_single_gpu import (
    BatchStats,
    EpochStepResult,
    PreparedSamples,
    apply_cli_overrides,
    build_actor,
    build_batch_facts,
    build_dataloader,
    build_epoch_facts,
    build_filter,
    build_model,
    compute_batch_stats,
    finish_run,
    load_config,
    parse_args,
    prepare_training_samples,
    save_config_snapshot,
    setup_chemistry_logging,
    update_policy,
)
from ddpo.rewards.docking import DockingReward
from ddpo.rewards.factory import build_rewarder
from ddpo.runtime import (
    DiffSBDDRollout,
    DistributedContext,
    all_gather_objects,
    barrier_if_needed,
    broadcast_object,
    destroy_distributed,
    init_distributed_from_env,
    is_main_process,
    reduce_max_scalar,
    reduce_mean_scalar,
    reduce_min_scalar,
    reduce_sum_scalar,
    set_seed_for_rank,
    setup_logging,
)
from ddpo.runtime.recording import CheckpointFacts, RunRecorder, RunStatus, create_run_context
from ddpo.runtime.recording.run_context import RunContext


@dataclass
class DistributedRuntimeBundle:
    args: Any
    config: dict[str, Any]
    seed: int
    device: str
    dataset_name: str
    dataset_info: dict[str, Any]
    run_context: Any
    logger: Any
    dist_ctx: DistributedContext
    model_impl: Any
    model_train: Any
    dataloader: Any
    rollout: DiffSBDDRollout
    rewarder: Any
    filters: Any
    actor: Any
    run_recorder: RunRecorder
    max_epochs: int
    checkpoint_interval: int
    per_rank_batch_size: int
    global_batch_size: int


@dataclass
class GlobalBatchStats:
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
    metrics: dict[str, float]


def _iter_wrapped_rewards(rewarder: Any):
    yield rewarder
    base_reward = getattr(rewarder, "base_reward", None)
    if base_reward is not None:
        yield from _iter_wrapped_rewards(base_reward)
    rewards = getattr(rewarder, "rewards", None)
    if isinstance(rewards, dict):
        for child in rewards.values():
            yield from _iter_wrapped_rewards(child)


def bind_reward_runtime_devices(rewarder: Any, gpu_id: int) -> None:
    for candidate in _iter_wrapped_rewards(rewarder):
        if isinstance(candidate, DockingReward):
            candidate.gpu_id = int(gpu_id)
            candidate._autodock_backend = None


def build_rank_run_context(
    config: dict[str, Any],
    save_path: str | None,
    dist_ctx: DistributedContext,
) -> Any:
    """Build one shared output directory with rank-local log subdirectories."""
    if is_main_process(dist_ctx):
        shared_context = create_run_context(config, save_path=save_path)
        shared_payload = {
            "run_id": shared_context.run_id,
            "run_name": shared_context.run_name,
            "started_at": shared_context.started_at,
            "output_dir": shared_context.output_dir,
            "logs_dir": shared_context.logs_dir,
            "epochs_dir": shared_context.epochs_dir,
            "checkpoints_dir": shared_context.checkpoints_dir,
            "lightning_logs_dir": shared_context.lightning_logs_dir,
            "config_snapshot_path": shared_context.config_snapshot_path,
            "training_summary_csv_path": shared_context.training_summary_csv_path,
            "run_manifest_path": shared_context.run_manifest_path,
            "checkpoint_manifest_path": shared_context.checkpoint_manifest_path,
        }
    else:
        shared_payload = None

    shared_payload = broadcast_object(shared_payload, dist_ctx)
    output_dir = Path(shared_payload["output_dir"])
    rank_logs_dir = output_dir / "logs" / f"rank_{dist_ctx.rank:02d}"
    rank_logs_dir.mkdir(parents=True, exist_ok=True)

    return RunContext(
        run_id=shared_payload["run_id"],
        run_name=shared_payload["run_name"],
        started_at=shared_payload["started_at"],
        output_dir=shared_payload["output_dir"],
        logs_dir=str(rank_logs_dir),
        epochs_dir=shared_payload["epochs_dir"],
        checkpoints_dir=shared_payload["checkpoints_dir"],
        lightning_logs_dir=shared_payload["lightning_logs_dir"],
        config_snapshot_path=shared_payload["config_snapshot_path"],
        training_summary_csv_path=shared_payload["training_summary_csv_path"],
        run_manifest_path=shared_payload["run_manifest_path"],
        checkpoint_manifest_path=shared_payload["checkpoint_manifest_path"],
    )


def build_runtime_ddp(args, config: dict[str, Any], dist_ctx: DistributedContext) -> DistributedRuntimeBundle:
    if args.resume or args.checkpoint_path is not None:
        raise NotImplementedError(
            "DDP resume/checkpoint restore is not implemented yet. "
            "Run without --resume / --checkpoint_path."
        )

    config = apply_cli_overrides(config, args)
    seed = int(config.get("train", {}).get("seed", 42))
    set_seed_for_rank(seed, dist_ctx.rank)
    setup_chemistry_logging()

    run_context = build_rank_run_context(config, save_path=args.save_path, dist_ctx=dist_ctx)
    logger = setup_logging(
        log_dir=run_context.logs_dir,
        log_level=args.log_level,
        console_enabled=is_main_process(dist_ctx),
    )
    if is_main_process(dist_ctx):
        save_config_snapshot(config, run_context.config_snapshot_path)
    barrier_if_needed(dist_ctx)

    dataset_name = config.get("dataset", "crossdock")
    dataset_info = dataset_params[dataset_name]
    device = str(dist_ctx.device)

    model_impl = build_model(config, device, dataset_name)
    model_train = DDP(
        model_impl,
        device_ids=[dist_ctx.local_rank],
        output_device=dist_ctx.local_rank,
        find_unused_parameters=bool(config.get("ddp", {}).get("find_unused_parameters", False)),
    )
    dataloader = build_dataloader(config, dataset_info, device)
    rollout = DiffSBDDRollout(model=model_impl, config=config)
    rewarder = build_rewarder(config=config, dataset_info=dataset_info, logger=logger)
    bind_reward_runtime_devices(rewarder, gpu_id=dist_ctx.local_rank)
    filters = build_filter(config, dataset_info)
    actor = build_actor(model_train, config, dataset_info, model_impl=model_impl)
    run_recorder = RunRecorder(run_context=run_context, logger=logger, world_size=dist_ctx.world_size)

    per_rank_batch_size = int(dataloader.batch_size)
    global_batch_size = int(per_rank_batch_size * dist_ctx.world_size)
    max_epochs = int(config.get("train", {}).get("epochs", config.get("dataloader", {}).get("epochs", 1)))
    checkpoint_interval = int(config.get("train", {}).get("checkpoint_interval", 1))
    logger.info(
        "Initialized DDP runtime | rank=%d local_rank=%d world_size=%d device=%s per_rank_batch=%d global_batch=%d",
        dist_ctx.rank,
        dist_ctx.local_rank,
        dist_ctx.world_size,
        device,
        per_rank_batch_size,
        global_batch_size,
    )

    return DistributedRuntimeBundle(
        args=args,
        config=config,
        seed=seed,
        device=device,
        dataset_name=dataset_name,
        dataset_info=dataset_info,
        run_context=run_context,
        logger=logger,
        dist_ctx=dist_ctx,
        model_impl=model_impl,
        model_train=model_train,
        dataloader=dataloader,
        rollout=rollout,
        rewarder=rewarder,
        filters=filters,
        actor=actor,
        run_recorder=run_recorder,
        max_epochs=max_epochs,
        checkpoint_interval=checkpoint_interval,
        per_rank_batch_size=per_rank_batch_size,
        global_batch_size=global_batch_size,
    )


def _merge_failure_counts(counts_per_rank: list[dict[str, int]]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for counts in counts_per_rank:
        for key, value in counts.items():
            merged[key] = merged.get(key, 0) + int(value)
    return merged


def _merge_metrics(metrics_per_rank: list[dict[str, Any]]) -> dict[str, float]:
    if not metrics_per_rank:
        return {}
    keys = set().union(*(metric.keys() for metric in metrics_per_rank))
    merged: dict[str, float] = {}
    for key in keys:
        values = [float(metric[key]) for metric in metrics_per_rank if key in metric]
        if values:
            merged[key] = float(sum(values) / len(values))
    return merged


def maybe_check_param_sync(runtime: DistributedRuntimeBundle, epoch: int) -> None:
    """Optionally verify that a small parameter summary matches across ranks.

    This is a lightweight smoke check for DDP correctness and is intentionally
    disabled by default. It compares only a small prefix of parameter norms to
    keep overhead low and avoid interfering with training semantics.
    """
    check_cfg = runtime.config.get("ddp", {})
    if not bool(check_cfg.get("param_sync_check", False)):
        return

    max_params = int(check_cfg.get("param_sync_check_max_params", 8))
    atol = float(check_cfg.get("param_sync_check_atol", 1e-6))

    summary: list[tuple[str, float]] = []
    with torch.no_grad():
        for idx, (name, param) in enumerate(runtime.model_impl.named_parameters()):
            if idx >= max_params:
                break
            summary.append((name, float(param.detach().float().norm().item())))

    gathered = all_gather_objects(summary, runtime.dist_ctx)
    reference = gathered[0] if gathered else summary
    mismatch_messages: list[str] = []
    for rank_idx, candidate in enumerate(gathered[1:], start=1):
        if len(candidate) != len(reference):
            mismatch_messages.append(
                f"rank{rank_idx}: summary length {len(candidate)} != rank0 length {len(reference)}"
            )
            continue
        for (ref_name, ref_norm), (cand_name, cand_norm) in zip(reference, candidate):
            if ref_name != cand_name:
                mismatch_messages.append(
                    f"rank{rank_idx}: param name mismatch {cand_name!r} != {ref_name!r}"
                )
                break
            if abs(cand_norm - ref_norm) > atol:
                mismatch_messages.append(
                    f"rank{rank_idx}: {ref_name} norm mismatch {cand_norm:.8f} != {ref_norm:.8f}"
                )
                break

    if mismatch_messages:
        raise RuntimeError(
            f"DDP parameter sync smoke check failed at epoch {epoch}: "
            + "; ".join(mismatch_messages)
        )

    if is_main_process(runtime.dist_ctx):
        runtime.logger.info(
            "[DDP CHECK] Epoch %d parameter sync smoke check passed for %d params",
            epoch,
            len(reference),
        )


def reduce_batch_stats(local_stats: BatchStats, prepared: PreparedSamples, metrics: dict[str, Any], dist_ctx: DistributedContext) -> GlobalBatchStats:
    count = float(local_stats.generated_count)
    generated_count = int(round(reduce_sum_scalar(float(local_stats.generated_count), dist_ctx)))
    valid_count = int(round(reduce_sum_scalar(float(local_stats.valid_count), dist_ctx)))
    reward_sum = reduce_sum_scalar(float(local_stats.reward_mean * count), dist_ctx)
    qed_sum = reduce_sum_scalar(float(local_stats.avg_qed * count), dist_ctx)
    sa_sum = reduce_sum_scalar(float(local_stats.avg_sa * count), dist_ctx)
    failure_counts = _merge_failure_counts(all_gather_objects(local_stats.failure_counts, dist_ctx))
    merged_metrics = _merge_metrics(all_gather_objects(metrics, dist_ctx))
    return GlobalBatchStats(
        reward_mean=(reward_sum / generated_count) if generated_count > 0 else 0.0,
        reward_min=reduce_min_scalar(float(local_stats.reward_min), dist_ctx),
        reward_max=reduce_max_scalar(float(local_stats.reward_max), dist_ctx),
        avg_qed=(qed_sum / generated_count) if generated_count > 0 else 0.0,
        avg_sa=(sa_sum / generated_count) if generated_count > 0 else 0.0,
        generated_count=generated_count,
        valid_count=valid_count,
        valid_ratio=(valid_count / generated_count) if generated_count > 0 else 0.0,
        filtering_ratio=reduce_mean_scalar(float(prepared.filtering_ratio), dist_ctx),
        novelty_penalty_ratio=reduce_mean_scalar(float(prepared.novelty_penalty_ratio), dist_ctx),
        failure_counts=failure_counts,
        metrics=merged_metrics,
    )


def save_checkpoint_ddp(runtime: DistributedRuntimeBundle, epoch: int, metrics: dict[str, Any]) -> CheckpointFacts | None:
    barrier_if_needed(runtime.dist_ctx)
    if not is_main_process(runtime.dist_ctx):
        barrier_if_needed(runtime.dist_ctx)
        return None
    if (epoch % runtime.checkpoint_interval) != 0:
        barrier_if_needed(runtime.dist_ctx)
        return None

    checkpoint_name = f"diffsbdd-ddpo-epoch={epoch:04d}.pt"
    checkpoint_path = str(Path(runtime.run_context.checkpoints_dir) / checkpoint_name)
    payload = {
        "epoch": epoch,
        "config": runtime.config,
        "seed": runtime.seed,
        "model_state_dict": runtime.model_impl.model.state_dict(),
        "optimizer_state_dict": runtime.actor.optimizer.state_dict(),
        "metrics": metrics,
    }
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
    barrier_if_needed(runtime.dist_ctx)
    return facts


def record_epoch_ddp(
    runtime: DistributedRuntimeBundle,
    epoch: int,
    prepared: PreparedSamples,
    local_stats: BatchStats,
    global_stats: GlobalBatchStats,
) -> None:
    rank = int(runtime.dist_ctx.rank)
    batch_facts = build_batch_facts(prepared=prepared, epoch=epoch, batch_idx=0, batch_stats=local_stats)
    batch_facts.rank = rank
    for row in batch_facts.molecule_rows:
        row["rank"] = rank
    runtime.run_recorder.on_batch_end(batch_facts)
    barrier_if_needed(runtime.dist_ctx)

    if not is_main_process(runtime.dist_ctx):
        return

    epoch_facts = build_epoch_facts(
        runtime=runtime,
        epoch=epoch,
        metrics=global_stats.metrics,
        batch_stats=BatchStats(
            reward_mean=global_stats.reward_mean,
            reward_min=global_stats.reward_min,
            reward_max=global_stats.reward_max,
            avg_qed=global_stats.avg_qed,
            avg_sa=global_stats.avg_sa,
            generated_count=global_stats.generated_count,
            valid_count=global_stats.valid_count,
            valid_ratio=global_stats.valid_ratio,
            failure_counts=global_stats.failure_counts,
        ),
        prepared=PreparedSamples(
            samples=prepared.samples,
            filtering_ratio=global_stats.filtering_ratio,
            novelty_penalty_ratio=global_stats.novelty_penalty_ratio,
        ),
    )
    runtime.run_recorder.on_epoch_end(epoch_facts)


def run_epoch_ddp(runtime: DistributedRuntimeBundle, epoch: int) -> EpochStepResult:
    prepared = prepare_training_samples(runtime, epoch)
    metrics = update_policy(runtime, prepared, epoch)
    maybe_check_param_sync(runtime, epoch)
    local_stats = compute_batch_stats(prepared)
    global_stats = reduce_batch_stats(local_stats, prepared, metrics, runtime.dist_ctx)
    record_epoch_ddp(runtime, epoch, prepared, local_stats, global_stats)
    save_checkpoint_ddp(runtime, epoch, global_stats.metrics)
    return EpochStepResult(
        epoch=epoch,
        metrics=global_stats.metrics,
        reward_mean=global_stats.reward_mean,
        reward_min=global_stats.reward_min,
        reward_max=global_stats.reward_max,
        avg_qed=global_stats.avg_qed,
        avg_sa=global_stats.avg_sa,
        generated_count=global_stats.generated_count,
        valid_count=global_stats.valid_count,
        valid_ratio=global_stats.valid_ratio,
        filtering_ratio=global_stats.filtering_ratio,
        novelty_penalty_ratio=global_stats.novelty_penalty_ratio,
        failure_counts=global_stats.failure_counts,
    )


def run_training_loop_ddp(runtime: DistributedRuntimeBundle) -> None:
    if is_main_process(runtime.dist_ctx):
        runtime.run_recorder.on_run_started()
    runtime.logger.info("Starting native DDP training loop for %d epochs", runtime.max_epochs)
    try:
        for epoch in range(1, runtime.max_epochs + 1):
            result = run_epoch_ddp(runtime, epoch)
            runtime.logger.info(
                "[Epoch %d/%d][rank=%d] complete | generated=%d valid=%d reward_mean=%.4f metrics=%s",
                epoch,
                runtime.max_epochs,
                runtime.dist_ctx.rank,
                result.generated_count,
                result.valid_count,
                result.reward_mean,
                result.metrics,
            )
    except KeyboardInterrupt:
        barrier_if_needed(runtime.dist_ctx)
        if is_main_process(runtime.dist_ctx):
            finish_run(runtime, RunStatus.INTERRUPTED, message="KeyboardInterrupt")
        raise
    except Exception as exc:
        barrier_if_needed(runtime.dist_ctx)
        if is_main_process(runtime.dist_ctx):
            finish_run(runtime, RunStatus.FAILED, message=f"{type(exc).__name__}: {exc}")
        raise
    else:
        barrier_if_needed(runtime.dist_ctx)
        if is_main_process(runtime.dist_ctx):
            finish_run(runtime, RunStatus.COMPLETED)


def main() -> None:
    args = parse_args()
    config = load_config(args.config_path)
    dist_ctx = init_distributed_from_env(config)
    try:
        runtime = build_runtime_ddp(args, config, dist_ctx)
        run_training_loop_ddp(runtime)
    finally:
        destroy_distributed()


if __name__ == "__main__":
    main()
