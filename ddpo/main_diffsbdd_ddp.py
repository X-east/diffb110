from __future__ import annotations

"""Native PyTorch DDP training entrypoint for DiffSBDD + DDPO."""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "RLPF-verl_diffusion"))
sys.path.insert(0, str(PROJECT_ROOT / "DiffSBDD"))

from constants import dataset_params
from ddpo.runtime.build import (
    apply_cli_overrides,
    build_actor,
    build_dataloader,
    build_filter,
    build_model,
    build_ref_model,
    build_reward_normalizer,
    load_config,
    save_config_snapshot,
)
from ddpo.runtime.cli import parse_args
from ddpo.runtime.checkpointing import (
    resolve_checkpoint_path,
    restore_checkpoint_if_requested,
    save_checkpoint_ddp,
)
from ddpo.runtime.train_helpers import (
    BatchStats,
    EpochStepResult,
    PreparedSamples,
    build_batch_facts,
    build_epoch_facts,
    compute_batch_stats,
    compute_epoch_molecular_descriptors,
    finish_run,
    prepare_training_samples,
    update_policy,
)
from ddpo.utils.suppress_warnings import setup_chemistry_logging
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
from ddpo.runtime.recording import RunRecorder, RunStatus, create_run_context
from ddpo.runtime.recording.run_context import RunContext
from ddpo.utils import PerfTimer


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
    reward_normalizer: Any
    filters: Any
    actor: Any
    run_recorder: RunRecorder
    max_epochs: int
    checkpoint_interval: int
    per_rank_batch_size: int
    global_batch_size: int
    ref_model: Any = None
    start_epoch: int = 0
    resumed_from_checkpoint: str | None = None
    resume_checkpoint_version: int | None = None


@dataclass
class GlobalBatchStats:
    reward_mean: float
    reward_min: float
    reward_max: float
    avg_qed: float
    avg_sa: float
    avg_docking: float
    avg_similarity: float
    avg_diversity: float
    avg_qed_norm: float
    avg_sa_norm: float
    avg_docking_norm: float
    avg_diversity_norm: float
    generated_count: int
    valid_count: int
    valid_ratio: float
    filtering_ratio: float
    novelty_penalty_ratio: float
    failure_counts: dict[str, int]
    metrics: dict[str, float]
    reward_raw_mean: float = 0.0
    avg_chembl_reward: float = 0.0
    avg_chembl_similarity: float = 0.0
    chembl_prior_metrics: dict[str, float] | None = None


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
            candidate._unidock_backend = None


def bind_reward_distributed_context(rewarder: Any, dist_ctx: DistributedContext) -> None:
    for candidate in _iter_wrapped_rewards(rewarder):
        setter = getattr(candidate, "set_distributed_context", None)
        if callable(setter):
            setter(dist_ctx)


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
    config = apply_cli_overrides(config, args)
    checkpoint_path = resolve_checkpoint_path(args)
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

    # --- Optional frozen reference model for KL penalty ---
    ref_model = build_ref_model(config, model_impl)

    dataloader = build_dataloader(config, dataset_info, device, model_impl=model_impl)
    rollout = DiffSBDDRollout(model=model_impl, config=config)
    rewarder = build_rewarder(config=config, dataset_info=dataset_info, logger=logger)
    reward_normalizer = build_reward_normalizer(config=config, logger=logger)
    bind_reward_runtime_devices(rewarder, gpu_id=dist_ctx.local_rank)
    bind_reward_distributed_context(rewarder, dist_ctx=dist_ctx)
    filters = build_filter(config, dataset_info)
    actor = build_actor(model_train, config, dataset_info,
                        model_impl=model_impl, ref_model=ref_model)
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
    logger.info("Initialized rewarder: %s", type(rewarder).__name__)
    logger.info("Initialized reward normalizer: %s", type(reward_normalizer).__name__)

    runtime = DistributedRuntimeBundle(
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
        reward_normalizer=reward_normalizer,
        filters=filters,
        actor=actor,
        run_recorder=run_recorder,
        ref_model=ref_model,
        max_epochs=max_epochs,
        checkpoint_interval=checkpoint_interval,
        per_rank_batch_size=per_rank_batch_size,
        global_batch_size=global_batch_size,
    )

    restore_checkpoint_if_requested(runtime, checkpoint_path)
    if runtime.start_epoch >= runtime.max_epochs:
        logger.warning(
            "Resume checkpoint epoch (%d) is >= configured max_epochs (%d); training loop will not run.",
            runtime.start_epoch,
            runtime.max_epochs,
        )
    return runtime


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


def maybe_log_ddp_probe(runtime: DistributedRuntimeBundle, epoch: int) -> None:
    """Log identity and local-gradient probes for DDP diagnosis."""
    probe_cfg = runtime.config.get("ddp", {})
    if not bool(probe_cfg.get("grad_sync_probe", False)):
        return

    model_train_module = getattr(runtime.model_train, "module", None)
    actor_model_module = getattr(runtime.actor.model, "module", None)
    local_probe = {
        "rank": int(runtime.dist_ctx.rank),
        "model_train_module_is_model_impl": model_train_module is runtime.model_impl,
        "actor_model_is_model_train": runtime.actor.model is runtime.model_train,
        "actor_model_module_is_model_impl": actor_model_module is runtime.model_impl,
        "actor_model_impl_is_model_impl": runtime.actor.model_impl is runtime.model_impl,
        "actor_model_impl_is_actor_model_module": actor_model_module is runtime.actor.model_impl,
        "grad_probe": list(getattr(runtime.actor, "last_grad_probe", [])),
    }
    gathered = all_gather_objects(local_probe, runtime.dist_ctx)
    barrier_if_needed(runtime.dist_ctx)
    if not is_main_process(runtime.dist_ctx):
        return

    runtime.logger.info("[DDP PROBE] Epoch %d identity/grad probe start", epoch)
    for item in gathered:
        runtime.logger.info(
            "[DDP PROBE][rank=%d] identity: train.module_is_impl=%s actor.model_is_train=%s actor.model.module_is_impl=%s actor.model_impl_is_impl=%s actor.model_impl_is_actor.model.module=%s",
            item["rank"],
            item["model_train_module_is_model_impl"],
            item["actor_model_is_model_train"],
            item["actor_model_module_is_model_impl"],
            item["actor_model_impl_is_model_impl"],
            item["actor_model_impl_is_actor_model_module"],
        )
        for probe in item["grad_probe"]:
            runtime.logger.info(
                "[DDP PROBE][rank=%d] %s | param_norm=%.8f grad_is_none=%s grad_norm=%s grad_mean=%s",
                item["rank"],
                probe["name"],
                probe["param_norm"],
                probe["grad_is_none"],
                "None" if probe["grad_norm"] is None else f"{probe['grad_norm']:.8f}",
                "None" if probe["grad_mean"] is None else f"{probe['grad_mean']:.8e}",
            )
    runtime.logger.info("[DDP PROBE] Epoch %d identity/grad probe end", epoch)


def reduce_batch_stats(local_stats: BatchStats, prepared: PreparedSamples, metrics: dict[str, Any], dist_ctx: DistributedContext) -> GlobalBatchStats:
    count = float(local_stats.generated_count)
    generated_count = int(round(reduce_sum_scalar(float(local_stats.generated_count), dist_ctx)))
    valid_count = int(round(reduce_sum_scalar(float(local_stats.valid_count), dist_ctx)))
    valid_count_local = float(local_stats.valid_count)
    reward_sum = reduce_sum_scalar(float(local_stats.reward_mean * count), dist_ctx)
    # Raw property averages are now over VALID molecules only; use valid_count as multiplier/divisor.
    qed_sum = reduce_sum_scalar(float(local_stats.avg_qed * valid_count_local), dist_ctx)
    sa_sum = reduce_sum_scalar(float(local_stats.avg_sa * valid_count_local), dist_ctx)
    docking_sum = reduce_sum_scalar(float(local_stats.avg_docking * valid_count_local), dist_ctx)
    similarity_sum = reduce_sum_scalar(float(local_stats.avg_similarity * valid_count_local), dist_ctx)
    diversity_sum = reduce_sum_scalar(float(local_stats.avg_diversity * valid_count_local), dist_ctx)
    qed_norm_sum = reduce_sum_scalar(float(local_stats.avg_qed_norm * count), dist_ctx)
    sa_norm_sum = reduce_sum_scalar(float(local_stats.avg_sa_norm * count), dist_ctx)
    docking_norm_sum = reduce_sum_scalar(float(local_stats.avg_docking_norm * count), dist_ctx)
    diversity_norm_sum = reduce_sum_scalar(float(local_stats.avg_diversity_norm * count), dist_ctx)
    reward_raw_sum = reduce_sum_scalar(float(local_stats.reward_raw_mean * count), dist_ctx)
    chembl_reward_sum = reduce_sum_scalar(float(local_stats.avg_chembl_reward * valid_count_local), dist_ctx)
    chembl_similarity_sum = reduce_sum_scalar(float(local_stats.avg_chembl_similarity * valid_count_local), dist_ctx)
    chembl_prior_metrics: dict[str, float] = {}
    for key, value in (local_stats.chembl_prior_metrics or {}).items():
        reduced = reduce_sum_scalar(float(value) * valid_count_local, dist_ctx)
        chembl_prior_metrics[key] = (reduced / valid_count) if valid_count > 0 else 0.0
    failure_counts = _merge_failure_counts(all_gather_objects(local_stats.failure_counts, dist_ctx))
    merged_metrics = _merge_metrics(all_gather_objects(metrics, dist_ctx))
    return GlobalBatchStats(
        reward_mean=(reward_sum / generated_count) if generated_count > 0 else 0.0,
        reward_min=reduce_min_scalar(float(local_stats.reward_min), dist_ctx),
        reward_max=reduce_max_scalar(float(local_stats.reward_max), dist_ctx),
        avg_qed=(qed_sum / valid_count) if valid_count > 0 else 0.0,
        avg_sa=(sa_sum / valid_count) if valid_count > 0 else 0.0,
        avg_docking=(docking_sum / valid_count) if valid_count > 0 else 0.0,
        avg_similarity=(similarity_sum / valid_count) if valid_count > 0 else 0.0,
        avg_diversity=(diversity_sum / valid_count) if valid_count > 0 else 0.0,
        avg_qed_norm=(qed_norm_sum / generated_count) if generated_count > 0 else 0.0,
        avg_sa_norm=(sa_norm_sum / generated_count) if generated_count > 0 else 0.0,
        avg_docking_norm=(docking_norm_sum / generated_count) if generated_count > 0 else 0.0,
        avg_diversity_norm=(diversity_norm_sum / generated_count) if generated_count > 0 else 0.0,
        generated_count=generated_count,
        valid_count=valid_count,
        valid_ratio=(valid_count / generated_count) if generated_count > 0 else 0.0,
        filtering_ratio=reduce_mean_scalar(float(prepared.filtering_ratio), dist_ctx),
        novelty_penalty_ratio=reduce_mean_scalar(float(prepared.novelty_penalty_ratio), dist_ctx),
        failure_counts=failure_counts,
        metrics=merged_metrics,
        reward_raw_mean=(reward_raw_sum / generated_count) if generated_count > 0 else 0.0,
        avg_chembl_reward=(chembl_reward_sum / valid_count) if valid_count > 0 else 0.0,
        avg_chembl_similarity=(chembl_similarity_sum / valid_count) if valid_count > 0 else 0.0,
        chembl_prior_metrics=chembl_prior_metrics,
    )


def record_epoch_ddp(
    runtime: DistributedRuntimeBundle,
    epoch: int,
    prepared: PreparedSamples,
    local_stats: BatchStats,
    global_stats: GlobalBatchStats,
) -> None:
    rank = int(runtime.dist_ctx.rank)
    batch_facts = build_batch_facts(prepared=prepared, epoch=epoch, batch_idx=0, batch_stats=local_stats, config=runtime.config)
    batch_facts.rank = rank
    for row in batch_facts.molecule_rows:
        row["rank"] = rank
    runtime.run_recorder.on_batch_end(batch_facts)

    # Write per-rank SDF with original 3D coordinates
    ntb = getattr(prepared.samples, "non_tensor_batch", None)
    if isinstance(ntb, dict):
        rdmols = ntb.get("rdmols")
        if rdmols is not None and len(rdmols) > 0:
            runtime.run_recorder.on_batch_sdf(batch_facts, rdmols)

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
            avg_docking=global_stats.avg_docking,
            avg_similarity=global_stats.avg_similarity,
            avg_diversity=global_stats.avg_diversity,
            avg_qed_norm=global_stats.avg_qed_norm,
            avg_sa_norm=global_stats.avg_sa_norm,
            avg_docking_norm=global_stats.avg_docking_norm,
            avg_diversity_norm=global_stats.avg_diversity_norm,
            generated_count=global_stats.generated_count,
            valid_count=global_stats.valid_count,
            valid_ratio=global_stats.valid_ratio,
            failure_counts=global_stats.failure_counts,
            reward_raw_mean=global_stats.reward_raw_mean,
            avg_chembl_reward=global_stats.avg_chembl_reward,
            avg_chembl_similarity=global_stats.avg_chembl_similarity,
            avg_chembl_prior_ecfp4_reward=(global_stats.chembl_prior_metrics or {}).get("avg_chembl_prior_ecfp4_reward", 0.0),
            avg_chembl_prior_ecfp4_similarity=(global_stats.chembl_prior_metrics or {}).get("avg_chembl_prior_ecfp4_similarity", 0.0),
            avg_chembl_prior_fcfp4_reward=(global_stats.chembl_prior_metrics or {}).get("avg_chembl_prior_fcfp4_reward", 0.0),
            avg_chembl_prior_fcfp4_similarity=(global_stats.chembl_prior_metrics or {}).get("avg_chembl_prior_fcfp4_similarity", 0.0),
            chembl_prior_metrics=global_stats.chembl_prior_metrics or {},
        ),
        prepared=PreparedSamples(
            samples=prepared.samples,
            filtering_ratio=global_stats.filtering_ratio,
            novelty_penalty_ratio=global_stats.novelty_penalty_ratio,
        ),
    )
    runtime.run_recorder.on_epoch_end(epoch_facts)


def run_epoch_ddp(runtime: DistributedRuntimeBundle, epoch: int) -> EpochStepResult:
    timer = PerfTimer() if bool(runtime.config.get("train", {}).get("timer", False)) else None
    prepared = prepare_training_samples(runtime, epoch, timer=timer)
    metrics = update_policy(runtime, prepared, epoch, timer=timer)
    maybe_log_ddp_probe(runtime, epoch)
    maybe_check_param_sync(runtime, epoch)
    # Merge timing metrics into the metrics dict for recording
    for step, elapsed in timer.summary().items():
        metrics[f"time_{step}"] = elapsed
    local_stats = compute_batch_stats(prepared, config=runtime.config)
    # Compute per-rank molecular descriptors for epoch-level monitoring;
    # _merge_metrics in reduce_batch_stats averages them across all ranks.
    descriptor_metrics = compute_epoch_molecular_descriptors(prepared)
    metrics.update(descriptor_metrics)
    global_stats = reduce_batch_stats(local_stats, prepared, metrics, runtime.dist_ctx)
    import sys; sys.stderr.flush()  # 刷新 stderr 确保 tqdm 残留完全写出后再打印 epoch 汇总
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
        avg_docking=global_stats.avg_docking,
        avg_similarity=global_stats.avg_similarity,
        generated_count=global_stats.generated_count,
        valid_count=global_stats.valid_count,
        valid_ratio=global_stats.valid_ratio,
        filtering_ratio=global_stats.filtering_ratio,
        novelty_penalty_ratio=global_stats.novelty_penalty_ratio,
        failure_counts=global_stats.failure_counts,
        avg_chembl_reward=global_stats.avg_chembl_reward,
        avg_chembl_similarity=global_stats.avg_chembl_similarity,
        avg_chembl_prior_ecfp4_reward=(global_stats.chembl_prior_metrics or {}).get("avg_chembl_prior_ecfp4_reward", 0.0),
        avg_chembl_prior_ecfp4_similarity=(global_stats.chembl_prior_metrics or {}).get("avg_chembl_prior_ecfp4_similarity", 0.0),
        avg_chembl_prior_fcfp4_reward=(global_stats.chembl_prior_metrics or {}).get("avg_chembl_prior_fcfp4_reward", 0.0),
        avg_chembl_prior_fcfp4_similarity=(global_stats.chembl_prior_metrics or {}).get("avg_chembl_prior_fcfp4_similarity", 0.0),
        chembl_prior_metrics=global_stats.chembl_prior_metrics or {},
    )


def run_training_loop_ddp(runtime: DistributedRuntimeBundle) -> None:
    if is_main_process(runtime.dist_ctx):
        runtime.run_recorder.on_run_started()
    if runtime.resumed_from_checkpoint:
        runtime.logger.info(
            "Starting native DDP training loop for epochs %d..%d (resumed from %s)",
            runtime.start_epoch + 1,
            runtime.max_epochs,
            runtime.resumed_from_checkpoint,
        )
    else:
        runtime.logger.info("Starting native DDP training loop for %d epochs", runtime.max_epochs)
    try:
        for epoch in range(runtime.start_epoch + 1, runtime.max_epochs + 1):
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
