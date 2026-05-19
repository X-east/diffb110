from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import torch

from ddpo.rewards.factory import get_enabled_rewards
from ddpo.runtime.recording import BatchFacts, EpochFacts, RunLifecycleFacts, RunStatus
from ddpo.utils import annotate_batch_molecule_semantics


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
    avg_docking: float
    avg_similarity: float
    generated_count: int
    valid_count: int
    valid_ratio: float
    filtering_ratio: float
    novelty_penalty_ratio: float
    failure_counts: dict[str, int]
    avg_chembl_reward: float = 0.0
    avg_chembl_similarity: float = 0.0


@dataclass
class BatchStats:
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
    failure_counts: dict[str, int]
    reward_raw_mean: float = 0.0
    avg_chembl_reward: float = 0.0
    avg_chembl_similarity: float = 0.0


def prepare_training_samples(runtime: Any, epoch: int) -> PreparedSamples:
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
    reward_normalizer = getattr(runtime, "reward_normalizer", None)
    if reward_normalizer is not None:
        logger.info("[Epoch %d/%d] reward_normalizer.apply", epoch, runtime.max_epochs)
        dist_ctx = getattr(runtime, "dist_ctx", None)
        samples = reward_normalizer.apply(samples, dist_ctx=dist_ctx)
    if "advantages" not in samples.batch:
        logger.info("[Epoch %d/%d] advantages = reward_total", epoch, runtime.max_epochs)
        samples.batch["advantages"] = samples.batch.get("reward_total_clamped", samples.batch.get("reward_total"))
    return PreparedSamples(samples=samples, filtering_ratio=filtering_ratio, novelty_penalty_ratio=novelty_penalty_ratio)


def update_policy(runtime: Any, prepared: PreparedSamples, epoch: int) -> dict[str, Any]:
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


def _mean_at_indices(values: list[float], indices: list[int]) -> float:
    selected = [values[i] for i in indices if i < len(values)]
    return float(sum(selected) / len(selected)) if selected else 0.0


def compute_batch_stats(prepared: PreparedSamples) -> BatchStats:
    samples = prepared.samples
    ntb = getattr(samples, "non_tensor_batch", None) if hasattr(samples, "non_tensor_batch") else None
    src = ntb if isinstance(ntb, dict) else samples.batch
    rewards = _to_float_list(samples.batch.get("reward_total_clamped", samples.batch.get("reward_total")))
    # 原始总奖励（归一化前），用于终端展示
    rewards_raw = _to_float_list(samples.batch.get("reward_total_raw"))
    # Raw values (for terminal display)
    qed_raw = _to_float_list(samples.batch.get("qed_raw"))
    sa_raw = _to_float_list(samples.batch.get("sa_raw"))
    docking_raw = _to_float_list(samples.batch.get("docking_raw"))
    similarity_raw = _to_float_list(samples.batch.get("similarity_raw"))
    diversity_raw = _to_float_list(samples.batch.get("diversity_raw"))
    # Normalised values (for recording)
    qed_norm = _to_float_list(samples.batch.get("qed_reward"))
    sa_norm = _to_float_list(samples.batch.get("sa_reward"))
    docking_norm = _to_float_list(samples.batch.get("docking_reward"))
    diversity_norm = _to_float_list(samples.batch.get("diversity_reward"))
    valid_flags = _to_int_list(src.get("rdkit_valid"))
    failure_types = _to_str_list(src.get("failure_type"))
    reward_mean = float(sum(rewards) / len(rewards)) if rewards else 0.0
    reward_min = float(min(rewards)) if rewards else 0.0
    reward_max = float(max(rewards)) if rewards else 0.0
    # ------------------------------------------------------------------
    # Raw property averages: compute over VALID molecules only.
    # Invalid molecules store 0.0 in qed_raw/sa_raw/docking_raw (see
    # SAReward/QEDReward/DockingReward), which would drag the average down
    # as a pure statistical artifact of validity rate.
    # ------------------------------------------------------------------
    _valid_indices = [i for i, f in enumerate(valid_flags) if f] if valid_flags else []
    avg_qed = _mean_at_indices(qed_raw, _valid_indices)
    avg_sa = _mean_at_indices(sa_raw, _valid_indices)
    avg_docking = _mean_at_indices(docking_raw, _valid_indices)
    avg_similarity = _mean_at_indices(similarity_raw, _valid_indices)
    avg_diversity = _mean_at_indices(diversity_raw, _valid_indices)
    # ChEMBL prior: dynamically detect top_k field name
    chembl_prior_reward = _to_float_list(samples.batch.get("chembl_prior_reward"))
    chembl_sim_field = next(
        (k for k in samples.batch.keys() if k.startswith("chembl_top") and k.endswith("_mean_similarity")),
        None,
    )
    chembl_similarity = _to_float_list(samples.batch.get(chembl_sim_field)) if chembl_sim_field else []
    avg_chembl_reward = _mean_at_indices(chembl_prior_reward, _valid_indices)
    avg_chembl_similarity = _mean_at_indices(chembl_similarity, _valid_indices)
    # Normalised reward averages: include ALL molecules (failures get explicit
    # failure_penalty, which is meaningful for reward-based monitoring).
    avg_qed_norm = float(sum(qed_norm) / len(qed_norm)) if qed_norm else 0.0
    avg_sa_norm = float(sum(sa_norm) / len(sa_norm)) if sa_norm else 0.0
    avg_docking_norm = float(sum(docking_norm) / len(docking_norm)) if docking_norm else 0.0
    avg_diversity_norm = float(sum(diversity_norm) / len(diversity_norm)) if diversity_norm else 0.0
    generated_count = len(valid_flags) if valid_flags else int(samples.batch.batch_size[0])
    valid_count = int(sum(valid_flags)) if valid_flags else 0
    valid_ratio = float(valid_count / generated_count) if generated_count > 0 else 0.0
    failure_counts = dict(Counter(failure_types)) if failure_types else {}
    reward_raw_mean = float(sum(rewards_raw) / len(rewards_raw)) if rewards_raw else 0.0
    return BatchStats(reward_mean=reward_mean, reward_min=reward_min, reward_max=reward_max, avg_qed=avg_qed, avg_sa=avg_sa, avg_docking=avg_docking, avg_similarity=avg_similarity, avg_diversity=avg_diversity, avg_qed_norm=avg_qed_norm, avg_sa_norm=avg_sa_norm, avg_docking_norm=avg_docking_norm, avg_diversity_norm=avg_diversity_norm, avg_chembl_reward=avg_chembl_reward, avg_chembl_similarity=avg_chembl_similarity, generated_count=generated_count, valid_count=valid_count, valid_ratio=valid_ratio, failure_counts=failure_counts, reward_raw_mean=reward_raw_mean)


def build_batch_facts(prepared: PreparedSamples, epoch: int, batch_idx: int, batch_stats: BatchStats) -> BatchFacts:
    samples = prepared.samples
    ntb = getattr(samples, "non_tensor_batch", None) if hasattr(samples, "non_tensor_batch") else None
    src = ntb if isinstance(ntb, dict) else samples.batch
    smiles_values = list(src.get("smiles", []))
    valid_flags = _to_int_list(src.get("rdkit_valid"))
    # Raw values
    qed_raw = _to_float_list(samples.batch.get("qed_raw"))
    sa_raw = _to_float_list(samples.batch.get("sa_raw"))
    docking_raw = _to_float_list(samples.batch.get("docking_raw"))
    similarity_raw = _to_float_list(samples.batch.get("similarity_raw"))
    diversity_raw = _to_float_list(samples.batch.get("diversity_raw"))
    # ChEMBL prior: dynamically detect top_k field name
    chembl_prior_reward = _to_float_list(samples.batch.get("chembl_prior_reward"))
    chembl_sim_field = next(
        (k for k in samples.batch.keys() if k.startswith("chembl_top") and k.endswith("_mean_similarity")),
        None,
    )
    chembl_similarity = _to_float_list(samples.batch.get(chembl_sim_field)) if chembl_sim_field else []
    # Normalised (reward) values
    qed_reward = _to_float_list(samples.batch.get("qed_reward"))
    sa_reward = _to_float_list(samples.batch.get("sa_reward"))
    docking_reward = _to_float_list(samples.batch.get("docking_reward"))
    diversity_reward = _to_float_list(samples.batch.get("diversity_reward"))
    # Total rewards
    rewards = _to_float_list(samples.batch.get("reward_total_clamped", samples.batch.get("reward_total")))
    rewards_raw = _to_float_list(samples.batch.get("reward_total_raw"))
    rewards_norm = _to_float_list(samples.batch.get("reward_total_norm"))
    molecule_rows = []
    for molecule_id in range(len(smiles_values)):
        molecule_rows.append({
            "epoch": epoch,
            "batch_idx": batch_idx,
            "rank": 0,
            "molecule_id": molecule_id,
            "smiles": smiles_values[molecule_id] if molecule_id < len(smiles_values) else "",
            "rdkit_valid": valid_flags[molecule_id] if molecule_id < len(valid_flags) else 0,
            "qed_raw": qed_raw[molecule_id] if molecule_id < len(qed_raw) else 0.0,
            "sa_raw": sa_raw[molecule_id] if molecule_id < len(sa_raw) else 0.0,
            "docking_raw": docking_raw[molecule_id] if molecule_id < len(docking_raw) else 0.0,
            "similarity_raw": similarity_raw[molecule_id] if molecule_id < len(similarity_raw) else 0.0,
            "diversity_raw": diversity_raw[molecule_id] if molecule_id < len(diversity_raw) else 0.0,
            "qed_reward": qed_reward[molecule_id] if molecule_id < len(qed_reward) else 0.0,
            "sa_reward": sa_reward[molecule_id] if molecule_id < len(sa_reward) else 0.0,
            "docking_reward": docking_reward[molecule_id] if molecule_id < len(docking_reward) else 0.0,
            "diversity_reward": diversity_reward[molecule_id] if molecule_id < len(diversity_reward) else 0.0,
            "chembl_prior_reward": chembl_prior_reward[molecule_id] if molecule_id < len(chembl_prior_reward) else 0.0,
            "chembl_mean_similarity": chembl_similarity[molecule_id] if molecule_id < len(chembl_similarity) else 0.0,
            "reward_total": rewards[molecule_id] if molecule_id < len(rewards) else 0.0,
            "reward_total_raw": rewards_raw[molecule_id] if molecule_id < len(rewards_raw) else 0.0,
            "reward_total_norm": rewards_norm[molecule_id] if molecule_id < len(rewards_norm) else 0.0,
        })
    return BatchFacts(epoch=epoch, batch_idx=batch_idx, rank=0, generated_count=batch_stats.generated_count, valid_count=batch_stats.valid_count, reward_mean=batch_stats.reward_mean, reward_min=batch_stats.reward_min, reward_max=batch_stats.reward_max, avg_qed=batch_stats.avg_qed, avg_sa=batch_stats.avg_sa, avg_docking=batch_stats.avg_docking, avg_similarity=batch_stats.avg_similarity, avg_diversity=batch_stats.avg_diversity, filtering_ratio=prepared.filtering_ratio, novelty_penalty_ratio=prepared.novelty_penalty_ratio, avg_qed_norm=batch_stats.avg_qed_norm, avg_sa_norm=batch_stats.avg_sa_norm, avg_docking_norm=batch_stats.avg_docking_norm, avg_diversity_norm=batch_stats.avg_diversity_norm, avg_chembl_reward=batch_stats.avg_chembl_reward, avg_chembl_similarity=batch_stats.avg_chembl_similarity, failure_counts=batch_stats.failure_counts, molecule_rows=molecule_rows)


# Metrics keys already consumed by EpochFacts' own fields — excluded from generic passthrough.
_CONSUMED_METRICS_KEYS = frozenset({
    "Loss", "loss", "ppo_loss", "ClipFrac", "PPO_Epoch", "PPO_Epochs",
})


def build_epoch_facts(runtime: Any, epoch: int, metrics: dict[str, Any], batch_stats: BatchStats, prepared: PreparedSamples) -> EpochFacts:
    avg_loss = float(metrics.get("Loss", metrics.get("loss", metrics.get("ppo_loss", 0.0))))
    checkpoint_due = (epoch % runtime.checkpoint_interval) == 0
    extra = {
        "max_epochs": runtime.max_epochs,
        "avg_qed_norm": batch_stats.avg_qed_norm,
        "avg_sa_norm": batch_stats.avg_sa_norm,
        "avg_docking_norm": batch_stats.avg_docking_norm,
        "avg_diversity_norm": batch_stats.avg_diversity_norm,
        "avg_similarity": batch_stats.avg_similarity,
        "avg_chembl_reward": batch_stats.avg_chembl_reward,
        "avg_chembl_similarity": batch_stats.avg_chembl_similarity,
        "avg_reward_raw": batch_stats.reward_raw_mean,
        "reward_enabled": get_enabled_rewards(runtime.config),
    }
    # Generic passthrough: Actor returns whatever metrics it produces;
    # unconsumed keys (e.g. KL_Penalty, PPO_Loss) automatically flow
    # into extra_metrics with their original key names.  No downstream
    # code should hard-code any specific key.
    for key, value in metrics.items():
        if key not in _CONSUMED_METRICS_KEYS:
            try:
                extra[key] = float(value)
            except (TypeError, ValueError):
                extra[key] = str(value) if not isinstance(value, (str, bool)) else value
    return EpochFacts(epoch=epoch, timestamp=datetime.now().isoformat(timespec="seconds"), avg_loss=avg_loss, reward_mean=batch_stats.reward_mean, reward_min=batch_stats.reward_min, reward_max=batch_stats.reward_max, avg_qed=batch_stats.avg_qed, avg_sa=batch_stats.avg_sa, avg_docking=batch_stats.avg_docking, avg_similarity=batch_stats.avg_similarity, avg_diversity=batch_stats.avg_diversity, generated_total=batch_stats.generated_count, rdkit_valid=batch_stats.valid_count, rdkit_valid_ratio=batch_stats.valid_ratio, filtering_ratio_mean=prepared.filtering_ratio, novelty_penalty_ratio_mean=prepared.novelty_penalty_ratio, avg_chembl_reward=batch_stats.avg_chembl_reward, avg_chembl_similarity=batch_stats.avg_chembl_similarity, failure_counts=batch_stats.failure_counts, extra_metrics=extra, checkpoint_due=checkpoint_due)


def build_run_lifecycle(runtime: Any, status: RunStatus, message: str | None = None) -> RunLifecycleFacts:
    return RunLifecycleFacts(run_id=runtime.run_context.run_id, run_name=runtime.run_context.run_name, output_dir=runtime.run_context.output_dir, status=status, started_at=runtime.run_context.started_at, ended_at=datetime.now().isoformat(timespec="seconds"), message=message)


def finish_run(runtime: Any, status: RunStatus, message: str | None = None) -> None:
    lifecycle = build_run_lifecycle(runtime, status=status, message=message)
    extra_lines = [f"Output Dir: {runtime.run_context.output_dir}"]
    if message:
        extra_lines.append(f"Message   : {message}")
    runtime.run_recorder.on_run_finished(lifecycle, extra_lines=extra_lines)
