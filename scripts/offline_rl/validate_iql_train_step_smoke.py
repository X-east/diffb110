#!/usr/bin/env python3
"""Run a real DiffSBDD policy + shard IQL engineering smoke on one GPU."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from ddpo.integration.diffsbdd_model import DiffSBDDModel
from ddpo.offline_rl.critic_adapter import PocketContext
from ddpo.offline_rl.dataset import load_trajectory_shard
from ddpo.offline_rl.models import (
    GeometricCriticConfig,
    GeometricValueNetwork,
    TwinQCritic,
)
from ddpo.offline_rl.policy_adapter import DiffSBDDPolicyAdapter
from ddpo.offline_rl.reward_sidecar import (
    REWARD_SIDECAR_SCHEMA_VERSION,
    EpisodeRewardFacts,
    RewardSidecar,
)
from ddpo.offline_rl.trainer import (
    ActorUpdateConfig,
    IQLTrainerConfig,
    OfflineActorUpdater,
    OfflineIQLTrainer,
)


LOGGER = logging.getLogger("RealIQLTrainStepSmoke")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pocket-pdb", type=Path, required=True)
    parser.add_argument("--ref-ligand", default="C:1101")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--trainable-parameter-tensors", type=int, default=2)
    return parser.parse_args()


def _configure_logging(output_dir: Path) -> None:
    """Configure durable file and terminal logging."""
    output_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    for handler in (
        logging.FileHandler(output_dir / "iql_train_step_smoke.log", encoding="utf-8"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)


def _build_contract_probe_sidecar(
    shard,
) -> RewardSidecar:
    """Build finite +/-1 labels solely for an engineering train-step smoke."""
    digest = shard.source_sha256()
    return RewardSidecar(
        EpisodeRewardFacts(
            sample_id=facts.sample_id,
            source_trajectory_format="diffsbdd_frozen_policy_trajectory_v1",
            source_checkpoint_sha256=facts.checkpoint_sha256,
            source_shard_sha256=digest,
            reward_schema_version=REWARD_SIDECAR_SCHEMA_VERSION,
            reward_version="validity_contract_probe_not_formal_reward_v1",
            reconstruction_success=facts.rdkit_valid,
            sanitize_success=facts.rdkit_valid,
            rdkit_valid=facts.rdkit_valid,
            pb_evaluated=False,
            pb_valid=False,
            docking_input_success=False,
            docking_success=False,
            actual_good=False,
            reward_total_train=1.0 if facts.rdkit_valid else -1.0,
            failure_reason=facts.failure_type,
        )
        for facts in shard.all_episode_facts()
    )


def _select_trainable_policy_parameters(
    model: DiffSBDDModel,
    count: int,
) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    """Freeze the policy and reopen the last dynamics parameter tensors."""
    if count <= 0:
        raise ValueError("trainable-parameter-tensors must be positive")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    candidates = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if "dynamics" in name
    ]
    if len(candidates) < count:
        raise RuntimeError(
            f"requested {count} dynamics tensors but found {len(candidates)}"
        )
    selected = tuple(candidates[-count:])
    for _, parameter in selected:
        parameter.requires_grad_(True)
    return selected


def _flatten_metrics(step: int, metrics) -> dict[str, object]:
    """Flatten nested actor metrics for CSV output."""
    row: dict[str, object] = {"step": step}
    payload = asdict(metrics)
    actor = payload.pop("actor")
    row.update(payload)
    if actor is not None:
        row.update({f"actor_{key}": value for key, value in actor.items()})
    return row


def main() -> None:
    """Execute the real-policy engineering smoke and write local evidence."""
    args = _parse_args()
    if args.batch_size <= 0 or args.steps <= 0:
        raise ValueError("batch-size and steps must be positive")
    _configure_logging(args.output_dir)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    LOGGER.info(
        "loading shard=%s checkpoint=%s device=%s",
        args.shard,
        args.checkpoint,
        device,
    )
    shard = load_trajectory_shard(args.shard, validate_full_tensors=True)
    if args.batch_size > shard.batch_size:
        raise ValueError("batch-size exceeds shard episode count")
    sidecar = _build_contract_probe_sidecar(shard)
    joined = sidecar.join_shard(shard)
    episode_indices = list(range(args.batch_size))
    transitions = joined.select_transitions(
        episode_indices,
        [shard.final_decision_index] * args.batch_size,
    ).to(device)

    policy = DiffSBDDModel(
        checkpoint_path=str(args.checkpoint),
        pocket_pdb_path=str(args.pocket_pdb),
        ref_ligand_path=args.ref_ligand,
        device=str(device),
        logprob_mode="mean_per_atom",
    )
    policy.train()
    selected_parameters = _select_trainable_policy_parameters(
        policy,
        args.trainable_parameter_tensors,
    )
    policy_before = {
        name: parameter.detach().clone()
        for name, parameter in selected_parameters
    }
    pocket = PocketContext.from_diffsbdd_mapping(
        policy.export_single_pocket_context()
    ).to(device)
    critic_config = GeometricCriticConfig(
        ligand_feature_dim=int(transitions.state_latent.shape[-1]) - 3,
        pocket_feature_dim=int(pocket.features.shape[-1]),
        hidden_dim=32,
        num_message_layers=1,
        num_rbf=8,
        time_embedding_dim=16,
        max_neighbors=16,
    )
    online_q = TwinQCritic(critic_config).to(device)
    value = GeometricValueNetwork(critic_config).to(device)
    actor_updater = OfflineActorUpdater(
        policy_adapter=DiffSBDDPolicyAdapter(
            policy,
            num_diffusion_steps=shard.num_diffusion_steps,
        ),
        optimizer=torch.optim.Adam(
            [parameter for _, parameter in selected_parameters],
            lr=1e-6,
        ),
        config=ActorUpdateConfig(
            objective="iql",
            temperature=1.0,
            max_weight=10.0,
            max_grad_norm=5.0,
        ),
    )
    trainer = OfflineIQLTrainer(
        online_q=online_q,
        value=value,
        q_optimizer=torch.optim.Adam(online_q.parameters(), lr=1e-3),
        value_optimizer=torch.optim.Adam(value.parameters(), lr=1e-3),
        actor_updater=actor_updater,
        pocket=pocket,
        config=IQLTrainerConfig(
            num_diffusion_steps=shard.num_diffusion_steps,
            discount=1.0,
            expectile=0.7,
            target_tau=0.005,
            q_loss_type="mse",
            critic_max_grad_norm=5.0,
        ),
    )

    metric_rows: list[dict[str, object]] = []
    for step in range(args.steps):
        metrics = trainer.step(transitions)
        metric_rows.append(_flatten_metrics(step, metrics))
        LOGGER.info(
            "step=%d value_loss=%.6f q_loss=%.6f actor_loss=%.6f "
            "advantage_mean=%.6f",
            step,
            metrics.value_loss,
            metrics.q_loss,
            0.0 if metrics.actor is None else metrics.actor.loss,
            metrics.advantage_mean,
        )

    parameter_delta_max = max(
        float(
            torch.max(
                torch.abs(parameter.detach() - policy_before[name])
            ).cpu()
        )
        for name, parameter in selected_parameters
    )
    if parameter_delta_max <= 0:
        raise RuntimeError("real DiffSBDD actor parameters did not change")
    metrics_path = args.output_dir / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)

    elapsed = time.perf_counter() - started
    peak_cuda = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
    summary = {
        "status": "passed",
        "scope": "engineering_smoke_proxy_reward_not_scientific_validation",
        "formal_training": False,
        "reward_version": sidecar.reward_version,
        "shard": str(args.shard.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "decision_index": shard.final_decision_index,
        "valid_count_in_batch": sum(
            shard.episode_facts(index).rdkit_valid
            for index in episode_indices
        ),
        "trainable_policy_parameter_names": [
            name for name, _ in selected_parameters
        ],
        "trainable_policy_parameter_count": sum(
            parameter.numel() for _, parameter in selected_parameters
        ),
        "policy_parameter_delta_max": parameter_delta_max,
        "final_metrics": metric_rows[-1],
        "elapsed_seconds": elapsed,
        "peak_cuda_allocated_bytes": peak_cuda,
        "warning": (
            "The +/-1 validity label is only an interface probe. This run does "
            "not validate the formal reward, held-out critic quality, or "
            "molecule-generation improvement."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    figure, axis = plt.subplots(figsize=(10, 5.5))
    steps = [int(row["step"]) for row in metric_rows]
    axis.plot(
        steps,
        [float(row["value_loss"]) for row in metric_rows],
        marker="o",
        linewidth=2.5,
        label="V loss",
    )
    axis.plot(
        steps,
        [float(row["q_loss"]) for row in metric_rows],
        marker="o",
        linewidth=2.5,
        label="Twin-Q loss",
    )
    axis.plot(
        steps,
        [float(row["actor_loss"]) for row in metric_rows],
        marker="o",
        linewidth=2.5,
        label="Actor loss",
    )
    axis.set_title("Real DiffSBDD IQL train-step smoke", fontsize=18)
    axis.set_xlabel("Optimization step", fontsize=15)
    axis.set_ylabel("Loss", fontsize=15)
    axis.tick_params(labelsize=13)
    axis.legend(fontsize=13)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(args.output_dir / "iql_train_step_losses.png", dpi=180)
    plt.close(figure)
    LOGGER.info(
        "passed parameter_delta_max=%.8g elapsed=%.2fs peak_cuda_bytes=%d",
        parameter_delta_max,
        elapsed,
        peak_cuda,
    )


if __name__ == "__main__":
    main()
