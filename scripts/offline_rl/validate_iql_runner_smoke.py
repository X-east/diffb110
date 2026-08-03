#!/usr/bin/env python3
"""Validate real DiffSBDD IQL epoch recording and checkpoint resume on GPU."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from ddpo.integration.diffsbdd_model import DiffSBDDModel
from ddpo.offline_rl.critic_adapter import PocketContext
from ddpo.offline_rl.dataset import load_trajectory_shard
from ddpo.offline_rl.index import (
    EpisodeIndex,
    load_episode_index,
    load_split_manifest,
)
from ddpo.offline_rl.models import (
    GeometricCriticConfig,
    GeometricValueNetwork,
    TwinQCritic,
)
from ddpo.offline_rl.policy_adapter import DiffSBDDPolicyAdapter
from ddpo.offline_rl.recording import LocalIQLRecorder
from ddpo.offline_rl.reward_sidecar import (
    REWARD_SIDECAR_SCHEMA_VERSION,
    EpisodeRewardFacts,
    RewardSidecar,
)
from ddpo.offline_rl.runner import IQLRunnerConfig, OfflineIQLRunner
from ddpo.offline_rl.sampler import DecisionTimeBin, OfflineTransitionSampler
from ddpo.offline_rl.trainer import (
    ActorUpdateConfig,
    IQLTrainerConfig,
    OfflineActorUpdater,
    OfflineIQLTrainer,
)


def _parse_args() -> argparse.Namespace:
    """Parse paths and bounded smoke controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--episode-index", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pocket-pdb", type=Path, required=True)
    parser.add_argument("--ref-ligand", default="C:1101")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps-per-epoch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--trainable-parameter-tensors", type=int, default=2)
    return parser.parse_args()


def _build_contract_probe_sidecar(
    run_dir: Path,
    episode_index: EpisodeIndex,
) -> RewardSidecar:
    """Build immutable +/-1 validity rewards for engineering verification."""
    rewards: list[EpisodeRewardFacts] = []
    shard_rows: dict[tuple[str, str], None] = {}
    for row in episode_index.rows:
        shard_rows[(row.shard_path, row.source_shard_sha256)] = None
    for shard_path, shard_sha256 in shard_rows:
        shard = load_trajectory_shard(
            run_dir / shard_path,
            expected_sha256=shard_sha256,
        )
        source_digest = shard.source_sha256()
        for facts in shard.all_episode_facts():
            rewards.append(
                EpisodeRewardFacts(
                    sample_id=facts.sample_id,
                    source_trajectory_format=(
                        "diffsbdd_frozen_policy_trajectory_v1"
                    ),
                    source_checkpoint_sha256=facts.checkpoint_sha256,
                    source_shard_sha256=source_digest,
                    reward_schema_version=REWARD_SIDECAR_SCHEMA_VERSION,
                    reward_version=(
                        "validity_contract_probe_not_formal_reward_v1"
                    ),
                    reconstruction_success=facts.rdkit_valid,
                    sanitize_success=facts.rdkit_valid,
                    rdkit_valid=facts.rdkit_valid,
                    pb_evaluated=False,
                    pb_valid=False,
                    docking_input_success=False,
                    docking_success=False,
                    actual_good=False,
                    reward_total_train=(
                        1.0 if facts.rdkit_valid else -1.0
                    ),
                    failure_reason=facts.failure_type,
                )
            )
    return RewardSidecar(rewards)


def _select_trainable_policy_parameters(
    model: DiffSBDDModel,
    count: int,
) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    """Freeze policy weights and reopen its last dynamics tensors."""
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


def _build_runtime(
    args: argparse.Namespace,
    *,
    episode_index: EpisodeIndex,
    split_manifest,
    sidecar: RewardSidecar,
) -> tuple[
    OfflineIQLTrainer,
    OfflineTransitionSampler,
    tuple[tuple[str, torch.nn.Parameter], ...],
]:
    """Build a fresh real policy, geometric critics, trainer, and sampler."""
    device = torch.device(args.device)
    policy = DiffSBDDModel(
        checkpoint_path=str(args.checkpoint),
        pocket_pdb_path=str(args.pocket_pdb),
        ref_ligand_path=args.ref_ligand,
        device=str(device),
        logprob_mode="mean_per_atom",
    )
    policy.train()
    selected = _select_trainable_policy_parameters(
        policy,
        args.trainable_parameter_tensors,
    )
    pocket = PocketContext.from_diffsbdd_mapping(
        policy.export_single_pocket_context()
    ).to(device)
    first_row = episode_index.rows[0]
    probe_shard = load_trajectory_shard(
        args.run_dir / first_row.shard_path,
        expected_sha256=first_row.source_shard_sha256,
    )
    ligand_feature_dim = int(probe_shard.latents.shape[-1]) - 3
    critic_config = GeometricCriticConfig(
        ligand_feature_dim=ligand_feature_dim,
        pocket_feature_dim=int(pocket.features.shape[-1]),
        hidden_dim=32,
        num_message_layers=1,
        num_rbf=8,
        time_embedding_dim=16,
        max_neighbors=16,
    )
    online_q = TwinQCritic(critic_config).to(device)
    value = GeometricValueNetwork(critic_config).to(device)
    trainer = OfflineIQLTrainer(
        online_q=online_q,
        value=value,
        q_optimizer=torch.optim.Adam(online_q.parameters(), lr=1e-3),
        value_optimizer=torch.optim.Adam(value.parameters(), lr=1e-3),
        actor_updater=OfflineActorUpdater(
            policy_adapter=DiffSBDDPolicyAdapter(
                policy,
                num_diffusion_steps=first_row.num_diffusion_steps,
            ),
            optimizer=torch.optim.Adam(
                [parameter for _, parameter in selected],
                lr=1e-6,
            ),
            config=ActorUpdateConfig(
                objective="iql",
                temperature=1.0,
                max_weight=10.0,
                max_grad_norm=5.0,
            ),
        ),
        pocket=pocket,
        config=IQLTrainerConfig(
            num_diffusion_steps=first_row.num_diffusion_steps,
            discount=1.0,
            expectile=0.7,
            target_tau=0.005,
            q_loss_type="mse",
            critic_max_grad_norm=5.0,
        ),
    )
    sampler = OfflineTransitionSampler(
        run_dir=args.run_dir,
        episode_index=episode_index,
        split_manifest=split_manifest,
        split="train",
        time_bins=(
            DecisionTimeBin(
                first_row.num_diffusion_steps,
                first_row.num_diffusion_steps,
            ),
        ),
        seed=args.seed,
        reward_sidecar=sidecar,
        cache_size=2,
    )
    return trainer, sampler, selected


def _read_metrics(path: Path) -> list[dict[str, str]]:
    """Read local step metrics emitted by the real runner."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _plot_losses(rows: list[dict[str, str]], output_path: Path) -> None:
    """Plot presentation-readable real-runner loss traces."""
    steps = [int(row["global_step"]) for row in rows]
    figure, axis = plt.subplots(figsize=(10, 5.5))
    for field, label in (
        ("value_loss", "V loss"),
        ("q_loss", "Twin-Q loss"),
        ("actor_loss", "Actor loss"),
    ):
        axis.plot(
            steps,
            [float(row[field]) for row in rows],
            marker="o",
            linewidth=2.5,
            label=label,
        )
    axis.set_title("Real DiffSBDD IQL runner and resume smoke", fontsize=18)
    axis.set_xlabel("Global optimization step", fontsize=15)
    axis.set_ylabel("Loss", fontsize=15)
    axis.tick_params(labelsize=13)
    axis.legend(fontsize=13)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    """Run epoch zero, rebuild all runtime objects, and resume epoch one."""
    args = _parse_args()
    if args.batch_size <= 0 or args.steps_per_epoch <= 0:
        raise ValueError("batch-size and steps-per-epoch must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but torch.cuda.is_available() is false"
        )
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)

    episode_index = load_episode_index(args.episode_index)
    split_manifest = load_split_manifest(args.split_manifest)
    sidecar = _build_contract_probe_sidecar(args.run_dir, episode_index)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    trainer, sampler, selected = _build_runtime(
        args,
        episode_index=episode_index,
        split_manifest=split_manifest,
        sidecar=sidecar,
    )
    parameter_before = {
        name: parameter.detach().clone() for name, parameter in selected
    }
    first_recorder = LocalIQLRecorder(args.output_dir, rank=0)
    first_runner = OfflineIQLRunner(
        trainer=trainer,
        sampler=sampler,
        recorder=first_recorder,
        config=IQLRunnerConfig(
            epochs=1,
            steps_per_epoch=args.steps_per_epoch,
            batch_size=args.batch_size,
            device=str(device),
            log_every_steps=1,
        ),
    )
    try:
        first_runner.run()
    finally:
        first_recorder.close()
    parameter_delta = max(
        float(
            torch.max(
                torch.abs(parameter.detach() - parameter_before[name])
            ).cpu()
        )
        for name, parameter in selected
    )
    if parameter_delta <= 0:
        raise RuntimeError("real DiffSBDD actor parameters did not change")
    checkpoint = (
        args.output_dir / "checkpoints" / "epoch_0000_rank_0000.pt"
    )
    del first_runner, first_recorder, trainer, sampler, selected
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    resumed_trainer, resumed_sampler, resumed_selected = _build_runtime(
        args,
        episode_index=episode_index,
        split_manifest=split_manifest,
        sidecar=sidecar,
    )
    resumed_recorder = LocalIQLRecorder(args.output_dir, rank=0)
    resumed_runner = OfflineIQLRunner(
        trainer=resumed_trainer,
        sampler=resumed_sampler,
        recorder=resumed_recorder,
        config=IQLRunnerConfig(
            epochs=2,
            steps_per_epoch=args.steps_per_epoch,
            batch_size=args.batch_size,
            device=str(device),
            log_every_steps=1,
        ),
    )
    try:
        resumed_runner.run(resume_checkpoint=checkpoint)
    finally:
        resumed_recorder.close()

    rows = _read_metrics(args.output_dir / "metrics" / "metrics.csv")
    expected_steps = 2 * args.steps_per_epoch
    observed_steps = [int(row["global_step"]) for row in rows]
    if observed_steps != list(range(expected_steps)):
        raise RuntimeError(
            f"unexpected resumed global steps: {observed_steps}"
        )
    checkpoint_paths = sorted(
        (args.output_dir / "checkpoints").glob("*.pt")
    )
    if len(checkpoint_paths) != 2:
        raise RuntimeError("runner smoke requires exactly two checkpoints")
    status = json.loads(
        (args.output_dir / "status.json").read_text(encoding="utf-8")
    )
    if status.get("status") != "completed":
        raise RuntimeError(f"runner status is not completed: {status}")
    _plot_losses(rows, args.output_dir / "iql_runner_resume_losses.png")

    summary = {
        "status": "passed",
        "scope": (
            "engineering_smoke_proxy_reward_not_scientific_validation"
        ),
        "formal_training": False,
        "reward_version": sidecar.reward_version,
        "device": str(device),
        "seed": args.seed,
        "eligible_train_episodes": resumed_sampler.num_eligible_episodes,
        "eligible_train_shards": resumed_sampler.num_eligible_shards,
        "batch_size": args.batch_size,
        "steps_per_epoch": args.steps_per_epoch,
        "epochs_completed": 2,
        "global_steps": expected_steps,
        "checkpoint_count": len(checkpoint_paths),
        "final_status": status,
        "policy_parameter_delta_after_first_epoch": parameter_delta,
        "trainable_policy_parameter_names": [
            name for name, _ in resumed_selected
        ],
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
        "warning": (
            "Validity +/-1 is only an interface probe. This validates runner "
            "recording and exact checkpoint restoration, not a formal reward "
            "or molecule-quality improvement."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
