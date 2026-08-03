"""Configuration-driven construction for DiffSBDD offline-IQL training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from ddpo.integration.diffsbdd_model import DiffSBDDModel
from ddpo.runtime import DistributedContext

from .critic_adapter import PocketContext
from .dataset import load_trajectory_shard
from .index import EpisodeIndex, SplitManifest, load_episode_index, load_split_manifest
from .models import GeometricCriticConfig, GeometricValueNetwork, TwinQCritic
from .policy_adapter import DiffSBDDPolicyAdapter
from .reward_sidecar import (
    REWARD_SIDECAR_SCHEMA_VERSION,
    EpisodeRewardFacts,
    RewardSidecar,
    load_reward_sidecar,
)
from .runner import IQLRunnerConfig
from .sampler import (
    DecisionTimeBin,
    OfflineTransitionSampler,
    SharedDecisionIndexScheduler,
)
from .trainer import (
    ActorUpdateConfig,
    IQLTrainerConfig,
    OfflineActorUpdater,
    OfflineIQLTrainer,
)


@dataclass(frozen=True)
class OfflineIQLRuntimeBundle:
    """Fully built rank-local runtime and immutable run identity."""

    trainer: OfflineIQLTrainer
    sampler: OfflineTransitionSampler
    runner_config: IQLRunnerConfig
    scheduler: SharedDecisionIndexScheduler
    run_identity: dict[str, Any]
    trainable_policy_parameter_names: tuple[str, ...]
    formal_training: bool
    reward_version: str


def load_offline_iql_config(path: str | Path) -> dict[str, Any]:
    """Load one mapping-valued YAML configuration."""
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("offline IQL config must contain a YAML mapping")
    return payload


def resolve_project_path(value: str | Path, project_root: Path) -> Path:
    """Resolve a configured path relative to the project checkout."""
    path = Path(value).expanduser()
    return (path if path.is_absolute() else project_root / path).resolve()


def sha256_file(path: str | Path) -> str:
    """Return the SHA256 digest of one required input file."""
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_sha256(module: nn.Module) -> str:
    """Hash a bare or DDP-wrapped module state for rank synchronization checks."""
    core = module.module if isinstance(module, DDP) else module
    digest = hashlib.sha256()
    for name, tensor in sorted(core.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    """Return one required mapping section with a precise error."""
    value = config.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"config section {key!r} must be a mapping")
    return value


def _select_trainable_policy_parameters(
    model: DiffSBDDModel,
    count: int,
) -> tuple[tuple[str, nn.Parameter], ...]:
    """Freeze the policy and reopen the configured final dynamics tensors."""
    if count <= 0:
        raise ValueError("trainable_parameter_tensors must be positive")
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


def _build_diagnostic_sidecar(
    run_dir: Path,
    episode_index: EpisodeIndex,
) -> RewardSidecar:
    """Build the explicit validity probe used only by engineering smoke tests."""
    rows: list[EpisodeRewardFacts] = []
    unique_shards = sorted(
        {
            (row.shard_path, row.source_shard_sha256)
            for row in episode_index.rows
        }
    )
    for shard_path, shard_sha256 in unique_shards:
        shard = load_trajectory_shard(
            run_dir / shard_path,
            expected_sha256=shard_sha256,
        )
        source_digest = shard.source_sha256()
        for facts in shard.all_episode_facts():
            rows.append(
                EpisodeRewardFacts(
                    sample_id=facts.sample_id,
                    source_trajectory_format="diffsbdd_frozen_policy_trajectory_v1",
                    source_checkpoint_sha256=facts.checkpoint_sha256,
                    source_shard_sha256=source_digest,
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
            )
    return RewardSidecar(rows)


def _load_reward(
    config: dict[str, Any],
    *,
    run_dir: Path,
    episode_index: EpisodeIndex,
    project_root: Path,
) -> tuple[RewardSidecar, bool, str | None]:
    """Load a formal immutable sidecar or an explicitly diagnostic probe."""
    formal_training = bool(config.get("formal_training", True))
    reward_config = _require_mapping(config, "reward")
    sidecar_value = reward_config.get("sidecar")
    expected_version = reward_config.get("expected_version")
    diagnostic_probe = bool(reward_config.get("diagnostic_validity_probe", False))
    if formal_training:
        if diagnostic_probe or not sidecar_value:
            raise ValueError(
                "formal_training=true requires reward.sidecar and forbids "
                "diagnostic_validity_probe"
            )
        sidecar_path = resolve_project_path(str(sidecar_value), project_root)
        sidecar = load_reward_sidecar(
            sidecar_path,
            expected_reward_version=(
                None if expected_version is None else str(expected_version)
            ),
        )
        return sidecar, True, sha256_file(sidecar_path)
    if not diagnostic_probe or sidecar_value:
        raise ValueError(
            "diagnostic runs require diagnostic_validity_probe=true and no sidecar"
        )
    return _build_diagnostic_sidecar(run_dir, episode_index), False, None


def _time_bins(config: dict[str, Any]) -> tuple[DecisionTimeBin, ...]:
    """Parse configured inclusive decision-index bins."""
    raw_bins = config.get("time_bins")
    if not isinstance(raw_bins, list) or not raw_bins:
        raise ValueError("sampler.time_bins must be a non-empty list")
    return tuple(
        DecisionTimeBin(
            start=int(item["start"]),
            end=int(item["end"]),
            weight=float(item.get("weight", 1.0)),
        )
        for item in raw_bins
        if isinstance(item, dict)
    )


def build_offline_iql_runtime(
    config: dict[str, Any],
    *,
    project_root: Path,
    dist_ctx: DistributedContext,
    epochs_override: int | None = None,
) -> OfflineIQLRuntimeBundle:
    """Build real DiffSBDD actor, geometric critics, sampler, and IQL trainer."""
    offline_config = _require_mapping(config, "offline_iql")
    data_config = _require_mapping(offline_config, "data")
    model_config = _require_mapping(offline_config, "model")
    critic_values = _require_mapping(offline_config, "critic")
    trainer_values = _require_mapping(offline_config, "trainer")
    actor_values = _require_mapping(offline_config, "actor")
    sampler_values = _require_mapping(offline_config, "sampler")
    run_values = _require_mapping(offline_config, "run")

    run_dir = resolve_project_path(str(data_config["run_dir"]), project_root)
    index_path = resolve_project_path(str(data_config["episode_index"]), project_root)
    split_path = resolve_project_path(str(data_config["split_manifest"]), project_root)
    checkpoint_path = resolve_project_path(str(model_config["checkpoint"]), project_root)
    pocket_path = resolve_project_path(str(model_config["pocket_pdb"]), project_root)
    for path in (run_dir, index_path, split_path, checkpoint_path, pocket_path):
        if not path.exists():
            raise FileNotFoundError(f"required offline IQL input not found: {path}")

    episode_index = load_episode_index(index_path)
    split_manifest: SplitManifest = load_split_manifest(split_path)
    split_manifest.validate_against(episode_index)
    reward_sidecar, formal_training, reward_sidecar_sha256 = _load_reward(
        offline_config,
        run_dir=run_dir,
        episode_index=episode_index,
        project_root=project_root,
    )
    first_row = episode_index.rows[0]
    num_diffusion_steps = first_row.num_diffusion_steps
    if any(row.num_diffusion_steps != num_diffusion_steps for row in episode_index.rows):
        raise ValueError("episode index mixes diffusion step counts")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if checkpoint_sha256.lower() != first_row.source_checkpoint_sha256.lower():
        raise ValueError(
            "configured policy checkpoint does not match trajectory provenance"
        )
    pocket_sha256 = sha256_file(pocket_path)
    expected_pocket_sha256 = str(
        data_config.get("expected_pocket_sha256", "")
    ).lower()
    if not expected_pocket_sha256:
        raise ValueError("data.expected_pocket_sha256 is required")
    if pocket_sha256.lower() != expected_pocket_sha256:
        raise ValueError("configured pocket PDB does not match trajectory provenance")

    rank_seed = dist_ctx.rank + 1
    torch.manual_seed(rank_seed)
    if dist_ctx.device.type == "cuda":
        torch.cuda.manual_seed_all(rank_seed)
    policy_core = DiffSBDDModel(
        checkpoint_path=str(checkpoint_path),
        pocket_pdb_path=str(pocket_path),
        ref_ligand_path=str(model_config.get("ref_ligand", "C:1101")),
        device=str(dist_ctx.device),
        dataset_name=str(model_config.get("dataset_name", "crossdock")),
        logprob_mode=str(model_config.get("logprob_mode", "mean_per_atom")),
    )
    policy_core.train()
    selected = _select_trainable_policy_parameters(
        policy_core,
        int(model_config.get("trainable_parameter_tensors", 2)),
    )
    pocket = PocketContext.from_diffsbdd_mapping(
        policy_core.export_single_pocket_context()
    ).to(dist_ctx.device)
    probe_shard = load_trajectory_shard(
        run_dir / first_row.shard_path,
        expected_sha256=first_row.source_shard_sha256,
    )
    critic_config = GeometricCriticConfig(
        ligand_feature_dim=int(probe_shard.latents.shape[-1]) - 3,
        pocket_feature_dim=int(pocket.features.shape[-1]),
        hidden_dim=int(critic_values.get("hidden_dim", 128)),
        num_message_layers=int(critic_values.get("num_message_layers", 3)),
        num_rbf=int(critic_values.get("num_rbf", 16)),
        time_embedding_dim=int(critic_values.get("time_embedding_dim", 32)),
        max_neighbors=int(critic_values.get("max_neighbors", 32)),
    )
    online_q_core = TwinQCritic(critic_config).to(dist_ctx.device)
    value_core = GeometricValueNetwork(critic_config).to(dist_ctx.device)
    if dist_ctx.world_size > 1:
        ddp_options = {
            "device_ids": [dist_ctx.local_rank],
            "output_device": dist_ctx.local_rank,
            "find_unused_parameters": bool(
                config.get("ddp", {}).get("find_unused_parameters", False)
            ),
        }
        policy: nn.Module = DDP(policy_core, **ddp_options)
        online_q: nn.Module = DDP(online_q_core, **ddp_options)
        value: nn.Module = DDP(value_core, **ddp_options)
    else:
        policy, online_q, value = policy_core, online_q_core, value_core

    actor_config = ActorUpdateConfig(
        objective="iql",
        temperature=float(actor_values.get("temperature", 1.0)),
        max_weight=float(actor_values.get("max_weight", 10.0)),
        max_grad_norm=float(actor_values.get("max_grad_norm", 5.0)),
    )
    trainer_config = IQLTrainerConfig(
        num_diffusion_steps=num_diffusion_steps,
        discount=float(trainer_values.get("discount", 1.0)),
        expectile=float(trainer_values.get("expectile", 0.7)),
        target_tau=float(trainer_values.get("target_tau", 0.005)),
        q_loss_type=str(trainer_values.get("q_loss_type", "mse")),
        critic_max_grad_norm=float(
            trainer_values.get("critic_max_grad_norm", 5.0)
        ),
        actor_warmup_steps=int(trainer_values.get("actor_warmup_steps", 0)),
        actor_update_interval=int(trainer_values.get("actor_update_interval", 1)),
    )
    trainer = OfflineIQLTrainer(
        online_q=online_q,
        value=value,
        q_optimizer=torch.optim.Adam(
            online_q_core.parameters(),
            lr=float(critic_values.get("learning_rate", 1e-4)),
            weight_decay=float(critic_values.get("weight_decay", 0.0)),
        ),
        value_optimizer=torch.optim.Adam(
            value_core.parameters(),
            lr=float(critic_values.get("learning_rate", 1e-4)),
            weight_decay=float(critic_values.get("weight_decay", 0.0)),
        ),
        actor_updater=OfflineActorUpdater(
            policy_adapter=DiffSBDDPolicyAdapter(
                policy,
                num_diffusion_steps=num_diffusion_steps,
            ),
            optimizer=torch.optim.Adam(
                [parameter for _, parameter in selected],
                lr=float(actor_values.get("learning_rate", 1e-6)),
                weight_decay=float(actor_values.get("weight_decay", 0.0)),
            ),
            config=actor_config,
        ),
        pocket=pocket,
        config=trainer_config,
    )
    time_bins = _time_bins(sampler_values)
    sampler = OfflineTransitionSampler(
        run_dir=run_dir,
        episode_index=episode_index,
        split_manifest=split_manifest,
        split=str(data_config.get("split", "train")),
        time_bins=time_bins,
        reward_sidecar=reward_sidecar,
        cache_size=int(sampler_values.get("cache_size", 2)),
        seed=rank_seed,
        rank=dist_ctx.rank,
        world_size=dist_ctx.world_size,
        same_decision_index_per_batch=True,
    )
    scheduler = SharedDecisionIndexScheduler(
        time_bins,
        seed=int(sampler_values.get("shared_scheduler_seed", 1)),
    )
    epochs = int(run_values["epochs"] if epochs_override is None else epochs_override)
    runner_config = IQLRunnerConfig(
        epochs=epochs,
        steps_per_epoch=int(run_values["steps_per_epoch"]),
        batch_size=int(run_values["batch_size"]),
        device=str(dist_ctx.device),
        log_every_steps=int(run_values.get("log_every_steps", 10)),
        checkpoint_every_epochs=int(
            run_values.get("checkpoint_every_epochs", 1)
        ),
        non_blocking_transfer=bool(run_values.get("non_blocking_transfer", True)),
    )
    run_identity = {
        "format": "diffsbdd_offline_iql_run_identity_v1",
        "formal_training": formal_training,
        "source_run_id": episode_index.source_run_id,
        "episode_index_sha256": episode_index.sha256,
        "episode_index_file_sha256": sha256_file(index_path),
        "split_manifest_sha256": sha256_file(split_path),
        "reward_version": reward_sidecar.reward_version,
        "reward_sidecar_sha256": reward_sidecar_sha256,
        "source_checkpoint_sha256": checkpoint_sha256,
        "pocket_pdb_sha256": pocket_sha256,
        "split": sampler.split,
        "world_size": dist_ctx.world_size,
        "time_bins": [asdict(time_bin) for time_bin in time_bins],
        "shared_scheduler_seed": scheduler.seed,
        "critic_config": asdict(critic_config),
        "trainer_config": asdict(trainer_config),
        "actor_config": asdict(actor_config),
        "trainable_policy_parameter_names": [name for name, _ in selected],
    }
    json.dumps(run_identity, sort_keys=True)
    return OfflineIQLRuntimeBundle(
        trainer=trainer,
        sampler=sampler,
        runner_config=runner_config,
        scheduler=scheduler,
        run_identity=run_identity,
        trainable_policy_parameter_names=tuple(name for name, _ in selected),
        formal_training=formal_training,
        reward_version=reward_sidecar.reward_version,
    )
