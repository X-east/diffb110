"""Decoupled offline reinforcement-learning primitives for DiffB110.

The package keeps trajectory I/O, policy integration, geometric critics, and
trainer logic in separate modules while exposing their typed public boundaries.
"""

from .critic_adapter import (
    GeometricCriticInput,
    PocketContext,
    build_geometric_critic_input,
    build_next_geometric_critic_input,
)
from .dataset import (
    DenoisingTrajectoryShard,
    OfflineTrajectoryDataError,
    load_trajectory_shard,
)
from .losses import (
    compute_actor_loss,
    compute_advantage,
    compute_advantage_weight,
    compute_q_loss,
    compute_td_target,
    compute_value_loss,
    effective_sample_size,
    expectile_loss,
)
from .index import (
    EPISODE_INDEX_SCHEMA_VERSION,
    SPLIT_MANIFEST_SCHEMA_VERSION,
    EpisodeIndex,
    EpisodeIndexError,
    EpisodeIndexRow,
    SplitAssignment,
    SplitManifest,
    build_episode_index,
    build_split_manifest,
    load_episode_index,
    load_split_manifest,
    write_episode_index,
    write_split_manifest,
)
from .policy_adapter import DiffSBDDPolicyAdapter, PolicyLogProbOutput
from .protocol import DenoisingEpisodeFacts, DenoisingTransitionBatch
from .reward_sidecar import (
    REWARD_SIDECAR_SCHEMA_VERSION,
    EpisodeRewardFacts,
    RewardJoinedTrajectoryShard,
    RewardSidecar,
    RewardSidecarError,
    load_reward_sidecar,
    write_reward_sidecar,
)
from .recording import (
    EPOCH_METRIC_FIELDS,
    STEP_METRIC_FIELDS,
    EpochMetrics,
    LocalIQLRecorder,
)
from .ddp import (
    DDP_LATEST_FORMAT,
    DDP_MANIFEST_FORMAT,
    OfflineIQLDDPCoordinator,
)
from .runner import IQLRunnerConfig, OfflineIQLRunner
from .sampler import (
    DecisionTimeBin,
    OfflineTransitionSampler,
    SamplerStats,
    SharedDecisionIndexScheduler,
    default_diffsbdd_time_bins,
)
from .target import polyak_update
from .trainer import (
    ActorStepMetrics,
    ActorUpdateConfig,
    IQLStepMetrics,
    IQLTrainerConfig,
    OfflineActorUpdater,
    OfflineIQLTrainer,
)

__all__ = [
    "DenoisingEpisodeFacts",
    "DenoisingTrajectoryShard",
    "DenoisingTransitionBatch",
    "DecisionTimeBin",
    "DiffSBDDPolicyAdapter",
    "ActorStepMetrics",
    "ActorUpdateConfig",
    "EPISODE_INDEX_SCHEMA_VERSION",
    "EpisodeRewardFacts",
    "EpochMetrics",
    "EpisodeIndex",
    "EpisodeIndexError",
    "EpisodeIndexRow",
    "GeometricCriticInput",
    "IQLStepMetrics",
    "IQLTrainerConfig",
    "IQLRunnerConfig",
    "LocalIQLRecorder",
    "OfflineTrajectoryDataError",
    "OfflineTransitionSampler",
    "OfflineActorUpdater",
    "OfflineIQLTrainer",
    "OfflineIQLRunner",
    "OfflineIQLDDPCoordinator",
    "PocketContext",
    "PolicyLogProbOutput",
    "REWARD_SIDECAR_SCHEMA_VERSION",
    "RewardJoinedTrajectoryShard",
    "RewardSidecar",
    "RewardSidecarError",
    "SamplerStats",
    "SharedDecisionIndexScheduler",
    "DDP_MANIFEST_FORMAT",
    "DDP_LATEST_FORMAT",
    "STEP_METRIC_FIELDS",
    "EPOCH_METRIC_FIELDS",
    "SPLIT_MANIFEST_SCHEMA_VERSION",
    "SplitAssignment",
    "SplitManifest",
    "build_episode_index",
    "build_geometric_critic_input",
    "build_next_geometric_critic_input",
    "build_split_manifest",
    "compute_actor_loss",
    "compute_advantage",
    "compute_advantage_weight",
    "compute_q_loss",
    "compute_td_target",
    "compute_value_loss",
    "effective_sample_size",
    "expectile_loss",
    "default_diffsbdd_time_bins",
    "load_reward_sidecar",
    "load_episode_index",
    "load_split_manifest",
    "load_trajectory_shard",
    "polyak_update",
    "write_reward_sidecar",
    "write_episode_index",
    "write_split_manifest",
]
