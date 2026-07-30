"""Decoupled offline reinforcement-learning primitives for DiffB110.

The package keeps trajectory I/O, policy integration, geometric critics, and
trainer logic in separate modules while exposing their typed public boundaries.
"""

from .critic_adapter import (
    GeometricCriticInput,
    PocketContext,
    build_geometric_critic_input,
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
from .target import polyak_update

__all__ = [
    "DenoisingEpisodeFacts",
    "DenoisingTrajectoryShard",
    "DenoisingTransitionBatch",
    "DiffSBDDPolicyAdapter",
    "EpisodeRewardFacts",
    "GeometricCriticInput",
    "OfflineTrajectoryDataError",
    "PocketContext",
    "PolicyLogProbOutput",
    "REWARD_SIDECAR_SCHEMA_VERSION",
    "RewardJoinedTrajectoryShard",
    "RewardSidecar",
    "RewardSidecarError",
    "build_geometric_critic_input",
    "compute_actor_loss",
    "compute_advantage",
    "compute_advantage_weight",
    "compute_q_loss",
    "compute_td_target",
    "compute_value_loss",
    "effective_sample_size",
    "expectile_loss",
    "load_reward_sidecar",
    "load_trajectory_shard",
    "polyak_update",
    "write_reward_sidecar",
]
