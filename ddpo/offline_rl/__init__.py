"""Decoupled offline reinforcement-learning primitives for DiffB110.

This package deliberately contains no trajectory I/O, DiffSBDD model, critic
architecture, or trainer imports. Those layers consume the stateless
mathematical helpers exported here.
"""

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
from .protocol import DenoisingEpisodeFacts, DenoisingTransitionBatch
from .target import polyak_update

__all__ = [
    "DenoisingEpisodeFacts",
    "DenoisingTrajectoryShard",
    "DenoisingTransitionBatch",
    "OfflineTrajectoryDataError",
    "compute_actor_loss",
    "compute_advantage",
    "compute_advantage_weight",
    "compute_q_loss",
    "compute_td_target",
    "compute_value_loss",
    "effective_sample_size",
    "expectile_loss",
    "load_trajectory_shard",
    "polyak_update",
]
