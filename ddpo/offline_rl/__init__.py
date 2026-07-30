"""Decoupled offline reinforcement-learning primitives for DiffB110.

This package deliberately contains no trajectory I/O, DiffSBDD model, critic
architecture, or trainer imports. Those layers consume the stateless
mathematical helpers exported here.
"""

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
from .target import polyak_update

__all__ = [
    "compute_actor_loss",
    "compute_advantage",
    "compute_advantage_weight",
    "compute_q_loss",
    "compute_td_target",
    "compute_value_loss",
    "effective_sample_size",
    "expectile_loss",
    "polyak_update",
]
