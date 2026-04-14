"""Reward functions for DDPO training.

Contains:
- BaseReward:          Abstract base class (from interfaces/)
- QEDReward:           QED-only reward (qed_reward.py)
- SAReward:            SA-only reward  (sa_reward.py)
- DruglikenessReward:  Weighted QED+SA composite (druglikeness.py)
- DockingReward:       AutoDock-GPU docking score (docking.py)
- CompositeReward:     Weighted combination of multiple rewards
- StabilityAwareReward: Wraps any reward to inject batch['stability']
"""

from .base import BaseReward
from .qed_reward import QEDReward
from .sa_reward import SAReward
from .druglikeness import DruglikenessReward
from .docking import DockingReward
from .composite import CompositeReward
from .stability import StabilityAwareReward

__all__ = [
    "BaseReward",
    "QEDReward",
    "SAReward",
    "DruglikenessReward",
    "DockingReward",
    "CompositeReward",
    "StabilityAwareReward",
]
