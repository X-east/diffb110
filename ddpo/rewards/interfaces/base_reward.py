"""BaseReward: abstract interface for all reward functions.

Pure contract — no implementation, no I/O.
All reward classes (DruglikenessReward, DockingReward, CompositeReward,
StabilityAwareReward) must satisfy this interface.

Usage:
    from ddpo.rewards.interfaces import BaseReward
    class MyReward(BaseReward):
        def calculate_rewards(self, data): ...
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ddpo.runtime.protocol import DataProto


class BaseReward(ABC):
    """Abstract base class for all DDPO reward functions.

    Contract:
        calculate_rewards(data) receives a DataProto whose batch contains at
        minimum:
            'x'            : coordinates [B, N, 3]
            'categorical'  : atom types  [B, N, num_atom_types]
            'nodesxsample' : nodes/sample [B]
            'rdkit_valid'  : canonical validity flags [B]  (set by
                             annotate_batch_molecule_semantics before reward)

        It must return the same DataProto with 'rewards' [B] added to batch.
        Failed molecules must receive an explicit penalty (not NaN/Inf).
        Rewards must be finite and in a PPO-compatible range.
    """

    @abstractmethod
    def calculate_rewards(self, data: "DataProto") -> "DataProto":
        """Calculate rewards for a batch of generated molecules.

        Args:
            data: DataProto with molecular geometry and pre-annotated semantics.

        Returns:
            DataProto with data.batch['reward_total'] tensor [B] added.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
