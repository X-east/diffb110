"""StabilityAwareReward: wraps any BaseReward to inject batch['stability'].

Responsibility:
    After the wrapped reward's calculate_rewards() runs, check whether
    batch['stability'] exists. If not, derive it from batch['rdkit_valid']
    (the canonical legality field set by annotate_batch_molecule_semantics).

Why a separate file:
    StabilityAwareReward is reward-domain logic, not runtime infrastructure.
    It was previously in ddpo/runtime/reward_wrappers.py (wrong layer).
    ddpo/runtime/reward_wrappers.py is now a legacy shim.

Usage:
    from ddpo.rewards.stability import StabilityAwareReward
    rewarder = StabilityAwareReward(composite_reward, dataset_info)
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ddpo.runtime.protocol import DataProto

log = logging.getLogger("DiffSBDD-DDPO")


class StabilityAwareReward:
    """Decorator that adds batch['stability'] after the wrapped reward runs.

    DDPOTrainer expects batch['stability'] for logging.
    This wrapper derives stability from batch['rdkit_valid'] — the canonical
    legality field produced by annotate_batch_molecule_semantics().

    Bug1 fix: previously this class independently ran build_molecule(
    use_openbabel=False), creating a second inconsistent molecule-build path.
    Now it reads the already-computed rdkit_valid field only.

    Invariant:
        annotate_batch_molecule_semantics() MUST be called before
        calculate_rewards() in the training forward pass. If rdkit_valid is
        absent, a warning is emitted and stability defaults to all-zeros
        (conservative — treats batch as fully invalid).
    """

    def __init__(self, base_reward, dataset_info: dict) -> None:
        """
        Args:
            base_reward: Any reward object with a calculate_rewards() method.
                         Typically a CompositeReward instance.
            dataset_info: Dataset info dict (passed through; not used internally).
        """
        self.base_reward = base_reward
        self.dataset_info = dataset_info

    def calculate_rewards(self, data: "DataProto", timer=None, **kwargs) -> "DataProto":
        """Run base reward then inject stability field."""
        data = self.base_reward.calculate_rewards(data, timer=timer)

        if "stability" not in data.batch:
            device = data.batch["x"].device
            stability = self._compute_stability(data)
            data.batch["stability"] = stability.to(device)

        return data

    def _compute_stability(self, data: "DataProto") -> torch.Tensor:
        """Derive stability tensor from non_tensor_batch['rdkit_valid']."""
        batch_size = data.batch["x"].shape[0]
        ntb = getattr(data, "non_tensor_batch", None)
        rdkit_valid = ntb.get("rdkit_valid") if ntb else data.batch.get("rdkit_valid")

        if rdkit_valid is not None:
            if isinstance(rdkit_valid, torch.Tensor):
                return rdkit_valid.float().cpu()
            return torch.tensor(rdkit_valid, dtype=torch.float32)

        # Should not be reached if mainline forward() calls annotate_batch first.
        log.warning(
            "[StabilityAwareReward] rdkit_valid not found; "
            "returning zeros. Ensure annotate_batch_molecule_semantics() "
            "is called before calculate_rewards()."
        )
        return torch.zeros(batch_size)

    def __repr__(self) -> str:
        return f"StabilityAwareReward({self.base_reward})"
