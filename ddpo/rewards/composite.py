"""
CompositeReward: Weighted combination of multiple reward functions.

Combines DockingReward, QEDReward, SAReward, and other rewards with configurable weights.
"""

import warnings
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import torch

from .interfaces import BaseReward

if TYPE_CHECKING:
    from verl_diffusion.protocol import DataProto


# ============================================================================
# USER CONFIGURABLE PARAMETERS
# ============================================================================

# Default weights for each reward component
DEFAULT_WEIGHTS = {
    "docking": 0.5,
    "druglikeness": 0.5,
}

# Fallback reward when all components fail
FALLBACK_REWARD = 0.0

# ============================================================================


class CompositeReward(BaseReward):
    """
    Weighted combination of multiple reward functions.
    
    Final reward = sum(weight_i * reward_i) for all reward components.
    
    Supports:
    - Multiple reward functions with configurable weights
    - Dynamic weight updates during training
    - Graceful fallback when individual rewards fail
    - Detailed logging of component rewards
    """

    def __init__(
        self,
        rewards: Dict[str, BaseReward],
        weights: Optional[Dict[str, float]] = None,
        fallback_reward: float = FALLBACK_REWARD,
        normalize_weights: bool = True,
    ):
        """
        Initialize CompositeReward.
        
        Args:
            rewards: Dict mapping reward names to BaseReward instances
                     e.g., {"docking": DockingReward(...), "druglikeness": DruglikenessReward(...)}
            weights: Dict mapping reward names to weights (default: equal weights)
            fallback_reward: Reward value when all components fail
            normalize_weights: If True, normalize weights to sum to 1.0
        """
        if not rewards:
            raise ValueError("At least one reward function must be provided")
        
        self.rewards = rewards
        self.fallback_reward = fallback_reward
        self.normalize_weights = normalize_weights
        
        # Initialize weights
        if weights is None:
            # Equal weights for all rewards
            n_rewards = len(rewards)
            self.weights = {name: 1.0 / n_rewards for name in rewards}
        else:
            # Validate weights match rewards
            missing = set(rewards.keys()) - set(weights.keys())
            if missing:
                warnings.warn(f"Missing weights for: {missing}. Using 0.0 for these.")
            self.weights = {name: weights.get(name, 0.0) for name in rewards}
        
        # Normalize if requested
        if normalize_weights:
            self._normalize_weights()
    
    def _normalize_weights(self) -> None:
        """Normalize weights to sum to 1.0."""
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}
        else:
            # All weights are 0, use equal weights
            n = len(self.weights)
            self.weights = {k: 1.0 / n for k in self.weights}
    
    def update_weights(self, new_weights: Dict[str, float]) -> None:
        """
        Dynamically update reward weights.
        
        Args:
            new_weights: Dict mapping reward names to new weights
        """
        for name, weight in new_weights.items():
            if name in self.weights:
                self.weights[name] = weight
            else:
                warnings.warn(f"Unknown reward name: {name}. Ignoring.")
        
        if self.normalize_weights:
            self._normalize_weights()
    
    def calculate_rewards(self, data: "DataProto") -> "DataProto":
        """
        Calculate weighted combination of all reward components.
        
        Each sub-reward writes its normalised value to batch['{name}_reward'].
        Composite reads those fields, computes the weighted sum, and writes:
            'reward_total_raw'  [B]  -- weighted combination (before normalizer)
            'reward_total'      [B]  -- same as reward_total_raw (may be overridden by normalizer)

        Args:
            data: DataProto containing molecule data

        Returns:
            DataProto with 'reward_total' and 'reward_total_raw' tensors [B] added.
        """
        batch = data.batch
        
        # Determine batch size from input
        if 'x' in batch:
            batch_size = batch['x'].shape[0]
            device = batch['x'].device
        elif 'nodesxsample' in batch:
            batch_size = batch['nodesxsample'].shape[0]
            device = batch['nodesxsample'].device
        else:
            raise ValueError("Cannot determine batch size from data")
        
        # Initialize combined rewards
        combined_rewards = torch.zeros(batch_size, device=device)
        component_rewards = {}
        successful_components = []
        
        # Calculate each component reward
        for name, reward_fn in self.rewards.items():
            weight = self.weights.get(name, 0.0)
            if weight == 0.0:
                continue
            
            try:
                # Calculate component reward (writes batch[f'{name}_reward'])
                data = reward_fn.calculate_rewards(data)
                
                # Read the normalised reward from the dedicated field
                reward_field = f'{name}_reward'
                if reward_field in data.batch:
                    component_reward = data.batch[reward_field].clone()
                    component_rewards[name] = component_reward
                    
                    # Add weighted contribution
                    combined_rewards += weight * component_reward
                    successful_components.append(name)
                else:
                    warnings.warn(f"Reward '{name}' did not return '{reward_field}' tensor")
                    
            except Exception as e:
                # Use full exception logging so that errors (e.g. cache load failure)
                # are not silently swallowed — traceback is essential for debugging.
                import logging
                logging.getLogger(__name__).exception(f"Failed to calculate '{name}' reward: {e}")
                # Use zeros for failed component
                component_rewards[name] = torch.zeros(batch_size, device=device)
        
        # Handle case where all components failed
        if not successful_components:
            warnings.warn("All reward components failed. Using fallback reward.")
            combined_rewards = torch.full(
                (batch_size,), self.fallback_reward, device=device
            )
        
        # Store results
        data.batch['reward_total_raw'] = combined_rewards.clone()
        data.batch['reward_total'] = combined_rewards
        
        return data
    
    def get_weights(self) -> Dict[str, float]:
        """Return current weights."""
        return self.weights.copy()
    
    def __repr__(self) -> str:
        weight_str = ", ".join(f"{k}={v:.2f}" for k, v in self.weights.items())
        return f"CompositeReward({weight_str})"
