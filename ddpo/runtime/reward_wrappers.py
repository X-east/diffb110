"""Legacy shim — StabilityAwareReward moved to ddpo.rewards.stability.

This file is kept for backward compatibility only.
Do not add new code here.

New code should import from:
    from ddpo.rewards.stability import StabilityAwareReward
"""
from ddpo.rewards.stability import StabilityAwareReward

__all__ = ["StabilityAwareReward"]
