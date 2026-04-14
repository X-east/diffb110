"""Legacy shim — BaseReward moved to ddpo.rewards.interfaces.base_reward.

Kept for backward compatibility. Do not add new code here.

New code should import from:
    from ddpo.rewards.interfaces import BaseReward
"""
from ddpo.rewards.interfaces.base_reward import BaseReward

__all__ = ["BaseReward"]
