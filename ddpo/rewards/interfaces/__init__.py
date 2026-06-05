"""Pure interface layer for the ddpo rewards subsystem.

This package contains only abstract contracts (ABCs and dataclasses).
No business logic, no I/O, no external library calls beyond stdlib typing.

Import from here when you need to type-annotate or subclass:
    from ddpo.rewards.interfaces import BaseReward, BaseDockingBackend, DockingResult
"""
from .base_reward import BaseReward
from .docking_backend import BaseDockingBackend
from .docking_result import DockingResult

__all__ = ["BaseReward", "BaseDockingBackend", "DockingResult"]
