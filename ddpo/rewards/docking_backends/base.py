"""Legacy shim — BaseDockingBackend moved to ddpo.rewards.interfaces.docking_backend.

Kept for backward compatibility. Do not add new code here.

New code should import from:
    from ddpo.rewards.interfaces import BaseDockingBackend, DockingResult
"""
from ddpo.rewards.interfaces.docking_backend import BaseDockingBackend
from ddpo.rewards.interfaces.docking_result import DockingResult

__all__ = ["BaseDockingBackend", "DockingResult"]
