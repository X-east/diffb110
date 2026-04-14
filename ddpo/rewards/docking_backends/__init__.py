"""
Docking backend implementations for molecular docking.

Supports multiple docking engines:
- AutoDockGPU: GPU-accelerated docking via AutoDock-GPU (recommended)
- UniDock: GPU-accelerated docking via Uni-Dock (legacy, currently broken)
- Smina: CPU-based docking (legacy)
"""

from .base import BaseDockingBackend
from ddpo.rewards.interfaces.docking_result import DockingResult
from .autodock_gpu import AutoDockGPUBackend
from .unidock import UniDockBackend
from .smina import SminaBackend

__all__ = ['BaseDockingBackend', 'DockingResult', 'AutoDockGPUBackend', 'UniDockBackend', 'SminaBackend']
