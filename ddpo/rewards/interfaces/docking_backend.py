"""BaseDockingBackend: abstract interface for docking backend implementations.

Pure contract — no implementation, no I/O, no docking logic.
All backend classes (AutoDockGPUBackend, UniDockBackend, SminaBackend)
must satisfy this interface.

Design:
    The primary interface is dock_batch_result() which returns structured
    DockingResult objects containing both the score and failure semantics.

    The legacy float interface dock_batch() is provided as a mixin default
    that calls dock_batch_result() — subclasses need not re-implement it.

Usage:
    from ddpo.rewards.interfaces import BaseDockingBackend, DockingResult
    class MyBackend(BaseDockingBackend):
        @property
        def backend_name(self) -> str: return 'my_backend'
        def dock_batch_result(self, ligand_files, receptor_file, ...) -> List[DockingResult]: ...
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from .docking_result import DockingResult


class BaseDockingBackend(ABC):
    """Abstract base class for molecular docking backends.

    Subclasses must implement:
        backend_name  (property)
        dock_batch_result()

    Subclasses may optionally override:
        dock_batch()  — defaults to extracting score_raw from dock_batch_result()
        dock_single_result()  — defaults to calling dock_batch_result() with 1 file
        dock_single()  — defaults to extracting score_raw from dock_single_result()
    """

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Short identifier string, e.g. 'autodock_gpu'."""
        ...

    @abstractmethod
    def dock_batch_result(
        self,
        ligand_files: List[str],
        receptor_file: str,
        center_x: Optional[float] = None,
        center_y: Optional[float] = None,
        center_z: Optional[float] = None,
        size_x: float = 20.0,
        size_y: float = 20.0,
        size_z: float = 20.0,
        exhaustiveness: int = 8,
        **kwargs,
    ) -> List[DockingResult]:
        """Dock a batch of ligands and return structured results.

        Args:
            ligand_files: Paths to ligand SDF/PDBQT files.
            receptor_file: Path to receptor PDBQT file (informational for some backends).
            center_x/y/z: Docking box centre (Angstroms).
            size_x/y/z:   Docking box size (Angstroms).
            exhaustiveness: Search exhaustiveness (backend-specific meaning).
            **kwargs: Additional backend-specific parameters.

        Returns:
            List[DockingResult] of length == len(ligand_files).
            Each entry is either DockingResult.succeeded(...) or
            DockingResult.failed(...).
        """
        ...

    # ------------------------------------------------------------------
    # Default implementations — subclasses may override for performance
    # ------------------------------------------------------------------

    def dock_single_result(
        self,
        ligand_file: str,
        receptor_file: str,
        center_x: Optional[float] = None,
        center_y: Optional[float] = None,
        center_z: Optional[float] = None,
        size_x: float = 20.0,
        size_y: float = 20.0,
        size_z: float = 20.0,
        exhaustiveness: int = 8,
        **kwargs,
    ) -> DockingResult:
        """Dock a single ligand. Delegates to dock_batch_result() by default."""
        results = self.dock_batch_result(
            ligand_files=[ligand_file],
            receptor_file=receptor_file,
            center_x=center_x,
            center_y=center_y,
            center_z=center_z,
            size_x=size_x,
            size_y=size_y,
            size_z=size_z,
            exhaustiveness=exhaustiveness,
            **kwargs,
        )
        return results[0]

    def dock_batch(
        self,
        ligand_files: List[str],
        receptor_file: str,
        center_x: Optional[float] = None,
        center_y: Optional[float] = None,
        center_z: Optional[float] = None,
        size_x: float = 20.0,
        size_y: float = 20.0,
        size_z: float = 20.0,
        exhaustiveness: int = 8,
        **kwargs,
    ) -> List[float]:
        """Legacy float interface. Returns score_raw for each result.

        Kept for backward compatibility with code that predates DockingResult.
        Prefer dock_batch_result() in new code.
        """
        return [
            r.score_raw
            for r in self.dock_batch_result(
                ligand_files=ligand_files,
                receptor_file=receptor_file,
                center_x=center_x,
                center_y=center_y,
                center_z=center_z,
                size_x=size_x,
                size_y=size_y,
                size_z=size_z,
                exhaustiveness=exhaustiveness,
                **kwargs,
            )
        ]

    def dock_single(
        self,
        ligand_file: str,
        receptor_file: str,
        center_x: Optional[float] = None,
        center_y: Optional[float] = None,
        center_z: Optional[float] = None,
        size_x: float = 20.0,
        size_y: float = 20.0,
        size_z: float = 20.0,
        exhaustiveness: int = 8,
        **kwargs,
    ) -> float:
        """Legacy float interface for single docking."""
        return self.dock_single_result(
            ligand_file=ligand_file,
            receptor_file=receptor_file,
            center_x=center_x,
            center_y=center_y,
            center_z=center_z,
            size_x=size_x,
            size_y=size_y,
            size_z=size_z,
            exhaustiveness=exhaustiveness,
            **kwargs,
        ).score_raw
