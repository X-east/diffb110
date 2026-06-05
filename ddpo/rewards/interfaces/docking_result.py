"""DockingResult: structured result from a single docking attempt.

Pure data contract — no business logic, no I/O.
Used by BaseDockingBackend.dock_batch_result() and consumed by DockingReward.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class DockingResult:
    """Structured result from one docking attempt.

    Attributes:
        score_raw:      Raw docking score in kcal/mol. More negative = stronger binding.
                        Set to 0.0 on failure.
        success:        True iff docking completed without error and a valid score
                        was obtained.
        failure_reason: None on success. On failure, one of the FAILURE_* constants
                        from ddpo.utils.molecule_semantics, e.g.:
                        - FAILURE_DOCKING_INPUT  (mol build / SDF write failed)
                        - FAILURE_DOCKING_BACKEND (backend execution failed)
        backend_name:   Name of the backend that produced this result, e.g.
                        'autodock_gpu', 'unidock', 'smina'.
    """
    score_raw: float
    success: bool
    failure_reason: Optional[str]
    backend_name: str

    @classmethod
    def failed(cls, backend_name: str, failure_reason: str) -> "DockingResult":
        """Convenience constructor for a failed docking result."""
        return cls(
            score_raw=0.0,
            success=False,
            failure_reason=failure_reason,
            backend_name=backend_name,
        )

    @classmethod
    def succeeded(cls, backend_name: str, score_raw: float) -> "DockingResult":
        """Convenience constructor for a successful docking result."""
        return cls(
            score_raw=score_raw,
            success=True,
            failure_reason=None,
            backend_name=backend_name,
        )
