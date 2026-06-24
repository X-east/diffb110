"""Configuration dataclasses for paired-sidechain symmetry constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SUPPORTED_SYMMETRY_MODES = frozenset({"single_sidechain_copy"})
SUPPORTED_COPY_CHANNELS = frozenset({"h"})
SUPPORTED_COORDINATE_POLICIES = frozenset({"free"})
SUPPORTED_COPY_SCHEDULES = frozenset({"every_step"})
SUPPORTED_TOPOLOGY_MODES = frozenset({"audit_only"})


@dataclass(frozen=True)
class SymmetrySpec:
    """Describe an R1-to-R2 sidechain copy constraint over padded ligand slots."""

    enabled: bool
    mode: str = "single_sidechain_copy"
    r1_indices: tuple[int, ...] = ()
    r2_indices: tuple[int, ...] = ()
    copy_channels: tuple[str, ...] = ("h",)
    coordinate_policy: str = "free"
    copy_schedule: str = "every_step"
    topology_mode: str = "audit_only"
    n_dims: int = 3
    atom_nf: int | None = None
    normalize_logprob_by_free_dims: bool = True

    def __post_init__(self) -> None:
        """Validate the constraint definition early and loudly."""
        if not self.enabled:
            return
        if self.mode not in SUPPORTED_SYMMETRY_MODES:
            raise ValueError(f"Unsupported symmetry mode: {self.mode!r}")
        if self.coordinate_policy not in SUPPORTED_COORDINATE_POLICIES:
            raise ValueError(f"Unsupported symmetry coordinate_policy: {self.coordinate_policy!r}")
        if self.copy_schedule not in SUPPORTED_COPY_SCHEDULES:
            raise ValueError(f"Unsupported symmetry copy_schedule: {self.copy_schedule!r}")
        if self.topology_mode not in SUPPORTED_TOPOLOGY_MODES:
            raise ValueError(f"Unsupported symmetry topology mode: {self.topology_mode!r}")
        if not self.r1_indices or not self.r2_indices:
            raise ValueError("symmetry.r1_indices and symmetry.r2_indices are required when enabled")
        if len(self.r1_indices) != len(self.r2_indices):
            raise ValueError("symmetry.r1_indices and symmetry.r2_indices must have the same length")
        if len(set(self.r1_indices)) != len(self.r1_indices):
            raise ValueError("symmetry.r1_indices must not contain duplicates")
        if len(set(self.r2_indices)) != len(self.r2_indices):
            raise ValueError("symmetry.r2_indices must not contain duplicates")
        if set(self.r1_indices).intersection(self.r2_indices):
            raise ValueError("symmetry.r1_indices and symmetry.r2_indices must be disjoint")
        if any(index < 0 for index in (*self.r1_indices, *self.r2_indices)):
            raise ValueError("symmetry indices must be non-negative padded ligand slots")
        unsupported_channels = set(self.copy_channels).difference(SUPPORTED_COPY_CHANNELS)
        if unsupported_channels:
            raise ValueError(f"Unsupported symmetry copy channel(s): {sorted(unsupported_channels)}")
        if "h" not in self.copy_channels:
            raise ValueError("The first implementation requires copy_channels to include 'h'")
        if self.n_dims <= 0:
            raise ValueError("symmetry.n_dims must be positive")
        if self.atom_nf is not None and self.atom_nf <= 0:
            raise ValueError("symmetry.atom_nf must be positive when provided")

    @property
    def h_start(self) -> int:
        """Return the first feature dimension occupied by atom-type channels."""
        return int(self.n_dims)

    def h_stop_for(self, feature_dim: int) -> int:
        """Return the exclusive h-channel stop index for a tensor feature width."""
        if self.atom_nf is None:
            return int(feature_dim)
        return min(int(feature_dim), int(self.n_dims) + int(self.atom_nf))

    def to_metadata(self) -> dict[str, Any]:
        """Return a compact JSON-serializable description for run records."""
        return {
            "enabled": bool(self.enabled),
            "mode": self.mode,
            "r1_indices": list(self.r1_indices),
            "r2_indices": list(self.r2_indices),
            "copy_channels": list(self.copy_channels),
            "coordinate_policy": self.coordinate_policy,
            "copy_schedule": self.copy_schedule,
            "topology_mode": self.topology_mode,
            "n_dims": int(self.n_dims),
            "atom_nf": None if self.atom_nf is None else int(self.atom_nf),
            "normalize_logprob_by_free_dims": bool(self.normalize_logprob_by_free_dims),
        }

