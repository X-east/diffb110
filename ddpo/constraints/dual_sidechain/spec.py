"""Specifications for dual-sidechain substitution and audit tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PhenolicOHSite:
    """A phenolic OH leaving group attached to an aromatic carbon."""

    core_atom_idx: int
    oxygen_atom_idx: int
    hydrogen_atom_idx: int | None = None

    def __post_init__(self) -> None:
        """Validate atom indices."""
        if self.core_atom_idx < 0:
            raise ValueError("core_atom_idx must be non-negative")
        if self.oxygen_atom_idx < 0:
            raise ValueError("oxygen_atom_idx must be non-negative")
        if self.hydrogen_atom_idx is not None and self.hydrogen_atom_idx < 0:
            raise ValueError("hydrogen_atom_idx must be non-negative when provided")
        if self.core_atom_idx == self.oxygen_atom_idx:
            raise ValueError("core_atom_idx and oxygen_atom_idx must differ")

    def to_dict(self) -> dict[str, int | None]:
        """Return a JSON-serializable representation."""
        return {
            "core_atom_idx": int(self.core_atom_idx),
            "oxygen_atom_idx": int(self.oxygen_atom_idx),
            "hydrogen_atom_idx": None if self.hydrogen_atom_idx is None else int(self.hydrogen_atom_idx),
        }


@dataclass(frozen=True)
class DualSubstitutionSpec:
    """Two aromatic-core sites whose outgoing substituent branches should match."""

    left_core_idx: int
    right_core_idx: int
    left_leaving_atom_idx: int | None = None
    right_leaving_atom_idx: int | None = None
    left_label: str = "R1"
    right_label: str = "R2"
    max_branch_atoms: int | None = None

    def __post_init__(self) -> None:
        """Validate static site metadata."""
        if self.left_core_idx < 0 or self.right_core_idx < 0:
            raise ValueError("core atom indices must be non-negative")
        if self.left_core_idx == self.right_core_idx:
            raise ValueError("left_core_idx and right_core_idx must differ")
        for field_name in ("left_leaving_atom_idx", "right_leaving_atom_idx"):
            value = getattr(self, field_name)
            if value is not None and value < 0:
                raise ValueError(f"{field_name} must be non-negative when provided")
        if self.max_branch_atoms is not None and self.max_branch_atoms < 1:
            raise ValueError("max_branch_atoms must be >= 1 when provided")

    @classmethod
    def from_phenolic_sites(
        cls,
        left_site: PhenolicOHSite,
        right_site: PhenolicOHSite,
        *,
        left_label: str = "R1",
        right_label: str = "R2",
        max_branch_atoms: int | None = None,
    ) -> DualSubstitutionSpec:
        """Build a substitution spec where phenolic OH atoms are leaving groups."""
        return cls(
            left_core_idx=left_site.core_atom_idx,
            right_core_idx=right_site.core_atom_idx,
            left_leaving_atom_idx=left_site.oxygen_atom_idx,
            right_leaving_atom_idx=right_site.oxygen_atom_idx,
            left_label=left_label,
            right_label=right_label,
            max_branch_atoms=max_branch_atoms,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "left_core_idx": int(self.left_core_idx),
            "right_core_idx": int(self.right_core_idx),
            "left_leaving_atom_idx": None
            if self.left_leaving_atom_idx is None
            else int(self.left_leaving_atom_idx),
            "right_leaving_atom_idx": None
            if self.right_leaving_atom_idx is None
            else int(self.right_leaving_atom_idx),
            "left_label": str(self.left_label),
            "right_label": str(self.right_label),
            "max_branch_atoms": None if self.max_branch_atoms is None else int(self.max_branch_atoms),
        }
