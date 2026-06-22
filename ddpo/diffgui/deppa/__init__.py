"""DePPA-style helpers for the decoupled DiffGui RL path."""

from __future__ import annotations

from ddpo.diffgui.deppa.advantage import (
    DePPAAdvantageResult,
    build_local_deppa_records,
    compute_deppa_advantages,
    is_deppa_mode_enabled,
)
from ddpo.diffgui.deppa.kl import compute_deppa_k3_kl

__all__ = [
    "DePPAAdvantageResult",
    "build_local_deppa_records",
    "compute_deppa_advantages",
    "compute_deppa_k3_kl",
    "is_deppa_mode_enabled",
]
