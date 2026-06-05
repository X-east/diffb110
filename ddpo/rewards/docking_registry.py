"""Docking backend registry: support-level policy and backend catalogue.

This module owns the canonical list of known docking backends and their
support status. It is the single source of truth for:
    - Which backends exist
    - Which are production-ready ('supported')
    - Which are legacy / debug only

Separated from factory.py so that the registry is a pure data/policy file
with no assembly logic, and from docking.py so that DockingReward has no
knowledge of which backends are approved for production.

Usage:
    from ddpo.rewards.docking_registry import (
        DOCKING_BACKEND_REGISTRY,
        MAINLINE_DOCKING_BACKENDS,
        DockingBackendSpec,
    )
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class DockingBackendSpec:
    """Metadata describing one docking backend's support status.

    Attributes:
        name:          Backend identifier string (matches config key).
        support_level: One of:
            'supported' — actively maintained, used in production training.
            'legacy'    — still runnable but not recommended; may be removed.
            'debug'     — CPU-only or low-quality; smoke tests only.
        description:   Human-readable note on current status.
    """
    name: str
    support_level: str
    description: str


# ---------------------------------------------------------------------------
# Canonical registry of all known docking backends.
# To add a new backend: add an entry here AND implement BaseDockingBackend.
# Only set support_level='supported' after production validation.
# ---------------------------------------------------------------------------
DOCKING_BACKEND_REGISTRY: Dict[str, DockingBackendSpec] = {
    "autodock_gpu": DockingBackendSpec(
        name="autodock_gpu",
        support_level="supported",
        description=(
            "AutoDock-GPU with pre-computed AutoGrid4 maps. "
            "Primary production backend. Binary at "
            "/usr/local/bin/autodock_gpu_64wi. "
            "Maps at data/8GJJ-C-Cip_charged.maps.fld."
        ),
    ),
"unidock": DockingBackendSpec(
    name="unidock",
    support_level="supported",
    description=(
        "UniDock GPU backend with --gpu_batch SDF mode. "
        "Verified: Uni-Dock v1.1.3 at /opt/unidock-1.1.3/bin/unidock."
    ),
),
    "smina": DockingBackendSpec(
        name="smina",
        support_level="debug",
        description=(
            "Smina CPU backend. Debug/smoke-test only. "
            "Not suitable for production due to speed."
        ),
    ),
}

# ---------------------------------------------------------------------------
# Convenience set: backends approved for mainline training runs.
# Use this to gate config validation in factory._build_docking_backend().
# ---------------------------------------------------------------------------
MAINLINE_DOCKING_BACKENDS: frozenset = frozenset(
    name
    for name, spec in DOCKING_BACKEND_REGISTRY.items()
    if spec.support_level == "supported"
)
