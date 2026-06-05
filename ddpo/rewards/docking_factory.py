"""DockingReward factory: config-driven construction of DockingReward.

Single responsibility:
    Read docking config, enforce backend policy via docking_registry,
    and construct a DockingReward instance.

Separated from factory.py so that:
    - factory.py is a pure assembler (no construction logic)
    - docking_factory.py owns all docking-specific wiring
    - DockingReward stays focused on reward semantics

Usage (from factory.py):
    from ddpo.rewards.docking_factory import build_docking_reward
    docking_reward = build_docking_reward(config, logger)
"""
from __future__ import annotations

import logging

from ddpo.rewards.docking_registry import DOCKING_BACKEND_REGISTRY


def build_docking_reward(config: dict, logger: logging.Logger):
    """Construct a DockingReward from config, enforcing backend registry policy.

    Policy:
        - Unknown backend names raise ValueError immediately.
        - Non-'supported' backends emit a warning but are allowed.

    Args:
        config:  Full training config dict.
        logger:  Logger for policy warnings and info.

    Returns:
        DockingReward instance.
        dataset_info is NOT injected here; caller sets it after construction.
    """
    from ddpo.rewards.docking import DockingReward

    reward_cfg = config.get("reward", {})
    docking_cfg = reward_cfg.get("docking", {})
    backend_name = docking_cfg.get("backend", "autodock_gpu").lower()

    if backend_name not in DOCKING_BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown docking backend '{backend_name}'. "
            f"Valid options: {list(DOCKING_BACKEND_REGISTRY.keys())}"
        )

    spec = DOCKING_BACKEND_REGISTRY[backend_name]
    if spec.support_level != "supported":
        logger.warning(
            "[docking_factory] Backend '%s' support_level='%s' (%s). "
            "Use 'autodock_gpu' for production training.",
            backend_name, spec.support_level, spec.description,
        )

    receptor_file = config.get("pocket", {}).get(
        "pdbqt_path", config.get("pocket", {}).get("pdb_path", "")
    )

    logger.info(
        "[docking_factory] Building DockingReward: backend=%s support_level=%s",
        backend_name, spec.support_level,
    )

    return DockingReward(
        backend=backend_name,
        receptor_file=receptor_file,
        fld_file=docking_cfg.get("fld_file"),
        normalization_factor=float(docking_cfg.get("normalization_factor", -10.0)),
        center_x=docking_cfg.get("center_x"),
        center_y=docking_cfg.get("center_y"),
        center_z=docking_cfg.get("center_z"),
        size_x=float(docking_cfg.get("size_x", 20.0)),
        size_y=float(docking_cfg.get("size_y", 20.0)),
        size_z=float(docking_cfg.get("size_z", 20.0)),
        exhaustiveness=int(docking_cfg.get("exhaustiveness", 8)),
        gpu_id=int(docking_cfg.get("gpu_id", 0)),
        search_mode=docking_cfg.get("search_mode", "fast"),
        nrun=int(docking_cfg.get("nrun", 20)),
        heuristics=bool(docking_cfg.get("heuristics", True)),
        dataset_info=None,
    )
