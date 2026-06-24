"""Tests for paired-slot dual-R inpainting constraints."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

DIFFSBDD_ROOT = Path(__file__).resolve().parents[1] / "DiffSBDD"
if str(DIFFSBDD_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFSBDD_ROOT))

from equivariant_diffusion.dual_r_constraints import (  # noqa: E402
    DualRConstraintConfig,
    apply_dual_r_latent_constraints,
    nudge_root_distances,
    should_apply_dual_r_constraints,
    sync_paired_atom_features,
)


def test_sync_paired_atom_features_per_sample() -> None:
    """Paired R1/R2 slots are synchronized within each sample only."""
    config = DualRConstraintConfig(
        enabled=True,
        fixed_atom_count=2,
        pair_count=2,
        left_core_idx=0,
        right_core_idx=1,
    )
    z_lig = torch.zeros((12, 6), dtype=torch.float32)
    lig_mask = torch.tensor([0] * 6 + [1] * 6, dtype=torch.long)
    z_lig[2, 3:] = torch.tensor([10.0, 0.0, 0.0])
    z_lig[4, 3:] = torch.tensor([0.0, 2.0, 0.0])
    z_lig[3, 3:] = torch.tensor([0.0, 0.0, 8.0])
    z_lig[5, 3:] = torch.tensor([0.0, 4.0, 0.0])
    z_lig[8, 3:] = torch.tensor([6.0, 0.0, 0.0])
    z_lig[10, 3:] = torch.tensor([2.0, 0.0, 0.0])

    synced = sync_paired_atom_features(z_lig, lig_mask, config, n_dims=3)

    assert torch.allclose(synced[2, 3:], synced[4, 3:])
    assert torch.allclose(synced[3, 3:], synced[5, 3:])
    assert torch.allclose(synced[8, 3:], synced[10, 3:])
    assert not torch.allclose(synced[2, 3:], synced[8, 3:])


def test_nudge_root_distances_moves_roots_toward_target() -> None:
    """Root coordinates move toward their configured target bond length."""
    config = DualRConstraintConfig(
        enabled=True,
        fixed_atom_count=2,
        pair_count=1,
        left_core_idx=0,
        right_core_idx=1,
        target_bond_length=1.5,
        distance_strength=1.0,
        max_distance_step=10.0,
    )
    z_lig = torch.zeros((4, 6), dtype=torch.float32)
    lig_mask = torch.zeros(4, dtype=torch.long)
    z_lig[0, :3] = torch.tensor([0.0, 0.0, 0.0])
    z_lig[1, :3] = torch.tensor([5.0, 0.0, 0.0])
    z_lig[2, :3] = torch.tensor([3.0, 0.0, 0.0])
    z_lig[3, :3] = torch.tensor([7.5, 0.0, 0.0])

    nudged = nudge_root_distances(z_lig, lig_mask, config, n_dims=3)

    left_distance = torch.linalg.norm(nudged[2, :3] - nudged[0, :3])
    right_distance = torch.linalg.norm(nudged[3, :3] - nudged[1, :3])
    assert torch.isclose(left_distance, torch.tensor(1.5), atol=1e-5)
    assert torch.isclose(right_distance, torch.tensor(1.5), atol=1e-5)


def test_apply_dual_r_latent_constraints_combines_h_and_distance() -> None:
    """Combined constraints synchronize h and adjust root distance together."""
    config = DualRConstraintConfig(
        enabled=True,
        fixed_atom_count=2,
        pair_count=1,
        left_core_idx=0,
        right_core_idx=1,
        target_bond_length=1.5,
        atom_sync_strength=1.0,
        distance_strength=1.0,
        max_distance_step=10.0,
    )
    z_lig = torch.zeros((4, 6), dtype=torch.float32)
    lig_mask = torch.zeros(4, dtype=torch.long)
    z_lig[0, :3] = torch.tensor([0.0, 0.0, 0.0])
    z_lig[1, :3] = torch.tensor([5.0, 0.0, 0.0])
    z_lig[2, :3] = torch.tensor([3.0, 0.0, 0.0])
    z_lig[3, :3] = torch.tensor([7.5, 0.0, 0.0])
    z_lig[2, 3:] = torch.tensor([4.0, 0.0, 0.0])
    z_lig[3, 3:] = torch.tensor([0.0, 4.0, 0.0])

    constrained = apply_dual_r_latent_constraints(
        z_lig,
        lig_mask,
        config,
        n_dims=3,
    )

    assert torch.allclose(constrained[2, 3:], constrained[3, 3:])
    assert torch.isclose(torch.linalg.norm(constrained[2, :3] - constrained[0, :3]), torch.tensor(1.5))
    assert torch.isclose(torch.linalg.norm(constrained[3, :3] - constrained[1, :3]), torch.tensor(1.5))


def test_should_apply_dual_r_constraints_respects_step_interval() -> None:
    """Step gating follows apply_every_n_steps."""
    config = DualRConstraintConfig(enabled=True, fixed_atom_count=2, pair_count=1, apply_every_n_steps=3)

    assert should_apply_dual_r_constraints(config, step=6)
    assert not should_apply_dual_r_constraints(config, step=5)
    assert not should_apply_dual_r_constraints(None, step=6)
