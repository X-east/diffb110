"""Unit tests for ddpo.constraints.symmetry — tensor layer + logprob + audit."""
from __future__ import annotations

import math
import pytest
import torch

from ddpo.constraints.symmetry import (
    SymmetryConstraintApplier,
    SymmetrySpec,
    build_symmetry_spec,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_spec(r1=(10, 11, 12), r2=(20, 21, 22), n_dims=3, atom_nf=5) -> SymmetrySpec:
    return SymmetrySpec(
        enabled=True,
        r1_indices=r1,
        r2_indices=r2,
        n_dims=n_dims,
        atom_nf=atom_nf,
    )


def _spec_config(r1, r2, **kw) -> dict:
    """Minimal YAML-like config dict."""
    return {
        "symmetry": {
            "enabled": True,
            "mode": "single_sidechain_copy",
            "r1_indices": list(r1),
            "r2_indices": list(r2),
            **kw,
        }
    }


# ---------------------------------------------------------------------------
# SymmetrySpec validation
# ---------------------------------------------------------------------------

class TestSpecValidation:
    def test_enabled_requires_indices(self):
        with pytest.raises(ValueError, match="r1_indices"):
            SymmetrySpec(enabled=True)

    def test_unequal_length_indices(self):
        with pytest.raises(ValueError, match="same length"):
            SymmetrySpec(enabled=True, r1_indices=(1,), r2_indices=(2, 3))

    def test_overlapping_indices(self):
        with pytest.raises(ValueError, match="disjoint"):
            SymmetrySpec(enabled=True, r1_indices=(1, 2), r2_indices=(2, 3))

    def test_negative_indices(self):
        with pytest.raises(ValueError, match="non-negative"):
            SymmetrySpec(enabled=True, r1_indices=(-1,), r2_indices=(0,))

    def test_duplicate_r1(self):
        with pytest.raises(ValueError, match="duplicates"):
            SymmetrySpec(enabled=True, r1_indices=(1, 1), r2_indices=(2, 3))

    def test_disabled_spec(self):
        spec = SymmetrySpec(enabled=False)
        assert spec.enabled is False


# ---------------------------------------------------------------------------
# config.py
# ---------------------------------------------------------------------------

class TestConfig:
    def test_disabled_returns_none(self):
        assert build_symmetry_spec({}) is None
        assert build_symmetry_spec({"symmetry": {"enabled": False}}) is None

    def test_minimal_yaml(self):
        spec = build_symmetry_spec(
            _spec_config(r1=[0, 1], r2=[10, 11]), n_dims=3, atom_nf=4
        )
        assert spec is not None
        assert spec.r1_indices == (0, 1)
        assert spec.r2_indices == (10, 11)
        assert spec.h_start == 3
        assert spec.atom_nf == 4

    def test_defaults(self):
        spec = build_symmetry_spec(
            _spec_config(r1=[0], r2=[1])
        )
        assert spec.mode == "single_sidechain_copy"
        assert spec.copy_schedule == "every_step"
        assert spec.topology_mode == "audit_only"


# ---------------------------------------------------------------------------
# SymmetryConstraintApplier — h-copy
# ---------------------------------------------------------------------------

class TestApplierHCopy:
    def test_h_copy_basic(self):
        spec = _make_spec(r1=(0, 1), r2=(2, 3), n_dims=3, atom_nf=2)
        applier = SymmetryConstraintApplier(spec)
        # [B=1, N=4, D=5]  (D = n_dims + atom_nf = 3+2=5)
        z = torch.randn(1, 4, 5)
        node_mask = torch.ones(1, 4, 1)
        z_out = applier.apply(z, node_mask)
        # R2 h (indices 2,3, dims 3:5) should equal R1 h (indices 0,1, dims 3:5)
        assert torch.equal(z_out[:, [2, 3], 3:5], z[:, [0, 1], 3:5])
        # x dims (0:3) should be unchanged for R2
        assert not torch.equal(z_out[:, [2, 3], :3], z[:, [0, 1], :3])

    def test_h_copy_disabled_returns_input(self):
        spec = SymmetrySpec(enabled=False)
        applier = SymmetryConstraintApplier(spec)
        z = torch.randn(1, 4, 5)
        assert torch.equal(applier.apply(z), z)

    def test_h_copy_respects_padding(self):
        spec = _make_spec(r1=(0,), r2=(1,), n_dims=3, atom_nf=2)
        applier = SymmetryConstraintApplier(spec)
        z = torch.randn(1, 3, 5)
        node_mask = torch.tensor([[[1.0], [1.0], [0.0]]])  # node 2 is padding
        z_out = applier.apply(z, node_mask)
        # padding node should be zeroed
        assert (z_out[:, 2, :] == 0).all()

    def test_h_copy_does_not_touch_x(self):
        spec = _make_spec(r1=(0,), r2=(1,), n_dims=3, atom_nf=2)
        applier = SymmetryConstraintApplier(spec)
        z = torch.randn(1, 2, 5)
        z_orig_x = z[:, :, :3].clone()
        z_out = applier.apply(z)
        assert torch.equal(z_out[:, :, :3], z_orig_x)

    def test_index_out_of_range(self):
        spec = _make_spec(r1=(0,), r2=(100,), n_dims=3, atom_nf=2)
        applier = SymmetryConstraintApplier(spec)
        z = torch.randn(1, 4, 5)
        with pytest.raises(ValueError, match="padded node dimension"):
            applier.validate_tensor(z)

    def test_h_start_out_of_range(self):
        spec = SymmetrySpec(enabled=True, r1_indices=(0,), r2_indices=(1,), n_dims=3, atom_nf=5)
        applier = SymmetryConstraintApplier(spec)
        z = torch.randn(1, 2, 3)  # D=3, h_start=3 → out of range
        with pytest.raises(ValueError, match="outside feature dimension"):
            applier.validate_tensor(z)


# ---------------------------------------------------------------------------
# SymmetryConstraintApplier — action_mask
# ---------------------------------------------------------------------------

class TestActionMask:
    def test_build_action_mask(self):
        spec = _make_spec(r1=(0, 1), r2=(2, 3), n_dims=3, atom_nf=2)
        applier = SymmetryConstraintApplier(spec)
        node_mask = torch.ones(2, 5, 1)  # [B=2, N=5, 1]
        am = applier.build_action_mask(node_mask, feature_dim=5)
        # Shape: [2, 5, 5]
        assert am.shape == (2, 5, 5)
        # R2 h dims (indices 2,3, dims 3:5) should be 0
        assert (am[:, [2, 3], 3:5] == 0).all()
        # R1 all dims should be 1
        assert (am[:, [0, 1], :] == 1).all()
        # R2 x dims (0:3) should be 1
        assert (am[:, [2, 3], :3] == 1).all()

    def test_action_mask_disabled_returns_full_mask(self):
        spec = SymmetrySpec(enabled=False)
        applier = SymmetryConstraintApplier(spec)
        node_mask = torch.ones(1, 3, 1)
        am = applier.build_action_mask(node_mask, feature_dim=4)
        assert am.shape == (1, 3, 4)
        assert (am == 1).all()


# ---------------------------------------------------------------------------
# Masked logprob — unit-level verification
# ---------------------------------------------------------------------------

class TestMaskedLogprob:
    """Test the logprob behaviour with action_mask.

    These tests verify the key invariance: changing R2.h should NOT affect
    logprob when action_mask zeros those dims, while changing R2.x or R1.h
    SHOULD change logprob.
    """

    def _compute_logprob(self, zs, mu, sigma, node_mask, action_mask=None):
        """Inline simplified logprob to avoid depending on DiffSBDDModel."""
        eps = 1e-6
        s = torch.max(sigma, torch.tensor(eps))
        log_p = -0.5 * ((zs.detach() - mu) ** 2) / (s ** 2)
        mask = action_mask if action_mask is not None else node_mask
        log_p = log_p * mask
        lp_sum = log_p.sum(dim=(1, 2))
        n_free = mask.sum(dim=(1, 2)) + eps
        return lp_sum / n_free

    def test_r2_h_delta_no_logprob_change(self):
        """Only changing R2 h should yield same logprob when action_mask zeros R2.h."""
        spec = _make_spec(r1=(0,), r2=(1,), n_dims=3, atom_nf=2)
        applier = SymmetryConstraintApplier(spec)
        node_mask = torch.ones(1, 2, 1)
        action_mask = applier.build_action_mask(node_mask, feature_dim=5)

        mu = torch.zeros(1, 2, 5)
        sigma = torch.tensor(1.0)
        zs = torch.randn(1, 2, 5)

        lp1 = self._compute_logprob(zs, mu, sigma, node_mask, action_mask)

        # Modify R2.h only (dim 3:5, atom 1)
        zs2 = zs.clone()
        zs2[:, 1, 3:5] += 10.0  # large change in R2.h
        lp2 = self._compute_logprob(zs2, mu, sigma, node_mask, action_mask)

        # Should be equal within tolerance (action_mask zeros those dims)
        assert torch.allclose(lp1, lp2, atol=1e-5), f"lp1={lp1} lp2={lp2}"

    def test_r2_x_delta_does_change_logprob(self):
        """Changing R2.x should change logprob (R2.x is free)."""
        spec = _make_spec(r1=(0,), r2=(1,), n_dims=3, atom_nf=2)
        applier = SymmetryConstraintApplier(spec)
        node_mask = torch.ones(1, 2, 1)
        action_mask = applier.build_action_mask(node_mask, feature_dim=5)

        mu = torch.zeros(1, 2, 5)
        sigma = torch.tensor(1.0)
        zs = torch.randn(1, 2, 5)

        lp1 = self._compute_logprob(zs, mu, sigma, node_mask, action_mask)
        zs2 = zs.clone()
        zs2[:, 1, :3] += 10.0  # R2.x change
        lp2 = self._compute_logprob(zs2, mu, sigma, node_mask, action_mask)

        assert not torch.allclose(lp1, lp2, atol=1e-5)

    def test_r1_h_delta_does_change_logprob(self):
        """Changing R1.h should change logprob (R1.h is free)."""
        spec = _make_spec(r1=(0,), r2=(1,), n_dims=3, atom_nf=2)
        applier = SymmetryConstraintApplier(spec)
        node_mask = torch.ones(1, 2, 1)
        action_mask = applier.build_action_mask(node_mask, feature_dim=5)

        mu = torch.zeros(1, 2, 5)
        sigma = torch.tensor(1.0)
        zs = torch.randn(1, 2, 5)

        lp1 = self._compute_logprob(zs, mu, sigma, node_mask, action_mask)
        zs2 = zs.clone()
        zs2[:, 0, 3:5] += 10.0  # R1.h change
        lp2 = self._compute_logprob(zs2, mu, sigma, node_mask, action_mask)

        assert not torch.allclose(lp1, lp2, atol=1e-5)

    def test_no_action_mask_uses_node_mask(self):
        """Without action_mask, node_mask broadcast behaviour is unchanged."""
        node_mask = torch.tensor([[[1.0], [1.0], [0.0]]])  # node 2 is padding
        mu = torch.zeros(1, 3, 5)
        sigma = torch.tensor(1.0)
        zs = torch.randn(1, 3, 5)

        lp = self._compute_logprob(zs, mu, sigma, node_mask, action_mask=None)
        assert lp.shape == (1,)
        assert not torch.isnan(lp).any()
        assert not torch.isinf(lp).any()
