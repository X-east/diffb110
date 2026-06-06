"""Tests for the optional DiffSBDD x/h/e interface path."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIFFSBDD_ROOT = PROJECT_ROOT / "DiffSBDD"
for extra in (PROJECT_ROOT, DIFFSBDD_ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from DiffSBDD.lightning_modules import LigandPocketDDPM
from DiffSBDD.train import validate_xhe_warm_start_policy
from equivariant_diffusion.conditional_model_xhe import BondDiffusionConfig, ConditionalDDPMXHE
from equivariant_diffusion.dynamics_xhe import EGNNDynamicsXHE, XHEEdgeHeadConfig


class _LigandPocketHarness:
    """Minimal object for testing LigandPocketDDPM batch conversion."""

    device = torch.device("cpu")
    virtual_nodes = False
    bond_diffusion_enabled = False
    bond_diffusion_config = BondDiffusionConfig(edge_loss_weight=1.0)
    get_ligand_and_pocket = LigandPocketDDPM.get_ligand_and_pocket
    _attach_ligand_halfedges = LigandPocketDDPM._attach_ligand_halfedges
    _apply_bond_diffusion_loss = LigandPocketDDPM._apply_bond_diffusion_loss


class TestDiffSBDDXHEInterfaces(unittest.TestCase):
    """Verify xhe fields and smoke logits stay decoupled from the old path."""

    def test_default_bond_config_matches_diffgui_mainline(self):
        config = BondDiffusionConfig.from_config(None, default_enabled=True)

        self.assertTrue(config.enabled)
        self.assertEqual(config.edge_prior, "absorb")
        self.assertEqual(config.beta_schedule, "segment")
        self.assertEqual(config.edge_loss_mode, "diffusion_kl")
        self.assertEqual(config.edge_class_weight_mode, "none")
        self.assertTrue(config.use_edge_message_passing)

    def test_xhe_warm_start_guard_rejects_xh_only_checkpoint(self):
        args = SimpleNamespace(diffusion_mode="xhe", xhe_warm_start_policy="xhe_only")

        with self.assertRaisesRegex(ValueError, "without xhe edge-head parameters"):
            validate_xhe_warm_start_policy(
                args,
                {"ddpm.dynamics.atom_encoder.weight": torch.zeros(1)},
                "xh.ckpt",
            )

        validate_xhe_warm_start_policy(
            args,
            {"ddpm.dynamics.chemical_edge_embedding.weight": torch.zeros(1)},
            "xhe.ckpt",
        )

    def test_xh_mode_does_not_require_halfedges(self):
        module = _LigandPocketHarness()
        module.bond_diffusion_enabled = False

        ligand, pocket = module.get_ligand_and_pocket(_make_batch(include_halfedges=False))

        self.assertNotIn("halfedge_index", ligand)
        self.assertEqual(tuple(ligand["x"].shape), (3, 3))
        self.assertEqual(tuple(pocket["x"].shape), (2, 3))

    def test_xhe_mode_requires_halfedge_fields(self):
        module = _LigandPocketHarness()
        module.bond_diffusion_enabled = True

        with self.assertRaisesRegex(ValueError, "requires halfedge batch fields"):
            module.get_ligand_and_pocket(_make_batch(include_halfedges=False))

    def test_xhe_mode_attaches_halfedge_fields(self):
        module = _LigandPocketHarness()
        module.bond_diffusion_enabled = True

        ligand, _ = module.get_ligand_and_pocket(_make_batch(include_halfedges=True))

        self.assertTrue(torch.equal(
            ligand["halfedge_index"],
            torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long),
        ))
        self.assertTrue(torch.equal(
            ligand["halfedge_type"],
            torch.tensor([1, 0, 2], dtype=torch.long),
        ))
        self.assertTrue(torch.equal(ligand["edge_batch"], torch.zeros(3, dtype=torch.long)))

    def test_dynamics_xhe_returns_old_outputs_by_default_and_edge_logits_on_request(self):
        dynamics = _make_dynamics()
        ligand, pocket = _make_ligand_pocket()
        xh_atoms = torch.cat([ligand["x"], ligand["one_hot"]], dim=-1)
        xh_residues = torch.cat([pocket["x"], pocket["one_hot"]], dim=-1)

        old_atoms, old_residues = dynamics(
            xh_atoms,
            xh_residues,
            torch.tensor([[0.25]]),
            ligand["mask"],
            pocket["mask"],
        )
        edge_atoms, edge_residues, edge_logits = dynamics(
            xh_atoms,
            xh_residues,
            torch.tensor([[0.25]]),
            ligand["mask"],
            pocket["mask"],
            halfedge_index=ligand["halfedge_index"],
            noised_edge_type=torch.zeros_like(ligand["halfedge_type"]),
            edge_batch=ligand["edge_batch"],
            return_edge_logits=True,
        )

        self.assertEqual(tuple(old_atoms.shape), tuple(edge_atoms.shape))
        self.assertEqual(tuple(old_residues.shape), tuple(edge_residues.shape))
        self.assertEqual(tuple(edge_logits.shape), (3, 5))
        self.assertTrue(torch.isfinite(edge_logits).all())

    def test_edge_message_passing_changes_xh_outputs(self):
        dynamics = _make_dynamics(use_edge_message_passing=True)
        ligand, pocket = _make_ligand_pocket()
        xh_atoms = torch.cat([ligand["x"], ligand["one_hot"]], dim=-1)
        xh_residues = torch.cat([pocket["x"], pocket["one_hot"]], dim=-1)

        atoms_zero, _, _ = dynamics(
            xh_atoms,
            xh_residues,
            torch.tensor([[0.25]]),
            ligand["mask"],
            pocket["mask"],
            halfedge_index=ligand["halfedge_index"],
            noised_edge_type=torch.zeros_like(ligand["halfedge_type"]),
            edge_batch=ligand["edge_batch"],
            return_edge_logits=True,
        )
        atoms_bond, _, _ = dynamics(
            xh_atoms,
            xh_residues,
            torch.tensor([[0.25]]),
            ligand["mask"],
            pocket["mask"],
            halfedge_index=ligand["halfedge_index"],
            noised_edge_type=ligand["halfedge_type"],
            edge_batch=ligand["edge_batch"],
            return_edge_logits=True,
        )

        self.assertFalse(torch.allclose(atoms_zero, atoms_bond))

    def test_conditional_ddpm_xhe_edge_smoke_loss_is_finite_and_backwardable(self):
        dynamics = _make_dynamics()
        ddpm = ConditionalDDPMXHE(
            dynamics=dynamics,
            atom_nf=4,
            residue_nf=3,
            n_dims=3,
            timesteps=8,
            noise_schedule="polynomial_2",
            noise_precision=1e-4,
            loss_type="l2",
            norm_values=(1.0, 1.0),
            size_histogram=np.ones((5, 5), dtype=np.float32),
            bond_diffusion_config=BondDiffusionConfig(
                enabled=True,
                num_edge_types=5,
                edge_hidden_dim=16,
                distance_basis_dim=8,
            ),
        )
        ligand, pocket = _make_ligand_pocket()

        output = ddpm.edge_forward_smoke(ligand, pocket)
        output.loss.backward()

        self.assertEqual(tuple(output.pred_edge_logits.shape), (3, 5))
        self.assertEqual(tuple(output.loss_per_molecule.shape), (1,))
        self.assertTrue(torch.isfinite(output.loss))
        self.assertTrue(torch.isfinite(output.real_bond_accuracy))
        first_grad = next(parameter.grad for parameter in ddpm.parameters() if parameter.grad is not None)
        self.assertTrue(torch.isfinite(first_grad).all())

    def test_conditional_ddpm_xhe_sampler_can_return_edge_bundle(self):
        dynamics = _make_dynamics(use_edge_message_passing=True)
        ddpm = ConditionalDDPMXHE(
            dynamics=dynamics,
            atom_nf=4,
            residue_nf=3,
            n_dims=3,
            timesteps=4,
            noise_schedule="polynomial_2",
            noise_precision=1e-4,
            loss_type="l2",
            norm_values=(1.0, 1.0),
            size_histogram=np.ones((5, 5), dtype=np.float32),
            bond_diffusion_config=BondDiffusionConfig(
                enabled=True,
                num_edge_types=5,
                edge_hidden_dim=16,
                distance_basis_dim=8,
                use_edge_message_passing=True,
                edge_message_dim=8,
            ),
        )
        pocket = {
            "x": torch.tensor([[0.0, 0.0, 1.0], [0.5, 1.0, 1.0]], dtype=torch.float32),
            "one_hot": torch.eye(3, dtype=torch.float32)[:2],
            "size": torch.tensor([2], dtype=torch.long),
            "mask": torch.zeros(2, dtype=torch.long),
        }

        xh_lig, xh_pocket, lig_mask, pocket_mask, edge_bundle = ddpm.sample_given_pocket(
            pocket,
            torch.tensor([3], dtype=torch.long),
            timesteps=2,
            return_edges=True,
            edge_temperature=0.0,
            generator=torch.Generator().manual_seed(3),
        )
        old_contract = ddpm.sample_given_pocket(
            pocket,
            torch.tensor([3], dtype=torch.long),
            timesteps=2,
            return_edges=False,
            edge_temperature=0.0,
            generator=torch.Generator().manual_seed(3),
        )

        self.assertEqual(tuple(xh_lig.shape), (3, 7))
        self.assertEqual(tuple(xh_pocket.shape), (2, 6))
        self.assertEqual(tuple(lig_mask.shape), (3,))
        self.assertEqual(tuple(pocket_mask.shape), (2,))
        self.assertEqual(tuple(edge_bundle["halfedge_index"].shape), (2, 3))
        self.assertEqual(tuple(edge_bundle["halfedge_type"].shape), (3,))
        self.assertEqual(tuple(edge_bundle["edge_batch"].shape), (3,))
        self.assertEqual(tuple(edge_bundle["num_halfedges"].shape), (1,))
        self.assertEqual(len(old_contract), 4)

    def test_edge_loss_is_added_to_nll_only_in_xhe_mode(self):
        module = _LigandPocketHarness()
        module.bond_diffusion_enabled = True
        module.bond_diffusion_config = BondDiffusionConfig(edge_loss_weight=2.0)
        nll = torch.tensor([1.0, 2.0])
        info = {"_edge_loss_batch": torch.tensor([0.5, 1.0])}

        updated = module._apply_bond_diffusion_loss(nll, info)

        self.assertTrue(torch.equal(updated, torch.tensor([2.0, 4.0])))
        self.assertNotIn("_edge_loss_batch", info)
        self.assertEqual(float(info["edge_loss_weight"]), 2.0)
        self.assertEqual(float(info["edge_loss_weighted"]), 1.5)

    def test_edge_loss_private_field_is_ignored_in_xh_mode(self):
        module = _LigandPocketHarness()
        module.bond_diffusion_enabled = False
        nll = torch.tensor([1.0, 2.0])
        info = {"_edge_loss_batch": torch.tensor([0.5, 1.0])}

        updated = module._apply_bond_diffusion_loss(nll, info)

        self.assertTrue(torch.equal(updated, nll))
        self.assertNotIn("_edge_loss_batch", info)
        self.assertNotIn("edge_loss_weighted", info)


def _make_batch(include_halfedges: bool) -> dict[str, torch.Tensor]:
    """Return a small collated DiffSBDD-like batch."""
    batch = {
        "lig_coords": torch.tensor([
            [0.0, 0.0, 0.0],
            [1.4, 0.0, 0.0],
            [2.7, 0.1, 0.0],
        ], dtype=torch.float32),
        "lig_one_hot": torch.eye(4, dtype=torch.float32)[:3],
        "num_lig_atoms": torch.tensor([3], dtype=torch.long),
        "lig_mask": torch.zeros(3, dtype=torch.long),
        "pocket_coords": torch.tensor([
            [0.0, 0.0, 1.0],
            [0.5, 1.0, 1.0],
        ], dtype=torch.float32),
        "pocket_one_hot": torch.eye(3, dtype=torch.float32)[:2],
        "num_pocket_nodes": torch.tensor([2], dtype=torch.long),
        "pocket_mask": torch.zeros(2, dtype=torch.long),
    }
    if include_halfedges:
        batch.update({
            "ligand_halfedge_index": torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long),
            "ligand_halfedge_type": torch.tensor([1, 0, 2], dtype=torch.long),
            "ligand_halfedge_type_batch": torch.zeros(3, dtype=torch.long),
            "num_ligand_halfedges": torch.tensor([3], dtype=torch.long),
        })
    return batch


def _make_ligand_pocket() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Return ligand and pocket dicts with xhe fields."""
    batch = _make_batch(include_halfedges=True)
    module = _LigandPocketHarness()
    module.bond_diffusion_enabled = True
    return module.get_ligand_and_pocket(batch)


def _make_dynamics(use_edge_message_passing: bool = False) -> EGNNDynamicsXHE:
    """Build a tiny xhe dynamics module for CPU smoke tests."""
    return EGNNDynamicsXHE(
        atom_nf=4,
        residue_nf=3,
        n_dims=3,
        joint_nf=8,
        hidden_nf=16,
        n_layers=1,
        attention=False,
        tanh=False,
        norm_constant=1,
        inv_sublayers=1,
        normalization_factor=10,
        aggregation_method="sum",
        update_pocket_coords=False,
        reflection_equivariant=True,
        xhe_edge_head_config=XHEEdgeHeadConfig(
            num_edge_types=5,
            num_timesteps=8,
            hidden_dim=16,
            distance_basis_dim=8,
            use_edge_message_passing=use_edge_message_passing,
            edge_message_dim=8,
        ),
    )


if __name__ == "__main__":
    unittest.main()
