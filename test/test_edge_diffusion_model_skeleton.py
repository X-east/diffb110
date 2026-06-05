"""Tests for the standalone x/h/e edge diffusion model skeleton."""

from __future__ import annotations

import unittest

import torch

from DiffSBDD.edge_diffusion import (
    EdgeBatch,
    EdgeDiffusionInputs,
    EdgeDiffusionLossConfig,
    EdgeDiffusionModel,
    EdgeDiffusionModelConfig,
    compute_edge_diffusion_loss,
)


class TestEdgeDiffusionModelSkeleton(unittest.TestCase):
    """Verify the decoupled edge diffusion x/h/e model contract."""

    def test_forward_shapes_and_backward(self):
        batch = _make_toy_batch()
        inputs = EdgeDiffusionInputs(
            ligand_pos_t=batch.ligand_pos + torch.tensor([
                [0.02, 0.00, 0.00],
                [0.00, 0.01, 0.00],
                [0.00, 0.00, 0.03],
            ]),
            ligand_atom_type_t=batch.ligand_atom_type,
            ligand_edge_type_t=torch.zeros_like(batch.target_edge_type),
            t=torch.tensor([5]),
        )
        model = EdgeDiffusionModel(EdgeDiffusionModelConfig(
            num_atom_types=3,
            hidden_dim=32,
            num_layers=2,
            distance_basis_dim=8,
        ))

        output = model(batch, inputs)
        loss = compute_edge_diffusion_loss(
            output,
            batch,
            EdgeDiffusionLossConfig(bond_length_weight=0.5),
        )
        loss.total.backward()

        self.assertEqual(tuple(output.pred_pos.shape), (3, 3))
        self.assertEqual(tuple(output.pred_atom_logits.shape), (3, 3))
        self.assertEqual(tuple(output.pred_edge_logits.shape), (3, 5))
        self.assertTrue(torch.isfinite(loss.total))
        self.assertTrue(torch.isfinite(loss.bond_length_loss))
        first_grad = next(parameter.grad for parameter in model.parameters() if parameter.grad is not None)
        self.assertTrue(torch.isfinite(first_grad).all())

    def test_loss_handles_no_real_bonds(self):
        batch = _make_toy_batch(target_edge_type=torch.zeros(3, dtype=torch.long))
        inputs = EdgeDiffusionInputs(
            ligand_pos_t=batch.ligand_pos,
            ligand_atom_type_t=batch.ligand_atom_type,
            ligand_edge_type_t=batch.target_edge_type,
            t=torch.tensor(0),
        )
        model = EdgeDiffusionModel(EdgeDiffusionModelConfig(
            num_atom_types=3,
            hidden_dim=16,
            num_layers=1,
            distance_basis_dim=8,
        ))

        output = model(batch, inputs)
        loss = compute_edge_diffusion_loss(output, batch)

        self.assertTrue(torch.isfinite(loss.total))
        self.assertEqual(float(loss.bond_length_loss), 0.0)

    def test_model_handles_single_atom_without_halfedges(self):
        batch = EdgeBatch(
            ligand_pos=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
            ligand_atom_type=torch.tensor([1], dtype=torch.long),
            halfedge_index=torch.empty((2, 0), dtype=torch.long),
            target_edge_type=torch.empty((0,), dtype=torch.long),
            atom_batch=torch.tensor([0], dtype=torch.long),
            edge_batch=torch.empty((0,), dtype=torch.long),
            names=["single_atom"],
        )
        inputs = EdgeDiffusionInputs(
            ligand_pos_t=batch.ligand_pos,
            ligand_atom_type_t=batch.ligand_atom_type,
            ligand_edge_type_t=batch.target_edge_type,
            t=torch.tensor([3]),
        )
        model = EdgeDiffusionModel(EdgeDiffusionModelConfig(
            num_atom_types=3,
            hidden_dim=16,
            num_layers=1,
            distance_basis_dim=8,
        ))

        output = model(batch, inputs)
        loss = compute_edge_diffusion_loss(output, batch)

        self.assertEqual(tuple(output.pred_pos.shape), (1, 3))
        self.assertEqual(tuple(output.pred_atom_logits.shape), (1, 3))
        self.assertEqual(tuple(output.pred_edge_logits.shape), (0, 5))
        self.assertTrue(torch.isfinite(loss.total))
        self.assertEqual(float(loss.edge_loss), 0.0)


def _make_toy_batch(target_edge_type: torch.Tensor | None = None) -> EdgeBatch:
    """Create a three-atom toy batch with full upper-triangle halfedges."""
    return EdgeBatch(
        ligand_pos=torch.tensor([
            [0.0, 0.0, 0.0],
            [1.4, 0.0, 0.0],
            [2.7, 0.1, 0.0],
        ], dtype=torch.float32),
        ligand_atom_type=torch.tensor([0, 1, 2], dtype=torch.long),
        halfedge_index=torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long),
        target_edge_type=torch.tensor([1, 0, 2], dtype=torch.long)
        if target_edge_type is None
        else target_edge_type,
        atom_batch=torch.zeros(3, dtype=torch.long),
        edge_batch=torch.zeros(3, dtype=torch.long),
        names=["toy"],
    )


if __name__ == "__main__":
    unittest.main()
