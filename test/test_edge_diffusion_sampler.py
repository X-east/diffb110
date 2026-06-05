"""Tests for the standalone edge diffusion sampler smoke path."""

from __future__ import annotations

import ast
import inspect
import unittest

import torch

import DiffSBDD.edge_diffusion.sampler as sampler_module
from DiffSBDD.edge_diffusion import (
    EdgeBatch,
    EdgeCategoricalTransition,
    EdgeDiffusionModel,
    EdgeDiffusionModelConfig,
    EdgeDiffusionSampler,
    EdgeSamplerConfig,
    EdgeTransitionConfig,
)


class TestEdgeDiffusionSampler(unittest.TestCase):
    """Verify sampler contracts without touching DDPO/RL modules."""

    def test_reconstruct_sampler_preserves_shapes_and_label_ranges(self):
        batch = _make_toy_batch()
        model = EdgeDiffusionModel(EdgeDiffusionModelConfig(
            num_atom_types=4,
            num_edge_types=5,
            num_timesteps=8,
            hidden_dim=16,
            num_layers=1,
            distance_basis_dim=8,
        ))
        transition = EdgeCategoricalTransition(EdgeTransitionConfig(
            num_classes=5,
            num_timesteps=8,
            prior="uniform",
        ))
        sampler = EdgeDiffusionSampler(
            model=model,
            transition=transition,
            config=EdgeSamplerConfig(
                mode="reconstruct",
                start_timestep=2,
                pos_noise_std=0.01,
                atom_corruption_prob=0.05,
            ),
        )

        result = sampler.sample(batch, generator=torch.Generator().manual_seed(1))

        self.assertEqual(tuple(result.pred_pos.shape), tuple(batch.ligand_pos.shape))
        self.assertEqual(tuple(result.pred_atom_type.shape), tuple(batch.ligand_atom_type.shape))
        self.assertEqual(tuple(result.pred_edge_type.shape), tuple(batch.target_edge_type.shape))
        self.assertGreaterEqual(int(result.pred_atom_type.min()), 0)
        self.assertLess(int(result.pred_atom_type.max()), model.config.num_atom_types)
        self.assertGreaterEqual(int(result.pred_edge_type.min()), 0)
        self.assertLess(int(result.pred_edge_type.max()), model.config.num_edge_types)
        self.assertEqual(result.steps, 3)

    def test_prior_sampler_preserves_batch_contract(self):
        batch = _make_toy_batch()
        model = EdgeDiffusionModel(EdgeDiffusionModelConfig(
            num_atom_types=4,
            num_edge_types=5,
            num_timesteps=8,
            hidden_dim=16,
            num_layers=1,
            distance_basis_dim=8,
        ))
        transition = EdgeCategoricalTransition(EdgeTransitionConfig(
            num_classes=5,
            num_timesteps=8,
            prior="uniform",
        ))
        atom_prior = torch.tensor([0.7, 0.1, 0.1, 0.1], dtype=torch.float32)
        sampler = EdgeDiffusionSampler(
            model=model,
            transition=transition,
            config=EdgeSamplerConfig(mode="prior", start_timestep=1),
            atom_prior=atom_prior,
        )

        result = sampler.sample(batch, generator=torch.Generator().manual_seed(2))

        self.assertEqual(result.mode, "prior")
        self.assertEqual(tuple(result.pred_pos.shape), (5, 3))
        self.assertEqual(tuple(result.pred_edge_type.shape), (4,))
        self.assertTrue(torch.isfinite(result.pred_pos).all())

    def test_sampler_does_not_import_ddpo(self):
        tree = ast.parse(inspect.getsource(sampler_module))
        imported_modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_modules.append(node.module)
        self.assertFalse(
            any(module == "ddpo" or module.startswith("ddpo.") for module in imported_modules)
        )


def _make_toy_batch() -> EdgeBatch:
    """Create a two-molecule batch with upper-triangle halfedges."""
    return EdgeBatch(
        ligand_pos=torch.tensor([
            [0.0, 0.0, 0.0],
            [1.4, 0.0, 0.0],
            [2.7, 0.1, 0.0],
            [0.0, 0.0, 0.0],
            [1.2, 0.2, 0.0],
        ], dtype=torch.float32),
        ligand_atom_type=torch.tensor([0, 1, 2, 0, 3], dtype=torch.long),
        halfedge_index=torch.tensor([
            [0, 0, 1, 3],
            [1, 2, 2, 4],
        ], dtype=torch.long),
        target_edge_type=torch.tensor([1, 0, 2, 1], dtype=torch.long),
        atom_batch=torch.tensor([0, 0, 0, 1, 1], dtype=torch.long),
        edge_batch=torch.tensor([0, 0, 0, 1], dtype=torch.long),
        names=["toy_a", "toy_b"],
    )


if __name__ == "__main__":
    unittest.main()
