"""Tests for the minimal edge denoiser dataset and model loop."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from DiffSBDD.edge_diffusion import (
    EdgeBatchCollator,
    EdgeBundleDataset,
    EdgeDenoiserConfig,
    EdgeDenoiserMLP,
    EdgeDenoiserNodeEdge,
)
from DiffSBDD.edge_diffusion.denoiser import (
    edge_focal_cross_entropy,
    make_inverse_sqrt_class_weight,
)


class TestEdgeDenoiserMiniLoop(unittest.TestCase):
    """Verify edge bundle collation and denoiser shape contracts."""

    def test_dataset_and_collator_offsets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            edge_npz = _write_toy_edge_bundle(Path(tmpdir) / "toy_edge.npz")
            dataset = EdgeBundleDataset(edge_npz)
            batch = EdgeBatchCollator()([dataset[0], dataset[1]])

            self.assertEqual(batch.names, ["mol_a", "mol_b"])
            self.assertEqual(tuple(batch.ligand_pos.shape), (5, 3))
            self.assertTrue(torch.equal(
                batch.halfedge_index,
                torch.tensor([[0, 0, 1, 3], [1, 2, 2, 4]]),
            ))
            self.assertTrue(torch.equal(batch.target_edge_type, torch.tensor([1, 0, 2, 1])))
            self.assertTrue(torch.equal(batch.edge_batch, torch.tensor([0, 0, 0, 1])))

    def test_denoiser_forward_and_loss(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            edge_npz = _write_toy_edge_bundle(Path(tmpdir) / "toy_edge.npz")
            dataset = EdgeBundleDataset(edge_npz)
            batch = EdgeBatchCollator()([dataset[0], dataset[1]])
            model = EdgeDenoiserMLP(EdgeDenoiserConfig(num_atom_types=4, hidden_dim=16))
            noised_edge_type = torch.zeros_like(batch.target_edge_type)

            logits = model(batch, noised_edge_type, torch.tensor([3, 5]))
            class_weight = make_inverse_sqrt_class_weight(torch.tensor([1, 2, 1, 0, 0]))
            loss = edge_focal_cross_entropy(
                logits,
                batch.target_edge_type,
                class_weight=class_weight,
                gamma=1.0,
            )

            self.assertEqual(tuple(logits.shape), (4, 5))
            self.assertTrue(torch.isfinite(loss))

    def test_node_edge_denoiser_forward_and_loss(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            edge_npz = _write_toy_edge_bundle(Path(tmpdir) / "toy_edge.npz")
            dataset = EdgeBundleDataset(edge_npz)
            batch = EdgeBatchCollator()([dataset[0], dataset[1]])
            model = EdgeDenoiserNodeEdge(EdgeDenoiserConfig(
                num_atom_types=4,
                hidden_dim=16,
                num_layers=2,
                distance_basis_dim=8,
            ))
            noised_edge_type = torch.zeros_like(batch.target_edge_type)

            logits = model(batch, noised_edge_type, torch.tensor([3, 5]))
            loss = edge_focal_cross_entropy(logits, batch.target_edge_type)

            self.assertEqual(tuple(logits.shape), (4, 5))
            self.assertTrue(torch.isfinite(loss))


def _object_array(values: list[object]) -> np.ndarray:
    """Create a NumPy object array."""
    array = np.empty(len(values), dtype=object)
    for idx, value in enumerate(values):
        array[idx] = value
    return array


def _write_toy_edge_bundle(path: Path) -> Path:
    """Write a tiny edge bundle with two molecules."""
    np.savez_compressed(
        path,
        names=_object_array(["mol_a", "mol_b"]),
        ligand_pos=_object_array([
            np.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [2.4, 0.0, 0.0]], dtype=np.float32),
            np.asarray([[0.0, 1.0, 0.0], [1.4, 1.0, 0.0]], dtype=np.float32),
        ]),
        ligand_atom_type=_object_array([
            np.asarray([0, 1, 2], dtype=np.int64),
            np.asarray([0, 0], dtype=np.int64),
        ]),
        ligand_halfedge_index=_object_array([
            np.asarray([[0, 0, 1], [1, 2, 2]], dtype=np.int64),
            np.asarray([[0], [1]], dtype=np.int64),
        ]),
        ligand_halfedge_type=_object_array([
            np.asarray([1, 0, 2], dtype=np.int64),
            np.asarray([1], dtype=np.int64),
        ]),
    )
    return path


if __name__ == "__main__":
    unittest.main()
