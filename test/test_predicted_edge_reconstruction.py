"""Tests for explicit halfedge-based molecule reconstruction."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIFFSBDD_ROOT = PROJECT_ROOT / "DiffSBDD"
for extra in (PROJECT_ROOT, DIFFSBDD_ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from DiffSBDD.reconstruction import reconstruct_molecule, reconstruct_molecule_from_edges


class TestPredictedEdgeReconstruction(unittest.TestCase):
    """Check RDKit reconstruction from explicit halfedge labels."""

    def test_predicted_edge_ethane_sanitizes(self):
        positions = torch.tensor([[0.0, 0.0, 0.0], [1.54, 0.0, 0.0]])
        atom_types = torch.tensor([0, 0], dtype=torch.long)
        halfedge_index = torch.tensor([[0], [1]], dtype=torch.long)
        halfedge_type = torch.tensor([1], dtype=torch.long)
        dataset_info = {"atom_decoder": ["C"]}

        mol = reconstruct_molecule(
            positions,
            atom_types,
            dataset_info,
            method="predicted_edge",
            add_coords=True,
            halfedge_index=halfedge_index,
            halfedge_type=halfedge_type,
        )
        Chem.SanitizeMol(mol)

        self.assertEqual(Chem.MolToSmiles(mol), "CC")
        self.assertEqual(mol.GetNumConformers(), 1)

    def test_oracle_edge_benzene_sanitizes(self):
        positions = torch.tensor([
            [1.40, 0.00, 0.00],
            [0.70, 1.21, 0.00],
            [-0.70, 1.21, 0.00],
            [-1.40, 0.00, 0.00],
            [-0.70, -1.21, 0.00],
            [0.70, -1.21, 0.00],
        ])
        atom_types = torch.zeros(6, dtype=torch.long)
        halfedge_index = torch.triu_indices(6, 6, offset=1)
        halfedge_type = torch.zeros(halfedge_index.shape[1], dtype=torch.long)
        aromatic_pairs = {
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (0, 5),
        }
        for edge_idx, pair in enumerate(zip(halfedge_index[0].tolist(), halfedge_index[1].tolist())):
            if pair in aromatic_pairs:
                halfedge_type[edge_idx] = 4
        dataset_info = {"atom_decoder": ["C"]}

        mol = reconstruct_molecule_from_edges(
            positions,
            atom_types,
            dataset_info,
            halfedge_index=halfedge_index,
            halfedge_type=halfedge_type,
            add_coords=True,
        )
        Chem.SanitizeMol(mol)

        self.assertEqual(Chem.MolToSmiles(mol), "c1ccccc1")
        self.assertTrue(all(atom.GetIsAromatic() for atom in mol.GetAtoms()))

    def test_edge_reconstruction_requires_halfedge_fields(self):
        with self.assertRaisesRegex(ValueError, "halfedge_index and halfedge_type"):
            reconstruct_molecule(
                torch.zeros(1, 3),
                torch.zeros(1, dtype=torch.long),
                {"atom_decoder": ["C"]},
                method="predicted_edge",
            )


if __name__ == "__main__":
    unittest.main()
