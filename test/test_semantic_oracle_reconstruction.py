"""Tests for semantic-aware oracle reconstruction."""

from __future__ import annotations

import unittest

import torch
from rdkit import Chem

from DiffSBDD.edge_diffusion import build_oracle_mol, build_semantic_oracle_mol


class TestSemanticOracleReconstruction(unittest.TestCase):
    """Verify original SDF semantics reduce reconstruction repairs."""

    def test_aromatic_nh_metadata_avoids_repair(self):
        atom_symbols = ["C", "C", "C", "N", "C"]
        positions = torch.zeros((5, 3), dtype=torch.float32)
        halfedge_index = torch.tensor(
            [[0, 1, 2, 3, 0], [1, 2, 3, 4, 4]],
            dtype=torch.long,
        )
        halfedge_type = torch.tensor([4, 4, 4, 4, 4], dtype=torch.long)

        default_result = build_oracle_mol(
            atom_symbols,
            positions,
            halfedge_index,
            halfedge_type,
        )
        semantic_result = build_semantic_oracle_mol(
            atom_symbols,
            positions,
            halfedge_index,
            halfedge_type,
            explicit_hs=[0, 0, 0, 1, 0],
            no_implicit_flags=[0, 0, 0, 1, 0],
            aromatic_atom_flags=[1, 1, 1, 1, 1],
        )

        self.assertIsNone(default_result.sanitize_error)
        self.assertNotEqual(default_result.repair_strategy, "none")
        self.assertIsNone(semantic_result.sanitize_error)
        self.assertEqual(semantic_result.repair_strategy, "none")
        self.assertIsNotNone(semantic_result.mol)
        Chem.SanitizeMol(semantic_result.mol)


if __name__ == "__main__":
    unittest.main()
