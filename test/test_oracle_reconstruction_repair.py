"""Tests for oracle reconstruction repair helpers."""

from __future__ import annotations

import unittest

import torch
from rdkit import Chem

from DiffSBDD.edge_diffusion.oracle_reconstruction import build_oracle_mol


class TestOracleReconstructionRepair(unittest.TestCase):
    """Verify common RDKit sanitize failures are repaired."""

    def test_n4_valence_is_repaired_with_formal_charge(self):
        atom_symbols = ["N", "C", "C", "C", "C"]
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.4, 0.0, 0.0],
                [-1.4, 0.0, 0.0],
                [0.0, 1.4, 0.0],
                [0.0, -1.4, 0.0],
            ],
            dtype=torch.float32,
        )
        halfedge_index = torch.triu_indices(5, 5, offset=1)
        halfedge_type = torch.zeros(halfedge_index.shape[1], dtype=torch.long)
        for edge_pos, (atom_i, atom_j) in enumerate(halfedge_index.t().tolist()):
            if 0 in (atom_i, atom_j):
                halfedge_type[edge_pos] = 1

        result = build_oracle_mol(atom_symbols, positions, halfedge_index, halfedge_type)

        self.assertIsNone(result.sanitize_error)
        self.assertEqual(result.initial_sanitize_error, "AtomValenceException")
        self.assertIn(result.repair_strategy, {"fix_valence", "fix_valence_after_aromatic"})
        self.assertIsNotNone(result.mol)
        self.assertEqual(result.mol.GetAtomWithIdx(0).GetFormalCharge(), 1)
        Chem.SanitizeMol(result.mol)


if __name__ == "__main__":
    unittest.main()
