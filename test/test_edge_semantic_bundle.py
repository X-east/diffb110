"""Tests for original-SDF semantic edge bundle helpers."""

from __future__ import annotations

import unittest

from rdkit import Chem

from DiffSBDD.analysis.build_edge_semantic_bundle import (
    _bond_map_from_semantics,
    _compare_semantics,
    extract_original_semantics,
)


class TestEdgeSemanticBundle(unittest.TestCase):
    """Verify semantic metadata extraction from original molecules."""

    def test_extract_formal_charge_and_h_counts(self):
        mol = Chem.MolFromSmiles("C[NH3+]")

        semantics = extract_original_semantics(mol)

        self.assertEqual(semantics.total_formal_charge, 1)
        self.assertEqual(semantics.formal_charges.tolist(), [0, 1])
        self.assertEqual(semantics.total_hs.tolist(), [3, 3])
        self.assertEqual(semantics.atom_symbols.tolist(), ["C", "N"])

    def test_extract_aromatic_bond_types(self):
        mol = Chem.MolFromSmiles("c1ccccc1")

        semantics = extract_original_semantics(mol)

        self.assertEqual(int(semantics.aromatic_atom_flags.sum()), 6)
        self.assertEqual(int(semantics.aromatic_bond_flags.sum()), 6)
        self.assertEqual(set(semantics.bond_types.tolist()), {4})

    def test_compare_processed_to_original_semantics(self):
        mol = Chem.MolFromSmiles("CCO")
        semantics = extract_original_semantics(mol)
        original_bond_map = _bond_map_from_semantics(semantics)

        comparison = _compare_semantics(
            processed_symbols=["C", "C", "O"],
            processed_bond_map={(0, 1): 1, (1, 2): 1},
            semantics=semantics,
            original_bond_map=original_bond_map,
        )

        self.assertEqual(comparison["atom_count_match"], 1)
        self.assertEqual(comparison["atom_symbol_match"], 1)
        self.assertEqual(comparison["bond_pair_set_match"], 1)
        self.assertEqual(comparison["bond_type_map_match"], 1)


if __name__ == "__main__":
    unittest.main()
