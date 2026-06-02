"""Tests for original/repaired oracle paired SDF export helpers."""

from __future__ import annotations

import argparse
import unittest
from pathlib import Path

from DiffSBDD.analysis.write_oracle_original_pair_sdf import (
    _format_sdf_value,
    build_sdf_properties,
    format_pair_id,
    make_record_name,
)


class TestOraclePairSdf(unittest.TestCase):
    """Verify paired SDF helper behavior."""

    def test_pair_and_record_names_are_stable(self):
        pair_id = format_pair_id(12)

        self.assertEqual(pair_id, "sample_000012")
        self.assertEqual(
            make_record_name(pair_id, "original", "/tmp/example_lig.sdf"),
            "sample_000012|original|example_lig",
        )

    def test_build_sdf_properties_switches_variant_perspective(self):
        row = _example_row()
        args = argparse.Namespace(
            npz_path=Path("data/example/train.npz"),
            dataset="crossdock_full",
        )

        original = build_sdf_properties(row, "original", args)
        repaired = build_sdf_properties(row, "repaired_oracle", args)

        self.assertEqual(original["this_smiles_no_stereo"], "CCN")
        self.assertEqual(original["counterpart_smiles_no_stereo"], "C[NH2+]C")
        self.assertEqual(repaired["this_smiles_no_stereo"], "C[NH2+]C")
        self.assertEqual(repaired["counterpart_smiles_no_stereo"], "CCN")
        self.assertEqual(original["visual_priority"], "mismatch")
        self.assertEqual(repaired["comparison_bond_type_map_match"], 0)

    def test_format_sdf_value_is_single_line(self):
        self.assertEqual(_format_sdf_value("a\nb"), "a b")
        self.assertEqual(_format_sdf_value({"b": 2, "a": 1}), '{"a": 1, "b": 2}')


def _example_row() -> dict[str, object]:
    """Return a minimal manifest row for property tests."""
    return {
        "pair_id": "sample_000003",
        "sample_idx": 3,
        "name": "processed_name",
        "original_sdf": "/tmp/example_lig.sdf",
        "original_record_name": "sample_000003|original|example_lig",
        "repaired_oracle_record_name": "sample_000003|repaired_oracle|example_lig",
        "repair_strategy": "fix_valence",
        "initial_sanitize_error": "AtomValenceException",
        "oracle_sanitize_error": "",
        "oracle_sanitize_success": 1,
        "original_load_success": 1,
        "original_load_error": "",
        "original_smiles_iso": "CCN",
        "oracle_smiles_iso": "C[NH2+]C",
        "original_smiles_no_stereo": "CCN",
        "oracle_smiles_no_stereo": "C[NH2+]C",
        "smiles_iso_match": 0,
        "smiles_no_stereo_match": 0,
        "num_atoms_original": 3,
        "num_atoms_oracle": 3,
        "num_atoms_match": 1,
        "num_bonds_original": 2,
        "num_bonds_oracle": 2,
        "num_bonds_match": 1,
        "total_formal_charge_original": 0,
        "total_formal_charge_oracle": 1,
        "total_formal_charge_match": 0,
        "formal_charge_vector_match": 0,
        "formal_charge_diff_count": 1,
        "aromatic_atom_count_original": 0,
        "aromatic_atom_count_oracle": 0,
        "aromatic_atom_count_match": 1,
        "aromatic_atom_vector_match": 1,
        "aromatic_atom_diff_count": 0,
        "aromatic_bond_count_original": 0,
        "aromatic_bond_count_oracle": 0,
        "aromatic_bond_count_match": 1,
        "bond_pair_set_match": 1,
        "bond_type_map_match": 0,
        "missing_bond_count": 0,
        "extra_bond_count": 0,
        "bond_type_mismatch_count": 1,
    }


if __name__ == "__main__":
    unittest.main()
