"""Tests for original SDF audit helpers."""

from __future__ import annotations

import unittest
from pathlib import Path

from DiffSBDD.analysis.oracle_original_sdf_audit import ligand_sdf_relpath_from_name


class TestOracleOriginalSdfAudit(unittest.TestCase):
    """Verify CrossDocked processed names are parsed safely."""

    def test_ligand_sdf_relpath_from_name(self):
        name = (
            "ABC/1abc_A_rec_1abc_lig_pocket10.pdb_"
            "ABC/1abc_A_rec_1abc_lig.sdf"
        )

        self.assertEqual(
            ligand_sdf_relpath_from_name(name),
            Path("ABC/1abc_A_rec_1abc_lig.sdf"),
        )

    def test_invalid_name_raises(self):
        with self.assertRaisesRegex(ValueError, "Cannot parse"):
            ligand_sdf_relpath_from_name("not_a_crossdock_name")


if __name__ == "__main__":
    unittest.main()
