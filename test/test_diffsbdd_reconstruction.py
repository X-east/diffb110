"""Tests for switchable DiffSBDD molecule reconstruction."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIFFSBDD_ROOT = PROJECT_ROOT / "DiffSBDD"
for extra in (PROJECT_ROOT, DIFFSBDD_ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from DiffSBDD.reconstruction import builder
from ddpo.utils import molecule_semantics


class TestReconstructionDispatch(unittest.TestCase):
    """Verify reconstruction method normalization and dispatch."""

    def test_openbabel_dispatch_is_default(self):
        positions = torch.zeros(1, 3)
        atom_types = torch.zeros(1, dtype=torch.long)
        dataset_info = {"atom_decoder": ["C"]}
        sentinel = object()

        with patch.object(builder, "make_mol_openbabel", return_value=sentinel) as mocked:
            mol = builder.reconstruct_molecule(positions, atom_types, dataset_info)

        self.assertIs(mol, sentinel)
        mocked.assert_called_once_with(positions, atom_types, dataset_info["atom_decoder"])

    def test_edm_alias_dispatches_to_distance_rule_builder(self):
        positions = torch.zeros(2, 3)
        atom_types = torch.zeros(2, dtype=torch.long)
        dataset_info = {"atom_decoder": ["C"]}
        sentinel = object()

        with patch.object(builder, "make_mol_edm", return_value=sentinel) as mocked:
            mol = builder.reconstruct_molecule(
                positions,
                atom_types,
                dataset_info,
                method="distance-rule",
                add_coords=True,
            )

        self.assertIs(mol, sentinel)
        mocked.assert_called_once_with(positions, atom_types, dataset_info, add_coords=True)

    def test_invalid_method_raises_clear_error(self):
        with self.assertRaisesRegex(ValueError, "Unsupported reconstruction method"):
            builder.normalize_reconstruction_method("made_up_method")

    def test_ddpo_semantics_uses_configured_reconstruction_method(self):
        positions = torch.zeros(1, 3)
        atom_types = torch.zeros(1, dtype=torch.long)
        dataset_info = {"atom_decoder": ["C"]}
        sentinel = object()

        with patch.object(
            molecule_semantics,
            "reconstruct_molecule",
            return_value=sentinel,
        ) as mocked:
            mol, ok, failure = molecule_semantics.build_rdmol_from_geometry(
                positions,
                atom_types,
                dataset_info,
                reconstruction_method="edm",
            )

        self.assertIs(mol, sentinel)
        self.assertTrue(ok)
        self.assertIsNone(failure)
        mocked.assert_called_once_with(
            positions=positions,
            atom_types=atom_types,
            dataset_info=dataset_info,
            add_coords=True,
            method="edm",
        )


if __name__ == "__main__":
    unittest.main()
