"""Tests for DiffSBDD fixed-core inpainting post-processing helpers."""

from __future__ import annotations

import sys
from pathlib import Path

from rdkit import Chem, Geometry
from rdkit.Chem import AllChem

DIFFSBDD_ROOT = Path(__file__).resolve().parents[1] / "DiffSBDD"
if str(DIFFSBDD_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFSBDD_ROOT))

from inpaint import _preserve_fixed_substructure_graph  # noqa: E402


def test_preserve_fixed_substructure_graph_restores_order_and_core_bonds() -> None:
    """Fixed atoms are restored even when reconstruction reorders atoms."""
    fixed = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"), addCoords=True)
    fixed = Chem.RemoveHs(fixed)
    assert AllChem.EmbedMolecule(fixed, randomSeed=17) == 0
    Chem.SanitizeMol(fixed)

    reordered = Chem.RWMol()
    old_order = [6, 0, 1, 2, 3, 4, 5]
    for old_idx in old_order:
        if old_idx == 6:
            reordered.AddAtom(Chem.Atom("N"))
        else:
            reordered.AddAtom(Chem.Atom("C"))
    for begin, end in [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1), (0, 1)]:
        reordered.AddBond(begin, end, Chem.BondType.SINGLE)

    conformer = Chem.Conformer(len(old_order))
    conformer.Set3D(True)
    fixed_conf = fixed.GetConformer()
    generated_pos = Geometry.Point3D(
        fixed_conf.GetAtomPosition(0).x + 1.4,
        fixed_conf.GetAtomPosition(0).y,
        fixed_conf.GetAtomPosition(0).z,
    )
    for new_idx, old_idx in enumerate(old_order):
        pos = generated_pos if old_idx == 6 else fixed_conf.GetAtomPosition(old_idx)
        conformer.SetAtomPosition(new_idx, Geometry.Point3D(float(pos.x), float(pos.y), float(pos.z)))
    reordered.AddConformer(conformer)

    preserved = _preserve_fixed_substructure_graph(reordered.GetMol(), fixed)

    assert preserved.GetNumAtoms() == 7
    assert [atom.GetSymbol() for atom in list(preserved.GetAtoms())[:6]] == ["C"] * 6
    assert preserved.GetAtomWithIdx(6).GetSymbol() == "N"
    for fixed_idx in range(6):
        expected = fixed_conf.GetAtomPosition(fixed_idx)
        actual = preserved.GetConformer().GetAtomPosition(fixed_idx)
        assert actual.Distance(expected) == 0.0
    assert sum(1 for bond in preserved.GetBonds() if bond.GetIsAromatic()) == 6
    assert preserved.GetBondBetweenAtoms(0, 6) is not None
