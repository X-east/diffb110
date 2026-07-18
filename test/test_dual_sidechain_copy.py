"""Unit tests for dual-sidechain copy and intramolecular clash check.

These tests are pure RDKit and do not require a GPU or a diffusion model.
They verify that a generated sidechain attached to one anchor can be replicated
onto the symmetric anchor, with correct bonding and no steric clash.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

from rdkit import Chem
from rdkit.Geometry import Point3D

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ddpo.constraints.dual_sidechain import (  # noqa: E402
    extract_sidechain_branch,
    intramolecular_clash_check,
    copy_sidechain_branch_to_anchor,
)

FIXED_CORE_SDF = (
    PROJECT_ROOT
    / "analysis/dual_sidechain_honokiol_9pc_20260617/inpaint_fixed_core_inputs/fixed_core_no_oh.sdf"
)
ONECOPY_SDF = (
    PROJECT_ROOT
    / "analysis/dual_sidechain_diffsbdd_onecopy_20260716/inpaint_raw.sdf"
)

ANCHOR1_IDX = 0  # R1 core atom in the fixed core / generated molecules
ANCHOR2_IDX = 11  # R2 core atom


def _load_fixed_core() -> Chem.Mol:
    """Read the honokiol fixed core (no phenolic OH) molecule."""
    mol = Chem.SDMolSupplier(str(FIXED_CORE_SDF), sanitize=False)[0]
    Chem.SanitizeMol(mol)
    return mol


def _ring_normal(mol: Chem.Mol, anchor_idx: int) -> tuple[float, float, float]:
    """Return the unit normal of the aromatic ring containing ``anchor_idx``."""
    ring_info = mol.GetRingInfo()
    target_ring = None
    for ring in ring_info.AtomRings():
        if anchor_idx in ring:
            target_ring = ring
            break
    if target_ring is None:
        return (0.0, 0.0, 1.0)
    conf = mol.GetConformer()
    pts = [conf.GetAtomPosition(i) for i in target_ring]
    nx = ny = nz = 0.0
    n = len(pts)
    for i in range(n):
        cur = pts[i]
        nxt = pts[(i + 1) % n]
        nx += (cur.y - nxt.y) * (cur.z + nxt.z)
        ny += (cur.z - nxt.z) * (cur.x + nxt.x)
        nz += (cur.x - nxt.x) * (cur.y + nxt.y)
    length = (nx * nx + ny * ny + nz * nz) ** 0.5
    if length < 1e-8:
        return (0.0, 0.0, 1.0)
    return (nx / length, ny / length, nz / length)


def _attach_ethyl(mol: Chem.Mol, anchor_idx: int) -> Chem.Mol:
    """Attach a two-carbon ethyl branch out of the ring plane for testing."""
    rw = Chem.RWMol(mol)
    conf = rw.GetConformer()
    p = conf.GetAtomPosition(anchor_idx)
    # Point away from the core centroid, projected onto the ring normal.
    positions = [
        conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms()) if i != anchor_idx
    ]
    cx = sum(q.x for q in positions) / len(positions)
    cy = sum(q.y for q in positions) / len(positions)
    cz = sum(q.z for q in positions) / len(positions)
    dx, dy, dz = p.x - cx, p.y - cy, p.z - cz
    nrm = _ring_normal(mol, anchor_idx)
    # component of (anchor-centroid) along the ring normal defines the sign
    sign = 1.0 if (dx * nrm[0] + dy * nrm[1] + dz * nrm[2]) >= 0 else -1.0
    ex, ey, ez = nrm[0] * sign, nrm[1] * sign, nrm[2] * sign
    root = rw.AddAtom(Chem.Atom("C"))
    child = rw.AddAtom(Chem.Atom("C"))
    rw.AddBond(anchor_idx, root, Chem.BondType.SINGLE)
    rw.AddBond(root, child, Chem.BondType.SINGLE)
    conf.SetAtomPosition(root, Point3D(p.x + ex * 1.50, p.y + ey * 1.50, p.z + ez * 1.50))
    conf.SetAtomPosition(
        child, Point3D(p.x + ex * 3.00, p.y + ey * 3.00, p.z + ez * 3.00)
    )
    out = rw.GetMol()
    Chem.SanitizeMol(out)
    return out


class DualSidechainCopyTest(unittest.TestCase):
    def test_copy_ethyl_makes_symmetric_molecule(self) -> None:
        core = _load_fixed_core()
        with_branch = _attach_ethyl(core, ANCHOR1_IDX)
        result = copy_sidechain_branch_to_anchor(
            with_branch, anchor1_idx=ANCHOR1_IDX, anchor2_idx=ANCHOR2_IDX
        )
        self.assertEqual(result.status, "success", result.error)
        self.assertIsNotNone(result.mol)
        self.assertFalse(result.clash)
        self.assertEqual(result.r1_atom_count, 2)
        self.assertEqual(result.r2_atom_count, 2)

        out = result.mol
        left = extract_sidechain_branch(out, core_atom_idx=ANCHOR1_IDX, label="R1")
        right = extract_sidechain_branch(out, core_atom_idx=ANCHOR2_IDX, label="R2")
        self.assertTrue(left.extractable)
        self.assertTrue(right.extractable)
        self.assertEqual(left.atom_count, right.atom_count)
        self.assertEqual(left.atom_symbols, right.atom_symbols)
        self.assertAlmostEqual(result.attachment_angle_1_deg, result.attachment_angle_2_deg, places=5)
        self.assertAlmostEqual(result.attachment_out_of_plane_deg, 0.0, places=5)
        self.assertGreater(result.attachment_angle_1_deg, 115.0)
        self.assertLess(result.attachment_angle_1_deg, 125.0)
        conf = out.GetConformer()
        for left_i, right_i in zip(left.atom_indices, right.atom_indices):
            for left_j, right_j in zip(left.atom_indices, right.atom_indices):
                left_distance = conf.GetAtomPosition(left_i).Distance(conf.GetAtomPosition(left_j))
                right_distance = conf.GetAtomPosition(right_i).Distance(conf.GetAtomPosition(right_j))
                self.assertAlmostEqual(left_distance, right_distance, places=6)

    def test_existing_lower_index_r2_is_removed_without_corrupting_r1(self) -> None:
        """Keep source indices valid when an earlier-added R2 branch is deleted."""
        core = _load_fixed_core()
        with_r2_first = _attach_ethyl(core, ANCHOR2_IDX)
        with_both = _attach_ethyl(with_r2_first, ANCHOR1_IDX)
        source = extract_sidechain_branch(with_both, core_atom_idx=ANCHOR1_IDX, label="R1")
        destination = extract_sidechain_branch(with_both, core_atom_idx=ANCHOR2_IDX, label="R2")
        self.assertGreater(min(source.atom_indices), max(destination.atom_indices))

        result = copy_sidechain_branch_to_anchor(
            with_both,
            anchor1_idx=ANCHOR1_IDX,
            anchor2_idx=ANCHOR2_IDX,
        )
        self.assertEqual(result.status, "success", result.error)
        self.assertEqual(result.r1_atom_count, 2)
        self.assertEqual(result.r2_atom_count, 2)
        copied = extract_sidechain_branch(result.mol, core_atom_idx=ANCHOR2_IDX, label="R2")
        self.assertEqual(copied.atom_symbols, ("C", "C"))

    def test_atom_identity_properties_are_copied(self) -> None:
        """Preserve isotope, atom-map, and custom atom properties on R2."""
        with_branch = _attach_ethyl(_load_fixed_core(), ANCHOR1_IDX)
        source = extract_sidechain_branch(with_branch, core_atom_idx=ANCHOR1_IDX, label="R1")
        source_root = with_branch.GetAtomWithIdx(source.root_atom_idx)
        source_root.SetIsotope(13)
        source_root.SetAtomMapNum(77)
        source_root.SetProp("copy_marker", "source_root")

        result = copy_sidechain_branch_to_anchor(
            with_branch,
            anchor1_idx=ANCHOR1_IDX,
            anchor2_idx=ANCHOR2_IDX,
        )
        self.assertEqual(result.status, "success", result.error)
        copied = extract_sidechain_branch(result.mol, core_atom_idx=ANCHOR2_IDX, label="R2")
        copied_root = result.mol.GetAtomWithIdx(copied.root_atom_idx)
        self.assertEqual(copied_root.GetIsotope(), 13)
        self.assertEqual(copied_root.GetAtomMapNum(), 77)
        self.assertEqual(copied_root.GetProp("copy_marker"), "source_root")

    def test_candidate9_cn_attachment_is_planar_and_angle_balanced(self) -> None:
        """Regress the real CN/CN candidate that exposed the translation error."""
        mols = [
            mol
            for mol in Chem.SDMolSupplier(str(ONECOPY_SDF), sanitize=False, removeHs=False)
            if mol is not None
        ]
        candidate = mols[9]
        Chem.SanitizeMol(candidate)
        source = extract_sidechain_branch(candidate, core_atom_idx=ANCHOR1_IDX, label="R1")
        self.assertTrue(source.extractable)
        source_conf = candidate.GetConformer()
        source_length = source_conf.GetAtomPosition(ANCHOR1_IDX).Distance(
            source_conf.GetAtomPosition(source.root_atom_idx)
        )

        result = copy_sidechain_branch_to_anchor(
            candidate,
            anchor1_idx=ANCHOR1_IDX,
            anchor2_idx=ANCHOR2_IDX,
        )
        self.assertEqual(result.status, "success", result.error)
        self.assertAlmostEqual(result.attachment_bond_length, source_length, places=5)
        self.assertAlmostEqual(result.attachment_angle_1_deg, 119.30283, places=4)
        self.assertAlmostEqual(result.attachment_angle_2_deg, 119.30283, places=4)
        self.assertAlmostEqual(result.attachment_out_of_plane_deg, 0.0, places=5)

    def test_r1_missing_reports_failure(self) -> None:
        core = _load_fixed_core()
        result = copy_sidechain_branch_to_anchor(
            core, anchor1_idx=ANCHOR1_IDX, anchor2_idx=ANCHOR2_IDX
        )
        self.assertEqual(result.status, "r1_missing")
        self.assertIsNone(result.mol)

    def test_clash_check_detects_overlap(self) -> None:
        core = _load_fixed_core()
        # Build a tiny molecule with two coincident heavy atoms.
        rw = Chem.RWMol()
        a = rw.AddAtom(Chem.Atom("C"))
        b = rw.AddAtom(Chem.Atom("C"))
        conf = Chem.Conformer(2)
        conf.SetAtomPosition(0, Point3D(0.0, 0.0, 0.0))
        conf.SetAtomPosition(1, Point3D(0.1, 0.0, 0.0))
        rw.AddConformer(conf, assignId=True)
        mol = rw.GetMol()
        has_clash, min_dist, _ = intramolecular_clash_check(mol, {a}, {b})
        self.assertTrue(has_clash)
        self.assertLess(min_dist, 0.5)

    def test_clash_check_uses_pair_specific_vdw_thresholds(self) -> None:
        """Detect a larger-radius clash even when another pair is closer."""
        rw = Chem.RWMol()
        carbon_a = rw.AddAtom(Chem.Atom("C"))
        fluorine = rw.AddAtom(Chem.Atom("F"))
        carbon_b = rw.AddAtom(Chem.Atom("C"))
        iodine = rw.AddAtom(Chem.Atom("I"))
        conf = Chem.Conformer(4)
        conf.SetAtomPosition(carbon_a, Point3D(0.0, 0.0, 0.0))
        conf.SetAtomPosition(fluorine, Point3D(2.23, 0.0, 0.0))
        conf.SetAtomPosition(carbon_b, Point3D(0.0, 5.0, 0.0))
        conf.SetAtomPosition(iodine, Point3D(2.50, 5.0, 0.0))
        rw.AddConformer(conf, assignId=True)
        mol = rw.GetMol()

        has_clash, distance, pair = intramolecular_clash_check(
            mol,
            {carbon_a, carbon_b},
            {fluorine, iodine},
        )
        self.assertTrue(has_clash)
        self.assertAlmostEqual(distance, 2.50, places=5)
        self.assertEqual(pair, (carbon_b, iodine))


if __name__ == "__main__":
    unittest.main(verbosity=2)
