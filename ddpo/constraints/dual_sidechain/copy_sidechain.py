"""Post-generation symmetric sidechain copy and intramolecular clash check.

Implements the "generate one sidechain, copy to the other anchor" strategy for
dual-sidechain molecules such as honokiol derivatives:

1. Generate a sidechain attached to one aromatic core anchor (R1) with a
   fragment-conditioned diffusion model (e.g. DiffSBDD inpaint with a fixed
   core).
2. Replicate that branch onto the second aromatic anchor (R2) with one proper
   rigid transform. The R2 attachment bond is placed along the in-plane outward
   bisector of the two aromatic ring bonds, while the copied branch keeps the
   exact internal 3D geometry of R1. A pre-existing branch at R2 is removed
   first, guaranteeing that both sides have identical branch chemistry.
3. A van-der-Waals clash check rejects copies that overlap the core or the
   original branch. Protein clashes are intentionally ignored: the docking
   backend performs energy optimization during scoring.

The copy is a pure RDKit post-processing step and does not require any
in-diffusion symmetry constraint.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

from ddpo.constraints.dual_sidechain.audit import extract_sidechain_branch

VDW_RADII: dict[str, float] = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "Cl": 1.75,
    "Br": 1.85,
    "I": 1.98,
    "B": 1.92,
    "Si": 2.10,
}

DEFAULT_CLASH_FRACTION = 0.70  # min allowed distance = fraction * (r_i + r_j)
GEOMETRY_EPS = 1.0e-8
PLANARITY_TOLERANCE_DEG = 1.0e-4
ANGLE_EQUALITY_TOLERANCE_DEG = 1.0e-4


def _vdw_radius(symbol: str) -> float:
    """Return the van-der-Waals radius for an element symbol."""
    return VDW_RADII.get(symbol, 1.70)


def _position_array(conf: Chem.Conformer, atom_idx: int) -> np.ndarray:
    """Return one conformer position as a float64 NumPy vector."""
    point = conf.GetAtomPosition(atom_idx)
    return np.asarray([point.x, point.y, point.z], dtype=np.float64)


def _unit_vector(vector: np.ndarray, *, label: str) -> np.ndarray:
    """Return a normalized vector and reject degenerate geometry."""
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm < GEOMETRY_EPS:
        raise ValueError(f"degenerate vector for {label}")
    return vector / norm


def _angle_degrees(left: np.ndarray, right: np.ndarray) -> float:
    """Return the unsigned angle between two vectors in degrees."""
    left_unit = _unit_vector(left, label="angle_left")
    right_unit = _unit_vector(right, label="angle_right")
    cosine = float(np.clip(np.dot(left_unit, right_unit), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _aromatic_ring_neighbors(mol: Chem.Mol, anchor_idx: int) -> tuple[int, int]:
    """Return the two aromatic ring neighbors of one substitution anchor."""
    anchor = mol.GetAtomWithIdx(anchor_idx)
    neighbors: list[int] = []
    for neighbor in anchor.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(anchor_idx, neighbor.GetIdx())
        if neighbor.GetIsAromatic() and bond is not None and bond.GetIsAromatic():
            neighbors.append(neighbor.GetIdx())
    if len(neighbors) != 2:
        raise ValueError(
            f"anchor {anchor_idx} must have exactly two aromatic ring neighbors, found {neighbors}"
        )
    return tuple(sorted(neighbors))


def _ring_geometry(
    mol: Chem.Mol,
    conf: Chem.Conformer,
    anchor_idx: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Return the in-plane outward bisector and ring normal at an anchor."""
    neighbor_indices = _aromatic_ring_neighbors(mol, anchor_idx)
    anchor_pos = _position_array(conf, anchor_idx)
    first = _unit_vector(
        _position_array(conf, neighbor_indices[0]) - anchor_pos,
        label="ring_neighbor_first",
    )
    second = _unit_vector(
        _position_array(conf, neighbor_indices[1]) - anchor_pos,
        label="ring_neighbor_second",
    )
    outward = _unit_vector(-(first + second), label="ring_outward_bisector")
    normal = _unit_vector(np.cross(first, second), label="ring_normal")
    # Numerical projection guarantees that the target bond is exactly in-plane.
    outward = _unit_vector(outward - np.dot(outward, normal) * normal, label="planar_outward")
    return outward, normal, neighbor_indices


def _attachment_frame(
    attachment_direction: np.ndarray,
    ring_normal: np.ndarray,
) -> np.ndarray:
    """Build a right-handed frame whose x axis is the attachment bond."""
    x_axis = _unit_vector(attachment_direction, label="attachment_axis")
    z_projection = ring_normal - np.dot(ring_normal, x_axis) * x_axis
    if float(np.linalg.norm(z_projection)) < GEOMETRY_EPS:
        # This only occurs for a pathological source bond normal to its ring.
        fallback = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(fallback, x_axis))) > 0.9:
            fallback = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        z_projection = fallback - np.dot(fallback, x_axis) * x_axis
    z_axis = _unit_vector(z_projection, label="attachment_frame_z")
    y_axis = _unit_vector(np.cross(z_axis, x_axis), label="attachment_frame_y")
    z_axis = _unit_vector(np.cross(x_axis, y_axis), label="attachment_frame_z_rebuilt")
    frame = np.column_stack([x_axis, y_axis, z_axis])
    if float(np.linalg.det(frame)) < 0.0:
        raise ValueError("attachment frame must be a proper rotation")
    return frame


def _remap_after_removals(atom_idx: int, removed_indices: set[int]) -> int:
    """Map an original atom index after lower-index atoms have been removed."""
    if atom_idx in removed_indices:
        raise ValueError(f"atom {atom_idx} was removed")
    return atom_idx - sum(removed_idx < atom_idx for removed_idx in removed_indices)


@dataclass(frozen=True)
class _BondCopyFacts:
    """Stored RDKit bond attributes used after destination-branch deletion."""

    begin_idx: int
    end_idx: int
    bond_type: Chem.BondType
    is_aromatic: bool
    is_conjugated: bool
    bond_dir: Chem.BondDir
    stereo: Chem.BondStereo
    stereo_atoms: tuple[int, ...]
    properties: tuple[tuple[str, str], ...]


def _bond_copy_facts(bond: Chem.Bond) -> _BondCopyFacts:
    """Capture one bond before atom indices are changed."""
    properties = tuple(
        (name, bond.GetProp(name))
        for name in bond.GetPropNames(includePrivate=True, includeComputed=False)
    )
    return _BondCopyFacts(
        begin_idx=bond.GetBeginAtomIdx(),
        end_idx=bond.GetEndAtomIdx(),
        bond_type=bond.GetBondType(),
        is_aromatic=bool(bond.GetIsAromatic()),
        is_conjugated=bool(bond.GetIsConjugated()),
        bond_dir=bond.GetBondDir(),
        stereo=bond.GetStereo(),
        stereo_atoms=tuple(int(idx) for idx in bond.GetStereoAtoms()),
        properties=properties,
    )


def _apply_bond_facts(
    bond: Chem.Bond,
    facts: _BondCopyFacts,
    atom_mapping: dict[int, int],
) -> None:
    """Restore stored bond metadata after adding a copied bond."""
    bond.SetIsAromatic(facts.is_aromatic)
    bond.SetIsConjugated(facts.is_conjugated)
    bond.SetBondDir(facts.bond_dir)
    if len(facts.stereo_atoms) == 2 and all(idx in atom_mapping for idx in facts.stereo_atoms):
        bond.SetStereoAtoms(
            atom_mapping[facts.stereo_atoms[0]],
            atom_mapping[facts.stereo_atoms[1]],
        )
        bond.SetStereo(facts.stereo)
    for name, value in facts.properties:
        bond.SetProp(name, value)


def _attachment_geometry(
    mol: Chem.Mol,
    anchor_idx: int,
    root_idx: int,
) -> tuple[float, float, float, float]:
    """Return bond length, two ring angles, and out-of-plane angle."""
    conf = mol.GetConformer()
    anchor_pos = _position_array(conf, anchor_idx)
    root_vector = _position_array(conf, root_idx) - anchor_pos
    _, normal, neighbor_indices = _ring_geometry(mol, conf, anchor_idx)
    first_vector = _position_array(conf, neighbor_indices[0]) - anchor_pos
    second_vector = _position_array(conf, neighbor_indices[1]) - anchor_pos
    root_unit = _unit_vector(root_vector, label="attachment_root")
    out_of_plane = float(np.degrees(np.arcsin(np.clip(abs(np.dot(root_unit, normal)), 0.0, 1.0))))
    return (
        float(np.linalg.norm(root_vector)),
        _angle_degrees(root_vector, first_vector),
        _angle_degrees(root_vector, second_vector),
        out_of_plane,
    )


def _within_two_bonds(mol: Chem.Mol, start: int, target: int) -> bool:
    """Return True if ``target`` is reachable from ``start`` within two bonds."""
    if start == target:
        return True
    visited: dict[int, int] = {start: 0}
    queue: deque[int] = deque([start])
    while queue:
        current = queue.popleft()
        depth = visited[current]
        if depth >= 2:
            continue
        for neighbor in mol.GetAtomWithIdx(current).GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx == target:
                return True
            if nbr_idx not in visited:
                visited[nbr_idx] = depth + 1
                queue.append(nbr_idx)
    return False


def intramolecular_clash_check(
    mol: Chem.Mol,
    group_a: set[int],
    group_b: set[int],
    clash_fraction: float = DEFAULT_CLASH_FRACTION,
) -> tuple[bool, float, tuple[int, int] | None]:
    """Check for steric clashes between two atom groups of one molecule.

    Returns ``(has_clash, min_distance, closest_pair)``. A pair is a clash when
    its distance is below ``clash_fraction * (r_i + r_j)`` and the two atoms are
    not within two bonds of each other. Directly bonded pairs and pairs that
    share a common neighbor (graph distance <= 2) are skipped, because a normal
    substituent's first atom legitimately sits ~2.0 A from the adjacent ring
    carbons and must not be flagged as a clash.
    """
    if mol.GetNumConformers() == 0:
        return False, float("inf"), None
    conf = mol.GetConformer()
    min_dist = float("inf")
    closest: tuple[int, int] | None = None
    worst_clearance = float("inf")
    worst_pair: tuple[int, int] | None = None
    worst_distance = float("inf")
    for i in group_a:
        if i < 0 or i >= mol.GetNumAtoms():
            continue
        pi = conf.GetAtomPosition(i)
        for j in group_b:
            if j < 0 or j >= mol.GetNumAtoms() or j == i:
                continue
            if mol.GetBondBetweenAtoms(i, j) is not None:
                continue
            if _within_two_bonds(mol, i, j):
                continue
            d = pi.Distance(conf.GetAtomPosition(j))
            if d < min_dist:
                min_dist = d
                closest = (i, j)
            threshold = clash_fraction * (
                _vdw_radius(mol.GetAtomWithIdx(i).GetSymbol())
                + _vdw_radius(mol.GetAtomWithIdx(j).GetSymbol())
            )
            clearance = d - threshold
            if clearance < worst_clearance:
                worst_clearance = clearance
                worst_pair = (i, j)
                worst_distance = d
    if min_dist == float("inf") or closest is None or worst_pair is None:
        return False, float("inf"), None
    if worst_clearance < 0.0:
        return True, worst_distance, worst_pair
    return False, min_dist, closest


@dataclass
class SidechainCopyResult:
    """Result of copying one sidechain branch onto the symmetric anchor."""

    mol: Chem.Mol | None
    status: str
    r1_atom_count: int = 0
    r2_atom_count: int = 0
    clash: bool = False
    min_clash_distance: float = float("inf")
    closest_pair: tuple[int, int] | None = None
    attachment_bond_length: float = float("nan")
    attachment_angle_1_deg: float = float("nan")
    attachment_angle_2_deg: float = float("nan")
    attachment_out_of_plane_deg: float = float("nan")
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a flat, JSON-serializable summary."""
        return {
            "status": self.status,
            "r1_atom_count": int(self.r1_atom_count),
            "r2_atom_count": int(self.r2_atom_count),
            "clash": int(self.clash),
            "min_clash_distance": float(self.min_clash_distance),
            "closest_pair": "" if self.closest_pair is None else " ".join(
                str(i) for i in self.closest_pair
            ),
            "attachment_bond_length": float(self.attachment_bond_length),
            "attachment_angle_1_deg": float(self.attachment_angle_1_deg),
            "attachment_angle_2_deg": float(self.attachment_angle_2_deg),
            "attachment_out_of_plane_deg": float(self.attachment_out_of_plane_deg),
            "error": self.error,
        }


def copy_sidechain_branch_to_anchor(
    mol: Chem.Mol,
    *,
    anchor1_idx: int,
    anchor2_idx: int,
    r1_atom_indices: tuple[int, ...] | None = None,
    bond_type: Chem.BondType | None = None,
    clash_fraction: float = DEFAULT_CLASH_FRACTION,
    sanitize: bool = True,
) -> SidechainCopyResult:
    """Copy the sidechain branch at ``anchor1_idx`` onto ``anchor2_idx``.

    The branch attached to ``anchor1_idx`` (R1) is replicated onto
    ``anchor2_idx`` (R2) with a proper rigid transform. The R2 attachment bond
    is forced into the aromatic-ring plane along the outward bisector of its two
    ring-neighbor bonds. This gives equal local angles (exactly 120 degrees for
    an ideal benzene ring) without distorting any internal R1 bond length,
    angle, or dihedral. Any pre-existing R2 branch is removed first. A
    van-der-Waals clash check then rejects copies that overlap the core or R1.

    Args:
        mol: Input molecule with a fixed core and a (possibly partial) branch at
            each anchor.
        anchor1_idx: Core atom index of the source branch (R1).
        anchor2_idx: Core atom index of the destination anchor (R2).
        r1_atom_indices: Optional explicit R1 atom indices (excluding the
            anchor). When omitted, the branch is detected automatically.
        bond_type: Optional bond-type override for the copied root attachment.
            When omitted, the R1 attachment bond type is preserved.
        clash_fraction: Minimum allowed inter-group distance as a fraction of
            the summed van-der-Waals radii.
        sanitize: Whether to sanitize the output molecule.

    Returns:
        A :class:`SidechainCopyResult`. ``mol`` is the symmetric molecule on
        success (``status == "success"``) or ``None`` on hard failure.
    """
    working = Chem.RWMol(mol)
    try:
        Chem.SanitizeMol(working)
    except Exception:
        # Continue; the molecule is re-sanitized at the end.
        pass
    if working.GetNumConformers() == 0:
        return SidechainCopyResult(mol=None, status="no_conformer", error="input has no conformer")

    # 1. Identify the R1 branch (source).
    if r1_atom_indices is None:
        left = extract_sidechain_branch(working, core_atom_idx=anchor1_idx, label="R1")
        if not left.extractable:
            return SidechainCopyResult(
                mol=None, status="r1_missing", error=left.failure_type
            )
        r1_atom_indices = left.atom_indices
        r1_root = left.root_atom_idx
    else:
        r1_set_in = set(r1_atom_indices)
        r1_root = None
        for neighbor in working.GetAtomWithIdx(anchor1_idx).GetNeighbors():
            if neighbor.GetIdx() in r1_set_in:
                r1_root = neighbor.GetIdx()
                break
        if r1_root is None:
            return SidechainCopyResult(
                mol=None,
                status="r1_not_attached",
                error="R1 branch not bonded to anchor1",
            )
    r1_set = set(r1_atom_indices)

    if anchor1_idx == anchor2_idx or anchor1_idx in r1_set or anchor2_idx in r1_set:
        return SidechainCopyResult(
            mol=None,
            status="invalid_indices",
            error="anchors must be distinct core atoms outside the R1 branch",
        )

    try:
        source_conf = working.GetConformer()
        source_anchor_pos = _position_array(source_conf, anchor1_idx)
        source_root_pos = _position_array(source_conf, r1_root)
        source_direction = source_root_pos - source_anchor_pos
        source_bond_length = float(np.linalg.norm(source_direction))
        _, source_ring_normal, _ = _ring_geometry(working, source_conf, anchor1_idx)
        source_frame = _attachment_frame(source_direction, source_ring_normal)
    except Exception as exc:  # noqa: BLE001 - return a structured geometry failure.
        return SidechainCopyResult(mol=None, status="source_geometry_failed", error=str(exc))

    source_positions = {
        idx: _position_array(source_conf, idx)
        for idx in r1_atom_indices
    }
    source_atoms = {
        idx: Chem.Atom(working.GetAtomWithIdx(idx))
        for idx in r1_atom_indices
    }
    internal_bonds = [
        _bond_copy_facts(bond)
        for bond in working.GetBonds()
        if bond.GetBeginAtomIdx() in r1_set and bond.GetEndAtomIdx() in r1_set
    ]
    source_attachment = working.GetBondBetweenAtoms(anchor1_idx, r1_root)
    if source_attachment is None:
        return SidechainCopyResult(
            mol=None,
            status="r1_not_attached",
            error="R1 attachment bond disappeared before copying",
        )
    attachment_facts = _bond_copy_facts(source_attachment)

    # 2. Remove any pre-existing branch at anchor2 so both sides equal R1.
    right = extract_sidechain_branch(working, core_atom_idx=anchor2_idx, label="R2")
    if not right.extractable and right.failure_type != "attachment_missing":
        return SidechainCopyResult(
            mol=None,
            status="r2_unremovable",
            r1_atom_count=len(r1_atom_indices),
            error=right.failure_type,
        )
    removed_indices = set(right.atom_indices) if right.extractable else set()
    if removed_indices & (r1_set | {anchor1_idx, anchor2_idx}):
        return SidechainCopyResult(
            mol=None,
            status="branch_overlap",
            r1_atom_count=len(r1_atom_indices),
            error="R1, R2, and anchor atom sets must be disjoint",
        )
    if removed_indices:
        for idx in sorted(removed_indices, reverse=True):
            working.RemoveAtom(idx)

    try:
        anchor1_current = _remap_after_removals(anchor1_idx, removed_indices)
        anchor2_current = _remap_after_removals(anchor2_idx, removed_indices)
        r1_set_current = {
            _remap_after_removals(idx, removed_indices)
            for idx in r1_set
        }
    except ValueError as exc:
        return SidechainCopyResult(mol=None, status="index_remap_failed", error=str(exc))

    # 3. Build a chemically correct aromatic attachment at R2. The root bond is
    # placed along the in-plane outward bisector, so its angles to the two ring
    # bonds are equal and are exactly 120 degrees for an ideal benzene geometry.
    try:
        conf = working.GetConformer()
        target_anchor_pos = _position_array(conf, anchor2_current)
        target_direction, target_ring_normal, _ = _ring_geometry(
            working,
            conf,
            anchor2_current,
        )
        if float(np.dot(source_frame[:, 2], target_ring_normal)) < 0.0:
            target_ring_normal = -target_ring_normal
        target_frame = _attachment_frame(target_direction, target_ring_normal)
        rotation = target_frame @ source_frame.T
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6):
            raise ValueError("attachment transform is not orthonormal")
        if float(np.linalg.det(rotation)) < 1.0 - 1.0e-6:
            raise ValueError("attachment transform must preserve chirality")
        target_root_pos = target_anchor_pos + source_bond_length * target_direction
    except Exception as exc:  # noqa: BLE001 - return a structured geometry failure.
        return SidechainCopyResult(
            mol=None,
            status="target_geometry_failed",
            r1_atom_count=len(r1_atom_indices),
            error=str(exc),
        )

    old_to_new: dict[int, int] = {}
    for old_idx in r1_atom_indices:
        new_idx = working.AddAtom(Chem.Atom(source_atoms[old_idx]))
        old_to_new[old_idx] = new_idx
        transformed = target_root_pos + rotation @ (source_positions[old_idx] - source_root_pos)
        working.GetConformer().SetAtomPosition(
            new_idx,
            Point3D(float(transformed[0]), float(transformed[1]), float(transformed[2])),
        )

    # 4. Copy internal bonds among the replicated atoms.
    for facts in internal_bonds:
        new_begin = old_to_new[facts.begin_idx]
        new_end = old_to_new[facts.end_idx]
        if working.GetBondBetweenAtoms(new_begin, new_end) is None:
            working.AddBond(new_begin, new_end, facts.bond_type)
            new_bond = working.GetBondBetweenAtoms(new_begin, new_end)
            _apply_bond_facts(new_bond, facts, old_to_new)

    # 5. Connect the copied root to anchor2 using the source attachment bond.
    root_new = old_to_new[r1_root]
    attachment_type = attachment_facts.bond_type if bond_type is None else bond_type
    if working.GetBondBetweenAtoms(anchor2_current, root_new) is None:
        working.AddBond(anchor2_current, root_new, attachment_type)
        new_attachment = working.GetBondBetweenAtoms(anchor2_current, root_new)
        attachment_mapping = {r1_root: root_new, anchor1_idx: anchor2_current}
        _apply_bond_facts(new_attachment, attachment_facts, attachment_mapping)
        if bond_type is not None and bond_type != attachment_facts.bond_type:
            new_attachment.SetBondType(bond_type)
            new_attachment.SetIsAromatic(bond_type == Chem.BondType.AROMATIC)

    out = working.GetMol()
    if sanitize:
        try:
            Chem.SanitizeMol(out)
        except Exception as exc:  # noqa: BLE001 - report sanitize failure clearly.
            return SidechainCopyResult(
                mol=None,
                status="sanitize_failed",
                r1_atom_count=len(r1_atom_indices),
                error=str(exc),
            )

    if len(Chem.GetMolFrags(out)) != 1:
        return SidechainCopyResult(
            mol=out,
            status="disconnected",
            r1_atom_count=len(r1_atom_indices),
            r2_atom_count=len(old_to_new),
            error="output contains multiple disconnected components",
        )

    try:
        bond_length_value, angle_1, angle_2, out_of_plane = _attachment_geometry(
            out,
            anchor2_current,
            root_new,
        )
    except Exception as exc:  # noqa: BLE001 - return a structured geometry failure.
        return SidechainCopyResult(
            mol=None,
            status="attachment_geometry_failed",
            r1_atom_count=len(r1_atom_indices),
            r2_atom_count=len(old_to_new),
            error=str(exc),
        )
    if (
        out_of_plane > PLANARITY_TOLERANCE_DEG
        or abs(angle_1 - angle_2) > ANGLE_EQUALITY_TOLERANCE_DEG
        or not math.isclose(bond_length_value, source_bond_length, abs_tol=1.0e-5)
    ):
        return SidechainCopyResult(
            mol=None,
            status="attachment_geometry_failed",
            r1_atom_count=len(r1_atom_indices),
            r2_atom_count=len(old_to_new),
            attachment_bond_length=bond_length_value,
            attachment_angle_1_deg=angle_1,
            attachment_angle_2_deg=angle_2,
            attachment_out_of_plane_deg=out_of_plane,
            error="R2 attachment is not planar, angle-balanced, and length-preserving",
        )

    # 6. Intramolecular clash check: copied branch vs (core + original R1).
    copied_indices = set(old_to_new.values())
    core_indices = set(range(out.GetNumAtoms())) - copied_indices - r1_set_current
    has_clash, min_dist, closest = intramolecular_clash_check(
        out, copied_indices, core_indices | r1_set_current, clash_fraction
    )
    status = "clash" if has_clash else "success"
    return SidechainCopyResult(
        mol=out,
        status=status,
        r1_atom_count=len(r1_atom_indices),
        r2_atom_count=len(copied_indices),
        clash=has_clash,
        min_clash_distance=min_dist,
        closest_pair=closest,
        attachment_bond_length=bond_length_value,
        attachment_angle_1_deg=angle_1,
        attachment_angle_2_deg=angle_2,
        attachment_out_of_plane_deg=out_of_plane,
    )
