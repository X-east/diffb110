"""Offline constructors for dual phenolic-OH replacement controls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

from ddpo.constraints.dual_sidechain.spec import DualSubstitutionSpec


_BOND_TYPE_NAMES = {
    Chem.BondType.SINGLE: "single",
    Chem.BondType.DOUBLE: "double",
    Chem.BondType.TRIPLE: "triple",
    Chem.BondType.AROMATIC: "aromatic",
}

_DEFAULT_ROOT_BOND_LENGTH = {
    "C": 1.50,
    "N": 1.40,
    "O": 1.36,
    "S": 1.76,
}


@dataclass(frozen=True)
class RGroupTemplate:
    """A small rooted R-group graph used for offline substitution controls."""

    name: str
    atom_symbols: tuple[str, ...]
    local_coords: tuple[tuple[float, float, float], ...]
    internal_bonds: tuple[tuple[int, int, Chem.BondType], ...] = ()
    root_atom_idx: int = 0
    root_bond_type: Chem.BondType = Chem.BondType.SINGLE

    def __post_init__(self) -> None:
        """Validate template graph and coordinates."""
        if not self.name:
            raise ValueError("R-group template name must not be empty")
        if not self.atom_symbols:
            raise ValueError("R-group template must contain at least one atom")
        if len(self.atom_symbols) != len(self.local_coords):
            raise ValueError("atom_symbols and local_coords must have the same length")
        if self.root_atom_idx < 0 or self.root_atom_idx >= len(self.atom_symbols):
            raise ValueError("root_atom_idx is out of range")
        for begin, end, _bond_type in self.internal_bonds:
            if begin < 0 or end < 0 or begin >= len(self.atom_symbols) or end >= len(self.atom_symbols):
                raise ValueError("internal bond atom index is out of range")
            if begin == end:
                raise ValueError("internal bond must connect two different atoms")

    @property
    def root_symbol(self) -> str:
        """Return the atomic symbol of the attachment root."""
        return self.atom_symbols[self.root_atom_idx]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "name": self.name,
            "atom_symbols": list(self.atom_symbols),
            "root_atom_idx": self.root_atom_idx,
            "root_bond_type": _bond_type_name(self.root_bond_type),
            "internal_bonds": [
                {
                    "begin": begin,
                    "end": end,
                    "bond_type": _bond_type_name(bond_type),
                }
                for begin, end, bond_type in self.internal_bonds
            ],
        }


@dataclass
class DualRSubstitutionResult:
    """Constructed dual-R control molecule plus atom-index bookkeeping."""

    mol: Chem.Mol
    original_spec: DualSubstitutionSpec
    adjusted_spec: DualSubstitutionSpec
    removed_atom_indices: tuple[int, ...]
    old_to_new_atom_indices: dict[int, int] = field(default_factory=dict)
    left_r_group: str | None = None
    right_r_group: str | None = None
    attach_left: bool = True
    attach_right: bool = True

    def to_manifest_row(self) -> dict[str, Any]:
        """Return flat metadata for CSV manifests."""
        return {
            "left_r_group": self.left_r_group or "",
            "right_r_group": self.right_r_group or "",
            "attach_left": int(self.attach_left),
            "attach_right": int(self.attach_right),
            "original_left_core_idx": self.original_spec.left_core_idx,
            "original_right_core_idx": self.original_spec.right_core_idx,
            "adjusted_left_core_idx": self.adjusted_spec.left_core_idx,
            "adjusted_right_core_idx": self.adjusted_spec.right_core_idx,
            "removed_atom_indices": " ".join(str(idx) for idx in self.removed_atom_indices),
            "num_atoms": self.mol.GetNumAtoms(),
            "num_heavy_atoms": self.mol.GetNumHeavyAtoms(),
        }


_R_GROUP_LIBRARY = {
    "methyl": RGroupTemplate(
        name="methyl",
        atom_symbols=("C",),
        local_coords=((0.0, 0.0, 0.0),),
    ),
    "ethyl": RGroupTemplate(
        name="ethyl",
        atom_symbols=("C", "C"),
        local_coords=((0.0, 0.0, 0.0), (1.52, 0.18, 0.0)),
        internal_bonds=((0, 1, Chem.BondType.SINGLE),),
    ),
    "propyl": RGroupTemplate(
        name="propyl",
        atom_symbols=("C", "C", "C"),
        local_coords=((0.0, 0.0, 0.0), (1.52, 0.18, 0.0), (3.02, -0.15, 0.0)),
        internal_bonds=((0, 1, Chem.BondType.SINGLE), (1, 2, Chem.BondType.SINGLE)),
    ),
    "allyl": RGroupTemplate(
        name="allyl",
        atom_symbols=("C", "C", "C"),
        local_coords=((0.0, 0.0, 0.0), (1.50, 0.18, 0.0), (2.84, -0.12, 0.0)),
        internal_bonds=((0, 1, Chem.BondType.SINGLE), (1, 2, Chem.BondType.DOUBLE)),
    ),
    "methoxy": RGroupTemplate(
        name="methoxy",
        atom_symbols=("O", "C"),
        local_coords=((0.0, 0.0, 0.0), (1.43, 0.18, 0.0)),
        internal_bonds=((0, 1, Chem.BondType.SINGLE),),
    ),
    "ethoxy": RGroupTemplate(
        name="ethoxy",
        atom_symbols=("O", "C", "C"),
        local_coords=((0.0, 0.0, 0.0), (1.43, 0.18, 0.0), (2.93, -0.15, 0.0)),
        internal_bonds=((0, 1, Chem.BondType.SINGLE), (1, 2, Chem.BondType.SINGLE)),
    ),
}


def available_r_group_templates() -> tuple[str, ...]:
    """Return names of built-in R-group templates."""
    return tuple(sorted(_R_GROUP_LIBRARY))


def get_r_group_template(name: str) -> RGroupTemplate:
    """Return a built-in R-group template by name."""
    key = name.strip().lower()
    try:
        return _R_GROUP_LIBRARY[key]
    except KeyError as exc:
        choices = ", ".join(available_r_group_templates())
        raise ValueError(f"Unknown R-group template '{name}'. Available templates: {choices}") from exc


def substitute_dual_r_groups(
    reference_mol: Chem.Mol,
    spec: DualSubstitutionSpec,
    *,
    left_r_group: str | RGroupTemplate | None,
    right_r_group: str | RGroupTemplate | None,
    attach_left: bool = True,
    attach_right: bool = True,
    sanitize: bool = True,
    add_hs: bool = True,
) -> DualRSubstitutionResult:
    """Replace two phenolic OH leaving groups with optional R-group templates.

    The constructor removes the recorded leaving atoms and their explicit
    hydrogens, then attaches independent R-group graphs to the two aromatic
    core atoms. It is intended for offline positive/negative controls, not for
    production diffusion sampling.
    """
    working = Chem.Mol(reference_mol)
    if sanitize:
        Chem.SanitizeMol(working)
    working = _ensure_conformer(working)

    remove_indices = _leaving_atoms_with_explicit_hydrogens(working, spec)
    old_to_new = _old_to_new_index_map(working.GetNumAtoms(), remove_indices)
    adjusted_spec = _adjust_spec_after_removal(spec, old_to_new)

    original_positions = _atom_positions(working)
    left_direction = _site_direction(
        original_positions,
        core_atom_idx=spec.left_core_idx,
        leaving_atom_idx=spec.left_leaving_atom_idx,
    )
    right_direction = _site_direction(
        original_positions,
        core_atom_idx=spec.right_core_idx,
        leaving_atom_idx=spec.right_leaving_atom_idx,
    )

    rw_mol = Chem.RWMol(working)
    for atom_idx in sorted(remove_indices, reverse=True):
        rw_mol.RemoveAtom(atom_idx)

    positions_by_new_idx = {
        new_idx: original_positions[old_idx]
        for old_idx, new_idx in old_to_new.items()
        if old_idx in original_positions
    }
    if attach_left and left_r_group is not None:
        _attach_r_group(
            rw_mol,
            core_atom_idx=adjusted_spec.left_core_idx,
            template=_resolve_template(left_r_group),
            direction=left_direction,
            positions_by_idx=positions_by_new_idx,
        )
    if attach_right and right_r_group is not None:
        _attach_r_group(
            rw_mol,
            core_atom_idx=adjusted_spec.right_core_idx,
            template=_resolve_template(right_r_group),
            direction=right_direction,
            positions_by_idx=positions_by_new_idx,
        )

    substituted = rw_mol.GetMol()
    _set_conformer(substituted, positions_by_new_idx)
    if sanitize:
        Chem.SanitizeMol(substituted)
    if add_hs:
        substituted = Chem.AddHs(substituted, addCoords=True)
        if sanitize:
            Chem.SanitizeMol(substituted)
    return DualRSubstitutionResult(
        mol=substituted,
        original_spec=spec,
        adjusted_spec=adjusted_spec,
        removed_atom_indices=tuple(sorted(remove_indices)),
        old_to_new_atom_indices=old_to_new,
        left_r_group=_template_name(left_r_group),
        right_r_group=_template_name(right_r_group),
        attach_left=attach_left,
        attach_right=attach_right,
    )


def _resolve_template(template: str | RGroupTemplate) -> RGroupTemplate:
    """Resolve a template name or pass through an explicit template."""
    if isinstance(template, RGroupTemplate):
        return template
    return get_r_group_template(template)


def _template_name(template: str | RGroupTemplate | None) -> str | None:
    """Return the stable name of a template input."""
    if template is None:
        return None
    if isinstance(template, RGroupTemplate):
        return template.name
    return get_r_group_template(template).name


def _ensure_conformer(mol: Chem.Mol) -> Chem.Mol:
    """Return a molecule with at least one conformer."""
    if mol.GetNumConformers() > 0:
        return mol
    with_coords = Chem.Mol(mol)
    try:
        AllChem.Compute2DCoords(with_coords)
    except Exception as exc:  # noqa: BLE001 - convert RDKit failures to clear constructor errors.
        raise ValueError("Reference molecule has no conformer and 2D coordinate generation failed") from exc
    return with_coords


def _leaving_atoms_with_explicit_hydrogens(mol: Chem.Mol, spec: DualSubstitutionSpec) -> set[int]:
    """Return leaving atoms plus explicit hydrogens attached to them."""
    remove_indices: set[int] = set()
    for leaving_idx in (spec.left_leaving_atom_idx, spec.right_leaving_atom_idx):
        if leaving_idx is None:
            continue
        if leaving_idx < 0 or leaving_idx >= mol.GetNumAtoms():
            continue
        remove_indices.add(leaving_idx)
        leaving_atom = mol.GetAtomWithIdx(leaving_idx)
        for neighbor in leaving_atom.GetNeighbors():
            if neighbor.GetAtomicNum() == 1:
                remove_indices.add(neighbor.GetIdx())
    return remove_indices


def _old_to_new_index_map(num_atoms: int, remove_indices: set[int]) -> dict[int, int]:
    """Map old atom indices to new atom indices after atom removal."""
    mapping: dict[int, int] = {}
    removed_before = 0
    for old_idx in range(num_atoms):
        if old_idx in remove_indices:
            removed_before += 1
            continue
        mapping[old_idx] = old_idx - removed_before
    return mapping


def _adjust_spec_after_removal(
    spec: DualSubstitutionSpec,
    old_to_new: dict[int, int],
) -> DualSubstitutionSpec:
    """Adjust core indices after deleting leaving atoms."""
    try:
        left_core_idx = old_to_new[spec.left_core_idx]
        right_core_idx = old_to_new[spec.right_core_idx]
    except KeyError as exc:
        raise ValueError("A substitution core atom was removed with the leaving group") from exc
    return DualSubstitutionSpec(
        left_core_idx=left_core_idx,
        right_core_idx=right_core_idx,
        left_leaving_atom_idx=None,
        right_leaving_atom_idx=None,
        left_label=spec.left_label,
        right_label=spec.right_label,
        max_branch_atoms=spec.max_branch_atoms,
    )


def _atom_positions(mol: Chem.Mol) -> dict[int, tuple[float, float, float]]:
    """Read atom coordinates from the first conformer."""
    conf = mol.GetConformer()
    positions = {}
    for atom_idx in range(mol.GetNumAtoms()):
        point = conf.GetAtomPosition(atom_idx)
        positions[atom_idx] = (float(point.x), float(point.y), float(point.z))
    return positions


def _site_direction(
    positions: dict[int, tuple[float, float, float]],
    *,
    core_atom_idx: int,
    leaving_atom_idx: int | None,
) -> tuple[float, float, float]:
    """Return the outgoing direction from aromatic core toward the old leaving atom."""
    if leaving_atom_idx is None or core_atom_idx not in positions or leaving_atom_idx not in positions:
        return (1.0, 0.0, 0.0)
    vector = _subtract(positions[leaving_atom_idx], positions[core_atom_idx])
    return _unit(vector, fallback=(1.0, 0.0, 0.0))


def _attach_r_group(
    rw_mol: Chem.RWMol,
    *,
    core_atom_idx: int,
    template: RGroupTemplate,
    direction: tuple[float, float, float],
    positions_by_idx: dict[int, tuple[float, float, float]],
) -> None:
    """Attach one R-group template to an aromatic core atom."""
    if core_atom_idx < 0 or core_atom_idx >= rw_mol.GetNumAtoms():
        raise ValueError(f"core_atom_idx out of range: {core_atom_idx}")
    core_position = positions_by_idx.get(core_atom_idx, (0.0, 0.0, 0.0))
    basis_x, basis_y, basis_z = _orthonormal_basis(direction)
    root_distance = _DEFAULT_ROOT_BOND_LENGTH.get(template.root_symbol, 1.50)
    root_origin = _add(core_position, _scale(basis_x, root_distance))

    local_to_global: dict[int, int] = {}
    for local_idx, symbol in enumerate(template.atom_symbols):
        atom_idx = rw_mol.AddAtom(Chem.Atom(symbol))
        local_to_global[local_idx] = atom_idx
        local_coord = template.local_coords[local_idx]
        positions_by_idx[atom_idx] = _add(
            root_origin,
            _add(
                _scale(basis_x, local_coord[0]),
                _add(_scale(basis_y, local_coord[1]), _scale(basis_z, local_coord[2])),
            ),
        )

    rw_mol.AddBond(core_atom_idx, local_to_global[template.root_atom_idx], template.root_bond_type)
    for begin, end, bond_type in template.internal_bonds:
        rw_mol.AddBond(local_to_global[begin], local_to_global[end], bond_type)


def _set_conformer(mol: Chem.Mol, positions_by_idx: dict[int, tuple[float, float, float]]) -> None:
    """Replace conformers with one conformer containing known atom positions."""
    mol.RemoveAllConformers()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for atom_idx in range(mol.GetNumAtoms()):
        x, y, z = positions_by_idx.get(atom_idx, (0.0, 0.0, 0.0))
        conf.SetAtomPosition(atom_idx, Point3D(float(x), float(y), float(z)))
    mol.AddConformer(conf, assignId=True)


def _orthonormal_basis(
    direction: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    """Build a right-handed local coordinate frame from an outgoing direction."""
    x_axis = _unit(direction, fallback=(1.0, 0.0, 0.0))
    reference = (0.0, 0.0, 1.0)
    if abs(_dot(x_axis, reference)) > 0.95:
        reference = (0.0, 1.0, 0.0)
    y_axis = _unit(_cross(reference, x_axis), fallback=(0.0, 1.0, 0.0))
    z_axis = _unit(_cross(x_axis, y_axis), fallback=(0.0, 0.0, 1.0))
    return x_axis, y_axis, z_axis


def _add(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Add two 3D vectors."""
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def _subtract(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Subtract two 3D vectors."""
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def _scale(vector: tuple[float, float, float], factor: float) -> tuple[float, float, float]:
    """Scale a 3D vector."""
    return (vector[0] * factor, vector[1] * factor, vector[2] * factor)


def _dot(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> float:
    """Return the dot product of two 3D vectors."""
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _cross(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Return the cross product of two 3D vectors."""
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _unit(
    vector: tuple[float, float, float],
    *,
    fallback: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Return a normalized vector or a fallback for near-zero input."""
    length = _dot(vector, vector) ** 0.5
    if length < 1e-8:
        return fallback
    return (vector[0] / length, vector[1] / length, vector[2] / length)


def _bond_type_name(bond_type: Chem.BondType) -> str:
    """Return a stable bond type name."""
    return _BOND_TYPE_NAMES.get(bond_type, str(bond_type).lower())
