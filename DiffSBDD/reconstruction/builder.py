"""Dispatch molecule reconstruction and bond inference methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from rdkit import Chem, Geometry

from analysis.molecule_builder import make_mol_edm, make_mol_openbabel

try:
    from DiffSBDD.edge_utils import RDKIT_BOND_TYPE_BY_ID
except ImportError:  # pragma: no cover - supports running with DiffSBDD on sys.path.
    from edge_utils import RDKIT_BOND_TYPE_BY_ID

DEFAULT_RECONSTRUCTION_METHOD = "openbabel"

_METHOD_ALIASES = {
    "openbabel": "openbabel",
    "obabel": "openbabel",
    "open_babel": "openbabel",
    "edm": "edm",
    "distance": "edm",
    "distance_rule": "edm",
    "distance_rules": "edm",
    "predicted_edge": "predicted_edge",
    "predicted_edges": "predicted_edge",
    "edge": "predicted_edge",
    "edge_pred": "predicted_edge",
    "oracle_edge": "oracle_edge",
    "oracle_edges": "oracle_edge",
    "true_edge": "oracle_edge",
    "true_edges": "oracle_edge",
}


@dataclass(frozen=True)
class ReconstructionConfig:
    """Configuration for molecule reconstruction and bond inference."""

    method: str = DEFAULT_RECONSTRUCTION_METHOD
    add_coords: bool = False

    @property
    def normalized_method(self) -> str:
        """Return the canonical reconstruction method name."""
        return normalize_reconstruction_method(self.method)


def normalize_reconstruction_method(method: str | None) -> str:
    """Normalize a reconstruction method name and validate support."""
    if method is None:
        return DEFAULT_RECONSTRUCTION_METHOD
    key = str(method).strip().lower().replace("-", "_")
    if key in _METHOD_ALIASES:
        return _METHOD_ALIASES[key]
    supported = ", ".join(sorted(set(_METHOD_ALIASES.values())))
    raise ValueError(
        f"Unsupported reconstruction method {method!r}. "
        f"Supported methods: {supported}."
    )


def reconstruct_molecule(
    positions: Any,
    atom_types: Any,
    dataset_info: dict[str, Any],
    *,
    method: str | None = DEFAULT_RECONSTRUCTION_METHOD,
    add_coords: bool = False,
    halfedge_index: Any | None = None,
    halfedge_type: Any | None = None,
) -> Chem.Mol:
    """Build an RDKit molecule with the selected bond inference backend."""
    selected_method = normalize_reconstruction_method(method)
    if selected_method == "openbabel":
        return make_mol_openbabel(
            positions,
            atom_types,
            dataset_info["atom_decoder"],
        )
    if selected_method == "edm":
        return make_mol_edm(
            positions,
            atom_types,
            dataset_info,
            add_coords=add_coords,
        )
    if selected_method in {"predicted_edge", "oracle_edge"}:
        return reconstruct_molecule_from_edges(
            positions=positions,
            atom_types=atom_types,
            dataset_info=dataset_info,
            halfedge_index=halfedge_index,
            halfedge_type=halfedge_type,
            add_coords=add_coords,
        )
    raise AssertionError(f"Unhandled reconstruction method: {selected_method}")


def reconstruct_molecule_from_edges(
    positions: Any,
    atom_types: Any,
    dataset_info: dict[str, Any],
    *,
    halfedge_index: Any,
    halfedge_type: Any,
    add_coords: bool = False,
) -> Chem.Mol:
    """Build an RDKit molecule from atom predictions and explicit halfedge labels."""
    if halfedge_index is None or halfedge_type is None:
        raise ValueError("halfedge_index and halfedge_type are required for edge reconstruction")
    pos_tensor = _as_tensor(positions, dtype=torch.float32)
    atom_type_tensor = _as_tensor(atom_types, dtype=torch.long)
    halfedge_index_tensor = _as_tensor(halfedge_index, dtype=torch.long)
    halfedge_type_tensor = _as_tensor(halfedge_type, dtype=torch.long)

    if pos_tensor.ndim != 2 or pos_tensor.shape[-1] != 3:
        raise ValueError("positions must have shape [num_atoms, 3]")
    if atom_type_tensor.ndim != 1:
        raise ValueError("atom_types must have shape [num_atoms]")
    if pos_tensor.shape[0] != atom_type_tensor.shape[0]:
        raise ValueError("positions and atom_types must contain the same number of atoms")
    if halfedge_index_tensor.ndim != 2 or halfedge_index_tensor.shape[0] != 2:
        raise ValueError("halfedge_index must have shape [2, num_halfedges]")
    if halfedge_type_tensor.ndim != 1:
        raise ValueError("halfedge_type must have shape [num_halfedges]")
    if halfedge_index_tensor.shape[1] != halfedge_type_tensor.shape[0]:
        raise ValueError("halfedge_index and halfedge_type lengths do not match")

    atom_symbols = _decode_atom_symbols(atom_type_tensor, dataset_info)
    rw_mol = Chem.RWMol()
    for symbol in atom_symbols:
        rw_mol.AddAtom(Chem.Atom(_atomic_number(symbol)))

    if add_coords:
        conformer = Chem.Conformer(len(atom_symbols))
        coords = pos_tensor.detach().cpu()
        for atom_idx in range(len(atom_symbols)):
            conformer.SetAtomPosition(
                atom_idx,
                Geometry.Point3D(*map(float, coords[atom_idx].tolist())),
            )
        rw_mol.AddConformer(conformer)

    for edge_idx, edge_type in enumerate(halfedge_type_tensor.detach().cpu().tolist()):
        bond_id = int(edge_type)
        if bond_id <= 0:
            continue
        begin_idx = int(halfedge_index_tensor[0, edge_idx].item())
        end_idx = int(halfedge_index_tensor[1, edge_idx].item())
        if begin_idx == end_idx:
            continue
        if begin_idx < 0 or end_idx < 0:
            continue
        if begin_idx >= len(atom_symbols) or end_idx >= len(atom_symbols):
            continue
        if rw_mol.GetBondBetweenAtoms(begin_idx, end_idx) is not None:
            continue
        rdkit_bond_type = RDKIT_BOND_TYPE_BY_ID.get(bond_id)
        if rdkit_bond_type is None:
            continue
        if rdkit_bond_type == Chem.BondType.AROMATIC:
            rw_mol.GetAtomWithIdx(begin_idx).SetIsAromatic(True)
            rw_mol.GetAtomWithIdx(end_idx).SetIsAromatic(True)
        rw_mol.AddBond(begin_idx, end_idx, rdkit_bond_type)
        if rdkit_bond_type == Chem.BondType.AROMATIC:
            bond = rw_mol.GetBondBetweenAtoms(begin_idx, end_idx)
            if bond is not None:
                bond.SetIsAromatic(True)

    return rw_mol.GetMol()


def _as_tensor(value: Any, *, dtype: torch.dtype) -> torch.Tensor:
    """Convert tensor-like values to a detached CPU-friendly tensor."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(dtype=dtype)
    return torch.as_tensor(value, dtype=dtype)


def _decode_atom_symbols(atom_types: torch.Tensor, dataset_info: dict[str, Any]) -> list[str]:
    """Decode atom type ids to element symbols."""
    atom_decoder = dataset_info.get("atom_decoder")
    if atom_decoder is None:
        raise KeyError("dataset_info must contain atom_decoder for edge reconstruction")
    symbols = []
    for atom_idx in atom_types.detach().cpu().tolist():
        try:
            symbol = atom_decoder[int(atom_idx)]
        except (IndexError, KeyError) as exc:
            raise ValueError(f"Unknown atom type id for edge reconstruction: {atom_idx}") from exc
        symbols.append(_normalize_atom_symbol(symbol))
    return symbols


def _normalize_atom_symbol(symbol: Any) -> str:
    """Normalize common atom decoder outputs to RDKit element symbols."""
    text = str(symbol)
    if "." in text:
        text = text.split(".", maxsplit=1)[0]
    if "_" in text:
        text = text.split("_", maxsplit=1)[0]
    return text


def _atomic_number(symbol: str) -> int:
    """Return atomic number for a valid element symbol."""
    atomic_number = Chem.GetPeriodicTable().GetAtomicNumber(symbol)
    if atomic_number <= 0:
        raise ValueError(f"Unsupported atom symbol for edge reconstruction: {symbol}")
    return int(atomic_number)
