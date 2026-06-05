"""Shared utilities for ligand bond and halfedge representations."""

from __future__ import annotations

import numpy as np
import torch
from rdkit import Chem


BOND_TYPE_NAMES = {
    0: "no_bond",
    1: "single",
    2: "double",
    3: "triple",
    4: "aromatic",
}

BOND_TYPE_TO_ID = {
    Chem.rdchem.BondType.SINGLE: 1,
    Chem.rdchem.BondType.DOUBLE: 2,
    Chem.rdchem.BondType.TRIPLE: 3,
    Chem.rdchem.BondType.AROMATIC: 4,
}

RDKIT_BOND_TYPE_BY_ID = {
    1: Chem.rdchem.BondType.SINGLE,
    2: Chem.rdchem.BondType.DOUBLE,
    3: Chem.rdchem.BondType.TRIPLE,
    4: Chem.rdchem.BondType.AROMATIC,
}


def extract_ligand_bonds(
    ligand: Chem.Mol,
    atom_index_map: dict[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return directed ligand bond indices and categorical bond types."""
    rows: list[int] = []
    cols: list[int] = []
    bond_types: list[int] = []

    for bond in ligand.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if begin not in atom_index_map or end not in atom_index_map:
            continue
        bond_type = BOND_TYPE_TO_ID.get(bond.GetBondType())
        if bond_type is None:
            raise ValueError(f"Unsupported bond type: {bond.GetBondType()}")
        begin_new = atom_index_map[begin]
        end_new = atom_index_map[end]
        rows.extend([begin_new, end_new])
        cols.extend([end_new, begin_new])
        bond_types.extend([bond_type, bond_type])

    if len(rows) == 0:
        return (
            np.zeros((2, 0), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )

    bond_index = np.array([rows, cols], dtype=np.int64)
    bond_type = np.array(bond_types, dtype=np.int64)
    order = (bond_index[0] * len(atom_index_map) + bond_index[1]).argsort()
    return bond_index[:, order], bond_type[order]


def build_ligand_halfedges(
    num_atoms: int,
    bond_index: torch.Tensor,
    bond_type: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build DiffGui-style upper-triangle ligand halfedge labels."""
    edge_type_mat = torch.zeros((num_atoms, num_atoms), dtype=torch.long)
    if bond_index.numel() > 0:
        edge_type_mat[bond_index[0].long(), bond_index[1].long()] = bond_type.long()

    halfedge_index = torch.triu_indices(num_atoms, num_atoms, offset=1)
    halfedge_type = edge_type_mat[halfedge_index[0], halfedge_index[1]]
    expected_bonds = int((bond_type > 0).sum().item() // 2)
    actual_bonds = int((halfedge_type > 0).sum().item())
    if actual_bonds != expected_bonds:
        raise ValueError(
            f"Halfedge bond mismatch: expected {expected_bonds}, got {actual_bonds}"
        )
    return {
        "ligand_halfedge_index": halfedge_index,
        "ligand_halfedge_type": halfedge_type,
        "num_ligand_halfedges": torch.tensor(len(halfedge_type), dtype=torch.long),
    }


def halfedges_to_undirected_bonds(
    halfedge_index: torch.Tensor,
    halfedge_type: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert nonzero halfedge labels to undirected bond index and type tensors."""
    is_bond = halfedge_type > 0
    return halfedge_index[:, is_bond], halfedge_type[is_bond]
