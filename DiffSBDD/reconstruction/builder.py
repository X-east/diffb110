"""Dispatch molecule reconstruction and bond inference methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rdkit import Chem

from analysis.molecule_builder import make_mol_edm, make_mol_openbabel

DEFAULT_RECONSTRUCTION_METHOD = "openbabel"

_METHOD_ALIASES = {
    "openbabel": "openbabel",
    "obabel": "openbabel",
    "open_babel": "openbabel",
    "edm": "edm",
    "distance": "edm",
    "distance_rule": "edm",
    "distance_rules": "edm",
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
    raise AssertionError(f"Unhandled reconstruction method: {selected_method}")
