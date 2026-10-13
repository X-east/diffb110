"""Lightweight, dependency-free ligand preparation primitives.

Authority: ``docs/评估口径规范.md`` §二.

These live in their own module on purpose: the docking evaluation scripts need
the largest-fragment reduction and the hydrogen step, but must **not** drag in
``torch`` / DiffSBDD (which ``ddpo.utils.molecule_semantics`` imports).
Only RDKit is required here.

The reduction used to be copy-pasted into four scripts, each with a slightly
different failure behaviour — most notably a silent ``return mol, 1`` fallback
that made ``fragments_reduced`` report 0 even when no reduction happened.
"""

from __future__ import annotations

from typing import Tuple

from rdkit import Chem

FAILURE_SUCCESS = "success"
FAILURE_DOCKING_INPUT = "docking_input_failed"

LargestFragmentResult = Tuple["Chem.Mol | None", int, str]


def extract_largest_fragment(
    mol: "Chem.Mol | None",
    *,
    sanitize_frags: bool = True,
) -> LargestFragmentResult:
    """Reduce a record to its largest connected fragment (SOP §二.2).

    Returns ``(fragment, original_fragment_count, failure_reason)``.

    On failure the fragment is ``None`` — never a silently passed-through
    molecule.  ``sanitizeFrags=False`` is retried once, because RDKit-legal but
    AutoDock-illegal fragments (abnormal S/O valence) raise on sanitisation and
    would otherwise abort Uni-Dock's internal assertion.
    """
    if mol is None:
        return None, 0, FAILURE_DOCKING_INPUT
    fragments = None
    if sanitize_frags:
        try:
            fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        except Exception:
            fragments = None
    if fragments is None:
        try:
            fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        except Exception:
            return None, 0, FAILURE_DOCKING_INPUT
    if not fragments:
        return None, 0, FAILURE_DOCKING_INPUT
    largest = max(fragments, key=lambda fragment: fragment.GetNumAtoms())
    return largest, len(fragments), FAILURE_SUCCESS


def add_hydrogens_for_docking(mol: Chem.Mol) -> Tuple[Chem.Mol | None, bool]:
    """``Chem.AddHs(mol, addCoords=True)`` (SOP §二.3).

    ``addCoords=True`` is mandatory: without it every hydrogen lands at the
    origin and the geometry degenerates (measured 100/100 ligands).

    Returns ``(mol_with_H, hydrogens_were_added)``; ``None`` marks failure so
    the caller can record a docking-input failure instead of docking garbage.
    """
    if mol is None:
        return None, False
    try:
        had_hydrogens = any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms())
        result = Chem.AddHs(Chem.Mol(mol), addCoords=True)
    except Exception:
        return None, False
    if result.GetNumConformers() == 0:
        return None, False
    return result, not had_hydrogens


__all__ = [
    "FAILURE_SUCCESS",
    "FAILURE_DOCKING_INPUT",
    "LargestFragmentResult",
    "extract_largest_fragment",
    "add_hydrogens_for_docking",
]
