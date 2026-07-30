"""Minimal reproduction test for largest-fragment extraction in benchmark scoring.

Both the score-only pipeline (score_qvina_only.py) and the full re-docking
pipeline (dock_sdf_unidock.py) must reduce disconnected ligand records to their
single largest connected fragment before docking, otherwise apparent affinity
is inflated and models are not comparable.

Run:
    conda run -n dif python test/test_largest_fragment_extraction.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SCORE_MODULE = _load(
    REPO_ROOT
    / "analysis/benchmark/benchmark5/8gjl_c1101_benchmark_20260706_1000steps_3rep_formal/qvina_score_only_20260721/score_qvina_only.py",
    "score_qvina_only_test",
)
DOCK_MODULE = _load(
    REPO_ROOT / "scripts/docking/dock_sdf_unidock.py",
    "dock_sdf_unidock_test",
)


def _with_3d(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid smiles: {smiles}")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=1)
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def test_disconnected_reduced_to_single_fragment() -> None:
    mol = _with_3d("c1ccccc1.c1ccccc1.CO")  # two benzene + methanol
    assert Chem.GetMolFrags(mol, asMols=True)  # 3 fragments

    score_mol, score_flag = SCORE_MODULE._extract_largest_fragment(mol)
    dock_mol, dock_count = DOCK_MODULE._extract_largest_fragment(mol)

    # score-only flags reduction, dock reports original fragment count
    assert score_flag is True
    assert dock_count == 3
    # both reduce to exactly one connected component
    assert len(Chem.GetMolFrags(score_mol, asMols=True)) == 1
    assert len(Chem.GetMolFrags(dock_mol, asMols=True)) == 1
    # the kept fragment is the largest (benzene, 12 atoms with Hs)
    assert score_mol.GetNumAtoms() == 12
    assert dock_mol.GetNumAtoms() == 12
    # 3D coordinates are preserved
    assert score_mol.GetNumConformers() > 0
    assert dock_mol.GetNumConformers() > 0


def test_single_fragment_untouched() -> None:
    mol = _with_3d("c1ccccc1O")  # phenol, single fragment
    n_atoms = mol.GetNumAtoms()

    score_mol, score_flag = SCORE_MODULE._extract_largest_fragment(mol)
    dock_mol, dock_count = DOCK_MODULE._extract_largest_fragment(mol)

    assert score_flag is False
    assert dock_count == 1
    assert score_mol.GetNumAtoms() == n_atoms
    assert dock_mol.GetNumAtoms() == n_atoms
    assert score_mol.GetNumConformers() > 0
    assert dock_mol.GetNumConformers() > 0


if __name__ == "__main__":
    test_disconnected_reduced_to_single_fragment()
    test_single_fragment_untouched()
    print("OK: largest-fragment extraction verified for score-only and re-dock pipelines")
