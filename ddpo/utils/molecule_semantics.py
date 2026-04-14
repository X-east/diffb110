from __future__ import annotations

import logging
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "DiffSBDD"))

import utils as diffsbdd_utils
from analysis.molecule_builder import build_molecule
from analysis.metrics import rdmol_to_smiles
from constants import dataset_params

from ddpo.utils.suppress_warnings import redirect_stderr_to_log

log = logging.getLogger("DiffSBDD-DDPO")

FAILURE_SUCCESS = "success"
FAILURE_OPENBABEL_RECONSTRUCT = "openbabel_reconstruct_failed"
FAILURE_SANITIZE = "sanitize_failed"
FAILURE_DRUGLIKELINESS = "druglikeness_failed"
FAILURE_DOCKING_INPUT = "docking_input_failed"
FAILURE_DOCKING_BACKEND = "docking_backend_failed"


@dataclass
class SampleMoleculeSemantics:
    sample_idx: int
    mol: Optional[Chem.Mol]
    obabel_reconstruct_success: bool
    build_success: bool
    sanitize_success: bool  # Whether RDKit sanitize passed; currently equivalent to rdkit_valid.
    rdkit_valid: bool
    smiles: Optional[str]
    failure_type: str


def extract_sample_geometry(
    x: torch.Tensor,
    categorical: torch.Tensor,
    nodesxsample: torch.Tensor,
    sample_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    n_nodes = int(nodesxsample[sample_idx])
    positions = x[sample_idx, :n_nodes].detach().cpu()
    atom_types = categorical[sample_idx, :n_nodes].argmax(dim=-1).detach().cpu()
    return positions, atom_types, n_nodes


def build_rdmol_from_geometry(
    positions: torch.Tensor,
    atom_types: torch.Tensor,
    dataset_info: Optional[dict] = None,
) -> Tuple[Optional[Chem.Mol], bool, Optional[str]]:
    dataset_info = dataset_info or dataset_params["crossdock"]
    try:
        with redirect_stderr_to_log():
            mol = build_molecule(
                positions=positions,
                atom_types=atom_types,
                dataset_info=dataset_info,
                add_coords=True,
                use_openbabel=True,
            )
        if mol is None:
            return None, False, FAILURE_OPENBABEL_RECONSTRUCT
        return mol, True, None
    except Exception as exc:
        log.warning("[OPENBABEL-RECONSTRUCT-ERROR] error=%s: %s", type(exc).__name__, exc)
        warnings.warn(f"Failed to reconstruct molecule with OpenBabel: {exc}")
        return None, False, FAILURE_OPENBABEL_RECONSTRUCT


def sanitize_rdmol(
    mol: Optional[Chem.Mol],
) -> Tuple[Optional[Chem.Mol], bool, bool, Optional[str]]:
    if mol is None:
        return None, False, False, FAILURE_OPENBABEL_RECONSTRUCT
    try:
        with redirect_stderr_to_log():
            Chem.SanitizeMol(mol)
        return mol, True, True, None
    except Exception as exc:
        log.warning("[RDKit-SANITIZE-INVALID] error=%s: %s", type(exc).__name__, exc)
        return mol, False, False, FAILURE_SANITIZE


def compute_smiles_from_rdmol(mol: Optional[Chem.Mol]) -> Optional[str]:
    if mol is None:
        return None
    try:
        with redirect_stderr_to_log():
            return rdmol_to_smiles(mol)
    except Exception as exc:
        log.warning("[RDKit-SMILES-ERROR] error=%s: %s", type(exc).__name__, exc)
        return None


def build_canonical_rdmol_from_sample(
    x: torch.Tensor,
    categorical: torch.Tensor,
    nodesxsample: torch.Tensor,
    sample_idx: int,
    dataset_info: Optional[dict] = None,
) -> Tuple[Optional[Chem.Mol], bool, bool, bool, Optional[str]]:
    """
    Build the canonical downstream RDKit molecule for one sample.

    Returns:
        mol,
        obabel_reconstruct_success,
        build_success,
        rdkit_valid,
        failure_type
    """
    positions, atom_types, _ = extract_sample_geometry(x, categorical, nodesxsample, sample_idx)
    mol, build_ok, build_failure = build_rdmol_from_geometry(positions, atom_types, dataset_info)
    if not build_ok or mol is None:
        return None, False, False, False, build_failure or FAILURE_OPENBABEL_RECONSTRUCT

    mol, sanitize_ok, is_valid, sanitize_failure = sanitize_rdmol(mol)
    if not sanitize_ok or not is_valid or mol is None:
        return None, True, True, False, sanitize_failure or FAILURE_SANITIZE

    return mol, True, True, True, None


def compute_sample_molecule_semantics(
    x: torch.Tensor,
    categorical: torch.Tensor,
    nodesxsample: torch.Tensor,
    sample_idx: int,
    dataset_info: Optional[dict] = None,
) -> SampleMoleculeSemantics:
    """Compute the full canonical molecule semantics for one sample."""
    mol, obabel_ok, build_ok, is_valid, failure_type = build_canonical_rdmol_from_sample(
        x=x,
        categorical=categorical,
        nodesxsample=nodesxsample,
        sample_idx=sample_idx,
        dataset_info=dataset_info,
    )
    smiles = compute_smiles_from_rdmol(mol) if is_valid and mol is not None else None
    return SampleMoleculeSemantics(
        sample_idx=sample_idx,
        mol=mol,
        obabel_reconstruct_success=obabel_ok,
        build_success=build_ok,
        sanitize_success=is_valid,
        rdkit_valid=is_valid,
        smiles=smiles,
        failure_type=failure_type or FAILURE_SUCCESS,
    )


def write_rdmol_to_sdf(mol: Optional[Chem.Mol], sdf_path: str) -> bool:
    """Write one canonical RDKit molecule to SDF."""
    if mol is None:
        return False
    try:
        with redirect_stderr_to_log():
            diffsbdd_utils.write_sdf_file(sdf_path, [mol])
        return True
    except Exception as exc:
        log.warning("[SDF-WRITE-ERROR] error=%s: %s", type(exc).__name__, exc)
        return False


def write_rdmol_to_pdbqt(mol: Optional[Chem.Mol], pdbqt_path: str) -> bool:
    """Convert one canonical RDKit molecule to PDBQT via SDF + obabel."""
    import os
    import tempfile

    if mol is None:
        return False

    with tempfile.NamedTemporaryFile(suffix='.sdf', delete=False) as tmp_sdf:
        sdf_path = tmp_sdf.name

    try:
        if not write_rdmol_to_sdf(mol, sdf_path):
            return False
        result = subprocess.run(
            ["obabel", sdf_path, "-opdbqt", "-O", pdbqt_path, "-h", "--partialcharge", "gasteiger"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if result.returncode != 0:
            log.warning("[PDBQT-CONVERT-ERROR] %s", result.stderr.strip())
            return False
        return Path(pdbqt_path).exists() and Path(pdbqt_path).stat().st_size > 0
    except Exception as exc:
        log.warning("[PDBQT-CONVERT-ERROR] error=%s: %s", type(exc).__name__, exc)
        return False
    finally:
        try:
            os.unlink(sdf_path)
        except OSError:
            pass


def annotate_batch_molecule_semantics(
    data: Any,
    dataset_info: Optional[dict] = None,
    include_legacy_valid_alias: bool = True,
) -> Any:
    dataset_info = dataset_info or dataset_params["crossdock"]
    batch = data.batch

    x = batch["x"]
    categorical = batch["categorical"]
    nodesxsample = batch["nodesxsample"]
    n_samples = x.shape[0]

    obabel_reconstruct_success: List[int] = []
    build_success: List[int] = []
    sanitize_success: List[int] = []
    rdkit_valid: List[int] = []
    smiles_list: List[Optional[str]] = []
    failure_types: List[str] = []

    all_semantics = []
    for sample_idx in range(n_samples):
        semantics = compute_sample_molecule_semantics(
            x=x,
            categorical=categorical,
            nodesxsample=nodesxsample,
            sample_idx=sample_idx,
            dataset_info=dataset_info,
        )
        all_semantics.append(semantics)
        obabel_reconstruct_success.append(1 if semantics.obabel_reconstruct_success else 0)
        build_success.append(1 if semantics.build_success else 0)
        sanitize_success.append(1 if semantics.sanitize_success else 0)
        rdkit_valid.append(1 if semantics.rdkit_valid else 0)
        smiles_list.append(semantics.smiles)
        failure_types.append(semantics.failure_type)

    batch["obabel_reconstruct_success"] = obabel_reconstruct_success
    batch["build_success"] = build_success
    batch["sanitize_success"] = sanitize_success
    batch["rdkit_valid"] = rdkit_valid
    batch["smiles"] = smiles_list
    batch["failure_type"] = failure_types
    # Store built mol objects so downstream rewards can reuse them without rebuilding.
    # Only valid molecules have a mol; invalid ones are None.
    batch["rdmols"] = [s.mol if s.rdkit_valid else None for s in all_semantics]
    if include_legacy_valid_alias:
        # Bug4 Fix-5: 'valid' kept as legacy alias for downstream compatibility only.
        # All internal logic reads from 'rdkit_valid'. Do not use 'valid' as primary field.
        # This alias will be removed once all downstream consumers are migrated.
        batch["valid"] = rdkit_valid
    return data


__all__ = [
    "FAILURE_SUCCESS",
    "FAILURE_OPENBABEL_RECONSTRUCT",
    "FAILURE_SANITIZE",
    "FAILURE_DRUGLIKELINESS",
    "FAILURE_DOCKING_INPUT",
    "FAILURE_DOCKING_BACKEND",
    "SampleMoleculeSemantics",
    "extract_sample_geometry",
    "build_rdmol_from_geometry",
    "sanitize_rdmol",
    "compute_smiles_from_rdmol",
    "build_canonical_rdmol_from_sample",
    "compute_sample_molecule_semantics",
    "write_rdmol_to_sdf",
    "write_rdmol_to_pdbqt",
    "annotate_batch_molecule_semantics",
]
