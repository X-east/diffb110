from __future__ import annotations

"""Ligand preparation helpers for docking reward backends.

Phase-2 purpose:
|- keep canonical molecule -> docking input preparation outside `DockingReward`
|- preserve current mainline behavior: canonical RDKit molecule exported to SDF
|- keep failure boundaries explicit as docking-input failures
|- **reuse pre-built rdmols from annotate_batch_molecule_semantics()**
"""

import os
import tempfile
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from ddpo.utils.molecule_semantics import (
    FAILURE_DOCKING_INPUT,
    build_canonical_rdmol_from_sample,
    prepare_mol_for_docking,
    write_rdmol_to_sdf,
)

if TYPE_CHECKING:
    from verl_diffusion.protocol import DataProto


@dataclass
class PreparedLigandBatch:
    tmpdir: str
    ligand_files: List[str]
    sample_indices: List[int]
    failures: Dict[int, str]

    def cleanup(self) -> None:
        import shutil
        try:
            shutil.rmtree(self.tmpdir, ignore_errors=True)
        except Exception:
            pass


@dataclass
class PreparedLigandFile:
    path: Optional[str]
    failure_type: Optional[str]

    def cleanup(self) -> None:
        if self.path and os.path.exists(self.path):
            try:
                os.unlink(self.path)
            except Exception:
                pass


def _build_sample_to_sdf_from_mol(
    mol,
    sdf_path: str,
    *,
    add_hydrogens: bool = True,
    assign_gasteiger: bool = True,
) -> Tuple[bool, Optional[str]]:
    """Write an existing RDKit Mol object to SDF with optional docking preparation.

    Args:
        mol: Canonical RDKit Mol (no H) from batch annotation.
        sdf_path: Destination SDF file path.
        add_hydrogens: If True, add explicit H before writing.
        assign_gasteiger: If True, assign Gasteiger charges before writing.
    """
    try:
        if mol is None:
            return False, FAILURE_DOCKING_INPUT
        # Prepare for docking: add H + charges before SDF export
        if add_hydrogens or assign_gasteiger:
            mol = prepare_mol_for_docking(mol, add_hydrogens, assign_gasteiger)
            if mol is None:
                return False, FAILURE_DOCKING_INPUT
        if not write_rdmol_to_sdf(mol, sdf_path):
            return False, FAILURE_DOCKING_INPUT
        return True, None
    except Exception as exc:
        warnings.warn(f"[DockingPreparation] write rdmol to SDF failed: {exc}")
        return False, FAILURE_DOCKING_INPUT


def build_sample_to_sdf(
    x,
    categorical,
    nodesxsample,
    sample_idx: int,
    sdf_path: str,
    dataset_info,
    *,
    add_hydrogens: bool = True,
    assign_gasteiger: bool = True,
) -> Tuple[bool, Optional[str]]:
    """Build canonical RDKit molecule for one sample and export to SDF.

    Args:
        x: Atom coordinates tensor.
        categorical: Atom type one-hot tensor.
        nodesxsample: Number of atoms per sample.
        sample_idx: Index of the sample in the batch.
        sdf_path: Destination SDF file path.
        dataset_info: Dataset parameter dict.
        add_hydrogens: If True, add explicit H before writing.
        assign_gasteiger: If True, assign Gasteiger charges before writing.
    """
    try:
        mol, _, _, is_valid, failure_type = build_canonical_rdmol_from_sample(
            x=x,
            categorical=categorical,
            nodesxsample=nodesxsample,
            sample_idx=sample_idx,
            dataset_info=dataset_info,
        )
        if not is_valid or mol is None:
            return False, failure_type or FAILURE_DOCKING_INPUT
        # Prepare for docking: add H + charges before SDF export
        if add_hydrogens or assign_gasteiger:
            mol = prepare_mol_for_docking(mol, add_hydrogens, assign_gasteiger)
            if mol is None:
                return False, FAILURE_DOCKING_INPUT
        if not write_rdmol_to_sdf(mol, sdf_path):
            return False, FAILURE_DOCKING_INPUT
        return True, None
    except Exception as exc:
        warnings.warn(f"[DockingPreparation] canonical mol build failed: {exc}")
        return False, FAILURE_DOCKING_INPUT


def prepare_batch_sdf_inputs(
    x,
    categorical,
    nodesxsample,
    batch_size: int,
    dataset_info,
    prefix: str,
    data: "DataProto" = None,
    *,
    add_hydrogens: bool = True,
    assign_gasteiger: bool = True,
) -> PreparedLigandBatch:
    """Prepare one SDF file per valid sample for backend batch execution.

    If data.batch contains pre-built 'rdmols' from
    annotate_batch_molecule_semantics(), they are reused directly.
    Otherwise falls back to building each mol from tensors (backward compat).

    Args:
        add_hydrogens: If True, add explicit H before SDF export for docking.
        assign_gasteiger: If True, assign Gasteiger charges before SDF export.
    """
    tmpdir = tempfile.mkdtemp(prefix=prefix)
    ligand_files: List[str] = []
    sample_indices: List[int] = []
    failures: Dict[int, str] = {}

    # Try to reuse mainline rdmols (stored in non_tensor_batch, not TensorDict)
    rdmols = None
    if data is not None:
        ntb = getattr(data, "non_tensor_batch", None)
        if isinstance(ntb, dict):
            rdmols = ntb.get("rdmols")

    for sample_idx in range(batch_size):
        sdf_path = os.path.join(tmpdir, f"mol_{sample_idx:04d}.sdf")

        # Path A: reuse pre-built canonical mol from mainline
        if rdmols is not None and sample_idx < len(rdmols):
            mol = rdmols[sample_idx]
            if mol is not None:
                ok, failure = _build_sample_to_sdf_from_mol(
                    mol, sdf_path,
                    add_hydrogens=add_hydrogens,
                    assign_gasteiger=assign_gasteiger,
                )
                if ok and os.path.exists(sdf_path) and os.path.getsize(sdf_path) > 0:
                    ligand_files.append(sdf_path)
                    sample_indices.append(sample_idx)
                    continue
                else:
                    failures[sample_idx] = failure or FAILURE_DOCKING_INPUT
                    continue

        # Path B: fallback — build locally (backward compatibility)
        ok, failure = build_sample_to_sdf(
            x=x,
            categorical=categorical,
            nodesxsample=nodesxsample,
            sample_idx=sample_idx,
            sdf_path=sdf_path,
            dataset_info=dataset_info,
            add_hydrogens=add_hydrogens,
            assign_gasteiger=assign_gasteiger,
        )
        if ok and os.path.exists(sdf_path) and os.path.getsize(sdf_path) > 0:
            ligand_files.append(sdf_path)
            sample_indices.append(sample_idx)
        else:
            failures[sample_idx] = failure or FAILURE_DOCKING_INPUT

    return PreparedLigandBatch(
        tmpdir=tmpdir,
        ligand_files=ligand_files,
        sample_indices=sample_indices,
        failures=failures,
    )


def prepare_single_sdf_input(
    x,
    categorical,
    nodesxsample,
    sample_idx: int,
    dataset_info,
    *,
    add_hydrogens: bool = True,
    assign_gasteiger: bool = True,
) -> PreparedLigandFile:
    """Prepare one temporary SDF file for sequential backend execution.

    Args:
        add_hydrogens: If True, add explicit H before SDF export for docking.
        assign_gasteiger: If True, assign Gasteiger charges before SDF export.
    """
    with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as tmp:
        sdf_path = tmp.name

    ok, failure = build_sample_to_sdf(
        x=x,
        categorical=categorical,
        nodesxsample=nodesxsample,
        sample_idx=sample_idx,
        sdf_path=sdf_path,
        dataset_info=dataset_info,
        add_hydrogens=add_hydrogens,
        assign_gasteiger=assign_gasteiger,
    )
    if not ok:
        prepared = PreparedLigandFile(path=sdf_path, failure_type=failure or FAILURE_DOCKING_INPUT)
        prepared.cleanup()
        return PreparedLigandFile(path=None, failure_type=failure or FAILURE_DOCKING_INPUT)

    if not os.path.exists(sdf_path) or os.path.getsize(sdf_path) == 0:
        prepared = PreparedLigandFile(path=sdf_path, failure_type=FAILURE_DOCKING_INPUT)
        prepared.cleanup()
        return PreparedLigandFile(path=None, failure_type=FAILURE_DOCKING_INPUT)

    return PreparedLigandFile(path=sdf_path, failure_type=None)
