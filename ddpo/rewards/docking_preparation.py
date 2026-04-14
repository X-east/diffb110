from __future__ import annotations

"""Ligand preparation helpers for docking reward backends.

Phase-2 purpose:
- keep canonical molecule -> docking input preparation outside `DockingReward`
- preserve current mainline behavior: canonical RDKit molecule exported to SDF
- keep failure boundaries explicit as docking-input failures
"""

import os
import tempfile
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ddpo.utils.molecule_semantics import (
    FAILURE_DOCKING_INPUT,
    build_canonical_rdmol_from_sample,
    write_rdmol_to_sdf,
)


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


def build_sample_to_sdf(
    x,
    categorical,
    nodesxsample,
    sample_idx: int,
    sdf_path: str,
    dataset_info,
) -> Tuple[bool, Optional[str]]:
    """Build canonical RDKit molecule for one sample and export to SDF."""
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
) -> PreparedLigandBatch:
    """Prepare one SDF file per valid sample for backend batch execution."""
    tmpdir = tempfile.mkdtemp(prefix=prefix)
    ligand_files: List[str] = []
    sample_indices: List[int] = []
    failures: Dict[int, str] = {}

    for sample_idx in range(batch_size):
        sdf_path = os.path.join(tmpdir, f"mol_{sample_idx:04d}.sdf")
        ok, failure = build_sample_to_sdf(
            x=x,
            categorical=categorical,
            nodesxsample=nodesxsample,
            sample_idx=sample_idx,
            sdf_path=sdf_path,
            dataset_info=dataset_info,
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
) -> PreparedLigandFile:
    """Prepare one temporary SDF file for sequential backend execution."""
    with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as tmp:
        sdf_path = tmp.name

    ok, failure = build_sample_to_sdf(
        x=x,
        categorical=categorical,
        nodesxsample=nodesxsample,
        sample_idx=sample_idx,
        sdf_path=sdf_path,
        dataset_info=dataset_info,
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
