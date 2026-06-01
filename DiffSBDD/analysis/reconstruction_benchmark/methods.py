"""Reconstruction method registry for generated atom-cloud benchmarks."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from rdkit import Chem

from reconstruction import reconstruct_molecule


@dataclass(frozen=True)
class ReconstructionAttempt:
    """Raw output from one reconstruction method before metric calculation."""

    method: str
    raw_mol: Chem.Mol | None
    build_error: str
    seconds: float


class ReconstructionMethod(Protocol):
    """Protocol implemented by benchmark reconstruction backends."""

    name: str

    def reconstruct(
        self,
        coords: torch.Tensor,
        atom_types: torch.Tensor,
        dataset_info: dict[str, Any],
    ) -> ReconstructionAttempt:
        """Reconstruct one RDKit molecule from one generated atom cloud."""


@dataclass(frozen=True)
class DiffSBDDReconstructionMethod:
    """Wrapper around the project-level DiffSBDD reconstruction dispatcher."""

    name: str
    backend_method: str

    def reconstruct(
        self,
        coords: torch.Tensor,
        atom_types: torch.Tensor,
        dataset_info: dict[str, Any],
    ) -> ReconstructionAttempt:
        """Run one configured DiffSBDD reconstruction backend."""
        started_at = time.perf_counter()
        try:
            mol = reconstruct_molecule(
                coords,
                atom_types,
                dataset_info,
                method=self.backend_method,
                add_coords=True,
            )
            build_error = ""
        except Exception as exc:  # noqa: BLE001 - benchmark records failures per method.
            mol = None
            build_error = type(exc).__name__
        return ReconstructionAttempt(
            method=self.name,
            raw_mol=mol,
            build_error=build_error,
            seconds=time.perf_counter() - started_at,
        )


def create_reconstruction_methods(method_names: list[str], dataset_info: dict[str, Any]) -> list[ReconstructionMethod]:
    """Build reconstruction method instances from CLI names."""
    del dataset_info
    aliases = {
        "openbabel": DiffSBDDReconstructionMethod("openbabel", "openbabel"),
        "obabel": DiffSBDDReconstructionMethod("openbabel", "openbabel"),
        "edm": DiffSBDDReconstructionMethod("edm", "edm"),
        "distance": DiffSBDDReconstructionMethod("edm", "edm"),
        "distance_rule": DiffSBDDReconstructionMethod("edm", "edm"),
    }
    methods: list[ReconstructionMethod] = []
    seen: set[str] = set()
    for requested_name in method_names:
        key = requested_name.strip().lower().replace("-", "_")
        if key not in aliases:
            supported = ", ".join(sorted(aliases))
            raise ValueError(f"Unsupported reconstruction benchmark method {requested_name!r}. Supported: {supported}")
        method = aliases[key]
        if method.name not in seen:
            methods.append(method)
            seen.add(method.name)
    return methods


def _resource_row(name: str, kind: str, checkpoint: Path | None, config: Path | None, source: Path) -> dict[str, object]:
    """Create one external-resource probe row."""
    checkpoint_exists = checkpoint.exists() if checkpoint is not None else False
    config_exists = config.exists() if config is not None else False
    source_exists = source.exists()
    if checkpoint is None:
        status = "source_found_no_checkpoint" if source_exists else "missing_source"
    elif checkpoint_exists and source_exists and (config is None or config_exists):
        status = "resource_found_adapter_pending"
    else:
        status = "missing_resource"
    return {
        "name": name,
        "kind": kind,
        "checkpoint": str(checkpoint) if checkpoint is not None else "",
        "checkpoint_exists": int(checkpoint_exists),
        "config": str(config) if config is not None else "",
        "config_exists": int(config_exists),
        "source": str(source),
        "source_exists": int(source_exists),
        "status": status,
    }


def probe_external_bond_resources(project_root: Path) -> list[dict[str, object]]:
    """Probe local reference-project resources for future bond-predictor adapters."""
    return [
        _resource_row(
            name="diffgui_pretrained_bond_predictor",
            kind="external_pretrained_bond_predictor",
            checkpoint=project_root / "reference project" / "DiffGui" / "ckpt" / "bond_trained.pt",
            config=project_root / "reference project" / "DiffGui" / "configs" / "train" / "train_bond.yml",
            source=project_root / "reference project" / "DiffGui" / "models" / "bond_predictor.py",
        ),
        _resource_row(
            name="molsnapper_pretrained_bond_predictor",
            kind="external_pretrained_bond_predictor",
            checkpoint=project_root / "reference project" / "modern_sbdd_20260531" / "MolSnapper" / "ckpt" / "bondpred.pt",
            config=None,
            source=project_root / "reference project" / "modern_sbdd_20260531" / "MolSnapper" / "models" / "bond_predictor.py",
        ),
        _resource_row(
            name="decompopt_generated_with_bond_reconstruct",
            kind="external_reconstruction_with_predicted_bonds",
            checkpoint=None,
            config=None,
            source=project_root
            / "reference project"
            / "modern_sbdd_20260531"
            / "DecompOpt"
            / "scripts"
            / "sample_diffusion_decomp_compose.py",
        ),
    ]
