"""Reconstruction method registry for generated atom-cloud benchmarks."""

from __future__ import annotations

import os
import pickle
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from rdkit import Chem, Geometry

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
        context: dict[str, Any] | None = None,
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
        context: dict[str, Any] | None = None,
    ) -> ReconstructionAttempt:
        """Run one configured DiffSBDD reconstruction backend."""
        del context
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


EXTERNAL_PROVIDER_DEFAULTS = {
    "diffgui": {
        "name": "diffgui_bond",
        "env": "diffgui",
        "root_parts": ("reference project", "DiffGui"),
        "checkpoint_parts": ("ckpt", "bond_trained.pt"),
    },
    "molsnapper": {
        "name": "molsnapper_bond",
        "env": "molsnapper_modern_ref",
        "root_parts": ("reference project", "modern_sbdd_20260531", "MolSnapper"),
        "checkpoint_parts": ("ckpt", "bondpred.pt"),
    },
}

_ATOMIC_NUMBERS = {
    "H": 1,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Br": 35,
    "I": 53,
}

_BOND_TYPE_BY_ID = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
    4: Chem.BondType.AROMATIC,
}


@dataclass(frozen=True)
class ExternalBondPredictorBatchMethod:
    """Batch adapter for pretrained external bond predictors in their own conda envs."""

    name: str
    provider: str
    project_root: Path
    provider_root: Path
    checkpoint: Path
    conda_env: str
    worker_script: Path
    cache_dir: Path
    device: str = "cpu"

    def reconstruct(
        self,
        coords: torch.Tensor,
        atom_types: torch.Tensor,
        dataset_info: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> ReconstructionAttempt:
        """Run one external predictor through the batch path for API compatibility."""
        sample = {
            "sample_idx": 0,
            "coords": coords,
            "atom_types": atom_types,
            "context": context or {},
        }
        return self.reconstruct_many([sample], dataset_info)[0][1]

    def reconstruct_many(
        self,
        samples: list[dict[str, Any]],
        dataset_info: dict[str, Any],
    ) -> list[tuple[dict[str, Any], ReconstructionAttempt]]:
        """Predict bonds for many atom clouds with one external worker process."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        input_path = self.cache_dir / f"{self.name}_input.pkl"
        output_path = self.cache_dir / f"{self.name}_output.pkl"
        stdout_path = self.cache_dir / f"{self.name}_worker.stdout.log"
        stderr_path = self.cache_dir / f"{self.name}_worker.stderr.log"

        payload = {
            "samples": [self._serialize_sample(sample, dataset_info) for sample in samples],
            "provider_root": str(self.provider_root),
            "checkpoint": str(self.checkpoint),
            "device": self.device,
        }
        with input_path.open("wb") as handle:
            pickle.dump(payload, handle)

        conda_exe = os.environ.get("CONDA_EXE") or "conda"
        command = [
            conda_exe,
            "run",
            "-n",
            self.conda_env,
            "python",
            str(self.worker_script),
            "--provider",
            self.provider,
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]
        started_at = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=self.project_root,
            check=False,
            text=True,
            capture_output=True,
        )
        worker_wall_seconds = time.perf_counter() - started_at
        stdout_path.write_text(completed.stdout)
        stderr_path.write_text(completed.stderr)
        if completed.returncode != 0:
            return [
                (
                    sample,
                    ReconstructionAttempt(
                        method=self.name,
                        raw_mol=None,
                        build_error=f"ExternalWorkerFailed:{completed.returncode}",
                        seconds=worker_wall_seconds / max(1, len(samples)),
                    ),
                )
                for sample in samples
            ]

        with output_path.open("rb") as handle:
            results = pickle.load(handle)
        result_by_sample_idx = {int(item["sample_idx"]): item for item in results}

        attempts: list[tuple[dict[str, Any], ReconstructionAttempt]] = []
        for sample in samples:
            sample_idx = int(sample["sample_idx"])
            result = result_by_sample_idx.get(sample_idx)
            if result is None:
                attempt = ReconstructionAttempt(
                    method=self.name,
                    raw_mol=None,
                    build_error="ExternalWorkerMissingResult",
                    seconds=0.0,
                )
            else:
                attempt = self._attempt_from_worker_result(sample, dataset_info, result)
            attempts.append((sample, attempt))
        return attempts

    @staticmethod
    def _serialize_sample(sample: dict[str, Any], dataset_info: dict[str, Any]) -> dict[str, Any]:
        """Convert one point-cloud sample to pickle-friendly numpy arrays."""
        atom_types = sample["atom_types"]
        atom_symbols = [dataset_info["atom_decoder"][int(atom_idx)] for atom_idx in atom_types]
        context = sample.get("context", {})
        serialized = {
            "sample_idx": int(sample["sample_idx"]),
            "coords": sample["coords"].detach().cpu().numpy().astype(np.float32),
            "atom_symbols": atom_symbols,
        }
        if context.get("protein_node") is not None and context.get("protein_pos") is not None:
            serialized["protein_node"] = context["protein_node"].detach().cpu().numpy().astype(np.float32)
            serialized["protein_pos"] = context["protein_pos"].detach().cpu().numpy().astype(np.float32)
        return serialized

    def _attempt_from_worker_result(
        self,
        sample: dict[str, Any],
        dataset_info: dict[str, Any],
        result: dict[str, Any],
    ) -> ReconstructionAttempt:
        """Build an RDKit molecule from external worker bond predictions."""
        started_at = time.perf_counter()
        worker_error = str(result.get("error") or "")
        worker_seconds = float(result.get("seconds") or 0.0)
        if worker_error:
            return ReconstructionAttempt(
                method=self.name,
                raw_mol=None,
                build_error=worker_error,
                seconds=worker_seconds,
            )
        try:
            atom_types = sample["atom_types"]
            atom_symbols = [dataset_info["atom_decoder"][int(atom_idx)] for atom_idx in atom_types]
            mol = _build_molecule_from_predicted_halfedges(
                sample["coords"],
                atom_symbols,
                np.asarray(result["bond_index"], dtype=np.int64),
                np.asarray(result["bond_type"], dtype=np.int64),
            )
            build_error = ""
        except Exception as exc:  # noqa: BLE001 - benchmark records failed predictions.
            mol = None
            build_error = type(exc).__name__
        return ReconstructionAttempt(
            method=self.name,
            raw_mol=mol,
            build_error=build_error,
            seconds=worker_seconds + time.perf_counter() - started_at,
        )


def _build_molecule_from_predicted_halfedges(
    coords: torch.Tensor,
    atom_symbols: list[str],
    bond_index: np.ndarray,
    bond_type: np.ndarray,
) -> Chem.Mol:
    """Build an RDKit molecule from atom coordinates and predicted halfedges."""
    rd_mol = Chem.RWMol()
    conformer = Chem.Conformer(len(atom_symbols))
    coords_np = coords.detach().cpu().numpy()
    for atom_idx, symbol in enumerate(atom_symbols):
        atomic_number = _ATOMIC_NUMBERS.get(symbol)
        if atomic_number is None:
            raise ValueError(f"Unsupported atom symbol for predicted-bond reconstruction: {symbol}")
        atom = Chem.Atom(atomic_number)
        rd_mol.AddAtom(atom)
        conformer.SetAtomPosition(atom_idx, Geometry.Point3D(*map(float, coords_np[atom_idx])))
    rd_mol.AddConformer(conformer)

    if bond_index.shape[0] != 2:
        raise ValueError(f"Expected bond_index shape [2, E], got {bond_index.shape}")
    for edge_idx, edge_type in enumerate(bond_type.tolist()):
        bond_id = int(edge_type)
        if bond_id <= 0:
            continue
        begin_idx = int(bond_index[0, edge_idx])
        end_idx = int(bond_index[1, edge_idx])
        if begin_idx == end_idx or begin_idx < 0 or end_idx < 0:
            continue
        if begin_idx >= len(atom_symbols) or end_idx >= len(atom_symbols):
            continue
        if rd_mol.GetBondBetweenAtoms(begin_idx, end_idx) is not None:
            continue
        rdkit_bond_type = _BOND_TYPE_BY_ID.get(bond_id)
        if rdkit_bond_type is None:
            continue
        if rdkit_bond_type == Chem.BondType.AROMATIC:
            rd_mol.GetAtomWithIdx(begin_idx).SetIsAromatic(True)
            rd_mol.GetAtomWithIdx(end_idx).SetIsAromatic(True)
        rd_mol.AddBond(begin_idx, end_idx, rdkit_bond_type)
        if rdkit_bond_type == Chem.BondType.AROMATIC:
            rd_mol.GetBondBetweenAtoms(begin_idx, end_idx).SetIsAromatic(True)
    return rd_mol.GetMol()


def create_reconstruction_methods(
    method_names: list[str],
    dataset_info: dict[str, Any],
    *,
    project_root: Path | None = None,
    cache_dir: Path | None = None,
    external_device: str = "cpu",
    diffgui_env: str = "diffgui",
    molsnapper_env: str = "molsnapper_modern_ref",
) -> list[ReconstructionMethod]:
    """Build reconstruction method instances from CLI names."""
    del dataset_info
    root = Path.cwd() if project_root is None else project_root
    external_cache_dir = root / "analysis" / "external_bond_predictor_cache" if cache_dir is None else cache_dir
    worker_script = Path(__file__).resolve().with_name("external_bond_predictor_worker.py")
    diffgui_defaults = EXTERNAL_PROVIDER_DEFAULTS["diffgui"]
    molsnapper_defaults = EXTERNAL_PROVIDER_DEFAULTS["molsnapper"]
    diffgui_method = ExternalBondPredictorBatchMethod(
        name=str(diffgui_defaults["name"]),
        provider="diffgui",
        project_root=root,
        provider_root=root.joinpath(*diffgui_defaults["root_parts"]),
        checkpoint=root.joinpath(*diffgui_defaults["root_parts"], *diffgui_defaults["checkpoint_parts"]),
        conda_env=diffgui_env,
        worker_script=worker_script,
        cache_dir=external_cache_dir / "diffgui",
        device=external_device,
    )
    molsnapper_method = ExternalBondPredictorBatchMethod(
        name=str(molsnapper_defaults["name"]),
        provider="molsnapper",
        project_root=root,
        provider_root=root.joinpath(*molsnapper_defaults["root_parts"]),
        checkpoint=root.joinpath(*molsnapper_defaults["root_parts"], *molsnapper_defaults["checkpoint_parts"]),
        conda_env=molsnapper_env,
        worker_script=worker_script,
        cache_dir=external_cache_dir / "molsnapper",
        device=external_device,
    )
    aliases = {
        "openbabel": DiffSBDDReconstructionMethod("openbabel", "openbabel"),
        "obabel": DiffSBDDReconstructionMethod("openbabel", "openbabel"),
        "edm": DiffSBDDReconstructionMethod("edm", "edm"),
        "distance": DiffSBDDReconstructionMethod("edm", "edm"),
        "distance_rule": DiffSBDDReconstructionMethod("edm", "edm"),
        "diffgui": diffgui_method,
        "diffgui_bond": diffgui_method,
        "diffgui_pretrained": diffgui_method,
        "molsnapper": molsnapper_method,
        "molsnapper_bond": molsnapper_method,
        "molsnapper_pretrained": molsnapper_method,
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


def _resource_row(
    name: str,
    kind: str,
    checkpoint: Path | None,
    config: Path | None,
    source: Path,
    *,
    adapter_available: bool = False,
) -> dict[str, object]:
    """Create one external-resource probe row."""
    checkpoint_exists = checkpoint.exists() if checkpoint is not None else False
    config_exists = config.exists() if config is not None else False
    source_exists = source.exists()
    if checkpoint is None:
        status = "source_found_no_checkpoint" if source_exists else "missing_source"
    elif adapter_available and checkpoint_exists and source_exists and (config is None or config_exists):
        status = "resource_found_adapter_available"
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
            adapter_available=True,
        ),
        _resource_row(
            name="molsnapper_pretrained_bond_predictor",
            kind="external_pretrained_bond_predictor",
            checkpoint=project_root / "reference project" / "modern_sbdd_20260531" / "MolSnapper" / "ckpt" / "bondpred.pt",
            config=None,
            source=project_root / "reference project" / "modern_sbdd_20260531" / "MolSnapper" / "models" / "bond_predictor.py",
            adapter_available=True,
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
