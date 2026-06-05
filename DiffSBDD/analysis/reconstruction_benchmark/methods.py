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
import torch.nn.functional as F
from rdkit import Chem, Geometry

from local_bond_predictor.dataset import (
    PairFeatureConfig,
    build_pair_features,
    pair_feature_dim,
)
from local_bond_predictor.model import PairBondPredictor
from edge_diffusion import (
    EdgeBatch,
    EdgeDenoiserConfig,
    EdgeDenoiserMLP,
    EdgeDenoiserNodeEdge,
)
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

DEFAULT_LOCAL_BOND_CHECKPOINT = Path("analysis/local_bond_predictor_v0_20260602_small/best_model.pt")
DEFAULT_EDGE_DENOISER_B_CHECKPOINT = Path(
    "analysis/edge_denoiser_ablation_20260603_B_diffgui_noise005/edge_denoiser_overfit.pt"
)
DEFAULT_EDGE_DENOISER_C_CHECKPOINT = Path(
    "analysis/edge_denoiser_ablation_20260603_C_diffgui_noise005_focal2_gpu/edge_denoiser_overfit.pt"
)


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


@dataclass
class LocalBondPredictorMethod:
    """Batch adapter for the decoupled local pair-wise bond predictor."""

    name: str
    checkpoint: Path
    device: str = "cpu"

    def __post_init__(self) -> None:
        """Load the predictor checkpoint once for all reconstruction calls."""
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"Local bond predictor checkpoint not found: {self.checkpoint}")

        self.torch_device = torch.device(self.device)
        checkpoint = torch.load(self.checkpoint, map_location=self.torch_device)
        feature_config_raw = checkpoint.get("feature_config", {})
        self.feature_config = PairFeatureConfig(**feature_config_raw)
        self.input_dim = int(checkpoint["input_dim"])
        distance_dim = self.feature_config.distance_feature_dim
        if (self.input_dim - distance_dim) % 2 != 0:
            raise ValueError(
                f"Invalid local bond predictor input_dim={self.input_dim} "
                f"for distance_dim={distance_dim}."
            )
        self.atom_feature_dim = (self.input_dim - distance_dim) // 2
        expected_input_dim = pair_feature_dim(self.atom_feature_dim, self.feature_config)
        if expected_input_dim != self.input_dim:
            raise ValueError(
                f"Local bond predictor input mismatch: checkpoint={self.input_dim}, "
                f"expected={expected_input_dim}."
            )

        args = checkpoint.get("args", {})
        model = PairBondPredictor(
            self.input_dim,
            hidden_dim=int(args.get("hidden_dim", 256)),
            num_layers=int(args.get("num_layers", 3)),
            dropout=float(args.get("dropout", 0.1)),
            num_edge_types=int(args.get("num_edge_types", 5)),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.torch_device)
        model.eval()
        self.model = model

    def reconstruct(
        self,
        coords: torch.Tensor,
        atom_types: torch.Tensor,
        dataset_info: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> ReconstructionAttempt:
        """Predict bonds and reconstruct one RDKit molecule."""
        sample = {
            "sample_idx": 0,
            "coords": coords,
            "atom_types": atom_types,
            "context": context or {},
        }
        return self.reconstruct_many([sample], dataset_info)[0][1]

    @torch.no_grad()
    def reconstruct_many(
        self,
        samples: list[dict[str, Any]],
        dataset_info: dict[str, Any],
    ) -> list[tuple[dict[str, Any], ReconstructionAttempt]]:
        """Predict local bond types for many generated atom clouds."""
        attempts: list[tuple[dict[str, Any], ReconstructionAttempt]] = []
        for sample in samples:
            started_at = time.perf_counter()
            try:
                halfedge_index, predicted_type = self._predict_halfedges(
                    sample["coords"],
                    sample["atom_types"],
                )
                atom_symbols = [
                    dataset_info["atom_decoder"][int(atom_idx)]
                    for atom_idx in sample["atom_types"]
                ]
                mol = _build_molecule_from_predicted_halfedges(
                    sample["coords"],
                    atom_symbols,
                    halfedge_index.detach().cpu().numpy(),
                    predicted_type.detach().cpu().numpy(),
                )
                build_error = ""
            except Exception as exc:  # noqa: BLE001 - benchmark records failed predictions.
                mol = None
                build_error = type(exc).__name__
            attempts.append((
                sample,
                ReconstructionAttempt(
                    method=self.name,
                    raw_mol=mol,
                    build_error=build_error,
                    seconds=time.perf_counter() - started_at,
                ),
            ))
        return attempts

    def _predict_halfedges(
        self,
        coords: torch.Tensor,
        atom_types: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return upper-triangle halfedge indices and predicted bond-type ids."""
        coords_device = coords.to(self.torch_device, dtype=torch.float32)
        atom_types_device = atom_types.to(self.torch_device, dtype=torch.long)
        if atom_types_device.numel() == 0:
            return (
                torch.zeros((2, 0), dtype=torch.long, device=self.torch_device),
                torch.zeros((0,), dtype=torch.long, device=self.torch_device),
            )
        max_atom_type = int(atom_types_device.max().item())
        if max_atom_type >= self.atom_feature_dim:
            raise ValueError(
                f"Generated atom type id {max_atom_type} exceeds local predictor "
                f"atom feature dim {self.atom_feature_dim}."
            )

        num_atoms = int(atom_types_device.shape[0])
        halfedge_index = torch.triu_indices(
            num_atoms,
            num_atoms,
            offset=1,
            device=self.torch_device,
        )
        atom_features = F.one_hot(
            atom_types_device,
            num_classes=self.atom_feature_dim,
        ).to(dtype=coords_device.dtype)
        features = build_pair_features(
            coords_device,
            atom_features,
            halfedge_index,
            self.feature_config,
        )
        if features.numel() == 0:
            return halfedge_index, torch.zeros((0,), dtype=torch.long, device=self.torch_device)
        logits = self.model(features)
        return halfedge_index, logits.argmax(dim=-1)


@dataclass
class EdgeDenoiserReconstructionMethod:
    """Use a standalone edge denoiser checkpoint to reconstruct generated atom clouds."""

    name: str
    checkpoint: Path
    device: str = "cpu"
    timestep: int = 999
    init_mode: str = "uniform_random"
    seed: int = 20260603

    def __post_init__(self) -> None:
        """Load the edge denoiser checkpoint once for all samples."""
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"Edge denoiser checkpoint not found: {self.checkpoint}")
        if self.init_mode not in {"uniform_random", "all_no_bond"}:
            raise ValueError(
                "edge denoiser init_mode must be one of: uniform_random, all_no_bond"
            )

        self.torch_device = torch.device(self.device)
        checkpoint = torch.load(self.checkpoint, map_location=self.torch_device)
        config = EdgeDenoiserConfig(**checkpoint["config"])
        summary = checkpoint.get("summary", {})
        model_kind = str(summary.get("model_kind", "node_edge"))
        if model_kind == "node_edge":
            model = EdgeDenoiserNodeEdge(config)
        elif model_kind == "mlp":
            model = EdgeDenoiserMLP(config)
        else:
            raise ValueError(f"Unsupported edge denoiser model_kind: {model_kind}")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.torch_device)
        model.eval()
        self.model = model
        self.config = config

    def reconstruct(
        self,
        coords: torch.Tensor,
        atom_types: torch.Tensor,
        dataset_info: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> ReconstructionAttempt:
        """Predict generated-cloud halfedge types and build one RDKit molecule."""
        sample = {
            "sample_idx": 0,
            "coords": coords,
            "atom_types": atom_types,
            "context": context or {},
        }
        return self.reconstruct_many([sample], dataset_info)[0][1]

    @torch.no_grad()
    def reconstruct_many(
        self,
        samples: list[dict[str, Any]],
        dataset_info: dict[str, Any],
    ) -> list[tuple[dict[str, Any], ReconstructionAttempt]]:
        """Predict edge-denoiser bond types for many generated atom clouds."""
        attempts: list[tuple[dict[str, Any], ReconstructionAttempt]] = []
        for sample in samples:
            started_at = time.perf_counter()
            try:
                halfedge_index, predicted_type = self._predict_halfedges(
                    sample["coords"],
                    sample["atom_types"],
                    int(sample.get("sample_idx", 0)),
                )
                atom_symbols = [
                    dataset_info["atom_decoder"][int(atom_idx)]
                    for atom_idx in sample["atom_types"]
                ]
                mol = _build_molecule_from_predicted_halfedges(
                    sample["coords"],
                    atom_symbols,
                    halfedge_index.detach().cpu().numpy(),
                    predicted_type.detach().cpu().numpy(),
                )
                build_error = ""
            except Exception as exc:  # noqa: BLE001 - benchmark records failed predictions.
                mol = None
                build_error = type(exc).__name__
            attempts.append((
                sample,
                ReconstructionAttempt(
                    method=self.name,
                    raw_mol=mol,
                    build_error=build_error,
                    seconds=time.perf_counter() - started_at,
                ),
            ))
        return attempts

    def _predict_halfedges(
        self,
        coords: torch.Tensor,
        atom_types: torch.Tensor,
        sample_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return upper-triangle halfedge indices and generated bond-type ids."""
        coords_device = coords.to(self.torch_device, dtype=torch.float32)
        atom_types_device = atom_types.to(self.torch_device, dtype=torch.long)
        num_atoms = int(atom_types_device.shape[0])
        halfedge_index = torch.triu_indices(
            num_atoms,
            num_atoms,
            offset=1,
            device=self.torch_device,
        )
        if num_atoms == 0 or halfedge_index.shape[1] == 0:
            return halfedge_index, torch.zeros((0,), dtype=torch.long, device=self.torch_device)

        max_atom_type = int(atom_types_device.max().item())
        if max_atom_type >= int(self.config.num_atom_types):
            raise ValueError(
                f"Generated atom type id {max_atom_type} exceeds edge denoiser "
                f"num_atom_types {self.config.num_atom_types}."
            )

        target_edge_type = torch.zeros(
            (halfedge_index.shape[1],),
            dtype=torch.long,
            device=self.torch_device,
        )
        batch = EdgeBatch(
            ligand_pos=coords_device,
            ligand_atom_type=atom_types_device,
            halfedge_index=halfedge_index,
            target_edge_type=target_edge_type,
            atom_batch=torch.zeros((num_atoms,), dtype=torch.long, device=self.torch_device),
            edge_batch=torch.zeros((halfedge_index.shape[1],), dtype=torch.long, device=self.torch_device),
            names=[str(sample_idx)],
        )
        noised_edge_type = self._initial_edge_type(
            halfedge_index.shape[1],
            sample_idx=sample_idx,
        )
        t = torch.tensor([int(self.timestep)], dtype=torch.long, device=self.torch_device)
        logits = self.model(batch, noised_edge_type, t)
        return halfedge_index, logits.argmax(dim=-1)

    def _initial_edge_type(self, num_halfedges: int, *, sample_idx: int) -> torch.Tensor:
        """Create generated-time edge inputs without using unavailable true labels."""
        if self.init_mode == "all_no_bond":
            return torch.zeros((num_halfedges,), dtype=torch.long, device=self.torch_device)
        generator = torch.Generator(device=self.torch_device)
        generator.manual_seed(int(self.seed) + int(sample_idx))
        return torch.randint(
            low=0,
            high=int(self.config.num_edge_types),
            size=(num_halfedges,),
            device=self.torch_device,
            generator=generator,
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
    local_bond_checkpoint: Path | None = None,
    local_bond_device: str = "cpu",
    edge_denoiser_b_checkpoint: Path | None = None,
    edge_denoiser_c_checkpoint: Path | None = None,
    edge_denoiser_device: str = "cpu",
    edge_denoiser_timestep: int = 999,
    edge_denoiser_init_mode: str = "uniform_random",
    edge_denoiser_seed: int = 20260603,
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
    local_checkpoint = DEFAULT_LOCAL_BOND_CHECKPOINT if local_bond_checkpoint is None else local_bond_checkpoint
    if not local_checkpoint.is_absolute():
        local_checkpoint = root / local_checkpoint
    local_method: LocalBondPredictorMethod | None = None
    edge_b_checkpoint = (
        DEFAULT_EDGE_DENOISER_B_CHECKPOINT
        if edge_denoiser_b_checkpoint is None
        else edge_denoiser_b_checkpoint
    )
    edge_c_checkpoint = (
        DEFAULT_EDGE_DENOISER_C_CHECKPOINT
        if edge_denoiser_c_checkpoint is None
        else edge_denoiser_c_checkpoint
    )
    if not edge_b_checkpoint.is_absolute():
        edge_b_checkpoint = root / edge_b_checkpoint
    if not edge_c_checkpoint.is_absolute():
        edge_c_checkpoint = root / edge_c_checkpoint
    edge_b_method: EdgeDenoiserReconstructionMethod | None = None
    edge_c_method: EdgeDenoiserReconstructionMethod | None = None

    def get_local_method() -> LocalBondPredictorMethod:
        """Lazily load the local predictor only when the benchmark requests it."""
        nonlocal local_method
        if local_method is None:
            local_method = LocalBondPredictorMethod(
                name="local_bond_v0",
                checkpoint=local_checkpoint,
                device=local_bond_device,
            )
        return local_method

    def get_edge_b_method() -> EdgeDenoiserReconstructionMethod:
        """Lazily load the B edge denoiser only when requested."""
        nonlocal edge_b_method
        if edge_b_method is None:
            edge_b_method = EdgeDenoiserReconstructionMethod(
                name="edge_denoiser_b",
                checkpoint=edge_b_checkpoint,
                device=edge_denoiser_device,
                timestep=edge_denoiser_timestep,
                init_mode=edge_denoiser_init_mode,
                seed=edge_denoiser_seed,
            )
        return edge_b_method

    def get_edge_c_method() -> EdgeDenoiserReconstructionMethod:
        """Lazily load the C edge denoiser only when requested."""
        nonlocal edge_c_method
        if edge_c_method is None:
            edge_c_method = EdgeDenoiserReconstructionMethod(
                name="edge_denoiser_c",
                checkpoint=edge_c_checkpoint,
                device=edge_denoiser_device,
                timestep=edge_denoiser_timestep,
                init_mode=edge_denoiser_init_mode,
                seed=edge_denoiser_seed,
            )
        return edge_c_method

    local_aliases = {
        "local",
        "local_bond",
        "local_bond_v0",
        "local_bond_predictor",
    }
    edge_b_aliases = {"edge_b", "edge_denoiser_b", "edge_denoiser_bond_b"}
    edge_c_aliases = {"edge_c", "edge_denoiser_c", "edge_denoiser_bond_c"}
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
        if key in local_aliases:
            method = get_local_method()
        elif key in edge_b_aliases:
            method = get_edge_b_method()
        elif key in edge_c_aliases:
            method = get_edge_c_method()
        elif key in aliases:
            method = aliases[key]
        else:
            supported = ", ".join(sorted(set(aliases) | local_aliases | edge_b_aliases | edge_c_aliases))
            raise ValueError(f"Unsupported reconstruction benchmark method {requested_name!r}. Supported: {supported}")
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
