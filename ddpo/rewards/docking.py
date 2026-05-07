"""
DockingReward: Reward function based on molecular docking scores.

Single responsibility: reward semantics layer.
- Prepares molecules to SDF via docking_preparation.prepare_batch_sdf_inputs()
- Delegates docking to a backend via dock_batch_result()
- Maps DockingResult scores to normalised rewards
- Writes rewards and failure_type back to batch

All backend-specific logic lives in docking_backends/.
All DEFAULT_* constants live in docking_config.py.
Backend support-level policy lives in docking_registry.py.
"""

import logging
import os
import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

# Add DiffSBDD to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "DiffSBDD"))

from constants import dataset_params

from ddpo.utils.molecule_semantics import (
    FAILURE_DOCKING_BACKEND,
    FAILURE_DOCKING_INPUT,
    FAILURE_SUCCESS,
)



from .interfaces import BaseReward
from .docking_backends import AutoDockGPUBackend, UniDockBackend, SminaBackend
from .docking_config import (
    ADD_HYDROGENS,
    DEFAULT_AUTOBOX_ADD,
    DEFAULT_AUTOBOX_LIGAND,
    DEFAULT_BACKEND,
    DEFAULT_CENTER_X,
    DEFAULT_CENTER_Y,
    DEFAULT_CENTER_Z,
    DEFAULT_EXHAUSTIVENESS,
    DEFAULT_FLD_FILE,
    DEFAULT_GPU_ID,
    DEFAULT_HEURISTICS,
    DEFAULT_NRUN,
    DEFAULT_RECEPTOR_FILE,
    DEFAULT_SEARCH_MODE,
    DEFAULT_SIZE_X,
    DEFAULT_SIZE_Y,
    DEFAULT_SIZE_Z,
    FAILURE_PENALTY,
    NORMALIZATION_FACTOR,
    SANITIZE_MOLECULES,
)
from .docking_preparation import prepare_batch_sdf_inputs

if TYPE_CHECKING:
    from verl_diffusion.protocol import DataProto


# Legacy alias kept for any external code that imports USE_RAY from here.
USE_RAY = False  # Ray removed from mainline docking path (Bug5 P3)

LOGGER = logging.getLogger("DiffSBDD-DDPO")


class DockingReward(BaseReward):
    """Reward function based on molecular docking scores.

    Normalises raw docking scores to a PPO-compatible range:
        reward = score_raw / normalization_factor
    Typical scores: -12 (strong) to 0 (weak) kcal/mol.
    normalization_factor = -10.0 maps this to ~[0, 1.2].

    Failed molecules (invalid structure or docking failure) receive
    failure_penalty (default -1.0), distinguishable from zero-score success.

    The backend is lazily initialised and cached; gpu_id can be updated after
    construction (e.g. by LightningDDPOModule.setup() to match local_rank).
    """

    def __init__(
        self,
        backend: str = DEFAULT_BACKEND,
        receptor_file: str = DEFAULT_RECEPTOR_FILE,
        normalization_factor: float = NORMALIZATION_FACTOR,
        failure_penalty: float = FAILURE_PENALTY,
        sanitize: bool = SANITIZE_MOLECULES,
        add_hydrogens: bool = ADD_HYDROGENS,
        dataset_info: Optional[dict] = None,
        center_x: Optional[float] = DEFAULT_CENTER_X,
        center_y: Optional[float] = DEFAULT_CENTER_Y,
        center_z: Optional[float] = DEFAULT_CENTER_Z,
        size_x: float = DEFAULT_SIZE_X,
        size_y: float = DEFAULT_SIZE_Y,
        size_z: float = DEFAULT_SIZE_Z,
        exhaustiveness: int = DEFAULT_EXHAUSTIVENESS,
        autobox_ligand: Optional[str] = DEFAULT_AUTOBOX_LIGAND,
        autobox_add: float = DEFAULT_AUTOBOX_ADD,
        chunk_size: Optional[int] = None,
        gpu_id: int = DEFAULT_GPU_ID,
        search_mode: str = DEFAULT_SEARCH_MODE,
        fld_file: Optional[str] = None,
        nrun: int = DEFAULT_NRUN,
        heuristics: bool = DEFAULT_HEURISTICS,
        autodock_gpu_bin: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        self.backend = backend.lower()
        self.receptor_file = str(Path(receptor_file).resolve())
        self.normalization_factor = normalization_factor
        self.failure_penalty = failure_penalty
        self.sanitize = sanitize
        self.add_hydrogens = add_hydrogens
        self.dataset_info = dataset_info or dataset_params["crossdock"]
        self.chunk_size = chunk_size

        # Docking box
        self.center_x = center_x
        self.center_y = center_y
        self.center_z = center_z
        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z
        self.exhaustiveness = exhaustiveness

        # Smina autobox
        self.autobox_ligand = str(Path(autobox_ligand).resolve()) if autobox_ligand else None
        self.autobox_add = autobox_add

        # GPU / backend params
        self.gpu_id = gpu_id
        self.search_mode = search_mode
        self.fld_file = str(Path(fld_file).resolve()) if fld_file else str(Path(DEFAULT_FLD_FILE).resolve())
        self.nrun = nrun
        self.heuristics = heuristics
        self.autodock_gpu_bin = autodock_gpu_bin
        self.verbose = verbose

        # Validate paths
        if not Path(self.receptor_file).exists():
            raise FileNotFoundError(f"Receptor file not found: {self.receptor_file}")
        if self.autobox_ligand and not Path(self.autobox_ligand).exists():
            raise FileNotFoundError(f"Autobox ligand not found: {self.autobox_ligand}")
        if self.backend == "autodock_gpu" and not Path(self.fld_file).exists():
            raise FileNotFoundError(
                f"AutoGrid4 map file not found: {self.fld_file}\n"
                "Generate with: prepare_receptor4.py + prepare_gpf4.py + autogrid4"
            )
        if self.backend not in ["autodock_gpu", "unidock", "smina"]:
            raise ValueError(f"Invalid backend: {self.backend}")

        # Lazy backend caches (re-initialise only when gpu_id changes)
        self._autodock_backend: Optional[AutoDockGPUBackend] = None
        self._unidock_backend: Optional[UniDockBackend] = None

        if self.verbose:
            LOGGER.debug(
                "DockingReward initialized: backend=%s gpu_id=%s receptor=%s box_center=(%s,%s,%s) box_size=(%s,%s,%s)",
                self.backend,
                self.gpu_id,
                Path(self.receptor_file).name,
                self.center_x,
                self.center_y,
                self.center_z,
                self.size_x,
                self.size_y,
                self.size_z,
            )
            if self.backend == "autodock_gpu":
                LOGGER.debug(
                    "DockingReward AutoDock-GPU settings: fld=%s nrun=%s heuristics=%s",
                    Path(self.fld_file).name,
                    self.nrun,
                    self.heuristics,
                )

    # ------------------------------------------------------------------
    # Backend lazy initialisation
    # ------------------------------------------------------------------

    def _get_autodock_backend(self) -> AutoDockGPUBackend:
        """Return cached AutoDockGPUBackend, re-initialising if gpu_id changed."""
        if self._autodock_backend is None or self._autodock_backend.gpu_id != self.gpu_id:
            self._autodock_backend = AutoDockGPUBackend(
                fld_file=self.fld_file,
                gpu_id=self.gpu_id,
                nrun=self.nrun,
                heuristics=self.heuristics,
                autodock_gpu_bin=self.autodock_gpu_bin,
            )
        return self._autodock_backend

    def _get_backend(self):
        """Return the appropriate backend instance for the current batch."""
        if self.backend == "autodock_gpu":
            return self._get_autodock_backend()
        if self.backend == "unidock":
            if self._unidock_backend is None or self._unidock_backend.gpu_id != self.gpu_id:
                self._unidock_backend = UniDockBackend(
                    gpu_id=self.gpu_id,
                    search_mode=self.search_mode,
                    num_modes=1,
                    exhaustiveness=self.exhaustiveness,
                )
            return self._unidock_backend
        if self.backend == "smina":
            return SminaBackend()
        raise ValueError(f"Unknown backend: {self.backend}")

    # ------------------------------------------------------------------
    # Reward calculation
    # ------------------------------------------------------------------

    def calculate_rewards(self, data: "DataProto") -> "DataProto":
        """Calculate docking rewards for a batch of molecules.

        Flow:
            1. Build all molecules to SDF (canonical RDKit path)
            2. Call backend.dock_batch_result() for all valid SDF files
            3. Map DockingResult scores to normalised rewards
            4. Write rewards and failure_type back to batch
        """
        batch = data.batch
        x = batch["x"]
        h = batch["categorical"]
        nodesxsample = batch["nodesxsample"]
        failure_type = batch.get("failure_type", None)
        batch_size = x.shape[0]
        device = x.device

        rewards = [self.failure_penalty] * batch_size
        docking_failures: Dict[int, str] = {}

        _prepared = prepare_batch_sdf_inputs(
            x=x, categorical=h, nodesxsample=nodesxsample,
            batch_size=batch_size,
            dataset_info=self.dataset_info,
            prefix="docking_batch_",
            data=data,
        )
        try:
            sdf_paths = _prepared.ligand_files
            mol_indices = _prepared.sample_indices
            input_failures = _prepared.failures
            docking_failures.update(input_failures)
            LOGGER.debug(
                "[DockingReward] Prepared %s/%s valid SDF inputs on gpu_id=%s",
                len(sdf_paths),
                batch_size,
                self.gpu_id,
            )

            if sdf_paths:
                backend = self._get_backend()
                try:
                    results = backend.dock_batch_result(
                        ligand_files=sdf_paths,
                        receptor_file=self.receptor_file,
                        center_x=self.center_x,
                        center_y=self.center_y,
                        center_z=self.center_z,
                        size_x=self.size_x,
                        size_y=self.size_y,
                        size_z=self.size_z,
                        exhaustiveness=self.exhaustiveness,
                    )
                except Exception as exc:
                    import traceback
                    warnings.warn(
                        f"[DockingReward] dock_batch_result failed: {exc}\n"
                        f"{traceback.format_exc()}"
                    )
                    results = []
                    for idx in mol_indices:
                        docking_failures[idx] = FAILURE_DOCKING_BACKEND

                for idx, result in zip(mol_indices, results):
                    if result.success and result.score_raw != 0.0:
                        rewards[idx] = result.score_raw / self.normalization_factor
                    else:
                        docking_failures[idx] = result.failure_reason or FAILURE_DOCKING_BACKEND
        finally:
            _prepared.cleanup()

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        data.batch["rewards"] = rewards_tensor
        data.batch["docking_reward"] = rewards_tensor.clone()
        if failure_type is not None:
            for idx, failure in docking_failures.items():
                if str(failure_type[idx]) == FAILURE_SUCCESS:
                    failure_type[idx] = failure
            data.batch["failure_type"] = failure_type
        return data

    def __repr__(self) -> str:
        return (
            f"DockingReward("
            f"backend='{self.backend}', "
            f"receptor='{Path(self.receptor_file).name}', "
            f"norm={self.normalization_factor})"
        )
