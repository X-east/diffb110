"""Established PoseBusters and UniDock adapters for offline labels."""

from __future__ import annotations

import multiprocessing as mp
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from rdkit import Chem
from posebusters import PoseBusters

from ddpo.diffgui.runtime.posebusters_filter import (
    _bust_one_molecule,
    evaluate_posebusters_molecule,
)
from ddpo.rewards.docking_backends.unidock import UNIDOCK_BIN, UniDockBackend
from ddpo.rewards.interfaces import DockingResult
from ddpo.utils.molecule_semantics import (
    FAILURE_DOCKING_BACKEND,
    FAILURE_DOCKING_INPUT,
    prepare_mol_for_docking,
    write_rdmol_to_sdf,
)

from .labeling import DockingLabelFacts, PoseBustersLabelFacts


@dataclass(frozen=True)
class PoseBustersBackendConfig:
    """Configuration for the existing per-molecule PoseBusters gate."""

    receptor_pdb: Path
    config_name: str = "dock_fast"
    molecule_timeout_seconds: float = 120.0
    execution_mode: str = "isolated_process"

    def __post_init__(self) -> None:
        """Validate required PoseBusters inputs."""
        if not self.receptor_pdb.is_file():
            raise FileNotFoundError(
                f"PoseBusters receptor not found: {self.receptor_pdb}"
            )
        if not self.config_name.strip():
            raise ValueError("PoseBusters config_name cannot be empty")
        if self.molecule_timeout_seconds <= 0:
            raise ValueError("PoseBusters molecule_timeout_seconds must be positive")
        if self.execution_mode not in {"isolated_process", "persistent_worker"}:
            raise ValueError(
                "PoseBusters execution_mode must be isolated_process or "
                "persistent_worker"
            )


_POOL_EVALUATOR: Any | None = None
_POOL_RECEPTOR: Path | None = None


def _initialize_posebusters_pool(config_name: str, receptor_pdb: str) -> None:
    """Initialize one reusable evaluator inside a dedicated child process."""
    global _POOL_EVALUATOR, _POOL_RECEPTOR
    _POOL_EVALUATOR = PoseBusters(config=config_name)
    _POOL_RECEPTOR = Path(receptor_pdb)


def _evaluate_posebusters_pool(mol_binary: bytes) -> dict[str, Any]:
    """Evaluate one molecule with the process-local reusable evaluator."""
    if _POOL_EVALUATOR is None or _POOL_RECEPTOR is None:
        raise RuntimeError("PoseBusters persistent worker is not initialized")
    return _bust_one_molecule(
        _POOL_EVALUATOR,
        _POOL_RECEPTOR,
        Chem.Mol(mol_binary),
    )


class PoseBustersLabeler:
    """Convert the established PoseBusters audit row into label facts."""

    def __init__(self, config: PoseBustersBackendConfig) -> None:
        """Store one validated backend configuration."""
        self.config = config
        self._pool: Any | None = None
        if config.execution_mode == "persistent_worker":
            self._start_pool()

    def _start_pool(self) -> None:
        """Start one spawn-based reusable evaluator process."""
        context = mp.get_context("spawn")
        self._pool = context.Pool(
            processes=1,
            initializer=_initialize_posebusters_pool,
            initargs=(
                self.config.config_name,
                str(self.config.receptor_pdb),
            ),
        )

    def _restart_pool(self) -> None:
        """Replace a timed-out or failed reusable evaluator process."""
        self.close()
        self._start_pool()

    def close(self) -> None:
        """Terminate and join the reusable evaluator process, if any."""
        if self._pool is None:
            return
        self._pool.terminate()
        self._pool.join()
        self._pool = None

    def __call__(self, molecule: Chem.Mol) -> PoseBustersLabelFacts:
        """Evaluate one sanitized molecule with timeout isolation."""
        if self.config.execution_mode == "persistent_worker":
            if self._pool is None:
                self._start_pool()
            try:
                pending = self._pool.apply_async(
                    _evaluate_posebusters_pool,
                    (molecule.ToBinary(),),
                )
                row = dict(
                    pending.get(timeout=self.config.molecule_timeout_seconds)
                )
            except mp.TimeoutError:
                self._restart_pool()
                row = {
                    "pb_evaluated": 1,
                    "pb_valid": 0,
                    "posebusters_error": (
                        "molecule_timeout:"
                        f"{self.config.molecule_timeout_seconds:.3f}s"
                    ),
                }
            except Exception as exc:
                self._restart_pool()
                row = {
                    "pb_evaluated": 1,
                    "pb_valid": 0,
                    "posebusters_error": f"{type(exc).__name__}:{exc}",
                }
        else:
            row = evaluate_posebusters_molecule(
                config_name=self.config.config_name,
                receptor_pdb=self.config.receptor_pdb,
                mol=molecule,
                timeout_seconds=self.config.molecule_timeout_seconds,
            )
        error = str(row.get("posebusters_error", "")).strip()
        valid = bool(int(row.get("pb_valid", 0)))
        evaluated = bool(int(row.get("pb_evaluated", 0)))
        if error:
            failure_reason = f"posebusters_error:{error}"
        elif not valid:
            failed_checks = str(row.get("posebusters_failed_checks", "")).strip()
            missing_checks = str(row.get("posebusters_missing_checks", "")).strip()
            details = failed_checks or missing_checks or "no_complete_pass"
            failure_reason = f"posebusters_invalid:{details}"
        else:
            failure_reason = None
        return PoseBustersLabelFacts(
            evaluated=evaluated,
            valid=valid,
            failure_reason=failure_reason,
        )


@dataclass(frozen=True)
class UniDockLabelingConfig:
    """Target and execution controls for canonical-molecule UniDock labels."""

    receptor_pdbqt: Path
    gpu_id: int
    center_x: float
    center_y: float
    center_z: float
    size_x: float = 20.0
    size_y: float = 20.0
    size_z: float = 20.0
    search_mode: str = "fast"
    num_modes: int = 1
    exhaustiveness: int = 8
    unidock_bin: Path = Path(UNIDOCK_BIN)
    add_hydrogens: bool = True
    assign_gasteiger: bool = True

    def __post_init__(self) -> None:
        """Validate target files and numeric controls."""
        if not self.receptor_pdbqt.is_file():
            raise FileNotFoundError(
                f"UniDock receptor not found: {self.receptor_pdbqt}"
            )
        if not self.unidock_bin.is_file():
            raise FileNotFoundError(f"UniDock binary not found: {self.unidock_bin}")
        if self.gpu_id < 0:
            raise ValueError("UniDock gpu_id must be non-negative")
        if min(self.size_x, self.size_y, self.size_z) <= 0:
            raise ValueError("UniDock box sizes must be positive")
        if self.num_modes <= 0 or self.exhaustiveness <= 0:
            raise ValueError("UniDock num_modes and exhaustiveness must be positive")


class UniDockLabeler:
    """Prepare canonical molecules and return per-sample UniDock label facts."""

    def __init__(self, config: UniDockLabelingConfig) -> None:
        """Build the existing UniDock backend once per worker."""
        self.config = config
        self._backend = UniDockBackend(
            gpu_id=config.gpu_id,
            search_mode=config.search_mode,
            num_modes=config.num_modes,
            exhaustiveness=config.exhaustiveness,
            unidock_bin=str(config.unidock_bin),
        )

    def __call__(
        self,
        molecules: Sequence[tuple[str, Chem.Mol]],
    ) -> Mapping[str, DockingLabelFacts]:
        """Prepare and dock one shard while preserving input failures."""
        results: dict[str, DockingLabelFacts] = {}
        submitted_ids: list[str] = []
        ligand_files: list[str] = []
        with tempfile.TemporaryDirectory(prefix="offline_iql_unidock_") as tmp_dir:
            output_dir = Path(tmp_dir) / "poses"
            for molecule_index, (sample_id, molecule) in enumerate(molecules):
                prepared = prepare_mol_for_docking(
                    molecule,
                    add_hydrogens=self.config.add_hydrogens,
                    assign_gasteiger=self.config.assign_gasteiger,
                )
                ligand_path = Path(tmp_dir) / f"ligand_{molecule_index:04d}.sdf"
                if prepared is None or not write_rdmol_to_sdf(
                    prepared,
                    str(ligand_path),
                ):
                    results[sample_id] = DockingLabelFacts(
                        input_success=False,
                        result=DockingResult.failed(
                            "unidock",
                            FAILURE_DOCKING_INPUT,
                        ),
                    )
                    continue
                submitted_ids.append(sample_id)
                ligand_files.append(str(ligand_path))

            if ligand_files:
                try:
                    backend_results = self._backend.dock_batch_result(
                        ligand_files,
                        str(self.config.receptor_pdbqt),
                        center_x=self.config.center_x,
                        center_y=self.config.center_y,
                        center_z=self.config.center_z,
                        size_x=self.config.size_x,
                        size_y=self.config.size_y,
                        size_z=self.config.size_z,
                        exhaustiveness=self.config.exhaustiveness,
                        output_dir=str(output_dir),
                    )
                except Exception as exc:
                    backend_results = [
                        DockingResult.failed(
                            "unidock",
                            f"{FAILURE_DOCKING_BACKEND}:{type(exc).__name__}",
                        )
                        for _ in ligand_files
                    ]
                if len(backend_results) != len(submitted_ids):
                    raise RuntimeError(
                        "UniDock result count does not match submitted ligand count"
                    )
                for sample_id, ligand_file, result in zip(
                    submitted_ids,
                    ligand_files,
                    backend_results,
                ):
                    ligand_path = Path(ligand_file)
                    pose_molecule = (
                        _load_unidock_pose(output_dir, ligand_path)
                        if result.success
                        else None
                    )
                    results[sample_id] = DockingLabelFacts(
                        input_success=True,
                        result=result,
                        pose_molecule=pose_molecule,
                    )
        return results


def _load_unidock_pose(output_dir: Path, ligand_path: Path) -> Chem.Mol | None:
    """Load the first UniDock output pose corresponding to one input ligand."""
    stem = ligand_path.stem
    candidates = (
        output_dir / f"{stem}_out.sdf",
        output_dir / f"{stem}.sdf",
        output_dir / f"{stem}_out.pdbqt",
    )
    output_path = next((path for path in candidates if path.is_file()), None)
    if output_path is None or output_path.suffix.lower() != ".sdf":
        return None
    supplier = Chem.SDMolSupplier(
        str(output_path),
        sanitize=False,
        removeHs=False,
    )
    return next((Chem.Mol(molecule) for molecule in supplier if molecule is not None), None)
