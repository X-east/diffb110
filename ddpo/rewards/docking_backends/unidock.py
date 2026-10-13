"""
UniDock GPU-accelerated docking backend.

Uses Uni-Dock v1.1.3 --gpu_batch mode which accepts SDF directly,
docking all ligands in a single GPU-parallelised run.
"""

import logging
import os
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

from ddpo.rewards.interfaces.docking_backend import BaseDockingBackend
from ddpo.rewards.interfaces.docking_result import DockingResult
from ddpo.utils.docking_protocol import (
    FRAME_CENTERED_ZERO,
    describe_protocol,
)
from ddpo.utils.unidock_score import parse_unidock_output

# Full path to unidock binary (conda dif env)
UNIDOCK_BIN = "/opt/unidock-1.1.3/bin/unidock"
LOGGER = logging.getLogger("DiffSBDD-DDPO")


class UniDockBackend(BaseDockingBackend):
    """
    UniDock GPU-accelerated docking backend.

    Key design:
    - Uses --gpu_batch with SDF input (no PDBQT conversion needed)
    - Docks the entire batch in ONE unidock call → maximum GPU utilisation
    - Parses per-molecule scores from the output SDF files in --dir
    """

    _binary_verified = False  # class-level flag: only verify once per process

    def __init__(
        self,
        gpu_id: int = 0,
        search_mode: str = "fast",
        num_modes: int = 1,
        exhaustiveness: int = 8,
        unidock_bin: str = UNIDOCK_BIN,
        scoring: str = "vina",
        seed: int = 42,
        verbosity: int = 1,
        cpu: int = 0,
        max_gpu_memory: int = 0,
        timeout_seconds: int = 3600,
        ligand_frame: str = FRAME_CENTERED_ZERO,
        ligand_charges: str = "none",
        **kwargs
    ):
        """
        Args:
            gpu_id: CUDA device index
            search_mode: "fast" | "balance" | "detail"
            num_modes: number of output poses per ligand
            exhaustiveness: search exhaustiveness (overridden by search_mode if set)
            unidock_bin: path to unidock executable
            scoring: scoring function; must be passed explicitly because Uni-Dock
                silently defaults to ``vina`` and would change if the version does
            seed: RNG seed; Uni-Dock defaults to 0, and the benchmark standard is 42
            verbosity, cpu, max_gpu_memory: passed explicitly so the reward search is
                identical to the evaluation search (docs/评估口径规范.md §3.1)
            timeout_seconds: subprocess timeout (was hard-coded to 600 s)
            ligand_frame: coordinate frame of the ligands handed to the backend;
                the DDPO rollout keeps coordinates zero-centred while the
                generation-evaluation line translates back to the pocket centre
        """
        self.gpu_id = gpu_id
        self.search_mode = search_mode
        self.num_modes = num_modes
        self.exhaustiveness = exhaustiveness
        self.unidock_bin = unidock_bin
        self.scoring = scoring
        self.seed = seed
        self.verbosity = verbosity
        self.cpu = cpu
        self.max_gpu_memory = max_gpu_memory
        self.timeout_seconds = timeout_seconds
        self.ligand_frame = ligand_frame
        self.ligand_charges = ligand_charges

        # Verify binary exists and works (once per process only)
        if not Path(self.unidock_bin).exists():
            raise RuntimeError(
                f"UniDock binary not found at: {self.unidock_bin}\n"
                f"Install via: conda install -c conda-forge unidock"
            )
        if not UniDockBackend._binary_verified:
            try:
                result = subprocess.run(
                    [self.unidock_bin, '--version'],
                    capture_output=True, text=True, timeout=10
                )
                version = result.stdout.strip() or result.stderr.strip()
                LOGGER.debug("[UniDockBackend] Binary OK: %s", version)
                UniDockBackend._binary_verified = True
            except Exception as e:
                raise RuntimeError(f"UniDock binary check failed: {e}")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def backend_name(self) -> str:
        return "unidock"

    def dock_batch_result(
        self,
        ligand_files: List[str],
        receptor_file: str,
        center_x: Optional[float] = None,
        center_y: Optional[float] = None,
        center_z: Optional[float] = None,
        size_x: float = 20.0,
        size_y: float = 20.0,
        size_z: float = 20.0,
        exhaustiveness: int = 8,
        **kwargs,
    ) -> List[DockingResult]:
        """Dock a batch of ligands and return structured DockingResult objects."""
        if not ligand_files:
            return []

        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)

        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                self.unidock_bin,
                '--receptor', receptor_file,
                '--gpu_batch', *ligand_files,
                '--scoring', self.scoring,
                '--num_modes', str(self.num_modes),
                '--seed', str(self.seed),
                '--verbosity', str(self.verbosity),
                '--cpu', str(self.cpu),
                '--max_gpu_memory', str(self.max_gpu_memory),
                '--dir', tmpdir,
            ]
            if self.search_mode:
                cmd += ['--search_mode', self.search_mode]
            else:
                cmd += ['--exhaustiveness', str(self.exhaustiveness)]
            if center_x is not None:
                box_payload: dict[str, object] = {
                    'center_x': float(center_x),
                    'center_y': float(center_y if center_y is not None else 0.0),
                    'center_z': float(center_z if center_z is not None else 0.0),
                    'size_x': float(size_x),
                    'size_y': float(size_y),
                    'size_z': float(size_z),
                }
                cmd += [
                    '--center_x', str(center_x),
                    '--center_y', str(center_y),
                    '--center_z', str(center_z),
                    '--size_x', str(size_x),
                    '--size_y', str(size_y),
                    '--size_z', str(size_z),
                ]
            else:
                box_payload = {'box_source': 'unset_let_unidock_decide'}

            protocol = describe_protocol(
                receptor_pdbqt=receptor_file,
                box=box_payload,
                scoring=self.scoring,
                search_mode=self.search_mode,
                num_modes=self.num_modes,
                seed=self.seed,
                ligand_prep={
                    'fragment_policy': 'largest_fragment',
                    'add_hydrogens': True,
                    'add_coords': True,
                    'charges': self.ligand_charges,
                },
                frame=self.ligand_frame,
                unidock_bin=self.unidock_bin,
            )
            LOGGER.info(
                "[UniDockBackend] docking_protocol_id=%s scoring=%s search_mode=%s seed=%s",
                protocol['docking_protocol_id'], self.scoring,
                self.search_mode, self.seed,
            )

            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True,
                    timeout=self.timeout_seconds, env=env,
                )
                if result.returncode != 0:
                    warnings.warn(
                        f"[UniDockBackend] unidock exited {result.returncode}:\n"
                        f"{result.stderr[:500]}"
                    )
                    return [
                        DockingResult.failed(self.backend_name, "FAILURE_DOCKING_BACKEND")
                        for _ in ligand_files
                    ]

                scores = self._parse_output_dir(tmpdir, ligand_files)
                results = []
                for score in scores:
                    if score is not None:
                        results.append(DockingResult.succeeded(self.backend_name, score))
                    else:
                        results.append(DockingResult.failed(self.backend_name, "FAILURE_DOCKING_BACKEND"))

                non_zero = sum(1 for r in results if r.success)
                mean_s = sum(r.score_raw for r in results) / len(results)
                LOGGER.debug(
                    "[UniDockBackend] Completed: %s/%s scored, mean=%.3f",
                    non_zero,
                    len(results),
                    mean_s,
                )
                return results

            except subprocess.TimeoutExpired:
                warnings.warn("[UniDockBackend] unidock timed out")
                return [DockingResult.failed(self.backend_name, "FAILURE_DOCKING_BACKEND") for _ in ligand_files]
            except Exception as e:
                warnings.warn(f"[UniDockBackend] dock_batch_result failed: {e}")
                return [DockingResult.failed(self.backend_name, "FAILURE_DOCKING_BACKEND") for _ in ligand_files]

    def dock_single(
        self,
        ligand_file: str,
        receptor_file: str,
        center_x: Optional[float] = None,
        center_y: Optional[float] = None,
        center_z: Optional[float] = None,
        size_x: float = 20.0,
        size_y: float = 20.0,
        size_z: float = 20.0,
        **kwargs
    ) -> float:
        """Legacy float interface — delegates to dock_batch_result()."""
        return self.dock_batch_result(
            [ligand_file], receptor_file,
            center_x, center_y, center_z,
            size_x, size_y, size_z, **kwargs,
        )[0].score_raw

    def dock_batch(
        self,
        ligand_files: List[str],
        receptor_file: str,
        center_x: Optional[float] = None,
        center_y: Optional[float] = None,
        center_z: Optional[float] = None,
        size_x: float = 20.0,
        size_y: float = 20.0,
        size_z: float = 20.0,
        **kwargs
    ) -> List[float]:
        """Legacy float interface — delegates to dock_batch_result()."""
        return [
            r.score_raw for r in self.dock_batch_result(
                ligand_files, receptor_file,
                center_x, center_y, center_z,
                size_x, size_y, size_z, **kwargs,
            )
        ]

    # ------------------------------------------------------------------
    # Score parsing
    # ------------------------------------------------------------------

    def _parse_output_dir(
        self,
        output_dir: str,
        ligand_files: List[str],
    ) -> List[Optional[float]]:
        """Parse best docking scores from UniDock output files.

        Delegates to the canonical parser (``ddpo.utils.unidock_score``); see
        ``docs/评估口径规范.md`` §3.3 / §3.3.1.  Returns ``None`` when a ligand
        has no usable affinity (parse error, or a suspect backend output), which
        the caller turns into ``FAILURE_DOCKING_BACKEND``.
        """
        scores: List[Optional[float]] = []
        out_path = Path(output_dir)

        for lig_file in ligand_files:
            stem = Path(lig_file).stem
            score: Optional[float] = None

            # Try SDF output first, then PDBQT
            for suffix in ('_out.sdf', '_out.pdbqt', '.sdf', '.pdbqt'):
                candidate = out_path / f"{stem}{suffix}"
                if candidate.exists():
                    score = self._extract_score(candidate)
                    if score is not None:
                        break

            # Also try scanning all files in output dir matching stem
            if score is None:
                for candidate_file in sorted(out_path.glob(f"{stem}*")):
                    candidate_score = self._extract_score(candidate_file)
                    if candidate_score is not None:
                        score = candidate_score
                        break

            scores.append(score)

        return scores

    @staticmethod
    def _extract_score(filepath: Path) -> Optional[float]:
        """Canonical Uni-Dock score accessor (see ``ddpo.utils.unidock_score``).

        Returns the affinity only when it is physically interpretable.  A
        ``suspect_backend`` output — Uni-Dock failures are written as a
        physically impossible, huge positive energy while rc stays 0 — and an
        unparseable file both yield ``None``.  This replaces the previous
        ``0.0`` sentinel, which conflated a genuine 0.0 score with failure.
        """
        return parse_unidock_output(filepath).as_legacy_float()
