"""
UniDock GPU-accelerated docking backend.

Uses Uni-Dock v1.1.3 --gpu_batch mode which accepts SDF directly,
docking all ligands in a single GPU-parallelised run.
"""

import os
import re
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

from .base import BaseDockingBackend
from ddpo.rewards.interfaces.docking_result import DockingResult

# Full path to unidock binary (conda dif env)
UNIDOCK_BIN = "/opt/unidock-1.1.3/bin/unidock"


class UniDockBackend(BaseDockingBackend):
    """
    UniDock GPU-accelerated docking backend.

    Key design:
    - Uses --gpu_batch with SDF input (no PDBQT conversion needed)
    - Docks the entire batch in ONE unidock call → maximum GPU utilisation
    - Parses per-molecule scores from the output SDF files in --dir
    """

    def __init__(
        self,
        gpu_id: int = 0,
        search_mode: str = "fast",
        num_modes: int = 1,
        exhaustiveness: int = 8,
        unidock_bin: str = UNIDOCK_BIN,
        **kwargs
    ):
        """
        Args:
            gpu_id: CUDA device index
            search_mode: "fast" | "balance" | "detail"
            num_modes: number of output poses per ligand
            exhaustiveness: search exhaustiveness (overridden by search_mode if set)
            unidock_bin: path to unidock executable
        """
        self.gpu_id = gpu_id
        self.search_mode = search_mode
        self.num_modes = num_modes
        self.exhaustiveness = exhaustiveness
        self.unidock_bin = unidock_bin

        # Verify binary exists and works
        if not Path(self.unidock_bin).exists():
            raise RuntimeError(
                f"UniDock binary not found at: {self.unidock_bin}\n"
                f"Install via: conda install -c conda-forge unidock"
            )
        try:
            result = subprocess.run(
                [self.unidock_bin, '--version'],
                capture_output=True, text=True, timeout=10
            )
            print(f"[UniDockBackend] Binary OK: {result.stdout.strip() or result.stderr.strip()}")
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
                '--num_modes', str(self.num_modes),
                '--dir', tmpdir,
            ]
            if self.search_mode:
                cmd += ['--search_mode', self.search_mode]
            else:
                cmd += ['--exhaustiveness', str(self.exhaustiveness)]
            if center_x is not None:
                cmd += [
                    '--center_x', str(center_x),
                    '--center_y', str(center_y),
                    '--center_z', str(center_z),
                    '--size_x', str(size_x),
                    '--size_y', str(size_y),
                    '--size_z', str(size_z),
                ]

            try:
                print(f"[UniDockBackend] Docking {len(ligand_files)} ligands on GPU {self.gpu_id}")
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=600, env=env,
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
                    if score != 0.0:
                        results.append(DockingResult.succeeded(self.backend_name, score))
                    else:
                        results.append(DockingResult.failed(self.backend_name, "FAILURE_DOCKING_BACKEND"))

                non_zero = sum(1 for r in results if r.success)
                mean_s = sum(r.score_raw for r in results) / len(results)
                print(
                    f"[UniDockBackend] Completed: {non_zero}/{len(results)} scored, "
                    f"mean={mean_s:.3f}"
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
        ligand_files: List[str]
    ) -> List[float]:
        """
        Parse best docking scores from UniDock output SDF/PDBQT files.

        UniDock writes <stem>_out.sdf (or _out.pdbqt) into --dir.
        Each file contains REMARK lines or SDF properties with the score.
        """
        scores = []
        out_path = Path(output_dir)

        print(f"[UniDockBackend] Output dir contents: {list(out_path.iterdir()) if out_path.exists() else 'EMPTY'}")

        for lig_file in ligand_files:
            stem = Path(lig_file).stem
            score = 0.0

            # Try SDF output first, then PDBQT
            for suffix in ('_out.sdf', '_out.pdbqt', '.sdf', '.pdbqt'):
                candidate = out_path / f"{stem}{suffix}"
                if candidate.exists():
                    score = self._extract_score(candidate)
                    if score != 0.0:
                        break

            # Also try scanning all files in output dir matching stem
            if score == 0.0:
                for f in out_path.glob(f"{stem}*"):
                    s = self._extract_score(f)
                    if s != 0.0:
                        score = s
                        break

            scores.append(score)

        return scores

    @staticmethod
    def _extract_score(filepath: Path) -> float:
        """Extract best docking score from a UniDock output file."""
        try:
            content = filepath.read_text(errors='ignore')

            # SDF property: > <docking_score>\n-7.5
            m = re.search(r'>\s*<[Dd]ocking[_\s][Ss]core>\s*\n([\-\+]?[0-9]*\.?[0-9]+)', content)
            if m:
                return float(m.group(1))

            # VINA RESULT in PDBQT: REMARK VINA RESULT: -7.5
            m = re.search(r'REMARK VINA RESULT:\s+([\-\+]?[0-9]*\.?[0-9]+)', content)
            if m:
                return float(m.group(1))

            # SDF score_only style: > <score>\n
            m = re.search(r'>\s*<[Ss]core>\s*\n([\-\+]?[0-9]*\.?[0-9]+)', content)
            if m:
                return float(m.group(1))

            # Uni-Dock v1.1 style: REMARK INTER + INTRA energy
            m = re.search(r'REMARK minimizedAffinity\s+([\-\+]?[0-9]*\.?[0-9]+)', content)
            if m:
                return float(m.group(1))

        except Exception as e:
            warnings.warn(f"[UniDockBackend] Score parse error for {filepath}: {e}")
        return 0.0
