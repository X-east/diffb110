"""
AutoDock-GPU docking backend.

Uses AutoDock-GPU (https://github.com/ccsb-scripps/AutoDock-GPU) which runs
Lamarckian Genetic Algorithm docking entirely on GPU, typically 10-100x faster
than AutoDock Vina on CPU.

Prerequisites
-------------
1. AutoDock-GPU binary installed and on PATH (or set AUTODOCK_GPU_BIN env var).
   Typical binary names: autodock_gpu_64wi, autodock_gpu_128wi, autodock_gpu
   Install: conda install -c conda-forge autodock-gpu  (or download from GitHub)

2. Receptor grid maps pre-computed with AutoGrid4:
       prepare_receptor4.py -r receptor.pdb -o receptor.pdbqt
       prepare_gpf4.py -l ligand.pdbqt -r receptor.pdbqt -o receptor.gpf
       autogrid4 -p receptor.gpf -l receptor.glg
   This produces receptor.maps.fld (the --ffile argument).

3. Ligands in PDBQT format (converted from SDF via obabel or meeko).

Output format
-------------
AutoDock-GPU writes a DLG (docking log) file per ligand. The best binding
energy is on the line:
    DOCKED: USER    Estimated Free Energy of Binding    =  -7.53 kcal/mol
or for newer versions:
    DOCKED: REMARK  VINA RESULT: -7.53
We parse both patterns.
"""

import os
import re
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import List, Optional

from .base import BaseDockingBackend
from ddpo.rewards.interfaces.docking_result import DockingResult

# --------------------------------------------------------------------------
# Binary discovery
# --------------------------------------------------------------------------
# Try common binary names in order of preference (wider warp = more poses).
_CANDIDATE_BINS = [
    "autodock_gpu_64wi",
    "autodock_gpu_128wi",
    "autodock_gpu_256wi",
    "autodock_gpu",
]


def _find_autodock_gpu_bin() -> Optional[str]:
    """Locate the AutoDock-GPU binary, checking env var then PATH."""
    candidates = [
        os.environ.get("AUTODOCK_GPU_BIN"),
        *[shutil.which(name) for name in _CANDIDATE_BINS],
        # Hardcoded fallback for this installation
        "/data/disk6/DiffB110/AutoDock-GPU/AutoDock-GPU-develop/bin/autodock_gpu_64wi",
        "/home/md666/data/disk6/DiffB110/AutoDock-GPU/AutoDock-GPU-develop/bin/autodock_gpu_64wi",
    ]
    checked = []
    for c in candidates:
        if not c:
            continue
        p = str(Path(c).expanduser())
        checked.append(p)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    # Log what was checked to aid diagnosis
    import logging
    logging.getLogger("DiffSBDD-DDPO").warning(
        f"[AutoDockGPU] Binary not found. Checked: {checked}; "
        f"AUTODOCK_GPU_BIN={os.environ.get('AUTODOCK_GPU_BIN')}; "
        f"PATH={os.environ.get('PATH')}"
    )
    return None


AUTODOCK_GPU_BIN: Optional[str] = None  # resolved lazily in AutoDockGPUBackend.__init__


# --------------------------------------------------------------------------


class AutoDockGPUBackend(BaseDockingBackend):
    """
    AutoDock-GPU docking backend.

    Key differences from UniDock
    ----------------------------
    - Requires pre-computed AutoGrid4 grid maps (.maps.fld file).
    - Accepts PDBQT ligands; SDF ligands are converted via obabel.
    - Runs each ligand in a separate subprocess (no native batch mode).
    - GPU device is selected via --devnum (1-indexed).

    Parameters
    ----------
    fld_file : str
        Path to the receptor .maps.fld file produced by autogrid4.
    gpu_id : int
        CUDA device index (0-based, converted to 1-based for AutoDock-GPU).
    nrun : int
        Number of LGA runs per ligand (higher = more thorough, slower).
        Typical range: 10–50. Default 20 is a good balance.
    heuristics : bool
        Use AutoDock-GPU heuristic (adaptive nrun based on ligand size).
        Recommended True for production.
    autodock_gpu_bin : str | None
        Explicit path to the binary. If None, auto-discovered from PATH /
        AUTODOCK_GPU_BIN environment variable.
    """

    def __init__(
        self,
        fld_file: str,
        gpu_id: int = 0,
        nrun: int = 20,
        heuristics: bool = True,
        autodock_gpu_bin: Optional[str] = None,
        **kwargs,
    ):
        self.fld_file = str(Path(fld_file).resolve())
        self.gpu_id = gpu_id
        # AutoDock-GPU uses 1-based device numbering (--devnum 1 = first GPU)
        # nvidia-smi and CUDA use 0-based indexing
        # Conversion: autodock_gpu --devnum = gpu_id + 1
        self._devnum = gpu_id + 1
        self.nrun = nrun
        self.heuristics = heuristics
        self.autodock_gpu_bin = autodock_gpu_bin or _find_autodock_gpu_bin()

        if self.autodock_gpu_bin is None:
            raise RuntimeError(
                "AutoDock-GPU binary not found.\n"
                "Options:\n"
                "  1. conda install -c conda-forge autodock-gpu\n"
                "  2. Download from https://github.com/ccsb-scripps/AutoDock-GPU/releases\n"
                "  3. Set env var AUTODOCK_GPU_BIN=/path/to/autodock_gpu_64wi"
            )

        if not Path(self.autodock_gpu_bin).exists():
            raise RuntimeError(
                f"AutoDock-GPU binary not found at: {self.autodock_gpu_bin}"
            )

        if not Path(self.fld_file).exists():
            raise FileNotFoundError(
                f"AutoGrid4 map file not found: {self.fld_file}\n"
                "Generate it with:\n"
                "  prepare_receptor4.py -r receptor.pdb -o receptor.pdbqt\n"
                "  prepare_gpf4.py -l ligand.pdbqt -r receptor.pdbqt -o receptor.gpf\n"
                "  autogrid4 -p receptor.gpf -l receptor.glg"
            )

        try:
            result = subprocess.run(
                [self.autodock_gpu_bin, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            ver_str = (result.stdout + result.stderr).strip().splitlines()
            ver_line = ver_str[0] if ver_str else "unknown version"
            print(f"[AutoDockGPUBackend] Binary OK: {ver_line}")
        except subprocess.TimeoutExpired:
            warnings.warn("[AutoDockGPUBackend] --version timed out; binary may be broken")
        except Exception as e:
            raise RuntimeError(f"AutoDock-GPU binary check failed: {e}")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def backend_name(self) -> str:
        """Identifier string used in DockingResult.backend_name."""
        return "autodock_gpu"

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
        """Dock a batch of ligands and return structured DockingResult objects.

        Wraps the existing sequential-per-ligand GPU docking logic.
        A score of 0.0 from _dock_one() is treated as a backend failure.

        Args:
            ligand_files:  list of PDBQT (or SDF) file paths.
            receptor_file: ignored — receptor box comes from fld_file.
            center/size:   ignored — box comes from fld_file.
            exhaustiveness: ignored — use nrun set at init time.

        Returns:
            List[DockingResult] of length == len(ligand_files).
        """
        if not ligand_files:
            return []

        env = os.environ.copy()
        # Do NOT set CUDA_VISIBLE_DEVICES: AutoDock-GPU uses --devnum (1-indexed)
        # and setting CUDA_VISIBLE_DEVICES renumbers devices, breaking --devnum mapping.
        if "CUDA_VISIBLE_DEVICES" in env:
            del env["CUDA_VISIBLE_DEVICES"]

        results = []
        for lig_file in ligand_files:
            score = self._dock_one(lig_file, env)
            if score != 0.0:
                results.append(DockingResult.succeeded(
                    backend_name=self.backend_name,
                    score_raw=score,
                ))
            else:
                results.append(DockingResult.failed(
                    backend_name=self.backend_name,
                    failure_reason="FAILURE_DOCKING_BACKEND",
                ))

        non_zero = sum(1 for r in results if r.success)
        if results:
            scores = [r.score_raw for r in results]
            mean_s = sum(scores) / len(scores)
            print(
                f"[AutoDockGPUBackend] Completed: {non_zero}/{len(results)} scored, "
                f"mean={mean_s:.3f}"
            )
        return results

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
        exhaustiveness: int = 8,
        **kwargs,
    ) -> float:
        """Legacy float interface — delegates to dock_batch_result()."""
        return self.dock_batch_result(
            [ligand_file], receptor_file,
            center_x, center_y, center_z,
            size_x, size_y, size_z, exhaustiveness,
            **kwargs,
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
        exhaustiveness: int = 8,
        **kwargs,
    ) -> List[float]:
        """Legacy float interface — delegates to dock_batch_result()."""
        return [
            r.score_raw
            for r in self.dock_batch_result(
                ligand_files, receptor_file,
                center_x, center_y, center_z,
                size_x, size_y, size_z, exhaustiveness,
                **kwargs,
            )
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dock_one(self, ligand_file: str, env: dict) -> float:
        """
        Run AutoDock-GPU for a single PDBQT ligand.
        Returns best binding energy (kcal/mol) or 0.0 on failure.
        """
        # Convert SDF → PDBQT if needed
        pdbqt_file, converted = self._ensure_pdbqt(ligand_file, env)
        if pdbqt_file is None:
            return 0.0

        with tempfile.TemporaryDirectory() as tmpdir:
            dlg_stem = Path(tmpdir) / Path(pdbqt_file).stem
            cmd = [
                self.autodock_gpu_bin,
                "--ffile", self.fld_file,
                "--lfile", pdbqt_file,
                "--resnam", str(dlg_stem),
                "--devnum", str(self._devnum),
                "--nrun", str(self.nrun),
            ]
            if self.heuristics:
                cmd += ["--heuristics", "1"]

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    env=env,
                )
                if result.returncode != 0:
                    warnings.warn(
                        f"[AutoDockGPUBackend] non-zero exit {result.returncode} "
                        f"for {Path(ligand_file).name}: "
                        f"{result.stderr[:300]}"
                    )
                    return 0.0

                # DLG file is written as <resnam>.dlg
                dlg_file = Path(str(dlg_stem) + ".dlg")
                if not dlg_file.exists():
                    # Some versions write to cwd
                    dlg_file = Path(tmpdir) / f"{Path(pdbqt_file).stem}.dlg"

                score = self._parse_dlg(dlg_file, result.stdout + result.stderr)
                return score

            except subprocess.TimeoutExpired:
                warnings.warn(
                    f"[AutoDockGPUBackend] Timeout docking {Path(ligand_file).name}"
                )
                return 0.0
            except Exception as e:
                warnings.warn(
                    f"[AutoDockGPUBackend] Error docking {Path(ligand_file).name}: {e}"
                )
                return 0.0
            finally:
                if converted and pdbqt_file and Path(pdbqt_file).exists():
                    try:
                        os.unlink(pdbqt_file)
                    except Exception:
                        pass

    def _ensure_pdbqt(self, ligand_file: str, env: dict):
        """
        If ligand_file is already PDBQT return (path, False).
        If it is SDF, convert with obabel and return (tmp_pdbqt_path, True).
        Returns (None, False) on conversion failure.
        """
        suffix = Path(ligand_file).suffix.lower()
        if suffix == ".pdbqt":
            return ligand_file, False

        # Convert SDF (or anything obabel understands) to PDBQT
        try:
            tmp_fd, tmp_pdbqt = tempfile.mkstemp(suffix=".pdbqt")
            os.close(tmp_fd)

            result = subprocess.run(
                ["obabel", ligand_file, "-opdbqt", "-O", tmp_pdbqt, "-h", "--partialcharge", "gasteiger"],
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
            )
            if result.returncode != 0 or not os.path.exists(tmp_pdbqt) or os.path.getsize(tmp_pdbqt) == 0:
                warnings.warn(
                    f"[AutoDockGPUBackend] obabel conversion failed for {ligand_file}: "
                    f"{result.stderr[:200]}"
                )
                try:
                    os.unlink(tmp_pdbqt)
                except Exception:
                    pass
                return None, False

            return tmp_pdbqt, True

        except subprocess.TimeoutExpired:
            warnings.warn(f"[AutoDockGPUBackend] obabel timeout for {ligand_file}")
            return None, False
        except Exception as e:
            warnings.warn(f"[AutoDockGPUBackend] obabel error for {ligand_file}: {e}")
            return None, False

    @staticmethod
    def _parse_dlg(dlg_file: Path, fallback_stdout: str) -> float:
        """
        Parse the best binding energy from an AutoDock-GPU DLG file.

        AutoDock-GPU DLG lines of interest (in order of preference):
          DOCKED: USER    Estimated Free Energy of Binding    =  -7.53 kcal/mol
          DOCKED: REMARK VINA RESULT:   -7.53   0.000   0.000
          Run   1 / 20   best energy:   -7.53 kcal/mol

        We scan the full file and return the minimum (best) energy found.
        Falls back to stdout/stderr scan if the DLG file is missing.
        """
        content = ""
        if dlg_file.exists():
            try:
                content = dlg_file.read_text(errors="ignore")
            except Exception:
                pass

        if not content:
            content = fallback_stdout

        if not content:
            return 0.0

        energies = []

        # Pattern 1: standard DLG free energy line
        # "Estimated Free Energy of Binding    =  -7.53 kcal/mol"
        for m in re.finditer(
            r"Estimated Free Energy of Binding\s*=\s*([\-\+]?[0-9]*\.?[0-9]+)\s*kcal",
            content,
        ):
            try:
                energies.append(float(m.group(1)))
            except ValueError:
                pass

        # Pattern 2: VINA-style remark
        # "DOCKED: REMARK VINA RESULT:   -7.53"
        for m in re.finditer(
            r"VINA RESULT:\s+([\-\+]?[0-9]*\.?[0-9]+)",
            content,
        ):
            try:
                energies.append(float(m.group(1)))
            except ValueError:
                pass

        # Pattern 3: per-run summary line
        # "best energy:   -7.53 kcal/mol"
        for m in re.finditer(
            r"best energy:\s+([\-\+]?[0-9]*\.?[0-9]+)\s*kcal",
            content,
        ):
            try:
                energies.append(float(m.group(1)))
            except ValueError:
                pass

        # Pattern 4: final docked energy summary
        # "DOCKED: USER    (1) Final Docked Energy:          -5.00 kcal/mol"
        for m in re.finditer(
            r"Final Docked Energy:\s+([\-\+]?[0-9]*\.?[0-9]+)\s*kcal",
            content,
        ):
            try:
                energies.append(float(m.group(1)))
            except ValueError:
                pass

        if not energies:
            return 0.0

        # Lower (more negative) = better binding; return best (min)
        return min(energies)
 