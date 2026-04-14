import os
import re
from typing import List, Optional

from .base import BaseDockingBackend
from ddpo.rewards.interfaces.docking_result import DockingResult


class SminaBackend(BaseDockingBackend):
    """
    Smina CPU-based docking backend.
    
    Legacy backend for compatibility. Slower than UniDock but doesn't require GPU.
    """
    
    def __init__(self, **kwargs):
        """Initialize Smina backend."""
        if os.system('which smina.static > /dev/null 2>&1') != 0:
            raise RuntimeError(
                "smina.static not found. Please install smina."
            )

    @property
    def backend_name(self) -> str:
        return "smina"

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
        results = []
        for ligand_file in ligand_files:
            score = self._dock_one(ligand_file, receptor_file,
                                   center_x, center_y, center_z,
                                   size_x, size_y, size_z, exhaustiveness, **kwargs)
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
        autobox_ligand: Optional[str] = None,
        autobox_add: float = 4.0,
        **kwargs
    ) -> float:
        """Legacy float interface."""
        return self._dock_one(ligand_file, receptor_file,
                              center_x, center_y, center_z,
                              size_x, size_y, size_z, exhaustiveness,
                              autobox_ligand=autobox_ligand, autobox_add=autobox_add)

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
        **kwargs
    ) -> List[float]:
        """Legacy float interface."""
        return [r.score_raw for r in self.dock_batch_result(
            ligand_files, receptor_file,
            center_x, center_y, center_z,
            size_x, size_y, size_z, exhaustiveness, **kwargs
        )]

    def _dock_one(
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
        autobox_ligand: Optional[str] = None,
        autobox_add: float = 4.0,
        **kwargs,
    ) -> float:
        """Run smina for one ligand. Returns score (kcal/mol) or 0.0 on failure."""
        cmd = f'smina.static -l {ligand_file} -r {receptor_file}'
        if autobox_ligand is not None:
            cmd += f' --autobox_ligand {autobox_ligand}'
            cmd += f' --autobox_add {autobox_add}'
        elif center_x is not None and center_y is not None and center_z is not None:
            cmd += f' --center_x {center_x} --center_y {center_y} --center_z {center_z}'
            cmd += f' --size_x {size_x} --size_y {size_y} --size_z {size_z}'
            cmd += f' --exhaustiveness {exhaustiveness}'
        else:
            cmd += ' --score_only'

        output = os.popen(cmd).read()
        matches = re.findall(
            r"Affinity:[ ]+([+-]?[0-9]*[.]?[0-9]+)[ ]+\(kcal/mol\)",
            output
        )
        return float(matches[0]) if matches else 0.0
