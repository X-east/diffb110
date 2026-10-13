"""Execute a PMDM script with its namespace isolated from global PYTHONPATH."""

from __future__ import annotations

import os
import runpy
import sys
import types
from pathlib import Path

import numpy as np


def main() -> None:
    """Install PMDM-local namespace packages, then run the requested script."""
    if len(sys.argv) < 3:
        raise SystemExit("usage: run_pmdm_script.py REPO SCRIPT [ARGS ...]")
    repo = Path(sys.argv[1]).resolve()
    script = (repo / sys.argv[2]).resolve()
    if not hasattr(np, "long"):
        np.long = np.int64
    if not hasattr(np, "bool"):
        np.bool = np.bool_
    if not hasattr(np, "int"):
        np.int = int
    if not hasattr(np, "float"):
        np.float = float
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    for module_name in ("utils", "models", "configs"):
        module = types.ModuleType(module_name)
        module.__path__ = [str(repo / module_name)]
        module.__package__ = module_name
        sys.modules[module_name] = module
    for module_name in tuple(sys.modules):
        if module_name == "evaluation" or module_name.startswith("evaluation."):
            del sys.modules[module_name]
    try:
        import six

        sys.modules.setdefault("rdkit.six", six)
        sys.modules.setdefault("rdkit.six.moves", six.moves)
    except ImportError:
        pass

    from utils.data import ProteinLigandData

    def _compatible_inc(self, key, value, *args, **kwargs):
        keys = self.keys if isinstance(self.keys, list) else self.keys()
        if key == "ligand_bond_index":
            if "ligand_element" in keys:
                return self["ligand_element"].size(0)
            return self["ligand_atom_feature"].size(0)
        if key == "protein_bond_index":
            if "protein_element" in keys:
                return self["protein_element"].size(0)
            return self["protein_atom_feature_full"].size(0)
        if key == "pocket_bond_index":
            if "pocket_element" in keys:
                return self["pocket_element"].size(0)
            return self["pocket_atom_type"].size(0)
        if key == "ligand_context_bond_index":
            return self["ligand_context_element"].size(0)
        return super(ProteinLigandData, self).__inc__(key, value)

    ProteinLigandData.__inc__ = _compatible_inc
    sys.argv = [str(script), *sys.argv[3:]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
