"""Export DiffSMol/KGDiff result pt files to SDF for uniform evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rdkit import Chem


def _to_index_array(v: Any) -> np.ndarray:
    arr = np.asarray(v)
    if arr.ndim == 2:
        arr = arr.argmax(axis=-1)
    return arr.astype(int)


def _load_project_modules(project: str):
    if project not in {"diffsmol", "kgdiff"}:
        raise ValueError(f"unsupported project: {project}")
    from utils import transforms  # type: ignore
    from utils.reconstruct import MolReconsError, reconstruct_from_generated  # type: ignore

    return transforms, reconstruct_from_generated, MolReconsError


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", choices=["diffsmol", "kgdiff"], required=True)
    parser.add_argument("--input-pt", type=Path, required=True)
    parser.add_argument("--out-sdf", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--atom-mode", default="add_aromatic")
    args = parser.parse_args()

    transforms, reconstruct_from_generated, mol_recons_error = _load_project_modules(args.project)
    result = torch.load(args.input_pt, map_location="cpu")
    positions = result["pred_ligand_pos"]
    values = result["pred_ligand_v"]
    exp_atom = result.get("pred_exp_atom_traj")
    bonds = result.get("pred_ligand_bond")

    args.out_sdf.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(args.out_sdf))
    rows = []
    failures = []
    for idx, (pos, v) in enumerate(zip(positions, values)):
        index = _to_index_array(v)
        atomic_nums = transforms.get_atomic_number_from_index(index, args.atom_mode)
        aromatic = transforms.is_aromatic_from_index(index, args.atom_mode)
        kwargs: dict[str, Any] = {}
        if args.project == "kgdiff":
            atom_affinity = []
            if exp_atom is not None:
                try:
                    atom_affinity = list(np.asarray(exp_atom[idx][-1]).reshape(-1))
                except Exception:
                    atom_affinity = [0.0] * len(atomic_nums)
            kwargs["atom_affinity"] = atom_affinity
        else:
            if bonds is not None and idx < len(bonds):
                kwargs["pred_bond"] = np.asarray(bonds[idx])
        try:
            mol = reconstruct_from_generated(np.asarray(pos), atomic_nums, aromatic, **kwargs)
            mol.SetProp("_Name", f"{args.project}_{idx}")
            writer.write(mol)
            rows.append({"idx": idx, "heavy_atoms": mol.GetNumHeavyAtoms(), "smiles": Chem.MolToSmiles(Chem.RemoveHs(mol))})
        except Exception as exc:  # project reconstruction raises several legacy exception types
            failures.append({"idx": idx, "error": type(exc).__name__, "message": str(exc)[:200]})
    writer.close()

    payload = {
        "project": args.project,
        "input_pt": str(args.input_pt),
        "out_sdf": str(args.out_sdf),
        "samples": len(positions),
        "exported": len(rows),
        "failed": len(failures),
        "time_list": [float(x) for x in result.get("time", [])],
        "rows": rows,
        "failures": failures,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
