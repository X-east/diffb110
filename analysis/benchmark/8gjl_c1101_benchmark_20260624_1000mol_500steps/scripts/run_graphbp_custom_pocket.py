"""Run GraphBP generation on one custom protein/ligand pocket."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import types
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule


def _install_pyg_swish_shim() -> None:
    """Provide the PyG 1.x swish import used by the archived GraphBP code."""

    acts = types.ModuleType("torch_geometric.nn.acts")
    acts.swish = lambda x: x * torch.sigmoid(x)
    sys.modules["torch_geometric.nn.acts"] = acts


def _make_graphbp_types_file(path: Path, protein: Path, ligand: Path) -> Path:
    """Create a one-row CrossDocked-style types file for GraphBP."""

    types_path = path / "selected_8gjl_c1101.types"
    ligand_field = f"{ligand.resolve()}.gz"
    line = f"0 0 0 {protein.resolve()} {ligand_field} 0\n"
    types_path.write_text(line, encoding="utf-8")
    return types_path


def _write_sdf_from_mol_dicts(all_mols_dict: dict[object, object], output_sdf: Path, uff: bool) -> dict[str, int]:
    """Postprocess GraphBP atom coordinates into one SDF file."""

    from utils import BondAdder  # type: ignore

    output_sdf.parent.mkdir(parents=True, exist_ok=True)
    sdf_dir = output_sdf.parent / "SDF"
    sdf_dir.mkdir(parents=True, exist_ok=True)

    bond_adder = BondAdder()
    writer = Chem.SDWriter(str(output_sdf))
    generated = 0
    reconstructed = 0
    valid_smiles = 0
    failed = 0

    for target_index, mol_dicts in all_mols_dict.items():
        if not isinstance(mol_dicts, dict):
            continue
        for num_atom, mols_by_atom_count in mol_dicts.items():
            if not isinstance(num_atom, int):
                continue
            atomic_numbers_block = mols_by_atom_count["_atomic_numbers"]
            positions_block = mols_by_atom_count["_positions"]
            for local_index, (atomic_numbers, positions) in enumerate(zip(atomic_numbers_block, positions_block)):
                generated += 1
                try:
                    rd_mol, _ = bond_adder.make_mol(atomic_numbers, positions)
                    if uff:
                        try:
                            UFFOptimizeMolecule(rd_mol)
                        except Exception:
                            pass
                    try:
                        rd_mol = Chem.RemoveHs(rd_mol)
                    except Exception:
                        pass
                    smiles = Chem.MolToSmiles(rd_mol, isomericSmiles=True)
                    valid_smiles += int(Chem.MolFromSmiles(smiles) is not None)
                    rd_mol.SetProp("benchmark_model", "GraphBP")
                    rd_mol.SetProp("graphbp_target_index", str(target_index))
                    rd_mol.SetProp("graphbp_num_atoms", str(num_atom))
                    rd_mol.SetProp("graphbp_local_index", str(local_index))
                    rd_mol.SetProp("smiles", smiles)
                    writer.write(rd_mol)
                    Chem.MolToMolFile(rd_mol, str(sdf_dir / f"{reconstructed:05d}.sdf"))
                    reconstructed += 1
                except Exception:
                    failed += 1
    writer.close()
    return {
        "generated_records": generated,
        "reconstructed_records": reconstructed,
        "valid_smiles_records": valid_smiles,
        "failed_records": failed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphbp-root", type=Path, required=True)
    parser.add_argument("--protein", type=Path, required=True)
    parser.add_argument("--ligand", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--binding-site-range", type=float, default=15.0)
    parser.add_argument("--min-atoms", type=int, default=10)
    parser.add_argument("--max-atoms", type=int, default=45)
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--node-temp", type=float, default=0.5)
    parser.add_argument("--dist-temp", type=float, default=0.3)
    parser.add_argument("--angle-temp", type=float, default=0.4)
    parser.add_argument("--torsion-temp", type=float, default=1.0)
    parser.add_argument("--focus-th", type=float, default=0.5)
    parser.add_argument("--contact-th", type=float, default=0.5)
    parser.add_argument("--no-uff", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    graphbp_pkg = args.graphbp_root / "GraphBP"
    _install_pyg_swish_shim()
    sys.path.insert(0, str(graphbp_pkg.resolve()))

    from config import conf  # type: ignore
    from runner import Runner  # type: ignore

    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    conf["model"]["use_gpu"] = use_cuda
    conf["chunk_size"] = args.chunk_size

    types_file = _make_graphbp_types_file(args.output_dir, args.protein, args.ligand)
    runner = Runner(conf)
    state = torch.load(args.checkpoint, map_location="cuda" if use_cuda else "cpu")
    runner.model.load_state_dict(state)
    runner.model.eval()

    all_mols_dict = runner.generate(
        args.num_samples,
        temperature=[args.node_temp, args.dist_temp, args.angle_temp, args.torsion_temp],
        max_atoms=args.max_atoms,
        min_atoms=args.min_atoms,
        focus_th=args.focus_th,
        contact_th=args.contact_th,
        add_final=True,
        known_binding_site=True,
        binding_site_range=args.binding_site_range,
        data_root=".",
        data_file=str(types_file),
    )

    raw_path = args.output_dir / "graphbp_mols.mol_dict"
    with raw_path.open("wb") as handle:
        pickle.dump(all_mols_dict, handle)

    output_sdf = args.output_dir / "graphbp_generated.sdf"
    summary = _write_sdf_from_mol_dicts(all_mols_dict, output_sdf, uff=not args.no_uff)
    summary.update(
        {
            "model": "GraphBP",
            "protein": str(args.protein.resolve()),
            "ligand": str(args.ligand.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "raw_mol_dict": str(raw_path),
            "output_sdf": str(output_sdf),
            "num_requested": args.num_samples,
            "seed": args.seed,
            "device": "cuda" if use_cuda else "cpu",
            "binding_site_range": args.binding_site_range,
        }
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
