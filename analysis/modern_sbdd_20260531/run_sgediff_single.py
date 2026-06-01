"""Run SGEDiff sampling on one local protein-ligand pair without editing SGEDiff."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict
from rdkit import Chem
from rdkit.Chem import BondType, HybridizationType
from torch_geometric.transforms import Compose


AA_NAME_NUMBER = {
    "ALA": 0,
    "CYS": 1,
    "ASP": 2,
    "GLU": 3,
    "PHE": 4,
    "GLY": 5,
    "HIS": 6,
    "ILE": 7,
    "LYS": 8,
    "LEU": 9,
    "MET": 10,
    "ASN": 11,
    "PRO": 12,
    "GLN": 13,
    "ARG": 14,
    "SER": 15,
    "THR": 16,
    "VAL": 17,
    "TRP": 18,
    "TYR": 19,
}

BACKBONE_ATOMS = {"N", "CA", "C", "O"}
PROTEIN_ATOMS = {1, 6, 7, 8, 16}
BOND_TYPES = {
    BondType.SINGLE: 1,
    BondType.DOUBLE: 2,
    BondType.TRIPLE: 3,
    BondType.AROMATIC: 4,
}
HYBRID_TYPES = {
    HybridizationType.UNSPECIFIED: "S",
    HybridizationType.S: "S",
    HybridizationType.SP: "SP",
    HybridizationType.SP2: "SP2",
    HybridizationType.SP3: "SP3",
    HybridizationType.SP3D: "SP3D",
    HybridizationType.SP3D2: "SP3D2",
}


def _repo_imports(repo: Path) -> None:
    sys.path.insert(0, str(repo))
    os.chdir(repo)


def _protein_arrays(protein_path: Path) -> dict[str, torch.Tensor]:
    mol = Chem.MolFromPDBFile(str(protein_path), sanitize=False, removeHs=False)
    if mol is None:
        raise ValueError(f"Could not read protein PDB: {protein_path}")
    conf = mol.GetConformer()
    keep: list[int] = []
    index_map: dict[int, int] = {}
    atom_class: list[int] = []
    atom_backbone: list[int] = []
    atom_res_name: list[int] = []
    atom_coords: list[np.ndarray] = []
    for atom in mol.GetAtoms():
        residue = atom.GetPDBResidueInfo()
        if residue is None:
            continue
        res_name = residue.GetResidueName().strip()
        atomic_num = atom.GetAtomicNum()
        if res_name not in AA_NAME_NUMBER or atomic_num not in PROTEIN_ATOMS:
            continue
        new_idx = len(keep)
        keep.append(atom.GetIdx())
        index_map[atom.GetIdx()] = new_idx
        atom_class.append(atomic_num)
        atom_backbone.append(int(residue.GetName().strip() in BACKBONE_ATOMS))
        atom_res_name.append(AA_NAME_NUMBER[res_name])
        pos = conf.GetAtomPosition(atom.GetIdx())
        atom_coords.append(np.array([pos.x, pos.y, pos.z], dtype=np.float32))

    bond_class: list[int] = []
    bond_idx: list[tuple[int, int]] = []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if begin not in index_map or end not in index_map:
            continue
        bond_type = BOND_TYPES.get(bond.GetBondType())
        if bond_type is None:
            continue
        bond_class.append(bond_type)
        bond_idx.append((index_map[begin], index_map[end]))

    if not atom_coords:
        raise ValueError(f"No usable protein atoms in {protein_path}")
    if not bond_idx:
        bond_idx = [(0, 0)]
        bond_class = [1]

    return {
        "protein_atom_class": torch.tensor(atom_class, dtype=torch.long),
        "protein_atom_is_backbone": torch.tensor(atom_backbone, dtype=torch.long),
        "protein_atom_coords": torch.tensor(np.asarray(atom_coords), dtype=torch.float32),
        "protein_atom_res_name": torch.tensor(atom_res_name, dtype=torch.long),
        "protein_bond_class": torch.tensor(bond_class, dtype=torch.long),
        "protein_bond_idx": torch.tensor(bond_idx, dtype=torch.long),
    }


def _ligand_arrays(ligand_path: Path) -> dict[str, torch.Tensor | list[str]]:
    mol = Chem.SDMolSupplier(str(ligand_path), removeHs=False)[0]
    if mol is None:
        raise ValueError(f"Could not read ligand SDF: {ligand_path}")
    mol = Chem.AddHs(mol, addCoords=True)
    conf = mol.GetConformer()
    atom_class: list[int] = []
    atom_aromatic: list[bool] = []
    atom_hybridization: list[str] = []
    atom_coords: list[np.ndarray] = []
    for atom in mol.GetAtoms():
        atom_class.append(atom.GetAtomicNum())
        atom_aromatic.append(bool(atom.GetIsAromatic()))
        atom_hybridization.append(HYBRID_TYPES[atom.GetHybridization()])
        pos = conf.GetAtomPosition(atom.GetIdx())
        atom_coords.append(np.array([pos.x, pos.y, pos.z], dtype=np.float32))

    bond_class: list[int] = []
    bond_idx: list[tuple[int, int]] = []
    for bond in mol.GetBonds():
        bond_type = BOND_TYPES.get(bond.GetBondType())
        if bond_type is None:
            continue
        bond_class.append(bond_type)
        bond_idx.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
    if not bond_idx:
        bond_idx = [(0, 0)]
        bond_class = [1]

    coords = np.asarray(atom_coords, dtype=np.float32)
    return {
        "ligand_center": torch.tensor([coords.mean(axis=0)], dtype=torch.float32),
        "ligand_atom_class": torch.tensor(atom_class, dtype=torch.long),
        "ligand_atom_aromatic": torch.tensor(atom_aromatic, dtype=torch.bool),
        "ligand_atom_coords": torch.tensor(coords, dtype=torch.float32),
        "ligand_atom_hybridization": atom_hybridization,
        "ligand_bond_class": torch.tensor(bond_class, dtype=torch.long),
        "ligand_bond_idx": torch.tensor(bond_idx, dtype=torch.long),
    }


def _build_data(repo: Path, protein_path: Path, ligand_path: Path):
    _repo_imports(repo)
    from utils.data_tools import BasedData

    fields: dict[str, object] = {
        "name": "single",
        "pn": str(protein_path),
        "ln": str(ligand_path),
        "score": 100.0,
        "interaction": [],
        "pures_idx": torch.tensor([0], dtype=torch.long),
    }
    fields.update(_protein_arrays(protein_path))
    fields.update(_ligand_arrays(ligand_path))
    return BasedData(**fields)


def _load_model(config, device: str):
    from models.SGEDiff.molopt_score_model import ScorePosNet3D
    from utils.transforms import FeaturizeLigandAtom, FeaturizeProteinAtom

    ckpt = torch.load(config.model["SGEDiff-NG"], map_location=device)
    model = ScorePosNet3D(
        ckpt["config"].model,
        protein_atom_feature_dim=FeaturizeProteinAtom().feature_dim,
        ligand_atom_feature_dim=FeaturizeLigandAtom("add_aromatic").feature_dim,
    ).to(device)
    state_dict = ckpt.get("model", ckpt.get("models"))
    if state_dict is None:
        raise KeyError("SGEDiff checkpoint has neither 'model' nor 'models' state dict")
    state = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }
    model.load_state_dict(state)
    model.eval()
    return model


def _write_sdf(result: dict[str, object], sdf_path: Path) -> tuple[int, int]:
    from utils import reconstruct, transforms

    sdf_path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(sdf_path))
    total = 0
    valid = 0
    for pred_pos, pred_v in zip(result["pred_ligand_pos"], result["pred_ligand_v"]):
        total += 1
        atom_type = transforms.get_atomic_number_from_index(
            torch.as_tensor(pred_v), mode="add_aromatic"
        )
        aromatic = transforms.is_aromatic_from_index(torch.as_tensor(pred_v), mode="add_aromatic")
        atom_affinity = [0.0 for _ in range(len(atom_type))]
        try:
            mol = reconstruct.reconstruct_from_generated(
                pred_pos, atom_type, aromatic, atom_affinity=atom_affinity
            )
            Chem.SanitizeMol(mol)
            if "." in Chem.MolToSmiles(mol):
                continue
        except Exception:
            continue
        writer.write(mol)
        valid += 1
    writer.close()
    return total, valid


def main() -> None:
    """Run SGEDiff-NG sampling and save both raw PT and reconstructed SDF."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protein", type=Path, required=True)
    parser.add_argument("--ligand", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    data = _build_data(args.repo.resolve(), args.protein.resolve(), args.ligand.resolve())

    from torch_geometric.data import Batch
    from utils import misc
    from utils.sample import sample_diffusion_ligand
    from utils.transforms import (
        FeaturizeLigandAtom,
        FeaturizeLigandBond,
        FeaturizeProteinAtom,
        FeaturizeProteinBond,
        ImportantEx,
        InteractionEx,
        NormalizeVina,
    )

    config = misc.load_config(str(args.config))
    misc.seed_all(config.sample.seed)
    transform = Compose(
        [
            FeaturizeLigandAtom("add_aromatic"),
            FeaturizeProteinAtom(),
            FeaturizeLigandBond(),
            FeaturizeProteinBond(),
            NormalizeVina(),
            InteractionEx(),
            ImportantEx(),
        ]
    )
    data = transform(data)
    # SGEDiff's sampling Batch call does not pass exclude_keys; keep only the
    # derived interaction tensors needed during sampling.
    for attr in ("interaction", "interaction_idx", "interaction_class"):
        if hasattr(data, attr):
            delattr(data, attr)
    model = _load_model(config, args.device)

    pred = sample_diffusion_ligand(
        model,
        data,
        config.sample.num_samples,
        batch_size=args.batch_size,
        device=args.device,
        num_steps=config.sample.num_steps,
        center_pos_mode=config.sample.center_pos_mode,
        sample_num_atoms=config.sample.sample_num_atoms,
        guide_mode="gic_no_score",
        value_model=None,
        sampling_model="SGEDiff-NG",
        type_grad_weight=100.0,
        pos_grad_weight=1.0,
        transform=None,
        init_mode="auto",
    )
    (
        pred_pos,
        pred_v,
        pred_exp,
        pred_pos_traj,
        pred_v_traj,
        pred_exp_traj,
        pred_v0_traj,
        pred_vt_traj,
        pred_exp_atom_traj,
        time_list,
    ) = pred
    result = {
        "data": data,
        "pred_ligand_pos": pred_pos,
        "pred_ligand_v": pred_v,
        "pred_exp": pred_exp,
        "pred_ligand_pos_traj": pred_pos_traj,
        "pred_ligand_v_traj": pred_v_traj,
        "pred_exp_traj": pred_exp_traj,
        "pred_exp_atom_traj": pred_exp_atom_traj,
        "time": time_list,
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.outdir / "sampled_molecule.pt")
    total, valid = _write_sdf(result, args.outdir / "sgediff_reconstructed.sdf")
    print(f"sampled={total} reconstructed_sdf={valid}")


if __name__ == "__main__":
    main()
