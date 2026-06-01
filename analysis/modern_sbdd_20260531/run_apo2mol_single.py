from __future__ import annotations

import argparse
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Batch
from torch_scatter import scatter_mean, scatter_sum


def _inject_local_package(repo: Path, name: str) -> None:
    module = types.ModuleType(name)
    module.__path__ = [str(repo / name)]
    sys.modules[name] = module


def _load_config(path: Path):
    from omegaconf import OmegaConf

    return OmegaConf.load(str(path))


def _build_data(repo: Path, pdb_path: Path, sdf_path: Path, train_config):
    from datasets.pl_data import ApoHoloLigandData, torchify_dict
    from utils.data import PDBProtein, compute_residue_transforms, parse_sdf_file
    import utils.transforms as trans

    protein_dict = PDBProtein(str(pdb_path)).to_dict_atom()
    ligand_dict = parse_sdf_file(str(sdf_path))
    data = ApoHoloLigandData.from_apo_holo_ligand_dicts(
        apo_dict=torchify_dict(protein_dict),
        holo_dict=torchify_dict(protein_dict),
        ligand_dict=torchify_dict(ligand_dict),
    )
    data.protein_rotations, _, data.protein_translations, data.protein_chi_apo, data.protein_chi_holo, data.protein_chi_mask = (
        compute_residue_transforms(
            protein_pos_apo=data.protein_pos,
            protein_pos_holo=data.protein_pos_holo,
            protein_atom_name=data.protein_atom_name,
            protein_atom_to_aa_name=data.protein_atom_to_aa_name,
            protein_atom_to_aa_group=data.protein_atom_to_aa_group,
        )
    )
    data.apo_filename = str(pdb_path)
    data.holo_filename = str(pdb_path)
    data.ligand_filename = str(sdf_path)

    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(train_config.data.transform.ligand_atom_mode)
    bond_featurizer = trans.FeaturizeLigandBond()
    data = protein_featurizer(data)
    data = ligand_featurizer(data)
    data = bond_featurizer(data)
    return data, protein_featurizer, ligand_featurizer


def _load_models(repo: Path, train_config, sample_config, protein_featurizer, ligand_featurizer, device: torch.device):
    from graphbap.bapnet import BAPNet
    from models.molopt_score_model import ScorePosNet3D

    checkpoint = repo / sample_config.model.checkpoint
    net_cond_path = repo / train_config.net_cond.ckpt_path

    net_cond = BAPNet(
        ckpt_path=str(net_cond_path),
        hidden_nf=train_config.net_cond.hidden_dim,
    ).to(device)
    net_cond.eval()

    ckpt = torch.load(str(checkpoint), map_location=device)
    model = ScorePosNet3D(
        train_config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
    ).to(device)
    state_dict = {
        key.replace("model.", ""): value
        for key, value in ckpt["state_dict"].items()
        if key.startswith("model.")
    }
    model.load_state_dict(state_dict)
    model.eval()
    return model, net_cond


def _reconstruct_and_write(pred_ligand_pos, pred_v, out_sdf: Path) -> tuple[int, int]:
    from utils import reconstruct, transforms

    writer = Chem.SDWriter(str(out_sdf))
    recon_success = 0
    complete = 0
    for pred_pos, pred_atom_index in zip(pred_ligand_pos, pred_v):
        pred_pos = np.asarray(pred_pos, dtype=np.float64)
        pred_atom_type = transforms.get_atomic_number_from_index(
            pred_atom_index,
            mode="add_aromatic",
        )
        pred_atom_type = [int(x) for x in pred_atom_type]
        pred_aromatic = transforms.is_aromatic_from_index(
            pred_atom_index,
            mode="add_aromatic",
        )
        pred_aromatic = [bool(x) for x in pred_aromatic]
        try:
            mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
        except reconstruct.MolReconsError:
            continue
        recon_success += 1
        if "." not in Chem.MolToSmiles(mol):
            complete += 1
        writer.write(mol)
    writer.close()
    return recon_success, complete


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--pdb", type=Path, required=True)
    parser.add_argument("--sdf", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    repo = args.repo.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(repo))
    _inject_local_package(repo, "utils")

    from datasets.pl_data import FOLLOW_BATCH

    sample_config = _load_config(repo / "configs" / "sampling.yaml")
    train_config = _load_config(repo / "configs" / "training.yaml")
    sample_config.sample.num_steps = args.num_steps
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(sample_config.sample.seed))

    data, protein_featurizer, ligand_featurizer = _build_data(repo, args.pdb.resolve(), args.sdf.resolve(), train_config)
    model, net_cond = _load_models(repo, train_config, sample_config, protein_featurizer, ligand_featurizer, device)

    all_pred_ligand_pos = []
    all_pred_v = []
    all_times = []
    remaining = args.num_samples
    while remaining > 0:
        n_data = min(args.batch_size, remaining)
        batch = Batch.from_data_list([data.clone() for _ in range(n_data)], follow_batch=FOLLOW_BATCH).to(device)
        batch_protein = batch.protein_element_batch.to(device)
        batch_ligand = batch.ligand_element_batch.to(device)
        ligand_num_atoms = scatter_sum(torch.ones_like(batch_ligand), batch_ligand, dim=0).tolist()
        center_pos = scatter_mean(batch.protein_pos_holo.to(device), batch_protein, dim=0).to(device)
        batch_center_pos = center_pos[batch_ligand].to(device)
        init_ligand_pos = batch_center_pos + torch.randn_like(batch_center_pos).to(device)
        init_ligand_v = torch.randint(
            model.num_classes,
            (len(batch_ligand),),
            device=device,
        )

        t1 = time.time()
        with torch.no_grad():
            result = model.sample_diffusion(
                data=batch,
                protein_pos_apo=batch.protein_pos.to(device),
                protein_pos_holo=batch.protein_pos_holo.to(device),
                protein_v=batch.protein_atom_feature.float().to(device),
                batch_protein=batch_protein,
                init_ligand_pos=init_ligand_pos,
                init_ligand_v=init_ligand_v,
                batch_ligand=batch_ligand,
                num_steps=args.num_steps,
                pos_only=sample_config.sample.pos_only,
                center_pos_mode=sample_config.sample.center_pos_mode,
                net_cond=net_cond,
                cond_dim=train_config.model.cond_dim,
            )
        all_times.append(time.time() - t1)
        ligand_cum_atoms = torch.tensor([0] + ligand_num_atoms).cumsum(0).tolist()
        ligand_pos = result["ligand_pos"].detach().cpu().numpy()
        ligand_v = result["v"].detach().cpu().numpy()
        for idx in range(n_data):
            all_pred_ligand_pos.append(ligand_pos[ligand_cum_atoms[idx] : ligand_cum_atoms[idx + 1]])
            all_pred_v.append(ligand_v[ligand_cum_atoms[idx] : ligand_cum_atoms[idx + 1]])
        remaining -= n_data

    out_sdf = args.out_dir / "apo2mol_generated.sdf"
    recon_success, complete = _reconstruct_and_write(all_pred_ligand_pos, all_pred_v, out_sdf)
    print(f"device={device}")
    print(f"num_samples={args.num_samples}")
    print(f"batch_size={args.batch_size}")
    print(f"num_steps={args.num_steps}")
    print(f"diffusion_seconds_total={sum(all_times):.4f}")
    print(f"diffusion_seconds_per_sample={sum(all_times) / args.num_samples:.4f}")
    print(f"reconstruct_success={recon_success}")
    print(f"complete={complete}")
    print(f"sdf={out_sdf}")


if __name__ == "__main__":
    main()
