"""Run PMDM pocket-conditioned sampling without editing the PMDM source tree."""

from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path

import torch
from rdkit import Chem
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
from tqdm import tqdm


def _repo_imports(repo: Path) -> None:
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    for module_name in ("utils", "models", "configs"):
        if module_name in sys.modules:
            del sys.modules[module_name]
        module = types.ModuleType(module_name)
        module.__path__ = [str(repo / module_name)]
        module.__package__ = module_name
        sys.modules[module_name] = module
    if "evaluation" in sys.modules:
        del sys.modules["evaluation"]
    try:
        import six

        sys.modules.setdefault("rdkit.six", six)
        sys.modules.setdefault("rdkit.six.moves", six.moves)
    except ImportError:
        pass


def main() -> None:
    """Sample molecules for one PMDM pocket and write an aggregate SDF."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--pdb-path", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--num-atom", type=int, default=24)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--sampling-type", default="generalized")
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (overrides checkpoint train.seed, required for multi-shard)")
    parser.add_argument("--fixed-attempts", action="store_true",
                        help="Stop after num_samples attempts regardless of success (for fair benchmarking)")
    args = parser.parse_args()

    _repo_imports(args.repo.resolve())

    import sample_for_pdb as pm
    from configs.dataset_config import get_dataset_info
    from models.epsnet import get_model
    from utils.sample import construct_dataset_pocket
    from utils.data import ProteinLigandData
    from utils.transforms import (
        ATOM_FAMILIES,
        CountNodesPerGraph,
        FeaturizeLigandAtom,
        FeaturizeProteinAtom,
        GetAdj,
        LigandCountNeighbors,
    )

    def _protein_ligand_inc(self, key, value, *inc_args, **inc_kwargs):
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

    ProteinLigandData.__inc__ = _protein_ligand_inc

    ckpt = torch.load(str(args.ckpt), map_location="cpu")
    config = ckpt["config"]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    pm.seed_all(args.seed if args.seed is not None else config.train.seed)

    if config.dataset.name != "crossdock":
        raise ValueError(f"Only crossdock checkpoint path is wired here, got {config.dataset.name}")
    dataset_info = get_dataset_info("crossdock_pocket", False)
    atomic_numbers = pm.atomic_numbers_crossdock

    protein_featurizer = FeaturizeProteinAtom(config.dataset.name, pocket=True)
    ligand_featurizer = FeaturizeLigandAtom(config.dataset.name, pocket=True)
    transform = Compose(
        [
            LigandCountNeighbors(),
            protein_featurizer,
            ligand_featurizer,
            CountNodesPerGraph(),
            GetAdj(only_prot=True),
        ]
    )

    data = pm.pdb_to_pocket_data(str(args.pdb_path), None)
    data = transform(data)

    model = get_model(config.model).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    data_list, _ = construct_dataset_pocket(
        args.num_samples * 3,
        args.batch_size,
        dataset_info,
        args.num_atom,
        args.num_atom,
        None,
        None,
        data.protein_atom_feature.float(),
        data.protein_atom_feature_full.float(),
        data.protein_pos,
        data.protein_bond_index,
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    sdf_path = args.outdir / "pmdm_generated.sdf"
    writer = Chem.SDWriter(str(sdf_path))

    attempted = 0
    valid = 0
    stable = 0
    smiles_seen: list[str] = []

    for datas in tqdm(data_list, desc="pmdm_batches"):
        if args.fixed_attempts:
            if attempted >= args.num_samples:
                break
        else:
            if stable >= args.num_samples:
                break
        batch = Batch.from_data_list(
            datas, follow_batch=["ligand_atom_feature", "protein_atom_feature_full"]
        ).to(device)
        with torch.no_grad():
            try:
                pos_gen, _, atom_type, _ = model.langevin_dynamics_sample(
                    ligand_atom_type=batch.ligand_atom_feature.float(),
                    ligand_pos_init=batch.ligand_pos,
                    ligand_bond_index=batch.ligand_bond_index,
                    ligand_bond_type=batch.ligand_bond_type,
                    ligand_num_node=batch.ligand_num_node,
                    ligand_batch=batch.ligand_atom_feature_batch,
                    protein_atom_type=batch.protein_atom_feature,
                    protein_atom_feature_full=batch.protein_atom_feature_full,
                    protein_pos=batch.protein_pos,
                    protein_bond_index=batch.protein_bond_index,
                    protein_backbone_mask=None,
                    protein_batch=batch.protein_atom_feature_full_batch,
                    num_graphs=batch.num_graphs,
                    extend_order=False,
                    n_steps=args.n_steps,
                    step_lr=1e-6,
                    w_global_pos=1.0,
                    w_global_node=1.0,
                    w_local_pos=1.0,
                    w_local_node=1.0,
                    global_start_sigma=float("inf"),
                    sampling_type=args.sampling_type,
                    eta=args.eta,
                    context=None,
                )
            except FloatingPointError:
                continue

        pos_list = pm.unbatch(pos_gen, batch.ligand_atom_feature_batch)
        atom_list = pm.unbatch(atom_type, batch.ligand_atom_feature_batch)
        for pos, atom_logits in zip(pos_list, atom_list):
            attempted += 1
            try:
                pos_cpu = pos.detach().cpu()
                atom_logits_cpu = atom_logits.detach().cpu()
                new_element = torch.tensor(
                    [atomic_numbers[i] for i in torch.argmax(atom_logits_cpu[:, :8], dim=1)]
                )
                indicator_idx = torch.argmax(atom_logits_cpu[:, 8:], dim=1)
                indicators = torch.zeros(
                    [pos_cpu.size(0), len(ATOM_FAMILIES)], dtype=torch.long
                )
                for atom_i, family_i in enumerate(indicator_idx):
                    indicators[atom_i, int(family_i)] = 1
                mol = pm.reconstruct_from_generated(pos_cpu, new_element, indicators)
                smiles = pm.mol2smiles(mol)
            except Exception:
                continue
            if smiles is None:
                continue
            valid += 1
            if "." in smiles:
                continue
            stable += 1
            smiles_seen.append(smiles)
            writer.write(mol)
            if not args.fixed_attempts and stable >= args.num_samples:
                break

    writer.close()
    print(
        f"attempted={attempted} valid={valid} stable={stable} "
        f"unique={len(set(smiles_seen))} sdf={sdf_path}"
    )


if __name__ == "__main__":
    main()
