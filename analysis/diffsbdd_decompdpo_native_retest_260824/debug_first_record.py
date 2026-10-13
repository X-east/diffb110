import pickle
import traceback

import torch

from utils.data import PDBProtein, ProteinLigandData, parse_sdf_file, torchify_dict
from utils.prior import compute_golden_prior_from_data


def main() -> None:
    """Trace preprocessing of the first official DecompDPO test record."""
    meta = pickle.load(open("/data/disk6/DiffB110/reference project/代码/DecompDPO/data/test_set/BSD_ASPTE_1_130_0/2z3h_A_rec_1wn6_bst_lig_tt_docked_3_meta.pkl", "rb"))["data"]
    try:
        num_arms, num_scaffold = meta["num_arms"], meta["num_scaffold"]
        protein = PDBProtein(meta["protein_file"])
        protein_dict = protein.to_dict_atom()
        ligand_dict = parse_sdf_file(meta["ligand_file"], kekulize=True)
        num_protein_atoms = len(protein.atoms)
        num_ligand_atoms = ligand_dict["rdmol"].GetNumAtoms()
        print("counts", num_protein_atoms, num_ligand_atoms, sum(len(x) for x in meta["all_submol_atom_idx"]))
        serial = [atom["atom_id"] for atom in protein.atoms]
        masks = []
        for pocket_serial in meta["all_pocket_atom_serial"]:
            pocket_idx = [serial.index(i) for i in pocket_serial]
            mask = torch.zeros(num_protein_atoms, dtype=torch.bool)
            mask[pocket_idx] = 1
            masks.append(mask)
        ligand_mask = torch.zeros(num_ligand_atoms, dtype=int)
        for arm_idx, atom_idx in enumerate(meta["all_submol_atom_idx"]):
            ligand_mask[atom_idx] = -1 if arm_idx == len(meta["all_submol_atom_idx"]) - 1 else arm_idx
        print("unique", ligand_mask.unique(), "expected", num_arms + num_scaffold)
        data = ProteinLigandData.from_protein_ligand_dicts(
            protein_dict=torchify_dict(protein_dict), ligand_dict=torchify_dict(ligand_dict)
        )
        for key, value in meta.items():
            data[key] = value
        data.num_arms, data.num_scaffold = num_arms, num_scaffold
        data.pocket_atom_masks, data.ligand_atom_mask = torch.stack(masks), ligand_mask
        data = compute_golden_prior_from_data(data)
        print("prior_ok", data.ligand_decomp_centers.shape)
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    main()
