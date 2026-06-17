import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from Bio.PDB import PDBParser
from rdkit import Chem, Geometry
from torch_scatter import scatter_mean
from openbabel import openbabel
openbabel.obErrorLog.StopLogging()  # suppress OpenBabel messages

import utils
from lightning_modules import LigandPocketDDPM
from constants import FLOAT_TYPE, INT_TYPE
from analysis.molecule_builder import process_molecule
from reconstruction import DEFAULT_RECONSTRUCTION_METHOD, reconstruct_molecule


def prepare_from_sdf_files(sdf_files, atom_encoder):

    ligand_coords = []
    atom_one_hot = []
    for file in sdf_files:
        rdmol = Chem.SDMolSupplier(str(file), sanitize=False)[0]
        ligand_coords.append(
            torch.from_numpy(rdmol.GetConformer().GetPositions()).float()
        )
        types = torch.tensor([atom_encoder[a.GetSymbol()] for a in rdmol.GetAtoms()])
        atom_one_hot.append(
            F.one_hot(types, num_classes=len(atom_encoder))
        )

    return torch.cat(ligand_coords, dim=0), torch.cat(atom_one_hot, dim=0)


def read_fixed_substructure_sdf(sdf_file):
    """Read a fixed substructure molecule from an SDF file."""
    rdmol = Chem.SDMolSupplier(str(sdf_file), sanitize=False)[0]
    if rdmol is None:
        raise ValueError(f"Could not read fixed substructure from {sdf_file}")
    try:
        Chem.SanitizeMol(rdmol)
    except ValueError:
        pass
    return rdmol


def prepare_ligand_from_pdb(biopython_atoms, atom_encoder):

    coord = torch.tensor(np.array([a.get_coord()
                                   for a in biopython_atoms]), dtype=FLOAT_TYPE)
    types = torch.tensor([atom_encoder[a.element.capitalize()]
                          for a in biopython_atoms])
    one_hot = F.one_hot(types, num_classes=len(atom_encoder))

    return coord, one_hot


def prepare_substructure(ref_ligand, fix_atoms, pdb_model):

    if fix_atoms[0].endswith(".sdf"):
        # ligand as sdf file
        coord, one_hot = prepare_from_sdf_files(fix_atoms, model.lig_type_encoder)

    else:
        # ligand contained in PDB; given in <chain>:<resi> format
        chain, resi = ref_ligand.split(':')
        ligand = utils.get_residue_with_resi(pdb_model[chain], int(resi))
        fixed_atoms = [a for a in ligand.get_atoms() if a.get_name() in set(fix_atoms)]
        coord, one_hot = prepare_ligand_from_pdb(fixed_atoms, model.lig_type_encoder)

    return coord, one_hot


def _restore_fixed_ligand_atoms(xh_lig, lig_mask, x_fixed, one_hot_fixed,
                                n_fixed, x_dims):
    """Restore fixed ligand atom coordinates and atom types before reconstruction."""
    if n_fixed == 0:
        return xh_lig

    x_fixed = x_fixed.to(device=xh_lig.device, dtype=xh_lig.dtype)
    one_hot_fixed = one_hot_fixed.to(device=xh_lig.device, dtype=xh_lig.dtype)
    for batch_idx in torch.unique(lig_mask, sorted=True):
        atom_indices = torch.nonzero(lig_mask == batch_idx, as_tuple=False).view(-1)
        if atom_indices.numel() < n_fixed:
            raise ValueError(
                f"Cannot restore {n_fixed} fixed atoms for ligand batch "
                f"{int(batch_idx)} with only {int(atom_indices.numel())} atoms"
            )
        fixed_indices = atom_indices[:n_fixed]
        xh_lig[fixed_indices, :x_dims] = x_fixed
        xh_lig[fixed_indices, x_dims:] = one_hot_fixed

    return xh_lig


def _preserve_fixed_substructure_graph(mol, fixed_mol, match_tolerance=0.05):
    """Preserve atom order, coordinates, and internal bonds of fixed atoms."""
    n_fixed = fixed_mol.GetNumAtoms()
    fixed_to_old = _match_fixed_atoms_by_coordinate(
        mol,
        fixed_mol,
        match_tolerance=match_tolerance,
    )
    fixed_old_indices = set(fixed_to_old.values())
    generated_old_indices = [
        atom.GetIdx() for atom in mol.GetAtoms()
        if atom.GetIdx() not in fixed_old_indices
    ]

    old_to_new = {
        old_idx: fixed_idx for fixed_idx, old_idx in fixed_to_old.items()
    }
    old_to_new.update(
        {
            old_idx: n_fixed + new_offset
            for new_offset, old_idx in enumerate(generated_old_indices)
        }
    )

    preserved = Chem.RWMol()
    for atom in fixed_mol.GetAtoms():
        preserved.AddAtom(Chem.Atom(atom))
    for old_idx in generated_old_indices:
        preserved.AddAtom(Chem.Atom(mol.GetAtomWithIdx(old_idx)))

    _copy_reordered_conformer(
        preserved,
        mol,
        fixed_mol,
        generated_old_indices,
    )
    _copy_fixed_core_bonds(preserved, fixed_mol)
    _copy_generated_and_attachment_bonds(preserved, mol, fixed_old_indices, old_to_new)

    out = preserved.GetMol()
    _copy_mol_properties(mol, out)
    return out


def _match_fixed_atoms_by_coordinate(mol, fixed_mol, match_tolerance):
    """Match fixed atoms to a reconstructed molecule using element and coordinates."""
    if mol.GetNumConformers() == 0 or fixed_mol.GetNumConformers() == 0:
        raise ValueError("Fixed-core graph preservation requires conformers")

    mol_conf = mol.GetConformer()
    fixed_conf = fixed_mol.GetConformer()
    unused_old_indices = {atom.GetIdx() for atom in mol.GetAtoms()}
    fixed_to_old = {}
    for fixed_atom in fixed_mol.GetAtoms():
        fixed_idx = fixed_atom.GetIdx()
        fixed_symbol = fixed_atom.GetSymbol()
        fixed_pos = fixed_conf.GetAtomPosition(fixed_idx)
        candidates = []
        for old_idx in unused_old_indices:
            old_atom = mol.GetAtomWithIdx(old_idx)
            if old_atom.GetSymbol() != fixed_symbol:
                continue
            old_pos = mol_conf.GetAtomPosition(old_idx)
            distance = fixed_pos.Distance(old_pos)
            candidates.append((distance, old_idx))
        if not candidates:
            raise ValueError(
                f"No reconstructed atom can match fixed atom {fixed_idx} "
                f"({fixed_symbol})"
            )
        distance, old_idx = min(candidates)
        if distance > match_tolerance:
            raise ValueError(
                f"Fixed atom {fixed_idx} ({fixed_symbol}) nearest match is "
                f"{distance:.4f} A away, above tolerance {match_tolerance:.4f} A"
            )
        fixed_to_old[fixed_idx] = old_idx
        unused_old_indices.remove(old_idx)

    return fixed_to_old


def _copy_reordered_conformer(preserved, mol, fixed_mol, generated_old_indices):
    """Copy fixed coordinates first, then generated coordinates."""
    fixed_conf = fixed_mol.GetConformer()
    mol_conf = mol.GetConformer()
    conformer = Chem.Conformer(preserved.GetNumAtoms())
    conformer.Set3D(True)
    n_fixed = fixed_mol.GetNumAtoms()
    for fixed_idx in range(n_fixed):
        pos = fixed_conf.GetAtomPosition(fixed_idx)
        conformer.SetAtomPosition(
            fixed_idx,
            Geometry.Point3D(float(pos.x), float(pos.y), float(pos.z)),
        )
    for new_offset, old_idx in enumerate(generated_old_indices):
        pos = mol_conf.GetAtomPosition(old_idx)
        conformer.SetAtomPosition(
            n_fixed + new_offset,
            Geometry.Point3D(float(pos.x), float(pos.y), float(pos.z)),
        )
    preserved.AddConformer(conformer)


def _copy_fixed_core_bonds(preserved, fixed_mol):
    """Copy all bonds between fixed atoms from the fixed-core molecule."""
    for bond in fixed_mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        preserved.AddBond(begin_idx, end_idx, bond.GetBondType())
        _copy_bond_flags(
            bond,
            preserved.GetBondBetweenAtoms(begin_idx, end_idx),
        )


def _copy_generated_and_attachment_bonds(preserved, mol, fixed_old_indices, old_to_new):
    """Copy generated-generated and fixed-generated bonds from reconstruction."""
    for bond in mol.GetBonds():
        old_begin = bond.GetBeginAtomIdx()
        old_end = bond.GetEndAtomIdx()
        if old_begin in fixed_old_indices and old_end in fixed_old_indices:
            continue
        if old_begin not in old_to_new or old_end not in old_to_new:
            continue
        new_begin = old_to_new[old_begin]
        new_end = old_to_new[old_end]
        if new_begin == new_end:
            continue
        if preserved.GetBondBetweenAtoms(new_begin, new_end) is not None:
            continue
        preserved.AddBond(new_begin, new_end, bond.GetBondType())
        _copy_bond_flags(
            bond,
            preserved.GetBondBetweenAtoms(new_begin, new_end),
        )


def _copy_bond_flags(source_bond, target_bond):
    """Copy non-query bond flags needed for aromatic core preservation."""
    if target_bond is None:
        return
    target_bond.SetIsAromatic(source_bond.GetIsAromatic())
    target_bond.SetIsConjugated(source_bond.GetIsConjugated())


def _copy_mol_properties(source, target):
    """Copy string properties from one RDKit molecule to another."""
    for prop_name in source.GetPropNames(includePrivate=True, includeComputed=False):
        target.SetProp(prop_name, source.GetProp(prop_name))


def inpaint_ligand(model, pdb_file, n_samples, ligand, fix_atoms,
                   add_n_nodes=None, center='ligand', sanitize=False,
                   largest_frag=False, relax_iter=0, timesteps=None,
                   resamplings=1, save_traj=False,
                   reconstruction_method=DEFAULT_RECONSTRUCTION_METHOD):
    """
    Generate ligands given a pocket
    Args:
        model: Lightning model
        pdb_file: PDB filename
        n_samples: number of samples
        ligand: reference ligand given in <chain>:<resi> format if the ligand is
                contained in the PDB file, or path to an SDF file that
                contains the ligand; used to define the pocket
        fix_atoms: ligand atoms that should be fixed, e.g. "C1 N6 C5 C12"
        center: 'ligand' or 'pocket'
        add_n_nodes: number of ligand nodes to add, sampled randomly if 'None'
        sanitize: whether to sanitize molecules or not
        largest_frag: only return the largest fragment
        relax_iter: number of force field optimization steps
        timesteps: number of denoising steps, use training value if None
        resamplings: number of resampling iterations
        save_traj: save intermediate states to visualize a denoising trajectory
        reconstruction_method: bond inference backend for RDKit reconstruction
    Returns:
        list of molecules
    """
    if save_traj and n_samples > 1:
        raise NotImplementedError("Can only visualize trajectory with "
                                  "n_samples=1.")
    frames = timesteps if save_traj else 1
    sanitize = False if save_traj else sanitize
    relax_iter = 0 if save_traj else relax_iter
    largest_frag = False if save_traj else largest_frag

    # Load PDB
    pdb_model = PDBParser(QUIET=True).get_structure('', pdb_file)[0]

    # Define pocket based on reference ligand
    residues = utils.get_pocket_from_ligand(pdb_model, ligand)
    pocket = model.prepare_pocket(residues, repeats=n_samples)

    # Get fixed ligand substructure
    x_fixed, one_hot_fixed = prepare_substructure(ligand, fix_atoms, pdb_model)
    n_fixed = len(x_fixed)
    fixed_mol = None
    if fix_atoms[0].endswith(".sdf"):
        fixed_mol = read_fixed_substructure_sdf(fix_atoms[0])
        if fixed_mol.GetNumAtoms() != n_fixed:
            raise ValueError(
                f"Fixed SDF atom count {fixed_mol.GetNumAtoms()} does not "
                f"match fixed tensor atom count {n_fixed}"
            )

    if add_n_nodes is None:
        num_nodes_lig = model.ddpm.size_distribution.sample_conditional(
            n1=None, n2=pocket['size'])
        num_nodes_lig = torch.clamp(num_nodes_lig, min=n_fixed)
    else:
        num_nodes_lig = torch.ones(n_samples, dtype=int) * n_fixed + add_n_nodes

    ligand_mask = utils.num_nodes_to_batch_mask(
        len(num_nodes_lig), num_nodes_lig, model.device)

    ligand = {
        'x': torch.zeros((len(ligand_mask), model.x_dims),
                         device=model.device, dtype=FLOAT_TYPE),
        'one_hot': torch.zeros((len(ligand_mask), model.atom_nf),
                               device=model.device, dtype=FLOAT_TYPE),
        'size': num_nodes_lig,
        'mask': ligand_mask
    }

    # fill in fixed atoms
    lig_fixed = torch.zeros_like(ligand_mask)
    for i in range(n_samples):
        sele = (ligand_mask == i)

        x_new = ligand['x'][sele]
        x_new[:n_fixed] = x_fixed
        ligand['x'][sele] = x_new

        h_new = ligand['one_hot'][sele]
        h_new[:n_fixed] = one_hot_fixed
        ligand['one_hot'][sele] = h_new

        fixed_new = lig_fixed[sele]
        fixed_new[:n_fixed] = 1
        lig_fixed[sele] = fixed_new

    # Pocket's center of mass
    pocket_com_before = scatter_mean(pocket['x'], pocket['mask'], dim=0)

    # Run sampling
    xh_lig, xh_pocket, lig_mask, pocket_mask = model.ddpm.inpaint(
        ligand, pocket, lig_fixed, center=center,
        resamplings=resamplings, timesteps=timesteps, return_frames=frames)

    # Treat intermediate states as molecules for downstream processing
    if save_traj:
        xh_lig = utils.reverse_tensor(xh_lig)
        xh_pocket = utils.reverse_tensor(xh_pocket)

        lig_mask = torch.arange(xh_lig.size(0), device=model.device
                                ).repeat_interleave(len(lig_mask))
        pocket_mask = torch.arange(xh_pocket.size(0), device=model.device
                                   ).repeat_interleave(len(pocket_mask))

        xh_lig = xh_lig.view(-1, xh_lig.size(2))
        xh_pocket = xh_pocket.view(-1, xh_pocket.size(2))

    # Move generated molecule back to the original pocket position
    pocket_com_after = scatter_mean(xh_pocket[:, :model.x_dims], pocket_mask, dim=0)

    xh_pocket[:, :model.x_dims] += \
        (pocket_com_before - pocket_com_after)[pocket_mask]
    xh_lig[:, :model.x_dims] += \
        (pocket_com_before - pocket_com_after)[lig_mask]
    xh_lig = _restore_fixed_ligand_atoms(
        xh_lig,
        lig_mask,
        x_fixed,
        one_hot_fixed,
        n_fixed,
        model.x_dims,
    )

    # Build mol objects
    x = xh_lig[:, :model.x_dims].detach().cpu()
    atom_type = xh_lig[:, model.x_dims:].argmax(1).detach().cpu()
    lig_mask_cpu = lig_mask.detach().cpu()

    molecules = []
    for mol_pc in zip(utils.batch_to_list(x, lig_mask_cpu),
                      utils.batch_to_list(atom_type, lig_mask_cpu)):

        mol = reconstruct_molecule(
            *mol_pc,
            model.dataset_info,
            method=reconstruction_method,
            add_coords=True,
        )
        if fixed_mol is not None:
            mol = _preserve_fixed_substructure_graph(mol, fixed_mol)
        mol = process_molecule(mol,
                               add_hydrogens=False,
                               sanitize=sanitize,
                               relax_iter=relax_iter,
                               largest_frag=largest_frag)
        if mol is not None:
            molecules.append(mol)

    return molecules


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--pdbfile', type=str)
    parser.add_argument('--ref_ligand', type=str, default=None)
    parser.add_argument('--fix_atoms', type=str, nargs='+', default=None)
    parser.add_argument('--center', type=str, default='ligand', choices={'ligand', 'pocket'})
    parser.add_argument('--outfile', type=Path)
    parser.add_argument('--n_samples', type=int, default=20)
    parser.add_argument('--add_n_nodes', type=int, default=None)
    parser.add_argument('--relax', action='store_true')
    parser.add_argument('--sanitize', action='store_true')
    parser.add_argument('--resamplings', type=int, default=20)
    parser.add_argument('--timesteps', type=int, default=50)
    parser.add_argument('--save_traj', action='store_true')
    parser.add_argument(
        '--reconstruction_method',
        type=str,
        default=DEFAULT_RECONSTRUCTION_METHOD,
        choices=('openbabel', 'obabel', 'edm', 'distance', 'distance_rule'),
    )
    args = parser.parse_args()

    pdb_id = Path(args.pdbfile).stem

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    model = LigandPocketDDPM.load_from_checkpoint(
        args.checkpoint, map_location=device)
    model = model.to(device)

    molecules = inpaint_ligand(model, args.pdbfile, args.n_samples,
                               args.ref_ligand, args.fix_atoms,
                               args.add_n_nodes, center=args.center,
                               sanitize=args.sanitize,
                               largest_frag=False,
                               relax_iter=(200 if args.relax else 0),
                               timesteps=args.timesteps,
                               resamplings=args.resamplings,
                               save_traj=args.save_traj,
                               reconstruction_method=args.reconstruction_method)

    # Make SDF files
    utils.write_sdf_file(args.outfile, molecules)
