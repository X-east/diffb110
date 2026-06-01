from itertools import accumulate
import numpy as np
import torch
from torch.utils.data import Dataset

from edge_utils import build_ligand_halfedges


class ProcessedLigandPocketDataset(Dataset):
    def __init__(self, npz_path, center=True, transform=None):

        self.transform = transform

        with np.load(npz_path, allow_pickle=True) as f:
            data = {key: val for key, val in f.items()}

        lig_sections = np.where(np.diff(data['lig_mask']))[0] + 1
        pocket_sections = np.where(np.diff(data['pocket_mask']))[0] + 1
        lig_lengths = [len(x) for x in np.split(data['lig_mask'], lig_sections)]
        lig_offsets = np.array([0] + list(accumulate(lig_lengths[:-1])), dtype=np.int64)
        has_bonds = 'lig_bond_index' in data and 'lig_bond_type' in data
        if has_bonds:
            if 'num_lig_bonds' in data:
                directed_bond_counts = np.asarray(data['num_lig_bonds'], dtype=np.int64) * 2
            else:
                directed_bond_counts = np.bincount(
                    data['lig_bond_mask'].astype(np.int64),
                    minlength=len(lig_lengths),
                )
            bond_sections = np.cumsum(directed_bond_counts)[:-1]

        # split data based on mask
        self.data = {}
        for (k, v) in data.items():
            if k == 'names' or k == 'receptors':
                self.data[k] = v
                continue
            if k == 'lig_bond_mask':
                continue
            if k == 'num_lig_bonds':
                self.data[k] = [torch.tensor(int(x), dtype=torch.long) for x in v]
                continue
            if k == 'lig_bond_index':
                bond_indices = np.split(v, bond_sections, axis=1)
                self.data[k] = [
                    torch.from_numpy(x.copy()).long() - int(offset)
                    if x.size > 0 else torch.zeros((2, 0), dtype=torch.long)
                    for x, offset in zip(bond_indices, lig_offsets)
                ]
                continue
            if k == 'lig_bond_type':
                self.data[k] = [
                    torch.from_numpy(x.copy()).long()
                    for x in np.split(v, bond_sections)
                ]
                continue

            sections = lig_sections if 'lig' in k else pocket_sections
            self.data[k] = [torch.from_numpy(x) for x in np.split(v, sections)]

            # add number of nodes for convenience
            if k == 'lig_mask':
                self.data['num_lig_atoms'] = \
                    torch.tensor([len(x) for x in self.data['lig_mask']])
            elif k == 'pocket_mask':
                self.data['num_pocket_nodes'] = \
                    torch.tensor([len(x) for x in self.data['pocket_mask']])

        if center:
            for i in range(len(self.data['lig_coords'])):
                mean = (self.data['lig_coords'][i].sum(0) +
                        self.data['pocket_coords'][i].sum(0)) / \
                       (len(self.data['lig_coords'][i]) + len(self.data['pocket_coords'][i]))
                self.data['lig_coords'][i] = self.data['lig_coords'][i] - mean
                self.data['pocket_coords'][i] = self.data['pocket_coords'][i] - mean

    def __len__(self):
        return len(self.data['names'])

    def __getitem__(self, idx):
        data = {key: val[idx] for key, val in self.data.items()}
        if 'lig_bond_index' in data and 'lig_bond_type' in data:
            data.update(build_ligand_halfedges(
                int(data['num_lig_atoms']),
                data['lig_bond_index'],
                data['lig_bond_type'],
            ))
        if self.transform is not None:
            data = self.transform(data)
        return data

    @staticmethod
    def collate_fn(batch):
        out = {}
        ligand_offsets = [0] + list(accumulate(
            int(x['num_lig_atoms']) for x in batch[:-1]
        ))
        for prop in batch[0].keys():

            if prop == 'names' or prop == 'receptors':
                out[prop] = [x[prop] for x in batch]
            elif prop == 'lig_bond_index' or prop == 'ligand_halfedge_index':
                out[prop] = torch.cat([
                    x[prop] + int(offset) if x[prop].numel() > 0 else x[prop]
                    for x, offset in zip(batch, ligand_offsets)
                ], dim=1)
            elif prop == 'num_lig_atoms' or prop == 'num_pocket_nodes' \
                    or prop == 'num_virtual_atoms' or prop == 'num_lig_bonds' \
                    or prop == 'num_ligand_halfedges':
                out[prop] = torch.tensor([int(x[prop]) for x in batch])
            elif 'mask' in prop:
                # make sure indices in batch start at zero (needed for
                # torch_scatter)
                out[prop] = torch.cat([i * torch.ones(len(x[prop]))
                                       for i, x in enumerate(batch)], dim=0)
            else:
                out[prop] = torch.cat([x[prop] for x in batch], dim=0)

        if 'lig_bond_type' in out:
            out['lig_bond_type_batch'] = torch.cat([
                i * torch.ones(len(x['lig_bond_type']), dtype=torch.long)
                for i, x in enumerate(batch)
            ], dim=0)
        if 'ligand_halfedge_type' in out:
            out['ligand_halfedge_type_batch'] = torch.cat([
                i * torch.ones(len(x['ligand_halfedge_type']), dtype=torch.long)
                for i, x in enumerate(batch)
            ], dim=0)
        return out
