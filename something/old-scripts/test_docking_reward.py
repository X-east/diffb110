#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test DockingReward using generated molecules.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "DiffSBDD"))

import torch
import numpy as np
from rdkit import Chem


def main():
    print("\n" + "="*60)
    print("  DOCKING REWARD TEST")
    print("="*60)
    
    # Load generated molecule data
    batch_dir = Path("generated_molecules/batch_data")
    if not batch_dir.exists():
        print("Error: Run generate_and_visualize.py first!")
        return
    
    print("\n[1] Loading generated molecules...", flush=True)
    x_final = np.load(batch_dir / "x_final.npy")
    h_categorical = np.load(batch_dir / "h_categorical.npy")
    node_counts = np.load(batch_dir / "node_counts.npy")
    
    print(f"    Loaded {len(node_counts)} molecules")
    print(f"    x shape: {x_final.shape}")
    print(f"    h shape: {h_categorical.shape}")
    
    # Test molecule building first
    print("\n[2] Testing molecule building...", flush=True)
    from analysis.molecule_builder import build_molecule, process_molecule
    from constants import dataset_params
    
    dataset_info = dataset_params['crossdock']
    atom_decoder = dataset_info['atom_decoder']
    
    valid_mols = []
    for i in range(len(node_counts)):
        n = node_counts[i]
        pos = torch.from_numpy(x_final[i, :n])
        atom_types = torch.from_numpy(h_categorical[i, :n].argmax(axis=-1))
        
        try:
            mol = build_molecule(
                positions=pos,
                atom_types=atom_types,
                dataset_info=dataset_info,
                add_coords=True,
                use_openbabel=False
            )
            
            if mol is not None:
                # Try to get SMILES (without sanitization)
                smiles = Chem.MolToSmiles(mol)
                valid_mols.append((i, mol, smiles))
                print(f"    Mol {i+1}: {n} atoms, SMILES={smiles[:50]}...")
            else:
                print(f"    Mol {i+1}: build_molecule returned None")
                
        except Exception as e:
            print(f"    Mol {i+1}: Error - {e}")
    
    print(f"\n    Valid molecules: {len(valid_mols)}/{len(node_counts)}")
    
    # Test DockingReward initialization
    print("\n[3] Testing DockingReward initialization...", flush=True)
    from ddpo.rewards.docking import DockingReward
    
    try:
        reward_fn = DockingReward(
            receptor_file="data/8GJJ-C-Cip.pdb",
            normalization_factor=-10.0,
            failure_penalty=0.0,
        )
        print(f"    {reward_fn}")
        print("    [PASS] DockingReward initialized successfully")
    except Exception as e:
        print(f"    [FAIL] {e}")
        return
    
    # Note about docking
    print("\n[4] Docking test skipped (Windows temp file issues)")
    print("    Docking requires smina and temp file access")
    print("    On Linux/CentOS, docking should work correctly")
    
    print("\n" + "="*60)
    print("  TEST SUMMARY")
    print("="*60)
    print(f"    Molecule building: {len(valid_mols)}/{len(node_counts)} valid")
    print(f"    DockingReward init: PASS")
    print(f"    Docking execution: SKIPPED (Windows)")
    print("="*60)


if __name__ == "__main__":
    main()
