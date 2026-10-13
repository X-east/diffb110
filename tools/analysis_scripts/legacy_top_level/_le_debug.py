"""Quick debug: logp_distribution reward effectiveness in early epochs."""
import pandas as pd
import sys
sys.path.insert(0, ".")
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np

runs = "runs/7gpu-dock-ecfp4-fcfp4_20260525_212140"

for ep in [1, 2, 3, 10, 20]:
    df = pd.read_csv(f"{runs}/epochs/epoch_{ep:04d}/molecules.csv")
    valid = df[df["rdkit_valid"].astype(int) == 1].copy()
    if len(valid) == 0:
        print(f"=== Epoch {ep}: NO valid molecules ===\n")
        continue

    # Compute LogP
    logs = []
    heavy_atoms = []
    for s in valid["smiles"]:
        mol = Chem.MolFromSmiles(s)
        if mol:
            logs.append(Descriptors.MolLogP(mol))
            heavy_atoms.append(mol.GetNumHeavyAtoms())
        else:
            logs.append(np.nan)
            heavy_atoms.append(np.nan)
    valid["LogP"] = logs
    valid["n_heavy"] = heavy_atoms

    cols = [
        "smiles", "logp_distribution_reward", "logp_distribution_gap",
        "docking_reward", "docking_raw", "LogP", "n_heavy",
    ]
    print(f"=== Epoch {ep} ({len(valid)} valid) ===")
    print(valid[cols].describe().to_string())
    print("  LogP bins:")
    bins = [-np.inf, 0.5, 1.5, 2.5, 3.0, 3.5, 4.5, 5.5, np.inf]
    labels = ["<0.5", "[0.5-1.5)", "[1.5-2.5)", "[2.5-3.0)", "[3.0-3.5)", "[3.5-4.5)", "[4.5-5.5)", ">5.5"]
    logp_arr = valid["LogP"].dropna()
    for i in range(len(bins) - 1):
        cnt = ((logp_arr >= bins[i]) & (logp_arr < bins[i+1])).sum()
        pct = cnt / len(logp_arr) * 100 if len(logp_arr) > 0 else 0
        print(f"    {labels[i]:12s}: {cnt:4d} ({pct:5.1f}%)")
    print("  Sample SMILES (with rewards):")
    for _, row in valid.head(5).iterrows():
        print(
            f"    {row.smiles:50s}  le_r={row.logp_distribution_reward:.3f}  "
            f"dock={row.docking_raw:.1f}  LogP={row.LogP:.1f}"
        )
    print()
