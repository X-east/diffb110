"""Directly run LogPDistributionReward on epoch 16 rdmols to verify. """
import sys, os
ROOT = '/home/md666/data/disk6/DiffB110'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/RLPF-verl_diffusion')
sys.path.insert(0, f'{ROOT}/DiffSBDD')

import pandas as pd, numpy as np, torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from ddpo.rewards.logp_distribution_reward import LogPDistributionReward, DEFAULT_LOGP_BINS

mdf = pd.read_csv(f'{ROOT}/runs/7gpu-dock-ecfp4-fcfp4_20260526_115341/epochs/epoch_0016/molecules.csv')
valid = mdf[mdf['rdkit_valid'] == 1].copy()

# Build rdmols from SMILES
rdmols, rdkit_valid = [], []
for _, row in mdf.iterrows():
    try:
        m = Chem.MolFromSmiles(row['smiles'])
        rdmols.append(m)
        rdkit_valid.append(1 if m is not None else 0)
    except:
        rdmols.append(None)
        rdkit_valid.append(0)

# Init reward
rew = LogPDistributionReward(
    chembl_csv='./data/chembl/chembl_36_properties.csv.gz',
    logp_bins=DEFAULT_LOGP_BINS,
    failure_penalty=-1.0,
    min_reward=-2.0, max_reward=2.0,
    scale=6.0, ema_alpha=0.0,
    reward_prefix='logp_distribution',
)
print(f'ChEMBL ref: {[f"{p*100:.1f}%" for p in rew.chmbl_proportions]}')

# Per-rank analysis
bins = DEFAULT_LOGP_BINS
n = len(bins) - 1
bin_labels = ['(-∞,0.5)','[0.5,1.5)','[1.5,2.5)','[2.5,3.0)','[3.0,3.5)','[3.5,4.5)','[4.5,5.5)','(5.5,+∞)']
ref = rew.chmbl_proportions

for rank in sorted(valid['rank'].unique()):
    rdf = mdf[mdf['rank'] == rank]
    r_rdmols = [Chem.MolFromSmiles(s) for s in rdf['smiles']]
    r_valid = [1 if m is not None else 0 for m in r_rdmols]
    
    # Simulate the reward computation
    B = 90
    valid_logps = []
    mol_to_bin = {}
    for i in range(B):
        if i >= len(r_rdmols) or r_rdmols[i] is None:
            continue
        try:
            lp = float(Descriptors.MolLogP(r_rdmols[i]))
            idx = int(np.digitize(lp, bins)) - 1
            if 0 <= idx < n:
                valid_logps.append(lp)
                mol_to_bin[i] = idx
        except:
            continue
    
    # Batch proportions (Laplace)
    counts = np.zeros(n, dtype=np.float64)
    for lp in valid_logps:
        idx = int(np.digitize(lp, bins)) - 1
        if 0 <= idx < n: counts[idx] += 1
    batch_props = (counts + 1.0) / (counts.sum() + n) if counts.sum() > 0 else np.ones(n)/n
    gaps = ref - batch_props
    
    # Per-molecule rewards
    rewards = np.full(B, -1.0)  # failure_penalty
    for i, bi in mol_to_bin.items():
        r = np.clip(6.0 * gaps[bi], -2.0, 2.0)
        rewards[i] = float(r)
    
    avg_valid = np.mean([rewards[i] for i in mol_to_bin]) if mol_to_bin else 0
    avg_all = np.mean(rewards)
    
    print(f'Rank {int(rank)}: valid={len(valid_logps)} valid_only_avg={avg_valid:.4f} all_avg={avg_all:.4f} logp_avg={np.mean(valid_logps):.2f}')
