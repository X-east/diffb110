"""验证 LogP 分布奖励是 per-rank 还是 global 计算。"""
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

ROOT = '/home/md666/data/disk6/DiffB110/OUT/7gpu-dock-ecfp4-fcfp4_20260526_115341'
mdf = pd.read_csv(f'{ROOT}/epochs/epoch_0016/molecules.csv')
valid = mdf[mdf['rdkit_valid'] == 1].copy()

bins = [-float('inf'), 0.5, 1.5, 2.5, 3.0, 3.5, 4.5, 5.5, float('inf')]
n = len(bins) - 1
scale = 6.0
ref = np.array([0.044, 0.093, 0.206, 0.140, 0.143, 0.227, 0.108, 0.039])
bin_labels = ['(-∞,0.5)','[0.5,1.5)','[1.5,2.5)','[2.5,3.0)',
              '[3.0,3.5)','[3.5,4.5)','[4.5,5.5)','(5.5,+∞)']

def compute_rank_reward(rank_valid_mols, n_per_rank=90):
    """Compute LogP distribution reward for one rank's molecules."""
    logps = []
    for smi in rank_valid_mols['smiles']:
        try:
            m = Chem.MolFromSmiles(smi)
            logps.append(Descriptors.MolLogP(m) if m else np.nan)
        except:
            logps.append(np.nan)
    logps_ok = np.array([l for l in logps if not np.isnan(l)])
    
    counts = np.zeros(n)
    for lp in logps_ok:
        idx = int(np.digitize(lp, bins)) - 1
        if 0 <= idx < n: counts[idx] += 1
    
    props = (counts + 1.0) / (counts.sum() + n)  # Laplace smoothed
    gaps = ref - props
    rews = np.clip(scale * gaps, -2, 2)
    
    # Per-molecule assignment + failures get -1.0
    mol_rews = []
    for lp in logps_ok:
        idx = int(np.digitize(lp, bins)) - 1
        mol_rews.append(float(rews[idx]) if 0 <= idx < n else -1.0)
    n_invalid = n_per_rank - len(logps_ok)
    if n_invalid > 0:
        mol_rews.extend([-1.0] * n_invalid)
    
    return {
        'counts': counts, 'props': props, 'gaps': gaps,
        'rews': rews, 'avg_logp': np.mean(logps_ok) if len(logps_ok) else 0,
        'avg_reward': np.mean(mol_rews), 'n_valid': len(logps_ok),
    }

print('=' * 70)
print('Per-Rank LogP Distribution (Epoch 16, 90 mol/rank × 7 ranks)')
print('=' * 70)
rank_results = {}
for rank in sorted(valid['rank'].unique()):
    rdf = valid[valid['rank'] == rank]
    res = compute_rank_reward(rdf, n_per_rank=90)
    rank_results[rank] = res
    print(f'\nRank {int(rank)}: {res["n_valid"]} valid, avg LogP={res["avg_logp"]:.2f}, avg reward={res["avg_reward"]:.4f}')
    for bi in range(n):
        c = int(res['counts'][bi])
        print(f'  {bin_labels[bi]:>12s}: count={c:3d}  prop={res["props"][bi]*100:5.1f}%  '
              f'gap={res["gaps"][bi]:+.3f}  rew={res["rews"][bi]:+.1f}')

# Cross-rank average (= actual training signal)
print('\n' + '=' * 70)
print('Comparison: Per-Rank (actual) vs Global 630-mol (my previous analysis)')
print('=' * 70)
per_rank_avg = np.mean([r['avg_reward'] for r in rank_results.values()])
per_rank_avg_logp = np.mean([r['avg_logp'] for r in rank_results.values()])

# Global 630
all_df = valid
all_res = compute_rank_reward(all_df, n_per_rank=630)
print(f'\nPer-Rank mean avg reward: {per_rank_avg:.4f}  |  LogP: {per_rank_avg_logp:.2f}')
print(f'Global 630 avg reward:    {all_res["avg_reward"]:.4f}  |  LogP: {all_res["avg_logp"]:.2f}')
print(f'CSV reported:             0.189')
print()
print('Per-rank bin props (mean ± std):')
per_rank_props = np.array([rank_results[r]['props'] for r in sorted(rank_results)])
for bi in range(n):
    m = per_rank_props[:, bi].mean()
    s = per_rank_props[:, bi].std()
    print(f'  {bin_labels[bi]:>12s}: mean={m*100:5.1f}% ± {s*100:4.1f}%  '
          f'(global={all_res["props"][bi]*100:5.1f}%)')
