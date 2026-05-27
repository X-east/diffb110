"""纯 Python 验证: per-rank vs global LogP 分布奖励."""
import csv,gzip,numpy as np,pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

ROOT='/home/md666/data/disk6/DiffB110'
os=f'{ROOT}/OUT/7gpu-dock-ecfp4-fcfp4_20260526_115341'
bins=[-np.inf,0.5,1.5,2.5,3.0,3.5,4.5,5.5,np.inf]; n=len(bins)-1
bin_labels=['(-∞,0.5)','[0.5,1.5)','[1.5,2.5)','[2.5,3.0)','[3.0,3.5)','[3.5,4.5)','[4.5,5.5)','(5.5,+∞)']
scale=6.0

# Load ChEMBL ref
ct=np.zeros(n,dtype=np.int64)
with gzip.open('data/chembl/chembl_36_properties.csv.gz','rt') as f:
    for row in csv.DictReader(f):
        try:
            lp=float(row['mol_log_p']);idx=np.digitize(lp,bins)-1
            if 0<=idx<n:ct[idx]+=1
        except:pass
ref=ct.astype(np.float64)/ct.sum()
print(f'ChEMBL ref: {[f"{p*100:.1f}%" for p in ref]}')
print()

def analyze_epoch(ep):
    mdf=pd.read_csv(f'{os}/epochs/epoch_{ep:04d}/molecules.csv')
    print(f'=== Epoch {ep} [{len(mdf)} total, {mdf["rdkit_valid"].sum()} valid] ===')
    for rank in sorted(mdf['rank'].unique()):
        rv=mdf[(mdf['rank']==rank)&(mdf['rdkit_valid']==1)]
        logps=[]
        for s in rv['smiles']:
            try:
                m=Chem.MolFromSmiles(s);logps.append(Descriptors.MolLogP(m) if m else np.nan)
            except:logps.append(np.nan)
        logps=np.array([l for l in logps if not np.isnan(l)])
        counts=np.zeros(n)
        for lp in logps:
            idx=np.digitize(lp,bins)-1
            if 0<=idx<n:counts[idx]+=1
        lap=(counts+1.0)/(counts.sum()+n)
        gaps=ref-lap;rews=np.clip(scale*gaps,-2,2)
        mol_rews=[]
        for lp in logps:
            idx=np.digitize(lp,bins)-1
            mol_rews.append(float(rews[idx]) if 0<=idx<n else -1.0)
        n_inv=90-len(rv);mol_rews.extend([-1]*n_inv)
        print(f'  R{int(rank)}: v={len(logps):2d} avgLogP={np.mean(logps):.2f} avgRew={np.mean(mol_rews):+.4f}')
        for bi in range(n):
            c=int(counts[bi])
            print(f'    {bin_labels[bi]:>12s}: n={c:2d} p={lap[bi]*100:5.1f}% gap={gaps[bi]:+.3f} rew={rews[bi]:+.1f}')
    print()

analyze_epoch(1)
analyze_epoch(5)
analyze_epoch(10)
analyze_epoch(14)
analyze_epoch(16)
