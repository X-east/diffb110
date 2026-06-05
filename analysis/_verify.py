import sys,os,numpy as np,pandas as pd
ROOT='/home/md666/data/disk6/DiffB110'; os.chdir(ROOT)
sys.path+=[ROOT,ROOT+'/RLPF-verl_diffusion',ROOT+'/DiffSBDD']
from rdkit import Chem
from rdkit.Chem import Descriptors
from ddpo.rewards.logp_distribution_reward import LogPDistributionReward,DEFAULT_LOGP_BINS

mdf=pd.read_csv(f'{ROOT}/OUT/7gpu-dock-ecfp4-fcfp4_20260526_115341/epochs/epoch_0016/molecules.csv')
valid=mdf[mdf['rdkit_valid']==1]
rw=LogPDistributionReward(chembl_csv='./data/chembl/chembl_36_properties.csv.gz',
    failure_penalty=-1.0,min_reward=-2.0,max_reward=2.0,scale=6.0,ema_alpha=0.0)
ref=rw.chmbl_proportions;bins=DEFAULT_LOGP_BINS;n=len(bins)-1
res=[]
for rank in sorted(valid['rank'].unique()):
    rdf_v=mdf[(mdf['rank']==rank)&(mdf['rdkit_valid']==1)]
    logps=[]
    for smi in rdf_v['smiles']:
        m=Chem.MolFromSmiles(smi)
        logps.append(Descriptors.MolLogP(m) if m else np.nan)
    logps=np.array([l for l in logps if not np.isnan(l)])
    ct=np.zeros(n)
    for lp in logps:
        idx=np.digitize(lp,bins)-1
        if 0<=idx<n:ct[idx]+=1
    pr=(ct+1.0)/(ct.sum()+n);gaps=ref-pr
    rews=np.clip(6.0*gaps,-2,2)
    mr=[]; [mr.append(float(rews[np.digitize(lp,bins)-1])) for lp in logps]
    n_inv=90-len(rdf_v);mr.extend([-1]*n_inv)
    res.append(np.mean(mr))
    print(f'R{int(rank)}: v={len(logps)} avgLP={np.mean(logps):.2f} avgR={np.mean(mr):.4f}')

print(f'\nPer-rank mean: {np.mean(res):.4f} | CSV: 0.189')
print('Epoch 1 check:')
mdf1=pd.read_csv(f'{ROOT}/OUT/7gpu-dock-ecfp4-fcfp4_20260526_115341/epochs/epoch_0001/molecules.csv')
res1=[]
for rank in sorted(mdf1[mdf1['rdkit_valid']==1]['rank'].unique()):
    rv=mdf1[(mdf1['rank']==rank)&(mdf1['rdkit_valid']==1)]
    lps=[]
    for s in rv['smiles']:
        m=Chem.MolFromSmiles(s);lps.append(Descriptors.MolLogP(m) if m else np.nan)
    lps=np.array([l for l in lps if not np.isnan(l)])
    ct=np.zeros(n)
    for lp in lps:
        idx=np.digitize(lp,bins)-1
        if 0<=idx<n:ct[idx]+=1
    pr=(ct+1.0)/(ct.sum()+n);gaps=ref-pr;rews=np.clip(6*gaps,-2,2)
    mr=[];[mr.append(float(rews[np.digitize(lp,bins)-1])) for lp in lps]
    mr.extend([-1]*(90-len(rv)))
    res1.append(np.mean(mr))
print(f'Epoch 1 per-rank mean: {np.mean(res1):.4f} | CSV: -1.044')
