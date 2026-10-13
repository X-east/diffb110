"""Evaluate a finished DiffGui sample pool without loading the DiffGui model."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import logging
import math
from pathlib import Path
import re
import subprocess
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, Crippen, Descriptors, QED, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Contrib.SA_Score import sascorer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.offline_rl.structure_pathology_rules import compute_structure_pathology

LOGGER = logging.getLogger(__name__)


def write_csv(path: Path, rows: list[dict]) -> None:
    """Save homogeneous per-record evidence."""
    keys=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,keys);writer.writeheader();writer.writerows(rows)


def features(spec: dict) -> dict:
    """Check the saved 3D SDF and extract project-standard structure facts."""
    RDLogger.DisableLog('rdApp.*')
    row={**spec,'readable':False,'sanitize':False}
    try:
        mol=Chem.SDMolSupplier(spec['source_sdf'],sanitize=False,removeHs=False)[0]
        if mol is None:return row
        row['readable']=True
        row['fragments']=len(Chem.GetMolFrags(mol))
        Chem.SanitizeMol(mol)
        mol=Chem.RemoveHs(mol)
        facts=compute_structure_pathology(mol)
        row.update(sanitize=True,smiles=Chem.MolToSmiles(mol),ha=mol.GetNumHeavyAtoms(),
                   single_fragment=row['fragments']==1,clean=not facts.is_pathological,
                   qed=QED.qed(mol),sa=sascorer.calculateScore(mol),mw=Descriptors.MolWt(mol),
                   logp=Crippen.MolLogP(mol),tpsa=rdMolDescriptors.CalcTPSA(mol),
                   hetero=rdMolDescriptors.CalcNumHeteroatoms(mol),carbon_fraction=facts.carbon_fraction,
                   element_diversity=len(set(a.GetSymbol() for a in mol.GetAtoms())),
                   rings=mol.GetRingInfo().NumRings(),aromatic_rings=rdMolDescriptors.CalcNumAromaticRings(mol),
                   three_ring=any(len(r)==3 for r in mol.GetRingInfo().AtomRings()),
                   spiro=rdMolDescriptors.CalcNumSpiroAtoms(mol)>0,
                   bridge=rdMolDescriptors.CalcNumBridgeheadAtoms(mol)>0,
                   r1=facts.r1_large_ring_violation,r2=facts.r2_ring_system_violation,
                   r3=facts.r3_four_ring_violation,r4=facts.r4_any_three_ring_violation,
                   r5=facts.r5_carbon_fraction_violation,
                   scaffold=MurckoScaffold.MurckoScaffoldSmiles(mol=mol))
    except Exception as exc:
        row['error']=f'{type(exc).__name__}: {exc}'
    return row


def pb_check(spec: dict) -> dict:
    """Reuse the formal benchmark's dock-config PB validity definition on raw poses."""
    import posebusters.posebusters as pbmodule
    from posebusters import PoseBusters
    RDLogger.DisableLog('rdApp.*')
    logging.getLogger('posebusters').setLevel(logging.ERROR)
    original=pbmodule.PoseBusters._get_name
    try:
        pbmodule.PoseBusters._get_name=lambda self,mol: mol.GetProp('_Name') if mol.HasProp('_Name') else 'mol'
        result=PoseBusters(config='dock').bust(mol_pred=spec['source_sdf'],mol_cond=spec['pocket'])
        if result.empty:return {'record_id':spec['record_id'],'pb_valid':False,'pb_error':'empty'}
        row=result.iloc[0].to_dict()
        checks={k:bool(v) for k,v in row.items() if isinstance(v,(bool,np.bool_))}
        return {'record_id':spec['record_id'],'pb_valid':bool(checks) and all(checks.values()),'pb_error':'',**checks}
    except Exception as exc:
        return {'record_id':spec['record_id'],'pb_valid':False,'pb_error':f'{type(exc).__name__}: {exc}'}
    finally:
        pbmodule.PoseBusters._get_name=original


def main() -> None:
    """Write structural evidence, PB checks and optional independent docking."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--pocket',type=Path,required=True)
    parser.add_argument('--receptor',type=Path,required=True)
    parser.add_argument('--num-workers',type=int,default=18)
    parser.add_argument('--dock',action='store_true')
    parser.add_argument('--gpu-id',type=int,default=0)
    parser.add_argument('--unidock-bin',type=Path,default=Path('/opt/unidock-1.1.3/bin/unidock'))
    args=parser.parse_args();out=args.out_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(name)s %(message)s',
                        handlers=[logging.StreamHandler(),logging.FileHandler(out/'evaluation.log')])
    logging.getLogger('fontTools').setLevel(logging.WARNING)
    started=time.time();run=args.run_dir.resolve()
    (out/'status.json').write_text(json.dumps({'status':'running','stage':'structure','run_dir':str(run)}))
    manifests=list(csv.DictReader((run/'run_manifest.csv').open()))
    attempts=sum(int(r['count']) for r in manifests);specs=[];shards=[]
    for entry in manifests:
        shard=int(entry['gpu_id']);files=sorted((run/f'outputs/diffgui/gpu_{shard}').glob('*/home_SDF/*.sdf'),key=lambda p:int(p.stem))
        log=(run/f'logs/diffgui/gpu_{shard}.log').read_text(errors='replace')
        failed=log.count('Reconstruction error encountered.');assert 'Using DiffGui sampling timesteps override: 500' in log
        shards.append({'shard':shard,'attempts':int(entry['count']),'written':len(files),'reconstruction_failures':failed,
                       'other_not_written':int(entry['count'])-len(files)-failed,'seed':int(entry['seed'])})
        source_smiles=next((run/f'outputs/diffgui/gpu_{shard}').glob('*/SMILES.txt')).read_text().splitlines()
        for file in files:specs.append({'record_id':f'g{shard}_{file.stem}','shard':shard,'sample_index':int(file.stem),'source_sdf':str(file),'saved_smiles':source_smiles[int(file.stem)]})
    write_csv(out/'shard_counts.csv',shards)
    LOGGER.info('source attempts=%s written=%s shards=%s',attempts,len(specs),shards)
    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:rows=list(pool.map(features,specs,chunksize=16))
    write_csv(out/'molecule_metrics.csv',rows)
    valid=[r for r in rows if r['sanitize']];clean=[r for r in valid if r['clean'] and r['single_fragment']]
    summary={'attempts':attempts,'written':len(specs),'sanitize_n':len(valid),
             'unique_smiles':len(set(r['saved_smiles'] for r in valid)),
             'sdf_unique_smiles':len(set(r['smiles'] for r in valid)),
             'single_fragment_n':sum(r['single_fragment'] for r in valid),
             'single_clean_n':len(clean),'single_clean_unique':len(set(r['smiles'] for r in clean)),
             'clean_scaffolds':len(set(r['scaffold'] for r in clean)),
             'reconstruction_failures':sum(r['reconstruction_failures'] for r in shards),
             'other_not_written':sum(r['other_not_written'] for r in shards),'shards':shards}
    summary['descriptor_means']={key:float(np.mean([r[key] for r in valid])) for key in ['ha','qed','sa','logp','tpsa','hetero','carbon_fraction','element_diversity','rings','aromatic_rings']}
    summary['rule_counts']={key:sum(r[key] for r in valid) for key in ['r1','r2','r3','r4','r5','three_ring','spiro','bridge']}
    # Exact pairwise Morgan similarity across unique saved structures, no subsampling.
    unique=list(dict.fromkeys(r['smiles'] for r in valid));fingerprints=[AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(sm),2,nBits=2048) for sm in unique]
    pair_sum=0.;pair_count=0
    for index,fp in enumerate(fingerprints[:-1]):
        values=DataStructs.BulkTanimotoSimilarity(fp,fingerprints[index+1:]);pair_sum+=sum(values);pair_count+=len(values)
    summary['morgan_radius']=2;summary['morgan_bits']=2048;summary['pair_count']=pair_count;summary['mean_pairwise_tanimoto']=pair_sum/pair_count
    writer=Chem.SDWriter(str(out/'all_saved_molecules.sdf'));clean_writer=Chem.SDWriter(str(out/'single_fragment_rule_compliant.sdf'))
    for row in valid:
        mol=Chem.SDMolSupplier(row['source_sdf'],removeHs=False)[0];mol.SetProp('_Name',row['record_id']);mol.SetProp('source_sdf',row['source_sdf'])
        mol.SetProp('record_id',row['record_id']);writer.write(mol)
        if row['clean'] and row['single_fragment']:clean_writer.write(mol)
    writer.close();clean_writer.close()
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    LOGGER.info('structure complete %s',summary)
    summary['posebusters']='not_run_user_requested'
    dockrows=[]
    if args.dock:
        (out/'status.json').write_text(json.dumps({'status':'running','stage':'independent_docking'}))
        command=[sys.executable,str(ROOT/'scripts/docking/dock_sdf_unidock.py'),'--model','DiffGui_epoch80','--input-sdf',str(out/'all_saved_molecules.sdf'),'--receptor',str(args.receptor.resolve()),'--out-dir',str(out/'docking'),'--unidock-bin',str(args.unidock_bin),'--gpu-id',str(args.gpu_id),'--center-x','207.293','--center-y','212.430','--center-z','172.771','--size-x','20','--size-y','20','--size-z','20','--search-mode','fast','--seed','42','--num-modes','1','--ligand-frame','receptor_frame','--max-gpu-memory','4096','--keep-split']
        # The canonical coordinate-frame identifier is imported from existing helpers.
        from ddpo.utils.docking_protocol import FRAME_RECEPTOR
        command[command.index('--ligand-frame')+1]=FRAME_RECEPTOR
        (out/'docking_command.json').write_text(json.dumps(command,ensure_ascii=False,indent=2))
        with (out/'docking.log').open('w') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        summary['docking_exit_code']=result.returncode
        if (out/'docking/docking_results.csv').exists():
            dockrows=list(csv.DictReader((out/'docking/docking_results.csv').open()))
            scored=[r for r in dockrows if r['score_status']=='ok' and r['score'] and math.isfinite(float(r['score']))]
            summary['docking_status_counts']=dict(Counter(r['score_status'] for r in dockrows))
            summary['docking_ok_n']=len(scored);summary['docking_ok_median']=float(np.median([float(r['score']) for r in scored])) if scored else None
            for dockrow in dockrows:
                index=int(dockrow['benchmark_index']);source=valid[index]
                dockrow.update(record_id=source['record_id'],r1r5_clean=source['clean'],qed=source['qed'],sa=source['sa'],source_sdf=source['source_sdf'])
            write_csv(out/'independent_docking_joined.csv',dockrows)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    plt.rcParams.update({'font.family':'Arial','font.size':15,'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(13,4.7))
    counts=[len(valid),summary['unique_smiles'],len(clean),summary['single_clean_unique']]
    axes[0].bar(['Saved valid','Unique SMILES','Rule clean','Unique clean'],np.array(counts)/attempts*100,color=['#A7B3BF','#A7B3BF','#044875','#2D8B8C'])
    axes[0].set_ylabel('Yield / 2000 attempts (%)');axes[0].tick_params(axis='x',labelrotation=20)
    axes[1].hist([r['qed'] for r in valid],bins=25,color='#A7B3BF',label='Saved valid')
    axes[1].hist([r['qed'] for r in clean],bins=25,histtype='step',lw=2,color='#044875',label='Rule clean')
    axes[1].set_xlabel('QED');axes[1].set_ylabel('Molecules');axes[1].legend();fig.tight_layout()
    fig.savefig(out/'candidate_quality_summary.png',dpi=300);fig.savefig(out/'candidate_quality_summary.svg');plt.close(fig)
    summary['elapsed_seconds']=time.time()-started
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    final_status='completed' if not args.dock or summary.get('docking_exit_code')==0 else 'partial_docking_failed'
    (out/'status.json').write_text(json.dumps({'status':final_status,'elapsed_seconds':summary['elapsed_seconds']}))
    LOGGER.info('evaluation status=%s summary=%s',final_status,summary)


if __name__=='__main__':main()
