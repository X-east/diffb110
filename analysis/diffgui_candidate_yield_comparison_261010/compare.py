"""Compare fixed-attempt candidate yields with frozen screening gates."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import logging
from pathlib import Path
import subprocess
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, QED, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Contrib.SA_Score import sascorer
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.offline_rl.structure_pathology_rules import compute_structure_pathology
from ddpo.utils.docking_protocol import FRAME_RECEPTOR

LOGGER=logging.getLogger(__name__)
BASE=ROOT/'analysis/benchmark/benchmark5/five_model_repeats_500steps_260826'
RL=ROOT/'analysis/diffgui_epoch80_independent_eval_261010'
REPEATS=['rep4_seed22_29','rep5_seed30_37','rep6_seed38_45']
GATES={'qed_min':.5,'sa_max':5.5,'heavy_atoms_min':10,'heavy_atoms_max':30,'ring_count_min':1,
       'require_single_fragment':True,'require_r1r5_clean':True,'dock_thresholds':[-8,-9,-10,-11,-12],
       'primary_baseline_repeats':REPEATS[:2],
       'basis':'Existing representative selection in 中期/scripts/prepare_diffgui_independent_slides.py; docking grid is exploratory sensitivity, not activity labels.'}


def read_csv(file: Path) -> list[dict]:
    """Read archived source records."""
    with file.open(encoding='utf-8-sig',newline='') as handle:return list(csv.DictReader(handle))


def write_csv(file: Path, rows: list[dict]) -> None:
    """Persist evidence and results."""
    keys=list(dict.fromkeys(key for row in rows for key in row))
    with file.open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,keys);writer.writeheader();writer.writerows(rows)


def baseline_features(spec: dict) -> dict:
    """Recompute identical screen features using each saved raw SDF record."""
    RDLogger.DisableLog('rdApp.*')
    supplier=Chem.SDMolSupplier(spec['source_sdf'],removeHs=True)
    mol=supplier[spec['sdf_index']]
    assert mol is not None
    facts=compute_structure_pathology(mol)
    return {**spec,'smiles':Chem.MolToSmiles(mol),'single':len(Chem.GetMolFrags(mol))==1,
            'clean':not facts.is_pathological,'qed':QED.qed(mol),'sa':sascorer.calculateScore(mol),
            'ha':mol.GetNumHeavyAtoms(),'rings':mol.GetRingInfo().NumRings(),
            'logp':Crippen.MolLogP(mol),'tpsa':rdMolDescriptors.CalcTPSA(mol),
            'scaffold':MurckoScaffold.MurckoScaffoldSmiles(mol=mol)}


def property_gate(row: dict) -> bool:
    """Apply the frozen screen rather than tune criteria to observed outcomes."""
    return (row['single'] and row['clean'] and row['qed']>=GATES['qed_min'] and row['sa']<=GATES['sa_max']
            and GATES['heavy_atoms_min']<=row['ha']<=GATES['heavy_atoms_max'] and row['rings']>=GATES['ring_count_min'])


def counts(rows: list[dict], group: str, attempts: int, stage: str) -> dict:
    """Count raw candidate yield and nonduplicated graphs/scaffolds."""
    return {'group':group,'attempts':attempts,'stage':stage,'count':len(rows),
            'yield_pct':len(rows)/attempts*100,'unique_structures':len(set(r['smiles'] for r in rows)),
            'scaffolds':len(set(r['scaffold'] for r in rows))}


def main() -> None:
    """Evaluate archived base-model samples and compare with epoch80."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--num-workers',type=int,default=18)
    parser.add_argument('--gpu-id',type=int,default=0)
    parser.add_argument('--receptor',type=Path,required=True)
    args=parser.parse_args();out=args.out_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',handlers=[logging.StreamHandler(),logging.FileHandler(out/'comparison.log')])
    logging.getLogger('fontTools').setLevel(logging.WARNING);RDLogger.DisableLog('rdApp.*')
    started=time.time()
    frozen=out/'frozen_screen.json'
    if frozen.exists():assert json.loads(frozen.read_text())==GATES
    else:frozen.write_text(json.dumps(GATES,ensure_ascii=False,indent=2))
    configs=[]
    for repeat in REPEATS:
        cfg=yaml.safe_load((BASE/repeat/'configs/diffgui/diffgui_c1101_gpu1.yml').read_text())
        configs.append({'group':repeat,'model':cfg['model'],'sample':cfg['sample']})
    rl_run=ROOT/'OUT/diffgui_epoch80_independent_500steps_6gpu_2000attempts_20261009_214739'
    cfg=yaml.safe_load((rl_run/'configs/diffgui/diffgui_c1101_gpu1.yml').read_text())
    configs.append({'group':'epoch80','model':cfg['model'],'sample':cfg['sample']})
    (out/'generation_config_audit.json').write_text(json.dumps(configs,ensure_ascii=False,indent=2))
    specs=[]
    for repeat in REPEATS:
        source=BASE/repeat/'filtered_sanitize_valid/diffgui.sanitize_valid.sdf'
        for index,mol in enumerate(Chem.SDMolSupplier(str(source),removeHs=False)):
            assert mol is not None
            specs.append({'group':repeat,'record_id':f'{repeat}_{index}','source_sdf':str(source),'sdf_index':index})
    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:base=list(pool.map(baseline_features,specs,chunksize=32))
    write_csv(out/'baseline_metrics.csv',base)
    trained=[]
    for row in read_csv(RL/'slide_assets/independent_molecules_joined.csv'):
        trained.append({'group':'epoch80','record_id':row['record_id'],'smiles':row['smiles'],
                        'single':row['single_fragment']=='True','clean':row['clean']=='True',
                        **{key:float(row[key]) for key in ['qed','sa','ha','rings','logp','tpsa']},
                        'scaffold':row['scaffold'],'score':float(row['score']),'score_status':row['score_status']})
    primary=[r for r in base if r['group'] in REPEATS[:2]]
    groups=[('Base first two repeats',primary,2000),('Epoch80',trained,2000),('Base all three repeats',base,3000)]
    stages=[]
    for name,rows,attempts in groups:
        for stage,selected in [('Saved valid',rows),('Single fragment + R1–R5',[r for r in rows if r['single'] and r['clean']]),('Existing property screen',[r for r in rows if property_gate(r)])]:
            stages.append(counts(selected,name,attempts,stage))
    write_csv(out/'structure_property_yields.csv',stages)
    LOGGER.info('Structure/property screening %s',stages)
    aggregate=out/'baseline_primary_raw.sdf'
    writer=Chem.SDWriter(str(aggregate))
    for row in primary:
        mol=Chem.SDMolSupplier(row['source_sdf'],removeHs=False)[row['sdf_index']];mol.SetProp('_Name',row['record_id']);writer.write(mol)
    writer.close()
    (out/'status.json').write_text(json.dumps({'status':'running','stage':'baseline_same_protocol_docking'}))
    docking_file=out/'docking/docking_results.csv'
    if not docking_file.exists():
        cmd=[sys.executable,str(ROOT/'scripts/docking/dock_sdf_unidock.py'),'--model','DiffGui_base_matched_eval','--input-sdf',str(aggregate),'--receptor',str(args.receptor.resolve()),'--out-dir',str(out/'docking'),'--gpu-id',str(args.gpu_id),'--center-x','207.293','--center-y','212.430','--center-z','172.771','--size-x','20','--size-y','20','--size-z','20','--search-mode','fast','--seed','42','--num-modes','1','--ligand-frame',FRAME_RECEPTOR,'--max-gpu-memory','4096','--keep-split']
        (out/'docking_command.json').write_text(json.dumps(cmd,ensure_ascii=False,indent=2))
        with (out/'docking.log').open('w') as log:result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        assert result.returncode==0,'Baseline docking failed; inspect docking.log'
    dock_rows=read_csv(docking_file)
    assert len(dock_rows)==len(primary)
    for row in dock_rows:
        record=primary[int(row['benchmark_index'])]
        record.update(score=float(row['score']) if row['score'] else None,score_status=row['score_status'])
    protocols=json.loads((out/'docking/run_summary.json').read_text())['docking_protocol']
    trained_protocol=json.loads((RL/'docking/run_summary.json').read_text())['docking_protocol']
    assert protocols==trained_protocol,(protocols,trained_protocol)
    (out/'matched_docking_protocol.json').write_text(json.dumps(protocols,indent=2))
    write_csv(out/'baseline_same_protocol_docking_joined.csv',primary)
    results=[];candidate_rows=[];diagnostics=[]
    for name,rows in [('Base first two repeats',primary),('Epoch80',trained)]:
        screened=[r for r in rows if property_gate(r)]
        for stage,subset in [('All saved',rows),('Single + R1–R5',[r for r in rows if r['single'] and r['clean']]),('Property screen',screened)]:
            scores=[r['score'] for r in subset if r['score'] is not None and r['score_status']=='ok']
            diagnostics.append({'group':name,'stage':stage,'count':len(subset),'scored':len(scores),
                                'mean':sum(scores)/len(scores) if scores else None,
                                'le_minus10':sum(v<=-10 for v in scores),
                                'conditional_le_minus10_pct':sum(v<=-10 for v in scores)/len(scores)*100 if scores else None})
        for threshold in GATES['dock_thresholds']:
            selected=[r for r in screened if r['score'] is not None and r['score_status']=='ok' and r['score']<=threshold]
            results.append({**counts(selected,name,2000,f'Docking <= {threshold}'),'dock_threshold':threshold,
                            'property_screen_count':len(screened),'conditional_pass_pct':len(selected)/len(screened)*100})
            candidate_rows.extend({**r,'screen_group':name,'dock_threshold':threshold} for r in selected)
    (out/'docking_stage_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    write_csv(out/'candidate_yield_by_threshold.csv',results);write_csv(out/'candidate_records_by_threshold.csv',candidate_rows)
    plt.rcParams.update({'font.family':'Arial','font.size':15,'svg.fonttype':'none','pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(9,5));fig.subplots_adjust(left=.14,right=.97,bottom=.19,top=.84)
    for name,color in [('Base first two repeats','#A7B3BF'),('Epoch80','#044875')]:
        part=[r for r in results if r['group']==name]
        ax.plot([-r['dock_threshold'] for r in part],[r['yield_pct'] for r in part],'o-',lw=2.4,color=color,label=name)
    ax.set_xticks([8,9,10,11,12],['−8','−9','−10','−11','−12']);ax.set_xlabel('Docking threshold (kcal/mol)')
    ax.set_ylabel('Screened candidates / 2000 attempts (%)');ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.18)
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.02),ncol=2,fontsize=13)
    for ext in ['png','svg','pdf']:fig.savefig(out/f'candidate_yield_threshold_sensitivity.{ext}',dpi=300)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5));fig.subplots_adjust(left=.14,right=.97,bottom=.19,top=.84)
    for name,color in [('Base first two repeats','#A7B3BF'),('Epoch80','#044875')]:
        part=[r for r in results if r['group']==name]
        ax.plot([-r['dock_threshold'] for r in part],[r['conditional_pass_pct'] for r in part],'o-',lw=2.4,color=color,label=name)
    ax.set_xticks([8,9,10,11,12],['−8','−9','−10','−11','−12']);ax.set_xlabel('Docking threshold (kcal/mol)')
    ax.set_ylabel('Docking pass / property-screened molecules (%)');ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.18)
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.02),ncol=2,fontsize=13)
    for ext in ['png','svg','pdf']:fig.savefig(out/f'conditional_docking_pass_comparison.{ext}',dpi=300)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5));fig.subplots_adjust(left=.12,right=.97,bottom=.22,top=.84)
    x=np.arange(3)
    for offset,name,color in [(-.18,'Base first two repeats','#A7B3BF'),(.18,'Epoch80','#044875')]:
        part=[r for r in stages if r['group']==name]
        ax.bar(x+offset,[r['yield_pct'] for r in part],width=.35,color=color,label=name)
    ax.set_xticks(x,['Saved valid','Single + R1–R5 clean','Property screen'],fontsize=13)
    ax.set_ylabel('Yield / 2000 attempts (%)');ax.legend(loc='lower left',bbox_to_anchor=(0,1.02),ncol=2,fontsize=13)
    for ext in ['png','svg','pdf']:fig.savefig(out/f'screening_stage_comparison.{ext}',dpi=300)
    plt.close(fig)
    summary={'frozen_screen':GATES,'stages':stages,'candidate_thresholds':results,
             'docking_diagnostics':diagnostics,'docking_protocol_id':protocols['docking_protocol_id'],'elapsed_seconds':time.time()-started,
             'limits':'Matched evaluation and attempt budget; historical generation seeds, shard sizes and single RL run differ. These are computational candidate yields, not confirmed activity.'}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    (out/'status.json').write_text(json.dumps({'status':'completed','elapsed_seconds':summary['elapsed_seconds']}))
    LOGGER.info('Completed candidates %s',results)


if __name__=='__main__':main()
