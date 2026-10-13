"""Apply literature-inspired physicochemical filters and the user-selected docking cutoff."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import logging
import math
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Contrib.SA_Score import sascorer

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.offline_rl.structure_pathology_rules import compute_structure_pathology

LOGGER=logging.getLogger(__name__)
OLD=ROOT/'analysis/diffgui_candidate_yield_comparison_261010'
RL=ROOT/'analysis/diffgui_epoch80_independent_eval_261010'
THRESHOLDS=[-7,-8,-9,-10,-11,-12]
RULES={'mw_max_exclusive':500,'logp_max_inclusive':5,'hbd_max_inclusive':5,'hba_max_inclusive':10,
       'rotatable_bonds_max_inclusive':10,'docking_max_inclusive':-10,
       'require_valid_original_single_fragment':True,
       'qed_hard_gate':None,'sa_hard_gate':None,'ha_hard_gate':None,'ring_hard_gate':None,
       'r1r5':'Separate supplementary stage, not a physicochemical literature criterion',
       'attempts_per_group':2000,'baseline_repeats':['rep4_seed22_29','rep5_seed30_37'],
       'literature':'Li et al., One Health Advances 2025, DOI10.1186/s44280-025-00102-1, Methods drug-like screening',
       'docking_cutoff_basis':'User explicitly selected -10, stricter than the cited paper -7; not a validated activity threshold',
       'definitions':'RDKit MolWt, Crippen MolLogP, CalcNumHBD/HBA, CalcNumRotatableBonds default; recomputed for both groups'}


def read_csv(path: Path) -> list[dict]:
    """Read frozen input records."""
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def write_csv(path: Path,rows: list[dict]) -> None:
    """Save detailed evidence with stable columns."""
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(rows)


def describe(row: dict) -> dict:
    """Recompute the same RDKit features and verify stored structure labels."""
    RDLogger.DisableLog('rdApp.*')
    mol=Chem.MolFromSmiles(row['smiles'])
    if mol is None:return {**row,'valid':False,'single':False,'physchem_pass':False,'clean':False}
    single=len(Chem.GetMolFrags(mol))==1
    facts=compute_structure_pathology(mol)
    clean=not facts.is_pathological
    assert clean==(str(row['clean']).lower()=='true'),row['record_id']
    hba=rdMolDescriptors.CalcNumHBA(mol);hbd=rdMolDescriptors.CalcNumHBD(mol)
    mw=Descriptors.MolWt(mol);logp=Crippen.MolLogP(mol)
    rot=rdMolDescriptors.CalcNumRotatableBonds(mol)
    flags={'mw_pass':mw<RULES['mw_max_exclusive'],'logp_pass':logp<=RULES['logp_max_inclusive'],
           'hbd_pass':hbd<=RULES['hbd_max_inclusive'],'hba_pass':hba<=RULES['hba_max_inclusive'],
           'rotatable_pass':rot<=RULES['rotatable_bonds_max_inclusive']}
    score=float(row['score']) if row['score'] else None
    score_ok=score is not None and math.isfinite(score) and row['score_status']=='ok'
    return {**row,'valid':True,'single':single,'clean':clean,'smiles':Chem.MolToSmiles(mol),
            'smiles_no_stereo':Chem.MolToSmiles(mol,isomericSmiles=False),
            'scaffold':MurckoScaffold.MurckoScaffoldSmiles(mol=mol),'mw':mw,'logp':logp,
            'qed':QED.qed(mol),'sa':sascorer.calculateScore(mol),'ha':mol.GetNumHeavyAtoms(),
            'rings':mol.GetRingInfo().NumRings(),'tpsa':rdMolDescriptors.CalcTPSA(mol),
            'hba':hba,'hbd':hbd,'rotatable_bonds':rot,**flags,
            'physchem_pass':all(flags.values()),'score':score,'score_ok':score_ok,
            'dock10_pass':score_ok and score<=RULES['docking_max_inclusive']}


def count(rows: list[dict],group: str,stage: str) -> dict:
    """Count yields and distinguish nonempty Murcko scaffolds from acyclic molecules."""
    return {'group':group,'stage':stage,'attempts':2000,'count':len(rows),'yield_pct':len(rows)/20,
            'unique_isomeric_smiles':len(set(r['smiles'] for r in rows)),
            'unique_connectivities':len(set(r['smiles_no_stereo'] for r in rows)),
            'nonempty_scaffolds':len(set(r['scaffold'] for r in rows if r['scaffold'])),
            'acyclic_count':sum(not r['scaffold'] for r in rows),
            'acyclic_unique_structures':len(set(r['smiles'] for r in rows if not r['scaffold']))}


def final_quality(rows: list[dict],group: str) -> list[dict]:
    """Describe the final supplementary candidates without asserting uniform improvement."""
    result=[]
    for prop in ['score','qed','sa','mw','logp','tpsa','ha','rings','hba','hbd','rotatable_bonds']:
        values=[float(row[prop]) for row in rows]
        result.append({'group':group,'property':prop,'n':len(values),
                       'mean':float(np.mean(values)) if values else None,
                       'median':float(np.median(values)) if values else None})
    return result


def main() -> None:
    """Evaluate saved molecules, export stage comparisons, and record completion."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=18)
    args=parser.parse_args();out=Path(__file__).parent;started=time.time()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(),logging.FileHandler(out/'evaluation.log')])
    logging.getLogger('fontTools').setLevel(logging.WARNING)
    (out/'status.json').write_text(json.dumps({'status':'running'}))
    frozen=out/'frozen_rules.json'
    if frozen.exists():assert json.loads(frozen.read_text())==RULES
    else:frozen.write_text(json.dumps(RULES,ensure_ascii=False,indent=2))
    base_protocol=json.loads((OLD/'docking/run_summary.json').read_text())['docking_protocol']
    rl_protocol=json.loads((RL/'docking/run_summary.json').read_text())['docking_protocol']
    assert base_protocol==rl_protocol
    (out/'matched_docking_protocol.json').write_text(json.dumps(base_protocol,ensure_ascii=False,indent=2))
    specs=[{**r,'comparison_group':'Base'} for r in read_csv(OLD/'baseline_same_protocol_docking_joined.csv')]
    specs.extend({**r,'comparison_group':'Epoch80'} for r in read_csv(RL/'slide_assets/independent_molecules_joined.csv'))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:rows=list(pool.map(describe,specs,chunksize=32))
    assert len(rows)==1734+1858 and all(r['valid'] and r['single'] for r in rows)
    write_csv(out/'molecule_evaluation.csv',rows)
    stages=[];failure_rows=[];sensitivity=[];properties=[];retained={}
    feature_names=['qed','sa','mw','logp','tpsa','ha','rings','hba','hbd','rotatable_bonds']
    for group in ['Base','Epoch80']:
        saved=[r for r in rows if r['comparison_group']==group and r['valid'] and r['single']]
        phys=[r for r in saved if r['physchem_pass']]
        primary=[r for r in phys if r['dock10_pass']]
        supplementary=[r for r in primary if r['clean']]
        retained[group]={'primary':primary,'with_r1r5':supplementary}
        for name,selected in [('Saved valid single',saved),('Literature physicochemical',phys),
                              ('Physicochemical + docking <= -10',primary),('Above + R1-R5 clean',supplementary)]:
            stages.append(count(selected,group,name))
            for prop in feature_names:
                vals=[r[prop] for r in selected]
                properties.append({'group':group,'stage':name,'property':prop,'n':len(vals),
                                   'mean':float(np.mean(vals)) if vals else None,'median':float(np.median(vals)) if vals else None})
        for key in ['mw','logp','hbd','hba','rotatable']:
            failure_rows.append({'group':group,'criterion':key,'fail_count':sum(not r[key+'_pass'] for r in saved),
                                 'denominator':len(saved),'fail_pct':sum(not r[key+'_pass'] for r in saved)/len(saved)*100})
        for threshold in THRESHOLDS:
            for with_rules in [False,True]:
                selected=[r for r in phys if r['score_ok'] and r['score']<=threshold and (r['clean'] or not with_rules)]
                sensitivity.append({**count(selected,group,'With R1-R5' if with_rules else 'Physchem only'),
                                    'threshold':threshold,'with_r1r5':with_rules,'physchem_denominator':len(phys),
                                    'conditional_pass_pct':len(selected)/len(phys)*100})
    write_csv(out/'screening_stages.csv',stages);write_csv(out/'physchem_failures.csv',failure_rows)
    write_csv(out/'threshold_sensitivity.csv',sensitivity);write_csv(out/'property_summary.csv',properties)
    write_csv(out/'final_candidate_quality.csv',[r for group in retained
              for r in final_quality(retained[group]['with_r1r5'],group)])
    RDLogger.DisableLog('rdApp.*')
    suppliers={'Base':OLD/'baseline_primary_raw.sdf','Epoch80':RL/'all_saved_molecules.sdf'}
    for group,source in suppliers.items():
        mols={m.GetProp('_Name'):m for m in Chem.SDMolSupplier(str(source),removeHs=False) if m is not None}
        for stage,selected in retained[group].items():
            write_csv(out/f'{group.lower()}_{stage}_candidates.csv',selected)
            writer=Chem.SDWriter(str(out/f'{group.lower()}_{stage}_candidates.sdf'))
            for row in selected:
                assert row['record_id'] in mols,(row['record_id'],list(mols)[:3])
                molecule=Chem.Mol(mols[row['record_id']])
                for key in ['score','qed','sa','mw','logp','hba','hbd','rotatable_bonds','clean']:
                    molecule.SetProp(key,str(row[key]))
                writer.write(molecule)
            writer.close()
    plt.rcParams.update({'font.family':'Arial','font.size':16,'pdf.fonttype':42,'svg.fonttype':'none'})
    colors={'Base':'#A7B3BF','Epoch80':'#044875'}
    fig,ax=plt.subplots(figsize=(10,5.4));fig.subplots_adjust(left=.12,right=.98,bottom=.18,top=.82)
    x=np.arange(4)
    for offset,group in [(-.18,'Base'),(.18,'Epoch80')]:
        data=[r for r in stages if r['group']==group]
        bars=ax.bar(x+offset,[r['yield_pct'] for r in data],width=.35,color=colors[group],label=group)
        ax.bar_label(bars,labels=[str(r['count']) for r in data],padding=4,fontsize=15)
    ax.set_xticks(x,['Valid single','Physchem pass','+ Docking ≤ −10','+ R1–R5'],fontsize=14)
    ax.set_ylabel('Yield / 2000 attempts (%)');ax.set_ylim(0,105);ax.grid(axis='y',alpha=.15)
    ax.set_axisbelow(True);ax.legend(loc='lower left',bbox_to_anchor=(0,1.02),ncol=2,frameon=False)
    for ext in ['png','svg','pdf']:fig.savefig(out/f'screening_stage_comparison.{ext}',dpi=300)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5.4));fig.subplots_adjust(left=.12,right=.98,bottom=.18,top=.75)
    for group in ['Base','Epoch80']:
        for with_rules in [False,True]:
            data=[r for r in sensitivity if r['group']==group and r['with_r1r5']==with_rules]
            ax.plot([-r['threshold'] for r in data],[r['yield_pct'] for r in data],
                    'o--' if with_rules else 'o-',lw=2.2,color=colors[group],label=group+(' + R1–R5' if with_rules else ' physchem'))
    ax.set_xticks([7,8,9,10,11,12],['−7','−8','−9','−10','−11','−12'])
    ax.set_xlabel('Docking cutoff (kcal/mol)');ax.set_ylabel('Candidate yield / 2000 attempts (%)')
    ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.15)
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.02),ncol=2,fontsize=14,frameon=False)
    for ext in ['png','svg','pdf']:fig.savefig(out/f'docking_threshold_sensitivity.{ext}',dpi=300)
    plt.close(fig)
    for with_rules,name in [(False,'docking_threshold_physchem'),(True,'docking_threshold_with_r1r5')]:
        fig,ax=plt.subplots(figsize=(9.5,5.4));fig.subplots_adjust(left=.13,right=.98,bottom=.19,top=.82)
        for group in ['Base','Epoch80']:
            data=[r for r in sensitivity if r['group']==group and r['with_r1r5']==with_rules]
            ax.plot([-r['threshold'] for r in data],[r['yield_pct'] for r in data],'o-',lw=2.3,color=colors[group],label=group)
        ax.set_xticks([7,8,9,10,11,12],['−7','−8','−9','−10','−11','−12'])
        ax.set_xlabel('Docking cutoff (kcal/mol)');ax.set_ylabel('Candidate yield / 2000 attempts (%)')
        ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.15)
        ax.legend(loc='lower left',bbox_to_anchor=(0,1.02),ncol=2,frameon=False)
        for ext in ['png','svg','pdf']:fig.savefig(out/f'{name}.{ext}',dpi=300)
        plt.close(fig)
    summary={'rules':RULES,'stages':stages,'thresholds':sensitivity,'failures':failure_rows,
             'elapsed_seconds':time.time()-started,'workers':args.workers,'rdkit_version':Chem.rdBase.rdkitVersion,
             'protocol_id':base_protocol['docking_protocol_id'],
             'limits':'Historical generation seeds/shards differ; single RL run; no activity confirmation; no QED/SA/HA/ring hard gates; nonempty Murcko scaffolds exclude acyclic graphs.'}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    (out/'status.json').write_text(json.dumps({'status':'completed','elapsed_seconds':summary['elapsed_seconds']}))
    LOGGER.info('Completed stage counts: %s',stages)
    print(json.dumps(stages,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
