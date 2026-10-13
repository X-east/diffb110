"""Render the requested DiffGui early-stage reference in the model panel."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read_rows(file: Path) -> list[dict]:
    """Read existing records without changing source results."""
    with file.open(encoding='utf-8-sig') as handle:return list(csv.DictReader(handle))


def main() -> None:
    """Plot actual distributions and store their provenance separately."""
    root=Path(__file__).resolve().parents[2];out=Path(__file__).parent
    run=root/'runs/diffgui-deppa-full-lr1e5-wd1e4-b8x8-t200-100ep_20260626_165809/epochs'
    files=sorted(run.glob('epoch_*/molecules.csv'))[:20]
    assert len(files)==20
    early=[]
    for file in files:
        early.extend(float(r['docking_raw']) for r in read_rows(file) if r['docking_success']=='1' and float(r['docking_raw'])<0)
    audit=read_rows(root/'analysis/midterm_protocol_audit_261009/rl_early_ep1_20.csv')
    reference=statistics.mean(float(r['mean']) for r in audit)
    assert round(reference,2)==-9.89
    names=['DiffGui','DiffSBDD','GCDM-SBDD','PMDM','TargetDiff']
    slug=['diffgui','diffsbdd','gcdm_sbdd','pmdm','targetdiff']
    all_data=[early];sources=[{'model':'DiffGui','n':len(early),'reference_mean':reference,'pooled_mean':statistics.mean(early),'source_files':[str(f) for f in files],'definition':'Training epochs1–20; mean of per-epoch means for marker'}]
    for model,model_slug in zip(names[1:],slug[1:]):
        data=[];paths=[]
        for repeat in ['rep4','rep5','rep6']:
            file=root/f'analysis/benchmark/benchmark5/five_model_500steps_full_eval_260903/docking/{repeat}/{model_slug}/docking_results.csv'
            paths.append(str(file))
            data.extend(float(r['score']) for r in read_rows(file) if r['success'].lower() in ['true','1'] and r['score'] and float(r['score'])<=0)
        all_data.append(data);sources.append({'model':model,'n':len(data),'reference_mean':statistics.mean(data),'source_files':paths,'definition':'Stored finite successful nonpositive docking scores'})
    plt.rcParams.update({'font.family':'Arial','font.size':16,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(9,4.5));fig.subplots_adjust(left=.12,right=.98,bottom=.18,top=.94)
    for i,(values,entry) in enumerate(zip(all_data,sources)):
        color='#044875' if i==0 else '#A7B3BF'
        violin=ax.violinplot(values,positions=[i],widths=.70,showextrema=False,bw_method=.25)
        for body in violin['bodies']:body.set_facecolor(color);body.set_edgecolor(color);body.set_alpha(.8)
        ax.scatter(i,entry['reference_mean'],s=32,c='white',edgecolors=color,zorder=4)
    ax.set_xticks(range(5),names,fontsize=14);ax.set_ylabel('Docking score (kcal/mol)',fontsize=18)
    ax.grid(axis='y',alpha=.18);ax.spines[['top','right']].set_visible(False)
    for ext in ['png','svg','pdf']:fig.savefig(out/f'model_comparison.{ext}',dpi=300)
    (out/'figure_source.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2))
    print('Reference means:',[(r['model'],r['reference_mean']) for r in sources])


if __name__=='__main__':main()
