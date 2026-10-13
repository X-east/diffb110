"""Export the matched DiffGui comparison violin from audited records."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


def main() -> None:
    """Prepare display statistics and a plot without any model execution."""
    root=Path(__file__).resolve().parents[2]
    out=Path(__file__).parent/'assets'
    sources=[root/'analysis/diffgui_candidate_yield_comparison_261010/baseline_same_protocol_docking_joined.csv',
             root/'analysis/diffgui_epoch80_independent_eval_261010/slide_assets/independent_molecules_joined.csv']
    groups=[]
    for source in sources:
        with source.open(encoding='utf-8-sig') as f:groups.append(list(csv.DictReader(f)))
    plt.rcParams.update({'font.family':'Arial','font.size':18,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(9,4.0))
    fig.subplots_adjust(left=.13,right=.99,bottom=.20,top=.77)
    stats=[]
    for j,(rows,color) in enumerate(zip(groups,['#A7B3BF','#044875'])):
        clean=[r for r in rows if r['clean']=='True']
        entry={'group':['Base','Epoch80'][j],'n':len(rows),'clean_n':len(clean)}
        for key in ['qed','sa','score']:
            entry[key+'_mean']=float(np.mean([float(r[key]) for r in rows]))
            entry['clean_'+key+'_mean']=float(np.mean([float(r[key]) for r in clean]))
        stats.append(entry)
        for k,subset in enumerate([rows,clean]):
            values=[float(r['score']) for r in subset if r['score_status']=='ok']
            pos=k+(-.20 if j==0 else .20)
            violin=ax.violinplot(values,positions=[pos],widths=.36,showextrema=False)
            for body in violin['bodies']:
                body.set_facecolor(color);body.set_edgecolor(color);body.set_alpha(.85)
            ax.scatter(pos,np.mean(values),s=42,c='white',edgecolors=color,zorder=3)
    ax.set_xticks([0,1],['All saved','R1–R5 compliant'])
    ax.set_ylabel('Docking score (kcal/mol)')
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.16)
    ax.legend(handles=[Patch(color='#A7B3BF',label='Base'),Patch(color='#044875',label='Epoch80')],
              loc='lower left',bbox_to_anchor=(0,1.04),ncol=2,frameon=False,fontsize=18)
    for suffix in ['png','svg','pdf']:fig.savefig(out/f'diffgui_matched_before_after.{suffix}',dpi=300)
    (out/'diffgui_display_stats.json').write_text(json.dumps(stats,indent=2))
    print(json.dumps(stats,indent=2))


if __name__=='__main__':main()
