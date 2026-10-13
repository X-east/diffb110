"""Scale post-selection maximum violin width by the within-model retained fraction."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODELS = ['DiffGui', 'DiffSBDD', 'GCDM-SBDD', 'PMDM', 'TargetDiff']
GRAY, BLUE = '#A6B3C1', '#064C76'


def main() -> None:
    """Keep distributions fixed while adding an explicit count-to-width mapping."""
    source = pd.read_csv(ROOT / 'analysis/midterm_ch2_results_261009/benchmark_per_molecule.csv')
    counts = pd.read_csv(HERE / 'selection_counts.csv').set_index('model')
    plt.rcParams.update({'font.family':'Arial','font.size':21,'pdf.fonttype':42,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False})
    fig, ax=plt.subplots(figsize=(15.5,7.1))
    fig.subplots_adjust(left=.10,right=.985,top=.82,bottom=.24)
    metadata=[]
    for i, model in enumerate(MODELS):
        raw=source[(source.model==model)&np.isfinite(source.score)&(source.score<=0)]
        ratio=float(counts.loc[model,'after_n']/counts.loc[model,'before_n'])
        for offset, clean, color in [(-.19,False,GRAY),(.19,True,BLUE)]:
            values=(raw[raw.clean] if clean else raw).score.to_numpy()
            width=.36*(ratio if clean else 1.0)
            parts=ax.violinplot(values,positions=[i+offset],widths=width,showextrema=False)
            for body in parts['bodies']:
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_linewidth(1.2)
                body.set_alpha(.68 if not clean else .88)
            # Keep mean markers small enough to fit the narrowest retained violin.
            marker_size=18
            ax.scatter(i+offset,values.mean(),s=marker_size,facecolor='white',edgecolor=color,linewidth=.8,zorder=4)
        metadata.append(dict(model=model,before_n=int(counts.loc[model,'before_n']),after_n=int(counts.loc[model,'after_n']),
                             retained_fraction=ratio,before_max_width=.36,after_max_width=.36*ratio))
    labels=[f"{m}\n{int(counts.loc[m,'before_n'])} → {int(counts.loc[m,'after_n'])}" for m in MODELS]
    ax.set_xticks(np.arange(5),labels,fontsize=20)
    ax.set_ylabel('Docking score (kcal/mol)',fontsize=23)
    ax.set_yticks([-15,-10,-5])
    ax.set_ylim(-16.5,-.8)
    ax.set_xlim(-.6,4.6)
    ax.grid(axis='y',color='#E8ECF0')
    ax.set_axisbelow(True)
    ax.tick_params(axis='x',pad=8)
    fig.legend(handles=[Patch(color=GRAY,label='Before rule selection'),Patch(color=BLUE,label='R1–R5 compliant')],
               loc='upper center',bbox_to_anchor=(.54,.995),ncol=2,frameon=False,fontsize=23)
    fig.text(.10,.855,'Blue maximum width / gray maximum width = retained fraction within each model',fontsize=18,color='#526272')
    fig.text(.10,.035,'Numbers below models: molecules before → after selection. White dots: mean docking scores.',fontsize=16,color='#526272')
    fig.savefig(HERE/'docking_violin_count_scaled.png',dpi=300)
    fig.savefig(HERE/'docking_violin_count_scaled.svg')
    fig.savefig(HERE/'docking_violin_count_scaled.pdf')
    (HERE/'violin_width_mapping.json').write_text(json.dumps(dict(mapping='Maximum width ratio, not area; within-model fractions, not absolute cross-model counts',
                                                               data=metadata),ensure_ascii=False,indent=2))
    print(json.dumps(metadata,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
