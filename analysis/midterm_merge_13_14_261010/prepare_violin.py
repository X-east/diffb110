"""Prepare a readable docking distribution from the existing epoch80 pool."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    """Render only stored scores, without docking or model execution."""
    root=Path(__file__).resolve().parents[2]
    source=root/'analysis/diffgui_epoch80_independent_eval_261010/slide_assets/independent_molecules_joined.csv'
    with source.open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    scored=[r for r in rows if r['score_status']=='ok']
    clean=[r for r in scored if r['clean']=='True']
    assert len(scored)==1858 and len(clean)==477
    plt.rcParams.update({'font.family':'Arial','font.size':19,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(6.7,2.75));fig.subplots_adjust(left=.17,right=.99,bottom=.23,top=.97)
    for i,(selected,color) in enumerate([(scored,'#A7B3BF'),(clean,'#044875')]):
        values=[float(r['score']) for r in selected]
        violin=ax.violinplot(values,positions=[i],widths=.80,showextrema=False,bw_method=.25)
        for body in violin['bodies']:body.set_facecolor(color);body.set_edgecolor(color);body.set_alpha(.85)
        ax.scatter(i,np.mean(values),s=54,color='white',edgecolors=color,zorder=4)
    ax.set_xticks([0,1],['All scored','R1–R5 compliant'],fontsize=18)
    ax.set_ylabel('Docking (kcal/mol)',fontsize=19)
    ax.set_yticks([-6,-9,-12,-15,-18]);ax.tick_params(axis='y',labelsize=17)
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
    for ext in ['png','svg','pdf']:fig.savefig(Path(__file__).parent/f'epoch80_docking.{ext}',dpi=300)
    print('Saved figure:',len(scored),len(clean))


if __name__=='__main__':main()
