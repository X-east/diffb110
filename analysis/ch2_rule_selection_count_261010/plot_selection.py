"""Show docking distributions and the explicit molecule loss after R1-R5 selection."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SKILL_SCRIPTS = Path('/home/md666/.codex/plugins/cache/nature-skills/nature-skills/0.1.0/skills/nature-figure/scripts')
sys.path.insert(0, str(SKILL_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment

MODELS = ['DiffGui', 'DiffSBDD', 'GCDM-SBDD', 'PMDM', 'TargetDiff']
GRAY, BLUE = '#A6B3C1', '#064C76'


def distribution(ax: plt.Axes, source: pd.DataFrame) -> None:
    """Retain the original non-positive finite-score distribution convention."""
    for i, model in enumerate(MODELS):
        raw = source[(source.model == model) & np.isfinite(source.score) & (source.score <= 0)]
        for offset, clean, color in [(-.19, False, GRAY), (.19, True, BLUE)]:
            rows = raw[raw.clean] if clean else raw
            values = rows.score.to_numpy()
            parts = ax.violinplot(values, positions=[i+offset], widths=.35,
                                  showmeans=False, showextrema=False, showmedians=False)
            for body in parts['bodies']:
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(.65 if not clean else .82)
            ax.scatter(i+offset, values.mean(), s=46, c='white', edgecolors=color, linewidths=1.5, zorder=4)
    ax.set_ylabel('Docking score (kcal/mol)', fontsize=22)
    ax.set_yticks([-15, -10, -5])
    ax.set_ylim(-16.5, -.8)
    ax.set_xlim(-.6, 4.6)
    ax.grid(axis='y', color='#E8ECF0')
    ax.set_axisbelow(True)


def main() -> None:
    """Export complementary distribution and count panels with exact source counts."""
    source = pd.read_csv(ROOT / 'analysis/midterm_ch2_results_261009/benchmark_per_molecule.csv')
    summary = pd.read_csv(ROOT / 'analysis/midterm_ch2_results_261009/benchmark_summary.csv').set_index('model')
    rows = []
    for model in MODELS:
        group = source[source.model == model]
        before, after = len(group), int(group.clean.sum())
        assert before == int(summary.loc[model, 'valid_n'])
        assert after == int(summary.loc[model, 'lf_clean_n'])
        rows.append(dict(model=model, before_n=before, after_n=after,
                         removed_n=before-after, retained_pct=after/before*100,
                         scored_before_n=int((np.isfinite(group.score)&(group.score<=0)).sum()),
                         scored_after_n=int((np.isfinite(group.score)&(group.score<=0)&group.clean).sum())))
    counts = pd.DataFrame(rows)
    counts.to_csv(HERE / 'selection_counts.csv', index=False)
    plt.rcParams.update({'font.family':'Arial', 'font.size':20, 'pdf.fonttype':42,
                         'svg.fonttype':'none', 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(2, 1, figsize=(15.5, 9.3), sharex=True,
                             gridspec_kw={'height_ratios':[1.8, 1]})
    fig.subplots_adjust(left=.10, right=.985, top=.89, bottom=.10, hspace=.20)
    distribution(axes[0], source)
    axes[0].text(0,1.035,'A  Docking score distribution',transform=axes[0].transAxes,fontsize=21,fontweight='bold')
    axes[0].tick_params(axis='x',bottom=False,labelbottom=False)
    ax=axes[1]
    x=np.arange(len(MODELS))
    ax.bar(x-.19,counts.before_n,width=.35,color=GRAY)
    ax.bar(x+.19,counts.after_n,width=.35,color=BLUE)
    for i, row in counts.iterrows():
        ax.text(i-.19,row.before_n+65,str(row.before_n),ha='center',fontsize=18,color='#526272')
        ax.text(i+.19,row.after_n+65,str(row.after_n),ha='center',fontsize=18,color=BLUE)
    ax.set_ylabel('Molecule count',fontsize=22)
    ax.set_ylim(0,3000)
    ax.set_yticks([0,1000,2000])
    ax.set_xticks(x,MODELS,fontsize=21)
    ax.text(0,1.03,'B  Molecules retained after R1–R5 selection',transform=ax.transAxes,fontsize=21,fontweight='bold')
    ax.grid(False)
    ax.set_axisbelow(True)
    fig.legend(handles=[Patch(color=GRAY,label='Before rule selection'),Patch(color=BLUE,label='R1–R5 compliant')],
               loc='upper center',bbox_to_anchor=(.54,.997),ncol=2,frameon=False,fontsize=22)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=HERE/'selection_distribution_and_count.alignment.json',strict=True)
    fig.savefig(HERE/'selection_distribution_and_count.png',dpi=300)
    fig.savefig(HERE/'selection_distribution_and_count.svg')
    fig.savefig(HERE/'selection_distribution_and_count.pdf')
    plt.close(fig)
    fig, ax=plt.subplots(figsize=(15.5,5.4))
    fig.subplots_adjust(left=.10,right=.985,top=.83,bottom=.15)
    distribution(ax,source)
    ax.set_xticks(np.arange(5),MODELS,fontsize=22)
    fig.legend(handles=[Patch(color=GRAY,label='Before rule selection'),Patch(color=BLUE,label='R1–R5 compliant')],
               loc='upper center',ncol=2,frameon=False,fontsize=23)
    fig.savefig(HERE/'distribution_only.png',dpi=300)
    ax.tick_params(axis='x',bottom=False,labelbottom=False)
    fig.subplots_adjust(bottom=.23)
    fig.savefig(HERE/'distribution_for_slide.png',dpi=300)
    plt.close(fig)
    provenance=dict(source='analysis/midterm_ch2_results_261009/benchmark_per_molecule.csv',
                    count_definition='All valid molecules before selection; clean means largest-fragment R1-R5 compliant',
                    score_plot='Retain original figure: finite non-positive docking scores; excludes +345.218 and +12.865 only',
                    counts=rows,white_dot='Mean of plotted scores',width='Same maximum width per distribution; count encoded only in lower panel')
    (HERE/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
    print(counts.to_string(index=False))


if __name__=='__main__':
    main()
