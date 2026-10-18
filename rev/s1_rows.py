"""One-page S1 with categorical targets and paired vertical CIs.

Layout reference: Fu et al., Nature Cancer (2020), Extended Data Fig. 8a.
https://www.nature.com/articles/s43018-020-0085-8/figures/16
The source uses target-wise cohort estimates and vertical confidence intervals;
this adaptation uses seven aligned cancer-type rows for all 30 SBS targets.
"""
from pathlib import Path
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CANCERS = ['BRCA', 'HNSC', 'KIRC', 'LUAD', 'LUSC', 'OV', 'UCEC']
SBS = [f'SBS{i}' for i in range(1, 31)]
COLORS = {'TCGA': '#0072B2', 'CPTAC': '#D55E00'}
TITLE = 'Figure S1. Signature associations across cancer types'
CAPTION = ('Pearson correlations between genomic SBS exposures and standardised image-derived scores are shown for all 210 cancer–signature combinations. '
           'Blue circles denote TCGA and orange squares denote CPTAC. Vertical bars show patient-level Fisher-z 95% confidence intervals. '
           'All panels use the same scale. Dashed lines mark r = 0. Crosses below the axes indicate unavailable correlations due to constant exposure or score values. '
           'Exact estimates, positive-direction one-sided P values and BH-adjusted P values across 210 tests separately in each cohort are provided in Supplementary Data S3 (paired_metrics).')


def make_figure():
    d = pd.read_csv(ROOT / 'rev/paired_metrics.tsv', sep='\t')
    assert len(d) == 210 and not d.duplicated(['cancer', 'signature']).any()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'pdf.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False})
    # A4 portrait: legibility is assessed at an ordinary one-page document size.
    fig, axes = plt.subplots(7, 1, figsize=(8.27, 11.69), sharex=True, sharey=True)
    fig.subplots_adjust(left=.10, right=.98, top=.88, bottom=.128, hspace=.38)
    counts = dict(TCGA=0, CPTAC=0)
    x = np.arange(1, 31)
    for ax, cancer in zip(axes, CANCERS):
        subset = d[d.cancer == cancer].set_index('signature').reindex(SBS)
        for start in [1, 11, 21]:
            ax.axvspan(start-.5, start+4.5, color='#F3F5F7', zorder=0, lw=0)
        ax.vlines(x, -.53, .78, color='#E9EDF0', lw=.35, zorder=0)
        ax.axhline(0, color='#6F7880', lw=.7, ls=(0, (3, 3)), zorder=1)
        for cohort, shift, marker in [('TCGA', -.16, 'o'), ('CPTAC', .16, 's')]:
            valid = subset[f'pearson_r_{cohort}'].notna().to_numpy()
            r = subset.loc[valid, f'pearson_r_{cohort}'].to_numpy()
            low = subset.loc[valid, f'ci_low_{cohort}'].to_numpy()
            high = subset.loc[valid, f'ci_high_{cohort}'].to_numpy()
            assert np.all(low <= r) and np.all(r <= high)
            assert low.min() > -.53 and high.max() < .78
            ax.errorbar(x[valid]+shift, r, yerr=[r-low, high-r], fmt=marker,
                        color=COLORS[cohort], markersize=2.7, capsize=1.4,
                        elinewidth=.85, markeredgewidth=.6, zorder=3)
            # A cross in the separate footer strip marks an unavailable estimate.
            ax.plot(x[~valid]+shift, np.full((~valid).sum(), -.665),
                    marker='x', ls='none', color=COLORS[cohort], ms=3.1,
                    mew=.7, clip_on=False, zorder=3)
            counts[cohort] += int(valid.sum())
        ax.set_ylim(-.53, .78)
        ax.set_yticks([-.4, 0, .4], ['−0.4', '0', '0.4'])
        ax.tick_params(axis='y', labelsize=7.5, width=.5, length=2)
        ax.tick_params(axis='x', labelsize=7.5, length=2, width=.5, pad=3)
        ax.spines['left'].set_linewidth(.5)
        ax.spines['bottom'].set_linewidth(.5)
        ax.spines['left'].set_color('#8B949C')
        ax.spines['bottom'].set_color('#8B949C')
        ax.set_title(cancer, loc='left', fontsize=10,
                     fontweight='bold', pad=5)
    axes[-1].set_xticks(x, [str(i) for i in x])
    axes[-1].tick_params(axis='x', pad=12)
    axes[-1].set_xlim(.5, 30.5)
    axes[-1].set_xlabel('SBS signature', fontsize=9, labelpad=6)
    axes[0].tick_params(axis='x', top=True, labeltop=True, bottom=False)
    for ax in axes[1:-1]:
        ax.tick_params(axis='x', bottom=False)
    fig.text(.025, .50, 'Pearson correlation (r)', rotation=90,
             va='center', ha='center', fontsize=10)
    fig.suptitle(TITLE,
                 x=.10, y=.978, ha='left', fontsize=12, fontweight='bold')
    fig.text(.10, .950, 'All 210 combinations · TCGA and CPTAC · 95% confidence intervals',
             fontsize=9, color='#424B53')
    handles = [Line2D([0], [0], marker=m, color=COLORS[c], markersize=4,
                      lw=.9, label=c) for c, m in [('TCGA', 'o'), ('CPTAC', 's')]]
    handles += [Line2D([0], [0], marker='x', color='#5C6670', ls='none',
                       markersize=4, label='Unavailable (below axis)')]
    fig.legend(handles=handles, loc='center left', bbox_to_anchor=(.09, .920),
               ncol=3, frameon=False, fontsize=8.5, columnspacing=2)
    fig.text(.10, .012, textwrap.fill(CAPTION, 116), fontsize=7.5, va='bottom', linespacing=1.2)
    assert counts == {'TCGA': 200, 'CPTAC': 187}
    return fig


def main():
    fig = make_figure()
    fig.savefig(ROOT / 'Figure_S1_rows.pdf')
    fig.savefig(ROOT / 'Figure_S1_rows.png', dpi=220, facecolor='white')
    # Word retains an editable title and legend around the plot image.
    for item in list(fig.texts):
        if item.get_text() != 'Pearson correlation (r)':
            item.remove()
    fig.savefig(ROOT / 'rev/s1_word.png', dpi=300, facecolor='white',
                bbox_inches=Bbox.from_extents(0, .075*11.69, 8.27, .94*11.69))
    plt.close(fig)
    print('One A4 page; 210 combinations; 387 estimated correlations and corresponding confidence intervals.')


if __name__ == '__main__':
    main()
