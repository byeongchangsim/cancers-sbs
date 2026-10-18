"""Consolidate existing revision results into five supplementary figure groups."""
from pathlib import Path
import json, textwrap
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.transforms import Bbox

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'rev'
CANCERS = ['BRCA', 'HNSC', 'KIRC', 'LUAD', 'LUSC', 'OV', 'UCEC']
SBS = ['SBS' + str(i) for i in range(1, 31)]
TEAL, GOLD, GREY = '#187c91', '#bd7734', '#87969e'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'pdf.fonttype': 42, 'axes.spines.top': False,
                     'axes.spines.right': False})
Image.MAX_IMAGE_PIXELS = None  # Previously verified local study mosaics.
wide = pd.read_csv(OUT / 'paired_metrics.tsv', sep='\t')
long = pd.read_csv(OUT / 'cohort_metrics.tsv', sep='\t')
scores = pd.read_csv(OUT / 'patient_scores.tsv', sep='\t')
mutations = pd.read_csv(OUT / 'mutation_summary.tsv', sep='\t')
sens = pd.read_csv(OUT / 'tumour_sensitivity.tsv', sep='\t')
clinical = pd.read_csv(OUT / 'clinical_revision.tsv', sep='\t')
cases = pd.read_csv(OUT / 'attention_cases.tsv', sep='\t')
weights = pd.read_csv(OUT / 'attention_tiles.tsv', sep='\t')
nominal = wide[wide.nominal_positive_both].copy()
nominal['order'] = nominal.signature.str[3:].astype(int)
nominal = nominal.sort_values(['cancer', 'order'])
labels = nominal.cancer + '–' + nominal.signature
pages = []


def finish(pdf, fig, name, group, caption, caption_width=155):
    from legends import LEGENDS, S5_CAPTIONS
    if group < 5:
        caption = LEGENDS[group - 1][1]
    else:
        order = ['supp5_ucec_sbs13', 'supp5_ucec_sbs3', 'supp5_brca_sbs3', 'supp5_luad_sbs4']
        caption = S5_CAPTIONS[order.index(name)]
    fig.text(.045, .015, textwrap.fill(caption, caption_width), fontsize=8,
             va='bottom', color='#263746')
    pdf.savefig(fig, dpi=200)
    fig.savefig(OUT / (name + '.png'), dpi=150, facecolor='white')
    pages.append({'figure': group, 'page': len(pages) + 1,
                  'preview': name + '.png', 'caption': caption})
    plt.close(fig)


def error(ax, x, low, high, y, color=TEAL, label=None, shift=0, size=4):
    ax.errorbar(x, np.asarray(y) + shift, xerr=[np.asarray(x)-np.asarray(low),
                np.asarray(high)-np.asarray(x)], fmt='o', color=color,
                ms=size, capsize=2, lw=1, label=label)


def landscape(pdf):
    from s1_rows import make_figure, CAPTION
    fig = make_figure()
    pdf.savefig(fig)
    fig.savefig(OUT / 'supp1_combined.png', dpi=220, facecolor='white')
    pages.append({'figure': 1, 'page': len(pages) + 1,
                  'preview': 'supp1_combined.png', 'caption': CAPTION})
    plt.close(fig)


def diagnostics(pdf):
    fig = plt.figure(figsize=(16.54, 11.69))
    gs = fig.add_gridspec(7, 4, left=.065, right=.985, top=.865,
                         bottom=.18, hspace=.75, wspace=.36)
    for j, co in enumerate(['TCGA', 'CPTAC']):
        fig.text(.28 + .48*j, .935, co, ha='center', fontsize=13, weight='bold')
        for col, title in enumerate(['Observed exposure', 'Standardised model score']):
            pos = gs[0, 2*j+col].get_position(fig)
            fig.text((pos.x0+pos.x1)/2, .905, title, ha='center', fontsize=10)
        for i, cancer in enumerate(CANCERS):
            data = scores[(scores.cohort == co) & (scores.cancer == cancer)]
            meta = mutations[(mutations.cohort == co) & (mutations.cancer == cancer)].iloc[0]
            sub = gs[i, 2*j].subgridspec(2, 1, height_ratios=[.84, .16], hspace=.04)
            ax = fig.add_subplot(sub[0]); strip = fig.add_subplot(sub[1], sharex=ax)
            bx = fig.add_subplot(gs[i, 2*j+1])
            values = [data[s].to_numpy() for s in SBS]
            for axis, vals, color in [(ax, values, '#d2e3e8'),
                                     (bx, [data['prediction_' + s].to_numpy() for s in SBS], '#b3d4df')]:
                axis.boxplot(vals, whis=(5, 95), showfliers=False, patch_artist=True,
                             boxprops={'facecolor': color}, medianprops={'color': '#263746'},
                             widths=.55, flierprops={'markersize': 0})
                axis.set_xlim(.5, 30.5); axis.tick_params(labelsize=7, length=2)
            ax.set_title(f'{cancer} | n={len(data)} | samples={int(meta.included_eligible_samples)}\nSNVs {meta["median"]:g} [{meta.q1:g}–{meta.q3:g}]',
                         loc='left', fontsize=8, pad=3)
            ax.set_ylabel('Exposure', fontsize=8); ax.tick_params(axis='x', labelbottom=False)
            zero = np.array([(v == 0).mean() for v in values])[None, :]
            strip.imshow(zero, extent=[.5, 30.5, 0, 1], vmin=0, vmax=1,
                         cmap='YlOrBr', aspect='auto', interpolation='none')
            strip.set_yticks([]); strip.set_xticks(range(1, 31), range(1, 31))
            strip.tick_params(axis='x', labelsize=7, length=2, labelbottom=i == 6,
                              labelrotation=90)
            bx.set_title(cancer, loc='left', fontsize=8, pad=3)
            bx.set_ylabel('Score', fontsize=8); bx.set_xticks(range(1, 31), range(1, 31))
            bx.tick_params(axis='x', labelbottom=i == 6, labelrotation=90)
            if i == 6:
                strip.set_xlabel('SBS signature', fontsize=8); bx.set_xlabel('SBS signature', fontsize=8)
    cax = fig.add_axes([.37, .095, .26, .012])
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap='YlOrBr'),
                 cax=cax, orientation='horizontal', ticks=[0, .5, 1],
                 label='Fraction of patients with zero observed exposure')
    fig.suptitle('Figure S2. Target distributions, sparsity and catalogue counts',
                 fontsize=15, y=.985)
    finish(pdf, fig, 'supp2_distributions', 2,
           'TCGA (left pair of columns) and CPTAC (right pair) each show all 30 observed SBS exposures and standardised model scores for every cancer. Boxes show medians and interquartile ranges; whiskers are the 5th–95th percentiles, with outliers omitted. The strip below each exposure panel gives its zero fraction. n denotes patients; samples denotes included eligible genomic samples. SNVs are sample-level median [IQR], not patient-averaged counts. Complete distributions and cohort-level mutation-count and eligibility summaries are in Supplementary Data S3.',
           caption_width=220)


def robustness(pdf):
    fig = plt.figure(figsize=(11.69, 11.69))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.7], left=.13,
                         right=.98, top=.93, bottom=.105, hspace=.43, wspace=.33)
    palette = plt.get_cmap('tab10')
    for j, co in enumerate(['TCGA', 'CPTAC']):
        ax = fig.add_subplot(gs[0, j]); df = long[(long.dataset == co) & long.pearson_r.notna() & long.spearman_rho.notna()]
        for k, cancer in enumerate(CANCERS):
            sub = df[df.cancer == cancer]
            ax.scatter(sub.pearson_r, sub.spearman_rho, s=13, alpha=.7,
                       color=palette(k), label=cancer)
        low = min(df.pearson_r.min(), df.spearman_rho.min()) - .05
        high = max(df.pearson_r.max(), df.spearman_rho.max()) + .05
        ax.plot([low, high], [low, high], '--', color='#a8b1b7', lw=.8)
        ax.set_xlim(low, high); ax.set_ylim(low, high)
        ax.set_xlabel('Pearson r'); ax.set_ylabel('Spearman ρ')
        agree = np.corrcoef(df.pearson_r, df.spearman_rho)[0, 1]
        ax.set_title(f'({"a" if j == 0 else "b"}) {co} effect agreement {agree:.3f}', loc='left')
    handles, names = ax.get_legend_handles_labels()
    fig.legend(handles, names, loc='upper center', bbox_to_anchor=(.56, .965), ncol=7, fontsize=8)
    subset = nominal[['cancer', 'signature']].merge(sens, on=['cancer', 'signature'])
    y = np.arange(14); ax = fig.add_subplot(gs[1, :])
    for prefix, color, shift, label in [('all', GREY, -.15, 'All slides (same patients)'),
                                       ('tumour', TEAL, .15, 'Tumour-labelled slides')]:
        error(ax, subset[prefix + '_r'], subset[prefix + '_ci_low'],
              subset[prefix + '_ci_high'], y, color, label, shift)
    ax.axvline(0, color='#a8b1b7', ls='--'); ax.set_yticks(y, labels)
    ax.invert_yaxis(); ax.set_xlabel('CPTAC Pearson r (95% Fisher-z CI)')
    ax.set_title('(c) Paired input sensitivity (14 nominal associations)', loc='left')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.11), ncol=2, fontsize=8)
    fig.suptitle('Figure S3. Robustness of score–exposure associations', fontsize=16, y=.99)
    finish(pdf, fig, 'supp3_robustness', 3,
           'Panels a–b compare Pearson and Spearman effects using the original scores. Panel c compares all-slide and tumour-labelled inputs in the same 570 retained patients; the original 580-patient analysis remains primary. All original 14 pairs are displayed with confidence intervals. Data S3 supplies input counts and the complete 210-pair sensitivity results.')


def genomic(pdf):
    fig, axes = plt.subplots(2, 2, figsize=(16.54, 11.69))
    fig.subplots_adjust(left=.12, right=.98, top=.9, bottom=.16, hspace=.48, wspace=.46)
    ax = axes[0, 0]; df = clinical[clinical.cohort == 'CPTAC']
    error(ax, df.auc, df.ci_low, df.ci_high, np.arange(2))
    ax.set_yticks(range(2), [f'{r.endpoint}\nn={r.n}, positive={r.positive_n}\nP={r.p:.3g}' for r in df.itertuples()])
    ax.invert_yaxis(); ax.axvline(.5, color=GREY, ls='--'); ax.set_xlim(.3, 1)
    ax.set_title('(a) CPTAC clinical endpoint assessment', loc='left')
    ax.set_xlabel('AUC (95% bootstrap CI)')
    ax = axes[0, 1]; df = clinical[clinical.cohort == 'TCGA']
    error(ax, df.auc, df.ci_low, df.ci_high, np.arange(2))
    ax.set_yticks(range(2), [f'{"Original exclusion" if i == 0 else "0.4 threshold"}\nn={r.n}, positive={r.positive_n}' for i, r in enumerate(df.itertuples())])
    ax.invert_yaxis(); ax.axvline(.5, color=GREY, ls='--'); ax.set_xlim(.3, 1)
    ax.set_title('(b) TCGA MANTIS definition sensitivity', loc='left')
    ax.set_xlabel('AUC (95% bootstrap CI)')
    from sklearn.metrics import roc_curve, roc_auc_score
    final_scores = pd.read_csv(OUT / 'clinical_revision_scores.tsv', sep='\t')
    ax = axes[1, 0]
    sample = final_scores[(final_scores.cohort == 'TCGA') & (final_scores.endpoint == 'MSI') & (final_scores.rule == 'Original gray-zone exclusion')]
    for label, color, text in [(0, GREY, 'MANTIS-negative'), (1, TEAL, 'MANTIS-positive')]:
        values = sample.loc[sample.label == label, 'score']
        ax.hist(values, bins=18, density=True, alpha=.5, color=color, label=f'{text} (n={len(values)})')
    ax.set_title('(c) TCGA final SBS6 clinical scores', loc='left')
    ax.set_xlabel('Exposure-trained SBS6 regression score'); ax.set_ylabel('Density'); ax.legend(fontsize=8)
    ax = axes[1, 1]
    for co, rule, color, text in [('TCGA', 'Original gray-zone exclusion', TEAL, 'TCGA primary MSI'), ('TCGA', 'Published 0.4 threshold', GOLD, 'TCGA MANTIS 0.4'), ('CPTAC', 'Source-published MSI-H versus MSS', GREY, 'CPTAC MSI')]:
        d = final_scores[(final_scores.cohort == co) & (final_scores.endpoint == 'MSI') & (final_scores.rule == rule)]
        fpr, tpr, _ = roc_curve(d.label, d.score)
        ax.plot(fpr, tpr, color=color, lw=1.4, label=f'{text}, AUC={roc_auc_score(d.label,d.score):.3f}')
    ax.plot([0,1],[0,1],'--',color='#b6bfc4',lw=.8);ax.set_xlim(0,1);ax.set_ylim(0,1)
    ax.set_title('(d) Final MSI discrimination', loc='left');ax.set_xlabel('False positive rate');ax.set_ylabel('True positive rate');ax.legend(loc='lower right',fontsize=8)
    fig.suptitle('Figure S4. Exploratory genomic-feature associations and MANTIS sensitivity', fontsize=16, y=.98)
    finish(pdf, fig, 'supp4_genomic', 4,
           'MSI is evaluated using the final continuous-SBS6-trained kernel regression readout; no MSI labels enter fitting or selection. The primary SBS10 score evaluates the original exact-list POLE OR POLD1 endpoint in the complete sequenced CPTAC profile: six positives (five P286R and one V411L) and 76 qualifying-list negatives. Panel b holds the clinical SBS6 score fixed while comparing the primary MANTIS rule and the 0.4 threshold. Panels c–d show the final MSI scores and ROC curves. Endpoint estimates and uncertainty are reported in Data S3; endpoint definitions remain in Data S2.')


def attention(pdf):
    cmap = plt.get_cmap('inferno')
    for cancer, sig in [('UCEC', 'SBS13'), ('UCEC', 'SBS3'), ('BRCA', 'SBS3'), ('LUAD', 'SBS4')]:
        fig = plt.figure(figsize=(16.54, 11.69))
        gs = fig.add_gridspec(3, 12, left=.03, right=.98, top=.80,
                             bottom=.16, height_ratios=[1.8, 1, 1], hspace=.48, wspace=.5)
        for j, co in enumerate(['TCGA', 'CPTAC']):
            row = cases[(cases.cohort == co) & (cases.cancer == cancer) & (cases.signature == sig)].iloc[0]
            t = weights[(weights.cohort == co) & (weights.cancer == cancer) & (weights.signature == sig)].copy()
            t['percentile'] = t.attention.rank(method='average', pct=True)
            show = t[t.FILENAME == row.displayed_slide].copy()
            order = show.sort_values(['attention', 'x', 'y'])
            selected = pd.concat([order.iloc[-3:].iloc[::-1].assign(group='High'), order.iloc[:3].assign(group='Low')])
            assert len(show) == row.displayed_slide_tiles and len(t) == row.sampled_patient_tiles
            with Image.open(row.raw_image) as raw:
                size = raw.size; thumb = raw.copy(); thumb.thumbnail((1600, 1050)); thumb = thumb.convert('RGB')
                crops = [raw.crop((int(r.x), int(r.y), int(r.x)+224, int(r.y)+224)).convert('RGB') for r in selected.itertuples()]
            layer = Image.new('RGBA', thumb.size, (0, 0, 0, 0)); draw = ImageDraw.Draw(layer)
            sx, sy = thumb.width/size[0], thumb.height/size[1]
            for r in show.itertuples():
                color = tuple(int(255*v) for v in cmap(r.percentile)[:3]) + (145,)
                draw.rectangle((int(r.x*sx), int(r.y*sy), max(int((r.x+224)*sx), int(r.x*sx)+1), max(int((r.y+224)*sy), int(r.y*sy)+1)), fill=color)
            overlay = Image.alpha_composite(thumb.convert('RGBA'), layer).convert('RGB')
            start = 6*j
            for block, image, title in [(gs[0, start:start+3], thumb, 'Original resampled H&E'),
                                        (gs[0, start+3:start+6], overlay, 'Attention on sampled tiles')]:
                ax = fig.add_subplot(block); ax.imshow(image, interpolation='none'); ax.axis('off'); ax.set_title(title, fontsize=11)
            for k, (r, crop) in enumerate(zip(selected.itertuples(), crops)):
                ax = fig.add_subplot(gs[1+k//3, start+2*(k%3):start+2*(k%3)+2])
                ax.imshow(crop, interpolation='none'); ax.axis('off')
                ax.set_title(f'{r.group} {k%3+1} | weight {r.attention:.4g}', fontsize=9)
                ax.plot([10, 80], [210, 210], color='black', lw=3)
                ax.text(10, 195, '80 µm', fontsize=8, bbox=dict(facecolor='white', alpha=.85, edgecolor='none'))
            xpos = .035 if j == 0 else .535
            fig.text(xpos, .91, f'{co} | {row.PATIENT} | exposure {row.observed_exposure:.3f}', fontsize=13, weight='bold')
            fig.text(xpos, .86, f'Model score {row.primary_ensemble_score:.3f}', fontsize=9)
        cax = fig.add_axes([.4, .108, .2, .012])
        bar = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap), cax=cax,
                           orientation='horizontal', ticks=[0, .5, 1], label='Within-patient attention percentile')
        bar.ax.set_xticklabels(['0', '50', '100'])
        fig.suptitle(f'Figure S5. Paired TCGA/CPTAC attention examples — {cancer}–{sig}', fontsize=16, y=.985)
        finish(pdf, fig, 'supp5_' + cancer.lower() + '_' + sig.lower(), 5,
               'Each case links measured genomic exposure and the patient-level standardised ensemble score with relative tile weighting in one MIL component. Higher-/lower-weight tiles are selected within the displayed slide. Colours represent within-patient rank, not tumour probability or exposure magnitude; uncoloured regions were not sampled. Cases were chosen by median positive exposure. Each crop spans 256 × 256 µm. Tissue identities and causal importance are not assigned. Patient values and displayed tile weights are included in each example.')
        fig_path = OUT / ('supp5_' + cancer.lower() + '_' + sig.lower() + '.png')
        # Reuse the updated figure in the presentation.
        import shutil
        shutil.copy2(fig_path, OUT / ('report_attention_' + cancer.lower() + '_' + sig.lower() + '.png'))
        print('Paired attention:', cancer, sig, flush=True)


if __name__ == '__main__':
    with PdfPages(OUT / 'supp_build.pdf') as pdf:
        landscape(pdf); diagnostics(pdf); robustness(pdf); genomic(pdf); attention(pdf)
    assert len(pages) == 8 and {p['figure'] for p in pages} == {1, 2, 3, 4, 5}
    (OUT / 'supp_build.pdf').replace(OUT / 'Revision_Figures.pdf')
    (OUT / 'supp_layout.json').write_text(json.dumps(pages, indent=2))
    (ROOT / 'Cancers_supp.pdf').write_bytes((OUT / 'Revision_Figures.pdf').read_bytes())
    print('Saved five figure groups across eight pages; existing numerical results retained.')
