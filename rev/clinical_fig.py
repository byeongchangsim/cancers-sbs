"""Render the final clinical Figure 4 with the original layout and endpoint labels.

HRD and the POLE/POLD1 composite retain the submitted patient scores. The MSI
panel uses the final continuous-SBS6 RBF regression readout on the same 365
patients and original MANTIS labels; it is not an old/new model comparison.
"""
from pathlib import Path
from types import ModuleType
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'rev'
SOURCE = ROOT / 'paper_ready_v2_46/reproducibility/build_v2_40_layout_figures.py'

code = SOURCE.read_text()
code = code.replace('"TCGA UCEC\\nMSI/MMR"', '"TCGA UCEC\\nMANTIS MSI"')
code = code.replace('"MSI/MMR · primary SBS6 score"',
                    '"MANTIS MSI · SBS6 regression score"')
builder = ModuleType('final_clinical_figure')
exec(compile(code, str(SOURCE), 'exec'), builder.__dict__)
builder.configure_style()

metrics = pd.read_excel(ROOT / 'can/Supplementary_Data_S2.xlsx',
                       sheet_name='clinical_results', header=3)
patients = pd.read_excel(ROOT / 'can/Supplementary_Data_S2.xlsx',
                        sheet_name='patient_scores', header=3)
metrics['analysis'] = ['HRD', 'HRD >=42', 'MSI/MMR', 'POLE/POLD1']
patients['endpoint'] = patients.endpoint.replace({'MANTIS-defined MSI': 'MSI/MMR'})
patients['cohort'] = np.where(patients.endpoint.eq('HRD'), 'TCGA BRCA', 'TCGA UCEC')

final = pd.read_csv(OUT / 'kread6_tcga_scores.tsv', sep='\t').set_index('PATIENT')
assert final.index.is_unique
msi = patients.endpoint.eq('MSI/MMR')
assert int(msi.sum()) == 365 and int(patients.loc[msi, 'label'].sum()) == 102
revised = patients.loc[msi, 'PATIENT'].map(final.prediction_SBS6)
assert revised.notna().all()
patients.loc[msi, 'score'] = revised.to_numpy()
record = next(r for r in json.loads((OUT / 'kread6_metrics.json').read_text())
              if r['cohort'] == 'TCGA' and r['model'] == 'RBF SBS6 readout'
              and r['definition'] == 'Original gray-zone exclusion')
mask = metrics.analysis.eq('MSI/MMR')
for key, value in [('n', record['n']), ('positives', record['positive_n']),
                   ('estimate', record['auc']), ('ci_low', record['ci_low']),
                   ('ci_high', record['ci_high']), ('p_value', record['p'])]:
    metrics.loc[mask, key] = value
labels = patients.loc[msi, 'label'].to_numpy(int)
values = patients.loc[msi, 'score'].to_numpy(float)
assert abs(roc_auc_score(labels, values) - record['auc']) < 1e-9
assert abs(mannwhitneyu(values[labels == 1], values[labels == 0],
                       alternative='two-sided').pvalue - record['p']) < 1e-9

auc = metrics[metrics.metric.eq('ROC AUC')].copy().reset_index(drop=True)
builder.clinical_data = lambda: (metrics, patients, auc)
figures = []
builder.save_figure = lambda fig, *args, **kwargs: figures.append(fig)
builder.figure4(OUT)
fig = figures[0]

# Preserve the submitted bold italic gene typography in the third panel.
matplotlib.rcParams['mathtext.it'] = 'Liberation Sans:style=italic:weight=bold'
matplotlib.rcParams['mathtext.rm'] = 'Liberation Sans:style=italic:weight=bold'
gene_title = fig.axes[-1]._left_title
assert gene_title.get_text() == r'$\it{POLE}$/$\it{POLD1}$ · primary SBS10 score'
gene_title.set_math_fontfamily('custom')

for extension, dpi in [('pdf', None), ('png', 300), ('tif', 600)]:
    options = dict(format='tiff' if extension == 'tif' else extension,
                   bbox_inches='tight', pad_inches=0.03, facecolor='white')
    if dpi:
        options['dpi'] = dpi
    if extension == 'tif':
        options['pil_kwargs'] = {'compression': 'tiff_lzw'}
    fig.savefig(OUT / ('Figure_4.' + extension), **options)
plt.close(fig)
print('Final Figure 4 saved: original HRD/polymerase scores retained; '
      'SBS6 regression score on the original 365 patients, AUC %.3f.' % record['auc'])
