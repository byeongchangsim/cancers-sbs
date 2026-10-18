"""Re-render the submitted Figure 4 with bold italic genes in panel c only."""
from pathlib import Path
from io import BytesIO, StringIO
from zipfile import ZipFile, ZIP_DEFLATED
from types import ModuleType
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.mathtext import MathTextParser
from matplotlib.font_manager import FontProperties, findfont
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path('/home/marugoto-1.0.0-regression')
OUT = ROOT / 'jpcr'
SOURCE = ROOT / 'paper_ready_v2_46/reproducibility/build_v2_40_layout_figures.py'
DATA = ROOT / 'paper_ready_v2_56/supplementary_data/Supplementary_Data_S2.xlsx'
code = SOURCE.read_text().replace('"TCGA UCEC\\nMSI/MMR"', '"TCGA UCEC\\nMANTIS MSI"')
code = code.replace('"MSI/MMR · primary SBS6 score"', '"MANTIS-defined MSI · primary SBS6 score"')
builder = ModuleType('figure4_builder')
exec(compile(code, str(SOURCE), 'exec'), builder.__dict__)
builder.configure_style()
metrics = pd.read_excel(DATA, sheet_name='TCGA_clinical_metrics')
metrics['analysis'] = ['HRD', 'HRD >=42', 'MSI/MMR', 'POLE/POLD1']
patients = pd.read_excel(DATA, sheet_name='TCGA_patient_scores')
patients['endpoint'] = patients.endpoint.replace({'MANTIS-defined MSI': 'MSI/MMR'})
# Match the original builder's TSV read/write path and column selection.
metrics = pd.read_csv(StringIO(metrics.to_csv(sep='\t', index=False)), sep='\t')
patients = pd.read_csv(StringIO(patients[['PATIENT', 'label', 'hrd_score', 'endpoint', 'score']].to_csv(sep='\t', index=False)), sep='\t')
patients['cohort'] = np.where(patients.endpoint.eq('HRD'), 'TCGA BRCA', 'TCGA UCEC')
auc = metrics[metrics.metric.eq('ROC AUC')].copy().reset_index(drop=True)
builder.clinical_data = lambda: (metrics, patients, auc)
figures = []
builder.save_figure = lambda fig, *args, **kwargs: figures.append(fig)
builder.figure4(OUT)
fig = figures[0]

def render(bbox='tight'):
    data = BytesIO()
    fig.savefig(data, format='tiff', dpi=600, bbox_inches=bbox, pad_inches=0.03,
                facecolor='white', pil_kwargs={'compression': 'tiff_lzw'})
    return Image.open(data).convert('RGB')

with ZipFile(ROOT / 'JPCR_submit.zip') as archive:
    original_bytes = archive.read('Figure_4.tif')
original = Image.open(BytesIO(original_bytes)).convert('RGB')
baseline = render()
print('Baseline size:', baseline.size, 'original:', original.size, flush=True)
print('Baseline equal:', np.array_equal(np.asarray(baseline), np.asarray(original)), flush=True)
assert np.array_equal(np.asarray(baseline), np.asarray(original)), 'Baseline must reproduce the submitted image exactly'

# Preserve the original canvas so this typography correction cannot rescale
# the neighbouring panels when the complete figure is placed in a document.
fig.set_dpi(600)
fig.canvas.draw()
original_bbox = fig.get_tightbbox(fig.canvas.get_renderer()).padded(0.03)
assert np.array_equal(np.asarray(render(original_bbox)), np.asarray(original))

title = fig.axes[-1]._left_title
assert title.get_text() == r'$\it{POLE}$/$\it{POLD1}$ · primary SBS10 score'
assert title.get_fontweight() == 'bold'
# Use the same family and point size as the two adjacent panel-c titles.
title_settings = []
for axis in fig.axes[-3:]:
    t = axis._left_title
    assert t.get_fontfamily() == ['Liberation Sans']
    assert t.get_fontsize() == 8.0 and t.get_fontweight() == 'bold'
    title_settings.append({'text': t.get_text(), 'family': t.get_fontfamily(),
                           'size_pt': t.get_fontsize(),
                           'font_file': findfont(t.get_fontproperties(), fallback_to_default=False)})
# Only this title uses custom math fonts; the other labels remain unchanged.
matplotlib.rcParams['mathtext.it'] = 'Liberation Sans:style=italic:weight=bold'
matplotlib.rcParams['mathtext.rm'] = 'Liberation Sans:style=italic:weight=bold'
title.set_math_fontfamily('custom')
parsed = MathTextParser('path').parse(r'$\it{POLE}$/$\it{POLD1}$',
    prop=FontProperties(size=8, math_fontfamily='custom'))
gene_fonts = [font.postscript_name for font, size, char, x, y in parsed.glyphs if chr(char) != '/']
assert set(gene_fonts) == {'LiberationSans-BoldItalic'}, gene_fonts
assert all(size == 8.0 for font, size, char, x, y in parsed.glyphs), parsed.glyphs
updated = render(original_bbox)
print('Updated size:', updated.size, flush=True)
updated.save(OUT / 'Figure_4_final.tif', compression='tiff_lzw', dpi=(600, 600))
preview = updated.copy()
preview.thumbnail((1600, 1800))
preview.save(OUT / 'figure4_final.png')
plt.close(fig)
assert updated.size == original.size
change = np.any(np.asarray(updated) != np.asarray(original), axis=2)
ys, xs = np.where(change)
bounds = [int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1]
print('Changed pixel bounds:', bounds, flush=True)
# The bottom-right panel title is the only permitted raster change.
assert bounds[0] > 2600 and 1850 < bounds[1] < bounds[3] < 2000, bounds
report = {'source_zip': 'JPCR_submit.zip', 'figure_change': 'Panel c POLE/POLD1 bold italic',
          'baseline_reproduced_exactly': True, 'changed_pixel_bounds': bounds,
          'gene_fonts': gene_fonts, 'original_size': original.size, 'size': updated.size,
          'panel_c_title_settings': title_settings}
(OUT / 'figure4_final_check.json').write_text(json.dumps(report, indent=2))
fixed_bytes = (OUT / 'Figure_4_final.tif').read_bytes()
with ZipFile(ROOT / 'JPCR_submit.zip') as source, ZipFile(ROOT / 'JPCR_submit_final.zip', 'w', ZIP_DEFLATED, compresslevel=6) as target:
    for name in source.namelist():
        target.writestr(name, fixed_bytes if name == 'Figure_4.tif' else source.read(name))
with ZipFile(ROOT / 'JPCR_submit.zip') as source, ZipFile(ROOT / 'JPCR_submit_final.zip') as target:
    assert target.testzip() is None
    assert target.namelist() == source.namelist()
    changed = [name for name in source.namelist() if source.read(name) != target.read(name)]
    assert changed == ['Figure_4.tif'], changed
print('PASS: Figure_4.tif only; all 10 submission filenames retained.')
