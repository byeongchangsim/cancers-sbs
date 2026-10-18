"""Compare saved/public SBS targets with the current, unchanged manuscript scores.

Run from the workspace with /opt/anaconda/bin/python rev/signature_compare.py.
No target refitting, image-model fitting, threshold tuning, or subtype summation.
"""
from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'rev'
SBS = [f'SBS{i}' for i in range(1, 31)]
LOCAL = dict(BRCA='Brca', HNSC='Hnscc', KIRC='Ccrcc', LUAD='Luad', LUSC='Lscc', OV='Ov', UCEC='Ucec')
SHORT = dict(BRCA='Brca', HNSC='Hnsc', KIRC='Kirc', LUAD='Luad', LUSC='Lusc', OV='Ov', UCEC='Ucec')
base = pd.read_csv(OUT / 'patient_scores.tsv', sep='\t')
assert not base.duplicated(['cohort', 'cancer', 'PATIENT']).any()
chosen = pd.read_csv(OUT / 'paired_metrics.tsv', sep='\t')
original14 = set(map(tuple, chosen.loc[chosen.nominal_positive_both, ['cancer', 'signature']].values))
sources, metrics, matches = [], [], []


def read(path):
    p = ROOT / path
    return pd.read_excel(p) if p.suffix == '.xlsx' else pd.read_csv(p, sep='\t' if p.suffix == '.tsv' else ',')


def corr(x, y):
    if len(x) < 4 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return [np.nan] * 4
    # Match rev/analyze.py; this environment's one-sided pearsonr uses abs(r).
    r, p_two = pearsonr(x, y)
    p = p_two / 2 if r >= 0 else 1 - p_two / 2
    lo, hi = np.tanh(np.arctanh(np.clip(r, -.999999999, .999999999)) + np.array([-1, 1]) * 1.959963984540054 / np.sqrt(len(x) - 3))
    return r, p, lo, hi


def compare(method, cohort, cancer, targets, path, note, url=''):
    targets = targets.rename(columns={'sample': 'PATIENT'})
    assert targets.PATIENT.is_unique, (method, cohort, cancer)
    sbcols = [s for s in targets if re.fullmatch(r'SBS\d+[a-z]?', s)]
    arr = targets[sbcols].to_numpy(float)
    assert np.isfinite(arr).all() and arr.min() >= -1e-8 and arr.max() <= 1 + 1e-8
    original = base[(base.cohort == cohort) & (base.cancer == cancer)]
    merged = original.merge(targets[['PATIENT'] + sbcols], on='PATIENT', suffixes=('', '_alt'), validate='one_to_one')
    sums = targets[sbcols].sum(axis=1)
    sources.append(dict(method=method, cohort=cohort, cancer=cancer, file=str(path),
                        sha256=hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),
                        source_patients=len(targets), matched_patients=len(merged), original_patients=len(original),
                        source_signatures=len(sbcols), signature_sum_min=sums.min(), signature_sum_max=sums.max(),
                        note=note, url=url))
    for patient in merged.PATIENT:
        matches.append(dict(method=method, cohort=cohort, cancer=cancer, PATIENT=patient))
    for sig in SBS:
        row = dict(method=method, cohort=cohort, cancer=cancer, signature=sig,
                   original14=(cancer, sig) in original14, matched_patients=len(merged),
                   n=0, status='signature not in source; no subtype aggregation')
        if sig in sbcols:
            d = merged[[sig, sig + '_alt', 'prediction_' + sig]].dropna()
            x, y, score = (d[c].to_numpy(float) for c in d)
            row.update(n=len(d), status='estimable' if len(d) >= 4 and np.ptp(x) and np.ptp(y) and np.ptp(score) else 'constant or insufficient values',
                       original_nonzero=int((x > 0).sum()), alternative_nonzero=int((y > 0).sum()),
                       exposure_r=corr(x, y)[0],
                       exposure_rho=spearmanr(x, y).statistic if len(d) >= 4 and np.ptp(x) and np.ptp(y) else np.nan,
                       mean_absolute_difference=float(np.mean(np.abs(x-y))) if len(d) else np.nan)
            for label, values in [('original', x), ('alternative', y)]:
                row.update(zip([label + '_r', label + '_p_one_sided', label + '_ci_low', label + '_ci_high'], corr(score, values)))
            row['delta_r'] = row['alternative_r'] - row['original_r']
        metrics.append(row)


for src in pd.read_csv(ROOT / 'data/alternative_sources.tsv', sep='\t').itertuples():
    compare(src.method, src.cohort, src.cancer, read(src.file), src.file, src.note)


data = pd.DataFrame(metrics)
for (method, _), idx in data.groupby(['method', 'cohort']).groups.items():
    # Complete planned 210 pair family, including absent/nonestimable targets as p=1.
    for arm in ['original', 'alternative']:
        p = data.loc[idx, arm + '_p_one_sided'].fillna(1).to_numpy()
        order = np.argsort(p)
        adjusted = np.minimum.accumulate((p[order] * 210 / np.arange(1, len(p)+1))[::-1])[::-1].clip(0, 1)
        q = np.empty(len(p)); q[order] = adjusted
        q[data.loc[idx, arm + '_p_one_sided'].isna().to_numpy()] = np.nan
        data.loc[idx, arm + '_q_bh210'] = q
summary = []
for (method, cohort), d in data.groupby(['method', 'cohort'], sort=False):
    eligible = d[d.alternative_r.notna() & d.original_r.notna()]
    f = eligible[eligible.original14]
    summary.append(dict(method=method, cohort=cohort, matched_patients=sum(s['matched_patients'] for s in sources if s['method']==method and s['cohort']==cohort),
                        comparable_pairs=len(eligible), exposure_r_median=eligible.exposure_r.median(),
                        original14_comparable=len(f), original14_positive=int(f.alternative_r.gt(0).sum()),
                        original14_original_nominal=int(f.original_p_one_sided.lt(.05).sum()),
                        original14_alternative_nominal=int(f.alternative_p_one_sided.lt(.05).sum()),
                        original14_exposure_r_median=f.exposure_r.median(),
                        original14_delta_r_median=f.delta_r.median(), original14_max_abs_delta_r=f.delta_r.abs().max()))
summary = pd.DataFrame(summary)
data.to_csv(OUT / 'signature_metrics.tsv', sep='\t', index=False)
pd.DataFrame(matches).to_csv(OUT / 'signature_matches.tsv', sep='\t', index=False)
pd.DataFrame(sources).to_csv(OUT / 'signature_sources.tsv', sep='\t', index=False)
summary.to_csv(OUT / 'signature_summary.tsv', sep='\t', index=False)
(OUT / 'signature_provenance.json').write_text(json.dumps(dict(
    baseline_scores='rev/patient_scores.tsv',
    baseline_sha256=hashlib.sha256((OUT / 'patient_scores.tsv').read_bytes()).hexdigest(),
    original14_source='rev/paired_metrics.tsv',
    original14_sha256=hashlib.sha256((OUT / 'paired_metrics.tsv').read_bytes()).hexdigest(),
    comparison='Fixed current manuscript scores; one-to-one patient matching within cohort and cancer; same matched patients for original and alternative targets.',
    statistics='Pearson r; Fisher 95% CI; positive-direction raw P; BH within each source/cohort planned210 screen, NE treated as1 for adjustment and displayed missing.',
    reference_mapping='Only exact SBS names compared; changed profiles are not assumed identical. Missing/split signatures are not summed or filled with zero.',
    interpretation='These are target-source sensitivity comparisons, not retrained-model performance estimates. Source catalogues, eligibility, normalization and sparsity can differ.',
), indent=2))
with pd.ExcelWriter(OUT / 'Signature_compare.xlsx', engine='xlsxwriter') as writer:
    for sheet, frame in [('Summary', summary), ('Comparisons', data), ('Sources', pd.DataFrame(sources))]:
        frame.to_excel(writer, sheet_name=sheet, index=False)
        ws = writer.sheets[sheet]
        ws.freeze_panes(1, 4 if sheet == 'Comparisons' else 2)
        ws.autofilter(0, 0, len(frame), len(frame.columns)-1)
        ws.set_column(0, 0, 30)
        ws.set_column(1, len(frame.columns)-1, 18)
        if sheet == 'Sources':
            ws.set_column(3, 3, 65)
            ws.set_column(len(frame.columns)-2, len(frame.columns)-1, 85)
print(summary.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
