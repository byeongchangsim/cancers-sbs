"""Recalculate manuscript statistics from the included derived patient data.

Run from any directory with Python 3.8+ and requirements.txt installed.
This command does not train models or change the supplied results.
"""
from pathlib import Path
import json
import hashlib
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, mannwhitneyu

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'rev'
OUT = ROOT / 'results'
SBS = [f'SBS{i}' for i in range(1, 31)]


def read(name):
    return pd.read_csv(DATA / (name + '.tsv'), sep='\t')


def bh(p):
    p = np.asarray(p, float)
    valid = np.isfinite(p)
    v = np.where(valid, p, 1.)
    order = np.argsort(v)
    q = np.empty(len(v))
    q[order] = np.minimum(1, np.minimum.accumulate(
        (v[order] * len(v) / np.arange(1, len(v) + 1))[::-1])[::-1])
    q[~valid] = np.nan
    return q


def effect(y, s, signature_comparison=False):
    y, s = np.asarray(y, float), np.asarray(s, float)
    if len(y) < 4 or np.ptp(y) == 0 or np.ptp(s) == 0:
        return dict(n=len(y), r=np.nan, p=np.nan, ci_low=np.nan, ci_high=np.nan)
    r, p = pearsonr(y, s)
    epsilon = 1e-9 if signature_comparison else 1e-15
    z = np.arctanh(np.clip(r, -1+epsilon, 1-epsilon))
    lo, hi = np.tanh(z + np.array([-1, 1]) * 1.959963984540054 / np.sqrt(len(y)-3))
    return dict(n=len(y), r=r, p=p/2 if r >= 0 else 1-p/2, ci_low=lo, ci_high=hi)


def compare(actual, expected, keys, columns):
    d = actual.merge(expected, on=keys, suffixes=('', '_expected'), validate='one_to_one')
    assert len(d) == len(actual) == len(expected), (keys, len(d), len(actual), len(expected))
    for a, b in columns.items():
        target = b + '_expected' if b in actual.columns else b
        np.testing.assert_allclose(d[a], d[target], rtol=1e-9, atol=1e-10,
                                   equal_nan=True, err_msg=a)


def main():
    manifest = pd.read_csv(ROOT / 'source_manifest.tsv', sep='\t')
    for r in manifest.itertuples():
        assert hashlib.sha256((ROOT / r.file).read_bytes()).hexdigest() == r.sha256, r.file
    patients = read('patient_scores')
    assert not patients.duplicated(['cohort', 'cancer', 'PATIENT']).any()
    assert patients.groupby('cohort').size().to_dict() == {'CPTAC': 580, 'TCGA': 2741}
    rows = []
    for (co, cancer), d in patients.groupby(['cohort', 'cancer']):
        for sig in SBS:
            m = effect(d[sig], d['prediction_' + sig])
            rho = spearmanr(d[sig], d['prediction_'+sig]).statistic if np.isfinite(m['r']) else np.nan
            rows.append(dict(dataset=co, cancer=cancer, signature=sig, spearman_rho=rho, **m))
    primary = pd.DataFrame(rows)
    primary['q_bh210'] = np.nan
    for _, idx in primary.groupby('dataset').groups.items():
        assert len(idx) == 210
        primary.loc[idx, 'q_bh210'] = bh(primary.loc[idx, 'p'])
    compare(primary, read('cohort_metrics'), ['dataset', 'cancer', 'signature'],
            dict(n='n', r='pearson_r', p='pearson_p_positive_direction',
                 ci_low='ci_low', ci_high='ci_high', q_bh210='q_bh210', spearman_rho='spearman_rho'))
    wide = primary.pivot(index=['cancer', 'signature'], columns='dataset', values=['r', 'p', 'q_bh210'])
    nominal = (wide['p'] < .05).all(axis=1)
    adjusted = (wide['q_bh210'] < .20).all(axis=1)
    assert int(nominal.sum()) == 14 and int((nominal & adjusted).sum()) == 9

    folds = read('fold_assignments')
    for (cancer, seed, fold), d in folds.groupby(['cancer', 'seed_index', 'fold']):
        assert d.PATIENT.is_unique
        assert set(d.role) == {'train', 'validation', 'test'}
        expected = patients[(patients.cohort == 'TCGA') & (patients.cancer == cancer)].PATIENT
        assert set(d.PATIENT) == set(expected)
    assert folds.groupby(['cancer', 'seed_index', 'fold']).ngroups == 105

    restricted = read('tumour_scores')
    assert len(restricted) == 570
    sensitivity = []
    for cancer, d in restricted.groupby('cancer'):
        all_slide = patients[(patients.cohort == 'CPTAC') & (patients.cancer == cancer)]
        paired = all_slide.merge(d, on='PATIENT', suffixes=('_all', '_tumour'), validate='one_to_one')
        for sig in SBS:
            a = effect(paired[sig], paired['prediction_'+sig+'_all'])
            t = effect(paired[sig], paired['prediction_'+sig+'_tumour'])
            sensitivity.append(dict(cancer=cancer, signature=sig,
                                    **{'all_'+k:v for k,v in a.items()},
                                    **{'tumour_'+k:v for k,v in t.items()}, delta_r=t['r']-a['r']))
    sensitivity = pd.DataFrame(sensitivity)
    sensitivity['tumour_q_bh210'] = bh(sensitivity.tumour_p)
    cols = [c for c in sensitivity if c not in ['cancer', 'signature']]
    compare(sensitivity, read('tumour_sensitivity'), ['cancer', 'signature'], {c:c for c in cols})

    clinical = []
    for (co, endpoint, rule), d in read('clinical_revision_scores').groupby(['cohort', 'endpoint', 'rule']):
        primary_scores = patients[(patients.cohort == co) & (patients.cancer == 'UCEC')].set_index('PATIENT')
        signature = 'SBS6' if endpoint == 'MSI' else 'SBS10'
        np.testing.assert_allclose(d.score, d.PATIENT.map(primary_scores['prediction_'+signature]), rtol=0, atol=1e-14)
        pos, neg = d.loc[d.label == 1, 'score'].to_numpy(), d.loc[d.label == 0, 'score'].to_numpy()
        wins = (pos[:,None] > neg).astype(float) + .5 * (pos[:,None] == neg)
        seed = 20261011 if co == 'TCGA' and endpoint == 'MSI' and rule == 'Original gray-zone exclusion' else 20261013
        rng = np.random.default_rng(seed)
        draws = [wins[np.ix_(rng.integers(len(pos), size=len(pos)),
                            rng.integers(len(neg), size=len(neg)))].mean() for _ in range(10000)]
        lo, hi = np.quantile(draws, [.025, .975])
        clinical.append(dict(cohort=co, endpoint=endpoint, rule=rule, n=len(d), positive_n=len(pos),
                             auc=wins.mean(), ci_low=lo, ci_high=hi,
                             p=mannwhitneyu(pos, neg, alternative='two-sided').pvalue))
    clinical = pd.DataFrame(clinical)
    compare(clinical, read('clinical_revision'), ['cohort', 'endpoint', 'rule'],
            {c:c for c in ['n', 'positive_n', 'auc', 'ci_low', 'ci_high', 'p']})

    s2 = ROOT/'data/Supplementary_Data_S2.xlsx'
    clinical_patients = pd.read_excel(s2, sheet_name='patient_scores', header=3)
    clinical_reported = pd.read_excel(s2, sheet_name='clinical_results', header=3)
    main_clinical = []
    for endpoint in ['HRD', 'MANTIS-defined MSI', 'POLE/POLD1']:
        d = clinical_patients[clinical_patients['Endpoint'] == endpoint]
        score = d['Image-derived score'].to_numpy()
        label = d['Endpoint label (0/1)'].to_numpy()
        if endpoint == 'HRD':
            r, p = pearsonr(score, d['Genomic-scar HRD score'])
            main_clinical.append((r, p))
        pos, neg = score[label == 1], score[label == 0]
        auc = ((pos[:,None] > neg).astype(float) + .5*(pos[:,None] == neg)).mean()
        main_clinical.append((auc, mannwhitneyu(pos, neg, alternative='two-sided').pvalue))
    np.testing.assert_allclose(np.array(main_clinical),
                               clinical_reported[['Estimate', 'P (two-sided)']], rtol=1e-9, atol=1e-10)

    alternatives = []
    sources = pd.read_csv(ROOT / 'data/alternative_sources.tsv', sep='\t')
    for src in sources.itertuples():
        p = ROOT / src.file
        target = pd.read_excel(p) if p.suffix == '.xlsx' else pd.read_csv(p, sep='\t')
        target = target.rename(columns={'sample':'PATIENT'})
        base = patients[(patients.cohort == src.cohort) & (patients.cancer == src.cancer)]
        d = base.merge(target, on='PATIENT', suffixes=('', '_alt'), validate='one_to_one')
        assert len(d) == src.matched_patients
        for sig in SBS:
            row = dict(method=src.method, cohort=src.cohort, cancer=src.cancer, signature=sig,
                       n=0, original_r=np.nan, original_p_one_sided=np.nan,
                       original_ci_low=np.nan, original_ci_high=np.nan,
                       alternative_r=np.nan, alternative_p_one_sided=np.nan,
                       alternative_ci_low=np.nan, alternative_ci_high=np.nan, delta_r=np.nan)
            if sig in target:
                vals = d[[sig, sig+'_alt', 'prediction_'+sig]].dropna()
                row['n'] = len(vals)
                for label, col in [('original',sig), ('alternative',sig+'_alt')]:
                    m = effect(vals[col], vals['prediction_'+sig], signature_comparison=True)
                    row.update({label+'_'+('p_one_sided' if k == 'p' else k):v
                                for k,v in m.items() if k != 'n'})
                row['delta_r'] = row['alternative_r'] - row['original_r']
            alternatives.append(row)
    alternatives = pd.DataFrame(alternatives)
    for _, idx in alternatives.groupby(['method', 'cohort']).groups.items():
        assert len(idx) == 210
        for label in ['original', 'alternative']:
            alternatives.loc[idx, label+'_q_bh210'] = bh(alternatives.loc[idx, label+'_p_one_sided'])
    expected = read('signature_metrics')
    expected = expected[expected.method.isin(sources.method)]
    keys = ['method', 'cohort', 'cancer', 'signature']
    compare(alternatives, expected, keys, {c:c for c in alternatives if c not in keys})

    # Check the actual submitted workbook, not only intermediate result tables.
    submitted = pd.read_excel(ROOT/'data/Supplementary_Data_S3.xlsx', sheet_name='paired_metrics', header=3)
    for co in ['TCGA', 'CPTAC']:
        tab = submitted[['Cancer type', 'SBS signature', co+'\nPearson r', co+'\nBH-adjusted P']].copy()
        tab.columns = ['cancer', 'signature', 'r', 'q_bh210']
        compare(primary[primary.dataset == co], tab, ['cancer', 'signature'], dict(r='r', q_bh210='q_bh210'))
    OUT.mkdir(exist_ok=True)
    for name, frame in [('cohort_metrics', primary), ('tumour_sensitivity', sensitivity),
                        ('clinical_revision', clinical), ('signature_comparison', alternatives)]:
        frame.to_csv(OUT/(name+'.tsv'), sep='\t', index=False)
    report = dict(source_files_checked=len(manifest), patients=dict(TCGA=2741, CPTAC=580),
                  primary_associations=210, original_nominal_associations=int(nominal.sum()),
                  original_associations_BH_below_020_in_each_cohort=int((nominal & adjusted).sum()),
                  training_contexts_checked=105, paired_sensitivity_patients=570,
                  clinical_analyses_checked=4, clinical_bootstrap_replicates=10000,
                  primary_clinical_estimates_checked=4,
                  signature_comparisons_checked=len(alternatives), submitted_workbook_checked=True,
                  model_training_rerun=False)
    (OUT/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='invalid value encountered in cast')
        main()
