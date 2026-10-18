"""Compare saved/public SBS targets with the current, unchanged manuscript scores.

Run from the workspace with /opt/anaconda/bin/python rev/signature_compare.py.
No target refitting, image-model fitting, threshold tuning, or subtype summation.
"""
from pathlib import Path
import hashlib
import json
import re
import zipfile

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


def local_method(name, template, note, mapping=LOCAL, cptac='CPTAC', corrected_brca=False):
    for cohort in ['TCGA', 'CPTAC']:
        for cancer in LOCAL:
            p = template.format(cohort=cohort if cohort == 'TCGA' else cptac, local=mapping[cancer])
            if corrected_brca and cohort == 'TCGA' and cancer == 'BRCA':
                p = 'sigfit_poulos2018_20260726_corrected_nonpreferred/model_inputs/TCGA/Brca/sbs30_targets.tsv'
            compare(name, cohort, cancer, read(p), p, note)


local_method('deconstruct_exome2genome', 'deconstructsigs_legacy30_exome2genome_full_paper_20260828/process_endpoint_inputs_legacy30_min50_exome2genome/{cohort}/{local}/{cohort}_{local}_process_endpoints.xlsx',
             'Saved legacy SBS1-30 targets; changes trinucleotide normalization to exome2genome; retain saved proportions and cutoff.')
local_method('sigfit_legacy30', 'sigfit_poulos2018_20260726/fits/{cohort}/{local}/patient_weights.tsv',
             'Local sigfit 1.2.0 reproduction using exome-corrected legacy30 reference; dense posterior means; input/sample rules also differ. TCGA BRCA uses verified 432-patient corrected model-input table only.', corrected_brca=True)
local_method('SPA_v2', 'sigprofiler_mg_assignment_v2_sbs30_currentmaf_v2_27_20260904/labels/{cohort}/{local}/{cohort}_{local}_SBS1_30.tsv',
             'Saved SPA1.1.5 COSMICv2 WES SBS1-30; MatrixGenerator1.3.6 catalogue. Catalogue and eligible sample changes mean this is not an algorithm-only comparison.')
for version, directory in [('3.5', 'sigprofilerassignment_v35_currentmaf_v2_27_20260903'), ('3.6', 'sigprofilerassignment_v115_v36_currentmaf_v2_27_20260903')]:
    local_method('SPA_v' + version, directory + '/labels/{cohort}/{local}/{cohort}_{local}_process_endpoints.xlsx',
                 'Saved COSMICv' + version + ' WES proportions; same-name signatures only, not identical reference profiles; split legacy signatures not summed.')
for caller in ['union', 'dual']:
    for prefix, directory, filename in [('SPA_standardized', 'all_sbs_standardized_v35_20260729', 'all_sbs_proportions.tsv'), ('SPSS', 'all_sbs_spss_v35_20260729', 'spss_proportions.tsv')]:
        # TCGA input is identical between caller branches; report it once.
        for cohort in (['TCGA', 'CPTAC'] if caller == 'union' else ['CPTAC']):
            for cancer in LOCAL:
                branch = cohort if cohort == 'TCGA' else 'CPTAC_' + caller
                p = f'{directory}/fits/{branch}/{SHORT[cancer]}/{filename}'
                compare(prefix + '_v3.5_' + caller, cohort, cancer, read(p), p,
                        'Saved GRCh38 WES catalogue; ' + ('TCGA GDC ensemble' if cohort == 'TCGA' else 'CPTAC caller ' + caller) + '. Full COSMICv3.5 reference; same-name comparisons only. Catalogue/sample definitions differ from manuscript.')

public_path = 'rev/published_tcga_author.csv'
public = read(public_path)
public = public[public['Sample Names'].str[13:15].eq('01')].copy()
public['PATIENT'] = public['Sample Names'].str[:12]
assert public.PATIENT.is_unique
public_cols = [c for c in public if re.fullmatch(r'SBS\d+[a-z]?', c)]
assert len(public_cols) == 65
public[public_cols] = public[public_cols].div(public[public_cols].sum(axis=1), axis=0)
public_cancers = dict(BRCA='Breast-cancer', HNSC='Head-SCC', KIRC='Kidney-RCC', LUAD='Lung-AdenoCa', LUSC='Lung-SCC', OV='Ovary-AdenoCa', UCEC='Uterus-AdenoCa')
for cancer, label in public_cancers.items():
    compare('published_Alexandrov2020', 'TCGA', cancer, public[public['Cancer Types'].eq(label)], public_path,
            'Published TCGA WES SigProfiler activities, downloaded from author repository; byte-identical second mirror. Primary01 sample only. Divide by all65 activities, no6%cutoff or split-signature summation.',
            'https://www.nature.com/articles/s41586-020-1943-3 ; https://raw.githubusercontent.com/judithabk6/Clonesig_analysis/master/external_data/TCGA_WES_sigProfiler_SBS_signatures_in_samples.csv')

with zipfile.ZipFile(OUT / 'published_poulos.zip') as z:
    luad = pd.concat([pd.read_csv(z.open('SI Fig5a Underlying Data (LUAD ' + group + ').txt'), sep='\t') for group in ['mutant', 'wildtype']])
luad = luad.rename(columns={'SampleID': 'PATIENT', 'Signature 4': 'SBS4'})
luad['PATIENT'] = luad.PATIENT.str[:12]
compare('published_Poulos2018_LUAD4', 'TCGA', 'LUAD', luad, 'rev/published_poulos.zip',
        'Published patient-identified LUAD SBS4 proportions from Fig5a (both groups). Full30x7815 S2 matrix has cancer labels but no sample IDs; do not assign IDs by row/column order.',
        'https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1007779')

data = pd.DataFrame(metrics)
for (method, _), idx in data.groupby(['method', 'cohort']).groups.items():
    if method == 'published_Poulos2018_LUAD4':
        continue  # One available published target, not a full-panel screen.
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
    statistics='Pearson r; Fisher 95% CI; positive-direction raw P; BH within each source/cohort planned210 screen, NE treated as1 for adjustment and displayed missing. Single published LUAD4 comparison has raw P only.',
    reference_mapping='Only exact SBS names compared; changed profiles are not assumed identical. Missing/split signatures are not summed or filled with zero.',
    interpretation='These are target-source sensitivity comparisons, not retrained-model performance estimates. Source catalogues, eligibility, normalization and sparsity can differ.',
    public_TCGA_mirror='https://raw.githubusercontent.com/xsliulab/Pan-cancer_CNA_signature/main/data/PCAWG/TCGA_WES_sigProfiler_SBS_signatures_in_samples.csv',
    public_TCGA_mirrors_identical=(OUT/'published_tcga_author.csv').read_bytes()==(OUT/'published_tcga_mirror.csv').read_bytes(),
    public_Poulos_download='https://journals.plos.org/plosgenetics/article/file?type=supplementary&id=info:doi/10.1371/journal.pgen.1007779.s010',
    published_CPTAC='Dou2020 method identified, but patient-level signature matrix not obtained/verified; no published CPTAC exposure comparison is claimed.',
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
