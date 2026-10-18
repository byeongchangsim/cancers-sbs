"""Evaluate MSI with the unchanged primary SBS6 scores and retained labels."""
from pathlib import Path
import pandas as pd
from analyze import binary

OUT = Path(__file__).resolve().parent


def main():
    scores = pd.read_csv(OUT / 'patient_scores.tsv', sep='\t')
    scores = scores[scores.cancer.eq('UCEC')].set_index(['cohort', 'PATIENT'])
    labels = pd.read_csv(OUT / 'clinical_revision_scores.tsv', sep='\t')
    metrics = pd.read_csv(OUT / 'clinical_revision.tsv', sep='\t')
    for cohort in ['TCGA', 'CPTAC']:
        mask = labels.cohort.eq(cohort) & labels.endpoint.eq('MSI')
        labels.loc[mask, 'score'] = labels.loc[mask, 'PATIENT'].map(
            scores.loc[cohort, 'prediction_SBS6'])
        assert labels.loc[mask, 'score'].notna().all()
        for rule, frame in labels[mask].groupby('rule', sort=False):
            # Retain the original TCGA bootstrap and the revision-analysis seed.
            seed = 20261011 if cohort == 'TCGA' and rule == 'Original gray-zone exclusion' else 20261013
            result = binary(frame.label, frame.score, seed=seed)
            selected = metrics.cohort.eq(cohort) & metrics.endpoint.eq('MSI') & metrics.rule.eq(rule)
            for key, value in result.items():
                metrics.loc[selected, key] = value
    if 'p_holm_two_endpoints' in metrics:
        metrics = metrics.drop(columns='p_holm_two_endpoints')
    labels.to_csv(OUT / 'clinical_revision_scores.tsv', sep='\t', index=False)
    metrics.to_csv(OUT / 'clinical_revision.tsv', sep='\t', index=False)
    print(metrics.to_string(index=False))


if __name__ == '__main__':
    main()
