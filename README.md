# Histology based SBS signature analysis

Code and derived data for *Deep learning-based inference of mutational signature exposures from histology across tumour types*, manuscript cancers-4641587.

The study evaluates 30 SBS signatures in seven cancer types using 2,741 TCGA patients and 580 CPTAC patients. The primary ensemble combines RetCCL mean-feature PCA-ridge regression with RetCCL/CTransPath attention MIL at two bag sizes. Its outputs are standardised SBS scores.

This repository is private during preparation. It will be made public before manuscript acceptance. Source-dataset access conditions continue to apply to data obtained from their original repositories.

## Reproduce the reported statistics

Use Python 3.8 or 3.9 in a separate environment.

```bash
python -m pip install -r requirements.txt
python reproduce.py
```

This runs on a CPU using the included patient scores. It checks the original 210-association screen, cohort-specific BH adjustment, training partitions, tumour-labelled input sensitivity, clinical comparisons and alternative signature estimates. Results are written to `results/`. The supplied data are not overwritten.

The recorded check reproduced the 14 positive associations with nominal P < 0.05 in both cohorts. Nine met BH-adjusted P < 0.20 in each cohort. It also reproduced the 570-patient sensitivity analysis, four revision clinical comparisons with 10,000 bootstrap replicates, and 1,680 original-versus-alternative signature comparisons. `verification.json` records the checks actually completed.

## Find the code and data

| Material | Location |
| --- | --- |
| Analysis steps, original commands and manuscript mapping | [WORKFLOW.md](WORKFLOW.md) |
| Included data, identifiers and external inputs | [DATA.md](DATA.md) |
| Current Supplementary Data S1–S3 | `data/` |
| Revision statistics, sensitivity, clinical assessment and attention | `rev/` |
| Model training, prediction and averaging | Root `run_full210_*` and `build_full210_*` scripts and their imported modules |
| Slide preprocessing and feature extraction | `preprocessing/`, `extract_ctranspath_from_existing_cache_20260527.py`, `sigprofiler_improved_20260818/scripts/` |
| Original SBS estimation and alternative estimates | `cptac_sigprof/`, `sigfit_poulos2018_20260726/`, `scripts/` |
| Versions and software requirements | [ENVIRONMENT.md](ENVIRONMENT.md), `rev/environment.json`, `rev/runtime_versions.tsv` |
| Original file locations and SHA-256 checksums | `source_manifest.tsv` |

Original driver names and import relationships are retained so that each reported analysis can be traced to its implementation. The drivers retain their original workspace paths. Read the path instructions in WORKFLOW.md before running extraction or training. Large WSI files, mutation files, feature caches and trained checkpoints must be obtained or generated separately. The result-level check above does not require them and does not rerun model training.

Upstream code notices are retained in `Marugoto_LICENSE.txt`, `preprocessing/LICENSE`, the RetCCL license and `TransPath/LICENSE.md`.
