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

## Repository folders

| Folder | Contents |
| --- | --- |
| `models/` | Model fitting, bag construction, prediction and fixed averaging |
| `preprocessing/` | Slide processing and RetCCL/CTransPath feature extraction |
| `signatures/` | deconstructSigs, sigfit and SigProfiler target estimation |
| `data/` | Supplementary Data S1–S3 and the 56 alternative target tables |
| `records/` | Original model protocols, selection records and signature-software versions |
| `rev/` | Cohort summaries, statistics, figures, clinical and input-sensitivity analyses |

[WORKFLOW.md](WORKFLOW.md) maps each manuscript analysis to its implementation. [DATA.md](DATA.md) describes included data and external inputs. [ENVIRONMENT.md](ENVIRONMENT.md) lists the recorded software environments. `source_manifest.tsv` maps distributed files to their original paths and checksums.

Training and preprocessing require the source mutation files, WSIs, feature caches and checkpoints described in WORKFLOW.md. The CPU result-level check uses the included derived data and does not rerun training.

Upstream notices are retained in `Marugoto_LICENSE.txt`, `preprocessing/LICENSE`, `preprocessing/RetCCL_LICENSE.md` and `preprocessing/CTransPath_LICENSE.md`.
