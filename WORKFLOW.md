# Analysis workflow

## Manuscript correspondence

| Manuscript analysis | Implementation | Inputs and records |
| --- | --- | --- |
| Methods 2.1, cohort and slide composition | `jpcr/cohorts.py` | `rev/slide_manifest.tsv`, `rev/eligibility_audit.tsv`, `rev/patient_flow.tsv` |
| Methods 2.2, original SBS1–SBS30 exposures | `cptac_sigprof/extract_cptac_sbs.R` and the TCGA/CPTAC shell drivers | `paper_deconstructsigs_configs.tsv`, `rev/mutation_counts.tsv` |
| Methods 2.3, image processing | `preprocessing/wsi-norm.py`, `preprocessing/wsi_cache_tiles.py` | Normalisation template and external source WSIs |
| Methods 2.3, frozen encoder features | `sigprofiler_improved_20260818/scripts/extract_end2end_retccl_qc_v1.py`, `extract_ctranspath_from_existing_cache_20260527.py` | Coordinate-aligned RetCCL and CTransPath features |
| Methods 2.4, MIL fitting and variant selection | `run_full210_exact25_multiscale_fusion_20260904.py` | `full210_exact25_multiscale_fusion_20260904/LOCKED_PROTOCOL.json`, `FROZEN_INNER_SELECTION.tsv` |
| Methods 2.4, final fixed ensemble | `build_full210_equal_model_average_20260904.py` | `full210_equal_model_average_20260904/METHOD_PROTOCOL.json`, `rev/fold_assignments.tsv` |
| Methods 2.5, final MSI score | `rev/sbs6_kread.py` with representation helper `rev/sbs6_probe.py` | `rev/kread6_protocol.json`, per-seed selection tables and scores |
| Methods 2.5–2.6, clinical endpoint evaluation | `rev/analyze.py`, `rev/fix_endpoint.py`, `rev/clinical_final.py` | Data S2, `rev/clinical_revision_scores.tsv`, `rev/cptac_endpoint_source.json` |
| Methods 2.6, full screen and BH adjustment | `rev/analyze.py`, portable `reproduce.py` | `rev/patient_scores.tsv`, `rev/cohort_metrics.tsv` |
| Methods 2.7, restricted image inputs | `rev/infer.py` and `rev/analyze.py` | `rev/tumour_scores.tsv`, `rev/input_counts.tsv`, source-slide annotations |
| Methods 2.8, attention visualisation | `rev/attention.py` | `rev/attention_cases.tsv`, `rev/attention_tiles.tsv`, original WSIs and checkpoints |
| Alternative target comparisons | `rev/signature_compare.py`, portable `reproduce.py` | `data/alternative_sources.tsv` and its referenced target tables |
| Figures 1–3 | `jpcr/figure.py` | Original patient scores and figure-source tables |
| Figure 4 | `rev/clinical_fig.py` and its retained layout dependency | Final clinical scores and Data S2 |
| Figure S1 | `rev/s1_rows.py` | All 210 associations in `rev/paired_metrics.tsv` |
| Figures S2–S5 | `rev/supp.py`, `rev/legends.py` | Distributions, sensitivity, clinical and attention records |
| Supplementary workbook presentation | `rev/readable_data.py`, `rev/data_published.py` | Published copies in `data/` |

## Original execution paths

For extraction and training, set the `BASE` or `ROOT` constants in the relevant original drivers to the repository or working-data root. The recorded workspace was `/home/marugoto-1.0.0-regression`. All paths beneath it retain their original relative names in this repository when the file is included. `source_manifest.tsv` identifies every copied file.

The original preprocessing path `/home/end2end-WSI-preprocessing-1.0.0-preprocessing` corresponds to the included `preprocessing/` source directory and a separately supplied WSI/cache location. The original `/home/model_repos/TransPath` points to the CTransPath implementation and pretrained weights. Adjust these constants, the `/opt/anaconda/...` Python/R interpreter paths and external data paths to the local installation. Do this in a working copy if retaining the checksum-verified source snapshot.

## Execution order

1. Obtain the source mutation and image data using the identifiers in DATA.md. Generate SBS targets with the R driver. Use the `legacy30_min50` TCGA configuration and the CPTAC legacy30 driver. Both use default trinucleotide normalisation, more than 50 substitutions and the 0.06 contribution cutoff. The reference genomes differ between the recorded TCGA and CPTAC mutation sources. Preserve the genome specified by each driver.
2. Process slides, extract frozen encoder features and build patient bags with `build_fused_bags.py` and `build_fused_bags_2048.py`. Retain coordinate alignment and the recorded patient-specific random seeds. The RetCCL aggregation driver creates the patient mean features.
3. In a fresh working directory, run `run_full210_exact25_multiscale_fusion_20260904.py register`, then its `launch` and `freeze-selection` commands. Keep the bundled original protocol/selection records separately. These drivers intentionally refuse to overwrite existing runs. Follow the CLI arguments in that file for the available GPUs. The selected models predict all 30 signatures in each cancer/seed/fold context. The retained nested-stack module supplies context preparation and PCA-ridge helper functions to the final averaging driver. Its optional fitted stacking results are not the final manuscript ensemble. The reported ensemble uses the unweighted average in `build_full210_equal_model_average_20260904.py`.
4. For tumour-labelled sensitivity, run `rev/infer.py` for each cancer using the original checkpoints. For the clinical SBS6 readout, run `rev/sbs6_kread.py 0`, `1` and `2`, followed by `rev/sbs6_kread.py aggregate`. The kernel-ridge readout is evaluated against MSI labels after exposure-based fitting and selection. SBS10 remains the primary ensemble score.
5. Use `python reproduce.py` to recalculate the reported result tables from included derived data. This is the supported self-contained entry point. The original revision scripts retain intermediate workbook dependencies and their historical output locations, so they should not be run indiscriminately against a submission folder.

The original `rev/signature_compare.py` also contains historical exploratory sources beyond the four alternatives in the final supplement. The portable entry point restricts its comparison to the four reported methods and the same patient matches. It does not refit image models or combine split signatures. The original source remains available to document the broader analysis history.

Figure S5 weights describe one MIL component at sampled coordinates. They are not tissue labels or the contribution of a region to the full ensemble. Expert pathology review was not performed.
