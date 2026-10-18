# Analysis workflow

## Manuscript correspondence

| Manuscript analysis | Implementation | Inputs and records |
| --- | --- | --- |
| Methods 2.1, cohort and slide composition | `rev/cohorts.py` | `rev/slide_manifest.tsv`, `rev/eligibility_audit.tsv`, `rev/patient_flow.tsv` |
| Methods 2.2, original SBS1–SBS30 exposures | `signatures/extract_cptac_sbs.R` and the TCGA/CPTAC shell drivers | `signatures/paper_deconstructsigs_configs.tsv`, `rev/mutation_counts.tsv` |
| Methods 2.3, image processing | `preprocessing/wsi-norm.py`, `preprocessing/wsi_cache_tiles.py` | Normalisation template and external source WSIs |
| Methods 2.3, frozen encoder features | `preprocessing/extract_end2end_retccl_qc_v1.py`, `preprocessing/extract_ctranspath_from_existing_cache_20260527.py` | Coordinate-aligned RetCCL and CTransPath features |
| Methods 2.4, MIL fitting and variant selection | `models/run_full210_exact25_multiscale_fusion_20260904.py` | `records/LOCKED_PROTOCOL.json`, `records/FROZEN_INNER_SELECTION.tsv` |
| Methods 2.4, final fixed ensemble | `models/build_full210_equal_model_average_20260904.py` | `records/METHOD_PROTOCOL.json`, `rev/fold_assignments.tsv` |
| Methods 2.5, MSI assessment | `rev/clinical_final.py` | Primary SBS6 scores in `rev/patient_scores.tsv` and retained MSI labels |
| Methods 2.5–2.6, clinical endpoint evaluation | `rev/analyze.py`, `rev/fix_endpoint.py`, `rev/clinical_final.py` | Data S2, `rev/clinical_revision_scores.tsv`, `rev/cptac_endpoint_source.json` |
| Methods 2.6, full screen and BH adjustment | `rev/analyze.py`, portable `reproduce.py` | `rev/patient_scores.tsv`, `rev/cohort_metrics.tsv` |
| Methods 2.7, restricted image inputs | `rev/infer.py` and `rev/analyze.py` | `rev/tumour_scores.tsv`, `rev/input_counts.tsv`, source-slide annotations |
| Methods 2.8, attention visualisation | `rev/attention.py` | `rev/attention_cases.tsv`, `rev/attention_tiles.tsv`, original WSIs and checkpoints |
| Alternative target comparisons | `rev/signature_compare.py`, portable `reproduce.py` | `data/alternative_sources.tsv` and its referenced target tables |
| Figures 1–3 | `rev/figures.py` | `rev/patient_scores.tsv`, `rev/cohort_metrics.tsv`, `rev/paired_metrics.tsv` |
| Figure 4 | `rev/clinical_fig.py` and `rev/figures.py` | Final clinical scores and Data S2 |
| Figure S1 | `rev/s1_rows.py` | All 210 associations in `rev/paired_metrics.tsv` |
| Figures S2–S5 | `rev/supp.py`, `rev/legends.py` | Distributions, sensitivity, clinical and attention records |
| Supplementary workbook presentation | `rev/readable_data.py` | Published copies in `data/` |

## Data and execution paths

Code is grouped by function. `source_manifest.tsv` records each distributed path, its original workspace path and both checksums. Training drivers resolve the repository root from their own locations. Original relative paths for large data, feature caches and training outputs are retained so that the recorded checkpoints and manifests remain traceable. These large inputs are obtained or generated separately.

The original workspace was `/home/marugoto-1.0.0-regression`. Adjust external WSI/cache, pretrained-weight and interpreter locations in the driver constants to the local installation. RetCCL and CTransPath source code is in `preprocessing/`. The recorded `/home/end2end-WSI-preprocessing-1.0.0-preprocessing` path identifies external caches and weights.

`records/` contains the original completed-run records. A new training run creates its own protocols and selection files in its output directory. To replay the original checkpoints, restore their recorded working-data layout and selection files using the original paths in `source_manifest.tsv`.

## Execution order

1. Obtain the source mutation and image data using the identifiers in DATA.md. Generate SBS targets with the R driver. Use the `legacy30_min50` TCGA configuration and the CPTAC legacy30 driver. Both use default trinucleotide normalisation, more than 50 substitutions and the 0.06 contribution cutoff. The reference genomes differ between the recorded TCGA and CPTAC mutation sources. Preserve the genome specified by each driver.
2. Process slides, extract frozen encoder features and build patient bags with `models/build_fused_bags.py` and `models/build_fused_bags_2048.py`. Retain coordinate alignment and the recorded patient-specific random seeds. The RetCCL aggregation driver creates the patient mean features.
3. In a fresh working directory, run `models/run_full210_exact25_multiscale_fusion_20260904.py register`, then its `launch` and `freeze-selection` commands. Keep the bundled original protocol/selection records separately. These drivers intentionally refuse to overwrite existing runs. Follow the CLI arguments in that file for the available GPUs. The selected models predict all 30 signatures in each cancer/seed/fold context. The retained nested-stack module supplies context preparation and PCA-ridge helper functions to the final averaging driver. Its optional fitted stacking results are not the final manuscript ensemble. The reported ensemble uses the unweighted average in `models/build_full210_equal_model_average_20260904.py`.
4. For tumour-labelled sensitivity, run `rev/infer.py` for each cancer using the original checkpoints. Clinical assessments use the unchanged primary SBS3, SBS6 and SBS10 ensemble scores. `rev/clinical_final.py` applies the retained MSI labels to the primary SBS6 scores, without fitting another model.
5. Use `python reproduce.py` to recalculate the reported result tables from included derived data. This is the supported self-contained entry point. The original revision scripts retain intermediate workbook dependencies and their historical output locations, so they should not be run indiscriminately against a submission folder.

`rev/signature_compare.py` and its saved comparison table are restricted to the four alternatives reported in the final supplement. The comparison calculations and patient matching are unchanged. `source_manifest.tsv` records the source-file checksum in `source_sha256` and the distributed-file checksum in `sha256`. The comparison does not refit image models or combine split signatures.

Figure S5 weights describe one MIL component at sampled coordinates. They are not tissue labels or the contribution of a region to the full ensemble. Expert pathology review was not performed.

## Render the main figures

With matplotlib and the plotting dependencies listed in ENVIRONMENT.md installed, run `python rev/figures.py` for Figures 1–3 and `python rev/clinical_fig.py` for Figure 4. Both commands use the included final tables and write to `results/`. Use `--output-root PATH` to choose another output folder.
