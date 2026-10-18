# Data and inputs

## Included derived data

Patient identifiers are the source repository identifiers used for matching. TCGA and CPTAC, patients and genomic samples, and patients and source slides are distinguished in the tables.

| File | Contents |
| --- | --- |
| `rev/patient_scores.tsv` | 2,741 TCGA and 580 CPTAC patients, original SBS1–SBS30 exposures and primary standardised scores |
| `rev/fold_assignments.tsv` | Train, inner-validation and outer-test roles in 105 cancer/seed/fold contexts |
| `rev/slide_manifest.tsv` | 5,135 TCGA and 2,239 CPTAC source slides with patient links and specimen annotations |
| `rev/tumour_scores.tsv` | Restricted-input scores for 570 CPTAC patients |
| `rev/clinical_revision_scores.tsv` | Primary SBS6/SBS10 scores and endpoint labels for the revision comparisons. Rows repeat when a patient participates in both MANTIS definitions |
| `rev/mutation_counts.tsv` | Genomic-sample substitution counts and eligibility indicators |
| `data/alternative_sources.tsv` | Index of the 56 method/cohort/cancer target tables stored alongside it in `data/` |
| `rev/attention_cases.tsv`, `rev/attention_tiles.tsv` | Eight displayed cases, selected source/model identifiers, sampled coordinates and attention weights |
| `data/Supplementary_Data_S1.xlsx` | Current model settings and cohort information |
| `data/Supplementary_Data_S2.xlsx` | Clinical definitions, results, patient scores and qualifying polymerase substitutions |
| `data/Supplementary_Data_S3.xlsx` | Current complete screen, sensitivity, distributions, specimen summaries and alternative-target comparisons |

The checked manuscript reports cohort-specific Benjamini–Hochberg adjustment across 210 tests. Some original intermediate files also preserve earlier exploratory joint-adjustment or Holm columns. These are historical calculations. The current workbooks and `reproduce.py` identify the statistics reported in the revision. Clinical P values reported in the manuscript are unadjusted, two-sided Mann–Whitney U tests.

## Source repositories

TCGA genomic and image data are obtained through the [NCI Genomic Data Commons](https://portal.gdc.cancer.gov/). CPTAC images are obtained through [The Cancer Imaging Archive](https://www.cancerimagingarchive.net/), and the recorded mutation-data source is [Zenodo 8394329](https://doi.org/10.5281/zenodo.8394329). CPTAC UCEC clinical/mutation queries are documented in `rev/cptac_endpoint_source.json`. The slide manifest retains annotation origins and source identifiers.

Raw WSIs, mutation-event files, pretrained encoder weights, patient feature bags and fitted checkpoints are not included. Training and image-level replay require these files. Checkpoint hashes and selection records identify the models used in the reported analyses. The code repository does not change the original data repositories' access or redistribution conditions.

Alternative target sources retain their distinct sample coverage and reference definitions. Equal signature names across reference versions do not guarantee equal profiles. The code uses the same matched patients within each original-versus-alternative comparison and does not sum split signatures.

## Publication status

The repository is private while the authors prepare the release. It will be made public before acceptance, in accordance with Reviewer 2, comment 8. Public release should retain the code, recorded environments, preprocessing details, fold assignments, manifests and shareable derived results. The corresponding repository link can then be added to the manuscript and response letter.
