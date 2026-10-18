#!/usr/bin/env python3
"""Select process-linked scores by TCGA SBS correlation, then evaluate clinical endpoints."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from recompute_clinical_endpoints import binary_metric, continuous_metric


BASE = Path("/home/marugoto-1.0.0-regression")
PACKAGE = BASE / "paper_ready_20260901_v2_25_standardized_upgrade"
OUT = PACKAGE / "process_linked_clinical"
GLOBAL = BASE / "legacy30_retccl_ctranspath_global_weight_20260829/results/predictions"
RET_ROOT = BASE / "legacy30_triscale_target_attention_20260829/results"
CTR_ROOT = BASE / "legacy30_ctranspath_target_attention_20260829"
CTR_LOCK = pd.read_csv(CTR_ROOT / "LOCKED_SELECTION.tsv", sep="\t")
OLD_SOURCE = BASE / "paper_ready_20260817_ver2/reproducibility/source_data"
OLD_FIGURE = BASE / "paper_ready_20260817_ver2/analysis_data/direct_figure_inputs/figure4_clinical_plot_data.tsv"


def candidate_predictions(cancer: str, signature: str, dataset: str) -> pd.DataFrame:
    global_frame = pd.read_csv(
        GLOBAL / f"{cancer.lower()}_{signature.lower()}_{dataset.lower()}.tsv", sep="\t"
    ).rename(columns={"prediction": "global_score", signature: "observed"})
    frame = global_frame[["PATIENT", "observed", "global_score"]]
    for branch, column in (
        ("uniform_four_way_equal", "ret4_score"),
        ("uniform_attention_scale_mean_equal", "retatt_score"),
    ):
        source = RET_ROOT / branch / "predictions" / f"{cancer.lower()}_{signature.lower()}_{dataset.lower()}.tsv"
        one = pd.read_csv(source, sep="\t")[["PATIENT", "prediction"]].rename(columns={"prediction": column})
        frame = frame.merge(one, on="PATIENT", validate="one_to_one")
    row = CTR_LOCK[(CTR_LOCK.cancer == cancer) & (CTR_LOCK.signature == signature)].iloc[0]
    suffix = "three_seed.tsv" if dataset == "TCGA" else "label_free_three_seed.tsv"
    cancer_key = {"BRCA": "brca", "UCEC": "ucec"}[cancer]
    source = CTR_ROOT / "variants" / row.variant / "aggregated" / f"{cancer_key}_{dataset.lower()}_{suffix}"
    column = f"{row.output_type}_{signature}"
    one = pd.read_csv(source, sep="\t")[["PATIENT", column]].rename(columns={column: "ctrans_score"})
    return frame.merge(one, on="PATIENT", validate="one_to_one")


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    mappings = [
        ("BRCA", "SBS3", "HRD"),
        ("UCEC", "SBS6", "MSI/MMR"),
        ("UCEC", "SBS10", "POLE/POLD1"),
    ]
    candidates = ["global_score", "ret4_score", "retatt_score", "ctrans_score"]
    selection_rows, exposure_rows, clinical_rows, patient_parts = [], [], [], []
    selections = {}
    frames = {}
    for cancer, signature, endpoint in mappings:
        tcga = candidate_predictions(cancer, signature, "TCGA")
        frames[(cancer, signature, "TCGA")] = tcga
        scores = {column: float(pearsonr(tcga.observed, tcga[column]).statistic) for column in candidates}
        selected = sorted(scores, key=lambda column: (-scores[column], candidates.index(column)))[0]
        selections[(cancer, signature)] = selected
        for column in candidates:
            selection_rows.append({
                "cancer": cancer, "signature": signature, "endpoint": endpoint,
                "candidate": column, "tcga_exposure_pearson_r": scores[column],
                "selected": column == selected,
            })
        for dataset in ("TCGA", "CPTAC"):
            frame = tcga if dataset == "TCGA" else candidate_predictions(cancer, signature, dataset)
            frames[(cancer, signature, dataset)] = frame
            value = continuous_metric(frame[selected], frame.observed)
            exposure_rows.append({
                "analysis": f"{cancer}-{signature} selected score versus observed exposure",
                "cohort": f"{dataset} {cancer}", "selected_source": selected, **value,
            })

    clinical = pd.read_csv(OLD_SOURCE / "clinical_patient_scores.tsv", sep="\t")
    # BRCA HRD.
    selected = selections[("BRCA", "SBS3")]
    brca = frames[("BRCA", "SBS3", "TCGA")]
    hrd_cont = pd.read_csv(OLD_SOURCE / "direct_sbs3_hrd_continuous.tsv", sep="\t")[["PATIENT", "hrd_score"]]
    hrd_labels = clinical[(clinical.cohort == "TCGA BRCA") & (clinical.endpoint == "HRD")][["PATIENT", "label"]]
    hrd = hrd_cont.merge(hrd_labels, on="PATIENT", validate="one_to_one").merge(
        brca[["PATIENT", "observed", selected]], on="PATIENT", validate="one_to_one"
    ).rename(columns={selected: "score"})
    clinical_rows.extend([
        {"analysis": "BRCA SBS3 selected score versus continuous HRD", "cohort": "TCGA BRCA", "selected_source": selected, **continuous_metric(hrd.score, hrd.hrd_score)},
        {"analysis": "BRCA SBS3 selected score for HRD score >=42", "cohort": "TCGA BRCA", "selected_source": selected, **binary_metric(hrd.label, hrd.score, 101)},
    ])
    patient_parts.append(hrd.assign(cohort="TCGA BRCA", endpoint="HRD", selected_source=selected))

    # UCEC MSI/MMR.
    selected = selections[("UCEC", "SBS6")]
    for dataset in ("TCGA", "CPTAC"):
        cohort = f"{dataset} UCEC"
        labels = clinical[(clinical.cohort == cohort) & (clinical.endpoint == "MSI/MMR")][["PATIENT", "label"]]
        frame = labels.merge(
            frames[("UCEC", "SBS6", dataset)][["PATIENT", "observed", selected]],
            on="PATIENT", validate="one_to_one",
        ).rename(columns={selected: "score"})
        clinical_rows.append({
            "analysis": "UCEC SBS6 selected score for MSI/MMR", "cohort": cohort,
            "selected_source": selected, **binary_metric(frame.label, frame.score, 110 if dataset == "TCGA" else 111),
        })
        patient_parts.append(frame.assign(cohort=cohort, endpoint="MSI/MMR", hrd_score=np.nan, selected_source=selected))

    # UCEC POLE/POLD1 exact-missense union.
    selected = selections[("UCEC", "SBS10")]
    figure = pd.read_csv(OLD_FIGURE, sep="\t")
    ids = {"TCGA": "ucec_pole_pold1_exact_missense_union", "CPTAC": "cptac_ucec_pole_pold1_exact_missense_union"}
    for dataset in ("TCGA", "CPTAC"):
        cohort = f"{dataset} UCEC"
        labels = figure[figure.endpoint_id == ids[dataset]][["PATIENT", "outcome", "pole_positive", "pold1_positive"]].rename(columns={"outcome": "label"})
        frame = labels.merge(
            frames[("UCEC", "SBS10", dataset)][["PATIENT", "observed", selected]],
            on="PATIENT", validate="one_to_one",
        ).rename(columns={selected: "score"})
        clinical_rows.append({
            "analysis": "UCEC SBS10 selected score for POLE/POLD1 exact-missense union", "cohort": cohort,
            "selected_source": selected, **binary_metric(frame.label, frame.score, 120 if dataset == "TCGA" else 121),
        })
        patient_parts.append(frame.assign(cohort=cohort, endpoint="POLE/POLD1", hrd_score=np.nan, selected_source=selected))

    selection_frame = pd.DataFrame(selection_rows)
    exposure_frame = pd.DataFrame(exposure_rows)
    clinical_frame = pd.DataFrame(clinical_rows)
    patient_frame = pd.concat(patient_parts, ignore_index=True, sort=False)
    selection_frame.to_csv(OUT / "TCGA_EXPOSURE_SELECTION.tsv", sep="\t", index=False)
    exposure_frame.to_csv(OUT / "selected_score_exposure_metrics.tsv", sep="\t", index=False)
    clinical_frame.to_csv(OUT / "selected_score_clinical_metrics.tsv", sep="\t", index=False)
    patient_frame.to_csv(OUT / "selected_score_clinical_patient_scores.tsv", sep="\t", index=False)
    audit = {
        "status": "complete", "selected_sources": {f"{cancer}-{signature}": value for (cancer, signature), value in selections.items()},
        "clinical_labels_used_for_score_selection": False, "cptac_used_for_score_selection": False,
        "metric_rows": len(clinical_frame), "audit_pass": len(clinical_frame) == 6,
    }
    (OUT / "AUDIT.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(selection_frame[selection_frame.selected].to_string(index=False))
    print("\nEXPOSURE")
    print(exposure_frame.to_string(index=False))
    print("\nCLINICAL")
    print(clinical_frame.to_string(index=False))


if __name__ == "__main__":
    main()
