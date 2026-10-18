#!/usr/bin/env python3
"""Recompute clinical endpoints from the locked global standardized ensemble."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, norm, pearsonr
from sklearn.metrics import roc_auc_score


BASE = Path("/home/marugoto-1.0.0-regression")
PACKAGE = BASE / "paper_ready_20260901_v2_25_standardized_upgrade"
PREDICTIONS = BASE / "legacy30_retccl_ctranspath_global_weight_20260829/results/predictions"
SUMMARY = BASE / "legacy30_retccl_ctranspath_global_weight_20260829/results/summary.tsv"
POSITIVE = BASE / "legacy30_retccl_ctranspath_global_weight_20260829/results/positive_raw_p_in_both.tsv"
SOURCE = BASE / "paper_ready_20260817_ver2/reproducibility/source_data"
FIGURE_SOURCE = BASE / "paper_ready_20260817_ver2/analysis_data/direct_figure_inputs/figure4_clinical_plot_data.tsv"
OUT = PACKAGE / "source_data"
BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 20260901


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fisher_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 3 or not np.isfinite(r):
        return np.nan, np.nan
    r = float(np.clip(r, -0.999999999, 0.999999999))
    z = np.arctanh(r)
    delta = norm.ppf(1 - alpha / 2) / np.sqrt(n - 3)
    return float(np.tanh(z - delta)), float(np.tanh(z + delta))


def continuous_metric(x: pd.Series, y: pd.Series) -> dict:
    keep = x.notna() & y.notna()
    x_values = x[keep].astype(float).to_numpy()
    y_values = y[keep].astype(float).to_numpy()
    r, p = pearsonr(x_values, y_values)
    low, high = fisher_ci(float(r), len(x_values))
    return {
        "n": int(len(x_values)),
        "positives": np.nan,
        "metric": "Pearson r",
        "estimate": float(r),
        "ci_low": low,
        "ci_high": high,
        "p_value": float(p),
    }


def binary_metric(label: pd.Series, score: pd.Series, seed_offset: int) -> dict:
    keep = label.notna() & score.notna()
    y = label[keep].astype(int).to_numpy()
    s = score[keep].astype(float).to_numpy()
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Binary endpoint must contain both classes")
    auc = float(roc_auc_score(y, s))
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    boot = np.empty(BOOTSTRAPS, dtype=float)
    for index in range(BOOTSTRAPS):
        chosen = np.concatenate([
            rng.choice(positive, size=len(positive), replace=True),
            rng.choice(negative, size=len(negative), replace=True),
        ])
        boot[index] = roc_auc_score(y[chosen], s[chosen])
    low, high = np.quantile(boot, [0.025, 0.975])
    p_value = mannwhitneyu(s[y == 1], s[y == 0], alternative="two-sided").pvalue
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "metric": "ROC AUC",
        "estimate": auc,
        "ci_low": float(low),
        "ci_high": float(high),
        "p_value": float(p_value),
    }


def prediction(cancer: str, signature: str, dataset: str) -> pd.DataFrame:
    path = PREDICTIONS / f"{cancer.lower()}_{signature.lower()}_{dataset.lower()}.tsv"
    frame = pd.read_csv(path, sep="\t")
    required = {"PATIENT", signature, "prediction"}
    if set(frame.columns) != required:
        raise ValueError(f"Unexpected columns in {path}: {frame.columns.tolist()}")
    if frame.PATIENT.duplicated().any():
        raise ValueError(f"Duplicate patients in {path}")
    return frame.rename(columns={signature: "observed", "prediction": "score"})


def tcga_percentile(reference: np.ndarray, values: np.ndarray, within_reference: bool) -> np.ndarray:
    reference = np.asarray(reference, dtype=float)
    values = np.asarray(values, dtype=float)
    if within_reference:
        return pd.Series(values).rank(method="average", pct=True).to_numpy()
    ordered = np.sort(reference)
    return np.searchsorted(ordered, values, side="right") / len(ordered)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)

    summary = pd.read_csv(SUMMARY, sep="\t")
    positive = pd.read_csv(POSITIVE, sep="\t")
    summary.to_csv(OUT / "standardized_210_results.tsv", sep="\t", index=False)
    positive.to_csv(OUT / "positive_raw_p_in_both.tsv", sep="\t", index=False)

    metrics: list[dict] = []
    patient_parts: list[pd.DataFrame] = []

    # BRCA SBS3 and continuous/binary HRD.
    brca = prediction("BRCA", "SBS3", "TCGA")
    hrd_continuous = pd.read_csv(SOURCE / "direct_sbs3_hrd_continuous.tsv", sep="\t")
    hrd = hrd_continuous[["PATIENT", "hrd_score"]].merge(
        brca[["PATIENT", "observed", "score"]], on="PATIENT", how="inner", validate="one_to_one"
    )
    metrics.append({
        "analysis": "BRCA SBS3 prediction versus continuous HRD score",
        "cohort": "TCGA BRCA",
        **continuous_metric(hrd["score"], hrd["hrd_score"]),
    })
    hrd_labels = pd.read_csv(SOURCE / "clinical_patient_scores.tsv", sep="\t")
    hrd_labels = hrd_labels[(hrd_labels.cohort == "TCGA BRCA") & (hrd_labels.endpoint == "HRD")]
    hrd_binary = hrd_labels[["PATIENT", "label"]].merge(
        brca[["PATIENT", "observed", "score"]], on="PATIENT", how="inner", validate="one_to_one"
    )
    metrics.append({
        "analysis": "BRCA HRD score >=42",
        "cohort": "TCGA BRCA",
        **binary_metric(hrd_binary["label"], hrd_binary["score"], 1),
    })
    patient_parts.extend([
        hrd.assign(endpoint="continuous HRD score", cohort="TCGA BRCA", label=np.nan)[
            ["PATIENT", "cohort", "endpoint", "label", "hrd_score", "observed", "score"]
        ],
        hrd_binary.assign(endpoint="HRD score >=42", cohort="TCGA BRCA", hrd_score=np.nan)[
            ["PATIENT", "cohort", "endpoint", "label", "hrd_score", "observed", "score"]
        ],
    ])

    # UCEC MSI/MMR: direct mapping to the canonical MMR-deficiency-associated SBS6 score.
    # The four-signature percentile average is retained separately as a sensitivity analysis.
    clinical = pd.read_csv(SOURCE / "clinical_patient_scores.tsv", sep="\t")
    msi_signatures = ["SBS6", "SBS15", "SBS20", "SBS26"]
    msi_scores: dict[str, pd.DataFrame] = {}
    for dataset in ("TCGA", "CPTAC"):
        merged = None
        for signature in msi_signatures:
            frame = prediction("UCEC", signature, dataset)[["PATIENT", "score"]].rename(
                columns={"score": signature}
            )
            merged = frame if merged is None else merged.merge(frame, on="PATIENT", validate="one_to_one")
        assert merged is not None
        msi_scores[dataset] = merged
    for signature in msi_signatures:
        reference = msi_scores["TCGA"][signature].to_numpy()
        msi_scores["TCGA"][f"{signature}_percentile"] = tcga_percentile(
            reference, reference, within_reference=True
        )
        msi_scores["CPTAC"][f"{signature}_percentile"] = tcga_percentile(
            reference, msi_scores["CPTAC"][signature].to_numpy(), within_reference=False
        )
    percentile_columns = [f"{signature}_percentile" for signature in msi_signatures]
    sensitivity_metrics: list[dict] = []
    sensitivity_parts: list[pd.DataFrame] = []
    for dataset, cohort in (("TCGA", "TCGA UCEC"), ("CPTAC", "CPTAC UCEC")):
        msi_scores[dataset]["score"] = msi_scores[dataset][percentile_columns].mean(axis=1)
        labels = clinical[(clinical.cohort == cohort) & (clinical.endpoint == "MSI/MMR")][
            ["PATIENT", "label"]
        ]
        evaluated = labels.merge(msi_scores[dataset], on="PATIENT", how="inner", validate="one_to_one")
        metrics.append({
            "analysis": "UCEC MSI/MMR direct SBS6 score",
            "cohort": cohort,
            **binary_metric(evaluated["label"], evaluated["SBS6"], 10 if dataset == "TCGA" else 11),
        })
        sensitivity_metrics.append({
            "analysis": "UCEC MSI/MMR equal-weight four-SBS percentile score",
            "cohort": cohort,
            **binary_metric(evaluated["label"], evaluated["score"], 12 if dataset == "TCGA" else 13),
        })
        evaluated["endpoint"] = "MSI/MMR"
        evaluated["cohort"] = cohort
        evaluated["hrd_score"] = np.nan
        evaluated["observed"] = np.nan
        evaluated["four_sbs_sensitivity_score"] = evaluated["score"]
        evaluated["score"] = evaluated["SBS6"]
        patient_parts.append(evaluated[
            ["PATIENT", "cohort", "endpoint", "label", "hrd_score", "observed", "score"]
            + msi_signatures + percentile_columns
        ])
        sensitivity_parts.append(evaluated[[
            "PATIENT", "cohort", "endpoint", "label", "four_sbs_sensitivity_score"
        ]])

    # UCEC SBS10 exposure and exact-missense POLE/POLD1 endpoint.
    figure_source = pd.read_csv(FIGURE_SOURCE, sep="\t")
    exact_ids = {
        "TCGA": "ucec_pole_pold1_exact_missense_union",
        "CPTAC": "cptac_ucec_pole_pold1_exact_missense_union",
    }
    for dataset, cohort in (("TCGA", "TCGA UCEC"), ("CPTAC", "CPTAC UCEC")):
        sbs10 = prediction("UCEC", "SBS10", dataset)
        metrics.append({
            "analysis": "UCEC SBS10 prediction versus observed SBS10 exposure",
            "cohort": cohort,
            **continuous_metric(sbs10["score"], sbs10["observed"]),
        })
        labels = figure_source[figure_source.endpoint_id == exact_ids[dataset]][
            ["PATIENT", "outcome", "pole_positive", "pold1_positive"]
        ].rename(columns={"outcome": "label"})
        evaluated = labels.merge(sbs10, on="PATIENT", how="inner", validate="one_to_one")
        metrics.append({
            "analysis": "UCEC POLE/POLD1 exact-missense union",
            "cohort": cohort,
            **binary_metric(evaluated["label"], evaluated["score"], 20 if dataset == "TCGA" else 21),
        })
        evaluated["endpoint"] = "POLE/POLD1 exact-missense union"
        evaluated["cohort"] = cohort
        evaluated["hrd_score"] = np.nan
        patient_parts.append(evaluated[[
            "PATIENT", "cohort", "endpoint", "label", "hrd_score", "observed", "score",
            "pole_positive", "pold1_positive",
        ]])

    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(OUT / "clinical_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(sensitivity_metrics).to_csv(
        OUT / "clinical_sensitivity_metrics.tsv", sep="\t", index=False
    )
    pd.concat(patient_parts, ignore_index=True, sort=False).to_csv(
        OUT / "clinical_patient_scores.tsv", sep="\t", index=False
    )
    pd.concat(sensitivity_parts, ignore_index=True).to_csv(
        OUT / "clinical_sensitivity_patient_scores.tsv", sep="\t", index=False
    )

    input_paths = [
        SUMMARY,
        POSITIVE,
        SOURCE / "direct_sbs3_hrd_continuous.tsv",
        SOURCE / "clinical_patient_scores.tsv",
        FIGURE_SOURCE,
    ] + sorted(PREDICTIONS.glob("*.tsv"))
    output_paths = sorted(OUT.glob("*.tsv"))
    audit = {
        "status": "complete",
        "analysis_contract": "CLINICAL_ANALYSIS_CONTRACT.json",
        "standardized_pairs": int(summary.pair.nunique()),
        "standardized_rows": int(len(summary)),
        "positive_raw_p_in_both": int(len(positive)),
        "clinical_metric_rows": int(len(metric_frame)),
        "clinical_endpoints_used_for_model_or_weight_selection": False,
        "cptac_used_for_model_or_weight_selection": False,
        "input_sha256": {str(path.relative_to(BASE)): sha256(path) for path in input_paths},
        "output_sha256": {str(path.relative_to(PACKAGE)): sha256(path) for path in output_paths},
        "audit_pass": bool(
            summary.pair.nunique() == 210
            and len(summary) == 420
            and len(positive) == 17
            and len(metric_frame) == 8
            and metric_frame[["n", "estimate", "p_value"]].notna().all().all()
        ),
    }
    (OUT / "AUDIT.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(metric_frame.to_string(index=False))
    print(json.dumps({"audit_pass": audit["audit_pass"], "positive_raw_p_in_both": len(positive)}, indent=2))
    if not audit["audit_pass"]:
        raise RuntimeError("Updated clinical analysis audit failed")


if __name__ == "__main__":
    main()
