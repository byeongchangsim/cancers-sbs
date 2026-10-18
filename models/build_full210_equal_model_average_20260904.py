#!/opt/anaconda/envs/cenv/bin/python
"""Build a fixed unweighted average of three frozen TCGA-developed SBS models.

The three components are a standard PCA-ridge model and two cancer-specific
multi-output gated-attention MIL models using 1,024- and 2,048-tile bags.  Each
component is standardized using the corresponding inner-validation partition,
then combined with the prespecified arithmetic mean (1/3, 1/3, 1/3).  No
ensemble weight is fitted.  Outer-test and CPTAC outcomes are not used to form
predictions.  The script reuses the already frozen inner-selected MIL
checkpoints and refits the fixed PCA16/ridge(alpha=100) component within the
same outer folds.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr


BASE = Path(__file__).resolve().parents[1]
MODEL = BASE / "full210_exact25_multiscale_fusion_20260904"
OLD_STACK = MODEL / "nested_regularized_stack"
SELECTION = MODEL / "FROZEN_INNER_SELECTION.tsv"
TRAINING_SCRIPT = BASE / "models/run_full210_exact25_multiscale_fusion_20260904.py"
STACK_SCRIPT = BASE / "models/run_full210_exact25_nested_stack_20260904.py"
OUTPUT = BASE / "full210_equal_model_average_20260904"
CONTEXTS = OUTPUT / "contexts"
FROZEN = OUTPUT / "frozen_predictions"
RESULTS = OUTPUT / "external_results"
SIGNATURES = tuple(f"SBS{i}" for i in range(1, 31))
COMPONENTS = ("retccl_pca_ridge", "mil_1024", "mil_2048")
CANCERS = ("BRCA", "HNSC", "KIRC", "LUAD", "LUSC", "OV", "UCEC")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selected_row(selection: pd.DataFrame, scale: int, cancer: str, seed: int, fold: int) -> pd.Series:
    rows = selection[
        selection.scale.eq(scale) & selection.cancer.eq(cancer)
        & selection.seed_index.eq(seed) & selection.fold.eq(fold)
    ]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one selection row: {scale}/{cancer}/{seed}/{fold}")
    return rows.iloc[0]


def component_moments(context: Path, old_name: str) -> tuple[np.ndarray, np.ndarray]:
    moments = pd.read_csv(context / "validation_moments.tsv", sep="\t")
    rows = moments[moments.component.eq(old_name)].set_index("signature").reindex(SIGNATURES)
    if len(rows) != 30 or rows.isna().any().any():
        raise RuntimeError(f"Incomplete moments: {context}/{old_name}")
    mean = rows.validation_mean.to_numpy(float)
    sd = rows.validation_population_sd.to_numpy(float)
    if np.any(~np.isfinite(sd)) or np.any(sd <= 0):
        raise RuntimeError(f"Invalid validation SD: {context}/{old_name}")
    return mean, sd


def standardize(values: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=float) - mean) / sd


def metric(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if len(target) < 3 or np.std(target) <= 0 or np.std(prediction) <= 0:
        return {
            "pearson_r": np.nan, "pearson_p_two_sided": np.nan,
            "pearson_p_positive_direction": np.nan,
            "spearman_rho": np.nan, "spearman_p_two_sided": np.nan,
        }
    r, p = pearsonr(target, prediction)
    rho, rho_p = spearmanr(target, prediction)
    p_positive = p / 2.0 if r >= 0 else 1.0 - p / 2.0
    return {
        "pearson_r": float(r), "pearson_p_two_sided": float(p),
        "pearson_p_positive_direction": float(p_positive),
        "spearman_rho": float(rho), "spearman_p_two_sided": float(rho_p),
    }


def bh(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    output = np.full(len(values), np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return output
    order = finite[np.argsort(values[finite])]
    adjusted = values[order] * len(values) / np.arange(1, len(order) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output[order] = np.minimum(adjusted, 1.0)
    return output


def build_contexts() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    CONTEXTS.mkdir(parents=True)
    selection = pd.read_csv(SELECTION, sep="\t")
    training = import_file(TRAINING_SCRIPT, "equal_average_training")
    stack = import_file(STACK_SCRIPT, "equal_average_stack")
    label_to_dir = {
        training.implementation("1024").CANCER_LABEL[item]: item
        for item in training.implementation("1024").CANCERS
    }
    audit_rows = []
    for cancer in CANCERS:
        cancer_dir = label_to_dir[cancer]
        for seed in range(3):
            for fold in range(5):
                old_context = OLD_STACK / "contexts" / cancer / f"seed-{seed}" / f"fold-{fold}"
                with np.load(old_context / "predictions.npz") as source:
                    test_patients = source["test_patients"].astype(str)
                    test_target = source["test_target"].astype(float)
                    cptac_patients = source["cptac_patients"].astype(str)

                mil = {}
                roles = None
                variants = {}
                for scale, old_name in ((1024, "fused1024"), (2048, "fused2048")):
                    row = selected_row(selection, scale, cancer, seed, fold)
                    module = training.implementation(str(scale))
                    directory = module.output_dir(
                        str(row.selected_variant), str(row.cancer_dir), seed, fold
                    )
                    checkpoint = torch.load(directory / "best.pt", map_location="cpu")
                    current_roles = {
                        role: [str(item) for item in checkpoint[f"{role}_patients"]]
                        for role in ("train", "validation", "test")
                    }
                    if roles is None:
                        roles = current_roles
                    elif roles != current_roles:
                        raise RuntimeError(f"Role mismatch: {cancer}/{seed}/{fold}")
                    if current_roles["test"] != test_patients.tolist():
                        raise RuntimeError(f"Test order mismatch: {cancer}/{seed}/{fold}/{scale}")
                    test = pd.read_csv(directory / "tcga_test_predictions.tsv", sep="\t", dtype={"PATIENT": str})
                    test = test.set_index("PATIENT").reindex(test_patients)
                    cptac = pd.read_csv(directory / "cptac_label_free_predictions.tsv", sep="\t", dtype={"PATIENT": str})
                    cptac = cptac.set_index("PATIENT").reindex(cptac_patients)
                    columns = [f"composition_{signature}" for signature in SIGNATURES]
                    if test[columns].isna().any().any() or cptac[columns].isna().any().any():
                        raise RuntimeError(f"Incomplete MIL prediction: {directory}")
                    mean, sd = component_moments(old_context, old_name)
                    mil[scale] = {
                        "test": standardize(test[columns].to_numpy(float), mean, sd),
                        "cptac": standardize(cptac[columns].to_numpy(float), mean, sd),
                    }
                    variants[scale] = str(row.selected_variant)

                ridge = stack.ridge_predictions(cancer, roles)
                if not np.array_equal(ridge["cptac_patients"], cptac_patients):
                    raise RuntimeError(f"CPTAC order mismatch: {cancer}/{seed}/{fold}/ridge")
                if not np.allclose(ridge["test_target"], test_target, atol=1e-7):
                    raise RuntimeError(f"Ridge target mismatch: {cancer}/{seed}/{fold}")
                mean, sd = component_moments(old_context, "retccl")
                ridge_test = standardize(ridge["test_prediction"], mean, sd)
                ridge_cptac = standardize(ridge["cptac_prediction"], mean, sd)

                test_prediction = (ridge_test + mil[1024]["test"] + mil[2048]["test"]) / 3.0
                cptac_prediction = (ridge_cptac + mil[1024]["cptac"] + mil[2048]["cptac"]) / 3.0
                destination = CONTEXTS / cancer / f"seed-{seed}" / f"fold-{fold}"
                destination.mkdir(parents=True)
                np.savez_compressed(
                    destination / "predictions.npz",
                    test_patients=test_patients,
                    test_target=test_target,
                    test_prediction=test_prediction,
                    cptac_patients=cptac_patients,
                    cptac_prediction=cptac_prediction,
                )
                audit_rows.append({
                    "cancer": cancer, "seed_index": seed, "fold": fold,
                    "retccl_pca_ridge_weight": 1 / 3,
                    "mil_1024_weight": 1 / 3, "mil_2048_weight": 1 / 3,
                    "mil_1024_variant": variants[1024], "mil_2048_variant": variants[2048],
                    "outer_test_used_for_averaging": False,
                    "cptac_outcome_used_for_averaging": False,
                })
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(OUTPUT / "MODEL_AVERAGE_SPECIFICATION.tsv", sep="\t", index=False)
    if len(audit) != 105:
        raise RuntimeError("Expected 105 cancer-seed-fold contexts")


def aggregate() -> None:
    FROZEN.mkdir()
    manifest = []
    for cancer in CANCERS:
        seed_frames = []
        cptac_parts = []
        cptac_order = None
        for seed in range(3):
            folds = []
            for fold in range(5):
                path = CONTEXTS / cancer / f"seed-{seed}" / f"fold-{fold}" / "predictions.npz"
                with np.load(path) as values:
                    folds.append(pd.DataFrame({
                        "PATIENT": values["test_patients"].astype(str),
                        **{signature: values["test_target"][:, i] for i, signature in enumerate(SIGNATURES)},
                        **{f"prediction_{signature}": values["test_prediction"][:, i] for i, signature in enumerate(SIGNATURES)},
                    }))
                    current = values["cptac_patients"].astype(str)
                    if cptac_order is None:
                        cptac_order = current
                    elif not np.array_equal(cptac_order, current):
                        raise RuntimeError(f"CPTAC order mismatch during aggregate: {cancer}")
                    cptac_parts.append(values["cptac_prediction"].astype(float))
            seed_frame = pd.concat(folds, ignore_index=True).sort_values("PATIENT")
            if seed_frame.PATIENT.duplicated().any():
                raise RuntimeError(f"Duplicate TCGA patient: {cancer}/{seed}")
            seed_frames.append(seed_frame)
        patients = seed_frames[0].PATIENT.to_numpy(str)
        tcga = pd.DataFrame({"PATIENT": patients})
        for signature in SIGNATURES:
            targets = [f.set_index("PATIENT").reindex(patients)[signature].to_numpy(float) for f in seed_frames]
            predictions = [f.set_index("PATIENT").reindex(patients)[f"prediction_{signature}"].to_numpy(float) for f in seed_frames]
            if any(not np.allclose(targets[0], item, atol=1e-7) for item in targets[1:]):
                raise RuntimeError(f"TCGA target mismatch: {cancer}/{signature}")
            tcga[signature] = targets[0]
            tcga[f"prediction_{signature}"] = np.mean(predictions, axis=0)
        cptac_matrix = np.mean(cptac_parts, axis=0)
        cptac = pd.DataFrame({
            "PATIENT": cptac_order,
            **{f"prediction_{signature}": cptac_matrix[:, i] for i, signature in enumerate(SIGNATURES)},
        })
        for cohort, frame in (("tcga_oof", tcga), ("cptac_label_free", cptac)):
            path = FROZEN / f"{cancer.lower()}_{cohort}.tsv"
            frame.to_csv(path, sep="\t", index=False)
            manifest.append({"file": str(path.relative_to(OUTPUT)), "sha256": sha256(path), "bytes": path.stat().st_size})
    pd.DataFrame(manifest).to_csv(OUTPUT / "FROZEN_PREDICTION_MANIFEST.tsv", sep="\t", index=False)


def evaluate() -> None:
    RESULTS.mkdir()
    prediction_root = RESULTS / "predictions"
    prediction_root.mkdir()
    training = import_file(TRAINING_SCRIPT, "equal_average_evaluate")
    module = training.implementation("1024")
    label_to_dir = {module.CANCER_LABEL[item]: item for item in module.CANCERS}
    rows = []
    for cancer in CANCERS:
        cancer_dir = label_to_dir[cancer]
        tcga = pd.read_csv(FROZEN / f"{cancer.lower()}_tcga_oof.tsv", sep="\t", dtype={"PATIENT": str})
        cptac_prediction = pd.read_csv(FROZEN / f"{cancer.lower()}_cptac_label_free.tsv", sep="\t", dtype={"PATIENT": str})
        label_path = BASE / "process_endpoint_inputs_legacy30_min50" / "CPTAC" / cancer_dir / f"CPTAC_{cancer_dir}_process_endpoints.xlsx"
        labels = pd.read_excel(label_path, usecols=["PATIENT", *SIGNATURES])
        labels["PATIENT"] = labels.PATIENT.astype(str)
        cptac = labels.merge(cptac_prediction, on="PATIENT", validate="one_to_one")
        for signature in SIGNATURES:
            pair = f"{cancer}–{signature}"
            for cohort, frame in (("TCGA", tcga), ("CPTAC", cptac)):
                prediction = frame[["PATIENT", signature, f"prediction_{signature}"]].rename(
                    columns={f"prediction_{signature}": "prediction"}
                )
                prediction.to_csv(
                    prediction_root / f"{cancer.lower()}_{signature.lower()}_{cohort.lower()}.tsv",
                    sep="\t", index=False,
                )
                rows.append({
                    "pair": pair, "cancer": cancer, "signature": signature,
                    "dataset": cohort, "n": len(prediction),
                    "nonzero": int((prediction[signature] > 0).sum()),
                    **metric(prediction[signature].to_numpy(float), prediction.prediction.to_numpy(float)),
                })
    long = pd.DataFrame(rows)
    long["pearson_q_bh_210_positive_direction"] = np.nan
    for cohort in ("TCGA", "CPTAC"):
        mask = long.dataset.eq(cohort)
        long.loc[mask, "pearson_q_bh_210_positive_direction"] = bh(
            long.loc[mask, "pearson_p_positive_direction"].to_numpy(float)
        )
    long.to_csv(RESULTS / "all_420_cohort_metrics.tsv", sep="\t", index=False)
    wide = long.pivot(index=["pair", "cancer", "signature"], columns="dataset")
    wide.columns = [f"{name}_{cohort}" for name, cohort in wide.columns]
    wide = wide.reset_index()
    wide["direction_concordant"] = np.sign(wide.pearson_r_TCGA) == np.sign(wide.pearson_r_CPTAC)
    wide["positive_directional_p_in_both"] = (
        (wide.pearson_r_TCGA > 0) & (wide.pearson_p_positive_direction_TCGA < 0.05)
        & (wide.pearson_r_CPTAC > 0) & (wide.pearson_p_positive_direction_CPTAC < 0.05)
    )
    wide["positive_two_sided_p_in_both"] = (
        (wide.pearson_r_TCGA > 0) & (wide.pearson_p_two_sided_TCGA < 0.05)
        & (wide.pearson_r_CPTAC > 0) & (wide.pearson_p_two_sided_CPTAC < 0.05)
    )
    wide.to_csv(RESULTS / "all_210_paired_metrics.tsv", sep="\t", index=False)
    selected = wide[wide.positive_directional_p_in_both].copy()
    selected.to_csv(RESULTS / "positive_directional_in_both.tsv", sep="\t", index=False)
    summary = {
        "status": "complete",
        "method": "inner-validation standardized fixed unweighted arithmetic mean",
        "components": list(COMPONENTS),
        "fixed_weights": {component: 1 / 3 for component in COMPONENTS},
        "contexts": 105, "pairs": 210,
        "positive_directional_p_in_both": int(wide.positive_directional_p_in_both.sum()),
        "positive_two_sided_p_in_both": int(wide.positive_two_sided_p_in_both.sum()),
        "direction_concordant_estimable": int(wide.direction_concordant.fillna(False).sum()),
        "brca_sbs3_selected": bool(selected.pair.eq("BRCA–SBS3").any()),
        "outer_test_used_for_model_average": False,
        "cptac_used_for_model_selection_weighting_or_calibration": False,
        "audit_pass": bool(len(long) == 420 and len(wide) == 210),
    }
    (OUTPUT / "AUDIT.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    protocol = {
        "analysis": "standard model averaging of three TCGA-developed component models",
        "scope": "seven cancers x SBS1-SBS30 = 210 pairs",
        "components": {
            "retccl_pca_ridge": "RetCCL patient-mean features; fold-internal standardization; PCA16; ridge alpha 100",
            "mil_1024": "inner-validation-selected multi-output gated-attention MIL using a 1,024-tile bag",
            "mil_2048": "inner-validation-selected multi-output gated-attention MIL using a 2,048-tile bag",
        },
        "combination": "component predictions standardized with inner-validation mean and population SD, followed by an unweighted 1:1:1 arithmetic mean",
        "fitted_ensemble_weights": False,
        "shared_stack_or_super_learner": False,
        "outer_test_used_for_combination": False,
        "cptac_outcome_used_for_combination": False,
        "tcga_aggregation": "unweighted mean of three seed-specific outer-fold predictions",
        "cptac_aggregation": "unweighted mean of 15 seed-fold predictions",
    }
    (OUTPUT / "METHOD_PROTOCOL.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["audit_pass"]:
        raise RuntimeError("Audit failed")


def main() -> None:
    build_contexts()
    aggregate()
    evaluate()


if __name__ == "__main__":
    main()
