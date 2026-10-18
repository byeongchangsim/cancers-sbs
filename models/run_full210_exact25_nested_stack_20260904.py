#!/opt/anaconda/envs/cenv/bin/python
"""Nested cancer-level stack and external evaluation for the exact25 run."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE / "full210_exact25_multiscale_fusion_20260904"
TRAINING_SCRIPT = BASE / "models/run_full210_exact25_multiscale_fusion_20260904.py"
RIDGE_SCRIPT = BASE / "models/run_standardized210_mean_ridge.py"
STACK_PROTOCOL = ROOT / "LOCKED_STACK_PROTOCOL.json"
SELECTION = ROOT / "FROZEN_INNER_SELECTION.tsv"
STACK_ROOT = ROOT / "nested_regularized_stack"
SIGNATURES = tuple(f"SBS{i}" for i in range(1, 31))
COMPONENTS = ("retccl", "fused1024", "fused2048")
SCALES = ("1024", "2048")
LAMBDA = 0.1


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


def indices_for(patients: np.ndarray, requested: list[str]) -> np.ndarray:
    lookup = {str(patient): index for index, patient in enumerate(patients)}
    missing = [patient for patient in requested if patient not in lookup]
    if missing:
        raise RuntimeError(f"Missing patients: {missing[:3]}")
    return np.asarray([lookup[patient] for patient in requested], dtype=np.int64)


def metric(target: np.ndarray, prediction: np.ndarray) -> dict:
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if len(target) < 3 or np.std(target) <= 0 or np.std(prediction) <= 0:
        return {"pearson_r": np.nan, "pearson_p": np.nan, "spearman_rho": np.nan, "spearman_p": np.nan}
    r, p = pearsonr(target, prediction)
    rho, rho_p = spearmanr(target, prediction)
    return {"pearson_r": float(r), "pearson_p": float(p), "spearman_rho": float(rho), "spearman_p": float(rho_p)}


def selected_row(selection: pd.DataFrame, scale: str, cancer: str, seed_index: int, fold: int):
    row = selection[
        selection.scale.eq(int(scale)) & selection.cancer.eq(cancer)
        & selection.seed_index.eq(seed_index) & selection.fold.eq(fold)
    ]
    if len(row) != 1:
        raise RuntimeError(f"Selection row count {len(row)}: {scale}/{cancer}/{seed_index}/{fold}")
    return row.iloc[0]


def load_selected_fused(
    training, selection: pd.DataFrame, scale: str, cancer: str,
    seed_index: int, fold: int, device: torch.device,
) -> dict:
    module = training.implementation(scale)
    row = selected_row(selection, scale, cancer, seed_index, fold)
    cancer_dir = str(row.cancer_dir)
    variant = str(row.selected_variant)
    sampling, hybrid = module.variant_parts(variant)
    directory = module.output_dir(variant, cancer_dir, seed_index, fold)
    checkpoint_path = directory / "best.pt"
    if sha256(checkpoint_path) != str(row.checkpoint_sha256):
        raise RuntimeError(f"Checkpoint hash mismatch: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    roles = {
        role: [str(value) for value in checkpoint[f"{role}_patients"]]
        for role in ("train", "validation", "test")
    }
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if set(roles[first]) & set(roles[second]):
            raise RuntimeError(f"Overlapping roles: {scale}/{cancer}/{seed_index}/{fold}")
    patients, bags_cpu, lens, target = module.load_tcga(cancer_dir, sampling)
    validation_index = indices_for(patients, roles["validation"])
    model = module.TargetAttentionMIL(hybrid).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    bags = torch.tensor(bags_cpu, device=device, dtype=torch.float16)
    validation_prediction = module.predict(model, bags, lens, validation_index)["composition"].astype(float)
    del bags, bags_cpu, model
    torch.cuda.empty_cache()
    gc.collect()
    test = pd.read_csv(directory / "tcga_test_predictions.tsv", sep="\t", dtype={"PATIENT": str}).set_index("PATIENT").reindex(roles["test"])
    cptac = pd.read_csv(directory / "cptac_label_free_predictions.tsv", sep="\t", dtype={"PATIENT": str})
    if test.isna().any().any() or cptac.isna().any().any():
        raise RuntimeError(f"Incomplete predictions: {directory}")
    return {
        "roles": roles,
        "validation_target": target[validation_index, :30].astype(float),
        "validation_prediction": validation_prediction,
        "test_target": test[list(SIGNATURES)].to_numpy(float),
        "test_prediction": test[[f"composition_{signature}" for signature in SIGNATURES]].to_numpy(float),
        "cptac_patients": cptac.PATIENT.astype(str).to_numpy(str),
        "cptac_prediction": cptac[[f"composition_{signature}" for signature in SIGNATURES]].to_numpy(float),
        "variant": variant,
    }


def ridge_predictions(cancer: str, roles: dict) -> dict:
    module = import_file(RIDGE_SCRIPT, f"exact25_ridge_{cancer}")
    patients, x, y, _, _ = module.load("TCGA", cancer)
    cptac_patients, cptac_x, _, _, _ = module.load("CPTAC", cancer)
    train_index = indices_for(patients, roles["train"])
    validation_index = indices_for(patients, roles["validation"])
    test_index = indices_for(patients, roles["test"])
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(x[train_index])
    validation_scaled = scaler.transform(x[validation_index])
    test_scaled = scaler.transform(x[test_index])
    cptac_scaled = scaler.transform(cptac_x)
    # Preserve the fixed seed of the previously registered multiscale stack.
    pca = PCA(n_components=16, random_state=20260901)
    train_reduced = pca.fit_transform(train_scaled)
    model = Ridge(alpha=100.0)
    model.fit(train_reduced, np.sqrt(np.clip(y[train_index], 0.0, None)))

    def predict(values: np.ndarray) -> np.ndarray:
        return np.square(np.clip(model.predict(pca.transform(values)), 0.0, None))

    return {
        "validation_target": y[validation_index].astype(float),
        "validation_prediction": predict(validation_scaled),
        "test_target": y[test_index].astype(float),
        "test_prediction": predict(test_scaled),
        "cptac_patients": cptac_patients.astype(str),
        "cptac_prediction": predict(cptac_scaled),
    }


def standardize(validation: np.ndarray, test: np.ndarray, cptac: np.ndarray):
    mean = validation.mean(axis=0)
    sd = validation.std(axis=0, ddof=0)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
    return (validation - mean) / sd, (test - mean) / sd, (cptac - mean) / sd, mean, sd


def weight_grid():
    for first in range(21):
        for second in range(21 - first):
            third = 20 - first - second
            yield np.asarray((first, second, third), dtype=float) / 20.0


def score_weights(target: np.ndarray, components: np.ndarray, weights: np.ndarray):
    prediction = np.tensordot(components, weights, axes=([2], [0]))
    values = []
    for index in range(30):
        if np.sum(target[:, index] > 0) < 3 or np.std(target[:, index]) <= 0 or np.std(prediction[:, index]) <= 0:
            continue
        value = float(pearsonr(target[:, index], prediction[:, index])[0])
        if np.isfinite(value):
            values.append(value)
    if not values:
        return -np.inf, -np.inf, 0
    return float(np.median(values)), float(np.mean(values)), len(values)


def fit_context(cancer: str, seed_index: int, fold: int, gpu: int) -> None:
    out = STACK_ROOT / "contexts" / cancer / f"seed-{seed_index}" / f"fold-{fold}"
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    training = import_file(TRAINING_SCRIPT, "exact25_training")
    selection = pd.read_csv(SELECTION, sep="\t")
    device = torch.device(f"cuda:{gpu}")
    fused = {
        scale: load_selected_fused(training, selection, scale, cancer, seed_index, fold, device)
        for scale in SCALES
    }
    first, second = fused["1024"], fused["2048"]
    for role in ("train", "validation", "test"):
        if first["roles"][role] != second["roles"][role]:
            raise RuntimeError(f"Role mismatch: {cancer}/{seed_index}/{fold}/{role}")
    ridge = ridge_predictions(cancer, first["roles"])
    for key in ("validation_target", "test_target"):
        if not np.allclose(first[key], second[key], atol=1e-7) or not np.allclose(first[key], ridge[key], atol=1e-7):
            raise RuntimeError(f"Target mismatch: {cancer}/{seed_index}/{fold}/{key}")
    if not np.array_equal(first["cptac_patients"], second["cptac_patients"]) or not np.array_equal(first["cptac_patients"], ridge["cptac_patients"]):
        raise RuntimeError(f"CPTAC order mismatch: {cancer}/{seed_index}/{fold}")
    raw = {"retccl": ridge, "fused1024": first, "fused2048": second}
    standardized = {}
    moment_rows = []
    for component, values in raw.items():
        validation, test, cptac, mean, sd = standardize(
            values["validation_prediction"], values["test_prediction"], values["cptac_prediction"]
        )
        standardized[component] = {"validation": validation, "test": test, "cptac": cptac}
        for index, signature in enumerate(SIGNATURES):
            moment_rows.append({
                "component": component, "signature": signature,
                "validation_mean": mean[index], "validation_population_sd": sd[index],
            })
    validation_components = np.stack([standardized[name]["validation"] for name in COMPONENTS], axis=2)
    equal = np.full(3, 1.0 / 3.0)
    grid_rows = []
    for weights in weight_grid():
        median_r, mean_r, eligible = score_weights(first["validation_target"], validation_components, weights)
        distance = float(np.sum((weights - equal) ** 2))
        grid_rows.append({
            **{f"{name}_weight": float(value) for name, value in zip(COMPONENTS, weights)},
            "validation_median_r": median_r, "validation_mean_r": mean_r,
            "eligible_signatures": eligible, "distance_from_equal": distance,
            "regularized_score": median_r - LAMBDA * distance,
        })
    grid = pd.DataFrame(grid_rows).sort_values(
        ["regularized_score", "validation_mean_r", "distance_from_equal",
         "retccl_weight", "fused1024_weight", "fused2048_weight"],
        ascending=[False, False, True, False, False, False],
    ).reset_index(drop=True)
    winner = grid.iloc[0]
    weights = np.asarray([winner[f"{name}_weight"] for name in COMPONENTS], dtype=float)
    test_components = np.stack([standardized[name]["test"] for name in COMPONENTS], axis=2)
    cptac_components = np.stack([standardized[name]["cptac"] for name in COMPONENTS], axis=2)
    np.savez_compressed(
        out / "predictions.npz",
        test_patients=np.asarray(first["roles"]["test"], dtype=str),
        test_target=first["test_target"],
        test_prediction=np.tensordot(test_components, weights, axes=([2], [0])),
        cptac_patients=first["cptac_patients"],
        cptac_prediction=np.tensordot(cptac_components, weights, axes=([2], [0])),
    )
    pd.DataFrame(moment_rows).to_csv(out / "validation_moments.tsv", sep="\t", index=False)
    grid.to_csv(out / "weight_grid.tsv", sep="\t", index=False)
    audit = {
        "status": "complete", "cancer": cancer, "seed_index": seed_index, "fold": fold,
        "selected_variants": {"fused1024": first["variant"], "fused2048": second["variant"]},
        "selected_weights": {name: float(value) for name, value in zip(COMPONENTS, weights)},
        "validation_median_r": float(winner.validation_median_r),
        "regularized_score": float(winner.regularized_score),
        "patient_roles_disjoint": True, "outer_test_used_for_weighting": False,
        "cptac_outcomes_read": False,
    }
    (out / "AUDIT.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, sort_keys=True))


def aggregate_and_freeze() -> None:
    if (STACK_ROOT / "FROZEN_PREDICTIONS").exists():
        raise FileExistsError(STACK_ROOT / "FROZEN_PREDICTIONS")
    frozen = STACK_ROOT / "FROZEN_PREDICTIONS"
    frozen.mkdir(parents=True)
    selection_audit = json.loads((ROOT / "FROZEN_SELECTION_AUDIT.json").read_text())
    if sha256(SELECTION) != selection_audit["selection_sha256"]:
        raise RuntimeError("Selection hash mismatch")
    training = import_file(TRAINING_SCRIPT, "exact25_training_aggregate")
    module = training.implementation("1024")
    cancer_labels = tuple(module.CANCER_LABEL[cancer] for cancer in module.CANCERS)
    weight_rows = []
    manifest_rows = []
    for cancer in cancer_labels:
        seed_frames = []
        cptac_parts = []
        cptac_order = None
        for seed_index in range(3):
            fold_frames = []
            for fold in range(5):
                context = STACK_ROOT / "contexts" / cancer / f"seed-{seed_index}" / f"fold-{fold}"
                if not (context / "AUDIT.json").exists():
                    raise FileNotFoundError(context / "AUDIT.json")
                with np.load(context / "predictions.npz") as data:
                    fold_frames.append(pd.DataFrame({
                        "PATIENT": data["test_patients"].astype(str),
                        **{signature: data["test_target"][:, index] for index, signature in enumerate(SIGNATURES)},
                        **{f"prediction_{signature}": data["test_prediction"][:, index] for index, signature in enumerate(SIGNATURES)},
                    }))
                    current_order = data["cptac_patients"].astype(str)
                    if cptac_order is None:
                        cptac_order = current_order
                    elif not np.array_equal(cptac_order, current_order):
                        raise RuntimeError(f"CPTAC order mismatch: {cancer}/{seed_index}/{fold}")
                    cptac_parts.append(data["cptac_prediction"].copy())
                audit = json.loads((context / "AUDIT.json").read_text())
                weight_rows.append({
                    "cancer": cancer, "seed_index": seed_index, "fold": fold,
                    **audit["selected_weights"],
                    "validation_median_r": audit["validation_median_r"],
                    "regularized_score": audit["regularized_score"],
                })
            seed_frame = pd.concat(fold_frames, ignore_index=True).sort_values("PATIENT")
            if seed_frame.PATIENT.duplicated().any():
                raise RuntimeError(f"Duplicate TCGA patient: {cancer}/{seed_index}")
            seed_frames.append(seed_frame)
        patient_order = seed_frames[0].PATIENT.to_numpy(str)
        tcga = pd.DataFrame({"PATIENT": patient_order})
        for signature in SIGNATURES:
            targets = [frame.set_index("PATIENT").reindex(patient_order)[signature].to_numpy(float) for frame in seed_frames]
            predictions = [frame.set_index("PATIENT").reindex(patient_order)[f"prediction_{signature}"].to_numpy(float) for frame in seed_frames]
            if any(not np.allclose(targets[0], target, atol=1e-7) for target in targets[1:]):
                raise RuntimeError(f"TCGA target mismatch: {cancer}/{signature}")
            tcga[signature] = targets[0]
            tcga[f"prediction_{signature}"] = np.mean(predictions, axis=0)
        cptac_matrix = np.mean(cptac_parts, axis=0)
        cptac = pd.DataFrame({
            "PATIENT": cptac_order,
            **{f"prediction_{signature}": cptac_matrix[:, index] for index, signature in enumerate(SIGNATURES)},
        })
        tcga_path = frozen / f"{cancer.lower()}_tcga_oof.tsv"
        cptac_path = frozen / f"{cancer.lower()}_cptac_label_free.tsv"
        tcga.to_csv(tcga_path, sep="\t", index=False)
        cptac.to_csv(cptac_path, sep="\t", index=False)
        for path in (tcga_path, cptac_path):
            manifest_rows.append({"file": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size})
    weights = pd.DataFrame(weight_rows)
    weights_path = STACK_ROOT / "FROZEN_STACK_WEIGHTS.tsv"
    weights.to_csv(weights_path, sep="\t", index=False)
    manifest_rows.append({"file": str(weights_path.relative_to(ROOT)), "sha256": sha256(weights_path), "bytes": weights_path.stat().st_size})
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = STACK_ROOT / "FROZEN_MANIFEST.tsv"
    manifest.to_csv(manifest_path, sep="\t", index=False)
    lock = {
        "status": "all_TCGA_OOF_and_label_free_CPTAC_predictions_frozen_before_external_outcome_join",
        "contexts": len(weights), "expected_contexts": 105,
        "prediction_files": 14, "complete_pairs": 210,
        "selection_sha256": sha256(SELECTION),
        "stack_protocol_sha256": sha256(STACK_PROTOCOL),
        "manifest_sha256": sha256(manifest_path),
        "outer_test_used_for_selection_or_weighting": False,
        "cptac_outcomes_read": False,
        "audit_pass": bool(len(weights) == 105 and len(manifest) == 15),
    }
    (STACK_ROOT / "PREDICTION_LOCK.json").write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))
    if not lock["audit_pass"]:
        raise RuntimeError("Prediction freeze audit failed")


def bh(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.full(len(values), np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return result
    ordered = finite[np.argsort(values[finite])]
    # The confirmatory family is the complete prespecified set of 210 pairs.
    # Non-estimable tests therefore remain missing but still count toward m.
    adjusted = values[ordered] * len(values) / np.arange(1, len(ordered) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result[ordered] = np.minimum(adjusted, 1.0)
    return result


def verify_frozen_manifest() -> None:
    lock = json.loads((STACK_ROOT / "PREDICTION_LOCK.json").read_text())
    manifest_path = STACK_ROOT / "FROZEN_MANIFEST.tsv"
    if sha256(manifest_path) != lock["manifest_sha256"]:
        raise RuntimeError("Frozen manifest hash mismatch")
    for row in pd.read_csv(manifest_path, sep="\t").itertuples(index=False):
        path = ROOT / row.file
        if sha256(path) != row.sha256:
            raise RuntimeError(f"Frozen file hash mismatch: {path}")


def evaluate_external() -> None:
    verify_frozen_manifest()
    result_root = STACK_ROOT / "external_results"
    if result_root.exists():
        raise FileExistsError(result_root)
    result_root.mkdir()
    prediction_root = result_root / "predictions"
    prediction_root.mkdir()
    training = import_file(TRAINING_SCRIPT, "exact25_training_evaluate")
    module = training.implementation("1024")
    rows = []
    for cancer_dir in module.CANCERS:
        cancer = module.CANCER_LABEL[cancer_dir]
        tcga = pd.read_csv(STACK_ROOT / "FROZEN_PREDICTIONS" / f"{cancer.lower()}_tcga_oof.tsv", sep="\t", dtype={"PATIENT": str})
        cptac_prediction = pd.read_csv(STACK_ROOT / "FROZEN_PREDICTIONS" / f"{cancer.lower()}_cptac_label_free.tsv", sep="\t", dtype={"PATIENT": str})
        label_path = BASE / "process_endpoint_inputs_legacy30_min50" / "CPTAC" / cancer_dir / f"CPTAC_{cancer_dir}_process_endpoints.xlsx"
        cptac_label = pd.read_excel(label_path, usecols=["PATIENT", *SIGNATURES])
        cptac_label["PATIENT"] = cptac_label.PATIENT.astype(str)
        cptac = cptac_label.merge(cptac_prediction, on="PATIENT", validate="one_to_one")
        for signature in SIGNATURES:
            pair = f"{cancer}–{signature}"
            for dataset, frame in (("TCGA", tcga), ("CPTAC", cptac)):
                output = frame[["PATIENT", signature, f"prediction_{signature}"]].rename(columns={f"prediction_{signature}": "prediction"})
                output.to_csv(prediction_root / f"{cancer.lower()}_{signature.lower()}_{dataset.lower()}.tsv", sep="\t", index=False)
                rows.append({
                    "pair": pair, "cancer": cancer, "signature": signature, "dataset": dataset,
                    "n": len(output), "nonzero": int(np.sum(output[signature] > 0)),
                    **metric(output[signature].to_numpy(float), output.prediction.to_numpy(float)),
                })
    long = pd.DataFrame(rows)
    long["pearson_q_bh_210"] = np.nan
    long["spearman_q_bh_210"] = np.nan
    for dataset in ("TCGA", "CPTAC"):
        mask = long.dataset.eq(dataset)
        long.loc[mask, "pearson_q_bh_210"] = bh(long.loc[mask, "pearson_p"].to_numpy(float))
        long.loc[mask, "spearman_q_bh_210"] = bh(long.loc[mask, "spearman_p"].to_numpy(float))
    long.to_csv(result_root / "all_210_long.tsv", sep="\t", index=False)
    wide = long.pivot(index=["pair", "cancer", "signature"], columns="dataset")
    wide.columns = [f"{metric_name}_{dataset}" for metric_name, dataset in wide.columns]
    wide = wide.reset_index()
    wide["direction_concordant"] = np.sign(wide.pearson_r_TCGA) == np.sign(wide.pearson_r_CPTAC)
    wide["positive_raw_p_in_both"] = (
        (wide.pearson_r_TCGA > 0) & (wide.pearson_p_TCGA < 0.05)
        & (wide.pearson_r_CPTAC > 0) & (wide.pearson_p_CPTAC < 0.05)
    )
    wide["positive_bh_q_in_both"] = (
        (wide.pearson_r_TCGA > 0) & (wide.pearson_q_bh_210_TCGA < 0.05)
        & (wide.pearson_r_CPTAC > 0) & (wide.pearson_q_bh_210_CPTAC < 0.05)
    )
    wide.to_csv(result_root / "all_210_wide.tsv", sep="\t", index=False)
    raw = wide[wide.positive_raw_p_in_both].copy()
    corrected = wide[wide.positive_bh_q_in_both].copy()
    raw.to_csv(result_root / "positive_raw_p_in_both.tsv", sep="\t", index=False)
    corrected.to_csv(result_root / "positive_bh_q_in_both.tsv", sep="\t", index=False)
    brca_sbs3 = wide[wide.pair.eq("BRCA–SBS3")].iloc[0].to_dict()
    audit = {
        "status": "complete_external_evaluation",
        "complete_pairs": len(wide),
        "positive_raw_p_in_both": len(raw),
        "positive_bh_q_in_both": len(corrected),
        "direction_concordant_regardless_of_p": int(wide.direction_concordant.sum()),
        "brca_sbs3": brca_sbs3,
        "cohorts_pooled": False,
        "cptac_used_only_as_external_test": True,
        "cptac_used_for_model_selection_weighting_or_calibration": False,
        "audit_pass": bool(len(wide) == 210),
    }
    (result_root / "AUDIT.json").write_text(json.dumps(audit, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True, default=str))
    if len(raw):
        print(raw[["pair", "pearson_r_TCGA", "pearson_p_TCGA", "pearson_r_CPTAC", "pearson_p_CPTAC"]].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit-context")
    fit.add_argument("--cancer", required=True)
    fit.add_argument("--seed-index", type=int, required=True)
    fit.add_argument("--fold", type=int, required=True)
    fit.add_argument("--gpu", type=int, default=0)
    sub.add_parser("aggregate-and-freeze")
    sub.add_parser("evaluate-external")
    args = parser.parse_args()
    if args.command == "fit-context":
        fit_context(args.cancer, args.seed_index, args.fold, args.gpu)
    elif args.command == "aggregate-and-freeze":
        aggregate_and_freeze()
    elif args.command == "evaluate-external":
        evaluate_external()


if __name__ == "__main__":
    main()
