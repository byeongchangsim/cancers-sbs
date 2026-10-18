#!/opt/anaconda/envs/cenv/bin/python
"""Apply the fixed mean-pooled RetCCL protocol to seven cancers x SBS1-SBS30."""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


BASE = Path("/home/marugoto-1.0.0-regression")
ROOT = BASE / "standardized210_retccl_mean_ridge_20260829"
AGG = BASE / "sigprofiler_improved_20260818/inputs/end2end_retccl_qc_v1/patient_aggregates"
LABELS = BASE / "process_endpoint_inputs_legacy30_min50"
PAPER = BASE / "paper_ready_20260817_ver2/reproducibility/source_data/full210_uniform_ensemble_results.tsv"
PROTOCOL = ROOT / "REGISTERED_PROTOCOL.json"
SEEDS = (20260829, 20260830, 20260831)
CANCERS = ("BRCA", "HNSC", "KIRC", "LUAD", "LUSC", "OV", "UCEC")
SIGNATURES = tuple(f"SBS{i}" for i in range(1, 31))
LABEL_CANCERS = {
    "BRCA": "Brca", "HNSC": "Hnscc", "KIRC": "Ccrcc", "LUAD": "Luad",
    "LUSC": "Lscc", "OV": "Ov", "UCEC": "Ucec",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict:
    if len(y) < 3 or np.std(y) == 0 or np.std(prediction) == 0:
        return {
            "n": int(len(y)), "nonzero": int(np.count_nonzero(y)),
            "pearson_r": np.nan, "pearson_p": np.nan,
            "spearman_rho": np.nan, "spearman_p": np.nan,
            "target_std": float(np.std(y)), "prediction_std": float(np.std(prediction)),
        }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        r, p = pearsonr(y, prediction)
        rho, rho_p = spearmanr(y, prediction)
    return {
        "n": int(len(y)), "nonzero": int(np.count_nonzero(y)),
        "pearson_r": float(r), "pearson_p": float(p),
        "spearman_rho": float(rho), "spearman_p": float(rho_p),
        "target_std": float(np.std(y)), "prediction_std": float(np.std(prediction)),
    }


def load(dataset: str, cancer: str):
    label_cancer = LABEL_CANCERS[cancer]
    archive = AGG / f"{dataset}_{cancer}_old_new_qc_mean_std.npz"
    labels = LABELS / dataset / label_cancer / f"{dataset}_{label_cancer}_process_endpoints.xlsx"
    z = np.load(archive)
    target = pd.read_excel(labels, usecols=["PATIENT", *SIGNATURES])
    target["PATIENT"] = target["PATIENT"].astype(str)
    index = pd.DataFrame({
        "PATIENT": z["patients"].astype(str),
        "source_index": np.arange(len(z["patients"])),
    }).merge(target, on="PATIENT", validate="one_to_one")
    x = z["new_all"][index.source_index.to_numpy(), :2048]
    y = index.loc[:, SIGNATURES].to_numpy(float)
    return index.PATIENT.to_numpy(), x, y, archive, labels


def main() -> None:
    if not PROTOCOL.exists():
        raise FileNotFoundError("The pre-registered protocol is missing")
    results = ROOT / "results"
    if results.exists():
        raise FileExistsError(f"Refusing to overwrite {results}")
    predictions_root = results / "predictions"
    predictions_root.mkdir(parents=True)

    summary, per_seed, hashes = [], [], {}
    for cancer in CANCERS:
        tp, x, y, ta, tl = load("TCGA", cancer)
        ep, xe, ye, ea, el = load("CPTAC", cancer)
        for name, path in {
            f"TCGA_{cancer}_aggregate": ta, f"CPTAC_{cancer}_aggregate": ea,
            f"TCGA_{cancer}_labels": tl, f"CPTAC_{cancer}_labels": el,
        }.items():
            hashes[name] = sha256(path)

        tcga_by_seed, cptac_by_seed = [], []
        transformed_y = np.sqrt(np.clip(y, 0.0, None))
        for seed_index, seed in enumerate(SEEDS):
            oof = np.zeros_like(y, dtype=np.float64)
            external = []
            for train, test in KFold(5, shuffle=True, random_state=seed).split(x):
                scaler = StandardScaler()
                train_scaled = scaler.fit_transform(x[train])
                test_scaled = scaler.transform(x[test])
                external_scaled = scaler.transform(xe)
                pca = PCA(n_components=16, random_state=seed)
                train_reduced = pca.fit_transform(train_scaled)
                test_reduced = pca.transform(test_scaled)
                external_reduced = pca.transform(external_scaled)
                model = Ridge(alpha=100.0)
                model.fit(train_reduced, transformed_y[train])
                oof[test] = np.square(np.clip(model.predict(test_reduced), 0.0, None))
                external.append(
                    np.square(np.clip(model.predict(external_reduced), 0.0, None))
                )
            ext = np.mean(external, axis=0, dtype=np.float64)
            tcga_by_seed.append(oof)
            cptac_by_seed.append(ext)
            for j, signature in enumerate(SIGNATURES):
                pair = f"{cancer}–{signature}"
                per_seed.extend([
                    {"pair": pair, "cancer": cancer, "signature": signature,
                     "dataset": "TCGA", "seed_index": seed_index, "seed": seed,
                     **metrics(y[:, j], oof[:, j])},
                    {"pair": pair, "cancer": cancer, "signature": signature,
                     "dataset": "CPTAC", "seed_index": seed_index, "seed": seed,
                     **metrics(ye[:, j], ext[:, j])},
                ])

        tcga_ensemble = np.mean(tcga_by_seed, axis=0, dtype=np.float64)
        cptac_ensemble = np.mean(cptac_by_seed, axis=0, dtype=np.float64)
        for j, signature in enumerate(SIGNATURES):
            pair = f"{cancer}–{signature}"
            safe_pair = pair.replace("–", "_").lower()
            for dataset, patients, target, seed_values, prediction in (
                ("TCGA", tp, y[:, j], tcga_by_seed, tcga_ensemble[:, j]),
                ("CPTAC", ep, ye[:, j], cptac_by_seed, cptac_ensemble[:, j]),
            ):
                frame = pd.DataFrame({
                    "PATIENT": patients, signature: target,
                    **{f"seed_{i}_prediction": values[:, j]
                       for i, values in enumerate(seed_values)},
                    "prediction": prediction,
                })
                frame.to_csv(
                    predictions_root / f"{safe_pair}_{dataset.lower()}_predictions.tsv",
                    sep="\t", index=False,
                )
                summary.append({
                    "pair": pair, "cancer": cancer, "signature": signature,
                    "dataset": dataset, **metrics(target, prediction),
                })

    summary_frame = pd.DataFrame(summary)
    summary_frame["positive_raw_p_lt_0_05"] = (
        summary_frame.pearson_r.gt(0) & summary_frame.pearson_p.lt(0.05)
    )
    summary_frame.to_csv(results / "summary.tsv", sep="\t", index=False)
    pd.DataFrame(per_seed).to_csv(results / "per_seed_metrics.tsv", sep="\t", index=False)

    wide = summary_frame.pivot(index=["pair", "cancer", "signature"], columns="dataset")
    wide.columns = [f"new_{dataset.lower()}_{metric}" for metric, dataset in wide.columns]
    wide = wide.reset_index()
    if PAPER is None:
        comparison = wide.copy()
    else:
        paper = pd.read_csv(PAPER, sep="\t")
        comparison = paper.merge(wide, on=["pair", "signature"], validate="one_to_one")
    comparison["new_positive_nominal_in_both"] = (
        comparison.new_tcga_pearson_r.gt(0)
        & comparison.new_tcga_pearson_p.lt(0.05)
        & comparison.new_cptac_pearson_r.gt(0)
        & comparison.new_cptac_pearson_p.lt(0.05)
    )
    comparison["new_positive_direction_in_both"] = (
        comparison.new_tcga_pearson_r.gt(0) & comparison.new_cptac_pearson_r.gt(0)
    )
    comparison["new_same_direction"] = (
        np.sign(comparison.new_tcga_pearson_r) == np.sign(comparison.new_cptac_pearson_r)
    ) & comparison.new_tcga_pearson_r.notna() & comparison.new_cptac_pearson_r.notna()
    if PAPER is not None:
        comparison["delta_tcga_r"] = comparison.new_tcga_pearson_r - comparison.tcga_pearson_r
        comparison["delta_cptac_r"] = comparison.new_cptac_pearson_r - comparison.cptac_pearson_r
        comparison["improved_r_in_both"] = (
            comparison.delta_tcga_r.gt(0) & comparison.delta_cptac_r.gt(0)
        )
        comparison.to_csv(results / "comparison_to_paper_ready_210.tsv", sep="\t", index=False)
    comparison.loc[comparison.new_positive_nominal_in_both].to_csv(
        results / "both_cohort_positive_raw_p.tsv", sep="\t", index=False,
    )

    counts = {
        "tasks": len(CANCERS) * len(SIGNATURES),
        "fold_model_fits": 7 * len(SEEDS) * 5,
        "equivalent_single_target_fits": len(CANCERS) * len(SIGNATURES) * len(SEEDS) * 5,
        "new_positive_raw_p_lt_0_05_in_both": int(comparison.new_positive_nominal_in_both.sum()),
        "new_positive_direction_in_both": int(comparison.new_positive_direction_in_both.sum()),
        "new_same_direction": int(comparison.new_same_direction.sum()),
        "new_nonestimable_tcga": int(comparison.new_tcga_pearson_r.isna().sum()),
        "new_nonestimable_cptac": int(comparison.new_cptac_pearson_r.isna().sum()),
    }
    if PAPER is not None:
        counts.update({
            "new_improved_r_over_paper_in_both": int(comparison.improved_r_in_both.sum()),
            "paper_positive_raw_p_lt_0_05_in_both": int(
                comparison.positive_nominal_p_lt_0_05_in_both_cohorts.sum()
            ),
        })
    audit = {
        "status": "complete",
        "protocol_sha256": sha256(PROTOCOL),
        "paper_comparison_source_sha256": sha256(PAPER) if PAPER is not None else None,
        "fixed_protocol_applied_without_task_specific_tuning": True,
        "historical_predictions_or_ensembles_used": False,
        "counts": counts,
        "input_sha256": hashes,
    }
    (results / "AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(counts, indent=2))
    print(comparison.loc[comparison.new_positive_nominal_in_both, [
        "pair", "new_tcga_n", "new_tcga_pearson_r", "new_tcga_pearson_p",
        "new_cptac_n", "new_cptac_pearson_r", "new_cptac_pearson_p",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
