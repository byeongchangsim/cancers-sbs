#!/opt/anaconda/envs/cenv/bin/python
"""Cancer-specific target-attention MIL for legacy deconstructSigs SBS1-SBS30.

One model is trained per cancer and predicts the complete 30-signature
composition.  Each SBS has its own gated-attention query.  A second head learns
signature presence.  CPTAC outcomes are not opened during training, aggregation,
or TCGA-only model selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import threading
from pathlib import Path
from queue import Empty, Queue

import numpy as np
import pandas as pd
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold, train_test_split
from torch import nn


BASE = Path("/home/marugoto-1.0.0-regression")
ROOT = BASE / "legacy30_cancer_specific_target_attention_mil_20260829"
CACHE = BASE / "sigprofiler_improved_20260818/inputs/end2end_retccl_qc_v1/fixed_bag_cache"
LABEL_ROOT = BASE / "process_endpoint_inputs_legacy30_min50"
FEATURE_CONTRACT = BASE / "sigprofiler_improved_20260818/inputs/end2end_retccl_qc_v1/CONTRACT.json"

CANCERS = ("Brca", "Ccrcc", "Hnscc", "Luad", "Lscc", "Ov", "Ucec")
CANCER_LABEL = {
    "Brca": "BRCA", "Ccrcc": "KIRC", "Hnscc": "HNSC", "Luad": "LUAD",
    "Lscc": "LUSC", "Ov": "OV", "Ucec": "UCEC",
}
SBS_COLUMNS = tuple(f"SBS{i}" for i in range(1, 31))
N_SIGNATURES = len(SBS_COLUMNS)
SEEDS = (20260829, 20260830, 20260831)
VARIANTS = (
    "qhybrid_target_attention",
    "qhybrid_target_attention_hybrid",
    "uniform_target_attention",
    "uniform_target_attention_hybrid",
)
OUTPUT_TYPES = ("composition", "hurdle", "presence")
FEATURE_DIM = 2048
HIDDEN_DIM = 256
ATTENTION_DIM = 128
BAG_SIZE = 1024
BATCH_SIZE = 24
MAX_EPOCHS = 80
PATIENCE = 12
WORKER_ARGS = ()
TARGET_DESCRIPTION = "deconstructSigs legacy30_min50 SBS1-SBS30 proportional exposures"
REQUIRE_COMPLETE_TCGA_LABELS = True


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def correlations(target: np.ndarray, prediction: np.ndarray) -> dict:
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if (len(target) < 3 or not np.isfinite(target).all()
            or not np.isfinite(prediction).all()
            or np.std(target) <= 0 or np.std(prediction) <= 0):
        return {
            "n": int(len(target)), "nonzero": int(np.sum(target > 0)),
            "pearson_r": np.nan, "pearson_p": np.nan,
            "spearman_rho": np.nan, "spearman_p": np.nan,
        }
    r, p = pearsonr(target, prediction)
    rho, rho_p = spearmanr(target, prediction)
    return {
        "n": int(len(target)), "nonzero": int(np.sum(target > 0)),
        "pearson_r": float(r), "pearson_p": float(p),
        "spearman_rho": float(rho), "spearman_p": float(rho_p),
    }


def variant_parts(variant: str) -> tuple[str, bool]:
    if variant not in VARIANTS:
        raise ValueError(variant)
    sampling = "qhybrid" if variant.startswith("qhybrid") else "uniform"
    return sampling, variant.endswith("_hybrid")


def cache_prefix(dataset: str, cancer: str, sampling: str) -> Path:
    return CACHE / sampling / f"{dataset}_{cancer}_bag{BAG_SIZE}"


def load_feature_bags(dataset: str, cancer: str, sampling: str):
    prefix = cache_prefix(dataset, cancer, sampling)
    meta = pd.read_csv(prefix.with_suffix(".meta.csv"), dtype={"PATIENT": str})
    source = np.load(prefix.with_suffix(".bags.npy"), mmap_mode="r")
    if len(meta) != len(source):
        raise ValueError(f"Feature/meta length mismatch: {prefix}")
    return (
        meta.PATIENT.astype(str).to_numpy(dtype=object),
        np.asarray(source, dtype=np.float16),
        meta.lens.to_numpy(dtype=np.int64),
    )


def load_tcga(cancer: str, sampling: str):
    patients, bags, lens = load_feature_bags("TCGA", cancer, sampling)
    labels = pd.read_excel(
        LABEL_ROOT / "TCGA" / cancer / f"TCGA_{cancer}_process_endpoints.xlsx",
        usecols=["PATIENT", *SBS_COLUMNS],
    )
    labels["PATIENT"] = labels.PATIENT.astype(str)
    meta = pd.DataFrame({"PATIENT": patients, "source_index": np.arange(len(patients))})
    merged = meta.merge(labels, on="PATIENT", how="inner", validate="one_to_one")
    if REQUIRE_COMPLETE_TCGA_LABELS and len(merged) != len(meta):
        raise ValueError(f"Missing TCGA labels for {cancer}: {len(merged)}/{len(meta)}")
    y30 = merged[list(SBS_COLUMNS)].to_numpy(dtype=np.float32)
    residual = np.maximum(1.0 - y30.sum(axis=1, keepdims=True), 0.0)
    composition = np.concatenate([y30, residual], axis=1)
    denominator = composition.sum(axis=1, keepdims=True)
    if np.any(denominator <= 0):
        raise ValueError(f"Invalid composition denominator: {cancer}")
    composition /= denominator
    idx = merged.source_index.to_numpy(dtype=np.int64)
    return (
        merged.PATIENT.to_numpy(dtype=object),
        bags[idx], lens[idx], composition.astype(np.float32),
    )


class TargetAttentionMIL(nn.Module):
    def __init__(self, hybrid: bool) -> None:
        super().__init__()
        self.hybrid = hybrid
        self.encoder = nn.Sequential(
            nn.Linear(FEATURE_DIM, HIDDEN_DIM), nn.ReLU(), nn.Dropout(0.1)
        )
        self.attention_v = nn.Sequential(nn.Linear(HIDDEN_DIM, ATTENTION_DIM), nn.Tanh())
        self.attention_u = nn.Sequential(nn.Linear(HIDDEN_DIM, ATTENTION_DIM), nn.Sigmoid())
        self.attention_w = nn.Linear(ATTENTION_DIM, N_SIGNATURES)
        if hybrid:
            self.fusion = nn.Sequential(
                nn.Linear(HIDDEN_DIM * 3, HIDDEN_DIM), nn.ReLU(), nn.Dropout(0.1)
            )
        self.magnitude_weight = nn.Parameter(torch.empty(N_SIGNATURES, HIDDEN_DIM))
        self.magnitude_bias = nn.Parameter(torch.zeros(N_SIGNATURES))
        self.presence_weight = nn.Parameter(torch.empty(N_SIGNATURES, HIDDEN_DIM))
        self.presence_bias = nn.Parameter(torch.zeros(N_SIGNATURES))
        self.residual_head = nn.Linear(HIDDEN_DIM, 1)
        nn.init.xavier_uniform_(self.magnitude_weight)
        nn.init.xavier_uniform_(self.presence_weight)

    def forward(self, x: torch.Tensor, lens: torch.Tensor):
        h = self.encoder(x)
        positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        valid = positions < lens.unsqueeze(1)
        attention_logits = self.attention_w(
            self.attention_v(h) * self.attention_u(h)
        )
        attention_logits = torch.where(
            valid.unsqueeze(-1), attention_logits,
            torch.full_like(attention_logits, -1e4),
        )
        attention = torch.softmax(attention_logits, dim=1)
        pooled = torch.einsum("btk,bth->bkh", attention, h)
        mask = valid.unsqueeze(-1).to(h.dtype)
        denominator = lens.clamp_min(1).to(h.dtype).unsqueeze(1)
        mean = torch.sum(h * mask, dim=1) / denominator
        if self.hybrid:
            variance = torch.sum(
                (h - mean.unsqueeze(1)).square() * mask, dim=1
            ) / denominator
            std = torch.sqrt(variance.clamp_min(1e-6))
            mean_expanded = mean.unsqueeze(1).expand(-1, N_SIGNATURES, -1)
            std_expanded = std.unsqueeze(1).expand(-1, N_SIGNATURES, -1)
            pooled = self.fusion(torch.cat([pooled, mean_expanded, std_expanded], dim=2))
        magnitude_logits = torch.sum(
            pooled * self.magnitude_weight.unsqueeze(0), dim=2
        ) + self.magnitude_bias
        residual_logit = self.residual_head(mean)
        composition = torch.softmax(
            torch.cat([magnitude_logits, residual_logit], dim=1), dim=1
        )
        presence_logits = torch.sum(
            pooled * self.presence_weight.unsqueeze(0), dim=2
        ) + self.presence_bias
        return composition, presence_logits


def make_predictions(composition: torch.Tensor, presence_logits: torch.Tensor) -> dict:
    exposure = composition[:, :N_SIGNATURES]
    presence = torch.sigmoid(presence_logits)
    return {
        "composition": exposure,
        "hurdle": exposure * presence,
        "presence": presence,
    }


def macro_pearson_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_centered = prediction - prediction.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    target_ss = torch.sum(target_centered.square(), dim=0)
    eligible = torch.sum(target > 0, dim=0) >= 3
    eligible &= target_ss > 1e-8
    if not torch.any(eligible):
        return prediction.sum() * 0.0
    numerator = torch.sum(
        pred_centered[:, eligible] * target_centered[:, eligible], dim=0
    )
    pred_ss = torch.sum(pred_centered[:, eligible].square(), dim=0)
    denominator = torch.sqrt(
        pred_ss.clamp_min(1e-8) * target_ss[eligible].clamp_min(1e-8)
    )
    correlation = numerator / denominator
    return 1.0 - correlation.mean()


def loss_function(
    composition: torch.Tensor,
    presence_logits: torch.Tensor,
    target: torch.Tensor,
    positive_weight: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    hellinger = torch.mean(torch.sum(
        (torch.sqrt(composition.clamp_min(1e-8))
         - torch.sqrt(target.clamp_min(1e-8))).square(), dim=1
    ))
    target30 = target[:, :N_SIGNATURES]
    presence_target = (target30 > 0).to(target30.dtype)
    presence = F.binary_cross_entropy_with_logits(
        presence_logits, presence_target, pos_weight=positive_weight
    )
    exposure = composition[:, :N_SIGNATURES]
    sqrt_mse = F.mse_loss(
        torch.sqrt(exposure.clamp_min(1e-8)),
        torch.sqrt(target30.clamp_min(1e-8)),
    )
    correlation = macro_pearson_loss(exposure, target30)
    total = hellinger + 0.25 * presence + 0.5 * sqrt_mse + 0.10 * correlation
    return total, {
        "hellinger": float(hellinger.detach()),
        "presence": float(presence.detach()),
        "sqrt_mse": float(sqrt_mse.detach()),
        "correlation": float(correlation.detach()),
    }


def batch_data(bags: torch.Tensor, lens: np.ndarray, indices: np.ndarray):
    idx = torch.as_tensor(indices, device=bags.device, dtype=torch.long)
    return bags[idx], torch.as_tensor(lens[indices], device=bags.device, dtype=torch.long)


def predict(
    model: nn.Module, bags: torch.Tensor, lens: np.ndarray, indices: np.ndarray
) -> dict[str, np.ndarray]:
    model.eval()
    values = {name: [] for name in OUTPUT_TYPES}
    with torch.inference_mode():
        for start in range(0, len(indices), BATCH_SIZE):
            batch = indices[start:start + BATCH_SIZE]
            x, x_lens = batch_data(bags, lens, batch)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                composition, presence_logits = model(x, x_lens)
            outputs = make_predictions(composition.float(), presence_logits.float())
            for name, output in outputs.items():
                values[name].append(output.cpu().numpy())
    return {name: np.concatenate(parts) for name, parts in values.items()}


def validation_score(target: np.ndarray, prediction: np.ndarray) -> float:
    values = []
    for index in range(N_SIGNATURES):
        if np.sum(target[:, index] > 0) < 3:
            continue
        result = correlations(target[:, index], prediction[:, index])
        if np.isfinite(result["pearson_r"]):
            values.append(result["pearson_r"])
    return float(np.median(values)) if values else -np.inf


def output_dir(variant: str, cancer: str, seed_index: int, fold: int) -> Path:
    return ROOT / "variants" / variant / "models" / cancer / f"seed-{seed_index}" / f"fold-{fold}"


def train_fold(variant: str, cancer: str, seed_index: int, fold: int, device_name: str) -> None:
    sampling, hybrid = variant_parts(variant)
    seed = SEEDS[seed_index]
    out = output_dir(variant, cancer, seed_index, fold)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    np.random.seed(seed + fold)
    torch.manual_seed(seed + fold)
    torch.cuda.manual_seed_all(seed + fold)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(device_name)

    patients, bags_cpu, lens, y = load_tcga(cancer, sampling)
    cpatients, cbags_cpu, clens = load_feature_bags("CPTAC", cancer, sampling)
    bags = torch.tensor(bags_cpu, device=device, dtype=torch.float16)
    cbags = torch.tensor(cbags_cpu, device=device, dtype=torch.float16)
    indices = np.arange(len(y))
    outer = KFold(5, shuffle=True, random_state=seed)
    train_outer, test = list(outer.split(indices))[fold]
    train, valid = train_test_split(train_outer, test_size=0.2, random_state=seed + fold)

    presence_count = np.sum(y[train, :N_SIGNATURES] > 0, axis=0)
    negative_count = len(train) - presence_count
    positive_weight = np.clip(
        negative_count / np.maximum(presence_count, 1), 1.0, 20.0
    ).astype(np.float32)
    positive_weight_tensor = torch.as_tensor(positive_weight, device=device)

    model = TargetAttentionMIL(hybrid).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best_score = -np.inf
    best_state = None
    best_epoch = None
    stale = 0
    history = []
    for epoch in range(MAX_EPOCHS):
        rng = np.random.default_rng(seed * 1000 + fold * 100 + epoch)
        epoch_indices = train.copy()
        rng.shuffle(epoch_indices)
        model.train()
        epoch_losses = []
        for start in range(0, len(epoch_indices), BATCH_SIZE):
            batch = epoch_indices[start:start + BATCH_SIZE]
            x, x_lens = batch_data(bags, lens, batch)
            target = torch.as_tensor(y[batch], device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                composition, presence_logits = model(x, x_lens)
            loss, components = loss_function(
                composition.float(), presence_logits.float(), target, positive_weight_tensor
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        valid_outputs = predict(model, bags, lens, valid)
        score = validation_score(y[valid, :N_SIGNATURES], valid_outputs["composition"])
        history.append({
            "epoch": epoch + 1,
            "train_loss": float(np.mean(epoch_losses)),
            "validation_median_pearson_r": score,
            **{f"last_batch_{key}": value for key, value in components.items()},
        })
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if best_state is None:
        raise RuntimeError("No finite validation checkpoint")
    model.load_state_dict(best_state)
    test_outputs = predict(model, bags, lens, test)
    cptac_outputs = predict(model, cbags, clens, np.arange(len(cpatients)))

    tcga_values = {"PATIENT": patients[test]}
    cptac_values = {"PATIENT": cpatients}
    for index, signature in enumerate(SBS_COLUMNS):
        tcga_values[signature] = y[test, index]
        for output_type in OUTPUT_TYPES:
            tcga_values[f"{output_type}_{signature}"] = test_outputs[output_type][:, index]
            cptac_values[f"{output_type}_{signature}"] = cptac_outputs[output_type][:, index]
    tcga_frame = pd.DataFrame(tcga_values)
    tcga_frame.to_csv(out / "tcga_test_predictions.tsv", sep="\t", index=False)
    cptac_frame = pd.DataFrame(cptac_values)
    cptac_frame.to_csv(out / "cptac_label_free_predictions.tsv", sep="\t", index=False)
    pd.DataFrame(history).to_csv(out / "history.tsv", sep="\t", index=False)
    torch.save({
        "state_dict": best_state, "variant": variant, "cancer": cancer,
        "seed": seed, "seed_index": seed_index, "fold": fold,
        "train_patients": patients[train].tolist(),
        "validation_patients": patients[valid].tolist(),
        "test_patients": patients[test].tolist(),
        "positive_weight": positive_weight.tolist(),
    }, out / "best.pt")
    result = {
        "status": "complete", "variant": variant, "cancer": cancer,
        "seed": seed, "seed_index": seed_index, "fold": fold,
        "best_epoch": best_epoch, "best_validation_median_r": best_score,
        "device": torch.cuda.get_device_name(device),
        "cptac_outcomes_opened": False,
    }
    (out / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def aggregate_one_variant(variant: str) -> list[dict]:
    aggregate_root = ROOT / "variants" / variant / "aggregated"
    if aggregate_root.exists():
        raise FileExistsError(aggregate_root)
    aggregate_root.mkdir(parents=True)
    metrics = []
    for cancer in CANCERS:
        seed_tcga = []
        seed_cptac = []
        for seed_index in range(3):
            tcga_parts = []
            cptac_parts = []
            for fold in range(5):
                out = output_dir(variant, cancer, seed_index, fold)
                tcga_parts.append(pd.read_csv(out / "tcga_test_predictions.tsv", sep="\t"))
                cptac = pd.read_csv(out / "cptac_label_free_predictions.tsv", sep="\t")
                pred_columns = [column for column in cptac if column != "PATIENT"]
                cptac_parts.append(cptac.rename(
                    columns={column: f"{column}_fold{fold}" for column in pred_columns}
                ))
            tcga = pd.concat(tcga_parts, ignore_index=True).sort_values("PATIENT")
            if tcga.PATIENT.duplicated().any():
                raise ValueError(f"Duplicate TCGA OOF patients: {variant}/{cancer}/seed{seed_index}")
            cptac = cptac_parts[0]
            for frame in cptac_parts[1:]:
                cptac = cptac.merge(frame, on="PATIENT", validate="one_to_one")
            final_cptac = pd.DataFrame({"PATIENT": cptac.PATIENT})
            for signature in SBS_COLUMNS:
                for output_type in OUTPUT_TYPES:
                    column = f"{output_type}_{signature}"
                    final_cptac[column] = cptac[
                        [f"{column}_fold{fold}" for fold in range(5)]
                    ].mean(axis=1)
            seed_tcga.append(tcga)
            seed_cptac.append(final_cptac)

        tcga_patients = seed_tcga[0].PATIENT.to_numpy(dtype=object)
        cptac_patients = seed_cptac[0].PATIENT.to_numpy(dtype=object)
        tcga_values = {
            "PATIENT": tcga_patients,
            **{signature: seed_tcga[0][signature].to_numpy() for signature in SBS_COLUMNS},
        }
        cptac_values = {"PATIENT": cptac_patients}
        for signature in SBS_COLUMNS:
            for output_type in OUTPUT_TYPES:
                column = f"{output_type}_{signature}"
                tcga_values[column] = np.mean(
                    [frame.set_index("PATIENT").reindex(tcga_patients)[column].to_numpy()
                     for frame in seed_tcga], axis=0,
                )
                cptac_values[column] = np.mean(
                    [frame.set_index("PATIENT").reindex(cptac_patients)[column].to_numpy()
                     for frame in seed_cptac], axis=0,
                )
        tcga_final = pd.DataFrame(tcga_values)
        cptac_final = pd.DataFrame(cptac_values)
        for signature in SBS_COLUMNS:
            for output_type in OUTPUT_TYPES:
                column = f"{output_type}_{signature}"
                values = correlations(tcga_final[signature], tcga_final[column])
                metrics.append({
                    "variant": variant, "cancer": CANCER_LABEL[cancer],
                    "signature": signature, "output_type": output_type,
                    "dataset": "TCGA", **values,
                })
        tcga_final.to_csv(
            aggregate_root / f"{cancer.lower()}_tcga_three_seed.tsv", sep="\t", index=False
        )
        cptac_final.to_csv(
            aggregate_root / f"{cancer.lower()}_cptac_label_free_three_seed.tsv",
            sep="\t", index=False,
        )
    return metrics


def aggregate_and_lock() -> None:
    selection_path = ROOT / "LOCKED_SELECTION.tsv"
    if selection_path.exists():
        raise FileExistsError(selection_path)
    all_metrics = []
    for variant in VARIANTS:
        all_metrics.extend(aggregate_one_variant(variant))
    metric_frame = pd.DataFrame(all_metrics)
    metric_frame.to_csv(ROOT / "TCGA_CANDIDATE_METRICS.tsv", sep="\t", index=False)
    selected = []
    for cancer in CANCER_LABEL.values():
        for signature in SBS_COLUMNS:
            candidates = metric_frame[
                metric_frame.cancer.eq(cancer) & metric_frame.signature.eq(signature)
            ].copy()
            candidates["selection_score"] = candidates.pearson_r.fillna(-np.inf)
            best = candidates.sort_values(
                ["selection_score", "variant", "output_type"],
                ascending=[False, True, True],
            ).iloc[0]
            selected.append(best.to_dict())
    locked = pd.DataFrame(selected).drop(columns="selection_score")
    locked.insert(0, "pair", locked.cancer + "\u2013" + locked.signature)
    locked.to_csv(selection_path, sep="\t", index=False)
    contract = {
        "status": "locked_before_cptac_outcomes",
        "selection": "highest three-seed TCGA OOF Pearson r among 12 candidates per pair",
        "candidate_definition": "4 sampling/pooling variants x composition/hurdle/presence outputs",
        "cptac_outcomes_opened": False,
        "locked_selection_sha256": sha256(selection_path),
        "tcga_candidate_metrics_sha256": sha256(ROOT / "TCGA_CANDIDATE_METRICS.tsv"),
    }
    (ROOT / "SELECTION_CONTRACT.json").write_text(json.dumps(contract, indent=2) + "\n")
    print(json.dumps(contract, indent=2))


def evaluate_cptac() -> None:
    result_root = ROOT / "results"
    if result_root.exists():
        raise FileExistsError(result_root)
    contract = json.loads((ROOT / "SELECTION_CONTRACT.json").read_text())
    selection_path = ROOT / "LOCKED_SELECTION.tsv"
    if sha256(selection_path) != contract["locked_selection_sha256"]:
        raise ValueError("Locked selection hash mismatch")
    locked = pd.read_csv(selection_path, sep="\t")
    rows = []
    prediction_root = result_root / "predictions"
    prediction_root.mkdir(parents=True)
    for row in locked.itertuples(index=False):
        cancer_key = next(key for key, value in CANCER_LABEL.items() if value == row.cancer)
        source = ROOT / "variants" / row.variant / "aggregated" / (
            f"{cancer_key.lower()}_cptac_label_free_three_seed.tsv"
        )
        prediction = pd.read_csv(source, sep="\t")[[
            "PATIENT", f"{row.output_type}_{row.signature}"
        ]].rename(columns={f"{row.output_type}_{row.signature}": "prediction"})
        labels = pd.read_excel(
            LABEL_ROOT / "CPTAC" / cancer_key / f"CPTAC_{cancer_key}_process_endpoints.xlsx",
            usecols=["PATIENT", row.signature],
        )
        labels.PATIENT = labels.PATIENT.astype(str)
        joined = labels.merge(prediction, on="PATIENT", validate="one_to_one")
        joined.to_csv(
            prediction_root / f"{row.cancer.lower()}_{row.signature.lower()}.tsv",
            sep="\t", index=False,
        )
        external = correlations(joined[row.signature], joined.prediction)
        rows.append({
            "pair": row.pair, "cancer": row.cancer, "signature": row.signature,
            "selected_variant": row.variant, "selected_output_type": row.output_type,
            "tcga_n": int(row.n), "tcga_nonzero": int(row.nonzero),
            "tcga_pearson_r": row.pearson_r, "tcga_pearson_p": row.pearson_p,
            "cptac_n": external["n"], "cptac_nonzero": external["nonzero"],
            "cptac_pearson_r": external["pearson_r"],
            "cptac_pearson_p": external["pearson_p"],
            "positive_raw_p_in_both": bool(
                row.pearson_r > 0 and row.pearson_p < 0.05
                and external["pearson_r"] > 0 and external["pearson_p"] < 0.05
            ),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(result_root / "summary.tsv", sep="\t", index=False)
    passed = summary[summary.positive_raw_p_in_both].copy()
    passed.to_csv(result_root / "positive_raw_p_in_both.tsv", sep="\t", index=False)
    failures = list(ROOT.glob("variants/*/models/*/seed-*/fold-*/FAILURE.json"))
    audit = {
        "status": "complete",
        "training_jobs": len(VARIANTS) * len(CANCERS) * len(SEEDS) * 5,
        "training_failures": [str(path.relative_to(ROOT)) for path in failures],
        "tcga_candidates_per_pair": len(VARIANTS) * len(OUTPUT_TYPES),
        "locked_pairs": int(len(locked)),
        "cptac_used_for_model_selection": False,
        "locked_selection_sha256_verified": True,
        "positive_raw_p_in_both": int(len(passed)),
        "audit_pass": bool(
            len(locked) == len(CANCERS) * len(SBS_COLUMNS) and not failures
        ),
    }
    (result_root / "AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    print(passed.to_string(index=False))
    if not audit["audit_pass"]:
        raise RuntimeError("Evaluation audit failed")


def register() -> None:
    path = ROOT / "REGISTERED_PROTOCOL.json"
    if path.exists():
        raise FileExistsError(path)
    counts = {"TCGA": {}, "CPTAC": {}}
    input_hashes = {"feature_contract": sha256(FEATURE_CONTRACT)}
    for dataset in ("TCGA", "CPTAC"):
        for cancer in CANCERS:
            meta = pd.read_csv(
                cache_prefix(dataset, cancer, "qhybrid").with_suffix(".meta.csv")
            )
            counts[dataset][CANCER_LABEL[cancer]] = int(len(meta))
            if dataset == "TCGA":
                input_hashes[f"{dataset}_{cancer}_labels"] = sha256(
                    LABEL_ROOT / dataset / cancer / f"{dataset}_{cancer}_process_endpoints.xlsx"
                )
            else:
                # Register only label-free CPTAC feature metadata before the
                # TCGA-only model choices are locked.
                input_hashes[f"{dataset}_{cancer}_feature_meta"] = sha256(
                    cache_prefix(dataset, cancer, "qhybrid").with_suffix(".meta.csv")
                )
    payload = {
        "status": "frozen_before_training",
        "target": TARGET_DESCRIPTION,
        "cancers_trained_separately": True,
        "encoder": "QC re-extracted RetCCL ResNet50 from End2End/Marugoto workflow",
        "architecture": (
            f"{N_SIGNATURES} target-specific gated-attention queries with "
            "composition and presence heads"
        ),
        "variants": list(VARIANTS),
        "output_types": list(OUTPUT_TYPES),
        "loss": "Hellinger composition + weighted BCE presence + square-root MSE + macro Pearson",
        "bag_size": BAG_SIZE, "batch_size": BATCH_SIZE,
        "seeds": list(SEEDS), "folds": 5,
        "checkpoint_selection": "inner-validation median Pearson across eligible SBS outputs",
        "model_selection": "highest pooled three-seed TCGA OOF Pearson per cancer-SBS",
        "require_complete_tcga_labels": REQUIRE_COMPLETE_TCGA_LABELS,
        "cptac_outcomes_opened_during_training_or_selection": False,
        "patient_counts": counts,
        "input_sha256": input_hashes,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(path)


def launch(gpus: str) -> None:
    gpu_ids = [value.strip() for value in gpus.split(",") if value.strip()]
    jobs: Queue[tuple[str, str, int, int]] = Queue()
    for variant in VARIANTS:
        for cancer in CANCERS:
            for seed_index in range(3):
                for fold in range(5):
                    if not (output_dir(variant, cancer, seed_index, fold) / "RESULT.json").exists():
                        jobs.put((variant, cancer, seed_index, fold))
    failures = []
    lock = threading.Lock()
    log_root = ROOT / "logs"
    log_root.mkdir(parents=True, exist_ok=True)

    def worker(gpu: str) -> None:
        while True:
            try:
                variant, cancer, seed_index, fold = jobs.get_nowait()
            except Empty:
                return
            log = log_root / f"{variant}_{cancer}_seed{seed_index}_fold{fold}.log"
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            command = [
                "/opt/anaconda/envs/cenv/bin/python", str(Path(__file__).resolve()),
                *WORKER_ARGS,
                "train-fold", "--variant", variant, "--cancer", cancer,
                "--seed-index", str(seed_index), "--fold", str(fold),
                "--device", "cuda:0",
            ]
            with log.open("w") as handle:
                completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=env)
            if completed.returncode:
                failure = {
                    "variant": variant, "cancer": cancer, "seed_index": seed_index,
                    "fold": fold, "gpu": gpu, "returncode": completed.returncode,
                    "log": str(log),
                }
                with lock:
                    failures.append(failure)
                    (ROOT / "FAILURES_RUNNING.json").write_text(
                        json.dumps(failures, indent=2) + "\n"
                    )
            jobs.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpu_ids]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    (ROOT / "FAILURES.json").write_text(json.dumps(failures, indent=2) + "\n")
    if failures:
        raise RuntimeError(f"{len(failures)} jobs failed")
    aggregate_and_lock()


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("register")
    train = sub.add_parser("train-fold")
    train.add_argument("--variant", choices=VARIANTS, required=True)
    train.add_argument("--cancer", choices=CANCERS, required=True)
    train.add_argument("--seed-index", type=int, choices=[0, 1, 2], required=True)
    train.add_argument("--fold", type=int, choices=range(5), required=True)
    train.add_argument("--device", default="cuda:0")
    launch_parser = sub.add_parser("launch")
    launch_parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    sub.add_parser("evaluate-cptac")
    args = parser.parse_args()
    if args.command == "register":
        register()
    elif args.command == "train-fold":
        train_fold(args.variant, args.cancer, args.seed_index, args.fold, args.device)
    elif args.command == "launch":
        launch(args.gpus)
    elif args.command == "evaluate-cptac":
        evaluate_cptac()


if __name__ == "__main__":
    main()
