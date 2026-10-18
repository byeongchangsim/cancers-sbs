#!/opt/anaconda/envs/cenv/bin/python
"""Run coordinate-aligned RetCCL+CTransPath multi-encoder target-attention MIL."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import torch
from torch import nn


BASE = Path("/home/marugoto-1.0.0-regression")
ROOT = BASE / "legacy30_retccl_ctranspath_feature_fusion_20260829"
SOURCE = (
    BASE / "legacy30_cancer_specific_target_attention_mil_20260829"
    / "run_target_attention_mil.py"
)
RETCCL_DIM = 2048
CTRANS_DIM = 768


def load_implementation():
    spec = importlib.util.spec_from_file_location("feature_fusion_impl", SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    module.CACHE = ROOT / "fixed_bag_cache"
    module.FEATURE_DIM = RETCCL_DIM + CTRANS_DIM
    module.BAG_SIZE = 1024
    module.BATCH_SIZE = 20
    module.FEATURE_CONTRACT = ROOT / "fixed_bag_cache/AUDIT.json"
    module.__file__ = str(Path(__file__).resolve())

    class MultiEncoderTargetAttentionMIL(nn.Module):
        def __init__(self, hybrid: bool) -> None:
            super().__init__()
            self.hybrid = hybrid
            branch_dim = module.HIDDEN_DIM // 2
            self.retccl_encoder = nn.Sequential(
                nn.Linear(RETCCL_DIM, branch_dim), nn.ReLU(),
                nn.LayerNorm(branch_dim), nn.Dropout(0.1),
            )
            self.ctranspath_encoder = nn.Sequential(
                nn.Linear(CTRANS_DIM, branch_dim), nn.ReLU(),
                nn.LayerNorm(branch_dim), nn.Dropout(0.1),
            )
            self.encoder_fusion = nn.Sequential(
                nn.Linear(branch_dim * 2, module.HIDDEN_DIM), nn.ReLU(),
                nn.LayerNorm(module.HIDDEN_DIM), nn.Dropout(0.1),
            )
            self.attention_v = nn.Sequential(
                nn.Linear(module.HIDDEN_DIM, module.ATTENTION_DIM), nn.Tanh(),
            )
            self.attention_u = nn.Sequential(
                nn.Linear(module.HIDDEN_DIM, module.ATTENTION_DIM), nn.Sigmoid(),
            )
            self.attention_w = nn.Linear(module.ATTENTION_DIM, 30)
            if hybrid:
                self.stat_fusion = nn.Sequential(
                    nn.Linear(module.HIDDEN_DIM * 3, module.HIDDEN_DIM), nn.ReLU(),
                    nn.Dropout(0.1),
                )
            self.magnitude_weight = nn.Parameter(torch.empty(30, module.HIDDEN_DIM))
            self.magnitude_bias = nn.Parameter(torch.zeros(30))
            self.presence_weight = nn.Parameter(torch.empty(30, module.HIDDEN_DIM))
            self.presence_bias = nn.Parameter(torch.zeros(30))
            self.residual_head = nn.Linear(module.HIDDEN_DIM, 1)
            nn.init.xavier_uniform_(self.magnitude_weight)
            nn.init.xavier_uniform_(self.presence_weight)

        def forward(self, x: torch.Tensor, lens: torch.Tensor):
            retccl = self.retccl_encoder(x[:, :, :RETCCL_DIM])
            ctranspath = self.ctranspath_encoder(x[:, :, RETCCL_DIM:])
            h = self.encoder_fusion(torch.cat([retccl, ctranspath], dim=2))
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
                mean_expanded = mean.unsqueeze(1).expand(-1, 30, -1)
                std_expanded = std.unsqueeze(1).expand(-1, 30, -1)
                pooled = self.stat_fusion(
                    torch.cat([pooled, mean_expanded, std_expanded], dim=2)
                )
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

    module.TargetAttentionMIL = MultiEncoderTargetAttentionMIL

    def register_feature_fusion() -> None:
        path = ROOT / "REGISTERED_PROTOCOL.json"
        if path.exists():
            raise FileExistsError(path)
        counts = {"TCGA": {}, "CPTAC": {}}
        input_hashes = {
            "fixed_bag_audit": module.sha256(module.FEATURE_CONTRACT),
            "bag_builder": module.sha256(ROOT / "build_fused_bags.py"),
            "training_wrapper": module.sha256(ROOT / "run_feature_fusion.py"),
        }
        for dataset in ("TCGA", "CPTAC"):
            for cancer in module.CANCERS:
                meta = pd.read_csv(
                    module.cache_prefix(dataset, cancer, "qhybrid").with_suffix(".meta.csv")
                )
                counts[dataset][module.CANCER_LABEL[cancer]] = int(len(meta))
                if dataset == "TCGA":
                    input_hashes[f"{dataset}_{cancer}_labels"] = module.sha256(
                        module.LABEL_ROOT / dataset / cancer
                        / f"{dataset}_{cancer}_process_endpoints.xlsx"
                    )
                else:
                    input_hashes[f"{dataset}_{cancer}_feature_meta"] = module.sha256(
                        module.cache_prefix(dataset, cancer, "qhybrid").with_suffix(".meta.csv")
                    )
        payload = {
            "status": "registered_before_feature-fusion_training",
            "target": "deconstructSigs legacy30_min50 SBS1-SBS30 proportional exposures",
            "cancers_trained_separately": True,
            "encoder": "coordinate-aligned RetCCL 2048-dimensional and CTransPath 768-dimensional End2End/Marugoto features",
            "encoder_fusion": "separate 128-dimensional learned branches followed by a shared 256-dimensional tile representation",
            "architecture": "30 target-specific gated-attention queries with composition and presence heads",
            "variants": list(module.VARIANTS),
            "output_types": list(module.OUTPUT_TYPES),
            "loss": "Hellinger composition + weighted BCE presence + square-root MSE + macro Pearson",
            "bag_size": module.BAG_SIZE, "batch_size": module.BATCH_SIZE,
            "seeds": list(module.SEEDS), "folds": 5,
            "checkpoint_selection": "inner-validation median Pearson across eligible SBS outputs",
            "model_selection": "highest pooled three-seed TCGA OOF Pearson per cancer-SBS",
            "same_candidate_rule_for_all_210_pairs": True,
            "pair_specific_sign_flips": False,
            "patient_counts": counts, "input_sha256": input_hashes,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(path)

    module.register = register_feature_fusion
    return module


if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    load_implementation().main()
