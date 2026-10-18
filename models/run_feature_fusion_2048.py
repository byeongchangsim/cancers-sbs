#!/opt/anaconda/envs/cenv/bin/python
"""Run 2,048-tile multi-encoder target-attention MIL."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE / "legacy30_retccl_ctranspath_feature_fusion_bag2048_20260829"
SOURCE = BASE / "models/run_feature_fusion.py"


def load_implementation():
    spec = importlib.util.spec_from_file_location("feature_fusion1024_wrapper", SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE)
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    module = wrapper.load_implementation()
    module.ROOT = ROOT
    module.CACHE = ROOT / "fixed_bag_cache"
    module.BAG_SIZE = 2048
    module.BATCH_SIZE = 10
    module.FEATURE_CONTRACT = ROOT / "fixed_bag_cache/AUDIT.json"
    module.__file__ = str(Path(__file__).resolve())

    def register() -> None:
        path = ROOT / "REGISTERED_PROTOCOL.json"
        if path.exists():
            raise FileExistsError(path)
        counts = {"TCGA": {}, "CPTAC": {}}
        input_hashes = {
            "fixed_bag_audit": module.sha256(module.FEATURE_CONTRACT),
            "bag_builder": module.sha256(ROOT / "build_fused_bags_2048.py"),
            "training_wrapper": module.sha256(ROOT / "run_feature_fusion_2048.py"),
        }
        for dataset in ("TCGA", "CPTAC"):
            for cancer in module.CANCERS:
                meta_path = module.cache_prefix(dataset, cancer, "qhybrid").with_suffix(".meta.csv")
                meta = pd.read_csv(meta_path)
                counts[dataset][module.CANCER_LABEL[cancer]] = int(len(meta))
                if dataset == "TCGA":
                    input_hashes[f"{dataset}_{cancer}_labels"] = module.sha256(
                        module.LABEL_ROOT / dataset / cancer / f"{dataset}_{cancer}_process_endpoints.xlsx"
                    )
                else:
                    input_hashes[f"{dataset}_{cancer}_feature_meta"] = module.sha256(meta_path)
        payload = {
            "status": "registered_before_bag2048_training",
            "target": "deconstructSigs legacy30_min50 SBS1-SBS30 proportional exposures",
            "cancers_trained_separately": True,
            "encoder": "coordinate-aligned RetCCL 2048d + CTransPath 768d End2End/Marugoto features",
            "encoder_fusion": "separate 128d branches followed by shared 256d tile representation",
            "architecture": "30 target-specific gated-attention queries with composition and presence heads",
            "variants": list(module.VARIANTS), "output_types": list(module.OUTPUT_TYPES),
            "loss": "Hellinger composition + weighted BCE presence + square-root MSE + macro Pearson",
            "bag_size": 2048, "batch_size": module.BATCH_SIZE,
            "seeds": list(module.SEEDS), "folds": 5,
            "checkpoint_selection": "inner-validation median Pearson across eligible SBS outputs",
            "model_selection": "highest pooled three-seed TCGA OOF Pearson per cancer-SBS",
            "same_candidate_rule_for_all_210_pairs": True,
            "pair_specific_sign_flips": False,
            "patient_counts": counts, "input_sha256": input_hashes,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(path)

    module.register = register
    return module


if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    load_implementation().main()
