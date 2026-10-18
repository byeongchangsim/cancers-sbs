#!/opt/anaconda/envs/cenv/bin/python
"""Build 2,048-tile coordinate-aligned RetCCL+CTransPath bags."""

from __future__ import annotations

import importlib.util
from pathlib import Path


BASE = Path("/home/marugoto-1.0.0-regression")
ROOT = BASE / "legacy30_retccl_ctranspath_feature_fusion_bag2048_20260829"
SOURCE = BASE / "legacy30_retccl_ctranspath_feature_fusion_20260829/build_fused_bags.py"


def load_implementation():
    spec = importlib.util.spec_from_file_location("fused_bag2048_impl", SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    module.OUT = ROOT / "fixed_bag_cache"
    module.BAG_SIZE = 2048
    module.BASE_SEED = 20260911
    return module


if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    load_implementation().main()
