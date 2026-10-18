#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path("/home/marugoto-1.0.0-regression")
SOURCE_ROOT = ROOT / "tcga_sigprof_paper_sweep" / "by_config" / "legacy30_min50"
OUT_ROOT = ROOT / "tcga_sigprof_paper_sweep" / "model_inputs" / "legacy30_min50_sbslabels"

COHORTS = {
    "Brca": "BRCA",
    "Ccrcc": "KIRC",
    "Hnscc": "HNSC",
    "Lscc": "LUSC",
    "Luad": "LUAD",
    "Ov": "OV",
    "Ucec": "UCEC",
}


def convert_one(cohort: str, cancer: str) -> dict[str, object]:
    src = SOURCE_ROOT / cohort / f"TCGA_{cancer}_legacy30_min50_raw.xlsx"
    if not src.exists():
        raise FileNotFoundError(src)

    df = pd.read_excel(src)
    rename = {f"Signature.{idx}": f"SBS{idx}" for idx in range(1, 31)}
    missing = [col for col in rename if col not in df.columns]
    if missing:
        raise ValueError(f"{src} missing expected columns: {missing}")

    out_df = df.rename(columns=rename)
    out_dir = OUT_ROOT / cohort
    out_dir.mkdir(parents=True, exist_ok=True)
    out_xlsx = out_dir / f"TCGA_{cancer}_legacy30_min50_sbslabels.xlsx"
    out_csv = out_dir / f"TCGA_{cancer}_legacy30_min50_sbslabels.csv"
    out_df.to_excel(out_xlsx, index=False)
    out_df.to_csv(out_csv, index=False)

    sbs_cols = [f"SBS{idx}" for idx in range(1, 31)]
    return {
        "cohort": cohort,
        "cancer": cancer,
        "rows": len(out_df),
        "sbs_columns": len(sbs_cols),
        "nonzero_sbs": int((out_df[sbs_cols].sum(axis=0) > 0).sum()),
        "xlsx": str(out_xlsx),
        "csv": str(out_csv),
    }


def main() -> None:
    rows = [convert_one(cohort, cancer) for cohort, cancer in COHORTS.items()]
    summary = pd.DataFrame(rows)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_ROOT / "legacy30_min50_model_input_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"[OK] wrote model inputs to {OUT_ROOT}")


if __name__ == "__main__":
    main()
