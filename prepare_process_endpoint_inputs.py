#!/opt/anaconda/envs/cenv/bin/python

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sbs_canonical import CANCER_PROCESS_MAP, PROCESS_DEFS


ROOT = Path("/home/marugoto-1.0.0-regression")
OUT = ROOT / "process_endpoint_inputs_legacy30_min50"

TCGA_LABELS = {
    "Brca": ROOT / "tcga_sigprof_paper_sweep/model_inputs/legacy30_min50_sbslabels/Brca/TCGA_BRCA_legacy30_min50_sbslabels.xlsx",
    "Ccrcc": ROOT / "tcga_sigprof_paper_sweep/model_inputs/legacy30_min50_sbslabels/Ccrcc/TCGA_KIRC_legacy30_min50_sbslabels.xlsx",
    "Hnscc": ROOT / "tcga_sigprof_paper_sweep/model_inputs/legacy30_min50_sbslabels/Hnscc/TCGA_HNSC_legacy30_min50_sbslabels.xlsx",
    "Lscc": ROOT / "tcga_sigprof_paper_sweep/model_inputs/legacy30_min50_sbslabels/Lscc/TCGA_LUSC_legacy30_min50_sbslabels.xlsx",
    "Luad": ROOT / "tcga_sigprof_paper_sweep/model_inputs/legacy30_min50_sbslabels/Luad/TCGA_LUAD_legacy30_min50_sbslabels.xlsx",
    "Ov": ROOT / "tcga_sigprof_paper_sweep/model_inputs/legacy30_min50_sbslabels/Ov/TCGA_OV_legacy30_min50_sbslabels.xlsx",
    "Ucec": ROOT / "tcga_sigprof_paper_sweep/model_inputs/legacy30_min50_sbslabels/Ucec/TCGA_UCEC_legacy30_min50_sbslabels.xlsx",
}

CPTAC_LABELS = {
    cancer: ROOT / f"cptac_inputs_legacy30_min50/by_cancer/{cancer}/CPTAC_{cancer}.xlsx"
    for cancer in ["Brca", "Ccrcc", "Hnscc", "Lscc", "Luad", "Ov", "Ucec"]
}

def add_processes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for proc, components in PROCESS_DEFS.items():
        available = [col for col in components if col in df.columns]
        if not available:
            df[proc] = pd.NA
            continue
        values = df[available].apply(pd.to_numeric, errors="coerce")
        df[proc] = values.sum(axis=1, min_count=1)
    return df


def write_one(kind: str, cancer: str, path: Path) -> list[dict[str, object]]:
    df = pd.read_excel(path)
    df = add_processes(df)
    out_dir = OUT / kind / cancer
    out_dir.mkdir(parents=True, exist_ok=True)
    out_xlsx = out_dir / f"{kind}_{cancer}_process_endpoints.xlsx"
    out_csv = out_dir / f"{kind}_{cancer}_process_endpoints.csv"
    df.to_excel(out_xlsx, index=False)
    df.to_csv(out_csv, index=False)

    rows = []
    for proc, components in PROCESS_DEFS.items():
        s = pd.to_numeric(df[proc], errors="coerce")
        rows.append(
            {
                "dataset": kind,
                "cancer": cancer,
                "process": proc,
                "associated_with_cancer": proc in CANCER_PROCESS_MAP.get(cancer, []),
                "components": ",".join(components),
                "n": int(s.notna().sum()),
                "nonzero": int((s.fillna(0) > 0).sum()),
                "nonzero_frac": float((s.fillna(0) > 0).mean()),
                "mean": float(s.mean()),
                "median": float(s.median()),
                "max": float(s.max()),
                "xlsx": str(out_xlsx.relative_to(ROOT)),
                "csv": str(out_csv.relative_to(ROOT)),
            }
        )
    return rows


def main() -> None:
    rows = []
    for cancer, path in TCGA_LABELS.items():
        rows.extend(write_one("TCGA", cancer, path))
    for cancer, path in CPTAC_LABELS.items():
        rows.extend(write_one("CPTAC", cancer, path))
    summary = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "process_endpoint_input_summary.csv", index=False)
    print(OUT)


if __name__ == "__main__":
    main()
