#!/usr/bin/env python3
"""Generate SBS96 with SigProfilerMatrixGenerator and fit exact COSMIC v2 SBS1-30.

The CPTAC MAFs in this workspace are column-name compatible but not in the fixed
column order expected by SigProfilerMatrixGenerator's legacy MAF reader.  This
script therefore reads the named MAF fields, writes the documented simple-text
format, generates exome SBS96 matrices, and then runs SigProfilerAssignment.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE / "sigprofiler_mg_assignment_v2_sbs30_currentmaf_v2_27_20260904"
SPA_PYTHON = BASE / "harmonized_sbs3_grch38_20260726/.venv/bin/python"
BASE_VENDOR = BASE / "harmonized_sbs3_grch38_20260726/python_packages"
SPA_VENDOR = BASE / "sigprofilerassignment_v1.1.5_vendor"
VENDORS = (SPA_VENDOR, BASE_VENDOR)
REFERENCE_ROOT = SPA_VENDOR / "SigProfilerAssignment/data/Reference_Signatures"
EXCEL_SITE = Path("/opt/anaconda/envs/cenv/lib/python3.8/site-packages")
AGG = BASE / "sigprofiler_improved_20260818/inputs/end2end_retccl_qc_v1/patient_aggregates"

CANCERS = {
    "Brca": "BRCA",
    "Ccrcc": "KIRC",
    "Hnscc": "HNSC",
    "Luad": "LUAD",
    "Lscc": "LUSC",
    "Ov": "OV",
    "Ucec": "UCEC",
}
CPTAC_NAMES = {
    "Brca": "BRCA",
    "Ccrcc": "CCRCC",
    "Hnscc": "HNSCC",
    "Luad": "LUAD",
    "Lscc": "LSCC",
    "Ov": "OV",
    "Ucec": "UCEC",
}
SBS30 = tuple(f"SBS{i}" for i in range(1, 31))
MAF_COLUMNS = (
    "Chromosome",
    "Start_Position",
    "Reference_Allele",
    "Tumor_Seq_Allele2",
    "Variant_Type",
    "Tumor_Sample_Barcode",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source(dataset: str, local: str) -> tuple[Path, str]:
    if dataset == "TCGA":
        code = CANCERS[local]
        path = BASE / f"tcga_sigprof_v34_mc3_rerun/input_maf/{local}/TCGA-{code}_mc3_public.maf"
        return path, "GRCh37"
    return (
        Path("/home/marugoto-1.0.0-classification")
        / f"{CPTAC_NAMES[local]}_somatic_mutation.maf",
        "GRCh38",
    )


def normalize_sample(values: pd.Series, dataset: str) -> pd.Series:
    out = values.astype(str).str.strip()
    if dataset == "TCGA":
        out = out.str.replace(r"-[0-9A-Z]{2}$", "", regex=True)
    return out


def patient_from_sample(sample: str, dataset: str) -> str:
    if dataset == "TCGA":
        return sample[:12]
    return re.sub(r"_(T|N)$", "", sample)


def make_simple_input(dataset: str, local: str) -> dict[str, object]:
    maf, genome_build = source(dataset, local)
    output = ROOT / "matrix_generator" / dataset / "source_simple" / f"{local}.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(
        maf,
        sep="\t",
        comment="#",
        dtype=str,
        usecols=list(MAF_COLUMNS),
        low_memory=False,
    )
    input_rows = len(frame)
    frame = frame.rename(
        columns={
            "Tumor_Sample_Barcode": "sample",
            "Chromosome": "chromosome",
            "Start_Position": "start",
            "Reference_Allele": "ref",
            "Tumor_Seq_Allele2": "alt",
            "Variant_Type": "variant_type",
        }
    )
    frame["sample"] = normalize_sample(frame["sample"], dataset)
    frame["chromosome"] = (
        frame["chromosome"].astype(str).str.strip().str.replace(r"^chr", "", regex=True, case=False)
    )
    frame.loc[frame.chromosome.str.upper().isin(["M", "MT"]), "chromosome"] = "MT"
    frame["start"] = pd.to_numeric(frame["start"], errors="coerce")
    frame["ref"] = frame["ref"].astype(str).str.upper()
    frame["alt"] = frame["alt"].astype(str).str.upper()
    frame["variant_type"] = frame["variant_type"].astype(str).str.upper()
    valid_chromosomes = {str(i) for i in range(1, 23)} | {"X", "Y", "MT"}
    keep = (
        frame.variant_type.eq("SNP")
        & frame.start.notna()
        & frame.chromosome.isin(valid_chromosomes)
        & frame.ref.str.fullmatch("[ACGT]", na=False)
        & frame.alt.str.fullmatch("[ACGT]", na=False)
        & frame.ref.ne(frame.alt)
        & frame["sample"].ne("")
    )
    frame = frame.loc[keep, ["sample", "chromosome", "start", "ref", "alt"]].copy()
    usable_before_dedup = len(frame)
    frame["start"] = frame["start"].astype(np.int64)
    frame = frame.drop_duplicates(["sample", "chromosome", "start", "ref", "alt"])
    frame = frame.sort_values(["chromosome", "start", "sample", "ref", "alt"])
    frame.insert(0, "project", f"{dataset}_{local}")
    frame.insert(2, "id", ".")
    frame.insert(3, "genome", genome_build)
    frame.insert(4, "mutation_type", "SNP")
    frame.insert(7, "end", frame["start"])
    frame.columns = [
        "project", "sample", "id", "genome", "mutation_type",
        "chromosome", "start", "end", "ref", "alt",
    ]
    frame.to_csv(output, sep="\t", index=False)
    sample_map = ROOT / "matrix_generator" / dataset / "sample_maps" / f"{local}.tsv"
    sample_map.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Sample": sorted(frame["sample"].unique())}).assign(
        PATIENT=lambda x: x.Sample.map(lambda value: patient_from_sample(value, dataset)),
        local_cancer=local,
        cancer=CANCERS[local],
    ).to_csv(sample_map, sep="\t", index=False)
    return {
        "dataset": dataset,
        "local_cancer": local,
        "cancer": CANCERS[local],
        "genome_build": genome_build,
        "maf": str(maf),
        "maf_sha256": sha256(maf),
        "input_rows": input_rows,
        "usable_snp_rows_before_dedup": usable_before_dedup,
        "exact_duplicate_rows_removed": usable_before_dedup - len(frame),
        "simple_rows": len(frame),
        "simple_samples": frame["sample"].nunique(),
        "simple_path": str(output),
        "simple_sha256": sha256(output),
    }


def run_matrix_generator(dataset: str) -> None:
    for vendor in reversed(VENDORS):
        sys.path.insert(0, str(vendor))
    from SigProfilerMatrixGenerator.scripts import SigProfilerMatrixGeneratorFunc as matgen

    genome_build = "GRCh37" if dataset == "TCGA" else "GRCh38"
    source_dir = ROOT / "matrix_generator" / dataset / "source_simple"
    output_dir = ROOT / "matrix_generator" / dataset / "run"
    combined_path = output_dir / f"{dataset}.SBS96.exome.tsv"
    if combined_path.exists():
        return
    matrices = matgen.SigProfilerMatrixGeneratorFunc(
        project=f"{dataset}_SBS96_WES",
        reference_genome=genome_build,
        path_to_input_files=str(source_dir),
        exome=True,
        bed_file=None,
        chrom_based=False,
        plot=False,
        tsb_stat=False,
        seqInfo=False,
        cushion=100,
        volume=None,
        output_directory=str(output_dir),
    )
    matrix = matrices["96"]
    if not isinstance(matrix, pd.DataFrame) or matrix.shape[0] != 96:
        raise RuntimeError(f"Unexpected SBS96 output for {dataset}: {type(matrix)}")
    matrix.index.name = "MutationType"
    matrix.to_csv(combined_path, sep="\t")


def split_catalogues(dataset: str) -> list[dict[str, object]]:
    combined_path = ROOT / "matrix_generator" / dataset / "run" / f"{dataset}.SBS96.exome.tsv"
    matrix = pd.read_csv(combined_path, sep="\t", index_col=0)
    if matrix.shape[0] != 96 or matrix.index.nunique() != 96:
        raise ValueError(f"Invalid SBS96 matrix: {combined_path}")
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for local in CANCERS:
        sample_map_path = ROOT / "matrix_generator" / dataset / "sample_maps" / f"{local}.tsv"
        sample_map = pd.read_csv(sample_map_path, sep="\t", dtype=str)
        samples = sample_map.Sample.tolist()
        overlap = seen.intersection(samples)
        if overlap:
            raise ValueError(f"Cross-cancer sample collision in {dataset}: {sorted(overlap)[:5]}")
        seen.update(samples)
        missing = sorted(set(samples) - set(matrix.columns.astype(str)))
        if missing:
            raise ValueError(f"MatrixGenerator omitted samples in {dataset}/{local}: {missing[:5]}")
        cohort = matrix.loc[:, samples].copy()
        counts = cohort.sum(axis=0).astype(int)
        selected_names = counts[counts > 50].index.tolist()
        selected = sample_map.copy()
        selected["snv_count"] = selected.Sample.map(counts).astype(int)
        selected["selected"] = selected.Sample.isin(selected_names)
        out_root = ROOT / "catalogues" / dataset / local
        out_root.mkdir(parents=True, exist_ok=True)
        matrix_path = out_root / "SBS96.tsv"
        selected_path = out_root / "selected_samples.csv"
        cohort.loc[:, selected_names].to_csv(matrix_path, sep="\t")
        selected.to_csv(selected_path, index=False)
        rows.append({
            "dataset": dataset,
            "local_cancer": local,
            "cancer": CANCERS[local],
            "genome_build": "GRCh37" if dataset == "TCGA" else "GRCh38",
            "input_samples": len(samples),
            "selected_samples": len(selected_names),
            "selected_patients": selected.loc[selected.selected, "PATIENT"].nunique(),
            "minimum_selected_sbs96_count": int(counts.loc[selected_names].min()),
            "maximum_selected_sbs96_count": int(counts.loc[selected_names].max()),
            "matrix_sha256": sha256(matrix_path),
            "selected_sha256": sha256(selected_path),
        })
    unexpected = sorted(set(matrix.columns.astype(str)) - seen)
    if unexpected:
        raise ValueError(f"Unexpected MatrixGenerator samples in {dataset}: {unexpected[:5]}")
    return rows


def build_catalogues() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    input_rows = [make_simple_input(dataset, local) for dataset in ("TCGA", "CPTAC") for local in CANCERS]
    pd.DataFrame(input_rows).to_csv(ROOT / "MAF_INPUT_AUDIT.tsv", sep="\t", index=False)
    # Run sequentially to avoid loading both multi-gigabyte chromosome references concurrently.
    for dataset in ("TCGA", "CPTAC"):
        run_matrix_generator(dataset)
    catalogue_rows = split_catalogues("TCGA") + split_catalogues("CPTAC")
    pd.DataFrame(catalogue_rows).to_csv(ROOT / "CATALOGUE_AUDIT.tsv", sep="\t", index=False)


def reference(genome_build: str) -> Path:
    return REFERENCE_ROOT / genome_build / f"COSMIC_v2_SBS_{genome_build}_exome.txt"


def rename_v2_signature(name: str) -> str:
    match = re.fullmatch(r"Signature_(\d+)", name)
    return f"SBS{match.group(1)}" if match else name


def fit_one(dataset: str, local: str) -> None:
    genome_build = "GRCh37" if dataset == "TCGA" else "GRCh38"
    fit_root = ROOT / "fits" / dataset / local
    marker = fit_root / "TARGET_AUDIT.json"
    if marker.exists():
        return
    assignment = fit_root / "sigprofilerassignment"
    activities_path = assignment / "Assignment_Solution/Activities/Assignment_Solution_Activities.txt"
    stats_path = assignment / "Assignment_Solution/Solution_Stats/Assignment_Solution_Samples_Stats.txt"
    matrix_path = ROOT / "catalogues" / dataset / local / "SBS96.tsv"
    selected_path = ROOT / "catalogues" / dataset / local / "selected_samples.csv"
    fit_root.mkdir(parents=True, exist_ok=True)
    if not (activities_path.exists() and stats_path.exists()):
        if assignment.exists():
            raise FileExistsError(f"Partial Assignment output: {assignment}")
        for vendor in reversed(VENDORS):
            sys.path.insert(0, str(vendor))
        from SigProfilerAssignment import Analyzer

        Analyzer.cosmic_fit(
            samples=str(matrix_path),
            output=str(assignment),
            signatures=None,
            signature_database=None,
            nnls_add_penalty=0.05,
            nnls_remove_penalty=0.01,
            initial_remove_penalty=0.05,
            genome_build=genome_build,
            cosmic_version=2,
            make_plots=False,
            collapse_to_SBS96=True,
            connected_sigs=True,
            verbose=False,
            devopts=None,
            exclude_signature_subgroups=None,
            exome=True,
            input_type="matrix",
            context_type="96",
            export_probabilities=True,
            export_probabilities_per_mutation=False,
            sample_reconstruction_plots=False,
            volume=None,
            cpu=2,
            add_background_signatures=True,
        )
    matrix = pd.read_csv(matrix_path, sep="\t", index_col=0)
    selected = pd.read_csv(selected_path, dtype={"Sample": str, "PATIENT": str})
    selected = selected[selected.selected.astype(str).str.upper().isin(["TRUE", "T", "1"])].copy()
    selected = selected.set_index("Sample").reindex(matrix.columns.astype(str)).rename_axis("Sample").reset_index()
    if selected.PATIENT.isna().any():
        raise ValueError(f"Selected sample mismatch: {dataset}/{local}")
    activities = pd.read_csv(activities_path, sep="\t").rename(columns={"Samples": "Sample"})
    activities = activities.rename(columns={column: rename_v2_signature(column) for column in activities.columns})
    activities.Sample = activities.Sample.astype(str)
    for signature in SBS30:
        if signature not in activities:
            activities[signature] = 0.0
    sample = selected[["Sample", "PATIENT", "snv_count"]].merge(
        activities[["Sample", *SBS30]], on="Sample", validate="one_to_one"
    )
    if not np.array_equal(matrix.sum(axis=0).astype(int).to_numpy(), sample.snv_count.astype(int)):
        raise ValueError(f"Mutation count mismatch: {dataset}/{local}")
    proportions = sample[["Sample", "PATIENT", "snv_count"]].copy()
    proportions[list(SBS30)] = sample[list(SBS30)].div(sample.snv_count, axis=0)
    patient = proportions.groupby("PATIENT", sort=True)[list(SBS30)].mean().reset_index()
    label_dir = ROOT / "labels" / dataset / local
    label_dir.mkdir(parents=True, exist_ok=True)
    label_path = label_dir / f"{dataset}_{local}_process_endpoints.xlsx"
    if str(EXCEL_SITE) not in sys.path:
        sys.path.append(str(EXCEL_SITE))
    patient.to_excel(label_path, index=False)
    patient.to_csv(label_dir / f"{dataset}_{local}_SBS1_30.tsv", sep="\t", index=False)
    proportions.to_csv(fit_root / "sample_SBS1_30_proportions.tsv", sep="\t", index=False)
    sample.to_csv(fit_root / "sample_SBS1_30_activities.tsv", sep="\t", index=False)
    stats = pd.read_csv(stats_path, sep="\t").rename(columns={"Sample Names": "Sample"})
    stats.Sample = stats.Sample.astype(str)
    stats.to_csv(fit_root / "sample_reconstruction_qc.tsv", sep="\t", index=False)
    audit = {
        "status": "complete",
        "dataset": dataset,
        "local_cancer": local,
        "cancer": CANCERS[local],
        "genome_build": genome_build,
        "assay": "WES",
        "catalogue_generator": "SigProfilerMatrixGenerator 1.3.6",
        "assignment_version": "1.1.5",
        "cosmic_version": "2",
        "fit_basis": list(SBS30),
        "samples": len(sample),
        "patients": len(patient),
        "signatures": len(SBS30),
        "minimum_mutations": int(sample.snv_count.min()),
        "mean_reconstruction_cosine": float(stats["Cosine Similarity"].mean()),
        "matrix_sha256": sha256(matrix_path),
        "reference_sha256": sha256(reference(genome_build)),
        "label_sha256": sha256(label_path),
        "audit_pass": bool(len(SBS30) == 30 and sample.snv_count.gt(50).all()),
    }
    marker.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit["audit_pass"]:
        raise RuntimeError(f"Target audit failed: {dataset}/{local}")


def run_fit_subprocess(job: tuple[str, str]) -> tuple[str, str]:
    dataset, local = job
    log = ROOT / "logs" / f"fit_{dataset}_{local}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(str(vendor) for vendor in VENDORS)
    command = [str(SPA_PYTHON), str(Path(__file__).resolve()), "fit-one", dataset, local]
    with log.open("w") as handle:
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=env)
    if completed.returncode:
        raise RuntimeError(f"Assignment failed: {dataset}/{local}; see {log}")
    return job


def freeze_eligibility() -> None:
    if str(EXCEL_SITE) not in sys.path:
        sys.path.append(str(EXCEL_SITE))
    rows = []
    for local, cancer in CANCERS.items():
        labels_path = ROOT / "labels" / "TCGA" / local / f"TCGA_{local}_process_endpoints.xlsx"
        labels = pd.read_excel(labels_path, dtype={"PATIENT": str})
        archive = np.load(AGG / f"TCGA_{cancer}_old_new_qc_mean_std.npz")
        feature_patients = pd.DataFrame({"PATIENT": archive["patients"].astype(str)})
        matched = feature_patients.merge(labels, on="PATIENT", validate="one_to_one")
        for signature in SBS30:
            target = matched[signature].to_numpy(float)
            nonzero = int(np.count_nonzero(target))
            unique = int(pd.Series(target).nunique())
            eligible = bool(
                len(target) >= 100
                and nonzero >= 30
                and nonzero / len(target) >= 0.10
                and unique >= 10
            )
            rows.append({
                "cancer": cancer,
                "local_cancer": local,
                "signature": signature,
                "matched_tcga_n": len(target),
                "nonzero_tcga_n": nonzero,
                "nonzero_fraction": nonzero / len(target),
                "unique_values": unique,
                "eligible": eligible,
                "exclusion_reason": "" if eligible else "TCGA_estimability",
            })
    frame = pd.DataFrame(rows)
    eligibility_path = ROOT / "TCGA_SIGNATURE_ELIGIBILITY.tsv"
    eligible_path = ROOT / "ELIGIBLE_PAIRS.tsv"
    frame.to_csv(eligibility_path, sep="\t", index=False)
    eligible = frame[frame.eligible].copy()
    eligible.to_csv(eligible_path, sep="\t", index=False)
    union = sorted(eligible.signature.unique(), key=lambda value: int(value[3:]))
    config = {
        "status": "frozen_using_tcga_targets_only",
        "target_method": (
            "SigProfilerMatrixGenerator 1.3.6 exome SBS96 followed by "
            "SigProfilerAssignment 1.1.5 COSMIC v2 WES exact SBS1-SBS30 basis"
        ),
        "fit_basis": list(SBS30),
        "eligibility": {
            "matched_tcga_n_min": 100,
            "nonzero_tcga_n_min": 30,
            "nonzero_fraction_min": 0.10,
            "unique_values_min": 10,
        },
        "eligible_pair_count": len(eligible),
        "union_signature_count": len(union),
        "union_signatures": union,
        "pairs": [f"{row.cancer}_{row.signature}" for row in eligible.itertuples(index=False)],
        "cptac_opened_for_eligibility": False,
        "eligibility_sha256": sha256(eligibility_path),
        "eligible_pairs_sha256": sha256(eligible_path),
    }
    (ROOT / "FROZEN_TARGET_CONFIG.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def finalize() -> None:
    audits = [json.loads(path.read_text()) for path in sorted((ROOT / "fits").glob("*/*/TARGET_AUDIT.json"))]
    if len(audits) != 14 or not all(audit["audit_pass"] for audit in audits):
        raise RuntimeError("Expected 14 passing target audits")
    freeze_eligibility()
    summary = pd.DataFrame(audits)
    summary.to_csv(ROOT / "TARGET_COHORT_SUMMARY.tsv", sep="\t", index=False)
    config = json.loads((ROOT / "FROZEN_TARGET_CONFIG.json").read_text())
    audit = {
        "status": "complete",
        "cohorts": 14,
        "fit_basis_count": 30,
        "minimum_mean_reconstruction_cosine": float(summary.mean_reconstruction_cosine.min()),
        "maximum_mean_reconstruction_cosine": float(summary.mean_reconstruction_cosine.max()),
        "eligible_pair_count": config["eligible_pair_count"],
        "union_signature_count": config["union_signature_count"],
        "cptac_used_for_target_eligibility": False,
        "audit_pass": True,
    }
    (ROOT / "TARGET_AUDIT.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


def jobs() -> list[tuple[str, str]]:
    return [(dataset, local) for dataset in ("TCGA", "CPTAC") for local in CANCERS]


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("catalogues")
    run = sub.add_parser("all")
    run.add_argument("--fit-jobs", type=int, default=7)
    fit = sub.add_parser("fit-one")
    fit.add_argument("dataset", choices=("TCGA", "CPTAC"))
    fit.add_argument("local", choices=tuple(CANCERS))
    args = parser.parse_args()
    if args.command == "fit-one":
        fit_one(args.dataset, args.local)
        return
    build_catalogues()
    if args.command == "catalogues":
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.fit_jobs) as pool:
        list(pool.map(run_fit_subprocess, jobs()))
    finalize()


if __name__ == "__main__":
    main()
