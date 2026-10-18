#!/opt/anaconda/envs/cenv/bin/python
"""Build current-MAF COSMIC v3.5 WES targets with SigProfilerAssignment 1.1.4."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path("/home/marugoto-1.0.0-regression")
R_SCRIPT = BASE / "cptac_sigprof/extract_cptac_sbs.R"
RSCRIPT = Path("/opt/anaconda/envs/cptac-sigprof-r/bin/Rscript")
SPA_PYTHON = BASE / "harmonized_sbs3_grch38_20260726/.venv/bin/python"
BASE_VENDOR = BASE / "harmonized_sbs3_grch38_20260726/python_packages"
SPA_PROFILE = os.environ.get("SPA_PROFILE", "v114_v35")
if SPA_PROFILE == "v115_v36":
    ROOT = BASE / "sigprofilerassignment_v115_v36_currentmaf_v2_27_20260903"
    SPA_VENDOR = BASE / "sigprofilerassignment_v1.1.5_vendor"
    COSMIC_VERSION = 3.6
    SPA_VERSION = "1.1.5"
    EXPECTED_SIGNATURES = 101
elif SPA_PROFILE == "v114_v35":
    ROOT = BASE / "sigprofilerassignment_v35_currentmaf_v2_27_20260903"
    SPA_VENDOR = BASE_VENDOR
    COSMIC_VERSION = 3.5
    SPA_VERSION = "1.1.4"
    EXPECTED_SIGNATURES = 97
else:
    raise ValueError(f"Unknown SPA_PROFILE: {SPA_PROFILE}")
VENDORS = tuple(dict.fromkeys((SPA_VENDOR, BASE_VENDOR)))
EXCEL_SITE = Path("/opt/anaconda/envs/cenv/lib/python3.8/site-packages")
REFERENCE_ROOT = SPA_VENDOR / "SigProfilerAssignment/data/Reference_Signatures"
AGG = BASE / "sigprofiler_improved_20260818/inputs/end2end_retccl_qc_v1/patient_aggregates"
CANCERS = {
    "Brca": "BRCA", "Ccrcc": "KIRC", "Hnscc": "HNSC", "Luad": "LUAD",
    "Lscc": "LUSC", "Ov": "OV", "Ucec": "UCEC",
}
CPTAC_MAFS = {
    "Brca": Path("/home/marugoto-1.0.0-classification/BRCA_somatic_mutation.maf"),
    "Ccrcc": Path("/home/marugoto-1.0.0-classification/CCRCC_somatic_mutation.maf"),
    "Hnscc": Path("/home/marugoto-1.0.0-classification/HNSCC_somatic_mutation.maf"),
    "Luad": Path("/home/marugoto-1.0.0-classification/LUAD_somatic_mutation.maf"),
    "Lscc": Path("/home/marugoto-1.0.0-classification/LSCC_somatic_mutation.maf"),
    "Ov": Path("/home/marugoto-1.0.0-classification/OV_somatic_mutation.maf"),
    "Ucec": Path("/home/marugoto-1.0.0-classification/UCEC_somatic_mutation.maf"),
}
ARTIFACT_SIGNATURES = {
    "SBS27", "SBS43", "SBS45", "SBS46", "SBS47", "SBS48", "SBS49",
    "SBS50", "SBS51", "SBS52", "SBS53", "SBS54", "SBS55", "SBS56",
    "SBS57", "SBS58", "SBS59", "SBS60", "SBS95",
}
PROCESSES = {
    "CLOCK": ("SBS1", "SBS5"),
    "APOBEC": ("SBS2", "SBS13"),
    "HRD": ("SBS3",),
    "TOBACCO": ("SBS4", "SBS29", "SBS92", "SBS100", "SBS109"),
    "MMR": ("SBS6", "SBS14", "SBS15", "SBS20", "SBS21", "SBS26", "SBS44"),
    "POL": ("SBS10a", "SBS10b", "SBS10c", "SBS10d", "SBS28"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source(dataset: str, local: str, code: str) -> tuple[Path, str, str, str]:
    if dataset == "TCGA":
        return (
            BASE / f"tcga_sigprof_v34_mc3_rerun/input_maf/{local}/TCGA-{code}_mc3_public.maf",
            "tcga", "BSgenome.Hsapiens.UCSC.hg19", "GRCh37",
        )
    return CPTAC_MAFS[local], "cptac", "BSgenome.Hsapiens.UCSC.hg38", "GRCh38"


def paths(dataset: str, local: str) -> dict[str, Path]:
    root = ROOT / "catalogues" / dataset / local
    return {
        "root": root,
        "matrix": root / "SBS96.tsv",
        "selected": root / "selected_samples.csv",
        "log": root / "catalogue.log",
    }


def build_catalogue(job: tuple[str, str, str]) -> dict[str, object]:
    dataset, local, code = job
    maf, patient_mode, genome_pkg, genome_build = source(dataset, local, code)
    output = paths(dataset, local)
    output["root"].mkdir(parents=True, exist_ok=True)
    if not output["matrix"].exists():
        if output["selected"].exists():
            raise FileExistsError(f"Partial catalogue: {dataset}/{local}")
        command = [
            str(RSCRIPT), str(R_SCRIPT), "--input-maf", str(maf),
            "--output-sbs96", str(output["matrix"]),
            "--output-selected-samples", str(output["selected"]),
            "--catalogue-only", "--min-mutations", "50",
            "--patient-mode", patient_mode, "--ref-genome-pkg", genome_pkg,
        ]
        with output["log"].open("w") as handle:
            completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f"Catalogue failed: {dataset}/{local}")
    matrix = pd.read_csv(output["matrix"], sep="\t")
    selected = pd.read_csv(output["selected"], dtype={"Sample": str, "PATIENT": str})
    selected = selected[selected.selected.astype(str).str.upper().isin(["TRUE", "T", "1"])].copy()
    matrix_samples = list(matrix.columns[1:].astype(str))
    if (len(matrix) != 96 or len(matrix_samples) != len(selected)
            or set(matrix_samples) != set(selected.Sample)):
        raise ValueError(f"Catalogue/sample mismatch: {dataset}/{local}")
    return {
        "dataset": dataset, "local_cancer": local, "cancer": code,
        "genome_build": genome_build, "samples": len(selected),
        "patients": selected.PATIENT.nunique(), "matrix_sha256": sha256(output["matrix"]),
        "selected_sha256": sha256(output["selected"]), "maf_sha256": sha256(maf),
    }


def reference(genome_build: str) -> Path:
    return (
        REFERENCE_ROOT / genome_build
        / f"COSMIC_v{COSMIC_VERSION}_SBS_{genome_build}_exome.txt"
    )


def fit_one(dataset: str, local: str) -> None:
    code = CANCERS[local]
    _, _, _, genome_build = source(dataset, local, code)
    catalogue = paths(dataset, local)
    fit_root = ROOT / "fits" / dataset / local
    target_marker = fit_root / "TARGET_AUDIT.json"
    if target_marker.exists():
        return
    assignment = fit_root / "sigprofilerassignment"
    activities_path = assignment / "Assignment_Solution/Activities/Assignment_Solution_Activities.txt"
    stats_path = assignment / "Assignment_Solution/Solution_Stats/Assignment_Solution_Samples_Stats.txt"
    fit_root.mkdir(parents=True, exist_ok=True)
    if not (activities_path.exists() and stats_path.exists()):
        if assignment.exists():
            raise FileExistsError(f"Partial SigProfilerAssignment output: {assignment}")
        for vendor in reversed(VENDORS):
            sys.path.insert(0, str(vendor))
        from SigProfilerAssignment import Analyzer

        Analyzer.cosmic_fit(
            samples=str(catalogue["matrix"]), output=str(assignment),
            signatures=None, signature_database=None,
            nnls_add_penalty=0.05, nnls_remove_penalty=0.01,
            initial_remove_penalty=0.05, genome_build=genome_build,
            cosmic_version=COSMIC_VERSION, make_plots=False, collapse_to_SBS96=True,
            connected_sigs=True, verbose=False, devopts=None,
            exclude_signature_subgroups=None, exome=True, input_type="matrix",
            context_type="96", export_probabilities=True,
            export_probabilities_per_mutation=False, sample_reconstruction_plots=False,
            volume=None, cpu=2, add_background_signatures=True,
        )
        if not (activities_path.exists() and stats_path.exists()):
            raise RuntimeError(f"SigProfilerAssignment outputs missing: {dataset}/{local}")

    matrix = pd.read_csv(catalogue["matrix"], sep="\t", index_col=0)
    selected = pd.read_csv(catalogue["selected"], dtype={"Sample": str, "PATIENT": str})
    selected = selected[selected.selected.astype(str).str.upper().isin(["TRUE", "T", "1"])].copy()
    selected = (
        selected.set_index("Sample")
        .reindex(matrix.columns.astype(str))
        .rename_axis("Sample")
        .reset_index()
    )
    if selected.PATIENT.isna().any():
        raise ValueError(f"Selected-sample reindex failed: {dataset}/{local}")
    activities = pd.read_csv(activities_path, sep="\t").rename(columns={"Samples": "Sample"})
    activities.Sample = activities.Sample.astype(str)
    signatures = [c for c in pd.read_csv(reference(genome_build), sep="\t", nrows=1) if c != "Type"]
    for signature in signatures:
        if signature not in activities:
            activities[signature] = 0.0
    sample = selected[["Sample", "PATIENT", "snv_count"]].merge(
        activities[["Sample", *signatures]], on="Sample", validate="one_to_one"
    )
    if not np.array_equal(matrix.sum(axis=0).astype(int).to_numpy(), sample.snv_count.astype(int)):
        raise ValueError(f"Mutation-count mismatch: {dataset}/{local}")
    proportions = sample[["Sample", "PATIENT", "snv_count"]].copy()
    proportions[signatures] = sample[signatures].div(sample.snv_count, axis=0)
    patient_sbs = proportions.groupby("PATIENT", sort=True)[signatures].mean().reset_index()
    patient = patient_sbs.copy()
    for process, members in PROCESSES.items():
        patient[f"PROC_{process}_proportion"] = patient_sbs[list(members)].sum(axis=1)
        sample_process_activity = sample[list(members)].sum(axis=1)
        process_activity = pd.DataFrame({
            "PATIENT": sample.PATIENT, "activity": sample_process_activity,
            "presence": sample_process_activity.gt(0).astype(float),
        }).groupby("PATIENT", sort=True).agg({"activity": "mean", "presence": "max"})
        patient[f"PROC_{process}_activity"] = process_activity.reindex(patient.PATIENT).activity.to_numpy()
        patient[f"PROC_{process}_presence"] = process_activity.reindex(patient.PATIENT).presence.to_numpy()

    label_dir = ROOT / "labels" / dataset / local
    label_dir.mkdir(parents=True, exist_ok=True)
    label_path = label_dir / f"{dataset}_{local}_process_endpoints.xlsx"
    if str(EXCEL_SITE) not in sys.path:
        sys.path.append(str(EXCEL_SITE))
    patient.to_excel(label_path, index=False)
    proportions.to_csv(fit_root / "sample_all_sbs_proportions.tsv", sep="\t", index=False)
    sample.to_csv(fit_root / "sample_all_sbs_activities.tsv", sep="\t", index=False)
    patient_sbs.to_csv(fit_root / "patient_all_sbs_proportions.tsv", sep="\t", index=False)
    stats = pd.read_csv(stats_path, sep="\t").rename(columns={"Sample Names": "Sample"})
    stats.Sample = stats.Sample.astype(str)
    stats.to_csv(fit_root / "sample_reconstruction_qc.tsv", sep="\t", index=False)
    audit = {
        "status": "complete", "dataset": dataset, "local_cancer": local,
        "cancer": code, "genome_build": genome_build, "assay": "WES",
        "cosmic_version": str(COSMIC_VERSION), "sigprofilerassignment_version": SPA_VERSION,
        "samples": len(sample), "patients": len(patient), "signatures": len(signatures),
        "minimum_mutations": int(sample.snv_count.min()),
        "mean_reconstruction_cosine": float(stats["Cosine Similarity"].mean()),
        "matrix_sha256": sha256(catalogue["matrix"]),
        "reference_sha256": sha256(reference(genome_build)),
        "label_sha256": sha256(label_path),
        "audit_pass": bool(
            len(signatures) == EXPECTED_SIGNATURES and sample.snv_count.gt(50).all()
        ),
    }
    target_marker.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit["audit_pass"]:
        raise RuntimeError(f"Target audit failed: {dataset}/{local}")


def run_fit_subprocess(job: tuple[str, str, str]) -> tuple[str, str]:
    dataset, local, _ = job
    log = ROOT / "logs" / f"fit_{dataset}_{local}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(str(vendor) for vendor in VENDORS)
    command = [str(SPA_PYTHON), str(Path(__file__).resolve()), "fit-one", dataset, local]
    with log.open("w") as handle:
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=env)
    if completed.returncode:
        raise RuntimeError(f"SigProfilerAssignment failed: {dataset}/{local}; {log}")
    return dataset, local


def freeze_eligibility() -> None:
    rows = []
    all_signatures = [
        c for c in pd.read_csv(reference("GRCh37"), sep="\t", nrows=1) if c != "Type"
    ]
    candidate_signatures = [s for s in all_signatures if s not in ARTIFACT_SIGNATURES]
    for local, cancer in CANCERS.items():
        labels_path = ROOT / "labels/TCGA" / local / f"TCGA_{local}_process_endpoints.xlsx"
        labels = pd.read_excel(labels_path, dtype={"PATIENT": str})
        archive = np.load(AGG / f"TCGA_{cancer}_old_new_qc_mean_std.npz")
        feature_patients = pd.DataFrame({"PATIENT": archive["patients"].astype(str)})
        matched = feature_patients.merge(labels, on="PATIENT", validate="one_to_one")
        for signature in candidate_signatures:
            target = matched[signature].to_numpy(float)
            nonzero = int(np.count_nonzero(target))
            unique = int(pd.Series(target).nunique())
            eligible = bool(
                len(target) >= 100 and nonzero >= 30 and nonzero / len(target) >= .10
                and unique >= 10
            )
            rows.append({
                "cancer": cancer, "local_cancer": local, "signature": signature,
                "matched_tcga_n": len(target), "nonzero_tcga_n": nonzero,
                "nonzero_fraction": nonzero / len(target), "unique_values": unique,
                "eligible": eligible, "exclusion_reason": "" if eligible else "TCGA_estimability",
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / "TCGA_SIGNATURE_ELIGIBILITY.tsv", sep="\t", index=False)
    eligible = frame[frame.eligible].copy()
    eligible.to_csv(ROOT / "ELIGIBLE_PAIRS.tsv", sep="\t", index=False)
    union = sorted(eligible.signature.unique(), key=lambda x: (int(''.join(filter(str.isdigit, x))), x))
    config = {
        "status": "frozen_using_tcga_targets_only",
        "target_method": (
            f"SigProfilerAssignment {SPA_VERSION} COSMIC v{COSMIC_VERSION} "
            f"WES full {EXPECTED_SIGNATURES} basis"
        ),
        "artifact_signatures_fit_as_nuisance_but_excluded_as_targets": sorted(ARTIFACT_SIGNATURES),
        "eligibility": {
            "matched_tcga_n_min": 100, "nonzero_tcga_n_min": 30,
            "nonzero_fraction_min": .10, "unique_values_min": 10,
        },
        "eligible_pair_count": len(eligible), "union_signature_count": len(union),
        "union_signatures": union,
        "pairs": [f"{row.cancer}_{row.signature}" for row in eligible.itertuples(index=False)],
        "cptac_opened_for_eligibility": False,
        "eligibility_sha256": sha256(ROOT / "TCGA_SIGNATURE_ELIGIBILITY.tsv"),
        "eligible_pairs_sha256": sha256(ROOT / "ELIGIBLE_PAIRS.tsv"),
    }
    (ROOT / "FROZEN_TARGET_CONFIG.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )


def freeze_process_eligibility() -> None:
    rows = []
    process_targets = tuple(f"PROC_{process}_proportion" for process in PROCESSES)
    for local, cancer in CANCERS.items():
        labels_path = ROOT / "labels/TCGA" / local / f"TCGA_{local}_process_endpoints.xlsx"
        labels = pd.read_excel(labels_path, dtype={"PATIENT": str})
        archive = np.load(AGG / f"TCGA_{cancer}_old_new_qc_mean_std.npz")
        feature_patients = pd.DataFrame({"PATIENT": archive["patients"].astype(str)})
        matched = feature_patients.merge(labels, on="PATIENT", validate="one_to_one")
        for target_name in process_targets:
            target = matched[target_name].to_numpy(float)
            nonzero = int(np.count_nonzero(target))
            unique = int(pd.Series(target).nunique())
            eligible = bool(
                len(target) >= 100 and nonzero >= 30 and nonzero / len(target) >= .10
                and unique >= 10
            )
            rows.append({
                "cancer": cancer, "local_cancer": local, "target": target_name,
                "matched_tcga_n": len(target), "nonzero_tcga_n": nonzero,
                "nonzero_fraction": nonzero / len(target), "unique_values": unique,
                "eligible": eligible, "exclusion_reason": "" if eligible else "TCGA_estimability",
            })
    frame = pd.DataFrame(rows)
    eligibility_path = ROOT / "TCGA_PROCESS_ELIGIBILITY.tsv"
    eligible_path = ROOT / "ELIGIBLE_PROCESS_PAIRS.tsv"
    frame.to_csv(eligibility_path, sep="\t", index=False)
    eligible = frame[frame.eligible].copy()
    eligible.to_csv(eligible_path, sep="\t", index=False)
    config = {
        "status": "frozen_using_tcga_targets_only",
        "target_method": (
            "prespecified biological process sums from the full "
            f"SigProfilerAssignment {SPA_VERSION} COSMIC v{COSMIC_VERSION} fit"
        ),
        "process_members": {key: list(value) for key, value in PROCESSES.items()},
        "eligibility": {
            "matched_tcga_n_min": 100, "nonzero_tcga_n_min": 30,
            "nonzero_fraction_min": .10, "unique_values_min": 10,
        },
        "eligible_pair_count": len(eligible),
        "targets": list(process_targets),
        "pairs": [f"{row.cancer}_{row.target}" for row in eligible.itertuples(index=False)],
        "cptac_opened_for_eligibility": False,
        "eligibility_sha256": sha256(eligibility_path),
        "eligible_pairs_sha256": sha256(eligible_path),
    }
    (ROOT / "FROZEN_PROCESS_CONFIG.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )


def finalize() -> None:
    audits = [
        json.loads(path.read_text())
        for path in sorted((ROOT / "fits").glob("*/*/TARGET_AUDIT.json"))
    ]
    if len(audits) != 14 or not all(a["audit_pass"] for a in audits):
        raise RuntimeError("Expected 14 passing target audits")
    freeze_eligibility()
    freeze_process_eligibility()
    config = json.loads((ROOT / "FROZEN_TARGET_CONFIG.json").read_text())
    process_config = json.loads((ROOT / "FROZEN_PROCESS_CONFIG.json").read_text())
    summary = pd.DataFrame(audits)
    summary.to_csv(ROOT / "TARGET_COHORT_SUMMARY.tsv", sep="\t", index=False)
    audit = {
        "status": "complete", "cohorts": 14,
        "minimum_mean_reconstruction_cosine": float(summary.mean_reconstruction_cosine.min()),
        "maximum_mean_reconstruction_cosine": float(summary.mean_reconstruction_cosine.max()),
        "eligible_pair_count": config["eligible_pair_count"],
        "union_signature_count": config["union_signature_count"],
        "eligible_process_pair_count": process_config["eligible_pair_count"],
        "cptac_used_for_target_eligibility": False,
        "audit_pass": True,
    }
    (ROOT / "TARGET_AUDIT.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


def jobs() -> list[tuple[str, str, str]]:
    return [(dataset, local, code) for dataset in ("TCGA", "CPTAC") for local, code in CANCERS.items()]


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("all")
    run.add_argument("--catalogue-jobs", type=int, default=14)
    run.add_argument("--fit-jobs", type=int, default=7)
    fit = sub.add_parser("fit-one")
    fit.add_argument("dataset", choices=("TCGA", "CPTAC"))
    fit.add_argument("local", choices=tuple(CANCERS))
    args = parser.parse_args()
    if args.command == "fit-one":
        fit_one(args.dataset, args.local)
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.catalogue_jobs) as pool:
        catalogue_rows = list(pool.map(build_catalogue, jobs()))
    pd.DataFrame(catalogue_rows).to_csv(ROOT / "CATALOGUE_AUDIT.tsv", sep="\t", index=False)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.fit_jobs) as pool:
        list(pool.map(run_fit_subprocess, jobs()))
    finalize()


if __name__ == "__main__":
    main()
