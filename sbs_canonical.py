#!/usr/bin/env python3
"""Canonical SBS biology definitions for the TCGA/CPTAC WSI project.

This file is intentionally conservative. It separates mechanistic/cancer
biology targets that can be defended in a manuscript from exploratory local
signals that should not be presented as established cancer-associated SBS.

Evidence base checked on 2026-05-25:
- COSMIC Mutational Signatures v3.6/v104.
- Alexandrov et al., Nature 2020, PCAWG mutational signature repertoire.
- Recent computational pathology studies using TCGA discovery and external
  validation designs for WSI molecular prediction.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SignatureTarget:
    cancer: str
    signature: str
    priority: str
    process: str
    etiology: str
    note: str


# Legacy30 labels in this project use SBS1..SBS30. Where COSMIC v3.6 has
# split signatures such as SBS10a/b/c/d or SBS7a/b/c/d, the legacy alias is
# included so the same definitions work on the local label tables.
PROCESS_DEFS: dict[str, list[str]] = {
    "PROC_clock_like": ["SBS1", "SBS5"],
    "PROC_APOBEC": ["SBS2", "SBS13"],
    "PROC_HRD": ["SBS3"],
    "PROC_tobacco": ["SBS4", "SBS29"],
    "PROC_MMR_MSI": ["SBS6", "SBS15", "SBS20", "SBS21", "SBS26", "SBS44"],
    "PROC_POLE_MMR": ["SBS10", "SBS10a", "SBS10b", "SBS10c", "SBS10d", "SBS14"],
    "PROC_UV": ["SBS7", "SBS7a", "SBS7b", "SBS7c", "SBS7d"],
    "PROC_BER_NTHL1": ["SBS30"],
}


PROCESS_LABELS: dict[str, str] = {
    "PROC_clock_like": "Clock-like / SBS1+5",
    "PROC_APOBEC": "APOBEC / SBS2+13",
    "PROC_HRD": "HRD / SBS3",
    "PROC_tobacco": "Tobacco / SBS4+29",
    "PROC_MMR_MSI": "MMR/MSI / SBS6+15+20+21+26",
    "PROC_POLE_MMR": "POLE/MMR / SBS10+14",
    "PROC_UV": "UV / SBS7",
    "PROC_BER_NTHL1": "BER/NTHL1 / SBS30",
}


SIGNATURE_ETIOLOGY: dict[str, str] = {
    "SBS1": "clock-like; spontaneous 5-methylcytosine deamination",
    "SBS2": "APOBEC cytidine deaminase activity",
    "SBS3": "defective homologous recombination DNA damage repair",
    "SBS4": "tobacco smoking",
    "SBS5": "clock-like; unknown etiology",
    "SBS6": "defective DNA mismatch repair / MSI",
    "SBS7": "ultraviolet light exposure",
    "SBS8": "unknown etiology; exploratory in this project",
    "SBS10": "POLE/POLD1 proofreading deficiency legacy aggregate",
    "SBS13": "APOBEC cytidine deaminase activity",
    "SBS14": "concurrent POLE mutation and defective mismatch repair",
    "SBS15": "defective DNA mismatch repair / MSI",
    "SBS20": "concurrent POLD1 mutation and defective mismatch repair",
    "SBS21": "defective DNA mismatch repair / MSI",
    "SBS26": "defective DNA mismatch repair / MSI",
    "SBS29": "tobacco chewing / tobacco-associated exposure",
    "SBS30": "base-excision repair deficiency / NTHL1-associated; rare",
    "SBS44": "defective DNA mismatch repair / MSI",
}


CANONICAL_TARGETS: tuple[SignatureTarget, ...] = (
    # Breast: HRD/SBS3 is the primary biology. APOBEC is common but secondary.
    SignatureTarget("Brca", "SBS3", "primary", "PROC_HRD", SIGNATURE_ETIOLOGY["SBS3"], "BRCA/HRD biology"),
    SignatureTarget("Brca", "SBS2", "secondary", "PROC_APOBEC", SIGNATURE_ETIOLOGY["SBS2"], "APOBEC activity in breast cancer"),
    SignatureTarget("Brca", "SBS13", "secondary", "PROC_APOBEC", SIGNATURE_ETIOLOGY["SBS13"], "APOBEC activity in breast cancer"),
    SignatureTarget("Brca", "SBS1", "background", "PROC_clock_like", SIGNATURE_ETIOLOGY["SBS1"], "clock-like background"),
    SignatureTarget("Brca", "SBS5", "background", "PROC_clock_like", SIGNATURE_ETIOLOGY["SBS5"], "clock-like background"),
    # Kidney clear-cell: no strong legacy30 cancer-specific SBS should be claimed.
    SignatureTarget("Ccrcc", "SBS1", "background", "PROC_clock_like", SIGNATURE_ETIOLOGY["SBS1"], "clock-like background"),
    SignatureTarget("Ccrcc", "SBS5", "background", "PROC_clock_like", SIGNATURE_ETIOLOGY["SBS5"], "clock-like background"),
    # Head and neck squamous: tobacco and APOBEC are defensible.
    SignatureTarget("Hnscc", "SBS4", "primary", "PROC_tobacco", SIGNATURE_ETIOLOGY["SBS4"], "tobacco exposure"),
    SignatureTarget("Hnscc", "SBS29", "secondary", "PROC_tobacco", SIGNATURE_ETIOLOGY["SBS29"], "tobacco chewing/oral exposure where present"),
    SignatureTarget("Hnscc", "SBS2", "secondary", "PROC_APOBEC", SIGNATURE_ETIOLOGY["SBS2"], "APOBEC epithelial cancer process"),
    SignatureTarget("Hnscc", "SBS13", "secondary", "PROC_APOBEC", SIGNATURE_ETIOLOGY["SBS13"], "APOBEC epithelial cancer process"),
    SignatureTarget("Hnscc", "SBS1", "background", "PROC_clock_like", SIGNATURE_ETIOLOGY["SBS1"], "clock-like background"),
    SignatureTarget("Hnscc", "SBS5", "background", "PROC_clock_like", SIGNATURE_ETIOLOGY["SBS5"], "clock-like background"),
    # Lung squamous: tobacco first, APOBEC second.
    SignatureTarget("Lscc", "SBS4", "primary", "PROC_tobacco", SIGNATURE_ETIOLOGY["SBS4"], "tobacco exposure"),
    SignatureTarget("Lscc", "SBS2", "secondary", "PROC_APOBEC", SIGNATURE_ETIOLOGY["SBS2"], "APOBEC lung cancer context"),
    SignatureTarget("Lscc", "SBS13", "secondary", "PROC_APOBEC", SIGNATURE_ETIOLOGY["SBS13"], "APOBEC lung cancer context"),
    SignatureTarget("Lscc", "SBS1", "background", "PROC_clock_like", SIGNATURE_ETIOLOGY["SBS1"], "clock-like background"),
    SignatureTarget("Lscc", "SBS5", "background", "PROC_clock_like", SIGNATURE_ETIOLOGY["SBS5"], "clock-like background"),
    # Lung adenocarcinoma: tobacco and APOBEC are both relevant.
    SignatureTarget("Luad", "SBS4", "primary", "PROC_tobacco", SIGNATURE_ETIOLOGY["SBS4"], "tobacco exposure"),
    SignatureTarget("Luad", "SBS2", "secondary", "PROC_APOBEC", SIGNATURE_ETIOLOGY["SBS2"], "APOBEC lung cancer context"),
    SignatureTarget("Luad", "SBS13", "secondary", "PROC_APOBEC", SIGNATURE_ETIOLOGY["SBS13"], "APOBEC lung cancer context"),
    SignatureTarget("Luad", "SBS1", "background", "PROC_clock_like", SIGNATURE_ETIOLOGY["SBS1"], "clock-like background"),
    SignatureTarget("Luad", "SBS5", "background", "PROC_clock_like", SIGNATURE_ETIOLOGY["SBS5"], "clock-like background"),
    # Ovarian: HRD/SBS3 is the main claim.
    SignatureTarget("Ov", "SBS3", "primary", "PROC_HRD", SIGNATURE_ETIOLOGY["SBS3"], "HRD ovarian cancer biology"),
    SignatureTarget("Ov", "SBS1", "background", "PROC_clock_like", SIGNATURE_ETIOLOGY["SBS1"], "clock-like background"),
    SignatureTarget("Ov", "SBS5", "background", "PROC_clock_like", SIGNATURE_ETIOLOGY["SBS5"], "clock-like background"),
    # Endometrial: MMR/MSI and POLE are primary; APOBEC is secondary.
    SignatureTarget("Ucec", "SBS6", "primary", "PROC_MMR_MSI", SIGNATURE_ETIOLOGY["SBS6"], "MMR/MSI endometrial cancer biology"),
    SignatureTarget("Ucec", "SBS15", "primary", "PROC_MMR_MSI", SIGNATURE_ETIOLOGY["SBS15"], "MMR/MSI endometrial cancer biology"),
    SignatureTarget("Ucec", "SBS20", "secondary", "PROC_MMR_MSI", SIGNATURE_ETIOLOGY["SBS20"], "MMR/POLD1-associated hypermutation"),
    SignatureTarget("Ucec", "SBS21", "secondary", "PROC_MMR_MSI", SIGNATURE_ETIOLOGY["SBS21"], "MMR/MSI-related"),
    SignatureTarget("Ucec", "SBS26", "secondary", "PROC_MMR_MSI", SIGNATURE_ETIOLOGY["SBS26"], "MMR/MSI-related"),
    SignatureTarget("Ucec", "SBS10", "primary", "PROC_POLE_MMR", SIGNATURE_ETIOLOGY["SBS10"], "POLE/POLD1 proofreading deficiency"),
    SignatureTarget("Ucec", "SBS14", "secondary", "PROC_POLE_MMR", SIGNATURE_ETIOLOGY["SBS14"], "POLE plus MMR-related hypermutation"),
    SignatureTarget("Ucec", "SBS2", "secondary", "PROC_APOBEC", SIGNATURE_ETIOLOGY["SBS2"], "APOBEC"),
    SignatureTarget("Ucec", "SBS13", "secondary", "PROC_APOBEC", SIGNATURE_ETIOLOGY["SBS13"], "APOBEC"),
    SignatureTarget("Ucec", "SBS1", "background", "PROC_clock_like", SIGNATURE_ETIOLOGY["SBS1"], "clock-like background"),
    SignatureTarget("Ucec", "SBS5", "background", "PROC_clock_like", SIGNATURE_ETIOLOGY["SBS5"], "clock-like background"),
)


CANCER_PROCESS_MAP: dict[str, list[str]] = {
    cancer: sorted(
        {target.process for target in CANONICAL_TARGETS if target.cancer == cancer},
        key=lambda proc: list(PROCESS_DEFS).index(proc),
    )
    for cancer in sorted({target.cancer for target in CANONICAL_TARGETS})
}


def canonical_targets_table(include_background: bool = True) -> pd.DataFrame:
    rows = [target.__dict__ for target in CANONICAL_TARGETS]
    df = pd.DataFrame(rows)
    if not include_background:
        df = df[df["priority"].ne("background")]
    return df.reset_index(drop=True)


def associated_sbs_map(include_background: bool = True) -> dict[str, list[str]]:
    df = canonical_targets_table(include_background=include_background)
    out: dict[str, list[str]] = {}
    for cancer, group in df.groupby("cancer", sort=False):
        out[str(cancer)] = sorted(group["signature"].astype(str).unique(), key=lambda s: int(s.replace("SBS", "")))
    return out


def associated_sbs_table(include_background: bool = True) -> pd.DataFrame:
    df = canonical_targets_table(include_background=include_background).copy()
    df["well_known_sbs"] = True
    df["cancer_associated"] = df["priority"].isin(["primary", "secondary"])
    df["cancer_association_note"] = df["note"]
    return df.rename(columns={"signature": "signature"})

