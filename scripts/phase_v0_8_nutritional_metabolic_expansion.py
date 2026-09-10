#!/usr/bin/env python3
"""
Phase v0.8 — TB–undernutrition proteomics nutritional-metabolic expansion.

Purpose
-------
1. Preserve the v0.7.1 BMI-defined manuscript as an immutable baseline.
2. Discover compatible proteomics and metadata files without overwriting prior outputs.
3. Define nutritional-metabolic and cellular pathogen-control modules.
4. Audit module coverage.
5. Score usable modules.
6. Run baseline, longitudinal, delta, BMI-group × time, and continuous BMI × time tests.
7. Build an integrated metabolic-undernutrition score when feasible.
8. Correlate new modules with compatible existing module-score tables.
9. Produce figure-ready tables, reports, decision logs, and draft figures.
10. Export gene lists and templates for later single-cell/spatial triangulation.

This script does NOT measure vitamin concentrations, intracellular calcium flux,
phagocytosis efficiency, or Mycobacterium tuberculosis killing. It analyzes
serum protein/gene module proxies only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import re
import shutil
import sys
import textwrap
import traceback
import warnings as pywarnings
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception as exc:
    raise SystemExit(
        "ERROR: scipy is required. Install with: python -m pip install scipy"
    ) from exc

try:
    import statsmodels.formula.api as smf
    from statsmodels.stats.multitest import multipletests
    HAVE_STATSMODELS = True
except Exception:
    HAVE_STATSMODELS = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MATPLOTLIB = True
except Exception:
    HAVE_MATPLOTLIB = False


EXPECTED_ROOT = Path(__file__).resolve().parents[1]
PHASE_TAG = "phase_v0_8_nutritional_metabolic_expansion"
SCRIPT_VERSION = "0.8.1"
BMI_CUTOFF = 18.5
MIN_QUANTITATIVE_PROTEINS = 5
MIN_LIMITED_PROXY_PROTEINS = 3
MIN_SCORE_PROTEINS = 3
MIN_GROUP_N = 3
RANDOM_SEED = 20260714

np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Module dictionary
# ---------------------------------------------------------------------------

MODULES: "OrderedDict[str, dict[str, Any]]" = OrderedDict({
    "vitamin_d_transport_metabolism_signalling": {
        "label": "Vitamin D transport, metabolism and signalling",
        "domain": "nutritional_metabolic",
        "expected_direction_in_recovery": "context_dependent",
        "proteins": [
            "GC", "VDR", "CYP2R1", "CYP27B1", "CYP24A1", "CYP27A1",
            "RXRA", "RXRB", "RXRG", "LRP2", "CUBN", "PDIA3", "DHCR7",
            "SEC14L2", "SULT2A1"
        ],
        "notes": (
            "GC/vitamin-D-binding protein is a serum transport proxy. VDR and "
            "vitamin-D metabolic enzymes may be poorly represented in serum."
        ),
    },
    "vitamin_d_linked_antimicrobial_autophagy_phagolysosome": {
        "label": "Vitamin D-linked antimicrobial, autophagy and phagolysosome machinery",
        "domain": "cellular_pathogen_control",
        "expected_direction_in_recovery": "context_dependent",
        "proteins": [
            "CAMP", "DEFB1", "NOD2", "TLR2", "SLC11A1", "LYZ", "CTSB", "CTSD",
            "CTSL", "LAMP1", "LAMP2", "RAB5A", "RAB7A", "RUBCN", "TFEB",
            "ATG3", "ATG5", "ATG7", "ATG12", "ATG16L1", "BECN1", "ULK1",
            "MAP1LC3A", "MAP1LC3B", "GABARAP", "SQSTM1", "OPTN", "CALCOCO2",
            "ATP6V0A1", "ATP6V0D1", "ATP6V1A", "ATP6V1B2", "ATP6V1E1",
            "CYBB", "NCF1", "NCF2", "NCF4", "RAC2", "LTF", "MPO"
        ],
        "notes": (
            "Serum abundance is an indirect proxy for intracellular antimicrobial "
            "and lysosomal biology and cannot establish Mtb killing capacity."
        ),
    },
    "calcium_calmodulin_calcineurin_signalling": {
        "label": "Calcium/calmodulin/calcineurin signalling",
        "domain": "signalling",
        "expected_direction_in_recovery": "context_dependent",
        "proteins": [
            "CALM1", "CALM2", "CALM3", "CAMK1", "CAMK2A", "CAMK2B", "CAMK2D",
            "CAMK2G", "CAMK4", "CAMKK1", "CAMKK2", "PPP3CA", "PPP3CB",
            "PPP3CC", "PPP3R1", "PPP3R2", "NFATC1", "NFATC2", "NFATC3",
            "NFATC4", "RCAN1", "CABIN1", "CHP1", "CHP2", "CALB1", "CALB2",
            "S100A8", "S100A9", "S100A10", "S100A12", "S100B"
        ],
        "notes": (
            "Detected proteins are pathway proxies and do not measure intracellular "
            "calcium concentration or calcineurin activity directly."
        ),
    },
    "calcium_channels_and_mobilization_proxies": {
        "label": "Calcium channels and calcium-mobilization proxies",
        "domain": "signalling",
        "expected_direction_in_recovery": "context_dependent",
        "proteins": [
            "ITPR1", "ITPR2", "ITPR3", "ORAI1", "ORAI2", "ORAI3", "STIM1",
            "STIM2", "TRPC1", "TRPC3", "TRPC6", "TRPM2", "TRPM7", "P2RX7",
            "RYR1", "RYR2", "RYR3", "ATP2A1", "ATP2A2", "ATP2A3", "ATP2B1",
            "ATP2B4", "SLC8A1", "SLC8A3", "MCU", "MICU1", "MICU2", "VDAC1",
            "VDAC2", "VDAC3", "CACNA1C", "CACNA1D", "CACNA2D1"
        ],
        "notes": (
            "Most channel proteins are membrane-associated and may be absent from "
            "serum proteomics. Any detected signal is a limited proxy."
        ),
    },
    "cytoskeleton_actin_immune_synapse_phagocytosis": {
        "label": "Cytoskeleton, actin remodelling, immune synapse and phagocytosis",
        "domain": "cellular_pathogen_control",
        "expected_direction_in_recovery": "context_dependent",
        "proteins": [
            "ACTB", "ACTG1", "ACTR2", "ACTR3", "ARPC1A", "ARPC1B", "ARPC2",
            "ARPC3", "ARPC4", "ARPC5", "WAS", "WASL", "WASF2", "WIPF1",
            "CDC42", "RAC1", "RAC2", "RHOA", "PFN1", "CFL1", "CFL2", "GSN",
            "CORO1A", "VASP", "EZR", "MSN", "RDX", "TLN1", "TLN2", "FERMT3",
            "MYH9", "MYH10", "MYL12A", "MYL12B", "DOCK2", "DOCK8", "VAV1",
            "SYK", "LYN", "HCK", "FGR", "LCP2", "PIK3CG", "FCGR1A", "FCGR2A",
            "FCGR2B", "FCGR3A", "FCGR3B", "ITGAM", "ITGAX", "ITGB2", "TLR2",
            "MARCO", "MRC1", "CLEC7A"
        ],
        "notes": (
            "Serum detection may reflect secretion, vesicles, cell turnover, injury, "
            "or leakage rather than intact phagocyte function."
        ),
    },
    "contractile_tissue_remodelling_markers": {
        "label": "Contractile and tissue-remodelling protein markers",
        "domain": "tissue_remodelling",
        "expected_direction_in_recovery": "context_dependent",
        "proteins": [
            "CALD1", "CALM1", "CALM2", "CALM3", "CNN1", "CNN2", "CNN3",
            "TNNT1", "TNNT2", "TNNT3", "TNNI1", "TNNI2", "TNNI3", "TNNC1",
            "TNNC2", "TPM1", "TPM2", "TPM3", "TPM4", "MYL6", "MYL9",
            "MYL12A", "MYL12B", "MYH9", "MYH10", "MYH11", "TAGLN", "ACTA2",
            "DES", "VIM", "FLNA", "FLNB", "FLNC", "PDLIM1", "PDLIM5",
            "TLN1", "TLN2", "PXN", "VCL"
        ],
        "notes": (
            "Interpret cautiously because serum contractile proteins may reflect "
            "tissue injury, vascular remodelling, platelet biology, or cell turnover."
        ),
    },
    "vitamin_a_retinol_transport": {
        "label": "Vitamin A/retinol transport",
        "domain": "nutritional_metabolic",
        "expected_direction_in_recovery": "increase",
        "proteins": [
            "RBP4", "TTR", "ALB", "APOA1", "APOB", "RBP1", "RBP2", "RBP3",
            "RBP7", "STRA6", "LRP2", "CUBN", "LRAT", "BCO1", "BCO2"
        ],
        "notes": (
            "RBP4, TTR and albumin are transport/host-state proxies; they do not "
            "directly quantify circulating retinol."
        ),
    },
    "retinoic_acid_signalling_lipid_cholesterol_trafficking": {
        "label": "Retinoic-acid signalling and lipid/cholesterol trafficking",
        "domain": "nutritional_metabolic",
        "expected_direction_in_recovery": "increase",
        "proteins": [
            "CRABP1", "CRABP2", "RARA", "RARB", "RARG", "RXRA", "RXRB",
            "RXRG", "CYP26A1", "CYP26B1", "CYP26C1", "RDH10", "ALDH1A1",
            "ALDH1A2", "ALDH1A3", "ABCA1", "ABCG1", "APOA1", "APOA2",
            "APOA4", "APOB", "APOC1", "APOC2", "APOC3", "APOC4", "APOD",
            "APOE", "LCAT", "CETP", "PLTP", "LPL", "LDLR", "LRP1", "SCARB1",
            "NPC1", "NPC2", "OSBPL1A", "OSBPL2", "OSBPL5", "STARD3",
            "STARD4", "SREBF1", "SREBF2", "HMGCR", "CYP27A1"
        ],
        "notes": (
            "Serum apolipoprotein and lipid-carrier recovery can be quantified, "
            "whereas nuclear retinoid receptor activity remains indirect."
        ),
    },
    "vitamin_c_redox_collagen_ecm_repair": {
        "label": "Vitamin C/redox/collagen maturation and ECM repair",
        "domain": "nutritional_metabolic",
        "expected_direction_in_recovery": "context_dependent",
        "proteins": [
            "SLC23A1", "SLC23A2", "P4HA1", "P4HA2", "P4HA3", "P4HB",
            "PLOD1", "PLOD2", "PLOD3", "COL1A1", "COL1A2", "COL3A1",
            "COL4A1", "COL4A2", "COL6A1", "COL6A2", "FN1", "VTN", "LUM",
            "DCN", "BGN", "MMP2", "MMP8", "MMP9", "TIMP1", "TIMP2", "PXDN",
            "PRDX1", "PRDX2", "PRDX3", "PRDX4", "PRDX5", "PRDX6", "GPX1",
            "GPX3", "GPX4", "SOD1", "SOD2", "SOD3", "CAT", "GSR", "TXN",
            "TXNRD1", "GSTP1", "AOC3"
        ],
        "notes": (
            "This is a redox/ECM protein proxy and does not directly measure "
            "ascorbate concentration or collagen hydroxylation flux."
        ),
    },
    "b_vitamin_nad_metabolism": {
        "label": "B-vitamin/cofactor metabolism — NAD",
        "domain": "nutritional_metabolic",
        "expected_direction_in_recovery": "increase",
        "proteins": [
            "NAMPT", "NADSYN1", "NMNAT1", "NMNAT2", "NMNAT3", "NMRK1",
            "NMRK2", "QPRT", "NAPRT", "NNT", "NNMT", "CD38", "ENPP1",
            "SIRT1", "SIRT2", "SIRT3", "SIRT4", "SIRT5", "SIRT6", "SIRT7",
            "PARP1", "PARP2"
        ],
        "notes": "Protein proxies for NAD synthesis, salvage and consumption.",
    },
    "b_vitamin_folate_one_carbon": {
        "label": "B-vitamin/cofactor metabolism — folate and one-carbon",
        "domain": "nutritional_metabolic",
        "expected_direction_in_recovery": "increase",
        "proteins": [
            "SLC19A1", "FOLR1", "FOLR2", "DHFR", "MTHFR", "MTHFD1",
            "MTHFD1L", "MTHFD2", "MTHFD2L", "SHMT1", "SHMT2", "TYMS",
            "GART", "ATIC", "CBS", "MTR", "MTRR", "MAT1A", "MAT2A",
            "AHCY", "BHMT"
        ],
        "notes": "Protein proxies for folate-dependent one-carbon metabolism.",
    },
    "b_vitamin_riboflavin_fad": {
        "label": "B-vitamin/cofactor metabolism — riboflavin/FAD",
        "domain": "nutritional_metabolic",
        "expected_direction_in_recovery": "increase",
        "proteins": [
            "SLC52A1", "SLC52A2", "SLC52A3", "RFK", "FLAD1", "ETFA",
            "ETFB", "ETFDH", "ACADVL", "ACADM", "ACADS", "GSR", "NQO1"
        ],
        "notes": "Protein proxies for riboflavin transport and FAD/FMN biology.",
    },
    "b_vitamin_thiamine": {
        "label": "B-vitamin/cofactor metabolism — thiamine",
        "domain": "nutritional_metabolic",
        "expected_direction_in_recovery": "increase",
        "proteins": [
            "SLC19A2", "SLC19A3", "TPK1", "SLC25A19", "PDHA1", "PDHB",
            "DLAT", "DLD", "OGDH", "DLST", "BCKDHA", "BCKDHB", "DBT",
            "TKT", "TKTL1"
        ],
        "notes": "Protein proxies for thiamine transport and dependent enzyme systems.",
    },
    "b_vitamin_b6": {
        "label": "B-vitamin/cofactor metabolism — vitamin B6",
        "domain": "nutritional_metabolic",
        "expected_direction_in_recovery": "increase",
        "proteins": [
            "PDXK", "PNPO", "ALPL", "GOT1", "GOT2", "BCAT1", "BCAT2",
            "CBS", "CTH", "SHMT1", "SHMT2", "KYNU", "KMO", "AOC1"
        ],
        "notes": "Protein proxies for pyridoxal-phosphate metabolism and dependent enzymes.",
    },
    "b_vitamin_b12": {
        "label": "B-vitamin/cofactor metabolism — vitamin B12",
        "domain": "nutritional_metabolic",
        "expected_direction_in_recovery": "increase",
        "proteins": [
            "TCN1", "TCN2", "CUBN", "LMBRD1", "ABCD4", "MMACHC", "MMAA",
            "MMAB", "MUT", "MTR", "MTRR"
        ],
        "notes": "Protein proxies for cobalamin transport, processing and utilization.",
    },
})

EXISTING_MODULE_KEYWORDS = [
    "inflamm", "acute_phase", "lipid", "apolipoprotein", "apo_", "retinol",
    "metal", "nutritional_immunity", "autophagy", "lysosome", "endocrine",
    "immunometabolic", "module_score", "modulescore"
]

PROTEIN_ID_ALIASES = [
    "gene", "gene_symbol", "genesymbol", "symbol", "protein", "protein_id",
    "proteinid", "protein_name", "proteinname", "accession", "uniprot",
    "uniprot_id", "uniprotid", "description", "feature", "analyte"
]
SAMPLE_ID_ALIASES = [
    "sample", "sample_id", "sampleid", "specimen", "specimen_id", "aliquot",
    "subject_sample", "sample_name", "samplename"
]
SUBJECT_ID_ALIASES = [
    "subject", "subject_id", "subjectid", "participant", "participant_id",
    "patient", "patient_id", "patientid", "individual", "individual_id",
    "study_id", "studyid"
]
BMI_ALIASES = [
    "bmi", "baseline_bmi", "body_mass_index", "bodymassindex", "bmi_baseline"
]
BMI_GROUP_ALIASES = [
    "bmi_group", "bmigroup", "nutritional_status", "nutrition_group",
    "undernutrition", "underweight_group", "bmi_stratum", "bmi_category"
]
TIME_ALIASES = [
    "time", "timepoint", "time_point", "visit", "visit_name", "study_visit",
    "day", "week", "month", "treatment_time", "treatment_timepoint"
]
VALUE_ALIASES = [
    "value", "abundance", "intensity", "log2_abundance", "log2_intensity",
    "normalized_abundance", "expression", "measurement"
]
ANNOTATION_ALIASES = [
    "gene", "gene_symbol", "genesymbol", "symbol", "protein_name",
    "proteinname", "description", "accession", "uniprot", "uniprot_id"
]


@dataclass
class InputCandidate:
    path: str
    file_type: str
    size_bytes: int
    modified_utc: str
    name_score: float
    inspection_status: str = "not_inspected"
    n_rows_preview: Optional[int] = None
    n_columns_preview: Optional[int] = None
    possible_role: Optional[str] = None
    notes: Optional[str] = None


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def slugify(value: str) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def normalize_colname(value: str) -> str:
    return slugify(value)


def normalize_symbol(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).upper().strip()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = text.replace("|", " ").replace(";", " ").replace(",", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def symbol_tokens(value: Any) -> set[str]:
    text = normalize_symbol(value)
    tokens = set(re.findall(r"\b[A-Z0-9][A-Z0-9\-\.]{1,20}\b", text))
    expanded = set(tokens)
    for token in list(tokens):
        expanded.add(token.split(".")[0])
        expanded.add(token.split("-")[0])
    return {x for x in expanded if x}


def extract_primary_label(value: Any, fallback: str) -> str:
    """Extract a stable protein/gene label without collapsing UniProt descriptions."""
    if pd.isna(value):
        return fallback
    raw = str(value).strip()
    # UniProt text commonly contains an authoritative GN= gene-symbol field.
    gn_match = re.search(r"(?:^|\s)GN=([A-Za-z0-9][A-Za-z0-9_.-]{1,30})", raw)
    if gn_match:
        return gn_match.group(1).upper()
    # Handle common key-value annotation forms.
    key_match = re.search(
        r"(?:gene(?:_symbol)?|symbol)\s*[:=]\s*([A-Za-z0-9][A-Za-z0-9_.-]{1,30})",
        raw,
        flags=re.IGNORECASE,
    )
    if key_match:
        return key_match.group(1).upper()
    # A clean single token is usually already a gene symbol or accession.
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{1,30}", raw):
        return raw.upper()
    # Prefer tokens that are not annotation keys or generic description words.
    stop = {
        "SP", "TR", "OS", "OX", "GN", "PE", "SV", "HUMAN", "HOMO",
        "SAPIENS", "PROTEIN", "ISOFORM", "CHAIN", "PRECURSOR", "FRAGMENT",
        "UNCHARACTERIZED", "PUTATIVE"
    }
    candidates = [
        token for token in re.findall(r"\b[A-Za-z][A-Za-z0-9_.-]{1,30}\b", raw)
        if token.upper() not in stop
    ]
    if candidates:
        # Gene symbols are often uppercase/digit-rich; preserve first plausible token.
        ranked = sorted(
            enumerate(candidates),
            key=lambda item: (
                0 if re.fullmatch(r"[A-Z0-9][A-Z0-9_.-]+", item[1]) else 1,
                item[0],
            ),
        )
        return ranked[0][1].upper()
    return fallback


def sha256sum(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def bh_fdr(values: Iterable[Any]) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(float)
    result = np.full(arr.shape, np.nan)
    mask = np.isfinite(arr)
    if not mask.any():
        return result
    if HAVE_STATSMODELS:
        result[mask] = multipletests(arr[mask], method="fdr_bh")[1]
        return result
    p = arr[mask]
    order = np.argsort(p)
    ranked = p[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    unorder = np.empty_like(adjusted)
    unorder[order] = adjusted
    result[mask] = unorder
    return result


def cohen_d(x: Iterable[float], y: Iterable[float]) -> float:
    x = np.asarray(pd.to_numeric(pd.Series(x), errors="coerce").dropna(), float)
    y = np.asarray(pd.to_numeric(pd.Series(y), errors="coerce").dropna(), float)
    if len(x) < 2 or len(y) < 2:
        return np.nan
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    pooled = ((len(x) - 1) * vx + (len(y) - 1) * vy) / (len(x) + len(y) - 2)
    if pooled <= 0:
        return np.nan
    return (np.mean(x) - np.mean(y)) / math.sqrt(pooled)


def paired_d(delta: Iterable[float]) -> float:
    delta = np.asarray(pd.to_numeric(pd.Series(delta), errors="coerce").dropna(), float)
    if len(delta) < 2 or np.std(delta, ddof=1) == 0:
        return np.nan
    return np.mean(delta) / np.std(delta, ddof=1)


def safe_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    pair = pd.concat([x, y], axis=1).dropna()
    if len(pair) < MIN_GROUP_N:
        return np.nan, np.nan, len(pair)
    rho, p = stats.spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
    return float(rho), float(p), len(pair)


def classify_coverage(detected: int, total: int) -> str:
    proportion = detected / total if total else 0.0
    if detected >= MIN_QUANTITATIVE_PROTEINS and proportion >= 0.20:
        return "candidate quantitative module"
    if detected >= MIN_LIMITED_PROXY_PROTEINS:
        return "limited proxy module"
    if detected >= 1:
        return "sparse proxy only"
    return "not detected"


def setup_logger(log_path: Path, verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger("phase_v0_8")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def scan_candidates(root: Path, output_root: Path) -> list[InputCandidate]:
    supported = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet", ".feather"}
    exclude_parts = {
        ".git", "__pycache__", "node_modules", "final_tiff", "final_eps",
        "final_svg", "preview_png", "draft_panels", "docx", "journal_targeting"
    }
    candidates: list[InputCandidate] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in supported:
            continue
        try:
            if output_root in path.parents:
                continue
        except Exception:
            pass
        rel_parts = {p.lower() for p in path.relative_to(root).parts}
        if rel_parts & exclude_parts:
            continue
        lower = path.name.lower()
        if any(token in lower for token in [
            "phase_v0_8", "coverage_audit", "module_score", "statistical_results",
            "candidate_inventory", "decision_log"
        ]):
            continue
        score = 0.0
        positive = {
            "proteom": 5, "protein": 4, "tmt": 4, "normalized": 4,
            "normalised": 4, "abundance": 4, "intensity": 3, "matrix": 3,
            "noref": 2, "serum": 2, "metadata": 4, "clinical": 3,
            "sample": 2, "manifest": 2
        }
        negative = {
            "result": -2, "figure": -5, "table_s": -2, "supplement": -2,
            "reference": -4, "author": -4, "manuscript": -5, "citation": -5
        }
        for token, weight in positive.items():
            if token in lower:
                score += weight
        for token, weight in negative.items():
            if token in lower:
                score += weight
        stat = path.stat()
        candidates.append(InputCandidate(
            path=str(path),
            file_type=path.suffix.lower(),
            size_bytes=stat.st_size,
            modified_utc=datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat(timespec="seconds"),
            name_score=score,
        ))
    return sorted(candidates, key=lambda x: (x.name_score, x.size_bytes), reverse=True)


def read_table(path: Path, sheet: Optional[str] = None, nrows: Optional[int] = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, nrows=nrows, low_memory=False)
    if suffix in {".tsv", ".txt"}:
        try:
            return pd.read_csv(path, sep="\t", nrows=nrows, low_memory=False)
        except Exception:
            return pd.read_csv(path, sep=None, engine="python", nrows=nrows)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet if sheet is not None else 0, nrows=nrows)
    if suffix == ".parquet":
        df = pd.read_parquet(path)
        return df.head(nrows) if nrows else df
    if suffix == ".feather":
        df = pd.read_feather(path)
        return df.head(nrows) if nrows else df
    raise ValueError(f"Unsupported tabular file: {path}")


def list_excel_sheets(path: Path) -> list[str]:
    if path.suffix.lower() not in {".xlsx", ".xls"}:
        return []
    try:
        return list(pd.ExcelFile(path).sheet_names)
    except Exception:
        return []


def first_matching_column(columns: Iterable[Any], aliases: Iterable[str]) -> Optional[str]:
    norm_map = {normalize_colname(c): str(c) for c in columns}
    for alias in aliases:
        if normalize_colname(alias) in norm_map:
            return norm_map[normalize_colname(alias)]
    # Partial matching only after exact matching.
    for norm, original in norm_map.items():
        if any(normalize_colname(alias) in norm for alias in aliases):
            return original
    return None


def numeric_fraction(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return pd.to_numeric(series, errors="coerce").notna().mean()


def inspect_candidate(candidate: InputCandidate) -> InputCandidate:
    path = Path(candidate.path)
    try:
        sheets = list_excel_sheets(path)
        sheet = sheets[0] if sheets else None
        df = read_table(path, sheet=sheet, nrows=100)
        candidate.inspection_status = "ok"
        candidate.n_rows_preview = len(df)
        candidate.n_columns_preview = len(df.columns)
        cols = list(df.columns)
        sample_col = first_matching_column(cols, SAMPLE_ID_ALIASES)
        protein_col = first_matching_column(cols, PROTEIN_ID_ALIASES)
        bmi_col = first_matching_column(cols, BMI_ALIASES)
        time_col = first_matching_column(cols, TIME_ALIASES)
        value_col = first_matching_column(cols, VALUE_ALIASES)

        numeric_cols = [c for c in cols if numeric_fraction(df[c]) >= 0.70]
        role = "unknown"
        notes = []
        if sample_col and bmi_col:
            role = "metadata"
        if sample_col and protein_col and value_col:
            role = "long_proteomics"
        elif protein_col and len(numeric_cols) >= 3:
            role = "wide_protein_rows"
        elif sample_col and len(numeric_cols) >= 10:
            role = "wide_sample_rows_or_scores"
        elif bmi_col or time_col:
            role = "metadata"
        if sheets:
            notes.append("sheets=" + ",".join(sheets[:20]))
        notes.append("columns=" + ",".join(map(str, cols[:30])))
        candidate.possible_role = role
        candidate.notes = " | ".join(notes)
    except Exception as exc:
        candidate.inspection_status = "failed"
        candidate.notes = f"{type(exc).__name__}: {exc}"
    return candidate


def choose_candidate(
    candidates: list[InputCandidate],
    role_preferences: set[str],
    override: Optional[Path] = None,
) -> Optional[InputCandidate]:
    if override:
        return InputCandidate(
            path=str(override),
            file_type=override.suffix.lower(),
            size_bytes=override.stat().st_size,
            modified_utc=datetime.fromtimestamp(
                override.stat().st_mtime, tz=timezone.utc
            ).isoformat(timespec="seconds"),
            name_score=999,
            inspection_status="override",
            possible_role="override",
        )
    eligible = [
        c for c in candidates
        if c.inspection_status == "ok" and c.possible_role in role_preferences
    ]
    if not eligible:
        return None
    return sorted(eligible, key=lambda c: (c.name_score, c.size_bytes), reverse=True)[0]


def identify_baseline(time_series: pd.Series) -> Any:
    values = [x for x in pd.unique(time_series.dropna())]
    if not values:
        return None
    normalized = {x: normalize_colname(x) for x in values}
    baseline_tokens = {
        "baseline", "base", "pre", "pretreatment", "pre_treatment", "day0",
        "day_0", "d0", "week0", "week_0", "w0", "month0", "month_0", "m0",
        "visit0", "visit_0", "t0"
    }
    for original, norm in normalized.items():
        if norm in baseline_tokens or norm.startswith("baseline"):
            return original
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    if numeric.notna().all():
        return values[int(np.nanargmin(numeric.to_numpy(float)))]
    # Try extracting the first numeric component.
    extracted = []
    for original in values:
        match = re.search(r"-?\d+(?:\.\d+)?", str(original))
        extracted.append(float(match.group()) if match else np.nan)
    if np.isfinite(extracted).any():
        return values[int(np.nanargmin(np.asarray(extracted, float)))]
    return sorted(values, key=lambda x: str(x))[0]


def order_timepoints(time_series: pd.Series, baseline: Any) -> list[Any]:
    values = list(pd.unique(time_series.dropna()))
    def key(v: Any) -> tuple[int, float, str]:
        if v == baseline:
            return (0, -np.inf, str(v))
        numeric = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
        if pd.notna(numeric):
            return (1, float(numeric), str(v))
        match = re.search(r"-?\d+(?:\.\d+)?", str(v))
        if match:
            return (1, float(match.group()), str(v))
        return (2, np.inf, str(v))
    return sorted(values, key=key)


def standardize_bmi_group(series: pd.Series) -> pd.Series:
    def convert(v: Any) -> Any:
        if pd.isna(v):
            return np.nan
        text = normalize_colname(v)
        under_tokens = [
            "under", "low_bmi", "bmi_lt_18_5", "lt18_5", "below_18_5",
            "malnour", "undernour", "thin"
        ]
        normal_tokens = [
            "normal", "adequate", "bmi_ge_18_5", "ge18_5", "above_18_5",
            "not_under", "non_under", "higher_bmi"
        ]
        if any(token in text for token in under_tokens):
            return "BMI<18.5"
        if any(token in text for token in normal_tokens):
            return "BMI>=18.5"
        num = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
        if pd.notna(num):
            return "BMI<18.5" if float(num) < BMI_CUTOFF else "BMI>=18.5"
        return str(v)
    return series.map(convert)


def derive_metadata(
    df: pd.DataFrame,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, Optional[str]]]:
    cols = list(df.columns)
    sample_col = args.sample_id_column or first_matching_column(cols, SAMPLE_ID_ALIASES)
    subject_col = args.subject_id_column or first_matching_column(cols, SUBJECT_ID_ALIASES)
    bmi_col = args.bmi_column or first_matching_column(cols, BMI_ALIASES)
    bmi_group_col = args.bmi_group_column or first_matching_column(cols, BMI_GROUP_ALIASES)
    time_col = args.time_column or first_matching_column(cols, TIME_ALIASES)

    if not sample_col:
        raise ValueError(
            "Could not identify a sample ID column in metadata. "
            "Use --sample-id-column."
        )

    meta = df.copy()
    meta = meta.rename(columns={sample_col: "sample_id"})
    if subject_col:
        meta = meta.rename(columns={subject_col: "subject_id"})
    else:
        meta["subject_id"] = meta["sample_id"].astype(str)

    if time_col:
        meta = meta.rename(columns={time_col: "timepoint"})
    else:
        meta["timepoint"] = "baseline_only"

    if bmi_col:
        meta = meta.rename(columns={bmi_col: "bmi"})
        meta["bmi"] = pd.to_numeric(meta["bmi"], errors="coerce")
    else:
        meta["bmi"] = np.nan

    if bmi_group_col:
        meta = meta.rename(columns={bmi_group_col: "bmi_group"})
        meta["bmi_group"] = standardize_bmi_group(meta["bmi_group"])
    elif meta["bmi"].notna().any():
        meta["bmi_group"] = pd.Series(np.nan, index=meta.index, dtype=object)
        observed_bmi = meta["bmi"].notna()
        meta.loc[observed_bmi, "bmi_group"] = np.where(
            meta.loc[observed_bmi, "bmi"] < BMI_CUTOFF,
            "BMI<18.5",
            "BMI>=18.5",
        )
    else:
        meta["bmi_group"] = np.nan

    meta["sample_id"] = meta["sample_id"].astype(str).str.strip()
    meta["subject_id"] = meta["subject_id"].astype(str).str.strip()

    # Baseline BMI is commonly recorded once per participant. Carry the available
    # participant-level value and derived stratum across longitudinal samples.
    meta["bmi"] = meta.groupby("subject_id", dropna=False)["bmi"].transform(
        lambda s: s.ffill().bfill()
    )
    meta["bmi_group"] = meta.groupby("subject_id", dropna=False)["bmi_group"].transform(
        lambda s: s.ffill().bfill()
    )
    missing_group = meta["bmi_group"].isna() & meta["bmi"].notna()
    meta.loc[missing_group, "bmi_group"] = np.where(
        meta.loc[missing_group, "bmi"] < BMI_CUTOFF,
        "BMI<18.5",
        "BMI>=18.5",
    )
    meta = meta.drop_duplicates(subset=["sample_id"], keep="first")

    mapping = {
        "sample_id": sample_col,
        "subject_id": subject_col,
        "bmi": bmi_col,
        "bmi_group": bmi_group_col,
        "timepoint": time_col,
    }
    logger.info("Metadata mapping: %s", mapping)
    return meta, mapping


def infer_annotation_column(df: pd.DataFrame, explicit: Optional[str] = None) -> str:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"Protein ID column not found: {explicit}")
        return explicit
    col = first_matching_column(df.columns, ANNOTATION_ALIASES)
    if col:
        return col
    return str(df.columns[0])


def wide_protein_rows_to_sample_by_protein(
    df: pd.DataFrame,
    annotation_col: str,
    metadata_sample_ids: Optional[set[str]],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    annotation_cols = []
    for col in df.columns:
        norm = normalize_colname(col)
        if col == annotation_col or any(normalize_colname(a) == norm for a in ANNOTATION_ALIASES):
            annotation_cols.append(col)

    numeric_cols = [
        col for col in df.columns
        if col not in annotation_cols and numeric_fraction(df[col]) >= 0.60
    ]
    if metadata_sample_ids:
        exact_sample_cols = [
            col for col in numeric_cols if str(col).strip() in metadata_sample_ids
        ]
        if len(exact_sample_cols) >= 2:
            numeric_cols = exact_sample_cols

    if len(numeric_cols) < 2:
        raise ValueError(
            "Fewer than two numeric sample columns were found in the protein matrix."
        )

    annotation_text = df[annotation_cols].astype(str).agg(" | ".join, axis=1)
    # Preserve the project protein-row identifier exactly. Gene-symbol mapping is
    # added later from the project protein inventory; premature parsing can turn
    # UniProt-style identifiers into accessions and lose the gene mnemonic.
    protein_labels = df[annotation_col].astype(str).str.strip().tolist()
    protein_labels = [label if label else f"FEATURE_{i+1}" for i, label in enumerate(protein_labels)]

    abundance = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    abundance.index = protein_labels
    abundance = abundance.groupby(level=0).mean()
    sample_by_protein = abundance.T
    sample_by_protein.index = sample_by_protein.index.astype(str).str.strip()
    sample_by_protein.index.name = "sample_id"

    annotation = pd.DataFrame({
        "feature_row": np.arange(len(df)),
        "primary_label": protein_labels,
        "annotation_text": annotation_text,
    })
    logger.info(
        "Converted wide protein-row matrix to %d samples × %d protein labels.",
        sample_by_protein.shape[0], sample_by_protein.shape[1]
    )
    return sample_by_protein, annotation


def wide_sample_rows_to_sample_by_protein(
    df: pd.DataFrame,
    sample_col: str,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric_cols = [
        col for col in df.columns
        if col != sample_col and numeric_fraction(df[col]) >= 0.60
    ]
    if len(numeric_cols) < 2:
        raise ValueError("Fewer than two numeric protein columns were found.")
    matrix = df[[sample_col] + numeric_cols].copy()
    matrix = matrix.rename(columns={sample_col: "sample_id"})
    matrix["sample_id"] = matrix["sample_id"].astype(str).str.strip()
    matrix = matrix.drop_duplicates("sample_id").set_index("sample_id")
    matrix = matrix.apply(pd.to_numeric, errors="coerce")
    rename = {}
    annotations = []
    for col in matrix.columns:
        label = str(col).strip()
        rename[col] = label
        annotations.append({
            "source_column": str(col),
            "primary_label": label,
            "annotation_text": str(col),
        })
    matrix = matrix.rename(columns=rename)
    matrix = matrix.T.groupby(level=0).mean().T
    logger.info(
        "Using wide sample-row matrix with %d samples × %d protein labels.",
        matrix.shape[0], matrix.shape[1]
    )
    return matrix, pd.DataFrame(annotations)


def long_to_sample_by_protein(
    df: pd.DataFrame,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_col = args.sample_id_column or first_matching_column(df.columns, SAMPLE_ID_ALIASES)
    protein_col = args.protein_id_column or first_matching_column(df.columns, PROTEIN_ID_ALIASES)
    value_col = args.value_column or first_matching_column(df.columns, VALUE_ALIASES)
    if not sample_col or not protein_col or not value_col:
        raise ValueError(
            "Long-format proteomics requires sample, protein, and value columns."
        )
    temp = df[[sample_col, protein_col, value_col]].copy()
    temp["sample_id"] = temp[sample_col].astype(str).str.strip()
    # Preserve exact protein_row_id values for lossless joining to the existing
    # project protein inventory and curated alias tables.
    temp["protein_label"] = temp[protein_col].astype(str).str.strip()
    empty_label = temp["protein_label"].eq("") | temp["protein_label"].eq("nan")
    temp.loc[empty_label, "protein_label"] = [
        f"FEATURE_{i+1}" for i in range(int(empty_label.sum()))
    ]
    temp["value"] = pd.to_numeric(temp[value_col], errors="coerce")
    matrix = temp.pivot_table(
        index="sample_id", columns="protein_label", values="value", aggfunc="mean"
    )
    annotation = temp[["protein_label", protein_col]].drop_duplicates()
    annotation = annotation.rename(columns={protein_col: "annotation_text"})
    logger.info(
        "Converted long matrix to %d samples × %d protein labels.",
        matrix.shape[0], matrix.shape[1]
    )
    return matrix, annotation


def load_proteomics_matrix(
    path: Path,
    metadata_sample_ids: Optional[set[str]],
    args: argparse.Namespace,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    sheet = args.protein_sheet
    if sheet is None and path.suffix.lower() in {".xlsx", ".xls"}:
        sheets = list_excel_sheets(path)
        # Prefer sheet names suggesting normalized abundance.
        ranked = sorted(
            sheets,
            key=lambda s: sum(
                token in s.lower()
                for token in ["normal", "abundance", "protein", "tmt", "matrix"]
            ),
            reverse=True,
        )
        sheet = ranked[0] if ranked else sheets[0]
    df = read_table(path, sheet=sheet)
    role = args.matrix_orientation
    if role == "auto":
        sample_col = first_matching_column(df.columns, SAMPLE_ID_ALIASES)
        protein_col = first_matching_column(df.columns, PROTEIN_ID_ALIASES)
        value_col = first_matching_column(df.columns, VALUE_ALIASES)
        if sample_col and protein_col and value_col:
            role = "long"
        elif protein_col:
            role = "protein_rows"
        elif sample_col:
            role = "sample_rows"
        else:
            # Heuristic: first column text, many subsequent numeric columns => proteins in rows.
            first_col = df.columns[0]
            if numeric_fraction(df[first_col]) < 0.30 and sum(
                numeric_fraction(df[c]) >= 0.60 for c in df.columns[1:]
            ) >= 2:
                role = "protein_rows"
            else:
                role = "sample_rows"

    if role == "long":
        matrix, annotation = long_to_sample_by_protein(df, args, logger)
    elif role == "protein_rows":
        annotation_col = infer_annotation_column(df, args.protein_id_column)
        matrix, annotation = wide_protein_rows_to_sample_by_protein(
            df, annotation_col, metadata_sample_ids, logger
        )
    elif role == "sample_rows":
        sample_col = args.sample_id_column or first_matching_column(
            df.columns, SAMPLE_ID_ALIASES
        )
        if not sample_col:
            sample_col = str(df.columns[0])
        matrix, annotation = wide_sample_rows_to_sample_by_protein(
            df, sample_col, logger
        )
    else:
        raise ValueError(f"Unsupported matrix orientation: {role}")

    matrix = matrix.replace([np.inf, -np.inf], np.nan)
    matrix = matrix.loc[:, matrix.notna().sum(axis=0) >= max(2, int(0.10 * len(matrix)))]
    return matrix, annotation, role


def enrich_protein_annotation(
    annotation: pd.DataFrame,
    annotation_paths: list[Path],
    args: argparse.Namespace,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Append project gene symbols/mnemonics to matrix feature annotations."""
    enriched = annotation.copy()
    label_col = first_matching_column(
        enriched.columns, ["primary_label", "protein_label", "source_column"]
    )
    text_col = first_matching_column(
        enriched.columns, ["annotation_text", "description", "source_column"]
    )
    if not label_col:
        raise ValueError("Protein annotation table has no feature-label column.")
    if not text_col:
        enriched["annotation_text"] = enriched[label_col].astype(str)
        text_col = "annotation_text"

    enriched[label_col] = enriched[label_col].astype(str).str.strip()
    enriched[text_col] = enriched[text_col].astype(str)
    audit_rows = []

    for annotation_path in annotation_paths:
        annotation_path = annotation_path.resolve()
        if not annotation_path.exists():
            raise FileNotFoundError(annotation_path)
        source = read_table(annotation_path)
        id_col = args.annotation_id_column or first_matching_column(
            source.columns,
            ["protein_row_id", "protein_id", "feature_id", "row_id", "primary_label"],
        )
        gene_col = args.annotation_gene_column or first_matching_column(
            source.columns,
            [
                "gene_symbol_curated", "protein_gene_like", "gene_symbol",
                "gene_like", "mnemonic", "symbol", "gene"
            ],
        )
        accession_col = first_matching_column(
            source.columns,
            ["protein_accession_like", "uniprot_id", "uniprot", "accession"],
        )
        if not id_col or not gene_col:
            audit_rows.append({
                "annotation_path": str(annotation_path),
                "status": "skipped",
                "id_column": id_col,
                "gene_column": gene_col,
                "n_source_rows": len(source),
                "n_mapped_features": 0,
                "reason": "required ID or gene-symbol column not found",
            })
            continue

        keep_cols = [id_col, gene_col] + ([accession_col] if accession_col else [])
        mapping = source[keep_cols].copy()
        mapping[id_col] = mapping[id_col].astype(str).str.strip()
        mapping[gene_col] = mapping[gene_col].astype(str).str.strip().str.upper()
        mapping = mapping[
            mapping[id_col].ne("")
            & mapping[gene_col].ne("")
            & mapping[gene_col].ne("NAN")
        ].drop_duplicates(subset=[id_col], keep="first")

        gene_map = dict(zip(mapping[id_col], mapping[gene_col]))
        mapped_gene = enriched[label_col].map(gene_map)
        n_mapped = int(mapped_gene.notna().sum())
        source_tag = normalize_colname(annotation_path.stem)
        out_col = f"mapped_gene__{source_tag}"
        enriched[out_col] = mapped_gene
        enriched[text_col] = (
            enriched[text_col].fillna("").astype(str)
            + mapped_gene.map(lambda x: f" | mapped_gene={x}" if pd.notna(x) else "")
        )

        if accession_col:
            accession_map = dict(zip(mapping[id_col], mapping[accession_col].astype(str)))
            mapped_accession = enriched[label_col].map(accession_map)
            acc_col = f"mapped_accession__{source_tag}"
            enriched[acc_col] = mapped_accession
            enriched[text_col] = (
                enriched[text_col].fillna("").astype(str)
                + mapped_accession.map(
                    lambda x: f" | mapped_accession={x}" if pd.notna(x) else ""
                )
            )

        audit_rows.append({
            "annotation_path": str(annotation_path),
            "status": "used",
            "id_column": id_col,
            "gene_column": gene_col,
            "n_source_rows": len(source),
            "n_unique_mapping_rows": len(mapping),
            "n_matrix_features": len(enriched),
            "n_mapped_features": n_mapped,
            "mapping_fraction": n_mapped / len(enriched) if len(enriched) else np.nan,
            "reason": "",
        })
        logger.info(
            "Protein annotation mapping %s: %d/%d matrix features mapped via %s -> %s.",
            annotation_path.name, n_mapped, len(enriched), id_col, gene_col
        )

    audit_columns = [
        "annotation_path", "status", "id_column", "gene_column",
        "n_source_rows", "n_unique_mapping_rows", "n_matrix_features",
        "n_mapped_features", "mapping_fraction", "reason"
    ]
    return enriched, pd.DataFrame(audit_rows, columns=audit_columns)


def build_feature_token_map(
    matrix_columns: Iterable[str],
    annotation: pd.DataFrame,
) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = {}
    annotation_by_label: dict[str, str] = {}
    if not annotation.empty:
        label_col = first_matching_column(
            annotation.columns, ["primary_label", "protein_label", "symbol"]
        )
        text_col = first_matching_column(
            annotation.columns, ["annotation_text", "description", "source_column"]
        )
        if label_col and text_col:
            annotation_by_label = {
                str(row[label_col]): str(row[text_col])
                for _, row in annotation[[label_col, text_col]].dropna().iterrows()
            }
    for col in matrix_columns:
        text = str(col) + " " + annotation_by_label.get(str(col), "")
        mapping[str(col)] = symbol_tokens(text)
    return mapping


def match_module_features(
    matrix: pd.DataFrame,
    annotation: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    token_map = build_feature_token_map(matrix.columns, annotation)
    rows = []
    matched: dict[str, list[str]] = {}
    for module_id, definition in MODULES.items():
        targets = {x.upper() for x in definition["proteins"]}
        detected_features = []
        detected_symbols = set()
        feature_symbol_pairs = []
        for feature, tokens in token_map.items():
            overlap = sorted(targets & tokens)
            if overlap:
                detected_features.append(feature)
                detected_symbols.update(overlap)
                feature_symbol_pairs.append(f"{feature}:{','.join(overlap)}")
        matched[module_id] = sorted(set(detected_features))
        total = len(targets)
        detected = len(detected_symbols)
        rows.append({
            "module_id": module_id,
            "module_label": definition["label"],
            "domain": definition["domain"],
            "expected_direction_in_recovery": definition["expected_direction_in_recovery"],
            "n_dictionary_proteins": total,
            "n_detected_dictionary_proteins": detected,
            "coverage_fraction": detected / total if total else np.nan,
            "n_matched_matrix_features": len(set(detected_features)),
            "feasibility_class": classify_coverage(detected, total),
            "detected_symbols": ";".join(sorted(detected_symbols)),
            "matched_features": ";".join(sorted(set(detected_features))),
            "feature_symbol_pairs": ";".join(feature_symbol_pairs),
            "interpretation_notes": definition["notes"],
        })
    return pd.DataFrame(rows), matched


def zscore_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)
    for col in df.columns:
        x = pd.to_numeric(df[col], errors="coerce")
        sd = x.std(ddof=0)
        if pd.isna(sd) or sd == 0:
            result[col] = np.nan
        else:
            result[col] = (x - x.mean()) / sd
    return result


def score_modules(
    matrix: pd.DataFrame,
    coverage: pd.DataFrame,
    matches: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_df = pd.DataFrame(index=matrix.index)
    membership_rows = []
    coverage_index = coverage.set_index("module_id")
    zmat = zscore_columns(matrix)
    for module_id, features in matches.items():
        feasibility = coverage_index.loc[module_id, "feasibility_class"]
        usable = feasibility in {
            "candidate quantitative module", "limited proxy module"
        } and len(features) >= MIN_SCORE_PROTEINS
        for feature in features:
            membership_rows.append({
                "module_id": module_id,
                "module_label": MODULES[module_id]["label"],
                "matrix_feature": feature,
                "used_in_score": bool(usable),
            })
        if usable:
            score_df[module_id] = zmat[features].mean(axis=1, skipna=True)
    score_df.index.name = "sample_id"
    return score_df, pd.DataFrame(membership_rows)


def import_existing_module_scores(
    root: Path,
    output_root: Path,
    sample_ids: set[str],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".tsv", ".txt", ".xlsx"}:
            continue
        if output_root in path.parents:
            continue
        lower = str(path).lower()
        if "phase_v0_8" in lower:
            continue
        if not any(keyword in lower for keyword in EXISTING_MODULE_KEYWORDS):
            continue
        if any(token in lower for token in ["manuscript", "citation", "reference"]):
            continue
        candidates.append(path)

    imported = pd.DataFrame(index=sorted(sample_ids))
    audit_rows = []
    used_names = set()
    for path in sorted(candidates)[:100]:
        try:
            df = read_table(path, nrows=None)
            sample_col = first_matching_column(df.columns, SAMPLE_ID_ALIASES)
            if not sample_col:
                audit_rows.append({
                    "path": str(path), "status": "skipped",
                    "reason": "no sample ID column", "n_imported_columns": 0
                })
                continue
            temp = df.copy()
            temp[sample_col] = temp[sample_col].astype(str).str.strip()
            overlap = set(temp[sample_col]) & sample_ids
            if len(overlap) < max(3, int(0.30 * len(sample_ids))):
                audit_rows.append({
                    "path": str(path), "status": "skipped",
                    "reason": f"insufficient sample overlap ({len(overlap)})",
                    "n_imported_columns": 0
                })
                continue
            numeric_cols = [
                c for c in temp.columns
                if c != sample_col and numeric_fraction(temp[c]) >= 0.70
                and any(k in normalize_colname(c) for k in EXISTING_MODULE_KEYWORDS)
            ]
            if not numeric_cols:
                audit_rows.append({
                    "path": str(path), "status": "skipped",
                    "reason": "no compatible module-score columns",
                    "n_imported_columns": 0
                })
                continue
            temp = temp[[sample_col] + numeric_cols].drop_duplicates(sample_col)
            temp = temp.set_index(sample_col)
            rename = {}
            for col in numeric_cols:
                base = "existing__" + normalize_colname(col)
                name = base
                counter = 2
                while name in used_names:
                    name = f"{base}_{counter}"
                    counter += 1
                used_names.add(name)
                rename[col] = name
            temp = temp.rename(columns=rename)
            imported = imported.join(temp, how="left")
            audit_rows.append({
                "path": str(path), "status": "imported", "reason": "",
                "n_imported_columns": len(numeric_cols),
                "imported_columns": ";".join(rename.values()),
            })
        except Exception as exc:
            audit_rows.append({
                "path": str(path), "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}", "n_imported_columns": 0
            })
    if imported.shape[1]:
        logger.info("Imported %d compatible existing module-score columns.", imported.shape[1])
    else:
        logger.info("No compatible existing module-score columns were imported.")
    imported.index.name = "sample_id"
    audit_columns = [
        "path", "status", "reason", "n_imported_columns", "imported_columns"
    ]
    return imported, pd.DataFrame(audit_rows, columns=audit_columns)


def baseline_group_tests(
    analysis: pd.DataFrame,
    score_cols: list[str],
    baseline: Any,
) -> pd.DataFrame:
    rows = []
    base = analysis[analysis["timepoint"] == baseline].copy()
    groups = ["BMI<18.5", "BMI>=18.5"]
    for score in score_cols:
        x = pd.to_numeric(
            base.loc[base["bmi_group"] == groups[0], score], errors="coerce"
        ).dropna()
        y = pd.to_numeric(
            base.loc[base["bmi_group"] == groups[1], score], errors="coerce"
        ).dropna()
        if len(x) >= MIN_GROUP_N and len(y) >= MIN_GROUP_N:
            t, p = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
            u, p_u = stats.mannwhitneyu(x, y, alternative="two-sided")
        else:
            t = p = u = p_u = np.nan
        rows.append({
            "test_family": "baseline_bmi_group_comparison",
            "module": score,
            "baseline_timepoint": baseline,
            "group_1": groups[0],
            "group_2": groups[1],
            "n_group_1": len(x),
            "n_group_2": len(y),
            "mean_group_1": x.mean() if len(x) else np.nan,
            "mean_group_2": y.mean() if len(y) else np.nan,
            "difference_group_1_minus_group_2": (
                x.mean() - y.mean() if len(x) and len(y) else np.nan
            ),
            "cohen_d": cohen_d(x, y),
            "welch_t": t,
            "p_value": p,
            "mann_whitney_u": u,
            "mann_whitney_p": p_u,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["fdr_bh"] = bh_fdr(out["p_value"])
        out["mann_whitney_fdr_bh"] = bh_fdr(out["mann_whitney_p"])
    return out


def within_group_change_tests(
    analysis: pd.DataFrame,
    score_cols: list[str],
    baseline: Any,
    ordered_timepoints: list[Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    deltas = []
    for group in ["BMI<18.5", "BMI>=18.5"]:
        group_df = analysis[analysis["bmi_group"] == group]
        for timepoint in ordered_timepoints:
            if timepoint == baseline:
                continue
            for score in score_cols:
                base = group_df[group_df["timepoint"] == baseline][
                    ["subject_id", score]
                ].dropna()
                post = group_df[group_df["timepoint"] == timepoint][
                    ["subject_id", score]
                ].dropna()
                merged = base.merge(post, on="subject_id", suffixes=("_baseline", "_post"))
                paired = len(merged) >= MIN_GROUP_N
                if paired:
                    delta = merged[f"{score}_post"] - merged[f"{score}_baseline"]
                    t, p = stats.ttest_rel(
                        merged[f"{score}_post"], merged[f"{score}_baseline"],
                        nan_policy="omit"
                    )
                    effect = paired_d(delta)
                    test_type = "paired_t"
                    for _, row in merged.iterrows():
                        deltas.append({
                            "subject_id": row["subject_id"],
                            "bmi_group": group,
                            "timepoint": timepoint,
                            "module": score,
                            "baseline_value": row[f"{score}_baseline"],
                            "post_value": row[f"{score}_post"],
                            "delta": row[f"{score}_post"] - row[f"{score}_baseline"],
                        })
                    n_base = len(merged)
                    n_post = len(merged)
                    mean_base = merged[f"{score}_baseline"].mean()
                    mean_post = merged[f"{score}_post"].mean()
                else:
                    x = pd.to_numeric(base[score], errors="coerce").dropna()
                    y = pd.to_numeric(post[score], errors="coerce").dropna()
                    if len(x) >= MIN_GROUP_N and len(y) >= MIN_GROUP_N:
                        t, p = stats.ttest_ind(y, x, equal_var=False, nan_policy="omit")
                    else:
                        t = p = np.nan
                    effect = cohen_d(y, x)
                    test_type = "unpaired_welch_t"
                    n_base, n_post = len(x), len(y)
                    mean_base = x.mean() if len(x) else np.nan
                    mean_post = y.mean() if len(y) else np.nan
                rows.append({
                    "test_family": "within_bmi_group_change_from_baseline",
                    "module": score,
                    "bmi_group": group,
                    "baseline_timepoint": baseline,
                    "post_timepoint": timepoint,
                    "test_type": test_type,
                    "n_baseline": n_base,
                    "n_post": n_post,
                    "mean_baseline": mean_base,
                    "mean_post": mean_post,
                    "mean_change": (
                        mean_post - mean_base
                        if pd.notna(mean_base) and pd.notna(mean_post) else np.nan
                    ),
                    "effect_size": effect,
                    "test_statistic": t,
                    "p_value": p,
                })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["fdr_bh"] = bh_fdr(out["p_value"])
    return out, pd.DataFrame(deltas)


def between_group_delta_tests(delta_long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if delta_long.empty:
        return pd.DataFrame()
    for (timepoint, module), subset in delta_long.groupby(["timepoint", "module"]):
        x = pd.to_numeric(
            subset.loc[subset["bmi_group"] == "BMI<18.5", "delta"], errors="coerce"
        ).dropna()
        y = pd.to_numeric(
            subset.loc[subset["bmi_group"] == "BMI>=18.5", "delta"], errors="coerce"
        ).dropna()
        if len(x) >= MIN_GROUP_N and len(y) >= MIN_GROUP_N:
            t, p = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
        else:
            t = p = np.nan
        rows.append({
            "test_family": "between_bmi_delta_comparison",
            "module": module,
            "timepoint": timepoint,
            "n_bmi_lt_18_5": len(x),
            "n_bmi_ge_18_5": len(y),
            "mean_delta_bmi_lt_18_5": x.mean() if len(x) else np.nan,
            "mean_delta_bmi_ge_18_5": y.mean() if len(y) else np.nan,
            "delta_difference": x.mean() - y.mean() if len(x) and len(y) else np.nan,
            "cohen_d": cohen_d(x, y),
            "welch_t": t,
            "p_value": p,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["fdr_bh"] = bh_fdr(out["p_value"])
    return out


def fit_interaction_models(
    analysis: pd.DataFrame,
    score_cols: list[str],
    continuous_bmi: bool,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not HAVE_STATSMODELS:
        logger.warning(
            "statsmodels is unavailable; longitudinal interaction models were skipped."
        )
        return pd.DataFrame(), pd.DataFrame()

    group_rows = []
    continuous_rows = []
    for score in score_cols:
        cols = ["sample_id", "subject_id", "timepoint", "bmi_group", "bmi", score]
        dat = analysis[cols].copy()
        dat = dat.rename(columns={score: "outcome"})
        dat["outcome"] = pd.to_numeric(dat["outcome"], errors="coerce")
        dat = dat.dropna(subset=["outcome", "timepoint", "subject_id"])

        if dat["bmi_group"].notna().sum() >= 2 * MIN_GROUP_N and dat["timepoint"].nunique() >= 2:
            formula = "outcome ~ C(bmi_group) * C(timepoint)"
            fit_type = ""
            try:
                with pywarnings.catch_warnings():
                    pywarnings.simplefilter("ignore")
                    model = smf.mixedlm(formula, dat, groups=dat["subject_id"])
                    fit = model.fit(reml=False, method="lbfgs", maxiter=500, disp=False)
                if hasattr(fit, "converged") and not fit.converged:
                    raise RuntimeError("mixed model did not converge")
                fit_type = "mixedlm_random_intercept"
            except Exception:
                try:
                    fit = smf.ols(formula, dat).fit(
                        cov_type="cluster", cov_kwds={"groups": dat["subject_id"]}
                    )
                    fit_type = "ols_cluster_robust_subject"
                except Exception:
                    fit = smf.ols(formula, dat).fit(cov_type="HC3")
                    fit_type = "ols_hc3"
            for term in fit.params.index:
                if ":" in term and "bmi_group" in term and "timepoint" in term:
                    group_rows.append({
                        "test_family": "longitudinal_bmi_group_by_time",
                        "module": score,
                        "fit_type": fit_type,
                        "n_observations": int(getattr(fit, "nobs", len(dat))),
                        "n_subjects": int(dat["subject_id"].nunique()),
                        "term": term,
                        "estimate": float(fit.params[term]),
                        "std_error": float(fit.bse[term]),
                        "test_statistic": float(fit.tvalues[term]),
                        "p_value": float(fit.pvalues[term]),
                    })

        if continuous_bmi:
            cdat = dat.dropna(subset=["bmi"]).copy()
            if cdat["bmi"].nunique() >= 5 and cdat["timepoint"].nunique() >= 2:
                formula = "outcome ~ bmi * C(timepoint)"
                fit_type = ""
                try:
                    with pywarnings.catch_warnings():
                        pywarnings.simplefilter("ignore")
                        model = smf.mixedlm(formula, cdat, groups=cdat["subject_id"])
                        fit = model.fit(reml=False, method="lbfgs", maxiter=500, disp=False)
                    if hasattr(fit, "converged") and not fit.converged:
                        raise RuntimeError("mixed model did not converge")
                    fit_type = "mixedlm_random_intercept"
                except Exception:
                    try:
                        fit = smf.ols(formula, cdat).fit(
                            cov_type="cluster",
                            cov_kwds={"groups": cdat["subject_id"]}
                        )
                        fit_type = "ols_cluster_robust_subject"
                    except Exception:
                        fit = smf.ols(formula, cdat).fit(cov_type="HC3")
                        fit_type = "ols_hc3"
                for term in fit.params.index:
                    if ":" in term and "bmi" in term and "timepoint" in term:
                        continuous_rows.append({
                            "test_family": "continuous_bmi_by_time",
                            "module": score,
                            "fit_type": fit_type,
                            "n_observations": int(getattr(fit, "nobs", len(cdat))),
                            "n_subjects": int(cdat["subject_id"].nunique()),
                            "term": term,
                            "estimate": float(fit.params[term]),
                            "std_error": float(fit.bse[term]),
                            "test_statistic": float(fit.tvalues[term]),
                            "p_value": float(fit.pvalues[term]),
                        })

    group_out = pd.DataFrame(group_rows)
    continuous_out = pd.DataFrame(continuous_rows)
    if not group_out.empty:
        group_out["fdr_bh"] = bh_fdr(group_out["p_value"])
    if not continuous_out.empty:
        continuous_out["fdr_bh"] = bh_fdr(continuous_out["p_value"])
    return group_out, continuous_out


def build_integrated_score(
    analysis: pd.DataFrame,
    module_cols: list[str],
    coverage: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    favorable_ids = [
        module_id for module_id in module_cols
        if MODULES.get(module_id, {}).get("domain") == "nutritional_metabolic"
        and MODULES.get(module_id, {}).get("expected_direction_in_recovery") == "increase"
    ]
    # Require at least three modules spanning at least two biological families.
    selected = [c for c in favorable_ids if analysis[c].notna().sum() >= MIN_GROUP_N]
    decision_rows = []
    for module_id in module_cols:
        reason = ""
        selected_flag = module_id in selected
        if module_id not in MODULES:
            reason = "existing module score; excluded from first-pass integrated score"
        elif MODULES[module_id]["domain"] != "nutritional_metabolic":
            reason = "non-nutritional domain"
        elif MODULES[module_id]["expected_direction_in_recovery"] != "increase":
            reason = "direction context-dependent"
        elif analysis[module_id].notna().sum() < MIN_GROUP_N:
            reason = "insufficient non-missing values"
        else:
            reason = "selected as recovery-favorable nutritional module"
        decision_rows.append({
            "module_id": module_id,
            "selected_for_integrated_score": selected_flag,
            "sign_in_undernutrition_burden_score": -1 if selected_flag else np.nan,
            "reason": reason,
        })

    out = analysis.copy()
    if len(selected) >= 3:
        z = zscore_columns(out[selected])
        # Higher value = greater metabolic-undernutrition burden.
        out["integrated_metabolic_undernutrition_score"] = -z.mean(axis=1, skipna=True)
        out["integrated_score_n_modules_available"] = z.notna().sum(axis=1)
        out.loc[
            out["integrated_score_n_modules_available"] < 2,
            "integrated_metabolic_undernutrition_score"
        ] = np.nan
    return out, pd.DataFrame(decision_rows)


def module_correlations(
    analysis: pd.DataFrame,
    module_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    cols = [c for c in module_cols if c in analysis.columns]
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            rho, p, n = safe_spearman(analysis[a], analysis[b])
            rows.append({
                "module_a": a,
                "module_b": b,
                "n_complete": n,
                "spearman_rho": rho,
                "p_value": p,
            })
    long = pd.DataFrame(rows)
    if not long.empty:
        long["fdr_bh"] = bh_fdr(long["p_value"])
    matrix = analysis[cols].corr(method="spearman") if cols else pd.DataFrame()
    return long, matrix


def make_trajectory_summary(
    analysis: pd.DataFrame,
    score_cols: list[str],
) -> pd.DataFrame:
    rows = []
    for (timepoint, group), subset in analysis.groupby(["timepoint", "bmi_group"], dropna=False):
        for score in score_cols:
            vals = pd.to_numeric(subset[score], errors="coerce").dropna()
            rows.append({
                "timepoint": timepoint,
                "bmi_group": group,
                "module": score,
                "n": len(vals),
                "mean": vals.mean() if len(vals) else np.nan,
                "sd": vals.std(ddof=1) if len(vals) > 1 else np.nan,
                "sem": vals.sem(ddof=1) if len(vals) > 1 else np.nan,
                "median": vals.median() if len(vals) else np.nan,
                "q1": vals.quantile(0.25) if len(vals) else np.nan,
                "q3": vals.quantile(0.75) if len(vals) else np.nan,
            })
    return pd.DataFrame(rows)


def plot_coverage(coverage: pd.DataFrame, path_png: Path, path_svg: Path) -> None:
    if not HAVE_MATPLOTLIB or coverage.empty:
        return
    plot_df = coverage.sort_values("coverage_fraction", ascending=True)
    fig_h = max(6, 0.38 * len(plot_df))
    fig, ax = plt.subplots(figsize=(10, fig_h))
    y = np.arange(len(plot_df))
    ax.barh(y, plot_df["coverage_fraction"] * 100)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["module_label"], fontsize=8)
    ax.set_xlabel("Dictionary proteins detected (%)")
    ax.set_title("Phase v0.8 nutritional-metabolic module coverage")
    for i, (_, row) in enumerate(plot_df.iterrows()):
        ax.text(
            row["coverage_fraction"] * 100 + 0.5,
            i,
            f'{int(row["n_detected_dictionary_proteins"])}/{int(row["n_dictionary_proteins"])}',
            va="center",
            fontsize=7,
        )
    fig.tight_layout()
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    fig.savefig(path_svg, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(
    corr: pd.DataFrame, path_png: Path, path_svg: Path
) -> None:
    if not HAVE_MATPLOTLIB or corr.empty or corr.shape[0] < 2:
        return
    n = corr.shape[0]
    fig_size = max(8, min(18, 0.55 * n + 4))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(corr.to_numpy(float), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.index, fontsize=7)
    ax.set_title("Spearman correlations among nutritional-metabolic and existing modules")
    fig.colorbar(im, ax=ax, label="Spearman rho", fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    fig.savefig(path_svg, bbox_inches="tight")
    plt.close(fig)


def plot_integrated_trajectory(
    analysis: pd.DataFrame,
    ordered_timepoints: list[Any],
    path_png: Path,
    path_svg: Path,
) -> None:
    score = "integrated_metabolic_undernutrition_score"
    if not HAVE_MATPLOTLIB or score not in analysis.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for group in ["BMI<18.5", "BMI>=18.5"]:
        sub = analysis[analysis["bmi_group"] == group]
        means = []
        sems = []
        for t in ordered_timepoints:
            vals = pd.to_numeric(
                sub.loc[sub["timepoint"] == t, score], errors="coerce"
            ).dropna()
            means.append(vals.mean() if len(vals) else np.nan)
            sems.append(vals.sem(ddof=1) if len(vals) > 1 else np.nan)
        x = np.arange(len(ordered_timepoints))
        ax.errorbar(x, means, yerr=sems, marker="o", capsize=3, label=group)
    ax.set_xticks(np.arange(len(ordered_timepoints)))
    ax.set_xticklabels([str(x) for x in ordered_timepoints], rotation=30, ha="right")
    ax.set_ylabel("Integrated metabolic-undernutrition burden score")
    ax.set_xlabel("Treatment timepoint")
    ax.set_title("Integrated metabolic-undernutrition score trajectory")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    fig.savefig(path_svg, bbox_inches="tight")
    plt.close(fig)


def export_single_cell_scaffolds(table_dir: Path) -> None:
    rows = []
    for module_id, definition in MODULES.items():
        for gene in definition["proteins"]:
            rows.append({
                "module_id": module_id,
                "module_label": definition["label"],
                "domain": definition["domain"],
                "gene_symbol": gene,
                "serum_detected": "",
                "single_cell_dataset": "",
                "cell_type": "",
                "detectable": "",
                "mean_expression": "",
                "fraction_expressing": "",
                "enrichment_statistic": "",
                "notes": "",
            })
    write_tsv(
        pd.DataFrame(rows),
        table_dir / "phase_v0_8_single_cell_gene_by_cell_type_detectability_template.tsv"
    )

    inventory = pd.DataFrame([
        {
            "dataset_id": "",
            "repository_accession": "",
            "publication": "",
            "country_or_setting": "",
            "disease_context": "active pulmonary TB / treatment / control",
            "assay": "single-cell RNA-seq / single-nucleus RNA-seq / spatial transcriptomics",
            "tissue": "",
            "n_samples": "",
            "n_cells_or_spots": "",
            "available_cell_types": (
                "monocytes/macrophages; dendritic cells; neutrophils; T/NK cells; "
                "epithelial cells; stromal/vascular compartments"
            ),
            "raw_or_processed_access": "",
            "priority": "",
            "eligibility_decision": "",
            "notes": "",
        }
    ])
    write_tsv(
        inventory,
        table_dir / "phase_v0_8_single_cell_spatial_dataset_inventory_template.tsv"
    )

    model = pd.DataFrame([
        {
            "serum_module": module_id,
            "serum_interpretation": definition["notes"],
            "priority_cell_types": (
                "monocytes/macrophages; dendritic cells; neutrophils; T/NK cells; "
                "epithelial cells; stromal/vascular compartments"
            ),
            "single_cell_validation_question": (
                "Are module genes detectably and preferentially expressed in biologically "
                "plausible TB-relevant cell compartments?"
            ),
            "prohibited_inference": (
                "Do not equate cell-type transcript enrichment with serum protein abundance "
                "or direct nutrient concentration."
            ),
        }
        for module_id, definition in MODULES.items()
    ])
    write_tsv(
        model,
        table_dir / "phase_v0_8_integrated_serum_single_cell_model_scaffold.tsv"
    )


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 30) -> str:
    if df.empty:
        return "_No rows available._"
    temp = df[columns].head(max_rows).copy()
    try:
        return temp.to_markdown(index=False)
    except Exception:
        return "```\n" + temp.to_csv(index=False) + "```"


def write_reports(
    report_path: Path,
    decision_log_path: Path,
    args: argparse.Namespace,
    inputs: dict[str, Any],
    coverage: pd.DataFrame,
    analysis: Optional[pd.DataFrame],
    stats_outputs: dict[str, pd.DataFrame],
    warnings: list[str],
    output_dirs: dict[str, Path],
) -> None:
    quantitative = coverage[
        coverage["feasibility_class"] == "candidate quantitative module"
    ]
    limited = coverage[
        coverage["feasibility_class"] == "limited proxy module"
    ]
    sparse = coverage[
        coverage["feasibility_class"] == "sparse proxy only"
    ]
    absent = coverage[
        coverage["feasibility_class"] == "not detected"
    ]
    significant_summary = []
    for name, df in stats_outputs.items():
        if not df.empty and "fdr_bh" in df.columns:
            significant_summary.append({
                "analysis": name,
                "n_tests": len(df),
                "n_fdr_lt_0_05": int((pd.to_numeric(df["fdr_bh"], errors="coerce") < 0.05).sum()),
            })

    report = f"""# Phase v0.8 nutritional-metabolic expansion report

Generated: {now_utc()}

## Baseline preservation

The v0.7.1 full co-author review manuscript remains the preserved baseline.
This Phase v0.8 run writes only to new `{PHASE_TAG}` directories and does not
overwrite the v0.7.1 manuscript, figures, tables, or decision logs.

## Input resolution

- Protein matrix: `{inputs.get("protein_matrix")}`
- Protein sheet/orientation: `{inputs.get("protein_sheet_or_orientation")}`
- Metadata: `{inputs.get("metadata")}`
- Samples in merged analysis: `{0 if analysis is None else len(analysis)}`
- Subjects: `{0 if analysis is None else analysis["subject_id"].nunique()}`
- Timepoints: `{"" if analysis is None else "; ".join(map(str, pd.unique(analysis["timepoint"].dropna())))}`
- BMI groups: `{"" if analysis is None else "; ".join(map(str, pd.unique(analysis["bmi_group"].dropna())))}`

## Coverage classification

- Candidate quantitative modules: {len(quantitative)}
- Limited proxy modules: {len(limited)}
- Sparse proxy only: {len(sparse)}
- Not detected: {len(absent)}

{markdown_table(coverage, ["module_label", "n_detected_dictionary_proteins", "n_dictionary_proteins", "coverage_fraction", "feasibility_class"], 50)}

## Statistical analyses

The script attempts:

1. Baseline BMI-group comparisons.
2. Within-group longitudinal changes from baseline.
3. Between-BMI delta comparisons.
4. Longitudinal BMI group × time models.
5. Continuous BMI × time models where continuous BMI is available.
6. BH-FDR correction within each test family.
7. Spearman correlation/network tables among new and compatible existing modules.
8. An integrated metabolic-undernutrition burden score when at least three
   recovery-favorable nutritional modules are scoreable.

### FDR summary

{markdown_table(pd.DataFrame(significant_summary), ["analysis", "n_tests", "n_fdr_lt_0_05"], 50) if significant_summary else "_No completed FDR-bearing test tables._"}

## Interpretation boundaries

- Serum proteomics does not directly measure vitamin D, vitamin A, vitamin C,
  B-vitamin, calcium, or other nutrient concentrations.
- Serum proteomics does not directly measure intracellular calcium flux.
- Serum proteomics does not directly measure phagocytosis efficiency,
  phagolysosome function, autophagic flux, or Mtb killing.
- Cytoskeletal, contractile, troponin, tropomyosin, myosin-light-chain,
  caldesmon, calmodulin, and related serum proteins may reflect secretion,
  extracellular vesicles, tissue injury, vascular remodelling, platelet
  biology, or cell turnover.
- Single-cell/spatial triangulation should validate cell-type plausibility of
  module genes, not serum protein abundance.

## Output locations

- Tables: `{output_dirs["tables"]}`
- Results/report: `{output_dirs["results"]}`
- Draft figures: `{output_dirs["figures"]}`
- Decision logs: `{output_dirs["logs"]}`

## Warnings and unresolved decisions

{chr(10).join("- " + x for x in warnings) if warnings else "- None recorded."}
"""
    report_path.write_text(report, encoding="utf-8")

    decision = f"""# Phase v0.8 decision log

Generated: {now_utc()}

## Immutable baseline

- Preserve v0.7.1 as the author-review baseline.
- Do not overwrite or silently edit v0.7.1 files.
- Treat Phase v0.8 as an additive biological and statistical expansion.

## Module-scoring decisions

- Candidate quantitative module: at least {MIN_QUANTITATIVE_PROTEINS} detected
  dictionary proteins and at least 20% dictionary coverage.
- Limited proxy module: at least {MIN_LIMITED_PROXY_PROTEINS} detected proteins.
- Sparse proxy only: one or two detected proteins; no composite score.
- Not detected: zero detected proteins.
- Composite module scores use the mean of protein-wise z-scores.
- A first-pass integrated metabolic-undernutrition burden score is created only
  when at least three recovery-favorable nutritional modules are available.
- Higher integrated score means greater relative metabolic-undernutrition burden;
  the score is internally standardized and is not a clinical diagnostic measure.

## Statistical decisions

- Baseline BMI comparison: Welch t-test, with Mann–Whitney sensitivity test.
- Within-group change: paired t-test when subject pairing is available;
  otherwise Welch t-test is reported explicitly as unpaired.
- Between-group delta comparison: Welch t-test on subject-level change values.
- Longitudinal interaction: random-intercept mixed model when estimable;
  cluster-robust or HC3 OLS fallback otherwise.
- Continuous BMI × time models are run only when sufficiently variable BMI is present.
- BH-FDR is applied within each related test family.

## Interpretation boundaries

- Protein/gene module proxy analysis only.
- No direct nutrient-concentration inference.
- No direct calcium-flux inference.
- No direct autophagic-flux, phagocytosis, intracellular killing, or Mtb-clearance inference.
- Single-cell/spatial follow-up validates cell-type plausibility, not serum abundance.

## Input decisions

```json
{json.dumps(inputs, indent=2, default=str)}
```

## Warnings

{chr(10).join("- " + x for x in warnings) if warnings else "- None recorded."}
"""
    decision_log_path.write_text(decision, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase v0.8 TB–undernutrition nutritional-metabolic expansion."
    )
    parser.add_argument("--project-root", type=Path, default=EXPECTED_ROOT)
    parser.add_argument("--protein-matrix", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument(
        "--protein-annotation",
        type=Path,
        action="append",
        default=[],
        help=(
            "Protein-row-to-gene mapping table. May be repeated; later tables "
            "supplement the annotation text without replacing the matrix IDs."
        ),
    )
    parser.add_argument("--annotation-id-column", default=None)
    parser.add_argument("--annotation-gene-column", default=None)
    parser.add_argument("--protein-sheet", default=None)
    parser.add_argument("--metadata-sheet", default=None)
    parser.add_argument(
        "--matrix-orientation",
        choices=["auto", "protein_rows", "sample_rows", "long"],
        default="auto",
    )
    parser.add_argument("--protein-id-column", default=None)
    parser.add_argument("--value-column", default=None)
    parser.add_argument("--sample-id-column", default=None)
    parser.add_argument("--subject-id-column", default=None)
    parser.add_argument("--bmi-column", default=None)
    parser.add_argument("--bmi-group-column", default=None)
    parser.add_argument("--time-column", default=None)
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help="Only scan and inspect candidate input files.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()

    if root != EXPECTED_ROOT:
        print(f"ERROR: Wrong project path: {root}")
        print(f"Expected: {EXPECTED_ROOT}")
        return 2
    if not root.exists():
        print(f"ERROR: Project root does not exist: {root}")
        return 2

    results_dir = root / "05_results" / PHASE_TAG
    tables_dir = root / "02_tables" / PHASE_TAG
    figures_dir = root / "03_figures" / "draft_panels" / PHASE_TAG
    logs_dir = root / "05_docs_decision_logs" / PHASE_TAG
    for directory in [results_dir, tables_dir, figures_dir, logs_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"phase_v0_8_run_{run_id}.log"
    logger = setup_logger(log_path, args.verbose)
    warnings: list[str] = []

    logger.info("Phase v0.8 started.")
    logger.info("v0.7.1 remains the preserved baseline.")
    logger.info("Project root guard passed: %s", root)

    # Export versioned module dictionary immediately.
    module_rows = []
    for module_id, definition in MODULES.items():
        for protein in definition["proteins"]:
            module_rows.append({
                "module_id": module_id,
                "module_label": definition["label"],
                "domain": definition["domain"],
                "expected_direction_in_recovery": definition["expected_direction_in_recovery"],
                "protein_gene_symbol": protein,
                "interpretation_notes": definition["notes"],
            })
    module_dictionary = pd.DataFrame(module_rows)
    write_tsv(
        module_dictionary,
        tables_dir / "phase_v0_8_nutritional_metabolic_module_dictionary.tsv"
    )
    write_json(
        {"script_version": SCRIPT_VERSION, "modules": MODULES},
        tables_dir / "phase_v0_8_nutritional_metabolic_module_dictionary.json"
    )
    export_single_cell_scaffolds(tables_dir)

    # Project inventory.
    candidates = scan_candidates(root, results_dir)
    inspected = []
    for candidate in candidates[:80]:
        inspected.append(inspect_candidate(candidate))
    inventory_df = pd.DataFrame([asdict(x) for x in inspected])
    write_tsv(
        inventory_df,
        tables_dir / "phase_v0_8_input_candidate_inventory.tsv"
    )

    if args.inventory_only:
        logger.info("Inventory-only mode completed.")
        return 0

    protein_override = args.protein_matrix.resolve() if args.protein_matrix else None
    metadata_override = args.metadata.resolve() if args.metadata else None
    annotation_overrides = [path.resolve() for path in args.protein_annotation]
    if protein_override and not protein_override.exists():
        raise FileNotFoundError(protein_override)
    if metadata_override and not metadata_override.exists():
        raise FileNotFoundError(metadata_override)
    for annotation_path in annotation_overrides:
        if not annotation_path.exists():
            raise FileNotFoundError(annotation_path)

    protein_candidate = choose_candidate(
        inspected,
        {"wide_protein_rows", "long_proteomics", "wide_sample_rows_or_scores"},
        override=protein_override,
    )

    # A workbook may open on a metadata sheet even though another sheet contains
    # the normalized abundance matrix. Search all workbook sheets before giving up.
    inferred_protein_sheet = None
    if protein_candidate is None:
        for candidate in inspected:
            candidate_path = Path(candidate.path)
            if candidate_path.suffix.lower() not in {".xlsx", ".xls"}:
                continue
            for candidate_sheet in list_excel_sheets(candidate_path):
                try:
                    preview = read_table(candidate_path, sheet=candidate_sheet, nrows=100)
                    sample_col = first_matching_column(preview.columns, SAMPLE_ID_ALIASES)
                    protein_col = first_matching_column(preview.columns, PROTEIN_ID_ALIASES)
                    value_col = first_matching_column(preview.columns, VALUE_ALIASES)
                    numeric_cols = [
                        c for c in preview.columns
                        if numeric_fraction(preview[c]) >= 0.60
                    ]
                    is_long = bool(sample_col and protein_col and value_col)
                    is_protein_rows = bool(protein_col and len(numeric_cols) >= 3)
                    is_sample_rows = bool(sample_col and len(numeric_cols) >= 10)
                    if is_long or is_protein_rows or is_sample_rows:
                        protein_candidate = InputCandidate(
                            path=str(candidate_path),
                            file_type=candidate_path.suffix.lower(),
                            size_bytes=candidate_path.stat().st_size,
                            modified_utc=datetime.fromtimestamp(
                                candidate_path.stat().st_mtime, tz=timezone.utc
                            ).isoformat(timespec="seconds"),
                            name_score=997,
                            inspection_status="workbook_sheet_search",
                            possible_role=(
                                "long_proteomics" if is_long
                                else "wide_protein_rows" if is_protein_rows
                                else "wide_sample_rows_or_scores"
                            ),
                            notes=f"protein sheet inferred as {candidate_sheet}",
                        )
                        inferred_protein_sheet = candidate_sheet
                        args.protein_sheet = args.protein_sheet or candidate_sheet
                        break
                except Exception:
                    continue
            if protein_candidate is not None:
                break

    metadata_candidate = choose_candidate(
        inspected,
        {"metadata"},
        override=metadata_override,
    )

    # If metadata is not a separate file, inspect other sheets in the selected
    # proteomics workbook for sample/BMI/time information.
    inferred_metadata_sheet = None
    if (
        metadata_candidate is None
        and protein_candidate is not None
        and Path(protein_candidate.path).suffix.lower() in {".xlsx", ".xls"}
    ):
        for candidate_sheet in list_excel_sheets(Path(protein_candidate.path)):
            if args.protein_sheet and candidate_sheet == args.protein_sheet:
                continue
            try:
                preview = read_table(
                    Path(protein_candidate.path), sheet=candidate_sheet, nrows=100
                )
                has_sample = first_matching_column(preview.columns, SAMPLE_ID_ALIASES)
                has_bmi = first_matching_column(preview.columns, BMI_ALIASES)
                has_group = first_matching_column(preview.columns, BMI_GROUP_ALIASES)
                has_time = first_matching_column(preview.columns, TIME_ALIASES)
                if has_sample and (has_bmi or has_group) and has_time:
                    metadata_candidate = InputCandidate(
                        path=protein_candidate.path,
                        file_type=Path(protein_candidate.path).suffix.lower(),
                        size_bytes=Path(protein_candidate.path).stat().st_size,
                        modified_utc=datetime.fromtimestamp(
                            Path(protein_candidate.path).stat().st_mtime,
                            tz=timezone.utc,
                        ).isoformat(timespec="seconds"),
                        name_score=998,
                        inspection_status="same_workbook_sheet",
                        possible_role="metadata",
                        notes=f"metadata sheet inferred as {candidate_sheet}",
                    )
                    inferred_metadata_sheet = candidate_sheet
                    break
            except Exception:
                continue

    inputs: dict[str, Any] = {
        "project_root": str(root),
        "script_version": SCRIPT_VERSION,
        "run_id": run_id,
        "protein_matrix": protein_candidate.path if protein_candidate else None,
        "metadata": metadata_candidate.path if metadata_candidate else None,
        "protein_annotation": [str(path) for path in annotation_overrides],
        "protein_sheet_or_orientation": (
            args.protein_sheet or inferred_protein_sheet or args.matrix_orientation
        ),
        "metadata_sheet": args.metadata_sheet or inferred_metadata_sheet,
        "user_overrides": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if value not in (None, False)
        },
    }

    if protein_candidate is None:
        warnings.append(
            "No compatible proteomics matrix was selected automatically. "
            "Re-run with --protein-matrix and, if needed, --matrix-orientation."
        )
        empty_coverage = pd.DataFrame([
            {
                "module_id": module_id,
                "module_label": d["label"],
                "domain": d["domain"],
                "expected_direction_in_recovery": d["expected_direction_in_recovery"],
                "n_dictionary_proteins": len(d["proteins"]),
                "n_detected_dictionary_proteins": 0,
                "coverage_fraction": 0.0,
                "n_matched_matrix_features": 0,
                "feasibility_class": "not assessed — no matrix selected",
                "detected_symbols": "",
                "matched_features": "",
                "feature_symbol_pairs": "",
                "interpretation_notes": d["notes"],
            }
            for module_id, d in MODULES.items()
        ])
        write_tsv(
            empty_coverage,
            tables_dir / "phase_v0_8_module_coverage_audit.tsv"
        )
        write_reports(
            results_dir / "phase_v0_8_nutritional_metabolic_expansion_report.md",
            logs_dir / "phase_v0_8_decision_log.md",
            args,
            inputs,
            empty_coverage,
            None,
            {},
            warnings,
            {
                "tables": tables_dir, "results": results_dir,
                "figures": figures_dir, "logs": logs_dir
            },
        )
        logger.warning(warnings[-1])
        return 0

    # Load metadata, if found.
    metadata = None
    metadata_mapping: dict[str, Optional[str]] = {}
    if metadata_candidate is not None:
        raw_meta = read_table(
            Path(metadata_candidate.path),
            sheet=args.metadata_sheet or inferred_metadata_sheet,
        )
        try:
            metadata, metadata_mapping = derive_metadata(raw_meta, args, logger)
        except Exception as exc:
            warnings.append(
                f"Metadata file was selected but could not be normalized: "
                f"{type(exc).__name__}: {exc}"
            )
            metadata = None
    else:
        warnings.append(
            "No separate metadata file was selected automatically. The script "
            "will attempt sample-only matrix analysis; inferential BMI/time tests "
            "will require --metadata."
        )

    metadata_sample_ids = (
        set(metadata["sample_id"]) if metadata is not None else None
    )

    matrix, annotation, orientation = load_proteomics_matrix(
        Path(protein_candidate.path),
        metadata_sample_ids,
        args,
        logger,
    )
    annotation, annotation_mapping_audit = enrich_protein_annotation(
        annotation,
        annotation_overrides,
        args,
        logger,
    )
    write_tsv(
        annotation_mapping_audit,
        tables_dir / "phase_v0_8_external_protein_annotation_mapping_audit.tsv",
    )
    inputs["resolved_matrix_orientation"] = orientation
    inputs["metadata_mapping"] = metadata_mapping
    inputs["protein_matrix_sha256"] = sha256sum(Path(protein_candidate.path))
    if metadata_candidate is not None:
        inputs["metadata_sha256"] = sha256sum(Path(metadata_candidate.path))
    inputs["protein_annotation_sha256"] = {
        str(path): sha256sum(path) for path in annotation_overrides
    }

    write_tsv(
        annotation,
        tables_dir / "phase_v0_8_protein_annotation_audit.tsv"
    )
    matrix_export = matrix.reset_index()
    try:
        matrix_export.to_parquet(
            tables_dir / "phase_v0_8_normalized_sample_by_protein_matrix.parquet",
            index=False,
        )
    except Exception:
        warnings.append(
            "Optional Parquet engine unavailable; wrote the normalized matrix as "
            "a compressed TSV file instead."
        )
        matrix_export.to_csv(
            tables_dir / "phase_v0_8_normalized_sample_by_protein_matrix.tsv.gz",
            sep="\t",
            index=False,
            compression="gzip",
        )

    coverage, matches = match_module_features(matrix, annotation)
    write_tsv(
        coverage,
        tables_dir / "phase_v0_8_module_coverage_audit.tsv"
    )
    plot_coverage(
        coverage,
        figures_dir / "phase_v0_8_module_coverage.png",
        figures_dir / "phase_v0_8_module_coverage.svg",
    )

    module_scores, membership = score_modules(matrix, coverage, matches)
    write_tsv(
        membership,
        tables_dir / "phase_v0_8_module_score_membership.tsv"
    )
    write_tsv(
        module_scores.reset_index(),
        tables_dir / "phase_v0_8_new_module_scores.tsv"
    )

    if module_scores.empty:
        warnings.append(
            "No module met the minimum criteria for a composite score. "
            "Coverage and sparse-proxy outputs were still generated."
        )

    existing_scores, existing_audit = import_existing_module_scores(
        root, results_dir, set(matrix.index), logger
    )
    write_tsv(
        existing_audit,
        tables_dir / "phase_v0_8_existing_module_score_import_audit.tsv"
    )

    combined_scores = module_scores.join(existing_scores, how="left")
    combined_scores.index.name = "sample_id"

    # Build analysis frame.
    if metadata is not None:
        analysis = metadata.merge(
            combined_scores.reset_index(), on="sample_id", how="inner"
        )
        if len(analysis) == 0:
            warnings.append(
                "Metadata and proteomics matrix had zero exact sample-ID overlap."
            )
            analysis = combined_scores.reset_index()
            analysis["subject_id"] = analysis["sample_id"]
            analysis["timepoint"] = "unknown"
            analysis["bmi"] = np.nan
            analysis["bmi_group"] = np.nan
    else:
        analysis = combined_scores.reset_index()
        analysis["subject_id"] = analysis["sample_id"]
        analysis["timepoint"] = "unknown"
        analysis["bmi"] = np.nan
        analysis["bmi_group"] = np.nan

    new_module_cols = list(module_scores.columns)
    existing_module_cols = list(existing_scores.columns)
    score_cols = new_module_cols + existing_module_cols

    analysis, integrated_decisions = build_integrated_score(
        analysis, new_module_cols, coverage
    )
    write_tsv(
        integrated_decisions,
        tables_dir / "phase_v0_8_integrated_score_decisions.tsv"
    )
    if "integrated_metabolic_undernutrition_score" in analysis.columns:
        score_cols.append("integrated_metabolic_undernutrition_score")
    else:
        warnings.append(
            "Integrated metabolic-undernutrition score was not created because "
            "fewer than three recovery-favorable nutritional modules were scoreable."
        )

    write_tsv(
        analysis,
        tables_dir / "phase_v0_8_sample_level_analysis_dataset.tsv"
    )

    stats_outputs: dict[str, pd.DataFrame] = {}
    can_test = (
        analysis["bmi_group"].isin(["BMI<18.5", "BMI>=18.5"]).sum() >= 2 * MIN_GROUP_N
        and analysis["timepoint"].notna().sum() >= MIN_GROUP_N
        and len(score_cols) > 0
    )

    baseline = identify_baseline(analysis["timepoint"])
    ordered_timepoints = order_timepoints(analysis["timepoint"], baseline)

    if can_test:
        baseline_tests = baseline_group_tests(analysis, score_cols, baseline)
        within_tests, delta_long = within_group_change_tests(
            analysis, score_cols, baseline, ordered_timepoints
        )
        delta_tests = between_group_delta_tests(delta_long)
        group_models, continuous_models = fit_interaction_models(
            analysis,
            score_cols,
            continuous_bmi=analysis["bmi"].notna().sum() >= MIN_GROUP_N,
            logger=logger,
        )
        stats_outputs = {
            "baseline_group_tests": baseline_tests,
            "within_group_change_tests": within_tests,
            "between_group_delta_tests": delta_tests,
            "longitudinal_bmi_group_by_time_models": group_models,
            "continuous_bmi_by_time_models": continuous_models,
        }
        write_tsv(
            baseline_tests,
            tables_dir / "phase_v0_8_baseline_bmi_group_comparisons.tsv"
        )
        write_tsv(
            within_tests,
            tables_dir / "phase_v0_8_within_group_longitudinal_changes.tsv"
        )
        write_tsv(
            delta_long,
            tables_dir / "phase_v0_8_subject_level_deltas.tsv"
        )
        write_tsv(
            delta_tests,
            tables_dir / "phase_v0_8_between_bmi_delta_comparisons.tsv"
        )
        write_tsv(
            group_models,
            tables_dir / "phase_v0_8_longitudinal_bmi_group_by_time_models.tsv"
        )
        write_tsv(
            continuous_models,
            tables_dir / "phase_v0_8_continuous_bmi_by_time_models.tsv"
        )
    else:
        warnings.append(
            "BMI/time inferential tests were not run because compatible metadata, "
            "both BMI strata, multiple observations, or scoreable modules were insufficient."
        )

    trajectory = make_trajectory_summary(analysis, score_cols)
    write_tsv(
        trajectory,
        tables_dir / "phase_v0_8_module_trajectory_summary.tsv"
    )

    corr_long, corr_matrix = module_correlations(analysis, score_cols)
    write_tsv(
        corr_long,
        tables_dir / "phase_v0_8_module_correlation_network_edges.tsv"
    )
    if not corr_matrix.empty:
        write_tsv(
            corr_matrix.reset_index().rename(columns={"index": "module"}),
            tables_dir / "phase_v0_8_module_spearman_correlation_matrix.tsv"
        )
        plot_correlation_heatmap(
            corr_matrix,
            figures_dir / "phase_v0_8_module_correlation_heatmap.png",
            figures_dir / "phase_v0_8_module_correlation_heatmap.svg",
        )

    plot_integrated_trajectory(
        analysis,
        ordered_timepoints,
        figures_dir / "phase_v0_8_integrated_score_trajectory.png",
        figures_dir / "phase_v0_8_integrated_score_trajectory.svg",
    )

    run_manifest = {
        "phase_tag": PHASE_TAG,
        "script_version": SCRIPT_VERSION,
        "generated_utc": now_utc(),
        "project_root": str(root),
        "baseline_preserved": "v0.7.1",
        "inputs": inputs,
        "n_matrix_samples": int(matrix.shape[0]),
        "n_matrix_features": int(matrix.shape[1]),
        "n_analysis_rows": int(len(analysis)),
        "n_subjects": int(analysis["subject_id"].nunique()),
        "n_new_module_scores": int(len(new_module_cols)),
        "n_existing_module_scores_imported": int(len(existing_module_cols)),
        "integrated_score_created": (
            "integrated_metabolic_undernutrition_score" in analysis.columns
        ),
        "warnings": warnings,
    }
    write_json(
        run_manifest,
        results_dir / "phase_v0_8_run_manifest.json"
    )

    write_reports(
        results_dir / "phase_v0_8_nutritional_metabolic_expansion_report.md",
        logs_dir / "phase_v0_8_decision_log.md",
        args,
        inputs,
        coverage,
        analysis,
        stats_outputs,
        warnings,
        {
            "tables": tables_dir, "results": results_dir,
            "figures": figures_dir, "logs": logs_dir
        },
    )

    logger.info("Phase v0.8 completed.")
    logger.info("Tables: %s", tables_dir)
    logger.info("Results: %s", results_dir)
    logger.info("Figures: %s", figures_dir)
    logger.info("Decision logs: %s", logs_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise
