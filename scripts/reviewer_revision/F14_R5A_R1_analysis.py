from __future__ import annotations

import json
import math
import platform
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.optimize import brentq
import statsmodels
import statsmodels.formula.api as smf

SEED = 20261122
N_PARALLEL = 2000

np.random.seed(SEED)

if len(sys.argv) != 10:
    raise SystemExit("HOLD_BAD_ARGUMENT_COUNT")

BASE = Path(sys.argv[1])
SAMPLE = Path(sys.argv[2])
LONG = Path(sys.argv[3])
S13 = Path(sys.argv[4])
S14 = Path(sys.argv[5])
S16 = Path(sys.argv[6])
S17 = Path(sys.argv[7])
OUT = Path(sys.argv[8])
AXIS_EXISTING = Path(sys.argv[9])

def read_tsv(p):
    return pd.read_csv(p, sep="\t", low_memory=False)

def norm_name(x):
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())

def pick(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c

    lookup = {norm_name(c): c for c in df.columns}

    for c in candidates:
        k = norm_name(c)
        if k in lookup:
            return lookup[k]

    if required:
        raise RuntimeError(
            f"Required column not found. candidates={candidates}; columns={list(df.columns)}"
        )

    return None

def normalize_group(x):
    s = str(x).strip().lower().replace(" ", "")

    if (
        "lt_18_5" in s
        or "lt18.5" in s
        or "<18.5" in s
        or "bmi<18.5" in s
    ):
        return "BMI_lt_18_5"

    if (
        "ge_18_5" in s
        or "ge18.5" in s
        or ">=18.5" in s
        or "bmi>=18.5" in s
    ):
        return "BMI_ge_18_5"

    raise RuntimeError(f"Unrecognized BMI group: {x!r}")

def best_id_column(df, target_ids):
    target = {str(x) for x in target_ids}

    preferred = [
        "participant_id",
        "participant_id_private",
        "participant",
        "PID",
        "Patient #",
    ]

    candidates = []

    for c in preferred + list(df.columns):
        if c not in df.columns:
            continue

        vals = set(df[c].dropna().astype(str))
        overlap = len(vals & target)

        if overlap:
            candidates.append((overlap, c))

    if not candidates:
        raise RuntimeError(
            f"No participant-ID column overlaps target IDs in columns={list(df.columns)}"
        )

    candidates.sort(reverse=True)

    best_overlap, best = candidates[0]

    if best_overlap < max(2, int(0.8 * len(target))):
        raise RuntimeError(
            f"Participant-ID overlap too small: {best} overlap={best_overlap}/{len(target)}"
        )

    return best

def bh(pvals):
    p = np.asarray(pvals, float)
    out = np.full(len(p), np.nan)

    ok = np.isfinite(p)
    idx = np.where(ok)[0]

    if len(idx) == 0:
        return out

    vals = p[idx]
    order = np.argsort(vals)
    ranked = vals[order]

    q = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1.0)

    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))

    out[idx] = q[inv]
    return out

base = read_tsv(BASE)
sample = read_tsv(SAMPLE)
long = read_tsv(LONG)
s13 = read_tsv(S13)
s14 = read_tsv(S14)
s16 = read_tsv(S16)
s17 = read_tsv(S17)

# -----------------------------------------------------------------
# Input-column adjudication
# -----------------------------------------------------------------

l_pid = pick(long, ["participant_id_private", "participant_id", "participant"])
l_week = pick(long, ["week", "visit_week", "timepoint", "visit"])
l_protein = pick(long, ["protein", "protein_row_id"])
l_abundance = pick(long, ["log2_normalized_abundance", "abundance", "value", "intensity"])
l_group = pick(long, ["bmi_group", "baseline_bmi_group"])
l_bmi = pick(long, ["bmi", "baseline_bmi"])

long = long.rename(
    columns={
        l_pid: "subject_id",
        l_week: "timepoint",
        l_protein: "protein",
        l_abundance: "abundance",
        l_group: "bmi_group",
        l_bmi: "bmi",
    }
)

long["subject_id"] = long["subject_id"].astype(str)
long["timepoint"] = pd.to_numeric(long["timepoint"], errors="raise").astype(int)
long["abundance"] = pd.to_numeric(long["abundance"], errors="coerce")
long["bmi"] = pd.to_numeric(long["bmi"], errors="coerce")
long["bmi_group"] = long["bmi_group"].map(normalize_group)

target_ids = sorted(long["subject_id"].unique())

b_pid = best_id_column(base, target_ids)
base = base.rename(columns={b_pid: "subject_id"})
base["subject_id"] = base["subject_id"].astype(str)

s_pid = best_id_column(sample, target_ids)
s_week = pick(sample, ["visit_week", "week", "timepoint", "visit"])
sample = sample.rename(columns={s_pid: "subject_id", s_week: "timepoint"})
sample["subject_id"] = sample["subject_id"].astype(str)
sample["timepoint"] = pd.to_numeric(sample["timepoint"], errors="raise").astype(int)

# -----------------------------------------------------------------
# 36-protein membership and frozen loadings
# -----------------------------------------------------------------

axis_family_col = pick(s13, ["axis_family"])
feature_col = pick(s13, ["matrix_feature", "feature_or_module", "protein"])

members = s13.loc[
    s13[axis_family_col].astype(str).eq("three_module_core"),
    feature_col,
].astype(str).tolist()

features36 = list(dict.fromkeys(members))

if len(features36) != 36:
    raise RuntimeError(f"HOLD_EXPECTED_36_FEATURES_FOUND_{len(features36)}")

component_col = pick(s14, ["component_type"])
loading_feature_col = pick(s14, ["feature_or_module", "matrix_feature", "protein"])
loading_col = pick(s14, ["pc1_loading", "loading"])

load36 = (
    s14.loc[
        s14[component_col].astype(str).eq(
            "three_module_core_unique_protein_pc1"
        ),
        [loading_feature_col, loading_col],
    ]
    .assign(**{loading_col: lambda d: pd.to_numeric(d[loading_col], errors="raise")})
    .set_index(loading_feature_col)[loading_col]
)

if len(load36) != 36:
    raise RuntimeError(f"HOLD_EXPECTED_36_LOADINGS_FOUND_{len(load36)}")

if set(features36) != set(load36.index.astype(str)):
    raise RuntimeError("HOLD_FEATURE_MEMBERSHIP_AND_LOADING_SET_DIFFER")

# -----------------------------------------------------------------
# Sample-level matrix
# -----------------------------------------------------------------

core_long = long.loc[long["protein"].isin(features36)].copy()

X = core_long.pivot(
    index=["subject_id", "timepoint"],
    columns="protein",
    values="abundance",
)

meta = (
    long[
        ["subject_id", "timepoint", "bmi_group", "bmi"]
    ]
    .drop_duplicates(["subject_id", "timepoint"])
    .sort_values(["subject_id", "timepoint"])
    .reset_index(drop=True)
)

if len(meta) != 150:
    raise RuntimeError(f"HOLD_EXPECTED_150_SAMPLE_VISITS_FOUND_{len(meta)}")

index = pd.MultiIndex.from_frame(meta[["subject_id", "timepoint"]])
X = X.reindex(index=index, columns=features36)

if X.shape != (150, 36):
    raise RuntimeError(f"HOLD_UNEXPECTED_AXIS_MATRIX_SHAPE={X.shape}")

# -----------------------------------------------------------------
# Add baseline covariates
# -----------------------------------------------------------------

def numeric_binary(series):
    x = pd.to_numeric(series, errors="coerce")
    return x

smear_col = pick(base, ["spsmear0.adj"], required=True)
hard_col = pick(base, ["treatcat0.adj"], required=True)
study_col = pick(base, ["STUDY", "study"], required=True)
female_col = pick(base, ["FEMALE", "female"], required=True)
black_col = pick(base, ["BLACK", "black"], required=True)
arm_col = pick(base, ["ARM", "arm"], required=True)

b = base.drop_duplicates("subject_id").copy()

b["smear3plus"] = (
    b[smear_col].astype(str).str.strip().str.upper().eq("3+").astype(int)
)

b["hard_to_treat"] = (
    b[hard_col].astype(str).str.strip().str.upper().eq("H").astype(int)
)

b["remox"] = (
    b[study_col].astype(str).str.strip().str.upper().eq("REMOX").astype(int)
)

b["female"] = numeric_binary(b[female_col])
b["black"] = numeric_binary(b[black_col])
b["arm"] = pd.to_numeric(b[arm_col], errors="coerce")

base_cov = b[
    [
        "subject_id",
        "smear3plus",
        "hard_to_treat",
        "remox",
        "female",
        "black",
        "arm",
    ]
]

meta = meta.merge(
    base_cov,
    on="subject_id",
    how="left",
    validate="many_to_one",
)

meta["undernutrition"] = (
    meta["bmi_group"].eq("BMI_lt_18_5").astype(int)
)

# TMT plex
plex_col = pick(
    sample,
    ["PLEX #", "PLEX", "plex", "plex_number"],
    required=False,
)

if plex_col is not None:
    plexmap = (
        sample[["subject_id", "timepoint", plex_col]]
        .drop_duplicates(["subject_id", "timepoint"])
        .rename(columns={plex_col: "plex"})
    )

    meta = meta.merge(
        plexmap,
        on=["subject_id", "timepoint"],
        how="left",
        validate="one_to_one",
    )
else:
    meta["plex"] = np.nan

# -----------------------------------------------------------------
# PCA utilities
# -----------------------------------------------------------------

frozen = load36.astype(float)

def build_axis(feature_list, require_complete=False):
    xv = X.loc[:, feature_list].astype(float)

    missing = int(xv.isna().sum().sum())

    mu = xv.mean(axis=0, skipna=True)
    sd = xv.std(axis=0, skipna=True, ddof=0)

    if (sd <= 0).any() or sd.isna().any():
        bad = sd[(sd <= 0) | sd.isna()]
        raise RuntimeError(f"HOLD_ZERO_OR_INVALID_SD={bad.to_dict()}")

    z = (xv - mu) / sd

    if require_complete and z.isna().any().any():
        raise RuntimeError(
            f"HOLD_REQUIRE_COMPLETE_BUT_MISSING={int(z.isna().sum().sum())}"
        )

    zi = z.fillna(0.0).to_numpy(float)

    u, s, vt = np.linalg.svd(zi, full_matrices=False)

    scores = zi @ vt[0, :]
    loadings = pd.Series(vt[0, :], index=feature_list)

    common = [p for p in feature_list if p in frozen.index]

    if float(np.dot(loadings.loc[common], frozen.loc[common])) < 0:
        scores *= -1
        loadings *= -1

    eigenvalues = (s ** 2) / max(1, zi.shape[0] - 1)
    explained = float(eigenvalues[0] / eigenvalues.sum())

    return {
        "scores": scores,
        "loadings": loadings,
        "eigenvalues": eigenvalues,
        "explained": explained,
        "missing_cells": missing,
        "z": zi,
    }

primary = build_axis(features36)

rho_load, p_load = stats.spearmanr(
    primary["loadings"].loc[frozen.index],
    frozen.values,
)

# expected variance explained
metric_col = pick(s17, ["metric"])
value_col = pick(s17, ["value"])

row = s17.loc[
    s17[metric_col].astype(str).eq(
        "unique_protein_pc1_variance_explained"
    )
]

if len(row) != 1:
    raise RuntimeError("HOLD_S17_PC1_VARIANCE_ROW_NOT_UNIQUE")

expected_variance = float(row.iloc[0][value_col])
variance_delta = abs(primary["explained"] - expected_variance)

if rho_load < 0.98:
    raise RuntimeError(
        f"HOLD_PRIMARY_AXIS_LOADING_RECONSTRUCTION_RHO_TOO_LOW={rho_load}"
    )

if variance_delta > 0.01:
    raise RuntimeError(
        f"HOLD_PRIMARY_AXIS_VARIANCE_RECONSTRUCTION_DELTA={variance_delta}"
    )

meta["axis36"] = primary["scores"]

# -----------------------------------------------------------------
# Model utility
# -----------------------------------------------------------------

def fit_model(df, formula, reml=False, allow_fallback=True):
    dat = df.dropna(
        subset=[
            "score",
            "subject_id",
            "timepoint",
            "undernutrition",
        ]
    ).copy()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            fit = smf.mixedlm(
                formula,
                dat,
                groups=dat["subject_id"],
            ).fit(
                reml=reml,
                method="lbfgs",
                maxiter=500,
                disp=False,
            )

        if hasattr(fit, "converged") and not fit.converged:
            raise RuntimeError("MixedLM did not converge")

        fit_type = "mixedlm_random_intercept"

    except Exception as e:
        if not allow_fallback:
            raise

        fit = smf.ols(
            formula,
            dat,
        ).fit(
            cov_type="cluster",
            cov_kwds={"groups": dat["subject_id"]},
        )

        fit_type = f"ols_cluster_robust_subject__fallback__{type(e).__name__}"

    terms = [
        t for t in fit.params.index
        if "undernutrition" in t and "C(timepoint)" in t
    ]

    beta = fit.params.loc[terms].to_numpy(float)
    cov = fit.cov_params().loc[terms, terms].to_numpy(float)

    rank = int(np.linalg.matrix_rank(cov))

    if rank > 0:
        stat = float(beta.T @ np.linalg.pinv(cov) @ beta)
        joint_p = float(stats.chi2.sf(stat, rank))
    else:
        stat = np.nan
        joint_p = np.nan

    rows = []

    for t in terms:
        rows.append(
            {
                "fit_type": fit_type,
                "term": t,
                "estimate": float(fit.params[t]),
                "std_error": float(fit.bse[t]),
                "p_value": float(fit.pvalues[t]),
                "n_observations": int(fit.nobs),
                "n_subjects": int(dat["subject_id"].nunique()),
                "joint_wald_chi2": stat,
                "joint_df": rank,
                "joint_p_value": joint_p,
            }
        )

    return fit, pd.DataFrame(rows)

# -----------------------------------------------------------------
# C03 simultaneous multivariable sensitivity
# -----------------------------------------------------------------

c03 = meta.copy()
c03["score"] = c03["axis36"]

required_c03 = [
    "smear3plus",
    "hard_to_treat",
    "remox",
    "black",
    "female",
]

if c03[required_c03].isna().any().any():
    raise RuntimeError(
        "HOLD_C03_REQUIRED_COVARIATE_MISSING"
    )

formula_c03 = (
    "score ~ undernutrition * C(timepoint)"
    " + smear3plus + hard_to_treat + remox + black + female"
)

fit_c03, tab_c03 = fit_model(
    c03,
    formula_c03,
    reml=False,
    allow_fallback=True,
)

tab_c03["analysis"] = (
    "reviewer_requested_simultaneous_multivariable_sensitivity"
)
tab_c03["formula"] = formula_c03

tab_c03.to_csv(
    OUT / "F14_R5A_C03_simultaneous_multivariable_interactions.tsv",
    sep="\t",
    index=False,
)

# -----------------------------------------------------------------
# C04 REMoxTB-only ARM-adjusted sensitivity
# -----------------------------------------------------------------

c04 = meta.loc[meta["remox"].eq(1)].copy()
c04["score"] = c04["axis36"]

if c04["subject_id"].nunique() != 21:
    raise RuntimeError(
        f"HOLD_C04_EXPECTED_21_REMOX_FOUND_{c04['subject_id'].nunique()}"
    )

if c04["arm"].isna().any():
    raise RuntimeError(
        "HOLD_C04_ARM_MISSING_WITHIN_REMOX"
    )

c04["arm"] = c04["arm"].astype(int).astype(str)

formula_c04 = (
    "score ~ undernutrition * C(timepoint) + C(arm)"
)

fit_c04, tab_c04 = fit_model(
    c04,
    formula_c04,
    reml=False,
    allow_fallback=True,
)

tab_c04["analysis"] = (
    "reviewer_requested_REMOX_only_ARM_adjusted_sensitivity"
)
tab_c04["formula"] = formula_c04

tab_c04.to_csv(
    OUT / "F14_R5A_C04_REMOX_ARM_adjusted_interactions.tsv",
    sep="\t",
    index=False,
)

arm_counts = (
    c04[
        ["subject_id", "undernutrition", "arm"]
    ]
    .drop_duplicates()
    .groupby(["undernutrition", "arm"])
    .size()
    .rename("n_participants")
    .reset_index()
)

arm_counts.to_csv(
    OUT / "F14_R5A_C04_REMOX_ARM_distribution.tsv",
    sep="\t",
    index=False,
)

# -----------------------------------------------------------------
# C05 MDE / power based on frozen submitted interaction SEs
# -----------------------------------------------------------------

axis_col = pick(s16, ["axis"])
family_col = pick(s16, ["model_family"])
term_col = pick(s16, ["term"])
se_col = pick(s16, ["std_error"])

power_source = s16.loc[
    s16[axis_col].astype(str).eq(
        "three_module_core_unique_protein_pc1"
    )
    & s16[family_col].astype(str).eq(
        "categorical_bmi_group_by_time"
    )
].copy()

if len(power_source) != 4:
    raise RuntimeError(
        f"HOLD_C05_EXPECTED_4_INTERACTION_TERMS_FOUND_{len(power_source)}"
    )

power_source[se_col] = pd.to_numeric(
    power_source[se_col],
    errors="raise",
)

alpha = 0.05
zcrit = stats.norm.ppf(1 - alpha / 2)

def two_sided_power(delta, se):
    mu = abs(delta) / se
    return float(
        stats.norm.cdf(-zcrit - mu)
        + stats.norm.sf(zcrit - mu)
    )

def exact_mde(se, target):
    f = lambda d: two_sided_power(d, se) - target
    return float(brentq(f, 0.0, 20.0 * se))

mde_rows = []
curve_rows = []

for _, r in power_source.iterrows():
    term = str(r[term_col])
    se = float(r[se_col])

    for target in (0.80, 0.90):
        mde_rows.append(
            {
                "term": term,
                "std_error": se,
                "alpha_two_sided": alpha,
                "target_power": target,
                "minimal_detectable_absolute_interaction": exact_mde(
                    se,
                    target,
                ),
            }
        )

    for delta in np.arange(0.0, 3.01, 0.25):
        curve_rows.append(
            {
                "term": term,
                "std_error": se,
                "true_absolute_interaction": float(delta),
                "power": two_sided_power(float(delta), se),
            }
        )

pd.DataFrame(mde_rows).to_csv(
    OUT / "F14_R5A_C05_interaction_MDE.tsv",
    sep="\t",
    index=False,
)

pd.DataFrame(curve_rows).to_csv(
    OUT / "F14_R5A_C05_interaction_power_curve.tsv",
    sep="\t",
    index=False,
)

# -----------------------------------------------------------------
# C07 PCA eigenvalues / fixed-seed parallel analysis
# -----------------------------------------------------------------

obs_corr = np.corrcoef(primary["z"], rowvar=False)
obs_eigs = np.linalg.eigvalsh(obs_corr)[::-1]

rng = np.random.default_rng(SEED)
sim_eigs = np.empty((N_PARALLEL, len(features36)), float)

for i in range(N_PARALLEL):
    g = rng.normal(
        size=(len(meta), len(features36))
    )

    g = (
        g - g.mean(axis=0, keepdims=True)
    ) / g.std(axis=0, ddof=0, keepdims=True)

    ev = np.linalg.eigvalsh(
        np.corrcoef(g, rowvar=False)
    )[::-1]

    sim_eigs[i, :] = ev

parallel95 = np.quantile(
    sim_eigs,
    0.95,
    axis=0,
)

parallel_mean = sim_eigs.mean(axis=0)

pca_table = pd.DataFrame(
    {
        "component": np.arange(1, len(features36) + 1),
        "observed_eigenvalue": obs_eigs,
        "parallel_mean_eigenvalue": parallel_mean,
        "parallel_95pct_eigenvalue": parallel95,
        "retain_observed_gt_parallel95": obs_eigs > parallel95,
    }
)

pca_table["observed_variance_fraction"] = (
    pca_table["observed_eigenvalue"]
    / pca_table["observed_eigenvalue"].sum()
)

pca_table.to_csv(
    OUT / "F14_R5A_C07_PCA_scree_parallel_analysis.tsv",
    sep="\t",
    index=False,
)

# -----------------------------------------------------------------
# C07/C08 axis sensitivity variants
# -----------------------------------------------------------------

variants = {
    "primary_36_mean_imputed": (features36, False),
    "drop_LCAT": (
        [x for x in features36 if x != "LCAT_HUMAN"],
        False,
    ),
    "drop_LEPR": (
        [x for x in features36 if x != "LEPR_HUMAN"],
        False,
    ),
    "drop_TRFE": (
        [x for x in features36 if x != "TRFE_HUMAN"],
        False,
    ),
    "drop_LCAT_LEPR_TRFE": (
        [
            x for x in features36
            if x not in {"LCAT_HUMAN", "LEPR_HUMAN", "TRFE_HUMAN"}
        ],
        False,
    ),
    "axis34_no_LEP_LEPR": (
        [
            x for x in features36
            if x not in {"LEP_HUMAN", "LEPR_HUMAN"}
        ],
        False,
    ),
    "axis33_complete_no_APOA5_LEP_LEPR": (
        [
            x for x in features36
            if x not in {
                "APOA5_HUMAN",
                "LEP_HUMAN",
                "LEPR_HUMAN",
            }
        ],
        True,
    ),
}

variant_scores = {}
variant_rows = []

for name, (features, require_complete) in variants.items():
    obj = build_axis(
        features,
        require_complete=require_complete,
    )

    variant_scores[name] = obj["scores"]

    rho, p = stats.spearmanr(
        primary["scores"],
        obj["scores"],
    )

    variant_rows.append(
        {
            "variant": name,
            "n_features": len(features),
            "missing_cells_before_imputation": obj["missing_cells"],
            "requires_complete_matrix": require_complete,
            "pc1_variance_explained": obj["explained"],
            "spearman_rho_vs_primary36": rho,
            "spearman_p_vs_primary36": p,
        }
    )

variant_table = pd.DataFrame(variant_rows)

variant_table.to_csv(
    OUT / "F14_R5A_C07_C08_axis_sensitivity_summary.tsv",
    sep="\t",
    index=False,
)

# -----------------------------------------------------------------
# C08 missingness audit
# -----------------------------------------------------------------

missing_by_protein = pd.DataFrame(
    {
        "protein": features36,
        "missing_n": X.isna().sum().reindex(features36).values,
        "total_n": len(X),
    }
)

missing_by_protein["missing_fraction"] = (
    missing_by_protein["missing_n"]
    / missing_by_protein["total_n"]
)

missing_by_protein.to_csv(
    OUT / "F14_R5A_C08_missingness_by_protein.tsv",
    sep="\t",
    index=False,
)

cell_missing = X.isna().sum(axis=1).to_numpy()
cell_total = X.shape[1]

miss_sample = meta[
    [
        "subject_id",
        "timepoint",
        "bmi_group",
        "plex",
    ]
].copy()

miss_sample["missing_cells"] = cell_missing
miss_sample["total_axis_cells"] = cell_total

by_visit = (
    miss_sample
    .groupby("timepoint", dropna=False)
    .agg(
        missing_cells=("missing_cells", "sum"),
        total_cells=("total_axis_cells", "sum"),
        n_sample_visits=("subject_id", "size"),
    )
    .reset_index()
)

by_visit["missing_fraction"] = (
    by_visit["missing_cells"] / by_visit["total_cells"]
)

by_visit.to_csv(
    OUT / "F14_R5A_C08_missingness_by_visit.tsv",
    sep="\t",
    index=False,
)

by_bmi = (
    miss_sample
    .groupby("bmi_group", dropna=False)
    .agg(
        missing_cells=("missing_cells", "sum"),
        total_cells=("total_axis_cells", "sum"),
        n_sample_visits=("subject_id", "size"),
    )
    .reset_index()
)

by_bmi["missing_fraction"] = (
    by_bmi["missing_cells"] / by_bmi["total_cells"]
)

by_bmi.to_csv(
    OUT / "F14_R5A_C08_missingness_by_BMI_stratum.tsv",
    sep="\t",
    index=False,
)

by_plex = (
    miss_sample
    .groupby("plex", dropna=False)
    .agg(
        missing_cells=("missing_cells", "sum"),
        total_cells=("total_axis_cells", "sum"),
        n_sample_visits=("subject_id", "size"),
    )
    .reset_index()
)

by_plex["missing_fraction"] = (
    by_plex["missing_cells"] / by_plex["total_cells"]
)

by_plex.to_csv(
    OUT / "F14_R5A_C08_missingness_by_TMT_plex.tsv",
    sep="\t",
    index=False,
)

# -----------------------------------------------------------------
# C08 34-/33-protein outcome robustness
# -----------------------------------------------------------------

robustness_summary = []
within_rows = []
interaction_rows = []

for name in [
    "axis34_no_LEP_LEPR",
    "axis33_complete_no_APOA5_LEP_LEPR",
]:
    d = meta.copy()
    d["score"] = variant_scores[name]

    baseline = d.loc[d["timepoint"].eq(0)].copy()

    a = baseline.loc[
        baseline["undernutrition"].eq(1),
        "score",
    ].to_numpy(float)

    b0 = baseline.loc[
        baseline["undernutrition"].eq(0),
        "score",
    ].to_numpy(float)

    welch = stats.ttest_ind(
        a,
        b0,
        equal_var=False,
        nan_policy="omit",
    )

    rho_bmi, p_bmi = stats.spearmanr(
        baseline["bmi"],
        baseline["score"],
        nan_policy="omit",
    )

    fit, tab = fit_model(
        d,
        "score ~ undernutrition * C(timepoint)",
        reml=False,
        allow_fallback=True,
    )

    if len(tab):
        joint_p = float(tab["joint_p_value"].iloc[0])
        fit_type = str(tab["fit_type"].iloc[0])
    else:
        joint_p = np.nan
        fit_type = "UNKNOWN"

    robustness_summary.append(
        {
            "variant": name,
            "baseline_Welch_p": float(welch.pvalue),
            "baseline_continuous_BMI_Spearman_rho": float(rho_bmi),
            "baseline_continuous_BMI_Spearman_p": float(p_bmi),
            "categorical_BMI_by_time_joint_p": joint_p,
            "interaction_fit_type": fit_type,
        }
    )

    tab["variant"] = name
    interaction_rows.append(tab)

    wide_score = d.pivot(
        index="subject_id",
        columns="timepoint",
        values="score",
    )

    subject_group = (
        d[
            ["subject_id", "bmi_group"]
        ]
        .drop_duplicates()
        .set_index("subject_id")["bmi_group"]
    )

    for grp in ["BMI_lt_18_5", "BMI_ge_18_5"]:
        ids = subject_group[
            subject_group.eq(grp)
        ].index

        for tp in [8, 17, 26, 52]:
            delta = (
                wide_score.loc[ids, tp]
                - wide_score.loc[ids, 0]
            ).dropna()

            if len(delta) >= 2:
                test = stats.ttest_1samp(
                    delta,
                    popmean=0.0,
                )

                within_rows.append(
                    {
                        "variant": name,
                        "bmi_group": grp,
                        "post_timepoint": tp,
                        "n": len(delta),
                        "mean_delta": float(delta.mean()),
                        "sd_delta": float(delta.std(ddof=1)),
                        "p_value": float(test.pvalue),
                    }
                )

pd.DataFrame(robustness_summary).to_csv(
    OUT / "F14_R5A_C08_axis34_axis33_robustness_summary.tsv",
    sep="\t",
    index=False,
)

within_df = pd.DataFrame(within_rows)
within_df["fdr_bh_within_variant"] = np.nan

for v in within_df["variant"].unique():
    mask = within_df["variant"].eq(v)
    within_df.loc[
        mask,
        "fdr_bh_within_variant",
    ] = bh(within_df.loc[mask, "p_value"])

within_df.to_csv(
    OUT / "F14_R5A_C08_axis34_axis33_within_group_changes.tsv",
    sep="\t",
    index=False,
)

pd.concat(
    interaction_rows,
    ignore_index=True,
).to_csv(
    OUT / "F14_R5A_C08_axis34_axis33_interaction_terms.tsv",
    sep="\t",
    index=False,
)

# -----------------------------------------------------------------
# Reproducibility / report
# -----------------------------------------------------------------

repro = {
    "analysis_label": "reviewer-requested/post hoc",
    "random_seed": SEED,
    "parallel_analysis_iterations": N_PARALLEL,
    "python": sys.version.replace("\n", " "),
    "platform": platform.platform(),
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scipy": scipy.__version__,
    "statsmodels": statsmodels.__version__,
}

(OUT / "F14_R5A_reproducibility.json").write_text(
    json.dumps(repro, indent=2) + "\n",
    encoding="utf-8",
)

report = []

report.append("TB PROTEOMICS F14 R5A REVIEWER-REQUESTED ANALYSIS")
report.append("LABEL=POST_HOC_REVIEWER_REQUESTED")
report.append(f"SEED={SEED}")
report.append(f"N_PARALLEL={N_PARALLEL}")
report.append("")
report.append("PRIMARY_AXIS_RECONSTRUCTION")
report.append(f"N_SAMPLE_VISITS={len(meta)}")
report.append(f"N_SUBJECTS={meta['subject_id'].nunique()}")
report.append(f"N_FEATURES={len(features36)}")
report.append(
    f"MISSING_CELLS={int(X.isna().sum().sum())}"
)
report.append(
    f"PC1_VARIANCE_EXPLAINED_RECONSTRUCTED={primary['explained']:.12g}"
)
report.append(
    f"PC1_VARIANCE_EXPLAINED_FROZEN_S17={expected_variance:.12g}"
)
report.append(
    f"PC1_VARIANCE_DELTA={variance_delta:.12g}"
)
report.append(
    f"LOADING_SPEARMAN_RHO_VS_FROZEN_S14={rho_load:.12g}"
)
report.append("")
report.append("C03")
report.append(
    f"SIMULTANEOUS_ADJUSTED_JOINT_P={tab_c03['joint_p_value'].iloc[0]:.12g}"
)
report.append(
    f"FIT_TYPE={tab_c03['fit_type'].iloc[0]}"
)
report.append("")
report.append("C04")
report.append(
    f"REMOX_N_SUBJECTS={c04['subject_id'].nunique()}"
)
report.append(
    f"REMOX_ARM_ADJUSTED_JOINT_P={tab_c04['joint_p_value'].iloc[0]:.12g}"
)
report.append(
    f"FIT_TYPE={tab_c04['fit_type'].iloc[0]}"
)
report.append("")
report.append("C07")
report.append(
    f"PARALLEL_COMPONENTS_RETAINED_95PCT={int(pca_table['retain_observed_gt_parallel95'].sum())}"
)
report.append("")
report.append("C08")
report.append(
    f"AXIS36_TOTAL_MISSING={int(X.isna().sum().sum())}"
)
report.append(
    "MISSING_PROTEINS="
    + ",".join(
        missing_by_protein.loc[
            missing_by_protein["missing_n"].gt(0),
            "protein",
        ].astype(str)
    )
)
report.append("")
report.append("C09")
report.append(
    "NO_NEW_SEMANTIC_CLAIM=XXX_PREFIX_MEANING_REMAINS_UNRESOLVED"
)
report.append("")
report.append(
    "NEXT_BOUNDARY=F14_R5B_PROTEIN_LEVEL_SPARSE_ROBUSTNESS_AND_TARGETED_MODEL_DIAGNOSTICS"
)

(OUT / "F14_R5A_analysis_report.txt").write_text(
    "\n".join(report) + "\n",
    encoding="utf-8",
)

print("\n".join(report))
