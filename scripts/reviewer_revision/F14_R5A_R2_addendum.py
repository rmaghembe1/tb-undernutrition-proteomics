import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

SEED = 20261122
N_PARALLEL = 2000

BASE = Path(sys.argv[1])
LONG = Path(sys.argv[2])
S13 = Path(sys.argv[3])
S14 = Path(sys.argv[4])
R5A = Path(sys.argv[5])
OUT = Path(sys.argv[6])

base = pd.read_csv(BASE, sep="\t")
long = pd.read_csv(LONG, sep="\t", low_memory=False)
s13 = pd.read_csv(S13, sep="\t")
s14 = pd.read_csv(S14, sep="\t")

# ------------------------------------------------------------
# Reconstruct exact 36-protein PC1
# ------------------------------------------------------------

features = (
    s13.loc[
        s13["axis_family"].eq("three_module_core"),
        "matrix_feature",
    ]
    .drop_duplicates()
    .astype(str)
    .tolist()
)

if len(features) != 36:
    raise SystemExit(f"HOLD_EXPECTED_36_FEATURES_FOUND_{len(features)}")

d = long.rename(
    columns={
        "participant_id_private": "subject_id",
        "visit_week": "timepoint",
        "protein_row_id": "protein",
        "log2_normalized_abundance": "abundance",
        "baseline_bmi": "bmi",
        "baseline_bmi_group": "bmi_group",
    }
)

d["subject_id"] = d["subject_id"].astype(str)

X = (
    d.loc[d["protein"].isin(features)]
    .pivot(
        index=["subject_id", "timepoint"],
        columns="protein",
        values="abundance",
    )
    .reindex(columns=features)
)

mu = X.mean(axis=0, skipna=True)
sd = X.std(axis=0, skipna=True, ddof=0)

Z = ((X - mu) / sd).fillna(0.0)

u, s, vt = np.linalg.svd(
    Z.to_numpy(float),
    full_matrices=False,
)

scores = Z.to_numpy(float) @ vt[0]

frozen = (
    s14.loc[
        s14["component_type"].eq(
            "three_module_core_unique_protein_pc1"
        ),
        ["feature_or_module", "pc1_loading"],
    ]
    .set_index("feature_or_module")["pc1_loading"]
    .astype(float)
)

load = pd.Series(vt[0], index=features)

if float(np.dot(load.loc[frozen.index], frozen)) < 0:
    scores *= -1
    load *= -1

rho = stats.spearmanr(
    load.loc[frozen.index],
    frozen,
).statistic

if rho < 0.999999:
    raise SystemExit(
        f"HOLD_AXIS_RECONSTRUCTION_NOT_EXACT_RHO={rho}"
    )

meta = (
    d[
        [
            "subject_id",
            "timepoint",
            "bmi",
            "bmi_group",
        ]
    ]
    .drop_duplicates(["subject_id", "timepoint"])
    .sort_values(["subject_id", "timepoint"])
    .reset_index(drop=True)
)

if len(meta) != 150:
    raise SystemExit(
        f"HOLD_EXPECTED_150_VISITS_FOUND_{len(meta)}"
    )

meta["score"] = scores

meta["undernutrition"] = (
    meta["bmi_group"].astype(str)
    .str.contains("lt_18_5", regex=False)
    .astype(int)
)

# ------------------------------------------------------------
# Attach simultaneous covariates
# ------------------------------------------------------------

b = base.copy()

b["subject_id"] = b["participant_id_private"].astype(str)

b["smear3plus"] = (
    b["spsmear0.adj"].astype(str)
    .str.strip()
    .str.upper()
    .eq("3+")
    .astype(int)
)

b["hard_to_treat"] = (
    b["treatcat0.adj"].astype(str)
    .str.strip()
    .str.upper()
    .eq("H")
    .astype(int)
)

b["remox"] = (
    b["STUDY"].astype(str)
    .str.strip()
    .str.upper()
    .eq("REMOX")
    .astype(int)
)

b["female"] = pd.to_numeric(
    b["FEMALE"],
    errors="raise",
)

b["black"] = pd.to_numeric(
    b["BLACK"],
    errors="raise",
)

covars = b[
    [
        "subject_id",
        "smear3plus",
        "hard_to_treat",
        "remox",
        "black",
        "female",
    ]
].drop_duplicates("subject_id")

meta = meta.merge(
    covars,
    on="subject_id",
    how="left",
    validate="many_to_one",
)

if meta[
    [
        "smear3plus",
        "hard_to_treat",
        "remox",
        "black",
        "female",
    ]
].isna().any().any():
    raise SystemExit(
        "HOLD_SIMULTANEOUS_COVARIATE_MISSING"
    )

# ------------------------------------------------------------
# C03 full interaction model with all fixed effects and CIs
# ------------------------------------------------------------

formula_interaction = (
    "score ~ undernutrition * C(timepoint)"
    " + smear3plus + hard_to_treat"
    " + remox + black + female"
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    m_inter = smf.mixedlm(
        formula_interaction,
        meta,
        groups=meta["subject_id"],
    ).fit(
        reml=False,
        method="lbfgs",
        maxiter=500,
        disp=False,
    )

ci = m_inter.conf_int()

full = pd.DataFrame(
    {
        "term": m_inter.params.index,
        "estimate": m_inter.params.values,
        "std_error": m_inter.bse.values,
        "ci95_low": ci.iloc[:, 0].values,
        "ci95_high": ci.iloc[:, 1].values,
        "p_value": m_inter.pvalues.values,
    }
)

full.to_csv(
    OUT / "F14_R5A_R2_C03_full_simultaneous_adjusted_model.tsv",
    sep="\t",
    index=False,
)

# ------------------------------------------------------------
# C03 simultaneous-adjustment common-offset model
# ------------------------------------------------------------

formula_offset = (
    "score ~ undernutrition + C(timepoint)"
    " + smear3plus + hard_to_treat"
    " + remox + black + female"
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    m_offset = smf.mixedlm(
        formula_offset,
        meta,
        groups=meta["subject_id"],
    ).fit(
        reml=False,
        method="lbfgs",
        maxiter=500,
        disp=False,
    )

ci2 = m_offset.conf_int()

offset = pd.DataFrame(
    {
        "term": m_offset.params.index,
        "estimate": m_offset.params.values,
        "std_error": m_offset.bse.values,
        "ci95_low": ci2.iloc[:, 0].values,
        "ci95_high": ci2.iloc[:, 1].values,
        "p_value": m_offset.pvalues.values,
    }
)

offset.to_csv(
    OUT / "F14_R5A_R2_C03_simultaneous_adjusted_common_offset.tsv",
    sep="\t",
    index=False,
)

# ------------------------------------------------------------
# C07 parallel analysis aligned to exact PCA covariance matrix
# ------------------------------------------------------------

A = Z.to_numpy(float)

obs_cov = np.cov(
    A,
    rowvar=False,
    ddof=1,
)

obs_eigs = np.linalg.eigvalsh(
    obs_cov
)[::-1]

rng = np.random.default_rng(SEED)

sim = np.empty(
    (N_PARALLEL, A.shape[1]),
    dtype=float,
)

for i in range(N_PARALLEL):

    G = rng.normal(
        size=A.shape
    )

    G = (
        G - G.mean(axis=0)
    ) / G.std(
        axis=0,
        ddof=0,
    )

    ev = np.linalg.eigvalsh(
        np.cov(
            G,
            rowvar=False,
            ddof=1,
        )
    )[::-1]

    sim[i, :] = ev

parallel95 = np.quantile(
    sim,
    0.95,
    axis=0,
)

parallel_mean = sim.mean(axis=0)

pa = pd.DataFrame(
    {
        "component": np.arange(
            1,
            A.shape[1] + 1,
        ),
        "observed_covariance_eigenvalue": obs_eigs,
        "observed_variance_fraction":
            obs_eigs / obs_eigs.sum(),
        "parallel_mean_eigenvalue":
            parallel_mean,
        "parallel_95pct_eigenvalue":
            parallel95,
        "retain_observed_gt_parallel95":
            obs_eigs > parallel95,
    }
)

pa.to_csv(
    OUT / "F14_R5A_R2_C07_covariance_parallel_analysis.tsv",
    sep="\t",
    index=False,
)

# ------------------------------------------------------------
# Copy existing missingness summaries for adjudication
# ------------------------------------------------------------

for name in [
    "F14_R5A_C08_missingness_by_visit.tsv",
    "F14_R5A_C08_missingness_by_BMI_stratum.tsv",
    "F14_R5A_C08_missingness_by_TMT_plex.tsv",
]:
    src = R5A / name
    dst = OUT / name
    dst.write_bytes(src.read_bytes())

# ------------------------------------------------------------
# Compact report
# ------------------------------------------------------------

u = full.loc[
    full["term"].eq("undernutrition")
].iloc[0]

uo = offset.loc[
    offset["term"].eq("undernutrition")
].iloc[0]

report = {
    "analysis_label":
        "post_hoc_reviewer_requested_addendum",
    "seed": SEED,
    "parallel_iterations": N_PARALLEL,
    "interaction_model_adjusted_baseline_BMI_group": {
        "estimate": float(u["estimate"]),
        "ci95_low": float(u["ci95_low"]),
        "ci95_high": float(u["ci95_high"]),
        "p_value": float(u["p_value"]),
    },
    "common_offset_simultaneous_adjustment": {
        "estimate": float(uo["estimate"]),
        "ci95_low": float(uo["ci95_low"]),
        "ci95_high": float(uo["ci95_high"]),
        "p_value": float(uo["p_value"]),
    },
    "parallel_components_retained_covariance_basis":
        int(
            pa[
                "retain_observed_gt_parallel95"
            ].sum()
        ),
    "pc1_variance_fraction_covariance_basis":
        float(
            pa.iloc[0][
                "observed_variance_fraction"
            ]
        ),
}

(
    OUT / "F14_R5A_R2_adjudication_report.json"
).write_text(
    json.dumps(
        report,
        indent=2,
    ) + "\n",
    encoding="utf-8",
)

print(
    json.dumps(
        report,
        indent=2,
    )
)
