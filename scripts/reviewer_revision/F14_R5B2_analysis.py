from __future__ import annotations

import json
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import statsmodels
import statsmodels.formula.api as smf

(
    LONG,
    SUPPORT,
    STRICTINT,
    FITSTATUS,
    FROZENINT,
    CALIB,
    CALSUM,
    OUT,
) = map(Path, sys.argv[1:9])

def qdesc(x):
    x = pd.to_numeric(
        pd.Series(x),
        errors="coerce",
    ).dropna()

    if len(x) == 0:
        return {
            "n": 0,
            "min": np.nan,
            "q25": np.nan,
            "median": np.nan,
            "q75": np.nan,
            "max": np.nan,
        }

    return {
        "n": int(len(x)),
        "min": float(x.min()),
        "q25": float(x.quantile(.25)),
        "median": float(x.median()),
        "q75": float(x.quantile(.75)),
        "max": float(x.max()),
    }

long = pd.read_csv(
    LONG,
    sep="\t",
    low_memory=False,
)

support = pd.read_csv(
    SUPPORT,
    sep="\t",
)

strictint = pd.read_csv(
    STRICTINT,
    sep="\t",
)

fitstatus = pd.read_csv(
    FITSTATUS,
    sep="\t",
)

frozen = pd.read_csv(
    FROZENINT,
    sep="\t",
)

calib = pd.read_csv(
    CALIB,
    sep="\t",
    low_memory=False,
)

# ---------------------------------------------------------
# Exact strict-support set
# ---------------------------------------------------------

eligible = (
    support.loc[
        support[
            "strict_interaction_allweeks_eligible"
        ].astype(bool),
        "protein",
    ]
    .astype(str)
    .tolist()
)

if len(eligible) != 1385:
    raise SystemExit(
        f"HOLD_EXPECTED_1385_STRICT_PROTEINS_FOUND_{len(eligible)}"
    )

# Identify the single strict within-week survivor.
survivor = strictint.loc[
    pd.to_numeric(
        strictint[
            "q_BH_strict_within_week"
        ],
        errors="coerce",
    ) < 0.05
].copy()

if len(survivor) != 1:
    raise SystemExit(
        f"HOLD_EXPECTED_ONE_STRICT_WEEK_SURVIVOR_FOUND_{len(survivor)}"
    )

survivor.to_csv(
    OUT / "F14_R5B2_single_strict_week_survivor.tsv",
    sep="\t",
    index=False,
)

# ---------------------------------------------------------
# Original strict-set fit status
# ---------------------------------------------------------

strict_status = fitstatus.loc[
    fitstatus["id"].astype(str).isin(
        eligible
    )
].copy()

strict_status.to_csv(
    OUT / "F14_R5B2_original_fit_status_strict_support.tsv",
    sep="\t",
    index=False,
)

status_summary = (
    strict_status.groupby(
        [
            "fit_type",
            "model_status",
            "fallback_used",
            "fallback_reason",
        ],
        dropna=False,
    )
    .size()
    .reset_index(name="n_proteins")
)

status_summary.to_csv(
    OUT / "F14_R5B2_original_fit_status_summary.tsv",
    sep="\t",
    index=False,
)

# ---------------------------------------------------------
# Long data setup
# ---------------------------------------------------------

required = {
    "protein_row_id",
    "participant_id_private",
    "visit_week",
    "log2_normalized_abundance",
    "baseline_bmi_group",
}

missing = required - set(long.columns)

if missing:
    raise SystemExit(
        "HOLD_LONG_SCHEMA_MISSING="
        + ",".join(sorted(missing))
    )

d = long.rename(
    columns={
        "protein_row_id": "protein",
        "participant_id_private": "subject",
        "visit_week": "week",
        "log2_normalized_abundance": "abundance",
        "baseline_bmi_group": "group",
    }
)

d = d.loc[
    d["protein"].astype(str).isin(
        eligible
    ),
    [
        "protein",
        "subject",
        "week",
        "abundance",
        "group",
    ],
].copy()

d["protein"] = d["protein"].astype(str)
d["subject"] = d["subject"].astype(str)

d["week"] = pd.to_numeric(
    d["week"],
    errors="raise",
).astype(int)

d["group"] = pd.Categorical(
    d["group"].astype(str),
    categories=[
        "BMI_ge_18_5",
        "BMI_lt_18_5",
    ],
    ordered=True,
)

d["week"] = pd.Categorical(
    d["week"],
    categories=[0, 8, 17, 26, 52],
    ordered=True,
)

formula = (
    "abundance ~ C(group) * C(week)"
)

diagnostic_rows = []
interaction_rows = []

for i, protein in enumerate(eligible, start=1):

    dat = d.loc[
        d["protein"].eq(protein)
        & d["abundance"].notna()
    ].copy()

    nobs = len(dat)
    nsub = dat["subject"].nunique()

    rec = {
        "protein": protein,
        "n_observations": nobs,
        "n_subjects": nsub,
    }

    fits = {}

    for method_name, reml in [
        ("ML", False),
        ("REML", True),
    ]:

        warning_text = []
        fit = None
        failure = ""

        try:
            with warnings.catch_warnings(
                record=True
            ) as caught:

                warnings.simplefilter(
                    "always"
                )

                fit = smf.mixedlm(
                    formula,
                    dat,
                    groups=dat["subject"],
                ).fit(
                    reml=reml,
                    method="lbfgs",
                    maxiter=500,
                    disp=False,
                )

                warning_text = [
                    f"{type(w.message).__name__}: {w.message}"
                    for w in caught
                ]

            if (
                hasattr(fit, "converged")
                and not fit.converged
            ):
                failure = "NONCONVERGED"

        except Exception as e:
            fit = None
            failure = (
                f"{type(e).__name__}: {e}"
            )

        rec[
            f"{method_name}_success"
        ] = int(
            fit is not None
            and failure == ""
        )

        rec[
            f"{method_name}_failure"
        ] = failure

        rec[
            f"{method_name}_warnings"
        ] = " | ".join(
            warning_text
        )

        if (
            fit is not None
            and failure == ""
        ):
            random_var = float(
                fit.cov_re.iloc[0, 0]
            )

            residual_var = float(
                fit.scale
            )

            denom = (
                random_var
                + residual_var
            )

            icc = (
                random_var / denom
                if denom > 0
                else np.nan
            )

            rec[
                f"{method_name}_random_intercept_variance"
            ] = random_var

            rec[
                f"{method_name}_residual_variance"
            ] = residual_var

            rec[
                f"{method_name}_ICC"
            ] = icc

            rec[
                f"{method_name}_loglik"
            ] = float(fit.llf)

            fits[
                method_name
            ] = fit

        else:
            rec[
                f"{method_name}_random_intercept_variance"
            ] = np.nan

            rec[
                f"{method_name}_residual_variance"
            ] = np.nan

            rec[
                f"{method_name}_ICC"
            ] = np.nan

            rec[
                f"{method_name}_loglik"
            ] = np.nan

    diagnostic_rows.append(rec)

    # Interaction-term diagnostics where available.
    for wk in [8, 17, 26, 52]:

        row = {
            "protein": protein,
            "week": wk,
        }

        frozen_row = frozen.loc[
            frozen["id"].astype(str).eq(
                protein
            )
            & pd.to_numeric(
                frozen["week"],
                errors="coerce",
            ).eq(wk)
        ]

        if len(frozen_row) != 1:
            raise SystemExit(
                f"HOLD_FROZEN_INTERACTION_NOT_UNIQUE_{protein}_{wk}"
            )

        fr = frozen_row.iloc[0]

        row[
            "frozen_fit_type"
        ] = str(fr["fit_type"])

        row[
            "frozen_estimate"
        ] = float(fr["estimate"])

        row[
            "frozen_std_error"
        ] = float(fr["std_error"])

        for method_name in [
            "ML",
            "REML",
        ]:
            fit = fits.get(
                method_name
            )

            if fit is None:
                row[
                    f"{method_name}_estimate"
                ] = np.nan

                row[
                    f"{method_name}_std_error"
                ] = np.nan

                continue

            target = (
                f"C(group)[T.BMI_lt_18_5]"
                f":C(week)[T.{wk}]"
            )

            if target not in fit.params.index:
                raise SystemExit(
                    f"HOLD_INTERACTION_TERM_MISSING_{protein}_{method_name}_{wk}"
                )

            row[
                f"{method_name}_estimate"
            ] = float(
                fit.params[target]
            )

            row[
                f"{method_name}_std_error"
            ] = float(
                fit.bse[target]
            )

        if np.isfinite(
            row.get(
                "ML_estimate",
                np.nan,
            )
        ):
            row[
                "abs_ML_minus_frozen_estimate"
            ] = abs(
                row[
                    "ML_estimate"
                ]
                - row[
                    "frozen_estimate"
                ]
            )

            row[
                "abs_ML_minus_frozen_SE"
            ] = abs(
                row[
                    "ML_std_error"
                ]
                - row[
                    "frozen_std_error"
                ]
            )
        else:
            row[
                "abs_ML_minus_frozen_estimate"
            ] = np.nan

            row[
                "abs_ML_minus_frozen_SE"
            ] = np.nan

        if (
            np.isfinite(
                row.get(
                    "ML_std_error",
                    np.nan,
                )
            )
            and np.isfinite(
                row.get(
                    "REML_std_error",
                    np.nan,
                )
            )
            and row[
                "ML_std_error"
            ] > 0
        ):
            row[
                "REML_to_ML_SE_ratio"
            ] = (
                row[
                    "REML_std_error"
                ]
                / row[
                    "ML_std_error"
                ]
            )

            row[
                "REML_minus_ML_estimate"
            ] = (
                row[
                    "REML_estimate"
                ]
                - row[
                    "ML_estimate"
                ]
            )
        else:
            row[
                "REML_to_ML_SE_ratio"
            ] = np.nan

            row[
                "REML_minus_ML_estimate"
            ] = np.nan

        interaction_rows.append(
            row
        )

    if (
        i % 100 == 0
        or i == len(eligible)
    ):
        print(
            f"PROGRESS_STRICT_PROTEINS={i}/{len(eligible)}"
        )

diag = pd.DataFrame(
    diagnostic_rows
)

ints = pd.DataFrame(
    interaction_rows
)

diag.to_csv(
    OUT / "F14_R5B2_ML_REML_random_intercept_diagnostics.tsv",
    sep="\t",
    index=False,
)

ints.to_csv(
    OUT / "F14_R5B2_ML_REML_interaction_comparison.tsv",
    sep="\t",
    index=False,
)

# ---------------------------------------------------------
# Reproduction check against frozen ML results
# Only frozen MixedLM rows are comparable.
# ---------------------------------------------------------

comparable = ints.loc[
    ints["frozen_fit_type"].eq(
        "mixedlm_random_intercept"
    )
    & ints["ML_estimate"].notna()
].copy()

if len(comparable) == 0:
    raise SystemExit(
        "HOLD_NO_COMPARABLE_ML_ROWS"
    )

max_beta_diff = float(
    comparable[
        "abs_ML_minus_frozen_estimate"
    ].max()
)

max_se_diff = float(
    comparable[
        "abs_ML_minus_frozen_SE"
    ].max()
)

print(
    f"MAX_ABS_ML_MINUS_FROZEN_ESTIMATE={max_beta_diff:.12g}"
)

print(
    f"MAX_ABS_ML_MINUS_FROZEN_SE={max_se_diff:.12g}"
)

if max_beta_diff > 1e-6:
    raise SystemExit(
        "HOLD_ML_COEFFICIENT_REPRODUCTION_MISMATCH"
    )

if max_se_diff > 1e-6:
    raise SystemExit(
        "HOLD_ML_SE_REPRODUCTION_MISMATCH"
    )

# ---------------------------------------------------------
# Existing mixed-vs-Welch calibration readback
# ---------------------------------------------------------

ratio_cols = [
    c for c in calib.columns
    if "ratio" in c.lower()
]

ratio_summary = []

for c in ratio_cols:
    desc = qdesc(
        calib[c]
    )

    ratio_summary.append(
        {
            "column": c,
            **desc,
        }
    )

pd.DataFrame(
    ratio_summary
).to_csv(
    OUT / "F14_R5B2_existing_calibration_ratio_summary.tsv",
    sep="\t",
    index=False,
)

# Preserve exact summary text as authenticated readback.
(
    OUT
    / "F14_R5B2_existing_calibration_summary_readback.txt"
).write_text(
    CALSUM.read_text(
        encoding="utf-8",
        errors="ignore",
    ),
    encoding="utf-8",
)

# ---------------------------------------------------------
# Random variance / ICC summaries
# ---------------------------------------------------------

variance_rows = []

for method in [
    "ML",
    "REML",
]:

    ok = diag.loc[
        diag[
            f"{method}_success"
        ].eq(1)
    ]

    for metric in [
        "random_intercept_variance",
        "residual_variance",
        "ICC",
    ]:
        desc = qdesc(
            ok[
                f"{method}_{metric}"
            ]
        )

        variance_rows.append(
            {
                "method": method,
                "metric": metric,
                **desc,
            }
        )

variance_summary = pd.DataFrame(
    variance_rows
)

variance_summary.to_csv(
    OUT / "F14_R5B2_variance_ICC_summary.tsv",
    sep="\t",
    index=False,
)

# ---------------------------------------------------------
# Failure summaries
# ---------------------------------------------------------

failure_rows = []

for method in [
    "ML",
    "REML",
]:

    x = (
        diag.groupby(
            [
                f"{method}_success",
                f"{method}_failure",
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="n_proteins"
        )
    )

    x.insert(
        0,
        "method",
        method,
    )

    x = x.rename(
        columns={
            f"{method}_success":
                "success",
            f"{method}_failure":
                "failure_reason",
        }
    )

    failure_rows.append(
        x
    )

failure_summary = pd.concat(
    failure_rows,
    ignore_index=True,
)

failure_summary.to_csv(
    OUT / "F14_R5B2_refit_failure_summary.tsv",
    sep="\t",
    index=False,
)

# ---------------------------------------------------------
# ML-vs-REML summary
# ---------------------------------------------------------

ml_ok = int(
    diag["ML_success"].sum()
)

reml_ok = int(
    diag["REML_success"].sum()
)

original_strict_mixed = int(
    strict_status[
        "fit_type"
    ].eq(
        "mixedlm_random_intercept"
    ).sum()
)

original_strict_fallback = int(
    strict_status[
        "fit_type"
    ].eq(
        "ols_cluster_robust_subject"
    ).sum()
)

se_ratio = qdesc(
    ints[
        "REML_to_ML_SE_ratio"
    ]
)

estimate_diff = qdesc(
    ints[
        "REML_minus_ML_estimate"
    ].abs()
)

ml_rv = qdesc(
    diag.loc[
        diag["ML_success"].eq(1),
        "ML_random_intercept_variance",
    ]
)

ml_icc = qdesc(
    diag.loc[
        diag["ML_success"].eq(1),
        "ML_ICC",
    ]
)

reml_rv = qdesc(
    diag.loc[
        diag["REML_success"].eq(1),
        "REML_random_intercept_variance",
    ]
)

reml_icc = qdesc(
    diag.loc[
        diag["REML_success"].eq(1),
        "REML_ICC",
    ]
)

report = []

report.append(
    "TB PROTEOMICS F14 R5B2 C12 TARGETED MIXED-MODEL DIAGNOSTICS"
)

report.append(
    "ANALYSIS_LABEL=POST_HOC_REVIEWER_REQUESTED"
)

report.append(
    f"STRICT_SUPPORT_PROTEINS={len(eligible)}"
)

report.append(
    f"ORIGINAL_STRICT_MIXEDLM={original_strict_mixed}"
)

report.append(
    f"ORIGINAL_STRICT_OLS_FALLBACK={original_strict_fallback}"
)

report.append(
    "ORIGINAL_FALLBACK_REASON=LinAlgError: Singular matrix"
)

report.append("")
report.append("R5B1_SURVIVOR")

sr = survivor.iloc[0]

for key in [
    "id",
    "week",
    "estimate",
    "std_error",
    "p_value",
    "q_BH_strict_within_week",
    "q_BH_strict_global_4weeks",
]:
    if key in survivor.columns:
        report.append(
            f"{key.upper()}={sr[key]}"
        )

report.append("")
report.append("ML_REFIT")

report.append(
    f"ML_SUCCESS={ml_ok}"
)

report.append(
    f"ML_RANDOM_INTERCEPT_VARIANCE_MEDIAN={ml_rv['median']}"
)

report.append(
    f"ML_RANDOM_INTERCEPT_VARIANCE_Q25={ml_rv['q25']}"
)

report.append(
    f"ML_RANDOM_INTERCEPT_VARIANCE_Q75={ml_rv['q75']}"
)

report.append(
    f"ML_ICC_MEDIAN={ml_icc['median']}"
)

report.append(
    f"ML_ICC_Q25={ml_icc['q25']}"
)

report.append(
    f"ML_ICC_Q75={ml_icc['q75']}"
)

report.append("")
report.append("REML_SENSITIVITY")

report.append(
    f"REML_SUCCESS={reml_ok}"
)

report.append(
    f"REML_RANDOM_INTERCEPT_VARIANCE_MEDIAN={reml_rv['median']}"
)

report.append(
    f"REML_RANDOM_INTERCEPT_VARIANCE_Q25={reml_rv['q25']}"
)

report.append(
    f"REML_RANDOM_INTERCEPT_VARIANCE_Q75={reml_rv['q75']}"
)

report.append(
    f"REML_ICC_MEDIAN={reml_icc['median']}"
)

report.append(
    f"REML_ICC_Q25={reml_icc['q25']}"
)

report.append(
    f"REML_ICC_Q75={reml_icc['q75']}"
)

report.append(
    f"REML_TO_ML_SE_RATIO_MEDIAN={se_ratio['median']}"
)

report.append(
    f"REML_TO_ML_SE_RATIO_Q25={se_ratio['q25']}"
)

report.append(
    f"REML_TO_ML_SE_RATIO_Q75={se_ratio['q75']}"
)

report.append(
    f"ABS_REML_MINUS_ML_INTERACTION_ESTIMATE_MEDIAN={estimate_diff['median']}"
)

report.append(
    f"ABS_REML_MINUS_ML_INTERACTION_ESTIMATE_MAX={estimate_diff['max']}"
)

report.append("")
report.append("FROZEN_ML_REPRODUCTION")

report.append(
    f"MAX_ABS_ML_MINUS_FROZEN_ESTIMATE={max_beta_diff}"
)

report.append(
    f"MAX_ABS_ML_MINUS_FROZEN_SE={max_se_diff}"
)

report.append("")
report.append(
    "ML_INTERPRETATION=ML was the fitting convention in the original post-hoc sensitivity; it is not asserted to be universally preferable to REML. REML is added here as a reviewer-requested sensitivity for variance-component and fixed-effect uncertainty assessment."
)

report.append(
    "NEXT_BOUNDARY=F14_ANALYTICAL_REVISION_BLOCK_CLOSED_IF_C12_DIAGNOSTICS_PASS"
)

(
    OUT
    / "F14_R5B2_analysis_report.txt"
).write_text(
    "\n".join(
        report
    ) + "\n",
    encoding="utf-8",
)

repro = {
    "analysis_label":
        "post_hoc_reviewer_requested",
    "strict_support_proteins":
        len(eligible),
    "formula":
        formula,
    "optimizer":
        "lbfgs",
    "maxiter":
        500,
    "ML_reml_flag":
        False,
    "REML_reml_flag":
        True,
    "python":
        sys.version.replace(
            "\n",
            " ",
        ),
    "platform":
        platform.platform(),
    "numpy":
        np.__version__,
    "pandas":
        pd.__version__,
    "scipy":
        scipy.__version__,
    "statsmodels":
        statsmodels.__version__,
}

(
    OUT
    / "F14_R5B2_reproducibility.json"
).write_text(
    json.dumps(
        repro,
        indent=2,
    ) + "\n",
    encoding="utf-8",
)

print(
    "\n".join(
        report
    )
)
