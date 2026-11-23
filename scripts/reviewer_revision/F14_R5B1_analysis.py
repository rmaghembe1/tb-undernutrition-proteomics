from __future__ import annotations

import json
import math
import platform
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import matplotlib

SEED = 20261122

(
    LONG,
    MISSAUDIT,
    MIXFIT,
    MIXINT,
    MIXJOINT,
    OUT,
) = map(Path, sys.argv[1:7])

OUTFIG = OUT / "figures"

def bh(values):
    p = np.asarray(values, dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)

    good = np.isfinite(p)

    if not good.any():
        return q

    x = p[good]
    order = np.argsort(x)
    ranked = x[order]

    adj = ranked * len(ranked) / np.arange(
        1,
        len(ranked) + 1,
    )

    adj = np.minimum.accumulate(
        adj[::-1]
    )[::-1]

    adj = np.minimum(adj, 1.0)

    back = np.empty_like(order)
    back[order] = np.arange(len(order))

    q[good] = adj[back]

    return q

def welch_summary(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)

    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if len(a) < 2 or len(b) < 2:
        return {
            "n_lt": len(a),
            "n_ge": len(b),
            "mean_lt": np.nanmean(a) if len(a) else np.nan,
            "mean_ge": np.nanmean(b) if len(b) else np.nan,
            "difference_lt_minus_ge": np.nan,
            "welch_se": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "welch_p": np.nan,
        }

    ma = float(np.mean(a))
    mb = float(np.mean(b))

    va = float(np.var(a, ddof=1))
    vb = float(np.var(b, ddof=1))

    se2 = va / len(a) + vb / len(b)
    se = math.sqrt(se2)

    if se == 0:
        p = 1.0 if ma == mb else 0.0
        lo = hi = ma - mb
    else:
        numerator = se2 ** 2
        denominator = (
            (va / len(a)) ** 2 / (len(a) - 1)
            + (vb / len(b)) ** 2 / (len(b) - 1)
        )

        df = (
            numerator / denominator
            if denominator > 0
            else np.inf
        )

        tcrit = stats.t.ppf(
            0.975,
            df,
        )

        diff = ma - mb

        lo = diff - tcrit * se
        hi = diff + tcrit * se

        p = float(
            stats.ttest_ind(
                a,
                b,
                equal_var=False,
            ).pvalue
        )

    return {
        "n_lt": len(a),
        "n_ge": len(b),
        "mean_lt": ma,
        "mean_ge": mb,
        "difference_lt_minus_ge": ma - mb,
        "welch_se": se,
        "ci95_low": lo,
        "ci95_high": hi,
        "welch_p": p,
    }

def mann_whitney(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)

    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if len(a) < 1 or len(b) < 1:
        return np.nan

    try:
        return float(
            stats.mannwhitneyu(
                a,
                b,
                alternative="two-sided",
                method="asymptotic",
            ).pvalue
        )
    except Exception:
        return np.nan

def wilcoxon_1samp(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]

    if len(x) < 2:
        return np.nan

    if np.allclose(x, 0):
        return 1.0

    try:
        return float(
            stats.wilcoxon(
                x,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            ).pvalue
        )
    except Exception:
        return np.nan

# ---------------------------------------------------------
# Load authenticated sources
# ---------------------------------------------------------

long = pd.read_csv(
    LONG,
    sep="\t",
    low_memory=False,
)

miss_frozen = pd.read_csv(
    MISSAUDIT,
    sep="\t",
)

fitstatus = pd.read_csv(
    MIXFIT,
    sep="\t",
)

mixint = pd.read_csv(
    MIXINT,
    sep="\t",
)

mixjoint = pd.read_csv(
    MIXJOINT,
    sep="\t",
)

expected_long = {
    "protein_row_id",
    "participant_id_private",
    "visit_week",
    "log2_normalized_abundance",
    "baseline_bmi_group",
}

missing = expected_long - set(long.columns)

if missing:
    raise SystemExit(
        "HOLD_LONG_SCHEMA_MISSING="
        + ",".join(sorted(missing))
    )

long = long.rename(
    columns={
        "protein_row_id": "protein",
        "participant_id_private": "subject",
        "visit_week": "week",
        "log2_normalized_abundance": "abundance",
        "baseline_bmi_group": "group",
    }
)

long["subject"] = long["subject"].astype(str)
long["week"] = pd.to_numeric(
    long["week"],
    errors="raise",
).astype(int)

if len(long) != 288450:
    raise SystemExit(
        f"HOLD_EXPECTED_288450_ROWS_FOUND_{len(long)}"
    )

if long["protein"].nunique() != 1923:
    raise SystemExit(
        f"HOLD_EXPECTED_1923_PROTEINS_FOUND_{long['protein'].nunique()}"
    )

weeks = [0, 8, 17, 26, 52]
postweeks = [8, 17, 26, 52]

group_lt = "BMI_lt_18_5"
group_ge = "BMI_ge_18_5"

base_subjects = (
    long.loc[
        long["week"].eq(0),
        ["subject", "group"],
    ]
    .drop_duplicates()
)

n_lt_total = int(
    base_subjects["group"].eq(group_lt).sum()
)

n_ge_total = int(
    base_subjects["group"].eq(group_ge).sum()
)

if (n_lt_total, n_ge_total) != (13, 17):
    raise SystemExit(
        f"HOLD_EXPECTED_GROUP_SIZES_13_17_FOUND_{n_lt_total}_{n_ge_total}"
    )

min_lt = math.ceil(0.50 * n_lt_total)
min_ge = math.ceil(0.50 * n_ge_total)

if (min_lt, min_ge) != (7, 9):
    raise SystemExit(
        f"HOLD_SUPPORT_THRESHOLDS_UNEXPECTED_{min_lt}_{min_ge}"
    )

# ---------------------------------------------------------
# C10 support audit reconstructed from primary matrix
# ---------------------------------------------------------

proteins = sorted(
    long["protein"].astype(str).unique()
)

support = pd.DataFrame(
    {"protein": proteins}
)

obs = (
    long.assign(
        observed=long["abundance"].notna()
    )
    .groupby("protein")["observed"]
    .sum()
)

subjects_any = (
    long.loc[
        long["abundance"].notna()
    ]
    .groupby("protein")["subject"]
    .nunique()
)

support["observed_n_of_150"] = (
    support["protein"].map(obs).fillna(0).astype(int)
)

support["observed_fraction"] = (
    support["observed_n_of_150"] / 150.0
)

support["subjects_any_observation"] = (
    support["protein"]
    .map(subjects_any)
    .fillna(0)
    .astype(int)
)

# Baseline support by BMI group.
baseline = long.loc[
    long["week"].eq(0)
].copy()

for grp, suffix in [
    (group_lt, "lt"),
    (group_ge, "ge"),
]:
    counts = (
        baseline.loc[
            baseline["group"].eq(grp)
            & baseline["abundance"].notna()
        ]
        .groupby("protein")["subject"]
        .nunique()
    )

    support[
        f"baseline_observed_{suffix}"
    ] = (
        support["protein"]
        .map(counts)
        .fillna(0)
        .astype(int)
    )

# Paired baseline-post support by week and group.
for wk in postweeks:

    b = (
        long.loc[
            long["week"].eq(0),
            [
                "protein",
                "subject",
                "group",
                "abundance",
            ],
        ]
        .rename(
            columns={
                "abundance": "baseline_value"
            }
        )
    )

    p = (
        long.loc[
            long["week"].eq(wk),
            [
                "protein",
                "subject",
                "group",
                "abundance",
            ],
        ]
        .rename(
            columns={
                "abundance": "post_value"
            }
        )
    )

    paired = b.merge(
        p,
        on=[
            "protein",
            "subject",
            "group",
        ],
        how="inner",
        validate="one_to_one",
    )

    paired = paired.loc[
        paired["baseline_value"].notna()
        & paired["post_value"].notna()
    ]

    for grp, suffix in [
        (group_lt, "lt"),
        (group_ge, "ge"),
    ]:
        counts = (
            paired.loc[
                paired["group"].eq(grp)
            ]
            .groupby("protein")["subject"]
            .nunique()
        )

        support[
            f"paired_{suffix}_week{wk}"
        ] = (
            support["protein"]
            .map(counts)
            .fillna(0)
            .astype(int)
        )

# Original 10% feature rule.
support["original_10pct_eligible"] = (
    support["observed_fraction"] >= 0.10
)

support["strict_baseline_eligible"] = (
    support["original_10pct_eligible"]
    & (
        support["baseline_observed_lt"]
        >= min_lt
    )
    & (
        support["baseline_observed_ge"]
        >= min_ge
    )
)

for wk in postweeks:
    support[
        f"strict_delta_week{wk}_eligible"
    ] = (
        support["original_10pct_eligible"]
        & (
            support[
                f"paired_lt_week{wk}"
            ]
            >= min_lt
        )
        & (
            support[
                f"paired_ge_week{wk}"
            ]
            >= min_ge
        )
    )

support["strict_interaction_allweeks_eligible"] = (
    support["original_10pct_eligible"]
)

for wk in postweeks:
    support[
        "strict_interaction_allweeks_eligible"
    ] &= (
        (
            support[
                f"paired_lt_week{wk}"
            ]
            >= min_lt
        )
        & (
            support[
                f"paired_ge_week{wk}"
            ]
            >= min_ge
        )
    )

support.to_csv(
    OUT / "F14_R5B1_C10_feature_support_audit.tsv",
    sep="\t",
    index=False,
)

# Sanity against frozen audit where overlapping.
frozen_cols = {
    "protein",
    "n_observed_of_150",
    "n_subjects_any_observation",
}

if frozen_cols.issubset(
    set(miss_frozen.columns)
):
    chk = support.merge(
        miss_frozen[
            list(frozen_cols)
        ],
        on="protein",
        how="inner",
    )

    if not np.array_equal(
        chk["observed_n_of_150"].to_numpy(),
        chk["n_observed_of_150"].to_numpy(),
    ):
        raise SystemExit(
            "HOLD_RECONSTRUCTED_OBSERVED_COUNTS_DIFFER_FROM_FROZEN_AUDIT"
        )

# Whole-matrix completeness distribution.
quantiles = (
    support[
        [
            "observed_n_of_150",
            "observed_fraction",
            "subjects_any_observation",
        ]
    ]
    .quantile(
        [0, .05, .10, .25, .50, .75, .90, .95, 1]
    )
    .reset_index()
    .rename(columns={"index": "quantile"})
)

quantiles.to_csv(
    OUT / "F14_R5B1_C10_whole_matrix_completeness_quantiles.tsv",
    sep="\t",
    index=False,
)

bins = [
    -0.001,
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    0.999999,
    1.000001,
]

labels = [
    "<=10%",
    ">10-25%",
    ">25-50%",
    ">50-75%",
    ">75-90%",
    ">90-<100%",
    "100%",
]

support["observed_fraction_bin"] = pd.cut(
    support["observed_fraction"],
    bins=bins,
    labels=labels,
    include_lowest=True,
)

(
    support["observed_fraction_bin"]
    .value_counts(sort=False, dropna=False)
    .rename_axis("observed_fraction_bin")
    .reset_index(name="n_proteins")
    .to_csv(
        OUT / "F14_R5B1_C10_whole_matrix_completeness_bins.tsv",
        sep="\t",
        index=False,
    )
)

# ---------------------------------------------------------
# C10 stricter interaction multiplicity family
# ---------------------------------------------------------

eligible_interaction = set(
    support.loc[
        support[
            "strict_interaction_allweeks_eligible"
        ],
        "protein",
    ].astype(str)
)

strict_int = mixint.loc[
    mixint["id"].astype(str).isin(
        eligible_interaction
    )
].copy()

if strict_int.empty:
    raise SystemExit(
        "HOLD_STRICT_INTERACTION_SET_EMPTY"
    )

strict_int["q_BH_strict_within_week"] = np.nan

for wk in postweeks:
    mask = (
        pd.to_numeric(
            strict_int["week"],
            errors="coerce",
        ).eq(wk)
    )

    strict_int.loc[
        mask,
        "q_BH_strict_within_week",
    ] = bh(
        strict_int.loc[
            mask,
            "p_value",
        ]
    )

strict_int["q_BH_strict_global_4weeks"] = bh(
    strict_int["p_value"]
)

strict_int = strict_int.merge(
    support[
        [
            "protein",
            "paired_lt_week8",
            "paired_ge_week8",
            "paired_lt_week17",
            "paired_ge_week17",
            "paired_lt_week26",
            "paired_ge_week26",
            "paired_lt_week52",
            "paired_ge_week52",
        ]
    ],
    left_on="id",
    right_on="protein",
    how="left",
    validate="many_to_one",
)

strict_int.to_csv(
    OUT / "F14_R5B1_C10_strict_support_mixed_model_interactions.tsv",
    sep="\t",
    index=False,
)

strict_joint = mixjoint.loc[
    mixjoint["id"].astype(str).isin(
        eligible_interaction
    )
].copy()

strict_joint["q_BH_strict_joint"] = bh(
    strict_joint["joint_p_value"]
)

strict_joint.to_csv(
    OUT / "F14_R5B1_C10_strict_support_joint_interactions.tsv",
    sep="\t",
    index=False,
)

# Original significant rows and support.
orig_sig = mixint.loc[
    pd.to_numeric(
        mixint["q_value_BH_week"],
        errors="coerce",
    ) < 0.05
].copy()

orig_sig = orig_sig.merge(
    support,
    left_on="id",
    right_on="protein",
    how="left",
    validate="many_to_one",
)

orig_sig.to_csv(
    OUT / "F14_R5B1_C10_original_week_BH_signals_with_support.tsv",
    sep="\t",
    index=False,
)

# ---------------------------------------------------------
# C13 baseline robust comparison
# ---------------------------------------------------------

baseline_rows = []

baseline_eligible = set(
    support.loc[
        support["strict_baseline_eligible"],
        "protein",
    ].astype(str)
)

for protein in sorted(baseline_eligible):

    x = baseline.loc[
        baseline["protein"].astype(str).eq(protein)
        & baseline["group"].eq(group_lt),
        "abundance",
    ].dropna().to_numpy(float)

    y = baseline.loc[
        baseline["protein"].astype(str).eq(protein)
        & baseline["group"].eq(group_ge),
        "abundance",
    ].dropna().to_numpy(float)

    w = welch_summary(x, y)

    baseline_rows.append(
        {
            "protein": protein,
            **w,
            "mann_whitney_p":
                mann_whitney(x, y),
        }
    )

base_res = pd.DataFrame(baseline_rows)

base_res["welch_q_BH"] = bh(
    base_res["welch_p"]
)

base_res["mann_whitney_q_BH"] = bh(
    base_res["mann_whitney_p"]
)

base_res = base_res.sort_values(
    ["welch_p", "protein"],
    na_position="last",
)

base_res.to_csv(
    OUT / "F14_R5B1_C13_baseline_effect_CI_and_nonparametric.tsv",
    sep="\t",
    index=False,
)

# ---------------------------------------------------------
# C13 week-specific delta robust comparison
# ---------------------------------------------------------

wide = (
    long.pivot(
        index=[
            "protein",
            "subject",
            "group",
        ],
        columns="week",
        values="abundance",
    )
)

delta_rows = []
within_rows = []

for wk in postweeks:

    if 0 not in wide.columns or wk not in wide.columns:
        raise SystemExit(
            f"HOLD_MISSING_WIDE_WEEK_{wk}"
        )

    tmp = wide[[0, wk]].copy()

    tmp["delta"] = (
        tmp[wk] - tmp[0]
    )

    tmp = (
        tmp["delta"]
        .dropna()
        .reset_index()
    )

    eligible_week = set(
        support.loc[
            support[
                f"strict_delta_week{wk}_eligible"
            ],
            "protein",
        ].astype(str)
    )

    for protein in sorted(eligible_week):

        x = tmp.loc[
            tmp["protein"].astype(str).eq(protein)
            & tmp["group"].eq(group_lt),
            "delta",
        ].to_numpy(float)

        y = tmp.loc[
            tmp["protein"].astype(str).eq(protein)
            & tmp["group"].eq(group_ge),
            "delta",
        ].to_numpy(float)

        w = welch_summary(x, y)

        delta_rows.append(
            {
                "week": wk,
                "protein": protein,
                **w,
                "mann_whitney_p":
                    mann_whitney(x, y),
            }
        )

        for grp, values in [
            (group_lt, x),
            (group_ge, y),
        ]:
            vals = np.asarray(values, float)

            one_t = (
                float(
                    stats.ttest_1samp(
                        vals,
                        0.0,
                    ).pvalue
                )
                if len(vals) >= 2
                else np.nan
            )

            within_rows.append(
                {
                    "week": wk,
                    "protein": protein,
                    "bmi_group": grp,
                    "n": len(vals),
                    "mean_delta": (
                        float(np.mean(vals))
                        if len(vals)
                        else np.nan
                    ),
                    "median_delta": (
                        float(np.median(vals))
                        if len(vals)
                        else np.nan
                    ),
                    "one_sample_t_p": one_t,
                    "wilcoxon_p":
                        wilcoxon_1samp(vals),
                }
            )

delta_res = pd.DataFrame(delta_rows)

delta_res["welch_q_BH_within_week"] = np.nan
delta_res["mann_whitney_q_BH_within_week"] = np.nan

for wk in postweeks:
    mask = delta_res["week"].eq(wk)

    delta_res.loc[
        mask,
        "welch_q_BH_within_week",
    ] = bh(
        delta_res.loc[
            mask,
            "welch_p",
        ]
    )

    delta_res.loc[
        mask,
        "mann_whitney_q_BH_within_week",
    ] = bh(
        delta_res.loc[
            mask,
            "mann_whitney_p",
        ]
    )

delta_res["welch_q_BH_global_4weeks"] = bh(
    delta_res["welch_p"]
)

delta_res[
    "mann_whitney_q_BH_global_4weeks"
] = bh(
    delta_res["mann_whitney_p"]
)

delta_res = delta_res.sort_values(
    [
        "week",
        "welch_p",
        "protein",
    ],
    na_position="last",
)

delta_res.to_csv(
    OUT / "F14_R5B1_C13_delta_effect_CI_and_nonparametric.tsv",
    sep="\t",
    index=False,
)

within_res = pd.DataFrame(
    within_rows
)

within_res["one_sample_t_q_BH_within_group_week"] = np.nan
within_res["wilcoxon_q_BH_within_group_week"] = np.nan

for wk in postweeks:
    for grp in [group_lt, group_ge]:

        mask = (
            within_res["week"].eq(wk)
            & within_res[
                "bmi_group"
            ].eq(grp)
        )

        within_res.loc[
            mask,
            "one_sample_t_q_BH_within_group_week",
        ] = bh(
            within_res.loc[
                mask,
                "one_sample_t_p",
            ]
        )

        within_res.loc[
            mask,
            "wilcoxon_q_BH_within_group_week",
        ] = bh(
            within_res.loc[
                mask,
                "wilcoxon_p",
            ]
        )

within_res.to_csv(
    OUT / "F14_R5B1_C13_within_group_t_vs_wilcoxon.tsv",
    sep="\t",
    index=False,
)

# ---------------------------------------------------------
# Multiplicity denominator table
# ---------------------------------------------------------

families = []

families.append(
    {
        "family":
            "baseline_between_BMI_strata",
        "denominator":
            len(base_res),
        "eligibility":
            f">=10% overall observed and baseline n>={min_lt} BMI<18.5 and n>={min_ge} BMI>=18.5",
    }
)

for wk in postweeks:
    n = int(
        delta_res["week"].eq(wk).sum()
    )

    families.append(
        {
            "family":
                f"week{wk}_delta_between_BMI_strata",
            "denominator": n,
            "eligibility":
                f">=10% overall observed and paired n>={min_lt} BMI<18.5 and n>={min_ge} BMI>=18.5",
        }
    )

families.append(
    {
        "family":
            "mixed_model_interaction_strict_allweeks",
        "denominator":
            len(eligible_interaction),
        "eligibility":
            f">=10% overall observed and paired n>={min_lt}/>={min_ge} in every post-baseline week",
    }
)

pd.DataFrame(families).to_csv(
    OUT / "F14_R5B1_multiplicity_denominators.tsv",
    sep="\t",
    index=False,
)

# ---------------------------------------------------------
# Diagnostic figures
# ---------------------------------------------------------

def save_volcano(df, xcol, qcol, stem):

    q = pd.to_numeric(
        df[qcol],
        errors="coerce",
    )

    x = pd.to_numeric(
        df[xcol],
        errors="coerce",
    )

    y = -np.log10(
        np.clip(
            q.to_numpy(float),
            1e-300,
            1.0,
        )
    )

    fig, ax = plt.subplots(
        figsize=(6.6, 5.2)
    )

    ax.scatter(
        x,
        y,
        s=11,
        alpha=0.65,
    )

    ax.axhline(
        -np.log10(0.05),
        linewidth=0.8,
        linestyle="--",
    )

    ax.axvline(
        0,
        linewidth=0.8,
    )

    ax.set_xlabel(
        "Mean difference: BMI <18.5 minus BMI >=18.5"
    )

    ax.set_ylabel(
        "-log10(BH-adjusted q)"
    )

    fig.tight_layout()

    fig.savefig(
        OUTFIG / f"{stem}.png",
        dpi=1200,
        bbox_inches="tight",
    )

    fig.savefig(
        OUTFIG / f"{stem}.svg",
        bbox_inches="tight",
    )

    plt.close(fig)

def save_qhist(df, qcol, stem):

    q = pd.to_numeric(
        df[qcol],
        errors="coerce",
    ).dropna()

    fig, ax = plt.subplots(
        figsize=(6.6, 5.2)
    )

    ax.hist(
        q,
        bins=np.linspace(0, 1, 21),
    )

    ax.set_xlabel(
        "BH-adjusted q"
    )

    ax.set_ylabel(
        "Protein tests"
    )

    fig.tight_layout()

    fig.savefig(
        OUTFIG / f"{stem}.png",
        dpi=1200,
        bbox_inches="tight",
    )

    fig.savefig(
        OUTFIG / f"{stem}.svg",
        bbox_inches="tight",
    )

    plt.close(fig)

save_volcano(
    base_res,
    "difference_lt_minus_ge",
    "welch_q_BH",
    "F14_R5B1_C13_baseline_volcano",
)

save_qhist(
    base_res,
    "welch_q_BH",
    "F14_R5B1_C13_baseline_q_histogram",
)

for wk in postweeks:

    sub = delta_res.loc[
        delta_res["week"].eq(wk)
    ].copy()

    save_volcano(
        sub,
        "difference_lt_minus_ge",
        "welch_q_BH_within_week",
        f"F14_R5B1_C13_week{wk}_delta_volcano",
    )

    save_qhist(
        sub,
        "welch_q_BH_within_week",
        f"F14_R5B1_C13_week{wk}_delta_q_histogram",
    )

# ---------------------------------------------------------
# Compact report
# ---------------------------------------------------------

original_filter_n = int(
    support[
        "original_10pct_eligible"
    ].sum()
)

strict_interaction_n = int(
    support[
        "strict_interaction_allweeks_eligible"
    ].sum()
)

orig_week_q_sig = int(
    (
        pd.to_numeric(
            mixint[
                "q_value_BH_week"
            ],
            errors="coerce",
        )
        < 0.05
    ).sum()
)

orig_global_q_sig = int(
    (
        pd.to_numeric(
            mixint[
                "q_value_BH_global"
            ],
            errors="coerce",
        )
        < 0.05
    ).sum()
)

strict_week_q_sig = int(
    (
        strict_int[
            "q_BH_strict_within_week"
        ]
        < 0.05
    ).sum()
)

strict_global_q_sig = int(
    (
        strict_int[
            "q_BH_strict_global_4weeks"
        ]
        < 0.05
    ).sum()
)

orig_joint_q_sig = int(
    (
        pd.to_numeric(
            mixjoint[
                "q_value_BH_joint"
            ],
            errors="coerce",
        )
        < 0.05
    ).sum()
)

strict_joint_q_sig = int(
    (
        strict_joint[
            "q_BH_strict_joint"
        ]
        < 0.05
    ).sum()
)

report = []

report.append(
    "TB PROTEOMICS F14 R5B1 C10/C13"
)
report.append(
    "ANALYSIS_LABEL=POST_HOC_REVIEWER_REQUESTED"
)
report.append(
    f"GROUP_SIZE_BMI_LT_18_5={n_lt_total}"
)
report.append(
    f"GROUP_SIZE_BMI_GE_18_5={n_ge_total}"
)
report.append(
    f"STRICT_MIN_PAIRED_BMI_LT_18_5={min_lt}"
)
report.append(
    f"STRICT_MIN_PAIRED_BMI_GE_18_5={min_ge}"
)
report.append(
    "STRICT_RULE=AT_LEAST_50_PERCENT_PAIRED_SUPPORT_WITHIN_EACH_BMI_STRATUM"
)
report.append("")
report.append("C10")
report.append(
    f"ORIGINAL_10PCT_FEATURES={original_filter_n}"
)
report.append(
    f"STRICT_ALLWEEKS_INTERACTION_FEATURES={strict_interaction_n}"
)
report.append(
    f"ORIGINAL_WEEK_BH_Q_LT_0_05={orig_week_q_sig}"
)
report.append(
    f"STRICT_WEEK_BH_Q_LT_0_05={strict_week_q_sig}"
)
report.append(
    f"ORIGINAL_GLOBAL_BH_Q_LT_0_05={orig_global_q_sig}"
)
report.append(
    f"STRICT_GLOBAL_BH_Q_LT_0_05={strict_global_q_sig}"
)
report.append(
    f"ORIGINAL_JOINT_BH_Q_LT_0_05={orig_joint_q_sig}"
)
report.append(
    f"STRICT_JOINT_BH_Q_LT_0_05={strict_joint_q_sig}"
)
report.append("")
report.append("C13")
report.append(
    f"STRICT_BASELINE_TESTS={len(base_res)}"
)
report.append(
    f"BASELINE_WELCH_NOMINAL_P_LT_0_05={int((base_res['welch_p'] < 0.05).sum())}"
)
report.append(
    f"BASELINE_WELCH_BH_Q_LT_0_05={int((base_res['welch_q_BH'] < 0.05).sum())}"
)
report.append(
    f"BASELINE_MANN_WHITNEY_BH_Q_LT_0_05={int((base_res['mann_whitney_q_BH'] < 0.05).sum())}"
)

for wk in postweeks:

    sub = delta_res.loc[
        delta_res["week"].eq(wk)
    ]

    report.append(
        f"WEEK{wk}_DELTA_TESTS={len(sub)}"
    )

    report.append(
        f"WEEK{wk}_WELCH_BH_Q_LT_0_05={int((sub['welch_q_BH_within_week'] < 0.05).sum())}"
    )

    report.append(
        f"WEEK{wk}_MANN_WHITNEY_BH_Q_LT_0_05={int((sub['mann_whitney_q_BH_within_week'] < 0.05).sum())}"
    )

report.append("")
report.append(
    "MIXED_MODEL_REFIT=NO"
)
report.append(
    "NEXT_BOUNDARY=F14_R5B2_TARGETED_MIXED_MODEL_DIAGNOSTICS"
)

(OUT / "F14_R5B1_analysis_report.txt").write_text(
    "\n".join(report) + "\n",
    encoding="utf-8",
)

repro = {
    "analysis_label":
        "post_hoc_reviewer_requested",
    "random_seed":
        SEED,
    "strict_support_rule":
        ">=50% paired support in each BMI stratum",
    "min_paired_lt":
        min_lt,
    "min_paired_ge":
        min_ge,
    "python":
        sys.version.replace("\n", " "),
    "platform":
        platform.platform(),
    "numpy":
        np.__version__,
    "pandas":
        pd.__version__,
    "scipy":
        scipy.__version__,
    "matplotlib":
        matplotlib.__version__,
}

(OUT / "F14_R5B1_reproducibility.json").write_text(
    json.dumps(
        repro,
        indent=2,
    ) + "\n",
    encoding="utf-8",
)

print("\n".join(report))
