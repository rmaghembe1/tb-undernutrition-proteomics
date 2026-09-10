#!/usr/bin/env python3
"""
Phase v0.8.3 — final nutritional-metabolic recovery-axis adjudication.

Compares:
1. Existing four-module candidate PC1.
2. Three-module metabolic-core PC1.
3. Unique-protein-union PC1 for the three-module core.
4. Unique-protein-union PC1 for the core plus redox/ECM extension.

The aim is to avoid double-counting shared proteins and determine whether a
stable serum nutritional-metabolic recovery axis can be retained.

This phase is additive and does not overwrite v0.7.1, v0.8.1, v0.8.2.1,
or v0.8.2.2.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import traceback
import warnings as pywarnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats

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
INPUT_V08 = "phase_v0_8_nutritional_metabolic_expansion"
INPUT_V0822 = "phase_v0_8_2_2_module_coherence_and_integration"
OUTPUT_TAG = "phase_v0_8_3_final_recovery_axis_adjudication"
SCRIPT_VERSION = "0.8.3"

CORE_MODULES = [
    "legacy__m13_curated_endocrine_immunometabolic",
    "legacy__m03b_curated_nutrition_carrier_broad",
    "legacy__m02_curated_lipid_apolipoprotein_transport",
]
REDOX_MODULE = "vitamin_c_redox_collagen_ecm_repair"
REDOX_SCORE = "vitamin_c_redox_collagen_ecm_repair__unique_only_mean_z"
EXISTING_CANDIDATE = "candidate_integrated_nutritional_metabolic_pc1"

AXIS_COLUMNS = [
    "existing_four_module_candidate_pc1",
    "three_module_core_score_pc1",
    "three_module_core_unique_protein_pc1",
    "core_plus_redox_unique_protein_pc1",
]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize(value: Any) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def bh(values: Iterable[Any]) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(float)
    out = np.full(arr.shape, np.nan)
    mask = np.isfinite(arr)
    if not mask.any():
        return out
    if HAVE_STATSMODELS:
        out[mask] = multipletests(arr[mask], method="fdr_bh")[1]
        return out
    p = arr[mask]
    order = np.argsort(p)
    ranked = p[order]
    n = len(ranked)
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    unorder = np.empty_like(adj)
    unorder[order] = adj
    out[mask] = unorder
    return out


def zscore_frame(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.apply(pd.to_numeric, errors="coerce")
    means = numeric.mean(axis=0)
    sds = numeric.std(axis=0, ddof=0).replace(0, np.nan)
    return numeric.subtract(means, axis=1).divide(sds, axis=1)


def pca_first_component(
    df: pd.DataFrame,
    orientation_reference: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series, float]:
    z = zscore_frame(df)
    keep = z.columns[z.notna().sum() >= 3]
    z = z[keep]
    if z.shape[1] < 2:
        return (
            pd.Series(np.nan, index=df.index),
            pd.Series(dtype=float),
            np.nan,
        )
    filled = z.fillna(z.mean())
    x = filled.to_numpy(float)
    x = x - x.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    pc1 = pd.Series(u[:, 0] * s[0], index=df.index)
    loadings = pd.Series(vt[0, :], index=keep)
    explained = float(
        (s[0] ** 2) / np.sum(s ** 2)
        if np.sum(s ** 2) > 0 else np.nan
    )

    if orientation_reference is None:
        orientation_reference = z.mean(axis=1, skipna=True)
    pair = pd.concat(
        [pc1.rename("pc1"), orientation_reference.rename("reference")],
        axis=1,
    ).dropna()
    if len(pair) >= 3:
        rho = pair["pc1"].corr(pair["reference"], method="spearman")
        if pd.notna(rho) and rho < 0:
            pc1 = -pc1
            loadings = -loadings

    return pc1, loadings, explained


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, default=str),
        encoding="utf-8",
    )


def setup_logger(path: Path, verbose: bool) -> logging.Logger:
    logger = logging.getLogger("phase_v0_8_3")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t", low_memory=False)


def read_matrix(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(
        path,
        sep="\t",
        compression="infer",
        low_memory=False,
    )
    sample_col = "sample_id" if "sample_id" in df.columns else df.columns[0]
    df[sample_col] = df[sample_col].astype(str)
    return (
        df.drop_duplicates(sample_col)
        .set_index(sample_col)
        .apply(pd.to_numeric, errors="coerce")
    )


def parse_feature_list(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    return sorted({
        x.strip()
        for x in str(value).split(";")
        if x.strip()
    })


def feature_sets(
    legacy_audit: pd.DataFrame,
    membership: pd.DataFrame,
) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}

    for _, row in legacy_audit.iterrows():
        module_id = str(row["module_id"])
        out[module_id] = set(parse_feature_list(row["matched_features"]))

    member = membership.copy()
    if "used_in_score" in member.columns:
        use = (
            member["used_in_score"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
        )
        member = member[use]

    for module_id, sub in member.groupby("module_id"):
        out[str(module_id)] = set(
            sub["matrix_feature"].dropna().astype(str)
        )

    return out


def build_feature_membership(
    feature_map: dict[str, set[str]],
    modules: list[str],
    axis_family: str,
) -> pd.DataFrame:
    all_features = sorted(
        set().union(*(feature_map.get(x, set()) for x in modules))
    )
    rows = []
    for feature in all_features:
        owners = [
            module
            for module in modules
            if feature in feature_map.get(module, set())
        ]
        rows.append({
            "axis_family": axis_family,
            "matrix_feature": feature,
            "n_source_modules": len(owners),
            "source_modules": ";".join(owners),
            "shared_between_selected_modules": len(owners) > 1,
        })
    return pd.DataFrame(rows)


def module_pc1(
    scores: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.Series, pd.DataFrame, float]:
    missing = [x for x in columns if x not in scores.columns]
    if missing:
        raise KeyError(
            "Missing module-score columns: " + ", ".join(missing)
        )
    orientation = zscore_frame(scores[columns]).mean(axis=1)
    pc1, loadings, explained = pca_first_component(
        scores[columns],
        orientation,
    )
    loading_df = pd.DataFrame({
        "component_type": "module_score_pc1",
        "feature_or_module": loadings.index,
        "pc1_loading": loadings.values,
    })
    return pc1, loading_df, explained


def protein_union_pc1(
    matrix: pd.DataFrame,
    features: list[str],
    orientation_reference: pd.Series,
    component_type: str,
) -> tuple[pd.Series, pd.DataFrame, float]:
    present = [x for x in features if x in matrix.columns]
    if len(present) < 3:
        raise ValueError(
            f"{component_type} has fewer than three matrix proteins."
        )
    pc1, loadings, explained = pca_first_component(
        matrix[present],
        orientation_reference,
    )
    loading_df = pd.DataFrame({
        "component_type": component_type,
        "feature_or_module": loadings.index,
        "pc1_loading": loadings.values,
    })
    return pc1, loading_df, explained


def axis_correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, a in enumerate(AXIS_COLUMNS):
        for b in AXIS_COLUMNS[i + 1:]:
            pair = df[[a, b]].dropna()
            if len(pair) >= 3:
                rho, p = stats.spearmanr(pair[a], pair[b])
            else:
                rho = p = np.nan
            rows.append({
                "axis_a": a,
                "axis_b": b,
                "n_complete": len(pair),
                "spearman_rho": rho,
                "p_value": p,
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["fdr_bh"] = bh(out["p_value"])
    return out


def baseline_validation(
    df: pd.DataFrame,
    axes: list[str],
) -> pd.DataFrame:
    baseline = sorted(pd.unique(df["timepoint"].dropna()))[0]
    base = df[df["timepoint"] == baseline].copy()
    rows = []
    for axis in axes:
        x = pd.to_numeric(
            base.loc[base["bmi_group"] == "BMI<18.5", axis],
            errors="coerce",
        ).dropna()
        y = pd.to_numeric(
            base.loc[base["bmi_group"] == "BMI>=18.5", axis],
            errors="coerce",
        ).dropna()
        if len(x) >= 3 and len(y) >= 3:
            t, p_group = stats.ttest_ind(
                x,
                y,
                equal_var=False,
                nan_policy="omit",
            )
        else:
            t = p_group = np.nan

        pair = base[["bmi", axis]].dropna()
        if len(pair) >= 3:
            rho, p_cont = stats.spearmanr(
                pair["bmi"],
                pair[axis],
            )
        else:
            rho = p_cont = np.nan

        rows.append({
            "axis": axis,
            "baseline_timepoint": baseline,
            "n_bmi_lt_18_5": len(x),
            "n_bmi_ge_18_5": len(y),
            "mean_bmi_lt_18_5": x.mean() if len(x) else np.nan,
            "mean_bmi_ge_18_5": y.mean() if len(y) else np.nan,
            "welch_t": t,
            "bmi_group_p_value": p_group,
            "n_continuous_bmi": len(pair),
            "spearman_rho_with_bmi": rho,
            "continuous_bmi_p_value": p_cont,
            "supports_undernutrition_naming": (
                (pd.notna(p_group) and p_group < 0.05)
                or (pd.notna(p_cont) and p_cont < 0.05)
            ),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["bmi_group_fdr_bh"] = bh(out["bmi_group_p_value"])
        out["continuous_bmi_fdr_bh"] = bh(
            out["continuous_bmi_p_value"]
        )
    return out


def paired_changes(
    df: pd.DataFrame,
    axes: list[str],
) -> pd.DataFrame:
    baseline = sorted(pd.unique(df["timepoint"].dropna()))[0]
    rows = []
    for axis in axes:
        for group in ["BMI<18.5", "BMI>=18.5"]:
            sub = df[df["bmi_group"] == group]
            base = sub[sub["timepoint"] == baseline][
                ["subject_id", axis]
            ].dropna()
            for tp in sorted(pd.unique(sub["timepoint"].dropna())):
                if tp == baseline:
                    continue
                post = sub[sub["timepoint"] == tp][
                    ["subject_id", axis]
                ].dropna()
                merged = base.merge(
                    post,
                    on="subject_id",
                    suffixes=("_baseline", "_post"),
                )
                if len(merged) < 3:
                    continue
                delta = (
                    merged[f"{axis}_post"]
                    - merged[f"{axis}_baseline"]
                )
                t, p = stats.ttest_1samp(
                    delta,
                    0.0,
                    nan_policy="omit",
                )
                rows.append({
                    "axis": axis,
                    "bmi_group": group,
                    "baseline_timepoint": baseline,
                    "post_timepoint": tp,
                    "n": len(delta),
                    "mean_delta": delta.mean(),
                    "sd_delta": delta.std(ddof=1),
                    "paired_effect_size": (
                        delta.mean() / delta.std(ddof=1)
                        if delta.std(ddof=1) > 0 else np.nan
                    ),
                    "t_statistic": t,
                    "p_value": p,
                })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["fdr_bh"] = bh(out["p_value"])
    return out


def fit_interactions(
    df: pd.DataFrame,
    axes: list[str],
    family: str,
) -> pd.DataFrame:
    if not HAVE_STATSMODELS:
        return pd.DataFrame()

    rows = []
    for axis in axes:
        dat = df[[
            "subject_id",
            "timepoint",
            "bmi_group",
            "bmi",
            axis,
        ]].copy()
        dat = dat.rename(columns={axis: "outcome"})
        dat["outcome"] = pd.to_numeric(
            dat["outcome"],
            errors="coerce",
        )
        dat = dat.dropna(
            subset=["outcome", "subject_id", "timepoint"]
        )
        if family == "continuous_bmi_by_time":
            dat = dat.dropna(subset=["bmi"])
            formula = "outcome ~ bmi * C(timepoint)"
        else:
            formula = "outcome ~ C(bmi_group) * C(timepoint)"

        if len(dat) < 20:
            continue

        try:
            with pywarnings.catch_warnings():
                pywarnings.simplefilter("ignore")
                fit = smf.mixedlm(
                    formula,
                    dat,
                    groups=dat["subject_id"],
                ).fit(
                    reml=False,
                    method="lbfgs",
                    maxiter=500,
                    disp=False,
                )
            if hasattr(fit, "converged") and not fit.converged:
                raise RuntimeError("Mixed model did not converge.")
            fit_type = "mixedlm_random_intercept"
        except Exception:
            fit = smf.ols(
                formula,
                dat,
            ).fit(
                cov_type="cluster",
                cov_kwds={"groups": dat["subject_id"]},
            )
            fit_type = "ols_cluster_robust_subject"

        for term in fit.params.index:
            if family == "continuous_bmi_by_time":
                keep = (
                    ":" in term
                    and "bmi" in term
                    and "timepoint" in term
                )
            else:
                keep = (
                    ":" in term
                    and "bmi_group" in term
                    and "timepoint" in term
                )
            if not keep:
                continue
            rows.append({
                "test_family": family,
                "axis": axis,
                "fit_type": fit_type,
                "n_observations": int(
                    getattr(fit, "nobs", len(dat))
                ),
                "n_subjects": int(dat["subject_id"].nunique()),
                "term": term,
                "estimate": float(fit.params[term]),
                "std_error": float(fit.bse[term]),
                "test_statistic": float(fit.tvalues[term]),
                "p_value": float(fit.pvalues[term]),
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["fdr_bh"] = bh(out["p_value"])
    return out


def trajectory_summary(
    df: pd.DataFrame,
    axes: list[str],
) -> pd.DataFrame:
    rows = []
    for axis in axes:
        for (group, tp), sub in df.groupby(
            ["bmi_group", "timepoint"],
            dropna=False,
        ):
            values = pd.to_numeric(
                sub[axis],
                errors="coerce",
            ).dropna()
            rows.append({
                "axis": axis,
                "bmi_group": group,
                "timepoint": tp,
                "n": len(values),
                "mean": values.mean() if len(values) else np.nan,
                "sd": values.std(ddof=1) if len(values) > 1 else np.nan,
                "sem": values.sem(ddof=1) if len(values) > 1 else np.nan,
                "median": values.median() if len(values) else np.nan,
                "q1": values.quantile(0.25) if len(values) else np.nan,
                "q3": values.quantile(0.75) if len(values) else np.nan,
            })
    return pd.DataFrame(rows)


def final_decision(
    comparison: pd.DataFrame,
    baseline: pd.DataFrame,
    group_models: pd.DataFrame,
    cont_models: pd.DataFrame,
    explained: dict[str, float],
    loadings: pd.DataFrame,
) -> pd.DataFrame:
    corr_lookup = {
        frozenset([row["axis_a"], row["axis_b"]]):
        row["spearman_rho"]
        for _, row in comparison.iterrows()
    }

    core_module_vs_union = corr_lookup.get(frozenset([
        "three_module_core_score_pc1",
        "three_module_core_unique_protein_pc1",
    ]), np.nan)
    core_vs_extended = corr_lookup.get(frozenset([
        "three_module_core_unique_protein_pc1",
        "core_plus_redox_unique_protein_pc1",
    ]), np.nan)

    core_module_loadings = loadings[
        loadings["component_type"] == "three_module_core_score_pc1"
    ]["pc1_loading"]
    same_sign = (
        len(core_module_loadings) == 3
        and (
            (core_module_loadings > 0).all()
            or (core_module_loadings < 0).all()
        )
    )
    minimum_abs_loading = (
        core_module_loadings.abs().min()
        if len(core_module_loadings) else np.nan
    )

    baseline_core = baseline[
        baseline["axis"] == "three_module_core_unique_protein_pc1"
    ]
    supports_under = (
        bool(baseline_core["supports_undernutrition_naming"].iloc[0])
        if len(baseline_core) else False
    )

    group_sig = (
        pd.to_numeric(
            group_models.loc[
                group_models["axis"]
                == "three_module_core_unique_protein_pc1",
                "fdr_bh",
            ],
            errors="coerce",
        ) < 0.05
    ).any() if not group_models.empty else False

    cont_sig = (
        pd.to_numeric(
            cont_models.loc[
                cont_models["axis"]
                == "three_module_core_unique_protein_pc1",
                "fdr_bh",
            ],
            errors="coerce",
        ) < 0.05
    ).any() if not cont_models.empty else False

    stable = (
        pd.notna(core_module_vs_union)
        and core_module_vs_union >= 0.80
        and pd.notna(core_vs_extended)
        and core_vs_extended >= 0.80
        and same_sign
        and pd.notna(minimum_abs_loading)
        and minimum_abs_loading >= 0.30
        and explained.get(
            "three_module_core_score_pc1",
            0.0,
        ) >= 0.50
    )

    if stable:
        preferred_axis = "three_module_core_unique_protein_pc1"
        retain_axis = True
        manuscript_name = "serum nutritional-metabolic recovery axis"
    else:
        preferred_axis = ""
        retain_axis = False
        manuscript_name = "retain component modules separately"

    if supports_under:
        undernutrition_label = (
            "baseline BMI association detected; undernutrition-linked "
            "terminology may be considered cautiously"
        )
    else:
        undernutrition_label = (
            "no baseline BMI association; do not call this an "
            "undernutrition axis"
        )

    if group_sig or cont_sig:
        interaction_label = (
            "BMI-related interaction detected after FDR correction"
        )
    else:
        interaction_label = (
            "no BMI-related interaction after FDR correction"
        )

    return pd.DataFrame([{
        "retain_integrated_axis": retain_axis,
        "preferred_axis": preferred_axis,
        "recommended_manuscript_name": manuscript_name,
        "core_module_pc1_vs_unique_protein_pc1_rho": core_module_vs_union,
        "core_unique_protein_vs_extended_unique_protein_rho": core_vs_extended,
        "core_module_loadings_same_sign": same_sign,
        "minimum_absolute_core_module_loading": minimum_abs_loading,
        "core_module_pc1_variance_explained": explained.get(
            "three_module_core_score_pc1",
            np.nan,
        ),
        "core_unique_protein_pc1_variance_explained": explained.get(
            "three_module_core_unique_protein_pc1",
            np.nan,
        ),
        "supports_undernutrition_naming": supports_under,
        "undernutrition_naming_decision": undernutrition_label,
        "bmi_interaction_decision": interaction_label,
        "redox_ecm_role": (
            "secondary extension/sensitivity component; not part of "
            "the primary metabolic core"
        ),
    }])


def plot_trajectory(
    df: pd.DataFrame,
    axis: str,
    path: Path,
    title: str,
) -> None:
    if not HAVE_MATPLOTLIB:
        return
    timepoints = sorted(pd.unique(df["timepoint"].dropna()))
    fig, ax = plt.subplots(figsize=(8, 5))
    for group in ["BMI<18.5", "BMI>=18.5"]:
        sub = df[df["bmi_group"] == group]
        means = []
        sems = []
        for tp in timepoints:
            values = pd.to_numeric(
                sub.loc[sub["timepoint"] == tp, axis],
                errors="coerce",
            ).dropna()
            means.append(values.mean() if len(values) else np.nan)
            sems.append(
                values.sem(ddof=1) if len(values) > 1 else np.nan
            )
        ax.errorbar(
            range(len(timepoints)),
            means,
            yerr=sems,
            marker="o",
            capsize=3,
            label=group,
        )
    ax.set_xticks(range(len(timepoints)))
    ax.set_xticklabels([str(x) for x in timepoints])
    ax.set_xlabel("Treatment week")
    ax.set_ylabel("PC1 score")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_loadings(
    loadings: pd.DataFrame,
    component_type: str,
    path: Path,
    top_n: int = 20,
) -> None:
    if not HAVE_MATPLOTLIB:
        return
    sub = loadings[
        loadings["component_type"] == component_type
    ].copy()
    if sub.empty:
        return
    sub["abs_loading"] = sub["pc1_loading"].abs()
    sub = sub.nlargest(top_n, "abs_loading").sort_values(
        "pc1_loading"
    )
    fig, ax = plt.subplots(
        figsize=(9, max(5, 0.35 * len(sub)))
    )
    ax.barh(
        range(len(sub)),
        sub["pc1_loading"],
    )
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(
        sub["feature_or_module"],
        fontsize=7,
    )
    ax.set_xlabel("PC1 loading")
    ax.set_title("Top loadings: serum nutritional-metabolic core")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_axis_comparison(
    df: pd.DataFrame,
    x: str,
    y: str,
    path: Path,
) -> None:
    if not HAVE_MATPLOTLIB:
        return
    pair = df[[x, y]].dropna()
    if len(pair) < 3:
        return
    rho, _ = stats.spearmanr(pair[x], pair[y])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pair[x], pair[y], alpha=0.75)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"Axis sensitivity comparison; Spearman rho={rho:.3f}")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(
    path: Path,
    decision: pd.DataFrame,
    explained: dict[str, float],
    baseline: pd.DataFrame,
) -> None:
    row = decision.iloc[0].to_dict()
    baseline_core = baseline[
        baseline["axis"] == "three_module_core_unique_protein_pc1"
    ]
    baseline_text = (
        baseline_core.to_markdown(index=False)
        if not baseline_core.empty
        else "No baseline validation result."
    )
    report = f"""# Phase v0.8.3 final recovery-axis adjudication

Generated: {now_utc()}

## Preservation

- v0.7.1 remains the preserved co-author-review manuscript baseline.
- v0.8.1, v0.8.2.1 and v0.8.2.2 remain preserved.
- Phase v0.8.3 writes only to `{OUTPUT_TAG}` directories.

## Axis constructions compared

1. Existing four-module candidate score.
2. Three-module score-level metabolic-core PC1.
3. Three-module unique-protein-union PC1.
4. Core-plus-redox unique-protein-union PC1.

The three-module core contains:

- curated endocrine and systemic immunometabolic signalling;
- curated broad nutrition-carrier biology;
- curated lipid, cholesterol and apolipoprotein transport.

The redox/ECM module is treated as a sensitivity extension because its prior
protein-level coherence was weak and its loading on the four-module candidate
axis was small.

## Variance explained

```json
{json.dumps(explained, indent=2, default=str)}
```

## Final decision

```json
{json.dumps(row, indent=2, default=str)}
```

## Baseline BMI validation of the preferred core axis

{baseline_text}

## Interpretation

The integrated axis may be retained only if score-level and unique-protein
constructions agree strongly, module loadings are directionally coherent, and
the result is insensitive to exclusion of the heterogeneous redox/ECM
extension.

Unless baseline BMI association is demonstrated, the axis should be called a
**serum nutritional-metabolic recovery axis**, not an undernutrition score or
metabolic-undernutrition axis.

Absence of BMI-group or continuous-BMI interaction means that the axis captures
shared treatment-associated recovery rather than a formally demonstrated
BMI-dependent recovery lag.
"""
    path.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=EXPECTED_ROOT,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
    )
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

    in_v08 = root / "02_tables" / INPUT_V08
    in_v0822 = root / "02_tables" / INPUT_V0822
    out_tables = root / "02_tables" / OUTPUT_TAG
    out_results = root / "05_results" / OUTPUT_TAG
    out_figures = (
        root / "03_figures" / "draft_panels" / OUTPUT_TAG
    )
    out_logs = root / "05_docs_decision_logs" / OUTPUT_TAG

    for path in [
        out_tables,
        out_results,
        out_figures,
        out_logs,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(
        out_logs / f"phase_v0_8_3_run_{run_id}.log",
        args.verbose,
    )
    logger.info("v0.7.1 remains preserved.")
    logger.info("Starting additive Phase v0.8.3.")

    matrix = read_matrix(
        in_v08
        / "phase_v0_8_normalized_sample_by_protein_matrix.tsv.gz"
    )
    combined = read_tsv(
        in_v0822
        / "phase_v0_8_2_combined_sample_level_scores.tsv"
    ).reset_index(drop=True)
    legacy_audit = read_tsv(
        in_v0822
        / "phase_v0_8_2_legacy_module_coverage_audit.tsv"
    )
    membership = read_tsv(
        in_v08
        / "phase_v0_8_module_score_membership.tsv"
    )

    combined["sample_id"] = combined["sample_id"].astype(str)
    combined["subject_id"] = combined["subject_id"].astype(str)
    combined["timepoint"] = pd.to_numeric(
        combined["timepoint"],
        errors="coerce",
    )
    combined["bmi"] = pd.to_numeric(
        combined["bmi"],
        errors="coerce",
    )
    matrix.index = matrix.index.astype(str)

    feature_map = feature_sets(
        legacy_audit,
        membership,
    )

    missing_core = [
        module for module in CORE_MODULES
        if module not in combined.columns
    ]
    if missing_core:
        raise KeyError(
            "Missing core module scores: " + ", ".join(missing_core)
        )
    if EXISTING_CANDIDATE not in combined.columns:
        raise KeyError(
            f"Missing existing candidate score: {EXISTING_CANDIDATE}"
        )

    core_membership = build_feature_membership(
        feature_map,
        CORE_MODULES,
        "three_module_core",
    )
    extended_membership = build_feature_membership(
        feature_map,
        CORE_MODULES + [REDOX_MODULE],
        "core_plus_redox_extension",
    )
    membership_out = pd.concat(
        [core_membership, extended_membership],
        ignore_index=True,
    )
    write_tsv(
        membership_out,
        out_tables / "phase_v0_8_3_axis_feature_membership.tsv",
    )

    core_features = sorted(
        set().union(*(feature_map[x] for x in CORE_MODULES))
    )
    extended_features = sorted(
        set(core_features) | feature_map.get(REDOX_MODULE, set())
    )

    score_indexed = combined.set_index("sample_id", drop=False)
    common_samples = [
        x for x in score_indexed.index
        if x in matrix.index
    ]
    score_indexed = score_indexed.loc[common_samples].copy()
    matrix_aligned = matrix.loc[common_samples].copy()

    core_module_reference = zscore_frame(
        score_indexed[CORE_MODULES]
    ).mean(axis=1)

    core_module_pc1, core_module_loadings, core_module_var = (
        module_pc1(
            score_indexed,
            CORE_MODULES,
        )
    )
    core_module_loadings["component_type"] = (
        "three_module_core_score_pc1"
    )

    core_protein_pc1, core_protein_loadings, core_protein_var = (
        protein_union_pc1(
            matrix_aligned,
            core_features,
            core_module_reference,
            "three_module_core_unique_protein_pc1",
        )
    )

    if REDOX_SCORE in score_indexed.columns:
        extended_reference = zscore_frame(
            score_indexed[CORE_MODULES + [REDOX_SCORE]]
        ).mean(axis=1)
    else:
        extended_reference = core_module_reference

    (
        extended_protein_pc1,
        extended_protein_loadings,
        extended_protein_var,
    ) = protein_union_pc1(
        matrix_aligned,
        extended_features,
        extended_reference,
        "core_plus_redox_unique_protein_pc1",
    )

    axes = score_indexed[[
        "sample_id",
        "subject_id",
        "timepoint",
        "bmi",
        "bmi_group",
    ]].copy()

    axes["existing_four_module_candidate_pc1"] = (
        score_indexed[EXISTING_CANDIDATE]
    )
    axes["three_module_core_score_pc1"] = core_module_pc1
    axes["three_module_core_unique_protein_pc1"] = core_protein_pc1
    axes["core_plus_redox_unique_protein_pc1"] = (
        extended_protein_pc1
    )
    axes = axes.reset_index(drop=True)

    loadings = pd.concat(
        [
            core_module_loadings,
            core_protein_loadings,
            extended_protein_loadings,
        ],
        ignore_index=True,
    )

    explained = {
        "three_module_core_score_pc1": core_module_var,
        "three_module_core_unique_protein_pc1": core_protein_var,
        "core_plus_redox_unique_protein_pc1": extended_protein_var,
        "existing_four_module_candidate_pc1": (
            0.6621191921322359
        ),
        "n_unique_core_proteins": len(core_features),
        "n_unique_extended_proteins": len(extended_features),
    }

    comparison = axis_correlation_table(axes)
    baseline = baseline_validation(axes, AXIS_COLUMNS)
    changes = paired_changes(axes, AXIS_COLUMNS)
    group_models = fit_interactions(
        axes,
        AXIS_COLUMNS,
        "bmi_group_by_time",
    )
    cont_models = fit_interactions(
        axes,
        AXIS_COLUMNS,
        "continuous_bmi_by_time",
    )
    trajectories = trajectory_summary(
        axes,
        AXIS_COLUMNS,
    )
    decision = final_decision(
        comparison,
        baseline,
        group_models,
        cont_models,
        explained,
        loadings,
    )

    write_tsv(
        axes,
        out_tables / "phase_v0_8_3_axis_scores.tsv",
    )
    write_tsv(
        loadings,
        out_tables / "phase_v0_8_3_axis_loadings.tsv",
    )
    write_tsv(
        comparison,
        out_tables / "phase_v0_8_3_axis_correlation_comparison.tsv",
    )
    write_tsv(
        baseline,
        out_tables / "phase_v0_8_3_axis_baseline_bmi_validation.tsv",
    )
    write_tsv(
        changes,
        out_tables / "phase_v0_8_3_axis_within_group_changes.tsv",
    )
    write_tsv(
        group_models,
        out_tables / "phase_v0_8_3_axis_bmi_group_time_models.tsv",
    )
    write_tsv(
        cont_models,
        out_tables / "phase_v0_8_3_axis_continuous_bmi_time_models.tsv",
    )
    write_tsv(
        trajectories,
        out_tables / "phase_v0_8_3_axis_trajectory_summary.tsv",
    )
    write_tsv(
        decision,
        out_tables / "phase_v0_8_3_final_axis_decision.tsv",
    )

    plot_trajectory(
        axes,
        "three_module_core_unique_protein_pc1",
        out_figures
        / "phase_v0_8_3_core_recovery_axis_trajectory.png",
        "Serum nutritional-metabolic core recovery axis",
    )
    plot_loadings(
        loadings,
        "three_module_core_unique_protein_pc1",
        out_figures
        / "phase_v0_8_3_core_recovery_axis_top_protein_loadings.png",
    )
    plot_loadings(
        loadings,
        "three_module_core_score_pc1",
        out_figures
        / "phase_v0_8_3_core_recovery_axis_module_loadings.png",
        top_n=10,
    )
    plot_axis_comparison(
        axes,
        "three_module_core_unique_protein_pc1",
        "core_plus_redox_unique_protein_pc1",
        out_figures
        / "phase_v0_8_3_core_vs_redox_extended_axis.png",
    )
    plot_axis_comparison(
        axes,
        "three_module_core_score_pc1",
        "three_module_core_unique_protein_pc1",
        out_figures
        / "phase_v0_8_3_module_score_vs_unique_protein_axis.png",
    )

    manifest = {
        "script_version": SCRIPT_VERSION,
        "generated_utc": now_utc(),
        "baseline_preserved": "v0.7.1",
        "input_phases": [INPUT_V08, INPUT_V0822],
        "output_phase": OUTPUT_TAG,
        "n_samples": len(axes),
        "n_subjects": axes["subject_id"].nunique(),
        "core_modules": CORE_MODULES,
        "redox_extension_module": REDOX_MODULE,
        "variance_explained": explained,
        "final_decision": decision.iloc[0].to_dict(),
    }
    write_json(
        manifest,
        out_results / "phase_v0_8_3_run_manifest.json",
    )
    write_report(
        out_results
        / "phase_v0_8_3_final_recovery_axis_report.md",
        decision,
        explained,
        baseline,
    )

    decision_log = f"""# Phase v0.8.3 decision log

Generated: {now_utc()}

## Preserved analyses

- v0.7.1 manuscript baseline.
- v0.8.1 nutritional-metabolic feasibility analysis.
- v0.8.2.1 coherence and vitamin D proxy dissection.
- v0.8.2.2 corrected legacy integration and candidate latent score.

## Primary analytical question

Does a stable serum nutritional-metabolic recovery axis remain after:

1. removing the weak heterogeneous redox/ECM extension;
2. avoiding double-counting proteins shared across lipid, nutrition-carrier and
   endocrine modules; and
3. comparing module-score PCA with PCA across the union of unique proteins?

## Naming rule

- Use **serum nutritional-metabolic recovery axis** if the core construction is
  stable.
- Use undernutrition-linked terminology only if the baseline axis is associated
  with baseline BMI after appropriate correction.
- Do not claim BMI-dependent recovery unless BMI-group or continuous-BMI
  interactions survive BH-FDR.

## Redox/ECM rule

The redox/ECM module remains a secondary extension or separate module because
its protein-level coherence was weak in v0.8.2.1.
"""
    (
        out_logs / "phase_v0_8_3_decision_log.md"
    ).write_text(decision_log, encoding="utf-8")

    logger.info("Phase v0.8.3 completed.")
    logger.info("Tables: %s", out_tables)
    logger.info("Results: %s", out_results)
    logger.info("Figures: %s", out_figures)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            f"FATAL: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        traceback.print_exc()
        raise
