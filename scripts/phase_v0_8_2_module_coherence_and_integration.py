#!/usr/bin/env python3
"""
Phase v0.8.2 — module coherence, overlap-sensitive rescoring, explicit legacy
module integration, vitamin-D-proxy dissection, and candidate integrated
nutritional-metabolic latent score.

This is an additive analysis. It does not overwrite Phase v0.7.1 or v0.8.1.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import traceback
import warnings as pywarnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

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
INPUT_TAG = "phase_v0_8_nutritional_metabolic_expansion"
OUTPUT_TAG = "phase_v0_8_2_2_module_coherence_and_integration"
SCRIPT_VERSION = "0.8.2.2"
MIN_MODULE_PROTEINS = 3
MIN_GROUP_N = 3
BMI_CUTOFF = 18.5

VITD_MODULE = "vitamin_d_linked_antimicrobial_autophagy_phagolysosome"

CORE_NEW_MODULES = [
    "vitamin_d_linked_antimicrobial_autophagy_phagolysosome",
    "cytoskeleton_actin_immune_synapse_phagocytosis",
    "contractile_tissue_remodelling_markers",
    "retinoic_acid_signalling_lipid_cholesterol_trafficking",
    "vitamin_c_redox_collagen_ecm_repair",
]

INTEGRATED_SCORE_POSITIVE_KEYWORDS = [
    "lipid", "apolipoprotein", "retinol", "retinoic", "cholesterol",
    "endocrine", "metabolic", "nutrient", "vitamin", "redox", "carrier",
]
INTEGRATED_SCORE_EXCLUDE_KEYWORDS = [
    "inflamm", "acute_phase", "acute phase", "cytoskeleton", "contractile",
    "phagocyt", "autophagy", "lysosome", "antimicrobial", "metal",
    "nutritional_immunity", "nutritional immunity",
]

LEGACY_MAPPING_CANDIDATES = [
    "resources/curated_legacy_module_mapping.tsv",
]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize(value: Any) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def symbol_tokens(value: Any) -> set[str]:
    if pd.isna(value):
        return set()
    text = str(value).upper()
    tokens = set(re.findall(r"\b[A-Z0-9][A-Z0-9_.-]{1,30}\b", text))
    expanded = set(tokens)
    for token in list(tokens):
        expanded.add(token.split("_")[0])
        expanded.add(token.split(".")[0])
        expanded.add(token.split("-")[0])
    return {x for x in expanded if x}


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
    """Vectorized column-wise z-scores without DataFrame fragmentation."""
    numeric = df.apply(pd.to_numeric, errors="coerce")
    means = numeric.mean(axis=0)
    sds = numeric.std(axis=0, ddof=0).replace(0, np.nan)
    return numeric.subtract(means, axis=1).divide(sds, axis=1)


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def setup_logger(path: Path, verbose: bool) -> logging.Logger:
    logger = logging.getLogger("phase_v0_8_2")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t", low_memory=False)


def read_matrix(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep="\t", compression="infer", low_memory=False)
    sample_col = "sample_id" if "sample_id" in df.columns else df.columns[0]
    df[sample_col] = df[sample_col].astype(str)
    df = df.drop_duplicates(sample_col).set_index(sample_col)
    return df.apply(pd.to_numeric, errors="coerce")


def cronbach_alpha(df: pd.DataFrame) -> float:
    clean = df.dropna(axis=0, how="any")
    k = clean.shape[1]
    if k < 2 or clean.shape[0] < 3:
        return np.nan
    item_var = clean.var(axis=0, ddof=1).sum()
    total_var = clean.sum(axis=1).var(ddof=1)
    if not np.isfinite(total_var) or total_var <= 0:
        return np.nan
    return float((k / (k - 1)) * (1 - item_var / total_var))


def pca_first_component(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, float]:
    z = zscore_frame(df)
    complete_cols = z.columns[z.notna().sum() >= 3]
    z = z[complete_cols]
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
    pc1 = u[:, 0] * s[0]
    loadings = vt[0, :]
    explained = (s[0] ** 2) / np.sum(s ** 2) if np.sum(s ** 2) > 0 else np.nan

    mean_score = z.mean(axis=1, skipna=True)
    corr = pd.Series(pc1, index=df.index).corr(mean_score)
    if pd.notna(corr) and corr < 0:
        pc1 = -pc1
        loadings = -loadings

    return (
        pd.Series(pc1, index=df.index),
        pd.Series(loadings, index=complete_cols),
        float(explained),
    )


def median_pairwise_spearman(df: pd.DataFrame) -> tuple[float, float, int]:
    corr = df.corr(method="spearman")
    if corr.shape[0] < 2:
        return np.nan, np.nan, 0
    vals = corr.to_numpy()[np.triu_indices(corr.shape[0], k=1)]
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, 0
    return float(np.median(vals)), float(np.mean(vals)), int(len(vals))


def leave_one_out_stability(df: pd.DataFrame) -> tuple[float, float, pd.DataFrame]:
    z = zscore_frame(df)
    full = z.mean(axis=1, skipna=True)
    rows = []
    corrs = []
    for col in z.columns:
        remain = [x for x in z.columns if x != col]
        if len(remain) < 2:
            continue
        loo = z[remain].mean(axis=1, skipna=True)
        pair = pd.concat([full, loo], axis=1).dropna()
        rho = pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman")
        corrs.append(rho)
        rows.append({
            "removed_feature": col,
            "n_remaining_features": len(remain),
            "spearman_full_vs_leave_one_out": rho,
        })
    return (
        float(np.nanmin(corrs)) if corrs else np.nan,
        float(np.nanmedian(corrs)) if corrs else np.nan,
        pd.DataFrame(rows),
    )


def build_membership_map(membership: pd.DataFrame) -> dict[str, list[str]]:
    df = membership.copy()
    if "used_in_score" in df.columns:
        use = (
            df["used_in_score"].astype(str).str.lower()
            .isin(["true", "1", "yes"])
        )
        df = df[use]
    result = {}
    for module, sub in df.groupby("module_id"):
        result[str(module)] = sorted(set(sub["matrix_feature"].astype(str)))
    return result


def identify_shared_features(module_map: dict[str, list[str]]) -> dict[str, set[str]]:
    feature_modules: dict[str, set[str]] = defaultdict(set)
    for module, features in module_map.items():
        for feature in features:
            feature_modules[feature].add(module)
    return {
        module: {
            feature for feature in features
            if len(feature_modules[feature]) > 1
        }
        for module, features in module_map.items()
    }


def score_module_variants(
    matrix: pd.DataFrame,
    analysis: pd.DataFrame,
    module_map: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    shared = identify_shared_features(module_map)
    score_out = analysis[[
        "sample_id", "subject_id", "timepoint", "bmi", "bmi_group"
    ]].copy()
    coherence_rows = []
    loading_rows = []
    loo_rows = []

    matrix_indexed = matrix.copy()
    matrix_indexed.index = matrix_indexed.index.astype(str)

    for module, requested in module_map.items():
        features = [x for x in requested if x in matrix_indexed.columns]
        if len(features) < MIN_MODULE_PROTEINS:
            continue
        sub = matrix_indexed[features]
        median_rho, mean_rho, n_pairs = median_pairwise_spearman(sub)
        alpha = cronbach_alpha(zscore_frame(sub))
        pc1, loadings, pc1_var = pca_first_component(sub)
        min_loo, median_loo, loo = leave_one_out_stability(sub)

        unique_features = [x for x in features if x not in shared.get(module, set())]
        mean_score = zscore_frame(sub).mean(axis=1, skipna=True)
        unique_score = (
            zscore_frame(sub[unique_features]).mean(axis=1, skipna=True)
            if len(unique_features) >= MIN_MODULE_PROTEINS
            else pd.Series(np.nan, index=sub.index)
        )

        # Convert all indexed Series to plain arrays. The matrix index is named
        # "sample_id"; retaining that index name while also creating a sample_id
        # column makes pandas merge keys ambiguous.
        score_table = pd.DataFrame({
            "sample_id": sub.index.astype(str).to_numpy(),
            f"{module}__mean_z": mean_score.to_numpy(),
            f"{module}__pc1": pc1.to_numpy(),
            f"{module}__unique_only_mean_z": unique_score.to_numpy(),
        }).reset_index(drop=True)
        score_out = score_out.reset_index(drop=True).merge(
            score_table,
            on="sample_id",
            how="left",
            validate="one_to_one",
        )

        coherence_rows.append({
            "module_id": module,
            "n_features": len(features),
            "n_unique_features": len(unique_features),
            "n_shared_features": len(shared.get(module, set())),
            "shared_features": ";".join(sorted(shared.get(module, set()))),
            "median_pairwise_spearman": median_rho,
            "mean_pairwise_spearman": mean_rho,
            "n_pairwise_correlations": n_pairs,
            "cronbach_alpha_zscores": alpha,
            "pc1_variance_explained": pc1_var,
            "minimum_leave_one_out_spearman": min_loo,
            "median_leave_one_out_spearman": median_loo,
            "coherence_flag": (
                "strong"
                if (
                    pd.notna(median_rho) and median_rho >= 0.30
                    and pd.notna(pc1_var) and pc1_var >= 0.35
                    and pd.notna(min_loo) and min_loo >= 0.80
                )
                else "moderate"
                if (
                    pd.notna(median_rho) and median_rho >= 0.15
                    and pd.notna(pc1_var) and pc1_var >= 0.25
                )
                else "weak_or_heterogeneous"
            ),
        })

        for feature, loading in loadings.items():
            loading_rows.append({
                "module_id": module,
                "feature": feature,
                "pc1_loading": loading,
                "shared_between_modules": feature in shared.get(module, set()),
            })

        if not loo.empty:
            loo["module_id"] = module
            loo_rows.extend(loo.to_dict("records"))

    return (
        score_out,
        pd.DataFrame(coherence_rows),
        pd.DataFrame(loading_rows),
        pd.DataFrame(loo_rows),
    )


def exact_or_token_match(
    matrix_columns: list[str],
    values: Iterable[Any],
) -> list[str]:
    matrix_tokens = {
        col: symbol_tokens(col)
        for col in matrix_columns
    }
    targets = set()
    for value in values:
        targets.update(symbol_tokens(value))
    matched = []
    for col, tokens in matrix_tokens.items():
        if col.upper() in targets or tokens & targets:
            matched.append(col)
    return sorted(set(matched))


def load_legacy_mapping(root: Path, explicit: Optional[Path]) -> tuple[pd.DataFrame, Path]:
    if explicit:
        path = explicit.resolve()
        return read_required(path), path

    for rel in LEGACY_MAPPING_CANDIDATES:
        path = root / rel
        if path.exists():
            return read_required(path), path
    raise FileNotFoundError(
        "No curated legacy module-protein mapping file was found."
    )


def reconstruct_legacy_modules(
    matrix: pd.DataFrame,
    mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    module_col = next(
        (c for c in ["module_id", "module", "module_name"] if c in mapping.columns),
        None,
    )
    label_col = next(
        (c for c in ["module_label", "short_name", "module_family"] if c in mapping.columns),
        module_col,
    )
    candidate_feature_cols = [
        c for c in [
            "mnemonic", "gene_symbol_curated", "gene_like",
            "protein_gene_like", "protein_row_id"
        ]
        if c in mapping.columns
    ]
    if module_col is None or not candidate_feature_cols:
        raise ValueError(
            "Legacy mapping needs a module column and at least one feature/gene column."
        )

    audit_rows = []
    module_map = {}
    label_map = {}
    matrix_cols = list(matrix.columns)

    for module, sub in mapping.groupby(module_col):
        source_values = []
        for col in candidate_feature_cols:
            source_values.extend(sub[col].dropna().astype(str).tolist())
        matched = exact_or_token_match(matrix_cols, source_values)
        label = (
            str(sub[label_col].dropna().iloc[0])
            if label_col and sub[label_col].notna().any()
            else str(module)
        )
        module_id = "legacy__" + normalize(module)
        module_map[module_id] = matched
        label_map[module_id] = label
        audit_rows.append({
            "module_id": module_id,
            "source_module_id": module,
            "module_label": label,
            "n_mapping_rows": len(sub),
            "n_matched_matrix_features": len(matched),
            "matched_features": ";".join(matched),
            "scoreable": len(matched) >= MIN_MODULE_PROTEINS,
        })

    score_df = pd.DataFrame(index=matrix.index)
    membership_rows = []
    zmat = zscore_frame(matrix)
    for module_id, features in module_map.items():
        if len(features) < MIN_MODULE_PROTEINS:
            continue
        score_df[module_id] = zmat[features].mean(axis=1, skipna=True)
        for feature in features:
            membership_rows.append({
                "module_id": module_id,
                "module_label": label_map[module_id],
                "matrix_feature": feature,
            })
    score_df.index.name = "sample_id"
    return (
        pd.DataFrame(audit_rows),
        score_df,
        module_map,
    )


def fit_interactions(
    df: pd.DataFrame,
    score_cols: list[str],
    family: str,
) -> pd.DataFrame:
    if not HAVE_STATSMODELS:
        return pd.DataFrame()
    rows = []
    for score in score_cols:
        dat = df[[
            "subject_id", "timepoint", "bmi_group", "bmi", score
        ]].copy()
        dat = dat.rename(columns={score: "outcome"})
        dat["outcome"] = pd.to_numeric(dat["outcome"], errors="coerce")
        dat = dat.dropna(subset=["outcome", "subject_id", "timepoint"])
        if len(dat) < 20 or dat["timepoint"].nunique() < 2:
            continue

        formula = (
            "outcome ~ C(bmi_group) * C(timepoint)"
            if family == "bmi_group_by_time"
            else "outcome ~ bmi * C(timepoint)"
        )
        if family == "continuous_bmi_by_time":
            dat = dat.dropna(subset=["bmi"])
            if dat["bmi"].nunique() < 5:
                continue
        try:
            with pywarnings.catch_warnings():
                pywarnings.simplefilter("ignore")
                fit = smf.mixedlm(
                    formula, dat, groups=dat["subject_id"]
                ).fit(
                    reml=False, method="lbfgs", maxiter=500, disp=False
                )
            if hasattr(fit, "converged") and not fit.converged:
                raise RuntimeError("mixed model did not converge")
            fit_type = "mixedlm_random_intercept"
        except Exception:
            fit = smf.ols(formula, dat).fit(
                cov_type="cluster",
                cov_kwds={"groups": dat["subject_id"]},
            )
            fit_type = "ols_cluster_robust_subject"

        for term in fit.params.index:
            keep = (
                family == "bmi_group_by_time"
                and ":" in term and "bmi_group" in term and "timepoint" in term
            ) or (
                family == "continuous_bmi_by_time"
                and ":" in term and "bmi" in term and "timepoint" in term
            )
            if keep:
                rows.append({
                    "test_family": family,
                    "score": score,
                    "fit_type": fit_type,
                    "n_observations": int(getattr(fit, "nobs", len(dat))),
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


def paired_change_table(
    df: pd.DataFrame,
    score_cols: list[str],
) -> pd.DataFrame:
    baseline = sorted(pd.unique(df["timepoint"].dropna()))[0]
    rows = []
    for score in score_cols:
        for group in ["BMI<18.5", "BMI>=18.5"]:
            sub = df[df["bmi_group"] == group]
            base = sub[sub["timepoint"] == baseline][["subject_id", score]].dropna()
            for tp in sorted(pd.unique(sub["timepoint"].dropna())):
                if tp == baseline:
                    continue
                post = sub[sub["timepoint"] == tp][["subject_id", score]].dropna()
                merged = base.merge(post, on="subject_id", suffixes=("_baseline", "_post"))
                if len(merged) < MIN_GROUP_N:
                    continue
                delta = merged[f"{score}_post"] - merged[f"{score}_baseline"]
                t, p = stats.ttest_1samp(delta, 0.0, nan_policy="omit")
                rows.append({
                    "score": score,
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


def vitamin_d_protein_dissection(
    matrix: pd.DataFrame,
    analysis: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    protein_df = analysis[[
        "sample_id", "subject_id", "timepoint", "bmi", "bmi_group"
    ]].merge(
        matrix[features].reset_index(),
        on="sample_id",
        how="left",
    )
    paired = paired_change_table(protein_df, features)
    group_models = fit_interactions(
        protein_df, features, "bmi_group_by_time"
    )
    continuous_models = fit_interactions(
        protein_df, features, "continuous_bmi_by_time"
    )
    return paired, group_models, continuous_models


def score_column_base_module_id(column: str) -> str:
    """
    Recover the biological module ID from a score column.

    New module variants use:
      module_id__mean_z
      module_id__pc1
      module_id__unique_only_mean_z

    Legacy module IDs themselves begin with "legacy__", so splitting at the
    first double underscore incorrectly collapses every legacy module to
    "legacy". Legacy columns must therefore be retained intact.
    """
    column = str(column)
    if column.startswith("legacy__"):
        return column
    for suffix in ("__unique_only_mean_z", "__mean_z", "__pc1"):
        if column.endswith(suffix):
            return column[:-len(suffix)]
    return column


def module_label_lookup(
    coverage: pd.DataFrame,
    legacy_audit: pd.DataFrame,
) -> dict[str, str]:
    labels = {}
    if {"module_id", "module_label"}.issubset(coverage.columns):
        labels.update(
            coverage.set_index("module_id")["module_label"].astype(str).to_dict()
        )
    if {"module_id", "module_label"}.issubset(legacy_audit.columns):
        labels.update(
            legacy_audit.set_index("module_id")["module_label"].astype(str).to_dict()
        )
    return labels


def select_integrated_modules(
    score_df: pd.DataFrame,
    labels: dict[str, str],
) -> pd.DataFrame:
    rows = []
    metadata_cols = {
        "sample_id", "subject_id", "timepoint", "bmi", "bmi_group"
    }
    for col in score_df.columns:
        if col in metadata_cols:
            continue
        base_id = score_column_base_module_id(col)
        label = labels.get(base_id, base_id)
        text = normalize(label + " " + base_id)
        positive_hit = any(normalize(k) in text for k in INTEGRATED_SCORE_POSITIVE_KEYWORDS)
        excluded = any(normalize(k) in text for k in INTEGRATED_SCORE_EXCLUDE_KEYWORDS)
        selected = positive_hit and not excluded
        rows.append({
            "score_column": col,
            "base_module_id": base_id,
            "module_label": label,
            "positive_keyword_hit": positive_hit,
            "excluded_keyword_hit": excluded,
            "selected_for_candidate_integrated_score": selected,
            "selection_reason": (
                "nutritional-metabolic transport/recovery module"
                if selected
                else "excluded from candidate latent score"
            ),
        })
    return pd.DataFrame(rows)


def combined_module_feature_map(
    new_module_map: dict[str, list[str]],
    legacy_module_map: dict[str, list[str]],
) -> dict[str, set[str]]:
    combined: dict[str, set[str]] = {}
    for module_id, features in new_module_map.items():
        combined[module_id] = set(features)
    for module_id, features in legacy_module_map.items():
        combined[module_id] = set(features)
    return combined


def integrated_module_overlap_audit(
    selection: pd.DataFrame,
    feature_map: dict[str, set[str]],
) -> pd.DataFrame:
    eligible = sorted(set(
        selection.loc[
            selection["selected_for_candidate_integrated_score"],
            "base_module_id",
        ].astype(str)
    ))
    rows = []
    for i, a in enumerate(eligible):
        fa = feature_map.get(a, set())
        for b in eligible[i + 1:]:
            fb = feature_map.get(b, set())
            inter = fa & fb
            union = fa | fb
            rows.append({
                "module_a": a,
                "module_b": b,
                "n_features_a": len(fa),
                "n_features_b": len(fb),
                "n_shared_features": len(inter),
                "shared_features": ";".join(sorted(inter)),
                "jaccard_overlap": len(inter) / len(union) if union else np.nan,
                "overlap_fraction_smaller_module": (
                    len(inter) / min(len(fa), len(fb))
                    if min(len(fa), len(fb)) > 0 else np.nan
                ),
            })
    return pd.DataFrame(rows)


def choose_nonredundant_integrated_columns(
    selected_columns: list[str],
    feature_map: dict[str, set[str]],
    labels: dict[str, str],
    max_smaller_module_overlap: float = 0.60,
) -> tuple[list[str], list[dict[str, Any]]]:
    """
    Choose one score variant per module and remove near-duplicate modules.

    Preference within a module:
      unique-only mean-z > PC1 > mean-z > legacy mean-z.

    Modules are considered near-duplicates when at least 60% of the smaller
    module's proteins overlap. The broader module is retained unless the
    narrower module is a dedicated nutrition-carrier or endocrine module.
    """
    grouped: dict[str, list[str]] = defaultdict(list)
    for col in selected_columns:
        grouped[score_column_base_module_id(col)].append(col)

    preferred = []
    for base, cols in grouped.items():
        unique = [c for c in cols if c.endswith("__unique_only_mean_z")]
        pc1 = [c for c in cols if c.endswith("__pc1")]
        mean_z = [c for c in cols if c.endswith("__mean_z")]
        chosen = (
            unique[0] if unique
            else pc1[0] if pc1
            else mean_z[0] if mean_z
            else cols[0]
        )
        preferred.append(chosen)

    # Prioritize explicitly nutritional carrier/endocrine modules, then larger
    # feature sets, then stable alphabetical order.
    def priority(col: str) -> tuple[int, int, str]:
        base = score_column_base_module_id(col)
        label = normalize(labels.get(base, base))
        special = int(
            "nutrition_carrier" in label
            or "retinol_carrier" in label
            or "endocrine" in label
        )
        return (-special, -len(feature_map.get(base, set())), base)

    kept: list[str] = []
    decisions: list[dict[str, Any]] = []
    for col in sorted(preferred, key=priority):
        base = score_column_base_module_id(col)
        features = feature_map.get(base, set())
        redundant_with = None
        overlap_value = 0.0
        for kept_col in kept:
            kept_base = score_column_base_module_id(kept_col)
            kept_features = feature_map.get(kept_base, set())
            denom = min(len(features), len(kept_features))
            overlap = (
                len(features & kept_features) / denom
                if denom > 0 else 0.0
            )
            if overlap >= max_smaller_module_overlap:
                redundant_with = kept_base
                overlap_value = overlap
                break

        if redundant_with is None:
            kept.append(col)
            decisions.append({
                "score_column": col,
                "base_module_id": base,
                "decision": "retained",
                "redundant_with": "",
                "overlap_fraction_smaller_module": np.nan,
            })
        else:
            decisions.append({
                "score_column": col,
                "base_module_id": base,
                "decision": "excluded_near_duplicate",
                "redundant_with": redundant_with,
                "overlap_fraction_smaller_module": overlap_value,
            })
    return kept, decisions


def build_integrated_pc1(
    score_df: pd.DataFrame,
    selection: pd.DataFrame,
    feature_map: dict[str, set[str]],
    labels: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    selected = selection.loc[
        selection["selected_for_candidate_integrated_score"],
        "score_column",
    ].astype(str).tolist()

    final_cols, redundancy_decisions = choose_nonredundant_integrated_columns(
        selected,
        feature_map,
        labels,
    )
    redundancy_df = pd.DataFrame(redundancy_decisions)

    result = score_df.copy()
    if len(final_cols) < 3:
        return result, pd.DataFrame(), redundancy_df, {
            "created": False,
            "reason": "fewer than three eligible nonredundant modules",
            "selected_columns": final_cols,
        }

    pc1, loadings, explained = pca_first_component(result[final_cols])
    result["candidate_integrated_nutritional_metabolic_pc1"] = pc1
    loading_df = pd.DataFrame({
        "score_column": loadings.index,
        "base_module_id": [
            score_column_base_module_id(x) for x in loadings.index
        ],
        "module_label": [
            labels.get(score_column_base_module_id(x),
                       score_column_base_module_id(x))
            for x in loadings.index
        ],
        "pc1_loading": loadings.values,
    })
    return result, loading_df, redundancy_df, {
        "created": True,
        "reason": "at least three eligible nonredundant modules",
        "selected_columns": final_cols,
        "pc1_variance_explained": explained,
        "interpretation": (
            "Exploratory latent nutritional-metabolic recovery axis. "
            "Not a validated clinical undernutrition score."
        ),
    }



def integrated_score_baseline_validation(
    df: pd.DataFrame,
    score: str,
) -> pd.DataFrame:
    if score not in df.columns:
        return pd.DataFrame()
    baseline = sorted(pd.unique(df["timepoint"].dropna()))[0]
    base = df[df["timepoint"] == baseline].copy()
    x = pd.to_numeric(
        base.loc[base["bmi_group"] == "BMI<18.5", score],
        errors="coerce",
    ).dropna()
    y = pd.to_numeric(
        base.loc[base["bmi_group"] == "BMI>=18.5", score],
        errors="coerce",
    ).dropna()
    if len(x) >= MIN_GROUP_N and len(y) >= MIN_GROUP_N:
        t, p_group = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
    else:
        t = p_group = np.nan
    pair = base[["bmi", score]].dropna()
    if len(pair) >= MIN_GROUP_N:
        rho, p_rho = stats.spearmanr(pair["bmi"], pair[score])
    else:
        rho = p_rho = np.nan
    return pd.DataFrame([{
        "score": score,
        "baseline_timepoint": baseline,
        "n_bmi_lt_18_5": len(x),
        "n_bmi_ge_18_5": len(y),
        "mean_bmi_lt_18_5": x.mean() if len(x) else np.nan,
        "mean_bmi_ge_18_5": y.mean() if len(y) else np.nan,
        "welch_t": t,
        "bmi_group_p_value": p_group,
        "n_continuous_bmi": len(pair),
        "spearman_rho_with_bmi": rho,
        "continuous_bmi_p_value": p_rho,
        "naming_decision": (
            "candidate metabolic-undernutrition axis"
            if (
                (pd.notna(p_group) and p_group < 0.05)
                or (pd.notna(p_rho) and p_rho < 0.05)
            )
            else "nutritional-metabolic recovery axis only"
        ),
    }])


def correlation_edges(
    df: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [c for c in columns if c in df.columns]
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            pair = df[[a, b]].dropna()
            if len(pair) < MIN_GROUP_N:
                rho = p = np.nan
            else:
                rho, p = stats.spearmanr(pair[a], pair[b])
            rows.append({
                "score_a": a,
                "score_b": b,
                "n_complete": len(pair),
                "spearman_rho": rho,
                "p_value": p,
            })
    edges = pd.DataFrame(rows)
    if not edges.empty:
        edges["fdr_bh"] = bh(edges["p_value"])
    matrix = df[cols].corr(method="spearman") if cols else pd.DataFrame()
    return edges, matrix


def plot_coherence(coherence: pd.DataFrame, path: Path) -> None:
    if not HAVE_MATPLOTLIB or coherence.empty:
        return
    plot_df = coherence.sort_values("pc1_variance_explained")
    fig, ax = plt.subplots(figsize=(10, max(5, 0.55 * len(plot_df))))
    y = np.arange(len(plot_df))
    ax.barh(y, plot_df["pc1_variance_explained"] * 100)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["module_id"], fontsize=8)
    ax.set_xlabel("PC1 variance explained (%)")
    ax.set_title("Phase v0.8.2 module coherence")
    for i, (_, row) in enumerate(plot_df.iterrows()):
        ax.text(
            row["pc1_variance_explained"] * 100 + 0.5,
            i,
            row["coherence_flag"],
            va="center",
            fontsize=7,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(corr: pd.DataFrame, path: Path) -> None:
    if not HAVE_MATPLOTLIB or corr.empty or corr.shape[0] < 2:
        return
    n = corr.shape[0]
    size = max(8, min(20, 0.45 * n + 5))
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(corr.to_numpy(float), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
    ax.set_yticklabels(corr.index, fontsize=6)
    ax.set_title("New and legacy module-score correlations")
    fig.colorbar(im, ax=ax, label="Spearman rho", fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory(
    df: pd.DataFrame,
    score: str,
    path: Path,
    title: str,
) -> None:
    if not HAVE_MATPLOTLIB or score not in df.columns:
        return
    timepoints = sorted(pd.unique(df["timepoint"].dropna()))
    fig, ax = plt.subplots(figsize=(8, 5))
    for group in ["BMI<18.5", "BMI>=18.5"]:
        sub = df[df["bmi_group"] == group]
        means, sems = [], []
        for tp in timepoints:
            vals = pd.to_numeric(
                sub.loc[sub["timepoint"] == tp, score],
                errors="coerce",
            ).dropna()
            means.append(vals.mean() if len(vals) else np.nan)
            sems.append(vals.sem(ddof=1) if len(vals) > 1 else np.nan)
        ax.errorbar(
            np.arange(len(timepoints)),
            means,
            yerr=sems,
            marker="o",
            capsize=3,
            label=group,
        )
    ax.set_xticks(np.arange(len(timepoints)))
    ax.set_xticklabels([str(x) for x in timepoints])
    ax.set_xlabel("Treatment week")
    ax.set_ylabel("Standardized score")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(
    path: Path,
    coherence: pd.DataFrame,
    legacy_audit: pd.DataFrame,
    vitd_models: pd.DataFrame,
    integrated_manifest: dict[str, Any],
    warnings: list[str],
) -> None:
    strong = int((coherence["coherence_flag"] == "strong").sum()) if not coherence.empty else 0
    moderate = int((coherence["coherence_flag"] == "moderate").sum()) if not coherence.empty else 0
    weak = int((coherence["coherence_flag"] == "weak_or_heterogeneous").sum()) if not coherence.empty else 0
    legacy_scoreable = int(legacy_audit["scoreable"].sum()) if not legacy_audit.empty else 0
    vitd_nominal = (
        int((pd.to_numeric(vitd_models.get("p_value"), errors="coerce") < 0.05).sum())
        if not vitd_models.empty else 0
    )
    vitd_fdr = (
        int((pd.to_numeric(vitd_models.get("fdr_bh"), errors="coerce") < 0.05).sum())
        if not vitd_models.empty else 0
    )

    text = f"""# Phase v0.8.2 module coherence and integration report

Generated: {now_utc()}

## Preservation rule

- v0.7.1 remains the preserved co-author-review baseline.
- v0.8.1 remains the first-pass nutritional-metabolic feasibility analysis.
- v0.8.2.2 writes only to new `{OUTPUT_TAG}` directories.

## Module-coherence audit

- Strong modules: {strong}
- Moderate modules: {moderate}
- Weak or heterogeneous modules: {weak}

The coherence audit includes median pairwise Spearman correlation, Cronbach alpha
on protein-wise z-scores, PC1 variance explained, leave-one-protein-out stability,
shared-protein counts, unique-only scores, and PC1 scores.

## Legacy-module reconstruction

- Legacy modules assessed: {len(legacy_audit)}
- Legacy modules with at least {MIN_MODULE_PROTEINS} detected proteins: {legacy_scoreable}

Legacy scores are reconstructed explicitly from the curated project
module-to-protein mapping rather than inferred from filenames. Legacy module
identifiers and labels are preserved during integrated-score selection.

## Vitamin D-linked proxy dissection

- Individual-protein categorical BMI × time terms with nominal p<0.05: {vitd_nominal}
- Individual-protein categorical BMI × time terms with BH-FDR<0.05: {vitd_fdr}

The four-protein module consists of CAMP, LAMP1, LAMP2 and RAB7A. A module-level
BMI interaction should enter the manuscript only if it is not attributable to a
single unstable protein and remains directionally coherent in leave-one-out and
individual-protein analyses.

## Candidate integrated nutritional-metabolic latent score

```json
{json.dumps(integrated_manifest, indent=2, default=str)}
```

This candidate score is an exploratory PCA-derived latent axis. It is not a
validated clinical nutritional score and should not be described as directly
measuring vitamin or micronutrient concentrations.

## Interpretation boundaries

- Serum proteins are proxies for pathway-related biology.
- Intracellular calcium flux, autophagic flux, phagocytosis efficiency and Mtb
  killing are not measured directly.
- Cytoskeletal and contractile proteins may reflect cell turnover, vesicles,
  tissue injury, platelet biology or vascular remodelling.
- Single-cell/spatial work will test cell-type plausibility, not serum abundance.

## Warnings

{chr(10).join("- " + x for x in warnings) if warnings else "- None."}
"""
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=EXPECTED_ROOT)
    parser.add_argument("--legacy-mapping", type=Path, default=None)
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
        print(f"ERROR: Missing project root: {root}")
        return 2

    input_tables = root / "02_tables" / INPUT_TAG
    output_tables = root / "02_tables" / OUTPUT_TAG
    output_results = root / "05_results" / OUTPUT_TAG
    output_figures = root / "03_figures" / "draft_panels" / OUTPUT_TAG
    output_logs = root / "05_docs_decision_logs" / OUTPUT_TAG
    for d in [output_tables, output_results, output_figures, output_logs]:
        d.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(
        output_logs / f"phase_v0_8_2_run_{run_id}.log",
        args.verbose,
    )
    warnings: list[str] = []

    logger.info("v0.7.1 remains preserved.")
    logger.info("Starting additive Phase v0.8.2 analysis.")

    matrix = read_matrix(
        input_tables / "phase_v0_8_normalized_sample_by_protein_matrix.tsv.gz"
    )
    analysis = read_required(
        input_tables / "phase_v0_8_sample_level_analysis_dataset.tsv"
    ).reset_index(drop=True)
    membership = read_required(
        input_tables / "phase_v0_8_module_score_membership.tsv"
    )
    coverage = read_required(
        input_tables / "phase_v0_8_module_coverage_audit.tsv"
    )

    for col in ["sample_id", "subject_id"]:
        analysis[col] = analysis[col].astype(str)
    analysis["timepoint"] = pd.to_numeric(
        analysis["timepoint"], errors="coerce"
    )
    analysis["bmi"] = pd.to_numeric(analysis["bmi"], errors="coerce")

    module_map = build_membership_map(membership)
    variant_scores, coherence, loadings, loo = score_module_variants(
        matrix, analysis, module_map
    )
    write_tsv(
        variant_scores,
        output_tables / "phase_v0_8_2_new_module_score_variants.tsv",
    )
    write_tsv(
        coherence,
        output_tables / "phase_v0_8_2_module_coherence_audit.tsv",
    )
    write_tsv(
        loadings,
        output_tables / "phase_v0_8_2_module_pc1_loadings.tsv",
    )
    write_tsv(
        loo,
        output_tables / "phase_v0_8_2_leave_one_protein_out_stability.tsv",
    )

    legacy_mapping, legacy_path = load_legacy_mapping(
        root,
        args.legacy_mapping,
    )
    legacy_audit, legacy_scores, legacy_map = reconstruct_legacy_modules(
        matrix,
        legacy_mapping,
    )
    write_tsv(
        legacy_audit,
        output_tables / "phase_v0_8_2_legacy_module_coverage_audit.tsv",
    )
    write_tsv(
        legacy_scores.reset_index(),
        output_tables / "phase_v0_8_2_legacy_module_scores.tsv",
    )

    combined = variant_scores.merge(
        legacy_scores.reset_index(),
        on="sample_id",
        how="left",
    )

    labels = module_label_lookup(coverage, legacy_audit)
    selection = select_integrated_modules(combined, labels)
    write_tsv(
        selection,
        output_tables / "phase_v0_8_2_integrated_score_module_selection.tsv",
    )

    integrated_feature_map = combined_module_feature_map(
        module_map,
        legacy_map,
    )
    overlap_audit = integrated_module_overlap_audit(
        selection,
        integrated_feature_map,
    )
    write_tsv(
        overlap_audit,
        output_tables / "phase_v0_8_2_integrated_module_overlap_audit.tsv",
    )

    (
        combined,
        integrated_loadings,
        integrated_redundancy,
        integrated_manifest,
    ) = build_integrated_pc1(
        combined,
        selection,
        integrated_feature_map,
        labels,
    )
    write_tsv(
        integrated_loadings,
        output_tables / "phase_v0_8_2_integrated_score_loadings.tsv",
    )
    write_tsv(
        integrated_redundancy,
        output_tables / "phase_v0_8_2_integrated_score_redundancy_decisions.tsv",
    )
    write_tsv(
        combined,
        output_tables / "phase_v0_8_2_combined_sample_level_scores.tsv",
    )

    if not integrated_manifest.get("created"):
        warnings.append(
            "Candidate integrated nutritional-metabolic PC1 was not created: "
            + integrated_manifest.get("reason", "unknown reason")
        )

    # Vitamin D-linked module dissection.
    vitd_features = module_map.get(VITD_MODULE, [])
    if len(vitd_features) >= MIN_MODULE_PROTEINS:
        vitd_paired, vitd_group_models, vitd_cont_models = (
            vitamin_d_protein_dissection(matrix, analysis, vitd_features)
        )
    else:
        vitd_paired = vitd_group_models = vitd_cont_models = pd.DataFrame()
        warnings.append(
            "Vitamin D-linked proxy had fewer than three mapped proteins."
        )

    write_tsv(
        vitd_paired,
        output_tables / "phase_v0_8_2_vitamin_d_proxy_individual_protein_changes.tsv",
    )
    write_tsv(
        vitd_group_models,
        output_tables / "phase_v0_8_2_vitamin_d_proxy_protein_bmi_group_time_models.tsv",
    )
    write_tsv(
        vitd_cont_models,
        output_tables / "phase_v0_8_2_vitamin_d_proxy_protein_continuous_bmi_time_models.tsv",
    )

    # Sensitivity models for new module score variants.
    variant_cols = [
        c for c in combined.columns
        if c.endswith("__mean_z")
        or c.endswith("__pc1")
        or c.endswith("__unique_only_mean_z")
    ]
    variant_group_models = fit_interactions(
        combined,
        variant_cols,
        "bmi_group_by_time",
    )
    variant_cont_models = fit_interactions(
        combined,
        variant_cols,
        "continuous_bmi_by_time",
    )
    write_tsv(
        variant_group_models,
        output_tables / "phase_v0_8_2_score_variant_bmi_group_time_models.tsv",
    )
    write_tsv(
        variant_cont_models,
        output_tables / "phase_v0_8_2_score_variant_continuous_bmi_time_models.tsv",
    )

    if "candidate_integrated_nutritional_metabolic_pc1" in combined.columns:
        integrated_group = fit_interactions(
            combined,
            ["candidate_integrated_nutritional_metabolic_pc1"],
            "bmi_group_by_time",
        )
        integrated_cont = fit_interactions(
            combined,
            ["candidate_integrated_nutritional_metabolic_pc1"],
            "continuous_bmi_by_time",
        )
        integrated_changes = paired_change_table(
            combined,
            ["candidate_integrated_nutritional_metabolic_pc1"],
        )
        integrated_baseline_validation = integrated_score_baseline_validation(
            combined,
            "candidate_integrated_nutritional_metabolic_pc1",
        )
    else:
        integrated_group = integrated_cont = integrated_changes = pd.DataFrame()
        integrated_baseline_validation = pd.DataFrame()

    write_tsv(
        integrated_group,
        output_tables / "phase_v0_8_2_integrated_score_bmi_group_time_models.tsv",
    )
    write_tsv(
        integrated_cont,
        output_tables / "phase_v0_8_2_integrated_score_continuous_bmi_time_models.tsv",
    )
    write_tsv(
        integrated_changes,
        output_tables / "phase_v0_8_2_integrated_score_within_group_changes.tsv",
    )
    write_tsv(
        integrated_baseline_validation,
        output_tables / "phase_v0_8_2_integrated_score_baseline_validation.tsv",
    )

    # Correlation network: mean-z new variants + all scoreable legacy modules.
    corr_cols = [
        c for c in combined.columns
        if c.endswith("__mean_z") or c.startswith("legacy__")
    ]
    if "candidate_integrated_nutritional_metabolic_pc1" in combined.columns:
        corr_cols.append("candidate_integrated_nutritional_metabolic_pc1")
    edges, corr = correlation_edges(combined, corr_cols)
    write_tsv(
        edges,
        output_tables / "phase_v0_8_2_new_legacy_module_correlation_edges.tsv",
    )
    if not corr.empty:
        write_tsv(
            corr.reset_index().rename(columns={"index": "score"}),
            output_tables / "phase_v0_8_2_new_legacy_module_correlation_matrix.tsv",
        )

    plot_coherence(
        coherence,
        output_figures / "phase_v0_8_2_module_coherence_pc1_variance.png",
    )
    plot_heatmap(
        corr,
        output_figures / "phase_v0_8_2_new_legacy_module_correlation_heatmap.png",
    )
    if "candidate_integrated_nutritional_metabolic_pc1" in combined.columns:
        plot_trajectory(
            combined,
            "candidate_integrated_nutritional_metabolic_pc1",
            output_figures / "phase_v0_8_2_candidate_integrated_score_trajectory.png",
            "Candidate integrated nutritional-metabolic latent score",
        )

    # Separate trajectory panel for each vitamin-D proxy protein.
    for feature in vitd_features:
        plot_trajectory(
            combined.merge(
                matrix[[feature]].reset_index(),
                on="sample_id",
                how="left",
            ),
            feature,
            output_figures / f"phase_v0_8_2_vitamin_d_proxy_{normalize(feature)}_trajectory.png",
            f"Vitamin D-linked proxy protein: {feature}",
        )

    figure_manifest = pd.DataFrame([
        {
            "panel_family": "module_coherence",
            "file": "phase_v0_8_2_module_coherence_pc1_variance.png",
            "purpose": "Compare coherence and latent dimensionality of scoreable modules.",
        },
        {
            "panel_family": "module_network",
            "file": "phase_v0_8_2_new_legacy_module_correlation_heatmap.png",
            "purpose": "Integrate new nutritional-metabolic modules with curated legacy modules.",
        },
        {
            "panel_family": "integrated_score",
            "file": "phase_v0_8_2_candidate_integrated_score_trajectory.png",
            "purpose": "Show exploratory nutritional-metabolic latent-score recovery.",
        },
        *[
            {
                "panel_family": "vitamin_d_proxy_dissection",
                "file": f"phase_v0_8_2_vitamin_d_proxy_{normalize(feature)}_trajectory.png",
                "purpose": "Determine whether the four-protein proxy is coherent or single-protein driven.",
            }
            for feature in vitd_features
        ],
    ])
    write_tsv(
        figure_manifest,
        output_tables / "phase_v0_8_2_figure_panel_manifest.tsv",
    )

    manifest = {
        "script_version": SCRIPT_VERSION,
        "generated_utc": now_utc(),
        "baseline_preserved": "v0.7.1",
        "input_phase": INPUT_TAG,
        "output_phase": OUTPUT_TAG,
        "legacy_mapping": str(legacy_path),
        "n_samples": int(len(combined)),
        "n_subjects": int(combined["subject_id"].nunique()),
        "n_new_modules_audited": int(len(coherence)),
        "n_scoreable_legacy_modules": int(legacy_audit["scoreable"].sum()),
        "vitamin_d_proxy_features": vitd_features,
        "integrated_score": integrated_manifest,
        "warnings": warnings,
    }
    write_json(
        manifest,
        output_results / "phase_v0_8_2_run_manifest.json",
    )
    write_report(
        output_results / "phase_v0_8_2_module_coherence_and_integration_report.md",
        coherence,
        legacy_audit,
        vitd_group_models,
        integrated_manifest,
        warnings,
    )

    decision = f"""# Phase v0.8.2 decision log

Generated: {now_utc()}

## Locked interpretations entering this phase

1. v0.7.1 remains the preserved manuscript baseline.
2. v0.8.1 demonstrated treatment-associated changes in several new serum-protein
   modules without BH-FDR-significant BMI-group or continuous-BMI interaction.
3. The vitamin D-linked antimicrobial/phagolysosome proxy showed nominal early
   BMI-related divergence but did not survive test-family BH-FDR.
4. No module-level BMI interaction should be promoted until protein-level and
   leave-one-out sensitivity analyses establish coherence.
5. An integrated score may be retained only as an exploratory latent axis unless
   it demonstrates stable loadings, adequate variance explained and biological
   interpretability.

## Methods added

- Protein-level module coherence.
- PC1 scores and loadings.
- Unique-protein-only sensitivity scores.
- Leave-one-protein-out stability.
- Explicit reconstruction of curated legacy module scores.
- New-to-legacy correlation network.
- Four-protein vitamin D proxy dissection.
- Exploratory PCA-derived integrated nutritional-metabolic score.

## Prohibited claims

- Direct vitamin concentration measurement.
- Direct calcium-flux measurement.
- Direct phagocytosis, autophagic-flux or Mtb-killing measurement.
- Clinical validation of the exploratory integrated score.
"""
    (
        output_logs / "phase_v0_8_2_decision_log.md"
    ).write_text(decision, encoding="utf-8")

    logger.info("Phase v0.8.2 completed.")
    logger.info("Tables: %s", output_tables)
    logger.info("Results: %s", output_results)
    logger.info("Figures: %s", output_figures)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise
