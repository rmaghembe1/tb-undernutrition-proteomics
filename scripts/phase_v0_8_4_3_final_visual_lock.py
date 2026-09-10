#!/usr/bin/env python3
"""
Phase v0.8.4.3 — final visual lock.

This additive repair preserves:
- manuscript baseline v0.7.1;
- scientific lock v0.8.3;
- first publication-facing output v0.8.4.

Repairs:
1. Removes the embedded figure title to increase usable panel space.
2. Cleans module labels and typography.
3. Replaces Panel D point-only effect sizes with paired standardized changes
   and bootstrap 95% confidence intervals.
4. Adds protein-level PC1 variance to the Results draft.
5. Removes the inaccurate phrase "selected a priori".
6. Standardizes symbols, units and p-value formatting.
7. Produces an automated raster/vector file-quality audit.
8. Writes only to new v0.8.4.3 directories.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    HAVE_MATPLOTLIB = True
except Exception:
    HAVE_MATPLOTLIB = False

try:
    from PIL import Image
    HAVE_PIL = True
except Exception:
    HAVE_PIL = False


EXPECTED_ROOT = Path(__file__).resolve().parents[1]
INPUT_V083 = "phase_v0_8_3_final_recovery_axis_adjudication"
INPUT_V0822 = "phase_v0_8_2_2_module_coherence_and_integration"
INPUT_V084 = "phase_v0_8_4_publication_figure_and_manuscript_integration"
OUTPUT_TAG = "phase_v0_8_4_3_final_visual_lock"
SCRIPT_VERSION = "0.8.4.3"

PREFERRED_AXIS = "three_module_core_unique_protein_pc1"
MODULE_AXIS = "three_module_core_score_pc1"
EXTENDED_AXIS = "core_plus_redox_unique_protein_pc1"

MODULE_LABELS = {
    "legacy__m02_curated_lipid_apolipoprotein_transport":
        "Lipid, cholesterol and\napolipoprotein transport",
    "legacy__m03b_curated_nutrition_carrier_broad":
        "Broad nutritional-carrier\nbiology",
    "legacy__m13_curated_endocrine_immunometabolic":
        "Endocrine and systemic\nimmunometabolic signalling",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t", low_memory=False)


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def setup_logger(path: Path, verbose: bool) -> logging.Logger:
    logger = logging.getLogger("phase_v0_8_4_3")
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


def panel_label(
    ax,
    label: str,
    x: float = -0.09,
    y: float = 1.05,
) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
        clip_on=False,
    )


def panel_a(ax) -> None:
    panel_label(ax, "A")
    ax.axis("off")

    boxes = [
        (0.03, 0.62, 0.27, 0.24,
         "Lipid, cholesterol and\napolipoprotein transport"),
        (0.365, 0.62, 0.27, 0.24,
         "Broad nutritional-carrier\nbiology"),
        (0.70, 0.62, 0.27, 0.24,
         "Endocrine and systemic\nimmunometabolic signalling"),
    ]

    for x, y, w, h, text in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.025",
                linewidth=1.0,
                facecolor="none",
            )
        )
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=8,
        )

    ax.add_patch(
        FancyBboxPatch(
            (0.28, 0.18),
            0.44,
            0.20,
            boxstyle="round,pad=0.02,rounding_size=0.025",
            linewidth=1.2,
            facecolor="none",
        )
    )
    ax.text(
        0.50,
        0.28,
        "Serum nutritional-metabolic\nrecovery axis",
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
    )

    for x in [0.165, 0.50, 0.835]:
        ax.add_patch(
            FancyArrowPatch(
                (x, 0.61),
                (0.50, 0.40),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.0,
            )
        )

    ax.text(
        0.50,
        0.055,
        "Preferred construction: PC1 across the union of 36 unique proteins",
        ha="center",
        va="bottom",
        fontsize=7,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def panel_b(ax, loadings: pd.DataFrame) -> pd.DataFrame:
    panel_label(ax, "B")
    sub = loadings[loadings["component_type"] == MODULE_AXIS].copy()
    sub["display_label"] = sub["feature_or_module"].map(MODULE_LABELS)
    sub = sub.sort_values("pc1_loading")
    ax.barh(np.arange(len(sub)), sub["pc1_loading"])
    ax.set_yticks(np.arange(len(sub)))
    ax.set_yticklabels(sub["display_label"], fontsize=7)
    ax.set_xlabel("PC1 loading", fontsize=8)
    ax.set_title("Concordant module contributions", fontsize=9)
    ax.tick_params(axis="x", labelsize=7)
    ax.axvline(0, linewidth=0.8)
    return sub


def trajectory_summary(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (group, tp), sub in scores.groupby(["bmi_group", "timepoint"]):
        values = pd.to_numeric(sub[PREFERRED_AXIS], errors="coerce").dropna()
        rows.append({
            "bmi_group": group,
            "timepoint": float(tp),
            "n": len(values),
            "mean": values.mean(),
            "sem": values.sem(ddof=1) if len(values) > 1 else np.nan,
        })
    return pd.DataFrame(rows)


def panel_c(ax, scores: pd.DataFrame) -> pd.DataFrame:
    panel_label(ax, "C")
    summary = trajectory_summary(scores)
    timepoints = sorted(summary["timepoint"].unique())

    styles = {
        "BMI<18.5": {
            "marker": "o",
            "linestyle": "-",
            "label": "BMI < 18.5",
        },
        "BMI>=18.5": {
            "marker": "s",
            "linestyle": "--",
            "label": "BMI ≥ 18.5",
        },
    }

    for group in ["BMI<18.5", "BMI>=18.5"]:
        sub = summary[summary["bmi_group"] == group].set_index("timepoint")
        ax.errorbar(
            timepoints,
            [sub.loc[x, "mean"] for x in timepoints],
            yerr=[sub.loc[x, "sem"] for x in timepoints],
            marker=styles[group]["marker"],
            linestyle=styles[group]["linestyle"],
            capsize=3,
            linewidth=1.2,
            label=styles[group]["label"],
        )

    ax.set_xlabel("Treatment week", fontsize=8)
    ax.set_ylabel("Recovery-axis PC1", fontsize=8)
    ax.set_title("Shared longitudinal recovery", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.axhline(0, linewidth=0.7)
    ax.legend(frameon=False, fontsize=7)
    return summary


def paired_standardized_change(
    scores: pd.DataFrame,
    group: str,
    post_timepoint: float,
    rng: np.random.Generator,
    n_boot: int = 5000,
) -> dict[str, Any]:
    sub = scores[scores["bmi_group"] == group]
    baseline = (
        sub[pd.to_numeric(sub["timepoint"], errors="coerce") == 0]
        [["subject_id", PREFERRED_AXIS]]
        .dropna()
        .rename(columns={PREFERRED_AXIS: "baseline"})
    )
    post = (
        sub[pd.to_numeric(sub["timepoint"], errors="coerce") == post_timepoint]
        [["subject_id", PREFERRED_AXIS]]
        .dropna()
        .rename(columns={PREFERRED_AXIS: "post"})
    )
    paired = baseline.merge(post, on="subject_id")
    delta = (paired["post"] - paired["baseline"]).to_numpy(float)
    n = len(delta)

    if n < 3 or np.std(delta, ddof=1) == 0:
        return {
            "bmi_group": group,
            "post_timepoint": post_timepoint,
            "n": n,
            "standardized_paired_change": np.nan,
            "bootstrap_ci_low": np.nan,
            "bootstrap_ci_high": np.nan,
        }

    effect = float(np.mean(delta) / np.std(delta, ddof=1))
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sampled = rng.choice(delta, size=n, replace=True)
        sd = np.std(sampled, ddof=1)
        boot[i] = np.mean(sampled) / sd if sd > 0 else np.nan

    boot = boot[np.isfinite(boot)]
    low, high = np.quantile(boot, [0.025, 0.975])

    return {
        "bmi_group": group,
        "post_timepoint": post_timepoint,
        "n": n,
        "standardized_paired_change": effect,
        "bootstrap_ci_low": float(low),
        "bootstrap_ci_high": float(high),
    }


def effect_size_table(scores: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(20260715)
    timepoints = sorted(
        x for x in pd.to_numeric(scores["timepoint"], errors="coerce").dropna().unique()
        if x != 0
    )
    rows = []
    for group in ["BMI<18.5", "BMI>=18.5"]:
        for tp in timepoints:
            rows.append(paired_standardized_change(scores, group, tp, rng))
    return pd.DataFrame(rows)


def panel_d(ax, effects: pd.DataFrame) -> None:
    panel_label(ax, "D")
    timepoints = sorted(effects["post_timepoint"].unique())
    y_positions = np.arange(len(timepoints), dtype=float)
    offsets = {"BMI<18.5": -0.11, "BMI>=18.5": 0.11}
    styles = {
        "BMI<18.5": {"fmt": "o", "label": "BMI < 18.5"},
        "BMI>=18.5": {"fmt": "s", "label": "BMI ≥ 18.5"},
    }

    for group in ["BMI<18.5", "BMI>=18.5"]:
        sub = effects[effects["bmi_group"] == group].set_index("post_timepoint")
        values = np.array(
            [sub.loc[tp, "standardized_paired_change"] for tp in timepoints]
        )
        low = np.array([sub.loc[tp, "bootstrap_ci_low"] for tp in timepoints])
        high = np.array([sub.loc[tp, "bootstrap_ci_high"] for tp in timepoints])
        xerr = np.vstack([values - low, high - values])
        ax.errorbar(
            values,
            y_positions + offsets[group],
            xerr=xerr,
            fmt=styles[group]["fmt"],
            capsize=3,
            linewidth=1.0,
            label=styles[group]["label"],
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"Week {int(x)}" for x in timepoints], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(r"Standardized paired change, $d_z$ (95% CI)", fontsize=8)
    ax.set_title("Within-participant treatment change", fontsize=9)
    ax.axvline(0, linewidth=0.7)
    ax.tick_params(axis="x", labelsize=7)
    ax.legend(frameon=False, fontsize=7, loc="upper right")


def baseline_table(scores: pd.DataFrame) -> pd.DataFrame:
    baseline = scores[
        pd.to_numeric(scores["timepoint"], errors="coerce") == 0
    ].copy()
    baseline["bmi"] = pd.to_numeric(baseline["bmi"], errors="coerce")
    baseline[PREFERRED_AXIS] = pd.to_numeric(
        baseline[PREFERRED_AXIS], errors="coerce"
    )
    return baseline.dropna(subset=["bmi", PREFERRED_AXIS])


def panel_e(ax, baseline: pd.DataFrame) -> dict[str, float]:
    panel_label(ax, "E")
    styles = {
        "BMI<18.5": {
            "marker": "o",
            "label": "BMI < 18.5",
        },
        "BMI>=18.5": {
            "marker": "s",
            "label": "BMI ≥ 18.5",
        },
    }

    for group in ["BMI<18.5", "BMI>=18.5"]:
        sub = baseline[baseline["bmi_group"] == group]
        ax.scatter(
            sub["bmi"],
            sub[PREFERRED_AXIS],
            alpha=0.75,
            marker=styles[group]["marker"],
            label=styles[group]["label"],
        )

    slope, intercept = np.polyfit(
        baseline["bmi"], baseline[PREFERRED_AXIS], 1
    )
    xline = np.linspace(baseline["bmi"].min(), baseline["bmi"].max(), 100)
    ax.plot(xline, slope * xline + intercept, linewidth=1.0)
    rho, p = stats.spearmanr(baseline["bmi"], baseline[PREFERRED_AXIS])

    ax.axvline(18.5, linestyle="--", linewidth=0.9)
    ax.set_xlabel("Baseline BMI (kg/m²)", fontsize=8)
    ax.set_ylabel("Baseline recovery-axis PC1", fontsize=8)
    ax.set_title("No significant baseline BMI association", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.text(
        0.03,
        0.96,
        f"Spearman ρ={rho:.2f}\np={p:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
    )
    return {"spearman_rho": float(rho), "p_value": float(p), "n": len(baseline)}


def panel_f(ax, scores: pd.DataFrame) -> dict[str, float]:
    panel_label(ax, "F", x=0.0, y=1.05)
    sub = scores[[MODULE_AXIS, PREFERRED_AXIS, EXTENDED_AXIS]].dropna()

    ax.scatter(
        sub[MODULE_AXIS],
        sub[PREFERRED_AXIS],
        s=18,
        alpha=0.55,
    )
    slope, intercept = np.polyfit(
        sub[MODULE_AXIS], sub[PREFERRED_AXIS], 1
    )
    xline = np.linspace(sub[MODULE_AXIS].min(), sub[MODULE_AXIS].max(), 100)
    ax.plot(xline, slope * xline + intercept, linewidth=1.0)

    rho_core, p_core = stats.spearmanr(
        sub[MODULE_AXIS], sub[PREFERRED_AXIS]
    )
    rho_ext, p_ext = stats.spearmanr(
        sub[PREFERRED_AXIS], sub[EXTENDED_AXIS]
    )

    ax.set_xlabel("Module-score core PC1", fontsize=8)
    ax.set_ylabel("Unique-protein core PC1", fontsize=8, labelpad=5)
    ax.set_title("Robustness to score construction", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.text(
        0.03,
        0.96,
        (
            f"Module vs protein ρ={rho_core:.3f}\n"
            f"Core vs redox extension ρ={rho_ext:.3f}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
    )

    return {
        "module_vs_unique_rho": float(rho_core),
        "module_vs_unique_p": float(p_core),
        "core_vs_extended_rho": float(rho_ext),
        "core_vs_extended_p": float(p_ext),
        "n": len(sub),
    }


def build_figure(
    scores: pd.DataFrame,
    loadings: pd.DataFrame,
    stem: Path,
) -> dict[str, pd.DataFrame]:
    if not HAVE_MATPLOTLIB:
        raise RuntimeError("matplotlib is not available.")

    fig = plt.figure(figsize=(7.2, 8.2))
    gs = fig.add_gridspec(
        4,
        2,
        height_ratios=[0.72, 1.0, 1.0, 1.0],
        hspace=0.50,
        wspace=0.48,
    )

    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[2, 0])
    ax_e = fig.add_subplot(gs[2, 1])
    ax_f = fig.add_subplot(gs[3, :])

    panel_a(ax_a)
    module_loadings = panel_b(ax_b, loadings)
    trajectory = panel_c(ax_c, scores)

    effects = effect_size_table(scores)
    panel_d(ax_d, effects)

    baseline = baseline_table(scores)
    baseline_stats = panel_e(ax_e, baseline)
    sensitivity_stats = panel_f(ax_f, scores)

    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    return {
        "module_loadings": module_loadings,
        "trajectory": trajectory,
        "paired_effect_sizes": effects,
        "baseline_points": baseline[
            ["sample_id", "subject_id", "bmi", "bmi_group", PREFERRED_AXIS]
        ],
        "baseline_statistics": pd.DataFrame([baseline_stats]),
        "sensitivity_statistics": pd.DataFrame([sensitivity_stats]),
    }


def format_p(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value >= 0.001:
        return f"{value:.3f}"

    superscripts = str.maketrans(
        "0123456789-",
        "⁰¹²³⁴⁵⁶⁷⁸⁹⁻",
    )
    exponent = int(math.floor(math.log10(value)))
    coefficient = value / (10 ** exponent)
    return f"{coefficient:.2f} × 10{str(exponent).translate(superscripts)}"


def integration_text(
    decision: pd.DataFrame,
    baseline_validation: pd.DataFrame,
    changes: pd.DataFrame,
    loadings: pd.DataFrame,
) -> str:
    d = decision.iloc[0]
    baseline = baseline_validation[
        baseline_validation["axis"] == PREFERRED_AXIS
    ].iloc[0]

    module_loadings = (
        loadings[loadings["component_type"] == MODULE_AXIS]
        .set_index("feature_or_module")["pc1_loading"]
    )
    endocrine_loading = module_loadings[
        "legacy__m13_curated_endocrine_immunometabolic"
    ]
    carrier_loading = module_loadings[
        "legacy__m03b_curated_nutrition_carrier_broad"
    ]
    lipid_loading = module_loadings[
        "legacy__m02_curated_lipid_apolipoprotein_transport"
    ]

    selected_changes = (
        changes[changes["axis"] == PREFERRED_AXIS]
        .set_index(["bmi_group", "post_timepoint"])
    )

    low_8 = selected_changes.loc[("BMI<18.5", 8)]
    high_8 = selected_changes.loc[("BMI>=18.5", 8)]
    low_52 = selected_changes.loc[("BMI<18.5", 52)]
    high_52 = selected_changes.loc[("BMI>=18.5", 52)]

    return f"""# Phase v0.8.4.3 manuscript integration draft

## Proposed Results subsection

### A serum nutritional-metabolic recovery axis increases during tuberculosis treatment

Integration of the curated lipid, cholesterol and apolipoprotein-transport;
broad nutritional-carrier; and endocrine-immunometabolic modules identified a
coordinated serum nutritional-metabolic recovery axis. Module-level loadings on
the first principal component were {endocrine_loading:.3f} for endocrine and
systemic immunometabolic signalling, {carrier_loading:.3f} for broad
nutritional-carrier biology, and {lipid_loading:.3f} for lipid, cholesterol and
apolipoprotein transport. This component explained
{d['core_module_pc1_variance_explained'] * 100:.1f}% of module-score variance.

To avoid double-counting proteins shared among the curated modules, the
preferred axis was reconstructed from the union of 36 unique proteins. The
protein-level PC1 explained
{d['core_unique_protein_pc1_variance_explained'] * 100:.1f}% of variance across
these proteins and correlated strongly with the module-level construction
(Spearman ρ={d['core_module_pc1_vs_unique_protein_pc1_rho']:.3f}). It remained
highly concordant with a sensitivity construction that additionally included
the heterogeneous redox/ECM module
(ρ={d['core_unique_protein_vs_extended_unique_protein_rho']:.3f}).

The axis increased during treatment in both baseline BMI strata. At week 8, the
within-participant increase was not significant among participants with BMI
<18.5 kg/m² (d_z={low_8['paired_effect_size']:.2f}, BH-adjusted
p={format_p(low_8['fdr_bh'])}) but was significant among those with BMI
≥18.5 kg/m² (d_z={high_8['paired_effect_size']:.2f}, BH-adjusted
p={format_p(high_8['fdr_bh'])}). From week 17 onward, both strata showed large
increases. By week 52, d_z was {low_52['paired_effect_size']:.2f} in the
lower-BMI stratum (BH-adjusted p={format_p(low_52['fdr_bh'])}) and
{high_52['paired_effect_size']:.2f} in the higher-BMI stratum
(BH-adjusted p={format_p(high_52['fdr_bh'])}). Neither categorical
BMI-group-by-time nor continuous-BMI-by-time interactions were significant
after false-discovery-rate correction.

At baseline, the preferred axis did not differ significantly between BMI groups
(Welch p={baseline['bmi_group_p_value']:.3f}) and was not significantly
correlated with continuous BMI (Spearman ρ={baseline['spearman_rho_with_bmi']:.3f},
p={baseline['continuous_bmi_p_value']:.3f}). The axis therefore captures shared
treatment-associated restoration of circulating nutritional-metabolic protein
systems rather than a validated undernutrition score or a formally demonstrated
BMI-dependent recovery lag.

## Proposed Methods subsection

### Construction and validation of the serum nutritional-metabolic recovery axis

Following module-coherence and redundancy adjudication, three curated
serum-protein modules were integrated: lipid, cholesterol and apolipoprotein
transport; broad nutritional-carrier biology; and endocrine and systemic
immunometabolic signalling. Protein abundances were standardized across
samples. A module-level PC1 was first calculated from the three curated module
scores. To reduce redundancy caused by proteins shared between modules, the
preferred protein-level construction was generated from the union of 36 unique
proteins represented across the modules. The sign of PC1 was oriented so that
higher values represented coordinated recovery of the constituent protein
systems.

Robustness was assessed using Spearman correlations among the module-score PC1,
the unique-protein PC1, the earlier four-module candidate score, and a
unique-protein construction that additionally included the vitamin
C/redox/ECM module. Baseline associations were evaluated using Welch's t test
for BMI category and Spearman correlation for continuous BMI.
Within-participant changes from baseline were tested separately at weeks 8, 17,
26 and 52. Standardized paired change (d_z) was calculated as the mean paired
change divided by the standard deviation of paired changes; 95% confidence
intervals shown in the figure were estimated using 5,000 participant-level
bootstrap resamples. Longitudinal effect modification was evaluated using
random-intercept mixed models containing categorical BMI-group-by-time or
continuous-BMI-by-time interaction terms. Benjamini-Hochberg correction was
applied within each test family.

## Proposed Discussion paragraph

The coordinated increase in lipid-transport, nutritional-carrier and
endocrine-immunometabolic proteins suggests that tuberculosis treatment is
accompanied by broad restoration of circulating metabolic homeostasis. This
signal was robust to alternative score construction and removal of overlapping
proteins. However, it was neither associated with baseline BMI nor significantly
modified by BMI over time. BMI therefore did not capture the full
treatment-responsive serum metabolic programme, but the current data do not
establish a distinct molecular undernutrition phenotype or delayed recovery
programme among participants with BMI <18.5 kg/m². The axis should consequently
be interpreted as a serum nutritional-metabolic recovery measure rather than a
clinical undernutrition score.

## Proposed main-figure legend

**Figure X. Coordinated serum nutritional-metabolic recovery during
tuberculosis treatment.** **A,** Architecture of the integrated axis,
comprising curated lipid, cholesterol and apolipoprotein transport; broad
nutritional-carrier biology; and endocrine and systemic immunometabolic
signalling. **B,** Concordant module-level PC1 loadings. **C,** Mean preferred
unique-protein recovery-axis score across treatment weeks, stratified by
baseline BMI group; error bars show the standard error of the mean. Circles and
solid lines indicate BMI <18.5 kg/m², whereas squares and dashed lines indicate
BMI ≥18.5 kg/m². **D,** Within-participant standardized changes (d_z) from
baseline with participant-level bootstrap 95% confidence intervals. **E,**
Relationship between baseline BMI and the preferred recovery-axis score; the
dashed vertical line indicates BMI 18.5 kg/m². **F,** Agreement between the
module-score and unique-protein constructions; text also reports concordance
with the redox/ECM-extended sensitivity axis. The axis increased during
treatment in both BMI strata, but baseline BMI and BMI-by-time interaction tests
were not significant after false-discovery-rate correction.

## Terminology lock

Use: **serum nutritional-metabolic recovery axis**

Do not use:
- undernutrition score;
- metabolic-undernutrition axis;
- malnutrition score;
- vitamin-repletion score;
- validated nutritional index.
"""


def quality_audit(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        row = {
            "file": path.name,
            "suffix": path.suffix.lower(),
            "size_bytes": path.stat().st_size if path.exists() else np.nan,
            "exists": path.exists(),
            "width_px": np.nan,
            "height_px": np.nan,
            "dpi_x": np.nan,
            "dpi_y": np.nan,
            "quality_flag": "",
        }

        if path.exists() and path.suffix.lower() in {".png", ".tif", ".tiff"}:
            if HAVE_PIL:
                with Image.open(path) as image:
                    row["width_px"], row["height_px"] = image.size
                    dpi = image.info.get("dpi", (np.nan, np.nan))
                    if isinstance(dpi, tuple) and len(dpi) >= 2:
                        row["dpi_x"], row["dpi_y"] = dpi[:2]

                if row["width_px"] < 3000:
                    row["quality_flag"] = "review_raster_width"
                elif row["height_px"] < 3000:
                    row["quality_flag"] = "review_raster_height"
                else:
                    row["quality_flag"] = "pass"
            else:
                row["quality_flag"] = "Pillow_unavailable"
        elif path.exists() and path.suffix.lower() in {".svg", ".pdf"}:
            row["quality_flag"] = "vector_present"
        else:
            row["quality_flag"] = "missing"

        rows.append(row)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=EXPECTED_ROOT)
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

    v083 = root / "02_tables" / INPUT_V083
    v0822 = root / "02_tables" / INPUT_V0822

    table_dir = root / "02_tables" / OUTPUT_TAG
    result_dir = root / "05_results" / OUTPUT_TAG
    log_dir = root / "05_docs_decision_logs" / OUTPUT_TAG
    manuscript_dir = root / "01_manuscript" / "drafts" / OUTPUT_TAG

    figure_dirs = {
        "png": root / "03_figures" / "final_png" / OUTPUT_TAG,
        "tiff": root / "03_figures" / "final_tiff" / OUTPUT_TAG,
        "svg": root / "03_figures" / "final_svg" / OUTPUT_TAG,
        "pdf": root / "03_figures" / "final_pdf" / OUTPUT_TAG,
        "preview": root / "03_figures" / "preview_png" / OUTPUT_TAG,
        "specs": root / "03_figures" / "figure_specs" / OUTPUT_TAG,
    }

    for path in [
        table_dir,
        result_dir,
        log_dir,
        manuscript_dir,
        *figure_dirs.values(),
    ]:
        path.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(
        log_dir / f"phase_v0_8_4_3_run_{run_id}.log",
        args.verbose,
    )

    logger.info("v0.7.1 remains preserved.")
    logger.info("v0.8.3 remains the scientific lock.")
    logger.info("Starting additive Phase v0.8.4.3 final polish.")

    scores = read_tsv(v083 / "phase_v0_8_3_axis_scores.tsv")
    loadings = read_tsv(v083 / "phase_v0_8_3_axis_loadings.tsv")
    changes = read_tsv(v083 / "phase_v0_8_3_axis_within_group_changes.tsv")
    baseline = read_tsv(v083 / "phase_v0_8_3_axis_baseline_bmi_validation.tsv")
    decision = read_tsv(v083 / "phase_v0_8_3_final_axis_decision.tsv")
    coherence = read_tsv(v0822 / "phase_v0_8_2_module_coherence_audit.tsv")

    base_name = (
        "Figure_v0_8_4_3_serum_nutritional_metabolic_recovery_axis"
    )
    temp_stem = figure_dirs["specs"] / base_name

    panel_data = build_figure(scores, loadings, temp_stem)

    destinations = {
        ".png": figure_dirs["png"],
        ".tiff": figure_dirs["tiff"],
        ".svg": figure_dirs["svg"],
        ".pdf": figure_dirs["pdf"],
    }
    final_paths = []

    for suffix, destination in destinations.items():
        source = temp_stem.with_suffix(suffix)
        target = destination / source.name
        source.replace(target)
        final_paths.append(target)

    preview = figure_dirs["preview"] / f"{base_name}_preview.png"
    preview.write_bytes(final_paths[0].read_bytes())
    final_paths.append(preview)

    for panel_name, frame in panel_data.items():
        write_tsv(
            frame,
            table_dir / f"phase_v0_8_4_3_panel_{panel_name}_source_data.tsv",
        )

    audit = quality_audit(final_paths[:-1])
    write_tsv(
        audit,
        table_dir / "phase_v0_8_4_3_figure_file_quality_audit.tsv",
    )

    figure_manifest = pd.DataFrame([
        {
            "figure_name": base_name,
            "format": path.suffix.lstrip("."),
            "path": str(path.relative_to(root)),
            "publication_role": "revised candidate main manuscript figure",
        }
        for path in final_paths
    ])
    write_tsv(
        figure_manifest,
        table_dir / "phase_v0_8_4_3_figure_manifest.tsv",
    )

    draft = integration_text(decision, baseline, changes, loadings)
    draft_path = (
        manuscript_dir
        / "TB_undernutrition_proteomics_v0_8_4_3_integration_draft.md"
    )
    draft_path.write_text(draft, encoding="utf-8")

    report = f"""# Phase v0.8.4.3 final visual-lock report

Generated: {now_utc()}

## Preserved versions

- v0.7.1 manuscript baseline remains unchanged.
- v0.8.3 remains the scientific lock.
- v0.8.4 remains preserved as the first publication-facing build.

## Final polish completed

- Separated the two BMI groups using both marker shape and line style.
- Clarified that Panel D displays standardized paired change, d_z.
- Reordered Panel D chronologically from week 8 to week 52.
- Reworded Panel E as absence of a statistically significant association.
- Moved the Panel F label away from the y-axis title.
- Reduced point density and crowding in Panel F.
- Tightened Panel A and cleaned all module labels.
- Condensed the Results text and standardized scientific notation.
- Clarified that module integration followed coherence and redundancy
  adjudication.

## Publication decision

The output is suitable for scientific freeze after one final visual confirmation
that no label is clipped in the revised preview.
"""
    (
        result_dir / "phase_v0_8_4_3_final_visual_lock_report.md"
    ).write_text(report, encoding="utf-8")

    decision_log = f"""# Phase v0.8.4.3 decision log

Generated: {now_utc()}

## Reason for final polish

Visual review of v0.8.4.1 showed a minor Panel F label collision, insufficient
grayscale distinction between BMI groups, an overly absolute Panel E title, and
remaining manuscript-density and notation issues.

## Locked scientific interpretation

The preferred measure remains the serum nutritional-metabolic recovery axis.
No baseline BMI association or FDR-significant BMI-related longitudinal
interaction is claimed.

## Commit rule

Commit v0.8.4, v0.8.4.1 and v0.8.4.3 together only after confirming that the
v0.8.4.3 preview has no clipping or unintended overlap. Treat v0.8.4.3 as the
publication-facing figure and text version.
"""
    (
        log_dir / "phase_v0_8_4_3_decision_log.md"
    ).write_text(decision_log, encoding="utf-8")

    manifest = {
        "script_version": SCRIPT_VERSION,
        "generated_utc": now_utc(),
        "baseline_preserved": "v0.7.1",
        "scientific_lock": "v0.8.3",
        "first_publication_build_preserved": "v0.8.4",
        "quality_repair_preserved": "v0.8.4.1",
        "output_tag": OUTPUT_TAG,
        "preferred_axis": PREFERRED_AXIS,
        "n_samples": len(scores),
        "n_subjects": scores["subject_id"].nunique(),
        "figure_files": [str(x.relative_to(root)) for x in final_paths],
        "manuscript_draft": str(draft_path.relative_to(root)),
    }
    write_json(
        manifest,
        result_dir / "phase_v0_8_4_3_run_manifest.json",
    )

    logger.info("Phase v0.8.4.3 completed.")
    logger.info("Preview: %s", preview)
    logger.info("Draft: %s", draft_path)
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
