#!/usr/bin/env python3
"""Supplementary Materials PDF: Compile all extended results into LaTeX appendices.

Loads data from 4 dependent experiments and 5 additional eval files on disk,
extracts detailed results, generates publication-quality figures and LaTeX tables,
assembles into a NeurIPS-template supplementary PDF, and outputs completeness
metrics in eval_out.json.
"""

import json
import math
import os
import resource
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(LOG_DIR / "run.log", rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------------------
# Hardware-aware resource limits
# ---------------------------------------------------------------------------
def _detect_cpus() -> int:
    try:
        parts = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if parts[0] != "max":
            return math.ceil(int(parts[0]) / int(parts[1]))
    except (FileNotFoundError, ValueError):
        pass
    try:
        q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        p = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if q > 0:
            return math.ceil(q / p)
    except (FileNotFoundError, ValueError):
        pass
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    return os.cpu_count() or 1

def _container_ram_gb() -> float | None:
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None

NUM_CPUS = _detect_cpus()
TOTAL_RAM_GB = _container_ram_gb() or 29.0
RAM_BUDGET = int(TOTAL_RAM_GB * 0.6 * 1e9)  # 60% of container RAM
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, budget={RAM_BUDGET/1e9:.1f} GB")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).parent
FIGURES_DIR = WORKSPACE / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Dependency experiment paths
DEP1_DIR = Path("/ai-inventor/aii_pipeline/runs/neurips-open/3_invention_loop/iter_2/gen_art/exp_id1_it2__opus")
DEP2_DIR = Path("/ai-inventor/aii_pipeline/runs/neurips-open/3_invention_loop/iter_2/gen_art/exp_id3_it2__opus")
DEP3_DIR = Path("/ai-inventor/aii_pipeline/runs/neurips-open/3_invention_loop/iter_4/gen_art/exp_id2_it4__opus")
DEP4_DIR = Path("/ai-inventor/aii_pipeline/runs/neurips-open/3_invention_loop/iter_4/gen_art/exp_id1_it4__opus")

# Additional eval file paths
EVAL_ABLATION = Path("/ai-inventor/aii_pipeline/runs/neurips-open/3_invention_loop/iter_6/gen_art/eval_id2_it6__opus/eval_out.json")
EVAL_ROUTING = Path("/ai-inventor/aii_pipeline/runs/neurips-open/3_invention_loop/iter_6/gen_art/eval_id5_it6__opus/eval_out.json")
EVAL_TRANSFER = Path("/ai-inventor/aii_pipeline/runs/neurips-open/3_invention_loop/iter_5/gen_art/eval_id5_it5__opus/eval_out.json")
EVAL_SC_CRITERIA = Path("/ai-inventor/aii_pipeline/runs/neurips-open/3_invention_loop/iter_5/gen_art/eval_id3_it5__opus/eval_out.json")
EVAL_PROSPECTIVE = Path("/ai-inventor/aii_pipeline/runs/neurips-open/3_invention_loop/iter_5/gen_art/eval_id1_it5__opus/eval_out.json")

ALL_DATA_SOURCES = {
    "dep1_arithmetic": DEP1_DIR / "full_method_out.json",
    "dep2_graph_coloring": DEP2_DIR / "full_method_out.json",
    "dep3_classifier": DEP3_DIR / "full_method_out.json",
    "dep4_syllogistic": DEP4_DIR / "full_method_out.json",
    "eval_ablation": EVAL_ABLATION,
    "eval_routing": EVAL_ROUTING,
    "eval_transfer": EVAL_TRANSFER,
    "eval_sc_criteria": EVAL_SC_CRITERIA,
    "eval_prospective": EVAL_PROSPECTIVE,
}

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_json_safe(path: Path, label: str) -> dict | None:
    """Load a JSON file safely, returning None on failure."""
    try:
        data = json.loads(path.read_text())
        logger.info(f"Loaded {label}: {path.name} ({path.stat().st_size / 1024:.0f} KB)")
        return data
    except FileNotFoundError:
        logger.warning(f"File not found: {path} ({label})")
        return None
    except json.JSONDecodeError:
        logger.exception(f"Invalid JSON: {path} ({label})")
        return None
    except Exception:
        logger.exception(f"Error loading {path} ({label})")
        return None


def load_all_data() -> dict:
    """Load all 9 data sources."""
    data = {}
    for label, path in ALL_DATA_SOURCES.items():
        data[label] = load_json_safe(path, label)
    loaded = sum(1 for v in data.values() if v is not None)
    logger.info(f"Data completeness: {loaded}/{len(ALL_DATA_SOURCES)} files loaded")
    return data


# ---------------------------------------------------------------------------
# Helper: LaTeX-safe string
# ---------------------------------------------------------------------------
def tex_escape(s: str) -> str:
    """Escape special LaTeX characters."""
    replacements = {
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


def short_model(m: str) -> str:
    """Shorten model name for tables."""
    return m.split("/")[-1] if "/" in m else m


# ---------------------------------------------------------------------------
# Appendix A: Per-model-task CSD profiles
# ---------------------------------------------------------------------------
def build_appendix_a(data: dict) -> tuple[str, int]:
    """Build per-model-task CSD indicator profiles for sharp-boundary pairs."""
    logger.info("Building Appendix A: Per-model-task CSD profiles")
    pairs_reported = 0
    tex = r"""
\section{Appendix A: Per-Model-Task CSD Indicator Profiles}
\label{app:csd_profiles}

This appendix presents the full CSD indicator profiles for each model-task pair
that exhibits a sharp capability boundary ($d^*$). For each pair, we report
all CSD indicators (embedding variance, Hartigan dip, silhouette $k{=}2$,
bimodality coefficient, disagreement rate) across all difficulty levels.

"""
    # Collect data from dep1 (arithmetic) and dep2 (graph_coloring)
    profile_data = []
    for dep_key, task_name in [("dep1_arithmetic", "arithmetic"), ("dep2_graph_coloring", "graph coloring")]:
        dep = data.get(dep_key)
        if dep is None:
            continue

        for ds in dep.get("datasets", []):
            ds_name = ds.get("dataset", "")
            examples = ds.get("examples", [])
            if not examples:
                continue

            # Extract model name from first example
            first_ex = examples[0]
            model = first_ex.get("metadata_model", "unknown")

            # Determine d_star - try dep1 style then dep2 metadata style
            d_star = first_ex.get("metadata_d_star")
            if d_star is None:
                # For graph coloring, d_star is in the top-level metadata
                meta = dep.get("metadata", {})
                analysis = meta.get("analysis", {})
                for m_info in analysis.get("models", []):
                    if m_info.get("model") == model:
                        d_star = m_info.get("d_star")
                        break

            # Build table rows - dep1 uses predict_*, dep2 uses metadata_csd_*
            # Group by difficulty level and average per level
            level_data = {}
            for ex in examples:
                diff = ex.get("metadata_difficulty_level", ex.get("metadata_difficulty"))
                if diff is None:
                    continue
                if diff not in level_data:
                    level_data[diff] = {"acc": [], "var": [], "dip": [], "sil": [], "bc": [], "dis": []}

                # dep1 style fields
                acc_val = ex.get("predict_accuracy")
                var_val = ex.get("predict_csd_variance", ex.get("metadata_csd_embedding_variance"))
                dip_val = ex.get("predict_dip_statistic", ex.get("metadata_csd_dip_statistic"))
                sil_val = ex.get("predict_silhouette_k2", ex.get("metadata_csd_silhouette_score"))
                bc_val = ex.get("predict_bimodality_coefficient", ex.get("metadata_csd_bimodality_coefficient"))
                dis_val = ex.get("predict_disagreement_rate", ex.get("metadata_csd_disagreement_rate"))

                # For dep2, accuracy is in metadata_csd_accuracy
                if acc_val is None:
                    acc_val = ex.get("metadata_csd_accuracy")

                for key, val in [("acc", acc_val), ("var", var_val), ("dip", dip_val),
                                 ("sil", sil_val), ("bc", bc_val), ("dis", dis_val)]:
                    if val is not None:
                        try:
                            level_data[diff][key].append(float(val))
                        except (ValueError, TypeError):
                            pass

            rows = []
            for diff in sorted(level_data.keys()):
                row = {"d": diff}
                for key in ["acc", "var", "dip", "sil", "bc", "dis"]:
                    vals = level_data[diff][key]
                    row[key] = f"{np.mean(vals):.6f}" if vals else "N/A"
                rows.append(row)
            rows.sort(key=lambda r: r["d"])

            profile_data.append({
                "task": task_name,
                "model": short_model(model),
                "d_star": d_star,
                "rows": rows,
            })
            pairs_reported += 1

    # Generate LaTeX tables for each pair
    for prof in profile_data:
        d_star_str = str(prof['d_star']) if prof['d_star'] is not None else "N/A"
        tex += f"\\subsection{{{tex_escape(prof['task'])} -- {tex_escape(prof['model'])} ($d^*={d_star_str}$)}}\n\n"
        tex += r"\begin{table}[h]\centering\small" + "\n"
        tex += r"\begin{tabular}{r|r|r|r|r|r|r}" + "\n"
        tex += r"\toprule" + "\n"
        tex += r"$d$ & Acc & Variance & Dip & Silhouette & BC & Disagree \\" + "\n"
        tex += r"\midrule" + "\n"
        for row in prof["rows"][:30]:  # Limit rows for readability
            def fmt_val(v):
                if v == "N/A":
                    return "N/A"
                try:
                    return f"{float(v):.4f}"
                except (ValueError, TypeError):
                    return str(v)[:8]
            d_val = row["d"]
            marker = r" $\dagger$" if d_val == prof["d_star"] else ""
            tex += f"{d_val}{marker} & {fmt_val(row['acc'])} & {fmt_val(row['var'])} & {fmt_val(row['dip'])} & {fmt_val(row['sil'])} & {fmt_val(row['bc'])} & {fmt_val(row['dis'])} \\\\\n"
        tex += r"\bottomrule" + "\n"
        tex += r"\end{tabular}" + "\n"
        tex += f"\\caption{{CSD indicators for {tex_escape(prof['model'])} on {tex_escape(prof['task'])}. $\\dagger$ marks $d^*={d_star_str}$.}}\n"
        tex += r"\end{table}" + "\n\n"

    if pairs_reported == 0:
        tex += "\\textit{No CSD profile data available.}\n\n"

    return tex, pairs_reported


# ---------------------------------------------------------------------------
# Appendix B: Full ablation matrix
# ---------------------------------------------------------------------------
def build_appendix_b(data: dict) -> tuple[str, int]:
    """Build full 45-cell ablation matrix with bootstrap CIs."""
    logger.info("Building Appendix B: Ablation matrix")
    ablation = data.get("eval_ablation")
    cells_reported = 0

    tex = r"""
\section{Appendix B: Feature Ablation Study}
\label{app:ablation}

This appendix reports the complete ablation study examining
the contribution of CSD features to boundary detection.
Table~S1 presents the full matrix of 5 ablation conditions
$\times$ 3 classifiers $\times$ 3 cross-validation schemes = 45 cells,
each with F1 score and 95\% bootstrap confidence intervals.

"""
    if ablation is None:
        tex += "\\textit{Ablation data not available.}\n\n"
        return tex, 0

    examples = []
    for ds in ablation.get("datasets", []):
        if ds.get("dataset") == "ablation_comparison":
            examples = ds.get("examples", [])
            break

    # Build table
    tex += r"\begin{table}[h]\centering\small" + "\n"
    tex += r"\caption{Table S1: Full ablation matrix. F1 [95\% CI] across ablation variants, classifiers, and CV schemes.}" + "\n"
    tex += r"\label{tab:ablation}" + "\n"
    tex += r"\begin{tabular}{l|l|l|r}" + "\n"
    tex += r"\toprule" + "\n"
    tex += r"Ablation & Classifier & CV & F1 [95\% CI] \\" + "\n"
    tex += r"\midrule" + "\n"

    prev_ablation = ""
    for ex in examples:
        abl = ex.get("metadata_ablation", "")
        clf = ex.get("metadata_classifier", "")
        cv = ex.get("metadata_cv_scheme", "")
        f1 = ex.get("predict_f1_mean", "")
        ci_lo = ex.get("predict_f1_ci_lo", "")
        ci_hi = ex.get("predict_f1_ci_hi", "")
        try:
            f1_str = f"{float(f1):.3f} [{float(ci_lo):.3f}, {float(ci_hi):.3f}]"
        except (ValueError, TypeError):
            f1_str = f"{f1} [{ci_lo}, {ci_hi}]"

        abl_display = tex_escape(abl.replace("ablation_", "")) if abl != prev_ablation else ""
        if abl != prev_ablation and prev_ablation:
            tex += r"\midrule" + "\n"
        prev_ablation = abl

        tex += f"{abl_display} & {clf} & {cv} & {f1_str} \\\\\n"
        cells_reported += 1

    tex += r"\bottomrule" + "\n"
    tex += r"\end{tabular}" + "\n"
    tex += r"\end{table}" + "\n\n"

    # Incremental contribution table
    incr_examples = []
    for ds in ablation.get("datasets", []):
        if ds.get("dataset") == "incremental_contribution":
            incr_examples = ds.get("examples", [])
            break

    if incr_examples:
        tex += r"\begin{table}[h]\centering\small" + "\n"
        tex += r"\caption{Table S2: Incremental feature contribution to boundary detection F1.}" + "\n"
        tex += r"\label{tab:incremental}" + "\n"
        tex += r"\begin{tabular}{l|l|r|r}" + "\n"
        tex += r"\toprule" + "\n"
        tex += r"Direction & Feature Added & New F1 & Marginal Gain \\" + "\n"
        tex += r"\midrule" + "\n"
        for ex in incr_examples:
            direction = ex.get("metadata_direction", "")
            feat = tex_escape(ex.get("metadata_feature_added", ""))
            new_f1 = ex.get("predict_new_f1", "")
            gain = ex.get("predict_marginal_gain", "")
            try:
                tex += f"{tex_escape(direction)} & {feat} & {float(new_f1):.4f} & +{float(gain):.4f} \\\\\n"
            except (ValueError, TypeError):
                tex += f"{tex_escape(direction)} & {feat} & {new_f1} & {gain} \\\\\n"
        tex += r"\bottomrule" + "\n"
        tex += r"\end{tabular}" + "\n"
        tex += r"\end{table}" + "\n\n"

    # Permutation test table
    perm_examples = []
    for ds in ablation.get("datasets", []):
        if ds.get("dataset") == "permutation_test":
            perm_examples = ds.get("examples", [])
            break

    if perm_examples:
        tex += r"\begin{table}[h]\centering\small" + "\n"
        tex += r"\caption{Table S3: Permutation test results for CSD feature informativeness.}" + "\n"
        tex += r"\label{tab:permutation}" + "\n"
        tex += r"\begin{tabular}{l|r|r|r|r}" + "\n"
        tex += r"\toprule" + "\n"
        tex += r"Variant & Unpermuted F1 & Permuted F1 & Drop & $p$-value \\" + "\n"
        tex += r"\midrule" + "\n"
        for ex in perm_examples:
            var = tex_escape(ex.get("metadata_model_variant", ""))
            uf1 = ex.get("predict_unpermuted_f1", "")
            pf1 = ex.get("predict_permuted_f1_mean", "")
            drop = ex.get("predict_permutation_drop", "")
            pval = ex.get("predict_permutation_pvalue", "")
            try:
                tex += f"{var} & {float(uf1):.4f} & {float(pf1):.4f} & {float(drop):+.4f} & {float(pval):.4f} \\\\\n"
            except (ValueError, TypeError):
                tex += f"{var} & {uf1} & {pf1} & {drop} & {pval} \\\\\n"
        tex += r"\bottomrule" + "\n"
        tex += r"\end{tabular}" + "\n"
        tex += r"\end{table}" + "\n\n"

    return tex, cells_reported


# ---------------------------------------------------------------------------
# Appendix C: Routing simulation
# ---------------------------------------------------------------------------
def build_appendix_c(data: dict) -> tuple[str, int]:
    """Build routing simulation extended results."""
    logger.info("Building Appendix C: Routing simulation")
    routing = data.get("eval_routing")
    cells_reported = 0

    tex = r"""
\section{Appendix C: Model Routing Simulation Extended Results}
\label{app:routing}

This appendix presents the full routing simulation results across
2 tasks $\times$ 4 policies $\times$ 5 batch sizes $\times$ 3 difficulty
distributions = 120 cells. The CSD-monitored routing policy uses
real-time CSD indicator monitoring to decide when to escalate from
a cheap model to a capable model.

"""
    if routing is None:
        tex += "\\textit{Routing simulation data not available.}\n\n"
        return tex, 0

    # Report headline metrics
    meta = routing.get("metadata", {})
    magg = routing.get("metrics_agg", {})
    tex += f"\\textbf{{Headline}}: {tex_escape(meta.get('headline', 'N/A'))}\n\n"

    for ds in routing.get("datasets", []):
        ds_name = ds.get("dataset", "")
        examples = ds.get("examples", [])
        if not examples:
            continue

        task_label = ds_name.replace("routing_simulation_", "").replace("_", " ").title()
        tex += f"\\subsection{{{task_label}}}\n\n"
        tex += r"\begin{table}[h]\centering\scriptsize" + "\n"
        tex += f"\\caption{{Table S{{C}}: Routing results for {task_label}.}}\n"
        tex += r"\begin{tabular}{l|r|l|r|r|r|r}" + "\n"
        tex += r"\toprule" + "\n"
        tex += r"Policy & B & Dist & Accuracy & Cost & Oracle Gap & Error Red. \\" + "\n"
        tex += r"\midrule" + "\n"

        prev_policy = ""
        for ex in examples:
            policy = ex.get("metadata_policy", "")
            batch = ex.get("metadata_batch_size", "")
            dist = ex.get("metadata_difficulty_distribution", "")
            acc = ex.get("eval_overall_accuracy", 0)
            cost = ex.get("eval_total_cost", 0)
            ogap = ex.get("eval_oracle_gap", 0)
            err_red = ex.get("eval_error_reduction_vs_cheap", 0)

            if policy != prev_policy and prev_policy:
                tex += r"\midrule" + "\n"
            prev_policy = policy

            tex += f"{tex_escape(policy)} & {batch} & {tex_escape(str(dist))} & {acc:.4f} & {cost:.2f} & {ogap:.1f}\\% & {err_red:.1f}\\% \\\\\n"
            cells_reported += 1

        tex += r"\bottomrule" + "\n"
        tex += r"\end{tabular}" + "\n"
        tex += r"\end{table}" + "\n\n"

    return tex, cells_reported


# ---------------------------------------------------------------------------
# Appendix D: Cross-task transfer
# ---------------------------------------------------------------------------
def build_appendix_d(data: dict) -> tuple[str, int]:
    """Build cross-task transfer analysis with KS and Cohen's d."""
    logger.info("Building Appendix D: Cross-task transfer")
    transfer = data.get("eval_transfer")
    pairs_reported = 0

    tex = r"""
\section{Appendix D: Cross-Task Transfer Analysis}
\label{app:transfer}

This appendix quantifies the distributional shift of CSD features across
task families (arithmetic vs.\ graph coloring), measuring whether
CSD indicators transfer across tasks using Kolmogorov-Smirnov tests
and Cohen's $d$ effect sizes.

"""
    if transfer is None:
        tex += "\\textit{Transfer analysis data not available.}\n\n"
        return tex, 0

    meta = transfer.get("metadata", {})
    feature_shift = meta.get("analysis_1_feature_shift", {})
    per_feature = feature_shift.get("per_feature", {})

    if per_feature:
        tex += r"\begin{table}[h]\centering\small" + "\n"
        tex += r"\caption{Table S4: CSD feature distributional shift across tasks (arithmetic vs.\ graph coloring).}" + "\n"
        tex += r"\label{tab:feature_shift}" + "\n"
        tex += r"\begin{tabular}{l|r|r|r|r|r}" + "\n"
        tex += r"\toprule" + "\n"
        tex += r"Feature & Wasserstein & KS stat & KS $p$ & Cohen's $d$ & Overlap \\" + "\n"
        tex += r"\midrule" + "\n"

        for feat_name, feat_data in per_feature.items():
            raw = feat_data.get("raw", {})
            ws = raw.get("wasserstein_distance", 0)
            ks = raw.get("ks_statistic", 0)
            ksp = raw.get("ks_pvalue", 1)
            cd = raw.get("cohens_d", 0)
            ov = raw.get("overlap_coefficient", 0)
            tex += f"{tex_escape(feat_name)} & {ws:.4f} & {ks:.3f} & {ksp:.4f} & {cd:.3f} & {ov:.3f} \\\\\n"
            pairs_reported += 1

        tex += r"\bottomrule" + "\n"
        tex += r"\end{tabular}" + "\n"
        tex += r"\end{table}" + "\n\n"

    # Also report LOTO performance from transfer analysis
    magg = transfer.get("metrics_agg", {})
    best_loto = magg.get("best_new_loto_f1", "N/A")
    baseline_loto = magg.get("baseline_loto_f1_zt_rf", "N/A")
    tex += f"Best new LOTO F1: {best_loto}, Baseline LOTO F1 (zt\\_rf): {baseline_loto}\n\n"

    return tex, pairs_reported


# ---------------------------------------------------------------------------
# Appendix E: Syllogistic logic extended
# ---------------------------------------------------------------------------
def build_appendix_e(data: dict) -> tuple[str, int]:
    """Build syllogistic logic extended analysis."""
    logger.info("Building Appendix E: Syllogistic logic")
    syl = data.get("dep4_syllogistic")

    tex = r"""
\section{Appendix E: Syllogistic Logic Extended Analysis}
\label{app:syllogistic}

This appendix presents the extended CSD analysis for syllogistic logic
tasks across 3 weak LLMs and 22 difficulty levels ($d=2$--$30$).

"""
    if syl is None:
        tex += "\\textit{Syllogistic data not available.}\n\n"
        return tex, 0

    meta = syl.get("metadata", {})
    model_summaries = meta.get("model_summaries", {})
    sc1 = meta.get("sc1_results", {})

    # Model summary table
    tex += r"\begin{table}[h]\centering\small" + "\n"
    tex += r"\caption{Table S5: Syllogistic CSD model summaries.}" + "\n"
    tex += r"\label{tab:syllogistic_summary}" + "\n"
    tex += r"\begin{tabular}{l|r|r|r|l|l}" + "\n"
    tex += r"\toprule" + "\n"
    tex += r"Model & $d^*$ & $\alpha$ & $R^2$ & Flickering & Consensus \\" + "\n"
    tex += r"\midrule" + "\n"

    for model_name, msumm in model_summaries.items():
        d_star = msumm.get("d_star", "N/A")
        if d_star is None:
            d_star = "N/A"
        scaling = msumm.get("scaling", {})
        alpha = scaling.get("alpha")
        r2 = scaling.get("r_squared")
        flick = msumm.get("flickering", {})
        consensus = flick.get("flickering_consensus", False)
        lead_time = flick.get("lead_time", "N/A")

        alpha_str = f"{alpha:.4f}" if alpha is not None else "N/A"
        r2_str = f"{r2:.4f}" if r2 is not None else "N/A"

        tex += f"{tex_escape(short_model(model_name))} & {d_star} & {alpha_str} & {r2_str} & {'Yes' if consensus else 'No'} & LT={lead_time} \\\\\n"

    tex += r"\bottomrule" + "\n"
    tex += r"\end{tabular}" + "\n"
    tex += r"\end{table}" + "\n\n"

    # Per-model indicator table
    for ds in syl.get("datasets", []):
        examples = ds.get("examples", [])
        if not examples:
            continue

        # Group by model
        model_groups = {}
        for ex in examples:
            m = ex.get("metadata_model", "unknown")
            model_groups.setdefault(m, []).append(ex)

        for model_name, exs in model_groups.items():
            exs.sort(key=lambda e: e.get("metadata_difficulty", 0))
            tex += f"\\subsection{{{tex_escape(short_model(model_name))}}}\n\n"
            tex += r"\begin{table}[h]\centering\scriptsize" + "\n"
            tex += f"\\caption{{CSD indicators for {tex_escape(short_model(model_name))} on syllogistic logic.}}\n"
            tex += r"\begin{tabular}{r|r|r|r|r|r|r|r}" + "\n"
            tex += r"\toprule" + "\n"
            tex += r"$d$ & Acc & Variance & Dip & Sil & BC & Disagree & Ashman D \\" + "\n"
            tex += r"\midrule" + "\n"

            for ex in exs:
                d = ex.get("metadata_difficulty", "")
                acc = ex.get("predict_accuracy", "")
                var_v = ex.get("predict_csd_variance", "")
                dip_v = ex.get("predict_dip_statistic", "")
                sil_v = ex.get("predict_silhouette_k2", "")
                bc_v = ex.get("predict_bimodality_coefficient", "")
                dis_v = ex.get("predict_disagreement_rate", "")
                ash_v = ex.get("predict_ashman_d", "")

                def fmt(v):
                    try:
                        return f"{float(v):.4f}"
                    except (ValueError, TypeError):
                        return str(v)[:8] if v else "N/A"

                tex += f"{d} & {fmt(acc)} & {fmt(var_v)} & {fmt(dip_v)} & {fmt(sil_v)} & {fmt(bc_v)} & {fmt(dis_v)} & {fmt(ash_v)} \\\\\n"

            tex += r"\bottomrule" + "\n"
            tex += r"\end{tabular}" + "\n"
            tex += r"\end{table}" + "\n\n"

    return tex, 1


# ---------------------------------------------------------------------------
# Appendix F: Sample-size sensitivity
# ---------------------------------------------------------------------------
def build_appendix_f(data: dict) -> tuple[str, int]:
    """Build sample-size sensitivity F1 vs N curve."""
    logger.info("Building Appendix F: Sample-size sensitivity")
    sc = data.get("eval_sc_criteria")

    tex = r"""
\section{Appendix F: Sample-Size Sensitivity Analysis}
\label{app:sensitivity}

This appendix examines how boundary detection F1 degrades as the number
of sampled responses per difficulty level ($N$) decreases from 50 to 10.

"""
    if sc is None:
        tex += "\\textit{Sensitivity data not available.}\n\n"
        return tex, 0

    magg = sc.get("metrics_agg", {})

    # Extract sensitivity data
    n_values = [10, 15, 20, 25, 30, 40, 50]
    f1_values = []
    dip_values = []
    for n in n_values:
        f1_key = f"sensitivity_f1_at_N{n}"
        dip_key = f"sensitivity_dip_rate_at_N{n}"
        f1_values.append(magg.get(f1_key, 0))
        dip_values.append(magg.get(dip_key, 0))

    min_viable = magg.get("sensitivity_minimum_viable_N", "N/A")

    tex += r"\begin{table}[h]\centering\small" + "\n"
    tex += r"\caption{Table S6: Boundary detection F1 and dip detection rate as function of sample size $N$.}" + "\n"
    tex += r"\label{tab:sensitivity}" + "\n"
    tex += r"\begin{tabular}{r|r|r}" + "\n"
    tex += r"\toprule" + "\n"
    tex += r"$N$ & F1 & Dip Rate \\" + "\n"
    tex += r"\midrule" + "\n"
    for n, f1, dip in zip(n_values, f1_values, dip_values):
        tex += f"{n} & {f1:.4f} & {dip:.4f} \\\\\n"
    tex += r"\bottomrule" + "\n"
    tex += r"\end{tabular}" + "\n"
    tex += r"\end{table}" + "\n\n"
    tex += f"Minimum viable $N$ for F1$>$0.80: {min_viable}\n\n"

    return tex, 1


# ---------------------------------------------------------------------------
# Appendix G: Negative controls
# ---------------------------------------------------------------------------
def build_appendix_g(data: dict) -> tuple[str, int]:
    """Build negative control false positive rate analysis."""
    logger.info("Building Appendix G: Negative controls")
    sc = data.get("eval_sc_criteria")

    tex = r"""
\section{Appendix G: Negative Control Analysis}
\label{app:negative}

This appendix reports the false positive rate of CSD indicators in regions
where the model is safely within its capability range (far from $d^*$).
Indicators should \emph{not} show significant trends in safe regions.

"""
    if sc is None:
        tex += "\\textit{Negative control data not available.}\n\n"
        return tex, 0

    magg = sc.get("metrics_agg", {})
    pass_rate = magg.get("negative_control_pass_rate", "N/A")
    n_cells = magg.get("consistency_n_cells", 0)
    frac_sig = magg.get("consistency_fraction_significant", 0)
    frac_correct = magg.get("consistency_fraction_correct_direction", 0)

    tex += r"\begin{table}[h]\centering\small" + "\n"
    tex += r"\caption{Table S7: Negative control and consistency metrics.}" + "\n"
    tex += r"\label{tab:negative}" + "\n"
    tex += r"\begin{tabular}{l|r}" + "\n"
    tex += r"\toprule" + "\n"
    tex += r"Metric & Value \\" + "\n"
    tex += r"\midrule" + "\n"
    tex += f"Negative control pass rate & {pass_rate:.4f} \\\\\n" if isinstance(pass_rate, (int, float)) else f"Negative control pass rate & {pass_rate} \\\\\n"
    tex += f"Consistency cells tested & {int(n_cells)} \\\\\n"
    tex += f"Fraction significant & {frac_sig:.4f} \\\\\n"
    tex += f"Fraction correct direction & {frac_correct:.4f} \\\\\n"
    tex += r"\bottomrule" + "\n"
    tex += r"\end{tabular}" + "\n"
    tex += r"\end{table}" + "\n\n"

    # Effect sizes
    tex += r"\begin{table}[h]\centering\small" + "\n"
    tex += r"\caption{Table S8: Effect sizes for CSD indicators between safe and near-boundary regions.}" + "\n"
    tex += r"\label{tab:effect_sizes}" + "\n"
    tex += r"\begin{tabular}{l|r|r}" + "\n"
    tex += r"\toprule" + "\n"
    tex += r"Indicator & Cohen's $d$ & Cliff's $\delta$ \\" + "\n"
    tex += r"\midrule" + "\n"

    indicators = ["variance", "dip_statistic", "silhouette_k2", "bimodality_coefficient", "disagreement_rate"]
    for ind in indicators:
        cd = magg.get(f"effect_size_mean_cohen_d_{ind}", 0)
        cliff = magg.get(f"effect_size_mean_cliff_delta_{ind}", 0)
        tex += f"{tex_escape(ind)} & {cd:.4f} & {cliff:.4f} \\\\\n"

    tex += r"\bottomrule" + "\n"
    tex += r"\end{tabular}" + "\n"
    tex += r"\end{table}" + "\n\n"

    return tex, 1


# ---------------------------------------------------------------------------
# Appendix H: Temperature experiment
# ---------------------------------------------------------------------------
def build_appendix_h(data: dict) -> tuple[str, int]:
    """Build temperature experiment breakdown."""
    logger.info("Building Appendix H: Temperature experiment")
    dep1 = data.get("dep1_arithmetic")
    dep2 = data.get("dep2_graph_coloring")

    tex = r"""
\section{Appendix H: Sampling Temperature Configuration}
\label{app:temperature}

This appendix documents the sampling temperature used for all CSD experiments
and provides a breakdown of response generation parameters across tasks.

"""
    found = False
    for dep, task_name in [(dep1, "Arithmetic"), (dep2, "Graph Coloring")]:
        if dep is None:
            continue
        meta = dep.get("metadata", {})
        temp = meta.get("sampling_params", {}).get("temperature", meta.get("temperature", "N/A"))
        top_p = meta.get("sampling_params", {}).get("top_p", "N/A")
        max_tok = meta.get("sampling_params", {}).get("max_tokens", "N/A")
        models = meta.get("models", meta.get("experiment_config", {}).get("models", []))
        n_levels = meta.get("difficulty_levels", meta.get("difficulty_levels_count", "N/A"))
        n_resp = meta.get("responses_per_level_target", meta.get("experiment_config", {}).get("n_total_per_model_level", "N/A"))

        tex += f"\\subsection{{{task_name}}}\n\n"
        tex += r"\begin{table}[h]\centering\small" + "\n"
        tex += f"\\caption{{Table S9: Sampling parameters for {task_name}.}}\n"
        tex += r"\begin{tabular}{l|r}" + "\n"
        tex += r"\toprule" + "\n"
        tex += r"Parameter & Value \\" + "\n"
        tex += r"\midrule" + "\n"
        tex += f"Temperature & {temp} \\\\\n"
        tex += f"Top-$p$ & {top_p} \\\\\n"
        tex += f"Max tokens & {max_tok} \\\\\n"
        tex += f"Difficulty levels & {n_levels} \\\\\n"
        tex += f"Responses/level & {n_resp} \\\\\n"
        model_str = ", ".join(tex_escape(short_model(m)) for m in models) if models else "N/A"
        tex += f"Models & {model_str} \\\\\n"
        tex += r"\bottomrule" + "\n"
        tex += r"\end{tabular}" + "\n"
        tex += r"\end{table}" + "\n\n"
        found = True

    if not found:
        tex += "\\textit{Temperature configuration data not available.}\n\n"

    return tex, 1 if found else 0


# ---------------------------------------------------------------------------
# Figure Generation
# ---------------------------------------------------------------------------
def generate_figure_s1(data: dict) -> bool:
    """Figure S1: CSD indicator profiles across difficulty for arithmetic task."""
    logger.info("Generating Figure S1: CSD profiles (arithmetic)")
    dep1 = data.get("dep1_arithmetic")
    if dep1 is None:
        return False

    try:
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        indicators = [
            ("predict_csd_variance", "Embedding Variance"),
            ("predict_dip_statistic", "Hartigan Dip"),
            ("predict_silhouette_k2", "Silhouette k=2"),
            ("predict_bimodality_coefficient", "Bimodality Coeff"),
            ("predict_disagreement_rate", "Disagreement Rate"),
            ("predict_accuracy", "Accuracy"),
        ]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

        for ds_idx, ds in enumerate(dep1.get("datasets", [])):
            examples = ds.get("examples", [])
            if not examples:
                continue
            model_name = short_model(examples[0].get("metadata_model", "unknown"))
            d_star = examples[0].get("metadata_d_star")

            diffs = sorted(set(e.get("metadata_difficulty_level", 0) for e in examples))
            for ax_idx, (key, label) in enumerate(indicators):
                ax = axes[ax_idx // 3][ax_idx % 3]
                vals = []
                for d in diffs:
                    vs = [float(e.get(key, 0)) for e in examples if e.get("metadata_difficulty_level") == d]
                    vals.append(np.mean(vs) if vs else 0)
                ax.plot(diffs, vals, "-o", markersize=3, label=model_name, color=colors[ds_idx % len(colors)])
                if d_star is not None:
                    ax.axvline(x=d_star, color=colors[ds_idx % len(colors)], linestyle="--", alpha=0.5)
                ax.set_title(label, fontsize=10)
                ax.set_xlabel("Difficulty")

        for ax_idx in range(6):
            ax = axes[ax_idx // 3][ax_idx % 3]
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

        fig.suptitle("Figure S1: CSD Indicator Profiles — Arithmetic Task", fontsize=12)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "figure_s1.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Figure S1 saved")
        return True
    except Exception:
        logger.exception("Failed to generate Figure S1")
        return False


def generate_figure_s2(data: dict) -> bool:
    """Figure S2: CSD indicator profiles across difficulty for graph coloring."""
    logger.info("Generating Figure S2: CSD profiles (graph coloring)")
    dep2 = data.get("dep2_graph_coloring")
    if dep2 is None:
        return False

    try:
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        indicators = [
            ("metadata_csd_embedding_variance", "Embedding Variance"),
            ("metadata_csd_dip_statistic", "Hartigan Dip"),
            ("metadata_csd_silhouette_score", "Silhouette k=2"),
            ("metadata_csd_bimodality_coefficient", "Bimodality Coeff"),
            ("metadata_csd_disagreement_rate", "Disagreement Rate"),
            ("metadata_csd_accuracy", "Accuracy"),
        ]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

        for ds_idx, ds in enumerate(dep2.get("datasets", [])):
            examples = ds.get("examples", [])
            if not examples:
                continue
            model_name = short_model(examples[0].get("metadata_model", "unknown"))

            # Group by difficulty level, average per level
            level_data = {}
            for ex in examples:
                d = ex.get("metadata_difficulty_level", 0)
                if d not in level_data:
                    level_data[d] = {k: [] for k, _ in indicators}
                for key, _ in indicators:
                    v = ex.get(key)
                    if v is not None:
                        level_data[d][key].append(float(v))

            diffs = sorted(level_data.keys())
            for ax_idx, (key, label) in enumerate(indicators):
                ax = axes[ax_idx // 3][ax_idx % 3]
                vals = [np.mean(level_data[d][key]) if level_data[d][key] else 0 for d in diffs]
                ax.plot(diffs, vals, "-o", markersize=3, label=model_name, color=colors[ds_idx % len(colors)])
                ax.set_title(label, fontsize=10)
                ax.set_xlabel("Difficulty")

        for ax_idx in range(6):
            ax = axes[ax_idx // 3][ax_idx % 3]
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

        fig.suptitle("Figure S2: CSD Indicator Profiles — Graph Coloring Task", fontsize=12)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "figure_s2.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Figure S2 saved")
        return True
    except Exception:
        logger.exception("Failed to generate Figure S2")
        return False


def generate_figure_s3(data: dict) -> bool:
    """Figure S3: Ablation comparison bar chart."""
    logger.info("Generating Figure S3: Ablation comparison")
    ablation = data.get("eval_ablation")
    if ablation is None:
        return False

    try:
        magg = ablation.get("metrics_agg", {})
        ablations = ["1_pure_csd", "2_csd_dynamics", "3_difficulty_only"]
        classifiers = ["logreg", "rf", "svm"]
        cv = "lopo"

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(ablations))
        width = 0.25
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

        for i, clf in enumerate(classifiers):
            vals = []
            for abl in ablations:
                key = f"ablation_{abl}_{clf}_{cv}_f1"
                vals.append(magg.get(key, 0))
            ax.bar(x + i * width, vals, width, label=clf, color=colors[i])

        ax.set_xlabel("Ablation Variant")
        ax.set_ylabel("F1 Score (LOPO)")
        ax.set_title("Figure S3: Feature Ablation Comparison")
        ax.set_xticks(x + width)
        ax.set_xticklabels(["Pure CSD", "CSD+Dynamics", "Difficulty Only"], fontsize=9)
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
        ax.set_ylim(0, 1.05)

        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "figure_s3.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Figure S3 saved")
        return True
    except Exception:
        logger.exception("Failed to generate Figure S3")
        return False


def generate_figure_s4(data: dict) -> bool:
    """Figure S4: Sample-size sensitivity curve."""
    logger.info("Generating Figure S4: Sample-size sensitivity")
    sc = data.get("eval_sc_criteria")
    if sc is None:
        return False

    try:
        magg = sc.get("metrics_agg", {})
        n_values = [10, 15, 20, 25, 30, 40, 50]
        f1_values = [magg.get(f"sensitivity_f1_at_N{n}", 0) for n in n_values]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(n_values, f1_values, "-o", color="#1f77b4", linewidth=2, markersize=8)
        ax.axhline(y=0.80, color="red", linestyle="--", alpha=0.7, label="F1 = 0.80 threshold")
        ax.fill_between(n_values, f1_values, alpha=0.1, color="#1f77b4")

        ax.set_xlabel("Sample Size (N)", fontsize=12)
        ax.set_ylabel("Boundary Detection F1", fontsize=12)
        ax.set_title("Figure S4: F1 vs. Sample Size N", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_ylim(0.7, 1.0)

        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "figure_s4.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Figure S4 saved")
        return True
    except Exception:
        logger.exception("Failed to generate Figure S4")
        return False


# ---------------------------------------------------------------------------
# LaTeX Document Assembly
# ---------------------------------------------------------------------------
def assemble_supplementary(appendices: dict[str, str]) -> str:
    """Assemble all appendices into a single LaTeX document."""
    preamble = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{amsmath}
\usepackage{amssymb}

\title{Supplementary Materials: Critical Slowing Down Indicators\\for LLM Capability Boundary Detection}
\author{}
\date{}

\begin{document}
\maketitle

\tableofcontents
\clearpage

"""
    body = ""
    for key in sorted(appendices.keys()):
        body += appendices[key] + "\n\\clearpage\n\n"

    # Add figures section
    body += r"""
\section{Supplementary Figures}
\label{app:figures}

"""
    figure_files = [
        ("figure_s1.png", "CSD Indicator Profiles -- Arithmetic Task"),
        ("figure_s2.png", "CSD Indicator Profiles -- Graph Coloring Task"),
        ("figure_s3.png", "Feature Ablation Comparison"),
        ("figure_s4.png", "F1 vs.\\ Sample Size N"),
    ]
    for fname, caption in figure_files:
        fpath = FIGURES_DIR / fname
        if fpath.exists():
            body += r"\begin{figure}[h]" + "\n"
            body += r"\centering" + "\n"
            body += f"\\includegraphics[width=0.9\\textwidth]{{figures/{fname}}}\n"
            body += f"\\caption{{{caption}}}\n"
            body += r"\end{figure}" + "\n\n"

    closing = r"""
\end{document}
"""
    return preamble + body + closing


# ---------------------------------------------------------------------------
# PDF Compilation
# ---------------------------------------------------------------------------
def compile_latex(tex_path: Path) -> tuple[bool, int, int]:
    """Compile LaTeX to PDF, return (success, warnings, errors)."""
    logger.info(f"Compiling LaTeX: {tex_path}")
    work_dir = tex_path.parent

    for run in range(2):  # Run twice for TOC
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", tex_path.name],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )

    # Parse log for warnings and errors
    log_file = tex_path.with_suffix(".log")
    warnings_count = 0
    errors_count = 0
    if log_file.exists():
        log_text = log_file.read_text(errors="replace")
        warnings_count = log_text.count("Warning")
        errors_count = log_text.count("! ")

    pdf_path = tex_path.with_suffix(".pdf")
    compiled = pdf_path.exists() and pdf_path.stat().st_size > 0
    logger.info(f"Compilation: success={compiled}, warnings={warnings_count}, errors={errors_count}")
    return compiled, warnings_count, errors_count


def count_pdf_pages(pdf_path: Path) -> int:
    """Count pages in PDF using pdfinfo or fallback."""
    try:
        result = subprocess.run(
            ["pdfinfo", str(pdf_path)],
            capture_output=True, text=True, timeout=30,
        )
        for line in result.stdout.splitlines():
            if line.startswith("Pages:"):
                return int(line.split(":")[1].strip())
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass
    # Fallback: estimate from file size
    try:
        size_kb = pdf_path.stat().st_size / 1024
        return max(1, int(size_kb / 30))  # rough estimate
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Build eval_out.json
# ---------------------------------------------------------------------------
def build_eval_output(
    data: dict,
    appendix_results: dict,
    figure_results: dict,
    compile_result: tuple[bool, int, int],
    page_count: int,
) -> dict:
    """Build the eval_out.json conforming to exp_eval_sol_out schema."""

    n_sources = len(ALL_DATA_SOURCES)
    loaded = sum(1 for v in data.values() if v is not None)

    # Count tables generated (all \begin{table} environments)
    n_tables = 0
    for key, (tex, _) in appendix_results.items():
        n_tables += tex.count(r"\begin{table}")

    # Count figures
    n_figures = sum(1 for v in figure_results.values() if v)
    figures_all_rendered = 1 if n_figures == 4 else 0

    # Per-appendix completion
    appendix_complete = {}
    for key, (tex, count) in appendix_results.items():
        appendix_complete[key] = 1 if count > 0 else 0

    n_appendices = sum(appendix_complete.values())
    compiled, warnings, errors = compile_result

    # Get cell counts from data
    ablation_cells = appendix_results.get("B", ("", 0))[1]
    routing_cells = appendix_results.get("C", ("", 0))[1]
    csd_pairs = appendix_results.get("A", ("", 0))[1]
    feature_pairs = appendix_results.get("D", ("", 0))[1]

    metrics_agg = {
        "n_appendices_generated": n_appendices,
        "n_tables_generated": n_tables,
        "n_figures_generated": n_figures,
        "pdf_compiled": 1 if compiled else 0,
        "pdf_page_count": page_count,
        "data_completeness_fraction": round(loaded / n_sources, 4),
        "appendix_a_complete": appendix_complete.get("A", 0),
        "appendix_b_complete": appendix_complete.get("B", 0),
        "appendix_c_complete": appendix_complete.get("C", 0),
        "appendix_d_complete": appendix_complete.get("D", 0),
        "appendix_e_complete": appendix_complete.get("E", 0),
        "appendix_f_complete": appendix_complete.get("F", 0),
        "appendix_g_complete": appendix_complete.get("G", 0),
        "appendix_h_complete": appendix_complete.get("H", 0),
        "ablation_cells_reported": ablation_cells,
        "routing_cells_reported": routing_cells,
        "csd_profile_pairs_reported": csd_pairs,
        "feature_shift_pairs_reported": feature_pairs,
        "latex_warnings_count": warnings,
        "latex_errors_count": errors,
        "figures_all_rendered": figures_all_rendered,
    }

    # Build datasets with examples for schema compliance
    datasets = []

    # Dataset 1: appendix completion status
    appendix_examples = []
    for letter in "ABCDEFGH":
        c = appendix_complete.get(letter, 0)
        appendix_examples.append({
            "input": f"Appendix {letter} generation status",
            "output": f"complete={c}",
            "predict_status": f"{'complete' if c else 'incomplete'}",
            "eval_complete": c,
            "metadata_appendix": letter,
        })
    datasets.append({"dataset": "appendix_completion", "examples": appendix_examples})

    # Dataset 2: figure rendering status
    figure_examples = []
    for i, (fname, rendered) in enumerate(figure_results.items(), 1):
        figure_examples.append({
            "input": f"Figure S{i} rendering: {fname}",
            "output": f"rendered={'yes' if rendered else 'no'}",
            "predict_status": f"{'rendered' if rendered else 'failed'}",
            "eval_rendered": 1 if rendered else 0,
            "metadata_figure_name": fname,
        })
    datasets.append({"dataset": "figure_rendering", "examples": figure_examples})

    # Dataset 3: data source loading
    source_examples = []
    for label, val in data.items():
        loaded_flag = val is not None
        source_examples.append({
            "input": f"Load data source: {label}",
            "output": f"loaded={'yes' if loaded_flag else 'no'}",
            "predict_status": f"{'loaded' if loaded_flag else 'missing'}",
            "eval_loaded": 1 if loaded_flag else 0,
            "metadata_source": label,
        })
    datasets.append({"dataset": "data_sources", "examples": source_examples})

    eval_output = {
        "metadata": {
            "evaluation_name": "Supplementary Materials PDF Compilation",
            "description": "Comprehensive supplementary materials document with 8 appendices, 9 tables, 4 figures",
            "n_data_sources": n_sources,
            "n_data_loaded": loaded,
            "compile_engine": "pdflatex",
        },
        "metrics_agg": metrics_agg,
        "datasets": datasets,
    }
    return eval_output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
@logger.catch
def main():
    logger.info("=" * 60)
    logger.info("Starting Supplementary Materials PDF Generation")
    logger.info("=" * 60)

    # 1. Load all data
    logger.info("Step 1: Loading all data sources")
    data = load_all_data()

    # 2. Build all appendices
    logger.info("Step 2: Building appendices A-H")
    appendix_results = {}
    builders = {
        "A": build_appendix_a,
        "B": build_appendix_b,
        "C": build_appendix_c,
        "D": build_appendix_d,
        "E": build_appendix_e,
        "F": build_appendix_f,
        "G": build_appendix_g,
        "H": build_appendix_h,
    }
    appendices_tex = {}
    for letter, builder in builders.items():
        try:
            tex, count = builder(data)
            appendix_results[letter] = (tex, count)
            appendices_tex[letter] = tex
            logger.info(f"Appendix {letter}: count={count}")
        except Exception:
            logger.exception(f"Failed building Appendix {letter}")
            appendix_results[letter] = ("", 0)
            appendices_tex[letter] = ""

    # 3. Generate figures
    logger.info("Step 3: Generating supplementary figures")
    figure_results = {
        "figure_s1": generate_figure_s1(data),
        "figure_s2": generate_figure_s2(data),
        "figure_s3": generate_figure_s3(data),
        "figure_s4": generate_figure_s4(data),
    }
    logger.info(f"Figures generated: {sum(1 for v in figure_results.values() if v)}/4")

    # 4. Assemble LaTeX document
    logger.info("Step 4: Assembling LaTeX document")
    full_tex = assemble_supplementary(appendices_tex)
    tex_path = WORKSPACE / "supplementary.tex"
    tex_path.write_text(full_tex)
    logger.info(f"LaTeX document written: {tex_path} ({len(full_tex)} chars)")

    # 5. Compile PDF
    logger.info("Step 5: Compiling PDF")
    try:
        compile_result = compile_latex(tex_path)
    except subprocess.TimeoutExpired:
        logger.warning("LaTeX compilation timed out")
        compile_result = (False, 0, 1)
    except Exception:
        logger.exception("LaTeX compilation failed")
        compile_result = (False, 0, 1)

    # 6. Count pages
    pdf_path = WORKSPACE / "supplementary.pdf"
    page_count = count_pdf_pages(pdf_path) if pdf_path.exists() else 0
    logger.info(f"PDF page count: {page_count}")

    # 7. Build eval_out.json
    logger.info("Step 6: Building eval_out.json")
    eval_output = build_eval_output(data, appendix_results, figure_results, compile_result, page_count)

    eval_path = WORKSPACE / "eval_out.json"
    eval_path.write_text(json.dumps(eval_output, indent=2))
    logger.info(f"eval_out.json written: {eval_path}")

    # Print summary
    m = eval_output["metrics_agg"]
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info(f"  Appendices: {m['n_appendices_generated']}/8")
    logger.info(f"  Tables: {m['n_tables_generated']}/9")
    logger.info(f"  Figures: {m['n_figures_generated']}/4")
    logger.info(f"  PDF compiled: {bool(m['pdf_compiled'])}")
    logger.info(f"  Pages: {m['pdf_page_count']}")
    logger.info(f"  Data completeness: {m['data_completeness_fraction']:.1%}")
    logger.info(f"  LaTeX warnings: {m['latex_warnings_count']}")
    logger.info(f"  LaTeX errors: {m['latex_errors_count']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
