#!/usr/bin/env python3
"""Cross-Task CSD Leading Indicator Evaluation.

Comprehensive statistical evaluation of CSD indicators as leading indicators
of LLM reasoning collapse across 4 task families (arithmetic, graph coloring,
syllogistic, multi-hop).
"""

import json
import math
import os
import resource
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import psutil
from loguru import logger
from scipy import optimize, stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------------------
# Hardware detection (cgroup-aware)
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
TOTAL_RAM_GB = _container_ram_gb() or psutil.virtual_memory().total / 1e9
RAM_BUDGET = int(min(TOTAL_RAM_GB * 0.5, 14) * 1e9)  # 50% of container RAM, max 14 GB
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, budget {RAM_BUDGET/1e9:.1f} GB")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).parent
EXP1_PATH = Path("/ai-inventor/aii_pipeline/runs/neurips-open/3_invention_loop/iter_2/gen_art/exp_id1_it2__opus/full_method_out.json")
EXP2_PATH = Path("/ai-inventor/aii_pipeline/runs/neurips-open/3_invention_loop/iter_2/gen_art/exp_id2_it2__opus/full_method_out.json")
EXP3_PATH = Path("/ai-inventor/aii_pipeline/runs/neurips-open/3_invention_loop/iter_2/gen_art/exp_id3_it2__opus/full_method_out.json")
EXP4_PATH = Path("/ai-inventor/aii_pipeline/runs/neurips-open/3_invention_loop/iter_2/gen_art/exp_id4_it2__opus/full_method_out.json")

BOOTSTRAP_B = 10000
RNG = np.random.default_rng(42)

# Short model names for output
def short_model(m: str) -> str:
    mapping = {
        "meta-llama/llama-3.1-8b-instruct": "llama",
        "google/gemini-2.0-flash-001": "gemini-flash",
        "openai/gpt-4o-mini": "gpt-4o-mini",
        "mistralai/ministral-3b-2512": "ministral-3b",
        "mistralai/ministral-8b-2512": "ministral-8b",
        "deepseek/deepseek-v3.2": "deepseek-v3.2",
        "google/gemini-2.0-flash-lite-001": "gemini-flash-lite",
    }
    return mapping.get(m, m.split("/")[-1])


# ===================================================================
# STEP 0: Data Loading & Normalization
# ===================================================================

def load_exp1(path: Path) -> list[dict]:
    """Load arithmetic experiment: 3 datasets, one per model, each example = one difficulty level."""
    logger.info(f"Loading Exp1 (arithmetic) from {path}")
    data = json.loads(path.read_text())
    rows = []
    for ds in data["datasets"]:
        for ex in ds["examples"]:
            rows.append({
                "task_family": "arithmetic",
                "model": ex["metadata_model"],
                "difficulty_level": int(ex["metadata_difficulty_level"]),
                "accuracy": float(ex["predict_accuracy"]),
                "embedding_variance": float(ex["predict_csd_variance"]),
                "dip_statistic": float(ex["predict_dip_statistic"]),
                "dip_pvalue": float(ex["predict_dip_pvalue"]),
                "silhouette_k2": float(ex["predict_silhouette_k2"]),
                "bimodality_coefficient": float(ex["predict_bimodality_coefficient"]),
                "ashman_d": float("nan"),
                "disagreement_rate": float(ex["predict_disagreement_rate"]),
                "chain_autocorrelation": float(ex.get("predict_step_correctness_autocorr", "nan")),
                "d_star": ex["metadata_d_star"],
            })
    logger.info(f"  Exp1: {len(rows)} rows loaded")
    return rows


def load_exp2(path: Path) -> list[dict]:
    """Load syllogistic experiment: 1 dataset, per-response. Aggregate by (model, difficulty)."""
    logger.info(f"Loading Exp2 (syllogistic) from {path}")
    data = json.loads(path.read_text())
    # Group by (model, difficulty) and take first row's CSD values (they're identical within level)
    groups: dict[tuple, dict] = {}
    for ds in data["datasets"]:
        for ex in ds["examples"]:
            key = (ex["metadata_model"], ex["metadata_difficulty"])
            if key not in groups:
                d_star = ex.get("metadata_analysis_d_star", None)
                groups[key] = {
                    "task_family": "syllogistic",
                    "model": ex["metadata_model"],
                    "difficulty_level": int(ex["metadata_difficulty"]),
                    "accuracy": float(ex["metadata_csd_accuracy"]),
                    "embedding_variance": float(ex["metadata_csd_embedding_variance"]),
                    "dip_statistic": float(ex["metadata_csd_dip_statistic"]),
                    "dip_pvalue": float(ex["metadata_csd_dip_pvalue"]),
                    "silhouette_k2": float(ex["metadata_csd_silhouette_k2"]),
                    "bimodality_coefficient": float(ex["metadata_csd_bimodality_coefficient"]),
                    "ashman_d": float("nan"),
                    "disagreement_rate": float(ex["metadata_csd_disagreement_rate"]),
                    "chain_autocorrelation": float(ex.get("metadata_csd_avg_chain_autocorrelation", "nan")),
                    "d_star": d_star,
                }
    rows = list(groups.values())
    logger.info(f"  Exp2: {len(rows)} aggregated rows")
    return rows


def load_exp3(path: Path) -> list[dict]:
    """Load graph coloring experiment: 3 datasets, per-response. Aggregate by (model, difficulty_level)."""
    logger.info(f"Loading Exp3 (graph_coloring) from {path}")
    data = json.loads(path.read_text())

    # Get d_star from top-level metadata
    model_dstar = {}
    if "metadata" in data and "analysis" in data["metadata"]:
        for m_info in data["metadata"]["analysis"].get("models", []):
            model_dstar[m_info["model"]] = m_info.get("d_star")

    groups: dict[tuple, dict] = {}
    for ds in data["datasets"]:
        for ex in ds["examples"]:
            key = (ex["metadata_model"], ex["metadata_difficulty_level"])
            if key not in groups:
                model = ex["metadata_model"]
                groups[key] = {
                    "task_family": "graph_coloring",
                    "model": model,
                    "difficulty_level": int(ex["metadata_difficulty_level"]),
                    "accuracy": float(ex["metadata_csd_accuracy"]),
                    "embedding_variance": float(ex["metadata_csd_embedding_variance"]),
                    "dip_statistic": float(ex["metadata_csd_dip_statistic"]),
                    "dip_pvalue": float(ex["metadata_csd_dip_pvalue"]),
                    "silhouette_k2": float(ex.get("metadata_csd_silhouette_score", 0)),
                    "bimodality_coefficient": float(ex["metadata_csd_bimodality_coefficient"]),
                    "ashman_d": float(ex.get("metadata_csd_ashman_d", "nan")),
                    "disagreement_rate": float(ex["metadata_csd_disagreement_rate"]),
                    "chain_autocorrelation": float("nan"),
                    "d_star": model_dstar.get(model),
                }
    rows = list(groups.values())
    logger.info(f"  Exp3: {len(rows)} aggregated rows")
    return rows


def load_exp4(path: Path) -> list[dict]:
    """Load multi-hop experiment: CSD indicators from top-level metadata.per_model_summary."""
    logger.info(f"Loading Exp4 (multi_hop) from {path}")
    data = json.loads(path.read_text())
    pms = data["metadata"]["per_model_summary"]
    rows = []
    for model, summary in pms.items():
        d_star_val = summary.get("variance_scaling", {}).get("d_star")
        for level_str, indicators in summary["per_level_indicators"].items():
            level = int(level_str)
            rows.append({
                "task_family": "multi_hop",
                "model": model,
                "difficulty_level": level,
                "accuracy": float(indicators["accuracy"]),
                "embedding_variance": float(indicators.get("embedding_variance_trace", "nan")),
                "dip_statistic": float(indicators.get("hartigan_dip_stat", "nan")),
                "dip_pvalue": float(indicators.get("hartigan_dip_pval", "nan")),
                "silhouette_k2": float(indicators.get("silhouette_score_k2", "nan")),
                "bimodality_coefficient": float(indicators.get("bimodality_coefficient", "nan")),
                "ashman_d": float(indicators.get("ashman_d", "nan")),
                "disagreement_rate": float(indicators.get("self_consistency_disagreement", "nan")),
                "chain_autocorrelation": float("nan"),
                "d_star": d_star_val,
            })
    logger.info(f"  Exp4: {len(rows)} rows loaded")
    return rows


def load_all_data() -> list[dict]:
    """Load and normalize all experiments into unified format."""
    all_rows = []
    all_rows.extend(load_exp1(EXP1_PATH))
    all_rows.extend(load_exp2(EXP2_PATH))
    all_rows.extend(load_exp3(EXP3_PATH))
    all_rows.extend(load_exp4(EXP4_PATH))
    logger.info(f"Total unified rows: {len(all_rows)}")
    return all_rows


def classify_pairs(all_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Classify model-task pairs into positive and negative."""
    # Group by (task, model) and get d_star
    pair_data: dict[tuple[str, str], list[dict]] = {}
    for r in all_rows:
        key = (r["task_family"], r["model"])
        pair_data.setdefault(key, []).append(r)

    positive_pairs = []
    negative_pairs = []

    for (task, model), rows in pair_data.items():
        d_star = rows[0]["d_star"]
        smodel = short_model(model)
        min_level = min(r["difficulty_level"] for r in rows)
        sorted_rows = sorted(rows, key=lambda x: x["difficulty_level"])

        if task == "arithmetic" and smodel == "gpt-4o-mini":
            # d*=2 is lowest level, no pre-boundary region — exclude from positive
            logger.info(f"  EXCLUDED: {task} x {smodel} (d*={d_star}, no pre-boundary region)")
            continue
        elif task == "syllogistic":
            # All d*=None → negative control
            negative_pairs.append({"task": task, "model": model, "smodel": smodel,
                                   "d_star": d_star, "rows": sorted_rows, "neg_type": "no_boundary"})
        elif task == "multi_hop":
            # d*=1, inverted difficulty → negative control
            negative_pairs.append({"task": task, "model": model, "smodel": smodel,
                                   "d_star": d_star, "rows": sorted_rows, "neg_type": "inverted"})
        elif d_star is not None and d_star > min_level:
            # Valid d_star with pre-boundary data → positive
            positive_pairs.append({"task": task, "model": model, "smodel": smodel,
                                   "d_star": d_star, "rows": sorted_rows})
        else:
            logger.warning(f"  Unclassified: {task} x {smodel}, d*={d_star}")

    logger.info(f"Positive pairs: {len(positive_pairs)}, Negative pairs: {len(negative_pairs)}")
    for p in positive_pairs:
        logger.info(f"  + {p['task']} x {p['smodel']} (d*={p['d_star']})")
    for p in negative_pairs:
        logger.info(f"  - {p['task']} x {p['smodel']} (d*={p['d_star']}, {p['neg_type']})")
    return positive_pairs, negative_pairs


# ===================================================================
# METRIC BLOCK 1: Leading Indicator Tests
# ===================================================================

def find_first_significant_level(rows: list[dict], d_star: int, indicator: str) -> dict:
    """Find the first difficulty level where an indicator crosses significance threshold."""
    pre_boundary = [r for r in rows if r["difficulty_level"] <= d_star]
    pre_boundary.sort(key=lambda x: x["difficulty_level"])

    thresholds = {
        "dip": lambda r: r["dip_pvalue"] < 0.05,
        "silhouette": lambda r: r["silhouette_k2"] > 0.3,
        "bimodality_coefficient": lambda r: r["bimodality_coefficient"] > 0.555,
        "ashman_d": lambda r: not np.isnan(r["ashman_d"]) and r["ashman_d"] > 2.0,
    }

    if indicator == "embedding_variance":
        # Use Kendall tau trend test across pre-d* levels
        levels = [r["difficulty_level"] for r in pre_boundary]
        variances = [r["embedding_variance"] for r in pre_boundary]
        if len(levels) >= 3:
            tau, p = stats.kendalltau(levels, variances)
            if p < 0.05:
                # Find first level where variance starts rising (above median of first half)
                half = max(1, len(pre_boundary) // 2)
                baseline_var = np.median([r["embedding_variance"] for r in pre_boundary[:half]])
                for r in pre_boundary:
                    if r["embedding_variance"] > baseline_var * 1.1:
                        return {
                            "d_lead": r["difficulty_level"],
                            "lead_time": d_star - r["difficulty_level"],
                            "accuracy_at_signal": r["accuracy"],
                            "significant": True,
                            "tau": tau, "p": p,
                        }
        return {"d_lead": None, "lead_time": None, "accuracy_at_signal": None, "significant": False}

    check_fn = thresholds.get(indicator)
    if check_fn is None:
        return {"d_lead": None, "lead_time": None, "accuracy_at_signal": None, "significant": False}

    for r in pre_boundary:
        try:
            if check_fn(r):
                return {
                    "d_lead": r["difficulty_level"],
                    "lead_time": d_star - r["difficulty_level"],
                    "accuracy_at_signal": r["accuracy"],
                    "significant": True,
                }
        except (ValueError, TypeError):
            continue

    return {"d_lead": None, "lead_time": None, "accuracy_at_signal": None, "significant": False}


def _bootstrap_one(args):
    """Single bootstrap resample for lead time estimation."""
    rows_pre, d_star, indicator, seed = args
    rng = np.random.default_rng(seed)
    n = len(rows_pre)
    indices = rng.choice(n, size=n, replace=True)
    resampled = [rows_pre[i] for i in indices]
    resampled.sort(key=lambda x: x["difficulty_level"])
    result = find_first_significant_level(resampled, d_star, indicator)
    return result.get("lead_time")


def bootstrap_lead_time_ci(rows: list[dict], d_star: int, indicator: str, B: int = BOOTSTRAP_B) -> dict:
    """Bootstrap CI on lead_time for a given indicator."""
    pre_boundary = [r for r in rows if r["difficulty_level"] <= d_star]
    if len(pre_boundary) < 4:
        return {"ci_lower": None, "ci_upper": None, "median": None}

    seeds = RNG.integers(0, 2**31, size=B)
    args_list = [(pre_boundary, d_star, indicator, int(s)) for s in seeds]

    lead_times = []
    # Use ProcessPoolExecutor for CPU parallelism
    with ProcessPoolExecutor(max_workers=max(1, NUM_CPUS - 1)) as executor:
        futures = {executor.submit(_bootstrap_one, args): i for i, args in enumerate(args_list)}
        for fut in as_completed(futures):
            try:
                lt = fut.result()
                if lt is not None:
                    lead_times.append(lt)
            except Exception:
                pass

    if len(lead_times) < B * 0.1:  # Too few valid resamples
        return {"ci_lower": None, "ci_upper": None, "median": None}

    lead_arr = np.array(lead_times)
    return {
        "ci_lower": float(np.percentile(lead_arr, 2.5)),
        "ci_upper": float(np.percentile(lead_arr, 97.5)),
        "median": float(np.median(lead_arr)),
    }


def compute_block1(pair: dict) -> dict:
    """Leading Indicator Tests for a single positive pair."""
    rows = pair["rows"]
    d_star = pair["d_star"]
    results = {}

    indicators = ["dip", "silhouette", "bimodality_coefficient", "ashman_d"]
    lead_times = {}
    earliest_signal_level = None
    earliest_accuracy = None

    for ind in indicators:
        res = find_first_significant_level(rows, d_star, ind)
        key_map = {"dip": "dip", "silhouette": "silhouette", "bimodality_coefficient": "bc", "ashman_d": "ashman_d"}
        short = key_map[ind]
        results[f"eval_lead_time_{short}"] = res["lead_time"] if res["lead_time"] is not None else float("nan")
        lead_times[ind] = res

        if res["significant"] and res["d_lead"] is not None:
            if earliest_signal_level is None or res["d_lead"] < earliest_signal_level:
                earliest_signal_level = res["d_lead"]
                earliest_accuracy = res["accuracy_at_signal"]

    # Embedding variance trend
    var_res = find_first_significant_level(rows, d_star, "embedding_variance")
    results["eval_variance_trend_significant"] = 1.0 if var_res.get("significant", False) else 0.0

    results["eval_accuracy_at_first_signal"] = earliest_accuracy if earliest_accuracy is not None else float("nan")
    is_leading = earliest_accuracy is not None and earliest_accuracy > 0.8
    results["eval_is_leading"] = 1.0 if is_leading else 0.0

    # Bootstrap CI for best indicator (the one with largest lead_time)
    best_ind = None
    best_lt = -1
    for ind in indicators:
        lt = lead_times[ind].get("lead_time")
        if lt is not None and lt > best_lt:
            best_lt = lt
            best_ind = ind

    if best_ind is not None and len(rows) >= 10:
        ci = bootstrap_lead_time_ci(rows, d_star, best_ind)
        results["eval_lead_time_ci_lower"] = ci["ci_lower"] if ci["ci_lower"] is not None else float("nan")
        results["eval_lead_time_ci_upper"] = ci["ci_upper"] if ci["ci_upper"] is not None else float("nan")
    else:
        results["eval_lead_time_ci_lower"] = float("nan")
        results["eval_lead_time_ci_upper"] = float("nan")

    return results


# ===================================================================
# METRIC BLOCK 2: Alternative Variance Models
# ===================================================================

def fit_variance_models(rows: list[dict], d_star: int) -> dict:
    """Fit 4 variance models and compare via AIC/BIC."""
    pre_boundary = sorted([r for r in rows if r["difficulty_level"] <= d_star],
                          key=lambda x: x["difficulty_level"])
    if len(pre_boundary) < 4:
        return {k: float("nan") for k in [
            "eval_best_variance_model", "eval_bifurcation_alpha",
            "eval_bifurcation_alpha_in_range", "eval_bifurcation_r2",
            "eval_aic_bifurcation", "eval_aic_gaussian", "eval_aic_logistic", "eval_aic_null",
            "eval_bic_bifurcation", "eval_bic_gaussian", "eval_bic_logistic", "eval_bic_null",
            "eval_delta_aic_vs_null",
        ]}

    d = np.array([r["difficulty_level"] for r in pre_boundary], dtype=float)
    v = np.array([r["embedding_variance"] for r in pre_boundary], dtype=float)
    n = len(d)

    def compute_aic_bic(rss, k, n):
        if rss <= 0 or n <= k:
            return float("inf"), float("inf")
        aic = n * np.log(rss / n) + 2 * k
        bic = n * np.log(rss / n) + k * np.log(n)
        return aic, bic

    results = {}

    # (a) Fold bifurcation power law: var(d) = A * (d_star - d)^alpha + C
    try:
        def power_law(x, A, alpha, C):
            diff = d_star - x
            diff = np.maximum(diff, 0.01)  # avoid zero/negative
            return A * np.power(diff, alpha) + C

        popt_a, _ = optimize.curve_fit(power_law, d, v,
                                        p0=[0.1, -0.5, np.mean(v)],
                                        bounds=([-10, -5, -10], [10, 5, 10]),
                                        maxfev=10000)
        pred_a = power_law(d, *popt_a)
        rss_a = np.sum((v - pred_a) ** 2)
        ss_tot = np.sum((v - np.mean(v)) ** 2)
        r2_a = 1 - rss_a / ss_tot if ss_tot > 0 else 0
        aic_a, bic_a = compute_aic_bic(rss_a, 3, n)
        alpha_val = popt_a[1]
    except Exception:
        rss_a = float("inf")
        aic_a = bic_a = float("inf")
        r2_a = 0.0
        alpha_val = float("nan")

    results["eval_bifurcation_alpha"] = float(alpha_val)
    results["eval_bifurcation_alpha_in_range"] = 1.0 if -0.7 <= alpha_val <= -0.3 else 0.0
    results["eval_bifurcation_r2"] = float(r2_a)
    results["eval_aic_bifurcation"] = float(aic_a)
    results["eval_bic_bifurcation"] = float(bic_a)

    # (b) Gaussian bump
    try:
        def gaussian_bump(x, A, mu, sigma, C):
            return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2 + 1e-10)) + C

        popt_b, _ = optimize.curve_fit(gaussian_bump, d, v,
                                        p0=[np.std(v), d_star, (d_star - d[0]) / 2, np.min(v)],
                                        bounds=([-10, d[0] - 5, 0.1, -10], [10, d_star + 5, d_star * 2, 10]),
                                        maxfev=10000)
        pred_b = gaussian_bump(d, *popt_b)
        rss_b = np.sum((v - pred_b) ** 2)
        aic_b, bic_b = compute_aic_bic(rss_b, 4, n)
    except Exception:
        aic_b = bic_b = float("inf")

    results["eval_aic_gaussian"] = float(aic_b)
    results["eval_bic_gaussian"] = float(bic_b)

    # (c) Logistic S-curve
    try:
        def logistic(x, L, k, d0, C):
            return L / (1 + np.exp(-k * (x - d0))) + C

        popt_c, _ = optimize.curve_fit(logistic, d, v,
                                        p0=[np.ptp(v), 0.5, np.median(d), np.min(v)],
                                        bounds=([-10, -10, d[0] - 5, -10], [10, 10, d_star + 5, 10]),
                                        maxfev=10000)
        pred_c = logistic(d, *popt_c)
        rss_c = np.sum((v - pred_c) ** 2)
        aic_c, bic_c = compute_aic_bic(rss_c, 4, n)
    except Exception:
        aic_c = bic_c = float("inf")

    results["eval_aic_logistic"] = float(aic_c)
    results["eval_bic_logistic"] = float(bic_c)

    # (d) Flat null: var(d) = C
    mean_v = np.mean(v)
    rss_null = np.sum((v - mean_v) ** 2)
    aic_null, bic_null = compute_aic_bic(rss_null, 1, n)
    results["eval_aic_null"] = float(aic_null)
    results["eval_bic_null"] = float(bic_null)

    # Best model by AIC
    models_aic = {"bifurcation": aic_a, "gaussian": aic_b, "logistic": aic_c, "null": aic_null}
    finite_models = {k: v for k, v in models_aic.items() if np.isfinite(v)}
    if finite_models:
        best = min(finite_models, key=finite_models.get)
    else:
        best = "null"

    # Encode best model as a number for schema compliance
    model_encoding = {"bifurcation": 1.0, "gaussian": 2.0, "logistic": 3.0, "null": 4.0}
    results["eval_best_variance_model"] = model_encoding.get(best, 4.0)
    results["eval_delta_aic_vs_null"] = float(aic_a - aic_null) if np.isfinite(aic_a) else float("nan")

    return results


# ===================================================================
# METRIC BLOCK 3: Cross-Task Consistency
# ===================================================================

def compute_block3(positive_pairs: list[dict], block1_results: list[dict]) -> dict:
    """Cross-task consistency metrics."""
    n_pairs = len(positive_pairs)
    results = {}

    # 3a. Leading indicator fraction
    n_leading = sum(1 for b in block1_results if b.get("eval_is_leading", 0) > 0.5)
    results["eval_fraction_pairs_with_leading_indicator"] = float(n_leading / n_pairs) if n_pairs > 0 else 0.0

    # Per-indicator fractions
    for ind_key in ["dip", "silhouette", "bc"]:
        n_sig = sum(1 for b in block1_results
                    if not np.isnan(b.get(f"eval_lead_time_{ind_key}", float("nan")))
                    and b.get(f"eval_lead_time_{ind_key}", float("nan")) > 0)
        results[f"eval_fraction_leading_{ind_key}"] = float(n_sig / n_pairs) if n_pairs > 0 else 0.0

    # 3b. Fisher's combined probability test
    for ind_key, ind_name in [("dip", "dip"), ("silhouette", "silhouette"), ("bc", "bimodality_coefficient")]:
        pvalues = []
        for pair, b1 in zip(positive_pairs, block1_results):
            rows = pair["rows"]
            d_star = pair["d_star"]
            # Find p-value at first level where accuracy drops below 0.8
            sorted_rows = sorted(rows, key=lambda x: x["difficulty_level"])
            for r in sorted_rows:
                if r["accuracy"] < 0.8:
                    if ind_name == "dip":
                        p = r["dip_pvalue"]
                    elif ind_name == "silhouette":
                        # Convert silhouette to a pseudo-p-value
                        p = max(0.001, 1.0 - r["silhouette_k2"]) if r["silhouette_k2"] > 0 else 1.0
                    elif ind_name == "bimodality_coefficient":
                        p = max(0.001, 1.0 - r["bimodality_coefficient"]) if r["bimodality_coefficient"] > 0 else 1.0
                    else:
                        p = 1.0
                    if not np.isnan(p) and p > 0:
                        pvalues.append(p)
                    break

        if len(pvalues) >= 2:
            chi2 = -2 * sum(np.log(p) for p in pvalues)
            df = 2 * len(pvalues)
            fisher_p = float(1 - stats.chi2.cdf(chi2, df))
            results[f"eval_fisher_chi2_{ind_key}"] = float(chi2)
            results[f"eval_fisher_pvalue_{ind_key}"] = fisher_p
        else:
            results[f"eval_fisher_chi2_{ind_key}"] = float("nan")
            results[f"eval_fisher_pvalue_{ind_key}"] = float("nan")

    # 3c. Variance trend consistency
    taus = []
    for pair in positive_pairs:
        rows = pair["rows"]
        d_star = pair["d_star"]
        pre = [r for r in rows if r["difficulty_level"] <= d_star]
        if len(pre) >= 3:
            levels = [r["difficulty_level"] for r in sorted(pre, key=lambda x: x["difficulty_level"])]
            variances = [r["embedding_variance"] for r in sorted(pre, key=lambda x: x["difficulty_level"])]
            tau, _ = stats.kendalltau(levels, variances)
            if not np.isnan(tau):
                taus.append(tau)

    results["eval_mean_kendall_tau_variance"] = float(np.mean(taus)) if taus else float("nan")
    results["eval_fraction_positive_tau"] = float(sum(1 for t in taus if t > 0) / len(taus)) if taus else float("nan")

    return results


# ===================================================================
# METRIC BLOCK 4: Effect Sizes
# ===================================================================

def compute_cohens_d(pre_vals: np.ndarray, near_vals: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    if len(pre_vals) < 2 or len(near_vals) < 2:
        return float("nan")
    n1, n2 = len(pre_vals), len(near_vals)
    s1, s2 = np.std(pre_vals, ddof=1), np.std(near_vals, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    if pooled_sd < 1e-10:
        return float("nan")
    return float((np.mean(near_vals) - np.mean(pre_vals)) / pooled_sd)


def compute_block4(pair: dict) -> dict:
    """Effect sizes for a single positive pair."""
    rows = pair["rows"]
    d_star = pair["d_star"]

    pre_zone = [r for r in rows if r["difficulty_level"] < d_star - 3]
    near_zone = [r for r in rows if d_star - 3 <= r["difficulty_level"] <= d_star]

    results = {}
    for field, key in [("embedding_variance", "variance"), ("dip_statistic", "dip"),
                       ("silhouette_k2", "silhouette"), ("bimodality_coefficient", "bc"),
                       ("disagreement_rate", "disagreement")]:
        pre_vals = np.array([r[field] for r in pre_zone if not np.isnan(r[field])])
        near_vals = np.array([r[field] for r in near_zone if not np.isnan(r[field])])
        results[f"eval_cohens_d_{key}"] = compute_cohens_d(pre_vals, near_vals)

    return results


# ===================================================================
# METRIC BLOCK 5: Negative Controls
# ===================================================================

def compute_block5(negative_pairs: list[dict]) -> dict:
    """Negative control analysis."""
    results = {}

    # Split by type
    syl_pairs = [p for p in negative_pairs if p["task"] == "syllogistic"]
    mhop_pairs = [p for p in negative_pairs if p["task"] == "multi_hop"]

    # 5a. Syllogistic (no boundary)
    syl_abs_tau_var = []
    syl_abs_tau_dip = []
    syl_sig_count = 0
    for pair in syl_pairs:
        rows = sorted(pair["rows"], key=lambda x: x["difficulty_level"])
        levels = [r["difficulty_level"] for r in rows]
        if len(levels) >= 3:
            # Variance trend
            var_vals = [r["embedding_variance"] for r in rows]
            tau_v, p_v = stats.kendalltau(levels, var_vals)
            if not np.isnan(tau_v):
                syl_abs_tau_var.append(abs(tau_v))
            # Dip trend
            dip_vals = [r["dip_statistic"] for r in rows]
            tau_d, p_d = stats.kendalltau(levels, dip_vals)
            if not np.isnan(tau_d):
                syl_abs_tau_dip.append(abs(tau_d))
            # Significance
            if (not np.isnan(p_v) and p_v < 0.05) or (not np.isnan(p_d) and p_d < 0.05):
                syl_sig_count += 1

    results["eval_neg_syl_mean_abs_tau_variance"] = float(np.mean(syl_abs_tau_var)) if syl_abs_tau_var else float("nan")
    results["eval_neg_syl_mean_abs_tau_dip"] = float(np.mean(syl_abs_tau_dip)) if syl_abs_tau_dip else float("nan")
    results["eval_neg_syl_fraction_significant"] = float(syl_sig_count / len(syl_pairs)) if syl_pairs else float("nan")

    # 5b. Multi-hop (inverted difficulty)
    mhop_abs_tau_var = []
    mhop_sig_count = 0
    mhop_bimodality_always = True
    for pair in mhop_pairs:
        rows = sorted(pair["rows"], key=lambda x: x["difficulty_level"])
        levels = [r["difficulty_level"] for r in rows]
        if len(levels) >= 3:
            var_vals = [r["embedding_variance"] for r in rows]
            tau_v, p_v = stats.kendalltau(levels, var_vals)
            if not np.isnan(tau_v):
                mhop_abs_tau_var.append(abs(tau_v))
            if not np.isnan(p_v) and p_v < 0.05:
                mhop_sig_count += 1
            # Check if bimodality always present
            for r in rows:
                if r["dip_pvalue"] >= 0.05:
                    mhop_bimodality_always = False

    results["eval_neg_mhop_mean_abs_tau_variance"] = float(np.mean(mhop_abs_tau_var)) if mhop_abs_tau_var else float("nan")
    results["eval_neg_mhop_bimodality_always_present"] = 1.0 if mhop_bimodality_always else 0.0
    results["eval_neg_mhop_fraction_significant_trend"] = float(mhop_sig_count / len(mhop_pairs)) if mhop_pairs else float("nan")

    # 5c. Summary
    all_abs_taus = syl_abs_tau_var + syl_abs_tau_dip + mhop_abs_tau_var
    mean_abs_tau = np.mean(all_abs_taus) if all_abs_taus else 1.0
    total_pairs = len(syl_pairs) + len(mhop_pairs)
    total_sig = syl_sig_count + mhop_sig_count
    frac_sig = total_sig / total_pairs if total_pairs > 0 else 1.0
    neg_control_pass = mean_abs_tau < 0.3 and frac_sig < 0.5
    results["eval_negative_control_pass"] = 1.0 if neg_control_pass else 0.0

    return results


# ===================================================================
# METRIC BLOCK 6: Hypothesis Verdict (includes SC3 classifier)
# ===================================================================

def compute_classifier_f1(pair: dict) -> tuple[float, float]:
    """Build logistic regression classifier for near-boundary detection. Returns (csd_f1, baseline_f1)."""
    rows = pair["rows"]
    d_star = pair["d_star"]

    # Features and labels
    X_csd = []
    X_base = []
    y = []
    for r in rows:
        feats_csd = [
            r["dip_statistic"] if not np.isnan(r["dip_statistic"]) else 0,
            r["silhouette_k2"] if not np.isnan(r["silhouette_k2"]) else 0,
            r["bimodality_coefficient"] if not np.isnan(r["bimodality_coefficient"]) else 0,
            r["embedding_variance"] if not np.isnan(r["embedding_variance"]) else 0,
        ]
        feats_base = [r["disagreement_rate"] if not np.isnan(r["disagreement_rate"]) else 0]
        label = 1 if abs(r["difficulty_level"] - d_star) <= 2 else 0
        X_csd.append(feats_csd)
        X_base.append(feats_base)
        y.append(label)

    X_csd = np.array(X_csd)
    X_base = np.array(X_base)
    y = np.array(y)

    if len(np.unique(y)) < 2 or len(y) < 4:
        return float("nan"), float("nan")

    # Leave-one-level-out CV
    loo = LeaveOneOut()
    preds_csd = np.zeros(len(y))
    preds_base = np.zeros(len(y))

    for train_idx, test_idx in loo.split(X_csd):
        # CSD classifier
        try:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_csd[train_idx])
            X_test = scaler.transform(X_csd[test_idx])
            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(X_train, y[train_idx])
            preds_csd[test_idx] = clf.predict(X_test)
        except Exception:
            preds_csd[test_idx] = 0

        # Baseline classifier
        try:
            scaler_b = StandardScaler()
            X_train_b = scaler_b.fit_transform(X_base[train_idx])
            X_test_b = scaler_b.transform(X_base[test_idx])
            clf_b = LogisticRegression(max_iter=1000, random_state=42)
            clf_b.fit(X_train_b, y[train_idx])
            preds_base[test_idx] = clf_b.predict(X_test_b)
        except Exception:
            preds_base[test_idx] = 0

    f1_csd = f1_score(y, preds_csd, zero_division=0)
    f1_base = f1_score(y, preds_base, zero_division=0)
    return float(f1_csd), float(f1_base)


def compute_block6(positive_pairs: list[dict], block1_results: list[dict],
                   block2_results: list[dict]) -> dict:
    """Hypothesis verdict: evaluate 3 success criteria."""
    results = {}

    # SC1: Flickering as leading indicator
    # Required: significant bimodality detectable where accuracy > 0.8, across >= 2 task families and >= 2 models
    leading_tasks = set()
    leading_models = set()
    for pair, b1 in zip(positive_pairs, block1_results):
        if b1.get("eval_is_leading", 0) > 0.5:
            leading_tasks.add(pair["task"])
            leading_models.add(pair["smodel"])

    n_task_families = len(leading_tasks)
    n_models = len(leading_models)
    sc1_met = n_task_families >= 2 and n_models >= 2
    results["eval_sc1_met"] = 1.0 if sc1_met else 0.0
    results["eval_sc1_n_task_families"] = float(n_task_families)
    results["eval_sc1_n_models"] = float(n_models)

    # SC2: Variance scaling exponent
    alphas = []
    n_in_range = 0
    for b2 in block2_results:
        a = b2.get("eval_bifurcation_alpha", float("nan"))
        if not np.isnan(a):
            alphas.append(a)
            if b2.get("eval_bifurcation_alpha_in_range", 0) > 0.5:
                n_in_range += 1

    mean_alpha = float(np.mean(alphas)) if alphas else float("nan")
    # SC2 met if any pair has alpha in range
    sc2_met = n_in_range > 0
    results["eval_sc2_met"] = 1.0 if sc2_met else 0.0
    results["eval_sc2_n_pairs_in_range"] = float(n_in_range)
    results["eval_sc2_mean_alpha"] = mean_alpha

    # SC3: Classifier improvement >= 15%
    csd_f1s = []
    base_f1s = []
    for pair in positive_pairs:
        f1_csd, f1_base = compute_classifier_f1(pair)
        if not np.isnan(f1_csd):
            csd_f1s.append(f1_csd)
            base_f1s.append(f1_base)

    if csd_f1s:
        mean_csd_f1 = float(np.mean(csd_f1s))
        mean_base_f1 = float(np.mean(base_f1s))
        improvement = ((mean_csd_f1 - mean_base_f1) / max(mean_base_f1, 0.01)) * 100
    else:
        mean_csd_f1 = mean_base_f1 = float("nan")
        improvement = float("nan")

    results["eval_sc3_csd_mean_f1"] = mean_csd_f1
    results["eval_sc3_baseline_mean_f1"] = mean_base_f1
    results["eval_sc3_improvement_pct"] = float(improvement) if not np.isnan(improvement) else float("nan")
    sc3_met = not np.isnan(improvement) and improvement >= 15
    results["eval_sc3_met"] = 1.0 if sc3_met else 0.0

    # Overall verdict
    results["eval_hypothesis_confirmed"] = 1.0 if (sc1_met and sc2_met and sc3_met) else 0.0
    results["eval_hypothesis_partially_confirmed"] = 1.0 if sc1_met else 0.0

    # Disconfirmed: accuracy transitions gradual everywhere AND CSD rises only concurrently
    all_concurrent = all(b.get("eval_is_leading", 0) < 0.5 for b in block1_results)
    results["eval_hypothesis_disconfirmed"] = 1.0 if all_concurrent else 0.0

    return results


# ===================================================================
# Build output datasets
# ===================================================================

def build_datasets(positive_pairs: list[dict], negative_pairs: list[dict],
                   block1_results: list[dict], block2_results: list[dict],
                   block4_results: list[dict]) -> list[dict]:
    """Build output datasets: one per model-task pair, each example = one difficulty level."""
    datasets = []

    # Positive pairs
    for i, pair in enumerate(positive_pairs):
        b1 = block1_results[i]
        b2 = block2_results[i]
        b4 = block4_results[i]
        examples = []
        for r in pair["rows"]:
            leading_str = "leading" if b1.get("eval_is_leading", 0) > 0.5 else "not_leading"
            ex = {
                "input": f"CSD evaluation at d={r['difficulty_level']} for {short_model(r['model'])} on {r['task_family']}",
                "output": f"accuracy={r['accuracy']:.4f}",
                "predict_csd_assessment": f"{leading_str}; dip_p={r['dip_pvalue']:.4f}, sil={r['silhouette_k2']:.4f}, bc={r['bimodality_coefficient']:.4f}, var={r['embedding_variance']:.4f}",
                "metadata_task_family": r["task_family"],
                "metadata_model": short_model(r["model"]),
                "metadata_difficulty_level": r["difficulty_level"],
                "metadata_d_star": r["d_star"],
                "metadata_pair_type": "positive",
            }
            # Add all eval metrics from block1
            for k, v in b1.items():
                if k.startswith("eval_"):
                    ex[k] = float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else float("nan")
            # Add block2 metrics
            for k, v in b2.items():
                if k.startswith("eval_"):
                    ex[k] = float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else float("nan")
            # Add block4 metrics
            for k, v in b4.items():
                if k.startswith("eval_"):
                    ex[k] = float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else float("nan")
            # Per-level raw CSD values
            ex["eval_accuracy_value"] = float(r["accuracy"])
            ex["eval_embedding_variance_value"] = float(r["embedding_variance"]) if not np.isnan(r["embedding_variance"]) else float("nan")
            ex["eval_dip_statistic_value"] = float(r["dip_statistic"]) if not np.isnan(r["dip_statistic"]) else float("nan")
            ex["eval_dip_pvalue_value"] = float(r["dip_pvalue"]) if not np.isnan(r["dip_pvalue"]) else float("nan")
            ex["eval_silhouette_value"] = float(r["silhouette_k2"]) if not np.isnan(r["silhouette_k2"]) else float("nan")
            ex["eval_bimodality_coeff_value"] = float(r["bimodality_coefficient"]) if not np.isnan(r["bimodality_coefficient"]) else float("nan")
            ex["eval_disagreement_value"] = float(r["disagreement_rate"]) if not np.isnan(r["disagreement_rate"]) else float("nan")

            examples.append(ex)

        ds_name = f"positive_{pair['task']}_{pair['smodel']}"
        datasets.append({"dataset": ds_name, "examples": examples})

    # Negative pairs
    for pair in negative_pairs:
        examples = []
        for r in pair["rows"]:
            ex = {
                "input": f"CSD evaluation at d={r['difficulty_level']} for {short_model(r['model'])} on {r['task_family']}",
                "output": f"accuracy={r['accuracy']:.4f}",
                "predict_csd_assessment": f"negative_control ({pair['neg_type']}); dip_p={r['dip_pvalue']:.4f}, sil={r['silhouette_k2']:.4f}, bc={r['bimodality_coefficient']:.4f}, var={r['embedding_variance']:.4f}",
                "metadata_task_family": r["task_family"],
                "metadata_model": short_model(r["model"]),
                "metadata_difficulty_level": r["difficulty_level"],
                "metadata_d_star": r["d_star"] if r["d_star"] is not None else -1,
                "metadata_pair_type": "negative",
                "metadata_neg_type": pair["neg_type"],
            }
            # Per-level raw CSD values
            ex["eval_accuracy_value"] = float(r["accuracy"])
            ex["eval_embedding_variance_value"] = float(r["embedding_variance"]) if not np.isnan(r["embedding_variance"]) else float("nan")
            ex["eval_dip_statistic_value"] = float(r["dip_statistic"]) if not np.isnan(r["dip_statistic"]) else float("nan")
            ex["eval_dip_pvalue_value"] = float(r["dip_pvalue"]) if not np.isnan(r["dip_pvalue"]) else float("nan")
            ex["eval_silhouette_value"] = float(r["silhouette_k2"]) if not np.isnan(r["silhouette_k2"]) else float("nan")
            ex["eval_bimodality_coeff_value"] = float(r["bimodality_coefficient"]) if not np.isnan(r["bimodality_coefficient"]) else float("nan")
            ex["eval_disagreement_value"] = float(r["disagreement_rate"]) if not np.isnan(r["disagreement_rate"]) else float("nan")
            examples.append(ex)

        ds_name = f"negative_{pair['task']}_{pair['smodel']}"
        datasets.append({"dataset": ds_name, "examples": examples})

    return datasets


def sanitize_for_json(obj):
    """Recursively convert NaN/Inf to None-safe JSON values. Schema requires numbers, use 0 for NaN."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return -999.0  # Sentinel for missing/invalid (schema requires number)
        return obj
    elif isinstance(obj, (np.floating, np.integer)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return -999.0
        return val
    elif isinstance(obj, np.bool_):
        return 1.0 if obj else 0.0
    elif isinstance(obj, bool):
        return 1.0 if obj else 0.0
    return obj


# ===================================================================
# Main
# ===================================================================

@logger.catch
def main():
    logger.info("=" * 60)
    logger.info("Cross-Task CSD Leading Indicator Evaluation")
    logger.info("=" * 60)

    # Step 0: Load & normalize
    logger.info("STEP 0: Loading and normalizing data...")
    all_rows = load_all_data()
    positive_pairs, negative_pairs = classify_pairs(all_rows)

    # Block 1: Leading Indicator Tests
    logger.info("BLOCK 1: Leading Indicator Tests...")
    block1_results = []
    for i, pair in enumerate(positive_pairs):
        logger.info(f"  Block1 [{i+1}/{len(positive_pairs)}]: {pair['task']} x {pair['smodel']}")
        b1 = compute_block1(pair)
        block1_results.append(b1)
        logger.info(f"    is_leading={b1['eval_is_leading']}, lead_times: "
                     f"dip={b1['eval_lead_time_dip']}, sil={b1['eval_lead_time_silhouette']}, "
                     f"bc={b1['eval_lead_time_bc']}, ashman={b1['eval_lead_time_ashman_d']}")

    # Block 2: Alternative Variance Models
    logger.info("BLOCK 2: Alternative Variance Models...")
    block2_results = []
    for i, pair in enumerate(positive_pairs):
        logger.info(f"  Block2 [{i+1}/{len(positive_pairs)}]: {pair['task']} x {pair['smodel']}")
        b2 = fit_variance_models(pair["rows"], pair["d_star"])
        block2_results.append(b2)
        logger.info(f"    best_model={b2['eval_best_variance_model']}, alpha={b2['eval_bifurcation_alpha']:.4f}, "
                     f"delta_aic={b2['eval_delta_aic_vs_null']:.2f}")

    # Block 3: Cross-Task Consistency
    logger.info("BLOCK 3: Cross-Task Consistency...")
    block3_results = compute_block3(positive_pairs, block1_results)
    logger.info(f"  fraction_leading={block3_results['eval_fraction_pairs_with_leading_indicator']:.2f}, "
                 f"mean_tau_var={block3_results['eval_mean_kendall_tau_variance']:.4f}")

    # Block 4: Effect Sizes
    logger.info("BLOCK 4: Effect Sizes...")
    block4_results = []
    for i, pair in enumerate(positive_pairs):
        logger.info(f"  Block4 [{i+1}/{len(positive_pairs)}]: {pair['task']} x {pair['smodel']}")
        b4 = compute_block4(pair)
        block4_results.append(b4)
        logger.info(f"    Cohen's d: var={b4['eval_cohens_d_variance']:.3f}, dip={b4['eval_cohens_d_dip']:.3f}")

    # Block 5: Negative Controls
    logger.info("BLOCK 5: Negative Controls...")
    block5_results = compute_block5(negative_pairs)
    logger.info(f"  neg_control_pass={block5_results['eval_negative_control_pass']}")

    # Block 6: Hypothesis Verdict
    logger.info("BLOCK 6: Hypothesis Verdict...")
    block6_results = compute_block6(positive_pairs, block1_results, block2_results)
    logger.info(f"  SC1={block6_results['eval_sc1_met']}, SC2={block6_results['eval_sc2_met']}, "
                 f"SC3={block6_results['eval_sc3_met']}")
    logger.info(f"  hypothesis_confirmed={block6_results['eval_hypothesis_confirmed']}, "
                 f"partially={block6_results['eval_hypothesis_partially_confirmed']}")

    # Build metrics_agg
    logger.info("Building metrics_agg...")
    metrics_agg = {}
    metrics_agg["eval_n_positive_pairs"] = float(len(positive_pairs))
    metrics_agg["eval_n_negative_pairs"] = float(len(negative_pairs))

    # Mean lead times across positive pairs
    for ind_key in ["dip", "silhouette", "bc", "ashman_d"]:
        lts = [b[f"eval_lead_time_{ind_key}"] for b in block1_results
               if not np.isnan(b.get(f"eval_lead_time_{ind_key}", float("nan")))
               and b.get(f"eval_lead_time_{ind_key}", float("nan")) > 0]
        metrics_agg[f"eval_mean_lead_time_{ind_key}"] = float(np.mean(lts)) if lts else -999.0

    all_lts = []
    for b in block1_results:
        for k in ["eval_lead_time_dip", "eval_lead_time_silhouette", "eval_lead_time_bc", "eval_lead_time_ashman_d"]:
            v = b.get(k, float("nan"))
            if not np.isnan(v) and v > 0:
                all_lts.append(v)
    metrics_agg["eval_mean_lead_time"] = float(np.mean(all_lts)) if all_lts else -999.0

    # Block 3 metrics
    metrics_agg.update(block3_results)

    # Mean effect sizes
    for key in ["variance", "dip", "silhouette", "bc", "disagreement"]:
        ds = [b[f"eval_cohens_d_{key}"] for b in block4_results
              if not np.isnan(b.get(f"eval_cohens_d_{key}", float("nan")))]
        metrics_agg[f"eval_mean_cohens_d_{key}"] = float(np.mean(ds)) if ds else -999.0

    max_cohens = -999.0
    for key in ["variance", "dip", "silhouette", "bc", "disagreement"]:
        v = metrics_agg.get(f"eval_mean_cohens_d_{key}", -999.0)
        if v > max_cohens and v != -999.0:
            max_cohens = v
    metrics_agg["eval_max_cohens_d_across_indicators"] = max_cohens

    # Block 5 metrics
    metrics_agg.update(block5_results)

    # Block 6 metrics
    metrics_agg.update(block6_results)

    # Best variance model consensus
    model_counts = {}
    for b2 in block2_results:
        m = b2.get("eval_best_variance_model", 4.0)
        model_counts[m] = model_counts.get(m, 0) + 1
    if model_counts:
        consensus = max(model_counts, key=model_counts.get)
    else:
        consensus = 4.0
    metrics_agg["eval_best_variance_model_consensus"] = float(consensus)

    # Build datasets
    logger.info("Building output datasets...")
    datasets = build_datasets(positive_pairs, negative_pairs, block1_results, block2_results, block4_results)

    total_examples = sum(len(ds["examples"]) for ds in datasets)
    logger.info(f"Total datasets: {len(datasets)}, total examples: {total_examples}")

    # Assemble output
    output = {
        "metadata": {
            "evaluation_name": "cross_task_csd_leading_indicator_evaluation",
            "description": "Comprehensive statistical evaluation of CSD indicators as leading indicators of LLM reasoning collapse",
            "task_families": ["arithmetic", "graph_coloring", "syllogistic", "multi_hop"],
            "n_positive_pairs": len(positive_pairs),
            "n_negative_pairs": len(negative_pairs),
            "bootstrap_B": BOOTSTRAP_B,
            "model_encoding": {"bifurcation": 1, "gaussian": 2, "logistic": 3, "null": 4},
        },
        "metrics_agg": metrics_agg,
        "datasets": datasets,
    }

    # Sanitize for JSON (NaN -> -999.0, booleans -> 0/1)
    output = sanitize_for_json(output)

    # Write output
    out_path = WORKSPACE / "eval_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Wrote eval_out.json ({out_path.stat().st_size / 1024:.1f} KB)")

    # Summary
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Positive pairs: {len(positive_pairs)}")
    logger.info(f"Negative pairs: {len(negative_pairs)}")
    logger.info(f"SC1 (leading indicator): {'MET' if block6_results['eval_sc1_met'] > 0.5 else 'NOT MET'}")
    logger.info(f"SC2 (variance scaling): {'MET' if block6_results['eval_sc2_met'] > 0.5 else 'NOT MET'}")
    logger.info(f"SC3 (classifier improvement): {'MET' if block6_results['eval_sc3_met'] > 0.5 else 'NOT MET'}")
    logger.info(f"Hypothesis: {'CONFIRMED' if block6_results['eval_hypothesis_confirmed'] > 0.5 else 'PARTIALLY' if block6_results['eval_hypothesis_partially_confirmed'] > 0.5 else 'DISCONFIRMED'}")
    logger.info(f"Total examples: {total_examples}")
    logger.info("DONE")


if __name__ == "__main__":
    main()
