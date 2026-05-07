#!/usr/bin/env python3
"""Mixture/Cusp/DDM Model Fitting to Empirical CSD Variance and Bimodality Data.

Fits 4 competing theoretical models (mixture, cusp catastrophe, drift-diffusion,
fold bifurcation baseline) to per-level accuracy and variance profiles from 10
series (3 arithmetic, 3 graph-coloring, 4 temperature-sweep conditions).
Produces paper-ready model comparison statistics (R2, AIC, BIC).
NO LLM API calls. NO GPU. Pure scipy optimization on ~240 data points.
"""

from __future__ import annotations

import json
import math
import os
import resource
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from loguru import logger
from scipy import integrate, optimize, stats

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------------------
# Hardware / memory limits
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
TOTAL_RAM_GB = _container_ram_gb() or 8.0
RAM_BUDGET = int(min(4, TOTAL_RAM_GB * 0.3) * 1024**3)  # 4 GB cap; this script is lightweight
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, budget {RAM_BUDGET/1e9:.1f} GB")

# ---------------------------------------------------------------------------
# Constants and Data Paths
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).resolve().parent

DATA_SOURCES = {
    "arithmetic": {
        "path": Path("/ai-inventor/aii_pipeline/runs/neurips-open/3_invention_loop/iter_2/gen_art/exp_id1_it2__opus/method_out.json"),
        "datasets": [
            "csd_indicators__llama-3.1-8b-instruct",
            "csd_indicators__gemini-2.0-flash-001",
            "csd_indicators__gpt-4o-mini",
        ],
        "field_prefix": "predict_",
        "difficulty_field": "metadata_difficulty_level",
    },
    "graph_coloring": {
        "path": Path("/ai-inventor/aii_pipeline/runs/neurips-open/3_invention_loop/iter_2/gen_art/exp_id3_it2__opus/method_out.json"),
        "datasets": [
            "graph_coloring_csd_gpt-4o-mini",
            "graph_coloring_csd_gemini-2.0-flash-001",
            "graph_coloring_csd_gemini-2.0-flash-lite-001",
        ],
        "field_prefix": "metadata_csd_",
        "difficulty_field": "metadata_difficulty_level",
        "needs_dedup": True,
    },
    "temp_sweep": {
        "path": Path("/ai-inventor/aii_pipeline/runs/neurips-open/3_invention_loop/iter_3/gen_art/exp_id3_it3__opus/method_out.json"),
        "datasets": [
            "csd_temp_T0.4__gemini-2.0-flash-001",
            "csd_temp_T0.7__gemini-2.0-flash-001",
            "csd_temp_T1.0__gemini-2.0-flash-001",
            "csd_temp_T1.3__gemini-2.0-flash-001",
        ],
        "field_prefix": "predict_",
        "difficulty_field": "metadata_difficulty_level",
    },
}

# d_star lookup from prior experiment metadata
D_STAR_LOOKUP: dict[str, int] = {}
SCALING_R2_LOOKUP: dict[str, float | None] = {}
TEMPERATURE_LOOKUP: dict[str, float | None] = {}

# Short IDs for series
SERIES_SHORT_ID = {
    "csd_indicators__llama-3.1-8b-instruct": "arith__llama-3.1-8b",
    "csd_indicators__gemini-2.0-flash-001": "arith__gemini-flash",
    "csd_indicators__gpt-4o-mini": "arith__gpt-4o-mini",
    "graph_coloring_csd_gpt-4o-mini": "gc__gpt-4o-mini",
    "graph_coloring_csd_gemini-2.0-flash-001": "gc__gemini-flash",
    "graph_coloring_csd_gemini-2.0-flash-lite-001": "gc__gemini-flash-lite",
    "csd_temp_T0.4__gemini-2.0-flash-001": "temp__T0.4",
    "csd_temp_T0.7__gemini-2.0-flash-001": "temp__T0.7",
    "csd_temp_T1.0__gemini-2.0-flash-001": "temp__T1.0",
    "csd_temp_T1.3__gemini-2.0-flash-001": "temp__T1.3",
}

MODEL_NAME_LOOKUP = {
    "csd_indicators__llama-3.1-8b-instruct": "meta-llama/llama-3.1-8b-instruct",
    "csd_indicators__gemini-2.0-flash-001": "google/gemini-2.0-flash-001",
    "csd_indicators__gpt-4o-mini": "openai/gpt-4o-mini",
    "graph_coloring_csd_gpt-4o-mini": "openai/gpt-4o-mini",
    "graph_coloring_csd_gemini-2.0-flash-001": "google/gemini-2.0-flash-001",
    "graph_coloring_csd_gemini-2.0-flash-lite-001": "google/gemini-2.0-flash-lite-001",
    "csd_temp_T0.4__gemini-2.0-flash-001": "google/gemini-2.0-flash-001",
    "csd_temp_T0.7__gemini-2.0-flash-001": "google/gemini-2.0-flash-001",
    "csd_temp_T1.0__gemini-2.0-flash-001": "google/gemini-2.0-flash-001",
    "csd_temp_T1.3__gemini-2.0-flash-001": "google/gemini-2.0-flash-001",
}

TASK_LOOKUP = {
    "csd_indicators__llama-3.1-8b-instruct": "arithmetic",
    "csd_indicators__gemini-2.0-flash-001": "arithmetic",
    "csd_indicators__gpt-4o-mini": "arithmetic",
    "graph_coloring_csd_gpt-4o-mini": "graph_coloring",
    "graph_coloring_csd_gemini-2.0-flash-001": "graph_coloring",
    "graph_coloring_csd_gemini-2.0-flash-lite-001": "graph_coloring",
    "csd_temp_T0.4__gemini-2.0-flash-001": "temp_sweep",
    "csd_temp_T0.7__gemini-2.0-flash-001": "temp_sweep",
    "csd_temp_T1.0__gemini-2.0-flash-001": "temp_sweep",
    "csd_temp_T1.3__gemini-2.0-flash-001": "temp_sweep",
}


# ===========================================================================
# STEP 1: DATA LOADING
# ===========================================================================

def load_all_series() -> dict[str, dict]:
    """Load and extract per-level CSD indicators for all 10 series."""
    all_series: dict[str, dict] = {}

    for source_name, source_cfg in DATA_SOURCES.items():
        logger.info(f"Loading {source_name} from {source_cfg['path']}")
        raw = json.loads(source_cfg["path"].read_text())

        # Build dataset lookup
        ds_by_name = {ds["dataset"]: ds for ds in raw["datasets"]}

        for ds_name in source_cfg["datasets"]:
            if ds_name not in ds_by_name:
                logger.warning(f"Dataset {ds_name} not found in {source_cfg['path']}")
                continue

            ds = ds_by_name[ds_name]
            examples = ds["examples"]
            prefix = source_cfg["field_prefix"]
            diff_field = source_cfg["difficulty_field"]

            # Extract d_star from metadata
            d_star = _extract_d_star(raw, ds_name, source_name, examples)
            D_STAR_LOOKUP[ds_name] = d_star

            # Extract scaling_r2
            scaling_r2 = _extract_scaling_r2(raw, ds_name, source_name, examples)
            SCALING_R2_LOOKUP[ds_name] = scaling_r2

            # Extract temperature if temp sweep
            temp = _extract_temperature(raw, ds_name, source_name, examples)
            TEMPERATURE_LOOKUP[ds_name] = temp

            # De-duplicate graph coloring: group by level, take first row
            if source_cfg.get("needs_dedup", False):
                seen_levels: dict[int, dict] = {}
                for ex in examples:
                    lev = ex[diff_field]
                    if lev not in seen_levels:
                        seen_levels[lev] = ex
                examples = [seen_levels[k] for k in sorted(seen_levels.keys())]
                logger.info(f"  Deduped {ds_name}: {len(examples)} unique levels")

            # Map field names
            if source_name == "graph_coloring":
                acc_field = f"{prefix}accuracy"
                var_field = f"{prefix}embedding_variance"
                dip_field = f"{prefix}dip_statistic"
                dip_p_field = f"{prefix}dip_pvalue"
                sil_field = f"{prefix}silhouette_score"
                bc_field = f"{prefix}bimodality_coefficient"
                dis_field = f"{prefix}disagreement_rate"
            else:
                acc_field = f"{prefix}accuracy"
                var_field = f"{prefix}csd_variance"
                dip_field = f"{prefix}dip_statistic"
                dip_p_field = f"{prefix}dip_pvalue"
                sil_field = f"{prefix}silhouette_k2"
                bc_field = f"{prefix}bimodality_coefficient"
                dis_field = f"{prefix}disagreement_rate"

            # Sort by difficulty level and extract arrays
            examples_sorted = sorted(examples, key=lambda ex: ex[diff_field])
            levels = np.array([ex[diff_field] for ex in examples_sorted], dtype=float)
            accuracy = np.array([_to_float(ex.get(acc_field, 0)) for ex in examples_sorted])
            variance = np.array([_to_float(ex.get(var_field, 0)) for ex in examples_sorted])
            dip_stat = np.array([_to_float(ex.get(dip_field, 0)) for ex in examples_sorted])
            dip_pval = np.array([_to_float(ex.get(dip_p_field, 1)) for ex in examples_sorted])
            silhouette = np.array([_to_float(ex.get(sil_field, 0)) for ex in examples_sorted])
            bimod_coef = np.array([_to_float(ex.get(bc_field, 0)) for ex in examples_sorted])
            disagreement = np.array([_to_float(ex.get(dis_field, 0)) for ex in examples_sorted])

            series_id = SERIES_SHORT_ID.get(ds_name, ds_name)

            all_series[series_id] = {
                "dataset_name": ds_name,
                "task": TASK_LOOKUP[ds_name],
                "model": MODEL_NAME_LOOKUP.get(ds_name, "unknown"),
                "temperature": temp,
                "levels": levels,
                "accuracy": accuracy,
                "variance": variance,
                "dip_stat": dip_stat,
                "dip_pvalue": dip_pval,
                "silhouette": silhouette,
                "bimod_coef": bimod_coef,
                "disagreement": disagreement,
                "d_star": d_star,
            }
            logger.info(
                f"  Series {series_id}: {len(levels)} levels, "
                f"d_star={d_star}, acc range=[{accuracy.min():.2f}, {accuracy.max():.2f}], "
                f"var range=[{variance.min():.4f}, {variance.max():.4f}]"
            )

    logger.info(f"Loaded {len(all_series)} series total")
    return all_series


def _to_float(val) -> float:
    """Convert string or numeric to float, handling None/NaN."""
    if val is None:
        return float("nan")
    try:
        return float(val)
    except (ValueError, TypeError):
        return float("nan")


def _extract_d_star(raw: dict, ds_name: str, source_name: str, examples: list) -> int:
    """Extract d_star from metadata or examples."""
    if source_name == "arithmetic":
        meta = raw.get("metadata", {}).get("model_summaries", {})
        for model_key, model_data in meta.items():
            if ds_name.endswith(model_key.split("/")[-1].replace("llama-", "llama-")):
                return int(model_data.get("d_star", 2))
            # Fuzzy match
            short = model_key.split("/")[-1]
            if short in ds_name:
                return int(model_data.get("d_star", 2))
        # fallback: from examples
        if examples and "metadata_d_star" in examples[0]:
            return int(examples[0]["metadata_d_star"])
    elif source_name == "graph_coloring":
        meta = raw.get("metadata", {}).get("analysis", {}).get("models", [])
        for m in meta:
            if m["model"].split("/")[-1] in ds_name:
                return int(m.get("d_star", 10))
    elif source_name == "temp_sweep":
        if examples and "metadata_d_star" in examples[0]:
            return int(examples[0]["metadata_d_star"])
    return 10  # fallback


def _extract_scaling_r2(raw: dict, ds_name: str, source_name: str, examples: list) -> float | None:
    """Extract scaling R2 from metadata."""
    if source_name == "arithmetic":
        meta = raw.get("metadata", {}).get("model_summaries", {})
        for model_key, model_data in meta.items():
            short = model_key.split("/")[-1]
            if short in ds_name:
                v = model_data.get("scaling_r2")
                return float(v) if v is not None else None
    elif source_name == "graph_coloring":
        meta = raw.get("metadata", {}).get("analysis", {}).get("models", [])
        for m in meta:
            if m["model"].split("/")[-1] in ds_name:
                v = m.get("scaling_r_squared")
                return float(v) if v is not None else None
    elif source_name == "temp_sweep":
        if examples and "metadata_scaling_r2" in examples[0]:
            v = examples[0]["metadata_scaling_r2"]
            return float(v) if v is not None else None
    return None


def _extract_temperature(raw: dict, ds_name: str, source_name: str, examples: list) -> float | None:
    """Extract temperature from dataset name or examples."""
    if source_name == "temp_sweep":
        if examples and "metadata_temperature" in examples[0]:
            return float(examples[0]["metadata_temperature"])
        # parse from name like csd_temp_T0.4__...
        for part in ds_name.split("_"):
            if part.startswith("T") and len(part) > 1:
                try:
                    return float(part[1:])
                except ValueError:
                    pass
    return None


# ===========================================================================
# STEP 2: MODEL FITTING
# ===========================================================================

def _compute_r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Compute R-squared, handling edge cases."""
    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    if ss_tot < 1e-15:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def _compute_aic_bic(n: int, k: int, ss_res: float) -> tuple[float, float]:
    """Compute AIC and BIC from residual sum of squares (Gaussian assumption)."""
    if n <= 0 or ss_res <= 0:
        return float("nan"), float("nan")
    mse = ss_res / n
    log_lik = -n / 2.0 * (np.log(2 * np.pi * mse) + 1.0)
    aic = 2.0 * k - 2.0 * log_lik
    bic = k * np.log(n) - 2.0 * log_lik
    return float(aic), float(bic)


# --- 2A: MIXTURE MODEL ---
def fit_mixture_model(d: np.ndarray, accuracy: np.ndarray, variance: np.ndarray) -> dict:
    """Fit both 2-param and 3-param mixture models. Return best.

    2-param: Var = p*(1-p)*D_sq + c  (equal within-cluster variance)
    3-param: Var = p*s_c + (1-p)*s_i + p*(1-p)*D_sq  (unequal within-cluster variance)

    The 3-param model uses the full law of total variance:
      Var_total = E[Var(X|Z)] + Var[E(X|Z)]
               = p*sigma_c^2 + (1-p)*sigma_i^2 + p*(1-p)*||mu_c - mu_i||^2
    """
    n = len(d)
    p = accuracy.copy()
    between = p * (1.0 - p)

    nan_result = {
        "D_sq": float("nan"), "s_c": float("nan"), "s_i": float("nan"),
        "c": float("nan"), "R2": float("nan"), "R2_2param": float("nan"),
        "AIC": float("nan"), "BIC": float("nan"),
        "predicted_variance": [float("nan")] * n,
        "peak_accuracy_level": None, "n_params": 0,
    }

    # --- 2-parameter model: Var = p*(1-p)*D_sq + c ---
    r2_2p = float("nan")
    pred_2p = np.full(n, float("nan"))
    d_sq_2p, c_2p = float("nan"), float("nan")

    if np.max(between) >= 1e-10:
        def model_2p(_, d_sq, c):
            return between * d_sq + c
        try:
            popt2, _ = optimize.curve_fit(
                model_2p, d, variance,
                p0=[4 * np.max(variance), np.min(variance)],
                bounds=([0.0, 0.0], [np.inf, np.inf]),
                maxfev=5000,
            )
            d_sq_2p, c_2p = popt2
            pred_2p = model_2p(d, d_sq_2p, c_2p)
            r2_2p = _compute_r2(variance, pred_2p)
        except (RuntimeError, ValueError):
            pass

    # --- 3-parameter model: Var = p*s_c + (1-p)*s_i + p*(1-p)*D_sq ---
    r2_3p = float("nan")
    pred_3p = np.full(n, float("nan"))
    s_c_3p, s_i_3p, d_sq_3p = float("nan"), float("nan"), float("nan")

    # Design matrix: columns are [p, (1-p), p*(1-p)]
    A = np.column_stack([p, 1.0 - p, between])
    try:
        # Use scipy.optimize.nnls for non-negative least squares
        # (s_c, s_i, D_sq must all be >= 0)
        x_nnls, residual = optimize.nnls(A, variance)
        s_c_3p, s_i_3p, d_sq_3p = x_nnls
        pred_3p = A @ x_nnls
        r2_3p = _compute_r2(variance, pred_3p)
    except Exception:
        pass

    # Pick the better model (prefer 3-param if it improves fit meaningfully)
    # Use AIC to decide: 3-param has more flexibility but more parameters
    ss_2p = float(np.sum((variance - pred_2p) ** 2)) if not np.isnan(r2_2p) else float("inf")
    ss_3p = float(np.sum((variance - pred_3p) ** 2)) if not np.isnan(r2_3p) else float("inf")

    aic_2p, bic_2p = _compute_aic_bic(n, k=2, ss_res=ss_2p) if np.isfinite(ss_2p) else (float("nan"), float("nan"))
    aic_3p, bic_3p = _compute_aic_bic(n, k=3, ss_res=ss_3p) if np.isfinite(ss_3p) else (float("nan"), float("nan"))

    # Select winner by AIC
    use_3p = (not np.isnan(aic_3p)) and (np.isnan(aic_2p) or aic_3p < aic_2p)

    if use_3p:
        predicted = pred_3p
        r2 = r2_3p
        aic, bic = aic_3p, bic_3p
        d_sq_best = d_sq_3p
        n_params = 3
    elif not np.isnan(r2_2p):
        predicted = pred_2p
        r2 = r2_2p
        aic, bic = aic_2p, bic_2p
        d_sq_best = d_sq_2p
        n_params = 2
    else:
        return nan_result

    # Peak: where p*(1-p) is maximized
    peak_idx = int(np.argmax(between)) if np.max(between) >= 1e-10 else 0
    peak_level = int(d[peak_idx]) if peak_idx < len(d) else None

    return {
        "D_sq": float(d_sq_best),
        "s_c": float(s_c_3p),
        "s_i": float(s_i_3p),
        "c": float(c_2p),
        "R2": float(r2),
        "R2_2param": float(r2_2p),
        "AIC": float(aic),
        "BIC": float(bic),
        "predicted_variance": predicted.tolist(),
        "peak_accuracy_level": peak_level,
        "n_params": n_params,
    }


# --- 2B: FOLD BIFURCATION MODEL (baseline) ---
def fit_fold_bifurcation(d: np.ndarray, variance: np.ndarray, d_star: int | None) -> dict:
    """Fit Var_fold(d) = A * (d_star - d)^gamma + B, baseline model."""
    n = len(d)
    if d_star is None:
        d_star = int(d[np.argmax(variance)])

    # Only fit to points where d < d_star
    mask_pre = d < d_star
    n_pre = int(np.sum(mask_pre))

    if n_pre < 3:
        logger.warning(f"Fold: insufficient pre-transition data (n_pre={n_pre}, d_star={d_star})")
        return {
            "A": float("nan"), "B": float("nan"), "gamma": -0.5,
            "R2_pre": float("nan"), "R2_all": float("nan"),
            "AIC": float("nan"), "BIC": float("nan"),
            "n_fitted": n_pre, "valid": False,
            "predicted_variance": [float("nan")] * n,
        }

    d_pre = d[mask_pre]
    var_pre = variance[mask_pre]

    # --- Fixed gamma = -0.5 ---
    def fold_fixed(x, a, b):
        dist = np.maximum(d_star - x, 1e-6)
        return a * dist ** (-0.5) + b

    try:
        popt_fixed, _ = optimize.curve_fit(
            fold_fixed, d_pre, var_pre,
            p0=[0.01, np.min(var_pre)],
            bounds=([0, 0], [np.inf, np.inf]),
            maxfev=5000,
        )
        a_f, b_f = popt_fixed
        pred_pre_fixed = fold_fixed(d_pre, a_f, b_f)
        r2_pre_fixed = _compute_r2(var_pre, pred_pre_fixed)
    except (RuntimeError, ValueError):
        a_f, b_f = float("nan"), float("nan")
        r2_pre_fixed = float("nan")

    # --- Free gamma ---
    def fold_free(x, a, b, gamma):
        dist = np.maximum(d_star - x, 1e-6)
        return a * dist ** gamma + b

    try:
        popt_free, _ = optimize.curve_fit(
            fold_free, d_pre, var_pre,
            p0=[0.01, np.min(var_pre), -0.5],
            bounds=([0, 0, -2], [np.inf, np.inf, 0]),
            maxfev=5000,
        )
        a_free, b_free, gamma_free = popt_free
        pred_pre_free = fold_free(d_pre, a_free, b_free, gamma_free)
        r2_pre_free = _compute_r2(var_pre, pred_pre_free)
    except (RuntimeError, ValueError):
        a_free, b_free, gamma_free = float("nan"), float("nan"), float("nan")
        r2_pre_free = float("nan")

    # Pick the better fit (free gamma usually better)
    if not np.isnan(r2_pre_free) and (np.isnan(r2_pre_fixed) or r2_pre_free > r2_pre_fixed):
        a_best, b_best, gamma_best = a_free, b_free, gamma_free
        r2_pre = r2_pre_free
        k = 3
    else:
        a_best, b_best, gamma_best = a_f, b_f, -0.5
        r2_pre = r2_pre_fixed
        k = 2

    # R2 on ALL points — cap predictions beyond d_star at boundary value
    try:
        boundary_val = a_best * max(1.0, 1e-6) ** gamma_best + b_best  # value at d = d_star - 1
        pred_all = np.zeros(n)
        for i_pt in range(n):
            if d[i_pt] < d_star:
                dist = max(d_star - d[i_pt], 1e-6)
                pred_all[i_pt] = a_best * dist ** gamma_best + b_best
            else:
                pred_all[i_pt] = boundary_val  # flat extrapolation beyond d_star
        r2_all = _compute_r2(variance, pred_all)
    except Exception:
        r2_all = float("nan")
        pred_all = np.full(n, float("nan"))

    ss_res_pre = float(np.sum((var_pre - (a_best * np.maximum(d_star - d_pre, 1e-6) ** gamma_best + b_best)) ** 2)) if n_pre > 0 else float("nan")
    aic, bic = _compute_aic_bic(n_pre, k=k, ss_res=ss_res_pre)

    return {
        "A": float(a_best),
        "B": float(b_best),
        "gamma": float(gamma_best),
        "R2_pre": float(r2_pre),
        "R2_all": float(r2_all),
        "AIC": float(aic),
        "BIC": float(bic),
        "n_fitted": n_pre,
        "valid": True,
        "predicted_variance": pred_all.tolist(),
    }


# --- 2C: CUSP CATASTROPHE MODEL (4 parameters) ---
def _cusp_stationary_density(x: np.ndarray, alpha: float, beta: float, sigma: float) -> np.ndarray:
    """Compute unnormalized cusp stationary density at points x."""
    exponent = (2.0 / sigma**2) * (alpha * x + beta * x**2 / 2.0 - x**4 / 4.0)
    # Stabilize by subtracting max
    exponent -= np.max(exponent)
    return np.exp(exponent)


def _cusp_moments(alpha: float, beta: float, sigma: float,
                  x_range: float = 5.0, n_pts: int = 500) -> tuple[float, float, float]:
    """Compute mean, variance, and P(x>0) for the cusp stationary density."""
    x = np.linspace(-x_range, x_range, n_pts)
    dx = x[1] - x[0]
    density = _cusp_stationary_density(x, alpha, beta, sigma)
    z = np.sum(density) * dx
    if z < 1e-30:
        return 0.0, 0.0, 0.5

    density_norm = density / z
    mean_x = float(np.sum(x * density_norm) * dx)
    var_x = float(np.sum(x**2 * density_norm) * dx - mean_x**2)
    var_x = max(var_x, 0.0)

    # P(correct) = P(x > 0)
    p_correct = float(np.sum(density_norm[x > 0]) * dx)
    p_correct = np.clip(p_correct, 0.01, 0.99)

    return mean_x, var_x, p_correct


def fit_cusp_model(d: np.ndarray, accuracy: np.ndarray, variance: np.ndarray) -> dict:
    """Fit cusp catastrophe model: alpha(d) = a0 + a1*d, with beta and sigma."""
    n = len(d)
    d_norm = d / np.max(d)  # normalize difficulty to [0,1] for numerical stability

    # Variance and accuracy scales for normalization
    var_scale = np.std(variance) if np.std(variance) > 1e-10 else 1.0
    acc_scale = np.std(accuracy) if np.std(accuracy) > 1e-10 else 1.0

    def cusp_loss(params):
        a0, a1, beta, sigma = params
        total_loss = 0.0
        for i, di in enumerate(d_norm):
            alpha_i = a0 + a1 * di
            try:
                _, var_pred, acc_pred = _cusp_moments(alpha_i, beta, sigma)
            except Exception:
                return 1e10
            # Normalize residuals
            total_loss += ((acc_pred - accuracy[i]) / acc_scale) ** 2
            total_loss += ((var_pred - variance[i]) / var_scale) ** 2
        return total_loss

    # Initial guesses
    best_result = None
    best_loss = float("inf")

    initial_guesses = [
        [2.0, -4.0, 1.5, 1.0],
        [1.5, -3.0, 2.0, 0.8],
        [3.0, -6.0, 1.0, 1.5],
        [1.0, -2.0, 2.5, 0.5],
    ]

    for p0 in initial_guesses:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = optimize.minimize(
                    cusp_loss, p0,
                    method="L-BFGS-B",
                    bounds=[(-10, 10), (-20, 0), (0.1, 10), (0.1, 5)],
                    options={"maxiter": 2000, "ftol": 1e-10},
                )
            if result.fun < best_loss:
                best_loss = result.fun
                best_result = result
        except Exception:
            continue

    if best_result is None or not best_result.success and best_loss > 1e8:
        # Try Nelder-Mead as fallback
        try:
            result = optimize.minimize(
                cusp_loss, [2.0, -4.0, 1.5, 1.0],
                method="Nelder-Mead",
                options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-8},
            )
            if result.fun < best_loss:
                best_loss = result.fun
                best_result = result
        except Exception:
            pass

    if best_result is None:
        logger.warning("Cusp model: all optimizers failed")
        return {
            "a0": float("nan"), "a1": float("nan"),
            "beta": float("nan"), "sigma": float("nan"),
            "R2_variance": float("nan"), "R2_accuracy": float("nan"),
            "AIC": float("nan"), "BIC": float("nan"),
            "bimodal_region": None,
            "cardan_discriminant": [],
            "predicted_variance": [float("nan")] * n,
            "predicted_accuracy": [float("nan")] * n,
        }

    a0, a1, beta, sigma = best_result.x

    # Compute raw predictions at each level
    raw_var = np.zeros(n)
    raw_acc = np.zeros(n)
    cardan = np.zeros(n)

    for i, di in enumerate(d_norm):
        alpha_i = a0 + a1 * di
        _, var_i, acc_i = _cusp_moments(alpha_i, beta, sigma)
        raw_var[i] = var_i
        raw_acc[i] = acc_i
        cardan[i] = 27 * alpha_i**2 - 4 * beta**3

    # Linear mapping from cusp internal scale to observed scale
    # This is standard practice: cusp gives shape, linear map gives scale
    # Variance mapping: obs_var = sv * cusp_var + ov
    if np.std(raw_var) > 1e-12:
        sv = np.cov(variance, raw_var)[0, 1] / np.var(raw_var)
        ov = np.mean(variance) - sv * np.mean(raw_var)
        pred_var = sv * raw_var + ov
    else:
        pred_var = np.full(n, np.mean(variance))

    # Accuracy mapping: obs_acc = sa * cusp_acc + oa
    if np.std(raw_acc) > 1e-12:
        sa = np.cov(accuracy, raw_acc)[0, 1] / np.var(raw_acc)
        oa = np.mean(accuracy) - sa * np.mean(raw_acc)
        pred_acc = sa * raw_acc + oa
    else:
        pred_acc = np.full(n, np.mean(accuracy))

    r2_var = _compute_r2(variance, pred_var)
    r2_acc = _compute_r2(accuracy, pred_acc)

    # AIC/BIC for variance prediction: k=6 (4 cusp params + 2 linear mapping params)
    ss_res_var = float(np.sum((variance - pred_var) ** 2))
    aic, bic = _compute_aic_bic(n, k=6, ss_res=ss_res_var)

    # Bimodal region: where Cardan discriminant < 0
    bimodal_mask = cardan < 0
    if np.any(bimodal_mask):
        bimodal_levels = d[bimodal_mask]
        bimodal_region = [int(bimodal_levels.min()), int(bimodal_levels.max())]
    else:
        bimodal_region = None

    return {
        "a0": float(a0),
        "a1": float(a1),
        "beta": float(beta),
        "sigma": float(sigma),
        "R2_variance": float(r2_var),
        "R2_accuracy": float(r2_acc),
        "AIC": float(aic),
        "BIC": float(bic),
        "bimodal_region": bimodal_region,
        "cardan_discriminant": cardan.tolist(),
        "predicted_variance": pred_var.tolist(),
        "predicted_accuracy": pred_acc.tolist(),
    }


# --- 2D: DRIFT-DIFFUSION MODEL (4 parameters) ---
def fit_ddm(d: np.ndarray, accuracy: np.ndarray, variance: np.ndarray) -> dict:
    """Fit DDM: logistic accuracy + mixture variance from fitted sigmoid."""
    n = len(d)

    # Step 1: Fit logistic to accuracy
    def sigmoid(x, c0, c1):
        z = c0 + c1 * x
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    try:
        # Guess c1 sign: accuracy typically decreases with d
        if accuracy[0] > accuracy[-1]:
            c1_init = -0.5
        else:
            c1_init = 0.5
        c0_init = -c1_init * np.mean(d)

        popt_sig, _ = optimize.curve_fit(
            sigmoid, d, accuracy,
            p0=[c0_init, c1_init],
            bounds=([-50, -50], [50, 50]),
            maxfev=5000,
        )
        c0, c1 = popt_sig
    except (RuntimeError, ValueError) as exc:
        logger.warning(f"DDM sigmoid fit failed: {exc}")
        return {
            "c0": float("nan"), "c1": float("nan"),
            "D_sq": float("nan"), "c_var": float("nan"),
            "R2_accuracy": float("nan"), "R2_variance": float("nan"),
            "AIC": float("nan"), "BIC": float("nan"),
            "implied_d_star": float("nan"),
            "predicted_variance": [float("nan")] * n,
            "predicted_accuracy": [float("nan")] * n,
        }

    # Accuracy predictions from sigmoid
    acc_pred = sigmoid(d, c0, c1)
    r2_acc = _compute_r2(accuracy, acc_pred)

    # Step 2: Fit variance using DDM-smoothed accuracy
    between_ddm = acc_pred * (1.0 - acc_pred)

    if np.max(between_ddm) < 1e-10:
        logger.warning("DDM: between-component term is zero")
        return {
            "c0": float(c0), "c1": float(c1),
            "D_sq": float("nan"), "c_var": float("nan"),
            "R2_accuracy": float(r2_acc), "R2_variance": float("nan"),
            "AIC": float("nan"), "BIC": float("nan"),
            "implied_d_star": float(-c0 / c1) if abs(c1) > 1e-10 else float("nan"),
            "predicted_variance": [float("nan")] * n,
            "predicted_accuracy": acc_pred.tolist(),
        }

    def var_model(_, d_sq, c_var):
        return between_ddm * d_sq + c_var

    try:
        popt_var, _ = optimize.curve_fit(
            var_model, d, variance,
            p0=[4 * np.max(variance), np.min(variance)],
            bounds=([0, 0], [np.inf, np.inf]),
            maxfev=5000,
        )
        d_sq, c_var = popt_var
    except (RuntimeError, ValueError):
        d_sq, c_var = float("nan"), float("nan")

    var_pred = var_model(d, d_sq, c_var) if not np.isnan(d_sq) else np.full(n, float("nan"))
    r2_var = _compute_r2(variance, var_pred) if not np.isnan(d_sq) else float("nan")

    # AIC/BIC with k=4 (c0, c1, d_sq, c_var)
    ss_res_var = float(np.sum((variance - var_pred) ** 2)) if not np.isnan(d_sq) else float("nan")
    aic, bic = _compute_aic_bic(n, k=4, ss_res=ss_res_var)

    implied_d_star = float(-c0 / c1) if abs(c1) > 1e-10 else float("nan")

    return {
        "c0": float(c0),
        "c1": float(c1),
        "D_sq": float(d_sq),
        "c_var": float(c_var),
        "R2_accuracy": float(r2_acc),
        "R2_variance": float(r2_var),
        "AIC": float(aic),
        "BIC": float(bic),
        "implied_d_star": float(implied_d_star),
        "predicted_variance": var_pred.tolist() if not np.isnan(d_sq) else [float("nan")] * n,
        "predicted_accuracy": acc_pred.tolist(),
    }


# ===========================================================================
# STEP 4: PER-SERIES ANALYSIS PIPELINE
# ===========================================================================

def analyze_series(series_id: str, series_data: dict) -> dict:
    """Run all 4 model fits on one series."""
    t0 = time.time()

    d = series_data["levels"]
    acc = series_data["accuracy"]
    var = series_data["variance"]
    d_star = series_data["d_star"]

    results: dict = {
        "series_id": series_id,
        "task": series_data["task"],
        "model": series_data["model"],
        "temperature": series_data.get("temperature"),
        "n_levels": len(d),
        "d_star": d_star,
    }

    # 4A: Mixture model fit
    logger.debug(f"  Fitting mixture model for {series_id}")
    results["mixture"] = fit_mixture_model(d, acc, var)

    # 4B: Fold bifurcation fit (baseline)
    logger.debug(f"  Fitting fold bifurcation for {series_id}")
    results["fold"] = fit_fold_bifurcation(d, var, d_star)

    # 4C: Cusp catastrophe fit
    logger.debug(f"  Fitting cusp model for {series_id}")
    results["cusp"] = fit_cusp_model(d, acc, var)

    # 4D: DDM fit
    logger.debug(f"  Fitting DDM for {series_id}")
    results["ddm"] = fit_ddm(d, acc, var)

    # 4E: PREDICTION TEST — peak variance vs 50% accuracy
    var_peak_idx = int(np.argmax(var))
    d_peak = int(d[var_peak_idx])
    d_50_idx = int(np.argmin(np.abs(acc - 0.5)))
    d_50 = int(d[d_50_idx])

    results["prediction_test"] = {
        "d_peak_empirical": d_peak,
        "d_50pct_accuracy": d_50,
        "accuracy_at_peak": float(acc[var_peak_idx]),
        "match_within_1": abs(d_peak - d_50) <= 1,
        "match_within_2": abs(d_peak - d_50) <= 2,
        "gap": abs(d_peak - d_50),
    }

    # 4F: MODEL RANKING by AIC
    # For mixture, fold, DDM: use variance-based AIC
    # For cusp: use joint AIC (fit to both acc and var)
    model_aics: dict[str, float] = {}
    for model_name in ["mixture", "fold", "cusp", "ddm"]:
        aic_val = results[model_name].get("AIC")
        if aic_val is not None and not np.isnan(aic_val):
            model_aics[model_name] = aic_val

    # Variance-only R2 comparison (fair: all evaluated on same data points)
    model_var_r2: dict[str, float] = {}
    for model_name in ["mixture", "fold", "cusp", "ddm"]:
        if model_name == "mixture":
            r2_val = results[model_name].get("R2")
        elif model_name == "fold":
            # Use R2_all (capped at boundary) for cross-model comparison
            r2_val = results[model_name].get("R2_all")
        elif model_name in ("cusp", "ddm"):
            r2_val = results[model_name].get("R2_variance")
        else:
            r2_val = None
        if r2_val is not None and not np.isnan(r2_val):
            model_var_r2[model_name] = r2_val

    if model_var_r2:
        best_r2 = max(model_var_r2, key=model_var_r2.get)
        results["best_model_r2"] = best_r2
        results["model_ranking_r2"] = sorted(model_var_r2.keys(), key=lambda k: -model_var_r2[k])

    if model_aics:
        best_aic = min(model_aics, key=model_aics.get)
        results["best_model_aic"] = best_aic
        results["model_ranking_aic"] = sorted(model_aics.keys(), key=lambda k: model_aics[k])

    elapsed = time.time() - t0
    logger.info(
        f"  {series_id}: mixture_R2={results['mixture'].get('R2', 'nan'):.3f}, "
        f"fold_R2_pre={results['fold'].get('R2_pre', 'nan')}, "
        f"cusp_R2_var={results['cusp'].get('R2_variance', 'nan'):.3f}, "
        f"ddm_R2_var={results['ddm'].get('R2_variance', 'nan'):.3f}, "
        f"best_R2={results.get('best_model_r2', 'N/A')}, "
        f"peak_match={results['prediction_test']['match_within_1']}, "
        f"elapsed={elapsed:.1f}s"
    )

    return results


# ===========================================================================
# STEP 5: AGGREGATION
# ===========================================================================

def compute_aggregate_stats(all_results: list[dict]) -> dict:
    """Cross-series aggregation for paper-ready statistics."""
    n_series = len(all_results)

    # Per-model R2 collection
    r2_by_model: dict[str, list[float]] = {"mixture": [], "fold": [], "cusp": [], "ddm": []}
    aic_wins: dict[str, int] = {"mixture": 0, "fold": 0, "cusp": 0, "ddm": 0}
    bic_wins: dict[str, int] = {"mixture": 0, "fold": 0, "cusp": 0, "ddm": 0}
    r2_wins: dict[str, int] = {"mixture": 0, "fold": 0, "cusp": 0, "ddm": 0}
    prediction_match_1 = 0
    prediction_match_2 = 0
    cusp_betas: list[float] = []
    cusp_bimodal_detected = 0

    for r in all_results:
        # Mixture R2
        mix_r2 = r["mixture"].get("R2")
        if mix_r2 is not None and not np.isnan(mix_r2):
            r2_by_model["mixture"].append(mix_r2)

        # Fold R2 — use R2_all but clamp to [-1, 1] to prevent pollution
        fold_r2 = r["fold"].get("R2_all")
        if fold_r2 is not None and not np.isnan(fold_r2):
            r2_by_model["fold"].append(max(fold_r2, -1.0))
        else:
            r2_by_model["fold"].append(float("nan"))

        # Cusp R2
        cusp_r2 = r["cusp"].get("R2_variance")
        if cusp_r2 is not None and not np.isnan(cusp_r2):
            r2_by_model["cusp"].append(cusp_r2)

        # DDM R2
        ddm_r2 = r["ddm"].get("R2_variance")
        if ddm_r2 is not None and not np.isnan(ddm_r2):
            r2_by_model["ddm"].append(ddm_r2)

        # AIC wins
        best_aic = r.get("best_model_aic")
        if best_aic:
            aic_wins[best_aic] = aic_wins.get(best_aic, 0) + 1

        # R2 wins
        best_r2 = r.get("best_model_r2")
        if best_r2:
            r2_wins[best_r2] = r2_wins.get(best_r2, 0) + 1

        # Prediction test
        pt = r.get("prediction_test", {})
        if pt.get("match_within_1"):
            prediction_match_1 += 1
        if pt.get("match_within_2"):
            prediction_match_2 += 1

        # Cusp beta
        cusp_beta = r["cusp"].get("beta")
        if cusp_beta is not None and not np.isnan(cusp_beta):
            cusp_betas.append(cusp_beta)

        if r["cusp"].get("bimodal_region") is not None:
            cusp_bimodal_detected += 1

    # Compute mean/median R2
    per_model_mean_r2 = {}
    per_model_median_r2 = {}
    for model_name, vals in r2_by_model.items():
        clean = [v for v in vals if not np.isnan(v)]
        if clean:
            per_model_mean_r2[model_name] = float(np.mean(clean))
            per_model_median_r2[model_name] = float(np.median(clean))
        else:
            per_model_mean_r2[model_name] = float("nan")
            per_model_median_r2[model_name] = float("nan")

    # Mixture vs fold paired comparison
    mixture_r2s = []
    fold_r2s = []
    for r in all_results:
        mr = r["mixture"].get("R2")
        fr = r["fold"].get("R2_all")
        if mr is not None and not np.isnan(mr) and fr is not None and not np.isnan(fr):
            mixture_r2s.append(mr)
            fold_r2s.append(fr)

    paired_result = {}
    if len(mixture_r2s) >= 2:
        mixture_arr = np.array(mixture_r2s)
        fold_arr = np.array(fold_r2s)
        improvement = float(np.mean(mixture_arr - fold_arr))
        n_wins = int(np.sum(mixture_arr > fold_arr))
        try:
            t_stat, p_val = stats.ttest_rel(mixture_arr, fold_arr)
            paired_result = {
                "mean_R2_improvement": improvement,
                "wins": n_wins,
                "total": len(mixture_r2s),
                "paired_t_statistic": float(t_stat),
                "paired_t_pvalue": float(p_val),
            }
        except Exception:
            paired_result = {
                "mean_R2_improvement": improvement,
                "wins": n_wins,
                "total": len(mixture_r2s),
            }

    stats_out = {
        "n_series": n_series,
        "per_model_mean_R2": per_model_mean_r2,
        "per_model_median_R2": per_model_median_r2,
        "aic_win_counts": aic_wins,
        "r2_win_counts": r2_wins,
        "mixture_vs_fold_paired": paired_result,
        "prediction_test_success_rate": prediction_match_1 / n_series if n_series > 0 else 0,
        "prediction_test_within_2_rate": prediction_match_2 / n_series if n_series > 0 else 0,
        "cusp_beta_range": [float(min(cusp_betas)), float(max(cusp_betas))] if cusp_betas else None,
        "cusp_bimodal_detected_fraction": cusp_bimodal_detected / n_series if n_series > 0 else 0,
    }

    # Temperature sweep specific: sigma vs temperature correlation
    temp_series = [r for r in all_results if r["task"] == "temp_sweep"]
    if len(temp_series) >= 2:
        temps = []
        sigmas = []
        for r in temp_series:
            t_val = r.get("temperature")
            s_val = r["cusp"].get("sigma")
            if t_val is not None and s_val is not None and not np.isnan(s_val):
                temps.append(t_val)
                sigmas.append(s_val)
        if len(temps) >= 2:
            try:
                rho, p_val = stats.spearmanr(temps, sigmas)
                stats_out["temp_sigma_correlation"] = {
                    "temperatures": temps,
                    "sigmas": sigmas,
                    "spearman_rho": float(rho),
                    "p_value": float(p_val),
                }
            except Exception:
                pass

    return stats_out


# ===========================================================================
# STEP 6: OUTPUT FORMATTING
# ===========================================================================

def format_output(all_results: list[dict], all_series: dict[str, dict],
                  aggregate: dict) -> dict:
    """Format results into exp_gen_sol_out.json schema."""
    datasets = []

    # --- Dataset 1: model_comparison_all_series ---
    comparison_examples = []
    for r in all_results:
        sid = r["series_id"]
        mix = r["mixture"]
        fold = r["fold"]
        cusp = r["cusp"]
        ddm = r["ddm"]
        pt = r.get("prediction_test", {})

        best_model = r.get("best_model_r2", "unknown")
        best_aic = r.get("best_model_aic", "unknown")

        # Build output summary
        summary_parts = [f"Best model by R2: {best_model}"]
        if not np.isnan(mix.get("R2", float("nan"))):
            summary_parts.append(f"mixture_R2={mix['R2']:.3f}")
        summary_parts.append(f"peak_gap={pt.get('gap', 'N/A')}")
        output_str = "; ".join(summary_parts)

        ex = {
            "input": f"Model fitting for series {sid} on {r['task']} task",
            "output": output_str,
            "predict_mixture_R2": str(round(mix.get("R2", float("nan")), 4)),
            "predict_fold_R2": str(round(fold.get("R2_all", float("nan")), 4)),
            "predict_cusp_R2_variance": str(round(cusp.get("R2_variance", float("nan")), 4)),
            "predict_ddm_R2_variance": str(round(ddm.get("R2_variance", float("nan")), 4)),
            "predict_mixture_AIC": str(round(mix.get("AIC", float("nan")), 2)),
            "predict_fold_AIC": str(round(fold.get("AIC", float("nan")), 2)),
            "predict_cusp_AIC": str(round(cusp.get("AIC", float("nan")), 2)),
            "predict_ddm_AIC": str(round(ddm.get("AIC", float("nan")), 2)),
            "predict_best_model_r2": str(best_model),
            "predict_best_model_aic": str(best_aic),
            "predict_mixture_D_sq": str(round(mix.get("D_sq", float("nan")), 4)),
            "predict_mixture_c": str(round(mix.get("c", float("nan")), 4)),
            "predict_mixture_s_c": str(round(mix.get("s_c", float("nan")), 4)),
            "predict_mixture_s_i": str(round(mix.get("s_i", float("nan")), 4)),
            "predict_mixture_n_params": str(mix.get("n_params", "")),
            "predict_mixture_R2_2param": str(round(mix.get("R2_2param", float("nan")), 4)),
            "predict_cusp_beta": str(round(cusp.get("beta", float("nan")), 4)),
            "predict_cusp_sigma": str(round(cusp.get("sigma", float("nan")), 4)),
            "predict_cusp_bimodal_region": str(cusp.get("bimodal_region", "none")),
            "predict_ddm_implied_d_star": str(round(ddm.get("implied_d_star", float("nan")), 2)),
            "predict_peak_variance_d": str(pt.get("d_peak_empirical", "")),
            "predict_half_accuracy_d": str(pt.get("d_50pct_accuracy", "")),
            "predict_peak_match_within_1": str(pt.get("match_within_1", False)).lower(),
            "predict_peak_match_within_2": str(pt.get("match_within_2", False)).lower(),
            "predict_accuracy_at_peak": str(round(pt.get("accuracy_at_peak", float("nan")), 3)),
            "metadata_task": r["task"],
            "metadata_model": r["model"],
            "metadata_d_star": r.get("d_star"),
            "metadata_n_levels": r["n_levels"],
            "metadata_fold": "test",
        }
        if r.get("temperature") is not None:
            ex["metadata_temperature"] = r["temperature"]

        comparison_examples.append(ex)

    datasets.append({
        "dataset": "model_comparison_all_series",
        "examples": comparison_examples,
    })

    # --- Dataset 2: aggregate_statistics ---
    agg = aggregate
    mix_mean = agg["per_model_mean_R2"].get("mixture", float("nan"))
    fold_mean = agg["per_model_mean_R2"].get("fold", float("nan"))
    cusp_mean = agg["per_model_mean_R2"].get("cusp", float("nan"))
    ddm_mean = agg["per_model_mean_R2"].get("ddm", float("nan"))

    paired = agg.get("mixture_vs_fold_paired", {})
    r2_imp = paired.get("mean_R2_improvement", float("nan"))
    p_t_val = paired.get("paired_t_pvalue", float("nan"))

    agg_output_parts = []
    agg_output_parts.append(f"Mixture model mean R2={mix_mean:.3f}")
    agg_output_parts.append(f"wins {agg['r2_win_counts'].get('mixture', 0)}/{agg['n_series']} by R2")
    agg_output_parts.append(f"prediction test success={agg['prediction_test_success_rate']:.0%}")
    agg_output = "; ".join(agg_output_parts)

    temp_corr = agg.get("temp_sigma_correlation", {})

    agg_example = {
        "input": f"Aggregate model comparison across {agg['n_series']} series",
        "output": agg_output,
        "predict_mixture_mean_R2": str(round(mix_mean, 4)),
        "predict_fold_mean_R2": str(round(fold_mean, 4)),
        "predict_cusp_mean_R2_variance": str(round(cusp_mean, 4)),
        "predict_ddm_mean_R2_variance": str(round(ddm_mean, 4)),
        "predict_mixture_r2_wins": str(agg["r2_win_counts"].get("mixture", 0)),
        "predict_fold_r2_wins": str(agg["r2_win_counts"].get("fold", 0)),
        "predict_cusp_r2_wins": str(agg["r2_win_counts"].get("cusp", 0)),
        "predict_ddm_r2_wins": str(agg["r2_win_counts"].get("ddm", 0)),
        "predict_mixture_aic_wins": str(agg["aic_win_counts"].get("mixture", 0)),
        "predict_fold_aic_wins": str(agg["aic_win_counts"].get("fold", 0)),
        "predict_cusp_aic_wins": str(agg["aic_win_counts"].get("cusp", 0)),
        "predict_ddm_aic_wins": str(agg["aic_win_counts"].get("ddm", 0)),
        "predict_mixture_vs_fold_R2_improvement": str(round(r2_imp, 4)),
        "predict_prediction_test_success_rate": str(round(agg["prediction_test_success_rate"], 3)),
        "predict_prediction_test_within_2_rate": str(round(agg["prediction_test_within_2_rate"], 3)),
        "predict_paired_t_pvalue": str(round(p_t_val, 6)),
        "predict_cusp_bimodal_fraction": str(round(agg["cusp_bimodal_detected_fraction"], 3)),
        "predict_temp_sigma_spearman": str(round(temp_corr.get("spearman_rho", float("nan")), 4)),
        "predict_temp_sigma_pvalue": str(round(temp_corr.get("p_value", float("nan")), 4)),
        "metadata_n_series": agg["n_series"],
        "metadata_fold": "test",
    }
    datasets.append({
        "dataset": "aggregate_statistics",
        "examples": [agg_example],
    })

    # --- Dataset 3: per_level_fits ---
    per_level_examples = []
    for r in all_results:
        sid = r["series_id"]
        sd = all_series[sid]
        d = sd["levels"]
        var_obs = sd["variance"]
        acc_obs = sd["accuracy"]

        mix_pred = r["mixture"].get("predicted_variance", [])
        cusp_pred = r["cusp"].get("predicted_variance", [])
        ddm_pred = r["ddm"].get("predicted_variance", [])

        # Fold prediction from stored predicted_variance
        fold_pred_arr = np.array(r["fold"].get("predicted_variance", [float("nan")] * len(d)))

        for i, di in enumerate(d):
            mix_v = mix_pred[i] if i < len(mix_pred) else float("nan")
            fold_v = float(fold_pred_arr[i]) if i < len(fold_pred_arr) else float("nan")
            cusp_v = cusp_pred[i] if i < len(cusp_pred) else float("nan")
            ddm_v = ddm_pred[i] if i < len(ddm_pred) else float("nan")

            ex = {
                "input": f"Variance fit at d={int(di)} for {sid}",
                "output": str(round(float(var_obs[i]), 6)),
                "predict_observed_variance": str(round(float(var_obs[i]), 6)),
                "predict_mixture_predicted": str(round(float(mix_v), 6)),
                "predict_fold_predicted": str(round(float(fold_v), 6)),
                "predict_cusp_predicted": str(round(float(cusp_v), 6)),
                "predict_ddm_predicted": str(round(float(ddm_v), 6)),
                "predict_observed_accuracy": str(round(float(acc_obs[i]), 4)),
                "metadata_series": sid,
                "metadata_difficulty_level": int(di),
                "metadata_fold": "test",
            }
            per_level_examples.append(ex)

    datasets.append({
        "dataset": "per_level_fits",
        "examples": per_level_examples,
    })

    return {"datasets": datasets}


# ===========================================================================
# MAIN
# ===========================================================================

@logger.catch
def main():
    t_start = time.time()
    logger.info("=" * 60)
    logger.info("Starting Model Fitting Experiment")
    logger.info("=" * 60)

    # --- Load all data ---
    all_series = load_all_series()
    series_ids = sorted(all_series.keys())
    logger.info(f"Series to analyze: {series_ids}")

    # --- STAGE 1: MINI (2 series) ---
    logger.info("=" * 60)
    logger.info("STAGE 1: MINI (2 series)")
    logger.info("=" * 60)
    mini_ids = []
    # Pick arith__gemini-flash and gc__gpt-4o-mini if available
    for candidate in ["arith__gemini-flash", "gc__gpt-4o-mini"]:
        if candidate in all_series:
            mini_ids.append(candidate)
    if len(mini_ids) < 2:
        mini_ids = series_ids[:2]

    mini_results = []
    for sid in mini_ids:
        try:
            r = analyze_series(sid, all_series[sid])
            mini_results.append(r)
        except Exception:
            logger.exception(f"Failed on mini series {sid}")

    # Validate mini
    for r in mini_results:
        mix_r2 = r["mixture"].get("R2", float("nan"))
        logger.info(f"MINI CHECK {r['series_id']}: mixture_R2={mix_r2:.4f}")
        if np.isnan(mix_r2):
            logger.warning(f"Mixture R2 is NaN for {r['series_id']}!")

    t_mini = time.time() - t_start
    logger.info(f"STAGE 1 completed in {t_mini:.1f}s")

    # --- STAGE 2: MEDIUM (6 series) ---
    logger.info("=" * 60)
    logger.info("STAGE 2: MEDIUM (6 series)")
    logger.info("=" * 60)
    medium_ids = [s for s in series_ids if all_series[s]["task"] in ("arithmetic", "graph_coloring")]

    medium_results = []
    for sid in medium_ids:
        try:
            r = analyze_series(sid, all_series[sid])
            medium_results.append(r)
        except Exception:
            logger.exception(f"Failed on medium series {sid}")

    # Check mixture vs fold
    mix_wins = sum(
        1 for r in medium_results
        if not np.isnan(r["mixture"].get("R2", float("nan")))
        and not np.isnan(r["fold"].get("R2_all", float("nan")))
        and r["mixture"]["R2"] > r["fold"]["R2_all"]
    )
    logger.info(f"STAGE 2: Mixture beats fold in {mix_wins}/{len(medium_results)} series")

    t_medium = time.time() - t_start
    logger.info(f"STAGE 2 completed in {t_medium:.1f}s")

    # --- STAGE 3: FULL (10 series) ---
    logger.info("=" * 60)
    logger.info("STAGE 3: FULL (all series)")
    logger.info("=" * 60)

    all_results = []
    for sid in series_ids:
        try:
            r = analyze_series(sid, all_series[sid])
            all_results.append(r)
        except Exception:
            logger.exception(f"Failed on full series {sid}")

    logger.info(f"Successfully analyzed {len(all_results)}/{len(series_ids)} series")

    # --- Aggregate ---
    logger.info("Computing aggregate statistics...")
    aggregate = compute_aggregate_stats(all_results)
    logger.info(f"Aggregate: mixture_mean_R2={aggregate['per_model_mean_R2'].get('mixture', 'nan'):.4f}")
    logger.info(f"Aggregate: fold_mean_R2={aggregate['per_model_mean_R2'].get('fold', 'nan')}")
    logger.info(f"Aggregate: r2_win_counts={aggregate['r2_win_counts']}")
    logger.info(f"Aggregate: prediction_test_success={aggregate['prediction_test_success_rate']:.0%}")

    paired = aggregate.get("mixture_vs_fold_paired", {})
    if paired:
        logger.info(f"Aggregate: mixture vs fold improvement={paired.get('mean_R2_improvement', 'nan'):.4f}, "
                     f"paired-t p={paired.get('paired_t_pvalue', 'nan')}")

    temp_corr = aggregate.get("temp_sigma_correlation", {})
    if temp_corr:
        logger.info(f"Temp-sigma correlation: rho={temp_corr.get('spearman_rho', 'nan'):.4f}, "
                     f"p={temp_corr.get('p_value', 'nan'):.4f}")

    # --- Format output ---
    logger.info("Formatting output...")
    output = format_output(all_results, all_series, aggregate)

    # --- Write output ---
    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

    t_total = time.time() - t_start
    logger.info(f"Total runtime: {t_total:.1f}s")
    logger.info("=" * 60)
    logger.info("DONE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
