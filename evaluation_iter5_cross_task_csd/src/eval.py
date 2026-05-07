#!/usr/bin/env python3
"""Cross-Task CSD Classifier Transfer Failure Diagnosis & Task-Agnostic Feature Engineering.

Diagnoses why CSD classifier LOTO F1 drops from 0.944 (with d*-dependent features)
to 0.43-0.58 (z-score-only), quantifies feature distribution shift between arithmetic
and graph coloring tasks, tests 5 task-agnostic normalization strategies, measures
few-shot calibration curves, and produces deployment recommendations.

Analyses:
  1 - Feature Distribution Shift Metrics
  2 - Normalized LOTO F1 Scores (5 new normalization strategies)
  3 - Transfer Learning Diagnosis (directional LOTO, per-pair LOPO, few-shot calibration)
  4 - Deployment Recommendation (threshold-based, composite, method ranking)
"""

import gc
import json
import math
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from scipy.integrate import trapezoid
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.svm import SVC

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
WORKSPACE = Path(__file__).resolve().parent
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------------------
# Hardware-aware resource limits
# ---------------------------------------------------------------------------
def _container_ram_gb() -> float | None:
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None

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

NUM_CPUS = _detect_cpus()
TOTAL_RAM_GB = _container_ram_gb() or 16.0
RAM_BUDGET_BYTES = int(TOTAL_RAM_GB * 0.4 * 1e9)  # Conservative for small data
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET_BYTES * 3, RAM_BUDGET_BYTES * 3))
logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, budget {RAM_BUDGET_BYTES/1e9:.1f} GB")

# ---------------------------------------------------------------------------
# Constants and paths
# ---------------------------------------------------------------------------
BASE = Path("/ai-inventor/aii_pipeline/runs/neurips-open/3_invention_loop")
ARITH_CSD = BASE / "iter_2/gen_art/exp_id1_it2__opus/full_method_out.json"
GRAPH_CSD = BASE / "iter_2/gen_art/exp_id3_it2__opus/full_method_out.json"
CLASSIFIER = BASE / "iter_4/gen_art/exp_id2_it4__opus/full_method_out.json"

VALID_PAIRS = [
    ("arithmetic", "meta-llama/llama-3.1-8b-instruct", 20, 24),
    ("arithmetic", "google/gemini-2.0-flash-001", 15, 24),
    ("graph_coloring", "openai/gpt-4o-mini", 10, 20),
    ("graph_coloring", "google/gemini-2.0-flash-001", 14, 20),
    ("graph_coloring", "google/gemini-2.0-flash-lite-001", 11, 20),
]

CSD_FEATURES = [
    "csd_variance", "dip_statistic", "silhouette_k2",
    "bimodality_coefficient", "disagreement_rate",
]


# ========================================================================
# STEP 0 — Data Loading
# ========================================================================

def load_arithmetic_csd(path: Path) -> pd.DataFrame:
    """Load arithmetic CSD indicators from iter_2 full_method_out.json."""
    logger.info(f"Loading arithmetic CSD from {path}")
    data = json.loads(path.read_text())
    rows = []
    for ds in data["datasets"]:
        for ex in ds["examples"]:
            model_name = ex.get("metadata_model", "")
            d_star = ex.get("metadata_d_star")
            # Skip gpt-4o-mini for arithmetic (d*=2, degenerate)
            if d_star is not None and d_star <= 2:
                continue
            rows.append({
                "task_family": "arithmetic",
                "model": model_name,
                "difficulty_level": ex["metadata_difficulty_level"],
                "accuracy": float(ex.get("predict_accuracy", 0)),
                "csd_variance": float(ex.get("predict_csd_variance", 0)),
                "dip_statistic": float(ex.get("predict_dip_statistic", 0)),
                "dip_pvalue": float(ex.get("predict_dip_pvalue", 1)),
                "silhouette_k2": float(ex.get("predict_silhouette_k2", 0)),
                "bimodality_coefficient": float(ex.get("predict_bimodality_coefficient", 0)),
                "disagreement_rate": float(ex.get("predict_disagreement_rate", 0)),
                "d_star": d_star,
            })
    df = pd.DataFrame(rows)
    logger.info(f"  Arithmetic CSD: {len(df)} rows, models={df['model'].unique().tolist()}")
    return df


def load_graph_csd(path: Path) -> pd.DataFrame:
    """Load graph coloring CSD indicators from iter_2 full_method_out.json."""
    logger.info(f"Loading graph coloring CSD from {path}")
    data = json.loads(path.read_text())
    rows = []
    seen = set()
    d_star_map = {}
    if "metadata" in data and "analysis" in data["metadata"]:
        for m_info in data["metadata"]["analysis"].get("models", []):
            d_star_map[m_info["model"]] = m_info["d_star"]
    for ds in data["datasets"]:
        for ex in ds["examples"]:
            model_name = ex.get("metadata_model", "")
            level = ex.get("metadata_difficulty_level")
            key = (model_name, level)
            if key in seen:
                continue
            seen.add(key)
            d_star = d_star_map.get(model_name)
            rows.append({
                "task_family": "graph_coloring",
                "model": model_name,
                "difficulty_level": level,
                "accuracy": float(ex.get("metadata_csd_accuracy", 0)),
                "csd_variance": float(ex.get("metadata_csd_embedding_variance", 0)),
                "dip_statistic": float(ex.get("metadata_csd_dip_statistic", 0)),
                "dip_pvalue": float(ex.get("metadata_csd_dip_pvalue", 1)),
                "silhouette_k2": float(ex.get("metadata_csd_silhouette_score", 0)),
                "bimodality_coefficient": float(ex.get("metadata_csd_bimodality_coefficient", 0)),
                "disagreement_rate": float(ex.get("metadata_csd_disagreement_rate", 0)),
                "d_star": d_star,
            })
    df = pd.DataFrame(rows)
    logger.info(f"  Graph CSD: {len(df)} rows, models={df['model'].unique().tolist()}")
    return df


def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary labels: 'near' if difficulty >= d_star - 2, else 'safe'."""
    df = df.copy()
    df["label"] = df.apply(
        lambda r: "near" if r["difficulty_level"] >= r["d_star"] - 2 else "safe",
        axis=1,
    )
    near_count = (df["label"] == "near").sum()
    safe_count = (df["label"] == "safe").sum()
    logger.info(f"  Labels: near={near_count}, safe={safe_count}, total={len(df)}")
    return df


def load_all_data() -> pd.DataFrame:
    """Load and merge all CSD data into a unified DataFrame."""
    logger.info("=== STEP 0: Loading data ===")
    arith_df = load_arithmetic_csd(ARITH_CSD)
    graph_df = load_graph_csd(GRAPH_CSD)
    df = pd.concat([arith_df, graph_df], ignore_index=True)
    df = create_labels(df)
    for (task, model), grp in df.groupby(["task_family", "model"]):
        d_s = grp["d_star"].iloc[0]
        near = (grp["label"] == "near").sum()
        safe = (grp["label"] == "safe").sum()
        logger.info(f"  {task}__{model}: d*={d_s}, near={near}, safe={safe}, n={len(grp)}")
    return df


# ========================================================================
# Helper: Classifier factories + evaluation utilities
# ========================================================================

def get_classifier_factories() -> dict:
    return {
        "rf": lambda: RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
        "logreg": lambda: LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"),
        "svm": lambda: SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42),
    }


def safe_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, zero_division=0))


def safe_auroc(y_true, y_prob) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return 0.5
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return 0.5


def train_eval_clf(clf_factory, X_train, y_train, X_test, y_test) -> dict:
    """Train a classifier and return metrics."""
    try:
        clf = clf_factory()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else y_pred.astype(float)
        return {
            "f1": safe_f1(y_test, y_pred),
            "auroc": safe_auroc(y_test, y_prob),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        }
    except Exception as e:
        logger.debug(f"Classifier training failed: {e}")
        return {"f1": 0.0, "auroc": 0.5, "precision": 0.0, "recall": 0.0}


def loto_eval(df: pd.DataFrame, features: list, clf_factory) -> dict:
    """Leave-One-Task-Out: train on task A, test on task B, average both directions."""
    y_col = (df["label"] == "near").astype(int)
    scores = []
    directional = {}
    for held_task in ["arithmetic", "graph_coloring"]:
        test_mask = df["task_family"] == held_task
        train_mask = ~test_mask
        if train_mask.sum() < 5 or test_mask.sum() < 2:
            continue
        X_train = df.loc[train_mask, features].values
        y_train = y_col[train_mask].values
        X_test = df.loc[test_mask, features].values
        y_test = y_col[test_mask].values
        if len(np.unique(y_train)) < 2:
            continue
        res = train_eval_clf(clf_factory, X_train, y_train, X_test, y_test)
        scores.append(res)
        train_task = "graph_coloring" if held_task == "arithmetic" else "arithmetic"
        directional[f"{train_task}_to_{held_task}"] = res
    if not scores:
        return {"f1": 0.0, "auroc": 0.5, "directional": directional}
    avg_f1 = np.mean([s["f1"] for s in scores])
    avg_auroc = np.mean([s["auroc"] for s in scores])
    return {"f1": float(avg_f1), "auroc": float(avg_auroc), "directional": directional}


def lopo_eval(df: pd.DataFrame, features: list, clf_factory) -> dict:
    """Leave-One-Pair-Out: hold out each model-task pair in turn."""
    y_col = (df["label"] == "near").astype(int)
    pairs = [(t, m) for t, m, _, _ in VALID_PAIRS]
    scores = []
    per_pair = {}
    for held_task, held_model in pairs:
        test_mask = (df["task_family"] == held_task) & (df["model"] == held_model)
        train_mask = ~test_mask
        if train_mask.sum() < 5 or test_mask.sum() < 2:
            continue
        X_train = df.loc[train_mask, features].values
        y_train = y_col[train_mask].values
        X_test = df.loc[test_mask, features].values
        y_test = y_col[test_mask].values
        if len(np.unique(y_train)) < 2:
            continue
        res = train_eval_clf(clf_factory, X_train, y_train, X_test, y_test)
        scores.append(res)
        pair_key = f"{held_task}__{held_model}"
        per_pair[pair_key] = res
    if not scores:
        return {"f1": 0.0, "auroc": 0.5, "per_pair": per_pair}
    avg_f1 = np.mean([s["f1"] for s in scores])
    avg_auroc = np.mean([s["auroc"] for s in scores])
    return {"f1": float(avg_f1), "auroc": float(avg_auroc), "per_pair": per_pair}


# ========================================================================
# STEP 1 — Feature Distribution Shift
# ========================================================================

def kl_divergence_binned(p_vals: np.ndarray, q_vals: np.ndarray, n_bins: int = 50) -> float:
    """Compute KL(P||Q) using histogram binning with Laplace smoothing."""
    lo = min(p_vals.min(), q_vals.min())
    hi = max(p_vals.max(), q_vals.max())
    if hi == lo:
        return 0.0
    bins = np.linspace(lo - 1e-10, hi + 1e-10, n_bins + 1)
    p_hist = np.histogram(p_vals, bins=bins)[0].astype(float)
    q_hist = np.histogram(q_vals, bins=bins)[0].astype(float)
    # Laplace smoothing
    p_hist = (p_hist + 1) / (p_hist.sum() + n_bins)
    q_hist = (q_hist + 1) / (q_hist.sum() + n_bins)
    return float(np.sum(p_hist * np.log(p_hist / q_hist)))


def overlap_coefficient(p_vals: np.ndarray, q_vals: np.ndarray, n_points: int = 500) -> float:
    """Compute overlap area between two KDE estimates."""
    combined = np.concatenate([p_vals, q_vals])
    lo, hi = combined.min(), combined.max()
    if hi - lo < 1e-12:
        return 1.0
    bw = max((hi - lo) / 20, 1e-8)
    try:
        kde_p = stats.gaussian_kde(p_vals, bw_method=bw / (hi - lo + 1e-8))
        kde_q = stats.gaussian_kde(q_vals, bw_method=bw / (hi - lo + 1e-8))
    except Exception:
        try:
            kde_p = stats.gaussian_kde(p_vals)
            kde_q = stats.gaussian_kde(q_vals)
        except Exception:
            return 0.5
    x_grid = np.linspace(lo - 2 * bw, hi + 2 * bw, n_points)
    overlap_vals = np.minimum(kde_p(x_grid), kde_q(x_grid))
    return float(trapezoid(overlap_vals, x_grid))


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cohen's d standardized mean difference."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    pooled_std = np.sqrt(((na - 1) * a.std(ddof=1)**2 + (nb - 1) * b.std(ddof=1)**2) / (na + nb - 2))
    if pooled_std < 1e-12:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def compute_feature_shift(df: pd.DataFrame) -> dict:
    """ANALYSIS 1: Compute distribution shift metrics for each CSD feature."""
    logger.info("=== ANALYSIS 1: Feature Distribution Shift ===")
    arith_mask = df["task_family"] == "arithmetic"
    graph_mask = df["task_family"] == "graph_coloring"

    results = {}
    ranking = []

    normalization_variants = {
        "raw": lambda feat: feat,
        "z_score": lambda feat: f"{feat}_zt",
        "percentile": lambda feat: f"{feat}_pct",
    }

    for feature in CSD_FEATURES:
        feature_results = {}
        for norm_name, norm_fn in normalization_variants.items():
            col = norm_fn(feature) if norm_name != "raw" else feature
            if col not in df.columns:
                continue
            arith_vals = df.loc[arith_mask, col].dropna().values
            graph_vals = df.loc[graph_mask, col].dropna().values
            if len(arith_vals) < 3 or len(graph_vals) < 3:
                continue

            wd = float(stats.wasserstein_distance(arith_vals, graph_vals))
            ks_stat, ks_pval = stats.ks_2samp(arith_vals, graph_vals)
            cd = cohens_d(arith_vals, graph_vals)
            kl_a2g = kl_divergence_binned(arith_vals, graph_vals)
            kl_g2a = kl_divergence_binned(graph_vals, arith_vals)
            oc = overlap_coefficient(arith_vals, graph_vals)

            entry = {
                "wasserstein_distance": round(wd, 6),
                "ks_statistic": round(float(ks_stat), 6),
                "ks_pvalue": round(float(ks_pval), 6),
                "cohens_d": round(cd, 6),
                "kl_arith_to_graph": round(kl_a2g, 6),
                "kl_graph_to_arith": round(kl_g2a, 6),
                "overlap_coefficient": round(oc, 6),
                "arith_mean": round(float(arith_vals.mean()), 6),
                "arith_std": round(float(arith_vals.std()), 6),
                "graph_mean": round(float(graph_vals.mean()), 6),
                "graph_std": round(float(graph_vals.std()), 6),
            }
            feature_results[norm_name] = entry
            if norm_name == "raw":
                ranking.append({"feature": feature, "wasserstein_distance": wd, "cohens_d": abs(cd)})

        results[feature] = feature_results
        logger.info(f"  {feature}: WD(raw)={feature_results.get('raw',{}).get('wasserstein_distance','N/A')}, "
                     f"Cohen's d={feature_results.get('raw',{}).get('cohens_d','N/A')}")

    ranking.sort(key=lambda x: x["wasserstein_distance"], reverse=True)
    top3_worst = [r["feature"] for r in ranking[:3]]
    logger.info(f"  Top-3 worst offenders (by Wasserstein): {top3_worst}")

    return {
        "per_feature": results,
        "feature_shift_ranking": ranking,
        "top3_worst_offenders": top3_worst,
    }


# ========================================================================
# STEP 2 — Normalization Strategies
# ========================================================================

def apply_strategy_a_rank_slope(df: pd.DataFrame) -> list:
    """Strategy A: Rank-Transform + Slope of last 3 ranks. Modifies df in-place."""
    feature_cols = []
    for feat in CSD_FEATURES:
        rank_col = f"{feat}_rank_a"
        slope_col = f"{feat}_slope_a"
        df[rank_col] = 0.0
        df[slope_col] = 0.0
        for (task, model), grp_idx in df.groupby(["task_family", "model"]).groups.items():
            subset = df.loc[grp_idx].sort_values("difficulty_level")
            n = len(subset)
            ranks = subset[feat].rank(pct=True).values
            df.loc[subset.index, rank_col] = ranks
            slopes = np.zeros(n)
            for i in range(n):
                window = ranks[max(0, i-2):i+1]
                if len(window) >= 2:
                    x = np.arange(len(window), dtype=float)
                    slopes[i] = np.polyfit(x, window, 1)[0]
            df.loc[subset.index, slope_col] = slopes
        feature_cols.extend([rank_col, slope_col])
    return feature_cols


def apply_strategy_b_running_zscore(df: pd.DataFrame) -> list:
    """Strategy B: Running (cumulative) Z-Score. Modifies df in-place."""
    feature_cols = []
    for feat in CSD_FEATURES:
        col = f"{feat}_runz_b"
        feature_cols.append(col)
        df[col] = 0.0
        for (task, model), grp_idx in df.groupby(["task_family", "model"]).groups.items():
            subset = df.loc[grp_idx].sort_values("difficulty_level")
            vals = subset[feat].values
            runz = np.zeros(len(vals))
            for i in range(len(vals)):
                window = vals[:i+1]
                if len(window) < 2:
                    runz[i] = 0.0
                else:
                    mu = window.mean()
                    sigma = window.std()
                    if sigma < 1e-12:
                        runz[i] = 0.0
                    else:
                        runz[i] = (vals[i] - mu) / sigma
            df.loc[subset.index, col] = runz
    return feature_cols


def apply_strategy_c_dimensionless_ratios(df: pd.DataFrame) -> list:
    """Strategy C: Dimensionless Ratios (fold-change relative to baseline + ratio to max). Modifies df in-place."""
    feature_cols = []
    eps = 1e-8
    for feat in CSD_FEATURES:
        base_col = f"{feat}_ratio_base_c"
        max_col = f"{feat}_ratio_max_c"
        feature_cols.extend([base_col, max_col])
        df[base_col] = 1.0
        df[max_col] = 0.0
        for (task, model), grp_idx in df.groupby(["task_family", "model"]).groups.items():
            subset = df.loc[grp_idx].sort_values("difficulty_level")
            vals = subset[feat].values
            baseline = vals[0] if len(vals) > 0 else eps
            max_val = vals.max() if len(vals) > 0 else eps
            df.loc[subset.index, base_col] = vals / (abs(baseline) + eps)
            df.loc[subset.index, max_col] = vals / (abs(max_val) + eps)
    return feature_cols


def apply_strategy_d_trend_derivative(df: pd.DataFrame) -> list:
    """Strategy D: Trend/Derivative Features (delta, slope of last 3, acceleration). Modifies df in-place."""
    feature_cols = []
    for feat in CSD_FEATURES:
        delta_col = f"{feat}_delta_d"
        slope_col = f"{feat}_slope3_d"
        accel_col = f"{feat}_accel_d"
        feature_cols.extend([delta_col, slope_col, accel_col])
        df[delta_col] = 0.0
        df[slope_col] = 0.0
        df[accel_col] = 0.0
        for (task, model), grp_idx in df.groupby(["task_family", "model"]).groups.items():
            subset = df.loc[grp_idx].sort_values("difficulty_level")
            vals = subset[feat].values
            n = len(vals)
            deltas = np.zeros(n)
            slopes = np.zeros(n)
            accels = np.zeros(n)
            for i in range(n):
                if i > 0:
                    deltas[i] = vals[i] - vals[i-1]
                window = vals[max(0, i-2):i+1]
                if len(window) >= 2:
                    x = np.arange(len(window), dtype=float)
                    slopes[i] = np.polyfit(x, window, 1)[0]
                if i >= 2:
                    accels[i] = deltas[i] - deltas[i-1]
            df.loc[subset.index, delta_col] = deltas
            df.loc[subset.index, slope_col] = slopes
            df.loc[subset.index, accel_col] = accels
    return feature_cols


def apply_strategy_e_binary_indicators(df: pd.DataFrame) -> list:
    """Strategy E: Binary Indicators using ecologically-motivated thresholds. Modifies df in-place."""
    feature_cols = []

    # is_dip_significant: dip_pvalue < 0.05
    col = "is_dip_significant_e"
    df[col] = (df["dip_pvalue"] < 0.05).astype(float)
    feature_cols.append(col)

    # is_silhouette_high: silhouette_k2 > 0.3
    col = "is_silhouette_high_e"
    df[col] = (df["silhouette_k2"] > 0.3).astype(float)
    feature_cols.append(col)

    # is_bimodality_above_threshold: bimodality_coefficient > 0.555
    col = "is_bimodality_above_e"
    df[col] = (df["bimodality_coefficient"] > 0.555).astype(float)
    feature_cols.append(col)

    # is_disagreement_high: disagreement_rate > 0.5
    col = "is_disagreement_high_e"
    df[col] = (df["disagreement_rate"] > 0.5).astype(float)
    feature_cols.append(col)

    # is_variance_increasing: within-pair variance delta > 0
    col = "is_variance_increasing_e"
    df[col] = 0.0
    for (task, model), grp_idx in df.groupby(["task_family", "model"]).groups.items():
        subset = df.loc[grp_idx].sort_values("difficulty_level")
        vals = subset["csd_variance"].values
        deltas = np.diff(vals, prepend=vals[0])
        df.loc[subset.index, col] = (deltas > 0).astype(float)
    feature_cols.append(col)

    return feature_cols


def apply_within_task_zscore(df: pd.DataFrame) -> list:
    """Apply within-task z-score normalization (baseline from iter_4)."""
    feature_cols = []
    for feat in CSD_FEATURES:
        col = f"{feat}_zt"
        feature_cols.append(col)
        for task in ["arithmetic", "graph_coloring"]:
            mask = df["task_family"] == task
            if mask.sum() == 0:
                continue
            mu = df.loc[mask, feat].mean()
            sigma = df.loc[mask, feat].std()
            df.loc[mask, col] = (df.loc[mask, feat] - mu) / (sigma + 1e-8)
    return feature_cols


def apply_percentile_rank(df: pd.DataFrame) -> list:
    """Apply within-task percentile rank normalization."""
    feature_cols = []
    for feat in CSD_FEATURES:
        col = f"{feat}_pct"
        feature_cols.append(col)
        for task in ["arithmetic", "graph_coloring"]:
            mask = df["task_family"] == task
            if mask.sum() == 0:
                continue
            df.loc[mask, col] = df.loc[mask, feat].rank(pct=True)
    return feature_cols


def apply_relative_difficulty(df: pd.DataFrame) -> None:
    """Add relative_difficulty and relative_dist_to_dstar columns."""
    for task in ["arithmetic", "graph_coloring"]:
        mask = df["task_family"] == task
        if mask.sum() == 0:
            continue
        d_min = df.loc[mask, "difficulty_level"].min()
        d_max = df.loc[mask, "difficulty_level"].max()
        df.loc[mask, "relative_difficulty"] = (
            (df.loc[mask, "difficulty_level"] - d_min) / max(d_max - d_min, 1)
        )
        for model in df.loc[mask, "model"].unique():
            m2 = mask & (df["model"] == model)
            if m2.sum() == 0:
                continue
            d_star = df.loc[m2, "d_star"].iloc[0]
            df.loc[m2, "relative_dist_to_dstar"] = (
                (d_star - df.loc[m2, "difficulty_level"]) / max(d_star - d_min, 1)
            )


def run_analysis_2(df: pd.DataFrame) -> dict:
    """ANALYSIS 2: Test 5 normalization strategies with 3 classifiers."""
    logger.info("=== ANALYSIS 2: Normalization Strategies ===")

    # Apply baseline normalizations first
    zt_feats = apply_within_task_zscore(df)
    pct_feats = apply_percentile_rank(df)
    apply_relative_difficulty(df)

    # Apply 5 new strategies
    strat_a_feats = apply_strategy_a_rank_slope(df)
    strat_b_feats = apply_strategy_b_running_zscore(df)
    strat_c_feats = apply_strategy_c_dimensionless_ratios(df)
    strat_d_feats = apply_strategy_d_trend_derivative(df)
    strat_e_feats = apply_strategy_e_binary_indicators(df)

    # New strategies alone
    new_strategy_names = ["A_rank_slope", "B_running_zscore", "C_dimensionless_ratios",
                          "D_trend_derivative", "E_binary_indicators"]
    new_strategy_feats = {
        "A_rank_slope": strat_a_feats,
        "B_running_zscore": strat_b_feats,
        "C_dimensionless_ratios": strat_c_feats,
        "D_trend_derivative": strat_d_feats,
        "E_binary_indicators": strat_e_feats,
    }
    strategies = dict(new_strategy_feats)
    # Combined: new strategy + relative_difficulty (task-agnostic, no d*)
    for name, feats in new_strategy_feats.items():
        strategies[f"{name}_reldiff"] = feats + ["relative_difficulty"]
    # Baselines for reference
    strategies["baseline_zt"] = zt_feats
    strategies["baseline_zt_reldist"] = zt_feats + ["relative_dist_to_dstar"]
    strategies["baseline_zt_reldiff"] = zt_feats + ["relative_difficulty"]

    clf_factories = get_classifier_factories()
    results = {}
    baseline_loto_f1 = 0.448  # csd_zt_rf LOTO from metadata

    for strat_name, features in strategies.items():
        # Check all features exist
        missing = [f for f in features if f not in df.columns]
        if missing:
            logger.warning(f"  Skipping {strat_name}: missing {missing}")
            continue

        feat_df = df[features]
        if feat_df.isna().any().any():
            # Fill NaN with 0
            df[features] = df[features].fillna(0)

        strat_results = {}
        for clf_name, clf_factory in clf_factories.items():
            key = f"{strat_name}_{clf_name}"
            loto_res = loto_eval(df, features, clf_factory)
            lopo_res = lopo_eval(df, features, clf_factory)
            delta_f1 = loto_res["f1"] - baseline_loto_f1

            strat_results[clf_name] = {
                "loto_f1": round(loto_res["f1"], 6),
                "loto_auroc": round(loto_res["auroc"], 6),
                "lopo_f1": round(lopo_res["f1"], 6),
                "lopo_auroc": round(lopo_res["auroc"], 6),
                "delta_vs_baseline": round(delta_f1, 6),
            }
            logger.info(f"  {key}: LOTO_F1={loto_res['f1']:.3f}, LOPO_F1={lopo_res['f1']:.3f}, delta={delta_f1:+.3f}")

        results[strat_name] = {
            "features": features,
            "n_features": len(features),
            "classifiers": strat_results,
        }

    # Find best NEW normalization (only A-E strategies and their _reldiff combos)
    best_new = None
    best_new_f1 = 0.0
    best_new_key = ""
    for strat_name, strat_data in results.items():
        # Only consider new strategies (A-E), not baselines
        is_new = any(strat_name.startswith(n) for n in new_strategy_names)
        if not is_new:
            continue
        for clf_name, clf_data in strat_data["classifiers"].items():
            if clf_data["loto_f1"] > best_new_f1:
                best_new_f1 = clf_data["loto_f1"]
                best_new_key = f"{strat_name}_{clf_name}"
                best_new = (strat_name, clf_name)

    logger.info(f"  Best new normalization: {best_new_key} with LOTO_F1={best_new_f1:.4f}")

    return {
        "strategies": results,
        "best_new_normalization": best_new_key,
        "best_new_loto_f1": round(best_new_f1, 6),
        "baseline_loto_f1_zt_rf": baseline_loto_f1,
    }


# ========================================================================
# STEP 3 — Transfer Learning Diagnosis
# ========================================================================

def run_analysis_3(df: pd.DataFrame, best_strat_name: str, best_clf_name: str,
                   all_strategies: dict) -> dict:
    """ANALYSIS 3: Transfer learning diagnosis."""
    logger.info("=== ANALYSIS 3: Transfer Learning Diagnosis ===")
    clf_factories = get_classifier_factories()

    # 3a: Directional LOTO
    logger.info("  3a: Directional LOTO")
    directional_results = {}

    # Get features for best new normalization + reference
    configs_to_test = {}
    if best_strat_name in all_strategies:
        configs_to_test["best_new"] = all_strategies[best_strat_name]["features"]

    # Also include the reference with d* and z-score only baseline
    zt_feats = [f"{f}_zt" for f in CSD_FEATURES]
    if "relative_dist_to_dstar" in df.columns:
        configs_to_test["csd_zt_reldist"] = zt_feats + ["relative_dist_to_dstar"]
    configs_to_test["csd_zt"] = zt_feats

    for config_name, features in configs_to_test.items():
        missing = [f for f in features if f not in df.columns]
        if missing:
            continue
        for clf_name, clf_factory in clf_factories.items():
            key = f"{config_name}_{clf_name}"
            loto_res = loto_eval(df, features, clf_factory)
            directional_results[key] = {
                "mean_loto_f1": round(loto_res["f1"], 6),
                "mean_loto_auroc": round(loto_res["auroc"], 6),
                "directional": {k: {m: round(v, 6) for m, v in d.items()}
                               for k, d in loto_res.get("directional", {}).items()},
            }
            dirs = loto_res.get("directional", {})
            for dir_name, dir_data in dirs.items():
                logger.info(f"    {key} {dir_name}: F1={dir_data.get('f1', 'N/A'):.3f}")

    # 3b: Per-Pair LOPO Matrix
    logger.info("  3b: Per-Pair LOPO Matrix")
    per_pair_results = {}
    best_features = configs_to_test.get("best_new", zt_feats)
    best_factory = clf_factories.get(best_clf_name, clf_factories["rf"])

    y_col = (df["label"] == "near").astype(int)
    for held_task, held_model, _, _ in VALID_PAIRS:
        pair_key = f"{held_task}__{held_model}"
        test_mask = (df["task_family"] == held_task) & (df["model"] == held_model)
        train_mask = ~test_mask
        if train_mask.sum() < 5 or test_mask.sum() < 2:
            continue
        X_train = df.loc[train_mask, best_features].values
        y_train = y_col[train_mask].values
        X_test = df.loc[test_mask, best_features].values
        y_test = y_col[test_mask].values
        if len(np.unique(y_train)) < 2:
            continue
        res = train_eval_clf(best_factory, X_train, y_train, X_test, y_test)
        # Determine if cross-task
        train_tasks = set(df.loc[train_mask, "task_family"].unique())
        is_cross_task = held_task not in train_tasks or len(train_tasks) > 1
        per_pair_results[pair_key] = {
            "f1": round(res["f1"], 6),
            "auroc": round(res["auroc"], 6),
            "is_cross_task": is_cross_task,
            "n_test": int(test_mask.sum()),
        }
        logger.info(f"    {pair_key}: F1={res['f1']:.3f}")

    # 3c: Few-Shot Calibration Curve
    logger.info("  3c: Few-Shot Calibration Curve")
    calibration_results = {}
    k_values = [1, 3, 5, 10]
    n_seeds = 20

    # Test with best new features + baseline z-score
    for config_name, features in [("best_new", best_features), ("csd_zt", zt_feats)]:
        missing = [f for f in features if f not in df.columns]
        if missing:
            continue
        config_results = {}
        for k in k_values:
            seed_f1s = []
            for seed in range(n_seeds):
                rng = np.random.RandomState(seed)
                fold_f1s = []
                for held_task in ["arithmetic", "graph_coloring"]:
                    test_mask = df["task_family"] == held_task
                    train_mask = ~test_mask
                    if train_mask.sum() < 5 or test_mask.sum() < 2:
                        continue
                    # Add k calibration samples from target task
                    target_indices = df.index[test_mask].tolist()
                    n_cal = min(k, len(target_indices) - 1)
                    if n_cal < 1:
                        continue
                    cal_indices = rng.choice(target_indices, size=n_cal, replace=False).tolist()
                    test_indices = [idx for idx in target_indices if idx not in cal_indices]
                    if len(test_indices) < 2:
                        continue

                    train_indices = df.index[train_mask].tolist() + cal_indices
                    X_train = df.loc[train_indices, features].values
                    y_train = y_col[train_indices].values
                    X_test = df.loc[test_indices, features].values
                    y_test = y_col[test_indices].values
                    if len(np.unique(y_train)) < 2:
                        continue

                    res = train_eval_clf(best_factory, X_train, y_train, X_test, y_test)
                    fold_f1s.append(res["f1"])

                if fold_f1s:
                    seed_f1s.append(np.mean(fold_f1s))

            if seed_f1s:
                mean_f1 = float(np.mean(seed_f1s))
                std_f1 = float(np.std(seed_f1s))
                ci_95 = 1.96 * std_f1 / max(np.sqrt(len(seed_f1s)), 1)
                config_results[str(k)] = {
                    "mean_f1": round(mean_f1, 6),
                    "std_f1": round(std_f1, 6),
                    "ci_95_lower": round(mean_f1 - ci_95, 6),
                    "ci_95_upper": round(mean_f1 + ci_95, 6),
                    "n_seeds": len(seed_f1s),
                }
                logger.info(f"    {config_name} k={k}: F1={mean_f1:.3f} +/- {ci_95:.3f}")

        calibration_results[config_name] = config_results

    # Find minimum k for F1 >= 0.8
    min_k_for_08 = None
    for k in k_values:
        best_cal = calibration_results.get("best_new", {}).get(str(k), {})
        if best_cal.get("mean_f1", 0) >= 0.8:
            min_k_for_08 = k
            break
    if min_k_for_08 is None:
        min_k_for_08 = -1  # Not achievable
    logger.info(f"  Minimum k for F1>=0.8: {min_k_for_08}")

    return {
        "directional_loto": directional_results,
        "per_pair_lopo_matrix": per_pair_results,
        "calibration_curve": calibration_results,
        "min_k_for_f1_ge_08": min_k_for_08,
    }


# ========================================================================
# STEP 4 — Deployment Recommendation
# ========================================================================

def run_analysis_4(df: pd.DataFrame, analysis_2: dict, analysis_3: dict) -> dict:
    """ANALYSIS 4: Deployment recommendations."""
    logger.info("=== ANALYSIS 4: Deployment Recommendation ===")
    y_col = (df["label"] == "near").astype(int)

    # 4a: Threshold-Based Detection
    logger.info("  4a: Threshold-Based Detection")
    threshold_results = {}
    for feature in CSD_FEATURES:
        if feature not in df.columns:
            continue
        # LOTO threshold: optimize on one task, test on other
        fold_f1s = []
        for held_task in ["arithmetic", "graph_coloring"]:
            test_mask = df["task_family"] == held_task
            train_mask = ~test_mask
            train_vals = df.loc[train_mask, feature].values
            train_y = y_col[train_mask].values
            test_vals = df.loc[test_mask, feature].values
            test_y = y_col[test_mask].values

            # Find optimal threshold on train data
            best_thresh = 0
            best_f1 = 0
            best_dir = 1  # Initialize to avoid unbound variable
            # Test both directions (feature > thresh => near, or feature < thresh => near)
            for direction in [1, -1]:
                thresholds = np.percentile(train_vals, np.arange(5, 100, 5))
                for thresh in thresholds:
                    if direction == 1:
                        pred = (train_vals > thresh).astype(int)
                    else:
                        pred = (train_vals < thresh).astype(int)
                    f1_val = safe_f1(train_y, pred)
                    if f1_val > best_f1:
                        best_f1 = f1_val
                        best_thresh = thresh
                        best_dir = direction

            # Apply to test
            if best_dir == 1:
                test_pred = (test_vals > best_thresh).astype(int)
            else:
                test_pred = (test_vals < best_thresh).astype(int)
            fold_f1s.append(safe_f1(test_y, test_pred))

        avg_f1 = float(np.mean(fold_f1s)) if fold_f1s else 0.0
        threshold_results[feature] = {
            "loto_f1": round(avg_f1, 6),
            "best_threshold": round(float(best_thresh), 6),
            "direction": "above" if best_dir == 1 else "below",
        }
        logger.info(f"    {feature}: LOTO_F1={avg_f1:.3f}")

    # Find best single feature
    best_single_feat = max(threshold_results.items(), key=lambda x: x[1]["loto_f1"])
    logger.info(f"  Best single feature: {best_single_feat[0]} (F1={best_single_feat[1]['loto_f1']:.3f})")

    # 4b: Composite Threshold (majority vote of top-3)
    logger.info("  4b: Composite Threshold")
    top3_features = sorted(threshold_results.items(), key=lambda x: x[1]["loto_f1"], reverse=True)[:3]
    top3_names = [f[0] for f in top3_features]

    composite_fold_f1s = []
    for held_task in ["arithmetic", "graph_coloring"]:
        test_mask = df["task_family"] == held_task
        train_mask = ~test_mask
        train_y = y_col[train_mask].values
        test_y = y_col[test_mask].values

        # Optimize each feature threshold on training data
        feature_preds_test = []
        for feat_name in top3_names:
            train_vals = df.loc[train_mask, feat_name].values
            test_vals = df.loc[test_mask, feat_name].values
            best_thresh, best_f1, best_dir = 0, 0, 1
            for direction in [1, -1]:
                thresholds = np.percentile(train_vals, np.arange(5, 100, 5))
                for thresh in thresholds:
                    pred = (train_vals > thresh).astype(int) if direction == 1 else (train_vals < thresh).astype(int)
                    f1_val = safe_f1(train_y, pred)
                    if f1_val > best_f1:
                        best_f1 = f1_val
                        best_thresh = thresh
                        best_dir = direction
            test_pred = (test_vals > best_thresh).astype(int) if best_dir == 1 else (test_vals < best_thresh).astype(int)
            feature_preds_test.append(test_pred)

        # Majority vote: classify as "near" if >= 2 of 3 agree
        votes = np.array(feature_preds_test)
        composite_pred = (votes.sum(axis=0) >= 2).astype(int)
        composite_fold_f1s.append(safe_f1(test_y, composite_pred))

    composite_f1 = float(np.mean(composite_fold_f1s)) if composite_fold_f1s else 0.0
    logger.info(f"  Composite (top-3 majority vote): LOTO_F1={composite_f1:.3f}")

    # 4c: Method Ranking Table
    logger.info("  4c: Method Ranking Table")

    # Get calibration result for k=5 if available
    cal_5 = analysis_3.get("calibration_curve", {}).get("best_new", {}).get("5", {})
    cal_5_f1 = cal_5.get("mean_f1", 0.0)

    method_ranking = [
        {
            "method": "csd_zt_reldist_rf (best existing)",
            "loto_f1": 0.944,
            "requires_d_star": True,
            "requires_calibration": False,
            "extra_api_cost": 0.0,
        },
        {
            "method": "csd_zt_reldiff_rf (existing, no d*)",
            "loto_f1": 0.860,
            "requires_d_star": False,
            "requires_calibration": False,
            "extra_api_cost": 0.0,
        },
        {
            "method": f"Best new normalization ({analysis_2.get('best_new_normalization', 'N/A')})",
            "loto_f1": round(analysis_2.get("best_new_loto_f1", 0.0), 4),
            "requires_d_star": False,
            "requires_calibration": False,
            "extra_api_cost": 0.0,
        },
        {
            "method": "Best new + 5-shot calibration",
            "loto_f1": round(cal_5_f1, 4),
            "requires_d_star": False,
            "requires_calibration": True,
            "calibration_samples": 5,
            "extra_api_cost": 0.0,
        },
        {
            "method": f"Threshold-based ({best_single_feat[0]})",
            "loto_f1": round(best_single_feat[1]["loto_f1"], 4),
            "requires_d_star": False,
            "requires_calibration": False,
            "extra_api_cost": 0.0,
        },
        {
            "method": "Composite threshold (top-3 vote)",
            "loto_f1": round(composite_f1, 4),
            "requires_d_star": False,
            "requires_calibration": False,
            "extra_api_cost": 0.0,
        },
        {
            "method": "SPUQ baseline",
            "loto_f1": 0.699,
            "requires_d_star": False,
            "requires_calibration": False,
            "extra_api_cost": 0.24,
        },
    ]

    # Sort by LOTO F1
    method_ranking.sort(key=lambda x: x["loto_f1"], reverse=True)
    for i, m in enumerate(method_ranking):
        logger.info(f"    #{i+1} {m['method']}: LOTO_F1={m['loto_f1']:.3f}, d*={m['requires_d_star']}")

    # 4d: Written Deployment Recommendation
    best_no_dstar = max([m for m in method_ranking if not m["requires_d_star"]],
                        key=lambda x: x["loto_f1"])
    recommendation = (
        f"For deployment without knowing d*: Use {best_no_dstar['method']} "
        f"(LOTO F1={best_no_dstar['loto_f1']:.3f}). "
        f"If 5 labeled target-task samples are available, few-shot calibration improves "
        f"LOTO F1 to {cal_5_f1:.3f}. "
        f"The composite threshold detector provides a simple rule-based alternative "
        f"(F1={composite_f1:.3f}) requiring no ML training. "
        f"SPUQ is not recommended as it costs $0.24 per evaluation with lower F1. "
        f"If d* is known (e.g., from a pilot study), csd_zt_reldist_rf achieves 0.944 LOTO F1."
    )
    logger.info(f"  Recommendation: {recommendation[:200]}...")

    return {
        "threshold_detection": threshold_results,
        "best_single_feature": {
            "feature": best_single_feat[0],
            "loto_f1": round(best_single_feat[1]["loto_f1"], 6),
        },
        "composite_threshold": {
            "features_used": top3_names,
            "loto_f1": round(composite_f1, 6),
        },
        "method_ranking": method_ranking,
        "deployment_recommendation": recommendation,
    }


# ========================================================================
# STEP 5 — Output Generation
# ========================================================================

def build_output(
    df: pd.DataFrame,
    analysis_1: dict,
    analysis_2: dict,
    analysis_3: dict,
    analysis_4: dict,
) -> dict:
    """Build the eval_out.json following exp_eval_sol_out schema."""
    logger.info("=== STEP 5: Building output ===")

    # Compute aggregate metrics
    best_new_f1 = analysis_2.get("best_new_loto_f1", 0.0)
    baseline_f1 = analysis_2.get("baseline_loto_f1_zt_rf", 0.448)
    composite_f1 = analysis_4["composite_threshold"]["loto_f1"]
    min_k = analysis_3.get("min_k_for_f1_ge_08", -1)
    best_single_f1 = analysis_4["best_single_feature"]["loto_f1"]

    # Average Wasserstein distance across features
    rankings = analysis_1.get("feature_shift_ranking", [])
    avg_wasserstein = np.mean([r["wasserstein_distance"] for r in rankings]) if rankings else 0.0

    metrics_agg = {
        "best_new_loto_f1": round(best_new_f1, 6),
        "baseline_loto_f1_zt_rf": round(baseline_f1, 6),
        "delta_best_new_vs_baseline": round(best_new_f1 - baseline_f1, 6),
        "best_existing_loto_f1_reldist_rf": 0.944,
        "composite_threshold_loto_f1": round(composite_f1, 6),
        "best_single_threshold_loto_f1": round(best_single_f1, 6),
        "min_calibration_k_for_f1_08": float(min_k),
        "n_model_task_pairs": 5,
        "n_total_rows": len(df),
        "avg_wasserstein_distance": round(float(avg_wasserstein), 6),
        "n_normalization_strategies_tested": 5,
        "n_classifier_types_tested": 3,
    }

    # Build datasets
    datasets = []

    # Dataset 1: Feature Shift Metrics
    ds1_examples = []
    for feat in CSD_FEATURES:
        feat_data = analysis_1.get("per_feature", {}).get(feat, {})
        raw = feat_data.get("raw", {})
        ds1_examples.append({
            "input": f"Feature distribution shift for {feat} between arithmetic and graph_coloring",
            "output": f"Wasserstein={raw.get('wasserstein_distance','N/A')}, Cohen_d={raw.get('cohens_d','N/A')}",
            "predict_wasserstein_distance": str(raw.get("wasserstein_distance", 0)),
            "predict_ks_statistic": str(raw.get("ks_statistic", 0)),
            "predict_ks_pvalue": str(raw.get("ks_pvalue", 1)),
            "predict_cohens_d": str(raw.get("cohens_d", 0)),
            "predict_kl_arith_to_graph": str(raw.get("kl_arith_to_graph", 0)),
            "predict_kl_graph_to_arith": str(raw.get("kl_graph_to_arith", 0)),
            "predict_overlap_coefficient": str(raw.get("overlap_coefficient", 0)),
            "eval_wasserstein_distance": raw.get("wasserstein_distance", 0),
            "eval_cohens_d_abs": abs(raw.get("cohens_d", 0)),
            "eval_ks_statistic": raw.get("ks_statistic", 0),
            "metadata_feature": feat,
            "metadata_normalization": "raw",
            "metadata_fold": "test",
        })
    datasets.append({"dataset": "feature_shift_metrics", "examples": ds1_examples})

    # Dataset 2: Normalization Comparison
    ds2_examples = []
    for strat_name, strat_data in analysis_2.get("strategies", {}).items():
        for clf_name, clf_data in strat_data.get("classifiers", {}).items():
            ds2_examples.append({
                "input": f"Normalization: {strat_name}, Classifier: {clf_name}",
                "output": f"LOTO_F1={clf_data['loto_f1']:.4f}, LOPO_F1={clf_data['lopo_f1']:.4f}",
                "predict_loto_f1": str(round(clf_data["loto_f1"], 6)),
                "predict_loto_auroc": str(round(clf_data["loto_auroc"], 6)),
                "predict_lopo_f1": str(round(clf_data["lopo_f1"], 6)),
                "predict_lopo_auroc": str(round(clf_data["lopo_auroc"], 6)),
                "predict_delta_vs_baseline": str(round(clf_data["delta_vs_baseline"], 6)),
                "eval_loto_f1": clf_data["loto_f1"],
                "eval_lopo_f1": clf_data["lopo_f1"],
                "eval_delta_vs_baseline": clf_data["delta_vs_baseline"],
                "metadata_strategy": strat_name,
                "metadata_classifier": clf_name,
                "metadata_n_features": strat_data["n_features"],
                "metadata_fold": "test",
            })
    datasets.append({"dataset": "normalization_comparison", "examples": ds2_examples})

    # Dataset 3: Calibration Curve
    ds3_examples = []
    for config_name, config_data in analysis_3.get("calibration_curve", {}).items():
        for k_str, k_data in config_data.items():
            ds3_examples.append({
                "input": f"Few-shot calibration: config={config_name}, k={k_str} target-task samples",
                "output": f"Mean LOTO F1={k_data['mean_f1']:.4f} +/- {k_data.get('std_f1',0):.4f}",
                "predict_mean_f1": str(round(k_data["mean_f1"], 6)),
                "predict_ci_95_lower": str(round(k_data["ci_95_lower"], 6)),
                "predict_ci_95_upper": str(round(k_data["ci_95_upper"], 6)),
                "eval_mean_f1": k_data["mean_f1"],
                "eval_ci_width": round(k_data["ci_95_upper"] - k_data["ci_95_lower"], 6),
                "metadata_config": config_name,
                "metadata_k": int(k_str),
                "metadata_n_seeds": k_data.get("n_seeds", 20),
                "metadata_fold": "test",
            })
    if not ds3_examples:
        ds3_examples.append({
            "input": "No calibration data available",
            "output": "N/A",
            "metadata_fold": "test",
        })
    datasets.append({"dataset": "calibration_curve", "examples": ds3_examples})

    # Dataset 4: Deployment Recommendation
    ds4_examples = []
    for method_data in analysis_4.get("method_ranking", []):
        ds4_examples.append({
            "input": f"Method: {method_data['method']}",
            "output": f"LOTO_F1={method_data['loto_f1']:.4f}, requires_d*={method_data['requires_d_star']}",
            "predict_loto_f1": str(round(method_data["loto_f1"], 6)),
            "predict_requires_d_star": str(method_data["requires_d_star"]),
            "predict_requires_calibration": str(method_data.get("requires_calibration", False)),
            "predict_extra_api_cost": str(method_data.get("extra_api_cost", 0.0)),
            "eval_loto_f1": method_data["loto_f1"],
            "metadata_method": method_data["method"],
            "metadata_fold": "test",
        })
    datasets.append({"dataset": "deployment_recommendation", "examples": ds4_examples})

    output = {
        "metadata": {
            "evaluation_name": "cross_task_transfer_diagnosis",
            "n_model_task_pairs": 5,
            "n_total_rows": len(df),
            "tasks": ["arithmetic", "graph_coloring"],
            "csd_features": CSD_FEATURES,
            "baseline_loto_f1_zt_rf": 0.448,
            "best_loto_f1_reldist_rf": 0.944,
            "analysis_1_feature_shift": analysis_1,
            "analysis_2_normalization_loto": {
                "best_new_normalization": analysis_2.get("best_new_normalization"),
                "best_new_loto_f1": analysis_2.get("best_new_loto_f1"),
            },
            "analysis_3_transfer_diagnosis": {
                "min_k_for_f1_ge_08": analysis_3.get("min_k_for_f1_ge_08"),
                "directional_loto_summary": {
                    k: v.get("mean_loto_f1", 0)
                    for k, v in analysis_3.get("directional_loto", {}).items()
                },
            },
            "analysis_4_deployment_recommendation": {
                "best_no_dstar_method": next(
                    (m["method"] for m in analysis_4.get("method_ranking", [])
                     if not m.get("requires_d_star", True)),
                    "N/A",
                ),
                "composite_threshold_f1": analysis_4.get("composite_threshold", {}).get("loto_f1", 0),
                "recommendation": analysis_4.get("deployment_recommendation", ""),
            },
        },
        "metrics_agg": metrics_agg,
        "datasets": datasets,
    }

    return output


# ========================================================================
# Main
# ========================================================================

@logger.catch
def main():
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("Cross-Task CSD Transfer Failure Diagnosis — Starting")
    logger.info("=" * 70)

    # STEP 0: Load data
    df = load_all_data()
    logger.info(f"Loaded {len(df)} total rows in {time.time()-t0:.1f}s")

    # STEP 1: Feature Distribution Shift
    t1 = time.time()
    analysis_1 = compute_feature_shift(df)
    logger.info(f"Analysis 1 complete in {time.time()-t1:.1f}s")

    # STEP 2: Normalization Strategies
    t2 = time.time()
    analysis_2 = run_analysis_2(df)
    logger.info(f"Analysis 2 complete in {time.time()-t2:.1f}s")

    # Parse best strategy info: key format is "strategy_name_clftype"
    # clf types are "rf", "logreg", "svm" — always the last segment
    best_new_key = analysis_2.get("best_new_normalization", "E_binary_indicators_rf")
    clf_suffixes = ["_rf", "_logreg", "_svm"]
    best_strat_name = best_new_key
    best_clf_name = "rf"
    for suffix in clf_suffixes:
        if best_new_key.endswith(suffix):
            best_strat_name = best_new_key[:-len(suffix)]
            best_clf_name = suffix[1:]  # strip leading underscore
            break

    # STEP 3: Transfer Learning Diagnosis
    t3 = time.time()
    analysis_3 = run_analysis_3(
        df, best_strat_name, best_clf_name,
        analysis_2.get("strategies", {}),
    )
    logger.info(f"Analysis 3 complete in {time.time()-t3:.1f}s")

    # STEP 4: Deployment Recommendation
    t4 = time.time()
    analysis_4 = run_analysis_4(df, analysis_2, analysis_3)
    logger.info(f"Analysis 4 complete in {time.time()-t4:.1f}s")

    # STEP 5: Build output
    output = build_output(df, analysis_1, analysis_2, analysis_3, analysis_4)

    # Save output
    out_path = WORKSPACE / "eval_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Saved eval_out.json ({out_path.stat().st_size / 1024:.1f} KB)")

    total_time = time.time() - t0
    logger.info(f"Total runtime: {total_time:.1f}s")
    logger.info("=" * 70)
    logger.info("DONE")


if __name__ == "__main__":
    main()
