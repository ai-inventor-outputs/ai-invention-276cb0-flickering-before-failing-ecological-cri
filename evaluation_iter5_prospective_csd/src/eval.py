#!/usr/bin/env python3
"""Prospective CSD Boundary Detection: Sequential Protocols Without Prior Knowledge of d*.

Evaluates three deployment-realistic sequential protocols for detecting approaching
LLM capability boundaries using CSD indicators, without requiring prior knowledge
of the critical difficulty d*. Uses existing per-level CSD data from 5 model-task
pairs across arithmetic and graph coloring experiments.

Protocols:
  A - Threshold-based (zero-training, zero-knowledge)
  B - CUSUM/EWMA change-point detection (standard SPC)
  C - d*-free classifier (logistic regression + random forest, LOPO CV)
"""

import json
import sys
import math
import os
import gc
import resource
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from loguru import logger
import psutil

# ─── Logging ───────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ─── Hardware Detection ────────────────────────────────────────────────────────
def _detect_cpus() -> int:
    """Detect actual CPU allocation (containers/pods/bare metal)."""
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
    """Read RAM limit from cgroup (containers/pods)."""
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

# Set RAM limit - lightweight evaluation, 4GB budget
_avail = psutil.virtual_memory().available
RAM_BUDGET = int(4 * 1024**3)  # 4 GB
assert RAM_BUDGET < _avail, f"Budget {RAM_BUDGET/1e9:.1f}GB > available {_avail/1e9:.1f}GB"
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))

# ─── Configuration ─────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent

# Dependency data paths
ARITH_DATA = Path(
    "/ai-inventor/aii_pipeline/runs/neurips-open/3_invention_loop/"
    "iter_2/gen_art/exp_id1_it2__opus/full_method_out.json"
)
GC_DATA = Path(
    "/ai-inventor/aii_pipeline/runs/neurips-open/3_invention_loop/"
    "iter_2/gen_art/exp_id3_it2__opus/full_method_out.json"
)

# 5 model-task pairs (same as iter_4 label_distribution)
VALID_PAIRS = [
    ("arithmetic", "meta-llama/llama-3.1-8b-instruct", 20),
    ("arithmetic", "google/gemini-2.0-flash-001", 15),
    ("graph_coloring", "openai/gpt-4o-mini", 10),
    ("graph_coloring", "google/gemini-2.0-flash-001", 14),
    ("graph_coloring", "google/gemini-2.0-flash-lite-001", 11),
]

# CSD indicators used across protocols
CSD_INDICATORS = [
    "csd_variance", "dip_statistic", "silhouette_k2",
    "bimodality_coefficient", "disagreement_rate",
]

# Protocol A parameters
PROTO_A_K = 3          # calibration window (first K levels)
PROTO_A_VAR_SIGMA = 2.0  # variance threshold in standard deviations
PROTO_A_DISAGREE_THRESH = 0.30  # disagreement rate threshold
PROTO_A_DIP_PVALUE_THRESH = 0.05  # dip p-value threshold

# Protocol B parameters
CUSUM_K_ALLOWANCE = 0.5  # allowance factor k (in units of sigma)
CUSUM_H_OPTIONS = [4.0, 5.0]  # threshold h options (in units of sigma)
EWMA_LAMBDA_OPTIONS = [0.2, 0.3]
EWMA_L_OPTIONS = [2.5, 3.0]

# Prior best result for retention ratio
PRIOR_BEST_F1 = 0.949


# ─── Data Loading ──────────────────────────────────────────────────────────────
def load_arithmetic_data() -> pd.DataFrame:
    """Load per-level CSD data from arithmetic experiment (exp_id1_it2).

    Each example is one (model, difficulty_level) pair with CSD indicators
    in predict_ fields.
    """
    logger.info(f"Loading arithmetic data from {ARITH_DATA}")
    raw = json.loads(ARITH_DATA.read_text())

    rows = []
    for ds in raw["datasets"]:
        for ex in ds["examples"]:
            row = {
                "task": "arithmetic",
                "model": ex["metadata_model"],
                "difficulty": int(ex["metadata_difficulty_level"]),
                "d_star": int(ex["metadata_d_star"]),
                "accuracy": float(ex["predict_accuracy"]),
                "csd_variance": float(ex["predict_csd_variance"]),
                "dip_statistic": float(ex["predict_dip_statistic"]),
                "dip_pvalue": float(ex["predict_dip_pvalue"]),
                "silhouette_k2": float(ex["predict_silhouette_k2"]),
                "bimodality_coefficient": float(ex["predict_bimodality_coefficient"]),
                "disagreement_rate": float(ex["predict_disagreement_rate"]),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    logger.info(
        f"Loaded {len(df)} arithmetic rows, "
        f"{df['model'].nunique()} models, "
        f"levels {df['difficulty'].min()}-{df['difficulty'].max()}"
    )
    return df


def load_graph_coloring_data() -> pd.DataFrame:
    """Load per-level CSD data from graph coloring experiment (exp_id3_it2).

    Individual responses share per-level CSD indicators; deduplicate to
    one row per (model, difficulty_level).
    """
    logger.info(f"Loading graph coloring data from {GC_DATA}")
    raw = json.loads(GC_DATA.read_text())

    # Get d_star from analysis metadata
    d_star_map = {}
    for m in raw["metadata"]["analysis"]["models"]:
        d_star_map[m["model"]] = int(m["d_star"])
    logger.info(f"Graph coloring d_star map: {d_star_map}")

    rows = []
    seen = set()
    for ds in raw["datasets"]:
        for ex in ds["examples"]:
            model = ex["metadata_model"]
            diff = int(ex["metadata_difficulty_level"])
            key = (model, diff)
            if key in seen:
                continue
            seen.add(key)

            row = {
                "task": "graph_coloring",
                "model": model,
                "difficulty": diff,
                "d_star": d_star_map.get(model, -1),
                "accuracy": float(ex["metadata_csd_accuracy"]),
                "csd_variance": float(ex["metadata_csd_embedding_variance"]),
                "dip_statistic": float(ex["metadata_csd_dip_statistic"]),
                "dip_pvalue": float(ex["metadata_csd_dip_pvalue"]),
                "silhouette_k2": float(ex["metadata_csd_silhouette_score"]),
                "bimodality_coefficient": float(ex["metadata_csd_bimodality_coefficient"]),
                "disagreement_rate": float(ex["metadata_csd_disagreement_rate"]),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    logger.info(
        f"Loaded {len(df)} graph coloring rows (deduplicated), "
        f"{df['model'].nunique()} models, "
        f"levels {df['difficulty'].min()}-{df['difficulty'].max()}"
    )
    return df


def build_all_pairs() -> dict[str, pd.DataFrame]:
    """Build sorted per-level DataFrame for each of the 5 model-task pairs."""
    arith_df = load_arithmetic_data()
    gc_df = load_graph_coloring_data()
    all_df = pd.concat([arith_df, gc_df], ignore_index=True)

    # Free originals
    del arith_df, gc_df
    gc.collect()

    pairs = {}
    for task, model, d_star in VALID_PAIRS:
        mask = (all_df["task"] == task) & (all_df["model"] == model)
        pair_df = all_df[mask].sort_values("difficulty").reset_index(drop=True)

        if len(pair_df) == 0:
            logger.warning(f"No data for pair ({task}, {model})")
            continue

        pair_key = f"{task}__{model}"
        pairs[pair_key] = pair_df
        logger.info(
            f"Pair {pair_key}: {len(pair_df)} levels, "
            f"d*={d_star}, "
            f"difficulty range [{pair_df['difficulty'].min()}, {pair_df['difficulty'].max()}]"
        )

    del all_df
    gc.collect()
    return pairs


# ─── Protocol A: Threshold-Based ──────────────────────────────────────────────
def protocol_a(
    pair_df: pd.DataFrame, d_star: int,
    variant: str = "any2", K: int = PROTO_A_K,
) -> dict:
    """Protocol A: Threshold-based sequential detection.

    Calibrate alarm thresholds from first K levels (assumed easy).
    At each subsequent level d, check three triggers:
      T1: Running embedding variance > mean(var_1..K) + 2*std(var_1..K)
      T2: Disagreement rate > 0.30
      T3: Dip p-value < 0.05
    Alarm fires based on variant: "any1" (>=1), "any2" (>=2), "all3" (==3).
    """
    levels = pair_df["difficulty"].values
    n = len(levels)

    if n <= K:
        return {"d_alarm": None, "alarms": {}, "triggers": {}}

    # Calibrate from first K levels
    cal = pair_df.iloc[:K]
    var_mean = cal["csd_variance"].mean()
    var_std = cal["csd_variance"].std(ddof=1) if K > 1 else 0.0
    var_threshold = var_mean + PROTO_A_VAR_SIGMA * max(var_std, 1e-10)

    alarms = {}
    triggers = {}
    d_alarm = None

    for i in range(K, n):
        d = int(levels[i])
        row = pair_df.iloc[i]

        t1 = bool(row["csd_variance"] > var_threshold)
        t2 = bool(row["disagreement_rate"] > PROTO_A_DISAGREE_THRESH)
        t3 = bool(row["dip_pvalue"] < PROTO_A_DIP_PVALUE_THRESH)

        n_triggers = int(t1) + int(t2) + int(t3)
        triggers[d] = {"t1_variance": t1, "t2_disagreement": t2, "t3_dip": t3}

        if variant == "any1":
            alarmed = n_triggers >= 1
        elif variant == "any2":
            alarmed = n_triggers >= 2
        elif variant == "all3":
            alarmed = n_triggers == 3
        else:
            raise ValueError(f"Unknown variant: {variant}")

        alarms[d] = alarmed
        if alarmed and d_alarm is None:
            d_alarm = d

    return {"d_alarm": d_alarm, "alarms": alarms, "triggers": triggers}


# ─── Protocol B: CUSUM / EWMA ─────────────────────────────────────────────────
def protocol_b_cusum(
    pair_df: pd.DataFrame, d_star: int,
    indicator: str = "csd_variance", K: int = 3, h_sigma: float = 4.0,
) -> dict:
    """Protocol B: CUSUM (Cumulative Sum) change-point detection.

    S_t = max(0, S_{t-1} + (x_t - mu_0 - k))
    where k = 0.5*sigma (allowance), alarm when S_t > h*sigma.
    """
    levels = pair_df["difficulty"].values
    n = len(levels)

    if n <= K:
        return {"d_alarm": None, "alarms": {}, "cusum_stats": {}}

    cal = pair_df.iloc[:K]
    mu_0 = cal[indicator].mean()
    sigma = cal[indicator].std(ddof=1) if K > 1 else 1e-10
    sigma = max(sigma, 1e-10)

    k = CUSUM_K_ALLOWANCE * sigma
    h = h_sigma * sigma

    S = 0.0
    alarms = {}
    cusum_stats = {}
    d_alarm = None

    for i in range(K, n):
        d = int(levels[i])
        x = pair_df.iloc[i][indicator]
        S = max(0.0, S + (x - mu_0 - k))
        alarmed = S > h
        alarms[d] = alarmed
        cusum_stats[d] = float(S)

        if alarmed and d_alarm is None:
            d_alarm = d

    return {"d_alarm": d_alarm, "alarms": alarms, "cusum_stats": cusum_stats}


def protocol_b_ewma(
    pair_df: pd.DataFrame, d_star: int,
    indicator: str = "csd_variance", K: int = 3,
    lam: float = 0.2, L: float = 2.5,
) -> dict:
    """Protocol B: EWMA (Exponentially Weighted Moving Average).

    Z_t = lambda*x_t + (1-lambda)*Z_{t-1}
    Alarm when Z_t > mu_0 + L * sigma_EWMA
    sigma_EWMA = sigma * sqrt(lambda/(2-lambda) * (1-(1-lambda)^(2t)))
    """
    levels = pair_df["difficulty"].values
    n = len(levels)

    if n <= K:
        return {"d_alarm": None, "alarms": {}, "ewma_stats": {}}

    cal = pair_df.iloc[:K]
    mu_0 = cal[indicator].mean()
    sigma = cal[indicator].std(ddof=1) if K > 1 else 1e-10
    sigma = max(sigma, 1e-10)

    Z = mu_0  # initialize to calibration mean
    alarms = {}
    ewma_stats = {}
    d_alarm = None

    for i in range(K, n):
        d = int(levels[i])
        t = i - K + 1  # time step since monitoring start
        x = pair_df.iloc[i][indicator]
        Z = lam * x + (1 - lam) * Z

        # Time-varying control limit
        sigma_ewma = sigma * math.sqrt(
            lam / (2 - lam) * (1 - (1 - lam) ** (2 * t))
        )
        ucl = mu_0 + L * sigma_ewma

        alarmed = Z > ucl
        alarms[d] = alarmed
        ewma_stats[d] = {"Z": float(Z), "ucl": float(ucl)}

        if alarmed and d_alarm is None:
            d_alarm = d

    return {"d_alarm": d_alarm, "alarms": alarms, "ewma_stats": ewma_stats}


# ─── Protocol C: d*-Free Classifier ───────────────────────────────────────────
def build_classifier_features(pair_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Build features and labels for Protocol C.

    Features (no d* required):
      - Raw CSD indicators at current level (5 features)
      - Delta features: change from previous level (5 features)
      - Running z-scores from all levels seen so far (5 features)
      - Running max ratio: current / running_max (5 features)
    Total: 20 features per level.

    Label: accuracy drops >20 percentage points at the NEXT level.
    Returns (X, y) where X has shape (n_levels-1, 20) and y has shape (n_levels-1,).
    """
    indicators = CSD_INDICATORS
    n = len(pair_df)

    if n < 2:
        return np.empty((0, 20)), np.empty(0)

    feature_rows = []
    labels = []
    accuracies = pair_df["accuracy"].values

    for i in range(n - 1):  # can't compute label for last level
        row = pair_df.iloc[i]
        features = []

        # 1) Raw CSD indicators (5)
        for ind in indicators:
            features.append(float(row[ind]))

        # 2) Delta features from previous level (5)
        if i > 0:
            prev = pair_df.iloc[i - 1]
            for ind in indicators:
                features.append(float(row[ind]) - float(prev[ind]))
        else:
            features.extend([0.0] * len(indicators))

        # 3) Running z-scores (5)
        history = pair_df.iloc[: i + 1]
        for ind in indicators:
            running_mean = history[ind].mean()
            running_std = history[ind].std(ddof=1) if i > 0 else 1e-10
            running_std = max(running_std, 1e-10)
            features.append((float(row[ind]) - running_mean) / running_std)

        # 4) Running max ratio (5)
        for ind in indicators:
            running_max = history[ind].max()
            running_max = max(abs(running_max), 1e-10)
            features.append(float(row[ind]) / running_max)

        feature_rows.append(features)

        # Label: accuracy drops >20pp at next level
        acc_curr = accuracies[i]
        acc_next = accuracies[i + 1]
        label = 1 if (acc_curr - acc_next) > 0.20 else 0
        labels.append(label)

    return np.array(feature_rows, dtype=np.float64), np.array(labels, dtype=np.int64)


def protocol_c_lopo(
    pairs_data: dict[str, pd.DataFrame],
    classifier_type: str = "logreg",
) -> dict:
    """Protocol C: d*-free classifier with Leave-One-Pair-Out CV.

    Train on 4 pairs, test on 1. Repeat for all 5 folds.
    """
    pair_keys = sorted(pairs_data.keys())
    all_results = {}

    for test_key in pair_keys:
        # Build train data from all other pairs
        X_train_parts, y_train_parts = [], []
        for train_key in pair_keys:
            if train_key == test_key:
                continue
            X, y = build_classifier_features(pairs_data[train_key])
            if len(X) > 0:
                X_train_parts.append(X)
                y_train_parts.append(y)

        if not X_train_parts:
            all_results[test_key] = {
                "d_alarm": None, "alarms": {},
                "y_pred": [], "y_true": [], "y_prob": [],
            }
            continue

        X_train = np.vstack(X_train_parts)
        y_train = np.concatenate(y_train_parts)

        # Build test data
        X_test, y_test = build_classifier_features(pairs_data[test_key])

        if len(X_test) == 0 or len(np.unique(y_train)) < 2:
            all_results[test_key] = {
                "d_alarm": None, "alarms": {},
                "y_pred": [], "y_true": [], "y_prob": [],
            }
            continue

        # Replace NaN/inf with 0
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale features
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Train classifier
        if classifier_type == "logreg":
            clf = LogisticRegression(
                max_iter=1000, class_weight="balanced",
                random_state=42, solver="lbfgs",
            )
        elif classifier_type == "rf":
            clf = RandomForestClassifier(
                n_estimators=100, class_weight="balanced",
                random_state=42, n_jobs=min(NUM_CPUS, 4),
            )
        else:
            raise ValueError(f"Unknown classifier: {classifier_type}")

        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)
        y_prob = (
            clf.predict_proba(X_test_s)[:, 1]
            if hasattr(clf, "predict_proba")
            else y_pred.astype(float)
        )

        # Convert predictions to alarm mapping
        # Feature at index j corresponds to level j (predicting drop at j+1)
        # Alarm fires at level j if classifier predicts drop
        test_df = pairs_data[test_key]
        levels = test_df["difficulty"].values
        alarms = {}
        d_alarm = None

        for j, pred in enumerate(y_pred):
            d = int(levels[j])
            alarmed = bool(pred == 1)
            alarms[d] = alarmed
            if alarmed and d_alarm is None:
                d_alarm = d

        all_results[test_key] = {
            "d_alarm": d_alarm,
            "alarms": alarms,
            "y_pred": y_pred.tolist(),
            "y_true": y_test.tolist(),
            "y_prob": y_prob.tolist(),
        }

    return all_results


# ─── Metrics Computation ──────────────────────────────────────────────────────
def compute_pair_metrics(
    d_alarm: Optional[int],
    alarms: dict,
    d_star: int,
    pair_df: pd.DataFrame,
) -> dict:
    """Compute per-pair alarm metrics as specified in the artifact plan."""
    # Lead time
    if d_alarm is not None:
        lead_time = d_star - d_alarm
    else:
        lead_time = None

    # Accuracy at alarm
    accuracy_at_alarm = None
    if d_alarm is not None:
        alarm_row = pair_df[pair_df["difficulty"] == d_alarm]
        if len(alarm_row) > 0:
            accuracy_at_alarm = float(alarm_row.iloc[0]["accuracy"])

    # Sensitivity: 1 if alarm fires at any level <= d*
    sensitivity = 0
    for d, alarmed in alarms.items():
        if alarmed and d <= d_star:
            sensitivity = 1
            break

    # False alarm rate: fraction of clearly-safe levels (d < d*-3) with alarm
    clearly_safe = [d for d in alarms.keys() if d < d_star - 3]
    if len(clearly_safe) > 0:
        false_alarm_rate = sum(
            1 for d in clearly_safe if alarms.get(d, False)
        ) / len(clearly_safe)
    else:
        false_alarm_rate = 0.0

    # Precision / Recall / F1
    # TP = alarm at d in [d*-2, d*], FP = alarm at d < d*-2
    # FN = no alarm at [d*-2, d*] levels that are monitored
    monitored = set(int(d) for d in alarms.keys())
    near_boundary = set(range(max(1, d_star - 2), d_star + 1))
    near_monitored = monitored & near_boundary

    tp = sum(1 for d in near_monitored if alarms.get(d, False))
    fp = sum(1 for d in monitored if alarms.get(d, False) and d not in near_boundary)
    fn = sum(1 for d in near_monitored if not alarms.get(d, False))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "d_alarm": d_alarm,
        "d_star": d_star,
        "lead_time": lead_time,
        "accuracy_at_alarm": accuracy_at_alarm,
        "sensitivity": sensitivity,
        "false_alarm_rate": round(false_alarm_rate, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def aggregate_metrics(pair_metrics_list: list[dict]) -> dict:
    """Compute aggregate metrics across all 5 pairs."""
    valid_lt = [m for m in pair_metrics_list if m["lead_time"] is not None]
    mean_lead_time = float(np.mean([m["lead_time"] for m in valid_lt])) if valid_lt else 0.0

    mean_sensitivity = float(np.mean([m["sensitivity"] for m in pair_metrics_list]))
    mean_false_alarm_rate = float(np.mean([m["false_alarm_rate"] for m in pair_metrics_list]))
    macro_f1 = float(np.mean([m["f1"] for m in pair_metrics_list]))
    mean_precision = float(np.mean([m["precision"] for m in pair_metrics_list]))
    mean_recall = float(np.mean([m["recall"] for m in pair_metrics_list]))

    # Lead time > 0 fraction
    lt_positive_frac = float(np.mean([
        1 if m["lead_time"] is not None and m["lead_time"] > 0 else 0
        for m in pair_metrics_list
    ]))

    # Retention ratio vs prior best (csd_zt_reldist_rf F1=0.949)
    retention_ratio = macro_f1 / PRIOR_BEST_F1 if PRIOR_BEST_F1 > 0 else 0.0

    # Deployment readiness score (composite)
    drs = (
        0.4 * mean_sensitivity
        + 0.3 * (1 - mean_false_alarm_rate)
        + 0.3 * lt_positive_frac
    )

    return {
        "mean_lead_time": round(mean_lead_time, 4),
        "mean_sensitivity": round(mean_sensitivity, 4),
        "mean_false_alarm_rate": round(mean_false_alarm_rate, 6),
        "mean_precision": round(mean_precision, 6),
        "mean_recall": round(mean_recall, 6),
        "macro_f1": round(macro_f1, 6),
        "retention_ratio": round(retention_ratio, 6),
        "deployment_readiness_score": round(drs, 6),
        "lead_time_positive_fraction": round(lt_positive_frac, 4),
    }


# ─── Output Building ──────────────────────────────────────────────────────────
def build_output(
    pairs_data: dict,
    proto_a_results: dict,
    proto_b_results: dict,
    proto_c_results: dict,
) -> dict:
    """Build exp_eval_sol_out-compliant JSON output."""

    # ── Find best configurations ──
    best_a_key = max(
        proto_a_results.keys(),
        key=lambda k: proto_a_results[k]["aggregate"]["macro_f1"],
    )
    best_a = proto_a_results[best_a_key]["aggregate"]

    best_b_key = max(
        proto_b_results.keys(),
        key=lambda k: proto_b_results[k]["aggregate"]["macro_f1"],
    )
    best_b = proto_b_results[best_b_key]["aggregate"]

    best_c_key = max(
        proto_c_results.keys(),
        key=lambda k: proto_c_results[k]["aggregate"]["macro_f1"],
    )
    best_c = proto_c_results[best_c_key]["aggregate"]

    best_overall_f1 = max(best_a["macro_f1"], best_b["macro_f1"], best_c["macro_f1"])
    best_overall_ret = max(
        best_a["retention_ratio"], best_b["retention_ratio"], best_c["retention_ratio"]
    )
    best_overall_drs = max(
        best_a["deployment_readiness_score"],
        best_b["deployment_readiness_score"],
        best_c["deployment_readiness_score"],
    )

    logger.info(f"Best Protocol A: {best_a_key}, F1={best_a['macro_f1']:.4f}")
    logger.info(f"Best Protocol B: {best_b_key}, F1={best_b['macro_f1']:.4f}")
    logger.info(f"Best Protocol C: {best_c_key}, F1={best_c['macro_f1']:.4f}")
    logger.info(f"Best overall F1={best_overall_f1:.4f}, retention={best_overall_ret:.4f}")

    # ── metrics_agg ──
    metrics_agg = {
        "protocol_a_best_macro_f1": best_a["macro_f1"],
        "protocol_a_best_sensitivity": best_a["mean_sensitivity"],
        "protocol_a_best_false_alarm_rate": best_a["mean_false_alarm_rate"],
        "protocol_a_best_lead_time": best_a["mean_lead_time"],
        "protocol_a_best_drs": best_a["deployment_readiness_score"],
        "protocol_b_best_macro_f1": best_b["macro_f1"],
        "protocol_b_best_sensitivity": best_b["mean_sensitivity"],
        "protocol_b_best_false_alarm_rate": best_b["mean_false_alarm_rate"],
        "protocol_b_best_lead_time": best_b["mean_lead_time"],
        "protocol_b_best_drs": best_b["deployment_readiness_score"],
        "protocol_c_best_macro_f1": best_c["macro_f1"],
        "protocol_c_best_sensitivity": best_c["mean_sensitivity"],
        "protocol_c_best_false_alarm_rate": best_c["mean_false_alarm_rate"],
        "protocol_c_best_lead_time": best_c["mean_lead_time"],
        "protocol_c_best_drs": best_c["deployment_readiness_score"],
        "protocol_c_lopo_clf_f1": best_c.get("lopo_clf_f1", 0.0),
        "protocol_c_lopo_clf_auroc": best_c.get("lopo_clf_auroc", 0.0),
        "best_overall_macro_f1": round(best_overall_f1, 6),
        "best_overall_retention_ratio": round(best_overall_ret, 6),
        "best_overall_drs": round(best_overall_drs, 6),
        "prior_best_f1": PRIOR_BEST_F1,
        "n_pairs": float(len(pairs_data)),
    }

    # ── Datasets ──
    datasets = []

    # Dataset 1: Protocol A per-pair results
    pa_examples = []
    for variant in sorted(proto_a_results.keys()):
        pp = proto_a_results[variant]["per_pair"]
        for pair_key in sorted(pp.keys()):
            m = pp[pair_key]
            pa_examples.append({
                "input": f"Protocol A ({variant}) for {pair_key}",
                "output": (
                    f"d_alarm={m['d_alarm']}, lead_time={m['lead_time']}, "
                    f"F1={m['f1']:.4f}"
                ),
                "predict_d_alarm": str(m["d_alarm"]) if m["d_alarm"] is not None else "None",
                "predict_lead_time": (
                    str(m["lead_time"]) if m["lead_time"] is not None else "None"
                ),
                "predict_accuracy_at_alarm": (
                    str(round(m["accuracy_at_alarm"], 4))
                    if m["accuracy_at_alarm"] is not None
                    else "None"
                ),
                "eval_sensitivity": m["sensitivity"],
                "eval_false_alarm_rate": m["false_alarm_rate"],
                "eval_precision": m["precision"],
                "eval_recall": m["recall"],
                "eval_f1": m["f1"],
                "metadata_protocol": "A",
                "metadata_variant": variant,
                "metadata_pair": pair_key,
                "metadata_d_star": m["d_star"],
                "metadata_fold": "test",
            })
    datasets.append({"dataset": "protocol_a_results", "examples": pa_examples})

    # Dataset 2: Protocol B per-pair results
    pb_examples = []
    for config_key in sorted(proto_b_results.keys()):
        pp = proto_b_results[config_key]["per_pair"]
        for pair_key in sorted(pp.keys()):
            m = pp[pair_key]
            pb_examples.append({
                "input": f"Protocol B ({config_key}) for {pair_key}",
                "output": (
                    f"d_alarm={m['d_alarm']}, lead_time={m['lead_time']}, "
                    f"F1={m['f1']:.4f}"
                ),
                "predict_d_alarm": str(m["d_alarm"]) if m["d_alarm"] is not None else "None",
                "predict_lead_time": (
                    str(m["lead_time"]) if m["lead_time"] is not None else "None"
                ),
                "predict_accuracy_at_alarm": (
                    str(round(m["accuracy_at_alarm"], 4))
                    if m["accuracy_at_alarm"] is not None
                    else "None"
                ),
                "eval_sensitivity": m["sensitivity"],
                "eval_false_alarm_rate": m["false_alarm_rate"],
                "eval_precision": m["precision"],
                "eval_recall": m["recall"],
                "eval_f1": m["f1"],
                "metadata_protocol": "B",
                "metadata_variant": config_key,
                "metadata_pair": pair_key,
                "metadata_d_star": m["d_star"],
                "metadata_fold": "test",
            })
    datasets.append({"dataset": "protocol_b_results", "examples": pb_examples})

    # Dataset 3: Protocol C per-pair results
    pc_examples = []
    for clf_type in sorted(proto_c_results.keys()):
        pp = proto_c_results[clf_type]["per_pair"]
        for pair_key in sorted(pp.keys()):
            m = pp[pair_key]
            pc_examples.append({
                "input": f"Protocol C ({clf_type}) for {pair_key}",
                "output": (
                    f"d_alarm={m['d_alarm']}, lead_time={m['lead_time']}, "
                    f"F1={m['f1']:.4f}, clf_F1={m.get('clf_f1', 0):.4f}"
                ),
                "predict_d_alarm": str(m["d_alarm"]) if m["d_alarm"] is not None else "None",
                "predict_lead_time": (
                    str(m["lead_time"]) if m["lead_time"] is not None else "None"
                ),
                "predict_accuracy_at_alarm": (
                    str(round(m["accuracy_at_alarm"], 4))
                    if m["accuracy_at_alarm"] is not None
                    else "None"
                ),
                "predict_clf_f1": str(round(m.get("clf_f1", 0.0), 6)),
                "predict_clf_auroc": str(round(m.get("clf_auroc", 0.0), 6)),
                "eval_sensitivity": m["sensitivity"],
                "eval_false_alarm_rate": m["false_alarm_rate"],
                "eval_precision": m["precision"],
                "eval_recall": m["recall"],
                "eval_f1": m["f1"],
                "metadata_protocol": "C",
                "metadata_variant": clf_type,
                "metadata_pair": pair_key,
                "metadata_d_star": m["d_star"],
                "metadata_fold": "test",
            })
    datasets.append({"dataset": "protocol_c_results", "examples": pc_examples})

    # Dataset 4: Aggregate comparison across protocols
    agg_examples = []
    for proto_name, best_key, agg_data in [
        ("A", best_a_key, best_a),
        ("B", best_b_key, best_b),
        ("C", best_c_key, best_c),
    ]:
        ex = {
            "input": f"Protocol {proto_name} (best: {best_key})",
            "output": (
                f"F1={agg_data['macro_f1']:.4f}, "
                f"Sensitivity={agg_data['mean_sensitivity']:.3f}, "
                f"DRS={agg_data['deployment_readiness_score']:.3f}"
            ),
            "predict_macro_f1": str(agg_data["macro_f1"]),
            "predict_mean_sensitivity": str(agg_data["mean_sensitivity"]),
            "predict_mean_false_alarm_rate": str(agg_data["mean_false_alarm_rate"]),
            "predict_mean_lead_time": str(agg_data["mean_lead_time"]),
            "predict_retention_ratio": str(agg_data["retention_ratio"]),
            "predict_deployment_readiness": str(agg_data["deployment_readiness_score"]),
            "eval_macro_f1": agg_data["macro_f1"],
            "eval_mean_sensitivity": agg_data["mean_sensitivity"],
            "eval_mean_false_alarm_rate": agg_data["mean_false_alarm_rate"],
            "eval_deployment_readiness_score": agg_data["deployment_readiness_score"],
            "eval_retention_ratio": agg_data["retention_ratio"],
            "metadata_protocol": proto_name,
            "metadata_best_variant": best_key,
            "metadata_fold": "test",
        }
        # Add classifier-specific metrics for Protocol C
        if proto_name == "C":
            ex["predict_lopo_clf_f1"] = str(agg_data.get("lopo_clf_f1", 0.0))
            ex["predict_lopo_clf_auroc"] = str(agg_data.get("lopo_clf_auroc", 0.0))
            ex["eval_lopo_clf_f1"] = agg_data.get("lopo_clf_f1", 0.0)
            ex["eval_lopo_clf_auroc"] = agg_data.get("lopo_clf_auroc", 0.0)
        agg_examples.append(ex)
    datasets.append({"dataset": "aggregate_comparison", "examples": agg_examples})

    # Dataset 5: All Protocol B configs (for parameter sensitivity analysis)
    pb_agg_examples = []
    for config_key in sorted(proto_b_results.keys()):
        agg = proto_b_results[config_key]["aggregate"]
        pb_agg_examples.append({
            "input": f"Protocol B config: {config_key}",
            "output": (
                f"F1={agg['macro_f1']:.4f}, "
                f"Sensitivity={agg['mean_sensitivity']:.3f}"
            ),
            "predict_macro_f1": str(agg["macro_f1"]),
            "predict_mean_sensitivity": str(agg["mean_sensitivity"]),
            "predict_mean_lead_time": str(agg["mean_lead_time"]),
            "eval_macro_f1": agg["macro_f1"],
            "eval_mean_sensitivity": agg["mean_sensitivity"],
            "eval_mean_false_alarm_rate": agg["mean_false_alarm_rate"],
            "eval_deployment_readiness_score": agg["deployment_readiness_score"],
            "metadata_protocol": "B",
            "metadata_config": config_key,
            "metadata_fold": "test",
        })
    datasets.append({"dataset": "protocol_b_parameter_sensitivity", "examples": pb_agg_examples})

    # ── Build metadata ──
    metadata = {
        "evaluation_name": "Prospective CSD Boundary Detection",
        "description": (
            "Three deployment-realistic sequential protocols for detecting "
            "approaching LLM capability boundaries using CSD indicators, "
            "without requiring prior knowledge of d*."
        ),
        "n_pairs": len(pairs_data),
        "pairs": sorted(pairs_data.keys()),
        "protocol_a_variants": sorted(proto_a_results.keys()),
        "protocol_a_best": best_a_key,
        "protocol_b_variants": sorted(proto_b_results.keys()),
        "protocol_b_best": best_b_key,
        "protocol_c_variants": sorted(proto_c_results.keys()),
        "protocol_c_best": best_c_key,
        "prior_best_classifier": "csd_zt_reldist_rf",
        "prior_best_f1": PRIOR_BEST_F1,
        "calibration_window_K": PROTO_A_K,
        "all_protocol_a_aggregates": {
            k: proto_a_results[k]["aggregate"] for k in sorted(proto_a_results.keys())
        },
        "all_protocol_b_aggregates": {
            k: proto_b_results[k]["aggregate"] for k in sorted(proto_b_results.keys())
        },
        "all_protocol_c_aggregates": {
            k: proto_c_results[k]["aggregate"] for k in sorted(proto_c_results.keys())
        },
    }

    return {
        "metadata": metadata,
        "metrics_agg": metrics_agg,
        "datasets": datasets,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────
@logger.catch
def main():
    logger.info("=" * 70)
    logger.info("Prospective CSD Boundary Detection Evaluation")
    logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM")
    logger.info(f"RAM budget: {RAM_BUDGET / 1e9:.1f} GB")
    logger.info("=" * 70)

    # Create output dirs
    (WORKSPACE / "logs").mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load data ──
    logger.info("Step 1: Loading data from dependency experiments")
    pairs_data = build_all_pairs()
    logger.info(f"Loaded {len(pairs_data)} model-task pairs")

    if len(pairs_data) == 0:
        logger.error("No valid pairs found!")
        sys.exit(1)

    # ── Step 2: Protocol A (3 variants) ──
    logger.info("")
    logger.info("=" * 60)
    logger.info("Step 2: Protocol A - Threshold-Based Detection")
    logger.info("=" * 60)

    proto_a_results = {}
    for variant in ["any1", "any2", "all3"]:
        variant_results = {}
        for pair_key, pair_df in sorted(pairs_data.items()):
            d_star = int(pair_df.iloc[0]["d_star"])
            result = protocol_a(pair_df, d_star, variant=variant)
            metrics = compute_pair_metrics(
                result["d_alarm"], result["alarms"], d_star, pair_df,
            )
            variant_results[pair_key] = {**result, **metrics}
            logger.info(
                f"  A-{variant} | {pair_key}: "
                f"d_alarm={result['d_alarm']}, "
                f"lead={metrics['lead_time']}, "
                f"sens={metrics['sensitivity']}, "
                f"FAR={metrics['false_alarm_rate']:.3f}, "
                f"F1={metrics['f1']:.3f}"
            )

            # Log trigger analysis
            triggers = result.get("triggers", {})
            if triggers:
                t1_count = sum(1 for t in triggers.values() if t["t1_variance"])
                t2_count = sum(1 for t in triggers.values() if t["t2_disagreement"])
                t3_count = sum(1 for t in triggers.values() if t["t3_dip"])
                n_levels = len(triggers)
                logger.debug(
                    f"    Triggers: T1(var)={t1_count}/{n_levels}, "
                    f"T2(dis)={t2_count}/{n_levels}, "
                    f"T3(dip)={t3_count}/{n_levels}"
                )

        agg = aggregate_metrics([variant_results[k] for k in sorted(variant_results)])
        proto_a_results[variant] = {"per_pair": variant_results, "aggregate": agg}
        logger.info(
            f"  >>> A-{variant} aggregate: "
            f"F1={agg['macro_f1']:.4f}, "
            f"sens={agg['mean_sensitivity']:.3f}, "
            f"FAR={agg['mean_false_alarm_rate']:.3f}, "
            f"DRS={agg['deployment_readiness_score']:.3f}"
        )

    # ── Step 3: Protocol B (CUSUM + EWMA) ──
    logger.info("")
    logger.info("=" * 60)
    logger.info("Step 3: Protocol B - CUSUM/EWMA Change-Point Detection")
    logger.info("=" * 60)

    proto_b_results = {}

    # CUSUM variants
    for indicator in ["csd_variance", "disagreement_rate"]:
        for h_sigma in CUSUM_H_OPTIONS:
            config_key = f"cusum_{indicator}_h{h_sigma}"
            variant_results = {}
            for pair_key, pair_df in sorted(pairs_data.items()):
                d_star = int(pair_df.iloc[0]["d_star"])
                result = protocol_b_cusum(
                    pair_df, d_star, indicator=indicator, h_sigma=h_sigma,
                )
                metrics = compute_pair_metrics(
                    result["d_alarm"], result["alarms"], d_star, pair_df,
                )
                variant_results[pair_key] = {**result, **metrics}

            agg = aggregate_metrics([variant_results[k] for k in sorted(variant_results)])
            proto_b_results[config_key] = {"per_pair": variant_results, "aggregate": agg}
            logger.info(
                f"  B-{config_key}: "
                f"F1={agg['macro_f1']:.4f}, "
                f"sens={agg['mean_sensitivity']:.3f}, "
                f"FAR={agg['mean_false_alarm_rate']:.3f}, "
                f"lead={agg['mean_lead_time']:.1f}"
            )

    # EWMA variants
    for indicator in ["csd_variance", "disagreement_rate"]:
        for lam in EWMA_LAMBDA_OPTIONS:
            for L_val in EWMA_L_OPTIONS:
                config_key = f"ewma_{indicator}_l{lam}_L{L_val}"
                variant_results = {}
                for pair_key, pair_df in sorted(pairs_data.items()):
                    d_star = int(pair_df.iloc[0]["d_star"])
                    result = protocol_b_ewma(
                        pair_df, d_star,
                        indicator=indicator, lam=lam, L=L_val,
                    )
                    metrics = compute_pair_metrics(
                        result["d_alarm"], result["alarms"], d_star, pair_df,
                    )
                    variant_results[pair_key] = {**result, **metrics}

                agg = aggregate_metrics([variant_results[k] for k in sorted(variant_results)])
                proto_b_results[config_key] = {"per_pair": variant_results, "aggregate": agg}
                logger.info(
                    f"  B-{config_key}: "
                    f"F1={agg['macro_f1']:.4f}, "
                    f"sens={agg['mean_sensitivity']:.3f}, "
                    f"FAR={agg['mean_false_alarm_rate']:.3f}, "
                    f"lead={agg['mean_lead_time']:.1f}"
                )

    # ── Step 4: Protocol C (d*-free Classifier, LOPO) ──
    logger.info("")
    logger.info("=" * 60)
    logger.info("Step 4: Protocol C - d*-Free Classifier (LOPO)")
    logger.info("=" * 60)

    proto_c_results = {}
    for clf_type in ["logreg", "rf"]:
        lopo_results = protocol_c_lopo(pairs_data, classifier_type=clf_type)

        # Compute per-pair metrics
        variant_results = {}
        all_y_true, all_y_pred, all_y_prob = [], [], []

        for pair_key in sorted(lopo_results.keys()):
            r = lopo_results[pair_key]
            d_star = int(pairs_data[pair_key].iloc[0]["d_star"])
            metrics = compute_pair_metrics(
                r["d_alarm"], r["alarms"], d_star, pairs_data[pair_key],
            )

            # Per-pair classifier F1/AUROC
            clf_f1 = 0.0
            clf_auroc = 0.0
            if len(r["y_true"]) > 0 and len(np.unique(r["y_true"])) > 1:
                clf_f1 = float(f1_score(r["y_true"], r["y_pred"]))
                try:
                    clf_auroc = float(roc_auc_score(r["y_true"], r["y_prob"]))
                except ValueError:
                    clf_auroc = 0.0
            elif len(r["y_true"]) > 0:
                clf_f1 = float(f1_score(r["y_true"], r["y_pred"], zero_division=0))

            variant_results[pair_key] = {
                **r, **metrics, "clf_f1": clf_f1, "clf_auroc": clf_auroc,
            }
            all_y_true.extend(r.get("y_true", []))
            all_y_pred.extend(r.get("y_pred", []))
            all_y_prob.extend(r.get("y_prob", []))

            logger.info(
                f"  C-{clf_type} | {pair_key}: "
                f"d_alarm={r['d_alarm']}, "
                f"lead={metrics['lead_time']}, "
                f"F1={metrics['f1']:.3f}, "
                f"clf_F1={clf_f1:.3f}, "
                f"clf_AUROC={clf_auroc:.3f}"
            )

        agg = aggregate_metrics([variant_results[k] for k in sorted(variant_results)])

        # Global LOPO classifier F1 and AUROC
        lopo_clf_f1 = 0.0
        lopo_clf_auroc = 0.0
        if len(all_y_true) > 0 and len(np.unique(all_y_true)) > 1:
            lopo_clf_f1 = float(f1_score(all_y_true, all_y_pred))
            try:
                lopo_clf_auroc = float(roc_auc_score(all_y_true, all_y_prob))
            except ValueError:
                lopo_clf_auroc = 0.0
        elif len(all_y_true) > 0:
            lopo_clf_f1 = float(f1_score(all_y_true, all_y_pred, zero_division=0))

        agg["lopo_clf_f1"] = round(lopo_clf_f1, 6)
        agg["lopo_clf_auroc"] = round(lopo_clf_auroc, 6)

        proto_c_results[clf_type] = {"per_pair": variant_results, "aggregate": agg}
        logger.info(
            f"  >>> C-{clf_type} aggregate: "
            f"F1={agg['macro_f1']:.4f}, "
            f"LOPO_clf_F1={lopo_clf_f1:.4f}, "
            f"AUROC={lopo_clf_auroc:.4f}, "
            f"DRS={agg['deployment_readiness_score']:.3f}"
        )

    # ── Step 5: Build and save output ──
    logger.info("")
    logger.info("=" * 60)
    logger.info("Step 5: Building output")
    logger.info("=" * 60)

    output = build_output(pairs_data, proto_a_results, proto_b_results, proto_c_results)

    out_path = WORKSPACE / "eval_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved output to {out_path}")

    # Log final summary
    ma = output["metrics_agg"]
    logger.info("")
    logger.info("=" * 70)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Protocol A best ({output['metadata']['protocol_a_best']}): "
                f"F1={ma['protocol_a_best_macro_f1']:.4f}, "
                f"DRS={ma['protocol_a_best_drs']:.4f}")
    logger.info(f"  Protocol B best ({output['metadata']['protocol_b_best']}): "
                f"F1={ma['protocol_b_best_macro_f1']:.4f}, "
                f"DRS={ma['protocol_b_best_drs']:.4f}")
    logger.info(f"  Protocol C best ({output['metadata']['protocol_c_best']}): "
                f"F1={ma['protocol_c_best_macro_f1']:.4f}, "
                f"DRS={ma['protocol_c_best_drs']:.4f}")
    logger.info(f"  Best overall: F1={ma['best_overall_macro_f1']:.4f}, "
                f"retention={ma['best_overall_retention_ratio']:.4f}")
    logger.info(f"  Prior best (csd_zt_reldist_rf): F1={PRIOR_BEST_F1}")
    logger.info("=" * 70)

    return output


if __name__ == "__main__":
    main()
