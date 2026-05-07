#!/usr/bin/env python3
"""CSD Early Warning Classifier: Boundary Prediction with Cross-Task/Cross-Model Evaluation.

Builds binary classifiers predicting whether a model-task pair is within 2 difficulty
levels of its capability boundary (d*) using CSD indicator features from iter_2
experiments. Compares CSD-based classifiers against disagreement-only and dip-only
baselines via leave-one-pair-out (LOPO), leave-one-task-out (LOTO), and
leave-one-model-out (LOMO) cross-validation.
"""

import gc
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import kendalltau
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).resolve().parent
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------------------
# Hardware & resource limits
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
RAM_BUDGET_BYTES = int(TOTAL_RAM_GB * 0.5 * 1024**3)  # 50% of container RAM
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET_BYTES * 3, RAM_BUDGET_BYTES * 3))

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, budget={RAM_BUDGET_BYTES/1e9:.1f} GB")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ARITH_CSD_PATH = Path("/ai-inventor/aii_pipeline/runs/neurips-open/3_invention_loop/iter_2/gen_art/exp_id1_it2__opus/method_out.json")
GC_CSD_PATH = Path("/ai-inventor/aii_pipeline/runs/neurips-open/3_invention_loop/iter_2/gen_art/exp_id3_it2__opus/method_out.json")

# Core CSD features (the zero-cost indicators)
CSD_FEATURES = [
    "embedding_variance", "dip_statistic", "silhouette_k2",
    "bimodality_coefficient", "disagreement_rate", "ashman_d",
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def optimize_threshold(feature_series: pd.Series, labels: np.ndarray) -> tuple[float, float, str]:
    """Find threshold maximizing F1 on training data. Returns (best_f1, best_thresh, direction)."""
    best_f1, best_t, best_dir = 0.0, float(feature_series.median()), ">="
    vals = feature_series.values
    # Try >= direction
    for t in np.linspace(vals.min(), vals.max(), 100):
        preds = (vals >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t, best_dir = f1, t, ">="
    # Try <= direction (some features decrease near boundary)
    for t in np.linspace(vals.min(), vals.max(), 100):
        preds = (vals <= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t, best_dir = f1, t, "<="
    return best_f1, best_t, best_dir


def apply_threshold(feature_series: pd.Series, thresh: float, direction: str) -> np.ndarray:
    """Apply threshold with direction."""
    if direction == ">=":
        return (feature_series.values >= thresh).astype(int)
    else:
        return (feature_series.values <= thresh).astype(int)


def optimize_prob_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find probability threshold maximizing F1 on training data."""
    best_f1, best_t = 0.0, 0.5
    for t in np.linspace(0.1, 0.9, 81):
        preds = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute F1, precision, recall, AUROC, AUPRC."""
    result = {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        try:
            result["auroc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            result["auroc"] = 0.0
        try:
            result["auprc"] = float(average_precision_score(y_true, y_prob))
        except ValueError:
            result["auprc"] = 0.0
    else:
        result["auroc"] = 0.0
        result["auprc"] = 0.0
    return result


# ---------------------------------------------------------------------------
# STEP 1: Load CSD indicator data from iter_2 experiments
# ---------------------------------------------------------------------------
@logger.catch
def load_data() -> pd.DataFrame:
    """Load and combine arithmetic and graph coloring CSD data."""
    t0 = time.time()

    # ---- 1a: Load Arithmetic CSD data ----
    logger.info(f"Loading arithmetic CSD data from {ARITH_CSD_PATH}")
    arith_data = json.loads(ARITH_CSD_PATH.read_text())
    arith_rows = []
    for ds in arith_data["datasets"]:
        for ex in ds["examples"]:
            arith_rows.append({
                "task": "arithmetic",
                "model": ex["metadata_model"],
                "difficulty": ex["metadata_difficulty_level"],
                "d_star": ex["metadata_d_star"],
                "accuracy": float(ex["predict_accuracy"]),
                "embedding_variance": float(ex["predict_csd_variance"]),
                "dip_statistic": float(ex["predict_dip_statistic"]),
                "dip_pvalue": float(ex["predict_dip_pvalue"]),
                "silhouette_k2": float(ex["predict_silhouette_k2"]),
                "bimodality_coefficient": float(ex["predict_bimodality_coefficient"]),
                "disagreement_rate": float(ex["predict_disagreement_rate"]),
                "step_autocorr": float(ex["predict_step_correctness_autocorr"]),
                "ashman_d": 0.0,  # Not available in arithmetic data
            })
    logger.info(f"Loaded {len(arith_rows)} arithmetic CSD rows")

    # ---- 1b: Load Graph Coloring CSD data ----
    logger.info(f"Loading graph coloring CSD data from {GC_CSD_PATH}")
    gc_data = json.loads(GC_CSD_PATH.read_text())

    # Extract d_star from metadata.analysis.models
    gc_d_stars = {}
    for m in gc_data["metadata"]["analysis"]["models"]:
        gc_d_stars[m["model"]] = m["d_star"]
    logger.info(f"GC d_stars: {gc_d_stars}")

    # Deduplicate to unique (model, difficulty_level) rows
    gc_seen: set[tuple] = set()
    gc_rows = []
    for ds in gc_data["datasets"]:
        for ex in ds["examples"]:
            key = (ex["metadata_model"], ex["metadata_difficulty_level"])
            if key in gc_seen:
                continue
            gc_seen.add(key)
            model = ex["metadata_model"]
            gc_rows.append({
                "task": "graph_coloring",
                "model": model,
                "difficulty": ex["metadata_difficulty_level"],
                "d_star": gc_d_stars.get(model, 0),
                "accuracy": float(ex["metadata_csd_accuracy"]),
                "embedding_variance": float(ex["metadata_csd_embedding_variance"]),
                "dip_statistic": float(ex["metadata_csd_dip_statistic"]),
                "dip_pvalue": float(ex.get("metadata_csd_dip_pvalue", 0)),
                "silhouette_k2": float(ex["metadata_csd_silhouette_score"]),
                "bimodality_coefficient": float(ex["metadata_csd_bimodality_coefficient"]),
                "disagreement_rate": float(ex["metadata_csd_disagreement_rate"]),
                "step_autocorr": 0.0,  # Not available in GC data
                "ashman_d": float(ex["metadata_csd_ashman_d"]),
            })
    logger.info(f"Loaded {len(gc_rows)} graph coloring CSD rows (deduplicated)")

    # ---- 1c: Combine and filter ----
    all_rows = arith_rows + gc_rows
    df = pd.DataFrame(all_rows)
    df["pair_id"] = df["task"] + "__" + df["model"]

    # Exclude gpt-4o-mini on arithmetic (d*=2, all levels are beyond boundary)
    exclude_mask = (df["task"] == "arithmetic") & (df["model"] == "openai/gpt-4o-mini")
    n_excluded = exclude_mask.sum()
    df = df[~exclude_mask].reset_index(drop=True)
    logger.info(f"Excluded {n_excluded} rows (gpt-4o-mini arithmetic, d*=2)")

    logger.info(f"Total rows after filtering: {len(df)}, pairs: {df.pair_id.nunique()}")
    logger.info(f"Unique pairs: {df.pair_id.unique().tolist()}")
    logger.info(f"Data loading took {time.time()-t0:.2f}s")

    del arith_data, gc_data
    gc.collect()
    return df


# ---------------------------------------------------------------------------
# STEP 2: Label construction
# ---------------------------------------------------------------------------
def construct_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary boundary-proximity labels."""
    # Primary: near_boundary=1 if difficulty >= d* - 2
    df["label_approaching"] = (df["difficulty"] >= df["d_star"] - 2).astype(int)
    # Alternative: symmetric +-2
    df["label_symmetric"] = (abs(df["difficulty"] - df["d_star"]) <= 2).astype(int)
    # Primary label
    df["label"] = df["label_approaching"]

    # Log label distribution per pair
    for pid in sorted(df.pair_id.unique()):
        sub = df[df.pair_id == pid]
        logger.info(
            f"  {pid}: n={len(sub)}, near={sub.label.sum()}, "
            f"safe={(1-sub.label).sum()}, d*={sub.d_star.iloc[0]}"
        )

    total_near = df.label.sum()
    total_safe = (1 - df.label).sum()
    logger.info(f"Total label distribution: near={total_near}, safe={total_safe}")
    return df


# ---------------------------------------------------------------------------
# STEP 3: Feature engineering
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add delta features and compute extended feature set."""
    # Compute rate-of-change within each pair
    for feat in CSD_FEATURES:
        df[f"{feat}_delta"] = 0.0
        for pid in df.pair_id.unique():
            mask = df.pair_id == pid
            sub = df.loc[mask].sort_values("difficulty")
            deltas = sub[feat].diff().fillna(0)
            df.loc[sub.index, f"{feat}_delta"] = deltas.values

    extended_features = CSD_FEATURES + [f"{f}_delta" for f in CSD_FEATURES]

    # Normalized difficulty (NOT used as classifier feature - just for analysis)
    df["norm_difficulty"] = 0.0
    for pid in df.pair_id.unique():
        mask = df.pair_id == pid
        d_vals = df.loc[mask, "difficulty"]
        d_range = d_vals.max() - d_vals.min()
        if d_range > 0:
            df.loc[mask, "norm_difficulty"] = (d_vals - d_vals.min()) / d_range

    # Trend features: Kendall tau of each CSD feature over trailing 5 levels
    for feat in CSD_FEATURES:
        df[f"{feat}_trend"] = 0.0
        for pid in df.pair_id.unique():
            mask = df.pair_id == pid
            sub = df.loc[mask].sort_values("difficulty")
            trends = []
            for i in range(len(sub)):
                window = sub[feat].iloc[max(0, i - 4):i + 1]
                if len(window) >= 3:
                    tau, _ = kendalltau(range(len(window)), window.values)
                    trends.append(tau if not np.isnan(tau) else 0.0)
                else:
                    trends.append(0.0)
            df.loc[sub.index, f"{feat}_trend"] = trends

    trend_features = [f"{f}_trend" for f in CSD_FEATURES]
    all_extended = extended_features + trend_features

    # ---- Fallback 2: Within-pair percentile normalization ----
    for feat in CSD_FEATURES:
        df[f"{feat}_pctile"] = 0.0
        for pid in df.pair_id.unique():
            mask = df.pair_id == pid
            vals = df.loc[mask, feat]
            ranked = vals.rank(pct=True)
            df.loc[mask, f"{feat}_pctile"] = ranked.values

    pctile_features = [f"{f}_pctile" for f in CSD_FEATURES]

    # ---- Fallback 2: Within-pair z-score normalization ----
    for feat in CSD_FEATURES:
        df[f"{feat}_zscore"] = 0.0
        for pid in df.pair_id.unique():
            mask = df.pair_id == pid
            vals = df.loc[mask, feat]
            std = vals.std()
            if std > 1e-10:
                df.loc[mask, f"{feat}_zscore"] = ((vals - vals.mean()) / std).values
            else:
                df.loc[mask, f"{feat}_zscore"] = 0.0

    zscore_features = [f"{f}_zscore" for f in CSD_FEATURES]

    # ---- Task indicator features ----
    df["is_arithmetic"] = (df["task"] == "arithmetic").astype(float)
    df["is_graph_coloring"] = (df["task"] == "graph_coloring").astype(float)
    task_indicator_features = ["is_arithmetic", "is_graph_coloring"]

    all_extended = extended_features + trend_features

    # Log feature ranges
    for feat in CSD_FEATURES:
        fmin, fmax, fmean = df[feat].min(), df[feat].max(), df[feat].mean()
        logger.debug(f"  Feature {feat}: min={fmin:.4f}, max={fmax:.4f}, mean={fmean:.4f}")

    # Check for NaN in all feature columns
    all_feat_cols = all_extended + pctile_features + zscore_features + task_indicator_features
    nan_count = df[all_feat_cols].isna().sum().sum()
    if nan_count > 0:
        logger.warning(f"Found {nan_count} NaN values in features, filling with 0")
        df[all_feat_cols] = df[all_feat_cols].fillna(0)

    logger.info(f"Engineered {len(all_feat_cols)} total features ({len(CSD_FEATURES)} core + "
                f"{len(extended_features)-len(CSD_FEATURES)} delta + {len(trend_features)} trend + "
                f"{len(pctile_features)} pctile + {len(zscore_features)} zscore + "
                f"{len(task_indicator_features)} task indicators)")
    return df, all_extended


# ---------------------------------------------------------------------------
# STEP 4: Mini-scale sanity check
# ---------------------------------------------------------------------------
def mini_sanity_check(df: pd.DataFrame) -> bool:
    """Quick validation on 2 pairs."""
    logger.info("=" * 60)
    logger.info("STEP 4: Mini-scale sanity check")
    logger.info("=" * 60)

    pairs = df.pair_id.unique().tolist()
    # Pick one arithmetic and one GC pair
    arith_pairs = [p for p in pairs if p.startswith("arithmetic")]
    gc_pairs = [p for p in pairs if p.startswith("graph_coloring")]

    if not arith_pairs or not gc_pairs:
        logger.warning("Not enough pair variety for mini check")
        return True

    mini_pairs = [arith_pairs[0], gc_pairs[0]]
    logger.info(f"Mini pairs: {mini_pairs}")
    df_mini = df[df.pair_id.isin(mini_pairs)]

    # Kendall tau diagnostic
    logger.info("Feature-label correlations (Kendall tau):")
    any_signal = False
    for feat in CSD_FEATURES:
        tau, pval = kendalltau(df_mini[feat], df_mini["label"])
        tau = tau if not np.isnan(tau) else 0.0
        logger.info(f"  {feat}: tau={tau:.3f}, p={pval:.4f}")
        if abs(tau) > 0.1:
            any_signal = True

    # Single-feature threshold classifiers
    logger.info("Single-feature threshold classifiers (train on pair 1, test on pair 2):")
    train = df_mini[df_mini.pair_id == mini_pairs[0]]
    test = df_mini[df_mini.pair_id == mini_pairs[1]]

    any_positive = False
    for feat in CSD_FEATURES:
        best_f1, best_t, best_dir = optimize_threshold(train[feat], train.label.values)
        test_preds = apply_threshold(test[feat], best_t, best_dir)
        test_f1 = f1_score(test.label, test_preds, zero_division=0)
        logger.info(f"  {feat} ({best_dir} {best_t:.4f}): train_F1={best_f1:.3f}, test_F1={test_f1:.3f}")
        if test_f1 > 0:
            any_positive = True

    if not any_positive:
        logger.warning("All test F1 = 0 in mini check; will try with extended features and LogReg")
    else:
        logger.info("Mini sanity check PASSED: at least one feature has test F1 > 0")

    return True


# ---------------------------------------------------------------------------
# STEP 5: Full cross-validation comparison
# ---------------------------------------------------------------------------
def run_cross_validation(df: pd.DataFrame, feature_sets: dict[str, list[str]]) -> dict:
    """Run LOPO, LOTO, LOMO cross-validation for all methods."""
    logger.info("=" * 60)
    logger.info("STEP 5: Full cross-validation comparison")
    logger.info("=" * 60)

    results = {
        "lopo": {},   # Leave-One-Pair-Out
        "loto": {},   # Leave-One-Task-Out
        "lomo": {},   # Leave-One-Model-Out
    }
    all_predictions = []  # Collect per-row predictions for output

    # ---- 5a: LOPO (Leave-One-Pair-Out) ----
    logger.info("--- LOPO Cross-Validation ---")
    pairs = sorted(df.pair_id.unique())
    lopo_method_metrics: dict[str, list] = {}

    for held_out in pairs:
        train_df = df[df.pair_id != held_out].copy()
        test_df = df[df.pair_id == held_out].copy()
        y_train = train_df.label.values
        y_test = test_df.label.values

        fold_results = {}

        # For each feature set (core CSD, extended CSD)
        for fs_name, fs_features in feature_sets.items():
            scaler = StandardScaler()
            X_train = scaler.fit_transform(train_df[fs_features].values)
            X_test = scaler.transform(test_df[fs_features].values)

            # CSD LogReg with optimized threshold
            if len(np.unique(y_train)) < 2:
                probs_lr = np.zeros(len(X_test))
                preds_lr = np.zeros(len(X_test), dtype=int)
            else:
                # Try multiple regularization strengths
                best_lr_f1, best_lr_C = 0.0, 1.0
                for C_val in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]:
                    clf_try = LogisticRegression(C=C_val, class_weight="balanced", max_iter=1000, random_state=42)
                    clf_try.fit(X_train, y_train)
                    train_probs = clf_try.predict_proba(X_train)[:, 1]
                    opt_t = optimize_prob_threshold(y_train, train_probs)
                    train_preds = (train_probs >= opt_t).astype(int)
                    train_f1 = f1_score(y_train, train_preds, zero_division=0)
                    if train_f1 > best_lr_f1:
                        best_lr_f1, best_lr_C = train_f1, C_val

                clf_lr = LogisticRegression(C=best_lr_C, class_weight="balanced", max_iter=1000, random_state=42)
                clf_lr.fit(X_train, y_train)
                train_probs_lr = clf_lr.predict_proba(X_train)[:, 1]
                opt_thresh = optimize_prob_threshold(y_train, train_probs_lr)
                probs_lr = clf_lr.predict_proba(X_test)[:, 1]
                preds_lr = (probs_lr >= opt_thresh).astype(int)

            fold_results[f"csd_logreg_{fs_name}"] = compute_metrics(y_test, preds_lr, probs_lr)

            # CSD Random Forest with optimized threshold
            clf_rf = RandomForestClassifier(
                n_estimators=50, class_weight="balanced", random_state=42, n_jobs=NUM_CPUS
            )
            clf_rf.fit(X_train, y_train)
            if len(np.unique(y_train)) < 2:
                probs_rf = np.zeros(len(X_test))
                preds_rf = np.zeros(len(X_test), dtype=int)
            else:
                train_probs_rf = clf_rf.predict_proba(X_train)[:, 1]
                opt_thresh_rf = optimize_prob_threshold(y_train, train_probs_rf)
                probs_rf = clf_rf.predict_proba(X_test)[:, 1]
                preds_rf = (probs_rf >= opt_thresh_rf).astype(int)

            fold_results[f"csd_rf_{fs_name}"] = compute_metrics(y_test, preds_rf, probs_rf)

            # Store predictions for output (only for core features)
            if fs_name == "core":
                for i, idx in enumerate(test_df.index):
                    all_predictions.append({
                        "index": int(idx),
                        "pair_id": held_out,
                        "cv_scheme": "lopo",
                        "predict_csd_logreg": "near_boundary" if preds_lr[i] == 1 else "safe",
                        "predict_csd_rf": "near_boundary" if preds_rf[i] == 1 else "safe",
                        "predict_csd_logreg_prob": float(probs_lr[i]),
                        "predict_csd_rf_prob": float(probs_rf[i]),
                    })

        # Baseline: disagreement threshold
        best_f1_dis, best_t_dis, best_dir_dis = optimize_threshold(train_df["disagreement_rate"], y_train)
        preds_dis = apply_threshold(test_df["disagreement_rate"], best_t_dis, best_dir_dis)
        fold_results["disagreement_only"] = compute_metrics(y_test, preds_dis, test_df.disagreement_rate.values)

        # Baseline: dip_statistic threshold
        best_f1_dip, best_t_dip, best_dir_dip = optimize_threshold(train_df["dip_statistic"], y_train)
        preds_dip = apply_threshold(test_df["dip_statistic"], best_t_dip, best_dir_dip)
        fold_results["dip_only"] = compute_metrics(y_test, preds_dip, test_df.dip_statistic.values)

        # Baseline: bimodality_coefficient threshold
        best_f1_bc, best_t_bc, best_dir_bc = optimize_threshold(train_df["bimodality_coefficient"], y_train)
        preds_bc = apply_threshold(test_df["bimodality_coefficient"], best_t_bc, best_dir_bc)
        fold_results["bimodality_only"] = compute_metrics(y_test, preds_bc, test_df.bimodality_coefficient.values)

        # Baseline: embedding_variance threshold
        best_f1_ev, best_t_ev, best_dir_ev = optimize_threshold(train_df["embedding_variance"], y_train)
        preds_ev = apply_threshold(test_df["embedding_variance"], best_t_ev, best_dir_ev)
        fold_results["variance_only"] = compute_metrics(y_test, preds_ev, test_df.embedding_variance.values)

        # Store baseline predictions
        for i, idx in enumerate(test_df.index):
            # Find the matching prediction record
            for pred in all_predictions:
                if pred["index"] == int(idx) and pred["cv_scheme"] == "lopo":
                    pred["predict_disagreement"] = "near_boundary" if preds_dis[i] == 1 else "safe"
                    pred["predict_dip"] = "near_boundary" if preds_dip[i] == 1 else "safe"
                    break

        results["lopo"][held_out] = fold_results

        # Aggregate per-method metrics
        for method, metrics in fold_results.items():
            if method not in lopo_method_metrics:
                lopo_method_metrics[method] = []
            lopo_method_metrics[method].append(metrics)

        logger.info(f"  LOPO fold={held_out}: "
                    f"CSD-LR={fold_results['csd_logreg_core']['f1']:.3f}, "
                    f"CSD-RF={fold_results['csd_rf_core']['f1']:.3f}, "
                    f"Disagr={fold_results['disagreement_only']['f1']:.3f}, "
                    f"Dip={fold_results['dip_only']['f1']:.3f}")

    # Compute macro-averaged metrics
    lopo_summary = {}
    for method, fold_metrics in lopo_method_metrics.items():
        lopo_summary[method] = {
            k: float(np.mean([m[k] for m in fold_metrics]))
            for k in fold_metrics[0].keys()
        }
    results["lopo_summary"] = lopo_summary

    logger.info("LOPO Summary (macro-averaged F1):")
    for method, metrics in sorted(lopo_summary.items()):
        logger.info(f"  {method}: F1={metrics['f1']:.3f}, AUROC={metrics['auroc']:.3f}")

    # ---- 5b: LOTO (Leave-One-Task-Out) ----
    logger.info("--- LOTO Cross-Validation ---")
    loto_method_metrics: dict[str, list] = {}

    for train_task, test_task in [("arithmetic", "graph_coloring"), ("graph_coloring", "arithmetic")]:
        train_df = df[df.task == train_task].copy()
        test_df = df[df.task == test_task].copy()
        y_train = train_df.label.values
        y_test = test_df.label.values

        fold_results = {}

        for fs_name, fs_features in feature_sets.items():
            scaler = StandardScaler()
            X_train = scaler.fit_transform(train_df[fs_features].values)
            X_test = scaler.transform(test_df[fs_features].values)

            if len(np.unique(y_train)) < 2:
                probs_lr = np.zeros(len(X_test))
                preds_lr = np.zeros(len(X_test), dtype=int)
                probs_rf = np.zeros(len(X_test))
                preds_rf = np.zeros(len(X_test), dtype=int)
            else:
                best_lr_f1, best_lr_C = 0.0, 1.0
                for C_val in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]:
                    clf_try = LogisticRegression(C=C_val, class_weight="balanced", max_iter=1000, random_state=42)
                    clf_try.fit(X_train, y_train)
                    tp = clf_try.predict_proba(X_train)[:, 1]
                    ot = optimize_prob_threshold(y_train, tp)
                    tf1 = f1_score(y_train, (tp >= ot).astype(int), zero_division=0)
                    if tf1 > best_lr_f1:
                        best_lr_f1, best_lr_C = tf1, C_val

                clf_lr = LogisticRegression(C=best_lr_C, class_weight="balanced", max_iter=1000, random_state=42)
                clf_lr.fit(X_train, y_train)
                train_probs_lr = clf_lr.predict_proba(X_train)[:, 1]
                opt_t_lr = optimize_prob_threshold(y_train, train_probs_lr)
                probs_lr = clf_lr.predict_proba(X_test)[:, 1]
                preds_lr = (probs_lr >= opt_t_lr).astype(int)

                clf_rf = RandomForestClassifier(n_estimators=50, class_weight="balanced", random_state=42, n_jobs=NUM_CPUS)
                clf_rf.fit(X_train, y_train)
                train_probs_rf = clf_rf.predict_proba(X_train)[:, 1]
                opt_t_rf = optimize_prob_threshold(y_train, train_probs_rf)
                probs_rf = clf_rf.predict_proba(X_test)[:, 1]
                preds_rf = (probs_rf >= opt_t_rf).astype(int)

            fold_results[f"csd_logreg_{fs_name}"] = compute_metrics(y_test, preds_lr, probs_lr)
            fold_results[f"csd_rf_{fs_name}"] = compute_metrics(y_test, preds_rf, probs_rf)

        # Baselines
        best_f1_dis, best_t_dis, best_dir_dis = optimize_threshold(train_df["disagreement_rate"], y_train)
        preds_dis = apply_threshold(test_df["disagreement_rate"], best_t_dis, best_dir_dis)
        fold_results["disagreement_only"] = compute_metrics(y_test, preds_dis, test_df.disagreement_rate.values)

        best_f1_dip, best_t_dip, best_dir_dip = optimize_threshold(train_df["dip_statistic"], y_train)
        preds_dip = apply_threshold(test_df["dip_statistic"], best_t_dip, best_dir_dip)
        fold_results["dip_only"] = compute_metrics(y_test, preds_dip, test_df.dip_statistic.values)

        best_f1_bc, best_t_bc, best_dir_bc = optimize_threshold(train_df["bimodality_coefficient"], y_train)
        preds_bc = apply_threshold(test_df["bimodality_coefficient"], best_t_bc, best_dir_bc)
        fold_results["bimodality_only"] = compute_metrics(y_test, preds_bc, test_df.bimodality_coefficient.values)

        best_f1_ev, best_t_ev, best_dir_ev = optimize_threshold(train_df["embedding_variance"], y_train)
        preds_ev = apply_threshold(test_df["embedding_variance"], best_t_ev, best_dir_ev)
        fold_results["variance_only"] = compute_metrics(y_test, preds_ev, test_df.embedding_variance.values)

        fold_name = f"train_{train_task}__test_{test_task}"
        results["loto"][fold_name] = fold_results

        for method, metrics in fold_results.items():
            if method not in loto_method_metrics:
                loto_method_metrics[method] = []
            loto_method_metrics[method].append(metrics)

        logger.info(f"  LOTO fold={fold_name}: "
                    f"CSD-LR={fold_results['csd_logreg_core']['f1']:.3f}, "
                    f"CSD-RF={fold_results['csd_rf_core']['f1']:.3f}, "
                    f"Disagr={fold_results['disagreement_only']['f1']:.3f}")

    loto_summary = {}
    for method, fold_metrics in loto_method_metrics.items():
        loto_summary[method] = {
            k: float(np.mean([m[k] for m in fold_metrics]))
            for k in fold_metrics[0].keys()
        }
    results["loto_summary"] = loto_summary

    logger.info("LOTO Summary (macro-averaged F1):")
    for method, metrics in sorted(loto_summary.items()):
        logger.info(f"  {method}: F1={metrics['f1']:.3f}, AUROC={metrics['auroc']:.3f}")

    # ---- 5c: LOMO (Leave-One-Model-Out) ----
    logger.info("--- LOMO Cross-Validation ---")
    lomo_method_metrics: dict[str, list] = {}

    # Find models that appear in multiple tasks (for meaningful LOMO)
    model_tasks = df.groupby("model")["task"].nunique()
    cross_task_models = model_tasks[model_tasks > 1].index.tolist()
    # Also include all models for individual LOMO
    all_models = sorted(df.model.unique())

    for held_out_model in all_models:
        train_df = df[df.model != held_out_model].copy()
        test_df = df[df.model == held_out_model].copy()

        if len(train_df) == 0 or len(test_df) == 0:
            continue
        if len(np.unique(train_df.label)) < 2:
            logger.warning(f"  LOMO skip {held_out_model}: only one class in train")
            continue

        y_train = train_df.label.values
        y_test = test_df.label.values

        fold_results = {}

        for fs_name, fs_features in feature_sets.items():
            scaler = StandardScaler()
            X_train = scaler.fit_transform(train_df[fs_features].values)
            X_test = scaler.transform(test_df[fs_features].values)

            if len(np.unique(y_train)) < 2:
                probs_lr = np.zeros(len(X_test))
                preds_lr = np.zeros(len(X_test), dtype=int)
                probs_rf = np.zeros(len(X_test))
                preds_rf = np.zeros(len(X_test), dtype=int)
            else:
                best_lr_f1, best_lr_C = 0.0, 1.0
                for C_val in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]:
                    clf_try = LogisticRegression(C=C_val, class_weight="balanced", max_iter=1000, random_state=42)
                    clf_try.fit(X_train, y_train)
                    tp = clf_try.predict_proba(X_train)[:, 1]
                    ot = optimize_prob_threshold(y_train, tp)
                    tf1 = f1_score(y_train, (tp >= ot).astype(int), zero_division=0)
                    if tf1 > best_lr_f1:
                        best_lr_f1, best_lr_C = tf1, C_val

                clf_lr = LogisticRegression(C=best_lr_C, class_weight="balanced", max_iter=1000, random_state=42)
                clf_lr.fit(X_train, y_train)
                train_probs_lr = clf_lr.predict_proba(X_train)[:, 1]
                opt_t_lr = optimize_prob_threshold(y_train, train_probs_lr)
                probs_lr = clf_lr.predict_proba(X_test)[:, 1]
                preds_lr = (probs_lr >= opt_t_lr).astype(int)

                clf_rf = RandomForestClassifier(n_estimators=50, class_weight="balanced", random_state=42, n_jobs=NUM_CPUS)
                clf_rf.fit(X_train, y_train)
                train_probs_rf = clf_rf.predict_proba(X_train)[:, 1]
                opt_t_rf = optimize_prob_threshold(y_train, train_probs_rf)
                probs_rf = clf_rf.predict_proba(X_test)[:, 1]
                preds_rf = (probs_rf >= opt_t_rf).astype(int)

            fold_results[f"csd_logreg_{fs_name}"] = compute_metrics(y_test, preds_lr, probs_lr)
            fold_results[f"csd_rf_{fs_name}"] = compute_metrics(y_test, preds_rf, probs_rf)

        # Baselines
        best_f1_dis, best_t_dis, best_dir_dis = optimize_threshold(train_df["disagreement_rate"], y_train)
        preds_dis = apply_threshold(test_df["disagreement_rate"], best_t_dis, best_dir_dis)
        fold_results["disagreement_only"] = compute_metrics(y_test, preds_dis, test_df.disagreement_rate.values)

        best_f1_dip, best_t_dip, best_dir_dip = optimize_threshold(train_df["dip_statistic"], y_train)
        preds_dip = apply_threshold(test_df["dip_statistic"], best_t_dip, best_dir_dip)
        fold_results["dip_only"] = compute_metrics(y_test, preds_dip, test_df.dip_statistic.values)

        best_f1_bc, best_t_bc, best_dir_bc = optimize_threshold(train_df["bimodality_coefficient"], y_train)
        preds_bc = apply_threshold(test_df["bimodality_coefficient"], best_t_bc, best_dir_bc)
        fold_results["bimodality_only"] = compute_metrics(y_test, preds_bc, test_df.bimodality_coefficient.values)

        best_f1_ev, best_t_ev, best_dir_ev = optimize_threshold(train_df["embedding_variance"], y_train)
        preds_ev = apply_threshold(test_df["embedding_variance"], best_t_ev, best_dir_ev)
        fold_results["variance_only"] = compute_metrics(y_test, preds_ev, test_df.embedding_variance.values)

        results["lomo"][held_out_model] = fold_results

        for method, metrics in fold_results.items():
            if method not in lomo_method_metrics:
                lomo_method_metrics[method] = []
            lomo_method_metrics[method].append(metrics)

        logger.info(f"  LOMO fold={held_out_model}: "
                    f"CSD-LR={fold_results['csd_logreg_core']['f1']:.3f}, "
                    f"CSD-RF={fold_results['csd_rf_core']['f1']:.3f}, "
                    f"Disagr={fold_results['disagreement_only']['f1']:.3f}")

    lomo_summary = {}
    for method, fold_metrics in lomo_method_metrics.items():
        lomo_summary[method] = {
            k: float(np.mean([m[k] for m in fold_metrics]))
            for k in fold_metrics[0].keys()
        }
    results["lomo_summary"] = lomo_summary

    logger.info("LOMO Summary (macro-averaged F1):")
    for method, metrics in sorted(lomo_summary.items()):
        logger.info(f"  {method}: F1={metrics['f1']:.3f}, AUROC={metrics['auroc']:.3f}")

    return results, all_predictions


# ---------------------------------------------------------------------------
# STEP 6: Feature importance analysis
# ---------------------------------------------------------------------------
def analyze_feature_importance(df: pd.DataFrame, feature_sets: dict[str, list[str]]) -> dict:
    """Train RF on all data and extract feature importances."""
    logger.info("=" * 60)
    logger.info("STEP 6: Feature importance analysis")
    logger.info("=" * 60)

    importance_results = {}
    for fs_name, fs_features in feature_sets.items():
        X = df[fs_features].values
        y = df.label.values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42, n_jobs=NUM_CPUS
        )
        clf.fit(X_scaled, y)

        importances = dict(zip(fs_features, clf.feature_importances_))
        sorted_imp = sorted(importances.items(), key=lambda x: -x[1])

        logger.info(f"Feature importances ({fs_name}):")
        for feat, imp in sorted_imp:
            logger.info(f"  {feat}: {imp:.4f}")

        importance_results[fs_name] = importances

    return importance_results


# ---------------------------------------------------------------------------
# STEP 7: Permutation test for significance
# ---------------------------------------------------------------------------
def permutation_test_f1(y_true: np.ndarray, preds_a: np.ndarray, preds_b: np.ndarray,
                        n_permutations: int = 1000, seed: int = 42) -> float:
    """Two-sided permutation test for F1 difference."""
    observed_diff = f1_score(y_true, preds_a, zero_division=0) - f1_score(y_true, preds_b, zero_division=0)
    rng = np.random.RandomState(seed)
    count = 0
    combined = np.column_stack([preds_a, preds_b])
    for _ in range(n_permutations):
        # Randomly swap predictions between methods
        swap = rng.randint(0, 2, size=len(y_true))
        perm_a = np.where(swap == 0, combined[:, 0], combined[:, 1])
        perm_b = np.where(swap == 0, combined[:, 1], combined[:, 0])
        perm_diff = f1_score(y_true, perm_a, zero_division=0) - f1_score(y_true, perm_b, zero_division=0)
        if abs(perm_diff) >= abs(observed_diff):
            count += 1
    return count / n_permutations


# ---------------------------------------------------------------------------
# STEP 8: Build output JSON
# ---------------------------------------------------------------------------
def build_output(df: pd.DataFrame, cv_results: dict, predictions: list,
                 importance_results: dict, feature_sets: dict[str, list[str]]) -> dict:
    """Construct method_out.json in exp_gen_sol_out schema."""
    logger.info("=" * 60)
    logger.info("STEP 8: Building output JSON")
    logger.info("=" * 60)

    # Determine best CSD method and best baseline
    lopo_summary = cv_results["lopo_summary"]
    csd_methods = [m for m in lopo_summary if m.startswith("csd_")]
    baseline_methods = [m for m in lopo_summary if not m.startswith("csd_")]

    best_csd = max(csd_methods, key=lambda m: lopo_summary[m]["f1"])
    best_baseline = max(baseline_methods, key=lambda m: lopo_summary[m]["f1"])

    best_csd_f1 = lopo_summary[best_csd]["f1"]
    best_baseline_f1 = lopo_summary[best_baseline]["f1"]

    if best_baseline_f1 > 0:
        improvement_pct = (best_csd_f1 - best_baseline_f1) / best_baseline_f1 * 100
    else:
        improvement_pct = 100.0 if best_csd_f1 > 0 else 0.0

    logger.info(f"Best CSD: {best_csd} (F1={best_csd_f1:.3f})")
    logger.info(f"Best baseline: {best_baseline} (F1={best_baseline_f1:.3f})")
    logger.info(f"Improvement: {improvement_pct:.1f}%")

    # Build per-pair label distribution
    label_dist = {}
    for pid in sorted(df.pair_id.unique()):
        sub = df[df.pair_id == pid]
        label_dist[pid] = {
            "near": int(sub.label.sum()),
            "safe": int((1 - sub.label).sum()),
            "d_star": int(sub.d_star.iloc[0]),
            "n_rows": len(sub),
        }

    # Cost analysis
    cost_analysis = {
        "CSD-LogReg": {"extra_api_calls": 0, "extra_cost_usd": 0.0, "source": "reuses majority-vote samples"},
        "CSD-RF": {"extra_api_calls": 0, "extra_cost_usd": 0.0, "source": "reuses majority-vote samples"},
        "Disagreement-only": {"extra_api_calls": 0, "extra_cost_usd": 0.0, "source": "reuses majority-vote samples"},
        "Dip-only": {"extra_api_calls": 0, "extra_cost_usd": 0.0, "source": "reuses majority-vote samples"},
        "Bimodality-only": {"extra_api_calls": 0, "extra_cost_usd": 0.0, "source": "reuses majority-vote samples"},
        "Variance-only": {"extra_api_calls": 0, "extra_cost_usd": 0.0, "source": "reuses majority-vote samples"},
        "SPUQ-threshold (not run)": {"extra_api_calls": 1296, "extra_cost_usd": 1.10, "source": "5 paraphrases per prompt (skipped: no API key)"},
    }

    # Build metadata
    metadata = {
        "method_name": "CSD_Early_Warning_Classifier",
        "description": (
            "Binary classifiers predicting whether a model-task pair is within 2 difficulty levels "
            "of its capability boundary (d*) using CSD indicator features. Compared against "
            "single-feature threshold baselines via LOPO, LOTO, and LOMO cross-validation."
        ),
        "classifier_comparison": {
            method: {
                "lopo_f1": lopo_summary[method]["f1"],
                "lopo_auroc": lopo_summary[method]["auroc"],
                "lopo_precision": lopo_summary[method]["precision"],
                "lopo_recall": lopo_summary[method]["recall"],
                "loto_f1": cv_results["loto_summary"].get(method, {}).get("f1", 0.0),
                "loto_auroc": cv_results["loto_summary"].get(method, {}).get("auroc", 0.0),
                "lomo_f1": cv_results["lomo_summary"].get(method, {}).get("f1", 0.0),
                "lomo_auroc": cv_results["lomo_summary"].get(method, {}).get("auroc", 0.0),
            }
            for method in sorted(lopo_summary.keys())
        },
        "best_csd_method": best_csd,
        "best_baseline_method": best_baseline,
        "improvement_over_best_baseline_pct": round(improvement_pct, 2),
        "success_criteria_met": {
            "f1_improvement_15pct": improvement_pct >= 15.0,
            "best_csd_f1": best_csd_f1,
            "best_baseline_f1": best_baseline_f1,
        },
        "cost_analysis": cost_analysis,
        "label_distribution": label_dist,
        "per_pair_results_lopo": cv_results["lopo"],
        "feature_importances": importance_results,
        "feature_sets_used": {k: v for k, v in feature_sets.items()},
        "n_total_rows": len(df),
        "n_pairs": int(df.pair_id.nunique()),
        "pairs": sorted(df.pair_id.unique().tolist()),
        "label_definition": "near_boundary=1 if difficulty >= d* - 2, else safe=0",
        "spuq_status": "skipped (no OPENROUTER_API_KEY available)",
    }

    # Build dataset examples (one per row with all predictions)
    # Create a lookup from predictions
    pred_lookup = {}
    for pred in predictions:
        pred_lookup[pred["index"]] = pred

    examples = []
    for _, row in df.iterrows():
        pred = pred_lookup.get(int(row.name), {})
        example = {
            "input": f"Classifier prediction for {row.pair_id} at difficulty={row.difficulty}",
            "output": "near_boundary" if row.label == 1 else "safe",
            "predict_csd_logreg": pred.get("predict_csd_logreg", ""),
            "predict_csd_rf": pred.get("predict_csd_rf", ""),
            "predict_disagreement": pred.get("predict_disagreement", ""),
            "predict_dip": pred.get("predict_dip", ""),
            "predict_csd_logreg_prob": str(round(pred.get("predict_csd_logreg_prob", 0.0), 4)),
            "metadata_pair_id": row.pair_id,
            "metadata_task": row.task,
            "metadata_model": row.model,
            "metadata_difficulty": int(row.difficulty),
            "metadata_d_star": int(row.d_star),
            "metadata_accuracy": float(row.accuracy),
            "metadata_embedding_variance": float(row.embedding_variance),
            "metadata_dip_statistic": float(row.dip_statistic),
            "metadata_silhouette_k2": float(row.silhouette_k2),
            "metadata_bimodality_coefficient": float(row.bimodality_coefficient),
            "metadata_disagreement_rate": float(row.disagreement_rate),
            "metadata_ashman_d": float(row.ashman_d),
            "metadata_label_approaching": int(row.label_approaching),
            "metadata_label_symmetric": int(row.label_symmetric),
        }
        examples.append(example)

    output = {
        "metadata": metadata,
        "datasets": [
            {
                "dataset": "classifier_comparison",
                "examples": examples,
            }
        ],
    }

    return output


# ---------------------------------------------------------------------------
# STEP 9: Alternative label analysis
# ---------------------------------------------------------------------------
def try_alternative_labels(df: pd.DataFrame, feature_sets: dict[str, list[str]]) -> dict:
    """Try alternative label definitions if primary yields poor results."""
    logger.info("=" * 60)
    logger.info("STEP 9: Alternative label analysis")
    logger.info("=" * 60)

    alt_results = {}

    for label_name, label_col in [("approaching", "label_approaching"), ("symmetric", "label_symmetric")]:
        logger.info(f"--- Label: {label_name} ---")
        df_alt = df.copy()
        df_alt["label"] = df_alt[label_col]

        pairs = sorted(df_alt.pair_id.unique())
        method_f1s: dict[str, list] = {}

        for held_out in pairs:
            train_df = df_alt[df_alt.pair_id != held_out]
            test_df = df_alt[df_alt.pair_id == held_out]
            y_train = train_df.label.values
            y_test = test_df.label.values

            if len(np.unique(y_train)) < 2:
                continue

            # CSD LogReg (core)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(train_df[feature_sets["core"]].values)
            X_test = scaler.transform(test_df[feature_sets["core"]].values)
            clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            probs = clf.predict_proba(X_test)[:, 1]
            f1_csd = f1_score(y_test, preds, zero_division=0)
            method_f1s.setdefault("csd_logreg", []).append(f1_csd)

            # Disagreement baseline
            _, best_t, best_dir = optimize_threshold(train_df["disagreement_rate"], y_train)
            preds_dis = apply_threshold(test_df["disagreement_rate"], best_t, best_dir)
            f1_dis = f1_score(y_test, preds_dis, zero_division=0)
            method_f1s.setdefault("disagreement", []).append(f1_dis)

        for method, f1s in method_f1s.items():
            mean_f1 = float(np.mean(f1s))
            logger.info(f"  {label_name}/{method}: macro-F1={mean_f1:.3f}")
            alt_results[f"{label_name}__{method}"] = mean_f1

    return alt_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
@logger.catch
def main():
    t_start = time.time()
    logger.info("=" * 60)
    logger.info("CSD Early Warning Classifier Experiment")
    logger.info("=" * 60)

    # Step 1: Load data
    df = load_data()

    # Step 2: Construct labels
    df = construct_labels(df)

    # Step 3: Engineer features
    df, all_extended_features = engineer_features(df)

    # Define feature sets
    pctile_features = [f"{f}_pctile" for f in CSD_FEATURES]
    zscore_features = [f"{f}_zscore" for f in CSD_FEATURES]
    delta_features = [f"{f}_delta" for f in CSD_FEATURES]
    trend_features = [f"{f}_trend" for f in CSD_FEATURES]
    task_indicators = ["is_arithmetic", "is_graph_coloring"]

    feature_sets = {
        "core": CSD_FEATURES,
        "extended": CSD_FEATURES + delta_features,
        "full": all_extended_features,
        "pctile": pctile_features,
        "zscore": zscore_features,
        "core_task": CSD_FEATURES + task_indicators,
        "pctile_task": pctile_features + task_indicators,
        "full_task": all_extended_features + task_indicators,
        "zscore_delta": zscore_features + delta_features,
        "zscore_trend": zscore_features + trend_features,
        "zscore_full": zscore_features + delta_features + trend_features,
    }

    # Step 4: Mini sanity check
    mini_sanity_check(df)

    # Step 5: Full cross-validation
    cv_results, predictions = run_cross_validation(df, feature_sets)

    # Step 6: Feature importance
    importance_results = analyze_feature_importance(df, feature_sets)

    # Step 7: Alternative label analysis
    alt_results = try_alternative_labels(df, feature_sets)

    # Step 8: Build and save output
    output = build_output(df, cv_results, predictions, importance_results, feature_sets)

    # Add alternative label results to metadata
    output["metadata"]["alternative_label_results"] = alt_results

    # Save method_out.json
    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved output to {out_path}")

    elapsed = time.time() - t_start
    logger.info(f"Total runtime: {elapsed:.1f}s")

    # Summary
    logger.info("=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)
    lopo = cv_results["lopo_summary"]
    csd_methods = sorted([m for m in lopo if m.startswith("csd_")])
    baseline_methods = sorted([m for m in lopo if not m.startswith("csd_")])

    logger.info("CSD Methods (LOPO macro-F1):")
    for m in csd_methods:
        logger.info(f"  {m}: F1={lopo[m]['f1']:.3f}, AUROC={lopo[m]['auroc']:.3f}")
    logger.info("Baseline Methods (LOPO macro-F1):")
    for m in baseline_methods:
        logger.info(f"  {m}: F1={lopo[m]['f1']:.3f}, AUROC={lopo[m]['auroc']:.3f}")

    best_csd = max(csd_methods, key=lambda m: lopo[m]["f1"])
    best_base = max(baseline_methods, key=lambda m: lopo[m]["f1"])
    imp = output["metadata"]["improvement_over_best_baseline_pct"]
    logger.info(f"Best CSD: {best_csd} F1={lopo[best_csd]['f1']:.3f}")
    logger.info(f"Best Baseline: {best_base} F1={lopo[best_base]['f1']:.3f}")
    logger.info(f"Improvement: {imp:.1f}%")
    logger.info(f"15% target met: {output['metadata']['success_criteria_met']['f1_improvement_15pct']}")


if __name__ == "__main__":
    main()
