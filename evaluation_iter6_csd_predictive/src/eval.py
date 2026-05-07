#!/usr/bin/env python3
"""CSD Predictive Independence Ablation Study.

Systematic ablation testing whether CSD ecological indicators have genuine
predictive power for boundary detection independent of difficulty-position encoding.
Loads the 108-row unified dataset from exp_id2_it4__opus, runs 5 ablation experiments
across 3 classifiers and 3 CV schemes with 1000-bootstrap 95% CIs.
"""

import json
import math
import os
import resource
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# ── Logging ──────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ── Hardware detection ───────────────────────────────────────────────────────
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
TOTAL_RAM_GB = _container_ram_gb() or 16.0
RAM_BUDGET = int(min(TOTAL_RAM_GB * 0.5, 8) * 1024**3)  # Conservative 50% or 8GB cap
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, budget={RAM_BUDGET/1e9:.1f} GB")

# ── Constants ────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent
DATA_PATH = Path("/ai-inventor/aii_pipeline/runs/neurips-open/3_invention_loop/iter_4/gen_art/exp_id2_it4__opus/full_method_out.json")

RAW_CSD_FEATURES = [
    "csd_variance", "dip_statistic", "silhouette_k2",
    "bimodality_coefficient", "disagreement_rate",
]
DELTA_FEATURES = [f"{f}_delta" for f in RAW_CSD_FEATURES]
ZT_FEATURES = [f"{f}_zt" for f in RAW_CSD_FEATURES]

VALID_PAIRS = [
    ("arithmetic", "meta-llama/llama-3.1-8b-instruct", 20),
    ("arithmetic", "google/gemini-2.0-flash-001", 15),
    ("graph_coloring", "openai/gpt-4o-mini", 10),
    ("graph_coloring", "google/gemini-2.0-flash-001", 14),
    ("graph_coloring", "google/gemini-2.0-flash-lite-001", 11),
]

SEED = 42
np.random.seed(SEED)


# ── Data Loading ─────────────────────────────────────────────────────────────
def load_unified_dataset() -> pd.DataFrame:
    """Load the csd_features_unified dataset from iter_4 output."""
    logger.info(f"Loading data from {DATA_PATH}")
    raw = json.loads(DATA_PATH.read_text())

    # Find the csd_features_unified dataset
    ds = None
    for d in raw.get("datasets", []):
        if d.get("dataset") == "csd_features_unified":
            ds = d
            break
    if ds is None:
        raise ValueError("csd_features_unified dataset not found in full_method_out.json")

    examples = ds["examples"]
    logger.info(f"Found {len(examples)} examples in csd_features_unified")

    rows = []
    for ex in examples:
        row = {
            "task_family": ex["metadata_task_family"],
            "model": ex["metadata_model"],
            "difficulty_level": int(ex["metadata_difficulty_level"]),
            "d_star": int(ex["metadata_d_star"]),
            "label": 1 if ex["output"] == "near" else 0,
            "pair_key": f"{ex['metadata_task_family']}__{ex['metadata_model']}",
        }
        # Parse predict_ fields as floats
        for k, v in ex.items():
            if k.startswith("predict_") and k != "predict_label":
                fname = k[len("predict_"):]
                try:
                    row[fname] = float(v)
                except (ValueError, TypeError):
                    row[fname] = np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    logger.info(f"DataFrame shape: {df.shape}, label distribution: near={df['label'].sum()}, safe={(1-df['label']).sum()}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features needed for ablations."""
    df = df.copy()

    # Difficulty-position features
    df["difficulty_level_squared"] = df["difficulty_level"] ** 2

    # Relative position within task family's difficulty range
    for tf in df["task_family"].unique():
        mask = df["task_family"] == tf
        dmin = df.loc[mask, "difficulty_level"].min()
        dmax = df.loc[mask, "difficulty_level"].max()
        rng = dmax - dmin
        if rng > 0:
            df.loc[mask, "relative_position_in_range"] = (
                (df.loc[mask, "difficulty_level"] - dmin) / rng
            )
        else:
            df.loc[mask, "relative_position_in_range"] = 0.0

    # Trend features: rolling OLS slope over window=3 for each CSD indicator
    # per (model, task_family) pair
    trend_cols = []
    for feat in RAW_CSD_FEATURES:
        col_name = f"{feat}_trend3"
        trend_cols.append(col_name)
        df[col_name] = 0.0

    for pair_key in df["pair_key"].unique():
        mask = df["pair_key"] == pair_key
        sub = df.loc[mask].sort_values("difficulty_level").copy()
        idx = sub.index
        levels = sub["difficulty_level"].values

        for feat in RAW_CSD_FEATURES:
            col_name = f"{feat}_trend3"
            vals = sub[feat].values
            slopes = np.zeros(len(vals))

            for i in range(2, len(vals)):
                x = levels[i-2:i+1].astype(float)
                y = vals[i-2:i+1]
                if np.any(np.isnan(y)):
                    slopes[i] = 0.0
                else:
                    try:
                        slopes[i] = np.polyfit(x, y, 1)[0]
                    except (np.linalg.LinAlgError, ValueError):
                        slopes[i] = 0.0

            df.loc[idx, col_name] = slopes

    logger.info(f"Engineered features. New columns: difficulty_level_squared, relative_position_in_range, {len(trend_cols)} trend features")
    return df


# ── Classifiers ──────────────────────────────────────────────────────────────
def get_classifiers() -> dict:
    return {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=SEED),
        "rf": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=SEED),
        "svm": SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=SEED),
    }


# ── Cross-Validation Schemes ────────────────────────────────────────────────
def get_cv_folds(df: pd.DataFrame, scheme: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return list of (train_idx, test_idx) for the given CV scheme."""
    folds = []

    if scheme == "lopo":
        # Leave-One-Pair-Out: 5 folds
        for pair_key in sorted(df["pair_key"].unique()):
            test_mask = df["pair_key"] == pair_key
            train_idx = df.index[~test_mask].values
            test_idx = df.index[test_mask].values
            folds.append((train_idx, test_idx))

    elif scheme == "loto":
        # Leave-One-Task-Out: 2 folds
        for task in sorted(df["task_family"].unique()):
            test_mask = df["task_family"] == task
            train_idx = df.index[~test_mask].values
            test_idx = df.index[test_mask].values
            folds.append((train_idx, test_idx))

    elif scheme == "lomo":
        # Leave-One-Model-Out
        unique_models = sorted(df["model"].unique())
        for model in unique_models:
            test_mask = df["model"] == model
            train_idx = df.index[~test_mask].values
            test_idx = df.index[test_mask].values
            if len(test_idx) > 0 and len(train_idx) > 0:
                folds.append((train_idx, test_idx))

    return folds


# ── Evaluation Utilities ─────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute F1, precision, recall, AUROC from predictions."""
    f1 = f1_score(y_true, y_pred, zero_division=0.0)
    prec = precision_score(y_true, y_pred, zero_division=0.0)
    rec = recall_score(y_true, y_pred, zero_division=0.0)
    try:
        if len(np.unique(y_true)) > 1:
            auroc = roc_auc_score(y_true, y_prob)
        else:
            auroc = 0.5
    except ValueError:
        auroc = 0.5
    return {"f1": f1, "precision": prec, "recall": rec, "auroc": auroc}


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = SEED,
) -> dict:
    """Compute bootstrap 95% CIs for all metrics."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    if n == 0:
        return {f"{m}_{s}": 0.0 for m in ["f1", "precision", "recall", "auroc"] for s in ["mean", "ci_lo", "ci_hi"]}

    metrics_list = {"f1": [], "precision": [], "recall": [], "auroc": []}

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        ypr = y_prob[idx]
        m = compute_metrics(yt, yp, ypr)
        for k in metrics_list:
            metrics_list[k].append(m[k])

    result = {}
    for k, vals in metrics_list.items():
        vals = np.array(vals)
        result[f"{k}_mean"] = float(np.mean(vals))
        result[f"{k}_ci_lo"] = float(np.percentile(vals, 2.5))
        result[f"{k}_ci_hi"] = float(np.percentile(vals, 97.5))

    return result


def run_cv_experiment(
    df: pd.DataFrame,
    feature_cols: list[str],
    clf_name: str,
    cv_scheme: str,
    n_bootstrap: int = 1000,
) -> dict:
    """Run a full CV experiment and return bootstrap CI results."""
    folds = get_cv_folds(df, cv_scheme)

    all_y_true = []
    all_y_pred = []
    all_y_prob = []

    for train_idx, test_idx in folds:
        X_train = df.loc[train_idx, feature_cols].values.astype(float)
        y_train = df.loc[train_idx, "label"].values.astype(int)
        X_test = df.loc[test_idx, feature_cols].values.astype(float)
        y_test = df.loc[test_idx, "label"].values.astype(int)

        # Handle NaN
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = get_classifiers()[clf_name]
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_test)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_prob = np.array(all_y_prob)

    # Point estimate
    point = compute_metrics(all_y_true, all_y_pred, all_y_prob)

    # Bootstrap CIs
    ci = bootstrap_ci(all_y_true, all_y_pred, all_y_prob, n_bootstrap=n_bootstrap)

    return {**ci, "f1_point": point["f1"], "auroc_point": point["auroc"],
            "n_test": len(all_y_true), "n_folds": len(folds)}


# ── Worker function for parallel permutation test ────────────────────────────
def _run_single_permutation(args: tuple) -> tuple:
    """Worker for parallel permutation iterations. Returns (perm_i, full_f1, csd_f1)."""
    import io
    perm_i, df_json, best_features, csd_features, clf_name, cv_scheme, seed = args
    df = pd.read_json(io.StringIO(df_json), orient="records")
    rng = np.random.RandomState(seed)

    # Permute CSD values within each task_family
    raw_csd = ["csd_variance", "dip_statistic", "silhouette_k2", "bimodality_coefficient", "disagreement_rate"]
    delta_feats = [f"{f}_delta" for f in raw_csd]
    zt_feats = [f"{f}_zt" for f in raw_csd]
    all_csd_cols = raw_csd + delta_feats + zt_feats

    for tf in df["task_family"].unique():
        tf_mask = df["task_family"] == tf
        tf_idx = df.index[tf_mask]
        for feat in all_csd_cols:
            if feat in df.columns:
                vals = df.loc[tf_idx, feat].values.copy()
                rng.shuffle(vals)
                df.loc[tf_idx, feat] = vals

    full_result = run_cv_experiment(df, best_features, clf_name, cv_scheme, n_bootstrap=50)
    csd_result = run_cv_experiment(df, list(csd_features), clf_name, cv_scheme, n_bootstrap=50)
    return (perm_i, full_result["f1_mean"], csd_result["f1_mean"])


# ── Ablation Definitions ────────────────────────────────────────────────────
def get_ablation_configs() -> dict:
    """Return feature sets for each ablation."""
    trend_features = [f"{f}_trend3" for f in RAW_CSD_FEATURES]

    return {
        "ablation_1_pure_csd": {
            "description": "Pure CSD features, zero difficulty knowledge",
            "features": RAW_CSD_FEATURES,
        },
        "ablation_2_csd_dynamics": {
            "description": "CSD + dynamics (deltas + trends), no position features",
            "features": RAW_CSD_FEATURES + DELTA_FEATURES + trend_features,
        },
        "ablation_3_difficulty_only": {
            "description": "Difficulty position features only, no CSD",
            "features": ["difficulty_level", "difficulty_level_squared", "relative_position_in_range"],
        },
    }


def run_ablations_1_to_3(
    df: pd.DataFrame,
    classifiers: list[str],
    cv_schemes: list[str],
    n_bootstrap: int = 1000,
) -> dict:
    """Run ablations 1-3 across all classifier x cv_scheme combos."""
    configs = get_ablation_configs()
    results = {}

    total_combos = len(configs) * len(classifiers) * len(cv_schemes)
    logger.info(f"Running {total_combos} ablation experiments (ablations 1-3)")
    t0 = time.time()

    for abl_name, abl_cfg in configs.items():
        results[abl_name] = {
            "description": abl_cfg["description"],
            "features": abl_cfg["features"],
            "results": {},
        }
        for clf_name in classifiers:
            for cv_scheme in cv_schemes:
                try:
                    result = run_cv_experiment(df, abl_cfg["features"], clf_name, cv_scheme, n_bootstrap)
                    key = f"{clf_name}_{cv_scheme}"
                    results[abl_name]["results"][key] = result
                    logger.debug(f"  {abl_name} {clf_name}_{cv_scheme}: F1={result['f1_mean']:.4f}")
                except Exception:
                    logger.exception(f"Failed: {abl_name} {clf_name}_{cv_scheme}")

    elapsed = time.time() - t0
    logger.info(f"Ablations 1-3 completed in {elapsed:.1f}s")
    return results


def run_ablation_4_incremental(
    df: pd.DataFrame,
    n_bootstrap: int = 1000,
) -> dict:
    """Ablation 4: Incremental contribution analysis using LOPO + RF."""
    logger.info("Running Ablation 4: Incremental contribution analysis")
    t0 = time.time()

    clf_name = "rf"
    cv_scheme = "lopo"

    # 4a: Forward from difficulty-only
    diff_features = ["difficulty_level", "difficulty_level_squared", "relative_position_in_range"]
    base_result = run_cv_experiment(df, diff_features, clf_name, cv_scheme, n_bootstrap)
    base_f1 = base_result["f1_mean"]
    logger.info(f"  Difficulty-only baseline F1: {base_f1:.4f}")

    additions_from_diff = []
    for csd_feat in RAW_CSD_FEATURES:
        features = diff_features + [csd_feat]
        result = run_cv_experiment(df, features, clf_name, cv_scheme, n_bootstrap)
        gain = result["f1_mean"] - base_f1
        additions_from_diff.append({
            "feature": csd_feat,
            "new_f1": result["f1_mean"],
            "new_f1_ci_lo": result["f1_ci_lo"],
            "new_f1_ci_hi": result["f1_ci_hi"],
            "marginal_gain": gain,
        })
        logger.info(f"    +{csd_feat}: F1={result['f1_mean']:.4f} (gain={gain:+.4f})")

    # Sort by marginal gain
    additions_from_diff.sort(key=lambda x: x["marginal_gain"], reverse=True)

    # 4b: Forward from pure CSD
    csd_features = list(RAW_CSD_FEATURES)
    csd_base_result = run_cv_experiment(df, csd_features, clf_name, cv_scheme, n_bootstrap)
    csd_base_f1 = csd_base_result["f1_mean"]
    logger.info(f"  Pure CSD baseline F1: {csd_base_f1:.4f}")

    additions_from_csd = []
    diff_features_ordered = ["difficulty_level", "difficulty_level_squared", "relative_position_in_range"]
    current_features = list(csd_features)

    for diff_feat in diff_features_ordered:
        current_features = current_features + [diff_feat]
        result = run_cv_experiment(df, current_features, clf_name, cv_scheme, n_bootstrap)
        gain = result["f1_mean"] - csd_base_f1
        additions_from_csd.append({
            "feature": diff_feat,
            "new_f1": result["f1_mean"],
            "new_f1_ci_lo": result["f1_ci_lo"],
            "new_f1_ci_hi": result["f1_ci_hi"],
            "marginal_gain_from_csd_base": gain,
        })
        logger.info(f"    +{diff_feat}: F1={result['f1_mean']:.4f} (gain from CSD base={gain:+.4f})")

    elapsed = time.time() - t0
    logger.info(f"Ablation 4 completed in {elapsed:.1f}s")

    return {
        "description": "Incremental contribution: forward feature addition",
        "forward_from_difficulty": {
            "base_features": diff_features,
            "base_f1": base_f1,
            "base_f1_ci_lo": base_result["f1_ci_lo"],
            "base_f1_ci_hi": base_result["f1_ci_hi"],
            "additions": additions_from_diff,
        },
        "forward_from_csd": {
            "base_features": csd_features,
            "base_f1": csd_base_f1,
            "base_f1_ci_lo": csd_base_result["f1_ci_lo"],
            "base_f1_ci_hi": csd_base_result["f1_ci_hi"],
            "additions": additions_from_csd,
        },
    }


def run_ablation_5_permutation(
    df: pd.DataFrame,
    n_permutations: int = 100,
    n_bootstrap: int = 1000,
) -> dict:
    """Ablation 5: Matched controls via permutation test."""
    logger.info(f"Running Ablation 5: Permutation test ({n_permutations} iterations)")
    t0 = time.time()

    clf_name = "rf"
    cv_scheme = "lopo"

    # Best classifier features from iter_4: csd_zt + relative_dist_to_dstar
    best_features = ZT_FEATURES + ["relative_dist_to_dstar"]

    # Unpermuted baseline
    unpermuted_result = run_cv_experiment(df, best_features, clf_name, cv_scheme, n_bootstrap)
    unpermuted_f1 = unpermuted_result["f1_mean"]
    logger.info(f"  Unpermuted F1 (csd_zt_reldist_rf): {unpermuted_f1:.4f}")

    # Pure CSD unpermuted baseline
    pure_csd_unpermuted = run_cv_experiment(df, RAW_CSD_FEATURES, clf_name, cv_scheme, n_bootstrap)
    pure_csd_f1 = pure_csd_unpermuted["f1_mean"]
    logger.info(f"  Unpermuted Pure CSD F1: {pure_csd_f1:.4f}")

    # Permutation iterations - run in parallel
    df_json = df.to_json(orient="records")
    tasks = [
        (i, df_json, best_features, tuple(RAW_CSD_FEATURES), clf_name, cv_scheme, SEED + i + 1)
        for i in range(n_permutations)
    ]

    permuted_full_f1s = [0.0] * n_permutations
    permuted_csd_f1s = [0.0] * n_permutations
    n_workers = min(NUM_CPUS, n_permutations, 8)
    logger.info(f"  Running {n_permutations} permutations with {n_workers} workers")

    completed = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_run_single_permutation, task): task[0] for task in tasks}
        for future in as_completed(futures):
            try:
                perm_i, full_f1, csd_f1 = future.result()
                permuted_full_f1s[perm_i] = full_f1
                permuted_csd_f1s[perm_i] = csd_f1
                completed += 1
                if completed % 10 == 0:
                    logger.info(f"  Permutation {completed}/{n_permutations}: last full_F1={full_f1:.4f}, csd_F1={csd_f1:.4f}")
            except Exception:
                logger.exception(f"Permutation {futures[future]} failed")

    permuted_full_f1s = np.array(permuted_full_f1s)
    permuted_csd_f1s = np.array(permuted_csd_f1s)

    # Compute permutation test statistics
    perm_pvalue_full = float(np.mean(permuted_full_f1s >= unpermuted_f1))
    perm_drop_full = float(unpermuted_f1 - np.mean(permuted_full_f1s))

    perm_pvalue_csd = float(np.mean(permuted_csd_f1s >= pure_csd_f1))
    perm_drop_csd = float(pure_csd_f1 - np.mean(permuted_csd_f1s))

    elapsed = time.time() - t0
    logger.info(f"Ablation 5 completed in {elapsed:.1f}s")
    logger.info(f"  Full model: drop={perm_drop_full:.4f}, p={perm_pvalue_full:.4f}")
    logger.info(f"  Pure CSD: drop={perm_drop_csd:.4f}, p={perm_pvalue_csd:.4f}")

    return {
        "description": "Permutation test: shuffle CSD within task_family to break CSD-boundary correlation",
        "full_model": {
            "features": best_features,
            "unpermuted_f1": unpermuted_f1,
            "unpermuted_f1_ci_lo": unpermuted_result["f1_ci_lo"],
            "unpermuted_f1_ci_hi": unpermuted_result["f1_ci_hi"],
            "permuted_f1_mean": float(np.mean(permuted_full_f1s)),
            "permuted_f1_std": float(np.std(permuted_full_f1s)),
            "permuted_f1_ci_lo": float(np.percentile(permuted_full_f1s, 2.5)),
            "permuted_f1_ci_hi": float(np.percentile(permuted_full_f1s, 97.5)),
            "permutation_drop": perm_drop_full,
            "permutation_pvalue": perm_pvalue_full,
        },
        "pure_csd": {
            "features": list(RAW_CSD_FEATURES),
            "unpermuted_f1": pure_csd_f1,
            "unpermuted_f1_ci_lo": pure_csd_unpermuted["f1_ci_lo"],
            "unpermuted_f1_ci_hi": pure_csd_unpermuted["f1_ci_hi"],
            "permuted_f1_mean": float(np.mean(permuted_csd_f1s)),
            "permuted_f1_std": float(np.std(permuted_csd_f1s)),
            "permuted_f1_ci_lo": float(np.percentile(permuted_csd_f1s, 2.5)),
            "permuted_f1_ci_hi": float(np.percentile(permuted_csd_f1s, 97.5)),
            "permutation_drop": perm_drop_csd,
            "permutation_pvalue": perm_pvalue_csd,
        },
        "n_permutations": n_permutations,
    }


def compute_summary_metrics(
    ablation_results: dict,
    near_proportion: float,
) -> dict:
    """Compute aggregate summary metrics across ablations."""
    # Random baseline F1 = 2*p*p/(p+p) = p for balanced prediction at the positive rate
    random_f1 = near_proportion  # stratified random: predicts near with prob=near_proportion

    # Get best F1s from each ablation (using RF + LOPO as primary)
    def get_best_f1(abl_name: str, metric: str = "f1_mean") -> float:
        abl = ablation_results.get(abl_name, {})
        results = abl.get("results", {})
        best = 0.0
        for key, val in results.items():
            if metric in val:
                best = max(best, val[metric])
        return best

    pure_csd_f1 = get_best_f1("ablation_1_pure_csd")
    csd_dynamics_f1 = get_best_f1("ablation_2_csd_dynamics")
    diff_only_f1 = get_best_f1("ablation_3_difficulty_only")

    # Full F1 from iter_4 best: csd_zt_reldist_rf LOPO F1=0.949
    full_f1 = 0.949

    # CSD Contribution Ratio
    denom = full_f1 - random_f1
    if denom > 0:
        csd_contribution_ratio = (full_f1 - diff_only_f1) / denom
    else:
        csd_contribution_ratio = 0.0

    # CSD Lift
    csd_lift = pure_csd_f1 - random_f1

    # Dynamics Lift
    dynamics_lift = csd_dynamics_f1 - pure_csd_f1

    # Permutation results
    perm_results = ablation_results.get("ablation_5_permutation", {})
    perm_pvalue = perm_results.get("full_model", {}).get("permutation_pvalue", 1.0)
    perm_drop = perm_results.get("full_model", {}).get("permutation_drop", 0.0)

    # Scientific verdict
    verdicts = []
    if csd_lift > 0.05:
        verdicts.append("CSD indicators beat random chance")
    if csd_contribution_ratio > 0.3:
        verdicts.append("CSD provides substantial unique signal beyond difficulty position")
    elif csd_contribution_ratio > 0.1:
        verdicts.append("CSD provides modest unique signal beyond difficulty position")
    else:
        verdicts.append("CSD signal is largely redundant with difficulty position")

    if dynamics_lift > 0.05:
        verdicts.append("Ecological dynamics (trends/deltas) add meaningful signal")
    else:
        verdicts.append("Dynamics do not substantially improve over static CSD")

    if perm_pvalue < 0.05:
        verdicts.append("Permutation test confirms CSD carries unique causal signal (p<0.05)")
    else:
        verdicts.append(f"Permutation test inconclusive (p={perm_pvalue:.3f})")

    verdict = "; ".join(verdicts)

    # Incremental analysis
    incr = ablation_results.get("ablation_4_incremental", {})
    top_csd_feature = "N/A"
    top_csd_gain = 0.0
    fwd_diff = incr.get("forward_from_difficulty", {})
    if fwd_diff.get("additions"):
        top_entry = fwd_diff["additions"][0]
        top_csd_feature = top_entry["feature"]
        top_csd_gain = top_entry["marginal_gain"]

    return {
        "csd_contribution_ratio": round(csd_contribution_ratio, 4),
        "csd_lift_above_random": round(csd_lift, 4),
        "dynamics_lift": round(dynamics_lift, 4),
        "permutation_pvalue": round(perm_pvalue, 4),
        "permutation_drop": round(perm_drop, 4),
        "pure_csd_best_f1": round(pure_csd_f1, 4),
        "difficulty_only_best_f1": round(diff_only_f1, 4),
        "full_model_f1": round(full_f1, 4),
        "random_baseline_f1": round(random_f1, 4),
        "pure_csd_above_random": bool(csd_lift > 0),
        "csd_adds_to_difficulty": bool(csd_contribution_ratio > 0.05),
        "dynamics_add_to_static": bool(dynamics_lift > 0.02),
        "top_csd_feature_over_difficulty": top_csd_feature,
        "top_csd_marginal_gain": round(top_csd_gain, 4),
        "scientific_verdict": verdict,
    }


# ── Output Formatting ────────────────────────────────────────────────────────
def format_output(
    ablation_results: dict,
    summary_metrics: dict,
    df: pd.DataFrame,
    run_mode: str,
    n_bootstrap: int,
    n_permutations: int,
) -> dict:
    """Format results into the exp_eval_sol_out schema."""

    # metrics_agg: flatten summary + key ablation results to numeric-only dict
    metrics_agg = {}
    for k, v in summary_metrics.items():
        # Check bool BEFORE int (bool is subclass of int in Python)
        if isinstance(v, bool):
            metrics_agg[k] = int(v)
        elif isinstance(v, (int, float)):
            safe_k = k.replace(".", "_")
            metrics_agg[safe_k] = v

    # Add per-ablation best F1 values
    for abl_name in ["ablation_1_pure_csd", "ablation_2_csd_dynamics", "ablation_3_difficulty_only"]:
        abl = ablation_results.get(abl_name, {})
        for key, val in abl.get("results", {}).items():
            safe_name = f"{abl_name}_{key}_f1"
            metrics_agg[safe_name] = round(val.get("f1_mean", 0.0), 4)

    # Dataset 1: Ablation comparison
    ablation_comparison_examples = []
    for abl_name in ["ablation_1_pure_csd", "ablation_2_csd_dynamics", "ablation_3_difficulty_only"]:
        abl = ablation_results.get(abl_name, {})
        for key, val in sorted(abl.get("results", {}).items()):
            clf, cv = key.split("_", 1)
            ablation_comparison_examples.append({
                "input": f"Ablation: {abl_name}, Classifier: {clf}, CV: {cv}",
                "output": f"F1={val.get('f1_mean', 0):.4f} [{val.get('f1_ci_lo', 0):.4f}, {val.get('f1_ci_hi', 0):.4f}]",
                "predict_f1_mean": str(round(val.get("f1_mean", 0), 6)),
                "predict_f1_ci_lo": str(round(val.get("f1_ci_lo", 0), 6)),
                "predict_f1_ci_hi": str(round(val.get("f1_ci_hi", 0), 6)),
                "predict_auroc_mean": str(round(val.get("auroc_mean", 0), 6)),
                "predict_precision_mean": str(round(val.get("precision_mean", 0), 6)),
                "predict_recall_mean": str(round(val.get("recall_mean", 0), 6)),
                "metadata_ablation": abl_name,
                "metadata_classifier": clf,
                "metadata_cv_scheme": cv,
                "metadata_n_features": str(len(abl.get("features", []))),
                "metadata_fold": "test",
                "eval_f1_mean": round(val.get("f1_mean", 0), 6),
                "eval_auroc_mean": round(val.get("auroc_mean", 0), 6),
            })

    # Dataset 2: Incremental contribution
    incremental_examples = []
    incr = ablation_results.get("ablation_4_incremental", {})

    fwd_diff = incr.get("forward_from_difficulty", {})
    for entry in fwd_diff.get("additions", []):
        incremental_examples.append({
            "input": f"Add {entry['feature']} to difficulty-only baseline",
            "output": f"F1={entry['new_f1']:.4f} (gain={entry['marginal_gain']:+.4f})",
            "predict_new_f1": str(round(entry["new_f1"], 6)),
            "predict_marginal_gain": str(round(entry["marginal_gain"], 6)),
            "predict_new_f1_ci_lo": str(round(entry.get("new_f1_ci_lo", 0), 6)),
            "predict_new_f1_ci_hi": str(round(entry.get("new_f1_ci_hi", 0), 6)),
            "metadata_direction": "forward_from_difficulty",
            "metadata_feature_added": entry["feature"],
            "metadata_fold": "test",
            "eval_marginal_gain": round(entry["marginal_gain"], 6),
        })

    fwd_csd = incr.get("forward_from_csd", {})
    for entry in fwd_csd.get("additions", []):
        incremental_examples.append({
            "input": f"Add {entry['feature']} to pure-CSD baseline",
            "output": f"F1={entry['new_f1']:.4f} (gain from CSD base={entry['marginal_gain_from_csd_base']:+.4f})",
            "predict_new_f1": str(round(entry["new_f1"], 6)),
            "predict_marginal_gain": str(round(entry["marginal_gain_from_csd_base"], 6)),
            "predict_new_f1_ci_lo": str(round(entry.get("new_f1_ci_lo", 0), 6)),
            "predict_new_f1_ci_hi": str(round(entry.get("new_f1_ci_hi", 0), 6)),
            "metadata_direction": "forward_from_csd",
            "metadata_feature_added": entry["feature"],
            "metadata_fold": "test",
            "eval_marginal_gain": round(entry["marginal_gain_from_csd_base"], 6),
        })

    # Dataset 3: Permutation test
    permutation_examples = []
    perm = ablation_results.get("ablation_5_permutation", {})

    for model_key in ["full_model", "pure_csd"]:
        pm = perm.get(model_key, {})
        if pm:
            permutation_examples.append({
                "input": f"Permutation test: {model_key}",
                "output": f"Unpermuted F1={pm.get('unpermuted_f1', 0):.4f}, Permuted F1={pm.get('permuted_f1_mean', 0):.4f}, p={pm.get('permutation_pvalue', 1):.4f}",
                "predict_unpermuted_f1": str(round(pm.get("unpermuted_f1", 0), 6)),
                "predict_permuted_f1_mean": str(round(pm.get("permuted_f1_mean", 0), 6)),
                "predict_permuted_f1_std": str(round(pm.get("permuted_f1_std", 0), 6)),
                "predict_permutation_drop": str(round(pm.get("permutation_drop", 0), 6)),
                "predict_permutation_pvalue": str(round(pm.get("permutation_pvalue", 1), 6)),
                "metadata_model_variant": model_key,
                "metadata_n_permutations": str(perm.get("n_permutations", 0)),
                "metadata_fold": "test",
                "eval_permutation_drop": round(pm.get("permutation_drop", 0), 6),
                "eval_permutation_pvalue": round(pm.get("permutation_pvalue", 1), 6),
            })

    # Build the output
    output = {
        "metadata": {
            "evaluation_name": "CSD_Predictive_Independence_Ablation",
            "n_total_rows": len(df),
            "n_model_task_pairs": len(df["pair_key"].unique()),
            "n_bootstrap": n_bootstrap,
            "n_permutations": n_permutations,
            "run_mode": run_mode,
            "classifiers": ["logreg", "rf", "svm"],
            "cv_schemes": ["lopo", "loto", "lomo"],
            "ablation_details": {
                k: {"description": v.get("description", ""), "features": v.get("features", [])}
                for k, v in ablation_results.items()
                if isinstance(v, dict) and "description" in v
            },
            "summary_metrics_detail": summary_metrics,
        },
        "metrics_agg": metrics_agg,
        "datasets": [],
    }

    if ablation_comparison_examples:
        output["datasets"].append({
            "dataset": "ablation_comparison",
            "examples": ablation_comparison_examples,
        })

    if incremental_examples:
        output["datasets"].append({
            "dataset": "incremental_contribution",
            "examples": incremental_examples,
        })

    if permutation_examples:
        output["datasets"].append({
            "dataset": "permutation_test",
            "examples": permutation_examples,
        })

    return output


# ── Main ─────────────────────────────────────────────────────────────────────
@logger.catch
def main():
    # Determine run mode from environment (default: full)
    run_mode = os.environ.get("RUN_MODE", "full")
    logger.info(f"=== CSD Predictive Independence Ablation ===")
    logger.info(f"Run mode: {run_mode}")

    # Load and prepare data
    df = load_unified_dataset()
    df = engineer_features(df)

    near_proportion = df["label"].mean()
    logger.info(f"Near proportion: {near_proportion:.4f}")

    # Verify feature columns exist
    trend_features = [f"{f}_trend3" for f in RAW_CSD_FEATURES]
    all_needed = set(RAW_CSD_FEATURES + DELTA_FEATURES + ZT_FEATURES + trend_features +
                     ["difficulty_level", "difficulty_level_squared", "relative_position_in_range",
                      "relative_dist_to_dstar"])
    missing = all_needed - set(df.columns)
    if missing:
        logger.warning(f"Missing columns: {missing}")

    # ── Gradual Scaling ──────────────────────────────────────────────────
    if run_mode == "mini":
        classifiers = ["logreg"]
        cv_schemes = ["lopo"]
        n_bootstrap = 100
        n_permutations = 0
        run_ablations_4 = False
        run_ablations_5 = False
    elif run_mode == "medium":
        classifiers = ["logreg", "rf", "svm"]
        cv_schemes = ["lopo", "loto"]
        n_bootstrap = 500
        n_permutations = 0
        run_ablations_4 = True
        run_ablations_5 = False
    else:  # full
        classifiers = ["logreg", "rf", "svm"]
        cv_schemes = ["lopo", "loto", "lomo"]
        n_bootstrap = 1000
        n_permutations = 100
        run_ablations_4 = True
        run_ablations_5 = True

    # Run ablations 1-3
    ablation_results = run_ablations_1_to_3(df, classifiers, cv_schemes, n_bootstrap)

    # Log key results
    for abl_name, abl_data in ablation_results.items():
        for key, val in sorted(abl_data.get("results", {}).items()):
            logger.info(f"  {abl_name} | {key}: F1={val['f1_mean']:.4f} [{val['f1_ci_lo']:.4f}, {val['f1_ci_hi']:.4f}]")

    # Run ablation 4 if applicable
    if run_ablations_4:
        ablation_results["ablation_4_incremental"] = run_ablation_4_incremental(df, n_bootstrap)

    # Run ablation 5 if applicable
    if run_ablations_5:
        ablation_results["ablation_5_permutation"] = run_ablation_5_permutation(df, n_permutations, n_bootstrap)

    # Compute summary metrics
    summary_metrics = compute_summary_metrics(ablation_results, near_proportion)
    logger.info(f"Summary metrics: {json.dumps(summary_metrics, indent=2)}")

    # Format output
    output = format_output(ablation_results, summary_metrics, df, run_mode, n_bootstrap, n_permutations)

    # Save output
    out_path = WORKSPACE / "eval_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved output to {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

    # Sanity checks
    logger.info("=== Sanity Checks ===")
    # Check Ablation 1 RF LOPO matches iter_4's csd_raw_rf LOPO F1 ~0.634
    abl1 = ablation_results.get("ablation_1_pure_csd", {}).get("results", {})
    rf_lopo = abl1.get("rf_lopo", {})
    if rf_lopo:
        logger.info(f"Ablation 1 RF LOPO F1: {rf_lopo.get('f1_mean', 0):.4f} (expected ~0.634 from iter_4)")

    logger.info("=== Done ===")
    return output


if __name__ == "__main__":
    main()
