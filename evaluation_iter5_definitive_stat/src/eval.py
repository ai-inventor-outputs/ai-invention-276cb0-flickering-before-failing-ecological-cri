#!/usr/bin/env python3
"""Definitive Statistical Summary Evaluation for CSD Paper Claims.

Computes 5 blocks:
  1. Bootstrap CIs for Success Criteria (SC1 flickering, SC2 mixture R2, SC3 classifier F1)
  2. Effect Sizes (Cohen's d, Cliff's delta) near vs far from boundary
  3. Feature Ablation (LOFO + forward selection)
  4. Sample-Size Sensitivity (variance inflation approach)
  5. Cross-Experiment Consistency (Kendall tau matrix + negative controls)
"""

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
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# ═══════════════════════════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════════════════════════

WORKSPACE = Path(__file__).parent
DEPS = WORKSPACE / "deps"
LOGDIR = WORKSPACE / "logs"
LOGDIR.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOGDIR / "run.log"), rotation="30 MB", level="DEBUG")


def _detect_cpus() -> int:
    try:
        q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        p = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if q > 0:
            return math.ceil(q / p)
    except (FileNotFoundError, ValueError):
        pass
    try:
        parts = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if parts[0] != "max":
            return math.ceil(int(parts[0]) / int(parts[1]))
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
TOTAL_RAM_GB = _container_ram_gb() or 57.0
RAM_BUDGET = int(min(TOTAL_RAM_GB * 0.5, 28) * 1e9)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))

warnings.filterwarnings("ignore")
np.random.seed(42)

B_MAIN = 10000   # bootstrap iters for main CIs
B_EFFECT = 10000  # bootstrap iters for effect sizes
B_SENS = 50       # replicates per N for sensitivity

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════


def safe_float(v, default=np.nan) -> float:
    """Convert value to float, handling None/nan/str."""
    if v is None:
        return default
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (ValueError, TypeError):
        return default


def load_json(path: Path) -> dict:
    logger.info(f"Loading {path.name} ({path.stat().st_size / 1024:.0f} KB)")
    data = json.loads(path.read_text())
    logger.info(f"  keys={list(data.keys()) if isinstance(data, dict) else 'array'}")
    return data


def wilson_ci(n: int, p: float, alpha: float = 0.05):
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    z = stats.norm.ppf(1 - alpha / 2)
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05):
    """Clopper-Pearson exact CI for binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    lo = float(stats.beta.ppf(alpha / 2, k, n - k + 1)) if k > 0 else 0.0
    hi = float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k)) if k < n else 1.0
    return lo, hi


def bootstrap_ci(data, func=np.mean, B=10000, alpha=0.05):
    """Percentile bootstrap CI (vectorised)."""
    arr = np.asarray(data, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return np.nan, np.nan, np.nan
    idx = np.random.randint(0, n, size=(B, n))
    boot = func(arr[idx], axis=1) if func in (np.mean, np.median) else np.array(
        [func(arr[idx[i]]) for i in range(B)]
    )
    point = func(arr)
    return float(point), float(np.percentile(boot, 100 * alpha / 2)), float(np.percentile(boot, 100 * (1 - alpha / 2)))


def bca_ci(data, func=np.mean, B=10000, alpha=0.05):
    """BCa bootstrap CI (for small samples)."""
    arr = np.asarray(data, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < 2:
        return float(func(arr)) if n == 1 else np.nan, np.nan, np.nan
    theta = float(func(arr))
    idx = np.random.randint(0, n, size=(B, n))
    if func in (np.mean, np.median):
        boot = func(arr[idx], axis=1)
    else:
        boot = np.array([func(arr[idx[i]]) for i in range(B)])
    # bias correction
    z0 = stats.norm.ppf(np.clip(np.mean(boot < theta), 1e-8, 1 - 1e-8))
    # acceleration (jackknife)
    jack = np.array([func(np.delete(arr, i)) for i in range(n)])
    jm = jack.mean()
    num = np.sum((jm - jack) ** 3)
    den = 6.0 * np.sum((jm - jack) ** 2) ** 1.5
    a = num / den if abs(den) > 1e-15 else 0.0
    za, zb = stats.norm.ppf(alpha / 2), stats.norm.ppf(1 - alpha / 2)
    a1 = stats.norm.cdf(z0 + (z0 + za) / max(1 - a * (z0 + za), 1e-8))
    a2 = stats.norm.cdf(z0 + (z0 + zb) / max(1 - a * (z0 + zb), 1e-8))
    a1, a2 = np.clip(a1, 0.5 / B, 1 - 0.5 / B), np.clip(a2, 0.5 / B, 1 - 0.5 / B)
    return theta, float(np.percentile(boot, 100 * a1)), float(np.percentile(boot, 100 * a2))


def cohens_d(near, far):
    n1, n2 = len(near), len(far)
    if n1 < 2 or n2 < 2:
        return np.nan
    m1, m2 = np.mean(near), np.mean(far)
    s1, s2 = np.var(near, ddof=1), np.var(far, ddof=1)
    sp = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    return float((m1 - m2) / sp) if sp > 1e-15 else 0.0


def cliffs_delta(near, far):
    n1, n2 = len(near), len(far)
    if n1 == 0 or n2 == 0:
        return np.nan
    # Vectorised
    diff = np.sign(near[:, None] - far[None, :])
    return float(diff.sum() / (n1 * n2))


# ═══════════════════════════════════════════════════════════════════════════════
# DATA EXTRACTION — build unified series list from 3 experiments
# ═══════════════════════════════════════════════════════════════════════════════

INDICATOR_KEYS = ["variance", "dip_statistic", "dip_pvalue",
                  "silhouette_k2", "bimodality_coefficient", "disagreement_rate"]


def _short_model(m: str) -> str:
    return m.split("/")[-1] if "/" in m else m


def extract_arithmetic_series(data: dict) -> list[dict]:
    """From exp_id1_it2: 3 datasets, one per model, each with 24 per-level rows."""
    series = []
    for ds in data["datasets"]:
        model_raw = ds["dataset"].replace("csd_indicators__", "")
        exs = ds["examples"]
        if not exs:
            continue
        d_star = exs[0].get("metadata_d_star")
        levels = []
        for ex in exs:
            levels.append({
                "d": ex["metadata_difficulty_level"],
                "accuracy": safe_float(ex["predict_accuracy"]),
                "variance": safe_float(ex["predict_csd_variance"]),
                "dip_statistic": safe_float(ex["predict_dip_statistic"]),
                "dip_pvalue": safe_float(ex["predict_dip_pvalue"]),
                "silhouette_k2": safe_float(ex["predict_silhouette_k2"]),
                "bimodality_coefficient": safe_float(ex["predict_bimodality_coefficient"]),
                "disagreement_rate": safe_float(ex["predict_disagreement_rate"]),
            })
        levels.sort(key=lambda x: x["d"])
        series.append({
            "task": "arithmetic", "model": model_raw,
            "full_model": exs[0].get("metadata_model", model_raw),
            "d_star": d_star, "levels": levels, "n_per_level": 50,
        })
    return series


def extract_gc_series(data: dict) -> list[dict]:
    """From exp_id3_it2: per-response data, aggregate to per-level."""
    models_meta = {m["model"]: m for m in data["metadata"]["analysis"]["models"]}
    series = []
    for ds in data["datasets"]:
        short = ds["dataset"].replace("graph_coloring_csd_", "")
        # Find full model name
        full_model = None
        for mname in models_meta:
            if short in mname:
                full_model = mname
                break
        if full_model is None:
            continue
        d_star = models_meta[full_model]["d_star"]
        # Aggregate per-level (CSD indicators are identical for all responses at same level)
        level_map: dict[int, dict] = {}
        for ex in ds["examples"]:
            d = ex["metadata_difficulty_level"]
            if d not in level_map:
                level_map[d] = {
                    "d": d,
                    "accuracy": safe_float(ex.get("metadata_csd_accuracy")),
                    "variance": safe_float(ex.get("metadata_csd_embedding_variance")),
                    "dip_statistic": safe_float(ex.get("metadata_csd_dip_statistic")),
                    "dip_pvalue": safe_float(ex.get("metadata_csd_dip_pvalue")),
                    "silhouette_k2": safe_float(ex.get("metadata_csd_silhouette_score")),
                    "bimodality_coefficient": safe_float(ex.get("metadata_csd_bimodality_coefficient")),
                    "disagreement_rate": safe_float(ex.get("metadata_csd_disagreement_rate")),
                }
        levels = sorted(level_map.values(), key=lambda x: x["d"])
        series.append({
            "task": "graph_coloring", "model": short, "full_model": full_model,
            "d_star": d_star, "levels": levels, "n_per_level": 50,
        })
    return series


def extract_syllogistic_series(data: dict) -> list[dict]:
    """From exp_id1_it4: single dataset with 66 rows (3 models x 22 levels)."""
    exs = data["datasets"][0]["examples"]
    by_model: dict[str, dict] = {}
    for ex in exs:
        model = ex["metadata_model"]
        if model not in by_model:
            by_model[model] = {"d_star": ex.get("metadata_d_star"), "levels": []}
        by_model[model]["levels"].append({
            "d": ex["metadata_difficulty"],
            "accuracy": safe_float(ex["predict_accuracy"]),
            "variance": safe_float(ex["predict_csd_variance"]),
            "dip_statistic": safe_float(ex["predict_dip_statistic"]),
            "dip_pvalue": safe_float(ex["predict_dip_pvalue"]),
            "silhouette_k2": safe_float(ex["predict_silhouette_k2"]),
            "bimodality_coefficient": safe_float(ex["predict_bimodality_coefficient"]),
            "disagreement_rate": safe_float(ex["predict_disagreement_rate"]),
        })
    series = []
    for model, info in by_model.items():
        info["levels"].sort(key=lambda x: x["d"])
        series.append({
            "task": "syllogistic", "model": _short_model(model),
            "full_model": model,
            "d_star": info["d_star"], "levels": info["levels"], "n_per_level": 50,
        })
    return series


def extract_all_series(arith_data, gc_data, syl_data) -> list[dict]:
    s = extract_arithmetic_series(arith_data)
    s += extract_gc_series(gc_data)
    s += extract_syllogistic_series(syl_data)
    logger.info(f"Total series extracted: {len(s)}")
    for sr in s:
        logger.info(f"  {sr['task']}/{sr['model']}: d*={sr['d_star']}, levels={len(sr['levels'])}")
    return s


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 1 — SUCCESS CRITERIA WITH BOOTSTRAP CIs
# ═══════════════════════════════════════════════════════════════════════════════

def compute_sc1(all_series: list[dict]) -> tuple[list[dict], dict]:
    """SC1: Flickering as leading indicator — does flickering appear while accuracy > 80%?"""
    logger.info("=== BLOCK 1 / SC1: Flickering as Leading Indicator ===")
    results = []
    for sr in all_series:
        d_star = sr["d_star"]
        if d_star is None or d_star < 3:
            logger.debug(f"  Skip {sr['task']}/{sr['model']}: d*={d_star}")
            continue
        # Find earliest flickering level before d_star
        earliest_d, acc_at_flicker = None, None
        for lv in sr["levels"]:
            if lv["d"] >= d_star:
                break
            if lv["dip_pvalue"] < 0.05 or lv["silhouette_k2"] > 0.3:
                if earliest_d is None:
                    earliest_d = lv["d"]
                    acc_at_flicker = lv["accuracy"]
        if acc_at_flicker is not None and np.isfinite(acc_at_flicker):
            ci_lo, ci_hi = wilson_ci(sr["n_per_level"], acc_at_flicker)
            meets = bool(acc_at_flicker > 0.8)
        else:
            ci_lo, ci_hi = np.nan, np.nan
            meets = False
            acc_at_flicker = acc_at_flicker if acc_at_flicker is not None else np.nan
        results.append({
            "task": sr["task"], "model": sr["model"], "d_star": d_star,
            "earliest_flickering_level": earliest_d,
            "accuracy_at_flickering": acc_at_flicker,
            "ci_lower": ci_lo, "ci_upper": ci_hi,
            "meets": meets,
        })
    total = len(results)
    meeting = sum(r["meets"] for r in results)
    frac = meeting / total if total > 0 else 0.0
    cp_lo, cp_hi = clopper_pearson_ci(meeting, total)
    logger.info(f"  SC1: {meeting}/{total} pairs meet criterion, frac={frac:.3f} CI=[{cp_lo:.3f},{cp_hi:.3f}]")
    agg = {
        "sc1_flickering_pairs_meeting": float(meeting),
        "sc1_total_pairs_with_dstar": float(total),
        "sc1_fraction": frac,
        "sc1_fraction_ci_lower": cp_lo,
        "sc1_fraction_ci_upper": cp_hi,
    }
    return results, agg


def compute_sc2(model_fit_data: dict) -> tuple[list[dict], dict]:
    """SC2: Mixture model R2 across series."""
    logger.info("=== BLOCK 1 / SC2: Mixture Model R2 ===")
    ds = None
    for d in model_fit_data["datasets"]:
        if d["dataset"] == "model_comparison_all_series":
            ds = d
            break
    if ds is None:
        logger.error("model_comparison_all_series not found!")
        return [], {}
    per_series = []
    r2_vals = []
    for ex in ds["examples"]:
        r2 = safe_float(ex.get("predict_mixture_R2"))
        sname = str(ex.get("metadata_task", "?")) + "__" + _short_model(str(ex.get("metadata_model", "?")))
        per_series.append({"series": sname, "r2": r2})
        if np.isfinite(r2):
            r2_vals.append(r2)
    r2a = np.array(r2_vals)
    mean_r2, ci_lo, ci_hi = bootstrap_ci(r2a, np.mean, B=B_MAIN)
    best = float(np.max(r2a)) if len(r2a) > 0 else 0.0
    logger.info(f"  SC2: mean_R2={mean_r2:.4f} CI=[{ci_lo:.4f},{ci_hi:.4f}], best={best:.4f}, n={len(r2a)}")
    agg = {
        "sc2_mean_mixture_r2": mean_r2,
        "sc2_mean_mixture_r2_ci_lower": ci_lo,
        "sc2_mean_mixture_r2_ci_upper": ci_hi,
        "sc2_n_series": float(len(r2a)),
        "sc2_best_series_r2": best,
    }
    return per_series, agg


def compute_sc3(classifier_meta: dict) -> dict:
    """SC3: CSD classifier LOPO F1 from pre-computed results, with BCa CI."""
    logger.info("=== BLOCK 1 / SC3: Classifier LOPO F1 ===")
    cc = classifier_meta["classifier_comparison"]
    csd = cc["csd_zt_reldist_rf"]
    csd_lopo = csd["lopo_f1"]
    csd_loto = csd["loto_f1"]
    csd_lomo = csd["lomo_f1"]
    # Best baselines
    spuq_best = cc["spuq_accuracy_rf"]
    disag_best = cc["disagreement_only_logreg"]
    best_base_lopo = max(spuq_best["lopo_f1"], disag_best["lopo_f1"])
    # BCa CI using LOPO/LOTO/LOMO as 3 estimates
    csd_ests = np.array([csd_lopo, csd_loto, csd_lomo])
    _, ci_lo, ci_hi = bca_ci(csd_ests, np.mean, B=B_MAIN)
    improvement = csd_lopo - best_base_lopo
    improvement_pct = improvement / best_base_lopo * 100 if best_base_lopo > 0 else 0.0
    # Paired improvement CI
    base_ests = np.array([
        max(spuq_best["lopo_f1"], disag_best["lopo_f1"]),
        max(spuq_best["loto_f1"], disag_best.get("loto_f1", disag_best["lopo_f1"])),
        max(spuq_best["lomo_f1"], disag_best.get("lomo_f1", disag_best["lopo_f1"])),
    ])
    diff_ests = csd_ests - base_ests
    _, imp_ci_lo, imp_ci_hi = bca_ci(diff_ests, np.mean, B=B_MAIN)
    logger.info(f"  SC3: CSD LOPO F1={csd_lopo:.4f} CI=[{ci_lo:.4f},{ci_hi:.4f}]")
    logger.info(f"  SC3: best_baseline={best_base_lopo:.4f}, improvement={improvement:.4f} ({improvement_pct:.1f}%)")
    return {
        "sc3_csd_lopo_f1": csd_lopo,
        "sc3_csd_lopo_f1_ci_lower": ci_lo,
        "sc3_csd_lopo_f1_ci_upper": ci_hi,
        "sc3_best_baseline_f1": best_base_lopo,
        "sc3_f1_improvement": improvement,
        "sc3_f1_improvement_ci_lower": imp_ci_lo,
        "sc3_f1_improvement_ci_upper": imp_ci_hi,
        "sc3_improvement_pct": improvement_pct,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 2 — EFFECT SIZES
# ═══════════════════════════════════════════════════════════════════════════════

def compute_effect_sizes(all_series: list[dict]) -> tuple[list[dict], dict]:
    """Cohen's d and Cliff's delta for near- vs far-boundary zones."""
    logger.info("=== BLOCK 2: Effect Sizes ===")
    indicators = ["variance", "dip_statistic", "silhouette_k2",
                  "bimodality_coefficient", "disagreement_rate"]
    results = []
    for sr in all_series:
        d_star = sr["d_star"]
        if d_star is None or d_star < 5:
            continue
        near = [lv for lv in sr["levels"] if d_star - 3 <= lv["d"] <= d_star]
        far = [lv for lv in sr["levels"] if lv["d"] <= d_star - 5]
        if len(far) < 3 or len(near) < 1:
            continue
        for ind in indicators:
            nv = np.array([lv[ind] for lv in near if np.isfinite(lv[ind])])
            fv = np.array([lv[ind] for lv in far if np.isfinite(lv[ind])])
            if len(nv) < 1 or len(fv) < 2:
                continue
            cd = cohens_d(nv, fv)
            cld = cliffs_delta(nv, fv)
            # Vectorised bootstrap for Cohen's d
            nn, nf = len(nv), len(fv)
            ni = np.random.randint(0, nn, size=(B_EFFECT, nn))
            fi = np.random.randint(0, nf, size=(B_EFFECT, nf))
            nb, fb = nv[ni], fv[fi]
            mn, mf = nb.mean(1), fb.mean(1)
            vn = nb.var(1, ddof=1) if nn > 1 else np.zeros(B_EFFECT)
            vf = fb.var(1, ddof=1) if nf > 1 else np.zeros(B_EFFECT)
            sp = np.sqrt(((nn - 1) * vn + (nf - 1) * vf) / max(nn + nf - 2, 1))
            sp = np.maximum(sp, 1e-15)
            cd_boot = (mn - mf) / sp
            cd_lo, cd_hi = float(np.percentile(cd_boot, 2.5)), float(np.percentile(cd_boot, 97.5))
            results.append({
                "task": sr["task"], "model": sr["model"], "indicator": ind,
                "cohen_d": cd, "cohen_d_ci_lower": cd_lo, "cohen_d_ci_upper": cd_hi,
                "cliff_delta": cld, "n_near": nn, "n_far": nf,
            })
    # Aggregate by indicator
    agg = {}
    for ind in indicators:
        cds = [r["cohen_d"] for r in results if r["indicator"] == ind and np.isfinite(r["cohen_d"])]
        clds = [r["cliff_delta"] for r in results if r["indicator"] == ind and np.isfinite(r["cliff_delta"])]
        agg[f"effect_size_mean_cohen_d_{ind}"] = float(np.mean(cds)) if cds else 0.0
        agg[f"effect_size_mean_cliff_delta_{ind}"] = float(np.mean(clds)) if clds else 0.0
    logger.info(f"  Computed {len(results)} effect-size cells")
    return results, agg


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 3 — FEATURE ABLATION
# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_NAMES = ["variance_zt", "dip_statistic_zt", "silhouette_k2_zt",
                 "bimodality_coefficient_zt", "disagreement_rate_zt",
                 "relative_dist_to_dstar"]
RAW_FEAT = ["variance", "dip_statistic", "silhouette_k2",
            "bimodality_coefficient", "disagreement_rate"]


def build_classifier_data(all_series: list[dict]):
    """Build feature matrix + labels matching csd_zt_reldist variant.

    Labelling (matched to exp_id2_it4):
      near = d >= d_star - 2   (label 1)
      safe = d <= d_star - 3   (label 0)
    Z-score normalise CSD indicators within each task, then add relative_dist = d / d_star.
    Only include arithmetic + graph_coloring series with d_star >= 5 (skip gpt-4o-mini arith d*=2).
    """
    logger.info("Building classifier feature matrix (csd_zt_reldist)")
    # Filter to arithmetic + graph_coloring with valid d_star
    valid = [s for s in all_series
             if s["task"] in ("arithmetic", "graph_coloring") and s["d_star"] is not None and s["d_star"] >= 5]
    if not valid:
        logger.warning("No valid series for classifier reconstruction!")
        return None, None, None

    # Group rows by task for z-score normalisation
    task_rows: dict[str, list] = {}
    for sr in valid:
        d_star = sr["d_star"]
        task = sr["task"]
        if task not in task_rows:
            task_rows[task] = []
        for lv in sr["levels"]:
            d = lv["d"]
            # Label
            if d >= d_star - 2:
                label = 1
            elif d <= d_star - 3:
                label = 0
            else:
                continue  # should not happen with this scheme but safety
            feats = [lv[f] for f in RAW_FEAT]
            task_rows[task].append({
                "feats": feats, "rel_dist": d / d_star, "label": label,
                "group": f"{sr['task']}__{sr['model']}",
            })

    # Z-score within each task
    all_rows = []
    for task, rows in task_rows.items():
        fm = np.array([r["feats"] for r in rows])
        fm = np.nan_to_num(fm, nan=0.0)
        mu = fm.mean(axis=0)
        sd = fm.std(axis=0)
        sd[sd < 1e-12] = 1.0
        fz = (fm - mu) / sd
        for i, r in enumerate(rows):
            all_rows.append({
                "x": list(fz[i]) + [r["rel_dist"]],
                "y": r["label"],
                "g": r["group"],
            })
    X = np.array([r["x"] for r in all_rows])
    y = np.array([r["y"] for r in all_rows])
    groups = [r["g"] for r in all_rows]
    n_near = int(y.sum())
    n_safe = int((1 - y).sum())
    logger.info(f"  X shape={X.shape}, near={n_near}, safe={n_safe}, groups={len(set(groups))}")
    return X, y, groups


def lopo_cv(X, y, groups, mask=None, n_est=100):
    """Leave-One-Pair-Out CV, return (macro_f1, per_fold_f1s)."""
    unique = sorted(set(groups))
    fold_f1s = []
    for holdout in unique:
        ti = [i for i, g in enumerate(groups) if g != holdout]
        vi = [i for i, g in enumerate(groups) if g == holdout]
        if not vi or not ti:
            continue
        Xtr, ytr = X[ti], y[ti]
        Xte, yte = X[vi], y[vi]
        if mask is not None:
            Xtr, Xte = Xtr[:, mask], Xte[:, mask]
        if len(np.unique(ytr)) < 2:
            fold_f1s.append(0.0)
            continue
        clf = RandomForestClassifier(n_estimators=n_est, random_state=42, n_jobs=1)
        clf.fit(Xtr, ytr)
        yp = clf.predict(Xte)
        fold_f1s.append(float(f1_score(yte, yp, zero_division=0)))
    macro = float(np.mean(fold_f1s)) if fold_f1s else 0.0
    return macro, fold_f1s


def compute_ablation(X, y, groups) -> dict:
    """LOFO + forward selection."""
    logger.info("=== BLOCK 3: Feature Ablation ===")
    nf = X.shape[1]
    full_f1, full_folds = lopo_cv(X, y, groups)
    logger.info(f"  Full model LOPO F1={full_f1:.4f} (folds={[round(f,3) for f in full_folds]})")

    # --- LOFO ---
    lofo = []
    for i in range(nf):
        mask = [j for j in range(nf) if j != i]
        minus_f1, _ = lopo_cv(X, y, groups, mask=mask)
        deg = full_f1 - minus_f1
        lofo.append({"feature": FEATURE_NAMES[i], "f1_without": minus_f1, "degradation": deg})
        logger.info(f"    LOFO -{FEATURE_NAMES[i]}: F1={minus_f1:.4f}, deg={deg:+.4f}")
    most_important = max(lofo, key=lambda r: r["degradation"])["feature"]

    # --- Forward selection ---
    selected: list[int] = []
    remaining = list(range(nf))
    forward = []
    while remaining:
        best_i, best_f1 = remaining[0], -1.0
        for fi in remaining:
            f1v, _ = lopo_cv(X, y, groups, mask=selected + [fi])
            if f1v > best_f1:
                best_f1, best_i = f1v, fi
        selected.append(best_i)
        remaining.remove(best_i)
        forward.append({
            "n": len(selected),
            "features": [FEATURE_NAMES[j] for j in selected],
            "f1": best_f1,
        })
        logger.info(f"    Forward +{FEATURE_NAMES[best_i]}: n={len(selected)} F1={best_f1:.4f}")
        if best_f1 >= 0.95 * full_f1:
            break
    min_set = forward[-1] if forward else {"n": nf, "f1": full_f1, "features": FEATURE_NAMES}
    logger.info(f"  Most important: {most_important}, min set: {min_set['n']} feats -> F1={min_set['f1']:.4f}")
    return {
        "full_f1": full_f1, "full_folds": full_folds,
        "lofo": lofo, "forward": forward,
        "most_important": most_important,
        "min_set_n": min_set["n"], "min_set_f1": min_set["f1"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 4 — SAMPLE-SIZE SENSITIVITY
# ═══════════════════════════════════════════════════════════════════════════════

def compute_sensitivity(X, y, groups, all_series: list[dict]) -> tuple[list[dict], list[dict], int]:
    """Variance-inflation approach: simulate reduced N by adding noise."""
    logger.info("=== BLOCK 4: Sample-Size Sensitivity ===")
    N_vals = [10, 15, 20, 25, 30, 40, 50]
    N_base = 50
    base_f1, _ = lopo_cv(X, y, groups)

    f1_results = []
    for N in N_vals:
        if N == N_base:
            f1_results.append({"N": N, "mean_f1": base_f1, "std_f1": 0.0, "ratio": 1.0})
            continue
        # Additional noise std from reducing N: sqrt(1/N - 1/50)
        noise_scale = math.sqrt(1.0 / N - 1.0 / N_base)
        rep_f1s = []
        for _ in range(B_SENS):
            X_noisy = X + np.random.randn(*X.shape) * noise_scale
            f1v, _ = lopo_cv(X_noisy, y, groups, n_est=50)  # fewer trees for speed
            rep_f1s.append(f1v)
        mf = float(np.mean(rep_f1s))
        sf = float(np.std(rep_f1s))
        f1_results.append({"N": N, "mean_f1": mf, "std_f1": sf, "ratio": mf / base_f1 if base_f1 > 0 else 0.0})
        logger.info(f"  N={N}: F1={mf:.4f}+/-{sf:.4f}, ratio={mf/base_f1:.3f}")

    # Dip detection rate (analytical approximation)
    dip_results = []
    # Count near-boundary dip detections at N=50
    n_det, n_tot = 0, 0
    for sr in all_series:
        d_star = sr["d_star"]
        if d_star is None or d_star < 5:
            continue
        for lv in sr["levels"]:
            if d_star - 3 <= lv["d"] <= d_star:
                n_tot += 1
                if lv["dip_pvalue"] < 0.05:
                    n_det += 1
    base_dip_rate = n_det / n_tot if n_tot > 0 else 0.0
    for N in N_vals:
        # Power scales approximately as (N/N_base)^0.3
        rate = min(1.0, base_dip_rate * (N / N_base) ** 0.3) if base_dip_rate > 0 else 0.0
        dip_results.append({"N": N, "dip_rate": rate})

    # Minimum viable N: smallest N where ratio >= 0.9
    min_N = N_base
    for r in f1_results:
        if r["ratio"] >= 0.9:
            min_N = r["N"]
            break
    logger.info(f"  Minimum viable N: {min_N}")
    return f1_results, dip_results, min_N


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 5 — CROSS-EXPERIMENT CONSISTENCY
# ═══════════════════════════════════════════════════════════════════════════════

EXPECTED_DIR = {
    "variance": 1, "dip_statistic": 1, "silhouette_k2": 1,
    "bimodality_coefficient": 1, "disagreement_rate": 1,
}  # 1 = expected increasing toward boundary


def compute_consistency(all_series: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """Kendall tau matrix + negative controls."""
    logger.info("=== BLOCK 5: Cross-Experiment Consistency ===")
    indicators = list(EXPECTED_DIR.keys())
    results = []

    for sr in all_series:
        d_star = sr["d_star"]
        for ind in indicators:
            if d_star is None:
                results.append({
                    "task": sr["task"], "model": sr["model"], "indicator": ind,
                    "tau": np.nan, "p": np.nan, "direction": "none",
                    "significant": False, "correct": False, "no_boundary": True,
                })
                continue
            pre = [lv for lv in sr["levels"] if lv["d"] < d_star]
            vals = [(lv["d"], lv[ind]) for lv in pre if np.isfinite(lv[ind])]
            if len(vals) < 4:
                results.append({
                    "task": sr["task"], "model": sr["model"], "indicator": ind,
                    "tau": np.nan, "p": np.nan, "direction": "insufficient",
                    "significant": False, "correct": False, "no_boundary": False,
                })
                continue
            ds = np.array([v[0] for v in vals])
            vs = np.array([v[1] for v in vals])
            tau, p = stats.kendalltau(ds, vs)
            sig = bool(p < 0.05)
            direction = "increasing" if tau > 0 else "decreasing"
            expected_inc = EXPECTED_DIR[ind] == 1
            correct = sig and ((tau > 0) == expected_inc)
            results.append({
                "task": sr["task"], "model": sr["model"], "indicator": ind,
                "tau": float(tau) if np.isfinite(tau) else 0.0,
                "p": float(p) if np.isfinite(p) else 1.0,
                "direction": direction, "significant": sig,
                "correct": correct, "no_boundary": False,
            })

    # Negative controls: indicators should be lower far from boundary
    neg_results = []
    for sr in all_series:
        d_star = sr["d_star"]
        if d_star is None or d_star < 6:
            continue
        far = [lv for lv in sr["levels"] if lv["d"] < d_star / 2]
        near = [lv for lv in sr["levels"] if d_star - 3 <= lv["d"] <= d_star]
        if len(far) < 2 or len(near) < 1:
            continue
        passes, total = 0, 0
        for ind in indicators:
            nv = [lv[ind] for lv in near if np.isfinite(lv[ind])]
            fv = [lv[ind] for lv in far if np.isfinite(lv[ind])]
            if len(nv) < 1 or len(fv) < 1:
                continue
            total += 1
            if len(nv) >= 2 and len(fv) >= 2:
                _, p = stats.mannwhitneyu(nv, fv, alternative="greater")
                if p < 0.05:
                    passes += 1
            elif np.mean(nv) > np.mean(fv):
                passes += 1
        neg_results.append({
            "task": sr["task"], "model": sr["model"],
            "passes": passes, "total": total,
            "rate": passes / total if total > 0 else 0.0,
        })

    # Aggregates
    valid = [r for r in results if not r["no_boundary"] and r["direction"] not in ("insufficient",)]
    n_sig = sum(r["significant"] for r in valid)
    n_cor = sum(r["correct"] for r in valid)
    n_all = len(valid)
    neg_rate = float(np.mean([r["rate"] for r in neg_results])) if neg_results else 0.0
    agg = {
        "consistency_fraction_significant": n_sig / n_all if n_all > 0 else 0.0,
        "consistency_fraction_correct_direction": n_cor / n_all if n_all > 0 else 0.0,
        "negative_control_pass_rate": neg_rate,
        "consistency_n_cells": float(n_all),
    }
    logger.info(f"  {n_sig}/{n_all} significant, {n_cor}/{n_all} correct direction, neg_ctrl={neg_rate:.3f}")
    return results, neg_results, agg


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT FORMATTING
# ═══════════════════════════════════════════════════════════════════════════════

def _clean_num(v) -> float:
    """Ensure value is a finite float for JSON schema compliance."""
    if isinstance(v, (np.integer, int)):
        return float(int(v))
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return f if math.isfinite(f) else 0.0
    return 0.0


def format_output(
    sc1_res, sc1_agg, sc2_res, sc2_agg, sc3_agg,
    eff_res, eff_agg, abl_res,
    sens_res, dip_res, min_N,
    cons_res, neg_res, cons_agg,
) -> dict:
    # ── metrics_agg (all values must be JSON number) ──
    ma: dict[str, float] = {}
    for d in [sc1_agg, sc2_agg, sc3_agg, eff_agg, cons_agg]:
        for k, v in d.items():
            ma[k] = _clean_num(v)
    if abl_res:
        ma["ablation_full_f1"] = _clean_num(abl_res["full_f1"])
        ma["ablation_minimum_viable_feature_count"] = _clean_num(abl_res["min_set_n"])
        ma["ablation_min_set_f1"] = _clean_num(abl_res["min_set_f1"])
        idx = FEATURE_NAMES.index(abl_res["most_important"]) if abl_res["most_important"] in FEATURE_NAMES else -1
        ma["ablation_most_important_feature_idx"] = _clean_num(idx)
    ma["sensitivity_minimum_viable_N"] = _clean_num(min_N)
    for r in sens_res:
        ma[f"sensitivity_f1_at_N{r['N']}"] = _clean_num(r["mean_f1"])
    for r in dip_res:
        ma[f"sensitivity_dip_rate_at_N{r['N']}"] = _clean_num(r["dip_rate"])

    # ── datasets ──
    datasets = []

    # 1. success_criteria_bootstrap
    sc_ex = []
    for r in sc1_res:
        sc_ex.append({
            "input": f"SC1 flickering: {r['task']}/{r['model']}",
            "output": f"d*={r['d_star']}, earliest_flicker={r['earliest_flickering_level']}, meets={r['meets']}",
            "eval_sc1_meets_criterion": 1.0 if r["meets"] else 0.0,
            "eval_sc1_accuracy_at_flickering": _clean_num(r["accuracy_at_flickering"]),
            "eval_sc1_ci_lower": _clean_num(r["ci_lower"]),
            "eval_sc1_ci_upper": _clean_num(r["ci_upper"]),
            "metadata_task": r["task"],
            "metadata_model": r["model"],
            "metadata_d_star": r["d_star"],
        })
    for r in sc2_res:
        sc_ex.append({
            "input": f"SC2 mixture R2: {r['series']}",
            "output": f"R2={_clean_num(r['r2']):.4f}",
            "eval_sc2_mixture_r2": _clean_num(r["r2"]),
            "metadata_series": r["series"],
        })
    if sc_ex:
        datasets.append({"dataset": "success_criteria_bootstrap", "examples": sc_ex})

    # 2. effect_sizes
    eff_ex = []
    for r in eff_res:
        eff_ex.append({
            "input": f"Effect {r['indicator']}: {r['task']}/{r['model']}",
            "output": f"d={_clean_num(r['cohen_d']):.3f}, delta={_clean_num(r['cliff_delta']):.3f}",
            "eval_cohen_d": _clean_num(r["cohen_d"]),
            "eval_cohen_d_ci_lower": _clean_num(r["cohen_d_ci_lower"]),
            "eval_cohen_d_ci_upper": _clean_num(r["cohen_d_ci_upper"]),
            "eval_cliff_delta": _clean_num(r["cliff_delta"]),
            "metadata_task": r["task"],
            "metadata_model": r["model"],
            "metadata_indicator": r["indicator"],
        })
    if eff_ex:
        datasets.append({"dataset": "effect_sizes", "examples": eff_ex})

    # 3. feature_ablation
    abl_ex = []
    if abl_res:
        for r in abl_res["lofo"]:
            abl_ex.append({
                "input": f"LOFO remove {r['feature']}",
                "output": f"F1={r['f1_without']:.4f}, deg={r['degradation']:+.4f}",
                "eval_f1_without": _clean_num(r["f1_without"]),
                "eval_degradation": _clean_num(r["degradation"]),
                "metadata_feature": r["feature"],
                "metadata_type": "lofo",
            })
        for r in abl_res["forward"]:
            abl_ex.append({
                "input": f"Forward select {r['n']} features",
                "output": f"F1={r['f1']:.4f}",
                "eval_forward_f1": _clean_num(r["f1"]),
                "eval_n_features": _clean_num(r["n"]),
                "metadata_features": str(r["features"]),
                "metadata_type": "forward",
            })
    if abl_ex:
        datasets.append({"dataset": "feature_ablation", "examples": abl_ex})

    # 4. sample_size_sensitivity
    sens_ex = []
    for r in sens_res:
        sens_ex.append({
            "input": f"Sample size N={r['N']}",
            "output": f"F1={r['mean_f1']:.4f}, ratio={r['ratio']:.3f}",
            "eval_mean_f1": _clean_num(r["mean_f1"]),
            "eval_std_f1": _clean_num(r["std_f1"]),
            "eval_f1_ratio": _clean_num(r["ratio"]),
            "metadata_N": r["N"],
        })
    if sens_ex:
        datasets.append({"dataset": "sample_size_sensitivity", "examples": sens_ex})

    # 5. cross_experiment_consistency
    cons_ex = []
    for r in cons_res:
        cons_ex.append({
            "input": f"Consistency {r['indicator']}: {r['task']}/{r['model']}",
            "output": f"tau={_clean_num(r['tau']):.3f}, p={_clean_num(r['p']):.4f}, sig={r['significant']}",
            "eval_kendall_tau": _clean_num(r["tau"]),
            "eval_p_value": _clean_num(r["p"]),
            "eval_significant": 1.0 if r["significant"] else 0.0,
            "eval_correct_direction": 1.0 if r["correct"] else 0.0,
            "metadata_task": r["task"],
            "metadata_model": r["model"],
            "metadata_indicator": r["indicator"],
            "metadata_direction": r["direction"],
        })
    if cons_ex:
        datasets.append({"dataset": "cross_experiment_consistency", "examples": cons_ex})

    return {"metrics_agg": ma, "datasets": datasets}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

@logger.catch
def main():
    t0 = time.time()
    logger.info(f"CSD Evaluation — workspace={WORKSPACE}")
    logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.0f}GB RAM, budget={RAM_BUDGET/1e9:.1f}GB")

    # ── Load data ──
    arith = load_json(DEPS / "exp_id1_it2__opus" / "full_method_out.json")
    gc = load_json(DEPS / "exp_id3_it2__opus" / "full_method_out.json")
    syl = load_json(DEPS / "exp_id1_it4__opus" / "full_method_out.json")
    clf_data = load_json(DEPS / "exp_id2_it4__opus" / "full_method_out.json")
    mfit = load_json(DEPS / "exp_id3_it4__opus" / "full_method_out.json")
    logger.info(f"Data loaded in {time.time()-t0:.1f}s")

    # ── Extract unified series ──
    all_series = extract_all_series(arith, gc, syl)

    # ── Block 1: Success Criteria ──
    sc1_res, sc1_agg = compute_sc1(all_series)
    sc2_res, sc2_agg = compute_sc2(mfit)
    sc3_agg = compute_sc3(clf_data["metadata"])

    # ── Block 2: Effect Sizes ──
    eff_res, eff_agg = compute_effect_sizes(all_series)

    # ── Block 3: Feature Ablation ──
    X, y, groups = build_classifier_data(all_series)
    abl_res = None
    if X is not None:
        abl_res = compute_ablation(X, y, groups)

    # ── Block 4: Sensitivity ──
    if X is not None:
        sens_res, dip_res, min_N = compute_sensitivity(X, y, groups, all_series)
    else:
        sens_res = [{"N": 50, "mean_f1": 0.0, "std_f1": 0.0, "ratio": 1.0}]
        dip_res = [{"N": 50, "dip_rate": 0.0}]
        min_N = 50

    # ── Block 5: Consistency ──
    cons_res, neg_res, cons_agg = compute_consistency(all_series)

    # ── Format + write ──
    output = format_output(
        sc1_res, sc1_agg, sc2_res, sc2_agg, sc3_agg,
        eff_res, eff_agg, abl_res,
        sens_res, dip_res, min_N,
        cons_res, neg_res, cons_agg,
    )
    out_path = WORKSPACE / "eval_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Wrote {out_path.name} ({out_path.stat().st_size/1024:.1f} KB)")

    # ── Summary ──
    elapsed = time.time() - t0
    logger.info(f"=== DONE in {elapsed:.1f}s ===")
    logger.info(f"metrics_agg ({len(output['metrics_agg'])} keys):")
    for k, v in sorted(output["metrics_agg"].items()):
        logger.info(f"  {k}: {v}")
    for ds in output["datasets"]:
        logger.info(f"  dataset '{ds['dataset']}': {len(ds['examples'])} examples")


if __name__ == "__main__":
    main()
