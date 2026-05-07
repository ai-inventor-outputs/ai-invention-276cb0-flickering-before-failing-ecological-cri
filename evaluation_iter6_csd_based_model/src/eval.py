#!/usr/bin/env python3
"""CSD-Based Model Routing Simulation: Downstream Value Quantification.

Simulates model-routing deployment using per-level accuracy and CSD indicator
data from two completed experiments (arithmetic x 3 models, graph coloring x 3
models). Compares four routing policies (always-cheap, always-capable,
CSD-monitored, oracle) across 100 Monte Carlo runs per policy to quantify
accuracy-vs-cost tradeoffs.
"""

import gc
import json
import math
import os
import resource
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

# ── Logging ──────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")


# ── Hardware detection ───────────────────────────────────────────────────────
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
TOTAL_RAM_GB = _container_ram_gb() or 57.0
RAM_BUDGET = int(TOTAL_RAM_GB * 0.4 * 1e9)  # 40% of container RAM
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, budget={RAM_BUDGET/1e9:.1f} GB")

# ── Constants ────────────────────────────────────────────────────────────────
N_QUERIES = 1000        # queries per MC run
N_MC_RUNS = 100         # Monte Carlo runs per policy
COST_CHEAP = 0.001      # $/query for cheap model
COST_CAPABLE = 0.010    # $/query for capable model
BATCH_SIZES = [10, 15, 20, 30, 50]
CSD_ALARM_THRESHOLD = 1.3   # alarm fires when variance > baseline * threshold
CSD_ALARM_MIN_LEVELS = 3    # need at least 3 levels to establish baseline
DIFFICULTY_DISTRIBUTIONS = ["uniform", "beta_easy", "beta_hard"]
POLICIES = ["always_cheap", "always_capable", "oracle", "csd_monitoring"]


# ── Data Loading ─────────────────────────────────────────────────────────────
def load_arithmetic_data(path: str) -> dict[str, dict]:
    """Load arithmetic CSD data: per-model per-level metrics.

    Returns dict mapping short model name -> {levels: {d: {accuracy, csd_variance, ...}}, d_star: int}
    """
    data = json.loads(Path(path).read_text())
    models: dict[str, dict] = {}
    model_summaries = data.get("metadata", {}).get("model_summaries", {})

    for ds in data["datasets"]:
        model_short = ds["dataset"].replace("csd_indicators__", "")
        levels: dict[int, dict] = {}
        d_star = None

        for ex in ds["examples"]:
            d = ex["metadata_difficulty_level"]
            levels[d] = {
                "accuracy": float(ex["predict_accuracy"]),
                "csd_variance": float(ex["predict_csd_variance"]),
                "dip_statistic": float(ex["predict_dip_statistic"]),
                "dip_pvalue": float(ex["predict_dip_pvalue"]),
                "silhouette_k2": float(ex["predict_silhouette_k2"]),
                "bimodality_coefficient": float(ex["predict_bimodality_coefficient"]),
                "disagreement_rate": float(ex["predict_disagreement_rate"]),
            }
            d_star = ex["metadata_d_star"]

        # Also try metadata model_summaries for d_star
        for full_name, summary in model_summaries.items():
            if model_short in full_name:
                d_star = summary.get("d_star", d_star)
                break

        models[model_short] = {"levels": levels, "d_star": d_star}

    return models


def load_graph_coloring_data(path: str) -> dict[str, dict]:
    """Load graph coloring data: aggregate per-model per-level metrics.

    Returns dict mapping short model name -> {levels: {d: {accuracy, csd_variance, ...}}, d_star: int}
    """
    data = json.loads(Path(path).read_text())
    models: dict[str, dict] = {}

    # Extract d_star from metadata
    model_d_stars: dict[str, int] = {}
    for m in data.get("metadata", {}).get("analysis", {}).get("models", []):
        model_d_stars[m["model"]] = m["d_star"]

    for ds in data["datasets"]:
        model_full = ds["examples"][0]["metadata_model"]
        model_short = model_full.split("/")[-1]

        level_correct: dict[int, list[float]] = defaultdict(list)
        level_csd: dict[int, dict] = {}

        for ex in ds["examples"]:
            d = ex["metadata_difficulty_level"]
            correct = 1.0 if ex["predict_is_correct"] == "true" else 0.0
            level_correct[d].append(correct)
            if d not in level_csd:
                level_csd[d] = {
                    "csd_variance": ex["metadata_csd_embedding_variance"],
                    "dip_statistic": ex["metadata_csd_dip_statistic"],
                    "dip_pvalue": ex["metadata_csd_dip_pvalue"],
                    "silhouette_score": ex.get("metadata_csd_silhouette_score", 0.0),
                    "bimodality_coefficient": ex.get("metadata_csd_bimodality_coefficient", 0.0),
                    "disagreement_rate": ex.get("metadata_csd_disagreement_rate", 0.0),
                }

        levels: dict[int, dict] = {}
        for d in sorted(level_correct.keys()):
            acc = sum(level_correct[d]) / len(level_correct[d])
            levels[d] = {"accuracy": acc, **level_csd[d]}

        d_star = model_d_stars.get(model_full)
        models[model_short] = {"levels": levels, "d_star": d_star}

    return models


# ── CSD Alarm ────────────────────────────────────────────────────────────────
def compute_csd_alarm_level(
    cheap_indicators: dict[int, dict],
    sorted_levels: list[int],
    batch_size: int,
    rng: np.random.Generator,
) -> int | None:
    """Multi-indicator CSD alarm using variance CUSUM + disagreement rate.

    Uses three alarm channels (first to trigger wins):
    1. CUSUM on embedding variance (detects upward shift in response diversity)
    2. Disagreement rate exceeding baseline * threshold
    3. Dip-significant bimodality with elevated variance

    Sampling noise scaled by 1/sqrt(batch_size) models the effect of smaller
    batch sizes on indicator reliability.

    Returns difficulty level where alarm fires, or None if never fires.
    """
    if len(sorted_levels) < CSD_ALARM_MIN_LEVELS + 1:
        return None

    # Collect noisy observations for all indicators
    obs_var: dict[int, float] = {}
    obs_disagr: dict[int, float] = {}
    obs_dip_p: dict[int, float] = {}

    noise_factor = 1.0 / math.sqrt(max(batch_size, 2))

    for d in sorted_levels:
        ind = cheap_indicators.get(d, {})

        # Variance with noise
        true_var = ind.get("csd_variance", 0.1)
        obs_var[d] = max(true_var + rng.normal(0, true_var * noise_factor * 0.3), 0.001)

        # Disagreement rate with noise
        true_dis = ind.get("disagreement_rate", 0.5)
        obs_disagr[d] = np.clip(true_dis + rng.normal(0, 0.05 * noise_factor), 0.0, 1.0)

        # Dip p-value (smaller batch = less power)
        true_dip_p = ind.get("dip_pvalue", 1.0)
        if true_dip_p is None:
            true_dip_p = 1.0
        power_adj = math.sqrt(50.0 / max(batch_size, 2))
        obs_dip_p[d] = min(true_dip_p * power_adj, 1.0)

    # Baselines from first few levels
    bl = sorted_levels[:CSD_ALARM_MIN_LEVELS]

    # Variance CUSUM baseline
    var_bl = [obs_var[d] for d in bl]
    var_target = float(np.mean(var_bl))
    var_std = max(float(np.std(var_bl)), var_target * 0.03)

    # Disagreement rate baseline
    dis_bl = [obs_disagr[d] for d in bl]
    dis_target = float(np.mean(dis_bl))
    dis_std = max(float(np.std(dis_bl)), 0.03)

    # CUSUM state
    cusum_var = 0.0
    cusum_h_var = 2.5 * var_std
    allow_var = 0.5 * var_std

    # Disagreement CUSUM state
    cusum_dis = 0.0
    cusum_h_dis = 2.5 * dis_std
    allow_dis = 0.5 * dis_std

    for d in sorted_levels[CSD_ALARM_MIN_LEVELS:]:
        # Channel 1: Variance CUSUM
        cusum_var = max(0.0, cusum_var + obs_var[d] - var_target - allow_var)
        if cusum_var > cusum_h_var:
            return d

        # Channel 2: Disagreement CUSUM
        cusum_dis = max(0.0, cusum_dis + obs_disagr[d] - dis_target - allow_dis)
        if cusum_dis > cusum_h_dis:
            return d

        # Channel 3: Dip-based bimodality + elevated variance
        if obs_dip_p[d] < 0.05 and obs_var[d] > var_target * 1.05:
            return d

    return None


# ── Monte Carlo Simulation ───────────────────────────────────────────────────
def _nearest_level(difficulty_val: int, sorted_levels: list[int]) -> int:
    """Map an integer difficulty to the nearest available level."""
    idx = np.searchsorted(sorted_levels, difficulty_val)
    if idx == 0:
        return sorted_levels[0]
    if idx >= len(sorted_levels):
        return sorted_levels[-1]
    lo, hi = sorted_levels[idx - 1], sorted_levels[idx]
    return lo if abs(difficulty_val - lo) <= abs(difficulty_val - hi) else hi


def generate_difficulties(
    rng: np.random.Generator,
    dist_name: str,
    n: int,
    min_level: int,
    max_level: int,
) -> np.ndarray:
    """Generate n query difficulty values from the specified distribution."""
    if dist_name == "uniform":
        raw = rng.uniform(0, 1, n)
    elif dist_name == "beta_easy":
        raw = rng.beta(2, 5, n)
    elif dist_name == "beta_hard":
        raw = rng.beta(5, 2, n)
    else:
        raw = rng.uniform(0, 1, n)

    scaled = np.round(raw * (max_level - min_level) + min_level).astype(int)
    return np.clip(scaled, min_level, max_level)


def run_single_mc(args: tuple) -> dict:
    """Run a single Monte Carlo simulation for a given policy + parameters.

    This function is designed to be called in a ProcessPoolExecutor.
    """
    (
        mc_idx, policy_name, cheap_levels_ser, capable_levels_ser,
        cheap_indicators_ser, sorted_levels, d_star_cheap,
        batch_size, dist_name, seed,
    ) = args

    rng = np.random.default_rng(seed)
    min_level = sorted_levels[0]
    max_level = sorted_levels[-1]

    # Generate query difficulties
    difficulties = generate_difficulties(rng, dist_name, N_QUERIES, min_level, max_level)

    # Compute CSD alarm level (only relevant for csd_monitoring)
    alarm_level = None
    if policy_name == "csd_monitoring":
        alarm_level = compute_csd_alarm_level(
            cheap_indicators_ser, sorted_levels, batch_size, rng,
        )

    # Simulate queries
    n_correct = 0
    total_cost = 0.0
    n_cheap = 0
    n_capable = 0

    for d_raw in difficulties:
        d_int = int(d_raw)
        d_mapped = _nearest_level(d_int, sorted_levels)

        cheap_acc = cheap_levels_ser.get(d_mapped, {}).get("accuracy", 0.0)
        cap_acc = capable_levels_ser.get(d_mapped, {}).get("accuracy", 0.0)

        # Routing decision
        if policy_name == "always_cheap":
            use_capable = False
        elif policy_name == "always_capable":
            use_capable = True
        elif policy_name == "oracle":
            use_capable = cap_acc > cheap_acc
        elif policy_name == "csd_monitoring":
            use_capable = alarm_level is not None and d_int >= alarm_level
        else:
            use_capable = False

        # Determine outcome
        acc = cap_acc if use_capable else cheap_acc
        cost = COST_CAPABLE if use_capable else COST_CHEAP
        correct = rng.random() < acc
        n_correct += int(correct)
        total_cost += cost
        if use_capable:
            n_capable += 1
        else:
            n_cheap += 1

    # CSD monitoring overhead: sampling B responses at each level up to alarm
    csd_overhead = 0.0
    n_levels_monitored = 0
    if policy_name == "csd_monitoring":
        if alarm_level is None:
            n_levels_monitored = len(sorted_levels)
        else:
            n_levels_monitored = len([lv for lv in sorted_levels if lv <= alarm_level])
        csd_overhead = n_levels_monitored * batch_size * COST_CHEAP
        total_cost += csd_overhead

    accuracy = n_correct / N_QUERIES
    return {
        "mc_idx": mc_idx,
        "policy": policy_name,
        "accuracy": accuracy,
        "error_rate": 1.0 - accuracy,
        "total_cost": total_cost,
        "csd_overhead": csd_overhead,
        "n_correct": n_correct,
        "n_cheap": n_cheap,
        "n_capable": n_capable,
        "alarm_level": alarm_level,
        "n_levels_monitored": n_levels_monitored,
        "batch_size": batch_size,
        "dist_name": dist_name,
    }


# ── Metrics ──────────────────────────────────────────────────────────────────
def _ci95(values: list[float]) -> float:
    """95% confidence interval half-width."""
    if len(values) < 2:
        return 0.0
    return float(1.96 * np.std(values, ddof=1) / math.sqrt(len(values)))


def compute_policy_metrics(
    mc_results: list[dict],
    cheap_results: list[dict],
    capable_results: list[dict],
    oracle_results: list[dict],
    d_star_cheap: int | None,
) -> dict[str, Any]:
    """Compute aggregate metrics for a policy from its MC results."""
    accuracies = [r["accuracy"] for r in mc_results]
    costs = [r["total_cost"] for r in mc_results]
    error_rates = [r["error_rate"] for r in mc_results]

    cheap_mean_err = float(np.mean([r["error_rate"] for r in cheap_results]))
    capable_mean_cost = float(np.mean([r["total_cost"] for r in capable_results]))
    cheap_mean_cost = float(np.mean([r["total_cost"] for r in cheap_results]))
    oracle_mean_acc = float(np.mean([r["accuracy"] for r in oracle_results]))
    cheap_mean_acc = float(np.mean([r["accuracy"] for r in cheap_results]))

    mean_acc = float(np.mean(accuracies))
    mean_cost = float(np.mean(costs))
    mean_err = float(np.mean(error_rates))

    # Error reduction vs cheap
    if cheap_mean_err > 1e-10:
        error_reduction = (cheap_mean_err - mean_err) / cheap_mean_err * 100
    else:
        error_reduction = 0.0

    # Cost relative to capable
    cost_relative = mean_cost / max(capable_mean_cost, 1e-10) * 100

    # Cost efficiency ratio: errors avoided per dollar extra
    extra_cost = mean_cost - cheap_mean_cost
    errors_avoided = cheap_mean_err - mean_err
    if extra_cost > 1e-6:
        cost_efficiency = errors_avoided / extra_cost
    else:
        cost_efficiency = 0.0

    # Oracle gap
    oracle_gap = _oracle_gap(mean_acc, cheap_mean_acc, oracle_mean_acc)

    # Alarm-level metrics (CSD monitoring only)
    alarm_levels = [r["alarm_level"] for r in mc_results if r["alarm_level"] is not None]
    if alarm_levels and d_star_cheap is not None:
        mean_alarm = float(np.mean(alarm_levels))
        alarm_lead_times = [d_star_cheap - a for a in alarm_levels]
        mean_lead = float(np.mean(alarm_lead_times))
        lead_pos_frac = float(sum(1 for t in alarm_lead_times if t > 0) / len(alarm_lead_times))
    else:
        mean_alarm = -1.0
        mean_lead = 0.0
        lead_pos_frac = 0.0

    return {
        "mean_accuracy": mean_acc,
        "ci95_accuracy": _ci95(accuracies),
        "mean_cost": mean_cost,
        "ci95_cost": _ci95(costs),
        "mean_error_rate": mean_err,
        "error_reduction_vs_cheap": error_reduction,
        "cost_relative_to_capable": cost_relative,
        "cost_efficiency_ratio": cost_efficiency,
        "oracle_gap": oracle_gap,
        "mean_alarm_level": mean_alarm,
        "mean_alarm_lead_time": mean_lead,
        "alarm_lead_pos_frac": lead_pos_frac,
    }


def _oracle_gap(policy_acc: float, cheap_acc: float, oracle_acc: float) -> float:
    """Oracle gap: fraction of oracle's improvement captured by this policy."""
    denom = oracle_acc - cheap_acc
    if abs(denom) < 1e-10:
        return 0.0
    return float((policy_acc - cheap_acc) / denom * 100)


def is_pareto_optimal(
    acc: float, cost: float, all_points: list[tuple[float, float]],
) -> bool:
    """Check if (acc, cost) is on the Pareto frontier (higher acc, lower cost)."""
    for a, c in all_points:
        if a >= acc and c <= cost and (a > acc or c < cost):
            return False
    return True


def compute_breakeven(
    csd_results: list[dict],
    capable_results: list[dict],
) -> float:
    """Minimum query volume where CSD routing beats always-capable on cost.

    Finds N such that CSD total cost <= always_capable total cost,
    accounting for CSD monitoring overhead.
    """
    # CSD average overhead per run (fixed cost)
    overheads = [r["csd_overhead"] for r in csd_results]
    mean_overhead = float(np.mean(overheads))

    # Per-query cost for CSD (excluding overhead)
    csd_total_no_overhead = [r["total_cost"] - r["csd_overhead"] for r in csd_results]
    csd_per_query = float(np.mean(csd_total_no_overhead)) / N_QUERIES

    # Per-query cost for always-capable
    cap_per_query = float(np.mean([r["total_cost"] for r in capable_results])) / N_QUERIES

    savings_per_query = cap_per_query - csd_per_query
    if savings_per_query <= 1e-10:
        return float("inf")

    breakeven = mean_overhead / savings_per_query
    return max(breakeven, 1.0)


# ── Main orchestration ───────────────────────────────────────────────────────
@logger.catch
def main() -> dict:
    t0 = time.time()
    logger.info("Starting CSD-Based Model Routing Simulation")

    # ── Load data ────────────────────────────────────────────────────────
    arith_path = Path("full_arithmetic_data.json")
    gc_path = Path("full_graph_coloring_data.json")

    logger.info(f"Loading arithmetic data from {arith_path}")
    arith_models = load_arithmetic_data(str(arith_path))
    logger.info(f"  Loaded {len(arith_models)} models: {list(arith_models.keys())}")
    for m, info in arith_models.items():
        logger.info(f"    {m}: {len(info['levels'])} levels, d*={info['d_star']}")

    logger.info(f"Loading graph coloring data from {gc_path}")
    gc_models = load_graph_coloring_data(str(gc_path))
    logger.info(f"  Loaded {len(gc_models)} models: {list(gc_models.keys())}")
    for m, info in gc_models.items():
        logger.info(f"    {m}: {len(info['levels'])} levels, d*={info['d_star']}")

    # Free raw JSON
    gc.collect()

    # ── Task configurations ──────────────────────────────────────────────
    tasks = [
        {
            "name": "arithmetic",
            "models": arith_models,
            "cheap": "gemini-2.0-flash-001",
            "capable": "llama-3.1-8b-instruct",
        },
        {
            "name": "graph_coloring",
            "models": gc_models,
            "cheap": "gemini-2.0-flash-lite-001",
            "capable": "gemini-2.0-flash-001",
        },
    ]

    all_datasets: list[dict] = []
    global_metrics: dict[str, float] = {}
    all_csd_lead_times: list[float] = []
    csd_pareto_flags: list[bool] = []

    for task_cfg in tasks:
        task_name = task_cfg["name"]
        cheap_name = task_cfg["cheap"]
        capable_name = task_cfg["capable"]
        models = task_cfg["models"]

        cheap_data = models[cheap_name]
        capable_data = models[capable_name]
        d_star_cheap = cheap_data["d_star"]
        d_star_capable = capable_data["d_star"]

        # Build sorted level list (union of both models)
        all_levels_set = set(cheap_data["levels"].keys()) | set(capable_data["levels"].keys())
        sorted_levels = sorted(all_levels_set)

        # Serializable indicator dicts for CSD alarm in workers
        cheap_indicators = cheap_data["levels"]  # full per-level dict

        logger.info(f"=== Task: {task_name} ===")
        logger.info(f"  Cheap: {cheap_name} (d*={d_star_cheap})")
        logger.info(f"  Capable: {capable_name} (d*={d_star_capable})")
        logger.info(f"  Levels: {sorted_levels[0]}..{sorted_levels[-1]} ({len(sorted_levels)} levels)")

        task_examples: list[dict] = []

        for dist_name in DIFFICULTY_DISTRIBUTIONS:
            for batch_size in BATCH_SIZES:
                # Build MC jobs for all policies
                base_seed = abs(hash((task_name, dist_name, batch_size))) % (2**31)
                mc_jobs: list[tuple] = []

                for pi, policy in enumerate(POLICIES):
                    for mc_idx in range(N_MC_RUNS):
                        seed = base_seed + mc_idx + pi * N_MC_RUNS * 2
                        mc_jobs.append((
                            mc_idx, policy,
                            cheap_data["levels"], capable_data["levels"],
                            cheap_indicators, sorted_levels, d_star_cheap,
                            batch_size, dist_name, seed,
                        ))

                # Run in parallel
                results_by_policy: dict[str, list[dict]] = defaultdict(list)
                with ProcessPoolExecutor(max_workers=max(NUM_CPUS - 1, 1)) as pool:
                    futures = [pool.submit(run_single_mc, job) for job in mc_jobs]
                    for fut in as_completed(futures):
                        try:
                            res = fut.result()
                            results_by_policy[res["policy"]].append(res)
                        except Exception:
                            logger.exception("MC run failed")

                # Compute per-policy metrics
                cheap_res = results_by_policy["always_cheap"]
                capable_res = results_by_policy["always_capable"]
                oracle_res = results_by_policy["oracle"]
                csd_res = results_by_policy["csd_monitoring"]

                policy_metrics: dict[str, dict] = {}
                for policy in POLICIES:
                    policy_metrics[policy] = compute_policy_metrics(
                        results_by_policy[policy],
                        cheap_res, capable_res, oracle_res,
                        d_star_cheap,
                    )

                # Pareto optimality
                points = [
                    (policy_metrics[p]["mean_accuracy"], policy_metrics[p]["mean_cost"])
                    for p in POLICIES
                ]
                for i, policy in enumerate(POLICIES):
                    policy_metrics[policy]["pareto_optimal"] = is_pareto_optimal(
                        points[i][0], points[i][1], points,
                    )

                # Breakeven
                breakeven_n = compute_breakeven(csd_res, capable_res)

                # Build output examples
                for policy in POLICIES:
                    pm = policy_metrics[policy]

                    input_str = (
                        f"Policy={policy.upper()}, task={task_name}, "
                        f"cheap={cheap_name}, capable={capable_name}, "
                        f"B={batch_size}, dist={dist_name}"
                    )
                    alarm_str = ""
                    if policy == "csd_monitoring" and pm["mean_alarm_level"] > 0:
                        alarm_str = f", alarm_d={pm['mean_alarm_level']:.1f}"
                    output_str = (
                        f"accuracy={pm['mean_accuracy']:.4f} +/- {pm['ci95_accuracy']:.4f}, "
                        f"cost=${pm['mean_cost']:.4f} +/- {pm['ci95_cost']:.4f}"
                        f"{alarm_str}"
                    )

                    # Routing decision summary for predict field
                    if policy == "csd_monitoring":
                        alarm_d = pm["mean_alarm_level"]
                        if alarm_d > 0:
                            predict_str = (
                                f"Use {cheap_name} for d<{alarm_d:.0f}, "
                                f"switch to {capable_name} at d>={alarm_d:.0f}"
                            )
                        else:
                            predict_str = f"Always use {cheap_name} (no alarm fired)"
                    elif policy == "always_cheap":
                        predict_str = f"Always use {cheap_name}"
                    elif policy == "always_capable":
                        predict_str = f"Always use {capable_name}"
                    else:
                        predict_str = f"Route per-query to best model (oracle)"

                    example: dict[str, Any] = {
                        "input": input_str,
                        "output": output_str,
                        "predict_routing_decision": predict_str,
                        "eval_overall_accuracy": round(pm["mean_accuracy"], 6),
                        "eval_total_cost": round(pm["mean_cost"], 6),
                        "eval_error_rate": round(pm["mean_error_rate"], 6),
                        "eval_error_reduction_vs_cheap": round(pm["error_reduction_vs_cheap"], 4),
                        "eval_cost_relative_to_capable": round(pm["cost_relative_to_capable"], 4),
                        "eval_cost_efficiency_ratio": round(pm["cost_efficiency_ratio"], 6),
                        "eval_oracle_gap": round(pm["oracle_gap"], 4),
                        "eval_pareto_optimal": 1.0 if pm["pareto_optimal"] else 0.0,
                        "eval_ci95_accuracy": round(pm["ci95_accuracy"], 6),
                        "eval_ci95_cost": round(pm["ci95_cost"], 6),
                        "eval_alarm_difficulty": round(pm["mean_alarm_level"], 2),
                        "eval_alarm_lead_time": round(pm["mean_alarm_lead_time"], 4),
                        "eval_batch_size_reliability": round(pm["alarm_lead_pos_frac"], 4),
                        "eval_breakeven_query_volume": round(min(breakeven_n, 1e7), 2),
                        "metadata_policy": policy,
                        "metadata_task": task_name,
                        "metadata_cheap_model": cheap_name,
                        "metadata_capable_model": capable_name,
                        "metadata_batch_size": batch_size,
                        "metadata_difficulty_distribution": dist_name,
                        "metadata_mc_runs": N_MC_RUNS,
                        "metadata_d_star_cheap": d_star_cheap if d_star_cheap is not None else -1,
                        "metadata_d_star_capable": d_star_capable if d_star_capable is not None else -1,
                    }
                    task_examples.append(example)

                # Track global metrics for default config (B=20, uniform)
                if batch_size == 20 and dist_name == "uniform":
                    csd_pm = policy_metrics["csd_monitoring"]
                    global_metrics[f"{task_name}_csd_oracle_gap_pct"] = round(csd_pm["oracle_gap"], 4)
                    global_metrics[f"{task_name}_csd_cost_relative_pct"] = round(csd_pm["cost_relative_to_capable"], 4)
                    global_metrics[f"{task_name}_csd_accuracy"] = round(csd_pm["mean_accuracy"], 6)
                    global_metrics[f"{task_name}_oracle_accuracy"] = round(
                        policy_metrics["oracle"]["mean_accuracy"], 6)
                    global_metrics[f"{task_name}_cheap_accuracy"] = round(
                        policy_metrics["always_cheap"]["mean_accuracy"], 6)
                    global_metrics[f"{task_name}_capable_accuracy"] = round(
                        policy_metrics["always_capable"]["mean_accuracy"], 6)
                    global_metrics[f"{task_name}_breakeven_n"] = round(min(breakeven_n, 1e7), 2)

                    csd_pareto_flags.append(csd_pm["pareto_optimal"])

                    if csd_pm["mean_alarm_lead_time"] != 0.0:
                        all_csd_lead_times.append(csd_pm["mean_alarm_lead_time"])

                    logger.info(
                        f"  B={batch_size} dist={dist_name}: "
                        + " | ".join(
                            f"{p}: acc={policy_metrics[p]['mean_accuracy']:.3f} "
                            f"cost=${policy_metrics[p]['mean_cost']:.3f} "
                            f"OG={policy_metrics[p]['oracle_gap']:.1f}%"
                            for p in POLICIES
                        )
                    )

        # Accuracy by distribution for CSD (B=20)
        for dist_name in DIFFICULTY_DISTRIBUTIONS:
            key = f"{task_name}_csd_accuracy_{dist_name}"
            # Find the example with matching config
            for ex in task_examples:
                if (ex["metadata_policy"] == "csd_monitoring"
                        and ex["metadata_batch_size"] == 20
                        and ex["metadata_difficulty_distribution"] == dist_name):
                    global_metrics[key] = ex["eval_overall_accuracy"]
                    break

        all_datasets.append({
            "dataset": f"routing_simulation_{task_name}",
            "examples": task_examples,
        })
        logger.info(f"  Generated {len(task_examples)} examples for {task_name}")

    # ── Aggregate metrics ────────────────────────────────────────────────
    csd_lead_mean = float(np.mean(all_csd_lead_times)) if all_csd_lead_times else 0.0
    csd_pareto = 1.0 if all(csd_pareto_flags) else 0.0

    arith_og = global_metrics.get("arithmetic_csd_oracle_gap_pct", 0.0)
    arith_cost = global_metrics.get("arithmetic_csd_cost_relative_pct", 0.0)
    gc_og = global_metrics.get("graph_coloring_csd_oracle_gap_pct", 0.0)
    gc_cost = global_metrics.get("graph_coloring_csd_cost_relative_pct", 0.0)
    avg_og = round((arith_og + gc_og) / 2, 4)
    avg_cost = round((arith_cost + gc_cost) / 2, 4)

    metrics_agg: dict[str, float] = {
        "arithmetic_csd_oracle_gap_pct": arith_og,
        "arithmetic_csd_cost_relative_pct": arith_cost,
        "graph_coloring_csd_oracle_gap_pct": gc_og,
        "graph_coloring_csd_cost_relative_pct": gc_cost,
        "csd_alarm_lead_time_mean": round(csd_lead_mean, 4),
        "csd_pareto_optimal": csd_pareto,
        "avg_oracle_gap_pct": avg_og,
        "avg_cost_relative_pct": avg_cost,
    }
    # Merge per-task metrics
    for k, v in global_metrics.items():
        safe_key = k.replace("-", "_").replace(".", "_")
        if safe_key not in metrics_agg:
            metrics_agg[safe_key] = v

    # Sanitize: ensure all values are finite numbers
    for k in list(metrics_agg.keys()):
        v = metrics_agg[k]
        if not isinstance(v, (int, float)):
            metrics_agg[k] = 0.0
        elif math.isinf(v) or math.isnan(v):
            metrics_agg[k] = 0.0

    headline = (
        f"CSD routing achieves {avg_og:.1f}% of oracle improvement "
        f"at {avg_cost:.1f}% of capable cost"
    )

    output = {
        "metadata": {
            "evaluation_name": "CSD-Based Model Routing Simulation",
            "description": (
                "Downstream value quantification of CSD indicators for model "
                "routing. Compares 4 policies (always-cheap, always-capable, "
                "CSD-monitored, oracle) across Monte Carlo runs."
            ),
            "n_queries_per_run": N_QUERIES,
            "n_mc_runs": N_MC_RUNS,
            "cost_cheap_per_query": COST_CHEAP,
            "cost_capable_per_query": COST_CAPABLE,
            "batch_sizes": BATCH_SIZES,
            "difficulty_distributions": DIFFICULTY_DISTRIBUTIONS,
            "csd_alarm_threshold": CSD_ALARM_THRESHOLD,
            "csd_alarm_min_levels": CSD_ALARM_MIN_LEVELS,
            "headline": headline,
            "tasks": [
                {
                    "name": t["name"],
                    "cheap_model": t["cheap"],
                    "capable_model": t["capable"],
                }
                for t in tasks
            ],
        },
        "metrics_agg": metrics_agg,
        "datasets": all_datasets,
    }

    # ── Save output ──────────────────────────────────────────────────────
    out_path = Path("eval_out.json")
    out_path.write_text(json.dumps(output, indent=2))
    elapsed = time.time() - t0
    total_examples = sum(len(ds["examples"]) for ds in all_datasets)
    logger.info(f"Saved {out_path} ({total_examples} examples) in {elapsed:.1f}s")
    logger.info(f"Headline: {headline}")

    return output


if __name__ == "__main__":
    main()
