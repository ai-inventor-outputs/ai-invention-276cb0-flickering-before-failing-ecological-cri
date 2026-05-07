#!/usr/bin/env python3
"""
Graph Coloring CSD Sampling Experiment
========================================
3 LLMs x 20 difficulty levels x 5 problems x 10 responses = 3,000 API calls.
Evaluates Critical Slowing Down (CSD) indicators near LLM capability boundaries
on graph coloring constraint satisfaction problems.

Produces method_out.json in exp_gen_sol_out schema format.
"""

import argparse
import asyncio
import gc
import json
import math
import os
import random
import re
import resource
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from loguru import logger

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
WORKSPACE = Path("/ai-inventor/aii_pipeline/runs/neurips-open/3_invention_loop/iter_2/gen_art/exp_id3_it2__opus")
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------------------
# HARDWARE DETECTION (cgroup-aware, per aii_use_hardware skill)
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

# Memory budget: 70% of container RAM for safety
RAM_BUDGET_BYTES = int(TOTAL_RAM_GB * 0.7 * 1024**3)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET_BYTES * 3, RAM_BUDGET_BYTES * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, budget={RAM_BUDGET_BYTES/1e9:.1f} GB")

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
DATA_DEP = Path("/ai-inventor/aii_pipeline/runs/neurips-open/3_invention_loop/iter_1/gen_art/data_id4_it1__opus")

# Load API key
from dotenv import load_dotenv
load_dotenv(Path("/ai-inventor/.claude/skills/aii_openrouter_llms/.env"))
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
if not OPENROUTER_API_KEY:
    logger.error("OPENROUTER_API_KEY not found!")
    sys.exit(1)

API_URL = "https://openrouter.ai/api/v1/responses"

# 3 models of varying capability, all affordable
MODELS = [
    "openai/gpt-4o-mini",               # Strong: $0.15/$0.60 per M tokens
    "google/gemini-2.0-flash-001",       # Medium: $0.10/$0.40 per M tokens
    "google/gemini-2.0-flash-lite-001",  # Weak:   $0.07/$0.30 per M tokens
]

MODEL_PRICING = {
    "openai/gpt-4o-mini":               {"input": 0.15, "output": 0.60},
    "google/gemini-2.0-flash-001":       {"input": 0.10, "output": 0.40},
    "google/gemini-2.0-flash-lite-001":  {"input": 0.07, "output": 0.30},
}

TEMPERATURE = 0.8
N_PROBLEMS_PER_LEVEL = 5
N_RESPONSES_PER_PROBLEM = 10
DIFFICULTY_LEVELS = 20
SEED = 42
MAX_OUTPUT_TOKENS = 1500
COST_LIMIT = 10.0   # USD hard limit
SEMAPHORE_LIMIT = 25  # concurrent API requests

SYSTEM_PROMPT = (
    "You are solving a graph coloring constraint satisfaction problem. "
    "Think through the problem step by step, then provide your final answer. "
    "Format your final answer as: Node 0: Color, Node 1: Color, Node 2: Color, ... "
    "Use only the colors specified in the problem."
)

COLOR_SETS = {
    3: {"red", "green", "blue"},
    4: {"red", "green", "blue", "yellow"},
}

# Global cost tracking
cumulative_cost = 0.0
total_input_tokens = 0
total_output_tokens = 0

# ===========================================================================
# STEP 1: LOAD DATASET & SELECT PROBLEMS
# ===========================================================================

def load_and_select_problems() -> dict:
    """Load 400 problems from full_data_out.json, select 5 per level deterministically."""
    data_path = DATA_DEP / "full_data_out.json"
    logger.info(f"Loading dataset from {data_path}")
    data = json.loads(data_path.read_text())
    all_examples = data["datasets"][0]["examples"]
    logger.info(f"Loaded {len(all_examples)} total problems")

    by_level: dict[int, list] = {}
    for ex in all_examples:
        lvl = ex["metadata_difficulty_level"]
        by_level.setdefault(lvl, []).append(ex)

    selected: dict[int, list] = {}
    rng = random.Random(SEED)
    for lvl in sorted(by_level.keys()):
        pool = by_level[lvl]
        selected[lvl] = rng.sample(pool, min(N_PROBLEMS_PER_LEVEL, len(pool)))

    total = sum(len(v) for v in selected.values())
    logger.info(f"Selected {total} problems across {len(selected)} levels")
    return selected


# ===========================================================================
# STEP 2: LLM RESPONSE GENERATION (async I/O-bound)
# ===========================================================================

async def call_openrouter(
    session: "aiohttp.ClientSession",
    semaphore: asyncio.Semaphore,
    model: str,
    prompt: str,
    temperature: float,
    metadata: dict,
) -> dict:
    """Single OpenRouter API call with 3-attempt retry and exponential backoff."""
    global cumulative_cost, total_input_tokens, total_output_tokens
    import aiohttp as _aio

    async with semaphore:
        for attempt in range(3):
            try:
                payload = {
                    "model": model,
                    "input": [
                        {
                            "type": "message",
                            "role": "system",
                            "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
                        },
                        {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": prompt}],
                        },
                    ],
                    "temperature": temperature,
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                }
                headers = {
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                }
                async with session.post(
                    API_URL,
                    json=payload,
                    headers=headers,
                    timeout=_aio.ClientTimeout(total=120),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()

                        # Extract response text
                        response_text = ""
                        if data.get("output_text"):
                            response_text = data["output_text"]
                        elif "output" in data:
                            for item in data.get("output", []):
                                if item.get("type") == "message":
                                    for content in item.get("content", []):
                                        if content.get("type") == "output_text":
                                            response_text = content.get("text", "")

                        tokens_in = data.get("usage", {}).get("input_tokens", 0)
                        tokens_out = data.get("usage", {}).get("output_tokens", 0)

                        # Track cost
                        pricing = MODEL_PRICING.get(model, {"input": 1.0, "output": 5.0})
                        call_cost = (tokens_in * pricing["input"] + tokens_out * pricing["output"]) / 1_000_000
                        cumulative_cost += call_cost
                        total_input_tokens += tokens_in
                        total_output_tokens += tokens_out

                        if cumulative_cost > COST_LIMIT:
                            logger.warning(f"COST LIMIT REACHED: ${cumulative_cost:.2f}")
                            return {
                                **metadata,
                                "response": "",
                                "error": "cost_limit_exceeded",
                                "tokens_in": 0,
                                "tokens_out": 0,
                            }

                        return {
                            **metadata,
                            "response": response_text,
                            "tokens_in": tokens_in,
                            "tokens_out": tokens_out,
                        }

                    elif resp.status == 429:
                        wait = 2**attempt * 5
                        logger.debug(f"Rate limited on {model}, waiting {wait}s (attempt {attempt+1})")
                        await asyncio.sleep(wait)
                    else:
                        error_text = await resp.text()
                        logger.warning(f"API error {resp.status} for {model}: {error_text[:200]}")
                        await asyncio.sleep(2**attempt * 2)

            except asyncio.TimeoutError:
                logger.warning(f"Timeout for {model} attempt {attempt+1}")
                await asyncio.sleep(2**attempt * 2)
            except Exception as e:
                logger.warning(f"Request failed {model} attempt {attempt+1}: {e}")
                await asyncio.sleep(2**attempt)

        return {
            **metadata,
            "response": "",
            "error": "max_retries_exceeded",
            "tokens_in": 0,
            "tokens_out": 0,
        }


async def generate_all_responses(
    selected_problems: dict,
    models: list[str] | None = None,
    max_levels: int | None = None,
    max_problems: int | None = None,
    max_responses: int | None = None,
) -> list[dict]:
    """Generate responses for all (model, problem) pairs with progress tracking."""
    import aiohttp as _aio

    if models is None:
        models = MODELS

    # Build task list
    tasks_meta = []
    for model in models:
        levels = sorted(selected_problems.keys())
        if max_levels:
            levels = levels[:max_levels]
        for level in levels:
            problems = selected_problems[level]
            if max_problems:
                problems = problems[:max_problems]
            for prob_idx, problem in enumerate(problems):
                n_resp = max_responses or N_RESPONSES_PER_PROBLEM
                for resp_idx in range(n_resp):
                    tasks_meta.append({
                        "model": model,
                        "level": level,
                        "prob_idx": prob_idx,
                        "resp_idx": resp_idx,
                        "prompt": problem["input"],
                    })

    total = len(tasks_meta)
    logger.info(f"Total API calls to make: {total}")

    connector = _aio.TCPConnector(limit=SEMAPHORE_LIMIT + 10)
    async with _aio.ClientSession(connector=connector) as session:
        semaphore = asyncio.Semaphore(SEMAPHORE_LIMIT)

        results: list[dict] = []
        batch_size = 100
        for i in range(0, total, batch_size):
            # Cost guard
            if cumulative_cost > COST_LIMIT * 0.95:
                logger.warning(f"Approaching cost limit ${cumulative_cost:.2f}, stopping early")
                break

            batch = tasks_meta[i : i + batch_size]
            coros = [
                call_openrouter(
                    session,
                    semaphore,
                    m["model"],
                    m["prompt"],
                    TEMPERATURE,
                    {"model": m["model"], "level": m["level"],
                     "prob_idx": m["prob_idx"], "resp_idx": m["resp_idx"]},
                )
                for m in batch
            ]
            batch_results = await asyncio.gather(*coros, return_exceptions=True)

            for r in batch_results:
                if isinstance(r, Exception):
                    logger.warning(f"Batch exception: {r}")
                    results.append({"response": "", "error": str(r),
                                    "model": "", "level": 0, "prob_idx": 0, "resp_idx": 0,
                                    "tokens_in": 0, "tokens_out": 0})
                else:
                    results.append(r)

            done = min(i + batch_size, total)
            logger.info(
                f"Progress: {done}/{total} calls "
                f"({done/total*100:.0f}%), cost: ${cumulative_cost:.3f}"
            )

            # Save checkpoint every batch
            ckpt_path = WORKSPACE / "checkpoint_responses.json"
            ckpt_path.write_text(json.dumps(results, default=str))

    return results


# ===========================================================================
# STEP 3: CONSTRAINT SATISFACTION EVALUATION
# ===========================================================================

def parse_coloring(response_text: str, num_nodes: int, num_colors: int) -> dict | None:
    """
    Robust parser: extract node->color assignments from LLM response.
    Handles multiple formats and focuses on the LAST occurrence of assignments.
    """
    if not response_text:
        return None

    allowed_colors = COLOR_SETS.get(num_colors, COLOR_SETS[3])
    color_pattern = "|".join(allowed_colors)

    coloring: dict[int, str] = {}

    # Strategy 1: "Node X: Color" / "Node X = Color" / "Node X -> Color"
    pattern1 = rf"[Nn]ode\s*(\d+)\s*[:=\->]+\s*({color_pattern})"
    matches = re.findall(pattern1, response_text, re.IGNORECASE)
    if matches:
        # Use LAST set of matches (final answer)
        # Group matches into contiguous sets
        all_match_positions = list(re.finditer(pattern1, response_text, re.IGNORECASE))
        if all_match_positions:
            # Find the last contiguous block of assignments
            last_block: list[tuple[str, str]] = []
            for m_obj in reversed(all_match_positions):
                node_id = int(m_obj.group(1))
                color = m_obj.group(2).lower()
                if node_id < num_nodes:
                    last_block.append((str(node_id), color))
                if len(last_block) >= num_nodes:
                    break
            for node_str, color in last_block:
                coloring[int(node_str)] = color

    # Strategy 2: "X: Color" without Node prefix
    if len(coloring) < num_nodes:
        pattern2 = rf"(?<!\w)(\d+)\s*[:=\->]+\s*({color_pattern})(?!\w)"
        matches2 = re.findall(pattern2, response_text, re.IGNORECASE)
        for node_str, color in matches2:
            node_id = int(node_str)
            if node_id < num_nodes and node_id not in coloring:
                coloring[node_id] = color.lower()

    # Strategy 3: Look in final answer section
    if len(coloring) < num_nodes:
        for marker in ["final answer", "answer:", "solution:", "coloring:"]:
            idx = response_text.lower().rfind(marker)
            if idx >= 0:
                tail = response_text[idx:]
                tail_matches = re.findall(pattern1, tail, re.IGNORECASE)
                for node_str, color in tail_matches:
                    node_id = int(node_str)
                    if node_id < num_nodes:
                        coloring[node_id] = color.lower()

    # Strategy 4: JSON-like {"0": "red", ...}
    if len(coloring) < num_nodes:
        json_pattern = rf'"?(\d+)"?\s*:\s*"?({color_pattern})"?'
        json_matches = re.findall(json_pattern, response_text, re.IGNORECASE)
        for node_str, color in json_matches:
            node_id = int(node_str)
            if node_id < num_nodes and node_id not in coloring:
                coloring[node_id] = color.lower()

    # Validate: all nodes 0..num_nodes-1 present, all colors valid
    if set(coloring.keys()) != set(range(num_nodes)):
        return None
    if not all(c in allowed_colors for c in coloring.values()):
        return None

    return coloring


def check_constraint_satisfaction(
    coloring: dict[int, str],
    adjacency: list[list[int]],
) -> bool:
    """Verify no two adjacent nodes share a color."""
    for edge in adjacency:
        u, v = edge[0], edge[1]
        if coloring.get(u) == coloring.get(v):
            return False
    return True


def evaluate_response(response_text: str, problem: dict) -> dict:
    """Full evaluation of a single response."""
    num_nodes = problem["metadata_num_nodes"]
    num_colors = problem["metadata_num_colors"]
    adjacency = problem["metadata_graph_adjacency"]

    coloring = parse_coloring(response_text, num_nodes, num_colors)

    if coloring is None:
        return {"parsed": False, "correct": False, "coloring": None}

    valid = check_constraint_satisfaction(coloring, adjacency)
    return {"parsed": True, "correct": valid, "coloring": coloring}


# ===========================================================================
# STEP 4: EMBEDDING
# ===========================================================================

def embed_responses(texts: list[str]) -> np.ndarray:
    """Embed all response texts using all-MiniLM-L6-v2."""
    from sentence_transformers import SentenceTransformer

    logger.info(f"Loading embedding model and encoding {len(texts)} texts...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    logger.info(f"Embedding shape: {embeddings.shape}")

    # Free model memory
    del model
    gc.collect()

    return embeddings


# ===========================================================================
# STEP 5: CSD INDICATOR COMPUTATION
# ===========================================================================

def _empty_indicators() -> dict:
    return {
        "accuracy": 0.0, "n_responses": 0, "n_correct": 0, "n_parsed": 0,
        "embedding_variance": 0.0, "pc1_variance": 0.0,
        "dip_statistic": 0.0, "dip_pvalue": 1.0,
        "silhouette_score": 0.0, "bimodality_coefficient": 0.0,
        "disagreement_rate": 1.0, "ashman_d": 0.0,
    }


def compute_csd_indicators(
    embeddings: np.ndarray,
    correctness: list[bool],
    colorings: list[dict | None],
) -> dict:
    """Compute full CSD indicator battery for one (model, level) group."""
    import diptest
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score as sil_fn
    from sklearn.decomposition import PCA
    from scipy.stats import skew, kurtosis

    N = len(correctness)
    if N == 0:
        return _empty_indicators()

    accuracy = sum(correctness) / N

    # --- 1. Embedding Variance ---
    if N > 1:
        sim_matrix = embeddings @ embeddings.T
        dist_matrix = 1.0 - sim_matrix
        mask = ~np.eye(N, dtype=bool)
        embedding_variance = float(np.mean(dist_matrix[mask]))
    else:
        embedding_variance = 0.0

    # PC1 projection
    if N >= 3:
        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(embeddings).flatten()
        pc1_variance = float(np.var(pc1))
    else:
        pc1 = np.zeros(N)
        pc1_variance = 0.0

    # --- 2. Hartigan's Dip Test ---
    if N >= 4:
        try:
            dip_stat, dip_pval = diptest.diptest(pc1)
            dip_stat = float(dip_stat)
            dip_pval = float(dip_pval)
        except Exception:
            dip_stat, dip_pval = 0.0, 1.0
    else:
        dip_stat, dip_pval = 0.0, 1.0

    # --- 3. Silhouette Score (k=2 k-means) ---
    if N >= 4:
        try:
            kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
            labels = kmeans.fit_predict(embeddings)
            if len(set(labels)) == 2:
                sil_score = float(sil_fn(embeddings, labels))
            else:
                sil_score = 0.0
        except Exception:
            sil_score = 0.0
    else:
        sil_score = 0.0

    # --- 4. Bimodality Coefficient ---
    if N >= 4:
        try:
            s = float(skew(pc1))
            k = float(kurtosis(pc1, fisher=True))
            n = N
            denom_factor = (n - 2) * (n - 3)
            if denom_factor != 0:
                denom = k + 3.0 * (n - 1) ** 2 / denom_factor
            else:
                denom = 1.0
            bc = (s**2 + 1) / denom if denom != 0 else 0.0
        except Exception:
            bc = 0.0
    else:
        bc = 0.0

    # --- 5. Disagreement Rate ---
    answer_strs = []
    for c in colorings:
        if c is not None:
            answer_strs.append(str(sorted(c.items())))
        else:
            answer_strs.append("PARSE_FAIL")
    answer_counts = Counter(answer_strs)
    most_common_count = answer_counts.most_common(1)[0][1] if answer_counts else 0
    disagreement = 1.0 - (most_common_count / N) if N > 0 else 1.0

    # --- 6. Ashman's D (correct vs incorrect cluster separation) ---
    correct_idx = [i for i, c in enumerate(correctness) if c]
    incorrect_idx = [i for i, c in enumerate(correctness) if not c]
    if len(correct_idx) >= 2 and len(incorrect_idx) >= 2:
        pc1_correct = pc1[correct_idx]
        pc1_incorrect = pc1[incorrect_idx]
        mu1, mu2 = float(np.mean(pc1_correct)), float(np.mean(pc1_incorrect))
        s1, s2 = float(np.std(pc1_correct)), float(np.std(pc1_incorrect))
        if (s1**2 + s2**2) > 0:
            ashman_d = float(np.sqrt(2) * abs(mu1 - mu2) / np.sqrt(s1**2 + s2**2))
        else:
            ashman_d = 0.0
    else:
        ashman_d = 0.0

    return {
        "accuracy": round(accuracy, 4),
        "n_responses": N,
        "n_correct": int(sum(correctness)),
        "n_parsed": int(sum(1 for c in colorings if c is not None)),
        "embedding_variance": round(embedding_variance, 6),
        "pc1_variance": round(pc1_variance, 6),
        "dip_statistic": round(dip_stat, 6),
        "dip_pvalue": round(dip_pval, 6),
        "silhouette_score": round(sil_score, 6),
        "bimodality_coefficient": round(bc, 6),
        "disagreement_rate": round(disagreement, 4),
        "ashman_d": round(ashman_d, 6),
    }


# ===========================================================================
# STEP 6: ANALYSIS -- d*, SCALING EXPONENT, LEADING INDICATORS
# ===========================================================================

def analyze_model_results(model_name: str, level_indicators: dict) -> dict:
    """Analyze CSD indicators across difficulty levels for one model."""
    from scipy.stats import linregress, kendalltau

    levels = sorted(level_indicators.keys())
    accuracies = [level_indicators[l]["accuracy"] for l in levels]
    variances = [level_indicators[l]["pc1_variance"] for l in levels]
    dip_pvals = [level_indicators[l]["dip_pvalue"] for l in levels]
    silhouettes = [level_indicators[l]["silhouette_score"] for l in levels]

    # --- d* estimation: first level where accuracy < 50% ---
    d_star = None
    for lvl, acc in zip(levels, accuracies):
        if acc < 0.5:
            d_star = lvl
            break
    if d_star is None:
        d_star = levels[-1] + 1  # model never drops below 50%

    # --- Variance scaling exponent ---
    scaling_exponent = None
    scaling_r_squared = None
    valid_points = []
    for lvl, var in zip(levels, variances):
        if lvl < d_star and var > 0:
            valid_points.append((d_star - lvl, var))
    if len(valid_points) >= 3:
        log_dist = np.log([p[0] for p in valid_points])
        log_var = np.log([p[1] for p in valid_points])
        try:
            slope, intercept, r_value, p_value, std_err = linregress(log_dist, log_var)
            scaling_exponent = float(slope)
            scaling_r_squared = float(r_value**2)
        except Exception as e:
            logger.warning(f"Scaling exponent fit failed for {model_name}: {e}")

    # --- Leading indicator tests ---
    leading_indicators: dict = {}

    # Dip test leading indicator
    for lvl, pval, acc in zip(levels, dip_pvals, accuracies):
        if pval < 0.05 and acc > 0.80:
            leading_indicators["dip_first_significant"] = lvl
            leading_indicators["dip_lead_time"] = d_star - lvl if d_star else None
            break

    # Silhouette leading indicator (threshold > 0.3)
    for lvl, sil, acc in zip(levels, silhouettes, accuracies):
        if sil > 0.3 and acc > 0.80:
            leading_indicators["silhouette_first_above_threshold"] = lvl
            leading_indicators["silhouette_lead_time"] = d_star - lvl if d_star else None
            break

    # Kendall tau trend test on variance
    pre_boundary = [i for i, l in enumerate(levels) if l < (d_star or levels[-1] + 1)]
    if len(pre_boundary) >= 5:
        var_pre = [variances[i] for i in pre_boundary]
        try:
            tau, tau_pval = kendalltau(range(len(var_pre)), var_pre)
            leading_indicators["variance_kendall_tau"] = round(float(tau), 4)
            leading_indicators["variance_kendall_pval"] = round(float(tau_pval), 6)
        except Exception:
            pass

    return {
        "model": model_name,
        "d_star": d_star,
        "scaling_exponent": round(scaling_exponent, 4) if scaling_exponent is not None else None,
        "scaling_r_squared": round(scaling_r_squared, 4) if scaling_r_squared is not None else None,
        "theoretical_exponent": -0.5,
        "exponent_in_range": (
            scaling_exponent is not None and -0.7 <= scaling_exponent <= -0.3
        ),
        "leading_indicators": leading_indicators,
        "per_level": {lvl: level_indicators[lvl] for lvl in levels},
    }


# ===========================================================================
# STEP 7: BASELINE CLASSIFIER COMPARISON
# ===========================================================================

def build_csd_classifier(all_model_results: list[dict]) -> dict:
    """
    Build logistic regression classifier: CSD features vs disagreement-only baseline.
    Evaluate via cross-validation. Report F1, precision, recall for each feature set.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score, precision_score, recall_score
    from sklearn.model_selection import cross_val_predict

    X_csd, X_baseline, y = [], [], []

    for model_result in all_model_results:
        d_star = model_result["d_star"]
        for lvl, indicators in model_result["per_level"].items():
            label = 1 if (d_star - lvl) <= 2 and lvl <= d_star else 0
            X_csd.append([
                indicators["pc1_variance"],
                indicators["dip_statistic"],
                indicators["silhouette_score"],
                indicators["bimodality_coefficient"],
                indicators["ashman_d"],
            ])
            X_baseline.append([indicators["disagreement_rate"]])
            y.append(label)

    X_csd_arr = np.array(X_csd)
    X_baseline_arr = np.array(X_baseline)
    y_arr = np.array(y)

    n_pos = int(y_arr.sum())
    n_neg = int((1 - y_arr).sum())

    if n_pos < 2 or n_neg < 2 or len(y_arr) < 10:
        return {
            "csd_f1": 0.0, "csd_precision": 0.0, "csd_recall": 0.0,
            "baseline_disagreement_f1": 0.0, "baseline_precision": 0.0, "baseline_recall": 0.0,
            "improvement_pct": 0.0,
            "note": "insufficient_data_for_classifier",
        }

    cv_folds = min(5, n_pos, n_neg)

    # CSD classifier
    try:
        scaler_csd = StandardScaler()
        X_csd_scaled = scaler_csd.fit_transform(X_csd_arr)
        clf_csd = LogisticRegression(random_state=42, max_iter=1000)
        y_pred_csd = cross_val_predict(clf_csd, X_csd_scaled, y_arr, cv=cv_folds)
        csd_f1 = float(f1_score(y_arr, y_pred_csd, zero_division=0))
        csd_prec = float(precision_score(y_arr, y_pred_csd, zero_division=0))
        csd_rec = float(recall_score(y_arr, y_pred_csd, zero_division=0))
    except Exception as e:
        logger.warning(f"CSD classifier failed: {e}")
        csd_f1 = csd_prec = csd_rec = 0.0

    # Baseline (disagreement only) classifier
    try:
        scaler_bl = StandardScaler()
        X_bl_scaled = scaler_bl.fit_transform(X_baseline_arr)
        clf_bl = LogisticRegression(random_state=42, max_iter=1000)
        y_pred_bl = cross_val_predict(clf_bl, X_bl_scaled, y_arr, cv=cv_folds)
        bl_f1 = float(f1_score(y_arr, y_pred_bl, zero_division=0))
        bl_prec = float(precision_score(y_arr, y_pred_bl, zero_division=0))
        bl_rec = float(recall_score(y_arr, y_pred_bl, zero_division=0))
    except Exception as e:
        logger.warning(f"Baseline classifier failed: {e}")
        bl_f1 = bl_prec = bl_rec = 0.0

    improvement = ((csd_f1 - bl_f1) / bl_f1 * 100) if bl_f1 > 0 else 0.0

    return {
        "csd_f1": round(csd_f1, 4),
        "csd_precision": round(csd_prec, 4),
        "csd_recall": round(csd_rec, 4),
        "baseline_disagreement_f1": round(bl_f1, 4),
        "baseline_precision": round(bl_prec, 4),
        "baseline_recall": round(bl_rec, 4),
        "improvement_pct": round(improvement, 2),
    }


# ===========================================================================
# STEP 8: OUTPUT FORMATTING (exp_gen_sol_out schema)
# ===========================================================================

def build_output(
    all_responses: list[dict],
    selected_problems: dict,
    model_analyses: list[dict],
    classifier_results: dict,
    active_models: list[str],
    level_indicators_lookup: dict,
) -> dict:
    """Build output in exp_gen_sol_out.json schema format."""

    # Build problem lookup
    problem_lookup: dict[tuple[int, int], dict] = {}
    for level, problems in selected_problems.items():
        for prob_idx, problem in enumerate(problems):
            problem_lookup[(level, prob_idx)] = problem

    # One dataset per model
    datasets = []
    for model in active_models:
        model_short = model.split("/")[-1]
        examples = []

        model_responses = [r for r in all_responses if r.get("model") == model]

        for resp in model_responses:
            level = resp.get("level")
            prob_idx = resp.get("prob_idx")
            problem = problem_lookup.get((level, prob_idx))
            if problem is None:
                continue

            response_text = resp.get("response", "")
            is_correct = resp.get("correct", False)
            coloring = resp.get("coloring")

            # Format parsed coloring as string
            if coloring:
                parsed_str = ", ".join(
                    f"Node {n}: {c.capitalize()}" for n, c in sorted(coloring.items())
                )
            else:
                parsed_str = "PARSE_FAILED"

            # Get CSD indicators for this (model, level)
            indicators = level_indicators_lookup.get((model, level), {})

            example = {
                "input": problem["input"],
                "output": problem["output"],
                "predict_model_response": response_text[:3000] if response_text else "",
                "predict_is_correct": str(is_correct).lower(),
                "predict_parsed_coloring": parsed_str,
                "metadata_difficulty_level": level,
                "metadata_model": model,
                "metadata_num_nodes": problem["metadata_num_nodes"],
                "metadata_num_colors": problem["metadata_num_colors"],
                "metadata_prob_idx": prob_idx,
                "metadata_resp_idx": resp.get("resp_idx", 0),
                "metadata_csd_accuracy": indicators.get("accuracy", 0.0),
                "metadata_csd_embedding_variance": indicators.get("embedding_variance", 0.0),
                "metadata_csd_dip_statistic": indicators.get("dip_statistic", 0.0),
                "metadata_csd_dip_pvalue": indicators.get("dip_pvalue", 1.0),
                "metadata_csd_silhouette_score": indicators.get("silhouette_score", 0.0),
                "metadata_csd_bimodality_coefficient": indicators.get("bimodality_coefficient", 0.0),
                "metadata_csd_disagreement_rate": indicators.get("disagreement_rate", 1.0),
                "metadata_csd_ashman_d": indicators.get("ashman_d", 0.0),
            }
            examples.append(example)

        if not examples:
            # Schema requires at least 1 example per dataset
            examples = [{"input": "N/A", "output": "N/A", "predict_model_response": "no_data"}]

        datasets.append({
            "dataset": f"graph_coloring_csd_{model_short}",
            "examples": examples,
        })

    # Success criteria
    success_criteria = {
        "flickering_detected": any(
            a.get("leading_indicators", {}).get("dip_first_significant") is not None
            for a in model_analyses
        ),
        "scaling_exponent_valid": any(
            a.get("exponent_in_range", False) for a in model_analyses
        ),
        "classifier_improvement": classifier_results.get("improvement_pct", 0) > 0,
    }

    output = {
        "metadata": {
            "method_name": "CSD_Sampling_Graph_Coloring",
            "description": (
                "Critical Slowing Down indicators for LLM capability boundary "
                "detection on graph coloring constraint satisfaction problems"
            ),
            "models": active_models,
            "temperature": TEMPERATURE,
            "n_problems_per_level": N_PROBLEMS_PER_LEVEL,
            "n_responses_per_problem": N_RESPONSES_PER_PROBLEM,
            "difficulty_levels": DIFFICULTY_LEVELS,
            "total_api_calls": len(all_responses),
            "total_cost_usd": round(cumulative_cost, 4),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "analysis": {
                "models": [
                    {k: v for k, v in a.items() if k != "per_level"}
                    for a in model_analyses
                ],
                "classifier_comparison": classifier_results,
                "success_criteria_met": success_criteria,
                "per_model_per_level": {
                    model: {
                        str(lvl): ind
                        for lvl, ind in level_indicators_lookup.items()
                        if isinstance(lvl, int)
                    }
                    for model in active_models
                    if any(
                        k[0] == model
                        for k in level_indicators_lookup.keys()
                        if isinstance(k, tuple)
                    )
                },
            },
        },
        "datasets": datasets,
    }

    return output


# ===========================================================================
# MAIN
# ===========================================================================

@logger.catch
def main():
    """Main execution with gradual scaling support."""
    parser = argparse.ArgumentParser(description="Graph Coloring CSD Sampling Experiment")
    parser.add_argument(
        "--scale",
        type=str,
        default="full",
        choices=["mini", "small", "medium", "two_models", "full"],
        help="Scale of experiment to run",
    )
    args = parser.parse_args()
    scale = args.scale

    t0 = time.time()
    logger.info(f"=== CSD Graph Coloring Experiment (scale={scale}) ===")

    # Load problems
    selected = load_and_select_problems()

    # Configure scale
    if scale == "mini":
        models = [MODELS[2]]  # cheapest
        max_levels, max_problems, max_responses = 1, 1, 2
    elif scale == "small":
        models = [MODELS[2]]
        max_levels, max_problems, max_responses = 3, 2, 3
    elif scale == "medium":
        models = [MODELS[2]]
        max_levels, max_problems, max_responses = 20, 5, 5
    elif scale == "two_models":
        models = [MODELS[1], MODELS[2]]
        max_levels, max_problems, max_responses = 20, 5, 10
    else:  # full
        models = MODELS
        max_levels, max_problems, max_responses = None, None, None

    expected_calls = (
        len(models)
        * min(max_levels or DIFFICULTY_LEVELS, DIFFICULTY_LEVELS)
        * min(max_problems or N_PROBLEMS_PER_LEVEL, N_PROBLEMS_PER_LEVEL)
        * min(max_responses or N_RESPONSES_PER_PROBLEM, N_RESPONSES_PER_PROBLEM)
    )
    logger.info(f"Expected API calls: {expected_calls}")

    # --- Step 2: Generate responses ---
    t_api = time.time()
    all_responses = asyncio.run(
        generate_all_responses(selected, models, max_levels, max_problems, max_responses)
    )
    api_time = time.time() - t_api
    n_success = sum(1 for r in all_responses if r.get("response"))
    logger.info(
        f"API calls done: {n_success}/{len(all_responses)} successful "
        f"in {api_time:.1f}s, cost=${cumulative_cost:.3f}"
    )

    # --- Step 3: Evaluate constraint satisfaction ---
    logger.info("Evaluating constraint satisfaction...")
    for resp in all_responses:
        if resp.get("response"):
            level = resp.get("level")
            prob_idx = resp.get("prob_idx")
            problem = None
            if level in selected and prob_idx is not None:
                probs = selected[level]
                if prob_idx < len(probs):
                    problem = probs[prob_idx]
            if problem:
                eval_result = evaluate_response(resp["response"], problem)
                resp.update(eval_result)
            else:
                resp["parsed"] = False
                resp["correct"] = False
                resp["coloring"] = None
        else:
            resp["parsed"] = False
            resp["correct"] = False
            resp["coloring"] = None

    n_total = len(all_responses)
    n_parsed = sum(1 for r in all_responses if r.get("parsed"))
    n_correct = sum(1 for r in all_responses if r.get("correct"))
    logger.info(
        f"Evaluation: {n_parsed}/{n_total} parsed ({n_parsed/max(n_total,1)*100:.1f}%), "
        f"{n_correct}/{n_total} correct ({n_correct/max(n_total,1)*100:.1f}%)"
    )

    # --- Step 4: Embed responses ---
    valid_texts = []
    valid_indices = []
    for i, r in enumerate(all_responses):
        if r.get("response"):
            valid_texts.append(r["response"])
            valid_indices.append(i)

    if not valid_texts:
        logger.error("No valid responses to embed! Exiting.")
        return

    embeddings = embed_responses(valid_texts)
    embed_idx_map = {orig_idx: emb_idx for emb_idx, orig_idx in enumerate(valid_indices)}

    # --- Step 5: Compute CSD indicators per (model, level) ---
    logger.info("Computing CSD indicators...")
    active_models = models
    model_level_indicators: dict[str, dict[int, dict]] = {}
    level_indicators_lookup: dict[tuple[str, int], dict] = {}

    for model in active_models:
        model_level_indicators[model] = {}
        active_levels = sorted(selected.keys())
        if max_levels:
            active_levels = active_levels[:max_levels]

        for level in active_levels:
            group_orig_indices = [
                i for i, r in enumerate(all_responses)
                if r.get("model") == model and r.get("level") == level
                and i in embed_idx_map
            ]

            if not group_orig_indices:
                indicators = _empty_indicators()
            else:
                group_emb_indices = [embed_idx_map[i] for i in group_orig_indices]
                group_emb = embeddings[group_emb_indices]
                group_correct = [
                    all_responses[i].get("correct", False) for i in group_orig_indices
                ]
                group_colorings = [
                    all_responses[i].get("coloring") for i in group_orig_indices
                ]
                indicators = compute_csd_indicators(group_emb, group_correct, group_colorings)

            model_level_indicators[model][level] = indicators
            level_indicators_lookup[(model, level)] = indicators

    # --- Step 6: Analysis ---
    logger.info("Running analysis...")
    model_analyses = []
    for model in active_models:
        analysis = analyze_model_results(model, model_level_indicators[model])
        model_analyses.append(analysis)
        logger.info(
            f"  {model}: d*={analysis['d_star']}, "
            f"scaling_exp={analysis['scaling_exponent']}, "
            f"exponent_valid={analysis['exponent_in_range']}"
        )

    # --- Step 7: Classifier comparison ---
    logger.info("Building classifier comparison...")
    classifier_results = build_csd_classifier(model_analyses)
    logger.info(
        f"  CSD F1={classifier_results['csd_f1']:.3f}, "
        f"Baseline F1={classifier_results['baseline_disagreement_f1']:.3f}, "
        f"Improvement={classifier_results['improvement_pct']:.1f}%"
    )

    # --- Step 8: Build and write output ---
    logger.info("Writing output...")
    output = build_output(
        all_responses, selected, model_analyses, classifier_results,
        active_models, level_indicators_lookup,
    )

    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Output written to {out_path}")

    total_time = time.time() - t0
    logger.info(
        f"=== DONE in {total_time:.1f}s | "
        f"Cost: ${cumulative_cost:.3f} | "
        f"Parsed: {n_parsed}/{n_total} | Correct: {n_correct}/{n_total} ==="
    )


if __name__ == "__main__":
    main()
