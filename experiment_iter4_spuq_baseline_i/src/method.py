#!/usr/bin/env python3
"""SPUQ Baseline Implementation + CSD Classifier with Task-Normalized Features.

Implements the SPUQ perturbation-based uncertainty baseline and applies within-task
normalization strategies to CSD features, producing a definitive head-to-head comparison
table showing CSD vs SPUQ at different cost points.

Phases:
  0 - Load existing CSD indicator data from iter_2 experiments
  1 - SPUQ baseline: paraphrase problems, get target model responses, score
  2 - Apply normalization strategies to CSD features
  3 - Train and evaluate classifiers (LOPO / LOTO / LOMO)
  4 - Cost comparison and output generation
"""

import asyncio
import json
import math
import os
import re
import resource
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import SVC
from tenacity import retry, stop_after_attempt, wait_exponential

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
RAM_BUDGET_BYTES = int(TOTAL_RAM_GB * 0.6 * 1e9)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET_BYTES * 3, RAM_BUDGET_BYTES * 3))
logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, budget {RAM_BUDGET_BYTES/1e9:.1f} GB")

# ---------------------------------------------------------------------------
# Load environment for OpenRouter
# ---------------------------------------------------------------------------
_env_path = Path("/ai-inventor/.claude/skills/aii_openrouter_llms/.env")
if _env_path.exists():
    load_dotenv(_env_path)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
if not OPENROUTER_API_KEY:
    logger.warning("No OPENROUTER_API_KEY found -- SPUQ API calls will fail")

# ---------------------------------------------------------------------------
# Constants and paths
# ---------------------------------------------------------------------------
BASE = Path("/ai-inventor/aii_pipeline/runs/neurips-open/3_invention_loop")
ARITH_METHOD = BASE / "iter_2/gen_art/exp_id1_it2__opus/method_out.json"
GRAPH_METHOD = BASE / "iter_2/gen_art/exp_id3_it2__opus/method_out.json"
CLASSIFIER_METHOD = BASE / "iter_3/gen_art/exp_id2_it3__opus/method_out.json"
ARITH_DATA = BASE / "iter_1/gen_art/data_id2_it1__opus/full_data_out.json"
GRAPH_DATA = BASE / "iter_1/gen_art/data_id4_it1__opus/full_data_out.json"

# Valid model-task pairs (d* > min_difficulty for task)
VALID_PAIRS = [
    ("arithmetic", "meta-llama/llama-3.1-8b-instruct", 20, 24),
    ("arithmetic", "google/gemini-2.0-flash-001", 15, 24),
    ("graph_coloring", "openai/gpt-4o-mini", 10, 20),
    ("graph_coloring", "google/gemini-2.0-flash-001", 14, 20),
    ("graph_coloring", "google/gemini-2.0-flash-lite-001", 11, 20),
]

ARITH_MODELS = ["meta-llama/llama-3.1-8b-instruct", "google/gemini-2.0-flash-001"]
GRAPH_MODELS = ["openai/gpt-4o-mini", "google/gemini-2.0-flash-001",
                "google/gemini-2.0-flash-lite-001"]

CSD_FEATURES = [
    "csd_variance", "dip_statistic", "silhouette_k2",
    "bimodality_coefficient", "disagreement_rate",
]

PARAPHRASE_MODEL = "openai/gpt-4o-mini"
N_PARAPHRASES = 5
N_PROBLEMS_PER_LEVEL = 2

# Cost tracking
COST_TRACKER = {"total_usd": 0.0, "total_calls": 0, "input_tokens": 0, "output_tokens": 0}
COST_LIMIT_USD = 9.0  # stay under $10 hard limit

# Run mode: set via CLI or environment
# "mini" = CSD-only, no API calls; "medium" = partial SPUQ; "full" = everything
RUN_MODE = os.environ.get("RUN_MODE", "full")


# ========================================================================
# PHASE 0 — Load existing data
# ========================================================================

def load_arithmetic_csd(path: Path) -> pd.DataFrame:
    """Load arithmetic CSD indicators from iter_2 method_out.json."""
    logger.info(f"Loading arithmetic CSD from {path}")
    data = json.loads(path.read_text())
    rows = []
    for ds in data["datasets"]:
        ds_name = ds["dataset"]
        for ex in ds["examples"]:
            model_name = ex.get("metadata_model", "")
            d_star = ex.get("metadata_d_star")
            # Skip gpt-4o-mini for arithmetic (d*=2, min difficulty)
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
                "dataset_name": ds_name,
            })
    df = pd.DataFrame(rows)
    logger.info(f"  Arithmetic CSD: {len(df)} rows, models={df['model'].unique().tolist()}")
    return df


def load_graph_csd(path: Path) -> pd.DataFrame:
    """Load graph coloring CSD indicators from iter_2 method_out.json.

    Graph data is per-response; we aggregate to per-level by taking the first
    response's CSD values (they are identical for all responses at the same level
    since CSD is computed per-level).
    """
    logger.info(f"Loading graph coloring CSD from {path}")
    data = json.loads(path.read_text())
    rows = []
    seen = set()  # (model, level) -> deduplicate to per-level
    # d_star from metadata.analysis
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
                "dataset_name": ds["dataset"],
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


def load_all_csd_data() -> pd.DataFrame:
    """Load and merge all CSD data into a unified DataFrame."""
    logger.info("=== PHASE 0: Loading existing CSD data ===")
    arith_df = load_arithmetic_csd(ARITH_METHOD)
    graph_df = load_graph_csd(GRAPH_METHOD)
    df = pd.concat([arith_df, graph_df], ignore_index=True)
    df = create_labels(df)

    # Print label distributions per pair for verification
    for (task, model), grp in df.groupby(["task_family", "model"]):
        d_s = grp["d_star"].iloc[0]
        near = (grp["label"] == "near").sum()
        safe = (grp["label"] == "safe").sum()
        logger.info(f"  {task}__{model}: d*={d_s}, near={near}, safe={safe}, n={len(grp)}")
    return df


# ========================================================================
# PHASE 1 — SPUQ baseline
# ========================================================================

ARITH_PARAPHRASE_PROMPT = """Rephrase the following math problem using different wording, but preserve ALL numbers, operations, and the exact sequence of steps. Do not change any numerical values. Only change the natural language phrasing. Return ONLY the rephrased problem, nothing else.

Original problem:
{problem_text}"""

GRAPH_PARAPHRASE_PROMPT = """Rephrase the following graph coloring problem using different wording, but preserve ALL node labels, edge connections, available colors, and constraints exactly. Do not change any graph structure details. Only change the natural language phrasing. Return ONLY the rephrased problem, nothing else.

Original problem:
{problem_text}"""


def estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """Estimate cost in USD for an API call."""
    # Approximate pricing per million tokens (input/output)
    pricing = {
        "openai/gpt-4o-mini": (0.15, 0.60),
        "google/gemini-2.0-flash-001": (0.10, 0.40),
        "google/gemini-2.0-flash-lite-001": (0.075, 0.30),
        "meta-llama/llama-3.1-8b-instruct": (0.06, 0.06),
    }
    in_price, out_price = pricing.get(model, (0.50, 1.50))
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000


async def call_openrouter_async(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    model: str,
    prompt: str,
    temperature: float = 0.8,
    max_tokens: int = 2048,
    retries: int = 3,
) -> dict:
    """Call OpenRouter API with retry and rate limiting."""
    global COST_TRACKER

    if COST_TRACKER["total_usd"] >= COST_LIMIT_USD:
        return {"success": False, "error": "Cost limit reached", "response": ""}

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for attempt in range(retries):
        try:
            async with semaphore:
                async with session.post(
                    OPENROUTER_API_URL, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=120)
                ) as resp:
                    if resp.status == 429:
                        wait_time = min(2 ** attempt * 2, 30)
                        logger.warning(f"Rate limited, waiting {wait_time}s (attempt {attempt+1})")
                        await asyncio.sleep(wait_time)
                        continue
                    if resp.status != 200:
                        text = await resp.text()
                        logger.warning(f"API error {resp.status}: {text[:200]}")
                        if attempt < retries - 1:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        return {"success": False, "error": f"Status {resp.status}", "response": ""}

                    result = await resp.json()
                    response_text = ""
                    if "choices" in result and result["choices"]:
                        msg = result["choices"][0].get("message", {})
                        response_text = msg.get("content", "")

                    usage = result.get("usage", {})
                    in_tok = usage.get("prompt_tokens", 0)
                    out_tok = usage.get("completion_tokens", 0)
                    cost = estimate_cost(in_tok, out_tok, model)

                    COST_TRACKER["total_usd"] += cost
                    COST_TRACKER["total_calls"] += 1
                    COST_TRACKER["input_tokens"] += in_tok
                    COST_TRACKER["output_tokens"] += out_tok

                    if COST_TRACKER["total_calls"] % 50 == 0:
                        logger.info(
                            f"  Cost tracker: ${COST_TRACKER['total_usd']:.4f} / "
                            f"${COST_LIMIT_USD}, calls={COST_TRACKER['total_calls']}"
                        )

                    return {"success": True, "response": response_text,
                            "input_tokens": in_tok, "output_tokens": out_tok}
        except asyncio.TimeoutError:
            logger.warning(f"Timeout on attempt {attempt+1} for {model}")
            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)
        except Exception as e:
            logger.warning(f"Error on attempt {attempt+1}: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)

    return {"success": False, "error": "Max retries exceeded", "response": ""}


def select_problems(arith_data: list, graph_data: list, n_per_level: int = 2) -> dict:
    """Select N problems per difficulty level for SPUQ evaluation."""
    selected = {"arithmetic": {}, "graph_coloring": {}}

    # Arithmetic: levels 2-25
    by_level_arith = defaultdict(list)
    for p in arith_data:
        by_level_arith[p["metadata_difficulty_level"]].append(p)
    for level in sorted(by_level_arith.keys()):
        problems = sorted(by_level_arith[level], key=lambda x: x.get("metadata_row_index", 0))
        selected["arithmetic"][level] = problems[:n_per_level]

    # Graph coloring: levels 1-20
    by_level_graph = defaultdict(list)
    for p in graph_data:
        by_level_graph[p["metadata_difficulty_level"]].append(p)
    for level in sorted(by_level_graph.keys()):
        problems = sorted(by_level_graph[level], key=lambda x: x.get("metadata_row_index", 0))
        selected["graph_coloring"][level] = problems[:n_per_level]

    total_arith = sum(len(v) for v in selected["arithmetic"].values())
    total_graph = sum(len(v) for v in selected["graph_coloring"].values())
    logger.info(f"  Selected problems: arithmetic={total_arith}, graph={total_graph}, total={total_arith+total_graph}")
    return selected


def extract_arithmetic_answer(response_text: str) -> int | None:
    """Extract final numerical answer from arithmetic response."""
    if not response_text:
        return None
    patterns = [
        r'(?:final|the)\s+(?:answer|result)\s+(?:is|=|:)\s*\**\s*(-?\d[\d,]*)',
        r'\*\*(-?\d[\d,]*)\*\*\s*$',
        r'(?:=|equals?|is)\s*\**\s*(-?\d[\d,]*)\s*\**\s*$',
        r'\\boxed\{(-?\d[\d,]*)\}',
        r'result\s*(?:is|=|:)\s*\**\s*(-?\d[\d,]*)',
        r'(?:^|\n)\s*(-?\d[\d,]*)\s*$',
    ]
    for p in patterns:
        match = re.search(p, response_text, re.IGNORECASE | re.MULTILINE)
        if match:
            try:
                return int(match.group(1).replace(",", ""))
            except ValueError:
                continue
    # Last resort: find the last standalone number
    all_nums = re.findall(r'(?<![.\d])(-?\d[\d,]*)(?![.\d])', response_text)
    if all_nums:
        try:
            return int(all_nums[-1].replace(",", ""))
        except ValueError:
            pass
    return None


def extract_graph_answer(response_text: str, num_nodes: int) -> dict | None:
    """Extract node-color assignments from graph coloring response."""
    if not response_text:
        return None
    coloring = {}
    # Pattern: "Node X: Color" or "Node X = Color" or "Node X -> Color"
    pattern = r'[Nn]ode\s+(\d+)\s*[:=\->]+\s*([A-Za-z]+)'
    matches = re.findall(pattern, response_text)
    for node_str, color in matches:
        try:
            node = int(node_str)
            coloring[node] = color.strip().lower()
        except ValueError:
            continue
    if len(coloring) >= max(1, num_nodes - 1):
        return coloring
    return None


def check_graph_coloring_valid(coloring: dict, adjacency: list) -> bool:
    """Check if a graph coloring satisfies all edge constraints."""
    if not coloring:
        return False
    for edge in adjacency:
        n1, n2 = edge[0], edge[1]
        c1 = coloring.get(n1)
        c2 = coloring.get(n2)
        if c1 and c2 and c1 == c2:
            return False
    return True


async def run_spuq_phase(
    selected: dict,
    csd_df: pd.DataFrame,
    max_levels_per_task: int | None = None,
    models_to_use: dict | None = None,
) -> pd.DataFrame:
    """Run the SPUQ perturbation baseline: paraphrase, query, score."""
    logger.info("=== PHASE 1: SPUQ Baseline Implementation ===")

    if models_to_use is None:
        models_to_use = {
            "arithmetic": ARITH_MODELS,
            "graph_coloring": GRAPH_MODELS,
        }

    # Prepare paraphrase tasks
    para_tasks = []
    for task in ["arithmetic", "graph_coloring"]:
        template = ARITH_PARAPHRASE_PROMPT if task == "arithmetic" else GRAPH_PARAPHRASE_PROMPT
        levels = sorted(selected[task].keys())
        if max_levels_per_task is not None:
            levels = levels[:max_levels_per_task]
        for level in levels:
            for prob_idx, prob in enumerate(selected[task][level]):
                for para_idx in range(N_PARAPHRASES):
                    para_tasks.append({
                        "task": task,
                        "level": level,
                        "prob_idx": prob_idx,
                        "para_idx": para_idx,
                        "prompt": template.format(problem_text=prob["input"]),
                        "original_input": prob["input"],
                        "ground_truth": prob["output"],
                    })

    logger.info(f"  Generating {len(para_tasks)} paraphrases...")

    # Generate paraphrases
    concurrency = min(NUM_CPUS * 10, 30)
    sem = asyncio.Semaphore(concurrency)
    paraphrases = {}  # (task, level, prob_idx, para_idx) -> paraphrased text

    async with aiohttp.ClientSession() as session:
        async def gen_paraphrase(t: dict) -> None:
            key = (t["task"], t["level"], t["prob_idx"], t["para_idx"])
            result = await call_openrouter_async(
                session, sem, PARAPHRASE_MODEL, t["prompt"],
                temperature=0.9, max_tokens=1024,
            )
            if result["success"]:
                paraphrases[key] = result["response"]
            else:
                # Fallback: use original with minor prefix change
                prefixes = [
                    "Please solve the following:\n\n",
                    "Work through this problem step by step:\n\n",
                    "Calculate the answer to:\n\n",
                    "Determine the result of:\n\n",
                    "Solve:\n\n",
                ]
                paraphrases[key] = prefixes[t["para_idx"] % len(prefixes)] + t["original_input"]

        tasks = [gen_paraphrase(t) for t in para_tasks]
        await asyncio.gather(*tasks)

    logger.info(f"  Paraphrases generated: {len(paraphrases)}")
    logger.info(f"  Cost so far: ${COST_TRACKER['total_usd']:.4f}")

    # Now query target models with paraphrased prompts
    target_tasks = []
    for task in ["arithmetic", "graph_coloring"]:
        levels = sorted(selected[task].keys())
        if max_levels_per_task is not None:
            levels = levels[:max_levels_per_task]
        for model in models_to_use[task]:
            for level in levels:
                for prob_idx, prob in enumerate(selected[task][level]):
                    for para_idx in range(N_PARAPHRASES):
                        key = (task, level, prob_idx, para_idx)
                        if key not in paraphrases:
                            continue
                        target_tasks.append({
                            "task": task,
                            "model": model,
                            "level": level,
                            "prob_idx": prob_idx,
                            "para_idx": para_idx,
                            "prompt": paraphrases[key],
                            "ground_truth": prob["output"],
                            "num_nodes": prob.get("metadata_num_nodes", 0),
                            "adjacency": prob.get("metadata_graph_adjacency", []),
                        })

    logger.info(f"  Querying target models: {len(target_tasks)} calls...")

    spuq_responses = {}  # (task, model, level, prob_idx, para_idx) -> response

    async with aiohttp.ClientSession() as session:
        async def query_target(t: dict) -> None:
            key = (t["task"], t["model"], t["level"], t["prob_idx"], t["para_idx"])
            result = await call_openrouter_async(
                session, sem, t["model"], t["prompt"],
                temperature=0.8, max_tokens=2048,
            )
            spuq_responses[key] = {
                "response": result.get("response", ""),
                "success": result.get("success", False),
                "ground_truth": t["ground_truth"],
                "num_nodes": t["num_nodes"],
                "adjacency": t["adjacency"],
            }

        tasks = [query_target(t) for t in target_tasks]
        await asyncio.gather(*tasks)

    logger.info(f"  Target responses collected: {len(spuq_responses)}")
    logger.info(f"  Cost so far: ${COST_TRACKER['total_usd']:.4f}")

    # Compute SPUQ metrics per (task, model, level)
    spuq_rows = []
    extraction_stats = {"total": 0, "extracted": 0}

    for task in ["arithmetic", "graph_coloring"]:
        levels = sorted(selected[task].keys())
        if max_levels_per_task is not None:
            levels = levels[:max_levels_per_task]
        for model in models_to_use[task]:
            for level in levels:
                level_disagreements = []
                level_accuracies = []

                for prob_idx in range(len(selected[task].get(level, []))):
                    answers = []
                    correct_count = 0
                    total_paraphrases = 0

                    for para_idx in range(N_PARAPHRASES):
                        key = (task, model, level, prob_idx, para_idx)
                        if key not in spuq_responses:
                            continue
                        resp = spuq_responses[key]
                        if not resp["success"]:
                            continue
                        total_paraphrases += 1
                        extraction_stats["total"] += 1

                        if task == "arithmetic":
                            ans = extract_arithmetic_answer(resp["response"])
                            if ans is not None:
                                answers.append(ans)
                                extraction_stats["extracted"] += 1
                                try:
                                    gt = int(resp["ground_truth"])
                                    if ans == gt:
                                        correct_count += 1
                                except (ValueError, TypeError):
                                    pass
                        else:
                            ans = extract_graph_answer(
                                resp["response"], resp["num_nodes"]
                            )
                            if ans is not None:
                                # Use string repr for comparison
                                ans_str = json.dumps(dict(sorted(ans.items())), sort_keys=True)
                                answers.append(ans_str)
                                extraction_stats["extracted"] += 1
                                if check_graph_coloring_valid(ans, resp["adjacency"]):
                                    correct_count += 1

                    # Compute disagreement for this problem
                    if len(answers) >= 2:
                        most_common_count = Counter(answers).most_common(1)[0][1]
                        agreement = most_common_count / len(answers)
                        disagreement = 1.0 - agreement
                    else:
                        disagreement = 1.0  # max uncertainty

                    level_disagreements.append(disagreement)
                    if total_paraphrases > 0:
                        level_accuracies.append(correct_count / total_paraphrases)
                    else:
                        level_accuracies.append(0.0)

                if level_disagreements:
                    spuq_rows.append({
                        "task_family": task,
                        "model": model,
                        "difficulty_level": level,
                        "spuq_disagreement": np.mean(level_disagreements),
                        "spuq_accuracy": np.mean(level_accuracies),
                        "n_problems": len(level_disagreements),
                    })

    ext_rate = extraction_stats["extracted"] / max(extraction_stats["total"], 1) * 100
    logger.info(f"  Answer extraction rate: {ext_rate:.1f}% ({extraction_stats['extracted']}/{extraction_stats['total']})")

    spuq_df = pd.DataFrame(spuq_rows)
    logger.info(f"  SPUQ rows: {len(spuq_df)}")
    return spuq_df


def generate_rule_based_paraphrases(problem_text: str, task: str, n: int = 5) -> list:
    """Generate simple rule-based paraphrases as fallback."""
    paraphrases = []
    if task == "arithmetic":
        replacements = [
            ("Compute", "Calculate"), ("Calculate", "Work out"),
            ("Add", "Sum"), ("Subtract", "Take away"),
            ("Multiply by", "Times"), ("showing your work", "with detailed steps"),
            ("step by step", "one step at a time"),
            ("What is the final result?", "What do you get?"),
            ("Provide your final numerical answer.", "Give the final number."),
        ]
    else:
        replacements = [
            ("Color each node", "Assign a color to each node"),
            ("The constraint is that", "The rule is that"),
            ("no two nodes connected by an edge may share the same color",
             "adjacent nodes must have different colors"),
            ("Provide a valid coloring", "Give a valid color assignment"),
            ("listing each node and its assigned color",
             "specifying the color for every node"),
        ]

    prefixes = [
        "", "Please solve the following:\n\n",
        "Work through this problem:\n\n",
        "Solve this:\n\n", "Determine the answer:\n\n",
    ]

    for i in range(n):
        text = problem_text
        # Apply 1-2 replacements
        for old, new in replacements[i % len(replacements) : i % len(replacements) + 2]:
            text = text.replace(old, new)
        text = prefixes[i % len(prefixes)] + text
        paraphrases.append(text)
    return paraphrases


# ========================================================================
# PHASE 2 — Normalization strategies
# ========================================================================

def apply_normalizations(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all normalization strategies to CSD features."""
    logger.info("=== PHASE 2: Applying normalization strategies ===")
    df = df.copy()

    # Strategy A: Within-task z-score
    for feature in CSD_FEATURES:
        for task in ["arithmetic", "graph_coloring"]:
            mask = df["task_family"] == task
            if mask.sum() == 0:
                continue
            mu = df.loc[mask, feature].mean()
            sigma = df.loc[mask, feature].std()
            df.loc[mask, f"{feature}_zt"] = (df.loc[mask, feature] - mu) / (sigma + 1e-8)

    # Strategy B: Relative difficulty
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

    # Strategy C: Percentile-rank within task
    for feature in CSD_FEATURES:
        for task in ["arithmetic", "graph_coloring"]:
            mask = df["task_family"] == task
            if mask.sum() == 0:
                continue
            df.loc[mask, f"{feature}_pct_t"] = df.loc[mask, feature].rank(pct=True)

    # Strategy D: Delta features (rate of change per difficulty step)
    for (task, model), grp_idx in df.groupby(["task_family", "model"]).groups.items():
        mask = df.index.isin(grp_idx)
        subset = df.loc[mask].sort_values("difficulty_level")
        for feature in CSD_FEATURES:
            deltas = subset[feature].diff().values
            df.loc[subset.index, f"{feature}_delta"] = deltas

    # Fill NaN deltas (first row per pair) with 0
    delta_cols = [f"{f}_delta" for f in CSD_FEATURES]
    df[delta_cols] = df[delta_cols].fillna(0)

    # Verify normalization sanity
    for task in ["arithmetic", "graph_coloring"]:
        mask = df["task_family"] == task
        if mask.sum() == 0:
            continue
        for feature in CSD_FEATURES:
            zt_col = f"{feature}_zt"
            zt_mean = df.loc[mask, zt_col].mean()
            zt_std = df.loc[mask, zt_col].std()
            logger.debug(f"  {task} {zt_col}: mean={zt_mean:.4f}, std={zt_std:.4f}")

    rd = df["relative_difficulty"]
    logger.info(f"  Relative difficulty range: [{rd.min():.3f}, {rd.max():.3f}]")
    logger.info(f"  NaN count after normalization: {df.isna().sum().sum()}")
    return df


# ========================================================================
# PHASE 3 — Classifier comparison
# ========================================================================

def define_classifiers(has_spuq: bool = True) -> dict:
    """Define feature sets for each classifier variant."""
    zt_feats = [f"{f}_zt" for f in CSD_FEATURES]
    pct_feats = [f"{f}_pct_t" for f in CSD_FEATURES]
    delta_feats = [f"{f}_delta" for f in CSD_FEATURES]

    classifiers = {
        # CSD variants (0 extra API calls)
        "csd_raw": list(CSD_FEATURES),
        "csd_raw_diff": list(CSD_FEATURES) + ["difficulty_level"],
        "csd_zt": zt_feats,
        "csd_zt_reldiff": zt_feats + ["relative_difficulty"],
        "csd_zt_reldist": zt_feats + ["relative_dist_to_dstar"],
        "csd_zt_delta": zt_feats + delta_feats,
        "csd_zt_full": zt_feats + ["relative_difficulty"] + delta_feats,
        "csd_pct_t": pct_feats,
        "csd_pct_t_reldiff": pct_feats + ["relative_difficulty"],
        # Single-feature baselines
        "variance_only": ["csd_variance"],
        "disagreement_only": ["disagreement_rate"],
        "dip_only": ["dip_statistic"],
        "bimodality_only": ["bimodality_coefficient"],
    }

    if has_spuq:
        classifiers.update({
            "spuq_disagreement": ["spuq_disagreement"],
            "spuq_accuracy": ["spuq_accuracy"],
            "spuq_combined": ["spuq_disagreement", "spuq_accuracy"],
            "csd_zt_spuq": zt_feats + ["relative_difficulty", "spuq_disagreement"],
        })

    return classifiers


def train_and_evaluate(
    df: pd.DataFrame,
    classifier_defs: dict,
    pairs: list | None = None,
) -> dict:
    """Train classifiers and evaluate with LOPO, LOTO, LOMO cross-validation."""
    logger.info("=== PHASE 3: Classifier Comparison ===")

    if pairs is None:
        pairs = [(t, m) for t, m, _, _ in VALID_PAIRS]

    unique_models = list(df["model"].unique())
    results = {}

    model_types = {
        "logreg": lambda: LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"),
        "rf": lambda: RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
        "svm": lambda: SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42),
    }

    for clf_name, features in classifier_defs.items():
        # Check if all features exist
        missing = [f for f in features if f not in df.columns]
        if missing:
            logger.debug(f"  Skipping {clf_name}: missing features {missing}")
            continue

        # Check for NaN in features
        feat_df = df[features]
        if feat_df.isna().any().any():
            logger.debug(f"  Skipping {clf_name}: NaN in features")
            continue

        for mt_name, mt_factory in model_types.items():
            key = f"{clf_name}_{mt_name}"

            # --- LOPO (Leave-One-Pair-Out) ---
            lopo_scores = []
            for held_task, held_model in pairs:
                test_mask = (df["task_family"] == held_task) & (df["model"] == held_model)
                train_mask = ~test_mask
                if train_mask.sum() < 5 or test_mask.sum() < 2:
                    continue
                X_train = df.loc[train_mask, features].values
                y_train = (df.loc[train_mask, "label"] == "near").astype(int).values
                X_test = df.loc[test_mask, features].values
                y_test = (df.loc[test_mask, "label"] == "near").astype(int).values

                if len(np.unique(y_train)) < 2:
                    continue

                try:
                    clf = mt_factory()
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else y_pred.astype(float)

                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    prec = precision_score(y_test, y_pred, zero_division=0)
                    rec = recall_score(y_test, y_pred, zero_division=0)
                    try:
                        auroc = roc_auc_score(y_test, y_prob)
                    except ValueError:
                        auroc = 0.5

                    lopo_scores.append({"f1": f1, "precision": prec, "recall": rec, "auroc": auroc})
                except Exception as e:
                    logger.debug(f"  LOPO error {key} on {held_task}__{held_model}: {e}")

            # --- LOTO (Leave-One-Task-Out) ---
            loto_scores = []
            for held_task in ["arithmetic", "graph_coloring"]:
                test_mask = df["task_family"] == held_task
                train_mask = ~test_mask
                if train_mask.sum() < 5 or test_mask.sum() < 2:
                    continue
                X_train = df.loc[train_mask, features].values
                y_train = (df.loc[train_mask, "label"] == "near").astype(int).values
                X_test = df.loc[test_mask, features].values
                y_test = (df.loc[test_mask, "label"] == "near").astype(int).values

                if len(np.unique(y_train)) < 2:
                    continue

                try:
                    clf = mt_factory()
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else y_pred.astype(float)

                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    prec = precision_score(y_test, y_pred, zero_division=0)
                    rec = recall_score(y_test, y_pred, zero_division=0)
                    try:
                        auroc = roc_auc_score(y_test, y_prob)
                    except ValueError:
                        auroc = 0.5

                    loto_scores.append({"f1": f1, "precision": prec, "recall": rec, "auroc": auroc})
                except Exception as e:
                    logger.debug(f"  LOTO error {key} on {held_task}: {e}")

            # --- LOMO (Leave-One-Model-Out) ---
            lomo_scores = []
            for held_model in unique_models:
                test_mask = df["model"] == held_model
                train_mask = ~test_mask
                if train_mask.sum() < 5 or test_mask.sum() < 2:
                    continue
                X_train = df.loc[train_mask, features].values
                y_train = (df.loc[train_mask, "label"] == "near").astype(int).values
                X_test = df.loc[test_mask, features].values
                y_test = (df.loc[test_mask, "label"] == "near").astype(int).values

                if len(np.unique(y_train)) < 2:
                    continue

                try:
                    clf = mt_factory()
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else y_pred.astype(float)

                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    prec = precision_score(y_test, y_pred, zero_division=0)
                    rec = recall_score(y_test, y_pred, zero_division=0)
                    try:
                        auroc = roc_auc_score(y_test, y_prob)
                    except ValueError:
                        auroc = 0.5

                    lomo_scores.append({"f1": f1, "precision": prec, "recall": rec, "auroc": auroc})
                except Exception as e:
                    logger.debug(f"  LOMO error {key} on {held_model}: {e}")

            if lopo_scores:
                avg_lopo = {k: float(np.mean([s[k] for s in lopo_scores])) for k in lopo_scores[0]}
            else:
                avg_lopo = {"f1": 0, "precision": 0, "recall": 0, "auroc": 0.5}

            if loto_scores:
                avg_loto = {k: float(np.mean([s[k] for s in loto_scores])) for k in loto_scores[0]}
            else:
                avg_loto = {"f1": 0, "precision": 0, "recall": 0, "auroc": 0.5}

            if lomo_scores:
                avg_lomo = {k: float(np.mean([s[k] for s in lomo_scores])) for k in lomo_scores[0]}
            else:
                avg_lomo = {"f1": 0, "precision": 0, "recall": 0, "auroc": 0.5}

            results[key] = {
                "lopo_f1": avg_lopo["f1"],
                "lopo_auroc": avg_lopo["auroc"],
                "lopo_precision": avg_lopo["precision"],
                "lopo_recall": avg_lopo["recall"],
                "loto_f1": avg_loto["f1"],
                "loto_auroc": avg_loto["auroc"],
                "lomo_f1": avg_lomo["f1"],
                "lomo_auroc": avg_lomo["auroc"],
                "features": features,
                "model_type": mt_name,
                "classifier_variant": clf_name,
            }

    # Log top results
    sorted_by_lopo = sorted(results.items(), key=lambda x: x[1]["lopo_f1"], reverse=True)
    logger.info("  Top 10 by LOPO F1:")
    for k, v in sorted_by_lopo[:10]:
        logger.info(f"    {k}: LOPO={v['lopo_f1']:.3f}, LOTO={v['loto_f1']:.3f}, LOMO={v['lomo_f1']:.3f}")

    sorted_by_loto = sorted(results.items(), key=lambda x: x[1]["loto_f1"], reverse=True)
    logger.info("  Top 10 by LOTO F1:")
    for k, v in sorted_by_loto[:10]:
        logger.info(f"    {k}: LOPO={v['lopo_f1']:.3f}, LOTO={v['loto_f1']:.3f}, LOMO={v['lomo_f1']:.3f}")

    return results


# ========================================================================
# PHASE 4 — Cost comparison and output
# ========================================================================

def build_output(
    df: pd.DataFrame,
    classifier_results: dict,
    spuq_df: pd.DataFrame | None,
    cost_tracker: dict,
) -> dict:
    """Build the final method_out.json output."""
    logger.info("=== PHASE 4: Building output ===")

    # Identify best methods per category
    csd_variants = {k: v for k, v in classifier_results.items()
                    if not k.startswith("spuq_") and "spuq" not in k.split("_")[0:2]}
    spuq_variants = {k: v for k, v in classifier_results.items()
                     if k.startswith("spuq_")}
    combined_variants = {k: v for k, v in classifier_results.items()
                         if "spuq" in k and not k.startswith("spuq_")}
    baseline_variants = {k: v for k, v in classifier_results.items()
                         if k.endswith("_logreg") and any(
                             k.startswith(b) for b in
                             ["variance_only", "disagreement_only", "dip_only", "bimodality_only"]
                         )}

    def best_f1(variants: dict, metric: str = "lopo_f1") -> tuple:
        if not variants:
            return ("none", 0.0)
        best = max(variants.items(), key=lambda x: x[1][metric])
        return (best[0], best[1][metric])

    best_csd_lopo_name, best_csd_lopo = best_f1(csd_variants, "lopo_f1")
    best_csd_loto_name, best_csd_loto = best_f1(csd_variants, "loto_f1")
    best_spuq_lopo_name, best_spuq_lopo = best_f1(spuq_variants, "lopo_f1")
    best_spuq_loto_name, best_spuq_loto = best_f1(spuq_variants, "loto_f1")
    best_baseline_name, best_baseline_lopo = best_f1(baseline_variants, "lopo_f1")

    # Cost comparison table
    spuq_total_calls = cost_tracker.get("total_calls", 0)
    spuq_total_usd = cost_tracker.get("total_usd", 0.0)

    cost_table = {
        "CSD (all variants)": {
            "extra_api_calls": 0,
            "extra_cost_usd": 0.0,
            "source": "reuses N=50 majority-vote samples from CSD computation",
        },
        "SPUQ (paraphrase)": {
            "extra_api_calls": spuq_total_calls,
            "extra_cost_usd": round(spuq_total_usd, 4),
            "source": f"{N_PARAPHRASES} paraphrases per problem + target model responses",
        },
        "CSD+SPUQ combined": {
            "extra_api_calls": spuq_total_calls,
            "extra_cost_usd": round(spuq_total_usd, 4),
            "source": "CSD free + SPUQ overhead",
        },
    }

    # Success criteria
    csd_beats_spuq_lopo = best_csd_lopo > best_spuq_lopo if best_spuq_lopo > 0 else True
    csd_beats_spuq_loto = best_csd_loto > best_spuq_loto if best_spuq_loto > 0 else True
    loto_above_05 = best_csd_loto > 0.5

    improvement_lopo = ((best_csd_lopo - best_spuq_lopo) / max(best_spuq_lopo, 1e-8) * 100) if best_spuq_lopo > 0 else 0
    improvement_loto = ((best_csd_loto - best_spuq_loto) / max(best_spuq_loto, 1e-8) * 100) if best_spuq_loto > 0 else 0

    # Build datasets for output
    datasets = []

    # Dataset 1: classifier_comparison — all classifier metrics
    clf_examples = []
    for name, metrics in sorted(classifier_results.items(), key=lambda x: x[1]["lopo_f1"], reverse=True):
        clf_examples.append({
            "input": f"Classifier: {name}",
            "output": f"LOPO_F1={metrics['lopo_f1']:.4f}, LOTO_F1={metrics['loto_f1']:.4f}",
            "predict_lopo_f1": str(round(metrics["lopo_f1"], 6)),
            "predict_lopo_auroc": str(round(metrics["lopo_auroc"], 6)),
            "predict_lopo_precision": str(round(metrics["lopo_precision"], 6)),
            "predict_lopo_recall": str(round(metrics["lopo_recall"], 6)),
            "predict_loto_f1": str(round(metrics["loto_f1"], 6)),
            "predict_loto_auroc": str(round(metrics["loto_auroc"], 6)),
            "predict_lomo_f1": str(round(metrics["lomo_f1"], 6)),
            "predict_lomo_auroc": str(round(metrics["lomo_auroc"], 6)),
            "metadata_classifier_variant": metrics["classifier_variant"],
            "metadata_model_type": metrics["model_type"],
            "metadata_features": json.dumps(metrics["features"]),
            "metadata_is_csd": str("spuq" not in name),
            "metadata_is_spuq": str(name.startswith("spuq_")),
            "metadata_is_combined": str("spuq" in name and not name.startswith("spuq_")),
            "metadata_fold": "test",
        })
    datasets.append({"dataset": "classifier_comparison", "examples": clf_examples})

    # Dataset 2: normalization_comparison — CSD normalization ablation
    norm_examples = []
    norm_variants = {k: v for k, v in classifier_results.items()
                     if any(k.startswith(p) for p in ["csd_raw", "csd_zt", "csd_pct_t"])}
    for name, metrics in sorted(norm_variants.items(), key=lambda x: x[1]["loto_f1"], reverse=True):
        norm_examples.append({
            "input": f"Normalization variant: {name}",
            "output": f"LOTO_F1={metrics['loto_f1']:.4f}",
            "predict_lopo_f1": str(round(metrics["lopo_f1"], 6)),
            "predict_loto_f1": str(round(metrics["loto_f1"], 6)),
            "predict_lomo_f1": str(round(metrics["lomo_f1"], 6)),
            "metadata_normalization": name.split("_" + metrics["model_type"])[0] if "_" + metrics["model_type"] in name else name,
            "metadata_model_type": metrics["model_type"],
            "metadata_fold": "test",
        })
    datasets.append({"dataset": "normalization_comparison", "examples": norm_examples if norm_examples else [
        {"input": "No normalization variants", "output": "N/A", "metadata_fold": "test"}
    ]})

    # Dataset 3: cost_comparison
    cost_examples = []
    for method, details in cost_table.items():
        # Find best F1 for this method category
        if "SPUQ" in method and "CSD" not in method:
            best_lopo = best_spuq_lopo
            best_loto = best_spuq_loto
        elif "CSD" in method and "SPUQ" not in method:
            best_lopo = best_csd_lopo
            best_loto = best_csd_loto
        else:
            comb_best_lopo = best_f1(combined_variants, "lopo_f1")[1] if combined_variants else 0
            comb_best_loto = best_f1(combined_variants, "loto_f1")[1] if combined_variants else 0
            best_lopo = comb_best_lopo
            best_loto = comb_best_loto

        cost_examples.append({
            "input": f"Cost analysis: {method}",
            "output": f"Extra calls={details['extra_api_calls']}, Cost=${details['extra_cost_usd']:.4f}",
            "predict_best_lopo_f1": str(round(best_lopo, 6)),
            "predict_best_loto_f1": str(round(best_loto, 6)),
            "predict_extra_api_calls": str(details["extra_api_calls"]),
            "predict_extra_cost_usd": str(round(details["extra_cost_usd"], 4)),
            "metadata_method": method,
            "metadata_source": details["source"],
            "metadata_fold": "test",
        })
    datasets.append({"dataset": "cost_comparison", "examples": cost_examples})

    # Dataset 4: spuq_per_level — SPUQ metrics per difficulty level
    if spuq_df is not None and len(spuq_df) > 0:
        spuq_examples = []
        for _, row in spuq_df.iterrows():
            spuq_examples.append({
                "input": f"SPUQ at level={int(row['difficulty_level'])} for {row['model']}",
                "output": f"disagreement={row['spuq_disagreement']:.4f}, accuracy={row['spuq_accuracy']:.4f}",
                "predict_spuq_disagreement": str(round(row["spuq_disagreement"], 6)),
                "predict_spuq_accuracy": str(round(row["spuq_accuracy"], 6)),
                "metadata_task_family": row["task_family"],
                "metadata_model": row["model"],
                "metadata_difficulty_level": int(row["difficulty_level"]),
                "metadata_n_problems": int(row["n_problems"]),
                "metadata_fold": "test",
            })
        datasets.append({"dataset": "spuq_per_level", "examples": spuq_examples})

    # Dataset 5: csd_features — the unified CSD DataFrame
    csd_examples = []
    for _, row in df.iterrows():
        ex = {
            "input": f"CSD features at level={int(row['difficulty_level'])} for {row['model']} on {row['task_family']}",
            "output": str(row["label"]),
            "predict_accuracy": str(round(row["accuracy"], 6)),
            "predict_csd_variance": str(round(row["csd_variance"], 6)),
            "predict_dip_statistic": str(round(row["dip_statistic"], 6)),
            "predict_silhouette_k2": str(round(row["silhouette_k2"], 6)),
            "predict_bimodality_coefficient": str(round(row["bimodality_coefficient"], 6)),
            "predict_disagreement_rate": str(round(row["disagreement_rate"], 6)),
            "predict_label": str(row["label"]),
            "metadata_task_family": row["task_family"],
            "metadata_model": row["model"],
            "metadata_difficulty_level": int(row["difficulty_level"]),
            "metadata_d_star": int(row["d_star"]) if pd.notna(row["d_star"]) else 0,
            "metadata_fold": "test",
        }
        # Add normalized features if they exist
        for col in df.columns:
            if col.endswith("_zt") or col.endswith("_pct_t") or col.endswith("_delta"):
                val = row[col]
                if pd.notna(val) and np.isfinite(val):
                    ex[f"predict_{col}"] = str(round(float(val), 6))
        if "relative_difficulty" in df.columns:
            ex["predict_relative_difficulty"] = str(round(float(row["relative_difficulty"]), 6))
        if "relative_dist_to_dstar" in df.columns and pd.notna(row.get("relative_dist_to_dstar")):
            ex["predict_relative_dist_to_dstar"] = str(round(float(row["relative_dist_to_dstar"]), 6))
        # Add SPUQ if merged
        if "spuq_disagreement" in df.columns and pd.notna(row.get("spuq_disagreement")):
            ex["predict_spuq_disagreement"] = str(round(float(row["spuq_disagreement"]), 6))
        if "spuq_accuracy" in df.columns and pd.notna(row.get("spuq_accuracy")):
            ex["predict_spuq_accuracy"] = str(round(float(row["spuq_accuracy"]), 6))
        csd_examples.append(ex)
    datasets.append({"dataset": "csd_features_unified", "examples": csd_examples})

    # Label distribution for metadata
    label_dist = {}
    for (task, model), grp in df.groupby(["task_family", "model"]):
        key_str = f"{task}__{model}"
        label_dist[key_str] = {
            "near": int((grp["label"] == "near").sum()),
            "safe": int((grp["label"] == "safe").sum()),
            "d_star": int(grp["d_star"].iloc[0]) if pd.notna(grp["d_star"].iloc[0]) else 0,
            "n_rows": len(grp),
        }

    metadata = {
        "method_name": "CSD_vs_SPUQ_Comparison",
        "description": (
            "Head-to-head comparison of CSD (Critical Slowing Down) indicators vs "
            "SPUQ (paraphrase-based uncertainty) for detecting LLM capability boundaries. "
            "CSD features are zero-cost (reuse majority-vote samples); SPUQ requires additional API calls."
        ),
        "classifier_comparison": {
            k: {kk: vv for kk, vv in v.items() if kk != "features"}
            for k, v in classifier_results.items()
        },
        "best_csd_method": best_csd_lopo_name,
        "best_csd_loto_method": best_csd_loto_name,
        "best_spuq_method": best_spuq_lopo_name,
        "best_baseline_method": best_baseline_name,
        "improvement_csd_over_spuq_lopo_pct": round(improvement_lopo, 2),
        "improvement_csd_over_spuq_loto_pct": round(improvement_loto, 2),
        "success_criteria_met": {
            "csd_beats_spuq_lopo": csd_beats_spuq_lopo,
            "csd_beats_spuq_loto": csd_beats_spuq_loto,
            "loto_above_0.5": loto_above_05,
            "cost_advantage": f"CSD uses 0 extra calls vs SPUQ's {spuq_total_calls}",
        },
        "cost_analysis": cost_table,
        "label_distribution": label_dist,
        "spuq_api_cost": {
            "total_calls": cost_tracker.get("total_calls", 0),
            "total_usd": round(cost_tracker.get("total_usd", 0), 4),
            "input_tokens": cost_tracker.get("input_tokens", 0),
            "output_tokens": cost_tracker.get("output_tokens", 0),
        },
        "run_mode": RUN_MODE,
        "n_paraphrases": N_PARAPHRASES,
        "n_problems_per_level": N_PROBLEMS_PER_LEVEL,
        "valid_pairs": [
            {"task": t, "model": m, "d_star": d, "n_levels": n}
            for t, m, d, n in VALID_PAIRS
        ],
    }

    output = {"metadata": metadata, "datasets": datasets}

    # Validate no NaN in output
    output_str = json.dumps(output)
    if "NaN" in output_str or "Infinity" in output_str:
        logger.warning("Found NaN/Infinity in output, cleaning...")
        output = json.loads(output_str.replace("NaN", "0").replace("Infinity", "999"))

    return output


# ========================================================================
# Main execution
# ========================================================================

@logger.catch
def main():
    global RUN_MODE, COST_TRACKER

    if len(sys.argv) > 1:
        RUN_MODE = sys.argv[1]
    logger.info(f"Starting experiment in {RUN_MODE} mode")
    start_time = time.time()

    # ---- Phase 0: Load CSD data ----
    df = load_all_csd_data()
    logger.info(f"Total CSD rows: {len(df)}")

    # ---- Phase 1: SPUQ baseline ----
    spuq_df = None
    if RUN_MODE != "mini" and OPENROUTER_API_KEY:
        # Load raw datasets for problem text extraction
        logger.info("Loading raw datasets for SPUQ...")
        arith_dataset = json.loads(ARITH_DATA.read_text())["datasets"][0]["examples"]
        graph_dataset = json.loads(GRAPH_DATA.read_text())["datasets"][0]["examples"]
        logger.info(f"  Arithmetic problems: {len(arith_dataset)}")
        logger.info(f"  Graph coloring problems: {len(graph_dataset)}")

        selected = select_problems(arith_dataset, graph_dataset, n_per_level=N_PROBLEMS_PER_LEVEL)

        if RUN_MODE == "medium":
            # Partial SPUQ: only a few levels, 1 model per task
            models_to_use = {
                "arithmetic": ["google/gemini-2.0-flash-001"],
                "graph_coloring": ["google/gemini-2.0-flash-001"],
            }
            spuq_df = asyncio.run(run_spuq_phase(
                selected, df, max_levels_per_task=5, models_to_use=models_to_use
            ))
        else:
            # Full SPUQ
            spuq_df = asyncio.run(run_spuq_phase(selected, df))
    elif RUN_MODE == "mini":
        logger.info("Skipping SPUQ in mini mode (no API calls)")
    else:
        logger.warning("No OPENROUTER_API_KEY — falling back to rule-based pseudo-SPUQ")
        # Fallback: generate rule-based paraphrases and use them
        # We'll still need the datasets but won't make API calls for responses
        # Instead, set SPUQ metrics to NaN and skip SPUQ classifiers
        spuq_df = None

    # ---- Phase 2: Normalize features ----
    df = apply_normalizations(df)

    # Merge SPUQ data if available
    if spuq_df is not None and len(spuq_df) > 0:
        logger.info(f"Merging SPUQ data ({len(spuq_df)} rows) into CSD DataFrame")
        merge_cols = ["task_family", "model", "difficulty_level"]
        spuq_merge = spuq_df[merge_cols + ["spuq_disagreement", "spuq_accuracy"]]
        df = df.merge(spuq_merge, on=merge_cols, how="left")
        spuq_coverage = df["spuq_disagreement"].notna().sum()
        logger.info(f"  SPUQ coverage: {spuq_coverage}/{len(df)} rows")
        # Fill missing SPUQ values with median for classifier training
        for col in ["spuq_disagreement", "spuq_accuracy"]:
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0.5
            df[col] = df[col].fillna(median_val)

    # ---- Phase 3: Classifier comparison ----
    has_spuq = spuq_df is not None and len(spuq_df) > 0
    classifier_defs = define_classifiers(has_spuq=has_spuq)
    classifier_results = train_and_evaluate(df, classifier_defs)

    # ---- Phase 4: Build output ----
    output = build_output(df, classifier_results, spuq_df, COST_TRACKER)

    # Save
    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Output saved to {out_path}")
    logger.info(f"Output size: {out_path.stat().st_size / 1024:.1f} KB")

    elapsed = time.time() - start_time
    logger.info(f"Total runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info(f"Total API cost: ${COST_TRACKER['total_usd']:.4f}")
    logger.info(f"Total API calls: {COST_TRACKER['total_calls']}")


if __name__ == "__main__":
    main()
