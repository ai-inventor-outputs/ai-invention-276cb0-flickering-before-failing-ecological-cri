#!/usr/bin/env python3
"""Syllogistic Logic CSD Sampling Experiment.

Runs the CSD (Critical Slowing Down) sampling experiment on a 280-problem
syllogistic deductive logic dataset. For each of 3 LLMs (small/medium/large),
generates N=50 responses at each of 14 difficulty levels (premise count 2-15),
evaluates TRUE/FALSE accuracy, computes sentence embeddings, and derives CSD
indicators plus baselines. Follows gradual scaling: mini -> medium -> full.
"""

import asyncio
import gc
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

import numpy as np
from loguru import logger

# ── LOGGING SETUP ────────────────────────────────────────────────────────────

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ── CONSTANTS ────────────────────────────────────────────────────────────────

WORKSPACE = Path(__file__).parent
DATA_PATH = WORKSPACE / "deps" / "data_id3_it1__opus" / "full_data_out.json"
ABILITY_SERVER_URL = os.environ.get("ABILITY_SERVICE_URL", "http://localhost:9100")

MODELS = [
    {
        "id": "mistralai/ministral-3b-2512",
        "tier": "small",
        "price_in": 0.10,   # per 1M tokens
        "price_out": 0.10,
    },
    {
        "id": "mistralai/ministral-8b-2512",
        "tier": "medium",
        "price_in": 0.15,
        "price_out": 0.15,
    },
    {
        "id": "deepseek/deepseek-v3.2",
        "tier": "large",
        "price_in": 0.26,
        "price_out": 0.38,
    },
]

TEMPERATURE = 0.8
N_PROBLEMS_PER_LEVEL = 5
N_RESPONSES_PER_PROBLEM = 10
DIFFICULTY_LEVELS = list(range(2, 16))  # d=2 through d=15, 14 levels
MAX_CONCURRENT = 10
BUDGET_LIMIT = 9.0  # Hard stop before $10
MAX_TOKENS_PER_CALL = 1024

SYSTEM_PROMPT = (
    "You are a logical reasoning assistant. Think step by step through "
    "the premises, then give your final answer as exactly TRUE or FALSE."
)


# ── HARDWARE DETECTION & RESOURCE LIMITS ─────────────────────────────────────

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
    for p in ["/sys/fs/cgroup/memory.max",
              "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None


NUM_CPUS = _detect_cpus()
TOTAL_RAM_GB = _container_ram_gb() or 16.0

# Set memory limit to 80% of available (leave room for OS + agent)
RAM_BUDGET_BYTES = int(TOTAL_RAM_GB * 0.80 * 1e9)
try:
    resource.setrlimit(resource.RLIMIT_AS,
                       (RAM_BUDGET_BYTES * 3, RAM_BUDGET_BYTES * 3))
except (ValueError, OSError):
    logger.warning("Could not set RLIMIT_AS")

# CPU time limit: 50 minutes
try:
    resource.setrlimit(resource.RLIMIT_CPU, (3000, 3000))
except (ValueError, OSError):
    pass

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, "
            f"budget {RAM_BUDGET_BYTES / 1e9:.1f} GB")


# ── DATA LOADING ─────────────────────────────────────────────────────────────

def load_dataset(data_path: Path) -> dict[int, list[dict]]:
    """Load full_data_out.json and group examples by metadata_difficulty."""
    logger.info(f"Loading dataset from {data_path}")
    data = json.loads(data_path.read_text())

    by_difficulty: dict[int, list[dict]] = defaultdict(list)
    for ex in data["datasets"][0]["examples"]:
        d = ex["metadata_difficulty"]
        by_difficulty[d].append(ex)

    for d in DIFFICULTY_LEVELS:
        count = len(by_difficulty[d])
        logger.debug(f"  d={d}: {count} problems")
        if count < N_PROBLEMS_PER_LEVEL:
            logger.warning(f"Only {count} problems at d={d}, need {N_PROBLEMS_PER_LEVEL}")

    total = sum(len(v) for v in by_difficulty.values())
    logger.info(f"Loaded {total} problems across {len(by_difficulty)} difficulty levels")
    return dict(by_difficulty)


def select_problems(by_difficulty: dict[int, list[dict]],
                    n_per_level: int = N_PROBLEMS_PER_LEVEL,
                    seed: int = 42) -> dict[int, list[dict]]:
    """Select n_per_level problems per difficulty with TRUE/FALSE balance.

    Target: 3 TRUE + 2 FALSE (or 2+3 alternating by parity of d).
    """
    rng = np.random.default_rng(seed)
    selected: dict[int, list[dict]] = {}

    for d in DIFFICULTY_LEVELS:
        problems = by_difficulty.get(d, [])
        if not problems:
            logger.warning(f"No problems at d={d}, skipping")
            continue

        true_probs = [p for p in problems if p["output"] == "TRUE"]
        false_probs = [p for p in problems if p["output"] == "FALSE"]

        n_true = min(3 if d % 2 == 0 else 2, len(true_probs))
        n_false = min(n_per_level - n_true, len(false_probs))

        # Adjust if not enough of one type
        if n_true + n_false < n_per_level:
            extra_needed = n_per_level - (n_true + n_false)
            if len(true_probs) > n_true:
                n_true = min(n_true + extra_needed, len(true_probs))
            elif len(false_probs) > n_false:
                n_false = min(n_false + extra_needed, len(false_probs))

        chosen_true_idx = rng.choice(len(true_probs), n_true, replace=False)
        chosen_false_idx = rng.choice(len(false_probs), n_false, replace=False)
        selected[d] = (
            [true_probs[i] for i in chosen_true_idx]
            + [false_probs[i] for i in chosen_false_idx]
        )

    logger.info(f"Selected {sum(len(v) for v in selected.values())} problems "
                f"across {len(selected)} levels")
    return selected


# ── API CALLING (ASYNC) ──────────────────────────────────────────────────────

class CostTracker:
    """Thread-safe cost tracker for API calls."""

    def __init__(self, budget_limit: float = BUDGET_LIMIT):
        self.total = 0.0
        self.by_model: dict[str, float] = defaultdict(float)
        self.call_count = 0
        self.budget_limit = budget_limit
        self._lock = asyncio.Lock()

    async def add(self, model: str, input_tokens: int, output_tokens: int,
                  price_in: float, price_out: float) -> None:
        async with self._lock:
            cost = (input_tokens / 1_000_000) * price_in + \
                   (output_tokens / 1_000_000) * price_out
            self.total += cost
            self.by_model[model] += cost
            self.call_count += 1

    async def check_budget(self) -> bool:
        async with self._lock:
            return self.total < self.budget_limit

    def summary(self) -> str:
        lines = [f"Total cost: ${self.total:.4f} ({self.call_count} calls)"]
        for m, c in self.by_model.items():
            lines.append(f"  {m}: ${c:.4f}")
        return "\n".join(lines)


async def call_openrouter_async(
    client: "httpx.AsyncClient",
    model_id: str,
    prompt: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    cost_tracker: CostTracker,
    model_prices: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    """Call OpenRouter via the ability server. Returns response dict."""
    if not await cost_tracker.check_budget():
        raise RuntimeError(f"Budget exceeded: ${cost_tracker.total:.2f}")

    async with semaphore:
        request_payload = {
            "model": model_id,
            "input_text": prompt,
            "instructions": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        for attempt in range(4):
            try:
                resp = await client.post(
                    f"{ABILITY_SERVER_URL}/aii_openrouter__call",
                    json=request_payload,
                    timeout=120.0,
                )
                resp.raise_for_status()
                result = resp.json()

                if not result.get("success"):
                    error_msg = result.get("error", "Unknown error")
                    if attempt < 3 and ("rate" in error_msg.lower()
                                        or "429" in error_msg
                                        or "503" in error_msg
                                        or "502" in error_msg):
                        wait = 2 ** (attempt + 1)
                        logger.warning(f"Transient error ({error_msg}), "
                                       f"retry {attempt + 1}/3 in {wait}s")
                        await asyncio.sleep(wait)
                        continue
                    raise RuntimeError(f"API error: {error_msg}")

                input_tokens = result.get("input_tokens", 0)
                output_tokens = result.get("output_tokens", 0)
                price_in, price_out = model_prices.get(model_id, (0.0, 0.0))
                await cost_tracker.add(model_id, input_tokens, output_tokens,
                                       price_in, price_out)

                return {
                    "response_text": result.get("response", ""),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                }

            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if attempt < 3 and status in (429, 502, 503, 504):
                    wait = 2 ** (attempt + 1)
                    logger.warning(f"HTTP {status}, retry {attempt + 1}/3 in {wait}s")
                    await asyncio.sleep(wait)
                    continue
                raise
            except (httpx.ConnectError, httpx.ReadTimeout) as e:
                if attempt < 3:
                    wait = 2 ** (attempt + 1)
                    logger.warning(f"Connection error ({e}), retry {attempt + 1}/3 in {wait}s")
                    await asyncio.sleep(wait)
                    continue
                raise

    raise RuntimeError("All retries exhausted")


async def generate_responses(
    selected_problems: dict[int, list[dict]],
    models: list[dict],
    *,
    n_problems: int | None = None,
    n_responses: int = N_RESPONSES_PER_PROBLEM,
    levels: list[int] | None = None,
) -> list[dict]:
    """Generate LLM responses for selected problems.

    Args:
        selected_problems: dict d -> list of problem dicts
        models: list of model config dicts
        n_problems: limit problems per level (None = all)
        n_responses: responses per problem
        levels: difficulty levels to run (None = all)

    Returns:
        list of result dicts with all metadata
    """
    import httpx

    if levels is None:
        levels = sorted(selected_problems.keys())

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    cost_tracker = CostTracker()
    model_prices = {m["id"]: (m["price_in"], m["price_out"]) for m in models}

    tasks = []
    task_meta = []

    async with httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=MAX_CONCURRENT + 5,
            max_keepalive_connections=MAX_CONCURRENT,
        ),
        timeout=httpx.Timeout(120.0, connect=30.0),
    ) as client:

        for model in models:
            model_id = model["id"]
            for d in levels:
                problems = selected_problems.get(d, [])
                n_probs = min(n_problems, len(problems)) if n_problems else len(problems)

                for prob_idx in range(n_probs):
                    problem = problems[prob_idx]
                    for sample_idx in range(n_responses):
                        meta = {
                            "model": model_id,
                            "tier": model["tier"],
                            "difficulty": d,
                            "problem_idx": prob_idx,
                            "sample_idx": sample_idx,
                            "input_text": problem["input"],
                            "ground_truth": problem["output"],
                            "template": problem.get("metadata_template", ""),
                            "quantifier_pattern": problem.get(
                                "metadata_quantifier_pattern", ""),
                            "problem_id": problem.get("metadata_problem_id", -1),
                        }
                        task = call_openrouter_async(
                            client, model_id, problem["input"],
                            SYSTEM_PROMPT, TEMPERATURE, MAX_TOKENS_PER_CALL,
                            semaphore, cost_tracker, model_prices,
                        )
                        tasks.append(task)
                        task_meta.append(meta)

        total_calls = len(tasks)
        logger.info(f"Launching {total_calls} API calls across "
                    f"{len(models)} models, {len(levels)} levels")

        # Run all tasks with gather
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Merge results with metadata
    results = []
    errors = 0
    for meta, raw in zip(task_meta, raw_results):
        if isinstance(raw, Exception):
            errors += 1
            logger.debug(f"Error for {meta['model']} d={meta['difficulty']} "
                         f"p={meta['problem_idx']} s={meta['sample_idx']}: "
                         f"{str(raw)[:100]}")
            continue
        result = {**meta, **raw}
        results.append(result)

    logger.info(f"Completed {len(results)}/{total_calls} calls "
                f"({errors} errors)")
    logger.info(cost_tracker.summary())
    return results


# ── ANSWER EXTRACTION ────────────────────────────────────────────────────────

def extract_answer(response_text: str) -> str | None:
    """Extract TRUE or FALSE from model response.

    Strategy: look for answer declarations first, then last standalone match.
    """
    if not response_text:
        return None

    # Priority 1: explicit answer declarations
    answer_patterns = [
        r'(?:final\s+)?answer\s*(?:is|:)\s*(TRUE|FALSE)',
        r'\*\*(?:answer|final answer)\s*(?:is|:)?\s*(TRUE|FALSE)\*\*',
        r'(?:therefore|thus|hence|so|conclusion)\s*(?:,\s*)?(?:the\s+(?:answer|statement)\s+is\s+)?(TRUE|FALSE)',
        r'(?:it\s+is\s+)(TRUE|FALSE)',
    ]
    for pat in answer_patterns:
        matches = re.findall(pat, response_text, re.IGNORECASE)
        if matches:
            return matches[-1].upper()

    # Priority 2: last standalone TRUE/FALSE
    matches = re.findall(r'\b(TRUE|FALSE)\b', response_text, re.IGNORECASE)
    if matches:
        return matches[-1].upper()

    # Priority 3: yes/no mapping
    yes_no = re.findall(r'\b(yes|no)\b', response_text, re.IGNORECASE)
    if yes_no:
        last = yes_no[-1].lower()
        return "TRUE" if last == "yes" else "FALSE"

    return None


# ── SEMANTIC EMBEDDING ───────────────────────────────────────────────────────

def embed_responses(results: list[dict]) -> np.ndarray:
    """Embed all response texts using all-MiniLM-L6-v2.

    Returns (N, 384) array of embeddings.
    """
    from sentence_transformers import SentenceTransformer

    logger.info(f"Embedding {len(results)} responses...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = []
    for r in results:
        text = r.get("response_text", "")
        if not text or text == "No output generated":
            text = "empty response"
        texts.append(text)

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    logger.info(f"Embeddings shape: {embeddings.shape}")

    # Free model memory
    del model
    gc.collect()

    return embeddings


# ── CSD INDICATOR COMPUTATION ────────────────────────────────────────────────

def compute_csd_indicators(
    embeddings_subset: np.ndarray,
    answers_subset: list[str | None],
    ground_truths: list[str],
) -> dict[str, Any]:
    """Compute full CSD indicator battery for responses at one difficulty level.

    Args:
        embeddings_subset: (N, 384) embeddings for this difficulty x model
        answers_subset: list of N extracted answers (TRUE/FALSE/None)
        ground_truths: list of N ground truth answers

    Returns:
        dict with all CSD indicators
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.metrics.pairwise import cosine_distances
    from scipy.stats import skew, kurtosis

    N = len(embeddings_subset)
    if N < 2:
        return _empty_csd_indicators(N)

    # --- A. Embedding variance (trace of covariance matrix) ---
    try:
        cov_matrix = np.cov(embeddings_subset.T)
        embedding_variance = float(np.trace(cov_matrix))
    except Exception:
        embedding_variance = 0.0

    # --- B. Mean pairwise cosine distance ---
    try:
        dist_matrix = cosine_distances(embeddings_subset)
        triu_idx = np.triu_indices(N, k=1)
        mean_cosine_distance = float(dist_matrix[triu_idx].mean())
    except Exception:
        mean_cosine_distance = 0.0

    # --- C. PCA projection to PC1 for 1D tests ---
    n_components = min(10, N - 1)
    try:
        pca = PCA(n_components=max(n_components, 1))
        pca_embeddings = pca.fit_transform(embeddings_subset)
        pc1 = pca_embeddings[:, 0]
        pc1_variance_explained = float(pca.explained_variance_ratio_[0])
    except Exception:
        pc1 = np.zeros(N)
        pc1_variance_explained = 0.0

    # --- D. Hartigan's Dip Test on PC1 ---
    try:
        import diptest
        dip_stat, dip_pvalue = diptest.diptest(pc1)
        dip_stat = float(dip_stat)
        dip_pvalue = float(dip_pvalue)
    except Exception:
        dip_stat = 0.0
        dip_pvalue = 1.0

    # --- E. Silhouette score (k=2 k-means on full embeddings) ---
    try:
        if N >= 4:
            km = KMeans(n_clusters=2, n_init=10, random_state=42)
            labels = km.fit_predict(embeddings_subset)
            if len(set(labels)) == 2:
                sil_score = float(silhouette_score(embeddings_subset, labels))
            else:
                sil_score = 0.0
        else:
            sil_score = 0.0
    except Exception:
        sil_score = 0.0

    # --- F. Bimodality Coefficient on PC1 ---
    try:
        s = float(skew(pc1))
        k = float(kurtosis(pc1))  # excess kurtosis
        n = len(pc1)
        bc_numerator = s ** 2 + 1
        denom_correction = 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)) if n > 3 else 3.0
        bc_denominator = k + denom_correction
        bimodality_coefficient = bc_numerator / bc_denominator if bc_denominator != 0 else 0.0
    except Exception:
        bimodality_coefficient = 0.0

    # --- G. Self-consistency disagreement (answer-level) ---
    valid_answers = [a for a in answers_subset if a in ("TRUE", "FALSE")]
    if valid_answers:
        counter = Counter(valid_answers)
        majority_fraction = counter.most_common(1)[0][1] / len(valid_answers)
        disagreement_rate = 1.0 - majority_fraction
        majority_answer = counter.most_common(1)[0][0]
    else:
        disagreement_rate = 1.0
        majority_answer = None

    # --- H. Answer-level bimodality ---
    n_true = sum(1 for a in valid_answers if a == "TRUE")
    n_false = sum(1 for a in valid_answers if a == "FALSE")
    answer_balance = min(n_true, n_false) / max(n_true + n_false, 1)

    # --- I. Accuracy ---
    correct = sum(1 for a, g in zip(answers_subset, ground_truths) if a == g)
    accuracy = correct / N if N > 0 else 0.0

    return {
        "embedding_variance": embedding_variance,
        "mean_cosine_distance": mean_cosine_distance,
        "pc1_variance_explained": pc1_variance_explained,
        "dip_statistic": dip_stat,
        "dip_pvalue": dip_pvalue,
        "silhouette_k2": sil_score,
        "bimodality_coefficient": bimodality_coefficient,
        "disagreement_rate": disagreement_rate,
        "answer_balance": answer_balance,
        "majority_answer": majority_answer,
        "accuracy": accuracy,
        "n_responses": N,
        "n_valid_answers": len(valid_answers),
        "n_extraction_failures": N - len(valid_answers),
    }


def _empty_csd_indicators(n: int = 0) -> dict[str, Any]:
    """Return empty CSD indicators for edge cases."""
    return {
        "embedding_variance": 0.0,
        "mean_cosine_distance": 0.0,
        "pc1_variance_explained": 0.0,
        "dip_statistic": 0.0,
        "dip_pvalue": 1.0,
        "silhouette_k2": 0.0,
        "bimodality_coefficient": 0.0,
        "disagreement_rate": 1.0,
        "answer_balance": 0.0,
        "majority_answer": None,
        "accuracy": 0.0,
        "n_responses": n,
        "n_valid_answers": 0,
        "n_extraction_failures": n,
    }


# ── WITHIN-CHAIN CONFIDENCE AUTOCORRELATION ──────────────────────────────────

def compute_chain_autocorrelation(response_text: str) -> float | None:
    """Parse step-by-step reasoning into steps, extract verbalized confidence,
    compute lag-1 autocorrelation of step 'certainty'.

    Heuristic: count hedging vs assertive words per step.
    """
    if not response_text:
        return None

    # Split into steps by numbered lines or "Step" markers
    step_pattern = r'(?:(?:^|\n)\s*(?:\d+[\.)\:]|Step\s+\d+|[-*]\s))'
    steps = re.split(step_pattern, response_text)
    steps = [s.strip() for s in steps if len(s.strip()) > 10]

    if len(steps) < 3:
        return None

    hedging = {"maybe", "perhaps", "might", "possibly", "uncertain",
               "could", "seems", "unclear", "assume", "suppose",
               "not sure", "likely", "probably"}
    assertive = {"therefore", "clearly", "must", "definitely", "certainly",
                 "thus", "hence", "conclude", "follows", "so",
                 "proven", "established", "confirmed", "means"}

    certainty = []
    for step in steps:
        words = step.lower().split()
        h = sum(1 for w in words if w in hedging)
        a = sum(1 for w in words if w in assertive)
        score = (a - h) / max(len(words), 1)
        certainty.append(score)

    c = np.array(certainty)
    if c.std() < 1e-8:
        return 0.0

    c_centered = c - c.mean()
    autocorr = float(np.correlate(c_centered[:-1], c_centered[1:])[0])
    autocorr /= (np.sum(c_centered ** 2) + 1e-10)
    return float(autocorr)


def compute_avg_autocorrelation(results_subset: list[dict]) -> float:
    """Average within-chain confidence autocorrelation across responses."""
    autocorrs = []
    for r in results_subset:
        ac = compute_chain_autocorrelation(r.get("response_text", ""))
        if ac is not None:
            autocorrs.append(ac)
    return float(np.mean(autocorrs)) if autocorrs else 0.0


# ── AGGREGATE CSD ANALYSIS ──────────────────────────────────────────────────

def find_critical_difficulty(accuracy_by_level: dict[int, float]) -> int | None:
    """Find d* = first difficulty where accuracy drops below 0.5."""
    for d in sorted(accuracy_by_level.keys()):
        if accuracy_by_level[d] < 0.5:
            return d
    return None


def fit_variance_scaling(
    difficulties: list[int],
    variances: list[float],
    d_star: int,
) -> dict[str, Any]:
    """Fit log(Var) ~ alpha * log(d* - d) via OLS.

    Only use levels where d < d_star. Tests if alpha in [-0.7, -0.3].
    """
    valid = [(d, v) for d, v in zip(difficulties, variances)
             if d < d_star and v > 0]
    if len(valid) < 3:
        return {"alpha": None, "r_squared": None, "consistent_with_fold": None,
                "theoretical_prediction": -0.5}

    ds, vs = zip(*valid)
    log_dist = np.log(np.array([d_star - d for d in ds], dtype=float))
    log_var = np.log(np.array(vs, dtype=float))

    A = np.vstack([log_dist, np.ones(len(log_dist))]).T
    result = np.linalg.lstsq(A, log_var, rcond=None)
    alpha = float(result[0][0])

    ss_res = float(np.sum((log_var - A @ result[0]) ** 2))
    ss_tot = float(np.sum((log_var - log_var.mean()) ** 2))
    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "alpha": alpha,
        "r_squared": r_sq,
        "consistent_with_fold": -0.7 <= alpha <= -0.3,
        "theoretical_prediction": -0.5,
    }


def test_leading_indicator(
    accuracy_by_level: dict[int, float],
    indicator_by_level: dict[int, float],
    d_star: int,
    accuracy_threshold: float = 0.8,
) -> dict[str, Any]:
    """Test whether CSD indicator becomes significant at d where accuracy > threshold.

    Uses Kendall tau trend test on indicator values.
    """
    from scipy.stats import kendalltau

    # Find levels where accuracy > threshold and d < d*
    pre_boundary = {d: v for d, v in indicator_by_level.items()
                    if d < d_star and accuracy_by_level.get(d, 0) > accuracy_threshold}

    if len(pre_boundary) < 3:
        return {"is_leading": None, "lead_distance": 0, "kendall_tau": None,
                "p_value": None, "levels_tested": len(pre_boundary)}

    sorted_d = sorted(pre_boundary.keys())
    tau, p_value = kendalltau(sorted_d, [pre_boundary[d] for d in sorted_d])

    return {
        "kendall_tau": float(tau),
        "p_value": float(p_value),
        "is_leading": p_value < 0.05 and tau > 0,
        "lead_distance": d_star - max(sorted_d),
        "levels_tested": len(sorted_d),
        "accuracy_at_last_preboundary": accuracy_by_level.get(max(sorted_d), 0.0),
    }


# ── OUTPUT ASSEMBLY ──────────────────────────────────────────────────────────

def build_output(
    all_results: list[dict],
    csd_by_model: dict[str, dict[int, dict]],
    analysis: dict[str, dict],
) -> dict:
    """Build exp_gen_sol_out.json format output.

    Schema: predict_* fields must be strings, metadata_* can be any type.
    """
    examples = []
    for r in all_results:
        model = r["model"]
        d = r["difficulty"]
        csd = csd_by_model.get(model, {}).get(d, _empty_csd_indicators())
        model_analysis = analysis.get(model, {})

        example = {
            "input": r["input_text"],
            "output": r["ground_truth"],
            # predict_* must be strings per schema
            "predict_response": r.get("response_text", ""),
            "predict_extracted_answer": r.get("extracted_answer", "") or "",
            "predict_correct": str(
                r.get("extracted_answer") == r["ground_truth"]
            ).lower(),
            # metadata fields
            "metadata_difficulty": d,
            "metadata_model": model,
            "metadata_tier": r.get("tier", ""),
            "metadata_sample_idx": r["sample_idx"],
            "metadata_problem_idx": r["problem_idx"],
            "metadata_problem_id": r.get("problem_id", -1),
            "metadata_template": r.get("template", ""),
            "metadata_quantifier_pattern": r.get("quantifier_pattern", ""),
            # CSD indicators (per model x difficulty level)
            "metadata_csd_embedding_variance": csd.get("embedding_variance", 0.0),
            "metadata_csd_mean_cosine_distance": csd.get("mean_cosine_distance", 0.0),
            "metadata_csd_dip_statistic": csd.get("dip_statistic", 0.0),
            "metadata_csd_dip_pvalue": csd.get("dip_pvalue", 1.0),
            "metadata_csd_silhouette_k2": csd.get("silhouette_k2", 0.0),
            "metadata_csd_bimodality_coefficient": csd.get(
                "bimodality_coefficient", 0.0),
            "metadata_csd_disagreement_rate": csd.get("disagreement_rate", 0.0),
            "metadata_csd_answer_balance": csd.get("answer_balance", 0.0),
            "metadata_csd_accuracy": csd.get("accuracy", 0.0),
            "metadata_csd_avg_chain_autocorrelation": csd.get(
                "avg_chain_autocorrelation", 0.0),
            # Per-model analysis
            "metadata_analysis_d_star": model_analysis.get("d_star"),
            "metadata_analysis_variance_scaling_alpha": (
                model_analysis.get("scaling", {}).get("alpha")
            ),
            "metadata_analysis_variance_scaling_r2": (
                model_analysis.get("scaling", {}).get("r_squared")
            ),
            "metadata_analysis_leading_dip": (
                model_analysis.get("leading_dip", {}).get("is_leading")
            ),
            "metadata_analysis_leading_variance": (
                model_analysis.get("leading_var", {}).get("is_leading")
            ),
        }
        examples.append(example)

    return {
        "metadata": {
            "method_name": "CSD_sampling_syllogistic_logic",
            "description": (
                "Critical Slowing Down indicators computed from LLM response "
                "distributions across difficulty levels of syllogistic logic tasks"
            ),
            "models": [m["id"] for m in MODELS],
            "n_responses_per_level": N_RESPONSES_PER_PROBLEM * N_PROBLEMS_PER_LEVEL,
            "temperature": TEMPERATURE,
            "difficulty_range": [min(DIFFICULTY_LEVELS), max(DIFFICULTY_LEVELS)],
        },
        "datasets": [{
            "dataset": "syllogistic_logic_csd",
            "examples": examples,
        }],
    }


# ── CSD COMPUTATION PIPELINE ────────────────────────────────────────────────

def run_csd_pipeline(
    all_results: list[dict],
    embeddings: np.ndarray,
    models: list[dict],
    levels: list[int],
) -> tuple[dict, dict]:
    """Run full CSD indicator computation and analysis.

    Returns:
        csd_by_model: dict[model_id, dict[difficulty, csd_dict]]
        analysis: dict[model_id, analysis_dict]
    """
    # Extract answers
    for r in all_results:
        if "extracted_answer" not in r:
            r["extracted_answer"] = extract_answer(r.get("response_text", ""))

    extraction_success = sum(1 for r in all_results
                             if r.get("extracted_answer") in ("TRUE", "FALSE"))
    logger.info(f"Answer extraction: {extraction_success}/{len(all_results)} "
                f"({100 * extraction_success / max(len(all_results), 1):.1f}%)")

    # Compute CSD indicators per model x difficulty
    csd_by_model: dict[str, dict[int, dict]] = {}
    for model in models:
        model_id = model["id"]
        csd_by_model[model_id] = {}

        for d in levels:
            indices = [i for i, r in enumerate(all_results)
                       if r["model"] == model_id and r["difficulty"] == d]

            if not indices:
                csd_by_model[model_id][d] = _empty_csd_indicators()
                continue

            emb_subset = embeddings[indices]
            ans_subset = [all_results[i].get("extracted_answer") for i in indices]
            gt_subset = [all_results[i]["ground_truth"] for i in indices]

            csd = compute_csd_indicators(emb_subset, ans_subset, gt_subset)

            # Add chain autocorrelation
            results_subset = [all_results[i] for i in indices]
            csd["avg_chain_autocorrelation"] = compute_avg_autocorrelation(
                results_subset)

            csd_by_model[model_id][d] = csd

            logger.debug(
                f"  {model_id} d={d}: acc={csd['accuracy']:.2f} "
                f"var={csd['embedding_variance']:.4f} "
                f"dip={csd['dip_statistic']:.4f} "
                f"dis={csd['disagreement_rate']:.2f}"
            )

    # Per-model aggregate analysis
    analysis: dict[str, dict] = {}
    for model in models:
        model_id = model["id"]
        acc_by_level = {d: csd_by_model[model_id][d]["accuracy"]
                        for d in levels if d in csd_by_model[model_id]}
        var_by_level = {d: csd_by_model[model_id][d]["embedding_variance"]
                        for d in levels if d in csd_by_model[model_id]}
        dip_by_level = {d: csd_by_model[model_id][d]["dip_statistic"]
                        for d in levels if d in csd_by_model[model_id]}
        sil_by_level = {d: csd_by_model[model_id][d]["silhouette_k2"]
                        for d in levels if d in csd_by_model[model_id]}
        dis_by_level = {d: csd_by_model[model_id][d]["disagreement_rate"]
                        for d in levels if d in csd_by_model[model_id]}

        d_star = find_critical_difficulty(acc_by_level)
        model_analysis: dict[str, Any] = {"d_star": d_star}

        logger.info(f"Model {model_id} ({model['tier']}): "
                    f"d*={d_star}, "
                    f"acc range: {min(acc_by_level.values()):.2f}-"
                    f"{max(acc_by_level.values()):.2f}")

        if d_star is not None:
            model_analysis["scaling"] = fit_variance_scaling(
                list(var_by_level.keys()),
                list(var_by_level.values()),
                d_star,
            )
            for name, ind in [
                ("leading_dip", dip_by_level),
                ("leading_var", var_by_level),
                ("leading_sil", sil_by_level),
                ("leading_dis", dis_by_level),
            ]:
                model_analysis[name] = test_leading_indicator(
                    acc_by_level, ind, d_star)
        else:
            logger.info(f"  No capability boundary found for {model_id}")
            model_analysis["scaling"] = {
                "alpha": None, "r_squared": None,
                "consistent_with_fold": None,
                "theoretical_prediction": -0.5,
            }
            for name in ["leading_dip", "leading_var",
                         "leading_sil", "leading_dis"]:
                model_analysis[name] = {
                    "is_leading": None, "lead_distance": 0,
                    "kendall_tau": None, "p_value": None,
                }

        analysis[model_id] = model_analysis

    return csd_by_model, analysis


# ── VALIDATION HELPERS ───────────────────────────────────────────────────────

def validate_mini_results(results: list[dict]) -> bool:
    """Validate mini run results before scaling up."""
    if not results:
        logger.error("MINI RUN: No results returned!")
        return False

    # Check answer extraction
    extracted = [r for r in results if r.get("extracted_answer") in ("TRUE", "FALSE")]
    rate = len(extracted) / len(results)
    logger.info(f"MINI RUN: extraction rate = {rate:.0%} ({len(extracted)}/{len(results)})")
    if rate < 0.5:
        logger.warning("MINI RUN: extraction rate very low, check model outputs")

    # Check responses are non-empty
    non_empty = sum(1 for r in results
                    if r.get("response_text", "")
                    and r["response_text"] != "No output generated")
    logger.info(f"MINI RUN: {non_empty}/{len(results)} non-empty responses")
    if non_empty == 0:
        logger.error("MINI RUN: All responses are empty!")
        return False

    return True


# ── MAIN ORCHESTRATION ───────────────────────────────────────────────────────

@logger.catch
def main():
    """Main experiment orchestration with gradual scaling."""
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("SYLLOGISTIC LOGIC CSD EXPERIMENT")
    logger.info("=" * 60)

    # 1. Load dataset
    by_difficulty = load_dataset(DATA_PATH)
    selected = select_problems(by_difficulty)

    # Log problem counts
    for d in sorted(selected.keys()):
        true_count = sum(1 for p in selected[d] if p["output"] == "TRUE")
        false_count = len(selected[d]) - true_count
        logger.debug(f"  d={d}: {len(selected[d])} problems "
                     f"({true_count}T/{false_count}F)")

    # ──────────────────────────────────────────────────────────────────────
    # PHASE A: MINI RUN — 1 model, 2 levels, 2 problems, 3 responses = 12
    # ──────────────────────────────────────────────────────────────────────
    logger.info("=" * 40)
    logger.info("=== PHASE A: MINI RUN (12 calls) ===")
    logger.info("=" * 40)

    mini_start = time.time()
    mini_results = asyncio.run(generate_responses(
        selected, [MODELS[0]],
        n_problems=2, n_responses=3, levels=[2, 8],
    ))
    mini_elapsed = time.time() - mini_start
    logger.info(f"Mini run: {len(mini_results)} results in {mini_elapsed:.1f}s")

    # Extract answers and validate
    for r in mini_results:
        r["extracted_answer"] = extract_answer(r.get("response_text", ""))

    if not validate_mini_results(mini_results):
        logger.error("Mini run validation failed! Attempting with backup model...")
        # Try with deepseek as backup
        mini_results = asyncio.run(generate_responses(
            selected, [MODELS[2]],
            n_problems=2, n_responses=3, levels=[2, 8],
        ))
        for r in mini_results:
            r["extracted_answer"] = extract_answer(r.get("response_text", ""))
        if not validate_mini_results(mini_results):
            logger.error("Backup model also failed. Aborting.")
            sys.exit(1)

    # Test embedding on mini results
    mini_embeddings = embed_responses(mini_results)
    assert mini_embeddings.shape[1] == 384, f"Bad embedding dim: {mini_embeddings.shape}"
    assert not np.any(np.isnan(mini_embeddings)), "NaN in embeddings!"
    logger.info("Mini run: embeddings OK")

    # Test CSD computation
    if len(mini_results) >= 3:
        test_csd = compute_csd_indicators(
            mini_embeddings[:3],
            [r.get("extracted_answer") for r in mini_results[:3]],
            [r["ground_truth"] for r in mini_results[:3]],
        )
        logger.info(f"Mini CSD test: variance={test_csd['embedding_variance']:.4f}, "
                    f"accuracy={test_csd['accuracy']:.2f}")
    logger.info("Phase A PASSED")

    # ──────────────────────────────────────────────────────────────────────
    # PHASE B: MEDIUM RUN — 1 model, 5 levels, 3 problems, 5 responses = 75
    # ──────────────────────────────────────────────────────────────────────
    logger.info("=" * 40)
    logger.info("=== PHASE B: MEDIUM RUN (75 calls) ===")
    logger.info("=" * 40)

    medium_start = time.time()
    medium_results = asyncio.run(generate_responses(
        selected, [MODELS[0]],
        n_problems=3, n_responses=5, levels=[2, 4, 6, 8, 10],
    ))
    medium_elapsed = time.time() - medium_start
    logger.info(f"Medium run: {len(medium_results)} results in {medium_elapsed:.1f}s")

    for r in medium_results:
        r["extracted_answer"] = extract_answer(r.get("response_text", ""))

    medium_embeddings = embed_responses(medium_results)

    # Compute CSD to check trends
    medium_csd, _ = run_csd_pipeline(
        medium_results, medium_embeddings,
        [MODELS[0]], [2, 4, 6, 8, 10],
    )

    model0_id = MODELS[0]["id"]
    for d in [2, 4, 6, 8, 10]:
        csd = medium_csd.get(model0_id, {}).get(d, {})
        logger.info(f"  d={d}: acc={csd.get('accuracy', 0):.2f} "
                    f"var={csd.get('embedding_variance', 0):.4f} "
                    f"dip={csd.get('dip_statistic', 0):.4f}")

    # Extrapolate timing
    calls_per_sec = len(medium_results) / max(medium_elapsed, 1)
    total_full_calls = (len(MODELS) * len(DIFFICULTY_LEVELS)
                        * N_PROBLEMS_PER_LEVEL * N_RESPONSES_PER_PROBLEM)
    estimated_full_time = total_full_calls / max(calls_per_sec, 0.1)
    logger.info(f"Rate: {calls_per_sec:.1f} calls/s, "
                f"full run estimate: {estimated_full_time:.0f}s "
                f"({estimated_full_time / 60:.1f}min)")

    elapsed_so_far = time.time() - start_time
    remaining_budget = 50 * 60 - elapsed_so_far  # 50 min budget
    if estimated_full_time > remaining_budget * 0.8:
        logger.warning(f"Full run may exceed time budget! "
                       f"Estimated: {estimated_full_time:.0f}s, "
                       f"remaining: {remaining_budget:.0f}s")

    logger.info("Phase B PASSED")
    del medium_results, medium_embeddings, medium_csd
    gc.collect()

    # ──────────────────────────────────────────────────────────────────────
    # PHASE C: FULL RUN — 3 models x 14 levels x 5 problems x 10 responses
    # ──────────────────────────────────────────────────────────────────────
    logger.info("=" * 40)
    logger.info(f"=== PHASE C: FULL RUN ({total_full_calls} calls) ===")
    logger.info("=" * 40)

    full_start = time.time()
    all_results = asyncio.run(generate_responses(
        selected, MODELS,
        n_problems=N_PROBLEMS_PER_LEVEL,
        n_responses=N_RESPONSES_PER_PROBLEM,
        levels=DIFFICULTY_LEVELS,
    ))
    full_elapsed = time.time() - full_start
    logger.info(f"Full run: {len(all_results)} results in {full_elapsed:.1f}s "
                f"({full_elapsed / 60:.1f}min)")

    # 6. Extract answers
    for r in all_results:
        r["extracted_answer"] = extract_answer(r.get("response_text", ""))

    extraction_ok = sum(1 for r in all_results
                        if r.get("extracted_answer") in ("TRUE", "FALSE"))
    logger.info(f"Full extraction: {extraction_ok}/{len(all_results)} "
                f"({100 * extraction_ok / max(len(all_results), 1):.1f}%)")

    # 7. Embed all responses
    logger.info("Embedding all responses...")
    embeddings = embed_responses(all_results)

    # 8-9. Compute CSD indicators and analysis
    logger.info("Computing CSD indicators...")
    csd_by_model, analysis = run_csd_pipeline(
        all_results, embeddings, MODELS, DIFFICULTY_LEVELS,
    )

    # 10. Build and save output
    logger.info("Building output...")
    output = build_output(all_results, csd_by_model, analysis)

    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Saved {len(all_results)} examples to {out_path}")

    # Log summary
    total_elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info(f"Total time: {total_elapsed:.0f}s ({total_elapsed / 60:.1f}min)")
    logger.info(f"Total examples: {len(all_results)}")
    for model in MODELS:
        mid = model["id"]
        ma = analysis.get(mid, {})
        logger.info(f"  {mid} ({model['tier']}): "
                    f"d*={ma.get('d_star')}, "
                    f"scaling_alpha={ma.get('scaling', {}).get('alpha')}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
