#!/usr/bin/env python3
"""Syllogistic CSD Sampling: 3 Weak LLMs x d=2-30 x N=50 with Flickering Leading Indicator Test.

Merges two syllogistic datasets (d=2-15 + d=16-30) into a unified 440-problem
benchmark, runs CSD sampling with 3 weaker LLMs (ministral-3b, llama-3.1-8b,
gemini-2.0-flash-lite) across 22 difficulty levels, computes full CSD indicator
battery, and tests whether flickering is a leading indicator of reasoning
collapse -- directly targeting Success Criterion 1 on a 3rd task family.
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

WORKSPACE = Path(__file__).parent
(WORKSPACE / "logs").mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(WORKSPACE / "logs" / "run.log"), rotation="30 MB", level="DEBUG")

# ── CONSTANTS ────────────────────────────────────────────────────────────────

DATA_PATH_1 = WORKSPACE / "deps" / "data_id3_it1__opus" / "full_data_out.json"
DATA_PATH_2 = WORKSPACE / "deps" / "data_id4_it3__opus" / "full_data_out.json"
ABILITY_SERVER_URL = os.environ.get("ABILITY_SERVICE_URL", "http://localhost:9100")

MODELS = [
    {
        "id": "mistralai/ministral-3b-2512",
        "tier": "small",
        "price_in": 0.10,
        "price_out": 0.10,
    },
    {
        "id": "meta-llama/llama-3.1-8b-instruct",
        "tier": "medium",
        "price_in": 0.06,
        "price_out": 0.06,
    },
    {
        "id": "google/gemini-2.0-flash-lite-001",
        "tier": "weak",
        "price_in": 0.07,
        "price_out": 0.30,
    },
]

# Fallback models in case primary ones fail
FALLBACK_MODELS = {
    "mistralai/ministral-3b-2512": {
        "id": "mistralai/ministral-8b-2512",
        "tier": "small_backup",
        "price_in": 0.15,
        "price_out": 0.15,
    },
    "meta-llama/llama-3.1-8b-instruct": {
        "id": "meta-llama/llama-3.3-70b-instruct",
        "tier": "medium_backup",
        "price_in": 0.30,
        "price_out": 0.30,
    },
    "google/gemini-2.0-flash-lite-001": {
        "id": "google/gemini-2.0-flash-001",
        "tier": "weak_backup",
        "price_in": 0.10,
        "price_out": 0.40,
    },
}

TEMPERATURE = 0.8
TOP_P = 0.95
MAX_TOKENS = 1024
N_PROBLEMS_PER_LEVEL = 5
N_RESPONSES_PER_PROBLEM = 10
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
BUDGET_LIMIT = 9.0
MAX_CONCURRENT = 15
SEED = 42

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

# Set memory limit to 80% of available
RAM_BUDGET_BYTES = int(TOTAL_RAM_GB * 0.80 * 1e9)
try:
    resource.setrlimit(resource.RLIMIT_AS,
                       (RAM_BUDGET_BYTES * 3, RAM_BUDGET_BYTES * 3))
except (ValueError, OSError):
    logger.warning("Could not set RLIMIT_AS")

# CPU time limit: 55 minutes
try:
    resource.setrlimit(resource.RLIMIT_CPU, (3300, 3300))
except (ValueError, OSError):
    pass

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, "
            f"budget {RAM_BUDGET_BYTES / 1e9:.1f} GB")


# ── DATA LOADING & MERGING ───────────────────────────────────────────────────

def load_and_merge_datasets(path1: Path, path2: Path) -> dict[int, list[dict]]:
    """Load two syllogistic datasets and merge by difficulty level.

    data_id3: d=2-15 (14 levels x 20 problems = 280)
    data_id4: d=16,18,20,22,24,26,28,30 (8 levels x 20 problems = 160)
    Combined: 22 difficulty levels, 440 problems total.
    """
    logger.info(f"Loading dataset 1 from {path1}")
    data1 = json.loads(path1.read_text())
    logger.info(f"Loading dataset 2 from {path2}")
    data2 = json.loads(path2.read_text())

    all_examples = (data1["datasets"][0]["examples"]
                    + data2["datasets"][0]["examples"])

    by_difficulty: dict[int, list[dict]] = defaultdict(list)
    for ex in all_examples:
        by_difficulty[ex["metadata_difficulty"]].append(ex)

    difficulty_levels = sorted(by_difficulty.keys())
    logger.info(f"Merged dataset: {len(difficulty_levels)} levels, "
                f"{len(all_examples)} problems total")
    logger.info(f"Difficulty levels: {difficulty_levels}")

    for d in difficulty_levels:
        n = len(by_difficulty[d])
        n_true = sum(1 for ex in by_difficulty[d] if ex["output"] == "TRUE")
        logger.debug(f"  d={d}: {n} problems ({n_true}T/{n - n_true}F)")
        if n < 20:
            logger.warning(f"Only {n} problems at d={d}, expected 20")

    return dict(by_difficulty)


def select_problems(
    by_difficulty: dict[int, list[dict]],
    levels: list[int],
    n_per_level: int = N_PROBLEMS_PER_LEVEL,
    seed: int = SEED,
) -> dict[int, list[dict]]:
    """Select n_per_level problems per difficulty with TRUE/FALSE balance.

    Target: 3 TRUE + 2 FALSE (or 2+3 alternating by parity of d).
    """
    rng = np.random.default_rng(seed)
    selected: dict[int, list[dict]] = {}

    for d in levels:
        problems = by_difficulty.get(d, [])
        if not problems:
            logger.warning(f"No problems at d={d}, skipping")
            continue

        true_probs = [p for p in problems if p["output"] == "TRUE"]
        false_probs = [p for p in problems if p["output"] == "FALSE"]

        n_true = min(3 if d % 2 == 0 else 2, len(true_probs))
        n_false = min(n_per_level - n_true, len(false_probs))

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

    total = sum(len(v) for v in selected.values())
    logger.info(f"Selected {total} problems across {len(selected)} levels")
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
    client: Any,
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
                                        or "502" in error_msg
                                        or "504" in error_msg):
                        wait = 2 ** (attempt + 1)
                        logger.warning(f"Transient error ({error_msg[:80]}), "
                                       f"retry {attempt + 1}/3 in {wait}s")
                        await asyncio.sleep(wait)
                        continue
                    raise RuntimeError(f"API error: {error_msg[:200]}")

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

            except Exception as e:
                import httpx
                if isinstance(e, httpx.HTTPStatusError):
                    status = e.response.status_code
                    if attempt < 3 and status in (429, 502, 503, 504):
                        wait = 2 ** (attempt + 1)
                        logger.warning(f"HTTP {status}, retry {attempt+1}/3 in {wait}s")
                        await asyncio.sleep(wait)
                        continue
                    raise
                elif isinstance(e, (httpx.ConnectError, httpx.ReadTimeout)):
                    if attempt < 3:
                        wait = 2 ** (attempt + 1)
                        logger.warning(f"Connection error ({str(e)[:60]}), "
                                       f"retry {attempt+1}/3 in {wait}s")
                        await asyncio.sleep(wait)
                        continue
                    raise
                elif isinstance(e, RuntimeError):
                    raise
                else:
                    if attempt < 3:
                        wait = 2 ** (attempt + 1)
                        logger.warning(f"Unexpected error ({str(e)[:60]}), "
                                       f"retry {attempt+1}/3 in {wait}s")
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
    cost_tracker: CostTracker | None = None,
) -> tuple[list[dict], CostTracker]:
    """Generate LLM responses for selected problems.

    Returns:
        tuple of (result dicts list, cost_tracker)
    """
    import httpx

    if levels is None:
        levels = sorted(selected_problems.keys())
    if cost_tracker is None:
        cost_tracker = CostTracker()

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
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
                            SYSTEM_PROMPT, TEMPERATURE, MAX_TOKENS,
                            semaphore, cost_tracker, model_prices,
                        )
                        tasks.append(task)
                        task_meta.append(meta)

        total_calls = len(tasks)
        logger.info(f"Launching {total_calls} API calls across "
                    f"{len(models)} models, {len(levels)} levels")

        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

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

    logger.info(f"Completed {len(results)}/{total_calls} calls ({errors} errors)")
    logger.info(cost_tracker.summary())
    return results, cost_tracker


# ── ANSWER EXTRACTION ────────────────────────────────────────────────────────

def extract_answer(response_text: str) -> str | None:
    """Extract TRUE or FALSE from model response.

    Priority 1: Explicit declarations like 'answer is TRUE/FALSE'
    Priority 2: Last standalone TRUE/FALSE in text
    Priority 3: yes/no mapping
    Priority 4: correct/incorrect mapping
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

    # Priority 3: case-sensitive True/False
    matches = re.findall(r'\b(True|False)\b', response_text)
    if matches:
        return matches[-1].upper()

    # Priority 4: yes/no mapping
    yes_no = re.findall(r'\b(yes|no)\b', response_text, re.IGNORECASE)
    if yes_no:
        last = yes_no[-1].lower()
        return "TRUE" if last == "yes" else "FALSE"

    # Priority 5: correct/incorrect mapping
    corr = re.findall(r'\b(correct|incorrect)\b', response_text, re.IGNORECASE)
    if corr:
        last = corr[-1].lower()
        return "TRUE" if last == "correct" else "FALSE"

    return None


# ── SEMANTIC EMBEDDING ───────────────────────────────────────────────────────

def embed_responses(results: list[dict], batch_size: int = 64) -> np.ndarray:
    """Embed all response texts using all-MiniLM-L6-v2.

    Returns (N, 384) array of embeddings.
    """
    from sentence_transformers import SentenceTransformer

    logger.info(f"Embedding {len(results)} responses...")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    texts = []
    for r in results:
        text = r.get("response_text", "")
        if not text or text == "No output generated":
            text = "empty response"
        texts.append(text)

    # Process in chunks to avoid OOM
    chunk_size = 500
    all_embs = []
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i + chunk_size]
        embs = model.encode(
            chunk,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        all_embs.append(embs)
        logger.debug(f"  Embedded chunk {i}-{i+len(chunk)}")

    embeddings = np.vstack(all_embs)
    logger.info(f"Embeddings shape: {embeddings.shape}")

    del model
    gc.collect()

    return embeddings


# ── CSD INDICATOR COMPUTATION ────────────────────────────────────────────────

def compute_csd_indicators(
    embeddings_subset: np.ndarray,
    answers_subset: list[str | None],
    ground_truths: list[str],
) -> dict[str, Any]:
    """Compute full CSD indicator battery for responses at one (model, difficulty).

    Args:
        embeddings_subset: (N, D) embeddings for this difficulty x model
        answers_subset: list of N extracted answers (TRUE/FALSE/None)
        ground_truths: list of N ground truth answers

    Returns:
        dict with all CSD indicators
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.metrics.pairwise import cosine_distances
    from sklearn.mixture import GaussianMixture
    from scipy.stats import skew, kurtosis

    N = len(embeddings_subset)
    if N < 2:
        return _empty_csd_indicators(N)

    # --- A. Embedding Variance: trace(cov(embeddings.T)) ---
    try:
        cov_matrix = np.cov(embeddings_subset.T)
        embedding_variance = float(np.trace(cov_matrix))
    except Exception:
        embedding_variance = 0.0

    # --- B. Mean Pairwise Cosine Distance ---
    try:
        dist_matrix = cosine_distances(embeddings_subset)
        triu_idx = np.triu_indices(N, k=1)
        mean_cosine_distance = float(dist_matrix[triu_idx].mean())
    except Exception:
        mean_cosine_distance = 0.0

    # --- C. PCA -> PC1 projection for 1D tests ---
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

    # --- E. Silhouette k=2: KMeans(2) on full embeddings ---
    try:
        if N >= 4:
            km = KMeans(n_clusters=2, n_init=10, random_state=SEED)
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
        if n > 3:
            denom_correction = 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
        else:
            denom_correction = 3.0
        bc_denominator = k + denom_correction
        bimodality_coefficient = bc_numerator / bc_denominator if bc_denominator != 0 else 0.0
    except Exception:
        bimodality_coefficient = 0.0

    # --- G. Ashman D: Fit GaussianMixture(2) on PC1 ---
    try:
        gmm = GaussianMixture(n_components=2, random_state=SEED)
        gmm.fit(pc1.reshape(-1, 1))
        mu1, mu2 = gmm.means_.flatten()
        s1, s2 = np.sqrt(gmm.covariances_.flatten())
        denom = s1 ** 2 + s2 ** 2
        ashman_d = float(np.sqrt(2) * abs(mu1 - mu2) / np.sqrt(denom)) if denom > 0 else 0.0
    except Exception:
        ashman_d = 0.0

    # --- H. Self-Consistency Disagreement: 1 - max_fraction ---
    valid_answers = [a for a in answers_subset if a in ("TRUE", "FALSE")]
    if valid_answers:
        counter = Counter(valid_answers)
        majority_fraction = counter.most_common(1)[0][1] / len(valid_answers)
        disagreement_rate = 1.0 - majority_fraction
        majority_answer = counter.most_common(1)[0][0]
    else:
        disagreement_rate = 1.0
        majority_answer = None

    # --- I. Answer Balance: min(n_true, n_false) / total ---
    n_true = sum(1 for a in valid_answers if a == "TRUE")
    n_false = sum(1 for a in valid_answers if a == "FALSE")
    answer_balance = min(n_true, n_false) / max(n_true + n_false, 1)

    # --- J. Accuracy ---
    correct = sum(1 for a, g in zip(answers_subset, ground_truths) if a == g)
    accuracy = correct / N if N > 0 else 0.0

    # --- K. Bimodality Consensus: >=2/3 of {dip_p<0.05, sil>0.3, BC>5/9} ---
    bimodality_flags = [
        dip_pvalue < 0.05,
        sil_score > 0.3,
        bimodality_coefficient > 5 / 9,
    ]
    bimodality_consensus = sum(bimodality_flags) >= 2

    return {
        "embedding_variance": embedding_variance,
        "mean_cosine_distance": mean_cosine_distance,
        "pc1_variance_explained": pc1_variance_explained,
        "dip_statistic": dip_stat,
        "dip_pvalue": dip_pvalue,
        "silhouette_k2": sil_score,
        "bimodality_coefficient": bimodality_coefficient,
        "ashman_d": ashman_d,
        "disagreement_rate": disagreement_rate,
        "answer_balance": answer_balance,
        "majority_answer": majority_answer,
        "accuracy": accuracy,
        "bimodality_consensus": bimodality_consensus,
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
        "ashman_d": 0.0,
        "disagreement_rate": 1.0,
        "answer_balance": 0.0,
        "majority_answer": None,
        "accuracy": 0.0,
        "bimodality_consensus": False,
        "n_responses": n,
        "n_valid_answers": 0,
        "n_extraction_failures": n,
    }


# ── WITHIN-CHAIN CONFIDENCE AUTOCORRELATION ──────────────────────────────────

def compute_chain_autocorrelation(response_text: str) -> float | None:
    """Parse step-by-step reasoning into steps, extract verbalized confidence,
    compute lag-1 autocorrelation of step 'certainty'.
    """
    if not response_text:
        return None

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

    Only use levels where d < d*. Tests if alpha in [-0.7, -0.3].
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
    """Test whether CSD indicator rises significantly before d*.

    Uses Kendall tau trend test on pre-boundary levels.
    """
    from scipy.stats import kendalltau

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


def test_flickering_leading(
    accuracy_by_level: dict[int, float],
    csd_by_level: dict[int, dict],
    d_star: int | None,
    accuracy_threshold: float = 0.8,
) -> dict[str, Any]:
    """Test flickering leading indicator (SC1 target).

    Check if bimodality signals flicker at levels where accuracy is still high.
    """
    if d_star is None:
        max_d = max(accuracy_by_level.keys()) + 1
    else:
        max_d = d_star

    high_acc_levels = [d for d, a in accuracy_by_level.items()
                       if a > accuracy_threshold and d < max_d]

    if not high_acc_levels:
        return {
            "flickering_dip": False,
            "flickering_sil": False,
            "flickering_bc": False,
            "flickering_consensus": False,
            "earliest_flickering_level": None,
            "lead_time": 0,
            "n_high_acc_levels": 0,
        }

    flicker_dip = False
    flicker_sil = False
    flicker_bc = False
    flicker_consensus = False
    earliest_level = None

    for d in sorted(high_acc_levels):
        csd = csd_by_level.get(d, {})
        if csd.get("dip_pvalue", 1.0) < 0.05:
            flicker_dip = True
            if earliest_level is None:
                earliest_level = d
        if csd.get("silhouette_k2", 0.0) > 0.3:
            flicker_sil = True
            if earliest_level is None:
                earliest_level = d
        if csd.get("bimodality_coefficient", 0.0) > 5 / 9:
            flicker_bc = True
            if earliest_level is None:
                earliest_level = d
        if csd.get("bimodality_consensus", False):
            flicker_consensus = True
            if earliest_level is None:
                earliest_level = d

    lead_time = (max_d - earliest_level) if earliest_level is not None else 0

    return {
        "flickering_dip": flicker_dip,
        "flickering_sil": flicker_sil,
        "flickering_bc": flicker_bc,
        "flickering_consensus": flicker_consensus,
        "earliest_flickering_level": earliest_level,
        "lead_time": lead_time,
        "n_high_acc_levels": len(high_acc_levels),
    }


def compute_mixture_correlation(
    accuracy_by_level: dict[int, float],
    variance_by_level: dict[int, float],
) -> dict[str, Any]:
    """Test whether variance tracks the binomial mixture prediction p(1-p).

    If responses are a mixture of correct/incorrect processes, variance should
    correlate with p*(1-p) where p is accuracy.
    """
    from scipy.stats import pearsonr

    common_levels = sorted(set(accuracy_by_level) & set(variance_by_level))
    if len(common_levels) < 4:
        return {"pearson_r": None, "p_value": None, "n_levels": len(common_levels)}

    accs = [accuracy_by_level[d] for d in common_levels]
    vars_ = [variance_by_level[d] for d in common_levels]
    mixture_pred = [p * (1 - p) for p in accs]

    try:
        r, p = pearsonr(mixture_pred, vars_)
        return {"pearson_r": float(r), "p_value": float(p),
                "n_levels": len(common_levels)}
    except Exception:
        return {"pearson_r": None, "p_value": None, "n_levels": len(common_levels)}


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
                f"sil={csd['silhouette_k2']:.4f} "
                f"ashman_d={csd['ashman_d']:.4f} "
                f"dis={csd['disagreement_rate']:.2f}"
            )

    # Per-model aggregate analysis
    analysis: dict[str, dict] = {}
    for model in models:
        model_id = model["id"]
        model_csd = csd_by_model.get(model_id, {})

        acc_by_level = {d: model_csd[d]["accuracy"]
                        for d in levels if d in model_csd}
        var_by_level = {d: model_csd[d]["embedding_variance"]
                        for d in levels if d in model_csd}
        dip_by_level = {d: model_csd[d]["dip_statistic"]
                        for d in levels if d in model_csd}
        sil_by_level = {d: model_csd[d]["silhouette_k2"]
                        for d in levels if d in model_csd}
        dis_by_level = {d: model_csd[d]["disagreement_rate"]
                        for d in levels if d in model_csd}

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
                    "levels_tested": 0,
                }

        # Flickering leading indicator test (SC1 target)
        model_analysis["flickering"] = test_flickering_leading(
            acc_by_level, model_csd, d_star)

        # Mixture model fit
        model_analysis["mixture_corr"] = compute_mixture_correlation(
            acc_by_level, var_by_level)

        analysis[model_id] = model_analysis

    return csd_by_model, analysis


# ── OUTPUT ASSEMBLY ──────────────────────────────────────────────────────────

def build_output(
    csd_by_model: dict[str, dict[int, dict]],
    analysis: dict[str, dict],
    models: list[dict],
    levels: list[int],
    cost_tracker: CostTracker,
) -> dict:
    """Build exp_gen_sol_out.json format output.

    One entry per (model, difficulty) pair = 66 entries.
    predict_* fields are strings, metadata_* can be any type.
    """
    examples = []
    for model in models:
        model_id = model["id"]
        model_analysis = analysis.get(model_id, {})
        d_star = model_analysis.get("d_star")
        scaling = model_analysis.get("scaling", {})
        flick = model_analysis.get("flickering", {})
        mixture = model_analysis.get("mixture_corr", {})

        for d in levels:
            csd = csd_by_model.get(model_id, {}).get(d, _empty_csd_indicators())

            # Leading indicator results
            leading_var = model_analysis.get("leading_var", {})
            leading_dip = model_analysis.get("leading_dip", {})

            example = {
                "input": f"CSD analysis at difficulty={d} for model={model_id}",
                "output": str(csd.get("accuracy", 0.0)),
                # predict_* fields must be strings
                "predict_accuracy": str(csd.get("accuracy", 0.0)),
                "predict_csd_variance": str(csd.get("embedding_variance", 0.0)),
                "predict_dip_statistic": str(csd.get("dip_statistic", 0.0)),
                "predict_dip_pvalue": str(csd.get("dip_pvalue", 1.0)),
                "predict_silhouette_k2": str(csd.get("silhouette_k2", 0.0)),
                "predict_bimodality_coefficient": str(csd.get("bimodality_coefficient", 0.0)),
                "predict_ashman_d": str(csd.get("ashman_d", 0.0)),
                "predict_disagreement_rate": str(csd.get("disagreement_rate", 1.0)),
                "predict_answer_balance": str(csd.get("answer_balance", 0.0)),
                "predict_chain_autocorrelation": str(csd.get("avg_chain_autocorrelation", 0.0)),
                "predict_bimodality_consensus": str(csd.get("bimodality_consensus", False)),
                "predict_mean_cosine_distance": str(csd.get("mean_cosine_distance", 0.0)),
                # metadata fields
                "metadata_difficulty": d,
                "metadata_model": model_id,
                "metadata_tier": model["tier"],
                "metadata_n_responses": csd.get("n_responses", 0),
                "metadata_n_valid_answers": csd.get("n_valid_answers", 0),
                "metadata_d_star": d_star,
                "metadata_scaling_alpha": scaling.get("alpha"),
                "metadata_scaling_r2": scaling.get("r_squared"),
                "metadata_consistent_with_fold": scaling.get("consistent_with_fold"),
                "metadata_flickering_leading": flick.get("flickering_dip", False),
                "metadata_sil_leading": flick.get("flickering_sil", False),
                "metadata_bc_leading": flick.get("flickering_bc", False),
                "metadata_consensus_leading": flick.get("flickering_consensus", False),
                "metadata_earliest_flickering_level": flick.get("earliest_flickering_level"),
                "metadata_lead_time": flick.get("lead_time", 0),
                "metadata_variance_mixture_corr": mixture.get("pearson_r"),
                "metadata_tau_variance": leading_var.get("kendall_tau"),
                "metadata_p_tau_variance": leading_var.get("p_value"),
                "metadata_tau_dip": leading_dip.get("kendall_tau"),
                "metadata_p_tau_dip": leading_dip.get("p_value"),
                "metadata_fold": "test",
            }
            examples.append(example)

    # Build model summaries
    model_summaries = {}
    for model in models:
        mid = model["id"]
        ma = analysis.get(mid, {})
        model_summaries[mid] = {
            "tier": model["tier"],
            "d_star": ma.get("d_star"),
            "scaling": ma.get("scaling", {}),
            "flickering": ma.get("flickering", {}),
            "mixture_corr": ma.get("mixture_corr", {}),
            "leading_var": ma.get("leading_var", {}),
            "leading_dip": ma.get("leading_dip", {}),
            "leading_sil": ma.get("leading_sil", {}),
            "leading_dis": ma.get("leading_dis", {}),
        }

    # SC1 summary
    sc1_results = {
        "any_flickering_detected": any(
            analysis.get(m["id"], {}).get("flickering", {}).get("flickering_dip", False)
            or analysis.get(m["id"], {}).get("flickering", {}).get("flickering_sil", False)
            for m in models
        ),
        "per_model": {
            m["id"]: analysis.get(m["id"], {}).get("flickering", {})
            for m in models
        },
    }

    return {
        "metadata": {
            "method_name": "CSD_sampling_syllogistic_extended",
            "description": (
                "Critical Slowing Down indicators computed from LLM response "
                "distributions across 22 difficulty levels (d=2-30) of syllogistic "
                "logic tasks with 3 weak LLMs. Tests flickering as leading indicator "
                "of reasoning collapse (SC1)."
            ),
            "task_family": "syllogistic_logic",
            "difficulty_range": [min(levels), max(levels)],
            "experiment_config": {
                "models": [m["id"] for m in models],
                "model_tiers": {m["id"]: m["tier"] for m in models},
                "n_difficulty_levels": len(levels),
                "n_problems_per_level": N_PROBLEMS_PER_LEVEL,
                "n_responses_per_problem": N_RESPONSES_PER_PROBLEM,
                "n_total_per_model_level": N_PROBLEMS_PER_LEVEL * N_RESPONSES_PER_PROBLEM,
                "temperature": TEMPERATURE,
                "embed_model": EMBED_MODEL_NAME,
            },
            "total_cost": cost_tracker.total,
            "cost_by_model": dict(cost_tracker.by_model),
            "model_summaries": model_summaries,
            "sc1_results": sc1_results,
        },
        "datasets": [{
            "dataset": "syllogistic_logic_csd_extended",
            "examples": examples,
        }],
    }


# ── VALIDATION HELPERS ───────────────────────────────────────────────────────

def validate_mini_results(results: list[dict]) -> bool:
    """Validate mini run results before scaling up."""
    if not results:
        logger.error("MINI RUN: No results returned!")
        return False

    # Extract answers
    for r in results:
        if "extracted_answer" not in r:
            r["extracted_answer"] = extract_answer(r.get("response_text", ""))

    extracted = [r for r in results if r.get("extracted_answer") in ("TRUE", "FALSE")]
    rate = len(extracted) / len(results)
    logger.info(f"MINI RUN: extraction rate = {rate:.0%} ({len(extracted)}/{len(results)})")
    if rate < 0.5:
        logger.warning("MINI RUN: extraction rate very low!")

    non_empty = sum(1 for r in results
                    if r.get("response_text", "")
                    and r["response_text"] != "No output generated")
    logger.info(f"MINI RUN: {non_empty}/{len(results)} non-empty responses")
    if non_empty == 0:
        logger.error("MINI RUN: All responses are empty!")
        return False

    # Log sample responses
    for r in results[:3]:
        resp = r.get("response_text", "")[:200]
        logger.debug(f"Sample response (d={r['difficulty']}, {r['model']}): {resp}")

    return True


# ── MAIN ORCHESTRATION ───────────────────────────────────────────────────────

@logger.catch
def main():
    """Main experiment orchestration with gradual scaling."""
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("SYLLOGISTIC LOGIC CSD EXPERIMENT (EXTENDED d=2-30)")
    logger.info("=" * 60)

    # 1. Load & merge datasets
    by_difficulty = load_and_merge_datasets(DATA_PATH_1, DATA_PATH_2)
    all_levels = sorted(by_difficulty.keys())
    logger.info(f"All difficulty levels ({len(all_levels)}): {all_levels}")

    # 2. Select problems for full run (5 per level, balanced TRUE/FALSE)
    selected_full = select_problems(by_difficulty, all_levels)

    for d in sorted(selected_full.keys()):
        true_count = sum(1 for p in selected_full[d] if p["output"] == "TRUE")
        false_count = len(selected_full[d]) - true_count
        logger.debug(f"  d={d}: {len(selected_full[d])} problems "
                     f"({true_count}T/{false_count}F)")

    # Shared cost tracker across all phases
    cost_tracker = CostTracker()

    # ──────────────────────────────────────────────────────────────────────
    # PHASE A: MINI RUN
    # 2 models x 8 spread levels x 2 problems x 5 responses = 160 calls
    # ──────────────────────────────────────────────────────────────────────
    logger.info("=" * 40)
    logger.info("=== PHASE A: MINI RUN ===")
    logger.info("=" * 40)

    mini_models = [MODELS[0], MODELS[2]]  # ministral-3b, gemini-flash-lite
    mini_levels = [2, 5, 8, 12, 16, 20, 24, 30]
    mini_levels = [l for l in mini_levels if l in selected_full]

    mini_start = time.time()
    mini_results, cost_tracker = asyncio.run(generate_responses(
        selected_full, mini_models,
        n_problems=2, n_responses=5, levels=mini_levels,
        cost_tracker=cost_tracker,
    ))
    mini_elapsed = time.time() - mini_start
    logger.info(f"Mini run: {len(mini_results)} results in {mini_elapsed:.1f}s")

    if not validate_mini_results(mini_results):
        logger.warning("Mini run validation issues. Trying backup models...")
        backup_models = [FALLBACK_MODELS[m["id"]] for m in mini_models
                         if m["id"] in FALLBACK_MODELS]
        mini_results_backup, cost_tracker = asyncio.run(generate_responses(
            selected_full, backup_models,
            n_problems=2, n_responses=5, levels=[2, 8],
            cost_tracker=cost_tracker,
        ))
        for r in mini_results_backup:
            r["extracted_answer"] = extract_answer(r.get("response_text", ""))
        if not validate_mini_results(mini_results_backup):
            logger.error("Backup models also failed. Using whatever we have.")
        else:
            logger.info("Backup models working, but continuing with primary models")

    # Test embedding on mini results
    mini_embeddings = embed_responses(mini_results)
    assert mini_embeddings.shape[1] == 384, f"Bad embedding dim: {mini_embeddings.shape}"
    assert not np.any(np.isnan(mini_embeddings)), "NaN in embeddings!"
    logger.info("Mini run: embeddings OK")

    # Test CSD on mini data
    if len(mini_results) >= 4:
        test_csd = compute_csd_indicators(
            mini_embeddings[:4],
            [r.get("extracted_answer") for r in mini_results[:4]],
            [r["ground_truth"] for r in mini_results[:4]],
        )
        logger.info(f"Mini CSD test: variance={test_csd['embedding_variance']:.4f}, "
                    f"accuracy={test_csd['accuracy']:.2f}, "
                    f"ashman_d={test_csd['ashman_d']:.4f}")

    # Log accuracy at easy vs hard
    for model in mini_models:
        mid = model["id"]
        for d in [2, 30]:
            model_d = [r for r in mini_results
                       if r["model"] == mid and r["difficulty"] == d]
            if model_d:
                correct = sum(1 for r in model_d
                              if r.get("extracted_answer") == r["ground_truth"])
                logger.info(f"  {mid} d={d}: {correct}/{len(model_d)} correct")

    logger.info("Phase A PASSED")
    del mini_embeddings
    gc.collect()

    # ──────────────────────────────────────────────────────────────────────
    # PHASE B: MEDIUM RUN
    # 3 models x 15 levels x 3 problems x 10 responses = 1350 calls
    # ──────────────────────────────────────────────────────────────────────
    logger.info("=" * 40)
    logger.info("=== PHASE B: MEDIUM RUN ===")
    logger.info("=" * 40)

    medium_levels = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    medium_levels = [l for l in medium_levels if l in selected_full]

    medium_start = time.time()
    medium_results, cost_tracker = asyncio.run(generate_responses(
        selected_full, MODELS,
        n_problems=3, n_responses=10, levels=medium_levels,
        cost_tracker=cost_tracker,
    ))
    medium_elapsed = time.time() - medium_start
    logger.info(f"Medium run: {len(medium_results)} results in {medium_elapsed:.1f}s")

    for r in medium_results:
        r["extracted_answer"] = extract_answer(r.get("response_text", ""))

    # Quick accuracy check
    for model in MODELS:
        mid = model["id"]
        for d in [2, 10, 20, 30]:
            model_d = [r for r in medium_results
                       if r["model"] == mid and r["difficulty"] == d]
            if model_d:
                correct = sum(1 for r in model_d
                              if r.get("extracted_answer") == r["ground_truth"])
                logger.info(f"  {mid} d={d}: {correct}/{len(model_d)} = "
                            f"{correct/len(model_d):.2f}")

    # Extrapolate timing for full run
    calls_per_sec = len(medium_results) / max(medium_elapsed, 1)
    total_full_calls = (len(MODELS) * len(all_levels)
                        * N_PROBLEMS_PER_LEVEL * N_RESPONSES_PER_PROBLEM)
    estimated_full_time = total_full_calls / max(calls_per_sec, 0.1)
    logger.info(f"Rate: {calls_per_sec:.1f} calls/s, "
                f"full run estimate: {estimated_full_time:.0f}s "
                f"({estimated_full_time / 60:.1f}min)")

    elapsed_so_far = time.time() - start_time
    remaining_budget = 55 * 60 - elapsed_so_far  # 55 min total budget
    if estimated_full_time > remaining_budget * 0.8:
        logger.warning(f"Full run may exceed time budget! "
                       f"Estimated: {estimated_full_time:.0f}s, "
                       f"remaining: {remaining_budget:.0f}s")

    logger.info(f"Cost so far: ${cost_tracker.total:.4f}")
    logger.info("Phase B PASSED")
    del medium_results
    gc.collect()

    # ──────────────────────────────────────────────────────────────────────
    # PHASE C: FULL RUN
    # 3 models x 22 levels x 5 problems x 10 responses = 3300 calls
    # ──────────────────────────────────────────────────────────────────────
    logger.info("=" * 40)
    logger.info(f"=== PHASE C: FULL RUN ({total_full_calls} calls) ===")
    logger.info("=" * 40)

    full_start = time.time()
    all_results, cost_tracker = asyncio.run(generate_responses(
        selected_full, MODELS,
        n_problems=N_PROBLEMS_PER_LEVEL,
        n_responses=N_RESPONSES_PER_PROBLEM,
        levels=all_levels,
        cost_tracker=cost_tracker,
    ))
    full_elapsed = time.time() - full_start
    logger.info(f"Full run: {len(all_results)} results in {full_elapsed:.1f}s "
                f"({full_elapsed / 60:.1f}min)")

    # Check response count >= 90% of target
    target = total_full_calls
    actual = len(all_results)
    logger.info(f"Response count: {actual}/{target} = {actual/target:.1%}")
    if actual < target * 0.9:
        logger.warning(f"Only {actual}/{target} responses, below 90% threshold")

    # Extract answers
    for r in all_results:
        r["extracted_answer"] = extract_answer(r.get("response_text", ""))

    extraction_ok = sum(1 for r in all_results
                        if r.get("extracted_answer") in ("TRUE", "FALSE"))
    logger.info(f"Full extraction: {extraction_ok}/{len(all_results)} "
                f"({100 * extraction_ok / max(len(all_results), 1):.1f}%)")

    # Embed all responses
    logger.info("Embedding all responses...")
    embeddings = embed_responses(all_results)

    # Compute CSD indicators and analysis
    logger.info("Computing CSD indicators...")
    csd_by_model, analysis = run_csd_pipeline(
        all_results, embeddings, MODELS, all_levels,
    )

    # Build and save output
    logger.info("Building output...")
    output = build_output(csd_by_model, analysis, MODELS, all_levels, cost_tracker)

    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Saved {len(output['datasets'][0]['examples'])} examples to {out_path}")

    # Log final summary
    total_elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info(f"Total time: {total_elapsed:.0f}s ({total_elapsed / 60:.1f}min)")
    logger.info(f"Total cost: ${cost_tracker.total:.4f}")
    logger.info(f"Total examples in output: {len(output['datasets'][0]['examples'])}")

    for model in MODELS:
        mid = model["id"]
        ma = analysis.get(mid, {})
        flick = ma.get("flickering", {})
        logger.info(
            f"  {mid} ({model['tier']}): "
            f"d*={ma.get('d_star')}, "
            f"alpha={ma.get('scaling', {}).get('alpha')}, "
            f"R2={ma.get('scaling', {}).get('r_squared')}, "
            f"flicker_dip={flick.get('flickering_dip')}, "
            f"flicker_sil={flick.get('flickering_sil')}"
        )

    sc1 = output["metadata"]["sc1_results"]
    logger.info(f"SC1 any flickering detected: {sc1['any_flickering_detected']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
