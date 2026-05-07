#!/usr/bin/env python3
"""Multi-Hop Factual Reasoning CSD Sampling Experiment (6 Difficulty Levels).

For 3 LLMs of varying capability, generates N=50 responses at each of 6
hop-count difficulty levels (1-6), evaluates correctness via fuzzy answer
matching, computes semantic embeddings and a full battery of CSD indicators
(variance, Hartigan dip, silhouette, bimodality coefficient, Ashman D,
self-consistency disagreement). Tests whether CSD signals are detectable
even at this coarse granularity.
"""

import gc
import json
import math
import os
import random
import re
import resource
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import psutil

# ---------------------------------------------------------------------------
# Hardware detection (cgroup-aware)
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
TOTAL_RAM_GB = _container_ram_gb() or psutil.virtual_memory().total / 1e9
AVAILABLE_RAM_GB = min(psutil.virtual_memory().available / 1e9, TOTAL_RAM_GB)

# Set memory limit: 20 GB budget (well within 29 GB container)
RAM_BUDGET = int(20 * 1024**3)
_avail = psutil.virtual_memory().available
assert RAM_BUDGET < _avail, f"Budget {RAM_BUDGET/1e9:.1f}GB > available {_avail/1e9:.1f}GB"
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))

# ---------------------------------------------------------------------------
# Logging (loguru)
# ---------------------------------------------------------------------------
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")

WORKSPACE = Path("/ai-inventor/aii_pipeline/runs/neurips-open/3_invention_loop/iter_2/gen_art/exp_id4_it2__opus")
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_PATH = Path("/ai-inventor/aii_pipeline/runs/neurips-open/3_invention_loop/iter_1/gen_art/data_id5_it1__opus/full_data_out.json")

MODELS = [
    "meta-llama/llama-3.1-8b-instruct",
    "openai/gpt-4o-mini",
    "google/gemini-2.0-flash-001",
]
MODEL_TIERS = ["small", "medium", "large"]

TEMPERATURE = 0.8
TOP_P = 0.95
MAX_TOKENS = 2048
N_PROBLEMS_PER_LEVEL = 5
N_RESPONSES_PER_PROBLEM = 10  # 5 problems x 10 = 50 per level
DIFFICULTY_LEVELS = [1, 2, 3, 4, 5, 6]
SEED = 42

# Cost tracking
TOTAL_COST_USD = 0.0
COST_LIMIT_USD = 9.0  # Hard stop at $9 to stay under $10 limit
TOTAL_INPUT_TOKENS = 0
TOTAL_OUTPUT_TOKENS = 0

# Approximate pricing per 1M tokens (conservative estimates)
MODEL_PRICING = {
    "meta-llama/llama-3.1-8b-instruct": {"input": 0.06, "output": 0.06},
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "google/gemini-2.0-flash-001": {"input": 0.10, "output": 0.40},
}

SYSTEM_PROMPT = """You are a helpful assistant. Answer the following question by reasoning step by step. Show your reasoning for each hop or sub-question. After your reasoning, provide your final answer on a new line starting with "ANSWER: " followed by just the answer (a short phrase or entity name)."""

USER_TEMPLATE = "Question: {question}"

# Number of concurrent threads for API calls
NUM_WORKERS = min(NUM_CPUS * 4, 16)

# ---------------------------------------------------------------------------
# Phase control: set via env var or default to "full"
# Allows gradual scaling: "mini", "single", "full"
# ---------------------------------------------------------------------------
PHASE = os.environ.get("CSD_PHASE", "full")


# ===========================================================================
# STEP 1: Load and prepare data
# ===========================================================================

def load_data(data_path: Path) -> dict[int, list[dict]]:
    """Load dataset, group by difficulty level, select 5 problems per level."""
    logger.info(f"Loading data from {data_path}")
    raw = json.loads(data_path.read_text())
    examples = raw["datasets"][0]["examples"]
    logger.info(f"Loaded {len(examples)} total examples")

    by_level: dict[int, list[dict]] = defaultdict(list)
    for ex in examples:
        level = ex["metadata_difficulty_level"]
        by_level[level].append(ex)

    random.seed(SEED)
    selected: dict[int, list[dict]] = {}
    for level in DIFFICULTY_LEVELS:
        pool = by_level[level]
        assert len(pool) >= N_PROBLEMS_PER_LEVEL, (
            f"Level {level} has only {len(pool)} problems, need {N_PROBLEMS_PER_LEVEL}"
        )
        selected[level] = random.sample(pool, N_PROBLEMS_PER_LEVEL)

    total = sum(len(v) for v in selected.values())
    logger.info(f"Selected {total} problems across {len(selected)} levels")

    for level in DIFFICULTY_LEVELS:
        sample = selected[level][0]
        logger.debug(f"  Level {level} sample: {sample['input'][:80]}... -> {sample['output']}")

    return selected


# ===========================================================================
# STEP 2: LLM Sampling (parallel via ThreadPoolExecutor + call_server)
# ===========================================================================

def call_llm(model: str, question: str) -> dict:
    """Call a single LLM via the ability server. Returns response dict."""
    global TOTAL_COST_USD, TOTAL_INPUT_TOKENS, TOTAL_OUTPUT_TOKENS

    from aii_lib.abilities.ability_server import call_server

    try:
        result = call_server("aii_openrouter__call", {
            "model": model,
            "input_text": USER_TEMPLATE.format(question=question),
            "instructions": SYSTEM_PROMPT,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
        }, timeout=120)

        if result is None:
            logger.warning(f"call_server returned None for {model}")
            return {"text": "", "error": "call_server returned None", "input_tokens": 0, "output_tokens": 0}

        if not result.get("success"):
            error_msg = result.get("error", "Unknown error")
            logger.warning(f"API error for {model}: {error_msg[:200]}")
            return {"text": "", "error": error_msg, "input_tokens": 0, "output_tokens": 0}

        in_tok = result.get("input_tokens", 0)
        out_tok = result.get("output_tokens", 0)

        # Track costs
        pricing = MODEL_PRICING.get(model, {"input": 0.5, "output": 1.0})
        cost = (in_tok * pricing["input"] + out_tok * pricing["output"]) / 1_000_000
        TOTAL_COST_USD += cost
        TOTAL_INPUT_TOKENS += in_tok
        TOTAL_OUTPUT_TOKENS += out_tok

        return {
            "text": result.get("response", ""),
            "error": None,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
        }

    except Exception as e:
        logger.exception(f"Exception calling {model}")
        return {"text": "", "error": str(e), "input_tokens": 0, "output_tokens": 0}


def sample_responses_for_model(
    selected_problems: dict[int, list[dict]],
    model: str,
    levels: list[int] | None = None,
    n_problems: int | None = None,
    n_reps: int | None = None,
) -> dict[int, list[dict]]:
    """Generate responses for one model across difficulty levels.

    Returns: {level: [{problem_id, question, ground_truth, aliases, responses: [str...]}]}
    """
    global TOTAL_COST_USD

    if levels is None:
        levels = DIFFICULTY_LEVELS
    if n_problems is None:
        n_problems = N_PROBLEMS_PER_LEVEL
    if n_reps is None:
        n_reps = N_RESPONSES_PER_PROBLEM

    results_by_level: dict[int, list[dict]] = {}
    total_calls = len(levels) * n_problems * n_reps
    logger.info(f"Sampling {total_calls} responses for {model} ({len(levels)} levels x {n_problems} problems x {n_reps} reps)")

    for level in levels:
        problems = selected_problems[level][:n_problems]
        level_results = []

        for prob_idx, problem in enumerate(problems):
            question = problem["input"]
            ground_truth = problem["output"]
            aliases = problem.get("metadata_answer_aliases", "[]")
            problem_id = problem.get("metadata_id", f"L{level}_P{prob_idx}")

            # Check cost before submitting
            if TOTAL_COST_USD >= COST_LIMIT_USD:
                logger.error(f"COST LIMIT REACHED: ${TOTAL_COST_USD:.2f} >= ${COST_LIMIT_USD:.2f}. Stopping.")
                level_results.append({
                    "problem_id": problem_id,
                    "question": question,
                    "ground_truth": ground_truth,
                    "aliases": aliases,
                    "responses": [],
                })
                continue

            # Submit all reps for this problem in parallel
            futures = {}
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                for rep_idx in range(n_reps):
                    future = executor.submit(call_llm, model, question)
                    futures[future] = rep_idx

                responses = [""] * n_reps
                errors = 0
                for future in as_completed(futures):
                    rep_idx = futures[future]
                    try:
                        result = future.result(timeout=130)
                        responses[rep_idx] = result["text"]
                        if result["error"]:
                            errors += 1
                    except Exception as e:
                        logger.exception(f"Future failed for {model} L{level} P{prob_idx} R{rep_idx}")
                        responses[rep_idx] = ""
                        errors += 1

            non_empty = sum(1 for r in responses if r.strip())
            logger.debug(
                f"  L{level} P{prob_idx}: {non_empty}/{n_reps} responses "
                f"(errors={errors}, cost=${TOTAL_COST_USD:.3f})"
            )

            level_results.append({
                "problem_id": problem_id,
                "question": question,
                "ground_truth": ground_truth,
                "aliases": aliases,
                "responses": responses,
            })

        results_by_level[level] = level_results

    logger.info(
        f"Completed {model}: total_cost=${TOTAL_COST_USD:.3f}, "
        f"tokens={TOTAL_INPUT_TOKENS}in/{TOTAL_OUTPUT_TOKENS}out"
    )
    return results_by_level


# ===========================================================================
# STEP 3: Fuzzy answer evaluation
# ===========================================================================

def extract_answer(response_text: str) -> str:
    """Extract final answer from response text."""
    if not response_text or not response_text.strip():
        return ""

    # Strategy 1: "ANSWER: " prefix (case-insensitive)
    match = re.search(r'ANSWER:\s*(.+?)(?:\n|$)', response_text, re.IGNORECASE)
    if match:
        return match.group(1).strip().rstrip('.')

    # Strategy 2: Common answer patterns
    patterns = [
        r'(?:the\s+)?(?:final\s+)?answer\s+is[:\s]+(.+?)(?:\.|$)',
        r'(?:therefore|thus|so|hence)[,\s]+(?:the\s+answer\s+is\s+)?(.+?)(?:\.|$)',
    ]
    for pat in patterns:
        m = re.search(pat, response_text, re.IGNORECASE)
        if m:
            return m.group(1).strip().rstrip('.')

    # Strategy 3: Last non-empty line (fallback)
    lines = [line.strip() for line in response_text.strip().split('\n') if line.strip()]
    if lines:
        last = lines[-1]
        # Truncate if too long
        return last[:100] if len(last) > 100 else last

    return ""


def _clean_answer(text: str) -> str:
    """Normalize answer for comparison."""
    t = text.lower().strip().rstrip('.')
    # Strip common articles/prepositions
    for prefix in ["the ", "a ", "an ", "in ", "at ", "on "]:
        if t.startswith(prefix):
            t = t[len(prefix):]
    return t.strip()


def fuzzy_match(predicted: str, ground_truth: str, aliases_str: str) -> tuple[bool, float]:
    """Multi-criteria fuzzy matching. Returns (is_correct, f1_score)."""
    pred_clean = _clean_answer(predicted)
    gt_clean = _clean_answer(ground_truth)

    if not pred_clean:
        return False, 0.0

    # Parse aliases
    try:
        aliases = json.loads(aliases_str) if aliases_str else []
    except (json.JSONDecodeError, TypeError):
        aliases = []
    all_answers = [gt_clean] + [_clean_answer(a) for a in aliases]

    for ans in all_answers:
        # Criterion 1: Exact match
        if pred_clean == ans:
            return True, 1.0

        # Criterion 2: Substring containment
        if ans and pred_clean and (ans in pred_clean or pred_clean in ans):
            return True, 0.9

    # Criterion 3: Token-level F1
    pred_tokens = set(pred_clean.split())
    gt_tokens = set(gt_clean.split())
    if pred_tokens and gt_tokens:
        common = pred_tokens & gt_tokens
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gt_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        if f1 >= 0.5:
            return True, f1

    return False, 0.0


def evaluate_responses(results_by_level: dict[int, list[dict]]) -> dict[int, list[dict]]:
    """Evaluate all responses: extract answers and compute fuzzy match.

    Mutates each problem dict to add 'extracted_answers', 'correctness', 'f1_scores'.
    """
    for level, problems in results_by_level.items():
        total_correct = 0
        total_responses = 0
        for prob in problems:
            extracted = [extract_answer(r) for r in prob["responses"]]
            correctness = []
            f1_scores = []
            for ans in extracted:
                is_correct, f1 = fuzzy_match(ans, prob["ground_truth"], prob["aliases"])
                correctness.append(is_correct)
                f1_scores.append(f1)

            prob["extracted_answers"] = extracted
            prob["correctness"] = correctness
            prob["f1_scores"] = f1_scores

            total_correct += sum(correctness)
            total_responses += len(correctness)

        acc = total_correct / total_responses if total_responses > 0 else 0.0
        logger.info(f"  Level {level}: accuracy={acc:.3f} ({total_correct}/{total_responses})")

    return results_by_level


# ===========================================================================
# STEP 4: Compute embeddings
# ===========================================================================

def compute_embeddings(results_by_level: dict[int, list[dict]]) -> dict[int, np.ndarray]:
    """Embed all responses using all-MiniLM-L6-v2. Returns {level: ndarray(N, 384)}."""
    from sentence_transformers import SentenceTransformer

    logger.info("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Model loaded.")

    embeddings_by_level: dict[int, np.ndarray] = {}
    for level in sorted(results_by_level.keys()):
        texts = []
        for prob in results_by_level[level]:
            texts.extend(prob["responses"])

        # Filter out empty responses for embedding, but track indices
        non_empty_indices = [i for i, t in enumerate(texts) if t.strip()]
        non_empty_texts = [texts[i] for i in non_empty_indices]

        if not non_empty_texts:
            logger.warning(f"Level {level}: no non-empty responses to embed")
            embeddings_by_level[level] = np.zeros((len(texts), 384))
            continue

        embs = model.encode(non_empty_texts, batch_size=64, show_progress_bar=False)

        # Reconstruct full embedding array (zeros for empty responses)
        full_embs = np.zeros((len(texts), embs.shape[1]))
        for idx, emb_idx in enumerate(non_empty_indices):
            full_embs[emb_idx] = embs[idx]

        embeddings_by_level[level] = full_embs
        logger.info(f"  Level {level}: embedded {len(non_empty_texts)}/{len(texts)} responses -> shape {full_embs.shape}")

    # Free model memory
    del model
    gc.collect()

    return embeddings_by_level


# ===========================================================================
# STEP 5: CSD Indicator Computation (per-level, no rolling windows)
# ===========================================================================

def compute_csd_indicators(
    embeddings: np.ndarray,
    correctness: list[bool],
    answers: list[str],
) -> dict:
    """Compute full CSD indicator battery for one difficulty level."""
    import diptest
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.mixture import GaussianMixture
    from scipy.stats import skew, kurtosis

    N = embeddings.shape[0]

    # Filter out zero-vector rows (from empty responses)
    norms = np.linalg.norm(embeddings, axis=1)
    valid_mask = norms > 1e-8
    valid_embeddings = embeddings[valid_mask]
    valid_correctness = [c for c, v in zip(correctness, valid_mask) if v]
    valid_answers = [a for a, v in zip(answers, valid_mask) if v]
    N_valid = valid_embeddings.shape[0]

    if N_valid < 4:
        logger.warning(f"Only {N_valid} valid embeddings, returning degenerate indicators")
        return _degenerate_indicators(N, N_valid, correctness, answers)

    # --- A. Embedding Variance ---
    cov = np.cov(valid_embeddings.T)
    variance_trace = float(np.trace(cov))

    from scipy.spatial.distance import pdist
    cos_dists = pdist(valid_embeddings, metric='cosine')
    mean_cosine_dist = float(np.mean(cos_dists))
    std_cosine_dist = float(np.std(cos_dists))

    # --- B. PC1 Projection for 1D Tests ---
    n_components = min(5, N_valid - 1)
    if n_components < 1:
        n_components = 1
    pca = PCA(n_components=n_components)
    pc_scores = pca.fit_transform(valid_embeddings)
    pc1 = pc_scores[:, 0]
    pc1_var_explained = float(pca.explained_variance_ratio_[0])

    # --- C. Hartigan's Dip Test (on PC1) ---
    try:
        dip_stat, dip_pval = diptest.diptest(pc1)
    except Exception:
        dip_stat, dip_pval = 0.0, 1.0

    # --- D. Silhouette Score (k=2 on full embeddings) ---
    if N_valid >= 4:
        try:
            kmeans = KMeans(n_clusters=2, random_state=SEED, n_init=10)
            labels = kmeans.fit_predict(valid_embeddings)
            if len(set(labels)) == 2:
                sil_score = float(silhouette_score(valid_embeddings, labels))
            else:
                sil_score = -1.0
        except Exception:
            sil_score = -1.0
    else:
        sil_score = -1.0

    # --- E. Bimodality Coefficient (on PC1) ---
    s = float(skew(pc1))
    k = float(kurtosis(pc1))  # excess kurtosis
    n = len(pc1)
    bc_numerator = s**2 + 1
    if n > 3:
        bc_denominator = k + 3 * (n - 1)**2 / ((n - 2) * (n - 3))
    else:
        bc_denominator = k + 3
    bimodality_coeff = bc_numerator / bc_denominator if bc_denominator != 0 else 0

    # --- F. Ashman's D (fit 2-component GMM on PC1) ---
    try:
        gmm = GaussianMixture(n_components=2, random_state=SEED)
        gmm.fit(pc1.reshape(-1, 1))
        mu1, mu2 = gmm.means_.flatten()
        s1, s2 = np.sqrt(gmm.covariances_.flatten())
        ashman_d = float(np.sqrt(2) * abs(mu1 - mu2) / np.sqrt(s1**2 + s2**2)) if (s1**2 + s2**2) > 0 else 0
    except Exception:
        ashman_d = 0.0

    # --- G. Self-Consistency Disagreement (BASELINE) ---
    if valid_answers:
        answer_counts = Counter(a.lower().strip() for a in valid_answers if a.strip())
        if answer_counts:
            most_common_count = answer_counts.most_common(1)[0][1]
            disagreement_rate = 1.0 - (most_common_count / len(valid_answers))
        else:
            disagreement_rate = 1.0
    else:
        disagreement_rate = 1.0

    # --- H. Accuracy ---
    accuracy = sum(valid_correctness) / len(valid_correctness) if valid_correctness else 0.0

    # --- I. Bimodality Consensus (>=2/3 agree) ---
    bimodality_flags = [
        dip_pval < 0.05,
        sil_score > 0.3,
        bimodality_coeff > 5 / 9,
    ]
    bimodality_consensus = sum(bimodality_flags) >= 2

    # --- J. Additional: Entropy of answer distribution ---
    if valid_answers:
        answer_counts_list = Counter(a.lower().strip() for a in valid_answers if a.strip())
        total_a = sum(answer_counts_list.values())
        if total_a > 0:
            probs = [c / total_a for c in answer_counts_list.values()]
            answer_entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        else:
            answer_entropy = 0.0
    else:
        answer_entropy = 0.0

    return {
        "accuracy": round(accuracy, 4),
        "n_responses": N,
        "n_valid_responses": N_valid,
        "n_correct": sum(valid_correctness),
        # Variance indicators
        "embedding_variance_trace": round(variance_trace, 6),
        "mean_cosine_distance": round(mean_cosine_dist, 6),
        "std_cosine_distance": round(std_cosine_dist, 6),
        # Bimodality indicators
        "hartigan_dip_stat": round(float(dip_stat), 6),
        "hartigan_dip_pval": round(float(dip_pval), 6),
        "silhouette_score_k2": round(float(sil_score), 4),
        "bimodality_coefficient": round(float(bimodality_coeff), 4),
        "ashman_d": round(float(ashman_d), 4),
        "bimodality_consensus": bimodality_consensus,
        # Baseline
        "self_consistency_disagreement": round(float(disagreement_rate), 4),
        # PCA info
        "pc1_variance_explained": round(float(pc1_var_explained), 4),
        # Answer distribution
        "unique_answers": len(set(a.lower().strip() for a in valid_answers if a.strip())),
        "answer_entropy": round(float(answer_entropy), 4),
    }


def _degenerate_indicators(N: int, N_valid: int, correctness: list[bool], answers: list[str]) -> dict:
    """Return degenerate indicator values when too few valid responses."""
    accuracy = sum(correctness) / len(correctness) if correctness else 0.0
    return {
        "accuracy": round(accuracy, 4),
        "n_responses": N,
        "n_valid_responses": N_valid,
        "n_correct": sum(correctness),
        "embedding_variance_trace": 0.0,
        "mean_cosine_distance": 0.0,
        "std_cosine_distance": 0.0,
        "hartigan_dip_stat": 0.0,
        "hartigan_dip_pval": 1.0,
        "silhouette_score_k2": -1.0,
        "bimodality_coefficient": 0.0,
        "ashman_d": 0.0,
        "bimodality_consensus": False,
        "self_consistency_disagreement": 1.0,
        "pc1_variance_explained": 0.0,
        "unique_answers": 0,
        "answer_entropy": 0.0,
    }


# ===========================================================================
# STEP 6: Cross-level trend analysis
# ===========================================================================

def compute_cross_level_trends(level_indicators: dict[int, dict]) -> dict:
    """Compute Kendall tau trends across the 6 difficulty levels."""
    from scipy.stats import kendalltau

    levels = sorted(level_indicators.keys())
    results = {}

    indicator_names = [
        "embedding_variance_trace", "mean_cosine_distance",
        "hartigan_dip_stat", "silhouette_score_k2",
        "bimodality_coefficient", "self_consistency_disagreement",
        "accuracy", "answer_entropy", "ashman_d",
    ]

    for name in indicator_names:
        values = [level_indicators[lv].get(name, 0) for lv in levels]
        try:
            tau, pval = kendalltau(levels, values)
            results[f"kendall_tau_{name}"] = round(float(tau), 4)
            results[f"kendall_pval_{name}"] = round(float(pval), 4)
        except Exception:
            results[f"kendall_tau_{name}"] = 0.0
            results[f"kendall_pval_{name}"] = 1.0

    return results


def compute_variance_scaling(level_indicators: dict[int, dict]) -> dict:
    """Fit log(Var) ~ alpha * log(d* - d) for levels below d*."""
    levels = sorted(level_indicators.keys())

    # Find d*: first level where accuracy < 0.5
    d_star_found = None
    for lv in levels:
        if level_indicators[lv]["accuracy"] < 0.5:
            d_star_found = lv
            break

    if d_star_found is None:
        d_star_found = max(levels) + 1

    fit_levels = [lv for lv in levels if lv < d_star_found]
    if len(fit_levels) < 3:
        return {"scaling_status": "insufficient_points", "d_star": d_star_found}

    try:
        log_dist = [np.log(d_star_found - lv) for lv in fit_levels]
        log_var = [np.log(max(level_indicators[lv]["embedding_variance_trace"], 1e-10)) for lv in fit_levels]
        coeffs = np.polyfit(log_dist, log_var, 1)
        alpha = float(coeffs[0])

        return {
            "scaling_status": "fitted",
            "d_star": d_star_found,
            "alpha_exponent": round(alpha, 4),
            "in_predicted_range": -0.7 <= alpha <= -0.3,
            "fit_levels_used": fit_levels,
            "n_fit_points": len(fit_levels),
        }
    except Exception as e:
        logger.exception("Variance scaling fit failed")
        return {"scaling_status": "fit_failed", "d_star": d_star_found, "error": str(e)}


# ===========================================================================
# STEP 7: Leading indicator test
# ===========================================================================

def test_leading_indicator(level_indicators: dict[int, dict]) -> dict:
    """Test if CSD indicators become significant BEFORE accuracy drops."""
    levels = sorted(level_indicators.keys())

    d_star = None
    for lv in levels:
        if level_indicators[lv]["accuracy"] < 0.5:
            d_star = lv
            break

    high_acc_levels = [lv for lv in levels if level_indicators[lv]["accuracy"] > 0.8]

    leading_signals = []
    for lv in high_acc_levels:
        ind = level_indicators[lv]
        if ind["hartigan_dip_pval"] < 0.05:
            leading_signals.append({"level": lv, "indicator": "hartigan_dip", "pval": ind["hartigan_dip_pval"]})
        if ind["silhouette_score_k2"] > 0.3:
            leading_signals.append({"level": lv, "indicator": "silhouette", "value": ind["silhouette_score_k2"]})
        if ind["bimodality_coefficient"] > 5 / 9:
            leading_signals.append({"level": lv, "indicator": "bimodality_coeff", "value": ind["bimodality_coefficient"]})

    return {
        "d_star": d_star,
        "high_accuracy_levels": high_acc_levels,
        "leading_signals_found": len(leading_signals) > 0,
        "leading_signals": leading_signals,
        "n_leading_signals": len(leading_signals),
        "caveat": "Only 6 difficulty levels - limited resolution for leading indicator detection",
    }


# ===========================================================================
# STEP 8: Assemble method_out.json
# ===========================================================================

def build_method_out(
    all_model_results: dict[str, dict[int, list[dict]]],
    all_model_indicators: dict[str, dict[int, dict]],
    all_model_trends: dict[str, dict],
    all_model_scaling: dict[str, dict],
    all_model_leading: dict[str, dict],
) -> dict:
    """Build the final output conforming to exp_gen_sol_out schema."""

    # Build per-model summary
    per_model_summary = {}
    for model, tier in zip(MODELS, MODEL_TIERS):
        if model not in all_model_indicators:
            continue
        per_model_summary[model] = {
            "tier": tier,
            "per_level_indicators": {
                str(lv): ind for lv, ind in all_model_indicators[model].items()
            },
            "trends": all_model_trends.get(model, {}),
            "variance_scaling": all_model_scaling.get(model, {}),
            "leading_indicator": all_model_leading.get(model, {}),
        }

    # Cross-model summary
    models_with_csd = []
    models_with_leading = []
    for model in MODELS:
        if model not in all_model_indicators:
            continue
        # Check if any level shows bimodality consensus
        for lv, ind in all_model_indicators[model].items():
            if ind.get("bimodality_consensus"):
                models_with_csd.append(model)
                break
        if all_model_leading.get(model, {}).get("leading_signals_found"):
            models_with_leading.append(model)

    # Success criteria assessment
    success = {
        "flickering_detected": len(models_with_csd) > 0,
        "n_models_with_bimodality": len(models_with_csd),
        "models_with_bimodality": models_with_csd,
        "variance_scaling_fit": any(
            s.get("scaling_status") == "fitted" and s.get("in_predicted_range", False)
            for s in all_model_scaling.values()
        ),
        "leading_indicator_found": len(models_with_leading) > 0,
        "models_with_leading_indicator": models_with_leading,
    }

    metadata = {
        "experiment_name": "multi_hop_reasoning_csd_6levels",
        "task_family": "multi_hop_factual_reasoning",
        "description": (
            "CSD sampling experiment on multi-hop factual reasoning with 6 difficulty "
            "levels. Tests whether Critical Slowing Down indicators (variance, dip test, "
            "silhouette, bimodality, Ashman D) can detect approaching capability boundaries "
            "before accuracy drops. Self-consistency disagreement serves as baseline."
        ),
        "models": MODELS,
        "model_tiers": MODEL_TIERS,
        "n_problems_per_level": N_PROBLEMS_PER_LEVEL,
        "n_responses_per_problem": N_RESPONSES_PER_PROBLEM,
        "n_total_per_level": N_PROBLEMS_PER_LEVEL * N_RESPONSES_PER_PROBLEM,
        "difficulty_levels": DIFFICULTY_LEVELS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "seed": SEED,
        "embedding_model": "all-MiniLM-L6-v2",
        "per_model_summary": per_model_summary,
        "cross_model_summary": {
            "models_with_csd_signal": models_with_csd,
            "models_with_leading_indicator": models_with_leading,
        },
        "reduced_resolution_caveat": (
            "Only 6 difficulty levels available. Statistical power for trend detection "
            "is limited (Kendall tau with n=6, min p-value ~0.014). Results are "
            "inconclusive (not disconfirming) if no signals appear due to limited resolution."
        ),
        "success_criteria_assessment": success,
        "total_cost_usd": round(TOTAL_COST_USD, 4),
        "total_input_tokens": TOTAL_INPUT_TOKENS,
        "total_output_tokens": TOTAL_OUTPUT_TOKENS,
    }

    # Build examples: one per (problem, model) pair
    examples = []
    for model, tier in zip(MODELS, MODEL_TIERS):
        if model not in all_model_results:
            continue
        for level in sorted(all_model_results[model].keys()):
            level_ind = all_model_indicators.get(model, {}).get(level, {})
            for prob in all_model_results[model][level]:
                responses = prob.get("responses", [])
                extracted = prob.get("extracted_answers", [])
                correctness = prob.get("correctness", [])
                f1_scores = prob.get("f1_scores", [])

                n_correct = sum(correctness) if correctness else 0
                n_total = len(correctness) if correctness else 0
                prob_accuracy = n_correct / n_total if n_total > 0 else 0.0

                example = {
                    "input": prob["question"],
                    "output": prob["ground_truth"],
                    "metadata_difficulty_level": level,
                    "metadata_model": model,
                    "metadata_model_tier": tier,
                    "metadata_problem_id": prob.get("problem_id", ""),
                    "metadata_answer_aliases": prob.get("aliases", "[]"),
                    "metadata_csd_indicators": json.dumps(level_ind),
                    "predict_responses": json.dumps(responses),
                    "predict_extracted_answers": json.dumps(extracted),
                    "predict_correctness": json.dumps(correctness),
                    "predict_f1_scores": json.dumps(f1_scores),
                    "predict_accuracy": str(round(prob_accuracy, 4)),
                    "predict_n_correct": str(n_correct),
                    "predict_n_total": str(n_total),
                }
                examples.append(example)

    output = {
        "metadata": metadata,
        "datasets": [
            {
                "dataset": "multi_hop_reasoning_csd",
                "examples": examples,
            }
        ],
    }

    return output


# ===========================================================================
# STEP 9: Unit tests
# ===========================================================================

def run_unit_tests():
    """Run unit tests for answer extraction and fuzzy matching."""
    logger.info("=== Running unit tests ===")

    # Test extract_answer
    test_cases_extract = [
        ("The answer is Thames.", "Thames"),
        ("ANSWER: Project Mercury", "Project Mercury"),
        ("...therefore, Poland.\n", "Poland"),
        ("Step 1: blah\nStep 2: blah\nParis", "Paris"),
        ("ANSWER: the Thames river", "the Thames river"),
        ("", ""),
    ]
    for text, expected in test_cases_extract:
        result = extract_answer(text)
        status = "PASS" if result.lower().strip('.') == expected.lower().strip('.') else "FAIL"
        logger.info(f"  extract_answer: {status} | input='{text[:40]}...' expected='{expected}' got='{result}'")

    # Test fuzzy_match
    test_cases_match = [
        ("Thames", "Thames", "[]", True),
        ("the thames", "Thames", "[]", True),
        ("Kensington", "in Kensington", "[]", True),
        ("wrong answer", "Thames", "[]", False),
        ("John Locke philosopher", "John Locke", "[]", True),
        ("Project Mercury", "Project Mercury", '["Mercury Program"]', True),
        ("Mercury Program", "Project Mercury", '["Mercury Program"]', True),
    ]
    for pred, gt, aliases, expected_correct in test_cases_match:
        is_correct, f1 = fuzzy_match(pred, gt, aliases)
        status = "PASS" if is_correct == expected_correct else "FAIL"
        logger.info(f"  fuzzy_match: {status} | pred='{pred}' gt='{gt}' expected={expected_correct} got={is_correct} (f1={f1:.2f})")

    logger.info("=== Unit tests complete ===")


# ===========================================================================
# MAIN: Gradual scaling execution
# ===========================================================================

@logger.catch
def main():
    t_start = time.time()
    logger.info(f"=== CSD Multi-Hop Reasoning Experiment ===")
    logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, Workers={NUM_WORKERS}")
    logger.info(f"Phase: {PHASE}")

    # Run unit tests first
    run_unit_tests()

    # Load data
    selected = load_data(DATA_PATH)

    if PHASE == "mini":
        # Phase A: Mini test - 1 model, 3 levels, N=5 (1 problem x 5 reps)
        logger.info("=== PHASE A: Mini Test ===")
        model = MODELS[1]  # gpt-4o-mini
        levels = [1, 3, 6]

        results = sample_responses_for_model(selected, model, levels=levels, n_problems=1, n_reps=5)
        results = evaluate_responses(results)
        embeddings = compute_embeddings(results)

        for lv in levels:
            all_correct = []
            all_answers = []
            for prob in results[lv]:
                all_correct.extend(prob["correctness"])
                all_answers.extend(prob["extracted_answers"])
            indicators = compute_csd_indicators(embeddings[lv], all_correct, all_answers)
            logger.info(f"Level {lv} indicators: {json.dumps(indicators, indent=2)}")

        elapsed = time.time() - t_start
        logger.info(f"Mini test completed in {elapsed:.1f}s, cost=${TOTAL_COST_USD:.4f}")
        return

    if PHASE == "single":
        # Phase B: Single model full - 1 model, all 6 levels, N=50
        logger.info("=== PHASE B: Single Model Full ===")
        model = MODELS[1]  # gpt-4o-mini

        results = sample_responses_for_model(selected, model)
        results = evaluate_responses(results)
        embeddings = compute_embeddings(results)

        level_indicators = {}
        for lv in DIFFICULTY_LEVELS:
            all_correct = []
            all_answers = []
            for prob in results[lv]:
                all_correct.extend(prob["correctness"])
                all_answers.extend(prob["extracted_answers"])
            level_indicators[lv] = compute_csd_indicators(embeddings[lv], all_correct, all_answers)

        trends = compute_cross_level_trends(level_indicators)
        logger.info(f"Trends: {json.dumps(trends, indent=2)}")

        elapsed = time.time() - t_start
        logger.info(f"Single model full completed in {elapsed:.1f}s, cost=${TOTAL_COST_USD:.4f}")
        return

    # Phase C+D: Full experiment - all 3 models
    logger.info("=== PHASE C: Full Experiment (3 models) ===")

    all_model_results: dict[str, dict[int, list[dict]]] = {}
    all_model_indicators: dict[str, dict[int, dict]] = {}
    all_model_trends: dict[str, dict] = {}
    all_model_scaling: dict[str, dict] = {}
    all_model_leading: dict[str, dict] = {}

    for model_idx, (model, tier) in enumerate(zip(MODELS, MODEL_TIERS)):
        if TOTAL_COST_USD >= COST_LIMIT_USD:
            logger.error(f"COST LIMIT: ${TOTAL_COST_USD:.2f} >= ${COST_LIMIT_USD}. Skipping remaining models.")
            break

        logger.info(f"--- Model {model_idx+1}/3: {model} ({tier}) ---")
        t_model = time.time()

        # Sample responses
        results = sample_responses_for_model(selected, model)

        # Evaluate
        logger.info(f"Evaluating responses for {model}...")
        results = evaluate_responses(results)

        # Store results
        all_model_results[model] = results

        # Compute embeddings
        logger.info(f"Computing embeddings for {model}...")
        embeddings = compute_embeddings(results)

        # Compute CSD indicators per level
        logger.info(f"Computing CSD indicators for {model}...")
        level_indicators = {}
        for lv in DIFFICULTY_LEVELS:
            all_correct = []
            all_answers = []
            for prob in results[lv]:
                all_correct.extend(prob.get("correctness", []))
                all_answers.extend(prob.get("extracted_answers", []))
            level_indicators[lv] = compute_csd_indicators(embeddings[lv], all_correct, all_answers)

        all_model_indicators[model] = level_indicators

        # Cross-level trends
        trends = compute_cross_level_trends(level_indicators)
        all_model_trends[model] = trends

        # Variance scaling
        scaling = compute_variance_scaling(level_indicators)
        all_model_scaling[model] = scaling

        # Leading indicator test
        leading = test_leading_indicator(level_indicators)
        all_model_leading[model] = leading

        elapsed_model = time.time() - t_model
        logger.info(
            f"Model {model} completed in {elapsed_model:.1f}s | "
            f"cost=${TOTAL_COST_USD:.3f} | "
            f"scaling={scaling.get('scaling_status')} | "
            f"leading={leading.get('leading_signals_found')}"
        )

        # Log per-level summary
        for lv in DIFFICULTY_LEVELS:
            ind = level_indicators[lv]
            logger.info(
                f"  L{lv}: acc={ind['accuracy']:.3f}, var_trace={ind['embedding_variance_trace']:.4f}, "
                f"dip_p={ind['hartigan_dip_pval']:.4f}, sil={ind['silhouette_score_k2']:.3f}, "
                f"bc={ind['bimodality_coefficient']:.4f}, disagree={ind['self_consistency_disagreement']:.3f}"
            )

        # Free embeddings memory
        del embeddings
        gc.collect()

    # Phase D: Build output
    logger.info("=== PHASE D: Building Output ===")
    output = build_method_out(
        all_model_results,
        all_model_indicators,
        all_model_trends,
        all_model_scaling,
        all_model_leading,
    )

    # Write method_out.json
    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    n_examples = len(output["datasets"][0]["examples"])
    logger.info(f"Wrote {out_path} with {n_examples} examples")

    elapsed_total = time.time() - t_start
    logger.info(
        f"=== Experiment complete ===\n"
        f"  Total time: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)\n"
        f"  Total cost: ${TOTAL_COST_USD:.4f}\n"
        f"  Total tokens: {TOTAL_INPUT_TOKENS} in / {TOTAL_OUTPUT_TOKENS} out\n"
        f"  Examples: {n_examples}"
    )


if __name__ == "__main__":
    main()
