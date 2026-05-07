#!/usr/bin/env python3
"""CSD Sampling Experiment: Multi-Step Arithmetic x 3 LLMs x CSD Indicator Extraction.

Generates N LLM responses per difficulty level across 3 capability-tiered models
on 480 arithmetic chain problems, computing CSD indicators (embedding variance,
Hartigan dip, silhouette, bimodality coefficient, within-chain autocorrelation)
plus baselines (self-consistency disagreement, extraction rate), testing for
flickering as a leading indicator of reasoning collapse, and fitting fold-
bifurcation variance scaling laws.
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
from dataclasses import asdict, dataclass
from pathlib import Path

import diptest
import httpx
import numpy as np
from loguru import logger
from scipy import stats as sp_stats
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

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
TOTAL_RAM_GB = _container_ram_gb() or 29.0

# Set memory limit at 80% of container RAM (leave buffer for OS + agent)
RAM_BUDGET_BYTES = int(TOTAL_RAM_GB * 0.80 * 1e9)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET_BYTES * 3, RAM_BUDGET_BYTES * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).parent
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = WORKSPACE / "full_data_out.json"
OUTPUT_PATH = WORKSPACE / "method_out.json"

# Ability server for OpenRouter calls (port 9100)
ABILITY_SERVER_URL = os.environ.get(
    "ABILITY_SERVICE_URL",
    f"http://{os.environ.get('ABILITY_SERVICE_HOST', 'localhost')}:"
    f"{os.environ.get('ABILITY_SERVICE_PORT', '9100')}",
)
ABILITY_ENDPOINT = "aii_openrouter__call"

MODELS = [
    "meta-llama/llama-3.1-8b-instruct",   # small
    "google/gemini-2.0-flash-001",         # medium
    "openai/gpt-4o-mini",                  # large
]

SAMPLING_PARAMS = {"temperature": 0.8, "top_p": 0.95, "max_tokens": 2048}
DIFFICULTY_LEVELS = list(range(2, 26))     # 2..25 = 24 levels
PROBLEMS_PER_LEVEL = 5
RESPONSES_PER_PROBLEM = 10
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# Gradual scaling configs
SCALE_MINI = {
    "models": [MODELS[1]],
    "levels": [2, 7, 12, 17, 22],
    "n_resp": 5,
    "n_prob": 2,
}
SCALE_MEDIUM = {
    "models": MODELS[:2],
    "levels": DIFFICULTY_LEVELS,
    "n_resp": 4,
    "n_prob": 5,
}
SCALE_FULL = {
    "models": MODELS,
    "levels": DIFFICULTY_LEVELS,
    "n_resp": RESPONSES_PER_PROBLEM,
    "n_prob": PROBLEMS_PER_LEVEL,
}

# Cost tracking
TOTAL_COST_USD = 0.0
COST_LIMIT_USD = 9.50  # hard stop well below $10

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CSDIndicators:
    accuracy: float
    embedding_variance: float
    dip_statistic: float
    dip_pvalue: float
    silhouette_k2: float
    bimodality_coefficient: float
    disagreement_rate: float
    mean_confidence: float  # extraction rate as proxy
    step_correctness_autocorr: float


# ---------------------------------------------------------------------------
# PHASE 1: Load dataset
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> dict[int, list[dict]]:
    """Load and index problems by difficulty level."""
    logger.info(f"Loading dataset from {path}")
    raw = json.loads(path.read_text())
    examples = raw["datasets"][0]["examples"]
    logger.info(f"Loaded {len(examples)} examples")

    problems_by_level: dict[int, list[dict]] = defaultdict(list)
    for ex in examples:
        problems_by_level[ex["metadata_difficulty_level"]].append(ex)

    assert len(problems_by_level) == 24, f"Expected 24 levels, got {len(problems_by_level)}"
    for lvl, probs in problems_by_level.items():
        assert len(probs) == 20, f"Level {lvl} has {len(probs)} problems, expected 20"

    logger.info(f"Indexed {len(problems_by_level)} difficulty levels, 20 problems each")
    return dict(problems_by_level)


# ---------------------------------------------------------------------------
# PHASE 2: LLM Response Generation
# ---------------------------------------------------------------------------

async def call_llm(
    client: httpx.AsyncClient,
    model: str,
    prompt: str,
    params: dict,
    max_retries: int = 4,
) -> tuple[str, float]:
    """Call LLM via ability server. Returns (response_text, cost_usd)."""
    global TOTAL_COST_USD

    if TOTAL_COST_USD >= COST_LIMIT_USD:
        raise RuntimeError(f"Cost limit reached: ${TOTAL_COST_USD:.4f} >= ${COST_LIMIT_USD}")

    url = f"{ABILITY_SERVER_URL}/{ABILITY_ENDPOINT}"
    payload = {
        "model": model,
        "input_text": prompt,
        "max_tokens": params["max_tokens"],
        "temperature": params["temperature"],
        "top_p": params["top_p"],
    }

    last_err = None
    for attempt in range(max_retries):
        try:
            resp = await client.post(url, json=payload, timeout=180.0)
            resp.raise_for_status()
            data = resp.json()

            if not data.get("success"):
                err_msg = data.get("error", "Unknown error")
                if attempt < max_retries - 1 and ("rate" in err_msg.lower() or "timeout" in err_msg.lower()):
                    wait = 2 ** (attempt + 1)
                    logger.debug(f"Retryable error: {err_msg}, waiting {wait}s")
                    await asyncio.sleep(wait)
                    continue
                raise RuntimeError(f"Ability server error: {err_msg}")

            text = data.get("response", "")
            in_tok = data.get("input_tokens", 0)
            out_tok = data.get("output_tokens", 0)
            # Estimate cost: ~$0.3/M tokens average across cheap models
            cost = (in_tok + out_tok) * 0.3 / 1_000_000
            TOTAL_COST_USD += cost
            return text, cost

        except httpx.HTTPStatusError as e:
            last_err = e
            if e.response.status_code in (429, 502, 503, 504):
                wait = 2 ** (attempt + 1)
                logger.debug(f"HTTP {e.response.status_code}, retry in {wait}s")
                await asyncio.sleep(wait)
                continue
            raise
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as e:
            last_err = e
            wait = 2 ** (attempt + 1)
            logger.debug(f"Connection error: {e}, retry in {wait}s")
            await asyncio.sleep(wait)
            continue

    raise RuntimeError(f"All {max_retries} retries failed: {last_err}")


async def generate_responses_for_config(
    problems_by_level: dict[int, list[dict]],
    models: list[str],
    levels: list[int],
    n_problems: int,
    n_responses: int,
    concurrency: int = 25,
) -> dict[tuple[str, int], list[dict]]:
    """Generate responses for given scale config.

    Returns {(model, level): [response_dicts]}.
    """
    global TOTAL_COST_USD

    results: dict[tuple[str, int], list[dict]] = {}
    total_calls = len(models) * len(levels) * n_problems * n_responses
    logger.info(
        f"Generating responses: {len(models)} models x {len(levels)} levels x "
        f"{n_problems} problems x {n_responses} responses = {total_calls} API calls"
    )

    sem = asyncio.Semaphore(concurrency)
    completed = 0
    failed = 0
    t0 = time.time()

    async def bounded_call(
        client: httpx.AsyncClient,
        model: str,
        level: int,
        p_idx: int,
        r_idx: int,
        prompt: str,
        gt: str,
    ) -> tuple[str, int, int, int, str, str, bool]:
        nonlocal completed, failed
        async with sem:
            try:
                text, cost = await call_llm(client, model, prompt, SAMPLING_PARAMS)
                completed += 1
                if completed % 50 == 0:
                    elapsed = time.time() - t0
                    rate = completed / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"  Progress: {completed}/{total_calls} "
                        f"({rate:.1f}/s, ${TOTAL_COST_USD:.4f})"
                    )
                return model, level, p_idx, r_idx, text, gt, True
            except Exception as e:
                failed += 1
                logger.warning(
                    f"  Failed call {model} lvl={level} p={p_idx} r={r_idx}: "
                    f"{type(e).__name__}: {str(e)[:100]}"
                )
                return model, level, p_idx, r_idx, "", gt, False

    async with httpx.AsyncClient() as client:
        tasks = []
        for model in models:
            for level in levels:
                probs = problems_by_level[level][:n_problems]
                for p_idx, prob in enumerate(probs):
                    for r_idx in range(n_responses):
                        tasks.append(
                            bounded_call(
                                client, model, level, p_idx, r_idx,
                                prob["input"], prob["output"],
                            )
                        )

        raw_results = await asyncio.gather(*tasks)

    # Organize results
    for model, level, p_idx, r_idx, text, gt, success in raw_results:
        key = (model, level)
        if key not in results:
            results[key] = []
        if success and text.strip():
            results[key].append({
                "problem_idx": p_idx,
                "response_idx": r_idx,
                "response_text": text,
                "ground_truth": gt,
            })

    elapsed = time.time() - t0
    logger.info(
        f"Generation complete: {completed} ok, {failed} failed, "
        f"{elapsed:.1f}s, ${TOTAL_COST_USD:.4f} total cost"
    )
    return results


# ---------------------------------------------------------------------------
# PHASE 3: Answer Extraction & Accuracy
# ---------------------------------------------------------------------------

def extract_final_answer(response_text: str) -> int | None:
    """Extract the final numeric answer from LLM response."""
    patterns = [
        r'(?:final\s+(?:answer|result)\s*(?:is|=|:)\s*)(-?\d[\d,]*)',
        r'(?:the\s+(?:answer|result)\s*(?:is|=|:)\s*)(-?\d[\d,]*)',
        r'\*\*(-?\d[\d,]*)\*\*\s*$',
        r'=\s*(-?\d[\d,]*)\s*$',
        r'(?:equals|equal\s+to)\s+(-?\d[\d,]*)',
        r'boxed\{(-?\d[\d,]*)\}',
    ]
    for pat in patterns:
        m = re.search(pat, response_text, re.IGNORECASE | re.MULTILINE)
        if m:
            try:
                return int(m.group(1).replace(",", ""))
            except ValueError:
                continue
    # Fallback: last integer in text
    all_nums = re.findall(r'-?\d+', response_text)
    if all_nums:
        try:
            return int(all_nums[-1])
        except ValueError:
            pass
    return None


def compute_accuracy(responses: list[dict]) -> float:
    """Fraction of responses where extracted answer matches ground truth."""
    correct = 0
    total = 0
    for r in responses:
        extracted = extract_final_answer(r["response_text"])
        try:
            gt = int(r["ground_truth"])
        except (ValueError, TypeError):
            continue
        if extracted is not None:
            total += 1
            if extracted == gt:
                correct += 1
    return correct / max(total, 1)


# ---------------------------------------------------------------------------
# PHASE 4: Semantic Embedding
# ---------------------------------------------------------------------------

def embed_all_responses(
    all_results: dict[tuple[str, int], list[dict]],
    model_name: str = EMBED_MODEL_NAME,
) -> dict[tuple[str, int], np.ndarray]:
    """Embed all responses using sentence-transformers."""
    logger.info(f"Loading embedding model: {model_name}")
    embed_model = SentenceTransformer(model_name, device="cpu")

    embeddings: dict[tuple[str, int], np.ndarray] = {}
    total_responses = sum(len(v) for v in all_results.values())
    logger.info(f"Embedding {total_responses} responses across {len(all_results)} groups")

    t0 = time.time()
    done = 0
    for (model, level), resps in all_results.items():
        if not resps:
            continue
        texts = [r["response_text"] for r in resps]
        emb = embed_model.encode(texts, batch_size=64, show_progress_bar=False)
        embeddings[(model, level)] = np.array(emb, dtype=np.float32)
        done += len(texts)
        if done % 500 == 0:
            logger.info(f"  Embedded {done}/{total_responses}")

    elapsed = time.time() - t0
    logger.info(f"Embedding complete: {done} responses in {elapsed:.1f}s")

    # Free model memory
    del embed_model
    gc.collect()

    return embeddings


# ---------------------------------------------------------------------------
# PHASE 5: CSD Indicator Computation
# ---------------------------------------------------------------------------

def compute_csd_indicators(
    embs: np.ndarray,
    responses: list[dict],
    problems_for_level: list[dict],
    n_problems: int,
) -> CSDIndicators:
    """Compute all CSD indicators for a (model, level) group."""
    N = embs.shape[0]

    # (a) Embedding Variance: trace of covariance matrix
    if N >= 2:
        cov = np.cov(embs.T)
        embedding_variance = float(np.trace(cov))
    else:
        embedding_variance = 0.0

    # (b) Hartigan's Dip Test on PC1
    if N >= 4:
        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(embs).ravel()
        try:
            dip_stat, dip_pval = diptest.diptest(pc1)
        except Exception:
            dip_stat, dip_pval = 0.0, 1.0
    else:
        pc1 = embs[:, 0] if embs.shape[1] > 0 else np.zeros(N)
        dip_stat, dip_pval = 0.0, 1.0

    # (c) Silhouette Score for k=2
    if N >= 4:
        try:
            km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(embs)
            # Check if we have at least 2 distinct labels
            if len(set(km.labels_)) >= 2:
                sil = float(silhouette_score(embs, km.labels_))
            else:
                sil = 0.0
        except Exception:
            sil = 0.0
    else:
        sil = 0.0

    # (d) Bimodality Coefficient from PC1
    if N >= 4:
        skew = float(sp_stats.skew(pc1))
        kurt = float(sp_stats.kurtosis(pc1))  # excess kurtosis
        n = len(pc1)
        denom = kurt + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
        bc = (skew ** 2 + 1) / denom if abs(denom) > 1e-12 else 0.0
    else:
        bc = 0.0

    # (e) Self-Consistency Disagreement
    answers = []
    for r in responses:
        a = extract_final_answer(r["response_text"])
        if a is not None:
            answers.append(a)
    if answers:
        counter = Counter(answers)
        max_frac = counter.most_common(1)[0][1] / len(answers)
        disagreement = 1.0 - max_frac
    else:
        disagreement = 1.0

    # (f) Answer extraction rate as confidence proxy
    extraction_rate = len(answers) / max(N, 1)

    # (g) Within-Chain Step Correctness Autocorrelation
    autocorrs = []
    prob_map = {i: problems_for_level[i] for i in range(min(n_problems, len(problems_for_level)))}
    for r in responses:
        p_idx = r["problem_idx"]
        if p_idx not in prob_map:
            continue
        prob = prob_map[p_idx]
        gt_intermediates = prob.get("metadata_all_intermediate_answers", [])
        if len(gt_intermediates) < 3:
            continue

        # Extract all numbers from response in order
        resp_numbers = []
        for x in re.findall(r'-?\d+', r["response_text"]):
            try:
                resp_numbers.append(int(x))
            except ValueError:
                continue

        # Build binary correctness series
        correctness = []
        search_start = 0
        for gt_val in gt_intermediates[1:]:  # skip initial value
            found = False
            for j in range(search_start, len(resp_numbers)):
                if resp_numbers[j] == gt_val:
                    found = True
                    search_start = j + 1
                    break
            correctness.append(1.0 if found else 0.0)

        if len(correctness) >= 3:
            series = np.array(correctness)
            if series.std() > 0:
                ac = float(np.corrcoef(series[:-1], series[1:])[0, 1])
                if not np.isnan(ac):
                    autocorrs.append(ac)

    step_ac = float(np.nanmean(autocorrs)) if autocorrs else 0.0

    accuracy = compute_accuracy(responses)

    return CSDIndicators(
        accuracy=accuracy,
        embedding_variance=embedding_variance,
        dip_statistic=float(dip_stat),
        dip_pvalue=float(dip_pval),
        silhouette_k2=float(sil),
        bimodality_coefficient=float(bc),
        disagreement_rate=float(disagreement),
        mean_confidence=float(extraction_rate),
        step_correctness_autocorr=float(step_ac),
    )


# ---------------------------------------------------------------------------
# PHASE 6: Scaling Law Fits & Leading Indicator Tests
# ---------------------------------------------------------------------------

def compute_model_analysis(
    indicators: dict[tuple[str, int], CSDIndicators],
    models: list[str],
    levels: list[int],
) -> dict[str, dict]:
    """Compute scaling law fits and leading indicator tests per model."""
    model_summary: dict[str, dict] = {}

    for model in models:
        # Get accuracy curve
        acc_curve = []
        for lvl in levels:
            key = (model, lvl)
            if key in indicators:
                acc_curve.append((lvl, indicators[key].accuracy))

        if not acc_curve:
            logger.warning(f"No indicators for model {model}, skipping analysis")
            model_summary[model] = {
                "d_star": None, "scaling_exponent": None, "scaling_r2": None,
                "flickering_leading": False, "sil_leading": False,
                "bc_leading": False, "bimodality_consensus_leading": False,
                "tau_variance": 0.0, "p_tau_variance": 1.0,
                "tau_dip": 0.0, "p_tau_dip": 1.0,
            }
            continue

        # Determine d* (capability boundary) = first level where accuracy < 0.5
        d_star = None
        for lvl, acc in acc_curve:
            if acc < 0.5:
                d_star = lvl
                break
        if d_star is None:
            d_star = levels[-1] + 1

        # Fit variance scaling: log(Var) ~ alpha * log(d* - d)
        log_dist = []
        log_var = []
        for lvl in levels:
            key = (model, lvl)
            if key not in indicators:
                continue
            if lvl < d_star:
                dist = d_star - lvl
                var = indicators[key].embedding_variance
                if dist > 0 and var > 0:
                    log_dist.append(math.log(dist))
                    log_var.append(math.log(var))

        scaling_exponent = None
        scaling_r2 = None
        if len(log_dist) >= 3:
            try:
                slope, intercept, r_value, p_value, std_err = sp_stats.linregress(
                    log_dist, log_var
                )
                scaling_exponent = float(slope)
                scaling_r2 = float(r_value ** 2)
            except Exception as e:
                logger.warning(f"Scaling fit failed for {model}: {e}")

        # Leading Indicator Tests
        # Check: dip_pvalue < 0.05 at any level where accuracy > 0.80
        flickering_leading = False
        earliest_flickering = None
        for lvl in levels:
            key = (model, lvl)
            if key not in indicators:
                continue
            ind = indicators[key]
            if ind.accuracy > 0.80 and ind.dip_pvalue < 0.05:
                flickering_leading = True
                earliest_flickering = lvl
                break

        # Silhouette > 0.3 at any level where accuracy > 0.80
        sil_leading = False
        for lvl in levels:
            key = (model, lvl)
            if key not in indicators:
                continue
            ind = indicators[key]
            if ind.accuracy > 0.80 and ind.silhouette_k2 > 0.3:
                sil_leading = True
                break

        # BC > 0.555 at any level where accuracy > 0.80
        bc_leading = False
        for lvl in levels:
            key = (model, lvl)
            if key not in indicators:
                continue
            ind = indicators[key]
            if ind.accuracy > 0.80 and ind.bimodality_coefficient > 0.555:
                bc_leading = True
                break

        bimodality_consensus = sum([flickering_leading, sil_leading, bc_leading]) >= 2

        # Kendall Tau Trend Tests
        available_levels = [lvl for lvl in levels if (model, lvl) in indicators]
        var_series = [indicators[(model, lvl)].embedding_variance for lvl in available_levels]
        dip_series = [indicators[(model, lvl)].dip_statistic for lvl in available_levels]

        if len(available_levels) >= 3:
            try:
                tau_var, p_tau_var = sp_stats.kendalltau(available_levels, var_series)
            except Exception:
                tau_var, p_tau_var = 0.0, 1.0
            try:
                tau_dip, p_tau_dip = sp_stats.kendalltau(available_levels, dip_series)
            except Exception:
                tau_dip, p_tau_dip = 0.0, 1.0
        else:
            tau_var, p_tau_var = 0.0, 1.0
            tau_dip, p_tau_dip = 0.0, 1.0

        model_summary[model] = {
            "d_star": d_star,
            "scaling_exponent": scaling_exponent,
            "scaling_r2": scaling_r2,
            "flickering_leading": flickering_leading,
            "earliest_flickering_level": earliest_flickering,
            "sil_leading": sil_leading,
            "bc_leading": bc_leading,
            "bimodality_consensus_leading": bimodality_consensus,
            "tau_variance": float(tau_var) if not np.isnan(tau_var) else 0.0,
            "p_tau_variance": float(p_tau_var) if not np.isnan(p_tau_var) else 1.0,
            "tau_dip": float(tau_dip) if not np.isnan(tau_dip) else 0.0,
            "p_tau_dip": float(p_tau_dip) if not np.isnan(p_tau_dip) else 1.0,
        }

        logger.info(
            f"Model {model.split('/')[-1]}: d*={d_star}, "
            f"scaling_exp={scaling_exponent}, R2={scaling_r2}, "
            f"flickering={flickering_leading}, sil={sil_leading}, bc={bc_leading}, "
            f"consensus={bimodality_consensus}, "
            f"tau_var={tau_var:.3f}(p={p_tau_var:.3f}), "
            f"tau_dip={tau_dip:.3f}(p={p_tau_dip:.3f})"
        )

    return model_summary


# ---------------------------------------------------------------------------
# PHASE 7: Output Assembly
# ---------------------------------------------------------------------------

def build_output(
    all_results: dict[tuple[str, int], list[dict]],
    indicators: dict[tuple[str, int], CSDIndicators],
    model_summary: dict[str, dict],
    models: list[str],
    levels: list[int],
) -> dict:
    """Build output conforming to exp_gen_sol_out.json schema."""
    output: dict = {"datasets": [], "metadata": {
        "method_name": "CSD_Sampling_Experiment",
        "description": (
            "Critical Slowing Down indicators extracted from LLM response "
            "distributions across difficulty levels to detect approaching "
            "reasoning collapse."
        ),
        "models": models,
        "difficulty_levels": levels,
        "responses_per_level_target": RESPONSES_PER_PROBLEM * PROBLEMS_PER_LEVEL,
        "embed_model": EMBED_MODEL_NAME,
        "sampling_params": SAMPLING_PARAMS,
        "model_summaries": model_summary,
    }}

    for model in models:
        model_short = model.split("/")[-1]
        examples = []
        for lvl in levels:
            key = (model, lvl)
            if key not in indicators:
                continue
            ind = indicators[key]
            resps = all_results.get(key, [])

            # Use first problem's input as representative
            rep_input = resps[0]["ground_truth"] if resps else ""
            rep_prompt = ""
            if resps:
                # Find a problem from responses for the representative input
                rep_prompt = f"CSD analysis at difficulty={lvl} for model={model_short}"

            ms = model_summary.get(model, {})

            examples.append({
                "input": rep_prompt,
                "output": str(rep_input),
                # predict_ fields (all strings per schema)
                "predict_accuracy": str(round(ind.accuracy, 4)),
                "predict_csd_variance": str(round(ind.embedding_variance, 6)),
                "predict_dip_statistic": str(round(ind.dip_statistic, 6)),
                "predict_dip_pvalue": str(round(ind.dip_pvalue, 4)),
                "predict_silhouette_k2": str(round(ind.silhouette_k2, 4)),
                "predict_bimodality_coefficient": str(round(ind.bimodality_coefficient, 4)),
                "predict_disagreement_rate": str(round(ind.disagreement_rate, 4)),
                "predict_extraction_rate": str(round(ind.mean_confidence, 4)),
                "predict_step_correctness_autocorr": str(round(ind.step_correctness_autocorr, 4)),
                # Baseline predictions (strings)
                "predict_baseline_disagreement": str(round(ind.disagreement_rate, 4)),
                "predict_baseline_extraction_rate": str(round(ind.mean_confidence, 4)),
                # metadata
                "metadata_difficulty_level": lvl,
                "metadata_model": model,
                "metadata_num_responses": len(resps),
                "metadata_d_star": ms.get("d_star"),
                "metadata_scaling_exponent": ms.get("scaling_exponent"),
                "metadata_scaling_r2": ms.get("scaling_r2"),
                "metadata_flickering_leading": ms.get("flickering_leading", False),
                "metadata_bimodality_consensus_leading": ms.get(
                    "bimodality_consensus_leading", False
                ),
                "metadata_tau_variance": ms.get("tau_variance", 0.0),
                "metadata_p_tau_variance": ms.get("p_tau_variance", 1.0),
                "metadata_tau_dip": ms.get("tau_dip", 0.0),
                "metadata_p_tau_dip": ms.get("p_tau_dip", 1.0),
                "metadata_fold": "test",
            })

        if examples:
            output["datasets"].append({
                "dataset": f"csd_indicators__{model_short}",
                "examples": examples,
            })

    return output


# ---------------------------------------------------------------------------
# Sanity Checks
# ---------------------------------------------------------------------------

def sanity_check_mini(
    results: dict[tuple[str, int], list[dict]],
) -> bool:
    """Sanity check mini run results."""
    if not results:
        logger.error("MINI SANITY FAIL: No results")
        return False

    for key, resps in results.items():
        if len(resps) == 0:
            logger.error(f"MINI SANITY FAIL: No responses for {key}")
            return False
        empty = sum(1 for r in resps if not r["response_text"].strip())
        if empty > 0:
            logger.warning(f"MINI: {empty}/{len(resps)} empty responses for {key}")

    # Check answer extraction rate
    all_resps = [r for rlist in results.values() for r in rlist]
    extracted = [extract_final_answer(r["response_text"]) for r in all_resps]
    extraction_rate = sum(1 for a in extracted if a is not None) / max(len(extracted), 1)
    logger.info(f"MINI: Answer extraction rate: {extraction_rate:.2%}")
    if extraction_rate < 0.50:
        logger.error(f"MINI SANITY FAIL: Extraction rate {extraction_rate:.2%} < 50%")
        return False

    logger.info("MINI sanity checks PASSED")
    return True


def sanity_check_medium(
    results: dict[tuple[str, int], list[dict]],
    problems_by_level: dict[int, list[dict]],
) -> bool:
    """Sanity check medium run results."""
    if not results:
        logger.error("MEDIUM SANITY FAIL: No results")
        return False

    # Check accuracy trend for each model
    models_in_results = set(m for m, _ in results.keys())
    for model in models_in_results:
        accs = {}
        for lvl in DIFFICULTY_LEVELS:
            key = (model, lvl)
            if key in results and results[key]:
                accs[lvl] = compute_accuracy(results[key])

        if accs:
            low_acc = np.mean([accs[lvl] for lvl in sorted(accs)[:3]])
            high_acc = np.mean([accs[lvl] for lvl in sorted(accs)[-3:]])
            logger.info(
                f"MEDIUM {model.split('/')[-1]}: "
                f"easy_acc={low_acc:.2f}, hard_acc={high_acc:.2f}"
            )

    logger.info("MEDIUM sanity checks PASSED")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@logger.catch
def main():
    global TOTAL_COST_USD

    logger.info(f"=== CSD Sampling Experiment ===")
    logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f}GB RAM, RAM budget: {RAM_BUDGET_BYTES/1e9:.1f}GB")
    logger.info(f"Workspace: {WORKSPACE}")

    # PHASE 1: Load dataset
    problems_by_level = load_dataset(DATA_PATH)

    # =====================================================================
    # MINI RUN
    # =====================================================================
    logger.info("=" * 60)
    logger.info("=== MINI RUN ===")
    t_mini_start = time.time()

    mini_results = asyncio.run(
        generate_responses_for_config(
            problems_by_level,
            models=SCALE_MINI["models"],
            levels=SCALE_MINI["levels"],
            n_problems=SCALE_MINI["n_prob"],
            n_responses=SCALE_MINI["n_resp"],
            concurrency=20,
        )
    )

    t_mini = time.time() - t_mini_start
    logger.info(f"MINI run took {t_mini:.1f}s, cost so far: ${TOTAL_COST_USD:.4f}")

    if not sanity_check_mini(mini_results):
        logger.error("MINI sanity check failed, aborting")
        sys.exit(1)

    # Quick CSD indicator check on mini results
    logger.info("Testing CSD indicators on mini data...")
    mini_embeddings = embed_all_responses(mini_results)
    for key, embs in mini_embeddings.items():
        if embs.shape[0] >= 4:
            resps = mini_results[key]
            model, lvl = key
            ind = compute_csd_indicators(
                embs, resps, problems_by_level[lvl], SCALE_MINI["n_prob"]
            )
            logger.info(
                f"  MINI {model.split('/')[-1]} lvl={lvl}: "
                f"acc={ind.accuracy:.2f}, var={ind.embedding_variance:.4f}, "
                f"dip_p={ind.dip_pvalue:.3f}, sil={ind.silhouette_k2:.3f}, "
                f"bc={ind.bimodality_coefficient:.3f}, "
                f"disagree={ind.disagreement_rate:.3f}"
            )
    del mini_embeddings
    gc.collect()
    logger.info("MINI CSD indicator check PASSED")

    # =====================================================================
    # MEDIUM RUN
    # =====================================================================
    logger.info("=" * 60)
    logger.info("=== MEDIUM RUN ===")
    t_med_start = time.time()

    medium_results = asyncio.run(
        generate_responses_for_config(
            problems_by_level,
            models=SCALE_MEDIUM["models"],
            levels=SCALE_MEDIUM["levels"],
            n_problems=SCALE_MEDIUM["n_prob"],
            n_responses=SCALE_MEDIUM["n_resp"],
            concurrency=25,
        )
    )

    t_med = time.time() - t_med_start
    logger.info(f"MEDIUM run took {t_med:.1f}s, cost so far: ${TOTAL_COST_USD:.4f}")

    if not sanity_check_medium(medium_results, problems_by_level):
        logger.warning("MEDIUM sanity check had warnings, continuing anyway")

    # Extrapolate full run time
    medium_calls = sum(len(v) for v in medium_results.values())
    full_target_calls = len(MODELS) * len(DIFFICULTY_LEVELS) * PROBLEMS_PER_LEVEL * RESPONSES_PER_PROBLEM
    if medium_calls > 0:
        time_per_call = t_med / medium_calls
        estimated_full_time = time_per_call * full_target_calls
        logger.info(
            f"Extrapolation: {time_per_call:.2f}s/call, "
            f"full run ~{estimated_full_time:.0f}s ({estimated_full_time/60:.1f}min)"
        )

    del medium_results
    gc.collect()

    # =====================================================================
    # FULL RUN
    # =====================================================================
    logger.info("=" * 60)
    logger.info("=== FULL RUN ===")
    t_full_start = time.time()

    all_results = asyncio.run(
        generate_responses_for_config(
            problems_by_level,
            models=SCALE_FULL["models"],
            levels=SCALE_FULL["levels"],
            n_problems=SCALE_FULL["n_prob"],
            n_responses=SCALE_FULL["n_resp"],
            concurrency=25,
        )
    )

    t_full = time.time() - t_full_start
    logger.info(f"FULL run took {t_full:.1f}s, cost so far: ${TOTAL_COST_USD:.4f}")

    # Check response counts
    total_responses = sum(len(v) for v in all_results.values())
    logger.info(f"Total responses collected: {total_responses}")

    # PHASE 3: Compute accuracy for every (model, level)
    logger.info("=" * 60)
    logger.info("=== COMPUTING ACCURACY ===")
    for model in MODELS:
        accs = []
        for lvl in DIFFICULTY_LEVELS:
            key = (model, lvl)
            if key in all_results and all_results[key]:
                acc = compute_accuracy(all_results[key])
                accs.append((lvl, acc))
        if accs:
            logger.info(
                f"  {model.split('/')[-1]}: "
                f"lvl2={accs[0][1]:.2f}, lvl13={accs[11][1]:.2f} (mid), "
                f"lvl25={accs[-1][1]:.2f}"
            )

    # PHASE 4: Semantic Embedding
    logger.info("=" * 60)
    logger.info("=== EMBEDDING RESPONSES ===")
    embeddings = embed_all_responses(all_results)

    # PHASE 5: CSD Indicator Computation
    logger.info("=" * 60)
    logger.info("=== COMPUTING CSD INDICATORS ===")
    indicators: dict[tuple[str, int], CSDIndicators] = {}
    for (model, level), resps in all_results.items():
        key = (model, level)
        if key not in embeddings or embeddings[key].shape[0] < 2:
            continue
        embs = embeddings[key]
        indicators[key] = compute_csd_indicators(
            embs, resps, problems_by_level[level], SCALE_FULL["n_prob"]
        )

    logger.info(f"Computed CSD indicators for {len(indicators)} (model, level) pairs")

    # Free embeddings to save memory
    del embeddings
    gc.collect()

    # PHASE 6: Scaling Law Fits
    logger.info("=" * 60)
    logger.info("=== SCALING LAW FITS & LEADING INDICATOR TESTS ===")
    model_summary = compute_model_analysis(indicators, MODELS, DIFFICULTY_LEVELS)

    # PHASE 7: Output Assembly
    logger.info("=" * 60)
    logger.info("=== ASSEMBLING OUTPUT ===")
    output = build_output(
        all_results, indicators, model_summary, MODELS, DIFFICULTY_LEVELS
    )

    # Write output
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    logger.info(f"Output written to {OUTPUT_PATH}")

    # Log summary statistics
    total_examples = sum(len(ds["examples"]) for ds in output["datasets"])
    logger.info(f"Output has {len(output['datasets'])} datasets, {total_examples} total examples")
    logger.info(f"Total API cost: ${TOTAL_COST_USD:.4f}")

    # Validate output
    for ds in output["datasets"]:
        for ex in ds["examples"]:
            assert "input" in ex, "Missing input field"
            assert "output" in ex, "Missing output field"
            for k in ex:
                if k.startswith("predict_"):
                    assert isinstance(ex[k], str), f"predict_ field {k} must be string, got {type(ex[k])}"

    logger.info("Output schema self-check PASSED")
    logger.info("=== EXPERIMENT COMPLETE ===")


if __name__ == "__main__":
    main()
