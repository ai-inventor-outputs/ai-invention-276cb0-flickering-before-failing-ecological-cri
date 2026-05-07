#!/usr/bin/env python3
"""Temperature-Dependent CSD Flickering Experiment.

Tests the CSD theory prediction that higher sampling temperature (noise intensity)
produces earlier and stronger flickering signals in LLM response distributions.
Uses gemini-2.0-flash-001 on the arithmetic chain dataset across 4 temperature
settings (T=0.4, 0.7, 1.0, 1.3). Compares d_lead (onset of flickering) and
d* (capability boundary) across temperatures for causal evidence of CSD dynamics.
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
TOTAL_RAM_GB = _container_ram_gb() or 29.0

# Set memory limit at 80% of container RAM
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

# Ability server for OpenRouter calls
ABILITY_SERVER_URL = os.environ.get(
    "ABILITY_SERVICE_URL",
    f"http://{os.environ.get('ABILITY_SERVICE_HOST', 'localhost')}:"
    f"{os.environ.get('ABILITY_SERVICE_PORT', '9100')}",
)
ABILITY_ENDPOINT = "aii_openrouter__call"

MODEL = "google/gemini-2.0-flash-001"
TEMPERATURES = [0.4, 0.7, 1.0, 1.3]
DIFFICULTY_LEVELS = list(range(2, 26))  # 24 levels: 2..25
PROBLEMS_PER_LEVEL = 5
RESPONSES_PER_PROBLEM = 10
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# Gradual scaling configs
SCALE_MINI = {
    "temps": [0.4, 1.0],
    "levels": [2, 5, 8, 11, 14, 17, 20, 23],
    "n_problems": 2,
    "n_responses": 5,
    # Total: 2 x 8 x 2 x 5 = 160 calls
}
SCALE_MEDIUM = {
    "temps": [0.4, 0.7, 1.0, 1.3],
    "levels": [2, 4, 6, 8, 10, 12, 13, 14, 15, 16, 17, 18, 20, 22, 24, 25],
    "n_problems": 5,
    "n_responses": 6,
    # Total: 4 x 16 x 5 x 6 = 1920 calls
}
SCALE_FULL = {
    "temps": TEMPERATURES,
    "levels": DIFFICULTY_LEVELS,
    "n_problems": PROBLEMS_PER_LEVEL,
    "n_responses": RESPONSES_PER_PROBLEM,
    # Total: 4 x 24 x 5 x 10 = 4800 calls
}

# Cost tracking
TOTAL_COST_USD = 0.0
COST_LIMIT_USD = 9.50  # hard stop well below $10

# Common sampling params (temperature varies per call)
BASE_SAMPLING_PARAMS = {"top_p": 0.95, "max_tokens": 2048}


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
    extraction_rate: float
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
# PHASE 2: LLM Response Generation (temperature-parameterized)
# ---------------------------------------------------------------------------

async def call_llm(
    client: httpx.AsyncClient,
    model: str,
    prompt: str,
    temperature: float,
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
        "max_tokens": BASE_SAMPLING_PARAMS["max_tokens"],
        "temperature": temperature,
        "top_p": BASE_SAMPLING_PARAMS["top_p"],
    }

    last_err = None
    for attempt in range(max_retries):
        try:
            resp = await client.post(url, json=payload, timeout=180.0)
            resp.raise_for_status()
            data = resp.json()

            if not data.get("success"):
                err_msg = data.get("error", "Unknown error")
                if attempt < max_retries - 1 and (
                    "rate" in err_msg.lower()
                    or "timeout" in err_msg.lower()
                    or "overloaded" in err_msg.lower()
                    or "429" in err_msg
                    or "503" in err_msg
                ):
                    wait = 2 ** (attempt + 1) + np.random.uniform(0, 1)
                    logger.debug(f"Retryable error: {err_msg[:100]}, waiting {wait:.1f}s")
                    await asyncio.sleep(wait)
                    continue
                raise RuntimeError(f"Ability server error: {err_msg}")

            text = data.get("response", "")
            in_tok = data.get("input_tokens", 0)
            out_tok = data.get("output_tokens", 0)
            # gemini-2.0-flash pricing: ~$0.10/M input, ~$0.40/M output
            cost = in_tok * 0.10 / 1_000_000 + out_tok * 0.40 / 1_000_000
            TOTAL_COST_USD += cost
            return text, cost

        except httpx.HTTPStatusError as e:
            last_err = e
            if e.response.status_code in (429, 502, 503, 504):
                wait = 2 ** (attempt + 1) + np.random.uniform(0, 1)
                logger.debug(f"HTTP {e.response.status_code}, retry in {wait:.1f}s")
                await asyncio.sleep(wait)
                continue
            raise
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as e:
            last_err = e
            wait = 2 ** (attempt + 1) + np.random.uniform(0, 1)
            logger.debug(f"Connection error: {e}, retry in {wait:.1f}s")
            await asyncio.sleep(wait)
            continue

    raise RuntimeError(f"All {max_retries} retries failed: {last_err}")


async def generate_responses_for_temp_config(
    problems_by_level: dict[int, list[dict]],
    temps: list[float],
    levels: list[int],
    n_problems: int,
    n_responses: int,
    concurrency: int = 25,
) -> dict[tuple[float, int], list[dict]]:
    """Generate responses for each (temperature, difficulty_level) combination.

    Returns {(temperature, difficulty_level): [response_dicts]}.
    Each response_dict has: problem_idx, response_idx, response_text, ground_truth.
    """
    global TOTAL_COST_USD

    results: dict[tuple[float, int], list[dict]] = {}
    total_calls = len(temps) * len(levels) * n_problems * n_responses
    logger.info(
        f"Generating responses: {len(temps)} temps x {len(levels)} levels x "
        f"{n_problems} problems x {n_responses} responses = {total_calls} API calls"
    )

    sem = asyncio.Semaphore(concurrency)
    completed = 0
    failed = 0
    t0 = time.time()

    async def bounded_call(
        client: httpx.AsyncClient,
        temp: float,
        level: int,
        p_idx: int,
        r_idx: int,
        prompt: str,
        gt: str,
    ) -> tuple[float, int, int, int, str, str, bool]:
        nonlocal completed, failed
        async with sem:
            try:
                text, cost = await call_llm(client, MODEL, prompt, temp)
                completed += 1
                if completed % 50 == 0:
                    elapsed = time.time() - t0
                    rate = completed / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"  Progress: {completed}/{total_calls} "
                        f"({rate:.1f}/s, ${TOTAL_COST_USD:.4f})"
                    )
                return temp, level, p_idx, r_idx, text, gt, True
            except Exception as e:
                failed += 1
                logger.warning(
                    f"  Failed call T={temp} lvl={level} p={p_idx} r={r_idx}: "
                    f"{type(e).__name__}: {str(e)[:100]}"
                )
                return temp, level, p_idx, r_idx, "", gt, False

    async with httpx.AsyncClient() as client:
        tasks = []
        for temp in temps:
            for level in levels:
                # Select SAME problems for ALL temperatures (deterministic)
                probs = problems_by_level[level][:n_problems]
                for p_idx, prob in enumerate(probs):
                    for r_idx in range(n_responses):
                        tasks.append(
                            bounded_call(
                                client, temp, level, p_idx, r_idx,
                                prob["input"], prob["output"],
                            )
                        )

        raw_results = await asyncio.gather(*tasks)

    # Organize results
    for temp, level, p_idx, r_idx, text, gt, success in raw_results:
        key = (temp, level)
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
    all_results: dict[tuple[float, int], list[dict]],
    model_name: str = EMBED_MODEL_NAME,
) -> dict[tuple[float, int], np.ndarray]:
    """Embed all responses using sentence-transformers."""
    logger.info(f"Loading embedding model: {model_name}")
    embed_model = SentenceTransformer(model_name, device="cpu")

    embeddings: dict[tuple[float, int], np.ndarray] = {}
    total_responses = sum(len(v) for v in all_results.values())
    logger.info(f"Embedding {total_responses} responses across {len(all_results)} groups")

    t0 = time.time()
    done = 0
    for (temp, level), resps in all_results.items():
        if not resps:
            continue
        texts = [r["response_text"] for r in resps]
        emb = embed_model.encode(texts, batch_size=64, show_progress_bar=False)
        embeddings[(temp, level)] = np.array(emb, dtype=np.float32)
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
    """Compute all CSD indicators for a (temp, level) group."""
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

    # (f) Answer extraction rate
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
        extraction_rate=float(extraction_rate),
        step_correctness_autocorr=float(step_ac),
    )


# ---------------------------------------------------------------------------
# PHASE 6: Per-Temperature Analysis (core of this experiment)
# ---------------------------------------------------------------------------

def compute_per_temperature_analysis(
    indicators: dict[tuple[float, int], CSDIndicators],
    temps: list[float],
    levels: list[int],
) -> dict[str, dict]:
    """For each temperature, compute d* and d_lead."""
    temp_results: dict[str, dict] = {}

    for T in temps:
        # 6a. Accuracy curve and d* determination
        acc_curve = [
            (lvl, indicators[(T, lvl)].accuracy)
            for lvl in levels
            if (T, lvl) in indicators
        ]

        # d* = first level where accuracy < 0.50
        d_star = None
        for lvl, acc in acc_curve:
            if acc < 0.50:
                d_star = lvl
                break
        if d_star is None:
            d_star = levels[-1] + 1  # never dropped below 50%

        # 6b. d_lead: earliest level where flickering is significant
        max_acc = max((acc for _, acc in acc_curve), default=0)
        acc_threshold_relaxed = max(0.6, max_acc * 0.7)

        d_lead_strict = None    # dip p<0.05 AND acc > 0.80
        d_lead_relaxed = None   # dip p<0.05 AND acc > max_acc*0.7

        for lvl in levels:
            key = (T, lvl)
            if key not in indicators:
                continue
            ind = indicators[key]
            if ind.dip_pvalue < 0.05:
                if d_lead_relaxed is None and ind.accuracy > acc_threshold_relaxed:
                    d_lead_relaxed = lvl
                if d_lead_strict is None and ind.accuracy > 0.80:
                    d_lead_strict = lvl

        # Also check silhouette-based d_lead (sil > 0.3)
        d_lead_sil = None
        for lvl in levels:
            key = (T, lvl)
            if key not in indicators:
                continue
            ind = indicators[key]
            if ind.silhouette_k2 > 0.3 and ind.accuracy > acc_threshold_relaxed:
                d_lead_sil = lvl
                break

        # 6c. Compute lead time
        lead_time_strict = (d_star - d_lead_strict) if (d_star and d_lead_strict) else None
        lead_time_relaxed = (d_star - d_lead_relaxed) if (d_star and d_lead_relaxed) else None

        # 6d. Variance scaling fit: log(Var) ~ alpha * log(d* - d)
        log_dist, log_var = [], []
        for lvl in levels:
            key = (T, lvl)
            if key not in indicators or lvl >= d_star:
                continue
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
                logger.warning(f"Scaling fit failed for T={T}: {e}")

        # 6e. Kendall tau trends for CSD indicators
        avail_levels = [l for l in levels if (T, l) in indicators]
        var_series = [indicators[(T, l)].embedding_variance for l in avail_levels]
        dip_series = [indicators[(T, l)].dip_statistic for l in avail_levels]
        sil_series = [indicators[(T, l)].silhouette_k2 for l in avail_levels]
        disagree_series = [indicators[(T, l)].disagreement_rate for l in avail_levels]

        def safe_kendalltau(x, y):
            if len(x) < 3:
                return 0.0, 1.0
            try:
                t, p = sp_stats.kendalltau(x, y)
                return (float(t) if not np.isnan(t) else 0.0,
                        float(p) if not np.isnan(p) else 1.0)
            except Exception:
                return 0.0, 1.0

        tau_var, p_tau_var = safe_kendalltau(avail_levels, var_series)
        tau_dip, p_tau_dip = safe_kendalltau(avail_levels, dip_series)
        tau_sil, p_tau_sil = safe_kendalltau(avail_levels, sil_series)
        tau_disagree, p_tau_disagree = safe_kendalltau(avail_levels, disagree_series)

        # 6f. Bimodality consensus at each level (2/3 of dip, sil, BC agree)
        bimodal_levels = []
        for lvl in levels:
            key = (T, lvl)
            if key not in indicators:
                continue
            ind = indicators[key]
            votes = sum([
                ind.dip_pvalue < 0.05,
                ind.silhouette_k2 > 0.3,
                ind.bimodality_coefficient > 0.555,
            ])
            if votes >= 2:
                bimodal_levels.append(lvl)

        temp_results[str(T)] = {
            "temperature": T,
            "d_star": d_star,
            "d_lead_strict": d_lead_strict,
            "d_lead_relaxed": d_lead_relaxed,
            "d_lead_sil": d_lead_sil,
            "lead_time_strict": lead_time_strict,
            "lead_time_relaxed": lead_time_relaxed,
            "scaling_exponent": scaling_exponent,
            "scaling_r2": scaling_r2,
            "tau_variance": tau_var, "p_tau_variance": p_tau_var,
            "tau_dip": tau_dip, "p_tau_dip": p_tau_dip,
            "tau_silhouette": tau_sil, "p_tau_silhouette": p_tau_sil,
            "tau_disagreement": tau_disagree, "p_tau_disagreement": p_tau_disagree,
            "bimodal_levels": bimodal_levels,
            "num_bimodal_levels": len(bimodal_levels),
            "max_accuracy": max_acc,
            "accuracy_curve": [(lvl, acc) for lvl, acc in acc_curve],
        }

        logger.info(
            f"T={T}: d*={d_star}, d_lead_r={d_lead_relaxed}, "
            f"lead_time_r={lead_time_relaxed}, "
            f"scaling_exp={scaling_exponent}, R2={scaling_r2}, "
            f"tau_var={tau_var:.3f}(p={p_tau_var:.3f}), "
            f"tau_dip={tau_dip:.3f}(p={p_tau_dip:.3f}), "
            f"bimodal_lvls={len(bimodal_levels)}, max_acc={max_acc:.2f}"
        )

    return temp_results


# ---------------------------------------------------------------------------
# PHASE 7: Cross-Temperature Dose-Response Analysis (key novelty)
# ---------------------------------------------------------------------------

def compute_dose_response_analysis(
    temp_results: dict[str, dict],
    indicators: dict[tuple[float, int], CSDIndicators],
    temps: list[float],
    levels: list[int],
) -> dict:
    """Test CSD predictions across temperature settings."""

    # 7a. D* STABILITY TEST
    d_stars = [temp_results[str(T)]["d_star"] for T in temps]
    d_star_stable = (max(d_stars) - min(d_stars)) <= 3
    d_star_mean = float(np.mean(d_stars))
    d_star_std = float(np.std(d_stars))

    # 7b. D_LEAD DOSE-RESPONSE
    lead_times_relaxed = [
        (T, temp_results[str(T)]["lead_time_relaxed"])
        for T in temps
        if temp_results[str(T)]["lead_time_relaxed"] is not None
    ]

    slope_lt, r_lt, p_lt = None, None, None
    dose_response_significant = False
    if len(lead_times_relaxed) >= 3:
        T_vals = [lt[0] for lt in lead_times_relaxed]
        lt_vals = [lt[1] for lt in lead_times_relaxed]
        try:
            slope_lt, intercept_lt, r_lt, p_lt, se_lt = sp_stats.linregress(T_vals, lt_vals)
            slope_lt = float(slope_lt)
            r_lt = float(r_lt)
            p_lt = float(p_lt)
            dose_response_significant = p_lt < 0.10 and slope_lt > 0
        except Exception as e:
            logger.warning(f"Lead time regression failed: {e}")

    # 7c. DIP STATISTIC MAGNITUDE AT MATCHED LEVELS
    dip_by_level: dict[int, dict] = {}
    for lvl in levels:
        dip_vals = []
        t_vals = []
        for T in temps:
            key = (T, lvl)
            if key in indicators:
                dip_vals.append(indicators[key].dip_statistic)
                t_vals.append(T)
        if len(t_vals) >= 3:
            try:
                rho, p_rho = sp_stats.spearmanr(t_vals, dip_vals)
                dip_by_level[lvl] = {
                    "spearman_rho": float(rho) if not np.isnan(rho) else 0.0,
                    "p_value": float(p_rho) if not np.isnan(p_rho) else 1.0,
                }
            except Exception:
                dip_by_level[lvl] = {"spearman_rho": 0.0, "p_value": 1.0}

    n_positive_dip = sum(1 for v in dip_by_level.values() if v["spearman_rho"] > 0)
    frac_positive_dip_trend = n_positive_dip / max(len(dip_by_level), 1)

    # 7d. VARIANCE MAGNITUDE AT MATCHED LEVELS
    var_by_level: dict[int, dict] = {}
    for lvl in levels:
        var_vals = []
        t_vals = []
        for T in temps:
            key = (T, lvl)
            if key in indicators:
                var_vals.append(indicators[key].embedding_variance)
                t_vals.append(T)
        if len(t_vals) >= 3:
            try:
                rho, p_rho = sp_stats.spearmanr(t_vals, var_vals)
                var_by_level[lvl] = {
                    "spearman_rho": float(rho) if not np.isnan(rho) else 0.0,
                    "p_value": float(p_rho) if not np.isnan(p_rho) else 1.0,
                }
            except Exception:
                var_by_level[lvl] = {"spearman_rho": 0.0, "p_value": 1.0}

    n_positive_var = sum(1 for v in var_by_level.values() if v["spearman_rho"] > 0)
    frac_positive_var_trend = n_positive_var / max(len(var_by_level), 1)

    # 7e. NUMBER OF BIMODAL LEVELS COMPARISON
    bimodal_counts = [temp_results[str(T)]["num_bimodal_levels"] for T in temps]
    rho_bimodal, p_bimodal = 0.0, 1.0
    if len(temps) >= 3:
        try:
            rho_bimodal, p_bimodal = sp_stats.spearmanr(temps, bimodal_counts)
            rho_bimodal = float(rho_bimodal) if not np.isnan(rho_bimodal) else 0.0
            p_bimodal = float(p_bimodal) if not np.isnan(p_bimodal) else 1.0
        except Exception:
            pass

    # 7f. DISAGREEMENT RATE MAGNITUDE AT MATCHED LEVELS
    disagree_by_level: dict[int, dict] = {}
    for lvl in levels:
        disagree_vals = []
        t_vals = []
        for T in temps:
            key = (T, lvl)
            if key in indicators:
                disagree_vals.append(indicators[key].disagreement_rate)
                t_vals.append(T)
        if len(t_vals) >= 3:
            try:
                rho, p_rho = sp_stats.spearmanr(t_vals, disagree_vals)
                disagree_by_level[lvl] = {
                    "spearman_rho": float(rho) if not np.isnan(rho) else 0.0,
                    "p_value": float(p_rho) if not np.isnan(p_rho) else 1.0,
                }
            except Exception:
                disagree_by_level[lvl] = {"spearman_rho": 0.0, "p_value": 1.0}

    n_positive_disagree = sum(1 for v in disagree_by_level.values() if v["spearman_rho"] > 0)
    frac_positive_disagree_trend = n_positive_disagree / max(len(disagree_by_level), 1)

    # 7g. OVERALL CSD EVIDENCE SCORE
    evidence_checks = {
        "d_star_stable": d_star_stable,
        "lead_time_increases_with_T": dose_response_significant,
        "dip_increases_with_T_majority": frac_positive_dip_trend > 0.5,
        "variance_increases_with_T_majority": frac_positive_var_trend > 0.5,
        "bimodal_zone_widens_with_T": rho_bimodal > 0 and p_bimodal < 0.10,
        "disagreement_increases_with_T_majority": frac_positive_disagree_trend > 0.5,
    }
    csd_evidence_score = sum(evidence_checks.values()) / len(evidence_checks)

    logger.info(f"CSD Evidence Score: {csd_evidence_score:.2f}")
    for check, result in evidence_checks.items():
        logger.info(f"  {check}: {'PASS' if result else 'FAIL'}")

    return {
        "d_star_analysis": {
            "d_stars_by_temp": {str(T): d_stars[i] for i, T in enumerate(temps)},
            "d_star_stable": d_star_stable,
            "d_star_mean": d_star_mean,
            "d_star_std": d_star_std,
        },
        "lead_time_dose_response": {
            "slope": slope_lt, "r_value": r_lt, "p_value": p_lt,
            "significant": dose_response_significant,
            "data_points": [(t, lt) for t, lt in lead_times_relaxed] if lead_times_relaxed else [],
        },
        "dip_temperature_effect": {
            "per_level": {str(k): v for k, v in dip_by_level.items()},
            "frac_positive_trend": frac_positive_dip_trend,
        },
        "variance_temperature_effect": {
            "per_level": {str(k): v for k, v in var_by_level.items()},
            "frac_positive_trend": frac_positive_var_trend,
        },
        "disagreement_temperature_effect": {
            "per_level": {str(k): v for k, v in disagree_by_level.items()},
            "frac_positive_trend": frac_positive_disagree_trend,
        },
        "bimodal_zone_widening": {
            "counts_by_temp": {str(T): bimodal_counts[i] for i, T in enumerate(temps)},
            "spearman_rho": rho_bimodal, "p_value": p_bimodal,
        },
        "evidence_checks": evidence_checks,
        "csd_evidence_score": csd_evidence_score,
    }


# ---------------------------------------------------------------------------
# PHASE 8: Output Assembly
# ---------------------------------------------------------------------------

def build_output(
    all_results: dict[tuple[float, int], list[dict]],
    indicators: dict[tuple[float, int], CSDIndicators],
    temp_analysis: dict[str, dict],
    dose_response: dict,
    temps: list[float],
    levels: list[int],
    total_cost: float,
) -> dict:
    """Build output conforming to exp_gen_sol_out.json schema."""
    output: dict = {
        "datasets": [],
        "metadata": {
            "method_name": "CSD_Temperature_Manipulation_Experiment",
            "description": (
                "Tests CSD prediction that higher temperature produces earlier "
                "flickering signals in LLM response distributions. Compares "
                "d_lead (onset) and d* (boundary) across 4 temperature settings."
            ),
            "model": MODEL,
            "temperatures": temps,
            "difficulty_levels": levels,
            "responses_per_level_target": RESPONSES_PER_PROBLEM * PROBLEMS_PER_LEVEL,
            "embed_model": EMBED_MODEL_NAME,
            "sampling_params": BASE_SAMPLING_PARAMS,
            "per_temperature_analysis": temp_analysis,
            "dose_response_analysis": dose_response,
            "total_cost_usd": total_cost,
        },
    }

    for T in temps:
        ta = temp_analysis.get(str(T), {})
        examples = []
        for lvl in levels:
            key = (T, lvl)
            if key not in indicators:
                continue
            ind = indicators[key]
            resps = all_results.get(key, [])

            # Representative input/output
            rep_gt = resps[0]["ground_truth"] if resps else ""
            rep_prompt = f"CSD temp analysis at T={T} d={lvl} for {MODEL.split('/')[-1]}"

            examples.append({
                "input": rep_prompt,
                "output": str(rep_gt),
                # predict_ fields (all strings per schema)
                "predict_accuracy": str(round(ind.accuracy, 4)),
                "predict_csd_variance": str(round(ind.embedding_variance, 6)),
                "predict_dip_statistic": str(round(ind.dip_statistic, 6)),
                "predict_dip_pvalue": str(round(ind.dip_pvalue, 4)),
                "predict_silhouette_k2": str(round(ind.silhouette_k2, 4)),
                "predict_bimodality_coefficient": str(round(ind.bimodality_coefficient, 4)),
                "predict_disagreement_rate": str(round(ind.disagreement_rate, 4)),
                "predict_extraction_rate": str(round(ind.extraction_rate, 4)),
                "predict_step_correctness_autocorr": str(round(ind.step_correctness_autocorr, 4)),
                # Baseline predictions (strings)
                "predict_baseline_disagreement": str(round(ind.disagreement_rate, 4)),
                "predict_baseline_extraction_rate": str(round(ind.extraction_rate, 4)),
                # metadata
                "metadata_difficulty_level": lvl,
                "metadata_temperature": T,
                "metadata_model": MODEL,
                "metadata_num_responses": len(resps),
                "metadata_d_star": ta.get("d_star"),
                "metadata_d_lead_relaxed": ta.get("d_lead_relaxed"),
                "metadata_lead_time_relaxed": ta.get("lead_time_relaxed"),
                "metadata_scaling_exponent": ta.get("scaling_exponent"),
                "metadata_scaling_r2": ta.get("scaling_r2"),
                "metadata_fold": "test",
            })

        if examples:
            output["datasets"].append({
                "dataset": f"csd_temp_T{T}__{MODEL.split('/')[-1]}",
                "examples": examples,
            })

    return output


# ---------------------------------------------------------------------------
# Sanity Checks
# ---------------------------------------------------------------------------

def sanity_check_mini(
    results: dict[tuple[float, int], list[dict]],
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

    # Check basic temperature effect at easiest available level
    temps_in_results = sorted(set(t for t, _ in results.keys()))
    levels_in_results = sorted(set(l for _, l in results.keys()))
    if len(temps_in_results) >= 2 and levels_in_results:
        easy_lvl = levels_in_results[0]
        t_low = temps_in_results[0]
        t_high = temps_in_results[-1]
        key_low = (t_low, easy_lvl)
        key_high = (t_high, easy_lvl)
        if key_low in results and key_high in results:
            acc_low = compute_accuracy(results[key_low])
            acc_high = compute_accuracy(results[key_high])
            logger.info(
                f"MINI: Easy level={easy_lvl}: "
                f"T={t_low} acc={acc_low:.2f}, T={t_high} acc={acc_high:.2f}"
            )
            if acc_low < acc_high:
                logger.warning(
                    f"MINI: T={t_low} has LOWER accuracy than T={t_high} at easy level "
                    f"(unexpected but may be noise at small N)"
                )

    logger.info("MINI sanity checks PASSED")
    return True


def sanity_check_medium(
    results: dict[tuple[float, int], list[dict]],
    problems_by_level: dict[int, list[dict]],
) -> bool:
    """Sanity check medium run results."""
    if not results:
        logger.error("MEDIUM SANITY FAIL: No results")
        return False

    # Check accuracy trend for each temperature
    temps_in_results = sorted(set(t for t, _ in results.keys()))
    for temp in temps_in_results:
        accs = {}
        for lvl in DIFFICULTY_LEVELS:
            key = (temp, lvl)
            if key in results and results[key]:
                accs[lvl] = compute_accuracy(results[key])

        if accs:
            sorted_lvls = sorted(accs)
            low_acc = np.mean([accs[lvl] for lvl in sorted_lvls[:3]])
            high_acc = np.mean([accs[lvl] for lvl in sorted_lvls[-3:]])
            logger.info(
                f"MEDIUM T={temp}: easy_acc={low_acc:.2f}, hard_acc={high_acc:.2f}"
            )

    # Check d* estimates across temperatures
    d_stars = {}
    for temp in temps_in_results:
        for lvl in sorted(set(l for _, l in results.keys())):
            key = (temp, lvl)
            if key in results and results[key]:
                acc = compute_accuracy(results[key])
                if acc < 0.50:
                    d_stars[temp] = lvl
                    break
        if temp not in d_stars:
            d_stars[temp] = 26  # never dropped

    if d_stars:
        d_star_vals = list(d_stars.values())
        d_star_range = max(d_star_vals) - min(d_star_vals)
        logger.info(f"MEDIUM d* estimates: {d_stars}, range={d_star_range}")
        if d_star_range > 6:
            logger.warning(f"MEDIUM: d* range={d_star_range} > 6 (large temperature effect on capability)")

    logger.info("MEDIUM sanity checks PASSED")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@logger.catch
def main():
    global TOTAL_COST_USD

    logger.info("=== CSD Temperature Manipulation Experiment ===")
    logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f}GB RAM, RAM budget: {RAM_BUDGET_BYTES/1e9:.1f}GB")
    logger.info(f"Workspace: {WORKSPACE}")
    logger.info(f"Model: {MODEL}")
    logger.info(f"Temperatures: {TEMPERATURES}")

    # PHASE 1: Load dataset
    problems_by_level = load_dataset(DATA_PATH)

    # =================================================================
    # MINI RUN
    # =================================================================
    logger.info("=" * 60)
    logger.info("=== MINI RUN ===")
    t_mini_start = time.time()

    mini_results = asyncio.run(
        generate_responses_for_temp_config(
            problems_by_level,
            temps=SCALE_MINI["temps"],
            levels=SCALE_MINI["levels"],
            n_problems=SCALE_MINI["n_problems"],
            n_responses=SCALE_MINI["n_responses"],
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
            temp, lvl = key
            ind = compute_csd_indicators(
                embs, resps, problems_by_level[lvl], SCALE_MINI["n_problems"]
            )
            logger.info(
                f"  MINI T={temp} lvl={lvl}: "
                f"acc={ind.accuracy:.2f}, var={ind.embedding_variance:.4f}, "
                f"dip_p={ind.dip_pvalue:.3f}, sil={ind.silhouette_k2:.3f}, "
                f"bc={ind.bimodality_coefficient:.3f}, "
                f"disagree={ind.disagreement_rate:.3f}"
            )
    del mini_embeddings
    gc.collect()
    logger.info("MINI CSD indicator check PASSED")

    # =================================================================
    # MEDIUM RUN
    # =================================================================
    logger.info("=" * 60)
    logger.info("=== MEDIUM RUN ===")
    t_med_start = time.time()

    medium_results = asyncio.run(
        generate_responses_for_temp_config(
            problems_by_level,
            temps=SCALE_MEDIUM["temps"],
            levels=SCALE_MEDIUM["levels"],
            n_problems=SCALE_MEDIUM["n_problems"],
            n_responses=SCALE_MEDIUM["n_responses"],
            concurrency=25,
        )
    )

    t_med = time.time() - t_med_start
    logger.info(f"MEDIUM run took {t_med:.1f}s, cost so far: ${TOTAL_COST_USD:.4f}")

    if not sanity_check_medium(medium_results, problems_by_level):
        logger.warning("MEDIUM sanity check had warnings, continuing anyway")

    # Extrapolate full run time
    medium_calls = sum(len(v) for v in medium_results.values())
    full_target_calls = len(TEMPERATURES) * len(DIFFICULTY_LEVELS) * PROBLEMS_PER_LEVEL * RESPONSES_PER_PROBLEM
    if medium_calls > 0:
        time_per_call = t_med / medium_calls
        estimated_full_time = time_per_call * full_target_calls
        logger.info(
            f"Extrapolation: {time_per_call:.2f}s/call, "
            f"full run ~{estimated_full_time:.0f}s ({estimated_full_time/60:.1f}min)"
        )

    # Cost extrapolation
    if TOTAL_COST_USD > 0:
        cost_per_call = TOTAL_COST_USD / (sum(len(v) for v in medium_results.values()) + sum(len(v) for v in mini_results.values()))
        estimated_full_cost = cost_per_call * full_target_calls
        logger.info(f"Cost extrapolation: ~${estimated_full_cost:.2f} for full run")
        if estimated_full_cost > COST_LIMIT_USD:
            logger.warning(
                f"Estimated full cost ${estimated_full_cost:.2f} > limit ${COST_LIMIT_USD}, "
                f"will monitor carefully"
            )

    del medium_results, mini_results
    gc.collect()

    # =================================================================
    # FULL RUN
    # =================================================================
    logger.info("=" * 60)
    logger.info("=== FULL RUN ===")
    t_full_start = time.time()

    all_results = asyncio.run(
        generate_responses_for_temp_config(
            problems_by_level,
            temps=SCALE_FULL["temps"],
            levels=SCALE_FULL["levels"],
            n_problems=SCALE_FULL["n_problems"],
            n_responses=SCALE_FULL["n_responses"],
            concurrency=25,
        )
    )

    t_full = time.time() - t_full_start
    logger.info(f"FULL run took {t_full:.1f}s, cost so far: ${TOTAL_COST_USD:.4f}")

    # Check response counts
    total_responses = sum(len(v) for v in all_results.values())
    logger.info(f"Total responses collected: {total_responses}")
    target_total = len(TEMPERATURES) * len(DIFFICULTY_LEVELS) * PROBLEMS_PER_LEVEL * RESPONSES_PER_PROBLEM
    if total_responses < target_total * 0.95:
        logger.warning(
            f"Only {total_responses}/{target_total} responses "
            f"({total_responses/target_total:.1%}), continuing with available data"
        )

    # PHASE 3: Compute accuracy for every (temp, level)
    logger.info("=" * 60)
    logger.info("=== COMPUTING ACCURACY ===")
    for T in TEMPERATURES:
        accs = []
        for lvl in DIFFICULTY_LEVELS:
            key = (T, lvl)
            if key in all_results and all_results[key]:
                acc = compute_accuracy(all_results[key])
                accs.append((lvl, acc))
        if accs:
            easy_acc = accs[0][1] if accs else 0
            mid_idx = len(accs) // 2
            mid_acc = accs[mid_idx][1] if accs else 0
            hard_acc = accs[-1][1] if accs else 0
            logger.info(
                f"  T={T}: lvl{accs[0][0]}={easy_acc:.2f}, "
                f"lvl{accs[mid_idx][0]}={mid_acc:.2f} (mid), "
                f"lvl{accs[-1][0]}={hard_acc:.2f}"
            )

    # PHASE 4: Semantic Embedding
    logger.info("=" * 60)
    logger.info("=== EMBEDDING RESPONSES ===")
    embeddings = embed_all_responses(all_results)

    # PHASE 5: CSD Indicator Computation
    logger.info("=" * 60)
    logger.info("=== COMPUTING CSD INDICATORS ===")
    indicators: dict[tuple[float, int], CSDIndicators] = {}
    for (temp, level), resps in all_results.items():
        key = (temp, level)
        if key not in embeddings or embeddings[key].shape[0] < 2:
            continue
        embs = embeddings[key]
        indicators[key] = compute_csd_indicators(
            embs, resps, problems_by_level[level], SCALE_FULL["n_problems"]
        )

    logger.info(f"Computed CSD indicators for {len(indicators)} (temp, level) pairs")

    # Free embeddings to save memory
    del embeddings
    gc.collect()

    # PHASE 6: Per-Temperature Analysis
    logger.info("=" * 60)
    logger.info("=== PER-TEMPERATURE ANALYSIS ===")
    temp_analysis = compute_per_temperature_analysis(
        indicators, TEMPERATURES, DIFFICULTY_LEVELS
    )

    # PHASE 7: Cross-Temperature Dose-Response Analysis
    logger.info("=" * 60)
    logger.info("=== DOSE-RESPONSE ANALYSIS ===")
    dose_response = compute_dose_response_analysis(
        temp_analysis, indicators, TEMPERATURES, DIFFICULTY_LEVELS
    )

    # PHASE 8: Output Assembly
    logger.info("=" * 60)
    logger.info("=== ASSEMBLING OUTPUT ===")
    output = build_output(
        all_results, indicators, temp_analysis, dose_response,
        TEMPERATURES, DIFFICULTY_LEVELS, TOTAL_COST_USD,
    )

    # Write output
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    logger.info(f"Output written to {OUTPUT_PATH}")

    # Log summary statistics
    total_examples = sum(len(ds["examples"]) for ds in output["datasets"])
    logger.info(f"Output has {len(output['datasets'])} datasets, {total_examples} total examples")
    logger.info(f"Total API cost: ${TOTAL_COST_USD:.4f}")

    # Validate output schema
    for ds in output["datasets"]:
        for ex in ds["examples"]:
            assert "input" in ex, "Missing input field"
            assert "output" in ex, "Missing output field"
            for k in ex:
                if k.startswith("predict_"):
                    assert isinstance(ex[k], str), (
                        f"predict_ field {k} must be string, got {type(ex[k])}"
                    )

    logger.info("Output schema self-check PASSED")
    logger.info("=== EXPERIMENT COMPLETE ===")


if __name__ == "__main__":
    main()
