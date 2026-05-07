#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["loguru"]
# ///
"""Multi-Hop Factual Reasoning Dataset Builder.

Loads MuSiQue and HotpotQA from temp/datasets/, assembles one dataset:
  multi_hop_reasoning_1to6 — Full combined (hops 1-6, ~180 examples)
    - 1-hop: extracted from MuSiQue sub-question decompositions
    - 2-hop: MuSiQue (20) + HotpotQA bridge (10)
    - 3-hop: MuSiQue
    - 4-hop: MuSiQue
    - 5-hop: synthetic (entity-chained from MuSiQue, composed via LLM)
    - 6-hop: synthetic (entity-chained from MuSiQue, composed via LLM)

Outputs in exp_sel_data_out.json schema format → full_data_out.json
"""

import gc
import json
import math
import os
import random
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

from loguru import logger

# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════
WORKSPACE = Path(
    "/ai-inventor/aii_pipeline/runs/neurips-open/3_invention_loop"
    "/iter_1/gen_art/data_id5_it1__opus"
)
DATA_DIR = WORKSPACE / "temp" / "datasets"
LOGS_DIR = WORKSPACE / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOGS_DIR / "data.log"), rotation="30 MB", level="DEBUG")

SEED = 42
random.seed(SEED)
TARGET_PER_HOP = 30

# ═══════════════════════════════════════════════════════════════════
# Hardware Detection & Memory Limits
# ═══════════════════════════════════════════════════════════════════
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
    for p in [
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None


NUM_CPUS = _detect_cpus()
TOTAL_RAM_GB = _container_ram_gb() or 29.0
RAM_BUDGET = int(TOTAL_RAM_GB * 0.5 * 1e9)  # 50 % of container
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f}GB RAM, budget={RAM_BUDGET / 1e9:.1f}GB")

# ═══════════════════════════════════════════════════════════════════
# File Paths
# ═══════════════════════════════════════════════════════════════════
MUSIQUE_VAL = DATA_DIR / "full_bdsaglam_musique_answerable_validation.json"
MUSIQUE_TRAIN = DATA_DIR / "full_bdsaglam_musique_answerable_train.json"
HOTPOTQA_VAL = DATA_DIR / "full_hotpotqa_hotpot_qa_distractor_validation.json"
EXISTING_DATA = WORKSPACE / "data_out.json"
OUTPUT_PATH = WORKSPACE / "full_data_out.json"


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════
def is_short_answer(answer: str) -> bool:
    """True if answer is a short factual entity (≤5 words, <100 chars)."""
    return 0 < len(answer) < 100 and len(answer.split()) <= 5


def get_hop_count(example: dict) -> int:
    return len(example.get("question_decomposition", []))


def fmt(
    question_text: str,
    answer: str,
    difficulty: int,
    source: str,
    example_id: str,
    answer_aliases: list,
    supporting_facts: list,
) -> dict:
    """Format one example in exp_sel_data_out schema."""
    return {
        "input": question_text,
        "output": answer,
        "metadata_fold": "test",
        "metadata_difficulty_level": difficulty,
        "metadata_source": source,
        "metadata_id": example_id,
        "metadata_task_type": "multi_hop_qa",
        "metadata_answer_aliases": json.dumps(answer_aliases, ensure_ascii=False),
        "metadata_supporting_facts": json.dumps(supporting_facts, ensure_ascii=False),
    }


# ═══════════════════════════════════════════════════════════════════
# Extraction Functions
# ═══════════════════════════════════════════════════════════════════
def extract_1hop(data: list[dict]) -> list[dict]:
    """Extract 1-hop Qs from MuSiQue first sub-question decompositions."""
    logger.info("Extracting 1-hop questions from MuSiQue decompositions...")
    candidates: list[dict] = []
    seen: set[str] = set()

    for ex in data:
        decomp = ex.get("question_decomposition", [])
        if not decomp:
            continue
        first = decomp[0]
        q, a = first["question"], first["answer"]

        # Only first sub-Q (self-contained, no #N refs)
        if "#" in q or not is_short_answer(a):
            continue
        q_norm = q.lower().strip()
        if q_norm in seen:
            continue
        seen.add(q_norm)

        para_idx = first.get("paragraph_support_idx")
        para_title = ""
        if para_idx is not None:
            paras = ex.get("paragraphs", [])
            if 0 <= para_idx < len(paras):
                para_title = paras[para_idx].get("title", "")

        sf = [{"hop": 1, "question": q, "answer": a, "paragraph_title": para_title}]
        candidates.append(
            fmt(q, a, 1, "musique_decomposition", f"1hop_{first['id']}", [], sf)
        )

    logger.info(f"Found {len(candidates)} unique 1-hop candidates")
    if len(candidates) > TARGET_PER_HOP:
        candidates = random.sample(candidates, TARGET_PER_HOP)
    logger.info(f"Selected {len(candidates)} 1-hop questions")
    return candidates


def sample_musique_nhop(
    data: list[dict], n_hops: int, count: int
) -> list[dict]:
    """Sample n-hop questions from MuSiQue answerable data."""
    logger.info(f"Sampling {count} {n_hops}-hop from MuSiQue ({len(data)} candidates pool)...")
    pool = [
        ex
        for ex in data
        if get_hop_count(ex) == n_hops
        and ex.get("answerable", True)
        and is_short_answer(ex.get("answer", ""))
    ]
    logger.info(f"  {len(pool)} valid {n_hops}-hop candidates with short answers")

    if len(pool) > count:
        pool = random.sample(pool, count)
    else:
        pool = pool[:count]

    results: list[dict] = []
    for ex in pool:
        decomp = ex.get("question_decomposition", [])
        sf: list[dict] = []
        for i, sq in enumerate(decomp):
            para_idx = sq.get("paragraph_support_idx")
            para_title = ""
            if para_idx is not None:
                paras = ex.get("paragraphs", [])
                if 0 <= para_idx < len(paras):
                    para_title = paras[para_idx].get("title", "")
            sf.append(
                {
                    "hop": i + 1,
                    "question": sq["question"],
                    "answer": sq["answer"],
                    "paragraph_title": para_title,
                }
            )
        results.append(
            fmt(
                ex["question"],
                ex["answer"],
                n_hops,
                "musique",
                ex["id"],
                ex.get("answer_aliases", []),
                sf,
            )
        )

    logger.info(f"  Selected {len(results)} {n_hops}-hop questions")
    return results


def sample_hotpotqa_2hop(data: list[dict], count: int) -> list[dict]:
    """Sample 2-hop bridge questions from HotpotQA distractor validation."""
    logger.info(f"Sampling {count} 2-hop bridge from HotpotQA ({len(data)} total)...")
    bridge = [
        ex
        for ex in data
        if ex.get("type") == "bridge"
        and is_short_answer(ex.get("answer", ""))
        and ex.get("level") in ("medium", "hard")
    ]
    logger.info(f"  {len(bridge)} bridge candidates (medium/hard, short answer)")

    if len(bridge) > count:
        bridge = random.sample(bridge, count)

    results: list[dict] = []
    for ex in bridge:
        sf_raw = ex.get("supporting_facts", {})
        titles_raw = sf_raw.get("title", [])

        # Deduplicate titles preserving order → each title = one hop
        title_order: list[str] = []
        seen: set[str] = set()
        for t in titles_raw:
            if t not in seen:
                title_order.append(t)
                seen.add(t)

        sf: list[dict] = []
        for i, title in enumerate(title_order):
            sf.append(
                {
                    "hop": i + 1,
                    "question": f"Information from: {title}",
                    "answer": ex["answer"]
                    if i == len(title_order) - 1
                    else "(intermediate fact)",
                    "paragraph_title": title,
                }
            )

        results.append(
            fmt(ex["question"], ex["answer"], 2, "hotpotqa", f"hotpot_{ex['id']}", [], sf)
        )

    logger.info(f"  Selected {len(results)} 2-hop HotpotQA questions")
    return results


def load_existing_synthetic(levels: list[int]) -> list[dict]:
    """Load pre-generated synthetic 5/6-hop questions from prior build."""
    logger.info(f"Loading existing synthetic questions for levels {levels}...")
    if not EXISTING_DATA.exists():
        logger.warning(f"No existing data at {EXISTING_DATA}")
        return []

    try:
        raw = json.loads(EXISTING_DATA.read_text())
    except json.JSONDecodeError:
        logger.exception("Failed to parse existing data")
        return []

    # Handle both schema-wrapped and flat-array formats
    if isinstance(raw, dict) and "datasets" in raw:
        examples: list[dict] = []
        for ds in raw["datasets"]:
            examples.extend(ds.get("examples", []))
    elif isinstance(raw, list):
        examples = raw
    else:
        logger.warning("Unexpected existing data format")
        return []

    synthetic = [
        ex
        for ex in examples
        if ex.get("metadata_difficulty_level") in levels
        and ex.get("metadata_source") == "synthetic"
    ]
    logger.info(f"  Loaded {len(synthetic)} existing synthetic questions")
    return synthetic


# ═══════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════
def validate_dataset_group(name: str, examples: list[dict]) -> list[str]:
    """Validate a single dataset group. Returns list of error messages."""
    errors: list[str] = []
    ids_seen: set[str] = set()
    hop_counts: dict[int, int] = defaultdict(int)

    for i, ex in enumerate(examples):
        if "input" not in ex:
            errors.append(f"[{name}] example {i}: missing 'input'")
        if "output" not in ex:
            errors.append(f"[{name}] example {i}: missing 'output'")
        if not ex.get("output"):
            errors.append(f"[{name}] example {i}: empty 'output'")
        if len(str(ex.get("output", ""))) > 100:
            errors.append(f"[{name}] example {i}: output >100 chars")

        eid = ex.get("metadata_id", "")
        if eid in ids_seen:
            errors.append(f"[{name}] example {i}: duplicate id '{eid}'")
        ids_seen.add(eid)

        dl = ex.get("metadata_difficulty_level")
        if isinstance(dl, int):
            hop_counts[dl] += 1
        else:
            errors.append(f"[{name}] example {i}: difficulty_level not int")

    return errors


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
@logger.catch
def main() -> None:
    t0 = time.time()

    # ── Load MuSiQue validation ──────────────────────────────────
    logger.info(f"Loading MuSiQue validation from {MUSIQUE_VAL.name}...")
    musique_val: list[dict] = json.loads(MUSIQUE_VAL.read_text())
    logger.info(f"  {len(musique_val)} validation examples")

    # ── Load MuSiQue train ───────────────────────────────────────
    logger.info(f"Loading MuSiQue train from {MUSIQUE_TRAIN.name} (~240 MB)...")
    musique_train: list[dict] = json.loads(MUSIQUE_TRAIN.read_text())
    logger.info(f"  {len(musique_train)} train examples")

    all_musique = musique_val + musique_train
    hop_dist: dict[int, int] = defaultdict(int)
    for ex in all_musique:
        hop_dist[get_hop_count(ex)] += 1
    logger.info(f"MuSiQue hop distribution: {dict(sorted(hop_dist.items()))}")

    # ── 1-hop: from MuSiQue decompositions ───────────────────────
    hop1 = extract_1hop(all_musique)

    # ── 2-hop: MuSiQue (20) ─────────────────────────────────────
    hop2_musique = sample_musique_nhop(musique_val, 2, 20)
    if len(hop2_musique) < 20:
        logger.info(f"  Only {len(hop2_musique)} 2-hop from val, supplementing from train...")
        extra = sample_musique_nhop(musique_train, 2, 20 - len(hop2_musique))
        hop2_musique.extend(extra)

    # ── 3-hop: MuSiQue ───────────────────────────────────────────
    hop3 = sample_musique_nhop(musique_val, 3, TARGET_PER_HOP)
    if len(hop3) < TARGET_PER_HOP:
        logger.info(f"  Only {len(hop3)} 3-hop from val, supplementing from train...")
        extra = sample_musique_nhop(musique_train, 3, TARGET_PER_HOP - len(hop3))
        hop3.extend(extra)

    # ── 4-hop: MuSiQue ───────────────────────────────────────────
    hop4 = sample_musique_nhop(musique_val, 4, TARGET_PER_HOP)
    if len(hop4) < TARGET_PER_HOP:
        logger.info(f"  Only {len(hop4)} 4-hop from val, supplementing from train...")
        extra = sample_musique_nhop(musique_train, 4, TARGET_PER_HOP - len(hop4))
        hop4.extend(extra)

    # Free large MuSiQue data
    del musique_val, musique_train, all_musique
    gc.collect()
    logger.info("Freed MuSiQue raw data from memory")

    # ── 2-hop: HotpotQA bridge (for combined ds only) ────────────
    logger.info(f"Loading HotpotQA validation from {HOTPOTQA_VAL.name}...")
    hotpotqa_val: list[dict] = json.loads(HOTPOTQA_VAL.read_text())
    logger.info(f"  {len(hotpotqa_val)} HotpotQA examples")
    hop2_hotpot = sample_hotpotqa_2hop(hotpotqa_val, 10)
    del hotpotqa_val
    gc.collect()

    # ── 5-hop & 6-hop: reuse existing synthetic ──────────────────
    synthetic_all = load_existing_synthetic([5, 6])
    hop5 = [e for e in synthetic_all if e["metadata_difficulty_level"] == 5][
        :TARGET_PER_HOP
    ]
    hop6 = [e for e in synthetic_all if e["metadata_difficulty_level"] == 6][
        :TARGET_PER_HOP
    ]
    logger.info(f"  5-hop: {len(hop5)}, 6-hop: {len(hop6)} synthetic questions available")

    # ══════════════════════════════════════════════════════════════
    # Assemble Single Dataset: multi_hop_reasoning_1to6
    # ══════════════════════════════════════════════════════════════
    hop2_combined = hop2_musique + hop2_hotpot
    all_examples = hop1 + hop2_combined + hop3 + hop4 + hop5 + hop6

    # ══════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("DATASET: multi_hop_reasoning_1to6 (hops 1-6, combined)")
    logger.info(f"  Total examples: {len(all_examples)}")
    for lvl in sorted({e["metadata_difficulty_level"] for e in all_examples}):
        cnt = sum(1 for e in all_examples if e["metadata_difficulty_level"] == lvl)
        logger.info(f"  Level {lvl} ({lvl}-hop): {cnt}")
    src_counts: dict[str, int] = defaultdict(int)
    for e in all_examples:
        src_counts[e["metadata_source"]] += 1
    logger.info(f"  Sources: {dict(src_counts)}")

    logger.info("-" * 60)
    logger.info("Sample questions:")
    for lvl in range(1, 7):
        lvl_qs = [e for e in all_examples if e["metadata_difficulty_level"] == lvl]
        if lvl_qs:
            ex = lvl_qs[0]
            logger.info(f"  [{lvl}-hop] Q: {ex['input'][:90]}...")
            logger.info(f"          A: {ex['output']}")

    # ══════════════════════════════════════════════════════════════
    # Validate
    # ══════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Validating...")
    all_errs = validate_dataset_group("multi_hop_reasoning_1to6", all_examples)
    if all_errs:
        for e in all_errs:
            logger.error(e)
        logger.warning(f"Validation found {len(all_errs)} error(s)")
    else:
        logger.info("Validation PASSED")

    # ══════════════════════════════════════════════════════════════
    # Build & Save Output
    # ══════════════════════════════════════════════════════════════
    output = {
        "metadata": {
            "description": "Multi-Hop Factual Reasoning Task Family Dataset (1-6 Hops)",
            "total_examples": len(all_examples),
            "hop_levels": "1-6",
            "target_per_level": TARGET_PER_HOP,
            "sources": ["musique_decomposition", "musique", "hotpotqa", "synthetic"],
            "seed": SEED,
        },
        "datasets": [
            {
                "dataset": "multi_hop_reasoning_1to6",
                "examples": all_examples,
            },
        ],
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    size_kb = OUTPUT_PATH.stat().st_size / 1024
    logger.info(f"Saved {OUTPUT_PATH.name} ({size_kb:.1f} KB)")
    logger.info(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
