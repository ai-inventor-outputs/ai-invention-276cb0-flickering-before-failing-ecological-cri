#!/usr/bin/env python3
"""Multi-Hop Factual Reasoning Dataset Assembly Script.

Assembles a multi-hop factual reasoning dataset spanning 1-6 hops from:
- MuSiQue (hops 1-4): question decomposition provides hop structure
- HotpotQA (hop 2): bridge-type questions supplement
- Synthetic (hops 5-6): entity-chained questions composed via LLM
"""

from loguru import logger
from pathlib import Path
import json
import sys
import random
import subprocess
import os
import resource
import math
import gc
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# === Setup ===
WORKSPACE = Path("/ai-inventor/aii_pipeline/runs/neurips-open/3_invention_loop/iter_1/gen_art/data_id5_it1__opus")
LOGS_DIR = WORKSPACE / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOGS_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# === Hardware Detection ===
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
RAM_BUDGET = int(TOTAL_RAM_GB * 0.5 * 1e9)  # 50% of container RAM
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f}GB RAM, budget={RAM_BUDGET/1e9:.1f}GB")

# === Paths ===
DATA_DIR = WORKSPACE / "temp" / "datasets"
MUSIQUE_VAL = DATA_DIR / "full_bdsaglam_musique_answerable_validation.json"
MUSIQUE_TRAIN = DATA_DIR / "full_bdsaglam_musique_answerable_train.json"
HOTPOTQA_VAL = DATA_DIR / "full_hotpotqa_hotpot_qa_distractor_validation.json"
OUTPUT_PATH = WORKSPACE / "data_out.json"

# OpenRouter config
SKILL_DIR = "/ai-inventor/.claude/skills/aii_openrouter_llms"
OR_PY = f"{SKILL_DIR}/../.ability_client_venv/bin/python"
OR_SCRIPT = f"{SKILL_DIR}/scripts/aii_or_call_llms.py"
LLM_MODEL = "google/gemini-2.0-flash-001"

SEED = 42
random.seed(SEED)
TARGET_PER_HOP = 30


# === Helper Functions ===
def is_short_answer(answer: str) -> bool:
    """Check if answer is a short factual entity (<=5 words, <100 chars)."""
    return len(answer.split()) <= 5 and len(answer) < 100 and len(answer) > 0


def get_hop_count(example: dict) -> int:
    """Determine hop count from question_decomposition length."""
    return len(example.get("question_decomposition", []))


# === Step 1: Extract 1-Hop Questions ===
def extract_1hop_questions(musique_data: list[dict]) -> list[dict]:
    """Extract 1-hop questions from MuSiQue decompositions.
    Only uses the FIRST sub-question in each chain (self-contained, no #N refs).
    """
    logger.info("Step 1: Extracting 1-hop questions from MuSiQue decompositions...")
    candidates = []
    seen_questions = set()

    for example in musique_data:
        decomp = example.get("question_decomposition", [])
        if not decomp:
            continue

        first_sq = decomp[0]
        q_text = first_sq["question"]
        answer = first_sq["answer"]

        # Skip if contains placeholder references
        if "#" in q_text:
            continue

        # Skip non-short answers
        if not is_short_answer(answer):
            continue

        # Deduplicate by normalized question text
        q_normalized = q_text.lower().strip()
        if q_normalized in seen_questions:
            continue
        seen_questions.add(q_normalized)

        # Get supporting paragraph title if available
        para_idx = first_sq.get("paragraph_support_idx")
        para_title = ""
        if para_idx is not None:
            paragraphs = example.get("paragraphs", [])
            if 0 <= para_idx < len(paragraphs):
                para_title = paragraphs[para_idx].get("title", "")

        candidates.append({
            "question_text": q_text,
            "ground_truth_answer": answer,
            "answer_aliases": [],
            "supporting_facts": [{
                "hop": 1,
                "question": q_text,
                "answer": answer,
                "paragraph_title": para_title
            }],
            "difficulty_level": 1,
            "source": "musique_decomposition",
            "id": f"1hop_{first_sq['id']}",
            "metadata_fold": "test"
        })

    logger.info(f"Found {len(candidates)} unique 1-hop candidates")

    if len(candidates) > TARGET_PER_HOP:
        selected = random.sample(candidates, TARGET_PER_HOP)
    else:
        selected = candidates[:TARGET_PER_HOP]

    logger.info(f"Selected {len(selected)} 1-hop questions")
    return selected


# === Steps 2-4: Sample N-Hop Questions from MuSiQue ===
def sample_musique_nhop(musique_data: list[dict], n_hops: int, count: int) -> list[dict]:
    """Sample n-hop questions from MuSiQue data."""
    logger.info(f"Sampling {count} {n_hops}-hop questions from MuSiQue...")

    candidates = [
        ex for ex in musique_data
        if get_hop_count(ex) == n_hops
        and ex.get("answerable", True)
        and is_short_answer(ex.get("answer", ""))
    ]
    logger.info(f"Found {len(candidates)} {n_hops}-hop candidates with short answers")

    if len(candidates) > count:
        selected = random.sample(candidates, count)
    else:
        selected = candidates[:count]

    results = []
    for ex in selected:
        decomp = ex.get("question_decomposition", [])
        supporting_facts = []
        for i, sq in enumerate(decomp):
            para_idx = sq.get("paragraph_support_idx")
            para_title = ""
            if para_idx is not None:
                paragraphs = ex.get("paragraphs", [])
                if 0 <= para_idx < len(paragraphs):
                    para_title = paragraphs[para_idx].get("title", "")
            supporting_facts.append({
                "hop": i + 1,
                "question": sq["question"],
                "answer": sq["answer"],
                "paragraph_title": para_title
            })

        results.append({
            "question_text": ex["question"],
            "ground_truth_answer": ex["answer"],
            "answer_aliases": ex.get("answer_aliases", []),
            "supporting_facts": supporting_facts,
            "difficulty_level": n_hops,
            "source": "musique",
            "id": ex["id"],
            "metadata_fold": "test"
        })

    logger.info(f"Selected {len(results)} {n_hops}-hop questions from MuSiQue")
    return results


# === Step 2 supplement: HotpotQA Bridge Questions ===
def sample_hotpotqa_2hop(hotpotqa_data: list[dict], count: int) -> list[dict]:
    """Sample 2-hop bridge questions from HotpotQA."""
    logger.info(f"Sampling {count} 2-hop bridge questions from HotpotQA...")

    bridge = [
        ex for ex in hotpotqa_data
        if ex.get("type") == "bridge"
        and is_short_answer(ex.get("answer", ""))
        and ex.get("level") in ("medium", "hard")
    ]
    logger.info(f"Found {len(bridge)} bridge candidates (medium/hard)")

    if len(bridge) > count:
        selected = random.sample(bridge, count)
    else:
        selected = bridge[:count]

    results = []
    for ex in selected:
        sf = ex.get("supporting_facts", {})
        titles = sf.get("title", [])
        sent_ids = sf.get("sent_id", [])

        # Group by title to form hops
        title_order = []
        seen_titles = set()
        for t in titles:
            if t not in seen_titles:
                title_order.append(t)
                seen_titles.add(t)

        # Build supporting facts with context sentences
        supporting_facts = []
        context = ex.get("context", {})
        ctx_titles = context.get("title", [])
        ctx_sentences = context.get("sentences", [])

        for i, title in enumerate(title_order):
            # Find matching context
            ctx_text = ""
            for ct_idx, ct in enumerate(ctx_titles):
                if ct == title and ct_idx < len(ctx_sentences):
                    # Get the relevant sentences
                    relevant_sents = []
                    for j, (st, sid) in enumerate(zip(titles, sent_ids)):
                        if st == title and sid < len(ctx_sentences[ct_idx]):
                            relevant_sents.append(ctx_sentences[ct_idx][sid])
                    ctx_text = " ".join(relevant_sents)
                    break

            supporting_facts.append({
                "hop": i + 1,
                "question": f"Information from: {title}",
                "answer": ex["answer"] if i == len(title_order) - 1 else "(intermediate fact)",
                "paragraph_title": title
            })

        results.append({
            "question_text": ex["question"],
            "ground_truth_answer": ex["answer"],
            "answer_aliases": [],
            "supporting_facts": supporting_facts,
            "difficulty_level": 2,
            "source": "hotpotqa",
            "id": f"hotpot_{ex['id']}",
            "metadata_fold": "test"
        })

    logger.info(f"Selected {len(results)} 2-hop questions from HotpotQA")
    return results


# === Steps 5-6: Entity Chaining ===
def build_entity_chains(musique_data: list[dict]) -> tuple[list, list]:
    """Build 5-hop and 6-hop entity chains from MuSiQue sub-questions."""
    logger.info("Building entity chains for 5-6 hop questions...")

    # Collect all self-contained sub-questions (no #N refs)
    # Map: entity_name (lowercase) -> list of (question, answer) pairs
    entity_to_sqs = defaultdict(list)
    all_self_contained = []

    for ex in musique_data:
        for sq in ex.get("question_decomposition", []):
            q = sq["question"]
            if "#" in q:
                continue
            if not is_short_answer(sq["answer"]):
                continue

            # Extract entity from "ENTITY >> relation" format
            if ">>" in q:
                entity = q.split(">>")[0].strip().lower()
            else:
                # Natural language - use the whole question as key
                entity = q.lower().strip()

            entity_to_sqs[entity].append({
                "question": q,
                "answer": sq["answer"],
                "source_id": ex["id"]
            })
            all_self_contained.append({
                "question": q,
                "answer": sq["answer"],
                "entity": entity,
                "source_id": ex["id"]
            })

    logger.info(f"Collected {len(all_self_contained)} self-contained sub-questions")
    logger.info(f"Unique entities: {len(entity_to_sqs)}")

    # Get 4-hop chains
    four_hop_examples = [
        ex for ex in musique_data
        if get_hop_count(ex) == 4 and ex.get("answerable", True)
    ]
    logger.info(f"Found {len(four_hop_examples)} 4-hop examples for chain extension")

    # Build answer -> sub-questions mapping (for matching final answers to entities)
    answer_to_entity_sqs = defaultdict(list)
    for sq_data in all_self_contained:
        answer_to_entity_sqs[sq_data["answer"].lower().strip()].append(sq_data)

    # Try to extend 4-hop chains to 5-hop
    five_hop_chains = []
    used_base_ids = set()

    for ex in four_hop_examples:
        decomp = ex.get("question_decomposition", [])
        if len(decomp) != 4:
            continue

        final_answer = decomp[-1]["answer"].lower().strip()

        # Find sub-questions whose ENTITY matches our final answer
        matching_sqs = entity_to_sqs.get(final_answer, [])

        for ext_sq in matching_sqs:
            if ext_sq["source_id"] == ex["id"]:
                continue  # Don't chain to self

            chain = []
            for sq in decomp:
                chain.append({"question": sq["question"], "answer": sq["answer"]})
            chain.append({"question": ext_sq["question"], "answer": ext_sq["answer"]})

            chain_key = f"{ex['id']}_{ext_sq['source_id']}"
            if chain_key not in used_base_ids:
                used_base_ids.add(chain_key)
                five_hop_chains.append({
                    "chain": chain,
                    "final_answer": ext_sq["answer"],
                    "base_id": ex["id"]
                })
                break

        if len(five_hop_chains) >= TARGET_PER_HOP * 3:
            break

    logger.info(f"Found {len(five_hop_chains)} potential 5-hop chains via entity chaining")

    # Try to extend 5-hop chains to 6-hop
    six_hop_chains = []
    for chain_data in five_hop_chains:
        chain = chain_data["chain"]
        final_answer = chain[-1]["answer"].lower().strip()

        matching_sqs = entity_to_sqs.get(final_answer, [])
        for ext_sq in matching_sqs:
            new_chain = list(chain) + [{"question": ext_sq["question"], "answer": ext_sq["answer"]}]
            six_hop_chains.append({
                "chain": new_chain,
                "final_answer": ext_sq["answer"],
                "base_id": chain_data["base_id"]
            })
            break

        if len(six_hop_chains) >= TARGET_PER_HOP * 3:
            break

    logger.info(f"Found {len(six_hop_chains)} potential 6-hop chains via entity chaining")

    # Also try: 3-hop + 2 extensions for more 5-hop chains
    if len(five_hop_chains) < TARGET_PER_HOP * 2:
        logger.info("Trying 3-hop + 2 extensions for more 5-hop chains...")
        three_hop_examples = [
            ex for ex in musique_data
            if get_hop_count(ex) == 3 and ex.get("answerable", True)
        ]
        for ex in three_hop_examples:
            decomp = ex.get("question_decomposition", [])
            if len(decomp) != 3:
                continue
            final_answer = decomp[-1]["answer"].lower().strip()
            matching_sqs = entity_to_sqs.get(final_answer, [])
            for ext_sq1 in matching_sqs:
                if ext_sq1["source_id"] == ex["id"]:
                    continue
                ext1_answer = ext_sq1["answer"].lower().strip()
                matching_sqs2 = entity_to_sqs.get(ext1_answer, [])
                for ext_sq2 in matching_sqs2:
                    if ext_sq2["source_id"] in (ex["id"], ext_sq1["source_id"]):
                        continue
                    chain = []
                    for sq in decomp:
                        chain.append({"question": sq["question"], "answer": sq["answer"]})
                    chain.append({"question": ext_sq1["question"], "answer": ext_sq1["answer"]})
                    chain.append({"question": ext_sq2["question"], "answer": ext_sq2["answer"]})
                    five_hop_chains.append({
                        "chain": chain,
                        "final_answer": ext_sq2["answer"],
                        "base_id": ex["id"]
                    })
                    break
                if len(five_hop_chains) >= TARGET_PER_HOP * 3:
                    break
            if len(five_hop_chains) >= TARGET_PER_HOP * 3:
                break
        logger.info(f"After 3-hop extension: {len(five_hop_chains)} 5-hop chains")

    return five_hop_chains, six_hop_chains


# === LLM Question Composition ===
def call_llm(prompt: str, temperature: float = 0.3, max_tokens: int = 300) -> str | None:
    """Call OpenRouter LLM and return response text."""
    try:
        result = subprocess.run(
            [OR_PY, OR_SCRIPT,
             "--model", LLM_MODEL,
             "--input", prompt,
             "--temperature", str(temperature),
             "--max-tokens", str(max_tokens)],
            capture_output=True, text=True, timeout=90
        )
        if result.returncode == 0 and "Response:" in result.stdout:
            response = result.stdout.split("Response:", 1)[1].strip()
            # Remove token info at the end
            lines = response.split("\n")
            clean_lines = []
            for line in lines:
                if line.startswith("Tokens:") or line.startswith("Model:"):
                    break
                clean_lines.append(line)
            return "\n".join(clean_lines).strip()
        else:
            logger.warning(f"LLM call failed: returncode={result.returncode}, stderr={result.stderr[:200]}")
    except subprocess.TimeoutExpired:
        logger.warning("LLM call timed out")
    except Exception:
        logger.exception("LLM call error")
    return None


def compose_question_via_llm(chain: list[dict], final_answer: str, chain_id: str) -> str | None:
    """Use LLM to compose a natural multi-hop question from a chain."""
    chain_text = ""
    for i, step in enumerate(chain):
        chain_text += f"{i+1}. Q: \"{step['question']}\" A: \"{step['answer']}\"\n"

    prompt = (
        f"Given these chained factual sub-questions and answers:\n{chain_text}\n"
        f"Compose a single natural multi-hop question whose answer is \"{final_answer}\" "
        f"and requires all {len(chain)} reasoning steps.\n\n"
        f"Rules:\n"
        f"1. Natural-sounding, grammatically correct English\n"
        f"2. Unambiguous with only one correct answer\n"
        f"3. Genuinely requires all hops\n"
        f"4. About factual knowledge only\n\n"
        f"Return ONLY the question text, nothing else."
    )

    response = call_llm(prompt, temperature=0.3, max_tokens=200)
    if response:
        # Clean up response
        q = response.strip().strip('"').strip("'").strip()
        q = q.split("\n")[0].strip()
        if q and len(q) > 15 and q.endswith("?"):
            return q
        elif q and len(q) > 15:
            return q + "?"
    return None


def generate_synthetic_questions(chains: list[dict], n_hops: int) -> list[dict]:
    """Generate synthetic n-hop questions from entity chains using LLM."""
    logger.info(f"Generating up to {TARGET_PER_HOP} synthetic {n_hops}-hop questions from {len(chains)} chains...")

    results = []
    to_try = chains[:TARGET_PER_HOP * 2]

    # Use ThreadPoolExecutor for parallel LLM calls (I/O bound)
    with ThreadPoolExecutor(max_workers=min(NUM_CPUS, 4)) as executor:
        futures = {}
        for i, chain_data in enumerate(to_try):
            chain_id = f"{n_hops}hop_synth_{i}"
            future = executor.submit(
                compose_question_via_llm,
                chain_data["chain"],
                chain_data["final_answer"],
                chain_id
            )
            futures[future] = (i, chain_data, chain_id)

        for future in as_completed(futures):
            idx, chain_data, chain_id = futures[future]
            try:
                question_text = future.result()
                if question_text and len(results) < TARGET_PER_HOP:
                    supporting_facts = []
                    for j, step in enumerate(chain_data["chain"]):
                        supporting_facts.append({
                            "hop": j + 1,
                            "question": step["question"],
                            "answer": step["answer"],
                            "paragraph_title": ""
                        })

                    results.append({
                        "question_text": question_text,
                        "ground_truth_answer": chain_data["final_answer"],
                        "answer_aliases": [],
                        "supporting_facts": supporting_facts,
                        "difficulty_level": n_hops,
                        "source": "synthetic",
                        "id": chain_id,
                        "metadata_fold": "test"
                    })
                    logger.info(f"[{n_hops}-hop {len(results)}/{TARGET_PER_HOP}] {question_text[:80]}...")
            except Exception:
                logger.exception(f"Failed to generate question for chain {chain_id}")

    logger.info(f"Generated {len(results)} synthetic {n_hops}-hop questions from entity chains")
    return results


def generate_fallback_chains_via_llm(n_hops: int, count: int) -> list[dict]:
    """Fallback: Use LLM to generate factual chains from scratch."""
    logger.info(f"Generating {count} fallback {n_hops}-hop chains via LLM...")

    topics = [
        "world capitals and geography", "famous scientists and their discoveries",
        "historical events and leaders", "classical music composers",
        "famous novels and their authors", "Olympic sports records",
        "award-winning films and directors", "important inventions",
        "major rivers and the cities on them", "Nobel Prize winners",
        "space exploration milestones", "ancient civilizations and empires",
        "famous paintings and museums", "Olympic host cities",
        "world religious sites", "famous architectural landmarks",
        "major wars and peace treaties", "famous explorers and expeditions",
        "programming languages and their creators", "chemical elements discovery",
        "US presidents and their policies", "European royal families",
        "famous universities founding", "endangered species habitats",
        "famous bridges around the world", "active volcanoes",
        "world deserts", "island nations", "world currencies",
        "famous speeches in history", "astronomical discoveries",
        "mountain ranges and peaks", "great lakes of the world",
        "famous operas and composers", "ancient trade routes",
        "major earthquakes in history", "famous philosophers",
        "world heritage sites", "telecommunications pioneers",
        "famous mathematicians", "renewable energy milestones"
    ]

    results = []
    attempts = 0
    max_attempts = count * 3

    def try_generate_one(topic_idx: int) -> dict | None:
        topic = topics[topic_idx % len(topics)]
        prompt = (
            f"Create a factual {n_hops}-hop reasoning chain about {topic}. "
            f"The chain must have EXACTLY {n_hops} steps where each answer leads to the next question.\n\n"
            f"Format as a JSON array:\n"
            f'[{{"question": "Q1", "answer": "A1"}}, {{"question": "Q2 (uses A1)", "answer": "A2"}}, ...]\n\n'
            f"Rules:\n"
            f"1. Each fact must be well-known and verifiable\n"
            f"2. Each answer must be 1-3 words (a named entity, place, date, or number)\n"
            f"3. Each question must depend on the previous answer\n"
            f"4. Exactly {n_hops} steps\n"
            f"5. No placeholder references\n\n"
            f"Return ONLY the JSON array."
        )

        response = call_llm(prompt, temperature=0.7, max_tokens=600)
        if not response:
            return None

        try:
            start = response.find("[")
            end = response.rfind("]") + 1
            if start < 0 or end <= start:
                return None
            chain_json = json.loads(response[start:end])
            if len(chain_json) != n_hops:
                return None

            # Validate chain structure
            for step in chain_json:
                if "question" not in step or "answer" not in step:
                    return None
                if not is_short_answer(step["answer"]):
                    return None

            return chain_json
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

    # Generate chains with parallel workers
    with ThreadPoolExecutor(max_workers=min(NUM_CPUS, 4)) as executor:
        futures = {}
        for i in range(max_attempts):
            if len(results) >= count:
                break
            future = executor.submit(try_generate_one, i)
            futures[future] = i

        for future in as_completed(futures):
            if len(results) >= count:
                break
            idx = futures[future]
            try:
                chain_json = future.result()
                if chain_json is None:
                    continue

                # Compose the question
                final_answer = chain_json[-1]["answer"]
                chain_id = f"{n_hops}hop_fallback_{idx}"
                question_text = compose_question_via_llm(chain_json, final_answer, chain_id)

                if question_text:
                    supporting_facts = []
                    for j, step in enumerate(chain_json):
                        supporting_facts.append({
                            "hop": j + 1,
                            "question": step["question"],
                            "answer": step["answer"],
                            "paragraph_title": ""
                        })

                    results.append({
                        "question_text": question_text,
                        "ground_truth_answer": final_answer,
                        "answer_aliases": [],
                        "supporting_facts": supporting_facts,
                        "difficulty_level": n_hops,
                        "source": "synthetic",
                        "id": chain_id,
                        "metadata_fold": "test"
                    })
                    logger.info(f"[Fallback {n_hops}-hop {len(results)}/{count}] {question_text[:80]}...")
            except Exception:
                logger.exception(f"Fallback generation error for index {idx}")

    logger.info(f"Generated {len(results)} fallback {n_hops}-hop questions")
    return results


# === Validation ===
def validate_dataset(dataset: list[dict]) -> bool:
    """Validate the assembled dataset against requirements."""
    logger.info("Validating dataset...")
    errors = []
    required_fields = [
        "question_text", "ground_truth_answer", "answer_aliases",
        "supporting_facts", "difficulty_level", "source", "id", "metadata_fold"
    ]

    ids_seen = set()
    hop_counts = defaultdict(int)

    for i, record in enumerate(dataset):
        for field in required_fields:
            if field not in record:
                errors.append(f"Record {i}: missing field '{field}'")

        if not isinstance(record.get("difficulty_level"), int):
            errors.append(f"Record {i}: difficulty_level must be int, got {type(record.get('difficulty_level'))}")

        if not isinstance(record.get("supporting_facts"), list):
            errors.append(f"Record {i}: supporting_facts must be list")

        answer = record.get("ground_truth_answer", "")
        if not answer or len(answer) > 100:
            errors.append(f"Record {i}: invalid answer (empty or >100 chars)")

        rid = record.get("id")
        if rid in ids_seen:
            errors.append(f"Record {i}: duplicate id '{rid}'")
        ids_seen.add(rid)

        if record.get("metadata_fold") != "test":
            errors.append(f"Record {i}: metadata_fold must be 'test'")

        hop_counts[record.get("difficulty_level", 0)] += 1

    for level in range(1, 7):
        cnt = hop_counts.get(level, 0)
        if cnt < 25:
            errors.append(f"Level {level}: only {cnt} questions (need >= 25)")

    total = len(dataset)
    if total < 150 or total > 200:
        errors.append(f"Total count {total} outside target range 150-200")

    if errors:
        for e in errors:
            logger.error(f"Validation: {e}")
        return False

    logger.info("Validation PASSED")
    return True


# === Main ===
@logger.catch
def main():
    t_start = time.time()

    # === Load data ===
    logger.info("Loading MuSiQue validation data...")
    musique_val = json.loads(MUSIQUE_VAL.read_text())
    logger.info(f"Loaded {len(musique_val)} MuSiQue validation examples")

    logger.info("Loading MuSiQue train data...")
    musique_train = json.loads(MUSIQUE_TRAIN.read_text())
    logger.info(f"Loaded {len(musique_train)} MuSiQue train examples")

    logger.info("Loading HotpotQA validation data...")
    hotpotqa_val = json.loads(HOTPOTQA_VAL.read_text())
    logger.info(f"Loaded {len(hotpotqa_val)} HotpotQA validation examples")

    all_musique = musique_val + musique_train

    # Log hop distribution in MuSiQue
    hop_dist = defaultdict(int)
    for ex in all_musique:
        hop_dist[get_hop_count(ex)] += 1
    logger.info(f"MuSiQue hop distribution: {dict(sorted(hop_dist.items()))}")

    # === Step 1: 1-hop ===
    hop1 = extract_1hop_questions(all_musique)

    # === Step 2: 2-hop (20 MuSiQue + 10 HotpotQA) ===
    hop2_musique = sample_musique_nhop(musique_val, 2, 20)
    if len(hop2_musique) < 20:
        logger.info(f"Only {len(hop2_musique)} 2-hop from val, supplementing from train...")
        extra = sample_musique_nhop(musique_train, 2, 20 - len(hop2_musique))
        hop2_musique.extend(extra)

    hop2_hotpot = sample_hotpotqa_2hop(hotpotqa_val, 10)
    hop2 = hop2_musique + hop2_hotpot

    # === Step 3: 3-hop ===
    hop3 = sample_musique_nhop(musique_val, 3, 30)
    if len(hop3) < 30:
        logger.info(f"Only {len(hop3)} 3-hop from val, supplementing from train...")
        extra = sample_musique_nhop(musique_train, 3, 30 - len(hop3))
        hop3.extend(extra)

    # === Step 4: 4-hop ===
    hop4 = sample_musique_nhop(musique_val, 4, 30)
    if len(hop4) < 30:
        logger.info(f"Only {len(hop4)} 4-hop from val, supplementing from train...")
        extra = sample_musique_nhop(musique_train, 4, 30 - len(hop4))
        hop4.extend(extra)

    logger.info(f"After steps 1-4: 1-hop={len(hop1)}, 2-hop={len(hop2)}, 3-hop={len(hop3)}, 4-hop={len(hop4)}")

    # Free memory before LLM-heavy steps
    del hotpotqa_val
    gc.collect()

    # === Steps 5-6: Entity chaining + LLM composition ===
    five_hop_chains, six_hop_chains = build_entity_chains(all_musique)

    # Free large data
    del all_musique, musique_val, musique_train
    gc.collect()

    # === Step 5: 5-hop questions ===
    hop5 = []
    if len(five_hop_chains) >= 1:
        selected_5 = five_hop_chains[:TARGET_PER_HOP * 2]
        random.shuffle(selected_5)
        hop5 = generate_synthetic_questions(selected_5, 5)

    if len(hop5) < TARGET_PER_HOP:
        needed = TARGET_PER_HOP - len(hop5)
        logger.info(f"Need {needed} more 5-hop questions, using LLM fallback...")
        fallback5 = generate_fallback_chains_via_llm(5, needed)
        hop5.extend(fallback5)

    # === Step 6: 6-hop questions ===
    hop6 = []
    if len(six_hop_chains) >= 1:
        selected_6 = six_hop_chains[:TARGET_PER_HOP * 2]
        random.shuffle(selected_6)
        hop6 = generate_synthetic_questions(selected_6, 6)

    if len(hop6) < TARGET_PER_HOP:
        needed = TARGET_PER_HOP - len(hop6)
        logger.info(f"Need {needed} more 6-hop questions, using LLM fallback...")
        fallback6 = generate_fallback_chains_via_llm(6, needed)
        hop6.extend(fallback6)

    # === Assemble ===
    dataset = hop1 + hop2 + hop3 + hop4 + hop5[:TARGET_PER_HOP] + hop6[:TARGET_PER_HOP]

    # === Summary ===
    logger.info("=" * 60)
    logger.info("DATASET SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total questions: {len(dataset)}")

    for level in range(1, 7):
        level_qs = [d for d in dataset if d["difficulty_level"] == level]
        logger.info(f"  Level {level} ({level}-hop): {len(level_qs)} questions")
        if level_qs:
            avg_ans_len = sum(len(d["ground_truth_answer"]) for d in level_qs) / len(level_qs)
            logger.info(f"    Avg answer length: {avg_ans_len:.1f} chars")
            logger.info(f"    Sample Q: {level_qs[0]['question_text'][:100]}...")
            logger.info(f"    Sample A: {level_qs[0]['ground_truth_answer']}")

    source_counts = defaultdict(int)
    for d in dataset:
        source_counts[d["source"]] += 1
    logger.info(f"Source distribution: {dict(source_counts)}")

    # === Validate ===
    is_valid = validate_dataset(dataset)

    # === Save ===
    OUTPUT_PATH.write_text(json.dumps(dataset, indent=2, ensure_ascii=False))
    logger.info(f"Saved dataset to {OUTPUT_PATH}")

    elapsed = time.time() - t_start
    logger.info(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    if not is_valid:
        logger.warning("Dataset validation had errors - review above")
        sys.exit(1)


if __name__ == "__main__":
    main()
