#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "z3-solver",
#   "numpy",
#   "loguru",
# ]
# ///
"""
Synthetic Syllogistic/Deductive Logic Dataset Generator.

Generates 280 problems (14 difficulty levels x 20 problems) with:
- Difficulty parameterized by premise count (2-15)
- 4 chain templates (A/B/C/D) with 5 problems each per level
- 50/50 TRUE/FALSE balance (10/10 per level)
- Z3 SMT solver ground truth verification
- Shuffled premise order
- Randomized entity names from 8 semantic categories
"""

import json
import random
import sys
import os
import math
import resource
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from loguru import logger

# ── Logging ──────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ── Hardware detection ───────────────────────────────────────────────────────
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
TOTAL_RAM_GB = _container_ram_gb() or 16.0

# Memory limit: 4GB is plenty for this task
RAM_BUDGET = int(4 * 1024**3)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f}GB RAM, budget={RAM_BUDGET/1e9:.1f}GB")

# ── Constants ────────────────────────────────────────────────────────────────
DIFFICULTY_RANGE = range(2, 16)  # d = 2 to 15
PROBLEMS_PER_LEVEL = 20
TEMPLATES = ["A", "B", "C", "D"]
PROBLEMS_PER_TEMPLATE = 5
UNIVERSE_SIZE = 32  # BitVec size for Z3

SEED = 42
random.seed(SEED)

# ── Entity Name Pools (8 semantic categories) ───────────────────────────────
ENTITY_POOLS = {
    "animals": [
        "cats", "dogs", "eagles", "sharks", "rabbits", "turtles", "wolves",
        "bears", "foxes", "owls", "dolphins", "parrots", "tigers", "snakes",
        "penguins", "horses", "deer", "whales", "hawks", "lizards",
    ],
    "professions": [
        "teachers", "doctors", "engineers", "artists", "scientists", "chefs",
        "pilots", "lawyers", "nurses", "architects", "musicians", "writers",
        "farmers", "judges", "soldiers",
    ],
    "nationalities": [
        "Europeans", "Canadians", "Brazilians", "Egyptians", "Australians",
        "Koreans", "Norwegians", "Mexicans", "Italians", "Greeks", "Japanese",
        "Peruvians", "Swedes", "Finns",
    ],
    "materials": [
        "metals", "crystals", "ceramics", "polymers", "diamonds", "rubies",
        "minerals", "alloys", "fabrics", "plastics", "stones", "glasses",
        "composites", "fibers",
    ],
    "academic": [
        "philosophers", "scholars", "athletes", "mathematicians", "poets",
        "linguists", "historians", "strategists", "diplomats", "theorists",
        "analysts", "virtuosos", "orators", "mentors",
    ],
    "foods_plants": [
        "fruits", "vegetables", "grains", "herbs", "flowers", "mushrooms",
        "berries", "legumes", "spices", "roots", "seeds", "vines", "cacti",
        "ferns",
    ],
    "vehicles_tools": [
        "vehicles", "instruments", "gadgets", "machines", "devices",
        "appliances", "drones", "rockets", "submarines", "tractors",
        "turbines", "lasers", "sensors", "robots",
    ],
    "nature_geography": [
        "mountains", "rivers", "islands", "glaciers", "deserts", "forests",
        "volcanoes", "canyons", "marshes", "plains", "reefs", "geysers",
        "fjords", "tundras",
    ],
}

# Flatten for total count check
ALL_CATEGORY_NAMES = list(ENTITY_POOLS.keys())
TOTAL_NAMES = sum(len(v) for v in ENTITY_POOLS.values())
logger.info(f"Entity pools: {len(ALL_CATEGORY_NAMES)} categories, {TOTAL_NAMES} total names")

# ── Singularization via explicit mapping (reliable for all entity names) ─────
SINGULAR_FORMS = {
    # Animals
    "cats": "cat", "dogs": "dog", "eagles": "eagle", "sharks": "shark",
    "rabbits": "rabbit", "turtles": "turtle", "wolves": "wolf",
    "bears": "bear", "foxes": "fox", "owls": "owl", "dolphins": "dolphin",
    "parrots": "parrot", "tigers": "tiger", "snakes": "snake",
    "penguins": "penguin", "horses": "horse", "deer": "deer",
    "whales": "whale", "hawks": "hawk", "lizards": "lizard",
    # Professions
    "teachers": "teacher", "doctors": "doctor", "engineers": "engineer",
    "artists": "artist", "scientists": "scientist", "chefs": "chef",
    "pilots": "pilot", "lawyers": "lawyer", "nurses": "nurse",
    "architects": "architect", "musicians": "musician", "writers": "writer",
    "farmers": "farmer", "judges": "judge", "soldiers": "soldier",
    # Nationalities
    "Europeans": "European", "Canadians": "Canadian", "Brazilians": "Brazilian",
    "Egyptians": "Egyptian", "Australians": "Australian", "Koreans": "Korean",
    "Norwegians": "Norwegian", "Mexicans": "Mexican", "Italians": "Italian",
    "Greeks": "Greek", "Japanese": "Japanese", "Peruvians": "Peruvian",
    "Swedes": "Swede", "Finns": "Finn",
    # Materials
    "metals": "metal", "crystals": "crystal", "ceramics": "ceramic",
    "polymers": "polymer", "diamonds": "diamond", "rubies": "ruby",
    "minerals": "mineral", "alloys": "alloy", "fabrics": "fabric",
    "plastics": "plastic", "stones": "stone", "glasses": "glass",
    "composites": "composite", "fibers": "fiber",
    # Academic
    "philosophers": "philosopher", "scholars": "scholar", "athletes": "athlete",
    "mathematicians": "mathematician", "poets": "poet", "linguists": "linguist",
    "historians": "historian", "strategists": "strategist", "diplomats": "diplomat",
    "theorists": "theorist", "analysts": "analyst", "virtuosos": "virtuoso",
    "orators": "orator", "mentors": "mentor",
    # Foods/Plants
    "fruits": "fruit", "vegetables": "vegetable", "grains": "grain",
    "herbs": "herb", "flowers": "flower", "mushrooms": "mushroom",
    "berries": "berry", "legumes": "legume", "spices": "spice",
    "roots": "root", "seeds": "seed", "vines": "vine",
    "cacti": "cactus", "ferns": "fern",
    # Vehicles/Tools
    "vehicles": "vehicle", "instruments": "instrument", "gadgets": "gadget",
    "machines": "machine", "devices": "device", "appliances": "appliance",
    "drones": "drone", "rockets": "rocket", "submarines": "submarine",
    "tractors": "tractor", "turbines": "turbine", "lasers": "laser",
    "sensors": "sensor", "robots": "robot",
    # Nature/Geography
    "mountains": "mountain", "rivers": "river", "islands": "island",
    "glaciers": "glacier", "deserts": "desert", "forests": "forest",
    "volcanoes": "volcano", "canyons": "canyon", "marshes": "marsh",
    "plains": "plain", "reefs": "reef", "geysers": "geyser",
    "fjords": "fjord", "tundras": "tundra",
}

def singularize(word: str) -> str:
    """Convert plural noun to singular using explicit mapping."""
    return SINGULAR_FORMS.get(word, word)

def article(word: str) -> str:
    """Return 'an' before vowel sounds, 'a' otherwise."""
    if word and word[0].lower() in "aeiou":
        return "an"
    return "a"

def _sg(word: str) -> str:
    """Shorthand: singularize."""
    return singularize(word)

def _art_sg(word: str) -> str:
    """Shorthand: article + singular form."""
    s = singularize(word)
    return f"{article(s)} {s}"


# ── Natural Language Phrasing Templates ──────────────────────────────────────
PHRASING = {
    "All": [
        lambda s, o: f"All {s} are {o}.",
        lambda s, o: f"Every {_sg(s)} is {_art_sg(o)}.",
    ],
    "Some": [
        lambda s, o: f"Some {s} are {o}.",
        lambda s, o: f"There exist {s} that are {o}.",
    ],
    "No": [
        lambda s, o: f"No {s} are {o}.",
        lambda s, o: f"No {_sg(s)} is {_art_sg(o)}.",
    ],
}

QUERY_PHRASING = {
    "all": [
        lambda s, o: f"all {s} are {o}",
        lambda s, o: f"every {_sg(s)} is {_art_sg(o)}",
    ],
    "some": [
        lambda s, o: f"some {s} are {o}",
        lambda s, o: f"there exist {s} that are {o}",
    ],
    "no": [
        lambda s, o: f"no {s} are {o}",
    ],
    "some_not": [
        lambda s, o: f"some {s} are not {o}",
    ],
}


def sample_entities(n_entities: int, rng: random.Random) -> list[str]:
    """Sample n_entities from different semantic categories."""
    categories = list(ALL_CATEGORY_NAMES)
    rng.shuffle(categories)
    entities = []
    cat_idx = 0
    used_names: set[str] = set()
    while len(entities) < n_entities:
        cat = categories[cat_idx % len(categories)]
        pool = [n for n in ENTITY_POOLS[cat] if n not in used_names]
        if pool:
            name = rng.choice(pool)
            entities.append(name)
            used_names.add(name)
        cat_idx += 1
        if cat_idx > n_entities * len(categories):
            # Fallback: pick from any unused name
            all_unused = [
                n for cat_names in ENTITY_POOLS.values()
                for n in cat_names if n not in used_names
            ]
            if all_unused:
                name = rng.choice(all_unused)
                entities.append(name)
                used_names.add(name)
            else:
                raise RuntimeError(f"Not enough unique entity names for {n_entities} entities")
    return entities


def get_quantifier_pattern(template: str, d: int) -> list[str]:
    """Get quantifier list for a chain of d premises given template."""
    if template == "A":
        return ["All"] * d
    elif template == "B":
        return ["Some"] + ["All"] * (d - 1)
    elif template == "C":
        return ["All"] * (d - 1) + ["No"]
    elif template == "D":
        return ["Some"] + ["All"] * (d - 2) + ["No"]
    else:
        raise ValueError(f"Unknown template: {template}")


def get_conclusion_type(template: str) -> str:
    """Get conclusion quantifier type for a template."""
    return {"A": "all", "B": "some", "C": "no", "D": "some_not"}[template]


def get_contradiction_type(conclusion_type: str) -> str:
    """Square of Opposition: get the contradictory quantifier type."""
    return {
        "all": "some_not",
        "some": "no",
        "no": "some",
        "some_not": "all",
    }[conclusion_type]


def format_premise(quantifier: str, subject: str, obj: str, rng: random.Random) -> str:
    """Format a premise in natural language with random phrasing."""
    templates = PHRASING[quantifier]
    return rng.choice(templates)(subject, obj)


def format_query(quantifier_type: str, subject: str, obj: str, rng: random.Random) -> str:
    """Format a query conclusion in natural language."""
    templates = QUERY_PHRASING[quantifier_type]
    return rng.choice(templates)(subject, obj)


def verify_with_z3(
    premises_structured: list[tuple[str, str, str]],
    conclusion: tuple[str, str, str],
    is_true_problem: bool,
    timeout_ms: int = 10000,
) -> bool:
    """Verify a problem using Z3 BitVec set-theoretic encoding.

    Returns True if verification passes, False otherwise.
    """
    from z3 import BitVec, Solver, sat, unsat

    s = Solver()
    s.set("timeout", timeout_ms)

    # Collect all entity names
    all_names: set[str] = set()
    for q, subj, obj in premises_structured:
        all_names.add(subj)
        all_names.add(obj)
    all_names.add(conclusion[1])
    all_names.add(conclusion[2])

    # Create BitVec for each entity type
    entity_sets = {name: BitVec(name, UNIVERSE_SIZE) for name in all_names}

    # Non-emptiness constraint
    for name, bv in entity_sets.items():
        s.add(bv != 0)

    # Encode premises
    for quantifier, subject, obj in premises_structured:
        S = entity_sets[subject]
        O = entity_sets[obj]
        if quantifier == "All":
            s.add((S & ~O) == 0)        # S ⊆ O
        elif quantifier == "No":
            s.add((S & O) == 0)          # S ∩ O = ∅
        elif quantifier == "Some":
            s.add((S & O) != 0)          # S ∩ O ≠ ∅

    # Encode NEGATION of conclusion
    cq, cs, co = conclusion
    S = entity_sets[cs]
    O = entity_sets[co]
    if cq == "All":
        neg_conclusion = (S & ~O) != 0
    elif cq == "No":
        neg_conclusion = (S & O) != 0
    elif cq == "Some":
        neg_conclusion = (S & O) == 0
    elif cq == "Some_not":
        neg_conclusion = (S & ~O) == 0
    else:
        raise ValueError(f"Unknown quantifier: {cq}")

    s.add(neg_conclusion)
    result = s.check()

    if is_true_problem:
        return result == unsat  # TRUE: conclusion IS entailed (negation is unsat)
    else:
        return result == sat    # FALSE: conclusion is NOT entailed (negation is sat)


def generate_problem(
    difficulty: int,
    template: str,
    is_true: bool,
    problem_id: int,
    seed: int,
) -> dict | None:
    """Generate a single syllogistic logic problem.

    Returns problem dict or None if verification fails.
    """
    rng = random.Random(seed)
    d = difficulty
    n_entities = d + 1

    # Sample entities
    entities = sample_entities(n_entities, rng)

    # Get quantifier pattern
    quantifiers = get_quantifier_pattern(template, d)

    # Build structured premises (chain order)
    premises_structured = []
    for i in range(d):
        premises_structured.append((quantifiers[i], entities[i], entities[i + 1]))

    # Determine conclusion
    conclusion_type = get_conclusion_type(template)
    subject = entities[0]
    obj = entities[-1]

    if is_true:
        # Query states the correct conclusion
        query_type = conclusion_type
        query_quantifier = {
            "all": "All", "some": "Some", "no": "No", "some_not": "Some_not"
        }[conclusion_type]
        conclusion_for_z3 = (query_quantifier, subject, obj)
    else:
        # Query states the contradictory
        query_type = get_contradiction_type(conclusion_type)
        query_quantifier = {
            "all": "All", "some": "Some", "no": "No", "some_not": "Some_not"
        }[query_type]
        conclusion_for_z3 = (query_quantifier, subject, obj)

    # Verify with Z3
    verified = verify_with_z3(premises_structured, conclusion_for_z3, is_true)
    if not verified:
        return None

    # Shuffle premise order
    premise_indices = list(range(d))
    rng.shuffle(premise_indices)
    shuffled_premises = [premises_structured[i] for i in premise_indices]

    # Format as natural language
    premise_lines = []
    for idx, (q, s, o) in enumerate(shuffled_premises, 1):
        premise_lines.append(f"{idx}. {format_premise(q, s, o, rng)}")

    query_text = format_query(query_type, subject, obj, rng)

    input_text = (
        "Consider the following statements:\n"
        + "\n".join(premise_lines)
        + f"\n\nBased on the above statements, is it true that {query_text}? Answer TRUE or FALSE."
    )

    output_text = "TRUE" if is_true else "FALSE"

    # Quantifier pattern string (chain order)
    pattern_str = "-".join(quantifiers)

    return {
        "input": input_text,
        "output": output_text,
        "metadata_fold": "test",
        "metadata_difficulty": d,
        "metadata_num_premises": d,
        "metadata_chain_depth": d,
        "metadata_quantifier_pattern": pattern_str,
        "metadata_template": template,
        "metadata_conclusion_type": conclusion_type if is_true else query_type,
        "metadata_conclusion_truth": is_true,
        "metadata_entities": entities,
        "metadata_query_subject": subject,
        "metadata_query_object": obj,
        "metadata_chain_order": entities,
        "metadata_premise_order_shuffled": True,
        "metadata_problem_id": problem_id,
    }


def generate_problem_with_retries(
    difficulty: int,
    template: str,
    is_true: bool,
    problem_id: int,
    base_seed: int,
    max_retries: int = 5,
) -> dict:
    """Generate a problem with retry logic."""
    for attempt in range(max_retries):
        seed = base_seed + attempt * 10000
        result = generate_problem(difficulty, template, is_true, problem_id, seed)
        if result is not None:
            return result
        logger.warning(f"Problem {problem_id} (d={difficulty}, {template}, {'T' if is_true else 'F'}) "
                      f"failed Z3 verification on attempt {attempt+1}")
    raise RuntimeError(
        f"Failed to generate problem {problem_id} after {max_retries} retries"
    )


def heuristic_baseline(problem: dict) -> str:
    """Simple keyword heuristic: if query says 'All' and any premise says 'No', answer FALSE."""
    input_text = problem["input"]
    lines = input_text.split("\n")

    # Extract query line
    query_line = ""
    for line in lines:
        if "Based on the above" in line:
            query_line = line
            break

    # Check if query mentions 'all'
    query_mentions_all = "all " in query_line.lower() or "every " in query_line.lower()

    # Check if any premise mentions 'no'
    premise_mentions_no = False
    for line in lines:
        if line.strip() and line[0].isdigit():
            if line.lower().startswith(f"{line.split('.')[0]}.") and ("no " in line.lower().split(". ", 1)[-1][:5]):
                premise_mentions_no = True
                break

    if query_mentions_all and premise_mentions_no:
        return "FALSE"
    return "TRUE"


def count_chain_reconstruction_steps(d: int) -> int:
    """Estimate entity-matching steps to reconstruct chain from shuffled premises."""
    # With d shuffled premises, reconstructing the chain requires:
    # Finding the start entity, then matching d-1 intermediate links
    # Average comparisons: sum of (d-i) for i in 1..d-1 ≈ d*(d-1)/2
    return d * (d - 1) // 2


@logger.catch
def main():
    logger.info("=" * 60)
    logger.info("Syllogistic Logic Dataset Generator")
    logger.info("=" * 60)

    # ── Generate all 280 problems ────────────────────────────────────────
    all_problems: list[dict] = []
    problem_id = 0
    generation_tasks = []

    for d in DIFFICULTY_RANGE:
        # For each template: 5 problems. TRUE/FALSE balance: 10/10 per level
        # Template A: 3 TRUE, 2 FALSE
        # Template B: 2 TRUE, 3 FALSE
        # Template C: 3 TRUE, 2 FALSE
        # Template D: 2 TRUE, 3 FALSE
        # Total: 10 TRUE + 10 FALSE per level
        true_false_per_template = {
            "A": (3, 2),
            "B": (2, 3),
            "C": (3, 2),
            "D": (2, 3),
        }

        for template in TEMPLATES:
            n_true, n_false = true_false_per_template[template]
            for _ in range(n_true):
                generation_tasks.append((d, template, True, problem_id))
                problem_id += 1
            for _ in range(n_false):
                generation_tasks.append((d, template, False, problem_id))
                problem_id += 1

    total_tasks = len(generation_tasks)
    logger.info(f"Generating {total_tasks} problems across {len(list(DIFFICULTY_RANGE))} difficulty levels...")

    # Use ProcessPoolExecutor for parallel Z3 verification
    num_workers = max(1, NUM_CPUS - 1)
    logger.info(f"Using {num_workers} parallel workers")

    results = [None] * total_tasks
    failed = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for idx, (d, template, is_true, pid) in enumerate(generation_tasks):
            base_seed = SEED + pid * 100
            future = executor.submit(
                generate_problem_with_retries, d, template, is_true, pid, base_seed
            )
            futures[future] = idx

        for i, future in enumerate(as_completed(futures)):
            idx = futures[future]
            try:
                result = future.result()
                results[idx] = result
            except Exception:
                logger.exception(f"Task {idx} failed")
                failed += 1

            if (i + 1) % 50 == 0:
                logger.info(f"Progress: {i + 1}/{total_tasks} problems generated")

    all_problems = [r for r in results if r is not None]
    logger.info(f"Generated {len(all_problems)} problems ({failed} failures)")

    if len(all_problems) != total_tasks:
        logger.error(f"Expected {total_tasks} problems, got {len(all_problems)}")

    # ── Validation Checks ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Running validation checks...")

    # 6c. Balance verification
    by_difficulty = defaultdict(list)
    for p in all_problems:
        by_difficulty[p["metadata_difficulty"]].append(p)

    logger.info(f"\n{'d':>3} | {'Total':>5} | {'TRUE':>5} | {'FALSE':>5} | {'A':>3} {'B':>3} {'C':>3} {'D':>3} | {'Heuristic':>9}")
    logger.info("-" * 65)

    heuristic_acc_by_level = {}
    for d in sorted(by_difficulty.keys()):
        problems = by_difficulty[d]
        n_true = sum(1 for p in problems if p["output"] == "TRUE")
        n_false = sum(1 for p in problems if p["output"] == "FALSE")

        template_counts = defaultdict(int)
        for p in problems:
            template_counts[p["metadata_template"]] += 1

        # Heuristic baseline
        correct = sum(1 for p in problems if heuristic_baseline(p) == p["output"])
        acc = correct / len(problems)
        heuristic_acc_by_level[d] = acc

        logger.info(
            f"{d:>3} | {len(problems):>5} | {n_true:>5} | {n_false:>5} | "
            f"{template_counts.get('A', 0):>3} {template_counts.get('B', 0):>3} "
            f"{template_counts.get('C', 0):>3} {template_counts.get('D', 0):>3} | "
            f"{acc:>8.1%}"
        )

    # Check unique entity name sets
    entity_sets = [frozenset(p["metadata_entities"]) for p in all_problems]
    unique_sets = len(set(entity_sets))
    logger.info(f"\nUnique entity name sets: {unique_sets}/{len(all_problems)}")

    # Chain reconstruction difficulty
    logger.info(f"\nChain reconstruction steps by difficulty:")
    for d in sorted(by_difficulty.keys()):
        steps = count_chain_reconstruction_steps(d)
        logger.info(f"  d={d}: ~{steps} matching steps")

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total problems: {len(all_problems)}")
    logger.info(f"Difficulty levels: {len(by_difficulty)} (d={min(by_difficulty.keys())}-{max(by_difficulty.keys())})")
    logger.info(f"Z3 verification: {len(all_problems)}/{total_tasks} passed")
    logger.info(f"Unique entity sets: {unique_sets}/{len(all_problems)}")

    # ── Output in exp_sel_data_out.json schema ───────────────────────────
    output = {
        "datasets": [
            {
                "dataset": "syllogistic_logic",
                "examples": all_problems,
            }
        ]
    }

    out_path = Path("full_data_out.json")
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    logger.info(f"Saved to {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

    # Also save the flat version as data_out.json for direct use
    flat_output = []
    for p in all_problems:
        flat_problem = {
            "input": p["input"],
            "output": p["output"],
            "metadata_fold": p["metadata_fold"],
            "difficulty": p["metadata_difficulty"],
            "num_premises": p["metadata_num_premises"],
            "chain_depth": p["metadata_chain_depth"],
            "quantifier_pattern": p["metadata_quantifier_pattern"],
            "template": p["metadata_template"],
            "conclusion_type": p["metadata_conclusion_type"],
            "conclusion_truth": p["metadata_conclusion_truth"],
            "entities": p["metadata_entities"],
            "query_subject": p["metadata_query_subject"],
            "query_object": p["metadata_query_object"],
            "chain_order": p["metadata_chain_order"],
            "premise_order_shuffled": p["metadata_premise_order_shuffled"],
        }
        flat_output.append(flat_problem)

    data_out_path = Path("data_out.json")
    data_out_path.write_text(json.dumps(flat_output, indent=2, ensure_ascii=False))
    logger.info(f"Saved flat version to {data_out_path} ({data_out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
