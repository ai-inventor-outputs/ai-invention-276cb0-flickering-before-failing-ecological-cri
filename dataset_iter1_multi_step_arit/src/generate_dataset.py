#!/usr/bin/env python3
"""Generate Multi-Step Arithmetic Chain Dataset with Parameterized Difficulty.

Produces 480 synthetic arithmetic chain problems spanning 24 difficulty levels
(2-25 operations), with deterministic ground truth, bounded intermediate values,
and structured metadata for CSD analysis.
"""

import json
import math
import os
import random
import resource
import sys
from collections import Counter, defaultdict
from pathlib import Path

from loguru import logger

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------------------
# Hardware-aware resource limits (cgroup v1 container)
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

# Set conservative RAM limit (this task is tiny - 1GB is more than enough)
RAM_BUDGET = int(1 * 1024**3)  # 1 GB
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, budget=1 GB")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MASTER_SEED = 42
MIN_DIFFICULTY = 2
MAX_DIFFICULTY = 25
PROBLEMS_PER_LEVEL = 20
INTERMEDIATE_BOUND = 10000
OP_SYMBOLS = ["+", "-", "\u00d7"]  # +, -, x (multiplication sign)
OP_WORDS = {"+": "Add", "-": "Subtract", "\u00d7": "Multiply by"}


# ---------------------------------------------------------------------------
# Problem generator
# ---------------------------------------------------------------------------
def generate_problem(difficulty_level: int, seed: int) -> dict:
    """Generate a single arithmetic chain problem.

    Args:
        difficulty_level: Number of sequential operations (2-25).
        seed: Random seed for reproducibility.

    Returns:
        Dict with input, output, difficulty_level, operations, operands, etc.
    """
    rng = random.Random(seed)

    # Starting value: 2-digit integer
    start = rng.randint(10, 99)
    current_value = start
    intermediates = [start]
    ops: list[str] = []
    operands_list: list[int] = []
    fallback_count = 0

    for _ in range(difficulty_level):
        # Build candidate (operation, operand) pairs
        candidates: list[tuple[str, int]] = []

        # Addition candidates: operand in [10, 99]
        for operand in range(10, 100):
            if abs(current_value + operand) <= INTERMEDIATE_BOUND:
                candidates.append(("+", operand))

        # Subtraction candidates: operand in [10, 99]
        for operand in range(10, 100):
            if abs(current_value - operand) <= INTERMEDIATE_BOUND:
                candidates.append(("-", operand))

        # Multiplication candidates: operand in [2, 9], exclude if current is 0
        if current_value != 0:
            for operand in range(2, 10):
                if abs(current_value * operand) <= INTERMEDIATE_BOUND:
                    candidates.append(("\u00d7", operand))

        # Filter and select
        if not candidates:
            # Fallback: use +1 or -1
            fallback_count += 1
            if current_value > 0:
                op, operand = "-", 1
            else:
                op, operand = "+", 1
            logger.debug(f"Fallback at seed={seed}, step={len(ops)}, val={current_value}")
        else:
            op, operand = rng.choice(candidates)

        # Apply operation
        if op == "+":
            current_value = current_value + operand
        elif op == "-":
            current_value = current_value - operand
        elif op == "\u00d7":
            current_value = current_value * operand

        intermediates.append(current_value)
        ops.append(op)
        operands_list.append(operand)

    # Format problem text
    lines = [
        "Compute the following step by step, showing your work for each operation. "
        "What is the final result?",
        "",
        f"Start with {start}.",
    ]
    for i, (op, operand) in enumerate(zip(ops, operands_list), 1):
        lines.append(f"Step {i}: {OP_WORDS[op]} {operand}")
    lines.append("")
    lines.append("Provide your final numerical answer.")

    input_text = "\n".join(lines)

    return {
        "input": input_text,
        "output": str(current_value),
        "difficulty_level": difficulty_level,
        "difficulty_param_name": "num_operations",
        "all_intermediate_answers": intermediates,
        "operations": ops,
        "operands": operands_list,
        "seed": seed,
        "metadata_fold": "test",
    }


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify_problem(problem: dict) -> list[str]:
    """Verify a single problem's correctness. Returns list of error messages."""
    errors: list[str] = []
    d = problem["difficulty_level"]
    ops = problem["operations"]
    operands = problem["operands"]
    intermediates = problem["all_intermediate_answers"]

    # Check lengths
    if len(ops) != d:
        errors.append(f"ops length {len(ops)} != difficulty {d}")
    if len(operands) != d:
        errors.append(f"operands length {len(operands)} != difficulty {d}")
    if len(intermediates) != d + 1:
        errors.append(f"intermediates length {len(intermediates)} != {d + 1}")

    # Re-execute the chain
    current = intermediates[0]
    for i, (op, operand) in enumerate(zip(ops, operands)):
        if op == "+":
            current = current + operand
        elif op == "-":
            current = current - operand
        elif op == "\u00d7":
            current = current * operand
        else:
            errors.append(f"Unknown op: {op}")

        if i + 1 < len(intermediates) and current != intermediates[i + 1]:
            errors.append(
                f"Step {i+1}: expected {intermediates[i+1]}, got {current}"
            )

    # Check final answer
    if str(current) != problem["output"]:
        errors.append(f"Final answer mismatch: {current} != {problem['output']}")

    # Check bounds
    for j, val in enumerate(intermediates):
        if abs(val) > INTERMEDIATE_BOUND:
            errors.append(f"Intermediate {j} out of bounds: {val}")

    # Check valid operations
    for op in ops:
        if op not in OP_SYMBOLS:
            errors.append(f"Invalid operation: {op}")

    return errors


def verify_dataset(dataset: list[dict]) -> bool:
    """Run full verification on the dataset. Returns True if all checks pass."""
    all_ok = True

    # Per-problem verification
    for i, problem in enumerate(dataset):
        errs = verify_problem(problem)
        if errs:
            all_ok = False
            logger.error(f"Problem {i} (seed={problem['seed']}): {errs}")

    # Check for duplicate input texts
    input_texts = [p["input"] for p in dataset]
    if len(set(input_texts)) != len(input_texts):
        all_ok = False
        logger.error("Duplicate problem texts found!")
    else:
        logger.info("No duplicate problem texts.")

    # Check counts per difficulty level
    level_counts: dict[int, int] = Counter(p["difficulty_level"] for p in dataset)
    for d in range(MIN_DIFFICULTY, MAX_DIFFICULTY + 1):
        count = level_counts.get(d, 0)
        if count != PROBLEMS_PER_LEVEL:
            all_ok = False
            logger.error(f"Difficulty {d}: expected {PROBLEMS_PER_LEVEL}, got {count}")
    logger.info(f"Difficulty level counts: {dict(sorted(level_counts.items()))}")

    # Intermediate value statistics
    all_intermediates = [
        v for p in dataset for v in p["all_intermediate_answers"]
    ]
    logger.info(
        f"Intermediate values: min={min(all_intermediates)}, "
        f"max={max(all_intermediates)}, "
        f"mean={sum(all_intermediates)/len(all_intermediates):.1f}"
    )

    # Operation distribution per difficulty level
    op_dist_by_level: dict[int, Counter] = defaultdict(Counter)
    for p in dataset:
        for op in p["operations"]:
            op_dist_by_level[p["difficulty_level"]][op] += 1

    logger.info("Operation distribution per difficulty level:")
    for d in sorted(op_dist_by_level.keys()):
        dist = op_dist_by_level[d]
        total = sum(dist.values())
        pcts = {op: f"{dist[op]/total*100:.0f}%" for op in OP_SYMBOLS}
        logger.info(f"  d={d:2d}: {pcts} (total ops: {total})")

    # Fallback usage check (global)
    logger.info(f"Total problems: {len(dataset)}")

    return all_ok


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------
@logger.catch
def main(
    *,
    max_difficulty: int = MAX_DIFFICULTY,
    problems_per_level: int = PROBLEMS_PER_LEVEL,
    output_path: str = "data_out.json",
) -> None:
    """Generate the full dataset, verify, and save."""
    logger.info(
        f"Generating dataset: d=[{MIN_DIFFICULTY},{max_difficulty}], "
        f"{problems_per_level} per level"
    )

    dataset: list[dict] = []
    for d in range(MIN_DIFFICULTY, max_difficulty + 1):
        for i in range(problems_per_level):
            seed = MASTER_SEED * 10000 + d * 100 + i
            problem = generate_problem(difficulty_level=d, seed=seed)
            dataset.append(problem)

    logger.info(f"Generated {len(dataset)} problems. Running verification...")

    # Verification pass
    ok = verify_dataset(dataset)
    if not ok:
        logger.error("VERIFICATION FAILED - see errors above")
        sys.exit(1)
    logger.info("All verifications PASSED.")

    # Save output
    out_path = Path(output_path)
    out_path.write_text(json.dumps(dataset, indent=2, ensure_ascii=False))
    size_kb = out_path.stat().st_size / 1024
    logger.info(f"Saved {len(dataset)} problems to {out_path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
