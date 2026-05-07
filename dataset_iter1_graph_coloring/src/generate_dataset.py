#!/usr/bin/env python3
"""Generate Graph Coloring Constraint Satisfaction Dataset with Parameterized Difficulty.

Produces 400 graph coloring problems (20 difficulty levels x 20 problems each)
using Erdos-Renyi random graphs. Every instance is verified solvable by an exact
backtracking solver, with connected-graph and uniqueness filters.
"""

import json
import math
import os
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
from loguru import logger

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
WORKSPACE = Path("/ai-inventor/aii_pipeline/runs/neurips-open/3_invention_loop/iter_1/gen_art/data_id4_it1__opus")
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOG_DIR / "generate.log"), rotation="30 MB", level="DEBUG")

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


def _container_ram_gb() -> Optional[float]:
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None


NUM_CPUS = _detect_cpus()
TOTAL_RAM_GB = _container_ram_gb() or 57.0

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM")

# Set memory limit (this task is lightweight, 4GB is generous)
RAM_BUDGET = int(4 * 1024**3)  # 4 GB
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))  # 1 hour CPU time

# ---------------------------------------------------------------------------
# Difficulty table (from artifact plan - EXACT parameters)
# ---------------------------------------------------------------------------
# (level, N, p, K)
DIFFICULTY_TABLE: list[tuple[int, int, float, int]] = [
    (1,  4,  0.30, 4),
    (2,  4,  0.50, 4),
    (3,  4,  0.60, 3),
    (4,  5,  0.30, 4),
    (5,  5,  0.40, 3),
    (6,  5,  0.60, 3),
    (7,  6,  0.30, 3),
    (8,  6,  0.50, 3),
    (9,  7,  0.30, 3),
    (10, 7,  0.40, 3),
    (11, 7,  0.55, 3),
    (12, 8,  0.30, 3),
    (13, 8,  0.40, 3),
    (14, 8,  0.50, 3),
    (15, 9,  0.35, 3),
    (16, 9,  0.45, 3),
    (17, 10, 0.30, 3),
    (18, 10, 0.40, 3),
    (19, 11, 0.35, 3),
    (20, 12, 0.35, 3),
]

PROBLEMS_PER_LEVEL = 20
MAX_ATTEMPTS_PER_PROBLEM = 200

# Color name mappings
COLOR_NAMES_3 = ["Red", "Green", "Blue"]
COLOR_NAMES_4 = ["Red", "Green", "Blue", "Yellow"]


# ---------------------------------------------------------------------------
# Exact backtracking k-coloring solver
# ---------------------------------------------------------------------------
def find_k_coloring(graph: nx.Graph, k: int) -> Optional[dict[int, int]]:
    """Backtracking solver: returns a valid k-coloring dict {node: color} or None."""
    nodes = sorted(graph.nodes())
    adj = {n: set(graph.neighbors(n)) for n in nodes}
    coloring: dict[int, int] = {}

    def backtrack(idx: int) -> bool:
        if idx == len(nodes):
            return True
        node = nodes[idx]
        for color in range(k):
            if all(coloring.get(neighbor) != color for neighbor in adj[node]):
                coloring[node] = color
                if backtrack(idx + 1):
                    return True
                del coloring[node]
        return False

    if backtrack(0):
        return dict(coloring)
    return None


def verify_coloring(graph: nx.Graph, coloring: dict[int, int], k: int) -> bool:
    """Check: all nodes colored, all colors in range, no adjacent same-color."""
    if set(coloring.keys()) != set(graph.nodes()):
        return False
    if any(c < 0 or c >= k for c in coloring.values()):
        return False
    for u, v in graph.edges():
        if coloring[u] == coloring[v]:
            return False
    return True


# ---------------------------------------------------------------------------
# Problem text formatting
# ---------------------------------------------------------------------------
def format_problem_text(n: int, edges: list[tuple[int, int]], k: int) -> str:
    """Create natural language problem statement."""
    edge_strs = [f"(Node {u}, Node {v})" for u, v in edges]
    edge_list = ", ".join(edge_strs)

    color_names = COLOR_NAMES_4[:k] if k == 4 else COLOR_NAMES_3[:k]
    color_str = ", ".join(color_names)

    text = (
        f"Given a graph with {n} nodes (labeled Node 0 through Node {n - 1}) "
        f"and the following edges:\n"
        f"{edge_list}\n\n"
        f"Color each node using exactly one of these colors: {color_str}.\n"
        f"The constraint is that no two nodes connected by an edge may share the same color.\n\n"
        f"Provide a valid coloring by listing each node and its assigned color."
    )
    return text


def format_coloring_output(coloring: dict[int, int], k: int) -> str:
    """Format coloring as ground truth output string."""
    color_names = COLOR_NAMES_4[:k] if k == 4 else COLOR_NAMES_3[:k]
    parts = [f"Node {node}: {color_names[color]}" for node, color in sorted(coloring.items())]
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Generate problems for a single difficulty level
# ---------------------------------------------------------------------------
def generate_level(level: int, n: int, p: float, k: int) -> list[dict]:
    """Generate PROBLEMS_PER_LEVEL valid graph coloring problems for a given level."""
    problems: list[dict] = []
    seen_edge_sets: set[frozenset[frozenset[int]]] = set()
    attempt = 0
    current_p = p

    while len(problems) < PROBLEMS_PER_LEVEL:
        attempt += 1

        if attempt > MAX_ATTEMPTS_PER_PROBLEM * PROBLEMS_PER_LEVEL:
            # Reduce density and retry
            logger.warning(f"Level {level}: exhausted {attempt} attempts at p={current_p:.2f}, reducing p by 0.05")
            current_p = max(current_p - 0.05, 0.10)
            attempt = 0
            seen_edge_sets.clear()
            problems.clear()
            continue

        seed = level * 10000 + attempt
        G = nx.erdos_renyi_graph(n, current_p, seed=seed)

        # Filter 1: Must be connected
        if not nx.is_connected(G):
            continue

        # Filter 2: Must have at least 2 edges
        if G.number_of_edges() < 2:
            continue

        # Filter 3: Must be k-colorable (exact solver)
        coloring = find_k_coloring(G, k)
        if coloring is None:
            continue

        # Filter 4: No duplicate edge sets within this level
        edge_set = frozenset(frozenset(e) for e in G.edges())
        if edge_set in seen_edge_sets:
            continue
        seen_edge_sets.add(edge_set)

        # Filter 5: Verify the coloring independently
        assert verify_coloring(G, coloring, k), f"Coloring verification failed for level {level}, attempt {attempt}"

        # Build the record
        edges_sorted = sorted([sorted(e) for e in G.edges()])
        problem_text = format_problem_text(n, [(u, v) for u, v in edges_sorted], k)
        coloring_str = format_coloring_output(coloring, k)

        record = {
            "input": problem_text,
            "output": coloring_str,
            "difficulty_level": level,
            "num_nodes": n,
            "num_edges": G.number_of_edges(),
            "num_colors": k,
            "edge_density": round(current_p, 2),
            "graph_adjacency": edges_sorted,
            "metadata_fold": f"level_{level:02d}",
        }
        problems.append(record)

    logger.info(f"Level {level:2d} (N={n}, p={current_p:.2f}, K={k}): generated {len(problems)} problems in {attempt} attempts")
    return problems


# ---------------------------------------------------------------------------
# Validation suite
# ---------------------------------------------------------------------------
def validate_dataset(data: list[dict]) -> bool:
    """Run all 7 validation checks on the generated dataset."""
    logger.info("=" * 60)
    logger.info("Running post-generation validation...")
    all_ok = True

    # 1. Count check
    if len(data) != 400:
        logger.error(f"[FAIL] Count check: expected 400 records, got {len(data)}")
        all_ok = False
    else:
        logger.info("[PASS] Count check: 400 records")

    # Check 20 per level
    from collections import Counter
    level_counts = Counter(r["difficulty_level"] for r in data)
    for lvl in range(1, 21):
        if level_counts.get(lvl, 0) != 20:
            logger.error(f"[FAIL] Level {lvl}: expected 20 records, got {level_counts.get(lvl, 0)}")
            all_ok = False
    if all(level_counts.get(lvl, 0) == 20 for lvl in range(1, 21)):
        logger.info("[PASS] Per-level count: 20 per level")

    # 2. Solvability re-verification
    verify_failures = 0
    for i, record in enumerate(data):
        G = nx.Graph()
        G.add_nodes_from(range(record["num_nodes"]))
        G.add_edges_from([tuple(e) for e in record["graph_adjacency"]])

        # Parse output coloring
        color_map = COLOR_NAMES_4 if record["num_colors"] == 4 else COLOR_NAMES_3
        coloring = {}
        for part in record["output"].split(", "):
            node_str, color_str = part.split(": ")
            node_id = int(node_str.replace("Node ", ""))
            color_id = color_map.index(color_str)
            coloring[node_id] = color_id

        if not verify_coloring(G, coloring, record["num_colors"]):
            logger.error(f"[FAIL] Solvability: record {i} (level {record['difficulty_level']}) has invalid coloring")
            verify_failures += 1

    if verify_failures == 0:
        logger.info("[PASS] Solvability re-verification: all 400 colorings valid")
    else:
        logger.error(f"[FAIL] Solvability: {verify_failures} invalid colorings")
        all_ok = False

    # 3. Connectivity check
    connectivity_failures = 0
    for i, record in enumerate(data):
        G = nx.Graph()
        G.add_nodes_from(range(record["num_nodes"]))
        G.add_edges_from([tuple(e) for e in record["graph_adjacency"]])
        if not nx.is_connected(G):
            logger.error(f"[FAIL] Connectivity: record {i} is disconnected")
            connectivity_failures += 1

    if connectivity_failures == 0:
        logger.info("[PASS] Connectivity: all 400 graphs are connected")
    else:
        logger.error(f"[FAIL] Connectivity: {connectivity_failures} disconnected graphs")
        all_ok = False

    # 4. Difficulty monotonicity (avg edges and nodes non-decreasing)
    level_avg_edges: dict[int, float] = {}
    level_avg_nodes: dict[int, float] = {}
    for lvl in range(1, 21):
        lvl_records = [r for r in data if r["difficulty_level"] == lvl]
        level_avg_edges[lvl] = sum(r["num_edges"] for r in lvl_records) / len(lvl_records)
        level_avg_nodes[lvl] = sum(r["num_nodes"] for r in lvl_records) / len(lvl_records)

    mono_ok = True
    for lvl in range(2, 21):
        if level_avg_nodes[lvl] < level_avg_nodes[lvl - 1]:
            logger.warning(f"[WARN] Monotonicity: avg nodes decrease at level {lvl} ({level_avg_nodes[lvl-1]:.1f} -> {level_avg_nodes[lvl]:.1f})")
            mono_ok = False

    if mono_ok:
        logger.info("[PASS] Difficulty monotonicity: avg nodes non-decreasing")
    else:
        logger.warning("[WARN] Difficulty monotonicity: some levels have decreasing avg nodes (may be expected for density changes)")

    # 5. No trivial instances (every graph has >= 2 edges)
    trivial_count = sum(1 for r in data if r["num_edges"] < 2)
    if trivial_count == 0:
        logger.info("[PASS] No trivial instances: all graphs have >= 2 edges")
    else:
        logger.error(f"[FAIL] Trivial instances: {trivial_count} graphs have < 2 edges")
        all_ok = False

    # 6. Color count check
    color_errors = 0
    for i, record in enumerate(data):
        valid_colors = set(COLOR_NAMES_4[:record["num_colors"]] if record["num_colors"] == 4 else COLOR_NAMES_3[:record["num_colors"]])
        for part in record["output"].split(", "):
            _, color_str = part.split(": ")
            if color_str not in valid_colors:
                logger.error(f"[FAIL] Color check: record {i} uses invalid color '{color_str}'")
                color_errors += 1

    if color_errors == 0:
        logger.info("[PASS] Color count: all outputs use valid palette colors")
    else:
        logger.error(f"[FAIL] Color check: {color_errors} invalid color usages")
        all_ok = False

    # 7. Schema check
    required_fields = {"input", "output", "difficulty_level", "num_nodes", "num_edges",
                       "num_colors", "edge_density", "graph_adjacency", "metadata_fold"}
    schema_errors = 0
    for i, record in enumerate(data):
        missing = required_fields - set(record.keys())
        if missing:
            logger.error(f"[FAIL] Schema: record {i} missing fields: {missing}")
            schema_errors += 1
        # Type checks
        if not isinstance(record.get("difficulty_level"), int):
            schema_errors += 1
        if not isinstance(record.get("num_nodes"), int):
            schema_errors += 1
        if not isinstance(record.get("num_edges"), int):
            schema_errors += 1
        if not isinstance(record.get("num_colors"), int):
            schema_errors += 1
        if not isinstance(record.get("edge_density"), float):
            schema_errors += 1
        if not isinstance(record.get("graph_adjacency"), list):
            schema_errors += 1
        if not isinstance(record.get("metadata_fold"), str):
            schema_errors += 1
        if not isinstance(record.get("input"), str):
            schema_errors += 1
        if not isinstance(record.get("output"), str):
            schema_errors += 1

    if schema_errors == 0:
        logger.info("[PASS] Schema: all fields present with correct types")
    else:
        logger.error(f"[FAIL] Schema: {schema_errors} errors")
        all_ok = False

    # Summary
    logger.info("=" * 60)
    if all_ok:
        logger.info("ALL VALIDATION CHECKS PASSED")
    else:
        logger.error("SOME VALIDATION CHECKS FAILED")
    logger.info("=" * 60)

    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
@logger.catch
def main():
    start_time = time.time()

    # Parse optional --levels arg for gradual scaling
    max_levels = 20
    if "--levels" in sys.argv:
        idx = sys.argv.index("--levels")
        max_levels = int(sys.argv[idx + 1])
        logger.info(f"Running with --levels={max_levels} (subset mode)")

    levels_to_generate = DIFFICULTY_TABLE[:max_levels]
    total_problems = PROBLEMS_PER_LEVEL * len(levels_to_generate)
    logger.info(f"Generating {total_problems} graph coloring problems across {len(levels_to_generate)} levels")

    # Use ProcessPoolExecutor for parallel generation across levels
    num_workers = min(NUM_CPUS, len(levels_to_generate))
    logger.info(f"Using {num_workers} parallel workers")

    all_records: list[dict] = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for level, n, p, k in levels_to_generate:
            future = executor.submit(generate_level, level, n, p, k)
            futures[future] = level

        for future in as_completed(futures):
            level = futures[future]
            try:
                records = future.result()
                all_records.extend(records)
            except Exception:
                logger.exception(f"Failed to generate level {level}")
                raise

    # Group by level, preserve generation order within each level
    grouped: dict[int, list[dict]] = {}
    for r in all_records:
        grouped.setdefault(r["difficulty_level"], []).append(r)
    all_records = []
    for lvl in sorted(grouped.keys()):
        all_records.extend(grouped[lvl])

    elapsed = time.time() - start_time
    logger.info(f"Generation complete: {len(all_records)} records in {elapsed:.1f}s")

    # Write output
    out_path = WORKSPACE / "data_out.json"
    out_path.write_text(json.dumps(all_records, indent=2))
    logger.info(f"Wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

    # Run validation (only on full dataset)
    if max_levels == 20 and len(all_records) == 400:
        validate_dataset(all_records)
    else:
        logger.info(f"Skipping full validation (subset mode: {len(all_records)} records)")
        # Quick sanity checks
        logger.info(f"  Levels generated: {sorted(grouped.keys())}")
        logger.info(f"  Records per level: {[len(grouped[k]) for k in sorted(grouped.keys())]}")

    # Print sample record
    if all_records:
        logger.info("Sample record (level 1):")
        sample = all_records[0]
        for key in ["difficulty_level", "num_nodes", "num_edges", "num_colors", "edge_density", "metadata_fold"]:
            logger.info(f"  {key}: {sample[key]}")
        logger.info(f"  input: {sample['input'][:120]}...")
        logger.info(f"  output: {sample['output']}")

    return all_records


if __name__ == "__main__":
    main()
