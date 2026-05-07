#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["loguru"]
# ///
"""Transform generated graph coloring data_out.json into exp_sel_data_out.json schema.

Loads the synthetically generated dataset and standardizes it into the
experiment selection data output format with proper metadata fields.

Produces one dataset group:
  1. "graph_coloring_full" — all 400 problems across 20 difficulty levels
"""

import json
import sys
from pathlib import Path

from loguru import logger

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
WORKSPACE = Path("/ai-inventor/aii_pipeline/runs/neurips-open/3_invention_loop/iter_1/gen_art/data_id4_it1__opus")
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOG_DIR / "data.log"), rotation="30 MB", level="DEBUG")


def transform_record(record: dict, row_index: int) -> dict:
    """Transform a single data_out.json record into exp_sel_data_out schema example."""
    return {
        "input": record["input"],
        "output": record["output"],
        "metadata_difficulty_level": record["difficulty_level"],
        "metadata_num_nodes": record["num_nodes"],
        "metadata_num_edges": record["num_edges"],
        "metadata_num_colors": record["num_colors"],
        "metadata_edge_density": record["edge_density"],
        "metadata_graph_adjacency": record["graph_adjacency"],
        "metadata_fold": record["metadata_fold"],
        "metadata_row_index": row_index,
        "metadata_task_type": "constraint_satisfaction",
    }


@logger.catch
def main() -> None:
    # Load generated dataset
    data_path = WORKSPACE / "data_out.json"
    logger.info(f"Loading data from {data_path}")

    try:
        raw_data = json.loads(data_path.read_text())
    except FileNotFoundError:
        logger.exception(f"Data file not found: {data_path}")
        raise
    except json.JSONDecodeError:
        logger.exception(f"Invalid JSON in: {data_path}")
        raise

    logger.info(f"Loaded {len(raw_data)} records from data_out.json")

    # Dataset 1: Full dataset (all 400 problems, levels 1-20)
    full_examples = []
    for idx, record in enumerate(raw_data):
        example = transform_record(record, row_index=idx)
        full_examples.append(example)

    logger.info(f"Dataset 'graph_coloring_full': {len(full_examples)} examples")

    # Build output in exp_sel_data_out schema (single dataset)
    output = {
        "metadata": {
            "source": "synthetic_generation",
            "description": "Graph Coloring Constraint Satisfaction Dataset with Parameterized Difficulty",
            "generation_method": "Erdos-Renyi random graphs with exact backtracking k-coloring solver",
            "total_problems": len(raw_data),
            "difficulty_levels": 20,
            "problems_per_level": 20,
            "node_range": "4-12",
            "color_range": "3-4",
        },
        "datasets": [
            {
                "dataset": "graph_coloring_full",
                "examples": full_examples,
            },
        ],
    }

    # Write output
    out_path = WORKSPACE / "full_data_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    file_size_kb = out_path.stat().st_size / 1024
    logger.info(f"Wrote {out_path} ({file_size_kb:.1f} KB)")

    # Validation summary
    total_examples = sum(len(d["examples"]) for d in output["datasets"])
    logger.info(f"Total examples across all datasets: {total_examples}")
    for ds in output["datasets"]:
        logger.info(f"  {ds['dataset']}: {len(ds['examples'])} examples")

    # Quick sanity check on first and last example
    first = output["datasets"][0]["examples"][0]
    last = output["datasets"][0]["examples"][-1]
    logger.info(f"First example: difficulty={first['metadata_difficulty_level']}, nodes={first['metadata_num_nodes']}")
    logger.info(f"Last example: difficulty={last['metadata_difficulty_level']}, nodes={last['metadata_num_nodes']}")
    logger.info(f"First input (truncated): {first['input'][:100]}...")
    logger.info(f"First output: {first['output']}")


if __name__ == "__main__":
    main()
