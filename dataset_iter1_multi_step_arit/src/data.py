#!/usr/bin/env python3
"""Transform synthetic arithmetic chain dataset into exp_sel_data_out.json schema.

Loads data_out.json (480 arithmetic chain problems) and converts to the
standardized full_data_out.json format grouped by dataset with metadata_ prefixed fields.
"""

import json
import sys
from pathlib import Path

from loguru import logger

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/data.log", rotation="30 MB", level="DEBUG")

WORKSPACE = Path(__file__).parent
INPUT_FILE = WORKSPACE / "data_out.json"
OUTPUT_FILE = WORKSPACE / "full_data_out.json"


@logger.catch
def main() -> None:
    """Load raw dataset, transform to exp_sel_data_out schema, and save."""

    # ------------------------------------------------------------------
    # 1. Load raw synthetic data
    # ------------------------------------------------------------------
    logger.info(f"Loading raw data from {INPUT_FILE}")
    if not INPUT_FILE.exists():
        logger.error(f"Input file not found: {INPUT_FILE}")
        sys.exit(1)

    raw_data: list[dict] = json.loads(INPUT_FILE.read_text())
    logger.info(f"Loaded {len(raw_data)} raw problems")

    # ------------------------------------------------------------------
    # 2. Transform each row into schema-compliant example
    # ------------------------------------------------------------------
    examples: list[dict] = []
    for idx, row in enumerate(raw_data):
        try:
            example = {
                # Required fields
                "input": row["input"],
                "output": row["output"],
                # Metadata fields (all use metadata_ prefix)
                "metadata_fold": row["metadata_fold"],
                "metadata_difficulty_level": row["difficulty_level"],
                "metadata_difficulty_param_name": row["difficulty_param_name"],
                "metadata_all_intermediate_answers": row["all_intermediate_answers"],
                "metadata_operations": row["operations"],
                "metadata_operands": row["operands"],
                "metadata_seed": row["seed"],
                "metadata_row_index": idx,
                "metadata_task_type": "arithmetic_chain",
                "metadata_num_operations": row["difficulty_level"],
            }
            examples.append(example)
        except KeyError:
            logger.exception(f"Missing key in row {idx}")
            continue

    logger.info(f"Transformed {len(examples)} examples")

    # ------------------------------------------------------------------
    # 3. Assemble into exp_sel_data_out schema
    # ------------------------------------------------------------------
    output_data = {
        "datasets": [
            {
                "dataset": "arithmetic_chains",
                "examples": examples,
            }
        ]
    }

    # ------------------------------------------------------------------
    # 4. Validate structure before saving
    # ------------------------------------------------------------------
    ds = output_data["datasets"]
    assert len(ds) == 1, f"Expected 1 dataset group, got {len(ds)}"
    assert ds[0]["dataset"] == "arithmetic_chains"
    assert len(ds[0]["examples"]) == 480, f"Expected 480 examples, got {len(ds[0]['examples'])}"

    # Spot-check first and last examples
    first = ds[0]["examples"][0]
    last = ds[0]["examples"][-1]
    assert "input" in first and "output" in first
    assert "input" in last and "output" in last
    assert first["metadata_difficulty_level"] == 2
    assert last["metadata_difficulty_level"] == 25
    assert first["metadata_fold"] == "test"

    # Check no bare non-metadata fields
    allowed_bare = {"input", "output"}
    for ex in ds[0]["examples"][:5]:
        for key in ex:
            if key not in allowed_bare and not key.startswith("metadata_"):
                logger.error(f"Unexpected bare field: {key}")
                sys.exit(1)

    logger.info("Structure validation passed")

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------
    OUTPUT_FILE.write_text(json.dumps(output_data, indent=2, ensure_ascii=False))
    size_kb = OUTPUT_FILE.stat().st_size / 1024
    logger.info(f"Saved {len(examples)} examples to {OUTPUT_FILE} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
