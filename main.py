"""
main.py
-------
CLI entry point and runnable test block for the Research Paper Harvester.

Usage
-----
    python -m harvester.main                         # run built-in test
    python -m harvester.main "your research query"  # custom query
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys

from core.engine import HarvesterEngine

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main coroutine
# ---------------------------------------------------------------------------

async def main(query: str) -> None:
    """
    Run the harvest pipeline for *query* and print the JSON result.

    Parameters
    ----------
    query:
        Natural-language research query.
    """
    engine = HarvesterEngine()

    logger.info("Harvesting papers for query: %s", query)
    result = await engine.harvest(query)

    # Pretty-print JSON output
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # ---------------------------------------------------------------------------
    # Built-in assertions (test block)
    # ---------------------------------------------------------------------------
    assert "session_id" in result, "Result missing 'session_id'"
    assert "original_query" in result, "Result missing 'original_query'"
    assert "corpus" in result, "Result missing 'corpus'"
    assert len(result["corpus"]) > 0, "Corpus is empty — no papers harvested"
    assert all(
        p["semantic_similarity_score"] >= 0 for p in result["corpus"]
    ), "Some papers have negative similarity scores"
    assert all(
        "deflection_flag" in p for p in result["corpus"]
    ), "Some papers are missing 'deflection_flag'"
    assert all(
        "is_redundant" in p for p in result["corpus"]
    ), "Some papers are missing 'is_redundant'"

    logger.info(
        "✓ All assertions passed. Session: %s | Papers: %d",
        result["session_id"],
        len(result["corpus"]),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_query = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "transformers for medical image segmentation"
    )
    asyncio.run(main(test_query))
