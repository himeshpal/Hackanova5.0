"""
main.py
-------
CLI entry point and runnable test block for the Research Paper Harvester.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import httpx

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

# ===========================================================================
# PERFORMANCE TUNING
# ===========================================================================
BATCH_SIZE = 2  # Send 2 papers to the LLM at a time for ultra-low latency

async def main(query: str) -> None:
    engine = HarvesterEngine()

    logger.info("Harvesting papers for query: %s", query)
    
    # 1. RUN THE HARVESTER
    result = await engine.harvest(query)
    
    # 2. FILTER FOR UNIQUE, ON-TOPIC PDFs
    valid_papers = [
        p for p in result["corpus"] 
        if not p["is_redundant"] 
        and p["deflection_flag"] == "ON_TOPIC"
        and ".pdf" in p["source_url"].lower()
    ]
    
    if not valid_papers:
        logger.warning("No valid, on-topic PDFs found to extract. Exiting.")
        return

    final_count = len(valid_papers)
    logger.info(f"Filtered down to {final_count} unique, high-quality PDFs.")
    logger.info(f"Starting Background Extraction in rapid batches of {BATCH_SIZE}...")
    
    # We will keep a master list of all intelligence as it streams in
    master_intelligence_list = []

    # 3. CHUNKED BATCH PROCESSING
    for i in range(0, final_count, BATCH_SIZE):
        batch = valid_papers[i : i + BATCH_SIZE]
        batch_number = (i // BATCH_SIZE) + 1
        
        logger.info(f"⏳ Sending Batch {batch_number} (Papers {i+1} to {min(i+BATCH_SIZE, final_count)} of {final_count}) to Extractor...")
        
        # Prepare the payload for just this tiny batch
        batch_payload = {
            "session_id": result["session_id"],
            "original_query": result["original_query"],
            "corpus": batch
        }
        
        try:
            # Timeout is shorter because we are only doing 2 papers!
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    "http://localhost:8001/extract-intelligence",
                    json=batch_payload
                )
                response.raise_for_status()
                
                batch_data = response.json()
                master_intelligence_list.extend(batch_data)
                
                # 🔥 INSTANT FEEDBACK: Print the result immediately so the UI can update!
                print(f"\n✅ BATCH {batch_number} COMPLETE! Streaming partial results:")
                print(json.dumps(batch_data, indent=2, ensure_ascii=False))
                print("-" * 50)
                
        except httpx.ConnectError:
            logger.error("❌ Could not connect! Is your extractor running on port 8001?")
            break # Stop processing if the server is completely dead
        except Exception as e:
            logger.error(f"❌ Error during extraction of batch {batch_number}: {e}")
            # The beauty of batches: if one batch fails, we don't crash. We just move to the next!
            continue

    # 4. FINAL COMPLETION
    logger.info("🎉 All batches successfully processed in the background!")
    
    # Optional: Save the master list to a JSON file for the Knowledge Graph Builder
    with open("final_extracted_intelligence.json", "w", encoding="utf-8") as f:
        json.dump(master_intelligence_list, f, indent=2, ensure_ascii=False)
        logger.info("💾 Saved full results to 'final_extracted_intelligence.json'")


if __name__ == "__main__":
    test_query = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "efficient fine tuning of large language models"
    )
    asyncio.run(main(test_query))