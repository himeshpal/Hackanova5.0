"""
main.py
-------
Unified Agentic AI Pipeline with Multi-Stage VectorDB Storage & Clean Slate Reset.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import uuid
import httpx
import io
import re
import fitz  # PyMuPDF

from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from utils.embeddings import encode 
from core.engine import HarvesterEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BATCH_SIZE = 2  

# ===========================================================================
# STRICT INTELLIGENCE SCHEMAS
# ===========================================================================
class ExtractedIntelligence(BaseModel):
    core_methodology: str = Field(description="The primary method, algorithm, or model used.")
    datasets_used: List[str] = Field(default_factory=list, description="Datasets used.")
    evaluation_metrics: List[str] = Field(default_factory=list, description="Metrics used.")
    identified_weaknesses: List[str] = Field(default_factory=list, description="Limitations.")
    unexplored_areas: str = Field(description="What was not tested.")

class PaperIntelligence(BaseModel):
    paper_id: str
    extracted_intelligence: Optional[ExtractedIntelligence] = None
    error: Optional[str] = None

# ===========================================================================
# HIGH-SPEED EXTRACTION ENGINE
# ===========================================================================
@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=3))
async def fetch_pdf_content(url: str, max_pages: int = 2) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True, headers=headers) as client:
        response = await client.get(url)
        response.raise_for_status()
        
        if b"%PDF" not in response.content[:10]:
            raise ValueError(f"URL did not return a valid PDF: {url}")
            
        pdf_stream = io.BytesIO(response.content)
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
        
        text = ""
        pages_to_process = min(max_pages, len(doc))
        for page_num in range(pages_to_process):
            text += doc.load_page(page_num).get_text("text") + "\n"
        return text

async def extract_intelligence_via_llm(text: str) -> ExtractedIntelligence:
    prompt = f"""
    Extract information from the text. Return ONLY valid JSON matching this exact structure:
    {{
      "core_methodology": "string",
      "datasets_used": ["string"],
      "evaluation_metrics": ["string"],
      "identified_weaknesses": ["string"],
      "unexplored_areas": "string"
    }}
    Text: {text[:8000]}
    """
    
    payload = {
        "model": "qwen2.5:7b", 
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.0}
    }
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post('http://localhost:11434/api/generate', json=payload)
        response.raise_for_status()
        result_text = response.json().get('response', '{}')
        
    result_text = re.sub(r'^```json\s*', '', result_text, flags=re.MULTILINE)
    result_text = re.sub(r'^```\s*$', '', result_text, flags=re.MULTILINE)
    
    try:
        return ExtractedIntelligence.model_validate_json(result_text.strip())
    except ValidationError:
        raise ValueError("LLM produced invalid JSON.")

async def process_single_paper(paper_dict: dict) -> PaperIntelligence:
    try:
        text_content = await fetch_pdf_content(paper_dict["source_url"])
        intelligence = await extract_intelligence_via_llm(text_content)
        return PaperIntelligence(paper_id=paper_dict["paper_id"], extracted_intelligence=intelligence)
    except Exception as e:
        logger.warning(f"⚠️ Skipped {paper_dict['paper_id']} (Reason: {str(e)})")
        return PaperIntelligence(paper_id=paper_dict["paper_id"], error=str(e))

# ===========================================================================
# MAIN PIPELINE ORCHESTRATOR
# ===========================================================================
async def main(query: str) -> None:
    logger.info("Initializing Multi-Stage Qdrant VectorDB...")
    qdrant = QdrantClient(path="./local_qdrant_db")
    
    # -----------------------------------------------------------------------
    # 🧹 CLEAN SLATE FOR HACKATHON DEMO
    # -----------------------------------------------------------------------
    for collection in ["harvested_metadata", "extracted_intelligence"]:
        if qdrant.collection_exists(collection):
            qdrant.delete_collection(collection_name=collection)
            logger.info(f"🧹 Cleared old '{collection}' data.")
            
    # Recreate empty collections
    qdrant.create_collection(
        collection_name="harvested_metadata",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    qdrant.create_collection(
        collection_name="extracted_intelligence",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    logger.info("✨ VectorDB reset and ready for new query!")

    engine = HarvesterEngine()
    logger.info(f"Harvesting papers for query: '{query}'")
    result = await engine.harvest(query)
    
    valid_papers = [
        p for p in result["corpus"] 
        if not p["is_redundant"] 
        and p["deflection_flag"] == "ON_TOPIC"
        and ".pdf" in p["source_url"].lower()
    ]
    
    final_count = len(valid_papers)
    if final_count == 0:
        logger.warning("No valid PDFs found. Exiting.")
        return

    # -----------------------------------------------------------------------
    # STAGE 1: BULK VECTORDB INSERTION
    # -----------------------------------------------------------------------
    logger.info(f"⚡ STAGE 1: Embedding {final_count} Harvested Papers for instant frontend access...")
    
    # Safe abstract fetching to prevent KeyErrors!
    abstract_texts = [p.get("abstract") if p.get("abstract") else p.get("title", "No Title") for p in valid_papers]
    harvested_vectors = encode(abstract_texts) 
    
    harvested_points = []
    for i, paper in enumerate(valid_papers):
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, paper["paper_id"]))
        harvested_points.append(PointStruct(
            id=point_id,
            vector=harvested_vectors[i].tolist(),
            payload={
                "paper_id": paper["paper_id"],
                "title": paper["title"],
                "authors": json.dumps(paper["authors"]),
                "year": paper.get("year", "N/A"),
                "source_url": paper["source_url"]
            }
        ))
    
    qdrant.upsert(collection_name="harvested_metadata", points=harvested_points)
    logger.info("✅ STAGE 1 COMPLETE! The frontend can now load the UI instantly.")
    
    # -----------------------------------------------------------------------
    # STAGE 2: ASYNC EXTRACTION BATCH PROCESSING
    # -----------------------------------------------------------------------
    logger.info("⚡ STAGE 2: Starting deep LLM Extraction in the background...")
    
    for i in range(0, final_count, BATCH_SIZE):
        batch = valid_papers[i : i + BATCH_SIZE]
        batch_number = (i // BATCH_SIZE) + 1
        
        logger.info(f"⏳ Extracting Batch {batch_number} (Papers {i+1} to {min(i+BATCH_SIZE, final_count)})...")
        
        tasks = [process_single_paper(paper) for paper in batch]
        batch_results = await asyncio.gather(*tasks)
        
        points = []
        for paper_res in batch_results:
            if paper_res.extracted_intelligence and not paper_res.error:
                intel = paper_res.extracted_intelligence.model_dump()
                
                text_to_embed = f"Methodology: {intel.get('core_methodology', '')}. " \
                                f"Datasets: {', '.join(intel.get('datasets_used', []))}."
                
                vector = encode(text_to_embed).tolist()
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, paper_res.paper_id))
                
                points.append(PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "paper_id": paper_res.paper_id,
                        "methodology": intel.get("core_methodology", ""),
                        "datasets": json.dumps(intel.get("datasets_used", [])),
                        "metrics": json.dumps(intel.get("evaluation_metrics", [])),
                        "weaknesses": json.dumps(intel.get("identified_weaknesses", []))
                    }
                ))
        
        if points:
            qdrant.upsert(collection_name="extracted_intelligence", points=points)
            logger.info(f"✅ Batch {batch_number} Intelligence saved to VectorDB!")

    logger.info("🎉 PIPELINE FULLY COMPLETE!")

if __name__ == "__main__":
    test_query = sys.argv[1] if len(sys.argv) > 1 else "efficient fine tuning of large language models"
    asyncio.run(main(test_query))