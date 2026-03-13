import io
import json
import re
import asyncio
import httpx
import fitz  # PyMuPDF
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential

app = FastAPI(title="Intelligence Extraction Microservice")

# ==========================================
# INPUT/OUTPUT SCHEMAS
# ==========================================
class PaperInput(BaseModel):
    paper_id: str
    title: str
    authors: List[str]
    year: Optional[int] = None
    source_url: str
    semantic_similarity_score: float
    is_redundant: bool
    deflection_flag: str

class IntelligenceRequest(BaseModel):
    session_id: str
    original_query: str
    corpus: List[PaperInput]

class ExtractedIntelligence(BaseModel):
    core_methodology: str = Field(description="The primary method, algorithm, or model used.")
    datasets_used: List[str] = Field(default_factory=list, description="Datasets used.")
    evaluation_metrics: List[str] = Field(default_factory=list, description="Metrics used.")
    identified_weaknesses: List[str] = Field(default_factory=list, description="Limitations.")
    unexplored_areas: str = Field(description="What was not tested.")

class PaperIntelligence(BaseModel):
    paper_id: str
    extracted_intelligence: Optional[ExtractedIntelligence] = None
    error: Optional[str] = None  # ROOT CAUSE FIX: We now send errors back to the main terminal!

# ==========================================
# RETRIEVAL & EXTRACTION LOGIC
# ==========================================
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def fetch_pdf_content(url: str, max_pages: int = 4) -> str:
    """Fetches PDF. Includes User-Agent to bypass arXiv bot-blocking."""
    
    # ROOT CAUSE FIX 1: Spoof a real browser so arXiv doesn't block the download
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True, headers=headers) as client:
        response = await client.get(url)
        response.raise_for_status()
        
        # Verify we actually got a PDF, not an HTML error page
        if b"%PDF" not in response.content[:10]:
            raise ValueError(f"URL did not return a valid PDF document: {url}")
            
        pdf_stream = io.BytesIO(response.content)
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
        
        extracted_text = ""
        pages_to_process = min(max_pages, len(doc))
        for page_num in range(pages_to_process):
            page = doc.load_page(page_num)
            extracted_text += page.get_text("text") + "\n"
            
        return extracted_text

async def extract_intelligence_via_llm(text: str) -> ExtractedIntelligence:
    """Uses Local Ollama and sanitizes the output."""
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
        "options": {"temperature": 0.0} # Set to 0.0 for absolute strictness
    }
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post('http://localhost:11434/api/generate', json=payload)
        response.raise_for_status()
        result_text = response.json().get('response', '{}')
        
    # ROOT CAUSE FIX 2: Strip markdown backticks injected by local models
    result_text = re.sub(r'^```json\s*', '', result_text, flags=re.MULTILINE)
    result_text = re.sub(r'^```\s*$', '', result_text, flags=re.MULTILINE)
    result_text = result_text.strip()
    
    try:
        return ExtractedIntelligence.model_validate_json(result_text)
    except ValidationError as e:
        raise ValueError(f"LLM produced invalid JSON structure: {result_text}")

# ==========================================
# API ENDPOINT
# ==========================================
async def process_single_paper(paper: PaperInput) -> PaperIntelligence:
    try:
        text_content = await fetch_pdf_content(paper.source_url)
        intelligence = await extract_intelligence_via_llm(text_content)
        return PaperIntelligence(
            paper_id=paper.paper_id,
            extracted_intelligence=intelligence
        )
    except Exception as e:
        # ROOT CAUSE FIX 3: Catch the error and return it to the frontend!
        error_msg = str(e)
        print(f"\n[ERROR] Failed to process {paper.paper_id}: {error_msg}\n")
        return PaperIntelligence(paper_id=paper.paper_id, error=error_msg)

@app.post("/extract-intelligence", response_model=List[PaperIntelligence])
async def extract_intelligence_endpoint(request: IntelligenceRequest):
    valid_papers = [
        paper for paper in request.corpus
        if not paper.is_redundant and paper.deflection_flag == "ON_TOPIC"
    ]
    if not valid_papers:
        return []

    tasks = [process_single_paper(paper) for paper in valid_papers]
    results = await asyncio.gather(*tasks)
    return results