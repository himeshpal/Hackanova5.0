"""
config/settings.py
------------------
Centralised configuration for the Research Paper Harvester.
All tunable constants live here so no magic numbers scatter through the codebase.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
REQUESTS_PER_SECOND: int = 1          # 1 req/s — safe for S2 without API key
REQUEST_TIMEOUT_SECONDS: int = 30     # aiohttp total timeout per request

# ---------------------------------------------------------------------------
# Retry / back-off
# ---------------------------------------------------------------------------
MAX_RETRIES: int = 2                  # reduced from 3 — fail faster on rate limits
BACKOFF_BASE_SECONDS: float = 2.0     # sleep = BACKOFF_BASE ** attempt

# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
ARXIV_API_URL: str = "http://export.arxiv.org/api/query"
PUBMED_ESEARCH_URL: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH_URL: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
SEMANTIC_SCHOLAR_SEARCH_URL: str = "https://api.semanticscholar.org/graph/v1/paper/search"
SEMANTIC_SCHOLAR_PAPER_URL: str = "https://api.semanticscholar.org/graph/v1/paper"

# ---------------------------------------------------------------------------
# Seed collection limits
# ---------------------------------------------------------------------------
ARXIV_MAX_RESULTS: int = 10           # reduced: fewer seeds = fewer BFS calls
PUBMED_MAX_RESULTS: int = 10
S2_MAX_RESULTS: int = 10

# ---------------------------------------------------------------------------
# Citation graph traversal
# ---------------------------------------------------------------------------
BFS_MAX_DEPTH: int = 1               # reduced from 2: single hop avoids call explosion
BFS_MAX_PAPERS_PER_HOP: int = 10     # reduced from 50: cap citations fetched per paper

# ---------------------------------------------------------------------------
# Semantic filtering thresholds
# ---------------------------------------------------------------------------
DEFLECTION_THRESHOLD: float = 0.25   # below -> DEFLECTED
REDUNDANCY_SIMILARITY_THRESHOLD: float = 0.92   # above -> redundant pair

# ---------------------------------------------------------------------------
# Review paper heuristic keywords (lower-cased)
# ---------------------------------------------------------------------------
REVIEW_KEYWORDS: tuple[str, ...] = (
    "review",
    "survey",
    "systematic review",
    "meta-analysis",
    "meta analysis",
    "overview",
    "literature review",
    "scoping review",
)

# ---------------------------------------------------------------------------
# Redundancy label keywords
# ---------------------------------------------------------------------------
METHOD_KEYWORDS: tuple[str, ...] = (
    "method",
    "approach",
    "algorithm",
    "framework",
    "architecture",
    "model",
    "technique",
    "procedure",
)
DATASET_KEYWORDS: tuple[str, ...] = (
    "dataset",
    "benchmark",
    "corpus",
    "collection",
    "database",
    "data set",
)