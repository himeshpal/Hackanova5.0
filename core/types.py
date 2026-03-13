"""
core/types.py
-------------
All shared dataclasses / enums for the Research Paper Harvester.
No business logic lives here — only data containers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional


# ---------------------------------------------------------------------------
# Deflection flag enum
# ---------------------------------------------------------------------------

class DeflectionFlag(str, Enum):
    """Categorical label describing how a paper relates to the query."""
    ON_TOPIC = "ON_TOPIC"
    DEFLECTED = "DEFLECTED"
    REDUNDANT_REVIEW_PAPER = "REDUNDANT_REVIEW_PAPER"
    REDUNDANT_METHOD = "REDUNDANT_METHOD"
    REDUNDANT_DATASET = "REDUNDANT_DATASET"
    EXACT_DUPLICATE = "EXACT_DUPLICATE"


# ---------------------------------------------------------------------------
# Paper
# ---------------------------------------------------------------------------

@dataclass
class Paper:
    """
    Canonical representation of a single academic paper, regardless of source.

    Attributes
    ----------
    paper_id:
        Namespaced identifier, e.g. ``arxiv:2401.09982`` or ``s2:abc123``.
    title:
        Full paper title.
    authors:
        Ordered list of author names.
    year:
        Publication year (None if unavailable).
    abstract:
        Paper abstract.  Falls back to title when the API omits the abstract.
    source_url:
        Direct URL to the PDF or landing page.
    doi:
        Digital Object Identifier, normalised to lowercase without URL prefix.
    semantic_similarity_score:
        Cosine similarity between the query embedding and the paper embedding.
        Populated by ``SemanticDeflectionFilter``.
    is_redundant:
        Set to ``True`` by ``RedundancyChecker`` for papers that duplicate
        another paper already in the corpus.
    deflection_flag:
        One of :class:`DeflectionFlag`.  Default ``ON_TOPIC``.
    """

    paper_id: str
    title: str
    authors: List[str]
    year: Optional[int]
    abstract: str
    source_url: str
    doi: Optional[str] = None
    semantic_similarity_score: float = 0.0
    is_redundant: bool = False
    deflection_flag: DeflectionFlag = DeflectionFlag.ON_TOPIC

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict matching the required output schema."""
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "source_url": self.source_url,
            "semantic_similarity_score": round(self.semantic_similarity_score, 4),
            "is_redundant": self.is_redundant,
            "deflection_flag": self.deflection_flag.value,
        }


# ---------------------------------------------------------------------------
# HarvestResult
# ---------------------------------------------------------------------------

@dataclass
class HarvestResult:
    """
    Top-level output returned by :class:`~core.engine.HarvesterEngine`.

    Attributes
    ----------
    session_id:
        Unique request identifier, e.g. ``req_a1b2c3d4``.
    original_query:
        The verbatim query string provided by the caller.
    corpus:
        Ranked, deduplicated list of :class:`Paper` objects.
    """

    session_id: str
    original_query: str
    corpus: List[Paper] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict matching the required output schema."""
        return {
            "session_id": self.session_id,
            "original_query": self.original_query,
            "corpus": [p.to_dict() for p in self.corpus],
        }


# ---------------------------------------------------------------------------
# HarvestSession
# ---------------------------------------------------------------------------

@dataclass
class HarvestSession:
    """
    Internal bookkeeping object for a single harvest run.

    Not included in the final JSON output; used for logging / diagnostics.

    Attributes
    ----------
    session_id:
        Mirrors :attr:`HarvestResult.session_id`.
    query:
        Original query string.
    started_at:
        UTC timestamp when the session was created.
    seed_count:
        Number of papers returned by initial API seed queries.
    expanded_count:
        Number of papers after citation graph expansion.
    final_count:
        Number of papers in the final ranked corpus.
    """

    session_id: str
    query: str
    started_at: datetime = field(default_factory=datetime.utcnow)
    seed_count: int = 0
    expanded_count: int = 0
    final_count: int = 0
