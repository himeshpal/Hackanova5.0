"""
connectors/semantic_scholar.py
-------------------------------
SemanticScholarConnector

Uses the Semantic Scholar Graph API v1:
  - /paper/search          → seed query results
  - /paper/{id}/citations  → citation traversal (BFS)

Citation graph is traversed using breadth-first search with:
  - max depth: BFS_MAX_DEPTH (2)
  - max papers per hop: BFS_MAX_PAPERS_PER_HOP (50)
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Any, Dict, List, Optional, Set

from config.settings import (
    BFS_MAX_DEPTH,
    BFS_MAX_PAPERS_PER_HOP,
    SEMANTIC_SCHOLAR_PAPER_URL,
    SEMANTIC_SCHOLAR_SEARCH_URL,
    S2_MAX_RESULTS,
)
from connectors.base import BaseConnector
from core.types import Paper

logger = logging.getLogger(__name__)

# Fields requested from S2 API
_PAPER_FIELDS = "paperId,title,authors,year,abstract,externalIds,url"
_CITATION_FIELDS = "paperId,title,authors,year,abstract,externalIds,url"


def _parse_s2_paper(data: Dict[str, Any]) -> Optional[Paper]:
    """
    Convert a Semantic Scholar paper dict into a :class:`Paper` object.

    Returns ``None`` if the record lacks a usable ID or title.
    """
    try:
        s2_id: str = data.get("paperId") or ""
        if not s2_id:
            return None

        title: str = (data.get("title") or "").strip()
        if not title:
            return None

        abstract: str = (data.get("abstract") or "").strip()
        if not abstract:
            abstract = title  # fallback

        authors: List[str] = [
            a.get("name", "Unknown") for a in (data.get("authors") or [])
        ]

        year: Optional[int] = data.get("year")

        # External IDs
        ext_ids: Dict[str, str] = data.get("externalIds") or {}
        doi: Optional[str] = ext_ids.get("DOI", "").lower() or None
        arxiv_id: Optional[str] = ext_ids.get("ArXiv")

        # Prefer an arXiv URL; fall back to S2 URL
        if arxiv_id:
            paper_id = f"arxiv:{arxiv_id}"
            source_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        else:
            paper_id = f"s2:{s2_id}"
            source_url = data.get("url") or f"https://www.semanticscholar.org/paper/{s2_id}"

        return Paper(
            paper_id=paper_id,
            title=title,
            authors=authors,
            year=year,
            abstract=abstract,
            source_url=source_url,
            doi=doi,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to parse S2 paper record: %s", exc)
        return None


class SemanticScholarConnector(BaseConnector):
    """
    Connector for the Semantic Scholar Graph API.

    Supports both paper search and BFS citation graph traversal.

    Parameters
    ----------
    max_results:
        Default search result cap.
    bfs_max_depth:
        Citation traversal depth.
    bfs_max_papers_per_hop:
        Maximum papers fetched at each BFS level.
    """

    def __init__(
        self,
        max_results: int = S2_MAX_RESULTS,
        bfs_max_depth: int = BFS_MAX_DEPTH,
        bfs_max_papers_per_hop: int = BFS_MAX_PAPERS_PER_HOP,
    ) -> None:
        super().__init__()
        self.default_max_results = max_results
        self.bfs_max_depth = bfs_max_depth
        self.bfs_max_papers_per_hop = bfs_max_papers_per_hop

    # ------------------------------------------------------------------
    # BaseConnector interface
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        max_results: int = 0,
    ) -> List[Paper]:
        """
        Search Semantic Scholar for papers matching *query*.

        Parameters
        ----------
        query:
            Natural-language search string.
        max_results:
            Number of results.  Uses connector default when 0.
        """
        max_results = max_results or self.default_max_results
        logger.info("S2 search: '%s' (max=%d)", query, max_results)

        params = {
            "query": query,
            "limit": max_results,
            "fields": _PAPER_FIELDS,
        }

        data = await self._get(
            SEMANTIC_SCHOLAR_SEARCH_URL,
            params=params,
            response_format="json",
        )
        if not data:
            logger.warning("S2 search returned no data for query: %s", query)
            return []

        raw_papers = data.get("data") or []
        papers: List[Paper] = []
        for raw in raw_papers:
            paper = _parse_s2_paper(raw)
            if paper:
                papers.append(paper)

        logger.info("S2 search: parsed %d papers.", len(papers))
        return papers

    async def get_citations(
        self,
        paper_id: str,
        max_results: int = 0,
    ) -> List[Paper]:
        """
        Retrieve papers that cite *paper_id* (one hop).

        Parameters
        ----------
        paper_id:
            Semantic Scholar internal paper ID (not the namespaced form).
        max_results:
            Cap on results.
        """
        max_results = max_results or self.bfs_max_papers_per_hop
        url = f"{SEMANTIC_SCHOLAR_PAPER_URL}/{paper_id}/citations"
        params = {
            "fields": _CITATION_FIELDS,
            "limit": max_results,
        }

        data = await self._get(url, params=params, response_format="json")
        if not data:
            return []

        papers: List[Paper] = []
        for item in data.get("data") or []:
            # The citing paper is in item["citingPaper"]
            raw = item.get("citingPaper") or {}
            paper = _parse_s2_paper(raw)
            if paper:
                papers.append(paper)

        return papers

    # ------------------------------------------------------------------
    # BFS citation graph traversal
    # ------------------------------------------------------------------

    async def expand_via_citations(self, seed_papers: List[Paper]) -> List[Paper]:
        """
        BFS expansion of the citation graph starting from *seed_papers*.

        Parameters
        ----------
        seed_papers:
            Initial set of papers (depth 0).

        Returns
        -------
        List[Paper]
            All papers discovered during traversal (excluding seeds).
        """
        logger.info(
            "S2 BFS: starting from %d seed papers (max_depth=%d, max_per_hop=%d).",
            len(seed_papers),
            self.bfs_max_depth,
            self.bfs_max_papers_per_hop,
        )

        # Track visited S2 IDs to prevent cycles; we use the raw s2 ID here
        # extracted from our namespaced paper_id
        visited: Set[str] = set()

        # Queue items: (paper, current_depth)
        queue: deque[tuple[Paper, int]] = deque()
        for p in seed_papers:
            s2_id = self._extract_s2_id(p)
            if s2_id:
                queue.append((p, 0))
                visited.add(s2_id)

        discovered: List[Paper] = []

        while queue:
            current_paper, depth = queue.popleft()
            if depth >= self.bfs_max_depth:
                continue

            s2_id = self._extract_s2_id(current_paper)
            if not s2_id:
                continue

            # Fetch citations for this paper
            citations = await self.get_citations(
                s2_id,
                max_results=self.bfs_max_papers_per_hop,
            )

            for cited in citations:
                cited_s2_id = self._extract_s2_id(cited)
                if cited_s2_id and cited_s2_id not in visited:
                    visited.add(cited_s2_id)
                    discovered.append(cited)
                    if depth + 1 < self.bfs_max_depth:
                        queue.append((cited, depth + 1))

        logger.info("S2 BFS: discovered %d additional papers.", len(discovered))
        return discovered

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_s2_id(paper: Paper) -> Optional[str]:
        """
        Extract the bare Semantic Scholar ID from the namespaced paper_id,
        or return None if it is not an S2 ID.

        We need the raw S2 ID to call the citations endpoint.
        For papers sourced from arXiv we must first look them up via S2 search
        to get their S2 ID — but for BFS we only follow papers already in our
        corpus with known S2 IDs.
        """
        if paper.paper_id.startswith("s2:"):
            return paper.paper_id[3:]
        # arXiv papers may have been returned from S2 with their ArXiv ID;
        # the S2 citations endpoint also accepts "arXiv:<id>"
        if paper.paper_id.startswith("arxiv:"):
            return f"arXiv:{paper.paper_id[6:]}"
        return None
