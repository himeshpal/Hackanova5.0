"""
core/engine.py
--------------
HarvesterEngine

Orchestrates the full Research Paper Harvester pipeline:

  1. Seed collection  — concurrent queries to arXiv, PubMed, Semantic Scholar
  2. Citation BFS     — expand via Semantic Scholar citation graph
  3. Deduplication    — 3-layer RedundancyChecker
  4. Semantic scoring — SemanticDeflectionFilter
  5. Ranking          — sort by score, push redundant/deflected to the end
  6. Serialisation    — return JSON-ready dict
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from datetime import datetime
from typing import Dict, List

import aiohttp

from connectors.arxiv import ArXivConnector
from connectors.pubmed import PubMedConnector
from connectors.semantic_scholar import SemanticScholarConnector
from core.types import DeflectionFlag, HarvestResult, HarvestSession, Paper
from filters.redundancy import RedundancyChecker
from filters.semantic import SemanticDeflectionFilter

logger = logging.getLogger(__name__)


def _make_session_id() -> str:
    """Generate a short random session identifier like ``req_a1b2c3d4``."""
    return f"req_{secrets.token_hex(4)}"


class HarvesterEngine:
    """
    Top-level orchestrator for the paper harvesting pipeline.

    All API calls are async; call :meth:`harvest` from an async context or
    use :meth:`harvest_sync` for synchronous callers.

    Parameters
    ----------
    arxiv_connector:
        Optional custom :class:`~harvester.connectors.arxiv.ArXivConnector`.
    pubmed_connector:
        Optional custom :class:`~harvester.connectors.pubmed.PubMedConnector`.
    s2_connector:
        Optional custom
        :class:`~harvester.connectors.semantic_scholar.SemanticScholarConnector`.
    semantic_filter:
        Optional custom :class:`~harvester.filters.semantic.SemanticDeflectionFilter`.
    redundancy_checker:
        Optional custom :class:`~harvester.filters.redundancy.RedundancyChecker`.
    """

    def __init__(
        self,
        arxiv_connector: ArXivConnector | None = None,
        pubmed_connector: PubMedConnector | None = None,
        s2_connector: SemanticScholarConnector | None = None,
        semantic_filter: SemanticDeflectionFilter | None = None,
        redundancy_checker: RedundancyChecker | None = None,
    ) -> None:
        self._arxiv = arxiv_connector or ArXivConnector()
        self._pubmed = pubmed_connector or PubMedConnector()
        self._s2 = s2_connector or SemanticScholarConnector()
        self._semantic_filter = semantic_filter or SemanticDeflectionFilter()
        self._redundancy_checker = redundancy_checker or RedundancyChecker()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def harvest(self, query: str) -> Dict:
        """
        Run the full harvesting pipeline for *query*.

        Parameters
        ----------
        query:
            Natural-language research query.

        Returns
        -------
        dict
            JSON-serialisable result matching the required output schema.
        """
        session_id = _make_session_id()
        session = HarvestSession(
            session_id=session_id,
            query=query,
            started_at=datetime.utcnow(),
        )
        logger.info("[%s] Starting harvest for: %s", session_id, query)

        async with aiohttp.ClientSession() as http_session:
            # Inject shared HTTP session into all connectors
            self._arxiv.session = http_session
            self._pubmed.session = http_session
            self._s2.session = http_session

            # -------------------------------------------------------
            # Step 1 — Parallel seed collection
            # -------------------------------------------------------
            seed_papers = await self._collect_seeds(query)
            session.seed_count = len(seed_papers)
            logger.info("[%s] Seed collection: %d papers.", session_id, session.seed_count)

            # -------------------------------------------------------
            # Step 2 — Citation graph expansion (BFS via S2)
            # -------------------------------------------------------
            citation_papers = await self._s2.expand_via_citations(seed_papers)

            all_papers: List[Paper] = seed_papers + citation_papers
            session.expanded_count = len(all_papers)
            logger.info(
                "[%s] After BFS expansion: %d papers total.", session_id, session.expanded_count
            )

        # -------------------------------------------------------
        # Step 3 — Redundancy detection (3-layer dedup)
        # -------------------------------------------------------
        all_papers = self._redundancy_checker.check(all_papers)

        # -------------------------------------------------------
        # Step 4 — Semantic scoring + deflection flagging
        # -------------------------------------------------------
        all_papers = self._semantic_filter.filter(query, all_papers)

        # -------------------------------------------------------
        # Step 5 — Ranking
        # -------------------------------------------------------
        ranked = self._rank(all_papers)
        session.final_count = len(ranked)

        logger.info(
            "[%s] Pipeline complete: %d papers in final corpus "
            "(seed=%d, expanded=%d).",
            session_id,
            session.final_count,
            session.seed_count,
            session.expanded_count,
        )

        # -------------------------------------------------------
        # Step 6 — Build output
        # -------------------------------------------------------
        result = HarvestResult(
            session_id=session_id,
            original_query=query,
            corpus=ranked,
        )
        return result.to_dict()

    def harvest_sync(self, query: str) -> Dict:
        """Synchronous convenience wrapper around :meth:`harvest`."""
        return asyncio.run(self.harvest(query))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _collect_seeds(self, query: str) -> List[Paper]:
        """
        Query arXiv, PubMed, and Semantic Scholar concurrently.

        Returns
        -------
        List[Paper]
            Combined, unfiltered results from all three sources.
        """
        arxiv_task = asyncio.create_task(self._arxiv.search(query))
        pubmed_task = asyncio.create_task(self._pubmed.search(query))
        s2_task = asyncio.create_task(self._s2.search(query))

        results = await asyncio.gather(
            arxiv_task, pubmed_task, s2_task, return_exceptions=True
        )

        papers: List[Paper] = []
        source_names = ["arXiv", "PubMed", "SemanticScholar"]
        for name, result in zip(source_names, results):
            if isinstance(result, Exception):
                logger.error("Seed collection from %s failed: %s", name, result)
            elif result:
                papers.extend(result)

        return papers

    @staticmethod
    def _rank(papers: List[Paper]) -> List[Paper]:
        """
        Sort papers for the final corpus.

        Ordering:
          1. Non-redundant, on-topic papers — descending similarity score
          2. Non-redundant, deflected papers — descending similarity score
          3. Redundant papers — descending similarity score
        """

        def sort_key(p: Paper):
            # Group: 0 = on-topic + not redundant, 1 = deflected + not redundant, 2 = redundant
            if p.is_redundant:
                group = 2
            elif p.deflection_flag == DeflectionFlag.DEFLECTED:
                group = 1
            else:
                group = 0
            # Within group: higher score first (negate for ascending sort)
            return (group, -p.semantic_similarity_score)

        return sorted(papers, key=sort_key)
