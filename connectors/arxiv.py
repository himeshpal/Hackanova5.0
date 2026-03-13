"""
connectors/arxiv.py
-------------------
ArXivConnector

Queries the arXiv Atom API at http://export.arxiv.org/api/query.
Response is an Atom XML feed parsed with feedparser.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

import feedparser

from config.settings import ARXIV_API_URL, ARXIV_MAX_RESULTS
from connectors.base import BaseConnector
from core.types import Paper

logger = logging.getLogger(__name__)

# arXiv paper ID regex: e.g.  2401.09982  or  cs/0612066
_ARXIV_ID_RE = re.compile(r"arxiv\.org/abs/([^v\s]+)")


def _extract_arxiv_id(entry_id: str) -> str:
    """Extract the bare arXiv ID from a full Atom entry id URL."""
    match = _ARXIV_ID_RE.search(entry_id)
    if match:
        return match.group(1)
    # Fallback: use the raw id field
    return entry_id.split("/")[-1]


def _parse_year(published: str) -> Optional[int]:
    """Parse the year out of an ISO-8601 date string like '2024-01-15T00:00:00Z'."""
    try:
        return int(published[:4])
    except (ValueError, TypeError, IndexError):
        return None


class ArXivConnector(BaseConnector):
    """
    Connector for the arXiv preprint API.

    Parameters
    ----------
    max_results:
        Default cap on results.  Can be overridden per-call.
    """

    def __init__(self, max_results: int = ARXIV_MAX_RESULTS) -> None:
        super().__init__()
        self.default_max_results = max_results

    # ------------------------------------------------------------------
    # BaseConnector interface
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        max_results: int = 0,
    ) -> List[Paper]:
        """
        Search arXiv for papers matching *query*.

        Parameters
        ----------
        query:
            Natural-language or field-qualified arXiv query.
        max_results:
            Number of results to request.  Uses connector default when 0.

        Returns
        -------
        List[Paper]
            Parsed paper objects.
        """
        max_results = max_results or self.default_max_results

        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        logger.info("arXiv search: '%s' (max=%d)", query, max_results)
        raw_text = await self._get(ARXIV_API_URL, params=params, response_format="text")

        if not raw_text:
            logger.warning("arXiv returned no data for query: %s", query)
            return []

        return self._parse_feed(raw_text)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_feed(raw_xml: str) -> List[Paper]:
        """Parse feedparser-compatible Atom XML into :class:`Paper` objects."""
        feed = feedparser.parse(raw_xml)
        papers: List[Paper] = []

        for entry in feed.entries:
            try:
                arxiv_id = _extract_arxiv_id(entry.get("id", ""))
                paper_id = f"arxiv:{arxiv_id}"

                title: str = entry.get("title", "").strip().replace("\n", " ")
                abstract: str = entry.get("summary", "").strip().replace("\n", " ")
                if not abstract:
                    abstract = title  # fallback

                authors: List[str] = [
                    a.get("name", "Unknown") for a in entry.get("authors", [])
                ]

                published: str = entry.get("published", "")
                year = _parse_year(published)

                # Build PDF URL
                source_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

                # Extract DOI if present in the links
                doi: Optional[str] = None
                for link in entry.get("links", []):
                    href = link.get("href", "")
                    if "doi.org" in href:
                        doi = href.split("doi.org/")[-1].lower()
                        break

                papers.append(
                    Paper(
                        paper_id=paper_id,
                        title=title,
                        authors=authors,
                        year=year,
                        abstract=abstract,
                        source_url=source_url,
                        doi=doi,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to parse arXiv entry: %s", exc)
                continue

        logger.info("arXiv: parsed %d papers.", len(papers))
        return papers
