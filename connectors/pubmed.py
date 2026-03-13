"""
connectors/pubmed.py
--------------------
PubMedConnector

Uses the NCBI E-Utilities:
  1. esearch — map a query to a list of PubMed IDs (PMIDs)
  2. efetch  — retrieve full records (XML) for those PMIDs

XML parsing is done with the stdlib ``xml.etree.ElementTree``.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import List, Optional

from config.settings import (
    PUBMED_EFETCH_URL,
    PUBMED_ESEARCH_URL,
    PUBMED_MAX_RESULTS,
)
from connectors.base import BaseConnector
from core.types import Paper

logger = logging.getLogger(__name__)


def _text(element: Optional[ET.Element]) -> str:
    """Safely extract .text from an optional Element, returning ''."""
    if element is None:
        return ""
    return (element.text or "").strip()


class PubMedConnector(BaseConnector):
    """
    Connector for NCBI PubMed via E-Utilities.

    Parameters
    ----------
    max_results:
        Default cap on results.
    """

    def __init__(self, max_results: int = PUBMED_MAX_RESULTS) -> None:
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
        Search PubMed and return a list of :class:`~harvester.core.types.Paper`.

        Parameters
        ----------
        query:
            Free-text search query.
        max_results:
            Number of results.  Uses connector default when 0.
        """
        max_results = max_results or self.default_max_results
        logger.info("PubMed search: '%s' (max=%d)", query, max_results)

        pmids = await self._esearch(query, max_results)
        if not pmids:
            logger.warning("PubMed esearch returned no IDs for query: %s", query)
            return []

        return await self._efetch(pmids)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _esearch(self, query: str, max_results: int) -> List[str]:
        """Call esearch and return a list of PMID strings."""
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "usehistory": "n",
        }

        data = await self._get(PUBMED_ESEARCH_URL, params=params, response_format="json")
        if not data:
            return []

        try:
            pmids: List[str] = data["esearchresult"]["idlist"]
            logger.info("PubMed esearch: found %d PMIDs.", len(pmids))
            return pmids
        except (KeyError, TypeError) as exc:
            logger.warning("PubMed esearch parse error: %s", exc)
            return []

    async def _efetch(self, pmids: List[str]) -> List[Paper]:
        """Call efetch for a list of PMIDs and parse the XML response."""
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract",
        }

        raw_xml = await self._get(PUBMED_EFETCH_URL, params=params, response_format="text")
        if not raw_xml:
            logger.warning("PubMed efetch returned no data.")
            return []

        return self._parse_pubmed_xml(raw_xml)

    @staticmethod
    def _parse_pubmed_xml(raw_xml: str) -> List[Paper]:
        """Parse PubMed XML into :class:`Paper` objects."""
        papers: List[Paper] = []

        try:
            root = ET.fromstring(raw_xml)
        except ET.ParseError as exc:
            logger.error("PubMed XML parse error: %s", exc)
            return []

        for article in root.findall(".//PubmedArticle"):
            try:
                medline = article.find("MedlineCitation")
                if medline is None:
                    continue

                # PMID
                pmid_el = medline.find("PMID")
                pmid = _text(pmid_el)
                paper_id = f"pubmed:{pmid}" if pmid else "pubmed:unknown"

                art = medline.find("Article")
                if art is None:
                    continue

                # Title
                title = _text(art.find("ArticleTitle"))
                if not title:
                    continue

                # Abstract — concatenate all AbstractText sections
                abstract_parts: List[str] = []
                abstract_el = art.find("Abstract")
                if abstract_el is not None:
                    for at in abstract_el.findall("AbstractText"):
                        label = at.get("Label", "")
                        text = (at.text or "").strip()
                        if label:
                            abstract_parts.append(f"{label}: {text}")
                        elif text:
                            abstract_parts.append(text)
                abstract = " ".join(abstract_parts) if abstract_parts else title

                # Authors
                authors: List[str] = []
                for author in art.findall(".//Author"):
                    last = _text(author.find("LastName"))
                    fore = _text(author.find("ForeName"))
                    if last:
                        authors.append(f"{fore} {last}".strip())

                # Year — try ArticleDate first, then PubDate
                year: Optional[int] = None
                for date_path in ("ArticleDate/Year", "Journal/JournalIssue/PubDate/Year"):
                    yr_el = art.find(date_path)
                    if yr_el is not None and yr_el.text:
                        try:
                            year = int(yr_el.text)
                            break
                        except ValueError:
                            pass

                # DOI
                doi: Optional[str] = None
                for el_id in article.findall(".//ArticleId"):
                    if el_id.get("IdType") == "doi":
                        doi = (el_id.text or "").lower().strip()
                        break

                # Source URL
                source_url = (
                    f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                    if pmid
                    else "https://pubmed.ncbi.nlm.nih.gov/"
                )

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
                logger.warning("Failed to parse PubMed article: %s", exc)
                continue

        logger.info("PubMed: parsed %d papers.", len(papers))
        return papers
