"""
filters/semantic.py
-------------------
SemanticDeflectionFilter

Embeds the original query and each paper's abstract (falling back to title),
computes cosine similarity, and assigns:

  - ``semantic_similarity_score`` on every paper
  - ``DeflectionFlag.DEFLECTED`` for papers whose score falls below the
    configured threshold
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

from config.settings import DEFLECTION_THRESHOLD
from core.types import DeflectionFlag, Paper
from utils.embeddings import encode

logger = logging.getLogger(__name__)


class SemanticDeflectionFilter:
    """
    Scores papers against the harvest query using dense embeddings.

    Parameters
    ----------
    deflection_threshold:
        Papers whose cosine similarity with the query falls below this value
        receive the ``DEFLECTED`` flag.  Defaults to
        :data:`~harvester.config.settings.DEFLECTION_THRESHOLD`.
    """

    def __init__(self, deflection_threshold: float = DEFLECTION_THRESHOLD) -> None:
        self.deflection_threshold = deflection_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter(self, query: str, papers: List[Paper]) -> List[Paper]:
        """
        Compute similarity scores and set deflection flags.

        Parameters
        ----------
        query:
            The original user query.
        papers:
            List of :class:`~harvester.core.types.Paper` objects to score.
            Mutated **in-place** and also returned.

        Returns
        -------
        List[Paper]
            The same list, with ``semantic_similarity_score`` and
            ``deflection_flag`` populated on every paper.
        """
        if not papers:
            return papers

        logger.info("Semantic filter: scoring %d papers against query.", len(papers))

        # Embed query (shape: (dim,))
        query_vec: np.ndarray = encode(query)

        # Build corpus texts — fall back to title when abstract is empty
        texts: List[str] = [
            p.abstract if p.abstract and p.abstract.strip() else p.title
            for p in papers
        ]

        # Embed all papers in a single batch (shape: (n, dim))
        paper_vecs: np.ndarray = encode(texts)

        # Cosine similarity = dot product because vectors are unit-normalised
        scores: np.ndarray = paper_vecs @ query_vec  # shape: (n,)

        deflected_count = 0
        for paper, score in zip(papers, scores):
            # Clip to [0, 1]: cosine on unit vectors is in [-1,1]; negative
            # similarity means orthogonal/opposite — treat as 0 relevance.
            paper.semantic_similarity_score = float(max(0.0, score))
            # Only override the flag when paper has not already been marked
            # redundant/duplicate by the RedundancyChecker
            if paper.deflection_flag == DeflectionFlag.ON_TOPIC:
                if paper.semantic_similarity_score < self.deflection_threshold:
                    paper.deflection_flag = DeflectionFlag.DEFLECTED
                    deflected_count += 1

        logger.info(
            "Semantic filter complete: %d deflected, %d on-topic.",
            deflected_count,
            len(papers) - deflected_count,
        )
        return papers
