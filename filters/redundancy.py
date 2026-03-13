"""
filters/redundancy.py
---------------------
RedundancyChecker — 3-layer deduplication pipeline.

Layer 1 — Exact identifier matching
    Remove papers whose DOI or paper_id has already been seen.
    Flag: EXACT_DUPLICATE

Layer 2 — Review / survey paper detection
    Title / abstract heuristic keyword scan.  If a very similar review paper
    is already in the accepted set it is marked redundant.
    Flag: REDUNDANT_REVIEW_PAPER

Layer 3 — Semantic similarity clustering
    Pairwise cosine similarity on abstract embeddings.  Within each
    near-duplicate pair the lower-scoring paper is marked redundant with an
    appropriate sub-flag (REDUNDANT_METHOD or REDUNDANT_DATASET).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Set, Tuple

import numpy as np

from config.settings import (
    DATASET_KEYWORDS,
    METHOD_KEYWORDS,
    REDUNDANCY_SIMILARITY_THRESHOLD,
    REVIEW_KEYWORDS,
)
from core.types import DeflectionFlag, Paper
from utils.embeddings import encode

logger = logging.getLogger(__name__)


def _normalise_id(paper: Paper) -> str:
    """Return a canonical identifier string for exact-match comparison."""
    if paper.doi:
        return f"doi:{paper.doi.lower().strip()}"
    return paper.paper_id.lower().strip()


def _text_contains_any(text: str, keywords: Tuple[str, ...]) -> bool:
    """Case-insensitive substring search for any keyword in *text*."""
    lower = text.lower()
    return any(kw in lower for kw in keywords)


def _infer_redundancy_flag(paper: Paper) -> DeflectionFlag:
    """
    Given a paper that has been identified as semantically redundant, decide
    which sub-flag to assign based on title / abstract keywords.
    """
    combined = f"{paper.title} {paper.abstract}".lower()
    if _text_contains_any(combined, DATASET_KEYWORDS):
        return DeflectionFlag.REDUNDANT_DATASET
    return DeflectionFlag.REDUNDANT_METHOD


class RedundancyChecker:
    """
    Deduplicates a list of :class:`~harvester.core.types.Paper` objects using
    a 3-layer pipeline.

    Parameters
    ----------
    similarity_threshold:
        Cosine similarity above which two papers are considered near-duplicates.
        Defaults to :data:`~harvester.config.settings.REDUNDANCY_SIMILARITY_THRESHOLD`.
    """

    def __init__(
        self,
        similarity_threshold: float = REDUNDANCY_SIMILARITY_THRESHOLD,
    ) -> None:
        self.similarity_threshold = similarity_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, papers: List[Paper]) -> List[Paper]:
        """
        Run the 3-layer deduplication pipeline.

        Parameters
        ----------
        papers:
            Input list.  Each paper is mutated **in-place** when marked
            redundant; the list itself is not filtered here so callers can
            inspect flags.

        Returns
        -------
        List[Paper]
            Same list with ``is_redundant`` and ``deflection_flag`` set.
        """
        if not papers:
            return papers

        papers = self._layer1_exact_ids(papers)
        papers = self._layer2_review_detection(papers)
        papers = self._layer3_semantic_clustering(papers)
        return papers

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _layer1_exact_ids(papers: List[Paper]) -> List[Paper]:
        """Mark exact duplicates by normalised DOI / paper_id."""
        seen: Set[str] = set()
        for paper in papers:
            uid = _normalise_id(paper)
            if uid in seen:
                paper.is_redundant = True
                paper.deflection_flag = DeflectionFlag.EXACT_DUPLICATE
                logger.debug("Layer 1 duplicate: %s", paper.paper_id)
            else:
                seen.add(uid)
        exact_count = sum(1 for p in papers if p.deflection_flag == DeflectionFlag.EXACT_DUPLICATE)
        logger.info("Layer 1 complete: %d exact duplicates flagged.", exact_count)
        return papers

    @staticmethod
    def _layer2_review_detection(papers: List[Paper]) -> List[Paper]:
        """
        Flag review/survey papers that arrive after an equivalent non-review
        paper is already accepted into the corpus.  The heuristic is simple:
        if the combined title+abstract contains a review keyword AND an
        accepted paper shares high title overlap, mark it redundant.

        Because we lack pairwise title similarity at this stage (we save
        embeddings for Layer 3), we apply a simpler rule: if the paper itself
        is a review and a non-review paper already accepted has a title that
        shares ≥60 % token overlap, mark the review redundant.
        """
        accepted_non_review: List[Paper] = []
        review_count = 0

        for paper in papers:
            if paper.is_redundant:
                continue  # already handled

            combined = f"{paper.title} {paper.abstract}"
            is_review = _text_contains_any(combined, REVIEW_KEYWORDS)

            if is_review:
                # Check token overlap against accepted non-reviews
                paper_tokens = set(paper.title.lower().split())
                for accepted in accepted_non_review:
                    accepted_tokens = set(accepted.title.lower().split())
                    if not paper_tokens or not accepted_tokens:
                        continue
                    overlap = len(paper_tokens & accepted_tokens) / max(
                        len(paper_tokens), len(accepted_tokens)
                    )
                    if overlap >= 0.5:
                        paper.is_redundant = True
                        paper.deflection_flag = DeflectionFlag.REDUNDANT_REVIEW_PAPER
                        review_count += 1
                        logger.debug(
                            "Layer 2 review paper: %s (overlap=%.2f with %s)",
                            paper.paper_id,
                            overlap,
                            accepted.paper_id,
                        )
                        break

            if not paper.is_redundant and not is_review:
                accepted_non_review.append(paper)

        logger.info("Layer 2 complete: %d review papers flagged.", review_count)
        return papers

    def _layer3_semantic_clustering(self, papers: List[Paper]) -> List[Paper]:
        """
        Pairwise cosine similarity on abstract embeddings for remaining papers.

        Within each near-duplicate pair the paper with the lower
        ``semantic_similarity_score`` is marked redundant.  If both scores are
        equal the second one encountered loses.
        """
        # Only consider papers not already marked redundant
        active: List[Tuple[int, Paper]] = [
            (i, p) for i, p in enumerate(papers) if not p.is_redundant
        ]

        if len(active) < 2:
            logger.info("Layer 3: fewer than 2 active papers, skipping.")
            return papers

        indices, active_papers = zip(*active)
        indices = list(indices)
        active_papers = list(active_papers)

        texts = [
            p.abstract if p.abstract and p.abstract.strip() else p.title
            for p in active_papers
        ]

        # Embeddings are unit-normalised → cosine = dot product
        vecs: np.ndarray = encode(texts)  # shape: (n, dim)
        sim_matrix: np.ndarray = vecs @ vecs.T  # shape: (n, n)

        redundant_local: Set[int] = set()  # indices into active_papers
        semantic_count = 0

        for i in range(len(active_papers)):
            if i in redundant_local:
                continue
            for j in range(i + 1, len(active_papers)):
                if j in redundant_local:
                    continue
                if sim_matrix[i, j] >= self.similarity_threshold:
                    # Keep the paper with higher semantic_similarity_score
                    score_i = active_papers[i].semantic_similarity_score
                    score_j = active_papers[j].semantic_similarity_score
                    loser = j if score_i >= score_j else i
                    redundant_local.add(loser)
                    loser_paper = active_papers[loser]
                    loser_paper.is_redundant = True
                    loser_paper.deflection_flag = _infer_redundancy_flag(loser_paper)
                    semantic_count += 1
                    logger.debug(
                        "Layer 3 near-duplicate: %s ≈ %s (sim=%.3f)",
                        active_papers[i].paper_id,
                        active_papers[j].paper_id,
                        sim_matrix[i, j],
                    )

        logger.info(
            "Layer 3 complete: %d semantic near-duplicates flagged.", semantic_count
        )
        return papers
