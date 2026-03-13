"""
utils/embeddings.py
-------------------
Singleton wrapper around SentenceTransformer.

The model is loaded once at first use and cached for the lifetime of the
process.  All downstream components call :func:`encode` rather than
instantiating the model themselves.
"""

from __future__ import annotations

import logging
from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# Module-level singleton — None until first call to get_model() / encode()
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """
    Return the shared :class:`SentenceTransformer` instance, loading it on
    first call.

    Returns
    -------
    SentenceTransformer
        The loaded embedding model.
    """
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        _model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded.")
    return _model


def encode(texts: Union[str, List[str]], batch_size: int = 64) -> np.ndarray:
    """
    Encode one or more texts into dense embedding vectors.

    Parameters
    ----------
    texts:
        A single string or list of strings to embed.
    batch_size:
        Batch size forwarded to the underlying model.

    Returns
    -------
    np.ndarray
        Shape ``(n_texts, embedding_dim)`` for a list input,
        or ``(embedding_dim,)`` for a single string.
    """
    model = get_model()
    if isinstance(texts, str):
        texts = [texts]
        result: np.ndarray = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,   # unit vectors → cosine = dot product
        )
        return result[0]

    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
