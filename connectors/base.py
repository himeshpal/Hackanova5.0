"""
connectors/base.py
------------------
BaseConnector

Provides:
  - shared aiohttp session management
  - global asyncio.Semaphore for rate limiting (1 req/s)
  - exponential back-off retry for transient failures
  - per-request sleep AFTER the call to stay within rate limit
  - timeout configuration
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import aiohttp

from config.settings import (
    BACKOFF_BASE_SECONDS,
    MAX_RETRIES,
    REQUESTS_PER_SECOND,
    REQUEST_TIMEOUT_SECONDS,
)
from core.types import Paper

logger = logging.getLogger(__name__)

# One global semaphore shared across all connectors — limits concurrent requests
_global_semaphore: asyncio.Semaphore | None = None

# Minimum seconds between requests (1 / REQUESTS_PER_SECOND)
_MIN_INTERVAL: float = 1.0 / REQUESTS_PER_SECOND


def get_semaphore() -> asyncio.Semaphore:
    """Return (or lazily create) the shared rate-limiting semaphore."""
    global _global_semaphore
    if _global_semaphore is None:
        _global_semaphore = asyncio.Semaphore(REQUESTS_PER_SECOND)
    return _global_semaphore


class BaseConnector(ABC):
    """
    Abstract base for all API connectors.

    Sub-classes must implement :meth:`search` and may override
    :meth:`get_citations`.
    """

    def __init__(self) -> None:
        self.session: Optional[aiohttp.ClientSession] = None

    @abstractmethod
    async def search(self, query: str, max_results: int) -> List[Paper]:
        """Query the upstream API and return a list of papers."""

    async def get_citations(self, paper_id: str, max_results: int) -> List[Paper]:
        """Return papers that cite *paper_id* (override in S2 connector)."""
        return []

    async def _get(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        response_format: str = "json",   # "json" | "text"
    ) -> Any:
        """
        HTTP GET with semaphore rate limiting, per-request sleep, and
        exponential back-off retry.

        Returns parsed JSON / raw text, or None on total failure.
        """
        semaphore = get_semaphore()
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)
        last_exc: Exception | None = None

        for attempt in range(MAX_RETRIES):
            async with semaphore:
                try:
                    assert self.session is not None, "HTTP session not initialised"
                    async with self.session.get(
                        url,
                        params=params,
                        headers=headers,
                        timeout=timeout,
                    ) as resp:

                        # Explicit rate-limit response — back off and retry
                        if resp.status == 429:
                            retry_after = int(resp.headers.get("Retry-After", "10"))
                            logger.warning(
                                "Rate limited by %s — sleeping %ds", url, retry_after
                            )
                            await asyncio.sleep(retry_after)
                            continue

                        resp.raise_for_status()

                        if response_format == "json":
                            data = await resp.json(content_type=None)
                        else:
                            data = await resp.text()

                        # Polite delay after every successful call
                        await asyncio.sleep(_MIN_INTERVAL)
                        return data

                except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                    last_exc = exc
                    backoff = BACKOFF_BASE_SECONDS ** attempt
                    logger.warning(
                        "Request to %s failed (attempt %d/%d): %s — retrying in %.1fs",
                        url, attempt + 1, MAX_RETRIES, exc, backoff,
                    )
                    await asyncio.sleep(backoff)

        logger.error("All %d retries failed for %s: %s", MAX_RETRIES, url, last_exc)
        return None