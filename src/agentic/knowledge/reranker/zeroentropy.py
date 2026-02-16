"""
ZeroEntropy Reranker - for zerank-1 and zerank-2 models.

ZeroEntropy models are not supported by litellm and require their own SDK.
The `zeroentropy` package is an optional dependency.
"""

import asyncio
import logging
from typing import Any

from agentic.knowledge.reranker.base import Reranker, RerankResult

logger = logging.getLogger(__name__)


class ZeroEntropyReranker(Reranker):
    """
    Reranker using the ZeroEntropy SDK (zerank-1, zerank-2).

    Args:
        model: ZeroEntropy model name (default: 'zerank-2')
        api_key: ZeroEntropy API key (defaults to ZEROENTROPY_API_KEY env var)

    Example:
        >>> reranker = ZeroEntropyReranker()  # Uses zerank-2
        >>> results = reranker.rerank("What is AI?", ["doc1", "doc2"])
    """

    def __init__(
        self,
        model: str = "zerank-2",
        api_key: str | None = None,
    ):
        self.model = model
        self.api_key = api_key

        logger.info(f"Initialized ZeroEntropy reranker with model: {model}")

    def _get_client(self) -> Any:
        """Lazy-import and instantiate the ZeroEntropy client."""
        try:
            from zeroentropy import ZeroEntropy
        except ImportError:
            raise ImportError(
                "zeroentropy is required for ZeroEntropyReranker. "
                "Install with: pip install zeroentropy"
            ) from None

        kwargs: dict[str, Any] = {}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        return ZeroEntropy(**kwargs)

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """Re-score documents against a query using ZeroEntropy."""
        if not documents:
            return []

        client = self._get_client()

        try:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "query": query,
                "documents": documents,
            }
            if top_n is not None:
                kwargs["top_n"] = top_n

            response = client.models.rerank(**kwargs)

            results = [
                RerankResult(
                    index=item.index,
                    relevance_score=item.relevance_score,
                    text=documents[item.index],
                )
                for item in response.results
            ]

            # Sort by relevance_score descending
            results.sort(key=lambda r: r.relevance_score, reverse=True)

            logger.debug(
                f"Reranked {len(documents)} documents using {self.model}, "
                f"returning {len(results)} results"
            )

            return results

        except Exception as e:
            logger.error(f"ZeroEntropy reranking failed with {self.model}: {e}")
            raise

    async def arerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """Async version of rerank (via run_in_executor)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.rerank, query, documents, top_n
        )
