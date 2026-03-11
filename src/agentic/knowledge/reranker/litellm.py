"""
LiteLLM Reranker - unified interface for multiple reranking providers.

Supports all providers via litellm.rerank():
- Cohere: rerank-english-v3.0, rerank-multilingual-v3.0
- Jina AI: jina-reranker-v2-base-multilingual
- Voyage: rerank-2.5, rerank-2.5-lite
- Together AI: Salesforce/Llama-Rank-V1
- Azure AI: various models
- VLLM (self-hosted): any model via hosted_vllm/ prefix
"""

import asyncio
import logging
from typing import Any

from agentic.knowledge.model_config import RERANKER_DEFAULT_MODEL
from agentic.knowledge.reranker.base import Reranker, RerankResult

logger = logging.getLogger(__name__)


class LiteLLMReranker(Reranker):
    """
    Universal reranker using LiteLLM for multi-provider support.

    Args:
        model: Model identifier (e.g., 'cohere/rerank-english-v3.0')
        api_key: Optional API key (defaults to environment variable)
        api_base: Optional custom API endpoint (for self-hosted VLLM, etc.)
        timeout: Request timeout in seconds (default: 120)
        **provider_params: Provider-specific parameters

    Example:
        >>> reranker = LiteLLMReranker(model="cohere/rerank-english-v3.0")
        >>> results = reranker.rerank("What is AI?", ["doc1", "doc2"])
    """

    def __init__(
        self,
        model: str = RERANKER_DEFAULT_MODEL,
        api_key: str | None = None,
        api_base: str | None = None,
        timeout: int = 120,
        **provider_params: Any,
    ):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.timeout = timeout
        self.provider_params = provider_params

        logger.info(f"Initialized LiteLLM reranker with model: {model}")

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """Re-score documents against a query using litellm.rerank()."""
        try:
            from litellm import rerank as litellm_rerank
        except ImportError:
            raise ImportError(
                "litellm is required for LiteLLMReranker. "
                "Install with: pip install litellm"
            ) from None

        if not documents:
            return []

        kwargs: dict[str, Any] = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "timeout": self.timeout,
        }

        if top_n is not None:
            kwargs["top_n"] = top_n
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base

        kwargs["num_retries"] = 3
        kwargs.update(self.provider_params)

        try:
            response = litellm_rerank(**kwargs)

            results = [
                RerankResult(
                    index=item["index"],
                    relevance_score=item["relevance_score"],
                    text=documents[item["index"]],
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
            logger.error(f"Reranking failed with {self.model}: {e}")
            raise

    async def arerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """Async version of rerank (via run_in_executor)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self.rerank, query, documents, top_n
        )


class CohereReranker(LiteLLMReranker):
    """Cohere reranker (convenience alias).

    Example:
        >>> reranker = CohereReranker()  # Uses rerank-english-v3.0
    """

    def __init__(self, model: str = RERANKER_DEFAULT_MODEL, **kwargs: Any):
        super().__init__(model=model, **kwargs)


class VoyageReranker(LiteLLMReranker):
    """Voyage AI reranker (convenience alias).

    Example:
        >>> reranker = VoyageReranker()  # Uses rerank-2.5
    """

    def __init__(self, model: str = "voyage/rerank-2.5", **kwargs: Any):
        super().__init__(model=model, **kwargs)


class JinaReranker(LiteLLMReranker):
    """Jina AI reranker (convenience alias).

    Example:
        >>> reranker = JinaReranker()  # Uses jina-reranker-v2-base-multilingual
    """

    def __init__(
        self,
        model: str = "jina_ai/jina-reranker-v2-base-multilingual",
        **kwargs: Any,
    ):
        super().__init__(model=model, **kwargs)
