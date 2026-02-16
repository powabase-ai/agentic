"""
Reranker - base class for reranking providers.

Rerankers re-score retrieved documents using cross-encoder models that
evaluate query-document pairs jointly, improving precision over
bi-encoder (embedding) similarity alone.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class RerankResult:
    """A single result from a reranking operation.

    Attributes:
        index: Position of this document in the original input list.
        relevance_score: Reranker-assigned relevance score (higher is better).
        text: The document text that was scored.
    """

    index: int
    relevance_score: float
    text: str


class Reranker(ABC):
    """
    Base class for reranking providers.

    A Reranker takes a query and a list of candidate documents, then
    returns them re-scored by a cross-encoder model.  Implementations
    use various model providers (Cohere, Jina, Voyage, ZeroEntropy, etc.).

    Subclasses must implement:
        - rerank()   (sync)
        - arerank()  (async)

    Example:
        >>> class CohereReranker(Reranker):
        ...     model = "cohere/rerank-english-v3.0"
        ...
        ...     def rerank(self, query, documents, top_n=None):
        ...         ...
    """

    model: str = "unknown"

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """
        Re-score documents against a query.

        Args:
            query: The search query.
            documents: Candidate document texts to rerank.
            top_n: Return only the top N results. If None, return all.

        Returns:
            List of RerankResult sorted by relevance_score descending.
        """
        ...

    @abstractmethod
    async def arerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """Async version of rerank()."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"
