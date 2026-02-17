"""
Reranker module - providers for re-scoring retrieved documents.

Rerankers use cross-encoder models to jointly evaluate query-document
pairs, improving retrieval precision beyond what bi-encoder embeddings
achieve alone.

Built-in rerankers:
- LiteLLMReranker: Universal reranker for Cohere, Jina, Voyage, VLLM, etc.
- CohereReranker: Cohere rerank models
- VoyageReranker: Voyage AI rerank models
- JinaReranker: Jina AI reranker models
- ZeroEntropyReranker: ZeroEntropy zerank models

Example:
    >>> from agentic.knowledge.reranker import CohereReranker
    >>>
    >>> reranker = CohereReranker()
    >>> results = reranker.rerank("What is AI?", ["doc1", "doc2", "doc3"])
    >>> print(results[0].relevance_score)
"""

from agentic.knowledge.reranker.base import Reranker, RerankResult
from agentic.knowledge.reranker.litellm import (
    CohereReranker,
    JinaReranker,
    LiteLLMReranker,
    VoyageReranker,
)
from agentic.knowledge.reranker.zeroentropy import ZeroEntropyReranker

__all__ = [
    "Reranker",
    "RerankResult",
    "LiteLLMReranker",
    "CohereReranker",
    "VoyageReranker",
    "JinaReranker",
    "ZeroEntropyReranker",
]
